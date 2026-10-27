//! `PreparedTable` definition, accessors, and target-stats helpers.

use super::descriptors::{compute_table_descriptors, Level, ShapeOverride, TableDescriptors};
use super::np::{pairwise_sum, squash};
use super::scaling::{
    err, ClassVocabPolicy, ColumnKind, FitOnlyTableTransform, PrepTask, TargetValue,
};
use crate::error::Result;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

/// Train-only prepared table: fitted transform plus the scaled training data,
/// physical row IDs, categorical metadata for [`crate::landmark::LandmarkFrame`],
/// and cached static [`TableDescriptors`] / classification `target_stats`.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedTable {
    pub task: PrepTask,
    pub rows: usize,
    pub features: usize,
    /// 0 for regression, 2 for binary classification, number of class labels for multiclass.
    pub classes: usize,
    /// Row-major `rows × features`.
    pub x_scaled: Vec<f64>,
    pub y_scaled: Vec<f64>,
    pub transform: FitOnlyTableTransform,
    /// Physical row identities of the training rows (defaults to `0..rows as u64`).
    pub row_ids: Vec<u64>,
    /// Per-column category cardinality (`None` = numeric, `Some(k)` = categorical with `k` levels)
    /// forwarded to [`crate::landmark::LandmarkFrame::from_columns`].
    pub categories: Vec<Option<usize>>,
    /// Optional per-column category vocabulary tokens when prepared from raw string cells
    /// or supplied via [`PreparedTable::with_category_tokens`].
    pub category_tokens: Option<Vec<Vec<String>>>,
    /// Column-major `features × rows` values suitable for [`crate::landmark::LandmarkFrame::from_columns`]
    /// (finite raw numeric values or `NaN`, and integer category codes `0..k` or `NaN`).
    pub frame_columns: Vec<Vec<f64>>,
    /// Target vector suitable for [`crate::landmark::LandmarkFrame::from_columns`].
    pub frame_y: Vec<f64>,
    /// Full-table native `target_stats` (`[skew, excess_kurt]` on raw `frame_y` for
    /// regression, `[entropy_norm, min_share]` on `y_scaled` for classification/multiclass;
    /// `[0.0, 0.0]` when degenerate), matching `landmark::Signature.target_stats` on the
    /// unsampled table so subsampled `full` reports can override `vector116[85:87]`.
    pub target_stats: [f64; 2],
    /// Observed-only `target_stats` (`Some([entropy_norm, min_share])` for
    /// classification/multiclass when `classes >= 2`, `None` for regression).
    pub observed_target_stats: Option<[f64; 2]>,
    /// Cached static `TableDescriptors` for [`Level::High`] so progressive updates
    /// never rescan or sort training columns.
    pub high_table_descriptors: TableDescriptors,
    /// Optional SHA-256 of the raw string cells + target values when constructed via
    /// [`super::scaling::prepare_table_cells`], [`super::scaling::prepare_table_cells_with_ids`],
    /// or [`super::scaling::prepare_table_mixed`].
    pub raw_cells_sha256: Option<[u8; 32]>,
    /// Optional caller-supplied source digest override returned by [`PreparedTable::source_rows_digest`].
    pub source_digest_override: Option<String>,
}

pub(crate) fn validate_row_ids(row_ids: &[u64], rows: usize) -> Result<()> {
    if row_ids.len() != rows {
        return Err(err("prepare_row_ids_length_mismatch"));
    }
    let mut seen = HashSet::with_capacity(row_ids.len());
    if !row_ids.iter().all(|&id| seen.insert(id)) {
        return Err(err("prepare_duplicate_row_ids"));
    }
    Ok(())
}

pub(crate) fn validate_categories(categories: &[Option<usize>], features: usize) -> Result<()> {
    if categories.len() != features {
        return Err(err("prepare_categories_length_mismatch"));
    }
    if categories
        .iter()
        .any(|c| c.is_some_and(|k| k == 0 || k > 257))
    {
        return Err(err("prepare_invalid_category_cardinality"));
    }
    Ok(())
}

#[inline]
pub(crate) fn canonical_f64_bits(v: f64) -> u64 {
    if v == 0.0 {
        0.0f64.to_bits()
    } else if v.is_nan() {
        f64::NAN.to_bits()
    } else {
        v.to_bits()
    }
}

pub(crate) fn hash_raw_cells(x: &[Vec<String>], yv: &[TargetValue<'_>]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"dope.prepared-cells/v1\0");
    h.update((x.len() as u64).to_le_bytes());
    for row in x {
        h.update((row.len() as u64).to_le_bytes());
        for cell in row {
            h.update((cell.len() as u64).to_le_bytes());
            h.update(cell.as_bytes());
        }
    }
    h.update((yv.len() as u64).to_le_bytes());
    for v in yv {
        match v {
            TargetValue::Num(f) => {
                h.update([0u8]);
                h.update(canonical_f64_bits(*f).to_le_bytes());
            }
            TargetValue::Str(s) => {
                h.update([1u8]);
                h.update((s.len() as u64).to_le_bytes());
                h.update(s.as_bytes());
            }
        }
    }
    h.finalize().into()
}

/// Native classification `target_stats` (`[normalized_entropy, min_class_share]`)
/// evaluated on `y_scaled` (`comms.md` / `optional_landmarks.observed_target_stats`).
/// Returns `None` for regression or when fewer than 2 classes exist.
pub fn compute_observed_target_stats(y_scaled: &[f64], task: PrepTask) -> Option<[f64; 2]> {
    if task == PrepTask::Regression || y_scaled.is_empty() {
        return None;
    }
    let n = y_scaled.len();
    let labels: Vec<i64> = y_scaled
        .iter()
        .map(|&v| v.round_ties_even() as i64)
        .collect();
    let classes: Vec<i64> = if task == PrepTask::Classification {
        vec![0, 1]
    } else {
        let mut c = labels.clone();
        c.sort_unstable();
        c.dedup();
        c
    };
    if classes.len() < 2 {
        return None;
    }
    let mut counts = vec![0.0f64; classes.len()];
    for &lab in &labels {
        if let Ok(idx) = classes.binary_search(&lab) {
            counts[idx] += 1.0 / n as f64;
        }
    }
    let pos_terms: Vec<f64> = counts
        .iter()
        .copied()
        .filter(|&c| c > 0.0)
        .map(|c| c * c.ln())
        .collect();
    let ent = -pairwise_sum(&pos_terms) / (counts.len() as f64).ln();
    let min_share = counts.iter().copied().reduce(f64::min).unwrap_or(0.0);
    Some([ent, min_share])
}

/// Native `Signature.target_stats` formula evaluated on the full prepared table:
/// for regression `(skew, excess_kurt)` on raw finite `frame_y` (or `[0.0, 0.0]` if `v <= 1e-24`);
/// for classification/multiclass `(normalized_entropy, min_class_share)` on `y_scaled`.
pub fn compute_native_target_stats(frame_y: &[f64], y_scaled: &[f64], task: PrepTask) -> [f64; 2] {
    if task == PrepTask::Regression {
        let n = frame_y.len();
        if n == 0 {
            return [0.0, 0.0];
        }
        let m = frame_y.iter().sum::<f64>() / n as f64;
        let v = frame_y.iter().map(|&val| (val - m).powi(2)).sum::<f64>() / n as f64;
        if !v.is_finite() || v <= 1e-24 {
            return [0.0, 0.0];
        }
        let skew =
            frame_y.iter().map(|&val| (val - m).powi(3)).sum::<f64>() / n as f64 / v.powf(1.5);
        let kurt =
            frame_y.iter().map(|&val| (val - m).powi(4)).sum::<f64>() / n as f64 / v.powi(2) - 3.0;
        [
            if skew.is_finite() { skew } else { 0.0 },
            if kurt.is_finite() { kurt } else { 0.0 },
        ]
    } else {
        compute_observed_target_stats(y_scaled, task).unwrap_or([0.0, 0.0])
    }
}

impl PreparedTable {
    #[inline]
    pub fn x(&self, i: usize, j: usize) -> f64 {
        self.x_scaled[i * self.features + j]
    }
    pub fn column(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.x(i, j)).collect()
    }

    /// Returns static [`TableDescriptors`] for `level`, reusing the cached
    /// [`PreparedTable::high_table_descriptors`] when `level == Level::High`.
    pub fn table_descriptors(&self, level: Level) -> TableDescriptors {
        if level == Level::High {
            self.high_table_descriptors.clone()
        } else {
            compute_table_descriptors(
                &self.x_scaled,
                self.rows,
                self.features,
                &self.y_scaled,
                level,
            )
        }
    }

    /// Borrow the cached [`Level::High`] static table descriptors without cloning.
    pub fn high_table_descriptors(&self) -> &TableDescriptors {
        &self.high_table_descriptors
    }

    /// Squashed 2-element full-table `target_stats` (`[squash(ts[0]), squash(ts[1])]`).
    pub fn squashed_target_stats(&self) -> [f64; 2] {
        let [a, b] = self.target_stats;
        [
            if a.is_finite() { squash(a) } else { 0.0 },
            if b.is_finite() { squash(b) } else { 0.0 },
        ]
    }

    /// Squashed 2-element observed-only `target_stats` (`[0.0, 0.0]` for regression).
    pub fn squashed_observed_target_stats(&self) -> [f64; 2] {
        match self.observed_target_stats {
            Some([a, b]) => [
                if a.is_finite() { squash(a) } else { 0.0 },
                if b.is_finite() { squash(b) } else { 0.0 },
            ],
            None => [0.0, 0.0],
        }
    }

    /// Native report-shaped 3-element `target_stats` for `observed-only-v1`.
    pub fn report_target_stats(&self) -> Vec<Option<f64>> {
        match self.observed_target_stats {
            Some([a, b]) => vec![Some(a), Some(b), None],
            None => vec![None, None, None],
        }
    }

    /// Full-table [`ShapeOverride`] (`rows`, `features`, `classes`, and native
    /// `target_stats`) used to override subsampled reports in `extract_report_descriptors`.
    pub fn shape_override(&self) -> ShapeOverride {
        ShapeOverride::from_table(self)
    }

    /// Extract row-major `x_scaled` (`positions.len() × features`) and `y_scaled`
    /// (`positions.len()`) for a subset of row positions `0..rows`, reusing the
    /// full train-fitted transform without refitting.
    pub fn select_scaled_rows(&self, positions: &[usize]) -> Result<(Vec<f64>, Vec<f64>)> {
        if positions.iter().any(|&r| r >= self.rows) {
            return Err(err("prepare_row_selection_out_of_bounds"));
        }
        let p = self.features;
        let mut x_sub = Vec::with_capacity(positions.len() * p);
        let mut y_sub = Vec::with_capacity(positions.len());
        for &r in positions {
            x_sub.extend_from_slice(&self.x_scaled[r * p..(r + 1) * p]);
            y_sub.push(self.y_scaled[r]);
        }
        Ok((x_sub, y_sub))
    }

    /// Replace physical `row_ids` (builder pattern).
    pub fn with_row_ids(mut self, row_ids: Vec<u64>) -> Result<Self> {
        self.set_row_ids(row_ids)?;
        Ok(self)
    }

    /// Mutably replace physical `row_ids`.
    pub fn set_row_ids(&mut self, row_ids: Vec<u64>) -> Result<()> {
        validate_row_ids(&row_ids, self.rows)?;
        self.row_ids = row_ids;
        Ok(())
    }

    /// Set per-column category cardinalities (`None` for numeric, `Some(k)` for categorical).
    pub fn with_categories(mut self, categories: Vec<Option<usize>>) -> Result<Self> {
        validate_categories(&categories, self.features)?;
        self.categories = categories;
        Ok(self)
    }

    /// Set per-column category vocabularies (empty inner Vec = numeric, non-empty = categorical).
    pub fn with_category_tokens(mut self, tokens: Vec<Vec<String>>) -> Result<Self> {
        if tokens.len() != self.features {
            return Err(err("prepare_categories_length_mismatch"));
        }
        let cats: Vec<Option<usize>> = tokens
            .iter()
            .map(|v| if v.is_empty() { None } else { Some(v.len()) })
            .collect();
        validate_categories(&cats, self.features)?;
        self.categories = cats;
        self.category_tokens = Some(tokens);
        Ok(self)
    }

    /// Override the digest returned by [`PreparedTable::source_rows_digest`] with an
    /// explicit caller-computed source digest (builder pattern).
    pub fn with_source_digest(mut self, digest: impl Into<String>) -> Self {
        self.set_source_digest(digest);
        self
    }

    /// Mutably set an explicit caller-computed source digest override.
    pub fn set_source_digest(&mut self, digest: impl Into<String>) {
        let d = digest.into();
        self.source_digest_override = if d.is_empty() { None } else { Some(d) };
    }

    /// Native [`crate::target::Head`] corresponding to this prepared table.
    pub fn head(&self) -> crate::target::Head {
        match self.task {
            PrepTask::Regression => crate::target::Head::Regression,
            PrepTask::Classification => crate::target::Head::Binary,
            PrepTask::Multiclass => crate::target::Head::Multiclass(self.classes),
        }
    }

    /// Construct a [`crate::landmark::LandmarkFrame`] preserving physical `row_ids`
    /// and categorical columns.
    pub fn to_landmark_frame(&self) -> Result<crate::landmark::LandmarkFrame> {
        self.to_landmark_frame_with_groups(None)
    }

    /// Alias for [`PreparedTable::to_landmark_frame`].
    pub fn landmark_frame(&self) -> Result<crate::landmark::LandmarkFrame> {
        self.to_landmark_frame_with_groups(None)
    }

    /// Construct a [`crate::landmark::LandmarkFrame`] with optional group/time-block IDs.
    pub fn to_landmark_frame_with_groups(
        &self,
        groups: Option<Vec<u64>>,
    ) -> Result<crate::landmark::LandmarkFrame> {
        crate::landmark::LandmarkFrame::from_columns(
            self.frame_columns.clone(),
            self.categories.clone(),
            self.frame_y.clone(),
            self.head(),
            self.row_ids.clone(),
            groups,
        )
    }

    /// Returns the explicit `source_digest_override` if set via [`PreparedTable::with_source_digest`],
    /// otherwise computes a deterministic SHA-256 hex digest.
    pub fn source_rows_digest(&self) -> String {
        if let Some(ref d) = self.source_digest_override {
            return d.clone();
        }
        self.computed_source_rows_digest()
    }

    /// Deterministic SHA-256 hex digest over all prepared table identity fields (ignoring
    /// `source_digest_override`).
    pub fn computed_source_rows_digest(&self) -> String {
        let mut h = Sha256::new();
        match self.transform.class_vocab_policy {
            ClassVocabPolicy::Cap32V1 => h.update(b"dope.prepared-table/v1\0"),
            ClassVocabPolicy::LosslessV1 => h.update(b"dope.prepared-table/lossless-v1\0"),
        }
        h.update(self.task.as_str().as_bytes());
        h.update(b"\0");
        for v in [self.rows as u64, self.features as u64, self.classes as u64] {
            h.update(v.to_le_bytes());
        }
        for &id in &self.row_ids {
            h.update(id.to_le_bytes());
        }
        for &c in &self.categories {
            h.update(c.map_or(u64::MAX, |k| k as u64).to_le_bytes());
        }
        for &k in &self.transform.kinds {
            h.update([match k {
                ColumnKind::Numeric => 0u8,
                ColumnKind::Categorical => 1u8,
            }]);
        }
        match &self.category_tokens {
            None => h.update([0u8]),
            Some(cols) => {
                h.update([1u8]);
                h.update((cols.len() as u64).to_le_bytes());
                for col in cols {
                    h.update((col.len() as u64).to_le_bytes());
                    for tok in col {
                        h.update((tok.len() as u64).to_le_bytes());
                        h.update(tok.as_bytes());
                    }
                }
            }
        }
        for opt_ce in &self.transform.categorical {
            match opt_ce {
                None => h.update([0u8]),
                Some(ce) => {
                    h.update([1u8]);
                    h.update((ce.vocab_tokens.len() as u64).to_le_bytes());
                    for tok in &ce.vocab_tokens {
                        h.update((tok.len() as u64).to_le_bytes());
                        h.update(tok.as_bytes());
                    }
                }
            }
        }
        match &self.transform.target.positive_label {
            None => h.update([0u8]),
            Some(pos) => {
                h.update([1u8]);
                h.update((pos.len() as u64).to_le_bytes());
                h.update(pos.as_bytes());
            }
        }
        h.update((self.transform.target.class_labels.len() as u64).to_le_bytes());
        for &lbl in &self.transform.target.class_labels {
            h.update(canonical_f64_bits(lbl).to_le_bytes());
        }
        match &self.raw_cells_sha256 {
            None => h.update([0u8]),
            Some(raw) => {
                h.update([1u8]);
                h.update(raw);
            }
        }
        for &v in self
            .x_scaled
            .iter()
            .chain(self.y_scaled.iter())
            .chain(self.frame_columns.iter().flatten())
            .chain(self.frame_y.iter())
        {
            h.update(canonical_f64_bits(v).to_le_bytes());
        }
        format!("{:x}", h.finalize())
    }
}

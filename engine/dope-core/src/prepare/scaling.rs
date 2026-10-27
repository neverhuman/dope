//! Rust port of `research/levels/scaling.py` (`FitOnlyTableTransform`).
//!
//! Train-only per-column scaling and categorical encoding producing the
//! `X_train_scaled` / `y_train_scaled` arrays consumed by the MLC-v3
//! conditioning builders. Statistics are fitted on the training rows only.

use super::descriptors::{compute_table_descriptors, Level, ShapeOverride, TableDescriptors};
use super::np::{
    argsort_i64, clip, interp, linspace, mean1d, median, pairwise_sum, quantiles, round6, sorted,
    squash, std1d,
};
use crate::error::{DopeError, Result};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

/// `scaling.QUANTILE_GRID`.
pub const QUANTILE_GRID: [f64; 9] = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];
/// `scaling.NORMAL_GRID`.
pub const NORMAL_GRID: [f64; 9] = [
    -2.32634787,
    -1.64485363,
    -1.28155157,
    -0.67448975,
    0.0,
    0.67448975,
    1.28155157,
    1.64485363,
    2.32634787,
];

fn err(msg: &str) -> DopeError {
    DopeError::Input(msg.to_string())
}

/// Python task string accepted by `FitOnlyTableTransform.fit(task=...)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrepTask {
    Regression,
    Classification,
    Multiclass,
}

impl PrepTask {
    pub fn as_str(self) -> &'static str {
        match self {
            PrepTask::Regression => "regression",
            PrepTask::Classification => "classification",
            PrepTask::Multiclass => "multiclass",
        }
    }
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "regression" => Ok(PrepTask::Regression),
            "classification" | "binary" => Ok(PrepTask::Classification),
            "multiclass" => Ok(PrepTask::Multiclass),
            _ => Err(err("prepare_unknown_task")),
        }
    }
}

impl From<crate::schema::Task> for PrepTask {
    fn from(t: crate::schema::Task) -> Self {
        match t {
            crate::schema::Task::Regression => PrepTask::Regression,
            crate::schema::Task::Classification => PrepTask::Classification,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnKind {
    Numeric,
    Categorical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformKind {
    Constant,
    Binary,
    SignedLog1pHybrid,
    AsinhHybrid,
    QuantileRobustHybrid,
}

impl TransformKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TransformKind::Constant => "constant",
            TransformKind::Binary => "binary",
            TransformKind::SignedLog1pHybrid => "signed_log1p_hybrid",
            TransformKind::AsinhHybrid => "asinh_hybrid",
            TransformKind::QuantileRobustHybrid => "quantile_robust_hybrid",
        }
    }
}

/// `scaling.NumericColumnScaler`.
#[derive(Clone, Debug, PartialEq)]
pub struct NumericColumnScaler {
    pub slot: usize,
    pub transform_kind: TransformKind,
    pub missing_rate: f64,
    pub center: f64,
    pub scale: f64,
    pub post_mean: f64,
    pub post_std: f64,
    pub quantiles: [f64; 9],
    pub skewness: f64,
    pub kurtosis: f64,
    pub blend_quantile_weight: f64,
}

impl NumericColumnScaler {
    fn raw_transform(&self, v: f64) -> f64 {
        let z_log = match self.transform_kind {
            TransformKind::Constant => return 0.0,
            TransformKind::Binary => {
                return clip((v - self.center) / self.scale.max(1e-9), -6.0, 6.0);
            }
            TransformKind::SignedLog1pHybrid => {
                let shifted = v - self.center;
                let scaled = np_sign(shifted) * (shifted.abs() / self.scale.max(1e-9)).ln_1p();
                (scaled - self.post_mean) / self.post_std.max(1e-9)
            }
            TransformKind::AsinhHybrid => {
                let shifted = v - self.center;
                let scaled = (shifted / self.scale.max(1e-9)).asinh();
                (scaled - self.post_mean) / self.post_std.max(1e-9)
            }
            TransformKind::QuantileRobustHybrid => (v - self.post_mean) / self.post_std.max(1e-9),
        };
        let q = &self.quantiles;
        let z = if q.windows(2).all(|w| w[1] - w[0] > 1e-12) {
            let iqr = (q[5] - q[3]).max(1e-9);
            let z_q = if v < q[0] {
                NORMAL_GRID[0] + (v - q[0]) / iqr * 1.3489795
            } else if v > q[8] {
                NORMAL_GRID[8] + (v - q[8]) / iqr * 1.3489795
            } else {
                interp(v, q, &NORMAL_GRID)
            };
            let w = self.blend_quantile_weight;
            (1.0 - w) * z_log + w * z_q
        } else {
            z_log
        };
        6.0 * (z / 6.0).tanh()
    }

    /// Returns (z, missing indicator) for one value.
    pub fn transform_value(&self, v: f64) -> (f64, f64) {
        if v.is_finite() {
            (self.raw_transform(v), 0.0)
        } else {
            (0.0, 1.0)
        }
    }
}

#[inline]
fn np_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x.is_nan() {
        f64::NAN
    } else {
        0.0
    }
}

/// `scaling.CategoricalColumnEncoder`.
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalColumnEncoder {
    pub slot: usize,
    pub missing_rate: f64,
    pub vocab_tokens: Vec<String>,
    pub token_to_id: HashMap<String, usize>,
    pub id_target_encoding: Vec<f64>,
    pub id_frequencies: Vec<f64>,
    pub post_mean: f64,
    pub post_std: f64,
}

impl CategoricalColumnEncoder {
    /// Returns (z, id, missing) for one raw cell.
    pub fn transform_cell(&self, cell: &str) -> (f64, usize, f64) {
        let s = py_strip(cell);
        let (miss, idx) = if is_missing_token(s) {
            (1.0, 0)
        } else {
            (0.0, self.token_to_id.get(s).copied().unwrap_or(0))
        };
        let enc = self.id_target_encoding[idx];
        let z = 6.0 * (((enc - self.post_mean) / self.post_std.max(1e-9)) / 6.0).tanh();
        (z, idx, miss)
    }
}

/// `scaling.TargetScaler`.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetScaler {
    pub task: PrepTask,
    pub mean: f64,
    pub std: f64,
    pub median: f64,
    pub iqr: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub positive_label: Option<String>,
    pub class_prior: f64,
    pub class_labels: Vec<f64>,
    pub class_priors: Vec<f64>,
}

/// One raw target value: numeric (ndarray / python number) or string (CSV cell).
#[derive(Clone, Copy, Debug)]
pub enum TargetValue<'a> {
    Num(f64),
    Str(&'a str),
}

impl TargetScaler {
    fn transform(&self, y: &[TargetValue<'_>]) -> Result<Vec<f64>> {
        if self.task == PrepTask::Multiclass && !self.class_labels.is_empty() {
            let labels = &self.class_labels;
            return Ok(y
                .iter()
                .map(|v| {
                    let mut a = match v {
                        TargetValue::Num(x) => *x,
                        TargetValue::Str(s) => parse_float_cell(s),
                    };
                    // np.nan_to_num(arr, nan=labels[0])
                    if a.is_nan() {
                        a = labels[0];
                    } else if a == f64::INFINITY {
                        a = f64::MAX;
                    } else if a == f64::NEG_INFINITY {
                        a = f64::MIN;
                    }
                    let mut best = 0usize;
                    let mut bd = (a - labels[0]).abs();
                    for (k, l) in labels.iter().enumerate().skip(1) {
                        let d = (a - l).abs();
                        if d < bd {
                            bd = d;
                            best = k;
                        }
                    }
                    best as f64
                })
                .collect());
        }
        if self.task == PrepTask::Classification {
            let pos = self.positive_label.as_deref().unwrap_or("1");
            return Ok(y
                .iter()
                .map(|v| match v {
                    TargetValue::Num(x) => {
                        if *x > 0.5 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    TargetValue::Str(s) => {
                        if py_strip(s) == pos {
                            1.0
                        } else {
                            0.0
                        }
                    }
                })
                .collect());
        }
        let arr = regression_array(y)?;
        Ok(arr
            .iter()
            .map(|&a| {
                if a.is_finite() {
                    8.0 * (((a - self.mean) / self.std.max(1e-9)) / 8.0).tanh()
                } else {
                    0.0
                }
            })
            .collect())
    }
}

/// `scaling.FitOnlyTableTransform` (fitted state).
#[derive(Clone, Debug, PartialEq)]
pub struct FitOnlyTableTransform {
    pub task: PrepTask,
    pub kinds: Vec<ColumnKind>,
    pub numeric: Vec<Option<NumericColumnScaler>>,
    pub categorical: Vec<Option<CategoricalColumnEncoder>>,
    pub target: TargetScaler,
    pub min_level_count: usize,
    pub prior_smoothing: f64,
}

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
    /// [`prepare_table_cells`], [`prepare_table_cells_with_ids`], or [`prepare_table_mixed`].
    pub raw_cells_sha256: Option<[u8; 32]>,
    /// Optional caller-supplied source digest override returned by [`PreparedTable::source_rows_digest`].
    pub source_digest_override: Option<String>,
}

fn validate_row_ids(row_ids: &[u64], rows: usize) -> Result<()> {
    if row_ids.len() != rows {
        return Err(err("prepare_row_ids_length_mismatch"));
    }
    let mut seen = HashSet::with_capacity(row_ids.len());
    if !row_ids.iter().all(|&id| seen.insert(id)) {
        return Err(err("prepare_duplicate_row_ids"));
    }
    Ok(())
}

fn validate_categories(categories: &[Option<usize>], features: usize) -> Result<()> {
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
fn canonical_f64_bits(v: f64) -> u64 {
    if v == 0.0 {
        0.0f64.to_bits()
    } else if v.is_nan() {
        f64::NAN.to_bits()
    } else {
        v.to_bits()
    }
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
    /// otherwise computes a deterministic SHA-256 hex digest binding task, shape, classes,
    /// physical `row_ids`, per-column kinds/categories/vocabularies, target label metadata,
    /// raw cells (when present), and both scaled and raw/frame table contents.
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
        h.update(b"dope.prepared-table/v1\0");
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

fn classes_of(t: &TargetScaler) -> usize {
    match t.task {
        PrepTask::Regression => 0,
        PrepTask::Classification => 2,
        PrepTask::Multiclass => t.class_labels.len(),
    }
}

/// Python `FitOnlyTableTransform.fit(X: float ndarray, y, task).transform(X, y)`
/// (all-numeric ndarray path) with explicit physical `row_ids`.
pub fn prepare_table_with_ids(
    x: &[Vec<f64>],
    y: &[f64],
    task: PrepTask,
    row_ids: Vec<u64>,
) -> Result<PreparedTable> {
    let n = x.len();
    let p = x.first().map_or(0, Vec::len);
    if x.iter().any(|r| r.len() != p) {
        return Err(err("prepare_ragged_rows"));
    }
    if y.len() != n {
        return Err(err("prepare_target_length_mismatch"));
    }
    validate_row_ids(&row_ids, n)?;
    let yv: Vec<TargetValue<'_>> = y.iter().map(|&v| TargetValue::Num(v)).collect();
    let target = fit_target_scaler(&yv, task)?;
    let mut numeric = Vec::with_capacity(p);
    let mut frame_columns = Vec::with_capacity(p);
    for j in 0..p {
        let col: Vec<f64> = x.iter().map(|r| r[j]).collect();
        numeric.push(Some(fit_numeric_column(j, &col)));
        frame_columns.push(
            col.into_iter()
                .map(|v| if v.is_finite() { v } else { f64::NAN })
                .collect(),
        );
    }
    let mut xs = vec![0.0; n * p];
    for j in 0..p {
        let sc = numeric[j].as_ref().expect("numeric");
        for i in 0..n {
            xs[i * p + j] = sc.transform_value(x[i][j]).0;
        }
    }
    let y_scaled = target.transform(&yv)?;
    let frame_y = if task == PrepTask::Regression {
        let fallback = if target.mean.is_finite() {
            target.mean
        } else {
            0.0
        };
        y.iter()
            .map(|&v| if v.is_finite() { v } else { fallback })
            .collect()
    } else {
        y_scaled.clone()
    };
    let observed_target_stats = compute_observed_target_stats(&y_scaled, task);
    let target_stats = compute_native_target_stats(&frame_y, &y_scaled, task);
    let high_table_descriptors = compute_table_descriptors(&xs, n, p, &y_scaled, Level::High);
    let transform = FitOnlyTableTransform {
        task,
        kinds: vec![ColumnKind::Numeric; p],
        numeric,
        categorical: vec![None; p],
        target,
        min_level_count: 5,
        prior_smoothing: 10.0,
    };
    Ok(PreparedTable {
        task,
        rows: n,
        features: p,
        classes: classes_of(&transform.target),
        x_scaled: xs,
        y_scaled,
        transform,
        row_ids,
        categories: vec![None; p],
        category_tokens: None,
        frame_columns,
        frame_y,
        target_stats,
        observed_target_stats,
        high_table_descriptors,
        raw_cells_sha256: None,
        source_digest_override: None,
    })
}

/// Python `FitOnlyTableTransform.fit(X: float ndarray, y, task).transform(X, y)`
/// (all-numeric ndarray path) using sequential `0..n as u64` row IDs.
pub fn prepare_table(x: &[Vec<f64>], y: &[f64], task: PrepTask) -> Result<PreparedTable> {
    let row_ids = (0..x.len() as u64).collect();
    prepare_table_with_ids(x, y, task, row_ids)
}

fn hash_raw_cells(x: &[Vec<String>], yv: &[TargetValue<'_>]) -> [u8; 32] {
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

fn prepare_cells_with_yv(
    x: &[Vec<String>],
    yv: &[TargetValue<'_>],
    declared_kinds: Option<&[ColumnKind]>,
    task: PrepTask,
    row_ids: Vec<u64>,
) -> Result<PreparedTable> {
    let n = x.len();
    let p = x.first().map_or(0, Vec::len);
    if x.iter().any(|r| r.len() != p) {
        return Err(err("prepare_ragged_rows"));
    }
    if yv.len() != n {
        return Err(err("prepare_target_length_mismatch"));
    }
    validate_row_ids(&row_ids, n)?;
    let raw_cells_sha256 = Some(hash_raw_cells(x, yv));
    let min_level_count = 5usize;
    let prior_smoothing = 10.0f64;
    let (kinds, mut parsed_numeric_cols): (Vec<ColumnKind>, Vec<Option<Vec<f64>>>) =
        match declared_kinds {
            Some(dk) => {
                if dk.len() != p {
                    return Err(err("prepare_kinds_length_mismatch"));
                }
                (dk.to_vec(), vec![None; p])
            }
            None => infer_feature_kinds_and_parse(x, p),
        };
    let target = fit_target_scaler(yv, task)?;
    let y_num = target.transform(yv)?;
    let y_global_mean = if y_num.is_empty() {
        0.0
    } else {
        mean1d(&y_num)
    };
    let mut numeric = vec![None; p];
    let mut categorical = vec![None; p];
    for (j, kind) in kinds.iter().enumerate() {
        match kind {
            ColumnKind::Numeric => {
                let col = parsed_numeric_cols[j]
                    .get_or_insert_with(|| x.iter().map(|r| parse_float_cell(&r[j])).collect());
                numeric[j] = Some(fit_numeric_column(j, col));
            }
            ColumnKind::Categorical => {
                let col: Vec<&str> = x.iter().map(|r| r[j].as_str()).collect();
                categorical[j] = Some(fit_categorical_column(
                    j,
                    &col,
                    &y_num,
                    y_global_mean,
                    min_level_count,
                    prior_smoothing,
                ));
            }
        }
    }
    let mut xs = vec![0.0; n * p];
    let mut categories = vec![None; p];
    let mut frame_columns = vec![vec![f64::NAN; n]; p];
    for j in 0..p {
        match kinds[j] {
            ColumnKind::Numeric => {
                let sc = numeric[j].as_ref().expect("numeric");
                let col = parsed_numeric_cols[j].as_ref().expect("parsed numeric");
                for (i, &raw) in col.iter().enumerate() {
                    xs[i * p + j] = sc.transform_value(raw).0;
                    frame_columns[j][i] = if raw.is_finite() { raw } else { f64::NAN };
                }
            }
            ColumnKind::Categorical => {
                let ce = categorical[j].as_ref().expect("categorical");
                categories[j] = Some(ce.vocab_tokens.len());
                for i in 0..n {
                    let (z, idx, miss) = ce.transform_cell(&x[i][j]);
                    xs[i * p + j] = z;
                    frame_columns[j][i] = if miss > 0.0 { f64::NAN } else { idx as f64 };
                }
            }
        }
    }
    let category_tokens = if kinds.contains(&ColumnKind::Categorical) {
        Some(
            (0..p)
                .map(|j| {
                    categorical[j]
                        .as_ref()
                        .map_or_else(Vec::new, |ce| ce.vocab_tokens.clone())
                })
                .collect(),
        )
    } else {
        None
    };
    let frame_y = if task == PrepTask::Regression {
        let raw_y = regression_array(yv)?;
        let fallback = if target.mean.is_finite() {
            target.mean
        } else {
            0.0
        };
        raw_y
            .into_iter()
            .map(|v| if v.is_finite() { v } else { fallback })
            .collect()
    } else {
        y_num.clone()
    };
    let observed_target_stats = compute_observed_target_stats(&y_num, task);
    let target_stats = compute_native_target_stats(&frame_y, &y_num, task);
    let high_table_descriptors = compute_table_descriptors(&xs, n, p, &y_num, Level::High);
    let transform = FitOnlyTableTransform {
        task,
        kinds,
        numeric,
        categorical,
        target,
        min_level_count,
        prior_smoothing,
    };
    Ok(PreparedTable {
        task,
        rows: n,
        features: p,
        classes: classes_of(&transform.target),
        x_scaled: xs,
        y_scaled: y_num,
        transform,
        row_ids,
        categories,
        category_tokens,
        frame_columns,
        frame_y,
        target_stats,
        observed_target_stats,
        high_table_descriptors,
        raw_cells_sha256,
        source_digest_override: None,
    })
}

/// Python `FitOnlyTableTransform.fit(X: list of str rows, y: list of str, task).transform(X, y)`
/// (kind inference + Bayesian-shrunk categorical encoding) with explicit physical `row_ids`.
pub fn prepare_table_cells_with_ids(
    x: &[Vec<String>],
    y: &[String],
    task: PrepTask,
    row_ids: Vec<u64>,
) -> Result<PreparedTable> {
    let yv: Vec<TargetValue<'_>> = y.iter().map(|s| TargetValue::Str(s.as_str())).collect();
    prepare_cells_with_yv(x, &yv, None, task, row_ids)
}

/// Mixed-column table preparation for typed/cell consumers: accepts string feature
/// cells `x_cells`, numeric target `y`, optional explicit per-column `declared_kinds`
/// (`None` infers kinds from `x_cells`), `task`, and optional physical `row_ids`
/// (`None` uses sequential `0..n as u64`).
pub fn prepare_table_mixed(
    x_cells: &[Vec<String>],
    y: &[f64],
    declared_kinds: Option<&[ColumnKind]>,
    task: PrepTask,
    row_ids: Option<Vec<u64>>,
) -> Result<PreparedTable> {
    let yv: Vec<TargetValue<'_>> = y.iter().map(|&v| TargetValue::Num(v)).collect();
    let row_ids = row_ids.unwrap_or_else(|| (0..x_cells.len() as u64).collect());
    prepare_cells_with_yv(x_cells, &yv, declared_kinds, task, row_ids)
}

/// Python `FitOnlyTableTransform.fit(X: list of str rows, y: list of str, task).transform(X, y)`
/// (kind inference + Bayesian-shrunk categorical encoding) using sequential `0..n as u64` row IDs.
pub fn prepare_table_cells(
    x: &[Vec<String>],
    y: &[String],
    task: PrepTask,
) -> Result<PreparedTable> {
    let row_ids = (0..x.len() as u64).collect();
    prepare_table_cells_with_ids(x, y, task, row_ids)
}

/// Python `str.strip()` (whitespace per `str.isspace` for ASCII + Unicode).
fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

#[inline]
fn is_missing_token(s: &str) -> bool {
    matches!(s.len(), 0..=4)
        && (s.is_empty()
            || s == "?"
            || s.eq_ignore_ascii_case("na")
            || s.eq_ignore_ascii_case("nan")
            || s.eq_ignore_ascii_case("null")
            || s.eq_ignore_ascii_case("none"))
}

/// Python `float(s)` on an already-stripped string. Python additionally
/// accepts `_` digit separators; these are not supported here.
#[inline]
fn py_float(s: &str) -> Option<f64> {
    if s.is_empty() {
        return None;
    }
    let b = s.as_bytes();
    let body = match b[0] {
        b'+' | b'-' => &s[1..],
        _ => s,
    };
    if body.is_empty() || matches!(body.as_bytes()[0], b'+' | b'-') {
        return None; // empty after sign or more than one sign
    }
    if !(body.eq_ignore_ascii_case("inf")
        || body.eq_ignore_ascii_case("infinity")
        || body.eq_ignore_ascii_case("nan")
        || body
            .bytes()
            .all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-')))
    {
        return None;
    }
    s.parse::<f64>().ok()
}

/// `scaling._parse_float` for a string cell.
#[inline]
pub fn parse_float_cell(cell: &str) -> f64 {
    let s = py_strip(cell);
    if is_missing_token(s) {
        return f64::NAN;
    }
    py_float(s).unwrap_or(f64::NAN)
}

fn regression_array(y: &[TargetValue<'_>]) -> Result<Vec<f64>> {
    // np.asarray(y, dtype=float64): numbers pass through, strings use float().
    y.iter()
        .map(|v| match v {
            TargetValue::Num(x) => Ok(*x),
            TargetValue::Str(s) => {
                py_float(py_strip(s)).ok_or_else(|| err("prepare_target_not_numeric"))
            }
        })
        .collect()
}

fn infer_feature_kinds_and_parse(
    rows: &[Vec<String>],
    p: usize,
) -> (Vec<ColumnKind>, Vec<Option<Vec<f64>>>) {
    let n = rows.len();
    let mut kinds = Vec::with_capacity(p);
    let mut parsed_cols = Vec::with_capacity(p);
    for j in 0..p {
        let mut col = Vec::with_capacity(n);
        let mut is_numeric = true;
        for r in rows {
            let s = py_strip(&r[j]);
            if is_missing_token(s) {
                col.push(f64::NAN);
                continue;
            }
            if let Some(v) = py_float(s) {
                col.push(v);
            } else {
                is_numeric = false;
                break;
            }
        }
        if is_numeric {
            kinds.push(ColumnKind::Numeric);
            parsed_cols.push(Some(col));
        } else {
            kinds.push(ColumnKind::Categorical);
            parsed_cols.push(None);
        }
    }
    (kinds, parsed_cols)
}

fn safe_skew_kurt(x: &[f64]) -> (f64, f64) {
    if x.len() < 4 {
        return (0.0, 0.0);
    }
    let m = mean1d(x);
    let s = std1d(x);
    if !s.is_finite() || s <= 1e-12 {
        return (0.0, 0.0);
    }
    let z3: Vec<f64> = x.iter().map(|&v| ((v - m) / s).powf(3.0)).collect();
    let z4: Vec<f64> = x.iter().map(|&v| ((v - m) / s).powf(4.0)).collect();
    let skew = mean1d(&z3);
    let kurt = mean1d(&z4) - 3.0;
    let skew = if skew.is_finite() {
        clip(skew, -50.0, 50.0)
    } else {
        0.0
    };
    let kurt = if kurt.is_finite() {
        clip(kurt, -50.0, 200.0)
    } else {
        0.0
    };
    (skew, kurt)
}

/// Python built-in `max(a, b, ...)` (first maximal argument wins).
fn py_max(vals: &[f64]) -> f64 {
    let mut m = vals[0];
    for &v in &vals[1..] {
        if v > m {
            m = v;
        }
    }
    m
}

fn fit_numeric_column(slot: usize, col: &[f64]) -> NumericColumnScaler {
    let finite: Vec<f64> = col.iter().copied().filter(|v| v.is_finite()).collect();
    let miss_rate = 1.0 - (finite.len() as f64 / col.len().max(1) as f64);
    if finite.is_empty() {
        return NumericColumnScaler {
            slot,
            transform_kind: TransformKind::Constant,
            missing_rate: 1.0,
            center: 0.0,
            scale: 1.0,
            post_mean: 0.0,
            post_std: 1.0,
            quantiles: [0.0; 9],
            skewness: 0.0,
            kurtosis: 0.0,
            blend_quantile_weight: 0.35,
        };
    }
    let s = sorted(&finite);
    let mut uniq = s.clone();
    uniq.dedup();
    let q_vals: [f64; 9] = QUANTILE_GRID.map(|q| super::np::quantile_sorted(&s, q));
    let (skew, kurt) = safe_skew_kurt(&finite);
    if uniq.len() <= 1 {
        return NumericColumnScaler {
            slot,
            transform_kind: TransformKind::Constant,
            missing_rate: miss_rate,
            center: uniq[0],
            scale: 1.0,
            post_mean: 0.0,
            post_std: 1.0,
            quantiles: q_vals,
            skewness: 0.0,
            kurtosis: 0.0,
            blend_quantile_weight: 0.35,
        };
    }
    let std_f = std1d(&finite);
    if uniq.len() == 2 {
        return NumericColumnScaler {
            slot,
            transform_kind: TransformKind::Binary,
            missing_rate: miss_rate,
            center: mean1d(&finite),
            scale: std_f.max(1e-6),
            post_mean: 0.0,
            post_std: 1.0,
            quantiles: q_vals,
            skewness: skew,
            kurtosis: kurt,
            blend_quantile_weight: 0.0,
        };
    }
    let med = q_vals[4];
    let iqr = (q_vals[5] - q_vals[3]).max(0.0);
    let span90 = (q_vals[7] - q_vals[1]).max(0.0);
    let tail_ratio = if iqr > 1e-9 {
        span90 / iqr.max(1e-9)
    } else {
        1.0
    };
    let k = s.len();
    let split = s.partition_point(|&v| v < med);
    let mut abs_sorted = Vec::with_capacity(k);
    let mut left = split;
    let mut right = split;
    while left > 0 && right < k {
        let dl = (s[left - 1] - med).abs();
        let dr = (s[right] - med).abs();
        if dl <= dr {
            abs_sorted.push(dl);
            left -= 1;
        } else {
            abs_sorted.push(dr);
            right += 1;
        }
    }
    while left > 0 {
        abs_sorted.push((s[left - 1] - med).abs());
        left -= 1;
    }
    while right < k {
        abs_sorted.push((s[right] - med).abs());
        right += 1;
    }
    let mad = if k % 2 == 1 {
        abs_sorted[k / 2]
    } else {
        (abs_sorted[k / 2 - 1] + abs_sorted[k / 2]) / 2.0
    };
    let robust_scale = py_max(&[iqr / 1.3489795, mad * 1.4826, std_f * 0.25, 1e-6]);
    if skew.abs() > 1.0 || kurt > 4.0 || tail_ratio > 4.2 {
        let logged: Vec<f64> = finite
            .iter()
            .map(|&v| {
                let sh = v - med;
                np_sign(sh) * (sh.abs() / robust_scale).ln_1p()
            })
            .collect();
        return NumericColumnScaler {
            slot,
            transform_kind: TransformKind::SignedLog1pHybrid,
            missing_rate: miss_rate,
            center: med,
            scale: robust_scale,
            post_mean: mean1d(&logged),
            post_std: std1d(&logged).max(1e-6),
            quantiles: q_vals,
            skewness: skew,
            kurtosis: kurt,
            blend_quantile_weight: 0.35,
        };
    }
    NumericColumnScaler {
        slot,
        transform_kind: TransformKind::QuantileRobustHybrid,
        missing_rate: miss_rate,
        center: med,
        scale: robust_scale,
        post_mean: mean1d(&finite),
        post_std: std_f.max(1e-6),
        quantiles: q_vals,
        skewness: skew,
        kurtosis: kurt,
        blend_quantile_weight: 0.30,
    }
}

fn fit_categorical_column(
    slot: usize,
    col: &[&str],
    y_num: &[f64],
    y_global_mean: f64,
    min_level_count: usize,
    prior_smoothing: f64,
) -> CategoricalColumnEncoder {
    let n = col.len().max(1);
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut miss_count = 0usize;
    let mut cleaned: Vec<&str> = Vec::with_capacity(col.len());
    for val in col {
        let s = py_strip(val);
        if is_missing_token(s) {
            miss_count += 1;
            cleaned.push("");
        } else {
            *counts.entry(s.to_string()).or_insert(0) += 1;
            cleaned.push(s);
        }
    }
    let mut retained: Vec<String> = counts
        .iter()
        .filter(|(_, &c)| c >= min_level_count)
        .map(|(t, _)| t.clone())
        .collect();
    retained.sort_by(|a, b| counts[b].cmp(&counts[a]).then_with(|| a.cmp(b)));
    retained.truncate(127);
    let mut vocab = vec!["__other__".to_string()];
    vocab.extend(retained.iter().cloned());
    let token_to_id: HashMap<String, usize> = retained
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i + 1))
        .collect();
    let v = vocab.len();
    let mut id_counts = vec![0.0f64; v];
    let mut id_y_sums = vec![0.0f64; v];
    for (i, s) in cleaned.iter().enumerate() {
        let idx = token_to_id.get(*s).copied().unwrap_or(0);
        id_counts[idx] += 1.0;
        id_y_sums[idx] += y_num[i];
    }
    let lin = linspace(-0.25, 0.25, v);
    let combined: Vec<f64> = (0..v)
        .map(|k| {
            let shrunk =
                (id_y_sums[k] + prior_smoothing * y_global_mean) / (id_counts[k] + prior_smoothing);
            shrunk + 0.15 * lin[k]
        })
        .collect();
    let freqs: Vec<f64> = id_counts.iter().map(|c| c / n as f64).collect();
    let train_enc: Vec<f64> = cleaned
        .iter()
        .map(|s| combined[token_to_id.get(*s).copied().unwrap_or(0)])
        .collect();
    CategoricalColumnEncoder {
        slot,
        missing_rate: miss_count as f64 / n as f64,
        vocab_tokens: vocab,
        token_to_id,
        id_target_encoding: combined,
        id_frequencies: freqs,
        post_mean: mean1d(&train_enc),
        post_std: std1d(&train_enc).max(1e-6),
    }
}

/// Python `repr(float)` for the classification positive-label heuristic.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // Shortest round-trip digits via LowerExp: "d.ddde±x".
    let e = format!("{x:e}");
    let (mant, exp) = e.split_once('e').expect("exp");
    let exp: i32 = exp.parse().expect("exp int");
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if neg { "-" } else { "" };
    if !(-4..16).contains(&exp) {
        let m = if digits.len() == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let es = if exp < 0 { "-" } else { "+" };
        return format!("{sign}{m}e{es}{:02}", exp.abs());
    }
    let nd = digits.len() as i32;
    let s = if exp >= 0 {
        if nd <= exp + 1 {
            format!("{}{}.0", digits, "0".repeat((exp + 1 - nd) as usize))
        } else {
            let k = (exp + 1) as usize;
            format!("{}.{}", &digits[..k], &digits[k..])
        }
    } else {
        format!("0.{}{}", "0".repeat((-exp - 1) as usize), digits)
    };
    format!("{sign}{s}")
}

fn fit_target_scaler(y: &[TargetValue<'_>], task: PrepTask) -> Result<TargetScaler> {
    match task {
        PrepTask::Multiclass => {
            let raw: Vec<f64> = y
                .iter()
                .map(|v| match v {
                    TargetValue::Num(x) => *x,
                    TargetValue::Str(s) => parse_float_cell(s),
                })
                .collect();
            let finite: Vec<f64> = raw.into_iter().filter(|v| v.is_finite()).collect();
            if finite.is_empty() {
                return Ok(TargetScaler {
                    task,
                    mean: 0.0,
                    std: 1.0,
                    median: 0.0,
                    iqr: 1.0,
                    skewness: 0.0,
                    kurtosis: 0.0,
                    positive_label: None,
                    class_prior: 0.5,
                    class_labels: vec![0.0, 0.5, 1.0],
                    class_priors: vec![0.34, 0.33, 0.33],
                });
            }
            let rounded = sorted(&finite.iter().map(|&v| round6(v)).collect::<Vec<_>>());
            let mut u_vals: Vec<f64> = Vec::new();
            let mut u_counts: Vec<i64> = Vec::new();
            for v in rounded {
                if u_vals.last() == Some(&v) {
                    *u_counts.last_mut().expect("count") += 1;
                } else {
                    u_vals.push(v);
                    u_counts.push(1);
                }
            }
            let counts_f: Vec<f64> = if u_vals.len() > 32 {
                let order = argsort_i64(&u_counts);
                let top: Vec<f64> = order.iter().rev().take(32).map(|&i| u_vals[i]).collect();
                u_vals = sorted(&top);
                u_vals
                    .iter()
                    .map(|&v| finite.iter().filter(|&&f| (f - v).abs() <= 1e-5).count() as f64)
                    .collect()
            } else {
                u_counts.iter().map(|&c| c as f64).collect()
            };
            let total = pairwise_sum(&counts_f).max(1.0);
            let priors: Vec<f64> = counts_f.iter().map(|c| c / total).collect();
            Ok(TargetScaler {
                task,
                mean: mean1d(&finite),
                std: std1d(&finite).max(1e-6),
                median: median(&finite),
                iqr: 1.0,
                skewness: 0.0,
                kurtosis: 0.0,
                positive_label: None,
                class_prior: priors[0],
                class_labels: u_vals,
                class_priors: priors,
            })
        }
        PrepTask::Classification => {
            let tokens: Vec<String> = y
                .iter()
                .map(|v| match v {
                    TargetValue::Num(x) => py_float_repr(*x),
                    TargetValue::Str(s) => py_strip(s).to_string(),
                })
                .collect();
            let mut uniq = tokens.clone();
            uniq.sort();
            uniq.dedup();
            let pos = if uniq.iter().any(|u| u == "1" || u == "1.0") {
                if uniq.iter().any(|u| u == "1") {
                    "1".to_string()
                } else {
                    "1.0".to_string()
                }
            } else if let Some(t) = uniq.iter().find(|u| u.to_lowercase() == "true") {
                t.clone()
            } else {
                uniq.last().cloned().unwrap_or_else(|| "1".to_string())
            };
            let bin: Vec<f64> = tokens
                .iter()
                .map(|t| if *t == pos { 1.0 } else { 0.0 })
                .collect();
            let prior = if bin.is_empty() { 0.5 } else { mean1d(&bin) };
            Ok(TargetScaler {
                task,
                mean: prior,
                std: (prior * (1.0 - prior)).sqrt().max(1e-6),
                median: if prior >= 0.5 { 1.0 } else { 0.0 },
                iqr: 1.0,
                skewness: 0.0,
                kurtosis: 0.0,
                positive_label: Some(pos),
                class_prior: prior,
                class_labels: Vec::new(),
                class_priors: Vec::new(),
            })
        }
        PrepTask::Regression => {
            let arr = regression_array(y)?;
            let finite: Vec<f64> = arr.into_iter().filter(|v| v.is_finite()).collect();
            if finite.is_empty() {
                return Ok(TargetScaler {
                    task,
                    mean: 0.0,
                    std: 1.0,
                    median: 0.0,
                    iqr: 1.0,
                    skewness: 0.0,
                    kurtosis: 0.0,
                    positive_label: None,
                    class_prior: 0.5,
                    class_labels: Vec::new(),
                    class_priors: Vec::new(),
                });
            }
            let m = mean1d(&finite);
            let s = std1d(&finite).max(1e-6);
            let q = quantiles(&finite, &[0.25, 0.5, 0.75]);
            let (skew, kurt) = safe_skew_kurt(&finite);
            Ok(TargetScaler {
                task,
                mean: m,
                std: s,
                median: q[1],
                iqr: (q[2] - q[0]).max(1e-6),
                skewness: skew,
                kurtosis: kurt,
                positive_label: None,
                class_prior: 0.5,
                class_labels: Vec::new(),
                class_priors: Vec::new(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_repr_matches_python() {
        assert_eq!(py_float_repr(1.0), "1.0");
        assert_eq!(py_float_repr(0.1), "0.1");
        assert_eq!(py_float_repr(1e16), "1e+16");
        assert_eq!(py_float_repr(1234.5), "1234.5");
        assert_eq!(py_float_repr(0.0001), "0.0001");
        assert_eq!(py_float_repr(0.00001), "1e-05");
        assert_eq!(py_float_repr(-2.5e-7), "-2.5e-07");
        assert_eq!(py_float_repr(123456789012345.0), "123456789012345.0");
    }

    #[test]
    fn parse_float_semantics() {
        assert!(parse_float_cell(" NA ").is_nan());
        assert_eq!(parse_float_cell(" 1e3 "), 1000.0);
        assert!(parse_float_cell("abc").is_nan());
        assert_eq!(parse_float_cell("-inf"), f64::NEG_INFINITY);
    }
}

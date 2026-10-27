//! Provenance/scope, mode/group enums, and deterministic stratified row
//! subsampling for `LandmarkBundle` v1.

use super::bundle::{bad, LANDMARK_SLOTS};
use super::{LandmarkOptions, PANEL};
use crate::{schema::Task, DopeError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::str::FromStr;

/// Provenance/scope attached to a bundle by its producer.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct BundleScope {
    /// `None` = full training partition.
    pub subsample_n: Option<usize>,
    /// Merge scope; normally [`LandmarkScope::digest`].
    pub source_digest: String,
    pub provenance: Option<String>,
}

impl BundleScope {
    pub fn new(source_digest: impl Into<String>) -> Self {
        Self {
            subsample_n: None,
            source_digest: source_digest.into(),
            provenance: None,
        }
    }
    pub fn with_subsample(mut self, subsample_n: Option<usize>) -> Self {
        self.subsample_n = subsample_n;
        self
    }
    pub fn with_provenance(mut self, provenance: impl Into<String>) -> Self {
        self.provenance = Some(provenance.into());
        self
    }
}

/// sha256 hex of `b"dope.landmark-options/v1\0" || LandmarkOptions JSON` (field order).
pub fn options_digest(options: &LandmarkOptions) -> String {
    let mut h = Sha256::new();
    h.update(b"dope.landmark-options/v1\0");
    h.update(serde_json::to_vec(options).expect("options serialize"));
    format!("{:x}", h.finalize())
}

/// Source-row scope for callers that hold the source bytes: sha256 hex of
/// `b"dope.landmark-rows/v1\0" || source_sha256 (ascii hex) || b"\0" ||
/// u64 LE count || ascending u64 LE physical row ids` (ids of the FULL training
/// partition, before any subsample).
pub fn source_rows_digest(source_sha256: &str, row_ids: &[u64]) -> String {
    let mut ids = row_ids.to_vec();
    ids.sort_unstable();
    let mut h = Sha256::new();
    h.update(b"dope.landmark-rows/v1\0");
    h.update(source_sha256.as_bytes());
    h.update(b"\0");
    h.update((ids.len() as u64).to_le_bytes());
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

/// Merge-compatibility scope of native landmark observations. Its
/// [`LandmarkScope::digest`] is a bundle's `source_digest`; bundles merge only
/// when panel, options, subsample (n AND seed) and source rows all agree.
/// Landmark ids/groups are deliberately NOT part of the scope.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LandmarkScope {
    pub panel: String,
    pub options_digest: String,
    pub subsample: Option<SubsampleSpec>,
    pub source_rows_digest: String,
}

impl LandmarkScope {
    pub fn new(
        options: &LandmarkOptions,
        subsample: Option<SubsampleSpec>,
        source_rows_digest: impl Into<String>,
    ) -> Self {
        Self {
            panel: options.panel.clone(),
            options_digest: options_digest(options),
            subsample,
            source_rows_digest: source_rows_digest.into(),
        }
    }

    /// Scope bound to an explicit pre-subsample training source digest (e.g.
    /// `PreparedTable::source_rows_digest()` or canonical training-cell digest).
    pub fn for_source_digest(
        options: &LandmarkOptions,
        subsample: Option<SubsampleSpec>,
        source_rows_digest: impl Into<String>,
    ) -> Self {
        Self::new(options, subsample, source_rows_digest)
    }

    /// Scope of a prepared (full, pre-subsample) training frame, bound by
    /// [`super::LandmarkFrame::content_digest`].
    pub fn for_frame(
        frame: &super::LandmarkFrame,
        options: &LandmarkOptions,
        subsample: Option<SubsampleSpec>,
    ) -> Self {
        Self::new(options, subsample, frame.content_digest())
    }

    /// sha256 hex of `b"dope.landmark-scope/v1\0" || panel || \0 || options_digest
    /// || \0 || ("none" | "n=<n>;seed=<seed>") || \0 || source_rows_digest`.
    pub fn digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"dope.landmark-scope/v1\0");
        h.update(self.panel.as_bytes());
        h.update(b"\0");
        h.update(self.options_digest.as_bytes());
        h.update(b"\0");
        match self.subsample {
            None => h.update(b"none"),
            Some(s) => h.update(format!("n={};seed={}", s.n, s.seed).as_bytes()),
        }
        h.update(b"\0");
        h.update(self.source_rows_digest.as_bytes());
        format!("{:x}", h.finalize())
    }

    /// `BundleScope` with `source_digest = digest()`, `subsample_n` and the
    /// default provenance (`native-v1` | `subsample-<n>-stratified`).
    pub fn bundle_scope(&self) -> BundleScope {
        BundleScope::new(self.digest())
            .with_subsample(self.subsample.map(|s| s.n))
            .with_provenance(default_provenance(self.subsample.map(|s| s.n)))
    }
}

pub(crate) fn default_provenance(subsample_n: Option<usize>) -> String {
    match subsample_n {
        None => "native-v1".to_string(),
        Some(n) => format!("subsample-{n}-stratified"),
    }
}

/// Landmark computation mode for optional/progressive conditioning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LandmarkMode {
    /// Skip native CV, spectrum and building blocks.
    None,
    /// Pure alias of `none` (no CV, spectrum or building blocks).
    Fast,
    /// zero → +macro → +larger → +fine (default subsample 300).
    Progressive,
    /// Current behaviour (unsampled unless a subsample is requested).
    Full,
}

impl LandmarkMode {
    pub const ALL: [Self; 4] = [Self::None, Self::Fast, Self::Progressive, Self::Full];
    /// `none` and `fast` (alias) run no native landmark work at all.
    pub fn is_landmark_free(&self) -> bool {
        matches!(self, Self::None | Self::Fast)
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Fast => "fast",
            Self::Progressive => "progressive",
            Self::Full => "full",
        }
    }
}

impl FromStr for LandmarkMode {
    type Err = DopeError;
    fn from_str(s: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|m| m.as_str() == s)
            .ok_or_else(|| bad("landmark mode must be none|fast|progressive|full"))
    }
}

impl std::fmt::Display for LandmarkMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Progressive landmark groups = `research/levels/descriptors.py` tiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LandmarkGroup {
    Macro,
    Larger,
    Fine,
}

/// descriptors.MACRO_INDICES (0-based slots).
pub const MACRO_SLOTS: [usize; 9] = [0, 1, 2, 3, 4, 5, 13, 14, 21];
/// descriptors.LARGER_ADDED_INDICES.
pub const LARGER_SLOTS: [usize; 9] = [6, 7, 8, 9, 10, 11, 12, 15, 16];
/// descriptors.FINE_ADDED_INDICES.
pub const FINE_SLOTS: [usize; 5] = [17, 18, 19, 20, 22];
/// Progressive computation order.
pub const PROGRESSIVE_ORDER: [LandmarkGroup; 3] = [
    LandmarkGroup::Macro,
    LandmarkGroup::Larger,
    LandmarkGroup::Fine,
];

impl LandmarkGroup {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Macro => "macro",
            Self::Larger => "larger",
            Self::Fine => "fine",
        }
    }
    /// 0-based panel slots.
    pub fn slots(&self) -> &'static [usize] {
        match self {
            Self::Macro => &MACRO_SLOTS,
            Self::Larger => &LARGER_SLOTS,
            Self::Fine => &FINE_SLOTS,
        }
    }
    /// 1-based landmark ids (slot + 1), ascending.
    pub fn ids(&self) -> Vec<u8> {
        let mut ids: Vec<u8> = self.slots().iter().map(|s| *s as u8 + 1).collect();
        ids.sort_unstable();
        ids
    }
}

impl FromStr for LandmarkGroup {
    type Err = DopeError;
    fn from_str(s: &str) -> Result<Self> {
        PROGRESSIVE_ORDER
            .into_iter()
            .find(|g| g.as_str() == s)
            .ok_or_else(|| bad("landmark group must be macro|larger|fine"))
    }
}

/// Parse `macro|larger|fine|<id 1..23>|<panel name>` comma lists into sorted,
/// de-duplicated 1-based ids. Empty selections are rejected.
pub fn parse_landmark_ids(spec: &str) -> Result<Vec<u8>> {
    let mut ids = Vec::new();
    for token in spec.split(',').map(str::trim) {
        if let Ok(group) = token.parse::<LandmarkGroup>() {
            ids.extend(group.ids());
        } else if let Ok(id) = token.parse::<u8>() {
            if !(1..=LANDMARK_SLOTS as u8).contains(&id) {
                return Err(bad("landmark id must be in 1..=23"));
            }
            ids.push(id);
        } else if let Some(r) = PANEL.iter().find(|r| r.name == token) {
            ids.push(r.id.0);
        } else {
            return Err(bad("unknown landmark id, name or group"));
        }
    }
    ids.sort_unstable();
    ids.dedup();
    if ids.is_empty() {
        return Err(bad("empty landmark selection"));
    }
    Ok(ids)
}

/// Deterministic stratified row subsample, applied to the frame BEFORE a
/// signature and kept outside `LandmarkOptions`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubsampleSpec {
    pub n: usize,
    pub seed: u64,
}

/// Regression y-quantile bins.
const REGRESSION_BINS: usize = 10;

/// Deterministic stratified row subsample (mirrors `optional_landmarks.stratified_subsample`).
pub fn stratified_subsample(y: &[f64], task: Task, n: usize, seed: u64) -> Result<Vec<usize>> {
    SubsampleSpec::new(n, seed).stratified_indices(y, task)
}

impl SubsampleSpec {
    /// Default progressive subsample.
    pub const PROGRESSIVE_DEFAULT: Self = Self { n: 300, seed: 0 };

    pub fn new(n: usize, seed: u64) -> Self {
        Self { n, seed }
    }

    /// Sorted row positions into `y` (keys = positions). See [`Self::select`].
    pub fn stratified_indices(&self, y: &[f64], task: Task) -> Result<Vec<usize>> {
        Ok(self.select(y, task, None)?.positions)
    }

    /// Deterministic stratified selection.
    pub fn select(&self, y: &[f64], task: Task, row_ids: Option<&[u64]>) -> Result<RowSelection> {
        let total = y.len();
        if self.n == 0 {
            return Err(bad("subsample size must be positive"));
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(bad("subsample target must be finite"));
        }
        let keys: Vec<u64> = match row_ids {
            Some(ids) => {
                let mut sorted = ids.to_vec();
                sorted.sort_unstable();
                sorted.dedup();
                if ids.len() != total || sorted.len() != total {
                    return Err(bad("subsample row ids must be unique, one per row"));
                }
                ids.to_vec()
            }
            None => (0..total as u64).collect(),
        };
        let positions = if self.n >= total {
            (0..total).collect()
        } else {
            self.stratified_positions(y, task, &keys)?
        };
        let row_ids: Vec<u64> = positions.iter().map(|&p| keys[p]).collect();
        Ok(RowSelection {
            spec: *self,
            digest: row_selection_digest(self, &row_ids),
            positions,
            row_ids,
        })
    }

    fn stratified_positions(&self, y: &[f64], task: Task, keys: &[u64]) -> Result<Vec<usize>> {
        let total = y.len();
        let strata: Vec<Vec<usize>> = match task {
            Task::Regression => {
                let mut order: Vec<usize> = (0..total).collect();
                order.sort_by(|&a, &b| y[a].total_cmp(&y[b]).then(keys[a].cmp(&keys[b])));
                let bins = REGRESSION_BINS.min(self.n);
                let q = total / bins;
                let r = total % bins;
                (0..bins)
                    .map(|b| {
                        let start = b * q + b.min(r);
                        let end = (b + 1) * q + (b + 1).min(r);
                        order[start..end].to_vec()
                    })
                    .collect()
            }
            Task::Classification => {
                let mut labels: Vec<f64> = y.to_vec();
                labels.sort_by(f64::total_cmp);
                labels.dedup_by(|a, b| a.total_cmp(b).is_eq());
                let mut strata = vec![Vec::new(); labels.len()];
                for (i, v) in y.iter().enumerate() {
                    let k = labels
                        .binary_search_by(|l| l.total_cmp(v))
                        .expect("label present");
                    strata[k].push(i);
                }
                if strata.iter().any(|s| s.len() < 2) || self.n < 2 * strata.len() {
                    return Err(bad(
                        "subsample too small for stratification (at least 2 rows per class)",
                    ));
                }
                strata
            }
        };
        let base: Vec<usize> = match task {
            Task::Regression => vec![0; strata.len()],
            Task::Classification => vec![2; strata.len()],
        };
        let capacity: Vec<usize> = strata.iter().zip(&base).map(|(s, b)| s.len() - b).collect();
        let extra = largest_remainder(self.n - base.iter().sum::<usize>(), &capacity);
        let seed_mix = crate::rng::splitmix64(self.seed);
        let mut out = Vec::with_capacity(self.n);
        for (k, stratum) in strata.iter().enumerate() {
            let mut keyed: Vec<(u64, u64, usize)> = stratum
                .iter()
                .map(|&i| (crate::rng::splitmix64(seed_mix ^ keys[i]), keys[i], i))
                .collect();
            keyed.sort_unstable();
            out.extend(keyed.iter().take(base[k] + extra[k]).map(|p| p.2));
        }
        out.sort_unstable();
        Ok(out)
    }
}

/// Result of [`SubsampleSpec::select`]: ascending positions, the matching
/// physical row ids (in position order) and a digest over the ids.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowSelection {
    pub spec: SubsampleSpec,
    pub positions: Vec<usize>,
    pub row_ids: Vec<u64>,
    pub digest: String,
}

/// sha256 hex of `b"dope.landmark-row-selection/v1\0" || u64 LE n || u64 LE seed
/// || u64 LE count || ascending u64 LE selected row ids`.
pub fn row_selection_digest(spec: &SubsampleSpec, row_ids: &[u64]) -> String {
    let mut ids = row_ids.to_vec();
    ids.sort_unstable();
    let mut h = Sha256::new();
    h.update(b"dope.landmark-row-selection/v1\0");
    h.update((spec.n as u64).to_le_bytes());
    h.update(spec.seed.to_le_bytes());
    h.update((ids.len() as u64).to_le_bytes());
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

/// Hamilton apportionment of `n` over `weights` (requires `n < sum(weights)`
/// or equality); result `[k] <= weights[k]`.
fn largest_remainder(n: usize, weights: &[usize]) -> Vec<usize> {
    let total: u128 = weights.iter().map(|w| *w as u128).sum();
    if total == 0 {
        return vec![0; weights.len()];
    }
    let mut quota: Vec<usize> = Vec::with_capacity(weights.len());
    let mut remainders: Vec<(u128, usize)> = Vec::with_capacity(weights.len());
    for (k, w) in weights.iter().enumerate() {
        let p = n as u128 * *w as u128;
        quota.push((p / total) as usize);
        remainders.push((p % total, k));
    }
    let left = n - quota.iter().sum::<usize>();
    // Descending remainder, ties → lower index.
    remainders.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    for &(_, k) in remainders.iter().take(left) {
        quota[k] += 1;
    }
    quota
}

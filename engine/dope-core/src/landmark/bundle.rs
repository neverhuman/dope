//! LandmarkBundle v1: the shared optional/progressive landmark contract.
//!
//! A bundle carries the 23 `landmarks-native-v1` coordinates as SQUASHED f32
//! gains (`s = g / (1 + |g|)`, computed in f32 exactly like the strict
//! `learned::landmark_features` loader) plus an explicit presence mask. Missing
//! slots are exactly `+0.0`. Status is always derived from the mask after
//! sanitation. Nothing here changes the strict full-panel paths: a full bundle
//! built from a native report reproduces the strict 46-d features bit-for-bit.
//!
//! Python mirror: `research/levels` (same JSON form, same digest bytes).

use super::{LandmarkOptions, PANEL, PANEL_V1};
use crate::{schema::Task, DopeError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::str::FromStr;

/// Number of coordinates in the `landmarks-native-v1` panel.
pub const LANDMARK_SLOTS: usize = 23;
/// Domain separator of [`LandmarkBundle::conditioning_digest`].
pub const BUNDLE_DIGEST_DOMAIN: &[u8] = b"dope.landmark-bundle/v1\0";
/// `source_digest` of the zero bundle.
pub const ZERO_SOURCE_DIGEST: &str = "zero";
/// Strict v1 report view: only `complete` cells are observations.
pub const REPORT_RECIPE_COMPLETE: &str = crate::learned::landmark_features::RECIPE;
/// Observed v2 report view: `complete` and `uninformative` cells are observations
/// (also what `research/levels/descriptors.py` treats as present).
pub const REPORT_RECIPE_OBSERVED: &str = crate::learned::landmark_features::OBSERVED_RECIPE;

fn bad(message: &str) -> DopeError {
    DopeError::Input(message.into())
}

/// Squash a raw f64 gain into `[-1, 1]` f32, or `None` if `gain` is non-finite.
///
/// Normal finite gains (`|gain| <= f32::MAX`) use strict `f32` arithmetic
/// (`let g = gain as f32; g / (1. + g.abs())`) for bit-identity with
/// `learned::landmark_features`. Finite `|gain| > f32::MAX` (e.g. `1e300`)
/// squashes in `f64` first so huge finite observations stay present.
#[inline]
pub fn squash_raw_gain(gain: f64) -> Option<f32> {
    if !gain.is_finite() {
        return None;
    }
    if gain.abs() > f32::MAX as f64 {
        Some((gain / (1.0 + gain.abs())) as f32)
    } else {
        let g = gain as f32;
        Some(g / (1. + g.abs()))
    }
}

/// Squash a raw f64 gain (returns `+0.0` if non-finite).
#[inline]
pub fn squash_gain(gain: f64) -> f32 {
    squash_raw_gain(gain).unwrap_or(0.0)
}

/// Derived coverage. JSON: `"none"` | `{"partial": n}` | `"full"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LandmarkStatus {
    None,
    Partial(u8),
    Full,
}

impl LandmarkStatus {
    pub fn from_mask(mask: &[bool; LANDMARK_SLOTS]) -> Self {
        match mask.iter().filter(|m| **m).count() {
            0 => Self::None,
            LANDMARK_SLOTS => Self::Full,
            n => Self::Partial(n as u8),
        }
    }
    /// Number of present coordinates.
    pub fn present(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Partial(n) => *n as usize,
            Self::Full => LANDMARK_SLOTS,
        }
    }
}

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

/// Shared optional-landmark contract (v1). See module docs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "BundleWire")]
pub struct LandmarkBundle {
    pub values: [f32; LANDMARK_SLOTS],
    pub mask: [bool; LANDMARK_SLOTS],
    pub status: LandmarkStatus,
    pub subsample_n: Option<usize>,
    pub source_digest: String,
    pub provenance: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleWire {
    values: [f32; LANDMARK_SLOTS],
    mask: [bool; LANDMARK_SLOTS],
    #[serde(default)]
    status: Option<LandmarkStatus>,
    subsample_n: Option<usize>,
    source_digest: String,
    provenance: Option<String>,
}

impl TryFrom<BundleWire> for LandmarkBundle {
    type Error = DopeError;
    fn try_from(w: BundleWire) -> Result<Self> {
        let _ = w.status;
        let b = Self::from_squashed(
            &w.values,
            &w.mask,
            BundleScope {
                subsample_n: w.subsample_n,
                source_digest: w.source_digest,
                provenance: w.provenance,
            },
        );
        b.validate()?;
        Ok(b)
    }
}

impl LandmarkBundle {
    /// The zero sentinel: nothing present; compatible with any scope on merge.
    pub fn zero() -> Self {
        Self {
            values: [0.; LANDMARK_SLOTS],
            mask: [false; LANDMARK_SLOTS],
            status: LandmarkStatus::None,
            subsample_n: None,
            source_digest: ZERO_SOURCE_DIGEST.into(),
            provenance: None,
        }
    }

    /// True only for the exact zero sentinel (`source_digest == "zero"`,
    /// `subsample_n == None`, and no observations). A scoped empty result is
    /// NOT a universal zero sentinel.
    pub fn is_zero(&self) -> bool {
        self.source_digest == ZERO_SOURCE_DIGEST
            && self.subsample_n.is_none()
            && !self.mask.iter().any(|m| *m)
            && self.values.iter().all(|v| v.to_bits() == 0)
    }

    /// Sanitize public fields (`values`, `mask`) and derive `status` from the
    /// sanitized mask, preserving scope fields.
    pub fn sanitized(&self) -> Self {
        Self::from_squashed(
            &self.values,
            &self.mask,
            BundleScope {
                subsample_n: self.subsample_n,
                source_digest: self.source_digest.clone(),
                provenance: self.provenance.clone(),
            },
        )
    }

    /// RAW gains (squashed here). Non-finite → `+0.0` / missing.
    pub fn from_raw_gains(gains: &[Option<f64>; LANDMARK_SLOTS], scope: BundleScope) -> Self {
        let mut values = [0f32; LANDMARK_SLOTS];
        let mut mask = [false; LANDMARK_SLOTS];
        for (i, g) in gains.iter().enumerate() {
            if let Some(s) = g.and_then(squash_raw_gain) {
                values[i] = s;
                mask[i] = true;
            }
        }
        Self::assemble(values, mask, scope)
    }

    /// Already-squashed values. Masked-out, non-finite or |v| > 1 → `+0.0` / missing.
    pub fn from_squashed(
        values: &[f32; LANDMARK_SLOTS],
        mask: &[bool; LANDMARK_SLOTS],
        scope: BundleScope,
    ) -> Self {
        let mut v = [0f32; LANDMARK_SLOTS];
        let mut m = [false; LANDMARK_SLOTS];
        for i in 0..LANDMARK_SLOTS {
            if mask[i] && values[i].is_finite() && values[i].abs() <= 1. {
                v[i] = values[i];
                m[i] = true;
            }
        }
        Self::assemble(v, m, scope)
    }

    fn assemble(
        values: [f32; LANDMARK_SLOTS],
        mask: [bool; LANDMARK_SLOTS],
        scope: BundleScope,
    ) -> Self {
        Self {
            values,
            mask,
            status: LandmarkStatus::from_mask(&mask),
            subsample_n: scope.subsample_n,
            source_digest: scope.source_digest,
            provenance: scope.provenance,
        }
    }

    /// Native private report (`Signature::private_report()` JSON or a saved
    /// `report.json`), optional/observed v2 view (`complete` | `uninformative`
    /// cells with finite gain, so finite measured zeros stay present).
    /// `source_digest` is derived from the full report payload (unbound,
    /// `provenance: None`); use [`Self::from_report_with_scope`] to bind a
    /// source table scope instead.
    pub fn from_report(report: &Value, subsample_n: Option<usize>) -> Result<Self> {
        Self::from_report_for_recipe(report, subsample_n, REPORT_RECIPE_OBSERVED)
    }

    /// Strict v1 report view (`complete` cells only; matches legacy v1
    /// `learned::landmark_features::load`).
    pub fn from_report_complete(report: &Value, subsample_n: Option<usize>) -> Result<Self> {
        Self::from_report_for_recipe(report, subsample_n, REPORT_RECIPE_COMPLETE)
    }

    /// Observed v2 view (`complete` | `uninformative` cells with finite gain).
    pub fn from_report_observed(report: &Value, subsample_n: Option<usize>) -> Result<Self> {
        Self::from_report_for_recipe(report, subsample_n, REPORT_RECIPE_OBSERVED)
    }

    pub fn from_report_for_recipe(
        report: &Value,
        subsample_n: Option<usize>,
        recipe: &str,
    ) -> Result<Self> {
        let digest = report_scope_digest(report)?;
        Self::from_report_with_scope(
            report,
            BundleScope::new(digest).with_subsample(subsample_n),
            recipe,
        )
    }

    pub fn from_report_with_scope(
        report: &Value,
        scope: BundleScope,
        recipe: &str,
    ) -> Result<Self> {
        if recipe != REPORT_RECIPE_COMPLETE && recipe != REPORT_RECIPE_OBSERVED {
            return Err(bad("unknown landmark report recipe"));
        }
        let cells = report["cells"]
            .as_array()
            .ok_or_else(|| bad("missing landmark report cells"))?;
        if report["panel"] != PANEL_V1 || cells.len() != LANDMARK_SLOTS {
            return Err(bad("landmark report panel mismatch"));
        }
        let mut values = [0f32; LANDMARK_SLOTS];
        let mut mask = [false; LANDMARK_SLOTS];
        for (i, cell) in cells.iter().enumerate() {
            if cell["id"] != i + 1 {
                return Err(bad("landmark report coordinate order mismatch"));
            }
            let observed = cell["state"] == "complete"
                || (recipe == REPORT_RECIPE_OBSERVED && cell["state"] == "uninformative");
            if observed && !cell["gain"].is_null() {
                let gain = cell["gain"]
                    .as_f64()
                    .ok_or_else(|| bad("invalid landmark report gain"))?;
                if let Some(s) = squash_raw_gain(gain) {
                    values[i] = s;
                    mask[i] = true;
                }
            }
        }
        Ok(Self::assemble(values, mask, scope))
    }

    /// Lattice merge. Incoming wins where present; mask = union; never downgrades.
    /// Both operands are sanitized and validated first. The exact zero sentinel is
    /// compatible with anything (and adopts the other side's scope); otherwise
    /// `subsample_n` and `source_digest` must match.
    pub fn merge(&self, incoming: &Self) -> Result<Self> {
        let a = self.sanitized();
        let b = incoming.sanitized();
        a.validate()?;
        b.validate()?;
        if b.is_zero() {
            return Ok(a);
        }
        if a.is_zero() {
            return Ok(b);
        }
        if a.subsample_n != b.subsample_n || a.source_digest != b.source_digest {
            return Err(bad("incompatible landmark bundle scope"));
        }
        let mut values = a.values;
        let mut mask = a.mask;
        for i in 0..LANDMARK_SLOTS {
            if b.mask[i] {
                values[i] = b.values[i];
                mask[i] = true;
            }
        }
        let provenance = match (&a.provenance, &b.provenance) {
            (pa, pb) if pa == pb => pa.clone(),
            (None, pb) => pb.clone(),
            (pa, None) => pa.clone(),
            (Some(pa), Some(pb)) => {
                let mut tokens: Vec<&str> = pa.split('+').chain(pb.split('+')).collect();
                tokens.sort_unstable();
                tokens.dedup();
                Some(tokens.join("+"))
            }
        };
        Ok(Self::assemble(
            values,
            mask,
            BundleScope {
                subsample_n: a.subsample_n,
                source_digest: a.source_digest,
                provenance,
            },
        ))
    }

    /// `cat(values, mask as 0/1)`: the 46-d strict-layout feature vector.
    pub fn features46(&self) -> [f32; 2 * LANDMARK_SLOTS] {
        let mut out = [0f32; 2 * LANDMARK_SLOTS];
        out[..LANDMARK_SLOTS].copy_from_slice(&self.values);
        for i in 0..LANDMARK_SLOTS {
            out[LANDMARK_SLOTS + i] = if self.mask[i] { 1. } else { 0. };
        }
        out
    }

    /// sha256 hex of `b"dope.landmark-bundle/v1\0" || b"landmarks-native-v1\0"
    /// || 23×f32 LE values || 23×u8 mask || u64 LE subsample_n (u64::MAX = None)`.
    pub fn conditioning_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(BUNDLE_DIGEST_DOMAIN);
        h.update(PANEL_V1.as_bytes());
        h.update(b"\0");
        for v in &self.values {
            h.update(v.to_le_bytes());
        }
        for m in &self.mask {
            h.update([u8::from(*m)]);
        }
        let n = self.subsample_n.map_or(u64::MAX, |n| n as u64);
        h.update(n.to_le_bytes());
        format!("{:x}", h.finalize())
    }

    /// Structural invariants (also enforced on deserialization).
    pub fn validate(&self) -> Result<()> {
        for i in 0..LANDMARK_SLOTS {
            let v = self.values[i];
            if !v.is_finite() || v.abs() > 1. {
                return Err(bad("invalid landmark bundle value"));
            }
            if !self.mask[i] && v.to_bits() != 0 {
                return Err(bad("missing landmark bundle slot must be +0.0"));
            }
        }
        if self.status != LandmarkStatus::from_mask(&self.mask) {
            return Err(bad("landmark bundle status does not match mask"));
        }
        if self.source_digest.is_empty() {
            return Err(bad("landmark bundle source digest required"));
        }
        if self.source_digest == ZERO_SOURCE_DIGEST {
            if self.mask.iter().any(|m| *m) {
                return Err(bad("zero landmark bundle cannot carry observations"));
            }
            if self.subsample_n.is_some() {
                return Err(bad("zero landmark bundle cannot carry subsample scope"));
            }
        }
        if self.subsample_n == Some(0) {
            return Err(bad("landmark bundle subsample must be positive"));
        }
        Ok(())
    }
}

/// Default unbound source digest for `from_report`: sha256 hex over the full
/// report (`panel`, `options`, `task`, `classes`, `rows`, `features`,
/// `target_stats`, `cells`, `contrasts`, `spectrum`), so different same-shaped
/// tables never collide.
fn report_scope_digest(report: &Value) -> Result<String> {
    let options: LandmarkOptions = serde_json::from_value(report["options"].clone())
        .map_err(|_| bad("invalid landmark report options"))?;
    let mut h = Sha256::new();
    h.update(b"dope.landmark-report-scope/v1\0");
    h.update(report["panel"].as_str().unwrap_or_default().as_bytes());
    h.update(b"\0");
    h.update(serde_json::to_vec(&options).expect("options serialize"));
    for key in [
        "task",
        "classes",
        "rows",
        "features",
        "target_stats",
        "cells",
        "contrasts",
        "spectrum",
    ] {
        h.update(b"\0");
        h.update(serde_json::to_vec(&report[key]).expect("report field serialize"));
    }
    Ok(format!("{:x}", h.finalize()))
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
    ///
    /// Each row has a key: its physical row id when `row_ids` is given (must be
    /// unique, same length as `y`), else its position. Selection depends only on
    /// `(y, key)` pairs, not on row order.
    /// * `n >= y.len()` → all rows.
    /// * regression: rows ordered by `(y.total_cmp, key)`, split into
    ///   `B = min(10, n)` contiguous rank bins (`np.array_split`: first `N % B`
    ///   bins one larger), quota per bin by largest remainder of `n*size_b/N`
    ///   (ties → lower bin).
    /// * classification: classes = distinct `y` (total order); error if any class
    ///   has `< 2` rows or `n < 2*K`; each class first takes `2`, the rest is
    ///   allocated by largest remainder over capacities `count - 2` (ties → lower class).
    /// * within a stratum: the `quota` rows with the smallest
    ///   `(splitmix64(splitmix64(seed) ^ key), key)`.
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

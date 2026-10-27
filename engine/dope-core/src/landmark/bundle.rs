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

use super::{LandmarkOptions, PANEL_V1};
use crate::{DopeError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub use super::bundle_scope::*;

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

pub(crate) fn bad(message: &str) -> DopeError {
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

//! Recipe selection, digest computation, and `DopeConditioning` assembly.

use sha2::{Digest, Sha256};

use crate::error::Result;

use super::building_blocks::compute_building_block_landmarks;
use super::conditioning::{
    apply_table_base_override, bad, level_landmark_mask, observed_target_stats, report_task,
    table_classes, COL_LANDMARK_DIM, CONDITIONING_DOMAIN, LEGACY_FULL_RECIPE, MAX_COLS,
    OBSERVED_RECIPE,
};
use super::conditioning_observed::{
    assemble_observed_conditioning_cached, assemble_v3_conditioning,
};
use super::inter_landmark::ReportLandmarkView;

/// `col_landmarks24[:p, :17]` zero-padded / truncated to `[64, 17]`, matching
/// `DopeEncoder629` / `ConditionalHyperTargetNet.compute_column_gates`.
pub fn pad_col_landmarks(col_landmarks24: &[[f32; 24]]) -> Vec<[f32; COL_LANDMARK_DIM]> {
    let mut out = vec![[0.0f32; COL_LANDMARK_DIM]; MAX_COLS];
    for (dst, src) in out.iter_mut().zip(col_landmarks24) {
        dst.copy_from_slice(&src[..COL_LANDMARK_DIM]);
    }
    out
}

/// Effective conditioning digest:
/// `sha256(b"dope.conditioning/v1\0" || recipe || 0x00 || level || 0x00 || vec256 f32 LE || col_landmarks 64×17 f32 LE)`,
/// lowercase hex (`optional_landmarks.effective_conditioning_digest`).
pub fn conditioning_digest(
    recipe: &str,
    level: &str,
    vec256: &[f32; 256],
    col_landmarks: &[[f32; COL_LANDMARK_DIM]],
) -> String {
    let mut h = Sha256::new();
    h.update(CONDITIONING_DOMAIN);
    h.update(recipe.as_bytes());
    h.update([0u8]);
    h.update(level.as_bytes());
    h.update([0u8]);
    for v in vec256 {
        h.update(v.to_le_bytes());
    }
    for row_idx in 0..MAX_COLS {
        let row = col_landmarks
            .get(row_idx)
            .copied()
            .unwrap_or([0.0f32; COL_LANDMARK_DIM]);
        for v in row {
            h.update(v.to_le_bytes());
        }
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Extract `(values23, mask23)` from native report cells using the strict
/// `LandmarkBundle::from_report` rule (`state == "complete"` and finite f32
/// gain, squashed in f32).
pub fn report_complete_landmarks<'a, I>(cells: I) -> ([f32; 23], [bool; 23])
where
    I: IntoIterator<Item = (i64, &'a str, Option<f64>)>,
{
    let mut values = [0.0f32; 23];
    let mut mask = [false; 23];
    for (id, state, gain) in cells {
        let idx = id - 1;
        if !(0..23).contains(&idx) || state != "complete" {
            continue;
        }
        if let Some(g64) = gain.filter(|g| g.is_finite()) {
            let g = g64 as f32;
            if g.is_finite() {
                let s = g / (1.0 + g.abs());
                if s.is_finite() {
                    values[idx as usize] = s + 0.0;
                    mask[idx as usize] = true;
                }
            }
        }
    }
    (values, mask)
}

/// Extract `(values23, mask23)` from native report cells using the optional
/// `REPORT_RECIPE_OBSERVED` rule (`state == "complete" || state == "uninformative"`
/// and finite f32 gain, squashed in f32, keeping finite `0.0` gains present).
pub fn report_observed_landmarks<'a, I>(cells: I) -> ([f32; 23], [bool; 23])
where
    I: IntoIterator<Item = (i64, &'a str, Option<f64>)>,
{
    let mut values = [0.0f32; 23];
    let mut mask = [false; 23];
    for (id, state, gain) in cells {
        let idx = id - 1;
        if !(0..23).contains(&idx) || (state != "complete" && state != "uninformative") {
            continue;
        }
        if let Some(g64) = gain.filter(|g| g.is_finite()) {
            let g = g64 as f32;
            if g.is_finite() {
                let s = g / (1.0 + g.abs());
                if s.is_finite() {
                    values[idx as usize] = s + 0.0;
                    mask[idx as usize] = true;
                }
            }
        }
    }
    (values, mask)
}

/// Bitwise f32 values + mask equality (`LandmarkBundle.same_landmarks`).
pub fn same_landmarks(
    values_a: &[f32; 23],
    mask_a: &[bool; 23],
    values_b: &[f32; 23],
    mask_b: &[bool; 23],
) -> bool {
    mask_a == mask_b
        && values_a
            .iter()
            .zip(values_b)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Gather subsampled rows `(sample_x_scaled, sample_rows, sample_y_scaled)`
/// from a complete `rows × features` row-major `x_scaled` and `y_scaled` slice
/// using 0-based `sample_positions` (e.g. `RowSelection.positions`), suitable
/// for [`LegacyReportConditioningInput::evidence_sample`].
pub fn subsample_scaled_rows(
    x_scaled: &[f64],
    rows: usize,
    features: usize,
    y_scaled: &[f64],
    sample_positions: &[usize],
) -> Result<(Vec<f64>, usize, Vec<f64>)> {
    if x_scaled.len() != rows * features || y_scaled.len() != rows {
        return Err(bad("prepare_subsample_shape_mismatch"));
    }
    let mut sx = Vec::with_capacity(sample_positions.len() * features);
    let mut sy = Vec::with_capacity(sample_positions.len());
    for &pos in sample_positions {
        if pos >= rows {
            return Err(bad("prepare_subsample_position_out_of_bounds"));
        }
        sx.extend_from_slice(&x_scaled[pos * features..(pos + 1) * features]);
        sy.push(y_scaled[pos]);
    }
    Ok((sx, sample_positions.len(), sy))
}

/// Bundle view consumed by [`build_conditioning_with_descriptors`].
#[derive(Debug, Clone, Copy)]
pub struct BundleConditioningInput<'a> {
    pub values: &'a [f32; 23],
    pub mask: &'a [bool; 23],
    pub subsample_n: Option<usize>,
}

/// Pre-extracted native report descriptors + complete/observed landmarks for
/// the `mode == "full"` (`legacy-full-v1`) path.
#[derive(Debug, Clone, Copy)]
pub struct LegacyReportConditioningInput<'a> {
    pub vector116: &'a [f32; 116],
    pub rep: ReportLandmarkView<'a>,
    pub column_prior_importance: &'a [f32],
    pub complete_values23: [f32; 23],
    pub complete_mask23: [bool; 23],
    pub observed_values23: [f32; 23],
    pub observed_mask23: [bool; 23],
    /// Optional full-table `target_stats` (`[Some(t0), Some(t1), None]`, e.g.
    /// from `PreparedTable.target_stats`) used to override `vector116[85..87]`
    /// when `mode == "full"` is run on a subsample (`subsample_n.is_some()`).
    pub full_table_target_stats: Option<[Option<f64>; 3]>,
    /// Optional subsampled `(x_scaled, sample_rows, y_scaled)` when the native
    /// report was computed on a subsample of size `N < rows`. When `Some`,
    /// building-block probes are computed on the evidence sample while
    /// `table_vec24`, `target_stats[85..87]` and `shape[87..91]` stay on the
    /// complete training table.
    pub evidence_sample: Option<(&'a [f64], usize, &'a [f64])>,
}

impl<'a> LegacyReportConditioningInput<'a> {
    /// True when `(values, mask)` matches either the strict `complete`-only
    /// view or the optional `observed` (`complete | uninformative`) view of
    /// the native report.
    pub fn matches_bundle(&self, values: &[f32; 23], mask: &[bool; 23]) -> bool {
        same_landmarks(values, mask, &self.complete_values23, &self.complete_mask23)
            || same_landmarks(values, mask, &self.observed_values23, &self.observed_mask23)
    }
}

/// Assembled MLC-v3 / `DopeEncoder629` conditioning output and effective digest.
#[derive(Debug, Clone, PartialEq)]
pub struct DopeConditioning {
    /// `"legacy-full-v1"` (`mode == "full"`) or `"observed-only-v1"` (`"none"|"fast"|"progressive"`).
    pub recipe: String,
    /// `"low" | "small" | "medium" | "high" | "ultra"`.
    pub level: String,
    /// `"none" | "fast" | "progressive" | "full"`.
    pub mode: String,
    /// Sanitized [`crate::landmark::LandmarkBundle`] carrying `source_digest`, `subsample_n`, `values`, `mask`, and `status`.
    pub bundle: crate::landmark::LandmarkBundle,
    /// 256-dim conditioning vector.
    pub vec256: [f32; 256],
    /// `col_landmarks24[:64, :17]` zero-padded to 64 rows.
    pub col_landmarks: Vec<[f32; COL_LANDMARK_DIM]>,
    /// Full `(p, 24)` column landmark matrix.
    pub col_landmarks24: Vec<[f32; 24]>,
    /// 48-dim inter-landmark summary vector.
    pub inter_landmark_vec48: [f32; 48],
    /// 68-dim building-block vector (zeros on `observed-only-v1`).
    pub building_block_vec68: [f32; 68],
    /// Normalised multi-primitive column driver weights.
    pub v3_driver_weights: Vec<f32>,
}

impl DopeConditioning {
    /// Effective conditioning digest:
    /// `sha256(b"dope.conditioning/v1\0" || recipe || 0x00 || level || 0x00 || vec256 f32 LE || col_landmarks 64×17 f32 LE)`.
    pub fn digest(&self) -> String {
        conditioning_digest(&self.recipe, &self.level, &self.vec256, &self.col_landmarks)
    }
}

/// Select and run the V3 conditioning recipe for an optional-landmark `mode`
/// (`multilevel_model.build_conditioning`).
#[allow(clippy::too_many_arguments)]
pub fn build_conditioning_with_descriptors(
    x_scaled: &[f64],
    rows: usize,
    features: usize,
    y_scaled: &[f64],
    task: &str,
    table_vec24: &[f32; 24],
    column_desc: &[[f32; 8]],
    empirical_driver_weights: &[f32],
    bundle: Option<BundleConditioningInput<'_>>,
    raw_report: Option<LegacyReportConditioningInput<'_>>,
    mode: &str,
    level: &str,
) -> Result<DopeConditioning> {
    build_conditioning_with_full_table_stats(
        x_scaled,
        rows,
        features,
        y_scaled,
        task,
        None,
        table_vec24,
        column_desc,
        empirical_driver_weights,
        bundle,
        raw_report,
        mode,
        level,
    )
}

/// Like [`build_conditioning_with_descriptors`], additionally accepting
/// `raw_target_stats: Option<[f64; 2]>` (e.g. `Some(prepared_table.target_stats)`)
/// so `none`/`fast`/`progressive` and sampled `full` all preserve the complete
/// training table's native `target_stats` (including regression raw-target
/// skewness and excess kurtosis) in `vec256[85..87]`.
#[allow(clippy::too_many_arguments)]
pub fn build_conditioning_with_full_table_stats(
    x_scaled: &[f64],
    rows: usize,
    features: usize,
    y_scaled: &[f64],
    task: &str,
    raw_target_stats: Option<[f64; 2]>,
    table_vec24: &[f32; 24],
    column_desc: &[[f32; 8]],
    empirical_driver_weights: &[f32],
    bundle: Option<BundleConditioningInput<'_>>,
    raw_report: Option<LegacyReportConditioningInput<'_>>,
    mode: &str,
    level: &str,
) -> Result<DopeConditioning> {
    build_conditioning_with_cached_table_stats(
        x_scaled,
        rows,
        features,
        y_scaled,
        task,
        None,
        raw_target_stats,
        table_vec24,
        column_desc,
        empirical_driver_weights,
        bundle,
        raw_report,
        mode,
        level,
    )
}

/// Like [`build_conditioning_with_full_table_stats`], additionally accepting
/// `classes_override: Option<usize>` (e.g. `Some(prepared_table.classes)`) so
/// progressive updates on multiclass tables reuse cached `PreparedTable.classes`
/// without calling `table_classes(y_scaled, task)`.
#[allow(clippy::too_many_arguments)]
pub fn build_conditioning_with_cached_table_stats(
    x_scaled: &[f64],
    rows: usize,
    features: usize,
    y_scaled: &[f64],
    task: &str,
    classes_override: Option<usize>,
    raw_target_stats: Option<[f64; 2]>,
    table_vec24: &[f32; 24],
    column_desc: &[[f32; 8]],
    empirical_driver_weights: &[f32],
    bundle: Option<BundleConditioningInput<'_>>,
    raw_report: Option<LegacyReportConditioningInput<'_>>,
    mode: &str,
    level: &str,
) -> Result<DopeConditioning> {
    let _ = level_landmark_mask(level)?;
    let (recipe, v3) = match mode {
        "full" => {
            let raw = raw_report.ok_or_else(|| bad("landmark_full_requires_native_report"))?;
            if let Some(b) = bundle {
                if !raw.matches_bundle(b.values, b.mask) {
                    return Err(bad("landmark_bundle_report_mismatch"));
                }
            }
            let mut vec116 = *raw.vector116;
            let is_sampled =
                bundle.and_then(|b| b.subsample_n).is_some() || raw.evidence_sample.is_some();
            if is_sampled {
                let classes = match classes_override {
                    Some(c) => {
                        let _ = report_task(task)?;
                        c
                    }
                    None => table_classes(y_scaled, task)?,
                };
                let t_stats = match raw.full_table_target_stats {
                    Some(ts) => ts,
                    None => match raw_target_stats {
                        Some([t0, t1]) => [Some(t0), Some(t1), None],
                        None => observed_target_stats(y_scaled, task)?,
                    },
                };
                apply_table_base_override(&mut vec116, rows, features, classes, task, &t_stats)?;
            }
            let (bb_x, bb_rows, bb_y) = raw.evidence_sample.unwrap_or((x_scaled, rows, y_scaled));
            let bb = compute_building_block_landmarks(bb_x, bb_rows, features, bb_y, level);
            let v3 = assemble_v3_conditioning(
                &vec116,
                table_vec24,
                column_desc,
                empirical_driver_weights,
                &raw.rep,
                raw.column_prior_importance,
                &bb,
                level,
            );
            (LEGACY_FULL_RECIPE, v3)
        }
        "none" | "fast" => {
            if let Some(b) = bundle {
                if b.mask.iter().any(|&m| m) {
                    return Err(bad("landmark_mode_none_rejects_bundle"));
                }
            }
            let zero_vals = [0.0f32; 23];
            let zero_mask = [false; 23];
            let v3 = assemble_observed_conditioning_cached(
                &zero_vals,
                &zero_mask,
                rows,
                features,
                y_scaled,
                task,
                classes_override,
                raw_target_stats,
                table_vec24,
                column_desc,
                empirical_driver_weights,
                level,
            )?;
            (OBSERVED_RECIPE, v3)
        }
        "progressive" => {
            let zero_vals = [0.0f32; 23];
            let zero_mask = [false; 23];
            let (vals, mask) = match bundle {
                Some(b) => (b.values, b.mask),
                None => (&zero_vals, &zero_mask),
            };
            let v3 = assemble_observed_conditioning_cached(
                vals,
                mask,
                rows,
                features,
                y_scaled,
                task,
                classes_override,
                raw_target_stats,
                table_vec24,
                column_desc,
                empirical_driver_weights,
                level,
            )?;
            (OBSERVED_RECIPE, v3)
        }
        _ => return Err(bad(format!("unknown_landmark_mode_{mode}"))),
    };
    let out_bundle = match bundle {
        Some(b) => {
            if b.subsample_n.is_none() && b.mask.iter().all(|&m| !m) {
                crate::landmark::LandmarkBundle::zero()
            } else {
                crate::landmark::LandmarkBundle::from_squashed(
                    b.values,
                    b.mask,
                    crate::landmark::BundleScope::new("unbound").with_subsample(b.subsample_n),
                )
            }
        }
        None => match raw_report {
            Some(raw) => crate::landmark::LandmarkBundle::from_squashed(
                &raw.observed_values23,
                &raw.observed_mask23,
                crate::landmark::BundleScope::new("unbound"),
            ),
            None => crate::landmark::LandmarkBundle::zero(),
        },
    };
    let col_landmarks = pad_col_landmarks(&v3.col_landmarks24);
    Ok(DopeConditioning {
        recipe: recipe.to_string(),
        level: level.to_string(),
        mode: mode.to_string(),
        bundle: out_bundle,
        vec256: v3.vec256,
        col_landmarks,
        col_landmarks24: v3.col_landmarks24,
        inter_landmark_vec48: v3.inter_landmark_vec48,
        building_block_vec68: v3.building_block_vec68,
        v3_driver_weights: v3.v3_driver_weights,
    })
}

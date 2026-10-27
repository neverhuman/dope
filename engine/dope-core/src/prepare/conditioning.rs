//! Rust port of the MLC-v3 conditioning assembly (`build_v3_conditioning_bundle`,
//! `build_observed_conditioning_bundle`, `build_conditioning`, and
//! `effective_conditioning_digest`) from
//! `research/levels/{building_block_landmarks,optional_landmarks,multilevel_model}.py`.
//!
//! Supports both conditioning recipes:
//! - [`LEGACY_FULL_RECIPE`] (`"legacy-full-v1"`): `mode == "full"`. Requires the
//!   native report; runs the exact legacy descriptors + building blocks +
//!   inter-landmark summary stats. If a landmark bundle is also supplied,
//!   verifies bitwise f32 values and mask match `from_report(native_report)`.
//! - [`OBSERVED_RECIPE`] (`"observed-only-v1"`): `mode` in
//!   `{"none", "fast", "progressive"}`. Selected strictly by `mode` (never by
//!   `mask.all()`). In `"none"`/`"fast"`, rejects any bundle with present slots.
//!   Uses only measured landmark gains from the bundle and train-only table
//!   descriptors (`bb68 = 0`, `col_spec9 = 0`, `col_bb7 = 0`, observed-only
//!   `vector116` and `inter48`).

use crate::error::{DopeError, Result};

use super::building_blocks::{np, squash};

pub use super::conditioning_builder::*;
pub use super::conditioning_observed::*;

/// Legacy full-report conditioning recipe (`mode == "full"`).
pub const LEGACY_FULL_RECIPE: &str = "legacy-full-v1";
/// Observed-only conditioning recipe (`mode` in `{"none", "fast", "progressive"}`).
pub const OBSERVED_RECIPE: &str = "observed-only-v1";
/// Domain separator for [`conditioning_digest`] / [`DopeConditioning::digest`].
pub const CONDITIONING_DOMAIN: &[u8] = b"dope.conditioning/v1\0";
/// Column slots consumed by the MLC-v3 nets (`MAX_COLS`).
pub const MAX_COLS: usize = 64;
/// Per-column landmark width fed to the column-gate head.
pub const COL_LANDMARK_DIM: usize = 17;
/// `f32::MAX` widened to `f64`, matching Python `F32_MAX`.
pub const F32_MAX_F64: f64 = 3.4028234663852886e38;

/// `descriptors.MACRO_INDICES` (0-based slots).
pub const MACRO_INDICES: [usize; 9] = [0, 1, 2, 3, 4, 5, 13, 14, 21];
/// `descriptors.LARGER_ADDED_INDICES` (0-based slots).
pub const LARGER_ADDED_INDICES: [usize; 9] = [6, 7, 8, 9, 10, 11, 12, 15, 16];
/// `descriptors.FINE_ADDED_INDICES` (0-based slots).
pub const FINE_ADDED_INDICES: [usize; 5] = [17, 18, 19, 20, 22];

pub(crate) fn bad(msg: impl Into<String>) -> DopeError {
    DopeError::Input(msg.into())
}

pub(crate) fn finite32(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Float64 inverse of the f32 squash (`optional_landmarks.unsquash`).
///
/// `|s| >= 1` maps to `±F32_MAX_F64`; otherwise `s / (1 - |s|)`.
pub fn unsquash(value: f64) -> f64 {
    if value.abs() >= 1.0 {
        F32_MAX_F64.copysign(value)
    } else {
        value / (1.0 - value.abs())
    }
}

/// `descriptors.level_landmark_mask(level)`.
pub fn level_landmark_mask(level: &str) -> Result<[bool; 23]> {
    let mut mask = [false; 23];
    match level {
        "low" | "small" => {
            for &i in &MACRO_INDICES[..6] {
                mask[i] = true;
            }
        }
        "medium" => {
            for &i in &MACRO_INDICES {
                mask[i] = true;
            }
        }
        "high" => {
            for &i in MACRO_INDICES.iter().chain(&LARGER_ADDED_INDICES) {
                mask[i] = true;
            }
        }
        "ultra" => {
            mask = [true; 23];
        }
        _ => return Err(bad(format!("unknown_level_{level}"))),
    }
    Ok(mask)
}

/// `descriptors.compute_level_contrasts(gains, mask, level=level)`.
pub fn compute_level_contrasts(
    gains: &[f64; 23],
    mask: &[bool; 23],
    level: &str,
) -> Result<[f64; 16]> {
    let lvl_mask = level_landmark_mask(level)?;
    let mut allowed = [false; 23];
    let mut g = [f64::NAN; 23];
    for i in 0..23 {
        allowed[i] = lvl_mask[i] && mask[i];
        if allowed[i] {
            g[i] = gains[i];
        }
    }
    let get = |idx: usize| -> f64 {
        if allowed[idx] && g[idx].is_finite() {
            g[idx]
        } else {
            f64::NAN
        }
    };
    let maxf = |idxs: &[usize]| -> f64 {
        let mut iter = idxs
            .iter()
            .filter(|&&i| allowed[i] && g[i].is_finite())
            .map(|&i| g[i]);
        match iter.next() {
            Some(first) => iter.fold(first, np::py_max),
            None => f64::NAN,
        }
    };
    let sub = |a: f64, b: f64, denom: f64| -> f64 {
        if a.is_finite() && b.is_finite() && denom >= 0.01 {
            np::clip((a - b) / denom, -1.0, 1.0)
        } else {
            0.0
        }
    };

    let all_idxs: [usize; 23] = std::array::from_fn(|i| i);
    let s = maxf(&all_idxs);
    if !s.is_finite() || s < 0.01 {
        return Ok([0.0; 16]);
    }

    let lin = get(0);
    let add_macro = maxf(&[0, 1, 2, 3, 4, 5, 14]);
    let add_all = maxf(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14]);
    let mono_excess = sub(maxf(&[1, 2, 4]), lin, s);
    let sym_excess = sub(maxf(&[3, 5, 9]), lin, s);
    let best1_share = if get(14).is_finite() {
        np::clip(get(14) / s, -1.0, 1.0)
    } else {
        0.0
    };
    let miss_share = if get(13).is_finite() {
        np::clip(get(13) / s, -1.0, 1.0)
    } else {
        0.0
    };
    let lin_lc = sub(lin, get(21), s);
    let hinge_step = sub(get(7), get(6), s);
    let periodic_ex = sub(get(8), maxf(&[0, 4]), s);
    let pairs_ex = sub(get(11), lin, s);
    let rff_ex = sub(get(12), add_all, s);
    let otree_ex = sub(maxf(&[15, 16]), add_all, s);
    let nonadd_ex = sub(maxf(&[17, 12, 20]), add_all, s);
    let knn_ratio = if get(18).is_finite() && get(19).is_finite() && get(19).abs() > 1e-6 {
        np::clip(get(18) / np::py_max(get(19), 1e-6), -1.0, 1.0)
    } else {
        0.0
    };
    let gbdt_lc = sub(get(17), get(22), s);
    let net_vs_gbdt = sub(get(20), maxf(&[17, 12]), s);

    let out = [
        np::clip(s, -1.0, 1.0),
        if lin.is_finite() {
            np::clip(lin / s, -1.0, 1.0)
        } else {
            0.0
        },
        if add_macro.is_finite() {
            np::clip(add_macro / s, -1.0, 1.0)
        } else {
            0.0
        },
        mono_excess,
        sym_excess,
        best1_share,
        miss_share,
        lin_lc,
        hinge_step,
        periodic_ex,
        pairs_ex,
        rff_ex,
        otree_ex,
        nonadd_ex,
        knn_ratio,
        gbdt_lc + 0.5 * net_vs_gbdt,
    ];
    Ok(out.map(np::nan_to_zero))
}

/// Native report task string (`optional_landmarks.report_task`).
pub fn report_task(task: &str) -> Result<&'static str> {
    match task {
        "regression" => Ok("regression"),
        "classification" | "binary" | "multiclass" => Ok("classification"),
        _ => Err(bad(format!("unknown_task_{task}"))),
    }
}

/// Native `classes`: 0 for regression, 2 for binary/classification, unique
/// `rint(y)` count for multiclass (`optional_landmarks.table_classes`).
pub fn table_classes(y: &[f64], task: &str) -> Result<usize> {
    match task {
        "regression" => Ok(0),
        "classification" | "binary" => Ok(2),
        "multiclass" => {
            let mut labels: Vec<i64> = y.iter().map(|&v| v.round_ties_even() as i64).collect();
            labels.sort_unstable();
            labels.dedup();
            Ok(labels.len())
        }
        _ => Err(bad(format!("unknown_task_{task}"))),
    }
}

/// Native `target_stats` formula evaluated on `y` (`optional_landmarks.table_target_stats`).
pub fn table_target_stats(y: &[f64], task: &str) -> Result<[Option<f64>; 3]> {
    let _ = report_task(task)?;
    let n = y.len();
    if n == 0 {
        return Ok([None, None, None]);
    }
    if task == "regression" {
        let m = np::sum(y) / n as f64;
        let d2: Vec<f64> = y
            .iter()
            .map(|&v| {
                let d = v - m;
                d * d
            })
            .collect();
        let v = np::sum(&d2) / n as f64;
        if v <= 1e-24 {
            return Ok([None, None, None]);
        }
        let d3: Vec<f64> = y.iter().map(|&x| (x - m).powf(3.0)).collect();
        let d4: Vec<f64> = y.iter().map(|&x| (x - m).powf(4.0)).collect();
        return Ok([
            Some(np::sum(&d3) / n as f64 / v.powf(1.5)),
            Some(np::sum(&d4) / n as f64 / (v * v) - 3.0),
            None,
        ]);
    }
    let labels: Vec<i64> = y.iter().map(|&v| v.round_ties_even() as i64).collect();
    let classes: Vec<i64> = if task == "classification" || task == "binary" {
        vec![0, 1]
    } else {
        let mut c = labels.clone();
        c.sort_unstable();
        c.dedup();
        c
    };
    let mut counts = vec![0.0f64; classes.len()];
    let inv_n = 1.0 / n as f64;
    for &lab in &labels {
        if let Ok(slot) = classes.binary_search(&lab) {
            counts[slot] += inv_n;
        }
    }
    if counts.len() < 2 {
        return Ok([None, None, None]);
    }
    let pos: Vec<f64> = counts.iter().copied().filter(|&c| c > 0.0).collect();
    let terms: Vec<f64> = pos.iter().map(|&c| c * c.ln()).collect();
    let ent = -np::sum(&terms) / (counts.len() as f64).ln();
    Ok([Some(ent), Some(np::min(&counts)), None])
}

/// `optional_landmarks.observed_target_stats`: classification/binary/multiclass
/// uses the exact native label-entropy/min-share formula; regression returns
/// `[None, None, None]` (`0.0`) when only tanh-scaled `y_train` is available.
/// When full-table raw `y` (or `PreparedTable.target_stats`) is available, use
/// [`native_target_stats`] or pass `raw_target_stats: Some(table.target_stats)`.
pub fn observed_target_stats(y: &[f64], task: &str) -> Result<[Option<f64>; 3]> {
    let _ = report_task(task)?;
    if task == "regression" {
        Ok([None, None, None])
    } else {
        table_target_stats(y, task)
    }
}

/// Native `Signature.target_stats` formula evaluated on raw `frame_y` (for
/// regression: skewness and excess kurtosis when variance > `1e-24`) or
/// `y_scaled` (for classification/binary/multiclass: normalized entropy and
/// min class share). Matches `PreparedTable.target_stats`.
pub fn native_target_stats(
    raw_y: &[f64],
    y_scaled: &[f64],
    task: &str,
) -> Result<[Option<f64>; 3]> {
    let _ = report_task(task)?;
    if task == "regression" {
        let n = raw_y.len();
        if n == 0 {
            return Ok([None, None, None]);
        }
        let m = raw_y.iter().sum::<f64>() / n as f64;
        let v = raw_y.iter().map(|&val| (val - m).powi(2)).sum::<f64>() / n as f64;
        if !v.is_finite() || v <= 1e-24 {
            return Ok([None, None, None]);
        }
        let skew = raw_y.iter().map(|&val| (val - m).powi(3)).sum::<f64>() / n as f64 / v.powf(1.5);
        let kurt =
            raw_y.iter().map(|&val| (val - m).powi(4)).sum::<f64>() / n as f64 / v.powi(2) - 3.0;
        Ok([
            if skew.is_finite() { Some(skew) } else { None },
            if kurt.is_finite() { Some(kurt) } else { None },
            None,
        ])
    } else {
        table_target_stats(y_scaled, task)
    }
}

/// Compute `[t_stats[0], t_stats[1], is_clf, log10(rows)/6, log2(feats)/12, log10(feats/rows)/4]`
/// (`vector116[85..91]`) from full-table metadata.
pub fn table_base_target_and_shape(
    rows: usize,
    features: usize,
    classes: usize,
    task: &str,
    target_stats: &[Option<f64>; 3],
) -> Result<[f32; 6]> {
    let r_task = report_task(task)?;
    let t0 = target_stats[0].filter(|v| v.is_finite()).unwrap_or(0.0);
    let t1 = target_stats[1].filter(|v| v.is_finite()).unwrap_or(0.0);
    let rows_c = rows.max(1);
    let feats_c = features.max(1);
    let is_clf = if classes >= 2 || r_task == "classification" {
        1.0
    } else {
        0.0
    };
    Ok([
        squash(t0) as f32,
        squash(t1) as f32,
        is_clf as f32,
        ((rows_c as f64).log10() / 6.0) as f32,
        ((feats_c as f64).log2() / 12.0) as f32,
        ((feats_c as f64 / rows_c as f64).log10() / 4.0) as f32,
    ])
}

/// Override `vector116[85..91]` (`target_stats2` + `shape4`) in-place with
/// the complete `PreparedTable`'s base descriptors.
pub fn apply_table_base_override(
    vector116: &mut [f32; 116],
    rows: usize,
    features: usize,
    classes: usize,
    task: &str,
    target_stats: &[Option<f64>; 3],
) -> Result<()> {
    let base6 = table_base_target_and_shape(rows, features, classes, task, target_stats)?;
    vector116[85..91].copy_from_slice(&base6);
    Ok(())
}

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

use sha2::{Digest, Sha256};

use crate::error::{DopeError, Result};

use super::building_blocks::{
    compute_building_block_landmarks, np, squash, BuildingBlockDescriptors,
};
use super::inter_landmark::{
    compute_inter_landmark_summary_stats, compute_observed_inter_landmark_stats, ReportLandmarkView,
};

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

fn bad(msg: impl Into<String>) -> DopeError {
    DopeError::Input(msg.into())
}

fn finite32(v: f32) -> f32 {
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
///
/// Used when `mode == "full"` is evaluated with a subsampled native report
/// (`subsample_n.is_some()`), so `vec256[85..91]` reflects the full training
/// table rather than the subsample size `N`, while keeping all native landmark
/// evidence fields lossless. Unsampled `full` (`subsample_n.is_none()`) does
/// NOT call this function, preserving bit-identity with legacy conditioning.
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

/// Output of [`observed_report_descriptors`] (`optional_landmarks.observed_report_descriptors`).
#[derive(Debug, Clone, PartialEq)]
pub struct ObservedReportDescriptors {
    pub level: String,
    pub gains23: [f64; 23],
    pub ses23: [f64; 23],
    pub lambdas23: [f64; 23],
    pub mask23: [bool; 23],
    pub midranks23: [f64; 23],
    pub contrasts16: [f64; 16],
    pub target_stats2: [f64; 2],
    pub shape4: [f64; 4],
    pub spectrum_summary16: [f64; 16],
    pub group_summary9: [f64; 9],
    pub vector116: [f32; 116],
    pub column_spectrum: Vec<[f32; 9]>,
    pub column_prior_importance: Vec<f32>,
}

/// Observed-only replacement for `extract_report_descriptors`
/// (`optional_landmarks.observed_report_descriptors`).
#[allow(clippy::too_many_arguments)]
pub fn observed_report_descriptors(
    bundle_values23: &[f32; 23],
    bundle_mask23: &[bool; 23],
    rows: usize,
    features: usize,
    classes: usize,
    task: &str,
    target_stats: &[Option<f64>; 3],
    level: &str,
) -> Result<ObservedReportDescriptors> {
    let lvl_mask = level_landmark_mask(level)?;
    let r_task = report_task(task)?;
    let mut present = [false; 23];
    let mut s = [0.0f64; 23];
    let mut raw = [0.0f64; 23];
    for i in 0..23 {
        present[i] = bundle_mask23[i] && lvl_mask[i];
        if present[i] {
            s[i] = bundle_values23[i] as f64;
            raw[i] = unsquash(s[i]);
        }
    }

    let mut midranks = [0.0f64; 23];
    let valid: Vec<usize> = (0..23).filter(|&i| present[i]).collect();
    if valid.len() > 1 {
        for &i in &valid {
            let less = valid.iter().filter(|&&j| raw[j] < raw[i]).count() as f64;
            let eq = valid.iter().filter(|&&j| raw[j] == raw[i]).count() as f64;
            midranks[i] = (less + 0.5 * (eq - 1.0)) / (valid.len() - 1) as f64;
        }
    } else if valid.len() == 1 {
        midranks[valid[0]] = 0.5;
    }

    let contrasts = compute_level_contrasts(&raw, &present, level)?;

    let t0 = target_stats[0].filter(|v| v.is_finite()).unwrap_or(0.0);
    let t1 = target_stats[1].filter(|v| v.is_finite()).unwrap_or(0.0);
    let t_stats = [squash(t0), squash(t1)];

    let rows_c = rows.max(1);
    let feats_c = features.max(1);
    let is_clf = if classes >= 2 || r_task == "classification" {
        1.0
    } else {
        0.0
    };
    let shape_vec = [
        is_clf,
        (rows_c as f64).log10() / 6.0,
        (feats_c as f64).log2() / 12.0,
        (feats_c as f64 / rows_c as f64).log10() / 4.0,
    ];

    let gmax = |idxs: &[usize], gate: bool| -> f64 {
        if !gate {
            return 0.0;
        }
        let mut iter = idxs.iter().filter(|&&i| present[i]).map(|&i| s[i]);
        match iter.next() {
            Some(first) => iter.fold(first, np::py_max),
            None => 0.0,
        }
    };
    let gmean = |idxs: &[usize], gate: bool| -> f64 {
        if !gate {
            return 0.0;
        }
        let sel: Vec<f64> = idxs
            .iter()
            .filter(|&&i| present[i])
            .map(|&i| s[i])
            .collect();
        if sel.is_empty() {
            0.0
        } else {
            np::mean(&sel)
        }
    };

    let larger_gate = matches!(level, "high" | "ultra");
    let fine_gate = level == "ultra";
    let group = [
        gmax(&MACRO_INDICES, true),
        gmax(&LARGER_ADDED_INDICES, larger_gate),
        gmax(&FINE_ADDED_INDICES, fine_gate),
        gmean(&MACRO_INDICES, true),
        gmean(&LARGER_ADDED_INDICES, larger_gate),
        gmean(&FINE_ADDED_INDICES, fine_gate),
        0.0,
        0.0,
        0.0,
    ];
    let spec_summary = [0.0f64; 16];

    let mut vector116 = [0.0f32; 116];
    let mut k = 0;
    let mut push = |v: f64| {
        vector116[k] = v as f32;
        k += 1;
    };
    s.iter().for_each(|&v| push(v));
    present
        .iter()
        .for_each(|&b| push(if b { 1.0 } else { 0.0 }));
    midranks.iter().for_each(|&v| push(v));
    contrasts.iter().for_each(|&v| push(v));
    t_stats.iter().for_each(|&v| push(v));
    shape_vec.iter().for_each(|&v| push(v));
    spec_summary.iter().for_each(|&v| push(v));
    group.iter().for_each(|&v| push(v));

    Ok(ObservedReportDescriptors {
        level: level.to_string(),
        gains23: s,
        ses23: [0.0; 23],
        lambdas23: [0.0; 23],
        mask23: present,
        midranks23: midranks,
        contrasts16: contrasts,
        target_stats2: t_stats,
        shape4: shape_vec,
        spectrum_summary16: spec_summary,
        group_summary9: group,
        vector116,
        column_spectrum: vec![[0.0f32; 9]; feats_c],
        column_prior_importance: vec![(1.0 / feats_c as f64) as f32; feats_c],
    })
}

/// Output of [`assemble_v3_conditioning`] and [`assemble_observed_conditioning`].
#[derive(Debug, Clone, PartialEq)]
pub struct V3ConditioningBundle {
    /// `[vector116 | table_vec24 | inter_landmark_vec48 | building_block_vec68]`.
    pub vec256: [f32; 256],
    /// `inter_landmark_vec48`.
    pub inter_landmark_vec48: [f32; 48],
    /// `building_block_vec68` (zeros on `observed-only-v1`).
    pub building_block_vec68: [f32; 68],
    /// `(p, 24)` = `[col_spec9 | col_desc8 | col_bb7]`, nan_to_num'd.
    pub col_landmarks24: Vec<[f32; 24]>,
    /// Normalised multi-primitive column driver weights.
    pub v3_driver_weights: Vec<f32>,
}

/// Assemble the legacy (`legacy-full-v1`) conditioning bundle from
/// already-computed descriptor blocks (`build_v3_conditioning_bundle`).
#[allow(clippy::too_many_arguments)]
pub fn assemble_v3_conditioning(
    vector116: &[f32; 116],
    table_vec24: &[f32; 24],
    column_desc: &[[f32; 8]],
    empirical_driver_weights: &[f32],
    rep: &ReportLandmarkView,
    column_prior_importance: &[f32],
    bb: &BuildingBlockDescriptors,
    level: &str,
) -> V3ConditioningBundle {
    let inter = compute_inter_landmark_summary_stats(rep, bb, level);
    assemble_conditioning_blocks(
        vector116,
        table_vec24,
        column_desc,
        empirical_driver_weights,
        rep.column_spectrum,
        column_prior_importance,
        bb,
        inter,
    )
}

#[allow(clippy::too_many_arguments)]
fn assemble_conditioning_blocks(
    vector116: &[f32; 116],
    table_vec24: &[f32; 24],
    column_desc: &[[f32; 8]],
    empirical_driver_weights: &[f32],
    column_spectrum: &[[f32; 9]],
    column_prior_importance: &[f32],
    bb: &BuildingBlockDescriptors,
    inter: [f32; 48],
) -> V3ConditioningBundle {
    let mut vec256 = [0.0f32; 256];
    vec256[..116].copy_from_slice(vector116);
    vec256[116..140].copy_from_slice(table_vec24);
    vec256[140..188].copy_from_slice(&inter);
    vec256[188..].copy_from_slice(&bb.building_block_vec68);
    let vec256 = vec256.map(finite32);

    let p = column_desc.len();
    let take_p = p.min(column_spectrum.len());
    let col_landmarks24: Vec<[f32; 24]> = (0..p)
        .map(|j| {
            let mut row = [0.0f32; 24];
            if j < take_p {
                row[..9].copy_from_slice(&column_spectrum[j]);
            }
            row[9..17].copy_from_slice(&column_desc[j]);
            if let Some(b) = bb.col_bb_probes7.get(j) {
                row[17..].copy_from_slice(b);
            }
            row.map(finite32)
        })
        .collect();

    // v3 driver weights
    let emp_base: Vec<f64> = (0..p)
        .map(|j| empirical_driver_weights.get(j).copied().unwrap_or(0.0) as f64)
        .collect();
    let take_prior = p.min(column_prior_importance.len());
    let rep_p: Vec<f64> = (0..p)
        .map(|j| {
            if j < take_prior {
                column_prior_importance[j] as f64
            } else {
                0.0
            }
        })
        .collect();
    let bb_col_max: Vec<f64> = (0..p)
        .map(|j| match bb.col_bb_probes7.get(j) {
            Some(r) => np::max(&r.map(|v| (v as f64).abs())),
            None => 0.0,
        })
        .collect();
    let total = if p == 0 { 0.0 } else { np::sum(&bb_col_max) };
    let bb_col_w: Vec<f64> = if total > 1e-9 {
        bb_col_max.iter().map(|&v| v / total).collect()
    } else {
        vec![1.0 / p.max(1) as f64; p]
    };
    let v3: Vec<f64> = (0..p)
        .map(|j| 0.35 * emp_base[j] + 0.35 * rep_p[j] + 0.30 * bb_col_w[j])
        .collect();
    let den = np::py_max(if p == 0 { 0.0 } else { np::sum(&v3) }, 1e-9);
    let v3_driver_weights = v3.iter().map(|&v| (v / den) as f32).collect();

    V3ConditioningBundle {
        vec256,
        inter_landmark_vec48: inter,
        building_block_vec68: bb.building_block_vec68,
        col_landmarks24,
        v3_driver_weights,
    }
}

/// Assemble the observed-only (`observed-only-v1`) conditioning bundle from a
/// landmark bundle (`values23`, `mask23`) and already-computed table
/// descriptors (`optional_landmarks.build_observed_conditioning_bundle`).
///
/// Used for `mode` in `{"none", "fast", "progressive"}`. When only scaled `y`
/// is available, regression `target_stats` default to `0.0`; use
/// [`assemble_observed_conditioning_with_target_stats`] when full-table raw
/// `target_stats` (from `PreparedTable.target_stats`) are available.
#[allow(clippy::too_many_arguments)]
pub fn assemble_observed_conditioning(
    bundle_values23: &[f32; 23],
    bundle_mask23: &[bool; 23],
    rows: usize,
    features: usize,
    y_scaled: &[f64],
    task: &str,
    table_vec24: &[f32; 24],
    column_desc: &[[f32; 8]],
    empirical_driver_weights: &[f32],
    level: &str,
) -> Result<V3ConditioningBundle> {
    assemble_observed_conditioning_with_target_stats(
        bundle_values23,
        bundle_mask23,
        rows,
        features,
        y_scaled,
        task,
        None,
        table_vec24,
        column_desc,
        empirical_driver_weights,
        level,
    )
}

/// Like [`assemble_observed_conditioning`], but accepts an optional full-table
/// `raw_target_stats` (`PreparedTable.target_stats`) so regression retains its
/// true raw-target skewness and excess kurtosis in `vec256[85..87]`.
#[allow(clippy::too_many_arguments)]
pub fn assemble_observed_conditioning_with_target_stats(
    bundle_values23: &[f32; 23],
    bundle_mask23: &[bool; 23],
    rows: usize,
    features: usize,
    y_scaled: &[f64],
    task: &str,
    raw_target_stats: Option<[f64; 2]>,
    table_vec24: &[f32; 24],
    column_desc: &[[f32; 8]],
    empirical_driver_weights: &[f32],
    level: &str,
) -> Result<V3ConditioningBundle> {
    assemble_observed_conditioning_cached(
        bundle_values23,
        bundle_mask23,
        rows,
        features,
        y_scaled,
        task,
        None,
        raw_target_stats,
        table_vec24,
        column_desc,
        empirical_driver_weights,
        level,
    )
}

/// Like [`assemble_observed_conditioning_with_target_stats`], additionally
/// accepting `classes_override: Option<usize>` (e.g. `Some(prepared_table.classes)`)
/// so progressive updates on multiclass tables reuse cached `PreparedTable.classes`
/// without calling `table_classes(y_scaled, task)`.
#[allow(clippy::too_many_arguments)]
pub fn assemble_observed_conditioning_cached(
    bundle_values23: &[f32; 23],
    bundle_mask23: &[bool; 23],
    rows: usize,
    features: usize,
    y_scaled: &[f64],
    task: &str,
    classes_override: Option<usize>,
    raw_target_stats: Option<[f64; 2]>,
    table_vec24: &[f32; 24],
    column_desc: &[[f32; 8]],
    empirical_driver_weights: &[f32],
    level: &str,
) -> Result<V3ConditioningBundle> {
    let classes = match classes_override {
        Some(c) => {
            let _ = report_task(task)?;
            c
        }
        None => table_classes(y_scaled, task)?,
    };
    let target_stats = match raw_target_stats {
        Some([t0, t1]) => [Some(t0), Some(t1), None],
        None => observed_target_stats(y_scaled, task)?,
    };
    let rep_desc = observed_report_descriptors(
        bundle_values23,
        bundle_mask23,
        rows,
        features,
        classes,
        task,
        &target_stats,
        level,
    )?;
    let bb = BuildingBlockDescriptors::zeroed(features);
    let inter = compute_observed_inter_landmark_stats(&rep_desc.gains23, &rep_desc.mask23, level);
    Ok(assemble_conditioning_blocks(
        &rep_desc.vector116,
        table_vec24,
        column_desc,
        empirical_driver_weights,
        &rep_desc.column_spectrum,
        &rep_desc.column_prior_importance,
        &bb,
        inter,
    ))
}

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
///
/// - `mode == "full"`: `raw_report` REQUIRED; if `bundle` is supplied, validates
///   `same_landmarks(bundle, raw_report)` (else error
///   `landmark_bundle_report_mismatch`); runs `legacy-full-v1`. When
///   `bundle.subsample_n.is_some()`, overrides `vec256[85..91]` with the
///   complete table's `target_stats` and `shape` while preserving all native
///   landmark evidence losslessly. When `subsample_n.is_none()`, preserves the
///   legacy report `vector116` untouched (bit-identical).
/// - `mode` in `{"none", "fast"}`: `bundle` must be `None` or have no present
///   slots (else error `landmark_mode_none_rejects_bundle`); runs
///   `observed-only-v1` on the zero bundle.
/// - `mode == "progressive"`: runs `observed-only-v1` on `bundle` (`None` =
///   zero bundle), ignoring `raw_report`.
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

//! Observed-only report descriptors and V3 conditioning bundle assembly.

use crate::error::Result;

use super::building_blocks::{np, squash, BuildingBlockDescriptors};
use super::conditioning::{
    compute_level_contrasts, finite32, level_landmark_mask, observed_target_stats, report_task,
    table_classes, unsquash, FINE_ADDED_INDICES, LARGER_ADDED_INDICES, MACRO_INDICES,
};
use super::inter_landmark::{
    compute_inter_landmark_summary_stats, compute_observed_inter_landmark_stats, ReportLandmarkView,
};

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
pub(crate) fn assemble_conditioning_blocks(
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

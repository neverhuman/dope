//! Equivalence coverage (`kappa` for pred, lift, calibration, and feature drivers).

use super::bundle::{PredictionBundle, TaskKind};
use super::harm::{donor_clusters, SplitMix64};
use crate::dvs::{ensure, DvsError};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KappaThresholds {
    pub pred_kappa_star: f64,
    pub lift_kappa_star: f64,
    pub calib_kappa_star: f64,
    pub driver_kappa_star: f64,
}

impl Default for KappaThresholds {
    fn default() -> Self {
        Self {
            pred_kappa_star: 1.25,
            lift_kappa_star: 1.25,
            calib_kappa_star: 1.25,
            driver_kappa_star: 1.25,
        }
    }
}

impl KappaThresholds {
    /// Calibrate `kappa*` thresholds at empirical 95th percentile of disjoint real split `C`
    /// comparisons (`[1.0, 2.5]` guard; falls back to `1.25` when empty).
    pub fn calibrate_from_real_c(
        pred_c: &[f64],
        lift_c: &[f64],
        calib_c: &[f64],
        driver_c: &[f64],
    ) -> Self {
        let q95 = |xs: &[f64]| -> f64 {
            let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
            if v.is_empty() {
                return 1.25;
            }
            v.sort_by(f64::total_cmp);
            let idx = ((0.95 * (v.len() as f64 - 1.0)).ceil() as usize).min(v.len() - 1);
            v[idx].clamp(1.0, 2.5)
        };
        Self {
            pred_kappa_star: q95(pred_c),
            lift_kappa_star: q95(lift_c),
            calib_kappa_star: q95(calib_c),
            driver_kappa_star: q95(driver_c),
        }
    }
}

/// Optional driver & ALE summaries computed on frozen query groups for `A`, `B`, and `S`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DriverEquivalenceInput {
    /// Normalized non-negative grouped permutation importances (sum to 1) for A, optional B, and S.
    pub perm_a: Vec<f64>,
    pub perm_b: Option<Vec<f64>>,
    pub perm_s: Vec<f64>,
    /// Optional flattened ALE curve grids for A, B, and S (same length).
    pub ale_a: Option<Vec<f64>>,
    pub ale_b: Option<Vec<f64>>,
    pub ale_s: Option<Vec<f64>>,
    /// Optional drop-group coalition loss deltas for A, B, and S.
    pub coalition_a: Option<Vec<f64>>,
    pub coalition_b: Option<Vec<f64>>,
    pub coalition_s: Option<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MetricKappa {
    pub error: f64,
    pub calibration_scale: f64,
    pub floor: f64,
    pub kappa_point: f64,
    pub kappa_ucb90: f64,
    pub kappa_star: f64,
    pub pass: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct EquivalenceAssessment {
    pub pred: MetricKappa,
    pub lift: MetricKappa,
    pub calib: MetricKappa,
    pub driver: Option<MetricKappa>,
    pub equivalence_pass: bool,
}

/// Cumulative gains max decile gap between two prediction vectors (`lift`).
pub fn max_decile_lift_gap(y: &[f64], p_ref: &[f64], p_cmp: &[f64]) -> f64 {
    let n = y.len();
    if n < 10 {
        return 0.0;
    }
    let gains = |scores: &[f64]| -> [f64; 10] {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| scores[j].total_cmp(&scores[i]).then(i.cmp(&j)));
        let total: f64 = y.iter().sum();
        let mean = total / n as f64;
        let denom = y.iter().map(|&v| (v - mean).abs()).sum::<f64>().max(1e-12);
        let mut out = [0.0; 10];
        let mut acc = 0.0;
        for d in 0..10 {
            let start = (d * n) / 10;
            let end = ((d + 1) * n) / 10;
            for &idx in &order[start..end] {
                acc += y[idx] - mean;
            }
            out[d] = acc / denom;
        }
        out
    };
    let g_ref = gains(p_ref);
    let g_cmp = gains(p_cmp);
    (0..10)
        .map(|d| (g_ref[d] - g_cmp[d]).abs())
        .fold(0.0_f64, f64::max)
}

/// 15-bin ECE for binary/multiclass probabilities; log residual-variance ratio for regression.
pub fn calibration_error(bundle: &PredictionBundle, preds: &[f64]) -> f64 {
    let n = bundle.n_query();
    match bundle.task {
        TaskKind::Regression => {
            let var_true = variance(&bundle.y_true).max(1e-12);
            let res: Vec<f64> = (0..n).map(|i| bundle.y_true[i] - preds[i]).collect();
            let var_res = variance(&res).max(1e-12);
            (var_res / var_true).ln().abs()
        }
        TaskKind::Binary => ece_15(&bundle.y_true, preds),
        TaskKind::Multiclass => {
            let k = bundle.num_outputs;
            let mut max_p = Vec::with_capacity(n);
            let mut hit = Vec::with_capacity(n);
            for i in 0..n {
                let row = &preds[i * k..(i + 1) * k];
                let (best_c, &best_p) = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap_or((0, &0.0));
                max_p.push(best_p.clamp(0.0, 1.0));
                hit.push(if (bundle.y_true[i].round() as usize) == best_c {
                    1.0
                } else {
                    0.0
                });
            }
            ece_15(&hit, &max_p)
        }
    }
}

fn ece_15(y_bin: &[f64], prob: &[f64]) -> f64 {
    let n = y_bin.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum_y = [0.0; 15];
    let mut sum_p = [0.0; 15];
    let mut cnt = [0usize; 15];
    for (&y, &p) in y_bin.iter().zip(prob) {
        let p = p.clamp(0.0, 1.0);
        let b = ((p * 15.0).floor() as usize).min(14);
        sum_y[b] += y;
        sum_p[b] += p;
        cnt[b] += 1;
    }
    (0..15)
        .filter(|&b| cnt[b] > 0)
        .map(|b| (sum_y[b] - sum_p[b]).abs() / n as f64)
        .sum()
}

fn variance(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    xs.iter().map(|&v| (v - m) * (v - m)).sum::<f64>() / xs.len() as f64
}

fn scalar_score_view(bundle: &PredictionBundle, preds: &[f64]) -> Vec<f64> {
    if bundle.num_outputs == 1 {
        return preds.to_vec();
    }
    let n = bundle.n_query();
    let k = bundle.num_outputs;
    (0..n)
        .map(|i| {
            let c = (bundle.y_true[i].round() as usize).min(k - 1);
            preds[i * k + c]
        })
        .collect()
}

fn make_kappa_ratio(
    error: f64,
    calib_scale: f64,
    floor: f64,
    kappa_ucb90: f64,
    kappa_star: f64,
) -> MetricKappa {
    let denom = calib_scale.max(floor);
    let kappa_point = error / denom;
    let ucb = kappa_ucb90.max(kappa_point);
    MetricKappa {
        error,
        calibration_scale: calib_scale,
        floor,
        kappa_point,
        kappa_ucb90: ucb,
        kappa_star,
        pass: ucb <= kappa_star,
    }
}

fn normalized_tv(a: &[f64], b: &[f64]) -> Result<f64, DvsError> {
    ensure(
        !a.is_empty() && a.len() == b.len(),
        "driver_length_mismatch",
    )?;
    let sa: f64 = a.iter().map(|v| v.max(0.0)).sum();
    let sb: f64 = b.iter().map(|v| v.max(0.0)).sum();
    let tv = if sa <= 1e-12 && sb <= 1e-12 {
        0.0
    } else if sa <= 1e-12 || sb <= 1e-12 {
        1.0
    } else {
        0.5 * a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x.max(0.0) / sa - y.max(0.0) / sb).abs())
            .sum::<f64>()
    };
    Ok(tv)
}

fn mean_abs_diff(a: &[f64], b: &[f64]) -> Result<f64, DvsError> {
    ensure(
        !a.is_empty() && a.len() == b.len(),
        "driver_length_mismatch",
    )?;
    Ok(a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).sum::<f64>() / a.len() as f64)
}

/// Evaluate `kappa` across `pred`, `lift`, `calib`, and optional `driver`.
pub fn assess_equivalence(
    bundle: &PredictionBundle,
    driver: Option<&DriverEquivalenceInput>,
    thresholds: &KappaThresholds,
    draws: usize,
    seed: u64,
) -> Result<EquivalenceAssessment, DvsError> {
    bundle.validate()?;
    let n = bundle.n_query();
    let k = bundle.num_outputs;
    let y_scale = variance(&bundle.y_true).sqrt().max(1e-6);

    // 1. Prediction distance (`pred`) per row normalized by target scale (or probability L1).
    let row_pred_dist = |p1: &[f64], p2: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| {
                let d: f64 = (0..k)
                    .map(|c| (p1[i * k + c] - p2[i * k + c]).abs())
                    .sum::<f64>()
                    / k as f64;
                if bundle.task == TaskKind::Regression {
                    d / y_scale
                } else {
                    d
                }
            })
            .collect()
    };
    let pred_s_rows = row_pred_dist(&bundle.pred_a, &bundle.pred_s);
    let pred_b_rows = bundle
        .pred_b
        .as_deref()
        .map(|pb| row_pred_dist(&bundle.pred_a, pb));
    let pred_err = pred_s_rows.iter().sum::<f64>() / pred_s_rows.len() as f64;
    let pred_scale = pred_b_rows
        .as_deref()
        .map(|v| v.iter().sum::<f64>() / v.len() as f64)
        .unwrap_or(0.01);

    // 2. Lift (`lift`) + Calibration (`calib`) + Paired ratio bootstrap over donor clusters.
    let sa = scalar_score_view(bundle, &bundle.pred_a);
    let ss = scalar_score_view(bundle, &bundle.pred_s);
    let sb = bundle
        .pred_b
        .as_deref()
        .map(|pb| scalar_score_view(bundle, pb));
    let lift_err = max_decile_lift_gap(&bundle.y_true, &sa, &ss);
    let lift_scale = sb
        .as_deref()
        .map(|b| max_decile_lift_gap(&bundle.y_true, &sa, b))
        .unwrap_or(0.01);

    let calib_a = calibration_error(bundle, &bundle.pred_a);
    let calib_s = calibration_error(bundle, &bundle.pred_s);
    let calib_err = (calib_s - calib_a).abs();
    let calib_scale = bundle
        .pred_b
        .as_deref()
        .map(|pb| (calibration_error(bundle, pb) - calib_a).abs())
        .unwrap_or(0.01);

    let clusters = donor_clusters(n, bundle.donor_ids.as_deref());
    let m = clusters.len();
    let mut rng = SplitMix64::new(seed ^ 0x41);
    let mut pred_ratio_boot = Vec::with_capacity(draws);
    let mut lift_ratio_boot = Vec::with_capacity(draws);
    let mut calib_ratio_boot = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut idxs = Vec::with_capacity(n);
        for _ in 0..m {
            idxs.extend_from_slice(&clusters[rng.gen_range(m)]);
        }
        let ps_m = idxs.iter().map(|&i| pred_s_rows[i]).sum::<f64>() / idxs.len() as f64;
        let pb_m = pred_b_rows
            .as_deref()
            .map(|r| idxs.iter().map(|&i| r[i]).sum::<f64>() / idxs.len() as f64)
            .unwrap_or(0.01);
        pred_ratio_boot.push(ps_m / pb_m.max(0.01));

        let yb: Vec<f64> = idxs.iter().map(|&i| bundle.y_true[i]).collect();
        let sab: Vec<f64> = idxs.iter().map(|&i| sa[i]).collect();
        let ssb: Vec<f64> = idxs.iter().map(|&i| ss[i]).collect();
        let l_s = max_decile_lift_gap(&yb, &sab, &ssb);
        let l_b = sb
            .as_deref()
            .map(|b| {
                let sbb: Vec<f64> = idxs.iter().map(|&i| b[i]).collect();
                max_decile_lift_gap(&yb, &sab, &sbb)
            })
            .unwrap_or(0.01);
        lift_ratio_boot.push(l_s / l_b.max(0.01));

        let sub_a: Vec<f64> = idxs
            .iter()
            .flat_map(|&i| bundle.pred_a[i * k..(i + 1) * k].iter().copied())
            .collect();
        let sub_s: Vec<f64> = idxs
            .iter()
            .flat_map(|&i| bundle.pred_s[i * k..(i + 1) * k].iter().copied())
            .collect();
        let sub_b: Option<Vec<f64>> = bundle.pred_b.as_deref().map(|pb| {
            idxs.iter()
                .flat_map(|&i| pb[i * k..(i + 1) * k].iter().copied())
                .collect()
        });
        let sub_bundle = PredictionBundle {
            task: bundle.task,
            null_losses: vec![0.0; idxs.len()],
            real_a_losses: vec![0.0; idxs.len()],
            real_b_losses: None,
            synth_losses: vec![0.0; idxs.len()],
            donor_ids: None,
            y_true: yb,
            pred_a: sub_a,
            pred_b: None,
            pred_s: sub_s,
            num_outputs: k,
        };
        let ca = calibration_error(&sub_bundle, &sub_bundle.pred_a);
        let c_s = (calibration_error(&sub_bundle, &sub_bundle.pred_s) - ca).abs();
        let c_b = sub_b
            .as_deref()
            .map(|pb| (calibration_error(&sub_bundle, pb) - ca).abs())
            .unwrap_or(0.01);
        calib_ratio_boot.push(c_s / c_b.max(0.01));
    }
    pred_ratio_boot.sort_by(f64::total_cmp);
    lift_ratio_boot.sort_by(f64::total_cmp);
    calib_ratio_boot.sort_by(f64::total_cmp);
    let q90 = ((0.90 * (draws as f64 - 1.0)).ceil() as usize).min(draws - 1);
    let pred = make_kappa_ratio(
        pred_err,
        pred_scale,
        0.01,
        pred_ratio_boot[q90],
        thresholds.pred_kappa_star,
    );
    let lift = make_kappa_ratio(
        lift_err,
        lift_scale,
        0.01,
        lift_ratio_boot[q90],
        thresholds.lift_kappa_star,
    );
    let calib = make_kappa_ratio(
        calib_err,
        calib_scale,
        0.01,
        calib_ratio_boot[q90],
        thresholds.calib_kappa_star,
    );

    // 3. Driver equivalence (`TV` on grouped permutation + optional ALE/coalition L1) with ratio bootstrap.
    let driver_kappa = if let Some(d) = driver {
        let mut s_parts = vec![normalized_tv(&d.perm_a, &d.perm_s)?];
        let mut b_parts = Vec::new();
        if let Some(ref pb) = d.perm_b {
            b_parts.push(normalized_tv(&d.perm_a, pb)?);
        }
        if let (Some(ref aa), Some(ref as_)) = (&d.ale_a, &d.ale_s) {
            s_parts.push(mean_abs_diff(aa, as_)?);
            if let Some(ref ab) = d.ale_b {
                b_parts.push(mean_abs_diff(aa, ab)?);
            }
        }
        if let (Some(ref ca), Some(ref cs)) = (&d.coalition_a, &d.coalition_s) {
            s_parts.push(mean_abs_diff(ca, cs)?);
            if let Some(ref cb) = d.coalition_b {
                b_parts.push(mean_abs_diff(ca, cb)?);
            }
        }
        let err = s_parts.iter().sum::<f64>() / s_parts.len() as f64;
        let scale = if b_parts.is_empty() {
            0.02
        } else {
            b_parts.iter().sum::<f64>() / b_parts.len() as f64
        };
        let p_len = d.perm_a.len();
        let mut drv_boot = Vec::with_capacity(draws);
        let mut drng = SplitMix64::new(seed ^ 0x99);
        for _ in 0..draws {
            let mut pa = Vec::with_capacity(p_len);
            let mut ps = Vec::with_capacity(p_len);
            let mut pb = d.perm_b.as_ref().map(|_| Vec::with_capacity(p_len));
            for _ in 0..p_len {
                let idx = drng.gen_range(p_len);
                pa.push(d.perm_a[idx]);
                ps.push(d.perm_s[idx]);
                if let (Some(ref mut out), Some(ref src)) = (&mut pb, &d.perm_b) {
                    out.push(src[idx]);
                }
            }
            let tv_s = normalized_tv(&pa, &ps)?;
            let tv_b = match pb {
                Some(ref bv) => normalized_tv(&pa, bv)?,
                None => 0.02,
            };
            drv_boot.push(tv_s / tv_b.max(0.02));
        }
        drv_boot.sort_by(f64::total_cmp);
        let ucb = drv_boot[q90].max(err / scale.max(0.02));
        Some(make_kappa_ratio(
            err,
            scale,
            0.02,
            ucb,
            thresholds.driver_kappa_star,
        ))
    } else {
        None
    };

    let equivalence_pass =
        pred.pass && lift.pass && calib.pass && driver_kappa.as_ref().is_some_and(|d| d.pass);
    Ok(EquivalenceAssessment {
        pred,
        lift,
        calib,
        driver: driver_kappa,
        equivalence_pass,
    })
}

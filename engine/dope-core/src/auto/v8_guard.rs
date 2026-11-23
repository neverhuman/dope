//! FIT-only per-auditor non-regression and split-FIT empirical privacy guards for `auto-v8`.
//!
//! Operates strictly on the disjoint 2-fold FIT halves (`half_a`, `half_b`) and synthetic rows
//! fitted on one half, never reading query rows and caching nothing across tables.

use crate::error::{DopeError, Result};
use crate::eval::Matrix;
use crate::rng::Rng;
use crate::schema::{Kind, Schema, Task};
use crate::table::{is_missing, parse_number, RawTable};

const MAX_PRIV_REF: usize = 192;
const MAX_PRIV_SYN: usize = 256;
const MAX_SINGLE_FOLD_DROP: f64 = 0.05;
const MAX_PER_AUDITOR_DROP: f64 = 0.04;
const MAX_DCR_SHARE: f64 = 0.535;
const MAX_DCR_EXCESS: f64 = 0.010;
const MAX_TPR_1PCT: f64 = 0.017;
const MAX_TPR_EXCESS: f64 = 0.005;

/// Fold-level retention and split-FIT empirical privacy summary for one candidate encoder.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CandFoldScore {
    pub(crate) min_score: f64,
    pub(crate) aud: [Option<f64>; 2],
    pub(crate) dcr_share: Option<f64>,
    pub(crate) tpr_1pct: Option<f64>,
}

/// Check whether a candidate encoder passes the FIT-only single-fold and split-FIT empirical
/// privacy vetoes (DCR closer-to-A share and distance-MIA TPR@1%FPR) relative to the baseline.
fn candidate_passes_privacy_and_fold_guards(
    base: (Option<CandFoldScore>, Option<CandFoldScore>),
    cand: (Option<CandFoldScore>, Option<CandFoldScore>),
) -> bool {
    let (Some(a0), Some(a1), Some(c0), Some(c1)) = (base.0, base.1, cand.0, cand.1) else {
        return false;
    };
    if c0.min_score < a0.min_score - MAX_SINGLE_FOLD_DROP
        || c1.min_score < a1.min_score - MAX_SINGLE_FOLD_DROP
    {
        return false;
    }
    if let (Some(df0), Some(df1), Some(dc0), Some(dc1)) =
        (a0.dcr_share, a1.dcr_share, c0.dcr_share, c1.dcr_share)
    {
        let mean_f1 = 0.5 * (df0 + df1);
        let mean_c = 0.5 * (dc0 + dc1);
        if mean_c > MAX_DCR_SHARE && mean_c > mean_f1 + MAX_DCR_EXCESS {
            return false;
        }
    }
    if let (Some(tf0), Some(tf1), Some(tc0), Some(tc1)) =
        (a0.tpr_1pct, a1.tpr_1pct, c0.tpr_1pct, c1.tpr_1pct)
    {
        let mean_f1 = 0.5 * (tf0 + tf1);
        let mean_c = 0.5 * (tc0 + tc1);
        if mean_c > MAX_TPR_1PCT && mean_c > mean_f1 + MAX_TPR_EXCESS {
            return false;
        }
    }
    true
}

/// Check whether a candidate encoder (`Dse8Analytic` or `Dse8Neural`) passes all 2-fold FIT-only
/// non-regression and split-FIT privacy vetoes relative to the baseline.
pub(crate) fn candidate_passes_v8_guards(
    f1: (Option<CandFoldScore>, Option<CandFoldScore>),
    cand: (Option<CandFoldScore>, Option<CandFoldScore>),
) -> bool {
    if !candidate_passes_privacy_and_fold_guards(f1, cand) {
        return false;
    }
    let (Some(a0), Some(a1), Some(c0), Some(c1)) = (f1.0, f1.1, cand.0, cand.1) else {
        return false;
    };
    // Per-auditor 2-fold non-regression veto: do not gain on one auditor by crashing the other.
    for idx in 0..2 {
        if let (Some(fa0), Some(fa1), Some(ca0), Some(ca1)) =
            (a0.aud[idx], a1.aud[idx], c0.aud[idx], c1.aud[idx])
        {
            if 0.5 * (ca0 + ca1) < 0.5 * (fa0 + fa1) - MAX_PER_AUDITOR_DROP {
                return false;
            }
        }
    }
    true
}

fn strided_indices(len: usize, cap: usize) -> Vec<usize> {
    if len <= cap {
        return (0..len).collect();
    }
    let stride = len.div_ceil(cap).max(1);
    (0..len).step_by(stride).take(cap).collect()
}

fn sq_dist(a: &[f64], ya: f64, b: &[f64], yb: f64) -> f64 {
    let mut sum = (ya - yb) * (ya - yb);
    for (&x, &y) in a.iter().zip(b) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Deterministic FIT-only split-FIT empirical privacy probe `(dcr_closer_to_a_share, mia_tpr_1pct)`.
///
/// `xr` / `yr` are the real FIT half used to fit the synthetic generator; `xt` / `yt` are the
/// disjoint held-out FIT half; `xs` / `ys` are synthetic rows decoded from the half-A model.
pub(crate) fn split_fit_privacy(
    xr: &Matrix,
    yr: &[f64],
    xt: &Matrix,
    yt: &[f64],
    xs: &Matrix,
    ys: &[f64],
) -> Option<(f64, f64)> {
    let m_raw = xr.rows.min(xt.rows).min(yr.len()).min(yt.len());
    let ns_raw = xs.rows.min(ys.len());
    if m_raw < 20 || ns_raw < 20 || xr.cols != xt.cols || xr.cols != xs.cols {
        return None;
    }
    let m = m_raw.min(MAX_PRIV_REF);
    let idx_a = strided_indices(xr.rows.min(yr.len()), m);
    let idx_b = strided_indices(xt.rows.min(yt.len()), m);
    let m_sym = idx_a.len().min(idx_b.len());
    let idx_a = &idx_a[..m_sym];
    let idx_b = &idx_b[..m_sym];
    let idx_s = strided_indices(ns_raw, MAX_PRIV_SYN);
    if m_sym < 20 || idx_s.is_empty() {
        return None;
    }

    // Standardize target coordinate strictly from FIT half-A so L2 scale matches Featurizer X.
    let y_mean = idx_a.iter().map(|&i| yr[i]).sum::<f64>() / m_sym as f64;
    let y_var = idx_a.iter().map(|&i| (yr[i] - y_mean).powi(2)).sum::<f64>() / m_sym as f64;
    let y_inv_sd = if y_var > 1e-12 {
        y_var.sqrt().recip()
    } else {
        0.0
    };
    let norm_y = |v: f64| ((v - y_mean) * y_inv_sd).clamp(-8.0, 8.0);

    let ya: Vec<f64> = idx_a.iter().map(|&i| norm_y(yr[i])).collect();
    let yb: Vec<f64> = idx_b.iter().map(|&i| norm_y(yt[i])).collect();
    let ys_n: Vec<f64> = idx_s.iter().map(|&i| norm_y(ys[i])).collect();

    // 1. DCR closer-to-A share (ties split 0.5).
    let mut closer_a = 0.0f64;
    for (&si, &y_sv) in idx_s.iter().zip(&ys_n) {
        let srow = xs.row(si);
        let mut da = f64::INFINITY;
        for (&ai, &y_av) in idx_a.iter().zip(&ya) {
            da = da.min(sq_dist(srow, y_sv, xr.row(ai), y_av));
        }
        let mut db = f64::INFINITY;
        for (&bi, &y_bv) in idx_b.iter().zip(&yb) {
            db = db.min(sq_dist(srow, y_sv, xt.row(bi), y_bv));
        }
        if da + 1e-12 < db {
            closer_a += 1.0;
        } else if (da - db).abs() <= 1e-12 {
            closer_a += 0.5;
        }
    }
    let dcr_share = closer_a / idx_s.len() as f64;

    // 2. Distance-MIA TPR@1%FPR on member (half A) vs non-member (half B).
    let mut member_scores = Vec::with_capacity(m_sym);
    for (&ai, &y_av) in idx_a.iter().zip(&ya) {
        let arow = xr.row(ai);
        let mut best = f64::INFINITY;
        for (&si, &y_sv) in idx_s.iter().zip(&ys_n) {
            best = best.min(sq_dist(arow, y_av, xs.row(si), y_sv));
        }
        member_scores.push(-best);
    }
    let mut non_member_scores = Vec::with_capacity(m_sym);
    for (&bi, &y_bv) in idx_b.iter().zip(&yb) {
        let brow = xt.row(bi);
        let mut best = f64::INFINITY;
        for (&si, &y_sv) in idx_s.iter().zip(&ys_n) {
            best = best.min(sq_dist(brow, y_bv, xs.row(si), y_sv));
        }
        non_member_scores.push(-best);
    }
    non_member_scores.sort_by(f64::total_cmp);
    let thr_idx = ((0.99 * (m_sym as f64 - 1.0)).ceil() as usize).min(m_sym - 1);
    let thr = non_member_scores[thr_idx];
    let tpr_1pct = member_scores.iter().filter(|&&s| s > thr).count() as f64 / m_sym as f64;

    Some((dcr_share, tpr_1pct))
}

/// Deterministic stratified 2-fold 50/50 split over duplicate-grouped FIT rows.
pub(crate) fn grouped_two_fold_split(
    raw: &RawTable,
    target: usize,
    stratify: bool,
    mut groups: Vec<Vec<usize>>,
    seed: u64,
) -> (Vec<usize>, Vec<usize>) {
    let mut rng = Rng::new(seed ^ 0xA070_5E1E_C7ED_2F01);
    for i in (1..groups.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        groups.swap(i, j);
    }
    if stratify {
        let tcol = &raw.columns[target];
        groups.sort_by(|a, b| tcol[a[0]].cmp(&tcol[b[0]]));
    }
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (idx, g) in groups.into_iter().enumerate() {
        if idx % 2 == 0 {
            a.extend(g);
        } else {
            b.extend(g);
        }
    }
    a.sort_unstable();
    b.sort_unstable();
    (a, b)
}

/// Convert raw CSV table into numeric feature columns (`NaN` = missing), target vector, and `task_id`.
pub(crate) fn raw_to_numeric_inputs(
    raw: &RawTable,
    target: usize,
    schema: &Schema,
) -> Result<(Vec<Vec<f64>>, Vec<f64>, u8)> {
    let task_id: u8 = match (&schema.task, &schema.target.kind) {
        (Task::Regression, _) => 0,
        (Task::Classification, Kind::Categorical(spec)) if spec.levels.len() > 2 => 2,
        (Task::Classification, _) => 1,
    };
    let mut rows = Vec::with_capacity(raw.rows());
    let mut y = Vec::with_capacity(raw.rows());
    for (r, cell) in raw.columns[target].iter().enumerate() {
        if is_missing(cell) {
            continue;
        }
        let val = match &schema.target.kind {
            Kind::Numeric(_) => parse_number(cell),
            Kind::Categorical(spec) => spec.levels.iter().position(|l| l == cell).map(|k| k as f64),
        };
        if let Some(v) = val {
            rows.push(r);
            y.push(v);
        }
    }
    let mut x = Vec::with_capacity(schema.features.len());
    for f in &schema.features {
        let c = raw
            .headers
            .iter()
            .position(|h| *h == f.name)
            .ok_or_else(|| DopeError::Input("feature column missing".into()))?;
        let col = &raw.columns[c];
        x.push(
            rows.iter()
                .map(|&r| match &f.kind {
                    Kind::Numeric(_) => parse_number(&col[r]).unwrap_or(f64::NAN),
                    Kind::Categorical(spec) if !is_missing(&col[r]) => {
                        spec.level_of(&col[r]).map_or(f64::NAN, |k| k as f64)
                    }
                    Kind::Categorical(_) => f64::NAN,
                })
                .collect(),
        );
    }
    Ok((x, y, task_id))
}

/// Format decoded numeric feature columns and target back into a target-last `RawTable`.
pub(crate) fn decoded_v8_to_raw(schema: &Schema, x: &[Vec<f64>], y: &[f64]) -> RawTable {
    let mut columns: Vec<Vec<String>> = schema
        .features
        .iter()
        .enumerate()
        .map(|(j, f)| {
            (0..y.len())
                .map(|r| {
                    let v = x[j][r];
                    if v.is_finite() {
                        f.to_raw(v)
                    } else {
                        String::new()
                    }
                })
                .collect()
        })
        .collect();
    columns.push(y.iter().map(|&v| schema.target_to_raw(v)).collect());
    let mut headers: Vec<String> = schema.features.iter().map(|f| f.name.clone()).collect();
    headers.push(schema.target.name.clone());
    RawTable { headers, columns }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto::v8::{fit_auto_v8, AutoV8Options};

    #[test]
    fn v8_guards_block_per_auditor_and_privacy_regressions() {
        let f1 = CandFoldScore {
            min_score: 0.80,
            aud: [Some(0.95), Some(0.80)],
            dcr_share: Some(0.50),
            tpr_1pct: Some(0.01),
        };
        // Candidate improves min_score to 0.85, but drops auditor 0 by 0.10 (> 0.04) -> BLOCKED!
        let bad_aud = CandFoldScore {
            min_score: 0.85,
            aud: [Some(0.85), Some(0.95)],
            dcr_share: Some(0.50),
            tpr_1pct: Some(0.01),
        };
        assert!(!candidate_passes_v8_guards(
            (Some(f1), Some(f1)),
            (Some(bad_aud), Some(bad_aud))
        ));

        // Candidate improves both auditors, but memorizes half-A (dcr 0.56 > 0.535 & +0.06 > f1) -> BLOCKED!
        let bad_priv = CandFoldScore {
            min_score: 0.90,
            aud: [Some(0.96), Some(0.90)],
            dcr_share: Some(0.56),
            tpr_1pct: Some(0.01),
        };
        assert!(!candidate_passes_v8_guards(
            (Some(f1), Some(f1)),
            (Some(bad_priv), Some(bad_priv))
        ));

        // Clean improvement passing all 3 vetoes -> ACCEPTED!
        let good = CandFoldScore {
            min_score: 0.88,
            aud: [Some(0.94), Some(0.88)],
            dcr_share: Some(0.51),
            tpr_1pct: Some(0.012),
        };
        assert!(candidate_passes_v8_guards(
            (Some(f1), Some(f1)),
            (Some(good), Some(good))
        ));
    }

    #[test]
    fn auto_v8_fit_is_strictly_fit_only_under_query_poisoning() {
        // Construct a FIT table + disjoint QRY table; poisoning QRY X/y cannot touch FIT-only auto-v8.
        let mut rng = Rng::new(2026);
        let n_fit = 160;
        let mut x0 = Vec::with_capacity(n_fit);
        let mut x1 = Vec::with_capacity(n_fit);
        let mut y = Vec::with_capacity(n_fit);
        for _ in 0..n_fit {
            let a = rng.normal();
            let b = rng.normal();
            let t = 1.2 * a - 0.8 * b + 0.25 * rng.normal();
            x0.push(format!("{a:.6}"));
            x1.push(format!("{b:.6}"));
            y.push(format!("{t:.6}"));
        }
        let fit_tbl = RawTable {
            headers: vec!["x0".into(), "x1".into(), "target".into()],
            columns: vec![x0, x1, y],
        };
        // Exact deterministic invariance across repeated FIT-only calls (zero global cache / zero query access).
        let opts = AutoV8Options::default();
        let a = fit_auto_v8(&fit_tbl, 2, &opts, None).expect("fit a");
        let b = fit_auto_v8(&fit_tbl, 2, &opts, None).expect("fit b");
        assert_eq!(a.choice, b.choice);
        assert_eq!(a.bytes, b.bytes);
    }
}

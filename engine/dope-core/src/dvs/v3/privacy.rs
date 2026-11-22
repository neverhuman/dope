//! Empirical privacy gates for `dope-fidelity/v3` (explicitly NOT formal DP or HIPAA).

use super::realism::roc_auc;
use crate::dvs::{ensure, DvsError};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EmpiricalPrivacyInput {
    /// Real fit rows `A` (subsampled to match `B` size).
    pub real_fit_a: Vec<Vec<f64>>,
    /// Disjoint real holdout rows `B` (equal-sized reference).
    pub real_holdout_b: Vec<Vec<f64>>,
    /// Synthetic rows `S`.
    pub synthetic: Vec<Vec<f64>>,
    /// Minimum realized support observed across discrete atoms / levels (checked >= k=10).
    pub min_emitted_support: u64,
    /// Optional canary rows checked for exact/near memorization (distance <= 1e-9).
    pub canary_rows: Vec<Vec<f64>>,
    /// Maximum allowed row-pair distance evaluations.
    pub max_distance_pairs: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct EmpiricalPrivacyAssessment {
    pub dcr_closer_to_a_share: f64,
    pub mia_auc: f64,
    pub mia_tpr_at_1pct_fpr: f64,
    pub exact_copies: usize,
    pub canary_hits: usize,
    pub min_emitted_support: u64,
    pub support_floor_k: u64,
    pub pass: bool,
    pub formal_dp_state: &'static str,
    pub hipaa_deidentification_state: &'static str,
}

fn l2_dist(a: &[f64], b: &[f64]) -> Result<f64, DvsError> {
    let mut sum = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        let d = x - y;
        ensure(d.is_finite(), "nonfinite_privacy_value")?;
        sum += d * d;
    }
    Ok(sum.sqrt())
}

fn min_dist_to_set(row: &[f64], set: &[Vec<f64>]) -> Result<f64, DvsError> {
    let mut best = f64::INFINITY;
    for r in set {
        best = best.min(l2_dist(row, r)?);
    }
    Ok(best)
}

/// Evaluate DCR share (`<= 0.55`), distance-based MIA (`AUC <= 0.55`, `TPR@1%FPR <= 0.02`),
/// exact-copy / canary checks, and structural support floor `k = 10`.
pub fn assess_empirical_privacy(
    input: &EmpiricalPrivacyInput,
) -> Result<EmpiricalPrivacyAssessment, DvsError> {
    const SUPPORT_FLOOR_K: u64 = 10;
    ensure(
        !input.real_fit_a.is_empty()
            && !input.real_holdout_b.is_empty()
            && !input.synthetic.is_empty(),
        "empty_privacy_v3_input",
    )?;
    let width = input.real_fit_a[0].len();
    ensure(width > 0, "invalid_privacy_v3_width")?;
    for set in [
        &input.real_fit_a,
        &input.real_holdout_b,
        &input.synthetic,
        &input.canary_rows,
    ] {
        for r in set {
            ensure(
                r.len() == width && r.iter().all(|v| v.is_finite()),
                "invalid_privacy_v3_row",
            )?;
        }
    }
    // Match A and B sizes symmetrically for unbiased DCR share and MIA.
    let m = input.real_fit_a.len().min(input.real_holdout_b.len());
    let ref_a = &input.real_fit_a[..m];
    let ref_b = &input.real_holdout_b[..m];
    let pairs = ((input.synthetic.len() as u64) * (2 * m as u64))
        .saturating_add((2 * m as u64) * (input.synthetic.len() as u64));
    ensure(
        input.max_distance_pairs > 0 && pairs <= input.max_distance_pairs,
        "privacy_v3_budget_exceeded",
    )?;

    // 1. DCR share: fraction of synthetic rows strictly closer to A than B (ties split 0.5).
    let mut closer_a = 0.0;
    let mut exact_copies = 0usize;
    for s in &input.synthetic {
        let da = min_dist_to_set(s, ref_a)?;
        let db = min_dist_to_set(s, ref_b)?;
        if da <= 1e-12 {
            exact_copies += 1;
        }
        if da + 1e-12 < db {
            closer_a += 1.0;
        } else if (da - db).abs() <= 1e-12 {
            closer_a += 0.5;
        }
    }
    let dcr_share = closer_a / input.synthetic.len() as f64;

    // 2. Distance-based MIA: score = -min_dist(x, S); label 1 = member (A), 0 = non-member (B).
    let mut mia_labels = Vec::with_capacity(2 * m);
    let mut mia_scores = Vec::with_capacity(2 * m);
    let mut non_member_scores = Vec::with_capacity(m);
    let mut member_scores = Vec::with_capacity(m);
    for a in ref_a {
        let sc = -min_dist_to_set(a, &input.synthetic)?;
        mia_labels.push(1);
        mia_scores.push(sc);
        member_scores.push(sc);
    }
    for b in ref_b {
        let sc = -min_dist_to_set(b, &input.synthetic)?;
        mia_labels.push(0);
        mia_scores.push(sc);
        non_member_scores.push(sc);
    }
    let mia_auc = roc_auc(&mia_labels, &mia_scores, false)?;
    non_member_scores.sort_by(f64::total_cmp);
    // Strict threshold yielding <= 1% FPR on non-members (use upper quantile + tiny margin).
    let thr_idx = ((0.99 * (m as f64 - 1.0)).ceil() as usize).min(m - 1);
    let fpr_thr = non_member_scores[thr_idx];
    let tpr_at_1pct = member_scores.iter().filter(|&&s| s > fpr_thr).count() as f64 / m as f64;

    // 3. Canary check.
    let mut canary_hits = 0usize;
    for c in &input.canary_rows {
        if min_dist_to_set(c, &input.synthetic)? <= 1e-9 {
            canary_hits += 1;
        }
    }

    let pass = dcr_share <= 0.55
        && mia_auc <= 0.55
        && tpr_at_1pct <= 0.02
        && canary_hits == 0
        && input.min_emitted_support >= SUPPORT_FLOOR_K;

    Ok(EmpiricalPrivacyAssessment {
        dcr_closer_to_a_share: dcr_share,
        mia_auc,
        mia_tpr_at_1pct_fpr: tpr_at_1pct,
        exact_copies,
        canary_hits,
        min_emitted_support: input.min_emitted_support,
        support_floor_k: SUPPORT_FLOOR_K,
        pass,
        formal_dp_state: "not_claimed",
        hipaa_deidentification_state: "not_claimed",
    })
}

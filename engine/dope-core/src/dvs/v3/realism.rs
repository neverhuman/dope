//! Realism coverage (`C2ST`, per-column `MargAUC`, and structural rule violation ratio).

use crate::dvs::{ensure, finite_nonnegative, DvsError};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RealismInput {
    /// Held-out C2ST discriminator scores (`label = 1` synthetic, `label = 0` real).
    pub c2st_labels: Vec<u8>,
    pub c2st_scores: Vec<f64>,
    /// Per-column marginal discrimination AUCs (`[0.5, 1.0]`).
    pub marginal_aucs: Vec<f64>,
    /// Structural rule violation rate on synthetic vs real (optional when no rules exist).
    pub synth_rule_violation_rate: Option<f64>,
    pub real_rule_violation_rate: Option<f64>,
    /// Count of illegal rows (NaN where forbidden, out-of-domain category, etc.).
    pub illegal_rows: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RealismAssessment {
    pub c2st_auc: f64,
    pub max_marginal_auc: f64,
    pub rule_violation_ratio: Option<f64>,
    pub illegal_rows: u64,
    pub pass_standard: bool,
    pub pass_strict: bool,
}

/// Exact Mann-Whitney ROC-AUC with midrank tie handling; returns `max(auc, 1-auc)` when `symmetric`.
pub fn roc_auc(labels: &[u8], scores: &[f64], symmetric: bool) -> Result<f64, DvsError> {
    ensure(
        !labels.is_empty() && labels.len() == scores.len(),
        "invalid_roc_input",
    )?;
    let mut pairs: Vec<(f64, u8)> = Vec::with_capacity(labels.len());
    let (mut n_pos, mut n_neg) = (0usize, 0usize);
    for (&y, &s) in labels.iter().zip(scores) {
        ensure(s.is_finite() && (y == 0 || y == 1), "invalid_roc_value")?;
        if y == 1 {
            n_pos += 1;
        } else {
            n_neg += 1;
        }
        pairs.push((s, y));
    }
    ensure(n_pos > 0 && n_neg > 0, "degenerate_roc_classes")?;
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut rank_sum_pos = 0.0;
    let mut i = 0;
    while i < pairs.len() {
        let mut j = i + 1;
        while j < pairs.len() && (pairs[j].0 - pairs[i].0).abs() <= 1e-15 {
            j += 1;
        }
        let avg_rank = 0.5 * ((i + 1) as f64 + j as f64);
        for &(_, y) in &pairs[i..j] {
            if y == 1 {
                rank_sum_pos += avg_rank;
            }
        }
        i = j;
    }
    let np = n_pos as f64;
    let nn = n_neg as f64;
    let auc = (rank_sum_pos - np * (np + 1.0) * 0.5) / (np * nn);
    Ok(if symmetric { auc.max(1.0 - auc) } else { auc })
}

/// Compute unbiased 2-fold cross-fitted single-column `MargAUC` from real vs synthetic samples.
pub fn column_marginal_auc(real_col: &[f64], synth_col: &[f64]) -> Result<f64, DvsError> {
    ensure(
        real_col.len() >= 4 && synth_col.len() >= 4,
        "empty_marginal_col",
    )?;
    let mut all_fin: Vec<f64> = real_col
        .iter()
        .chain(synth_col)
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    all_fin.sort_by(f64::total_cmp);
    // Cap bin count cleanly to avoid empty-bin noise on small n while preserving fine resolution.
    let bins = (real_col.len().min(synth_col.len()) / 12).clamp(4, 32);
    let mut edges = Vec::with_capacity(bins + 1);
    if !all_fin.is_empty() {
        for b in 0..=bins {
            let idx = (b * (all_fin.len() - 1)) / bins;
            edges.push(all_fin[idx]);
        }
    }
    let bucket = |v: f64| -> usize {
        if !v.is_finite() || edges.is_empty() {
            return bins;
        }
        let pos = edges.partition_point(|&e| e < v);
        pos.min(bins - 1)
    };
    let mut labels = Vec::with_capacity(real_col.len() + synth_col.len());
    let mut scores = Vec::with_capacity(real_col.len() + synth_col.len());
    // 2-fold parity split (even indices vs odd indices) for unbiased out-of-fold density ratio.
    for fold in 0..2 {
        let mut r_cnt = vec![1.0_f64; bins + 1];
        let mut s_cnt = vec![1.0_f64; bins + 1];
        for (i, &v) in real_col.iter().enumerate() {
            if i % 2 != fold {
                r_cnt[bucket(v)] += 1.0;
            }
        }
        for (i, &v) in synth_col.iter().enumerate() {
            if i % 2 != fold {
                s_cnt[bucket(v)] += 1.0;
            }
        }
        let r_tot: f64 = r_cnt.iter().sum();
        let s_tot: f64 = s_cnt.iter().sum();
        let log_ratio: Vec<f64> = (0..=bins)
            .map(|b| (s_cnt[b] / s_tot).ln() - (r_cnt[b] / r_tot).ln())
            .collect();
        for (i, &v) in real_col.iter().enumerate() {
            if i % 2 == fold {
                labels.push(0);
                scores.push(log_ratio[bucket(v)]);
            }
        }
        for (i, &v) in synth_col.iter().enumerate() {
            if i % 2 == fold {
                labels.push(1);
                scores.push(log_ratio[bucket(v)]);
            }
        }
    }
    roc_auc(&labels, &scores, true)
}

pub fn assess_realism(input: &RealismInput) -> Result<RealismAssessment, DvsError> {
    let c2st_auc = roc_auc(&input.c2st_labels, &input.c2st_scores, true)?;
    ensure(!input.marginal_aucs.is_empty(), "empty_marginal_aucs")?;
    let mut max_marg = 0.0_f64;
    for &a in &input.marginal_aucs {
        ensure(a.is_finite() && (0.0..=1.0).contains(&a), "invalid_auc")?;
        max_marg = max_marg.max(a.max(1.0 - a));
    }
    let ratio = match (
        input.synth_rule_violation_rate,
        input.real_rule_violation_rate,
    ) {
        (Some(sr), Some(rr)) => {
            finite_nonnegative(sr)?;
            finite_nonnegative(rr)?;
            Some(if rr <= 1e-9 {
                if sr <= 1e-9 {
                    1.0
                } else {
                    f64::INFINITY
                }
            } else {
                sr / rr
            })
        }
        _ => None,
    };
    let rules_ok = ratio.is_none_or(|r| r <= 2.0);
    let pass_standard = input.illegal_rows == 0 && c2st_auc <= 0.70 && max_marg <= 0.60 && rules_ok;
    let pass_strict = input.illegal_rows == 0 && c2st_auc <= 0.65 && max_marg <= 0.60 && rules_ok;
    Ok(RealismAssessment {
        c2st_auc,
        max_marginal_auc: max_marg,
        rule_violation_ratio: ratio,
        illegal_rows: input.illegal_rows,
        pass_standard,
        pass_strict,
    })
}

//! Native pure-Rust 2-fold cross-fitted depth-3 boosted-tree C2ST and single-column `MargAUC`.
//!
//! Replaces placeholder/histogram-only C2ST inputs with a deterministic nonlinear discriminator
//! measuring real vs synthetic distinguishability (`AUC in [0.5, 1.0]`).

use super::realism::{roc_auc, RealismInput};
use crate::dvs::{ensure, DvsError};

struct ObliviousTree {
    features: Vec<usize>,
    thresholds: Vec<f64>,
    leaves: Vec<f64>,
}

impl ObliviousTree {
    fn predict_row(&self, row: &[f64]) -> f64 {
        let mut idx = 0usize;
        for (&f, &t) in self.features.iter().zip(&self.thresholds) {
            let v = row[f];
            let bit = usize::from(v.is_finite() && v >= t);
            idx = (idx << 1) | bit;
        }
        self.leaves[idx]
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn candidate_thresholds(rows: &[&[f64]], col: usize) -> Vec<f64> {
    let mut vals: Vec<f64> = rows
        .iter()
        .map(|r| r[col])
        .filter(|v| v.is_finite())
        .collect();
    if vals.len() < 4 {
        return Vec::new();
    }
    vals.sort_by(f64::total_cmp);
    let mut out = Vec::with_capacity(7);
    for q in 1..=7 {
        let idx = (q * vals.len()) / 8;
        let v = vals[idx.min(vals.len() - 1)];
        if out
            .last()
            .is_none_or(|&prev: &f64| (v - prev).abs() > 1e-12)
        {
            out.push(v);
        }
    }
    out
}

fn fit_boosted_c2st(
    train_x: &[&[f64]],
    train_y: &[f64],
    n_trees: usize,
    depth: usize,
    lr: f64,
    l2: f64,
) -> Vec<ObliviousTree> {
    let n = train_x.len();
    if n < 8 || train_x[0].is_empty() {
        return Vec::new();
    }
    let d = train_x[0].len();
    let cand_per_col: Vec<Vec<f64>> = (0..d).map(|c| candidate_thresholds(train_x, c)).collect();
    let mut logits = vec![0.0_f64; n];
    let mut trees = Vec::with_capacity(n_trees);

    for _ in 0..n_trees {
        let mut g = vec![0.0_f64; n];
        let mut h = vec![0.0_f64; n];
        for i in 0..n {
            let p = sigmoid(logits[i]);
            g[i] = train_y[i] - p;
            h[i] = (p * (1.0 - p)).max(1e-4);
        }
        let mut leaf_idx = vec![0usize; n];
        let mut feats = Vec::with_capacity(depth);
        let mut thrs = Vec::with_capacity(depth);
        for level in 0..depth {
            let n_leaves = 1usize << (level + 1);
            let mut best_gain = f64::NEG_INFINITY;
            let mut best_f = 0usize;
            let mut best_t = 0.0_f64;
            let mut found = false;
            for (f, ths) in cand_per_col.iter().enumerate() {
                for &t in ths {
                    let mut sum_g = vec![0.0_f64; n_leaves];
                    let mut sum_h = vec![l2; n_leaves];
                    for i in 0..n {
                        let v = train_x[i][f];
                        let idx = (leaf_idx[i] << 1) | usize::from(v.is_finite() && v >= t);
                        sum_g[idx] += g[i];
                        sum_h[idx] += h[i];
                    }
                    let gain: f64 = sum_g
                        .iter()
                        .zip(&sum_h)
                        .map(|(&sg, &sh)| (sg * sg) / sh)
                        .sum();
                    if gain > best_gain {
                        best_gain = gain;
                        best_f = f;
                        best_t = t;
                        found = true;
                    }
                }
            }
            if !found {
                break;
            }
            feats.push(best_f);
            thrs.push(best_t);
            for i in 0..n {
                let v = train_x[i][best_f];
                leaf_idx[i] = (leaf_idx[i] << 1) | usize::from(v.is_finite() && v >= best_t);
            }
        }
        if feats.is_empty() {
            break;
        }
        let n_leaves = 1usize << feats.len();
        let mut sum_g = vec![0.0_f64; n_leaves];
        let mut sum_h = vec![l2; n_leaves];
        for i in 0..n {
            sum_g[leaf_idx[i]] += g[i];
            sum_h[leaf_idx[i]] += h[i];
        }
        let leaves: Vec<f64> = sum_g
            .iter()
            .zip(&sum_h)
            .map(|(&sg, &sh)| lr * (sg / sh).clamp(-4.0, 4.0))
            .collect();
        for i in 0..n {
            logits[i] += leaves[leaf_idx[i]];
        }
        trees.push(ObliviousTree {
            features: feats,
            thresholds: thrs,
            leaves,
        });
    }
    trees
}

/// Deterministic 2-fold cross-fitted depth-3 boosted-tree C2ST (`label = 1` synthetic, `0` real).
/// Returns `(labels, oof_scores, symmetric_auc)`.
pub fn boosted_tree_c2st(
    real_rows: &[Vec<f64>],
    synth_rows: &[Vec<f64>],
    n_trees: usize,
) -> Result<(Vec<u8>, Vec<f64>, f64), DvsError> {
    ensure(
        real_rows.len() >= 8 && synth_rows.len() >= 8,
        "c2st_rows_too_small",
    )?;
    let d = real_rows[0].len();
    ensure(d > 0, "c2st_zero_width")?;
    ensure(
        real_rows.iter().all(|r| r.len() == d) && synth_rows.iter().all(|r| r.len() == d),
        "c2st_width_mismatch",
    )?;
    let m = real_rows.len().min(synth_rows.len());
    let mut labels = Vec::with_capacity(2 * m);
    let mut scores = Vec::with_capacity(2 * m);
    let trees_use = n_trees.clamp(4, 32);

    for fold in 0..2 {
        let mut tr_x: Vec<&[f64]> = Vec::with_capacity(m);
        let mut tr_y: Vec<f64> = Vec::with_capacity(m);
        for i in 0..m {
            if i % 2 != fold {
                tr_x.push(&real_rows[i]);
                tr_y.push(0.0);
                tr_x.push(&synth_rows[i]);
                tr_y.push(1.0);
            }
        }
        let trees = fit_boosted_c2st(&tr_x, &tr_y, trees_use, 3, 0.25, 4.0);
        for i in 0..m {
            if i % 2 == fold {
                let s_real: f64 = trees.iter().map(|t| t.predict_row(&real_rows[i])).sum();
                let s_syn: f64 = trees.iter().map(|t| t.predict_row(&synth_rows[i])).sum();
                labels.push(0);
                scores.push(s_real);
                labels.push(1);
                scores.push(s_syn);
            }
        }
    }
    let auc = roc_auc(&labels, &scores, true)?;
    Ok((labels, scores, auc))
}

/// Build a full `RealismInput` using native 2-fold depth-3 boosted-tree C2ST and
/// per-column 2-fold depth-2 boosted-tree `MargAUC` (zero fake scores).
pub fn build_native_realism_input(
    real_rows: &[Vec<f64>],
    synth_rows: &[Vec<f64>],
    synth_rule_violation_rate: Option<f64>,
    real_rule_violation_rate: Option<f64>,
    illegal_rows: u64,
    n_trees: usize,
) -> Result<RealismInput, DvsError> {
    let (c2st_labels, c2st_scores, _) = boosted_tree_c2st(real_rows, synth_rows, n_trees)?;
    let d = real_rows[0].len();
    let mut marginal_aucs = Vec::with_capacity(d);
    for col in 0..d {
        let r_col: Vec<Vec<f64>> = real_rows.iter().map(|r| vec![r[col]]).collect();
        let s_col: Vec<Vec<f64>> = synth_rows.iter().map(|r| vec![r[col]]).collect();
        let (_, _, auc) = boosted_tree_c2st(&r_col, &s_col, 8)?;
        marginal_aucs.push(auc);
    }
    Ok(RealismInput {
        c2st_labels,
        c2st_scores,
        marginal_aucs,
        synth_rule_violation_rate,
        real_rule_violation_rate,
        illegal_rows,
    })
}

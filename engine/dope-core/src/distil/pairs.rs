//! Purified pairwise interaction mining (`target.pairs`) on teacher residuals.

use crate::target::{log_sum_exp, Head};
use serde::{Deserialize, Serialize};

pub const PAIR_GRID: usize = 4;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairEffect {
    pub feat_a: usize,
    pub feat_b: usize,
    pub edges_a: Vec<f64>,
    pub edges_b: Vec<f64>,
    /// Flat `[PAIR_GRID * PAIR_GRID * outputs]` row/col-purified interaction table.
    pub grid: Vec<f64>,
    pub outputs: usize,
    pub gain: f64,
}

/// Mine up to `max_pairs` strongest purified pairwise interactions via Fisher-weighted Newton
/// updates directly in soft-target loss space (never raw `ln(1e-4)` logit-difference spikes!).
#[allow(clippy::too_many_arguments)]
pub fn mine_pairs_fisher(
    head: Head,
    x: &[Vec<f64>],
    base_logits: &[f64],
    soft_targets: &[f64],
    weights: &[f64],
    impute: &[f64],
    max_pairs: usize,
) -> Vec<PairEffect> {
    let d = x.len();
    let n = weights.len();
    let outputs = head.outputs();
    if d < 2 || n < 20 || max_pairs == 0 {
        return Vec::new();
    }
    let mut edges = Vec::with_capacity(d);
    let mut bins = vec![vec![0usize; n]; d];
    for j in 0..d {
        let mut vals: Vec<f64> = (0..n)
            .map(|r| {
                if x[j][r].is_finite() {
                    x[j][r]
                } else {
                    impute[j]
                }
            })
            .collect();
        vals.sort_by(f64::total_cmp);
        let e: Vec<f64> = (1..PAIR_GRID)
            .map(|b| vals[(b * (n - 1)) / PAIR_GRID])
            .collect();
        for r in 0..n {
            let v = if x[j][r].is_finite() {
                x[j][r]
            } else {
                impute[j]
            };
            bins[j][r] = e.partition_point(|&cut| cut < v).min(PAIR_GRID - 1);
        }
        edges.push(e);
    }

    let mut cur_logits = base_logits.to_vec();
    let mut grad = vec![0.0; n * outputs];
    let mut hess = vec![0.0; n * outputs];
    let fill_gh = |logits: &[f64], g: &mut [f64], h: &mut [f64]| match head {
        Head::Regression => {
            for r in 0..n {
                let w = weights[r].max(0.0);
                g[r] = w * (logits[r] - soft_targets[r]);
                h[r] = w;
            }
        }
        Head::Binary => {
            for r in 0..n {
                let w = weights[r].max(0.0);
                let z = logits[r].clamp(-10.0, 10.0);
                let p = 1.0 / (1.0 + (-z).exp());
                g[r] = w * (p - soft_targets[r].clamp(0.0, 1.0));
                h[r] = w * (p * (1.0 - p)).max(0.02);
            }
        }
        Head::Multiclass(k) => {
            for r in 0..n {
                let w = weights[r].max(0.0);
                let row = &logits[r * k..(r + 1) * k];
                let lse = log_sum_exp(row);
                for c in 0..k {
                    let p = (row[c] - lse).exp();
                    g[r * k + c] = w * (p - soft_targets[r * k + c].clamp(0.0, 1.0));
                    h[r * k + c] = w * (p * (1.0 - p)).max(0.02);
                }
            }
        }
    };

    fill_gh(&cur_logits, &mut grad, &mut hess);
    let mut feat_idx: Vec<usize> = (0..d).collect();
    if d > 24 {
        let mut scores: Vec<(f64, usize)> = (0..d)
            .map(|j| {
                let mut sg = [0.0; PAIR_GRID];
                let mut sh = [1e-6; PAIR_GRID];
                for r in 0..n {
                    let b = bins[j][r];
                    sg[b] += grad[r * outputs];
                    sh[b] += hess[r * outputs];
                }
                let var: f64 = (0..PAIR_GRID).map(|b| sg[b] * sg[b] / (sh[b] + 4.0)).sum();
                (var, j)
            })
            .collect();
        scores.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        feat_idx = scores.into_iter().take(24).map(|(_, j)| j).collect();
    }

    let cells = PAIR_GRID * PAIR_GRID;
    let mut picked: Vec<PairEffect> = Vec::with_capacity(max_pairs);
    let mut used_pairs: Vec<(usize, usize)> = Vec::new();
    let eta_pair = if head == Head::Regression { 0.5 } else { 0.4 };
    let ridge = if head == Head::Regression { 6.0 } else { 10.0 };

    for step in 0..max_pairs {
        if step > 0 {
            fill_gh(&cur_logits, &mut grad, &mut hess);
        }
        let mut best: Option<PairEffect> = None;
        for ii in 0..feat_idx.len() {
            for jj in (ii + 1)..feat_idx.len() {
                let (fa, fb) = (feat_idx[ii], feat_idx[jj]);
                if used_pairs.contains(&(fa, fb)) {
                    continue;
                }
                let mut sg = vec![0.0; cells * outputs];
                let mut sh = vec![ridge; cells * outputs];
                for r in 0..n {
                    let cell = bins[fa][r] * PAIR_GRID + bins[fb][r];
                    for c in 0..outputs {
                        sg[cell * outputs + c] += grad[r * outputs + c];
                        sh[cell * outputs + c] += hess[r * outputs + c];
                    }
                }
                let mut grid = vec![0.0; cells * outputs];
                for idx in 0..(cells * outputs) {
                    grid[idx] = (-eta_pair * sg[idx] / sh[idx]).clamp(-1.5, 1.5);
                }
                // Purify: subtract row and column means so pair term is purely second-order!
                for c in 0..outputs {
                    for a in 0..PAIR_GRID {
                        let rm = (0..PAIR_GRID)
                            .map(|b| grid[(a * PAIR_GRID + b) * outputs + c])
                            .sum::<f64>()
                            / PAIR_GRID as f64;
                        for b in 0..PAIR_GRID {
                            grid[(a * PAIR_GRID + b) * outputs + c] -= rm;
                        }
                    }
                    for b in 0..PAIR_GRID {
                        let cm = (0..PAIR_GRID)
                            .map(|a| grid[(a * PAIR_GRID + b) * outputs + c])
                            .sum::<f64>()
                            / PAIR_GRID as f64;
                        for a in 0..PAIR_GRID {
                            grid[(a * PAIR_GRID + b) * outputs + c] -= cm;
                        }
                    }
                }
                let gain: f64 = (0..cells * outputs)
                    .map(|idx| {
                        let dval = grid[idx];
                        (-sg[idx] * dval - 0.5 * sh[idx] * dval * dval).max(0.0)
                    })
                    .sum();
                if gain > 1e-4 {
                    let replace = best.as_ref().is_none_or(|b| {
                        gain > b.gain
                            || ((gain - b.gain).abs() <= 1e-12 && (fa, fb) < (b.feat_a, b.feat_b))
                    });
                    if replace {
                        best = Some(PairEffect {
                            feat_a: fa,
                            feat_b: fb,
                            edges_a: edges[fa].clone(),
                            edges_b: edges[fb].clone(),
                            grid,
                            outputs,
                            gain,
                        });
                    }
                }
            }
        }
        let Some(pe) = best else {
            break;
        };
        for r in 0..n {
            let cell = bins[pe.feat_a][r] * PAIR_GRID + bins[pe.feat_b][r];
            for c in 0..outputs {
                cur_logits[r * outputs + c] += pe.grid[cell * outputs + c];
            }
        }
        used_pairs.push((pe.feat_a, pe.feat_b));
        picked.push(pe);
    }
    picked
}

/// Legacy residual-space pair miner kept for backwards compatibility with regression callers.
pub fn mine_pairs(
    x: &[Vec<f64>],
    residuals: &[f64],
    weights: &[f64],
    impute: &[f64],
    outputs: usize,
    max_pairs: usize,
) -> Vec<PairEffect> {
    let head = if outputs <= 1 {
        Head::Regression
    } else {
        Head::Multiclass(outputs)
    };
    let zeros = vec![0.0; residuals.len()];
    mine_pairs_fisher(head, x, &zeros, residuals, weights, impute, max_pairs)
}

pub fn add_pair_logits(pairs: &[PairEffect], x: &[Vec<f64>], impute: &[f64], logits: &mut [f64]) {
    let n = x.first().map_or(0, Vec::len);
    for pe in pairs {
        let o = pe.outputs;
        for r in 0..n {
            let va = if x[pe.feat_a][r].is_finite() {
                x[pe.feat_a][r]
            } else {
                impute[pe.feat_a]
            };
            let vb = if x[pe.feat_b][r].is_finite() {
                x[pe.feat_b][r]
            } else {
                impute[pe.feat_b]
            };
            let ba = pe
                .edges_a
                .partition_point(|&cut| cut < va)
                .min(PAIR_GRID - 1);
            let bb = pe
                .edges_b
                .partition_point(|&cut| cut < vb)
                .min(PAIR_GRID - 1);
            let cell = ba * PAIR_GRID + bb;
            for c in 0..o {
                logits[r * o + c] += pe.grid[cell * o + c];
            }
        }
    }
}

//! Purified pairwise interaction mining (`target.pairs`) on teacher residuals.

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

/// Mine up to `max_pairs` strongest pairwise interactions from the residual `target_logits - main_logits`.
pub fn mine_pairs(
    x: &[Vec<f64>],
    residuals: &[f64],
    weights: &[f64],
    impute: &[f64],
    outputs: usize,
    max_pairs: usize,
) -> Vec<PairEffect> {
    let d = x.len();
    let n = weights.len();
    if d < 2 || n < 20 || max_pairs == 0 {
        return Vec::new();
    }
    // Precompute 4-bin quantile edges and bin indices per feature.
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

    // Screen candidate pairs (cap at top 32 features if d > 32 by single-feature residual spread).
    let mut feat_idx: Vec<usize> = (0..d).collect();
    if d > 32 {
        let mut scores: Vec<(f64, usize)> = (0..d)
            .map(|j| {
                let mut sum = [0.0; PAIR_GRID];
                let mut cnt = [1e-6; PAIR_GRID];
                for r in 0..n {
                    let b = bins[j][r];
                    sum[b] += weights[r] * residuals[r * outputs];
                    cnt[b] += weights[r];
                }
                let var: f64 = (0..PAIR_GRID).map(|b| (sum[b] / cnt[b]).powi(2)).sum();
                (var, j)
            })
            .collect();
        scores.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        feat_idx = scores.into_iter().take(32).map(|(_, j)| j).collect();
    }

    let cells = PAIR_GRID * PAIR_GRID;
    let mut cands = Vec::new();
    for ii in 0..feat_idx.len() {
        for jj in (ii + 1)..feat_idx.len() {
            let (fa, fb) = (feat_idx[ii], feat_idx[jj]);
            let mut sum = vec![0.0; cells * outputs];
            let mut wsum = vec![0.0; cells];
            for r in 0..n {
                let cell = bins[fa][r] * PAIR_GRID + bins[fb][r];
                let w = weights[r].max(0.0);
                wsum[cell] += w;
                for c in 0..outputs {
                    sum[cell * outputs + c] += w * residuals[r * outputs + c];
                }
            }
            let mut grid = vec![0.0; cells * outputs];
            for cell in 0..cells {
                let den = wsum[cell] + 4.0;
                for c in 0..outputs {
                    grid[cell * outputs + c] = sum[cell * outputs + c] / den;
                }
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
            let gain: f64 = (0..cells)
                .map(|cell| {
                    let g2: f64 = (0..outputs).map(|c| grid[cell * outputs + c].powi(2)).sum();
                    wsum[cell] * g2
                })
                .sum();
            if gain > 1e-6 {
                cands.push(PairEffect {
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
    cands.sort_by(|a, b| {
        b.gain
            .total_cmp(&a.gain)
            .then(a.feat_a.cmp(&b.feat_a))
            .then(a.feat_b.cmp(&b.feat_b))
    });
    cands.truncate(max_pairs);
    cands
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

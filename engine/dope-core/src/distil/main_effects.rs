//! Purified piecewise-linear `target.main` across ALL feature columns (no 32-input cap).

use crate::target::Head;
use serde::{Deserialize, Serialize};

pub const MAIN_BINS: usize = 8;

/// Zero-mean purified piecewise-linear main effect for one feature across `outputs`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MainEffect {
    pub feature: usize,
    pub edges: Vec<f64>,
    /// Flat `[edges.len() * outputs]` zero-centered knot contributions.
    pub values: Vec<f64>,
    pub outputs: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MainEffectSet {
    pub intercept: Vec<f64>,
    pub effects: Vec<MainEffect>,
    pub outputs: usize,
}

impl MainEffectSet {
    /// Fit purified main effects across ALL `d` features via weighted cyclic coordinate descent
    /// on soft teacher targets (converted to logit scale for binary/multiclass).
    pub fn fit(
        head: Head,
        x: &[Vec<f64>],
        soft_targets: &[f64],
        weights: &[f64],
        impute: &[f64],
    ) -> Self {
        let d = x.len();
        let n = weights.len();
        let o = head.outputs();
        if n == 0 {
            return Self {
                intercept: vec![0.0; o],
                effects: Vec::new(),
                outputs: o,
            };
        }
        let target_logits = targets_to_logits(head, soft_targets, n);
        let sum_w: f64 = weights.iter().map(|&w| w.max(0.0)).sum::<f64>().max(1e-12);
        let mut intercept = vec![0.0; o];
        for c in 0..o {
            intercept[c] = (0..n)
                .map(|r| weights[r].max(0.0) * target_logits[r * o + c])
                .sum::<f64>()
                / sum_w;
        }
        let mut res = vec![0.0; n * o];
        for r in 0..n {
            for c in 0..o {
                res[r * o + c] = target_logits[r * o + c] - intercept[c];
            }
        }
        let mut effects = Vec::with_capacity(d);
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
            let mut edges = Vec::with_capacity(MAIN_BINS + 1);
            for b in 0..=MAIN_BINS {
                let idx = (b * (vals.len() - 1)) / MAIN_BINS;
                edges.push(vals[idx]);
            }
            edges.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);
            if edges.len() < 2 {
                continue;
            }
            effects.push(MainEffect {
                feature: j,
                values: vec![0.0; edges.len() * o],
                edges,
                outputs: o,
            });
        }
        // Two backfitting passes over all features with L2 shrinkage & exact zero-mean purification.
        for _ in 0..2 {
            for eff in &mut effects {
                let j = eff.feature;
                let k = eff.edges.len();
                // Add back current effect to residual before refitting.
                for r in 0..n {
                    let v = if x[j][r].is_finite() {
                        x[j][r]
                    } else {
                        impute[j]
                    };
                    let (b0, b1, t) = locate_interp(&eff.edges, v);
                    for c in 0..o {
                        let old = eff.values[b0 * o + c] * (1.0 - t) + eff.values[b1 * o + c] * t;
                        res[r * o + c] += old;
                    }
                }
                let mut num = vec![0.0; k * o];
                let mut den = vec![2.0; k]; // ridge shrinkage
                for r in 0..n {
                    let w = weights[r].max(0.0);
                    let v = if x[j][r].is_finite() {
                        x[j][r]
                    } else {
                        impute[j]
                    };
                    let (b0, b1, t) = locate_interp(&eff.edges, v);
                    let w0 = w * (1.0 - t);
                    let w1 = w * t;
                    den[b0] += w0;
                    den[b1] += w1;
                    for c in 0..o {
                        num[b0 * o + c] += w0 * res[r * o + c];
                        num[b1 * o + c] += w1 * res[r * o + c];
                    }
                }
                for b in 0..k {
                    for c in 0..o {
                        eff.values[b * o + c] = num[b * o + c] / den[b];
                    }
                }
                // Purify: subtract weighted mean of this feature's contribution into intercept.
                let mut mean_c = vec![0.0; o];
                for r in 0..n {
                    let w = weights[r].max(0.0);
                    let v = if x[j][r].is_finite() {
                        x[j][r]
                    } else {
                        impute[j]
                    };
                    let (b0, b1, t) = locate_interp(&eff.edges, v);
                    for c in 0..o {
                        let val = eff.values[b0 * o + c] * (1.0 - t) + eff.values[b1 * o + c] * t;
                        mean_c[c] += w * val / sum_w;
                    }
                }
                for c in 0..o {
                    intercept[c] += mean_c[c];
                    for b in 0..k {
                        eff.values[b * o + c] -= mean_c[c];
                    }
                }
                // Subtract BOTH purified effect AND mean_c shifted to intercept from residual (BUG-R2).
                for r in 0..n {
                    let v = if x[j][r].is_finite() {
                        x[j][r]
                    } else {
                        impute[j]
                    };
                    let (b0, b1, t) = locate_interp(&eff.edges, v);
                    for c in 0..o {
                        let val = eff.values[b0 * o + c] * (1.0 - t) + eff.values[b1 * o + c] * t;
                        res[r * o + c] -= val + mean_c[c];
                    }
                }
            }
        }
        Self {
            intercept,
            effects,
            outputs: o,
        }
    }

    pub fn predict_logits(&self, x: &[Vec<f64>], impute: &[f64]) -> Vec<f64> {
        let n = x.first().map_or(0, Vec::len);
        let o = self.outputs;
        let mut out = vec![0.0; n * o];
        for r in 0..n {
            for c in 0..o {
                out[r * o + c] = self.intercept[c];
            }
        }
        for eff in &self.effects {
            let j = eff.feature;
            for r in 0..n {
                let v = if x[j][r].is_finite() {
                    x[j][r]
                } else {
                    impute[j]
                };
                let (b0, b1, t) = locate_interp(&eff.edges, v);
                for c in 0..o {
                    out[r * o + c] +=
                        eff.values[b0 * o + c] * (1.0 - t) + eff.values[b1 * o + c] * t;
                }
            }
        }
        out
    }
}

pub(crate) fn targets_to_logits(head: Head, soft_targets: &[f64], n: usize) -> Vec<f64> {
    match head {
        Head::Regression => soft_targets.to_vec(),
        Head::Binary => soft_targets
            .iter()
            .map(|&p| {
                let pc = p.clamp(1e-4, 1.0 - 1e-4);
                (pc / (1.0 - pc)).ln()
            })
            .collect(),
        Head::Multiclass(k) => {
            let mut out = vec![0.0; n * k];
            for r in 0..n {
                let mut logs = vec![0.0; k];
                let mut mean = 0.0;
                for c in 0..k {
                    logs[c] = soft_targets[r * k + c].clamp(1e-4, 1.0).ln();
                    mean += logs[c] / k as f64;
                }
                for c in 0..k {
                    out[r * k + c] = logs[c] - mean;
                }
            }
            out
        }
    }
}

fn locate_interp(edges: &[f64], v: f64) -> (usize, usize, f64) {
    let k = edges.len();
    if v <= edges[0] {
        return (0, 0, 0.0);
    }
    if v >= edges[k - 1] {
        return (k - 1, k - 1, 0.0);
    }
    let idx = edges
        .partition_point(|&e| e <= v)
        .saturating_sub(1)
        .min(k - 2);
    let span = (edges[idx + 1] - edges[idx]).max(1e-12);
    (idx, idx + 1, ((v - edges[idx]) / span).clamp(0.0, 1.0))
}

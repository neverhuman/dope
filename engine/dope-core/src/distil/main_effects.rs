//! Purified piecewise-linear `target.main` across ALL feature columns (no 32-input cap).

use crate::target::{log_sum_exp, Head};
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
    /// on soft teacher targets (`MAIN_BINS` default knots).
    pub fn fit(
        head: Head,
        x: &[Vec<f64>],
        soft_targets: &[f64],
        weights: &[f64],
        impute: &[f64],
    ) -> Self {
        Self::fit_with_bins(head, x, soft_targets, weights, impute, MAIN_BINS)
    }

    /// Fit purified main effects with configurable knot budget (`max_bins` in `1..=MAIN_BINS`).
    pub fn fit_with_bins(
        head: Head,
        x: &[Vec<f64>],
        soft_targets: &[f64],
        weights: &[f64],
        impute: &[f64],
        max_bins: usize,
    ) -> Self {
        let bins = max_bins.clamp(1, MAIN_BINS);
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
        let sum_w: f64 = weights.iter().map(|&w| w.max(0.0)).sum::<f64>().max(1e-12);
        // Jensen-unbiased intercept initialization: logit(E_w[p]) for classification (BUG-R12).
        let mut intercept = vec![0.0; o];
        match head {
            Head::Regression => {
                intercept[0] = (0..n)
                    .map(|r| weights[r].max(0.0) * soft_targets[r])
                    .sum::<f64>()
                    / sum_w;
            }
            Head::Binary => {
                let mp = ((0..n)
                    .map(|r| weights[r].max(0.0) * soft_targets[r].clamp(0.0, 1.0))
                    .sum::<f64>()
                    / sum_w)
                    .clamp(1e-4, 1.0 - 1e-4);
                intercept[0] = (mp / (1.0 - mp)).ln();
            }
            Head::Multiclass(k) => {
                let mut logs = vec![0.0; k];
                let mut mean_l = 0.0;
                for c in 0..k {
                    let mp = ((0..n)
                        .map(|r| weights[r].max(0.0) * soft_targets[r * k + c].clamp(0.0, 1.0))
                        .sum::<f64>()
                        / sum_w)
                        .clamp(1e-4, 1.0);
                    logs[c] = mp.ln();
                    mean_l += logs[c] / k as f64;
                }
                for c in 0..k {
                    intercept[c] = logs[c] - mean_l;
                }
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
            let mut edges = Vec::with_capacity(bins + 1);
            for b in 0..=bins {
                let idx = (b * (vals.len() - 1)) / bins;
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

        // Track live linear predictor `f[r*o + c]` starting at `intercept`.
        let mut pred = vec![0.0; n * o];
        for r in 0..n {
            for c in 0..o {
                pred[r * o + c] = intercept[c];
            }
        }
        // Regression: damped weighted Gauss-Seidel; Classification: Fisher-weighted IRLS Newton (BUG-R12).
        // True L2 ridge + 1st-difference P-spline roughness penalty (`g += l2*v + smooth*Δ²v`)
        // prevents collinear explosion and wide-table small-n memorization!
        let passes = if head == Head::Regression { 6 } else { 8 };
        let eta = if head == Head::Regression { 0.80 } else { 0.60 };
        let l2 = if head == Head::Regression {
            2.0 + 0.35 * d as f64
        } else {
            3.0 + 0.45 * d as f64
        };
        let smooth = if head == Head::Regression { 1.5 } else { 2.5 };
        for _ in 0..passes {
            for eff in &mut effects {
                let j = eff.feature;
                let k = eff.edges.len();
                let mut g_num = vec![0.0; k * o];
                let mut h_den = vec![1e-6; k * o];
                for b in 0..k {
                    for c in 0..o {
                        let idx = b * o + c;
                        let v = eff.values[idx];
                        g_num[idx] += l2 * v;
                        h_den[idx] += l2;
                        if b > 0 {
                            g_num[idx] += smooth * (v - eff.values[(b - 1) * o + c]);
                            h_den[idx] += smooth;
                        }
                        if b + 1 < k {
                            g_num[idx] += smooth * (v - eff.values[(b + 1) * o + c]);
                            h_den[idx] += smooth;
                        }
                    }
                }
                for r in 0..n {
                    let w = weights[r].max(0.0);
                    let v = if x[j][r].is_finite() {
                        x[j][r]
                    } else {
                        impute[j]
                    };
                    let (b0, b1, t) = locate_interp(&eff.edges, v);
                    let (w0, w1) = (w * (1.0 - t), w * t);
                    match head {
                        Head::Regression => {
                            let err = pred[r] - soft_targets[r];
                            g_num[b0] += w0 * err;
                            g_num[b1] += w1 * err;
                            h_den[b0] += w0;
                            h_den[b1] += w1;
                        }
                        Head::Binary => {
                            let z = pred[r].clamp(-10.0, 10.0);
                            let p = 1.0 / (1.0 + (-z).exp());
                            let err = p - soft_targets[r].clamp(0.0, 1.0);
                            let h = (p * (1.0 - p)).max(0.02);
                            g_num[b0] += w0 * err;
                            g_num[b1] += w1 * err;
                            h_den[b0] += w0 * h;
                            h_den[b1] += w1 * h;
                        }
                        Head::Multiclass(nc) => {
                            let row = &pred[r * nc..(r + 1) * nc];
                            let lse = log_sum_exp(row);
                            for c in 0..nc {
                                let p = (row[c] - lse).exp();
                                let err = p - soft_targets[r * nc + c].clamp(0.0, 1.0);
                                let h = (p * (1.0 - p)).max(0.02);
                                g_num[b0 * nc + c] += w0 * err;
                                g_num[b1 * nc + c] += w1 * err;
                                h_den[b0 * nc + c] += w0 * h;
                                h_den[b1 * nc + c] += w1 * h;
                            }
                        }
                    }
                }
                let mut delta = vec![0.0; k * o];
                for idx in 0..(k * o) {
                    let raw_d = -eta * g_num[idx] / h_den[idx];
                    let next_v = if head == Head::Regression {
                        eff.values[idx] + raw_d
                    } else {
                        (eff.values[idx] + raw_d.clamp(-1.5, 1.5)).clamp(-4.0, 4.0)
                    };
                    delta[idx] = next_v - eff.values[idx];
                    eff.values[idx] = next_v;
                }
                // Update `pred` with `delta`, then purify weighted mean into `intercept` (leaves `pred` exact).
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
                        pred[r * o + c] += delta[b0 * o + c] * (1.0 - t) + delta[b1 * o + c] * t;
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

/// L2-regularized classification temperature + intercept calibration on held-out `CAL` (`BUG-R13`).
pub(crate) fn fit_temperature_and_bias(head: Head, logits: &[f64], y: &[f64]) -> (f64, Vec<f64>) {
    let n = y.len();
    let o = head.outputs();
    let mut best_t = 1.0;
    let mut best_b = vec![0.0; o];
    let mut best_nll = f64::INFINITY;
    for &t in &[0.5_f64, 0.65, 0.8, 1.0, 1.25, 1.6, 2.2] {
        let mut b = vec![0.0; o];
        let t_prior = 0.5 * t.ln().powi(2);
        match head {
            Head::Binary => {
                let reg_b = 3.0_f64;
                for _ in 0..8 {
                    let (mut g, mut h) = (reg_b * b[0], reg_b);
                    for r in 0..n {
                        let z = (logits[r] / t + b[0]).clamp(-12.0, 12.0);
                        let p = 1.0 / (1.0 + (-z).exp());
                        g += p - y[r].clamp(0.0, 1.0);
                        h += (p * (1.0 - p)).max(1e-4);
                    }
                    b[0] = (b[0] - 0.7 * g / h).clamp(-1.5, 1.5);
                }
                let mut nll = t_prior + 0.5 * reg_b * b[0] * b[0];
                for r in 0..n {
                    let z = (logits[r] / t + b[0]).clamp(-12.0, 12.0);
                    let ce = if z >= 0.0 {
                        z + (-z).exp().ln_1p()
                    } else {
                        z.exp().ln_1p()
                    } - y[r].clamp(0.0, 1.0) * z;
                    nll += ce;
                }
                if nll < best_nll {
                    best_nll = nll;
                    best_t = t;
                    best_b = b;
                }
            }
            Head::Multiclass(k) => {
                let reg_b = 5.0_f64;
                for _ in 0..8 {
                    let mut g: Vec<f64> = b.iter().map(|&v| reg_b * v).collect();
                    let mut h = vec![reg_b; k];
                    for r in 0..n {
                        let row: Vec<f64> = (0..k).map(|c| logits[r * o + c] / t + b[c]).collect();
                        let lse = log_sum_exp(&row);
                        let yc = (y[r].round() as usize).min(k - 1);
                        for c in 0..k {
                            let p = (row[c] - lse).exp();
                            g[c] += p - if c == yc { 1.0 } else { 0.0 };
                            h[c] += (p * (1.0 - p)).max(1e-4);
                        }
                    }
                    let mut mean_b = 0.0;
                    for c in 0..k {
                        b[c] = (b[c] - 0.65 * g[c] / h[c]).clamp(-1.5, 1.5);
                        mean_b += b[c] / k as f64;
                    }
                    for v in &mut b {
                        *v -= mean_b;
                    }
                }
                let mut nll = t_prior + 0.5 * reg_b * b.iter().map(|v| v * v).sum::<f64>();
                for r in 0..n {
                    let row: Vec<f64> = (0..k).map(|c| logits[r * o + c] / t + b[c]).collect();
                    let lse = log_sum_exp(&row);
                    let c = (y[r].round() as usize).min(k - 1);
                    nll += lse - row[c];
                }
                if nll < best_nll {
                    best_nll = nll;
                    best_t = t;
                    best_b = b;
                }
            }
            Head::Regression => return (1.0, vec![0.0]),
        }
    }
    (best_t, best_b)
}

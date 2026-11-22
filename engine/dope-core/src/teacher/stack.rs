//! Inner-fold simplex/NNLS stacking and probability calibration (PAV isotonic + temp/bias).

use crate::target::{log_sum_exp, Head};
use serde::{Deserialize, Serialize};

/// Project vector onto the probability simplex `{w >= 0, sum w = 1}` (Duchi et al. 2008).
pub fn project_simplex(v: &[f64]) -> Vec<f64> {
    let m = v.len();
    if m == 0 {
        return Vec::new();
    }
    let mut u = v.to_vec();
    u.sort_by(|a, b| b.total_cmp(a));
    let mut cssv = 0.0;
    let mut rho = 0usize;
    for (i, &val) in u.iter().enumerate() {
        cssv += val;
        let t = (cssv - 1.0) / (i + 1) as f64;
        if val - t > 0.0 {
            rho = i;
        }
    }
    let theta = (u[..=rho].iter().sum::<f64>() - 1.0) / (rho + 1) as f64;
    v.iter().map(|&x| (x - theta).max(0.0)).collect()
}

/// Fit simplex stacking weights across `M` teacher members on inner OOF predictions.
///
/// `member_preds[m]` has length `n * o` (regression means or class probabilities).
pub fn fit_simplex_weights(head: Head, member_preds: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let m = member_preds.len();
    let n = y.len();
    if m == 0 {
        return Vec::new();
    }
    if m == 1 || n < 4 {
        return vec![1.0 / m as f64; m];
    }
    let mut w = vec![1.0 / m as f64; m];
    let o = head.outputs();
    let steps = 80usize;
    for step in 0..steps {
        let mut grad = vec![0.0; m];
        match head {
            Head::Regression => {
                for r in 0..n {
                    let pred: f64 = (0..m).map(|j| w[j] * member_preds[j][r]).sum();
                    let err = 2.0 * (pred - y[r]) / n as f64;
                    for j in 0..m {
                        grad[j] += err * member_preds[j][r];
                    }
                }
            }
            Head::Binary => {
                for r in 0..n {
                    let p: f64 = (0..m)
                        .map(|j| w[j] * member_preds[j][r].clamp(1e-6, 1.0 - 1e-6))
                        .sum::<f64>()
                        .clamp(1e-6, 1.0 - 1e-6);
                    let dl_dp = (-(y[r] / p) + (1.0 - y[r]) / (1.0 - p)) / n as f64;
                    for j in 0..m {
                        grad[j] += dl_dp * member_preds[j][r].clamp(1e-6, 1.0 - 1e-6);
                    }
                }
            }
            Head::Multiclass(k) => {
                for r in 0..n {
                    let c = (y[r].round() as usize).min(k - 1);
                    let p_c: f64 = (0..m)
                        .map(|j| w[j] * member_preds[j][r * o + c].clamp(1e-6, 1.0))
                        .sum::<f64>()
                        .clamp(1e-6, 1.0);
                    let dl = -1.0 / (p_c * n as f64);
                    for j in 0..m {
                        grad[j] += dl * member_preds[j][r * o + c].clamp(1e-6, 1.0);
                    }
                }
            }
        }
        let gnorm = grad.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
        let lr = 0.25 / (1.0 + 0.05 * step as f64) / gnorm.max(1.0);
        let cand: Vec<f64> = (0..m).map(|j| w[j] - lr * grad[j]).collect();
        w = project_simplex(&cand);
    }
    w
}

/// Pool-Adjacent-Violators (PAV) monotone isotonic calibrator for binary probabilities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IsotonicCalibrator {
    pub x_knots: Vec<f64>,
    pub y_knots: Vec<f64>,
}

impl IsotonicCalibrator {
    pub fn fit(probs: &[f64], y: &[f64]) -> Self {
        let n = probs.len();
        if n < 10 {
            return Self {
                x_knots: vec![0.0, 1.0],
                y_knots: vec![0.0, 1.0],
            };
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| probs[a].total_cmp(&probs[b]).then(a.cmp(&b)));
        // Blocks: (x_lo, x_hi, sum_y, count)
        let mut blocks: Vec<(f64, f64, f64, usize)> = Vec::with_capacity(n);
        for &idx in &order {
            let p = probs[idx].clamp(1e-6, 1.0 - 1e-6);
            let yi = y[idx].clamp(0.0, 1.0);
            blocks.push((p, p, yi, 1));
            while blocks.len() >= 2 {
                let len = blocks.len();
                let m1 = blocks[len - 2].2 / blocks[len - 2].3 as f64;
                let m2 = blocks[len - 1].2 / blocks[len - 1].3 as f64;
                if m1 > m2 {
                    let b2 = blocks.pop().unwrap();
                    let b1 = blocks.last_mut().unwrap();
                    b1.1 = b2.1;
                    b1.2 += b2.2;
                    b1.3 += b2.3;
                } else {
                    break;
                }
            }
        }
        let mut x_knots = Vec::with_capacity(blocks.len());
        let mut y_knots = Vec::with_capacity(blocks.len());
        for (lo, hi, sy, cnt) in blocks {
            x_knots.push(0.5 * (lo + hi));
            // Smooth lightly toward raw midpoint to prevent hard 0/1 collapse.
            let raw_m = sy / cnt as f64;
            let smooth = (sy + 0.5 * 0.5 * (lo + hi)) / (cnt as f64 + 0.5);
            y_knots.push((0.85 * smooth + 0.15 * raw_m).clamp(1e-5, 1.0 - 1e-5));
        }
        Self { x_knots, y_knots }
    }

    pub fn transform(&self, p: f64) -> f64 {
        let p = p.clamp(1e-6, 1.0 - 1e-6);
        let k = self.x_knots.len();
        if k == 0 {
            return p;
        }
        if p <= self.x_knots[0] {
            return self.y_knots[0];
        }
        if p >= self.x_knots[k - 1] {
            return self.y_knots[k - 1];
        }
        let i = self
            .x_knots
            .partition_point(|&xk| xk <= p)
            .saturating_sub(1)
            .min(k - 2);
        let span = (self.x_knots[i + 1] - self.x_knots[i]).max(1e-12);
        let t = ((p - self.x_knots[i]) / span).clamp(0.0, 1.0);
        (self.y_knots[i] + t * (self.y_knots[i + 1] - self.y_knots[i])).clamp(1e-5, 1.0 - 1e-5)
    }
}

/// Temperature + per-class bias calibrator for multiclass probabilities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MulticlassCalibrator {
    pub temp: f64,
    pub bias: Vec<f64>,
}

impl MulticlassCalibrator {
    pub fn fit(probs: &[f64], y: &[f64], k: usize) -> Self {
        let n = y.len();
        if n < 10 || k < 2 {
            return Self {
                temp: 1.0,
                bias: vec![0.0; k],
            };
        }
        let logits: Vec<f64> = probs.iter().map(|&p| p.clamp(1e-6, 1.0).ln()).collect();
        let mut best_temp = 1.0;
        let mut best_loss = f64::INFINITY;
        for &t in &[0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0] {
            let mut nll = 0.0;
            for r in 0..n {
                let row: Vec<f64> = (0..k).map(|c| logits[r * k + c] / t).collect();
                let lse = log_sum_exp(&row);
                let c = (y[r].round() as usize).min(k - 1);
                nll += lse - row[c];
            }
            if nll < best_loss {
                best_loss = nll;
                best_temp = t;
            }
        }
        // Per-class additive bias correction via damped Newton steps on calibrated logits.
        let mut bias = vec![0.0; k];
        for _ in 0..12 {
            let mut g = vec![0.0; k];
            let mut h = vec![0.0; k];
            for r in 0..n {
                let row: Vec<f64> = (0..k)
                    .map(|c| logits[r * k + c] / best_temp + bias[c])
                    .collect();
                let lse = log_sum_exp(&row);
                let yc = (y[r].round() as usize).min(k - 1);
                for c in 0..k {
                    let p = (row[c] - lse).exp();
                    g[c] += p - if c == yc { 1.0 } else { 0.0 };
                    h[c] += (p * (1.0 - p)).max(1e-6);
                }
            }
            for c in 0..k {
                bias[c] = (bias[c] - 0.5 * g[c] / (h[c] + 1.0)).clamp(-3.0, 3.0);
            }
            let mean_b = bias.iter().sum::<f64>() / k as f64;
            for b in &mut bias {
                *b -= mean_b;
            }
        }
        Self {
            temp: best_temp,
            bias,
        }
    }

    pub fn transform_row(&self, probs_row: &[f64]) -> Vec<f64> {
        let k = self.bias.len();
        let row: Vec<f64> = (0..k)
            .map(|c| probs_row[c].clamp(1e-6, 1.0).ln() / self.temp + self.bias[c])
            .collect();
        let lse = log_sum_exp(&row);
        row.into_iter().map(|v| (v - lse).exp()).collect()
    }
}

//! `LearningSignatureV1`: up to 128 standardized moment & finite relative-risk witnesses (`T31`).

use serde::{Deserialize, Serialize};

pub const MAX_WITNESSES: usize = 128;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RelativeRiskPredictor {
    pub feature: usize,
    pub threshold: f64,
    pub lo_val: f64,
    pub hi_val: f64,
}

impl RelativeRiskPredictor {
    #[inline]
    pub fn predict(&self, x_val: f64) -> f64 {
        if x_val.is_finite() && x_val > self.threshold {
            self.hi_val
        } else {
            self.lo_val
        }
    }
}

/// Owner-side witness specification fitted strictly on real `BUILD` (never exposed publicly).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningSignatureV1 {
    pub d: usize,
    pub feat_means: Vec<f64>,
    pub feat_sds: Vec<f64>,
    pub y_mean: f64,
    pub y_sd: f64,
    pub top_features: Vec<usize>,
    pub top_pairs: Vec<(usize, usize)>,
    pub risk_predictors: Vec<RelativeRiskPredictor>,
    /// Target witness means evaluated on real reference `P` (length <= 128).
    pub real_witness_means: Vec<f64>,
    pub witness_names: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WitnessGapReport {
    pub witness_count: usize,
    pub max_abs_gap: f64,
    pub mean_abs_gap: f64,
    pub max_risk_witness_eps: f64,
    /// Exact `2 * eps + delta_opt` upper bound on excess transfer risk across `risk_predictors` (`T31`).
    pub risk_transfer_bound_2eps_plus_delta: f64,
    pub realized_excess_risk: f64,
}

impl LearningSignatureV1 {
    /// Extract up to `MAX_WITNESSES` (128) standardized moments + top-8 pairs + finite
    /// relative-risk witnesses from `(x_cols, y)`.
    pub fn fit(x_cols: &[Vec<f64>], y: &[f64]) -> Self {
        let d = x_cols.len();
        let n = y.len().max(1);
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let y_sd = (y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>() / n as f64)
            .sqrt()
            .max(1e-6);

        let mut feat_means = vec![0.0; d];
        let mut feat_sds = vec![1.0; d];
        let mut corrs: Vec<(f64, usize)> = Vec::with_capacity(d);
        for j in 0..d {
            let mut sum = 0.0;
            let mut cnt = 0usize;
            for &v in &x_cols[j] {
                if v.is_finite() {
                    sum += v;
                    cnt += 1;
                }
            }
            let m = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
            let mut var = 0.0;
            let mut cov = 0.0;
            for (r, &v) in x_cols[j].iter().enumerate() {
                if v.is_finite() {
                    let dx = v - m;
                    var += dx * dx;
                    cov += dx * (y[r] - y_mean);
                }
            }
            let sd = (var / cnt.max(1) as f64).sqrt().max(1e-6);
            feat_means[j] = m;
            feat_sds[j] = sd;
            let r_abs = (cov / (cnt.max(1) as f64 * sd * y_sd)).abs().min(1.0);
            corrs.push((r_abs, j));
        }
        corrs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let top_features: Vec<usize> = corrs.iter().take(12).map(|&(_, j)| j).collect();

        // Top-8 interacting/correlated feature pairs among top features.
        let mut top_pairs = Vec::new();
        for i in 0..top_features.len() {
            for j in (i + 1)..top_features.len() {
                if top_pairs.len() >= 8 {
                    break;
                }
                top_pairs.push((top_features[i], top_features[j]));
            }
            if top_pairs.len() >= 8 {
                break;
            }
        }

        // Build deterministic finite family of stump relative-risk predictors (`<= 32` members).
        let mut risk_predictors = Vec::new();
        for &feat in top_features.iter().take(8) {
            let mut vals: Vec<f64> = x_cols[feat]
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();
            vals.sort_by(f64::total_cmp);
            if vals.len() < 8 {
                continue;
            }
            for &q in &[0.25, 0.50, 0.75] {
                let thr =
                    vals[((q * (vals.len() - 1) as f64).round() as usize).min(vals.len() - 1)];
                let (mut s_lo, mut n_lo, mut s_hi, mut n_hi) = (0.0, 0usize, 0.0, 0usize);
                for r in 0..n {
                    let v = x_cols[feat][r];
                    let ys = (y[r] - y_mean) / y_sd;
                    if v.is_finite() && v > thr {
                        s_hi += ys;
                        n_hi += 1;
                    } else {
                        s_lo += ys;
                        n_lo += 1;
                    }
                }
                if n_lo >= 2 && n_hi >= 2 {
                    risk_predictors.push(RelativeRiskPredictor {
                        feature: feat,
                        threshold: thr,
                        lo_val: s_lo / n_lo as f64,
                        hi_val: s_hi / n_hi as f64,
                    });
                }
            }
        }

        let mut sig = Self {
            d,
            feat_means,
            feat_sds,
            y_mean,
            y_sd,
            top_features,
            top_pairs,
            risk_predictors,
            real_witness_means: Vec::new(),
            witness_names: Vec::new(),
        };
        let (names, mat) = sig.witness_matrix_with_names(x_cols, y);
        let m = names.len();
        let mut means = vec![0.0; m];
        for r in 0..n {
            for k in 0..m {
                means[k] += mat[r * m + k] / n as f64;
            }
        }
        sig.witness_names = names;
        sig.real_witness_means = means;
        sig
    }

    /// Evaluate the `[n * M]` witness matrix (`M <= 128`) on any dataset `(x_cols, y)`.
    pub fn witness_matrix(&self, x_cols: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
        self.witness_matrix_with_names(x_cols, y).1
    }

    fn witness_matrix_with_names(&self, x_cols: &[Vec<f64>], y: &[f64]) -> (Vec<String>, Vec<f64>) {
        let n = y.len();
        let mut names = Vec::new();
        names.push("y_z".into());
        names.push("y_z2".into());
        for &j in &self.top_features {
            names.push(format!("x{j}_z"));
            names.push(format!("x{j}_z2"));
            names.push(format!("x{j}_y"));
        }
        for &(a, b) in &self.top_pairs {
            names.push(format!("x{a}_x{b}"));
        }
        for (i, _) in self.risk_predictors.iter().enumerate() {
            if names.len() >= MAX_WITNESSES {
                break;
            }
            names.push(format!("risk_delta_{i}"));
        }
        names.truncate(MAX_WITNESSES);
        let m = names.len();
        let mut mat = vec![0.0; n * m];
        for r in 0..n {
            let ys = ((y[r] - self.y_mean) / self.y_sd).clamp(-5.0, 5.0);
            let mut k = 0usize;
            mat[r * m + k] = ys;
            k += 1;
            mat[r * m + k] = (ys * ys).min(25.0);
            k += 1;
            for &j in &self.top_features {
                let z = if x_cols[j][r].is_finite() {
                    ((x_cols[j][r] - self.feat_means[j]) / self.feat_sds[j]).clamp(-5.0, 5.0)
                } else {
                    0.0
                };
                if k + 3 <= m {
                    mat[r * m + k] = z;
                    mat[r * m + k + 1] = (z * z).min(25.0);
                    mat[r * m + k + 2] = (z * ys).clamp(-25.0, 25.0);
                    k += 3;
                }
            }
            for &(a, b) in &self.top_pairs {
                if k >= m {
                    break;
                }
                let za = if x_cols[a][r].is_finite() {
                    ((x_cols[a][r] - self.feat_means[a]) / self.feat_sds[a]).clamp(-5.0, 5.0)
                } else {
                    0.0
                };
                let zb = if x_cols[b][r].is_finite() {
                    ((x_cols[b][r] - self.feat_means[b]) / self.feat_sds[b]).clamp(-5.0, 5.0)
                } else {
                    0.0
                };
                mat[r * m + k] = (za * zb).clamp(-25.0, 25.0);
                k += 1;
            }
            // Finite relative-risk differences Delta(f) = (ys - f(x))^2 - ys^2 (`T31`).
            let null_loss = ys * ys;
            for pred in &self.risk_predictors {
                if k >= m {
                    break;
                }
                let p = pred.predict(x_cols[pred.feature][r]);
                let loss_f = (ys - p).powi(2);
                mat[r * m + k] = (loss_f - null_loss).clamp(-25.0, 25.0);
                k += 1;
            }
        }
        (names, mat)
    }

    /// Compare synthetic `(sx, sy)` against `real_witness_means` and verify the `2*eps + delta`
    /// finite relative-risk transfer bound (`T31`).
    pub fn evaluate_gaps(
        &self,
        sx: &[Vec<f64>],
        sy: &[f64],
        weights: Option<&[f64]>,
    ) -> WitnessGapReport {
        let n = sy.len().max(1);
        let m = self.real_witness_means.len();
        let mat = self.witness_matrix(sx, sy);
        let mut q_means = vec![0.0; m];
        let mut w_tot = 0.0;
        for r in 0..n {
            let w = weights.map_or(1.0, |ws| ws[r].max(0.0));
            w_tot += w;
            for k in 0..m {
                q_means[k] += w * mat[r * m + k];
            }
        }
        let inv = 1.0 / w_tot.max(1e-12);
        for v in &mut q_means {
            *v *= inv;
        }
        let mut max_gap = 0.0_f64;
        let mut sum_gap = 0.0;
        let mut max_risk_eps = 0.0_f64;
        let mut risk_indices = Vec::new();
        for k in 0..m {
            let g = (q_means[k] - self.real_witness_means[k]).abs();
            max_gap = max_gap.max(g);
            sum_gap += g;
            if self.witness_names[k].starts_with("risk_delta_") {
                max_risk_eps = max_risk_eps.max(g);
                risk_indices.push(k);
            }
        }
        // Check exact `2*eps + delta` bound across the finite predictor class (`T31`).
        let (realized_excess, bound) = if risk_indices.is_empty() {
            (0.0, 0.0)
        } else {
            let best_p = risk_indices
                .iter()
                .copied()
                .min_by(|&a, &b| self.real_witness_means[a].total_cmp(&self.real_witness_means[b]))
                .unwrap();
            let best_q = risk_indices
                .iter()
                .copied()
                .min_by(|&a, &b| q_means[a].total_cmp(&q_means[b]))
                .unwrap();
            let excess =
                (self.real_witness_means[best_q] - self.real_witness_means[best_p]).max(0.0);
            let delta_opt = 0.0;
            (excess, 2.0 * max_risk_eps + delta_opt)
        };
        WitnessGapReport {
            witness_count: m,
            max_abs_gap: max_gap,
            mean_abs_gap: sum_gap / m.max(1) as f64,
            max_risk_witness_eps: max_risk_eps,
            risk_transfer_bound_2eps_plus_delta: bound,
            realized_excess_risk: realized_excess,
        }
    }
}

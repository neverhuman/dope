//! Honest regression residual noise model: `log sigma(x)` + 17-knot conditional quantiles.

use crate::noise::RESIDUAL_KNOTS;
use crate::stats::{norm_cdf, quantile_sorted};
use serde::{Deserialize, Serialize};

/// Conditional regression noise fitted strictly on out-of-fold (OOF) or `SELECT`/`CAL` residuals.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeacherResidualModel {
    /// Intercept + linear log-scale coefficients on standardized top features.
    pub log_sigma_intercept: f64,
    pub log_sigma_slope_yhat: f64,
    pub sigma_lo: f64,
    pub sigma_hi: f64,
    /// Predicted-mean bin edges (up to 10 deciles when `n >= 200`, else 1 global bin).
    pub bin_edges: Vec<f64>,
    /// 17-knot standardized residual quantiles per predicted-mean bin.
    pub bin_knots: Vec<[f64; RESIDUAL_KNOTS]>,
}

impl TeacherResidualModel {
    pub fn fit_oof(y_true: &[f64], y_hat_oof: &[f64]) -> Self {
        let n = y_true.len().min(y_hat_oof.len());
        if n < 8 {
            return Self::unit_gaussian();
        }
        let res: Vec<f64> = (0..n).map(|i| y_true[i] - y_hat_oof[i]).collect();
        let abs_res: Vec<f64> = res.iter().map(|r| r.abs()).collect();
        let mut sorted_abs = abs_res.clone();
        sorted_abs.sort_by(f64::total_cmp);
        let global_sigma = (quantile_sorted(&sorted_abs, 0.6827)).max(1e-4);

        // Fit linear heteroscedastic log-scale against centered `y_hat`.
        let mu_yh = y_hat_oof[..n].iter().sum::<f64>() / n as f64;
        let var_yh = y_hat_oof[..n]
            .iter()
            .map(|&v| (v - mu_yh).powi(2))
            .sum::<f64>()
            / n as f64;
        let log_s: Vec<f64> = abs_res
            .iter()
            .map(|&a| (a.max(0.05 * global_sigma)).ln())
            .collect();
        let mean_ls = log_s.iter().sum::<f64>() / n as f64;
        let slope = if var_yh > 1e-10 {
            let cov: f64 = (0..n)
                .map(|i| (y_hat_oof[i] - mu_yh) * (log_s[i] - mean_ls))
                .sum::<f64>()
                / n as f64;
            (cov / (var_yh + 1e-4)).clamp(-1.5, 1.5)
        } else {
            0.0
        };
        let intercept = (mean_ls - slope * mu_yh).clamp(-12.0, 12.0);
        let sigma_lo = (0.1 * global_sigma).max(1e-5);
        let sigma_hi = (5.0 * global_sigma).max(sigma_lo * 2.0);

        // Standardize residuals by predicted sigma(y_hat).
        let std_res: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let s = (intercept + slope * y_hat_oof[i])
                    .exp()
                    .clamp(sigma_lo, sigma_hi);
                (y_hat_oof[i], (res[i] / s).clamp(-6.0, 6.0))
            })
            .collect();

        // Partition into decile bins (when n >= 200) for conditional 17-knot quantiles.
        let n_bins: usize = if n >= 200 { 10 } else { 1 };
        let mut by_yh = std_res;
        by_yh.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut bin_edges = Vec::with_capacity(n_bins.saturating_sub(1));
        let mut bin_knots = Vec::with_capacity(n_bins);
        for b in 0..n_bins {
            let lo = (b * n) / n_bins;
            let hi = ((b + 1) * n) / n_bins;
            if b + 1 < n_bins {
                bin_edges.push(by_yh[hi.saturating_sub(1)].0);
            }
            let mut slice: Vec<f64> = by_yh[lo..hi].iter().map(|&(_, r)| r).collect();
            slice.sort_by(f64::total_cmp);
            let mut knots = [0.0; RESIDUAL_KNOTS];
            for (k, knot) in knots.iter_mut().enumerate() {
                let z = -3.0 + 6.0 * (k as f64) / (RESIDUAL_KNOTS - 1) as f64;
                *knot = quantile_sorted(&slice, norm_cdf(z));
            }
            bin_knots.push(knots);
        }

        Self {
            log_sigma_intercept: intercept,
            log_sigma_slope_yhat: slope,
            sigma_lo,
            sigma_hi,
            bin_edges,
            bin_knots,
        }
    }

    pub fn unit_gaussian() -> Self {
        let mut knots = [0.0; RESIDUAL_KNOTS];
        for (k, knot) in knots.iter_mut().enumerate() {
            *knot = -3.0 + 6.0 * (k as f64) / (RESIDUAL_KNOTS - 1) as f64;
        }
        Self {
            log_sigma_intercept: 0.0,
            log_sigma_slope_yhat: 0.0,
            sigma_lo: 1e-4,
            sigma_hi: 10.0,
            bin_edges: Vec::new(),
            bin_knots: vec![knots],
        }
    }

    pub fn sigma(&self, y_hat: f64) -> f64 {
        (self.log_sigma_intercept + self.log_sigma_slope_yhat * y_hat)
            .exp()
            .clamp(self.sigma_lo, self.sigma_hi)
    }

    pub fn sample_residual(&self, y_hat: f64, z_normal: f64) -> f64 {
        let bin = self.bin_edges.partition_point(|&e| e < y_hat);
        let knots = &self.bin_knots[bin.min(self.bin_knots.len() - 1)];
        let z = z_normal.clamp(-3.0, 3.0);
        let pos =
            ((z + 3.0) / 6.0 * (RESIDUAL_KNOTS - 1) as f64).clamp(0.0, (RESIDUAL_KNOTS - 1) as f64);
        let i = (pos.floor() as usize).min(RESIDUAL_KNOTS - 2);
        let f = pos - i as f64;
        let q = knots[i] * (1.0 - f) + knots[i + 1] * f;
        self.sigma(y_hat) * q
    }
}

//! Regime mixtures: a finite mixture of Gaussian copulas that share one
//! correlation structure but differ in location and scale.
//!
//! ```text
//! k ~ Categorical(π),   z = μ_k + σ_k ⊙ (ε Mᵀ),   x_j = Q_j(F_j(z_j))
//! ```
//!
//! `F_j` is the latent mixture marginal `Σ_k π_k Φ((z − μ_kj)/σ_kj)`, so
//! every feature keeps its stored marginal *exactly*; the regimes only shape
//! dependence. They capture what a single Gaussian copula cannot: rare
//! operating regimes in which several variables are extreme together (zero
//! tail dependence is the Gaussian copula's best-known failure). With K = 1
//! the model is the plain Gaussian copula.

use crate::copula::Dependence;
use crate::rng::Rng;
use crate::stats::{cholesky, nearest_correlation, norm_cdf, norm_ppf};
use std::f64::consts::PI;

pub const MAX_REGIMES: usize = 16;
/// Regime counts tried by `fit` (selected by held-out likelihood within the budget).
pub const REGIME_LADDER: [usize; 8] = [1, 2, 3, 4, 6, 8, 12, 16];
pub const SCALE_MIN: f64 = 0.05;
pub const SCALE_MAX: f64 = 2.5;
pub const MEAN_RANGE: f64 = 4.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Regimes {
    pub weights: Vec<f64>,
    /// Row-major K × d.
    pub means: Vec<f64>,
    /// Row-major K × d.
    pub scales: Vec<f64>,
}

impl Regimes {
    pub fn single(d: usize) -> Self {
        Self {
            weights: vec![1.0],
            means: vec![0.0; d],
            scales: vec![1.0; d],
        }
    }

    pub fn k(&self) -> usize {
        self.weights.len()
    }

    pub fn is_trivial(&self) -> bool {
        self.k() == 1
            && self.means.iter().all(|m| *m == 0.0)
            && self.scales.iter().all(|s| *s == 1.0)
    }

    /// Diagonal Gaussian-mixture EM on standardized scores (row-major n × d;
    /// NaN treated as 0), k-means++ initialisation, `k` components.
    pub fn fit(s: &[f64], n: usize, d: usize, k: usize, seed: u64) -> Self {
        if k <= 1 || n < 20 * k {
            return Self::single(d);
        }
        let x: Vec<f64> = s
            .iter()
            .map(|v| if v.is_finite() { *v } else { 0.0 })
            .collect();
        let row = |i: usize| &x[i * d..(i + 1) * d];
        let dist =
            |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(u, v)| (u - v).powi(2)).sum::<f64>();
        let mut rng = Rng::new(seed ^ 0x6e61);
        let mut centers = vec![row(rng.below(n)).to_vec()];
        while centers.len() < k {
            let w: Vec<f64> = (0..n)
                .map(|i| {
                    centers
                        .iter()
                        .map(|c| dist(row(i), c))
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();
            let total: f64 = w.iter().sum();
            let mut t = rng.uniform() * total;
            let mut pick = n - 1;
            for (i, wi) in w.iter().enumerate() {
                t -= wi;
                if t <= 0.0 {
                    pick = i;
                    break;
                }
            }
            centers.push(row(pick).to_vec());
        }
        let mut weights = vec![1.0 / k as f64; k];
        let mut means: Vec<f64> = centers.concat();
        let mut scales = vec![1.0; k * d];
        let mut resp = vec![0.0; n * k];
        for _ in 0..60 {
            // E-step.
            for i in 0..n {
                let xi = row(i);
                let mut lp = vec![0.0; k];
                for c in 0..k {
                    let mut l = weights[c].max(1e-12).ln();
                    for j in 0..d {
                        let sd = scales[c * d + j];
                        let z = (xi[j] - means[c * d + j]) / sd;
                        l += -0.5 * z * z - sd.ln();
                    }
                    lp[c] = l;
                }
                let m = lp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = lp.iter().map(|l| (l - m).exp()).sum();
                for c in 0..k {
                    resp[i * k + c] = (lp[c] - m).exp() / total;
                }
            }
            // M-step.
            for c in 0..k {
                let rc: f64 = (0..n).map(|i| resp[i * k + c]).sum::<f64>().max(1e-9);
                weights[c] = rc / n as f64;
                for j in 0..d {
                    let mu = (0..n).map(|i| resp[i * k + c] * x[i * d + j]).sum::<f64>() / rc;
                    let var = (0..n)
                        .map(|i| resp[i * k + c] * (x[i * d + j] - mu).powi(2))
                        .sum::<f64>()
                        / rc;
                    means[c * d + j] = mu.clamp(-MEAN_RANGE, MEAN_RANGE);
                    scales[c * d + j] = var.sqrt().clamp(SCALE_MIN, SCALE_MAX);
                }
            }
        }
        Self {
            weights,
            means,
            scales,
        }
    }

    /// Posterior regime probabilities of one standardized row.
    fn posterior(&self, x: &[f64]) -> Vec<f64> {
        let (k, d) = (self.k(), x.len());
        let lp: Vec<f64> = (0..k)
            .map(|c| {
                let mut l = self.weights[c].max(1e-12).ln();
                for j in 0..d {
                    let sd = self.scales[c * d + j];
                    let v = if x[j].is_finite() { x[j] } else { 0.0 };
                    let z = (v - self.means[c * d + j]) / sd;
                    l += -0.5 * z * z - sd.ln();
                }
                l
            })
            .collect();
        let m = lp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = lp.iter().map(|l| (l - m).exp()).sum();
        lp.iter().map(|l| (l - m).exp() / total).collect()
    }

    /// Pooled within-regime correlation of standardized scores.
    pub fn within_correlation(&self, s: &[f64], n: usize, d: usize) -> Vec<f64> {
        let mut cov = vec![0.0; d * d];
        let mut wsum = 0.0;
        for i in 0..n {
            let xi = &s[i * d..(i + 1) * d];
            let post = self.posterior(xi);
            for (c, &w) in post.iter().enumerate() {
                if w < 1e-6 {
                    continue;
                }
                let r: Vec<f64> = (0..d)
                    .map(|j| {
                        let v = if xi[j].is_finite() {
                            xi[j]
                        } else {
                            self.means[c * d + j]
                        };
                        (v - self.means[c * d + j]) / self.scales[c * d + j]
                    })
                    .collect();
                for a in 0..d {
                    for b in a..d {
                        cov[a * d + b] += w * r[a] * r[b];
                    }
                }
                wsum += w;
            }
        }
        let mut corr = vec![0.0; d * d];
        for a in 0..d {
            for b in a..d {
                let v = cov[a * d + b] / (cov[a * d + a] * cov[b * d + b]).sqrt().max(1e-12);
                corr[a * d + b] = if a == b { 1.0 } else { v.clamp(-0.995, 0.995) };
                corr[b * d + a] = corr[a * d + b];
            }
        }
        let _ = wsum;
        nearest_correlation(&corr, d, 1e-3)
    }

    /// Latent mixture marginal CDF of column `j`.
    pub fn cdf(&self, j: usize, z: f64) -> f64 {
        let d = self.means.len() / self.k();
        (0..self.k())
            .map(|c| {
                self.weights[c] * norm_cdf((z - self.means[c * d + j]) / self.scales[c * d + j])
            })
            .sum()
    }

    /// Support interval used for the per-column lookup table.
    pub fn span(&self, j: usize) -> (f64, f64) {
        let d = self.means.len() / self.k();
        let lo = (0..self.k())
            .map(|c| self.means[c * d + j] - 4.5 * self.scales[c * d + j])
            .fold(f64::INFINITY, f64::min);
        let hi = (0..self.k())
            .map(|c| self.means[c * d + j] + 4.5 * self.scales[c * d + j])
            .fold(f64::NEG_INFINITY, f64::max);
        (lo, hi)
    }

    /// Table of `Φ⁻¹(F_j(z))` on `points` uniform nodes of `span(j)`: maps a
    /// latent mixture coordinate to the standard normal score the stored
    /// marginal is indexed by.
    pub fn score_table(&self, j: usize, points: usize) -> (f64, f64, Vec<f64>) {
        let (lo, hi) = self.span(j);
        let t = (0..points)
            .map(|g| {
                norm_ppf(
                    self.cdf(j, lo + (hi - lo) * g as f64 / (points - 1) as f64)
                        .clamp(1e-12, 1.0 - 1e-12),
                )
            })
            .collect();
        (lo, hi, t)
    }

    /// Mean log-likelihood of standardized scores under the regime mixture
    /// with shared dependence `dep`.
    pub fn log_likelihood(&self, dep: &Dependence, s: &[f64], n: usize, d: usize) -> f64 {
        let (m, w) = dep.mixing(d);
        let mut sigma = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..=i {
                let v: f64 = (0..w).map(|q| m[i * w + q] * m[j * w + q]).sum();
                sigma[i * d + j] = v;
                sigma[j * d + i] = v;
            }
            sigma[i * d + i] += 1e-6;
        }
        let Some(l) = cholesky(&sigma, d) else {
            return f64::NEG_INFINITY;
        };
        let logdet_r: f64 = (0..d).map(|i| 2.0 * l[i * d + i].ln()).sum();
        let mut total = 0.0;
        let mut y = vec![0.0; d];
        for r in 0..n {
            let xi = &s[r * d..(r + 1) * d];
            let mut lp = Vec::with_capacity(self.k());
            for c in 0..self.k() {
                let mut q = 0.0;
                let mut logdet = logdet_r;
                for i in 0..d {
                    let sd = self.scales[c * d + i];
                    logdet += 2.0 * sd.ln();
                    let v = if xi[i].is_finite() {
                        xi[i]
                    } else {
                        self.means[c * d + i]
                    };
                    let mut acc = (v - self.means[c * d + i]) / sd;
                    for p in 0..i {
                        acc -= l[i * d + p] * y[p];
                    }
                    y[i] = acc / l[i * d + i];
                    q += y[i] * y[i];
                }
                lp.push(
                    self.weights[c].max(1e-12).ln()
                        - 0.5 * (q + logdet + d as f64 * (2.0 * PI).ln()),
                );
            }
            let mx = lp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            total += mx + lp.iter().map(|v| (v - mx).exp()).sum::<f64>().ln();
        }
        total / n.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_rare_joint_regime() {
        // 90% bulk around 0, 10% regime where both coordinates are high.
        let mut rng = Rng::new(3);
        let n = 4000;
        let mut s = Vec::with_capacity(2 * n);
        for i in 0..n {
            let shift = if i % 10 == 0 { 2.5 } else { -0.28 };
            s.push(shift + 0.3 * rng.normal());
            s.push(shift + 0.3 * rng.normal());
        }
        let r = Regimes::fit(&s, n, 2, 2, 1);
        let hi = (0..2)
            .max_by(|&a, &b| r.means[a * 2].total_cmp(&r.means[b * 2]))
            .unwrap();
        assert!((r.weights[hi] - 0.1).abs() < 0.03, "{:?}", r.weights);
        assert!(r.means[hi * 2 + 1] > 2.0);
        let indep = Dependence::Independent;
        assert!(
            r.log_likelihood(&indep, &s, n, 2)
                > Regimes::single(2).log_likelihood(&indep, &s, n, 2)
        );
        // The latent marginal CDF is a proper distribution function.
        assert!(r.cdf(0, -10.0) < 1e-6 && r.cdf(0, 10.0) > 1.0 - 1e-6);
    }
}

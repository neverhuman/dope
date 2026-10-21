//! What the conditional model does not explain.
//!
//! * Regression: `y = f(x) + σ(x)·e`, with `log σ(x)` linear in up to four
//!   screened inputs and `e` drawn from a stored standardized-residual
//!   quantile function (heavy tails and skew survive). Fitted on held-out
//!   residuals, so the noise is not underestimated by in-sample fit.
//! * Classification: labels drawn from `softmax(logits / τ)` with τ chosen by
//!   held-out likelihood (temperature scaling). Under log-loss the retention
//!   gap equals E KL(p* ‖ p_f) / I(X;Y), so calibration matters.

use crate::marginal::interp_grid;
use crate::marginal::{grid_z, Z_GRID};
use crate::stats::{norm_cdf, quantile_sorted, solve_spd};
use crate::target::{log_sum_exp, Head};

pub const RESIDUAL_KNOTS: usize = 9;
/// E[log |N(0,1)|].
const LOG_ABS_NORMAL_MEAN: f64 = -0.635_181_422_730_739;

#[derive(Clone, Debug, PartialEq)]
pub enum Noise {
    Gaussianish {
        /// log σ(x) = coef[0] + Σ_k coef[k] · x_c[k-1] (first t screened inputs).
        coef: Vec<f64>,
        lo: f64,
        hi: f64,
        /// Standardized residual quantiles on the z grid.
        knots: Vec<f64>,
    },
    Temperature(f64),
}

impl Noise {
    pub fn sigma(&self, xc_row: &[f64]) -> f64 {
        match self {
            Noise::Gaussianish { coef, lo, hi, .. } => {
                let mut z = coef[0];
                for (k, c) in coef.iter().enumerate().skip(1) {
                    z += c * xc_row.get(k - 1).copied().unwrap_or(0.0);
                }
                z.exp().clamp(*lo, *hi)
            }
            Noise::Temperature(_) => 0.0,
        }
    }

    pub fn residual(&self, z: f64) -> f64 {
        match self {
            Noise::Gaussianish { knots, .. } => interp_grid(knots, z),
            Noise::Temperature(_) => 0.0,
        }
    }
}

/// Fit the regression noise from held-out residuals `r` and inputs `xc`.
pub fn fit_regression(r: &[f64], xc: &[f64], inputs: usize) -> Noise {
    let n = r.len();
    let sd = (r.iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64)
        .sqrt()
        .max(1e-9);
    let t = inputs.min(4);
    let p = 1 + t;
    let mut g = vec![0.0; p * p];
    let mut rhs = vec![0.0; p];
    for i in 0..n {
        let target = r[i].abs().max(1e-6 * sd).ln() - LOG_ABS_NORMAL_MEAN;
        let mut row = vec![1.0; p];
        for k in 0..t {
            row[1 + k] = xc[i * inputs + k];
        }
        for a in 0..p {
            rhs[a] += row[a] * target;
            for b in 0..p {
                g[a * p + b] += row[a] * row[b];
            }
        }
    }
    for a in 1..p {
        g[a * p + a] += 1e-2 * n as f64;
    }
    let mut coef = if n >= 20 * p {
        solve_spd(&g, &rhs, p)
    } else {
        vec![0.0; p]
    };
    if n < 20 * p {
        coef[0] = sd.ln();
    }
    let (lo, hi) = (0.2 * sd, 4.0 * sd);
    let mut model = Noise::Gaussianish {
        coef,
        lo,
        hi,
        knots: vec![0.0; RESIDUAL_KNOTS],
    };
    // Standardize, renormalize to unit variance, then store the quantiles.
    let mut e: Vec<f64> = (0..n)
        .map(|i| r[i] / model.sigma(&xc[i * inputs..(i + 1) * inputs]))
        .collect();
    let scale = (e.iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64)
        .sqrt()
        .max(1e-9);
    if let Noise::Gaussianish { coef, .. } = &mut model {
        coef[0] += scale.ln();
    }
    e.iter_mut().for_each(|v| *v /= scale);
    e.sort_by(f64::total_cmp);
    let knots = (0..RESIDUAL_KNOTS)
        .map(|k| {
            if e.is_empty() {
                grid_z(k, RESIDUAL_KNOTS)
            } else {
                quantile_sorted(&e, norm_cdf(grid_z(k, RESIDUAL_KNOTS)))
            }
        })
        .collect();
    if let Noise::Gaussianish { knots: slot, .. } = &mut model {
        *slot = knots;
    }
    model
}

/// Temperature for class sampling, by golden-section search of held-out NLL.
pub fn fit_temperature(head: Head, logits: &[f64], y: &[f64]) -> Noise {
    let nll = |tau: f64| -> f64 {
        let n = y.len().max(1) as f64;
        match head {
            Head::Binary => {
                logits
                    .iter()
                    .zip(y)
                    .map(|(z, t)| {
                        let z = z / tau;
                        crate::target::softplus(z) - t * z
                    })
                    .sum::<f64>()
                    / n
            }
            Head::Multiclass(k) => {
                let mut total = 0.0;
                for (r, t) in y.iter().enumerate() {
                    let row: Vec<f64> =
                        logits[r * k..(r + 1) * k].iter().map(|z| z / tau).collect();
                    total += log_sum_exp(&row) - row[(*t as usize).min(k - 1)];
                }
                total / n
            }
            Head::Regression => 0.0,
        }
    };
    if y.is_empty() {
        return Noise::Temperature(1.0);
    }
    let (mut a, mut b) = (0.2f64.ln(), 5.0f64.ln());
    let phi = 0.618_033_988_749_895;
    for _ in 0..40 {
        let c = b - phi * (b - a);
        let d = a + phi * (b - a);
        if nll(c.exp()) < nll(d.exp()) {
            b = d;
        } else {
            a = c;
        }
    }
    Noise::Temperature(((a + b) / 2.0).exp())
}

/// Clamp range of a standardized residual draw (matches the knot grid).
pub const RESIDUAL_Z: f64 = Z_GRID;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn learns_heteroscedastic_scale() {
        let mut rng = Rng::new(4);
        let n = 4000;
        let xc: Vec<f64> = (0..n).map(|_| rng.uniform() - 0.5).collect();
        let r: Vec<f64> = xc
            .iter()
            .map(|x| rng.normal() * (0.1 + 0.4 * (x + 0.5)))
            .collect();
        let m = fit_regression(&r, &xc, 1);
        assert!(m.sigma(&[0.4]) > 2.0 * m.sigma(&[-0.4]));
        let mid = m.residual(0.0).abs();
        assert!(mid < 0.1, "median residual {mid}");
    }

    #[test]
    fn temperature_cools_overconfident_logits() {
        let mut rng = Rng::new(9);
        let n = 2000;
        let p: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
        let y: Vec<f64> = p
            .iter()
            .map(|q| (rng.uniform() < *q) as u8 as f64)
            .collect();
        let logits: Vec<f64> = p.iter().map(|q| 3.0 * (q / (1.0 - q)).ln()).collect();
        match fit_temperature(Head::Binary, &logits, &y) {
            Noise::Temperature(t) => assert!((t - 3.0).abs() < 0.6, "tau {t}"),
            _ => panic!(),
        }
    }
}

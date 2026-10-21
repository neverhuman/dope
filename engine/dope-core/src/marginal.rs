//! Per-feature marginals expressed as functions of a normal score `z`.
//!
//! A continuous marginal is a monotone piecewise-linear quantile function on
//! a *uniform grid in z* (z_k evenly spaced on [-Z, Z]); because the grid is
//! uniform, sampling needs one multiply and one gather per cell, which runs
//! identically on CPU and GPU. Discrete marginals (categorical codes, small
//! integer supports) are thresholds `t_k = Φ⁻¹(cum_k)` in the same z space.

use crate::stats::{norm_cdf, norm_pdf, norm_ppf, normal_scores, quantile_sorted};
use serde::{Deserialize, Serialize};

/// Half-width of the normal-score knot grid (Φ(-3) ≈ 0.00135).
pub const Z_GRID: f64 = 3.0;
/// Finite supports up to this size are modelled exactly.
pub const MAX_DISCRETE: usize = 24;
/// A value repeated in at least this share of rows is kept as an exact atom.
const ATOM_SHARE: f64 = 0.10;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Atom {
    pub value: f64,
    pub z_lo: f64,
    pub z_hi: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Marginal {
    Continuous { knots: Vec<f64>, atoms: Vec<Atom> },
    Discrete { values: Vec<f64>, probs: Vec<f64> },
}

#[inline]
pub fn grid_z(k: usize, knots: usize) -> f64 {
    -Z_GRID + 2.0 * Z_GRID * k as f64 / (knots - 1) as f64
}

impl Marginal {
    /// Fit from model-space values (NaN = missing). `categorical_levels`
    /// forces a discrete marginal over codes k/(L-1).
    pub fn fit(column: &[f64], knots: usize, categorical_levels: Option<usize>) -> Self {
        let mut obs: Vec<f64> = column.iter().copied().filter(|v| v.is_finite()).collect();
        obs.sort_by(f64::total_cmp);
        if let Some(levels) = categorical_levels {
            let mut counts = vec![0.0; levels.max(1)];
            for v in &obs {
                let k = if levels <= 1 {
                    0
                } else {
                    (v * (levels - 1) as f64).round() as usize
                };
                counts[k.min(levels.max(1) - 1)] += 1.0;
            }
            let values = (0..levels.max(1))
                .map(|k| {
                    if levels <= 1 {
                        0.5
                    } else {
                        k as f64 / (levels - 1) as f64
                    }
                })
                .collect();
            return Self::discrete(values, counts);
        }
        if obs.is_empty() {
            return Self::Discrete {
                values: vec![0.0],
                probs: vec![1.0],
            };
        }
        let mut uniq = obs.clone();
        uniq.dedup();
        if uniq.len() <= MAX_DISCRETE {
            let counts = uniq
                .iter()
                .map(|u| (obs.partition_point(|v| v <= u) - obs.partition_point(|v| v < u)) as f64)
                .collect();
            return Self::discrete(uniq, counts);
        }
        let n = obs.len() as f64;
        let knots_v = (0..knots)
            .map(|k| quantile_sorted(&obs, norm_cdf(grid_z(k, knots))))
            .collect();
        let mut atoms = Vec::new();
        let mut i = 0;
        while i < obs.len() {
            let j = obs.partition_point(|v| *v <= obs[i]);
            if (j - i) as f64 >= ATOM_SHARE * n {
                atoms.push(Atom {
                    value: obs[i],
                    z_lo: norm_ppf((i as f64 / n).max(1e-9)),
                    z_hi: norm_ppf((j as f64 / n).min(1.0 - 1e-9)),
                });
            }
            i = j;
        }
        Self::Continuous {
            knots: knots_v,
            atoms,
        }
    }

    fn discrete(values: Vec<f64>, counts: Vec<f64>) -> Self {
        let total: f64 = counts.iter().sum::<f64>().max(1.0);
        Self::Discrete {
            values,
            probs: counts.iter().map(|c| c / total).collect(),
        }
    }

    /// Thresholds t_1..t_{L-1} in z space for a discrete marginal.
    pub fn thresholds(probs: &[f64]) -> Vec<f64> {
        let mut cum = 0.0;
        probs[..probs.len().saturating_sub(1)]
            .iter()
            .map(|p| {
                cum += p;
                norm_ppf(cum.clamp(1e-12, 1.0 - 1e-12))
            })
            .collect()
    }

    /// Quantile function evaluated at a normal score.
    pub fn at(&self, z: f64) -> f64 {
        match self {
            Self::Continuous { knots, atoms } => {
                for a in atoms {
                    if z >= a.z_lo && z <= a.z_hi {
                        return a.value;
                    }
                }
                interp_grid(knots, z)
            }
            Self::Discrete { values, probs } => {
                let t = Self::thresholds(probs);
                values[t.iter().filter(|&&tk| z > tk).count()]
            }
        }
    }

    /// Normal scores of a column for dependence fitting, plus the standard
    /// deviation of those scores under the model (1 for continuous data;
    /// below 1 for discrete data, used to undo the attenuation of
    /// correlations between coarse variables).
    pub fn scores(&self, column: &[f64]) -> (Vec<f64>, f64) {
        match self {
            Self::Continuous { .. } => {
                let idx: Vec<usize> = (0..column.len())
                    .filter(|&i| column[i].is_finite())
                    .collect();
                let vals: Vec<f64> = idx.iter().map(|&i| column[i]).collect();
                let s = normal_scores(&vals);
                let sd = (s.iter().map(|v| v * v).sum::<f64>() / s.len().max(1) as f64).sqrt();
                let mut out = vec![f64::NAN; column.len()];
                for (k, &i) in idx.iter().enumerate() {
                    out[i] = s[k];
                }
                (out, sd.max(1e-6))
            }
            Self::Discrete { values, probs } => {
                let t = Self::thresholds(probs);
                let mut means = Vec::with_capacity(probs.len());
                let mut var = 0.0;
                for (k, &p) in probs.iter().enumerate() {
                    let lo = if k == 0 { f64::NEG_INFINITY } else { t[k - 1] };
                    let hi = if k + 1 == probs.len() {
                        f64::INFINITY
                    } else {
                        t[k]
                    };
                    let (plo, phi) = (pdf_or_zero(lo), pdf_or_zero(hi));
                    let m = if p > 1e-12 { (plo - phi) / p } else { 0.0 };
                    var += p * m * m;
                    means.push(m);
                }
                let out = column
                    .iter()
                    .map(|v| {
                        if !v.is_finite() {
                            return f64::NAN;
                        }
                        let k = nearest(values, *v);
                        means[k]
                    })
                    .collect();
                (out, var.sqrt().max(1e-6))
            }
        }
    }

    /// Inverse of [`Marginal::at`] on the continuous part (used by quantile encoding).
    pub fn cdf(&self, x: f64) -> f64 {
        match self {
            Self::Continuous { knots, .. } => {
                let k = knots.len();
                if x <= knots[0] {
                    return norm_cdf(grid_z(0, k));
                }
                if x >= knots[k - 1] {
                    return norm_cdf(grid_z(k - 1, k));
                }
                let i = knots
                    .partition_point(|v| *v <= x)
                    .saturating_sub(1)
                    .min(k - 2);
                let span = knots[i + 1] - knots[i];
                let t = if span > 0.0 {
                    (x - knots[i]) / span
                } else {
                    0.5
                };
                norm_cdf(grid_z(i, k) + t * (grid_z(i + 1, k) - grid_z(i, k)))
            }
            Self::Discrete { values, probs } => {
                let k = nearest(values, x);
                probs[..k].iter().sum::<f64>() + 0.5 * probs[k]
            }
        }
    }
}

fn pdf_or_zero(z: f64) -> f64 {
    if z.is_finite() {
        norm_pdf(z)
    } else {
        0.0
    }
}

pub fn nearest(values: &[f64], v: f64) -> usize {
    let mut best = 0;
    for (k, u) in values.iter().enumerate() {
        if (u - v).abs() < (values[best] - v).abs() {
            best = k;
        }
    }
    best
}

/// Piecewise-linear interpolation on the uniform z grid, clamped at the ends.
#[inline]
pub fn interp_grid(knots: &[f64], z: f64) -> f64 {
    let k = knots.len();
    let pos = ((z + Z_GRID) / (2.0 * Z_GRID) * (k - 1) as f64).clamp(0.0, (k - 1) as f64);
    let i = (pos.floor() as usize).min(k - 2);
    let t = pos - i as f64;
    knots[i] * (1.0 - t) + knots[i + 1] * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_quantiles_track_the_data() {
        let col: Vec<f64> = (0..1000).map(|i| (i as f64 / 999.0).powi(2)).collect();
        let m = Marginal::fit(&col, 17, None);
        assert!((m.at(0.0) - 0.25).abs() < 0.02);
        assert!(m.at(-10.0) >= 0.0 && m.at(10.0) <= 1.0);
        let p = m.cdf(0.25);
        assert!((p - 0.5).abs() < 0.02);
    }

    #[test]
    fn discrete_scores_have_attenuated_sd() {
        let col: Vec<f64> = (0..300).map(|i| (i % 3) as f64 / 2.0).collect();
        let m = Marginal::fit(&col, 9, Some(3));
        let (s, sd) = m.scores(&col);
        assert!(sd < 1.0 && sd > 0.7);
        assert!(s[0] < 0.0 && s[2] > 0.0);
        assert_eq!(m.at(-2.0), 0.0);
        assert_eq!(m.at(2.0), 1.0);
    }

    #[test]
    fn zero_inflation_is_an_atom() {
        let col: Vec<f64> = (0..1000)
            .map(|i| if i < 400 { 0.0 } else { i as f64 })
            .collect();
        match Marginal::fit(&col, 9, None) {
            Marginal::Continuous { atoms, .. } => assert_eq!(atoms.len(), 1),
            _ => panic!("expected continuous"),
        }
    }
}

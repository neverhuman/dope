//! Unified Phase P3 Column Realism Marginal Codec (`ColumnMarginalV2`).
//!
//! Covers BOTH features and targets with exact atoms (`<= 32`, `>= k`), decimal-exact
//! lattice (`q0` + `f32` heritage), heaping multiples `Q`, explicit off-grid mixture weight
//! `pi_offgrid` (`T28`), tier-scaled discrete cap (`24/64/256/4096`), first-class missingness,
//! and Wasserstein-1 DP knots + log-spaced tails to `k`-th order stats (`k = 10`).

use super::atoms::{detect_atoms, AtomV2, DEFAULT_SUPPORT_K};
use super::lattice::{detect_heaping, detect_lattice, HeapingSpec, LatticeSpec};
use super::tails::{fit_w1_knots, interp_uv, invert_uv};
use crate::stats::{norm_cdf, norm_ppf};
use serde::{Deserialize, Serialize};

/// Tier-scaled discrete cardinality cap (4 KiB -> 24, 16 KiB -> 64, 64 KiB -> 256, Max -> 4096).
pub fn tier_discrete_cap(budget_bytes: usize) -> usize {
    if budget_bytes <= 4_096 {
        24
    } else if budget_bytes <= 16_384 {
        64
    } else if budget_bytes <= 65_536 {
        256
    } else {
        4_096
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ColumnMarginalV2 {
    Discrete {
        values: Vec<f64>,
        probs: Vec<f64>,
        p_missing: f64,
        support_k: usize,
    },
    Continuous {
        atoms: Vec<AtomV2>,
        lattice: Option<LatticeSpec>,
        heaping: Option<HeapingSpec>,
        smooth_u: Vec<f64>,
        smooth_x: Vec<f64>,
        offgrid_u: Vec<f64>,
        offgrid_x: Vec<f64>,
        p_missing: f64,
        support_k: usize,
    },
}

impl ColumnMarginalV2 {
    /// Fit a v2 column marginal on raw numeric values (NaN = missing) for features OR targets.
    pub fn fit(column: &[f64], max_knots: usize, max_discrete: usize, support_k: usize) -> Self {
        let support_k = support_k.max(DEFAULT_SUPPORT_K);
        let total = column.len().max(1);
        let mut obs: Vec<f64> = column.iter().copied().filter(|v| v.is_finite()).collect();
        let p_missing = (total - obs.len()) as f64 / total as f64;
        if obs.is_empty() {
            return Self::Discrete {
                values: vec![0.0],
                probs: vec![1.0],
                p_missing: 1.0,
                support_k,
            };
        }
        obs.sort_by(f64::total_cmp);

        // Count unique values; pool rare levels (< support_k) only if discrete cardinality fits.
        let mut uniq_counts: Vec<(f64, usize)> = Vec::new();
        for &v in &obs {
            if let Some(last) = uniq_counts.last_mut() {
                if last.0 == v {
                    last.1 += 1;
                    continue;
                }
            }
            uniq_counts.push((v, 1));
        }

        if uniq_counts.len() <= max_discrete {
            // Scale discrete classification floor smoothly on small n (< 250) so genuine
            // low-cardinality binary/ordinal columns aren't smeared as continuous (`BUG-R16`).
            let disc_k = if obs.len() < support_k * 25 {
                support_k.min((obs.len() / 25).max(3))
            } else {
                support_k
            };
            let kept: Vec<(f64, usize)> = uniq_counts
                .iter()
                .copied()
                .filter(|&(_, c)| c >= disc_k)
                .collect();
            let kept_mass: usize = kept.iter().map(|&(_, c)| c).sum();
            // Never truncate multi-level ordinal/lattice columns by dropping > 1 level or > 5% mass;
            // let `Continuous + Lattice + Atoms` smoothly interpolate the full grid instead!
            let genuinely_discrete = uniq_counts.len() == 1
                || (kept.len() >= 2 && kept.len() == uniq_counts.len())
                || (kept.len() >= 2
                    && kept_mass * 20 >= obs.len() * 19
                    && uniq_counts.len() <= kept.len() + 1)
                || (obs.len() < disc_k * 2 && uniq_counts.len() <= 4);
            if genuinely_discrete {
                let final_counts = if kept.is_empty() {
                    vec![(obs[obs.len() / 2], obs.len())]
                } else {
                    let mut acc = kept;
                    for &(v, c) in &uniq_counts {
                        if c < disc_k {
                            let best = acc
                                .iter()
                                .enumerate()
                                .min_by(|a, b| (a.1 .0 - v).abs().total_cmp(&(b.1 .0 - v).abs()))
                                .map_or(0, |(i, _)| i);
                            acc[best].1 += c;
                        }
                    }
                    acc
                };
                let tot = final_counts.iter().map(|&(_, c)| c).sum::<usize>() as f64;
                return Self::Discrete {
                    values: final_counts.iter().map(|&(v, _)| v).collect(),
                    probs: final_counts
                        .iter()
                        .map(|&(_, c)| c as f64 / tot.max(1.0))
                        .collect(),
                    p_missing,
                    support_k,
                };
            }
        }

        let atoms = detect_atoms(&obs, support_k);
        let lattice = detect_lattice(&obs, support_k);
        let heaping = lattice.as_ref().and_then(|l| detect_heaping(&obs, l));
        let (smooth_u, smooth_x) = fit_w1_knots(&obs, max_knots, support_k);
        let (offgrid_u, offgrid_x) = if let Some(ref lat) = lattice {
            if lat.pi_offgrid > 0.0 {
                let off: Vec<f64> = obs
                    .iter()
                    .copied()
                    .filter(|&v| !lat.is_on_grid(v))
                    .collect();
                if off.len() >= 4 {
                    fit_w1_knots(&off, max_knots.min(16), support_k.min(off.len()))
                } else {
                    (Vec::new(), Vec::new())
                }
            } else {
                (Vec::new(), Vec::new())
            }
        } else {
            (Vec::new(), Vec::new())
        };

        Self::Continuous {
            atoms,
            lattice,
            heaping,
            smooth_u,
            smooth_x,
            offgrid_u,
            offgrid_x,
            p_missing,
            support_k,
        }
    }

    /// Stateless direct inversion given uniform draws `(u_main, u_mix, u_miss)` in `(0, 1)`.
    pub fn sample_from_uniform(&self, u_main: f64, u_mix: f64, u_miss: f64) -> f64 {
        let p_miss = match self {
            Self::Discrete { p_missing, .. } | Self::Continuous { p_missing, .. } => *p_missing,
        };
        if u_miss < p_miss {
            return f64::NAN;
        }
        let u = u_main.clamp(1e-9, 1.0 - 1e-9);
        match self {
            Self::Discrete { values, probs, .. } => {
                let mut cum = 0.0;
                for (v, p) in values.iter().zip(probs) {
                    cum += p;
                    if u <= cum {
                        return *v;
                    }
                }
                *values.last().unwrap_or(&0.0)
            }
            Self::Continuous {
                atoms,
                lattice,
                heaping,
                smooth_u,
                smooth_x,
                offgrid_u,
                offgrid_x,
                ..
            } => {
                // 1. Exact atoms take priority over smooth interpolation.
                for a in atoms {
                    if u >= a.u_lo && u <= a.u_hi {
                        return a.value;
                    }
                }
                // 2. Hybrid off-grid vs on-grid mixture (`T28`).
                if let Some(lat) = lattice {
                    if lat.pi_offgrid > 0.0 && !offgrid_u.is_empty() && u_mix < lat.pi_offgrid {
                        let raw_off = interp_uv(offgrid_u, offgrid_x, u);
                        // Ensure off-grid draw stays off-grid when smooth touches a lattice node.
                        if !lat.is_on_grid(raw_off) {
                            return raw_off;
                        }
                        return raw_off + 0.17 * lat.step;
                    }
                    let base = interp_uv(smooth_u, smooth_x, u);
                    // 3. Heaping boost onto multiple `Q = mult * q0` with exact excess probability.
                    if let Some(heap) = heaping {
                        let heap_u = ((u_mix - lat.pi_offgrid) / (1.0 - lat.pi_offgrid).max(1e-6))
                            .clamp(0.0, 1.0);
                        if heap_u < heap.excess {
                            let big_step = lat.step * heap.multiple as f64;
                            let big_idx = ((base - lat.origin) / big_step).round();
                            let heaped_raw = lat.origin + big_idx * big_step;
                            return lat.snap_canonical(heaped_raw);
                        }
                    }
                    return lat.snap_canonical(base);
                }
                interp_uv(smooth_u, smooth_x, u)
            }
        }
    }

    /// Evaluate from normal copula score `z` directly via `u = Phi(z)`.
    pub fn sample_from_z(&self, z: f64, u_mix: f64, u_miss: f64) -> f64 {
        self.sample_from_uniform(norm_cdf(z), u_mix, u_miss)
    }

    /// Approximate CDF in `(0, 1)` for a finite observation `x`.
    pub fn cdf(&self, x: f64) -> f64 {
        match self {
            Self::Discrete { values, probs, .. } => {
                let mut cum = 0.0;
                for (v, p) in values.iter().zip(probs) {
                    if (v - x).abs() <= 1e-12 {
                        return (cum + 0.5 * p).clamp(1e-9, 1.0 - 1e-9);
                    }
                    if *v < x {
                        cum += p;
                    }
                }
                cum.clamp(1e-9, 1.0 - 1e-9)
            }
            Self::Continuous {
                atoms,
                smooth_u,
                smooth_x,
                ..
            } => {
                for a in atoms {
                    if (a.value - x).abs() <= 1e-12 {
                        return 0.5 * (a.u_lo + a.u_hi);
                    }
                }
                invert_uv(smooth_u, smooth_x, x).clamp(1e-9, 1.0 - 1e-9)
            }
        }
    }

    pub fn z_score(&self, x: f64) -> f64 {
        if !x.is_finite() {
            f64::NAN
        } else {
            norm_ppf(self.cdf(x))
        }
    }
}

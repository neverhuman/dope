//! Decimal-exact lattice detection (`T27`), heaping multiples, and off-grid mixture (`T28`).

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LatticeSpec {
    /// Grid origin (typically `0.0`, or shifted min when non-zero offset aligns cleanly).
    pub origin: f64,
    /// Mantissa in `{1.0, 2.0, 2.5, 5.0}`.
    pub mantissa: f64,
    /// Decimal exponent `d in [-4, 6]` such that `step = mantissa * 10^(-d)`.
    pub dec_exp: i8,
    /// Exact step `q0 = mantissa * 10^(-dec_exp)`.
    pub step: f64,
    /// True when source values exhibit IEEE-754 `f32` rounding heritage (`T27`).
    pub f32_heritage: bool,
    /// Share of non-atom observations that fall off-grid (`T28`, 0.0 for pure lattices).
    pub pi_offgrid: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeapingSpec {
    /// Coarser multiple `mult in {2, 5, 10, 20, 25, 50, 100}` of `q0`.
    pub multiple: u32,
    /// Observed probability of falling on a multiple of `Q = multiple * q0`.
    pub observed_share: f64,
    /// Excess mass above uniform `1 / multiple` (>= 0.05).
    pub excess: f64,
}

impl LatticeSpec {
    /// Canonical decimal-exact emission avoiding binary float drift (e.g. `0.30000000000000004`).
    pub fn snap_canonical(&self, raw: f64) -> f64 {
        let idx = ((raw - self.origin) / self.step).round();
        let snapped = if self.mantissa > 0.0 && self.origin == 0.0 && self.dec_exp >= 0 {
            let scale = 10_f64.powi(self.dec_exp as i32);
            // Compute exact scaled integer when mantissa * 10 is integer (1, 2, 2.5, 5).
            let half_m = (self.mantissa * 2.0).round();
            let num = idx * half_m;
            let raw_dec = num / (2.0 * scale);
            // Decimal round at `dec_exp + 1` digits to eliminate 1-ulp artifacts.
            let round_scale = 10_f64.powi((self.dec_exp as i32) + 1);
            (raw_dec * round_scale).round() / round_scale
        } else {
            self.origin + idx * self.step
        };
        let snapped = if snapped == 0.0 { 0.0 } else { snapped };
        if self.f32_heritage {
            f64::from(snapped as f32)
        } else {
            snapped
        }
    }

    pub fn is_on_grid(&self, v: f64) -> bool {
        let idx = ((v - self.origin) / self.step).round();
        let target = self.origin + idx * self.step;
        (v - target).abs() <= 2e-3 * self.step
    }
}

/// True when finite observations exhibit nontrivial IEEE-754 `f32` rounding heritage (`T27`).
pub fn is_f32_heritage(values: &[f64]) -> bool {
    let mut n = 0usize;
    let mut f32_exact = 0usize;
    let mut f32_nontrivial = false;
    for &v in values {
        if !v.is_finite() {
            continue;
        }
        n += 1;
        let f = f64::from(v as f32);
        if (v - f).abs() <= 1e-12 * v.abs().max(1.0) {
            f32_exact += 1;
            let r6 = (v * 1e6).round() / 1e6;
            if (v - r6).abs() > 1e-10 {
                f32_nontrivial = true;
            }
        }
    }
    n >= 4 && f32_nontrivial && (f32_exact as f64 / n as f64) >= 0.995
}

/// Detect the coarsest decimal-exact lattice `q0 = m * 10^(-d)` (`m in {5, 2.5, 2, 1}`,
/// `d in [-4, 6]`, `2 <= bulk_range/q0 <= 1e6`), falling back to general uniform arithmetic
/// lattices (handling scaled/rational grids like `1/6, 1/7, 1/9` and model-space affine steps;
/// `M01/M02`, `T27`, `T28`).
pub fn detect_lattice(sorted: &[f64], support_k: usize) -> Option<LatticeSpec> {
    let n = sorted.len();
    if n < support_k.max(10) {
        return None;
    }
    // Use k-th order statistics for bulk range so a single extreme outlier never
    // inflates `step` and collapses the entire bulk to one grid node (`T08`).
    let k_ord = support_k.min(n / 10).max(1);
    let bulk_lo = sorted[k_ord];
    let bulk_hi = sorted[n - 1 - k_ord];
    let range = bulk_hi - bulk_lo;
    let f32_heritage = is_f32_heritage(sorted);

    if range.is_finite() && range > 0.0 {
        let origins = [0.0, bulk_lo];
        let mut best_hybrid: Option<(f64, LatticeSpec)> = None;
        for d in -4i8..=6i8 {
            let pow10 = 10_f64.powi(-(d as i32));
            for &m in &[5.0_f64, 2.5, 2.0, 1.0] {
                let step = m * pow10;
                let n_steps = range / step;
                if !(2.0..=1e6).contains(&n_steps) {
                    continue;
                }
                let tol = 2e-3 * step;
                for &origin in &origins {
                    let mut on_grid = 0usize;
                    let mut distinct_idx = std::collections::BTreeSet::new();
                    for &v in sorted {
                        let idx = ((v - origin) / step).round();
                        if (v - (origin + idx * step)).abs() <= tol {
                            on_grid += 1;
                            if v >= bulk_lo && v <= bulk_hi && distinct_idx.len() < 4 {
                                distinct_idx.insert(idx as i64);
                            }
                        }
                    }
                    // Require at least 3 distinct occupied lattice indices in the bulk!
                    if distinct_idx.len() < 3 {
                        continue;
                    }
                    let share = on_grid as f64 / n as f64;
                    let off_cnt = n - on_grid;
                    if share >= 0.995 {
                        // Do not override a strong coarse hybrid lattice (`>= 85%`) by refining
                        // more than 100x finer into noisy sub-digits (`T28`).
                        if let Some((h_share, ref h_spec)) = best_hybrid {
                            if h_share >= 0.85 && step <= h_spec.step * 0.01 {
                                return Some(h_spec.clone());
                            }
                        }
                        let pi_off = if off_cnt >= support_k {
                            off_cnt as f64 / n as f64
                        } else {
                            0.0
                        };
                        return Some(LatticeSpec {
                            origin,
                            mantissa: m,
                            dec_exp: d,
                            step,
                            f32_heritage,
                            pi_offgrid: pi_off,
                        });
                    }
                    if share >= 0.80 && off_cnt >= support_k {
                        let cand = LatticeSpec {
                            origin,
                            mantissa: m,
                            dec_exp: d,
                            step,
                            f32_heritage,
                            pi_offgrid: off_cnt as f64 / n as f64,
                        };
                        if best_hybrid.as_ref().is_none_or(|(s, _)| share > *s + 0.05) {
                            best_hybrid = Some((share, cand));
                        }
                    }
                }
            }
        }
        if let Some((_, spec)) = best_hybrid {
            return Some(spec);
        }
    }

    detect_uniform_lattice(sorted, support_k, f32_heritage)
}

/// Detect a general uniform arithmetic lattice `origin + k * step` (e.g. `k/7`, `k/9`, or
/// model-space affine-scaled integer/decimal lattices) using repeated (`count >= 2`) values
/// so private singletons never define the grid (`M01/M02`).
fn detect_uniform_lattice(
    sorted: &[f64],
    support_k: usize,
    f32_heritage: bool,
) -> Option<LatticeSpec> {
    let n = sorted.len();
    let mut uniq: Vec<(f64, usize)> = Vec::new();
    for &v in sorted {
        if let Some(last) = uniq.last_mut() {
            if (last.0 - v).abs() <= 1e-12 * v.abs().max(1.0) {
                last.1 += 1;
                continue;
            }
        }
        uniq.push((v, 1));
    }
    let rep: Vec<f64> = uniq
        .iter()
        .filter(|&&(_, c)| c >= 2)
        .map(|&(v, _)| v)
        .collect();
    if rep.len() < 3 || rep.len() > 2_000 {
        return None;
    }
    let full_range = rep[rep.len() - 1] - rep[0];
    if !full_range.is_finite() || full_range <= 0.0 {
        return None;
    }
    let mut gaps: Vec<f64> = rep
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&g| g > 1e-9 * full_range.max(1.0))
        .collect();
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_by(f64::total_cmp);
    let g0 = gaps[0];
    let mut cands = vec![g0, g0 / 2.0, g0 / 3.0];
    for &g in gaps.iter().take(6) {
        if cands.iter().all(|&c| (c - g).abs() > 1e-4 * c.max(1e-9)) {
            cands.push(g);
        }
    }
    cands.sort_by(|a, b| b.total_cmp(a));
    let origins = [0.0, rep[0]];
    for step in cands {
        let n_steps = full_range / step;
        if !(2.0..=2_000.0).contains(&n_steps) {
            continue;
        }
        let tol = 2e-3 * step;
        for &origin in &origins {
            let mut on_grid = 0usize;
            let mut distinct_idx = std::collections::BTreeSet::new();
            for &v in sorted {
                let idx = ((v - origin) / step).round();
                if (v - (origin + idx * step)).abs() <= tol {
                    on_grid += 1;
                    if distinct_idx.len() < 4 {
                        distinct_idx.insert(idx as i64);
                    }
                }
            }
            if distinct_idx.len() < 3 {
                continue;
            }
            let share = on_grid as f64 / n as f64;
            let off_cnt = n - on_grid;
            if share >= 0.99 {
                let pi_off = if off_cnt >= support_k {
                    off_cnt as f64 / n as f64
                } else {
                    0.0
                };
                return Some(LatticeSpec {
                    origin,
                    mantissa: 0.0,
                    dec_exp: 0,
                    step,
                    f32_heritage,
                    pi_offgrid: pi_off,
                });
            }
        }
    }
    None
}

/// Detect heaping at coarser multiples `Q = mult * q0` (`mult in {2,5,10,20,25,50,100}`)
/// when observed share on `Q` exceeds `1 / mult` by `>= 0.05`.
pub fn detect_heaping(sorted: &[f64], lattice: &LatticeSpec) -> Option<HeapingSpec> {
    let on_grid: Vec<i64> = sorted
        .iter()
        .filter(|&&v| lattice.is_on_grid(v))
        .map(|&v| ((v - lattice.origin) / lattice.step).round() as i64)
        .collect();
    if on_grid.len() < 20 {
        return None;
    }
    let nf = on_grid.len() as f64;
    let mut best: Option<HeapingSpec> = None;
    for &mult in &[100u32, 50, 25, 20, 10, 5, 2] {
        let m64 = mult as i64;
        let hits = on_grid.iter().filter(|&&idx| idx % m64 == 0).count();
        let obs = hits as f64 / nf;
        let expected = 1.0 / mult as f64;
        let excess = obs - expected;
        if excess >= 0.05 && obs < 0.995 && best.as_ref().is_none_or(|b| excess > b.excess) {
            best = Some(HeapingSpec {
                multiple: mult,
                observed_share: obs,
                excess,
            });
        }
    }
    best
}

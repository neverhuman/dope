//! Typed marginals (`cont`, `lattice + heaping`, `atoms k>=10`, `zero_inflated`, `const`)
//! and adaptive tail extrapolation for `DSE8` v1.

use super::rng_v8::{counter_uniform, STREAM_ATOM, STREAM_MISSING};
use super::scale_v8::Dse8ColumnMeta;
use crate::evidence::primitives::{support_u_floor, SUPPORT_FLOOR_K};
use crate::marginal::atoms::detect_atoms;
use crate::marginal::lattice::detect_lattice;
use crate::single_encoding::math::{f16_bits_to_f32, f32_to_f16_bits, quant_f16_roundtrip};
use crate::single_encoding::KNOT_COUNT;
use crate::stats::{norm_cdf, quantile_sorted};

pub use super::nominal::{decode_nominal_group, fit_nominal_groups, NominalGroupV8};

pub const MARG_CONT: u8 = 0;
pub const MARG_LATTICE: u8 = 1;
pub const MARG_ATOMS: u8 = 2;
pub const MARG_ZERO_INFLATED: u8 = 3;
pub const MARG_CONST: u8 = 4;

pub const TAIL_BOUNDED: u8 = 0;
pub const TAIL_EXP: u8 = 1;
pub const TAIL_PARETO: u8 = 2;

/// Compact per-column typed marginal specification stored inside `DSE8` `TAG_MARG`.
#[derive(Clone, Debug)]
pub struct MarginalColV8 {
    pub kind: u8,
    pub tail_family: u8,
    pub is_int: bool,
    pub miss_rate: f32,
    pub lo: f32,
    pub span: f32,
    pub knots17: [f32; KNOT_COUNT],
    pub knot_level: u8,
    pub rel_q: Vec<u8>,
    // Lattice parameters (when kind == MARG_LATTICE)
    pub lattice_q0: f32,
    pub lattice_origin: f32,
    pub heaping_step: f32,
    pub heaping_excess: f32,
    // Atom / zero-inflated parameters (k >= 10 enforced; max 8 atoms)
    pub atom_vals: Vec<f32>,
    pub atom_probs: Vec<f32>,
}

impl MarginalColV8 {
    /// Fit a typed marginal strictly from FIT column values (`x_unit`, optional `x_raw` for lattice).
    pub fn fit(
        x_unit: &[f32],
        meta: Option<&Dse8ColumnMeta>,
        knot_level: usize,
        z_max: f32,
        is_int: bool,
    ) -> Self {
        let n_tot = x_unit.len().max(1);
        let mut sorted: Vec<f64> = x_unit
            .iter()
            .map(|&v| v as f64)
            .filter(|v| v.is_finite())
            .collect();
        let n_obs = sorted.len();
        let miss_rate = ((n_tot - n_obs) as f32 / n_tot as f32).clamp(0.0, 0.99);
        sorted.sort_by(f64::total_cmp);

        if n_obs < 4 {
            return Self::constant(0.0, miss_rate, knot_level);
        }

        // Distinct value runs with k >= 10 support floor checked BEFORE constant collapse so
        // imbalanced binary/sparse columns with >= 10 minority rows are never crushed to constants.
        let mut runs: Vec<(f64, usize)> = Vec::new();
        for &v in &sorted {
            if let Some(last) = runs.last_mut() {
                if (last.0 - v).abs() <= 1e-7 * v.abs().max(1.0) {
                    last.1 += 1;
                    continue;
                }
            }
            runs.push((v, 1));
        }
        let kept_runs: Vec<(f64, usize)> = runs
            .iter()
            .copied()
            .filter(|&(_, c)| c >= SUPPORT_FLOOR_K)
            .collect();
        let is_discrete_atoms = runs.len() <= 8
            && kept_runs.len() >= 2
            && (kept_runs.iter().map(|r| r.1).sum::<usize>() as f64 / n_obs as f64) >= 0.85;

        let u_floor = support_u_floor(n_obs);
        let mut lo_raw = quantile_sorted(&sorted, u_floor);
        let mut hi_raw = quantile_sorted(&sorted, 1.0 - u_floor);
        if is_discrete_atoms {
            lo_raw = kept_runs[0].0;
            hi_raw = kept_runs[kept_runs.len() - 1].0;
        } else if (hi_raw - lo_raw) < 1e-6 && n_obs >= 2 * SUPPORT_FLOOR_K {
            // Support-safe k-th order statistics preserve variance on sparse heavy-tail columns
            // without ever leaking rare extrema (strictly k >= 10 on each side).
            let lo_k = sorted[SUPPORT_FLOOR_K - 1];
            let hi_k = sorted[n_obs - SUPPORT_FLOOR_K];
            if (hi_k - lo_k) >= 1e-6 {
                lo_raw = lo_k;
                hi_raw = hi_k;
            }
        }
        if (hi_raw - lo_raw) < 1e-6 {
            let val = quant_f16_roundtrip(quantile_sorted(&sorted, 0.5) as f32);
            return Self::constant(val, miss_rate, knot_level);
        }

        let lo = quant_f16_roundtrip(lo_raw as f32);
        let span = quant_f16_roundtrip((hi_raw - lo_raw).max(1e-5) as f32).max(1e-5);

        // Build 17-knot quantile curve.
        let mut knots_f64 = [0.0f64; KNOT_COUNT];
        for (ki, slot) in knots_f64.iter_mut().enumerate() {
            let z = -(z_max as f64) + (2.0 * z_max as f64) * (ki as f64 / (KNOT_COUNT - 1) as f64);
            *slot = quantile_sorted(&sorted, norm_cdf(z));
        }

        // Detect atoms (strictly k >= 10 support floor!) and lattice.
        let atoms = detect_atoms(&sorted, SUPPORT_FLOOR_K);
        let tot_atom_p: f64 = atoms.iter().map(|a| a.prob).sum();
        let lattice = detect_lattice(&sorted, SUPPORT_FLOOR_K);

        // Adaptive tail family from Pickands / excess kurtosis.
        let tail_family = if meta
            .map(|m| m.has_lower_bound > 0 && m.has_upper_bound > 0)
            .unwrap_or(false)
            || is_int
        {
            TAIL_BOUNDED
        } else {
            let u_p = u_floor.max(0.05);
            if 4.0 * u_p < 0.45 {
                let q1 = quantile_sorted(&sorted, 1.0 - u_p);
                let q2 = quantile_sorted(&sorted, 1.0 - 2.0 * u_p);
                let q4 = quantile_sorted(&sorted, 1.0 - 4.0 * u_p);
                let pick =
                    ((q1 - q2).max(1e-6) / (q2 - q4).max(1e-6)).ln() / std::f64::consts::LN_2;
                if pick > 0.35 {
                    TAIL_PARETO
                } else if pick > 0.05 {
                    TAIL_EXP
                } else {
                    TAIL_BOUNDED
                }
            } else {
                TAIL_EXP
            }
        };

        // Choose marginal kind.
        let mut kind = MARG_CONT;
        let mut atom_vals = Vec::new();
        let mut atom_probs = Vec::new();
        let mut lattice_q0 = 0.0f32;
        let mut lattice_origin = 0.0f32;
        let heaping_step = 0.0f32;
        let heaping_excess = 0.0f32;

        if is_discrete_atoms {
            kind = MARG_ATOMS;
            let tot_k: f64 = kept_runs.iter().map(|r| r.1 as f64).sum::<f64>().max(1.0);
            for (v, c) in kept_runs.into_iter().take(8) {
                atom_vals.push(quant_f16_roundtrip(v as f32));
                atom_probs.push((c as f64 / tot_k) as f32);
            }
            // Snap knots to kept atoms for clean copula coupling.
            for kv in &mut knots_f64 {
                if let Some(&near) = atom_vals.iter().min_by(|a, b| {
                    (**a as f64 - *kv)
                        .abs()
                        .total_cmp(&(**b as f64 - *kv).abs())
                }) {
                    *kv = near as f64;
                }
            }
        } else if let Some(lat) = lattice {
            let meta_q0 = meta.map(|m| m.lattice_q0_unit).unwrap_or(0.0);
            let q0 = if meta_q0 > 1e-5 {
                meta_q0
            } else {
                lat.step as f32
            };
            // Require q0 >= 1.5% of span so f16 rounding of q0 cannot cause cumulative grid drift,
            // and avoid post-hoc stochastic heaping distortion that inflates marginal C2ST.
            if q0 >= span * 0.015 && q0 < span * 0.5 {
                kind = MARG_LATTICE;
                lattice_q0 = quant_f16_roundtrip(q0);
                lattice_origin = quant_f16_roundtrip(lat.origin as f32);
            }
        } else if !atoms.is_empty() && tot_atom_p >= 0.12 {
            // Zero-inflated or semi-continuous with k>=10 spike atom(s).
            kind = MARG_ZERO_INFLATED;
            for a in atoms.iter().take(2) {
                atom_vals.push(quant_f16_roundtrip(a.value as f32));
                atom_probs.push((a.prob as f32).clamp(0.01, 0.95));
            }
        }

        // Quantize knot profile at knot_level (5, 9, or 17).
        let kl = match knot_level {
            5 | 9 | 17 => knot_level,
            _ => 9,
        };
        let step = (KNOT_COUNT - 1) / (kl - 1);
        let mut rel_q = Vec::with_capacity(kl);
        for ki in 0..kl {
            let rel = ((knots_f64[ki * step] - lo as f64) / span as f64).clamp(0.0, 1.0);
            rel_q.push((rel * 255.0).round().clamp(0.0, 255.0) as u8);
        }
        let knots17 = reconstruct_knots17(lo, span, &rel_q);

        Self {
            kind,
            tail_family,
            is_int,
            miss_rate,
            lo,
            span,
            knots17,
            knot_level: kl as u8,
            rel_q,
            lattice_q0,
            lattice_origin,
            heaping_step,
            heaping_excess,
            atom_vals,
            atom_probs,
        }
    }

    fn constant(val: f32, miss_rate: f32, knot_level: usize) -> Self {
        let kl = match knot_level {
            5 | 9 | 17 => knot_level,
            _ => 9,
        };
        Self {
            kind: MARG_CONST,
            tail_family: TAIL_BOUNDED,
            is_int: false,
            miss_rate,
            lo: val,
            span: 0.0,
            knots17: [val; KNOT_COUNT],
            knot_level: kl as u8,
            rel_q: vec![0u8; kl],
            lattice_q0: 0.0,
            lattice_origin: 0.0,
            heaping_step: 0.0,
            heaping_excess: 0.0,
            atom_vals: Vec::new(),
            atom_probs: Vec::new(),
        }
    }

    /// Decode one scalar feature value from latent Gaussian score `z_lat`, respecting type, lattice,
    /// atoms (`k >= 10`), zero-inflation, adaptive tails, and missingness!
    pub fn decode_value(
        &self,
        z_lat: f32,
        knots_eff: &[f32; KNOT_COUNT],
        z_max: f32,
        seed: u64,
        row_idx: u64,
        col_idx: u64,
    ) -> (f32, bool) {
        if self.miss_rate > 1e-4 {
            let u_m = counter_uniform(seed, row_idx, STREAM_MISSING, col_idx) as f32;
            if u_m < self.miss_rate {
                return (0.0, true);
            }
        }
        if self.kind == MARG_CONST {
            return (self.lo, false);
        }

        // Pure discrete atoms: map Φ(z_lat) directly through the k>=10 atom CDF!
        if self.kind == MARG_ATOMS && !self.atom_vals.is_empty() {
            let u = norm_cdf(z_lat as f64) as f32;
            let mut acc = 0.0f32;
            for (v, p) in self.atom_vals.iter().zip(&self.atom_probs) {
                acc += *p;
                if u <= acc {
                    return (*v, false);
                }
            }
            return (*self.atom_vals.last().unwrap(), false);
        }

        // Continuous / lattice quantile interpolation with adaptive tail extrapolation.
        let dz = (2.0 * z_max) / (KNOT_COUNT - 1) as f32;
        let pos = (z_lat.clamp(-z_max, z_max) + z_max) / dz;
        let i0 = (pos.floor() as isize).clamp(0, (KNOT_COUNT - 2) as isize) as usize;
        let frac = (pos - i0 as f32).clamp(0.0, 1.0);
        let mut x = knots_eff[i0] + frac * (knots_eff[i0 + 1] - knots_eff[i0]);

        if z_lat < -z_max || z_lat > z_max {
            let slope_lo = ((knots_eff[1] - knots_eff[0]) / dz).max(0.0);
            let slope_hi = ((knots_eff[KNOT_COUNT - 1] - knots_eff[KNOT_COUNT - 2]) / dz).max(0.0);
            let mult = match self.tail_family {
                TAIL_BOUNDED => 0.0,
                TAIL_EXP => 0.28,
                _ => 0.55, // Pareto heavy tail
            };
            x += mult * slope_lo * (z_lat + z_max).min(0.0)
                + mult * slope_hi * (z_lat - z_max).max(0.0);
        }

        // Semi-continuous / zero-inflated: snap exact boundary atoms strictly on their exact
        // Gaussian CDF mass interval [0, p] or [1-p, 1] (zero double-counting), and snap
        // interior plateaus only when already within u8 knot quantization tolerance.
        if self.kind == MARG_ZERO_INFLATED && !self.atom_vals.is_empty() {
            let u = norm_cdf(z_lat as f64) as f32;
            let tol = (self.span * 0.012).max(1e-4);
            let hi = self.lo + self.span;
            for (&v, &p) in self.atom_vals.iter().zip(&self.atom_probs) {
                if (v - self.lo).abs() <= (self.span * 0.005).max(1e-4) {
                    if u <= p {
                        return (v, false);
                    }
                } else if (v - hi).abs() <= (self.span * 0.005).max(1e-4) {
                    if u >= 1.0 - p {
                        return (v, false);
                    }
                } else if (x - v).abs() <= tol {
                    return (v, false);
                }
            }
        }

        if self.kind == MARG_LATTICE && self.lattice_q0 > 1e-5 {
            let q0 = self.lattice_q0;
            let orig = self.lattice_origin;
            x = orig + ((x - orig) / q0).round() * q0;
            if self.heaping_excess > 0.05 && self.heaping_step > q0 {
                let u_h = counter_uniform(seed, row_idx, STREAM_ATOM, col_idx) as f32;
                if u_h < self.heaping_excess {
                    let hs = self.heaping_step;
                    x = orig + ((x - orig) / hs).round() * hs;
                }
            }
        }
        (x.clamp(-6.0, 6.0), false)
    }

    pub fn encode_bytes(&self, out: &mut Vec<u8>) {
        let flags =
            (self.kind & 0x07) | ((self.tail_family & 0x03) << 3) | (u8::from(self.is_int) << 5);
        out.push(flags);
        out.push((self.miss_rate * 255.0).round().clamp(0.0, 255.0) as u8);
        out.extend_from_slice(&f32_to_f16_bits(self.lo).to_le_bytes());
        out.extend_from_slice(&f32_to_f16_bits(self.span).to_le_bytes());
        if self.kind != MARG_CONST {
            out.extend_from_slice(&self.rel_q);
        }
        match self.kind {
            MARG_LATTICE => {
                out.extend_from_slice(&f32_to_f16_bits(self.lattice_q0).to_le_bytes());
                out.extend_from_slice(&f32_to_f16_bits(self.lattice_origin).to_le_bytes());
                out.extend_from_slice(&f32_to_f16_bits(self.heaping_step).to_le_bytes());
                out.push((self.heaping_excess * 255.0).round().clamp(0.0, 255.0) as u8);
            }
            MARG_ATOMS | MARG_ZERO_INFLATED => {
                let k = self.atom_vals.len().min(8);
                out.push(k as u8);
                for i in 0..k {
                    out.extend_from_slice(&f32_to_f16_bits(self.atom_vals[i]).to_le_bytes());
                    out.push((self.atom_probs[i] * 255.0).round().clamp(1.0, 255.0) as u8);
                }
            }
            _ => {}
        }
    }

    pub fn decode_bytes(bytes: &[u8], cur: &mut usize, knot_level: usize) -> Option<Self> {
        if *cur + 6 > bytes.len() {
            return None;
        }
        let flags = bytes[*cur];
        let kind = flags & 0x07;
        let tail_family = (flags >> 3) & 0x03;
        let is_int = (flags & (1 << 5)) != 0;
        let miss_rate = bytes[*cur + 1] as f32 / 255.0;
        let lo = f16_bits_to_f32(u16::from_le_bytes([bytes[*cur + 2], bytes[*cur + 3]]));
        let span = f16_bits_to_f32(u16::from_le_bytes([bytes[*cur + 4], bytes[*cur + 5]]));
        *cur += 6;

        let rel_q = if kind == MARG_CONST {
            vec![0u8; knot_level]
        } else {
            if *cur + knot_level > bytes.len() {
                return None;
            }
            let r = bytes[*cur..*cur + knot_level].to_vec();
            *cur += knot_level;
            r
        };
        let knots17 = if kind == MARG_CONST {
            [lo; KNOT_COUNT]
        } else {
            reconstruct_knots17(lo, span.max(1e-5), &rel_q)
        };

        let (mut lattice_q0, mut lattice_origin, mut heaping_step, mut heaping_excess) =
            (0.0, 0.0, 0.0, 0.0);
        let mut atom_vals = Vec::new();
        let mut atom_probs = Vec::new();
        match kind {
            MARG_LATTICE => {
                if *cur + 7 > bytes.len() {
                    return None;
                }
                lattice_q0 = f16_bits_to_f32(u16::from_le_bytes([bytes[*cur], bytes[*cur + 1]]));
                lattice_origin =
                    f16_bits_to_f32(u16::from_le_bytes([bytes[*cur + 2], bytes[*cur + 3]]));
                heaping_step =
                    f16_bits_to_f32(u16::from_le_bytes([bytes[*cur + 4], bytes[*cur + 5]]));
                heaping_excess = bytes[*cur + 6] as f32 / 255.0;
                *cur += 7;
            }
            MARG_ATOMS | MARG_ZERO_INFLATED => {
                if *cur >= bytes.len() {
                    return None;
                }
                let k = bytes[*cur] as usize;
                *cur += 1;
                if k > 8 || *cur + k * 3 > bytes.len() {
                    return None;
                }
                let mut sum_p = 0.0f32;
                for _ in 0..k {
                    let v = f16_bits_to_f32(u16::from_le_bytes([bytes[*cur], bytes[*cur + 1]]));
                    let p = bytes[*cur + 2] as f32 / 255.0;
                    *cur += 3;
                    atom_vals.push(v);
                    atom_probs.push(p);
                    sum_p += p;
                }
                if kind == MARG_ATOMS && sum_p > 1e-6 {
                    for p in &mut atom_probs {
                        *p /= sum_p;
                    }
                }
            }
            _ => {}
        }
        Some(Self {
            kind,
            tail_family,
            is_int,
            miss_rate,
            lo,
            span,
            knots17,
            knot_level: knot_level as u8,
            rel_q,
            lattice_q0,
            lattice_origin,
            heaping_step,
            heaping_excess,
            atom_vals,
            atom_probs,
        })
    }
}

pub fn reconstruct_knots17(lo: f32, span: f32, rel_q: &[u8]) -> [f32; KNOT_COUNT] {
    let kl = rel_q.len().max(2);
    let step = (KNOT_COUNT - 1) / (kl - 1).max(1);
    let mut knots17 = [0.0f32; KNOT_COUNT];
    for (ki, slot) in knots17.iter_mut().enumerate() {
        let seg = (ki / step).min(kl - 2);
        let frac = (ki - seg * step) as f32 / step as f32;
        let r0 = rel_q[seg] as f32 / 255.0;
        let r1 = rel_q[seg + 1] as f32 / 255.0;
        *slot = lo + span * (r0 + frac * (r1 - r0));
    }
    for ki in 1..KNOT_COUNT {
        if knots17[ki] < knots17[ki - 1] {
            knots17[ki] = knots17[ki - 1];
        }
    }
    knots17
}

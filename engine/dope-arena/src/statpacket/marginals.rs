//! Per-column marginals: constant, continuous (nested 5 / 9 / 17 smoothed
//! knots) or discrete (atom table with a support floor of 10 rows).
//!
//! Everything is in unit space (`scale::ScaleCard`), where the FIT median of a
//! column is 0. Knot `i` of 17 sits at the normal score
//! `z_i = z_hi * (i - 8) / 8`, `z_hi = -ppf(u_lo)`, so the knots cover the
//! clipped probability range `[u_lo, 1 - u_lo]` and are equally spaced in the
//! coordinate the sampler works in. A knot is the mean of `2 * half + 1`
//! consecutive order statistics and its window never contains the smallest or
//! the largest FIT row. Nothing here stores a minimum, a maximum, or a level
//! carried by fewer than 10 rows.
//!
//! Stored increments: knot 8 is pinned to the origin (the FIT median); `lo`
//! and `hi` are log codes of the distances down to knot 0 and up to knot 16;
//! two 8-bit splits place knots 4 and 12 inside their half; every refinement
//! knot is a 6-bit split of its parent interval. Each side of the median
//! therefore keeps its own relative precision, however skewed the column is.

use super::packet::{
    f16_to_f64, f64_to_f16, AtomTable, AtomValues, ColumnMarginal, PacketLevel, CODE6_MAX,
    HALF_SPAN_CODE_MAX, MAX_ATOMS, SUPPORT_K,
};
use dope_core::stats::{norm_cdf, norm_ppf};

pub const KNOTS: usize = 17;
/// Unit-space clip shared with the scaler.
pub const UNIT_CLIP: f64 = 6.0;
/// Half-span code: `0` for code 0, else `6 * 2^((code - 4095) / 128)`.
pub const HALF_SPAN_MAX: f64 = 6.0;
pub const HALF_SPAN_STEPS_PER_OCTAVE: f64 = 128.0;
/// Split code: knot = inner end + `code / 255` of the half-span.
pub const SPLIT_MAX: f64 = 255.0;
/// A lattice is used only when it rebuilds every atom within this unit-space error.
pub const LATTICE_TOL: f64 = 0.004;
const LATTICE_MIN_ATOMS: usize = 5;
const LATTICE_DIVISORS: usize = 8;

/// Clipping and smoothing constants; a pure function of header `floor(log2 n_fit)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KnotPlan {
    /// `k = min(10, max(3, 2^e / 10))`.
    pub window_k: usize,
    /// Window half-width `ceil(k / 2)` in order statistics.
    pub half: usize,
    /// Lowest knot probability `max(0.01, k / 2^e)`.
    pub u_lo: f64,
    pub z_hi: f64,
}

pub fn knot_plan(log2_n: u8) -> KnotPlan {
    let pow = 2f64.powi(log2_n.min(62) as i32);
    let window_k = ((pow / 10.0).floor() as usize).clamp(3, SUPPORT_K);
    let u_lo = (window_k as f64 / pow).max(0.01);
    KnotPlan {
        window_k,
        half: window_k.div_ceil(2),
        u_lo,
        z_hi: -norm_ppf(u_lo),
    }
}

/// Probability of knot `i` (0..17).
pub fn knot_probability(plan: &KnotPlan, i: usize) -> f64 {
    norm_cdf(plan.z_hi * (i as f64 - 8.0) / 8.0)
}

/// Seventeen smoothed order statistics of an ascending column with at least 16 rows.
pub fn smoothed_knots(sorted: &[f64], plan: &KnotPlan) -> [f64; KNOTS] {
    let n = sorted.len();
    let mut out = [0.0; KNOTS];
    let mut floor = f64::NEG_INFINITY;
    for (i, slot) in out.iter_mut().enumerate() {
        let centre = (knot_probability(plan, i) * (n - 1) as f64 + 0.5).floor() as usize;
        let centre = centre.clamp(1, n - 2);
        // The window never reaches rank 0 or rank n-1.
        let lo = centre.saturating_sub(plan.half).max(1);
        let hi = (centre + plan.half).min(n - 2);
        let window = &sorted[lo..=hi];
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        floor = floor.max(mean);
        *slot = floor;
    }
    out
}

/// 6-bit weights `c` with `c_i / sum(c)` close to `w_i / sum(w)`: for every
/// multiplier `m` in 1..=63 take `c_i = clamp(round(m * w_i / max(w)))` and keep
/// the first `m` with the smallest maximum error of the cumulative shares.
pub fn simplex_codes(w: &[f64], min_code: u8) -> Vec<u8> {
    let total: f64 = w.iter().sum();
    let top = w.iter().fold(0.0f64, |m, &v| m.max(v));
    if top <= 0.0 || !total.is_finite() {
        return vec![min_code; w.len()];
    }
    let mut best = (f64::INFINITY, vec![min_code; w.len()]);
    for m in 1..=CODE6_MAX {
        let codes: Vec<u8> = w
            .iter()
            .map(|&v| ((m as f64 * v / top).round() as u8).clamp(min_code, CODE6_MAX))
            .collect();
        let sum: u32 = codes.iter().map(|&c| c as u32).sum();
        if sum == 0 {
            continue;
        }
        let (mut want, mut have, mut err) = (0.0, 0u32, 0.0f64);
        for (&v, &c) in w.iter().zip(&codes) {
            want += v / total;
            have += c as u32;
            err = err.max((have as f64 / sum as f64 - want).abs());
        }
        if err < best.0 - 1e-12 {
            best = (err, codes);
        }
    }
    best.1
}

/// Dequantised half-span of a 12-bit code.
pub fn half_span_value(code: u16) -> f64 {
    if code == 0 {
        return 0.0;
    }
    let octaves = (code as f64 - HALF_SPAN_CODE_MAX as f64) / HALF_SPAN_STEPS_PER_OCTAVE;
    HALF_SPAN_MAX * octaves.exp2()
}

/// Nearest half-span code on the log scale; 0 only for a half-span of zero.
fn half_span_code(v: f64) -> u16 {
    if v <= 0.0 {
        return 0;
    }
    let code = HALF_SPAN_CODE_MAX as f64 + HALF_SPAN_STEPS_PER_OCTAVE * (v / HALF_SPAN_MAX).log2();
    code.round().clamp(1.0, HALF_SPAN_CODE_MAX as f64) as u16
}

fn fraction_code(x: f64, lo: f64, hi: f64) -> u8 {
    if hi <= lo {
        return 0;
    }
    (CODE6_MAX as f64 * ((x - lo) / (hi - lo)).clamp(0.0, 1.0)).round() as u8
}

/// The five coarse knots (grid indices 0, 4, 8, 12, 16); knot 8 is the origin.
fn coarse_knots(lo: u16, hi: u16, split: &[u8; 2]) -> [f64; 5] {
    let (below, above) = (half_span_value(lo), half_span_value(hi));
    [
        -below,
        -below * (1.0 - split[0] as f64 / SPLIT_MAX),
        0.0,
        above * (split[1] as f64 / SPLIT_MAX),
        above,
    ]
}

/// Inserts one knot inside every interval: `lo + code / 63 * (hi - lo)`.
fn refine(parents: &[f64], mids: &[u8]) -> Vec<f64> {
    let mut out = Vec::with_capacity(parents.len() + mids.len());
    for (pair, &code) in parents.windows(2).zip(mids) {
        out.push(pair[0]);
        out.push(pair[0] + code as f64 / CODE6_MAX as f64 * (pair[1] - pair[0]));
    }
    out.push(parents[parents.len() - 1]);
    out
}

/// Dequantised knots of a continuous column at `count` in {5, 9, 17}.
pub fn decoded_knots(col: &ColumnMarginal, count: usize) -> Vec<f64> {
    let ColumnMarginal::Continuous {
        lo,
        hi,
        split,
        mid9,
        mid17,
    } = col
    else {
        return Vec::new();
    };
    let coarse = coarse_knots(*lo, *hi, split).to_vec();
    match count {
        5 => coarse,
        9 => refine(&coarse, mid9),
        _ => refine(&refine(&coarse, mid9), mid17),
    }
}

/// Closed-loop quantisation: every code is chosen against the already
/// dequantised parents, so errors do not accumulate down the nesting. The
/// input must be median-centred (the scale card guarantees it); a side that
/// does not reach the origin is stored as an empty half.
pub fn quantise_knots(k: &[f64; KNOTS]) -> ColumnMarginal {
    let (lo, hi) = (half_span_code(-k[0]), half_span_code(k[16]));
    let (below, above) = (half_span_value(lo), half_span_value(hi));
    let share = |part: f64, whole: f64| -> u8 {
        if whole <= 0.0 {
            return 0;
        }
        (SPLIT_MAX * (part / whole).clamp(0.0, 1.0)).round() as u8
    };
    let split = [share(k[4] + below, below), share(k[12], above)];
    let coarse = coarse_knots(lo, hi, &split);
    let mut mid9 = [0u8; 4];
    for (i, code) in mid9.iter_mut().enumerate() {
        *code = fraction_code(k[4 * i + 2], coarse[i], coarse[i + 1]);
    }
    let nine = refine(&coarse, &mid9);
    let mut mid17 = [0u8; 8];
    for (i, code) in mid17.iter_mut().enumerate() {
        *code = fraction_code(k[2 * i + 1], nine[i], nine[i + 1]);
    }
    ColumnMarginal::Continuous {
        lo,
        hi,
        split,
        mid9,
        mid17,
    }
}

/// Dequantised atom values (ascending, clipped to the unit range).
pub fn atom_values(table: &AtomTable) -> Vec<f64> {
    let raw: Vec<f64> = match &table.values {
        AtomValues::Explicit(v) => v.iter().map(|&h| f16_to_f64(h)).collect(),
        AtomValues::Lattice { base, step, index } => index
            .iter()
            .map(|&i| f16_to_f64(*base) + f16_to_f64(*step) * i as f64)
            .collect(),
    };
    raw.into_iter()
        .map(|v| v.clamp(-UNIT_CLIP, UNIT_CLIP))
        .collect()
}

/// Dequantised atom probabilities (they sum to 1).
pub fn atom_masses(table: &AtomTable) -> Vec<f64> {
    let total: u32 = table.prob.iter().map(|&c| c as u32).sum();
    table
        .prob
        .iter()
        .map(|&c| c as f64 / total.max(1) as f64)
        .collect()
}

/// `(base, step, indices)` if the values sit on `base + step * i`, `i <= 255`,
/// with `step = smallest gap / d` for the smallest `d` in 1..=8 that works.
fn lattice(values: &[f64]) -> Option<AtomValues> {
    let gap = values
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min);
    for d in 1..=LATTICE_DIVISORS {
        let step = gap / d as f64;
        let index: Vec<f64> = values
            .iter()
            .map(|v| ((v - values[0]) / step).round())
            .collect();
        if index[index.len() - 1] > 255.0 {
            return None;
        }
        let (base16, step16) = (f64_to_f16(values[0]), f64_to_f16(step));
        let rebuilt = |i: f64| f16_to_f64(base16) + f16_to_f64(step16) * i;
        let worst = values
            .iter()
            .zip(&index)
            .map(|(v, &i)| (rebuilt(i) - v).abs())
            .fold(0.0f64, f64::max);
        if worst <= LATTICE_TOL && index.windows(2).all(|w| w[0] < w[1]) {
            return Some(AtomValues::Lattice {
                base: base16,
                step: step16,
                index: index.into_iter().map(|i| i as u8).collect(),
            });
        }
    }
    None
}

/// Atom table of a column with at most 16 distinct levels (`levels` ascending,
/// with row counts). Levels under the support floor merge into the nearest
/// surviving level (the lower one on a tie). `None` if fewer than two survive
/// or two survivors collapse to the same binary16 value.
fn fit_atoms(levels: &[(f64, usize)]) -> Option<AtomTable> {
    let keep: Vec<usize> = (0..levels.len())
        .filter(|&i| levels[i].1 >= SUPPORT_K)
        .collect();
    if keep.len() < 2 {
        return None;
    }
    let mut counts: Vec<usize> = keep.iter().map(|&i| levels[i].1).collect();
    for (i, &(value, count)) in levels.iter().enumerate() {
        if keep.contains(&i) {
            continue;
        }
        let mut nearest = 0usize;
        for (slot, &s) in keep.iter().enumerate() {
            if (levels[s].0 - value).abs() < (levels[keep[nearest]].0 - value).abs() {
                nearest = slot;
            }
        }
        counts[nearest] += count;
    }
    let values: Vec<f64> = keep.iter().map(|&i| levels[i].0).collect();
    let explicit: Vec<u16> = values.iter().map(|&v| f64_to_f16(v)).collect();
    if !explicit
        .windows(2)
        .all(|w| f16_to_f64(w[0]) < f16_to_f64(w[1]))
    {
        return None;
    }
    let on_lattice = if values.len() >= LATTICE_MIN_ATOMS {
        lattice(&values)
    } else {
        None
    };
    let weights: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
    Some(AtomTable {
        values: on_lattice.unwrap_or(AtomValues::Explicit(explicit)),
        prob: simplex_codes(&weights, 1),
    })
}

/// Fits one unit-space column (any order, at least 16 finite rows).
pub fn fit_column(values: &[f64], plan: &KnotPlan) -> ColumnMarginal {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut levels: Vec<(f64, usize)> = Vec::new();
    for &v in &sorted {
        let distinct = levels.len();
        match levels.last_mut() {
            Some(last) if last.0 == v => last.1 += 1,
            _ if distinct > MAX_ATOMS => break,
            _ => levels.push((v, 1)),
        }
    }
    if levels.len() == 1 {
        return ColumnMarginal::Constant;
    }
    if levels.len() <= MAX_ATOMS {
        if let Some(table) = fit_atoms(&levels) {
            return ColumnMarginal::Discrete(table);
        }
    }
    quantise_knots(&smoothed_knots(&sorted, plan))
}

/// Extension beyond an end knot, `t` knot steps out. `d0` is the end segment
/// and `d1` its neighbour. A widening or even end (`d1 <= d0`) continues
/// linearly; a narrowing end continues as the geometric series
/// `d0 * (r + r^2 + ...)`, `r = d0 / d1`, which stays inside `d0 * r / (1 - r)`.
fn tail(d0: f64, d1: f64, t: f64) -> f64 {
    if d0 <= 0.0 {
        return 0.0;
    }
    if d1 <= d0 {
        return d0 * t;
    }
    let r = d0 / d1;
    d0 * r * (1.0 - r.powf(t)) / (1.0 - r)
}

/// Decode-side quantile map of one column: normal score in, unit value out.
#[derive(Clone, Debug)]
pub enum ColumnSampler {
    Constant,
    Knots {
        knots: Vec<f64>,
        z_hi: f64,
        dz: f64,
    },
    Atoms {
        values: Vec<f64>,
        thresholds: Vec<f64>,
    },
}

impl ColumnSampler {
    pub fn new(col: &ColumnMarginal, level: PacketLevel, plan: &KnotPlan) -> Self {
        match col {
            ColumnMarginal::Constant => Self::Constant,
            ColumnMarginal::Continuous { .. } => {
                let knots = decoded_knots(col, level.knots());
                let dz = 2.0 * plan.z_hi / (knots.len() - 1) as f64;
                Self::Knots {
                    knots,
                    z_hi: plan.z_hi,
                    dz,
                }
            }
            ColumnMarginal::Discrete(table) => {
                let masses = atom_masses(table);
                let mut cum = 0.0;
                let thresholds = masses[..masses.len() - 1]
                    .iter()
                    .map(|m| {
                        cum += m;
                        norm_ppf(cum)
                    })
                    .collect();
                Self::Atoms {
                    values: atom_values(table),
                    thresholds,
                }
            }
        }
    }

    /// Value at normal score `zq` (the probit of the row's uniform).
    pub fn eval(&self, zq: f64) -> f64 {
        match self {
            Self::Constant => 0.0,
            Self::Atoms { values, thresholds } => {
                values[thresholds.iter().take_while(|&&t| t <= zq).count()]
            }
            Self::Knots { knots: k, z_hi, dz } => {
                let last = k.len() - 1;
                let t = (zq + z_hi) / dz;
                let x = if t <= 0.0 {
                    k[0] - tail(k[1] - k[0], k[2] - k[1], -t)
                } else if t >= last as f64 {
                    k[last]
                        + tail(
                            k[last] - k[last - 1],
                            k[last - 1] - k[last - 2],
                            t - last as f64,
                        )
                } else {
                    let i = (t.floor() as usize).min(last - 1);
                    k[i] + (t - i as f64) * (k[i + 1] - k[i])
                };
                x.clamp(-UNIT_CLIP, UNIT_CLIP)
            }
        }
    }
}

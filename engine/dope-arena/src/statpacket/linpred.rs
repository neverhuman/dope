//! Dequantised target model, shared by the fitter (closed loop) and the
//! sampler: `eta = b + w . x + main effects + pairs` on unit-space features.
//!
//! Dequantisation (normative text: section 5 of the specification):
//!
//! ```text
//! S3   w_kj   = f16(scale_k) * code / 127,   b_k = f16(intercept_k)
//! S8   effect = g(code),  bin(x_j) = #{m : t_jm <= x_j},
//!      t_j    = column quantiles at 1/4, 1/2, 3/4 (the 17-knot curve or the atom table)
//! S10  entries 0..14: g(code) * x_a * x_b;  entry 15: the constant g(code)
//! g(c) = 4 * sign(c) * (c / 127)^2
//! S1   regression: sd = 2^((n0 - 240) / 32) (0 for n0 = 0), times 2^(-d/5) from
//!      level D and 2^(-e/5) at level E, n1 = 16 d + e
//!      classes:    T = 2^((n1 - 128) / 32)
//! ```

use super::marginals::ColumnSampler;
use super::packet::{f16_to_f64, Level, Packet, Task, MAIN_BINS, PAIRS};

/// Weight value = scale * code / 127.
pub const WEIGHT_SCALE: f64 = 127.0;
/// Largest main-effect or pair coefficient.
pub const EFFECT_MAX: f64 = 4.0;
/// Normal score of the upper quartile.
pub const QUARTILE_SCORE: f64 = 0.674_489_750_196_081_7;
/// Residual-SD code of 1.0 and codes per octave.
pub const RESID_SD_UNIT: f64 = 240.0;
pub const RESID_SD_STEPS: f64 = 32.0;
/// Ratio nibbles per octave.
pub const RATIO_STEPS: f64 = 5.0;
/// Temperature code of 1.0 and codes per octave.
pub const TEMPERATURE_UNIT: u8 = 128;
pub const TEMPERATURE_STEPS: f64 = 32.0;
/// Slot of S10 that holds the constant term.
pub const PAIR_OFFSET_SLOT: usize = PAIRS - 1;

/// Main-effect and pair coefficient of an int8 code.
pub fn effect_value(code: i8) -> f64 {
    let t = code as f64 / 127.0;
    EFFECT_MAX * t * t.abs()
}

/// Nearest code by value (the lower magnitude on a tie).
pub fn effect_code(v: f64) -> i8 {
    let want = v.abs().min(EFFECT_MAX);
    let lo = ((want / EFFECT_MAX).sqrt() * 127.0).floor().min(127.0);
    let gap = |c: f64| (effect_value(c as i8) - want).abs();
    let mag = if lo < 127.0 && gap(lo + 1.0) < gap(lo) {
        lo + 1.0
    } else {
        lo
    };
    if v < 0.0 {
        -(mag as i8)
    } else {
        mag as i8
    }
}

pub fn resid_sd_value(code: u8) -> f64 {
    if code == 0 {
        return 0.0;
    }
    ((code as f64 - RESID_SD_UNIT) / RESID_SD_STEPS).exp2()
}

/// Nearest code on the log scale; 0 only for a residual of zero.
pub fn resid_sd_code(sd: f64) -> u8 {
    if sd.is_nan() || sd <= 0.0 {
        return 0;
    }
    (RESID_SD_UNIT + RESID_SD_STEPS * sd.log2())
        .round()
        .clamp(1.0, 255.0) as u8
}

pub fn ratio_value(nibble: u8) -> f64 {
    (-(nibble.min(15) as f64) / RATIO_STEPS).exp2()
}

/// Nearest nibble on the log scale for a ratio in (0, 1].
pub fn ratio_code(ratio: f64) -> u8 {
    if ratio.is_nan() || ratio <= 0.0 {
        return 15;
    }
    (-RATIO_STEPS * ratio.log2()).round().clamp(0.0, 15.0) as u8
}

pub fn temperature_value(code: u8) -> f64 {
    ((code as f64 - TEMPERATURE_UNIT as f64) / TEMPERATURE_STEPS).exp2()
}

pub fn temperature_code(t: f64) -> u8 {
    if t.is_nan() || t <= 0.0 {
        return TEMPERATURE_UNIT;
    }
    (TEMPERATURE_UNIT as f64 + TEMPERATURE_STEPS * t.log2())
        .round()
        .clamp(0.0, 255.0) as u8
}

/// Noise SD of the standardised regression target at `level` (0 for other tasks).
pub fn noise_sd(packet: &Packet, level: Level) -> f64 {
    if packet.task != Task::Regression {
        return 0.0;
    }
    let mut sd = resid_sd_value(packet.resid_sd);
    if level >= Level::D {
        sd *= ratio_value(packet.temperature >> 4);
    }
    if level >= Level::E {
        sd *= ratio_value(packet.temperature & 15);
    }
    sd
}

/// Label temperature (1 for regression, whose second noise byte holds ratios).
pub fn temperature(packet: &Packet) -> f64 {
    if packet.task == Task::Regression {
        1.0
    } else {
        temperature_value(packet.temperature)
    }
}

/// Dequantised S3 weight of output `k`, column `j`.
pub fn weight_value(packet: &Packet, k: usize, j: usize) -> f64 {
    let scale = f16_to_f64(packet.target_scale[k]);
    scale * packet.target_weight[k * packet.cols() + j] as f64 / WEIGHT_SCALE
}

/// Column importance for the budget fallback: the summed absolute S3 weight
/// over outputs, `sum_k f16(scale_k) * |code_kj|`.
pub fn importance(packet: &Packet) -> Vec<f64> {
    let (p, o) = (packet.cols(), packet.outputs());
    let scale: Vec<f64> = packet.target_scale.iter().map(|&h| f16_to_f64(h)).collect();
    (0..p)
        .map(|j| {
            (0..o)
                .map(|k| scale[k] * packet.target_weight[k * p + j].unsigned_abs() as f64)
                .sum()
        })
        .collect()
}

/// Columns by decreasing [`importance`]; ties keep the lower index first.
pub fn importance_order(packet: &Packet) -> Vec<usize> {
    let weight = importance(packet);
    let mut order: Vec<usize> = (0..packet.cols()).collect();
    order.sort_by(|&a, &b| weight[b].total_cmp(&weight[a]).then(a.cmp(&b)));
    order
}

/// Quartile thresholds of one column in unit space.
pub fn bin_thresholds(sampler: &ColumnSampler) -> [f64; 3] {
    [
        sampler.eval(-QUARTILE_SCORE),
        sampler.eval(0.0),
        sampler.eval(QUARTILE_SCORE),
    ]
}

/// Bin 0..=3 of a unit value: the number of thresholds at or below it.
pub fn bin_of(thresholds: &[f64; 3], x: f64) -> usize {
    thresholds.iter().filter(|&&t| t <= x).count()
}

#[derive(Clone, Debug)]
pub struct TargetModel {
    outputs: usize,
    intercept: Vec<f64>,
    /// Columns with a non-zero weight and `outputs` weights for each.
    linear_cols: Vec<usize>,
    linear: Vec<f64>,
    /// Columns with main effects, their thresholds and `outputs x 4` effects for each.
    main_cols: Vec<usize>,
    thresholds: Vec<[f64; 3]>,
    main: Vec<f64>,
    /// Product terms with a non-zero coefficient and `outputs` coefficients for each.
    pair_cols: Vec<(usize, usize)>,
    pair: Vec<f64>,
    offset: Vec<f64>,
}

impl TargetModel {
    /// The model a decoder of `level` evaluates: S3 always, S8 from level D,
    /// S10 at level E. `columns` are the samplers of the packet's columns
    /// (17 knots from level D on, where the thresholds are needed).
    pub fn new(packet: &Packet, level: Level, columns: &[ColumnSampler]) -> Self {
        let (p, o) = (packet.cols(), packet.outputs());
        let intercept = packet
            .target_intercept
            .iter()
            .map(|&h| f16_to_f64(h))
            .collect();
        let mut model = Self {
            outputs: o,
            intercept,
            linear_cols: Vec::new(),
            linear: Vec::new(),
            main_cols: Vec::new(),
            thresholds: Vec::new(),
            main: Vec::new(),
            pair_cols: Vec::new(),
            pair: Vec::new(),
            offset: vec![0.0; o],
        };
        for j in 0..p {
            if (0..o).any(|k| packet.target_weight[k * p + j] != 0) {
                model.linear_cols.push(j);
                model
                    .linear
                    .extend((0..o).map(|k| weight_value(packet, k, j)));
            }
        }
        let width = o * MAIN_BINS;
        if level >= Level::D && level <= packet.level {
            for j in (0..p).filter(|&j| packet.main_mask[j]) {
                let cell = &packet.main_effect[j * width..(j + 1) * width];
                if cell.iter().all(|&c| c == 0) {
                    continue;
                }
                model.main_cols.push(j);
                model.thresholds.push(bin_thresholds(&columns[j]));
                model.main.extend(cell.iter().map(|&c| effect_value(c)));
            }
        }
        if level >= Level::E && packet.pairs.len() == PAIRS {
            for term in &packet.pairs[..PAIR_OFFSET_SLOT] {
                if term.coef.iter().any(|&c| c != 0) {
                    model.pair_cols.push((term.a as usize, term.b as usize));
                    model
                        .pair
                        .extend(term.coef.iter().map(|&c| effect_value(c)));
                }
            }
            let last = &packet.pairs[PAIR_OFFSET_SLOT];
            model.offset = last.coef.iter().map(|&c| effect_value(c)).collect();
        }
        model
    }

    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// Writes `eta` (length `outputs`) for one unit-space row.
    pub fn eta(&self, x: &[f64], out: &mut [f64]) {
        let o = self.outputs;
        for (k, slot) in out.iter_mut().enumerate().take(o) {
            *slot = self.intercept[k] + self.offset[k];
        }
        for (i, &j) in self.linear_cols.iter().enumerate() {
            let w = &self.linear[i * o..(i + 1) * o];
            out.iter_mut().zip(w).for_each(|(e, w)| *e += w * x[j]);
        }
        for (i, &j) in self.main_cols.iter().enumerate() {
            let bin = bin_of(&self.thresholds[i], x[j]);
            let cell = &self.main[i * o * MAIN_BINS..(i + 1) * o * MAIN_BINS];
            for (k, e) in out.iter_mut().enumerate().take(o) {
                *e += cell[k * MAIN_BINS + bin];
            }
        }
        for (i, &(a, b)) in self.pair_cols.iter().enumerate() {
            let product = x[a] * x[b];
            let c = &self.pair[i * o..(i + 1) * o];
            out.iter_mut().zip(c).for_each(|(e, c)| *e += c * product);
        }
    }

    /// `eta` of every row of a row-major `n x p` matrix, row-major `n x outputs`.
    pub fn eta_rows(&self, x: &[f64], p: usize) -> Vec<f64> {
        let mut out = vec![0.0; x.len() / p.max(1) * self.outputs];
        for (row, eta) in x.chunks(p).zip(out.chunks_mut(self.outputs.max(1))) {
            self.eta(row, eta);
        }
        out
    }
}

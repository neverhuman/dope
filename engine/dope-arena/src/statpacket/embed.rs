//! Fixed embedding of a packet: 128 int8 values whose first 64 are the
//! embedding prefix. It is a function of the packet bytes alone (no weights,
//! no rows), so it reveals nothing the packet does not.
//!
//! Construction. A summary of 128 numbers is computed from the packet; every
//! entry is a symmetric function of the columns (moments and quantiles across
//! columns, sorted values, eigenvalues), so reordering the columns of a table
//! changes nothing. The summary is split into a primary and a secondary half
//! of 64 entries; each half is rotated by its own fixed orthonormal 64 x 64
//! matrix and quantised: `e = clamp(round(48 * R s), -127, 127)`. Because the
//! primary half is rotated on its own, the 64-D embedding is exactly the
//! prefix of the 128-D one and loses none of the primary summary.
//!
//! Rotation `b` (0 primary, 1 secondary): rows `r = 0..63` are filled with
//! `normal(EMBED_SEED, 64 * b + r, column)` and orthonormalised in row order
//! by modified Gram-Schmidt.

use super::dependence::{LOADING_SCALE, MEAN_STEP};
use super::linpred::{effect_value, importance, noise_sd, temperature_value, PAIR_OFFSET_SLOT};
use super::marginals::{atom_masses, half_span_value, knot_plan, ColumnSampler};
use super::packet::{
    f16_to_f64, AtomValues, ColumnMarginal, Packet, Task, MAIN_BINS, MAX_RANK, PAIRS,
};
use dope_core::rng::normal;
use dope_core::stats::sym_eigen;

pub const EMBED_DIM: usize = 128;
pub const EMBED_PREFIX: usize = 64;
/// ASCII `DSE1EMBD`.
pub const EMBED_SEED: u64 = 0x4453_4531_454d_4244;
/// Code = `round(EMBED_GAIN * rotated summary)`.
pub const EMBED_GAIN: f64 = 48.0;
/// Normal scores at which every column's quantile curve is read.
pub const EMBED_SCORES: [f64; 9] = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];

/// Mean, SD and quartiles of a list (zeros if empty); the list is sorted
/// first, so the result does not depend on the order of the input.
fn moments(values: &mut [f64]) -> [f64; 5] {
    if values.is_empty() {
        return [0.0; 5];
    }
    values.sort_by(f64::total_cmp);
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let at = |q: f64| {
        let h = q * (values.len() - 1) as f64;
        let lo = h.floor() as usize;
        let hi = (lo + 1).min(values.len() - 1);
        values[lo] + (h - lo as f64) * (values[hi] - values[lo])
    };
    [mean, var.sqrt(), at(0.25), at(0.5), at(0.75)]
}

/// The `count` largest values, descending, padded with zeros.
fn top(mut values: Vec<f64>, count: usize) -> Vec<f64> {
    values.sort_by(|a, b| b.total_cmp(a));
    values.resize(count, 0.0);
    values
}

/// Writes `values` into consecutive slots.
struct Cursor<'a> {
    slots: &'a mut [f64],
    at: usize,
}

impl Cursor<'_> {
    fn put(&mut self, values: &[f64], weight: f64) {
        for v in values {
            self.slots[self.at] = weight * v;
            self.at += 1;
        }
    }
}

/// The permutation-invariant summary: primary half in `0..64`, secondary in `64..128`.
pub fn summary(packet: &Packet) -> [f64; EMBED_DIM] {
    let (p, o, rank) = (packet.cols(), packet.outputs(), packet.level.rank());
    let plan = knot_plan(packet.log2_n);
    let mut out = [0.0; EMBED_DIM];
    let (head, tail) = out.split_at_mut(EMBED_PREFIX);
    let mut first = Cursor { slots: head, at: 0 };
    let mut second = Cursor { slots: tail, at: 0 };

    // Task, class count, width, column kinds.
    let mut task = [0.0; 4];
    task[packet.task as usize] = 1.0;
    first.put(&task, 0.3);
    first.put(&[(1.0 + packet.k as f64).ln() / 256f64.ln()], 1.0);
    // Width, centred at 16 columns.
    first.put(&[((p as f64).log2() - 4.0) / 8.0], 1.0);
    let share = |keep: fn(&ColumnMarginal) -> bool| {
        packet.columns.iter().filter(|c| keep(c)).count() as f64 / p as f64
    };
    // Column kinds: the share that is not continuous, discrete, constant.
    first.put(
        &[
            1.0 - share(|c| c.is_continuous()),
            share(|c| matches!(c, ColumnMarginal::Discrete(_))),
            share(|c| matches!(c, ColumnMarginal::Constant)),
        ],
        1.0,
    );

    // Class shares or label rates, sorted, and their entropy.
    let mut prior: Vec<f64> = packet.prior.iter().map(|&c| c as f64 / 255.0).collect();
    if matches!(packet.task, Task::Binary | Task::Multiclass) {
        prior.push(1.0 - prior.iter().sum::<f64>());
    }
    let entropy = |q: &[f64]| -> f64 {
        let h: f64 = q.iter().filter(|&&v| v > 0.0).map(|v| -v * v.ln()).sum();
        h / (q.len().max(2) as f64).ln()
    };
    let spread = if packet.task == Task::Multilabel {
        0.0
    } else {
        entropy(&prior)
    };
    // Sorted shares as their distance from the uniform share (zero beyond the classes).
    let even = 1.0 / prior.len().max(1) as f64;
    let classes = prior.len().min(8);
    let mut sorted = top(prior, 8);
    if packet.task != Task::Multilabel {
        sorted[..classes].iter_mut().for_each(|q| *q -= even);
    }
    first.put(&sorted, 1.5);
    first.put(&[spread], 1.0);

    // Quantile curves of the varying columns, read at fixed normal scores.
    let samplers: Vec<ColumnSampler> = packet
        .columns
        .iter()
        .filter(|c| !matches!(c, ColumnMarginal::Constant))
        .map(|c| ColumnSampler::new(c, packet.level, &plan))
        .collect();
    for z in EMBED_SCORES {
        let mut gap: Vec<f64> = samplers.iter().map(|s| s.eval(z) - z).collect();
        let [mean, sd, q25, q50, q75] = moments(&mut gap);
        first.put(&[mean, sd, q50], 1.0);
        second.put(&[q25, q75], 1.0);
    }

    // Dependence: uniquenesses, spectrum of the loadings, strength of the
    // modelled correlations, class factor means. With 12 columns or fewer the
    // loadings reproduce the whole matrix, so every measure is taken against
    // the mean communality rather than against zero.
    let scale = LOADING_SCALE * LOADING_SCALE;
    let communality: Vec<f64> = packet
        .loading
        .chunks(MAX_RANK)
        .map(|l| l[..rank].iter().map(|&c| c as i64 * c as i64).sum::<i64>() as f64 / scale)
        .collect();
    let shared = top(communality.clone(), p).iter().sum::<f64>() / p as f64;
    let mut uniqueness: Vec<f64> = communality.iter().map(|h| (1.0 - h).max(0.0)).collect();
    first.put(&moments(&mut uniqueness), 1.5);
    let mut gram = vec![0.0; rank * rank];
    for f in 0..rank {
        for g in 0..rank {
            let sum: i64 = packet
                .loading
                .chunks(MAX_RANK)
                .map(|l| l[f] as i64 * l[g] as i64)
                .sum();
            gram[f * rank + g] = sum as f64 / scale;
        }
    }
    let mut spectrum = [0.0; MAX_RANK];
    if rank > 0 {
        let values = top(sym_eigen(&gram, rank).0, MAX_RANK);
        for (slot, value) in spectrum.iter_mut().zip(values).take(rank.min(p)) {
            *slot = (value - shared) / p as f64;
        }
    }
    first.put(&spectrum[..8], 2.0);
    // Root mean square of the modelled correlations between different columns.
    let squares: f64 = gram.iter().map(|v| v * v).sum::<f64>()
        - top(communality.iter().map(|h| h * h).collect(), p)
            .iter()
            .sum::<f64>();
    let pairs_of_columns = (p * p.saturating_sub(1)).max(1) as f64;
    let coupling = (squares.max(0.0) / pairs_of_columns).sqrt();
    let regime_norm: Vec<f64> = packet
        .class_mean
        .chunks(MAX_RANK)
        .map(|m| {
            let sum: i64 = m[..rank].iter().map(|&c| c as i64 * c as i64).sum();
            (sum as f64).sqrt() * MEAN_STEP
        })
        .collect();

    // Target: how the weight is spread over columns, noise, residual sections.
    let weight = importance(packet);
    let total: f64 = top(weight.clone(), p).iter().sum();
    let shares: Vec<f64> = if total > 0.0 {
        weight.iter().map(|w| w / total).collect()
    } else {
        vec![0.0; p]
    };
    let squares: f64 = top(shares.iter().map(|s| s * s).collect(), p).iter().sum();
    let participation = if squares > 0.0 {
        1.0 / (squares * p as f64)
    } else {
        0.0
    };
    first.put(&top(shares, 3), 1.0);
    first.put(&[participation], 1.0);
    let intercept: Vec<f64> = packet
        .target_intercept
        .iter()
        .map(|&h| f16_to_f64(h))
        .collect();
    let noise = if packet.task == Task::Regression {
        noise_sd(packet, packet.level)
    } else {
        let size: f64 = top(intercept.iter().map(|b| b.abs()).collect(), o)
            .iter()
            .sum();
        (size / o as f64 / 4.0).min(2.0)
    };
    let main_cols = packet.main_mask.iter().filter(|&&on| on).count();
    first.put(&[noise], 1.0);
    first.put(&[coupling], 3.0);

    // Secondary half: atoms, regimes, the rest of the spectrum, target detail, skew.
    let (mut atoms, mut mass_entropy, mut lattice) = (Vec::new(), Vec::new(), 0usize);
    let (mut skew, mut discrete) = (Vec::new(), 0usize);
    for col in &packet.columns {
        match col {
            ColumnMarginal::Discrete(table) => {
                discrete += 1;
                atoms.push(table.prob.len() as f64 / 16.0);
                mass_entropy.push(entropy(&atom_masses(table)));
                lattice += matches!(table.values, AtomValues::Lattice { .. }) as usize;
            }
            ColumnMarginal::Continuous { lo, hi, .. } if *lo > 0 && *hi > 0 => {
                skew.push((half_span_value(*hi) / half_span_value(*lo)).ln());
            }
            _ => {}
        }
    }
    let atom = moments(&mut atoms);
    let lattice_share = lattice as f64 / discrete.max(1) as f64;
    second.put(
        &[
            atom[0],
            atom[1],
            moments(&mut mass_entropy)[0],
            lattice_share,
        ],
        1.0,
    );
    second.put(&top(regime_norm, 8), 1.0 / 6.0);
    second.put(&spectrum[8..], 2.0);
    let mean_abs = |values: Vec<f64>| -> f64 {
        let n = values.len().max(1) as f64;
        top(values.iter().map(|v| v.abs()).collect(), values.len())
            .iter()
            .sum::<f64>()
            / n
    };
    let scales: Vec<f64> = packet
        .target_scale
        .iter()
        .map(|&h| f16_to_f64(h).ln_1p())
        .collect();
    let effects: Vec<f64> = (0..p)
        .filter(|&j| packet.main_mask[j])
        .flat_map(|j| &packet.main_effect[j * o * MAIN_BINS..(j + 1) * o * MAIN_BINS])
        .map(|&c| effect_value(c))
        .collect();
    let (mut products, mut constant) = (Vec::new(), Vec::new());
    if packet.pairs.len() == PAIRS {
        for term in &packet.pairs[..PAIR_OFFSET_SLOT] {
            products.extend(term.coef.iter().map(|&c| effect_value(c)));
        }
        constant.extend(
            packet.pairs[PAIR_OFFSET_SLOT]
                .coef
                .iter()
                .map(|&c| effect_value(c)),
        );
    }
    let heat = if packet.task == Task::Regression {
        0.0
    } else {
        temperature_value(packet.temperature) - 1.0
    };
    let skewness = moments(&mut skew);
    second.put(
        &[
            mean_abs(scales),
            moments(&mut intercept.clone())[1].min(4.0) / 4.0,
            mean_abs(effects),
            mean_abs(products),
            mean_abs(constant),
            heat,
            skewness[0],
            skewness[1],
            main_cols as f64 / p as f64,
        ],
        1.0,
    );
    out
}

/// Orthonormal 64 x 64 rotation of block `block`, row-major.
pub fn rotation(block: u64) -> Vec<f64> {
    let d = EMBED_PREFIX;
    let mut rows = vec![0.0; d * d];
    for r in 0..d {
        let mut row: Vec<f64> = (0..d)
            .map(|c| normal(EMBED_SEED, block * d as u64 + r as u64, c as u64))
            .collect();
        for earlier in rows[..r * d].chunks(d) {
            let dot: f64 = row.iter().zip(earlier).map(|(a, b)| a * b).sum();
            row.iter_mut().zip(earlier).for_each(|(a, b)| *a -= dot * b);
        }
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        row.iter_mut().for_each(|v| *v /= norm);
        rows[r * d..(r + 1) * d].copy_from_slice(&row);
    }
    rows
}

/// Rotates and quantises a summary: each half of 64 entries by its own rotation.
pub fn embed_summary(source: &[f64; EMBED_DIM]) -> [i8; EMBED_DIM] {
    let mut out = [0i8; EMBED_DIM];
    for (block, (half, codes)) in source
        .chunks(EMBED_PREFIX)
        .zip(out.chunks_mut(EMBED_PREFIX))
        .enumerate()
    {
        let rows = rotation(block as u64);
        for (row, code) in rows.chunks(EMBED_PREFIX).zip(codes) {
            let value: f64 = row.iter().zip(half).map(|(a, b)| a * b).sum();
            *code = (EMBED_GAIN * value).round().clamp(-127.0, 127.0) as i8;
        }
    }
    out
}

/// The 128-D embedding; its first 64 values are the 64-D embedding.
pub fn embed(packet: &Packet) -> [i8; EMBED_DIM] {
    embed_summary(&summary(packet))
}

/// Cosine similarity of two embeddings (0 if either is all zero).
pub fn cosine(a: &[i8], b: &[i8]) -> f64 {
    let dot =
        |x: &[i8], y: &[i8]| -> f64 { x.iter().zip(y).map(|(&u, &v)| u as f64 * v as f64).sum() };
    let norm = (dot(a, a) * dot(b, b)).sqrt();
    if norm > 0.0 {
        dot(a, b) / norm
    } else {
        0.0
    }
}

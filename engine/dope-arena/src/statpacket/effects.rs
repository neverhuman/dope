//! Residual target sections: S8 (four-bin main effects) and S10 (fifteen
//! pairwise products and one constant). Both are fitted on top of the
//! dequantised sections before them, so the stored S3 model stays the plain
//! linear fit at every level.
//!
//! S8. The bins of a column are cut at its own decoded quartiles (the 17-knot
//! curve or the atom table), so the sampler needs no stored threshold. A
//! column enters only if the between-bin sum of squares of the response
//! residual passes a chi-square gate; the columns that pass are fitted jointly
//! with a unit ridge. A bin with fewer than 10 FIT rows gets no coefficient,
//! and a column whose codes are all zero is left out of the section.
//!
//! S10. Products `x_a * x_b` (`a <= b`, so squares are allowed) are chosen
//! greedily by their score statistic against the response residual, each
//! choice deflating the residual, until the best candidate fails the gate;
//! the coefficients are then fitted jointly with a free constant, which is
//! stored in the last entry.
//!
//! The gate ([`gate`]) is what keeps a table without such structure from
//! storing fitted noise: on a purely linear table both sections stay empty.

use super::glm::{Dense, OneHot, NO_COEF};
use super::linpred::{
    bin_of, bin_thresholds, effect_code, importance_order, temperature, TargetModel,
    PAIR_OFFSET_SLOT,
};
use super::marginals::ColumnSampler;
use super::packet::{ColumnMarginal, Level, Packet, PairTerm, MAIN_BINS, SUPPORT_K};
use super::target::{fit_block, FitRows, Resp};
use dope_core::stats::norm_ppf;

/// Most columns that receive main effects, taken in importance order.
pub const MAIN_FIT_COLS_MAX: usize = 96;
/// Ridge weight of every bin coefficient.
pub const MAIN_RIDGE: f64 = 1.0;
/// Columns whose products are scanned, taken in importance order.
pub const PAIR_CANDIDATE_COLS: usize = 48;
/// A product needs at least this variance to be a candidate.
pub const PAIR_VARIANCE_FLOOR: f64 = 1e-9;
/// Family-wise level of the gate on main effects and pairs.
pub const GATE_ALPHA: f64 = 0.05;
/// An output whose residual mean square is below this takes no part in a gate.
pub const VARIANCE_FLOOR: f64 = 1e-12;

/// The first `limit` non-constant columns by importance, ascending.
fn leading_columns(packet: &Packet, limit: usize) -> Vec<usize> {
    let mut cols: Vec<usize> = importance_order(packet)
        .into_iter()
        .filter(|&j| !matches!(packet.columns[j], ColumnMarginal::Constant))
        .take(limit)
        .collect();
    cols.sort_unstable();
    cols
}

/// `T * eta` of the FIT rows for the sections a decoder of `level` uses.
fn scaled_eta(
    packet: &Packet,
    level: Level,
    rows: &FitRows,
    samplers: &[ColumnSampler],
) -> Vec<f64> {
    let t = temperature(packet);
    let mut eta = TargetModel::new(packet, level, samplers).eta_rows(rows.x, rows.p);
    eta.iter_mut().for_each(|e| *e *= t);
    eta
}

/// Chi-square bound a candidate must reach to be stored: the upper
/// `GATE_ALPHA / candidates` point of a chi-square with `df` degrees of
/// freedom (Wilson-Hilferty). It keeps sampling noise out of S8 and S10.
pub fn gate(df: f64, candidates: usize) -> f64 {
    let z = norm_ppf(1.0 - GATE_ALPHA / candidates.max(1) as f64);
    let spread = 2.0 / (9.0 * df);
    df * (1.0 - spread + z * spread.sqrt()).powi(3)
}

/// Mean square of every residual column.
fn residual_variance(resid: &[f64], o: usize) -> Vec<f64> {
    let n = (resid.len() / o.max(1)).max(1) as f64;
    let mut out = vec![0.0; o];
    for r in resid.chunks(o) {
        out.iter_mut().zip(r).for_each(|(v, r)| *v += r * r / n);
    }
    out
}

pub fn fit_main_effects(
    packet: &mut Packet,
    rows: &FitRows,
    resp: &Resp,
    samplers: &[ColumnSampler],
) {
    let (n, p, o) = (rows.n, rows.p, packet.outputs());
    let cols = leading_columns(packet, MAIN_FIT_COLS_MAX);
    let t = temperature(packet);
    let offset = scaled_eta(packet, Level::A, rows, samplers);
    let resid = resp.residuals(&offset);
    let variance = residual_variance(&resid, o);
    let live_outputs = variance.iter().filter(|&&v| v > VARIANCE_FLOOR).count();
    // Coefficient slots (column, bin) of the columns that pass the gate.
    let mut slots: Vec<(usize, usize)> = Vec::new();
    let mut kept: Vec<(usize, Vec<usize>, [u32; MAIN_BINS])> = Vec::new();
    for &j in &cols {
        let thresholds = bin_thresholds(&samplers[j]);
        let bins: Vec<usize> = (0..n)
            .map(|i| bin_of(&thresholds, rows.x[i * p + j]))
            .collect();
        let mut count = [0usize; MAIN_BINS];
        bins.iter().for_each(|&b| count[b] += 1);
        let live = |b: usize| count[b] >= SUPPORT_K;
        if (0..MAIN_BINS).filter(|&b| live(b)).count() < 2 {
            continue;
        }
        // Groups: every live bin, and one group for the rows of all other bins.
        let group = |b: usize| if live(b) { b } else { MAIN_BINS };
        let mut size = [0.0f64; MAIN_BINS + 1];
        let mut sum = vec![0.0f64; (MAIN_BINS + 1) * o];
        for (i, &b) in bins.iter().enumerate() {
            size[group(b)] += 1.0;
            for k in 0..o {
                sum[group(b) * o + k] += resid[i * o + k];
            }
        }
        let groups = size.iter().filter(|&&s| s > 0.0).count();
        let mut statistic = 0.0;
        for (k, &v) in variance
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > VARIANCE_FLOOR)
        {
            let total: f64 = (0..=MAIN_BINS).map(|g| sum[g * o + k]).sum();
            let between: f64 = (0..=MAIN_BINS)
                .filter(|&g| size[g] > 0.0)
                .map(|g| sum[g * o + k].powi(2) / size[g])
                .sum();
            statistic += (between - total * total / n as f64) / v;
        }
        let df = ((groups - 1) * live_outputs) as f64;
        if df <= 0.0 || statistic < gate(df, cols.len()) {
            continue;
        }
        let mut slot = [NO_COEF; MAIN_BINS];
        for (b, s) in slot.iter_mut().enumerate() {
            if live(b) {
                *s = slots.len() as u32;
                slots.push((j, b));
            }
        }
        kept.push((j, bins, slot));
    }
    let (d, groups) = (slots.len(), kept.len());
    if d == 0 {
        return;
    }
    let mut id = vec![NO_COEF; n * groups];
    for (g, (_, bins, slot)) in kept.iter().enumerate() {
        for (i, &b) in bins.iter().enumerate() {
            id[i * groups + g] = slot[b];
        }
    }
    let basis = OneHot {
        id,
        n,
        groups,
        d,
        penalty: vec![MAIN_RIDGE; d],
    };
    let coef = fit_block(&basis, resp, &offset, &vec![0.0; o]);
    for k in 0..o {
        for (s, &(j, b)) in slots.iter().enumerate() {
            let code = effect_code(coef[k * d + s] / t);
            packet.main_effect[(j * o + k) * MAIN_BINS + b] = code;
            if code != 0 {
                packet.main_mask[j] = true;
            }
        }
    }
}

/// Sums of one candidate product against the residual columns.
struct ProductSums {
    /// Centred sum of squares of the product.
    spread: f64,
    /// Centred cross products with every residual column.
    cross: Vec<f64>,
    mean: f64,
}

fn product_sums(a: &[f64], b: &[f64], resid: &[f64], resid_sum: &[f64]) -> ProductSums {
    let (n, o) = (a.len() as f64, resid_sum.len());
    let (mut sum, mut squares) = (0.0, 0.0);
    let mut cross = vec![0.0; o];
    for ((x, y), r) in a.iter().zip(b).zip(resid.chunks(o)) {
        let z = x * y;
        sum += z;
        squares += z * z;
        cross.iter_mut().zip(r).for_each(|(c, r)| *c += z * r);
    }
    for (c, total) in cross.iter_mut().zip(resid_sum) {
        *c -= sum * total / n;
    }
    ProductSums {
        spread: squares - sum * sum / n,
        cross,
        mean: sum / n,
    }
}

/// Greedy choice of up to 15 products of candidate columns (indices into
/// `columns`): each round takes the product with the largest score statistic
/// `sum_k cross_k^2 / (spread * var_k)` against the current residual, stops
/// when it is below the gate, and deflates the residual by the chosen product.
fn choose_pairs(columns: &[Vec<f64>], mut resid: Vec<f64>, o: usize) -> Vec<(usize, usize)> {
    let n = columns.first().map_or(0, Vec::len);
    let candidates = columns.len() * (columns.len() + 1) / 2;
    let mut chosen: Vec<(usize, usize)> = Vec::new();
    while chosen.len() < PAIR_OFFSET_SLOT {
        let mut resid_sum = vec![0.0; o];
        for r in resid.chunks(o) {
            resid_sum.iter_mut().zip(r).for_each(|(s, r)| *s += r);
        }
        let variance = residual_variance(&resid, o);
        let live_outputs = variance.iter().filter(|&&v| v > VARIANCE_FLOOR).count();
        let mut best: Option<(f64, usize, usize)> = None;
        for a in 0..columns.len() {
            for b in a..columns.len() {
                if chosen.contains(&(a, b)) {
                    continue;
                }
                let sums = product_sums(&columns[a], &columns[b], &resid, &resid_sum);
                if sums.spread <= PAIR_VARIANCE_FLOOR * n as f64 {
                    continue;
                }
                let score: f64 = sums
                    .cross
                    .iter()
                    .zip(&variance)
                    .filter(|(_, &v)| v > VARIANCE_FLOOR)
                    .map(|(c, v)| c * c / (sums.spread * v))
                    .sum();
                if best.is_none_or(|(top, ..)| score > top) {
                    best = Some((score, a, b));
                }
            }
        }
        let Some((score, a, b)) = best else {
            break;
        };
        if live_outputs == 0 || score.is_nan() || score < gate(live_outputs as f64, candidates) {
            break;
        }
        chosen.push((a, b));
        let sums = product_sums(&columns[a], &columns[b], &resid, &resid_sum);
        for (i, r) in resid.chunks_mut(o).enumerate() {
            let z = columns[a][i] * columns[b][i] - sums.mean;
            for (r, c) in r.iter_mut().zip(&sums.cross) {
                *r -= z * c / sums.spread;
            }
        }
    }
    chosen
}

pub fn fit_pairs(packet: &mut Packet, rows: &FitRows, resp: &Resp, samplers: &[ColumnSampler]) {
    let (n, p, o) = (rows.n, rows.p, packet.outputs());
    let cand = leading_columns(packet, PAIR_CANDIDATE_COLS);
    let columns: Vec<Vec<f64>> = cand
        .iter()
        .map(|&j| (0..n).map(|i| rows.x[i * p + j]).collect())
        .collect();
    let t = temperature(packet);
    let offset = scaled_eta(packet, Level::D, rows, samplers);
    let chosen = choose_pairs(&columns, resp.residuals(&offset), o);
    if chosen.is_empty() {
        return;
    }
    // Joint fit of the chosen products with a free constant.
    let d = chosen.len() + 1;
    let mut a = Vec::with_capacity(n * d);
    for i in 0..n {
        a.push(1.0);
        a.extend(chosen.iter().map(|&(u, v)| columns[u][i] * columns[v][i]));
    }
    let mut penalty = vec![0.0];
    for s in 1..d {
        let mean = (0..n).map(|i| a[i * d + s]).sum::<f64>() / n as f64;
        let var = (0..n).map(|i| (a[i * d + s] - mean).powi(2)).sum::<f64>() / n as f64;
        penalty.push(var);
    }
    let basis = Dense { a, n, d, penalty };
    let coef = fit_block(&basis, resp, &offset, &vec![0.0; o]);
    for (s, &(u, v)) in chosen.iter().enumerate() {
        let codes: Vec<i8> = (0..o)
            .map(|k| effect_code(coef[k * d + 1 + s] / t))
            .collect();
        if codes.iter().any(|&c| c != 0) {
            packet.pairs[s] = PairTerm {
                a: cand[u] as u16,
                b: cand[v] as u16,
                coef: codes,
            };
        }
    }
    // Closed loop: the constant is refitted against the quantised products.
    let with_pairs = scaled_eta(packet, Level::E, rows, samplers);
    let ones = Dense {
        a: vec![1.0; n],
        n,
        d: 1,
        penalty: vec![0.0],
    };
    let constant = fit_block(&ones, resp, &with_pairs, &vec![0.0; o]);
    packet.pairs[PAIR_OFFSET_SLOT].coef = constant.iter().map(|c| effect_code(c / t)).collect();
}

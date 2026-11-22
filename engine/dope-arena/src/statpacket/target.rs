//! Target sections: the S3 linear target, the S1 noise bytes and the order in
//! which the residual sections (S8 and S10, see `effects`) are fitted.
//!
//! Every block is fitted against the *dequantised* blocks before it (closed
//! loop), on unit-space FIT rows:
//!
//! 1. S3: ridge / logistic / multinomial fit with an unpenalised intercept and
//!    the penalty `s_j^2 w_j^2` (`s_j` the SD of column `j`), which is
//!    `Ridge(alpha = 1)` and `LogisticRegression(C = 1)` on standardised
//!    inputs; int8 weights, then the intercept is refitted on its own.
//! 2. Classification temperature (one Newton search on the stored model).
//! 3. S8 main effects, 4. S10 pairs and constant, 5. regression noise codes.
//!
//! Support floor: a class (or a multilabel side) with fewer than 10 FIT rows
//! takes no part in any fit and gets no parameters.

use super::effects::{fit_main_effects, fit_pairs};
use super::glm::Dense;
use super::linpred::{
    ratio_code, ratio_value, resid_sd_code, resid_sd_value, temperature_code, weight_value,
    TargetModel, TEMPERATURE_UNIT, WEIGHT_SCALE,
};
use super::marginals::{knot_plan, ColumnSampler};
use super::packet::{f16_to_f64, f64_to_f16, Level, Packet, PairTerm, Task, PAIRS};
use super::response::temperature_slope;
pub use super::response::{fit_block, Resp, Response};
use super::StatError;

/// Most columns that enter the S3 fit; wider tables are screened first.
pub const TARGET_FIT_COLS_MAX: usize = 512;
/// A column whose unit-space SD is below this carries no weight.
pub const SD_FLOOR: f64 = 1e-6;
/// Newton steps of the temperature search and its clamp.
pub const TEMPERATURE_SEARCH_STEPS: usize = 6;
pub const TEMPERATURE_RANGE: (f64, f64) = (0.25, 4.0);

/// Unit-space FIT rows, row-major `n x p`.
#[derive(Clone, Copy)]
pub struct FitRows<'a> {
    pub x: &'a [f64],
    pub n: usize,
    pub p: usize,
}

/// Which residual sections to fit (skipped when the byte budget cannot hold them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetPlan {
    pub main_effects: bool,
    pub pairs: bool,
}

/// Population SD of every unit-space column.
pub fn column_sd(rows: &FitRows) -> Vec<f64> {
    let (n, p) = (rows.n as f64, rows.p);
    let mut mean = vec![0.0; p];
    for row in rows.x.chunks(p) {
        mean.iter_mut().zip(row).for_each(|(m, v)| *m += v);
    }
    mean.iter_mut().for_each(|m| *m /= n);
    let mut var = vec![0.0; p];
    for row in rows.x.chunks(p) {
        for (j, v) in row.iter().enumerate() {
            var[j] += (v - mean[j]) * (v - mean[j]);
        }
    }
    var.into_iter().map(|v| (v / n).sqrt()).collect()
}

/// The `TARGET_FIT_COLS_MAX` columns with the largest summed squared
/// correlation with the response residuals of the intercept-only model.
fn screen(rows: &FitRows, resp: &Resp, sd: &[f64], cols: Vec<usize>) -> Vec<usize> {
    let o = resp.outputs();
    let start = resp.intercept_start();
    let flat: Vec<f64> = (0..rows.n * o).map(|i| start[i % o]).collect();
    let resid = resp.residuals(&flat);
    let mut score = vec![0.0; rows.p];
    for &j in &cols {
        for k in 0..o {
            let cov: f64 = (0..rows.n)
                .map(|i| rows.x[i * rows.p + j] * resid[i * o + k])
                .sum();
            score[j] += (cov / sd[j]).powi(2);
        }
    }
    let mut order = cols;
    order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
    order.truncate(TARGET_FIT_COLS_MAX);
    order.sort_unstable();
    order
}

fn fit_linear(packet: &mut Packet, rows: &FitRows, resp: &Resp, sd: &[f64]) {
    let (n, p, o) = (rows.n, rows.p, packet.outputs());
    let mut cols: Vec<usize> = (0..p).filter(|&j| sd[j] > SD_FLOOR).collect();
    if cols.len() > TARGET_FIT_COLS_MAX {
        cols = screen(rows, resp, sd, cols);
    }
    let d = cols.len() + 1;
    let mut a = Vec::with_capacity(n * d);
    for row in rows.x.chunks(p) {
        a.push(1.0);
        a.extend(cols.iter().map(|&j| row[j]));
    }
    let mut penalty = vec![0.0];
    penalty.extend(cols.iter().map(|&j| sd[j] * sd[j]));
    let basis = Dense { a, n, d, penalty };
    let start = resp.intercept_start();
    let coef = fit_block(&basis, resp, &vec![0.0; n * o], &start);
    packet.target_weight.fill(0);
    for k in 0..o {
        let w = &coef[k * d + 1..(k + 1) * d];
        let bits = f64_to_f16(w.iter().fold(0.0f64, |m, v| m.max(v.abs())));
        let scale = f16_to_f64(bits);
        packet.target_scale[k] = bits;
        packet.target_intercept[k] = 0;
        if scale > 0.0 {
            for (&j, &v) in cols.iter().zip(w) {
                let code = (WEIGHT_SCALE * v / scale).round().clamp(-127.0, 127.0);
                packet.target_weight[k * p + j] = code as i8;
            }
        }
    }
    // Closed loop: the intercept is refitted against the quantised weights.
    let linear = TargetModel::new(packet, Level::A, &[]).eta_rows(rows.x, p);
    let ones = Dense {
        a: vec![1.0; n],
        n,
        d: 1,
        penalty: vec![0.0],
    };
    let first: Vec<f64> = (0..o).map(|k| coef[k * d]).collect();
    let intercept = fit_block(&ones, resp, &linear, &first);
    for (bits, b) in packet.target_intercept.iter_mut().zip(intercept) {
        *bits = f64_to_f16(b);
    }
}

/// Ridge quadratic `sum_j s_j^2 |w_j|^2` of the stored weights in the
/// parametrisation the fit penalised (centred over the supported classes).
fn penalty_quadratic(packet: &Packet, resp: &Resp, sd: &[f64]) -> f64 {
    let (p, o) = (packet.cols(), packet.outputs());
    let mut total = 0.0;
    for (j, s) in sd.iter().enumerate().take(p) {
        let w: Vec<f64> = (0..o).map(|k| weight_value(packet, k, j)).collect();
        let squares: f64 = w.iter().map(|v| v * v).sum();
        let centre = match resp {
            Resp::Softmax { classes, .. } if !classes.is_empty() => {
                w.iter().sum::<f64>().powi(2) / classes.len() as f64
            }
            _ => 0.0,
        };
        total += s * s * (squares - centre);
    }
    total
}

/// Temperature code: the maximiser over `T` of the penalised likelihood of the
/// stored S3 model. It is 1 at the unquantised optimum, so it only absorbs the
/// scale error of the int8 weights.
fn fit_temperature(packet: &mut Packet, rows: &FitRows, resp: &Resp, sd: &[f64]) {
    if matches!(resp, Resp::Gaussian(_)) {
        return;
    }
    let eta = TargetModel::new(packet, Level::A, &[]).eta_rows(rows.x, rows.p);
    let quad = penalty_quadratic(packet, resp, sd);
    let mut t = 1.0f64;
    for _ in 0..TEMPERATURE_SEARCH_STEPS {
        let (slope, curve) = temperature_slope(resp, &eta, t);
        let denominator = curve + quad;
        if denominator <= 1e-12 {
            break;
        }
        t = (t + (slope - t * quad) / denominator).clamp(TEMPERATURE_RANGE.0, TEMPERATURE_RANGE.1);
    }
    packet.temperature = temperature_code(t);
}

/// Samplers of every column at 17 knots (the thresholds of S8 use them).
pub fn column_samplers(packet: &Packet) -> Vec<ColumnSampler> {
    let plan = knot_plan(packet.log2_n);
    packet
        .columns
        .iter()
        .map(|col| ColumnSampler::new(col, Level::E, &plan))
        .collect()
}

/// Regression noise codes against the packet as it stands (its level and
/// masks): `resid_sd` for S3 alone, two ratio nibbles for levels D and E.
/// `sd^2 = RSS / max(n - df, n / 2)` with `df` the non-zero codes in use.
fn set_noise(packet: &mut Packet, rows: &FitRows, resp: &Resp, samplers: &[ColumnSampler]) {
    let Resp::Gaussian(y) = resp else {
        packet.resid_sd = 0;
        return;
    };
    let n = rows.n as f64;
    let count = |codes: &[i8]| codes.iter().filter(|&&c| c != 0).count() as f64;
    let pair_codes: f64 = packet.pairs.iter().map(|t| count(&t.coef)).sum();
    let df = [
        1.0 + count(&packet.target_weight),
        count(&packet.main_effect),
        pair_codes,
    ];
    let mut sd = [0.0f64; 3];
    let mut used = 0.0;
    for (tier, level) in [Level::A, Level::D, Level::E].into_iter().enumerate() {
        used += df[tier];
        let eta = TargetModel::new(packet, level, samplers).eta_rows(rows.x, rows.p);
        let rss: f64 = y.iter().zip(&eta).map(|(y, e)| (y - e) * (y - e)).sum();
        sd[tier] = (rss / (n - used).max(n / 2.0)).sqrt();
    }
    packet.resid_sd = resid_sd_code(sd[0]);
    let base = resid_sd_value(packet.resid_sd);
    let (mut d, mut e) = (0u8, 0u8);
    if base > 0.0 {
        d = ratio_code(sd[1] / base);
        e = ratio_code(sd[2] / (base * ratio_value(d)));
    }
    packet.temperature = d << 4 | e;
}

fn clear_residual_sections(packet: &mut Packet) {
    let o = packet.outputs();
    packet.main_mask.fill(false);
    packet.main_effect.fill(0);
    packet.pairs = (0..PAIRS)
        .map(|_| PairTerm {
            a: 0,
            b: 0,
            coef: vec![0; o],
        })
        .collect();
}

/// Fits S3, the temperature, S8, S10 and the noise codes of a level-E packet
/// whose feature sections are complete.
pub fn fit_target(
    packet: &mut Packet,
    rows: &FitRows,
    response: Response,
    plan: TargetPlan,
) -> Result<(), StatError> {
    if packet.level != Level::E || rows.p != packet.cols() || rows.x.len() != rows.n * rows.p {
        return Err(StatError::Invalid(
            "target fit needs a level-E packet of the same shape",
        ));
    }
    let resp = Resp::new(packet, response, rows.n)?;
    let sd = column_sd(rows);
    let samplers = column_samplers(packet);
    packet.resid_sd = 0;
    packet.temperature = if packet.task == Task::Regression {
        0
    } else {
        TEMPERATURE_UNIT
    };
    clear_residual_sections(packet);
    fit_linear(packet, rows, &resp, &sd);
    fit_temperature(packet, rows, &resp, &sd);
    if plan.main_effects {
        fit_main_effects(packet, rows, &resp, &samplers);
    }
    if plan.pairs {
        fit_pairs(packet, rows, &resp, &samplers);
    }
    set_noise(packet, rows, &resp, &samplers);
    Ok(())
}

/// Recomputes the regression noise codes after the budget step changed the
/// level or the masks of `packet` (a no-op for the other tasks).
pub fn refresh_noise(
    packet: &mut Packet,
    rows: &FitRows,
    response: Response,
) -> Result<(), StatError> {
    let resp = Resp::new(packet, response, rows.n)?;
    let samplers = column_samplers(packet);
    if packet.task == Task::Regression {
        set_noise(packet, rows, &resp, &samplers);
    }
    Ok(())
}

//! Decode-aligned CDF interval coordinates, true-loss validation helpers, and
//! deterministic synthetic-X intercept calibration for `DSE8` v1.
//!
//! Eliminates:
//! 1. Discrete/tied quantile-bin misalignment between FIT and decode `Phi(z_lat)` bins.
//! 2. Asymmetric clamped classification residual mean bias on imbalanced classes.
//! 3. Post-nonlinear class prior & regression mean drift on decoded synthetic X rows
//!    (decode remains 100% row-local!).

mod midrank;

pub use midrank::project_fit_midrank;

use super::copula_regimes::{RegimesSectionV8, V8_MAX_RANK};
use super::decode::{build_decode_plan_v8, decode_dse8_row_at};
use super::marginals::MarginalColV8;
use super::nominal::{decode_nominal_group, NominalGroupV8};
use super::packet::Dse8Packet;
use super::rng_v8::{
    counter_normal, counter_uniform, sample_categorical, STREAM_FACTOR, STREAM_NOISE, STREAM_REGIME,
};
use super::scale_v8::V8_MAX_CLASSES;

const CAL_SYNTH_ROWS: usize = 512;
const CAL_SEED: u64 = 0xCA11_B8A7_D5E8_0001;
const X_PROBE_ROWS: usize = 256;
const X_PROBE_SEED: u64 = 0x51A7_E0B1_D5E8_0002;

/// Decode `X_PROBE_ROWS` deterministic synthetic X rows (in active unit space, missing -> 0.0)
/// directly from `(marginals, loadings, nominal_groups, regimes)` to guard against collinear
/// errors-in-variables weight explosion on quantized synthetic X!
pub fn sample_synthetic_x_probe(
    marginals: &[MarginalColV8],
    loadings: &[[f32; V8_MAX_RANK]],
    nominal_groups: &[NominalGroupV8],
    regimes: &RegimesSectionV8,
    rank_use: usize,
    z_max: f32,
) -> Vec<Vec<f32>> {
    let p = marginals.len();
    let mut noise_sd = vec![1.0f32; p];
    for j in 0..p {
        let sq: f32 = loadings[j].iter().map(|v| v * v).sum();
        noise_sd[j] = (1.0 - sq).max(1e-4).sqrt();
    }
    let reg_weights: Vec<f64> = regimes.regimes.iter().map(|rg| rg.weight as f64).collect();
    let mut x_syn = vec![vec![0.0f32; p]; X_PROBE_ROWS];
    let mut z_factor = [0.0f32; V8_MAX_RANK];
    let mut z_lat = vec![0.0f32; p];
    let mut miss = vec![false; p];
    for (row_idx, row_out) in x_syn.iter_mut().enumerate() {
        let ri = row_idx as u64;
        for (r, slot) in z_factor.iter_mut().enumerate().take(rank_use) {
            *slot = counter_normal(X_PROBE_SEED, ri, STREAM_FACTOR, r as u64) as f32;
        }
        if regimes.regimes.len() >= 2 {
            let u_rg = counter_uniform(X_PROBE_SEED, ri, STREAM_REGIME, 0);
            let chosen = sample_categorical(&reg_weights, u_rg);
            let reg = &regimes.regimes[chosen];
            for (r, &mu) in reg.mean_f.iter().enumerate().take(rank_use) {
                let shr = (1.0 - (mu * mu).min(0.75)).max(0.25).sqrt();
                z_factor[r] = mu + shr * z_factor[r];
            }
        }
        for j in 0..p {
            let eps = counter_normal(X_PROBE_SEED, ri, STREAM_NOISE, j as u64) as f32;
            let mut z = eps * noise_sd[j];
            for r in 0..rank_use {
                z += z_factor[r] * loadings[j][r];
            }
            z_lat[j] = z;
        }
        for j in 0..p {
            let (val, is_m) = marginals[j].decode_value(
                z_lat[j],
                &marginals[j].knots17,
                z_max,
                X_PROBE_SEED,
                ri,
                j as u64,
            );
            row_out[j] = if is_m { 0.0 } else { val };
            miss[j] = is_m;
        }
        for (gi, grp) in nominal_groups.iter().enumerate() {
            decode_nominal_group(grp, &z_lat, row_out, &mut miss, X_PROBE_SEED, ri, gi as u64);
        }
        for j in 0..p {
            if miss[j] || !row_out[j].is_finite() {
                row_out[j] = 0.0;
            }
        }
    }
    x_syn
}

/// Ratio `max_c std(W_c x_syn) / max(std(W_c x_fit), 0.25)` detecting collinear nullspace explosion.
pub fn linear_syn_to_fit_sd_ratio(
    w: &[Vec<f64>],
    x_unit: &[Vec<f32>],
    x_syn_probe: &[Vec<f32>],
) -> f64 {
    let p = x_unit.len();
    let n_fit = x_unit.first().map(|c| c.len()).unwrap_or(0);
    let n_syn = x_syn_probe.len();
    if p == 0 || n_fit < 4 || n_syn < 8 {
        return 1.0;
    }
    let mut worst = 1.0f64;
    for wc in w {
        let (mut mf, mut vf) = (0.0, 0.0);
        let mut pf = Vec::with_capacity(n_fit);
        for r in 0..n_fit {
            let mut s = 0.0;
            for j in 0..p {
                let v = x_unit[j][r];
                if v.is_finite() {
                    s += v as f64 * wc[j];
                }
            }
            pf.push(s);
            mf += s;
        }
        mf /= n_fit as f64;
        for &s in &pf {
            vf += (s - mf) * (s - mf);
        }
        let sd_fit = (vf / n_fit as f64).sqrt();

        let (mut ms, mut vs) = (0.0, 0.0);
        let mut ps = Vec::with_capacity(n_syn);
        for row in x_syn_probe {
            let mut s = 0.0;
            for j in 0..p {
                s += row[j] as f64 * wc[j];
            }
            ps.push(s);
            ms += s;
        }
        ms /= n_syn as f64;
        for &s in &ps {
            vs += (s - ms) * (s - ms);
        }
        let sd_syn = (vs / n_syn as f64).sqrt();
        worst = worst.max(sd_syn / sd_fit.max(0.25));
    }
    worst
}

/// Exact empirical CDF tie-group intervals `[u_lo, u_hi]` per column matching decode `Phi(z_lat)`.
#[derive(Clone, Debug)]
pub struct CdfIntervals {
    pub p: usize,
    pub n: usize,
    pub u_lo: Vec<Vec<f32>>,
    pub u_hi: Vec<Vec<f32>>,
    pub u_mid: Vec<Vec<f32>>,
    pub z_mid: Vec<Vec<f64>>,
}

impl CdfIntervals {
    pub fn build(x_unit: &[Vec<f32>]) -> Self {
        let p = x_unit.len();
        let n = x_unit.first().map(|c| c.len()).unwrap_or(0);
        let mut u_lo = vec![vec![0.5f32; n]; p];
        let mut u_hi = vec![vec![0.5f32; n]; p];
        let mut u_mid = vec![vec![0.5f32; n]; p];
        let mut z_mid = vec![vec![0.0f64; n]; p];

        for j in 0..p {
            let obs = midrank::sorted_observations(&x_unit[j]);
            if obs.len() < 2 {
                continue;
            }
            let nf = obs.len() as f64;
            for group in midrank::groups(&obs) {
                let lo = (group.start as f64 / nf) as f32;
                let hi = (group.end as f64 / nf) as f32;
                let mid = 0.5 * (lo + hi);
                let zm = midrank::normal_score(mid);
                for &(r, _) in &obs[group] {
                    u_lo[j][r] = lo;
                    u_hi[j][r] = hi;
                    u_mid[j][r] = mid;
                    z_mid[j][r] = zm;
                }
            }
        }
        Self {
            p,
            n,
            u_lo,
            u_hi,
            u_mid,
            z_mid,
        }
    }

    /// Distribute a tie-group interval `[lo, hi]` proportionally across `N_BINS` equal-width `[0,1]` bins.
    #[inline]
    pub fn bin_weights<const N_BINS: usize>(lo: f32, hi: f32) -> (usize, usize, [f64; N_BINS]) {
        let mut w = [0.0f64; N_BINS];
        let span = (hi - lo) as f64;
        if span <= 1e-9 {
            let b = ((lo as f64 * N_BINS as f64).floor() as isize).clamp(0, (N_BINS - 1) as isize)
                as usize;
            w[b] = 1.0;
            return (b, b, w);
        }
        let b0 =
            ((lo as f64 * N_BINS as f64).floor() as isize).clamp(0, (N_BINS - 1) as isize) as usize;
        let b1 = (((hi as f64 * N_BINS as f64).ceil() as isize) - 1)
            .clamp(b0 as isize, (N_BINS - 1) as isize) as usize;
        let inv_n = 1.0 / N_BINS as f64;
        let lo_f = lo as f64;
        let hi_f = hi as f64;
        for (b, slot) in w.iter_mut().enumerate().take(b1 + 1).skip(b0) {
            let bl = b as f64 * inv_n;
            let bh = (b + 1) as f64 * inv_n;
            let ov = (hi_f.min(bh) - lo_f.max(bl)).max(0.0);
            *slot = ov / span;
        }
        (b0, b1, w)
    }
}

/// Exact proper-scoring-rule loss on `rows` (MSE for reg, log-loss for binary/multiclass).
pub fn eval_true_loss(
    logits: &[Vec<f64>],
    y_sc: &[f64],
    rows: &[usize],
    task_id: u8,
    temp: f32,
) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    let t = (temp as f64).clamp(0.25, 4.0);
    let mut sum = 0.0;
    match task_id {
        0 => {
            for &r in rows {
                let d = y_sc[r] - logits[0][r];
                sum += d * d;
            }
        }
        1 => {
            for &r in rows {
                let z = (t * logits[0][r]).clamp(-12.0, 12.0);
                let yr = if y_sc[r] > 0.5 { 1.0 } else { 0.0 };
                // Numerically stable softplus(-z) + (1 - y)*z
                let log_loss = if z >= 0.0 {
                    (1.0 + (-z).exp()).ln() + (1.0 - yr) * z
                } else {
                    (1.0 + z.exp()).ln() - yr * z
                };
                sum += log_loss;
            }
        }
        _ => {
            let o_dim = logits.len();
            for &r in rows {
                let yi = (y_sc[r].round() as isize).clamp(0, (o_dim - 1) as isize) as usize;
                let mut max_l = f64::NEG_INFINITY;
                let mut ls = [0.0f64; V8_MAX_CLASSES];
                for c in 0..o_dim {
                    let v = (t * logits[c][r]).clamp(-12.0, 12.0);
                    ls[c] = v;
                    if v > max_l {
                        max_l = v;
                    }
                }
                let mut se = 0.0;
                for &v in ls.iter().take(o_dim) {
                    se += (v - max_l).exp();
                }
                sum += (max_l + se.max(1e-12).ln()) - ls[yi];
            }
        }
    }
    sum / rows.len() as f64
}

/// Zero-centered Newton working residuals on `rows` (eliminates minority-class clamping bias!).
pub fn centered_working_resid(
    logits: &[Vec<f64>],
    y_sc: &[f64],
    rows: &[usize],
    task_id: u8,
    temp: f32,
) -> Vec<Vec<f64>> {
    let o_dim = logits.len();
    let n = y_sc.len();
    let mut resid = vec![vec![0.0f64; n]; o_dim];
    if rows.is_empty() {
        return resid;
    }
    let t = (temp as f64).clamp(0.25, 4.0);
    match task_id {
        0 => {
            for &r in rows {
                resid[0][r] = y_sc[r] - logits[0][r];
            }
        }
        1 => {
            let mut sum = 0.0;
            for &r in rows {
                let pr = 1.0 / (1.0 + (-(t * logits[0][r]).clamp(-12.0, 12.0)).exp());
                let yr = if y_sc[r] > 0.5 { 1.0 } else { 0.0 };
                let g = ((yr - pr) / ((pr * (1.0 - pr)).max(0.15) * t)).clamp(-2.5, 2.5);
                resid[0][r] = g;
                sum += g;
            }
            let mean = sum / rows.len() as f64;
            for &r in rows {
                resid[0][r] -= mean;
            }
        }
        _ => {
            let mut sums = vec![0.0f64; o_dim];
            let mut ls = [0.0f64; V8_MAX_CLASSES];
            for &r in rows {
                let yi = (y_sc[r].round() as isize).clamp(0, (o_dim - 1) as isize) as usize;
                let mut max_l = f64::NEG_INFINITY;
                for c in 0..o_dim {
                    let v = (t * logits[c][r]).clamp(-12.0, 12.0);
                    ls[c] = v;
                    if v > max_l {
                        max_l = v;
                    }
                }
                let mut se = 0.0;
                for slot in ls.iter_mut().take(o_dim) {
                    *slot = (*slot - max_l).exp();
                    se += *slot;
                }
                let inv = se.max(1e-12).recip();
                for c in 0..o_dim {
                    let pc = ls[c] * inv;
                    let yc = if yi == c { 1.0 } else { 0.0 };
                    let g = ((yc - pc) / ((pc * (1.0 - pc)).max(0.15) * t)).clamp(-2.5, 2.5);
                    resid[c][r] = g;
                    sums[c] += g;
                }
            }
            let inv_n = 1.0 / rows.len() as f64;
            for c in 0..o_dim {
                let mean = sums[c] * inv_n;
                for &r in rows {
                    resid[c][r] -= mean;
                }
            }
        }
    }
    resid
}

/// Calibrate `pkt.linear.b` on deterministic synthetic rows decoded from `pkt`'s own X sections
/// so expected decoded class shares match FIT shares (and regression mean matches FIT mean).
/// Decode stays 100% row-local and wire length is unchanged!
pub fn calibrate_intercepts_on_packet(pkt: &mut Dse8Packet, y_sc: &[f64]) {
    let n_fit = y_sc.len();
    let o_dim = pkt.linear.o_dim;
    if n_fit < 4 || o_dim == 0 {
        return;
    }
    let Ok(plan) = build_decode_plan_v8(pkt, None) else {
        return;
    };
    let t = pkt.linear.temp.clamp(0.25, 4.0) as f64;
    let inv_t = if pkt.header.task_id != 0 {
        1.0 / t
    } else {
        1.0
    };

    // Collect pre-intercept scores s_i[c] = (logits_i[c] / temp) - b_old[c] on CAL_SYNTH_ROWS.
    let mut scores = vec![[0.0f64; V8_MAX_CLASSES]; CAL_SYNTH_ROWS];
    for (i, slot) in scores.iter_mut().enumerate() {
        let row = decode_dse8_row_at(&plan, CAL_SEED, i as u64);
        for c in 0..o_dim {
            slot[c] = (row.logits[c] as f64) * inv_t - pkt.linear.b[c] as f64;
        }
    }

    match pkt.header.task_id {
        0 => {
            let y_mean = y_sc.iter().sum::<f64>() / n_fit as f64;
            let s_mean = scores.iter().map(|s| s[0]).sum::<f64>() / CAL_SYNTH_ROWS as f64;
            let b_new = (y_mean - s_mean).clamp(-12.0, 12.0);
            if b_new.is_finite() {
                pkt.linear.b[0] = b_new as f32;
            }
        }
        1 => {
            let pi1 = (y_sc.iter().filter(|&&v| v > 0.5).count() as f64 / n_fit as f64)
                .clamp(0.005, 0.995);
            let mut b = pkt.linear.b[0] as f64;
            for _ in 0..15 {
                let mut p_sum = 0.0;
                let mut d_sum = 0.0;
                for s in &scores {
                    let pr = 1.0 / (1.0 + (-(t * (s[0] + b)).clamp(-12.0, 12.0)).exp());
                    p_sum += pr;
                    d_sum += pr * (1.0 - pr) * t;
                }
                let err = p_sum / CAL_SYNTH_ROWS as f64 - pi1;
                if err.abs() < 5e-4 {
                    break;
                }
                let step = (err / (d_sum / CAL_SYNTH_ROWS as f64).max(1e-4)).clamp(-2.0, 2.0);
                b = (b - step).clamp(-12.0, 12.0);
            }
            if b.is_finite() {
                pkt.linear.b[0] = b as f32;
            }
        }
        _ => {
            let mut cnts = vec![0.0f64; o_dim];
            for &v in y_sc {
                let c = (v.round() as isize).clamp(0, (o_dim - 1) as isize) as usize;
                cnts[c] += 1.0;
            }
            let tot = cnts.iter().sum::<f64>().max(1.0);
            let pi: Vec<f64> = cnts.iter().map(|&c| (c / tot).max(1e-4)).collect();
            let pi_s: f64 = pi.iter().sum();
            let pi: Vec<f64> = pi.iter().map(|&v| v / pi_s).collect();

            let mut b: Vec<f64> = pkt.linear.b.iter().map(|&v| v as f64).collect();
            let mut lgs = [0.0f64; V8_MAX_CLASSES];
            for _ in 0..24 {
                let mut p_sum = vec![0.0f64; o_dim];
                for s in &scores {
                    let mut max_l = f64::NEG_INFINITY;
                    for c in 0..o_dim {
                        let v = (t * (s[c] + b[c])).clamp(-12.0, 12.0);
                        lgs[c] = v;
                        if v > max_l {
                            max_l = v;
                        }
                    }
                    let mut se = 0.0;
                    for slot in lgs.iter_mut().take(o_dim) {
                        *slot = (*slot - max_l).exp();
                        se += *slot;
                    }
                    let inv = se.max(1e-12).recip();
                    for c in 0..o_dim {
                        p_sum[c] += lgs[c] * inv;
                    }
                }
                let mut max_err = 0.0f64;
                for c in 0..o_dim {
                    let p_bar = (p_sum[c] / CAL_SYNTH_ROWS as f64).max(1e-6);
                    max_err = max_err.max((p_bar - pi[c]).abs());
                    let delta = ((pi[c] / p_bar).ln() / t).clamp(-1.5, 1.5);
                    b[c] += 0.80 * delta;
                }
                let mb = b.iter().sum::<f64>() / o_dim as f64;
                for slot in &mut b {
                    *slot = (*slot - mb).clamp(-12.0, 12.0);
                }
                if max_err < 1e-3 {
                    break;
                }
            }
            for c in 0..o_dim {
                if b[c].is_finite() {
                    pkt.linear.b[c] = b[c] as f32;
                }
            }
        }
    }
}

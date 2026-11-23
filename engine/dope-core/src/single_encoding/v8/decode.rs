//! Row-local `DecodePlanV8`, single-row & bulk decoding, and 64-D JopeDime embedding export
//! for `DSE8` v1 (`dope_core::single_encoding::v8`).

use super::copula_regimes::V8_MAX_RANK;
use super::marginals::MARG_CONT;
use super::model::{NeuralPlanV8, SingleEncV8Weights, DEC_CODE_DIM, DEC_GLOB_DIM};
use super::model_ops::{apply_row_local_flow, build_decoder_inputs, eval_neural_target_residual};
use super::nominal::decode_nominal_group;
use super::packet::{pack_dse8_packet, Dse8Packet};
use super::rng_v8::{
    counter_normal, counter_student_t_scale, counter_uniform, sample_categorical,
    sample_softmax_categorical, STREAM_FACTOR, STREAM_NOISE, STREAM_REGIME, STREAM_TARGET,
};
use super::scale_v8::V8_MAX_CLASSES;
use super::target_v8::eval_effects_and_pairs;
use crate::error::{DopeError, Result};
use crate::single_encoding::KNOT_COUNT;

/// Precomputed row-local decoding state built once from a `Dse8Packet` + optional neural weights.
#[derive(Clone, Debug)]
pub struct DecodePlanV8 {
    pub pkt: Dse8Packet,
    pub l_eff: Vec<[f32; V8_MAX_RANK]>,
    pub noise_sd: Vec<f32>,
    pub knots_eff: Vec<[f32; KNOT_COUNT]>,
    pub w_eff: Vec<Vec<f32>>,
    pub resid_sd_eff: f32,
    pub neural: Option<NeuralPlanV8>,
}

/// One decoded synthetic row in BOTH normalized active space and full raw-unit space.
#[derive(Clone, Debug)]
pub struct DecodedV8Row {
    pub x_unit: Vec<f32>,
    pub missing_active: Vec<bool>,
    pub x_raw: Vec<f64>,
    pub logits: [f32; V8_MAX_CLASSES],
    pub y_unit: f32,
    pub y_raw: f64,
}

/// Column-major decoded synthetic table returned by `decode_dse8_rows`.
#[derive(Clone, Debug)]
pub struct DecodedV8Table {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub missing: Vec<Vec<bool>>,
    pub packet_bytes: usize,
}

/// Build a precomputed row-local `DecodePlanV8` from an unpacked `Dse8Packet` + optional weights.
pub fn build_decode_plan_v8(
    packet: &Dse8Packet,
    weights: Option<&SingleEncV8Weights>,
) -> Result<DecodePlanV8> {
    if weights.is_some_and(|w| {
        (w.active_rank_mask || packet.weights_fp.is_some())
            && packet.weights_fp != Some(w.fingerprint())
    }) {
        return Err(DopeError::Artifact(
            "dse8_weights_fingerprint_mismatch".into(),
        ));
    }
    let p = packet.header.p_eff as usize;
    let o_dim = packet.linear.o_dim;
    let neural = weights.map(|w| w.build_neural_plan(packet));
    let rank = weights.map_or(V8_MAX_RANK, |w| w.loading_rank(packet));

    let mut l_eff = packet.loadings.clone();
    let mut noise_sd = vec![1.0f32; p];
    let mut knots_eff: Vec<[f32; KNOT_COUNT]> =
        packet.marginals.iter().map(|m| m.knots17).collect();
    let mut w_eff = packet.linear.w.clone();
    let mut resid_sd_eff = packet.header.resid_sd;

    for loading in &mut l_eff {
        loading[rank..].fill(0.0);
    }
    if let Some(ref np) = neural {
        if !np.is_identity {
            resid_sd_eff = (resid_sd_eff * np.resid_sd_mult).clamp(0.002, 6.0);
            for j in 0..p {
                apply_loading_delta(&mut l_eff[j], &np.delta_l[j], rank);
                if packet.marginals[j].kind == MARG_CONT {
                    let base = &packet.marginals[j].knots17;
                    let mut w_raw = [0.0f32; KNOT_COUNT - 1];
                    let mut span_b = 0.0f32;
                    let mut w_sum = 0.0f32;
                    for k in 0..(KNOT_COUNT - 1) {
                        let bd = (base[k + 1] - base[k]).max(0.0);
                        let wr = bd * (0.25 * np.delta_knots[j][k + 1].tanh()).exp();
                        w_raw[k] = wr;
                        span_b += bd;
                        w_sum += wr;
                    }
                    knots_eff[j][0] = base[0];
                    let mut acc = base[0];
                    for k in 0..(KNOT_COUNT - 1) {
                        let step = if span_b > 1e-7 {
                            w_raw[k] * (span_b / w_sum.max(1e-7))
                        } else {
                            0.0
                        };
                        acc += step;
                        knots_eff[j][k + 1] = acc;
                    }
                }
                for c in 0..o_dim {
                    w_eff[c][j] += np.delta_w[c][j];
                }
            }
        }
    }

    for j in 0..p {
        let sq: f32 = l_eff[j].iter().map(|v| v * v).sum();
        noise_sd[j] = (1.0 - sq).max(1e-4).sqrt();
    }

    Ok(DecodePlanV8 {
        pkt: packet.clone(),
        l_eff,
        noise_sd,
        knots_eff,
        w_eff,
        resid_sd_eff,
        neural,
    })
}

pub(super) fn apply_loading_delta(
    loading: &mut [f32; V8_MAX_RANK],
    delta: &[f32; V8_MAX_RANK],
    rank: usize,
) {
    let (active, inactive) = loading.split_at_mut(rank);
    inactive.fill(0.0);
    let mut square_sum = 0.0f32;
    for (value, correction) in active.iter_mut().zip(delta) {
        *value += correction;
        square_sum += *value * *value;
    }
    let scale = (square_sum.sqrt() / 0.999).max(1.0);
    for value in active {
        *value /= scale;
    }
}

/// Strictly row-local decode of synthetic row `row_idx` from `(plan, seed, row_idx)`.
pub fn decode_dse8_row_at(plan: &DecodePlanV8, seed: u64, row_idx: u64) -> DecodedV8Row {
    decode_row(plan, seed, row_idx, true, true, |_, _| {})
}

pub(super) fn decode_row(
    plan: &DecodePlanV8,
    seed: u64,
    row_idx: u64,
    flow: bool,
    head: bool,
    observe_latents: impl FnOnce([f32; V8_MAX_RANK], Vec<f32>),
) -> DecodedV8Row {
    let pkt = &plan.pkt;
    let p = pkt.header.p_eff as usize;
    let r_use = pkt.header.rank_use as usize;
    let o_dim = pkt.linear.o_dim;

    let mut z_factor = [0.0f32; V8_MAX_RANK];
    let t_scale = counter_student_t_scale(seed, row_idx, pkt.header.inv_nu);
    for r in 0..r_use {
        z_factor[r] = counter_normal(seed, row_idx, STREAM_FACTOR, r as u64) as f32 * t_scale;
    }
    if pkt.regimes.regimes.len() >= 2 {
        let weights: Vec<f64> = pkt
            .regimes
            .regimes
            .iter()
            .map(|rg| rg.weight as f64)
            .collect();
        let u_rg = counter_uniform(seed, row_idx, STREAM_REGIME, 0);
        let chosen = sample_categorical(&weights, u_rg);
        let reg = &pkt.regimes.regimes[chosen];
        for (r, &mu) in reg.mean_f.iter().enumerate().take(r_use) {
            let shr = (1.0 - (mu * mu).min(0.75)).max(0.25).sqrt();
            z_factor[r] = mu + shr * z_factor[r];
        }
    }

    let mut z_lat = vec![0.0f32; p];
    for j in 0..p {
        let eps = counter_normal(seed, row_idx, STREAM_NOISE, j as u64) as f32 * t_scale;
        let mut z = eps * plan.noise_sd[j];
        for r in 0..r_use {
            z += z_factor[r] * plan.l_eff[j][r];
        }
        z_lat[j] = z;
    }
    if let Some(np) = plan.neural.as_ref().filter(|_| flow) {
        apply_row_local_flow(np, &mut z_lat);
    }

    let mut x_unit = vec![0.0f32; p];
    let mut missing_active = vec![false; p];
    for j in 0..p {
        let (val, is_miss) = pkt.marginals[j].decode_value(
            z_lat[j],
            &plan.knots_eff[j],
            pkt.header.z_max,
            seed,
            row_idx,
            j as u64,
        );
        x_unit[j] = val;
        missing_active[j] = is_miss;
    }

    for (gi, grp) in pkt.nominal_groups.iter().enumerate() {
        decode_nominal_group(
            grp,
            &z_lat,
            &mut x_unit,
            &mut missing_active,
            seed,
            row_idx,
            gi as u64,
        );
    }

    let mut logits = [0.0f32; V8_MAX_CLASSES];
    for c in 0..o_dim {
        let mut s = pkt.linear.b[c];
        for j in 0..p {
            if !missing_active[j] {
                s += x_unit[j] * plan.w_eff[c][j];
            }
        }
        logits[c] = s;
    }
    eval_effects_and_pairs(&pkt.effects, &pkt.pairs, &z_lat, &mut logits[..o_dim]);
    pkt.trees.eval_row(&z_lat, &mut logits[..o_dim]);
    if let Some(np) = plan.neural.as_ref().filter(|_| head) {
        let x_clean: Vec<f32> = x_unit
            .iter()
            .zip(&missing_active)
            .map(|(&v, &m)| if m { 0.0 } else { v })
            .collect();
        eval_neural_target_residual(np, &x_clean, o_dim, &mut logits[..o_dim]);
    }
    if pkt.header.task_id != 0 {
        for slot in logits.iter_mut().take(o_dim) {
            *slot *= pkt.linear.temp;
        }
    }

    let (y_unit, y_raw) = match pkt.header.task_id {
        0 => {
            let mut eps_y = counter_normal(seed, row_idx, STREAM_TARGET, 0) as f32;
            let rs = &pkt.resid_model;
            if rs.q5_resid.iter().any(|v| v.abs() > 1e-4) {
                let u = crate::stats::norm_cdf(eps_y as f64) as f32;
                let pos = ((u - 0.10) / 0.20).clamp(0.0, 4.0);
                let i0 = (pos.floor() as usize).min(3);
                let f = pos - i0 as f32;
                eps_y += rs.q5_resid[i0] + f * (rs.q5_resid[i0 + 1] - rs.q5_resid[i0]);
            }
            let mut log_mult = 0.0f32;
            for (&col_u8, &w) in rs.cols.iter().zip(&rs.log_sd_w) {
                let j = col_u8 as usize;
                if j < p {
                    log_mult += w * z_lat[j].clamp(-3.0, 3.0);
                }
            }
            let sd_row = plan.resid_sd_eff * log_mult.clamp(-1.0, 1.0).exp();
            let yu = (logits[0] + sd_row * eps_y).clamp(-8.0, 8.0);
            let yr = if pkt.card.normalized_only {
                yu as f64
            } else {
                pkt.card.y_mean as f64 + (pkt.card.y_sd as f64) * yu as f64
            };
            (yu, yr)
        }
        1 => {
            let p1 = 1.0 / (1.0 + (-(logits[0].clamp(-12.0, 12.0)) as f64).exp());
            let u = counter_uniform(seed, row_idx, STREAM_TARGET, 0);
            let idx = usize::from(u < p1);
            let cls_val = pkt.card.classes.get(idx).copied().unwrap_or(idx as i32) as f64;
            (idx as f32, cls_val)
        }
        _ => {
            let u = counter_uniform(seed, row_idx, STREAM_TARGET, 0);
            let pick = sample_softmax_categorical(&logits[..o_dim], u);
            let cls_val = pkt.card.classes.get(pick).copied().unwrap_or(pick as i32) as f64;
            (pick as f32, cls_val)
        }
    };

    let x_raw = pkt.card.inverse_row_x(&x_unit, &missing_active);
    observe_latents(z_factor, z_lat);
    DecodedV8Row {
        x_unit,
        missing_active,
        x_raw,
        logits,
        y_unit,
        y_raw,
    }
}

/// Bulk row-local decode into column-major `DecodedV8Table`.
pub fn decode_dse8_rows(
    packet: &Dse8Packet,
    weights: Option<&SingleEncV8Weights>,
    n_rows: usize,
    seed: u64,
) -> Result<DecodedV8Table> {
    let plan = build_decode_plan_v8(packet, weights)?;
    let x_cols = packet.card.x_cols_raw;
    let mut x = vec![vec![0.0f64; n_rows]; x_cols];
    let mut missing = vec![vec![false; n_rows]; x_cols];
    let mut y = vec![0.0f64; n_rows];
    for r in 0..n_rows {
        let row = decode_dse8_row_at(&plan, seed, r as u64);
        for c in 0..x_cols {
            let v = row.x_raw[c];
            x[c][r] = v;
            missing[c][r] = !v.is_finite();
        }
        y[r] = row.y_raw;
    }
    Ok(DecodedV8Table {
        x,
        y,
        missing,
        packet_bytes: pack_dse8_packet(packet, 65536)
            .map(|b| b.len())
            .unwrap_or(0),
    })
}

/// Extract a deterministic 64-D JopeDime dataset embedding directly from a `Dse8Packet`.
pub fn extract_jopedime_embedding(packet: &Dse8Packet) -> Vec<f32> {
    if packet.code.tier > 0 && !packet.code.packed.is_empty() {
        return packet.code.dequantize(DEC_CODE_DIM);
    }
    let (_, _, glob, _) = build_decoder_inputs(packet);
    let mut emb = vec![0.0f32; DEC_CODE_DIM];
    emb[..DEC_GLOB_DIM].copy_from_slice(&glob);
    let p = packet.header.p_eff as usize;
    if p > 0 {
        for r in 0..V8_MAX_RANK {
            let rms = (packet.loadings.iter().map(|l| l[r] * l[r]).sum::<f32>() / p as f32).sqrt();
            emb[32 + r] = rms;
        }
        for c in 0..packet.linear.o_dim.min(8) {
            let rms = (packet.linear.w[c].iter().map(|w| w * w).sum::<f32>() / p as f32).sqrt();
            emb[44 + c] = rms.tanh();
        }
        let disc_share = packet
            .marginals
            .iter()
            .filter(|m| m.kind != MARG_CONT)
            .count() as f32
            / p as f32;
        let miss_mean = packet.marginals.iter().map(|m| m.miss_rate).sum::<f32>() / p as f32;
        emb[52] = disc_share;
        emb[53] = miss_mean;
    }
    emb
}

//! Pure-Rust self-contained `DSE7` v2 binary packet encoder/decoder (`<= 2 KiB`, CRC32-verified).
//!
//! Embeds the compact raw-unit scale card (`ScaleCardV7`) directly inside the wire packet so
//! `m.bin.len() <= budget_bytes` covers 100% of everything required to synthesize raw-unit rows
//! (`decode(unpack(pack(encode(A))))`, strictly `source_free`, zero separate sidecar).

use super::math::{
    count_unique_sorted, f16_bits_to_f32, f32_to_f16_bits, fit_factor_loadings_i8,
    fit_target_model_i8, norm_cdf_f64, normal_scores_conditional, percentile_sorted,
    quant_f16_roundtrip,
};
use super::scale_card::{
    bf16_bits_to_f32, choose_dse7_layout, dse7_wire_len, f32_to_bf16_bits, fit_q_nl_i8, Dse7Layout,
    ScaleCardV7, DSE7_HEADER_BYTES,
};
use super::{
    SingleEncV7Packet, COL_FEAT_DIM, GLOB_FEAT_DIM, KNOT_COUNT, MAX_CLASSES, MAX_COLS, MAX_RANK,
};
use crate::artifact::format2::crc32_ieee;
use crate::error::{DopeError, Result};
use crate::stats::norm_ppf;

pub const DSE7_MAGIC: &[u8; 4] = b"DSE7";
pub const DSE7_VERSION: u8 = 2;

const FLAG_HAS_Q_NL: u8 = 0x01;
const FLAG_HAS_COL_MAP: u8 = 0x02;
const FLAG_HAS_CUSTOM_CLASSES: u8 = 0x04;
const FLAG_HAS_DROPPED_MEDS: u8 = 0x08;

#[derive(Clone, Debug)]
pub struct Dse7Header {
    pub task_id: u8,
    pub k_eff: u8,
    pub p_eff: u8,
    pub x_cols_raw: u16,
    pub rank_use: u8,
    pub knot_level: u8,
    pub has_q_nl: bool,
    pub z_max: f32,
    pub resid_sd: f32,
}

/// Fit and pack a strict `<= budget_bytes` self-contained CRC32-verified `DSE7` v2 wire packet.
pub fn fit_dse7_packet(
    x_raw: &[Vec<f64>],
    y_raw: &[f64],
    task_id: u8,
    budget_bytes: usize,
) -> Result<Vec<u8>> {
    let x_cols_raw = x_raw.len();
    if x_cols_raw == 0 || x_cols_raw > u16::MAX as usize {
        return Err(DopeError::Input("dse7 invalid column count".into()));
    }
    // Initial card probe at MAX_COLS to determine exact class list & custom-class flag.
    let probe = ScaleCardV7::fit(x_raw, y_raw, task_id, MAX_COLS)?;
    let k_eff: usize = match task_id {
        0 => 1,
        1 => 2,
        _ => probe.classes.len(),
    };
    let has_custom_classes = task_id != 0
        && probe
            .classes
            .iter()
            .enumerate()
            .any(|(i, &cid)| cid != i as i32);
    let lay = choose_dse7_layout(x_cols_raw, task_id, k_eff, has_custom_classes, budget_bytes)
        .ok_or_else(|| DopeError::Budget("over_budget".into()))?;
    let card = if lay.p_eff == probe.col_indices.len() {
        probe
    } else {
        ScaleCardV7::fit(x_raw, y_raw, task_id, lay.p_eff)?
    };
    let x_sc = card.forward_x(x_raw);
    let p_eff = x_sc.len();
    let n_fit = y_raw.len();
    let y_sc: Vec<f64> = match task_id {
        0 => y_raw
            .iter()
            .map(|&v| ((v - card.y_mean as f64) / card.y_sd as f64).clamp(-6.0, 6.0))
            .collect(),
        1 => {
            let hi_cls = card.classes.get(1).copied().unwrap_or(1);
            y_raw
                .iter()
                .map(|&v| {
                    if (v.round() as i32) == hi_cls {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect()
        }
        _ => y_raw
            .iter()
            .map(|&v| {
                let iv = v.round() as i32;
                card.classes.binary_search(&iv).unwrap_or(0) as f64
            })
            .collect(),
    };
    let o_dim = if task_id == 2 { k_eff } else { 1 };

    let u_floor = (10.0 / (n_fit as f64 + 1.0).max(20.0)).min(0.20);
    let z_max = norm_ppf(1.0 - u_floor).clamp(1.2, 2.6) as f32;
    let mut u_grid = [0.0f64; KNOT_COUNT];
    for (i, slot) in u_grid.iter_mut().enumerate() {
        let z = -(z_max as f64) + (2.0 * z_max as f64) * (i as f64) / (KNOT_COUNT - 1) as f64;
        *slot = norm_cdf_f64(z);
    }

    let mut col_flags = vec![0u8; p_eff];
    let mut lo_vals = vec![0.0f32; p_eff];
    let mut span_vals = vec![0.0f32; p_eff];
    let mut rel_q = vec![vec![0u8; lay.knot_level]; p_eff];
    let mut z_mat = vec![vec![0.0f64; n_fit]; p_eff];

    for j in 0..p_eff {
        let mut sorted: Vec<f64> = x_sc[j].iter().map(|&v| v as f64).collect();
        sorted.sort_by(f64::total_cmp);
        let lo = percentile_sorted(&sorted, u_floor);
        let hi = percentile_sorted(&sorted, 1.0 - u_floor);
        let is_const = (hi - lo) < 1e-6;
        let uniq = count_unique_sorted(&sorted);
        let is_disc = !is_const && uniq.len() <= 12;
        let is_int = card.is_int_active[j];
        col_flags[j] = u8::from(is_const) | (u8::from(is_disc) << 1) | (u8::from(is_int) << 2);
        let lo_f = quant_f16_roundtrip(lo as f32);
        let span_f = quant_f16_roundtrip((hi - lo).max(1e-5) as f32).max(1e-5);
        lo_vals[j] = lo_f;
        span_vals[j] = span_f;

        let mut knots17 = [0.0f64; KNOT_COUNT];
        for (ki, &u) in u_grid.iter().enumerate() {
            knots17[ki] = percentile_sorted(&sorted, u);
        }
        if is_disc {
            let kept: Vec<f64> = uniq
                .iter()
                .filter(|&&(_, cnt)| cnt >= 10)
                .map(|&(v, _)| v)
                .collect();
            if kept.len() >= 2 {
                for kv in &mut knots17 {
                    *kv = kept
                        .iter()
                        .copied()
                        .min_by(|a, b| (a - *kv).abs().total_cmp(&(b - *kv).abs()))
                        .unwrap_or(*kv);
                }
            }
        }
        let step = (KNOT_COUNT - 1) / (lay.knot_level - 1);
        for ki in 0..lay.knot_level {
            let rel = ((knots17[ki * step] - lo_f as f64) / span_f as f64).clamp(0.0, 1.0);
            rel_q[j][ki] = (rel * 255.0).round().clamp(0.0, 255.0) as u8;
        }
        if !is_const {
            z_mat[j] = normal_scores_conditional(&x_sc[j], u_floor);
        }
    }

    let l_i8 = fit_factor_loadings_i8(&z_mat, lay.rank_use);
    let (w_scales, b_vals, w_i8, resid_sd) = fit_target_model_i8(&x_sc, &y_sc, task_id, k_eff);
    let q_nl_i8 = if lay.has_q_nl {
        fit_q_nl_i8(
            &x_sc, &y_sc, &z_mat, &w_scales, &b_vals, &w_i8, resid_sd, task_id, k_eff,
        )
    } else {
        Vec::new()
    };

    let mut pkt_flags = 0u8;
    if lay.has_q_nl {
        pkt_flags |= FLAG_HAS_Q_NL;
    }
    if lay.has_col_map {
        pkt_flags |= FLAG_HAS_COL_MAP;
    }
    if lay.has_custom_classes {
        pkt_flags |= FLAG_HAS_CUSTOM_CLASSES;
    }
    if lay.has_dropped_meds {
        pkt_flags |= FLAG_HAS_DROPPED_MEDS;
    }

    let log2_nfit_u8 = ((n_fit.max(2) as f32).log2() / 12.0 * 255.0)
        .round()
        .clamp(0.0, 255.0) as u8;
    let p1_u8: u8 = if task_id == 1 {
        let p1 = (y_sc.iter().sum::<f64>() / n_fit.max(1) as f64).clamp(0.02, 0.98);
        (p1 * 255.0).round().clamp(1.0, 254.0) as u8
    } else {
        0
    };
    let mut buf = Vec::with_capacity(budget_bytes);
    buf.extend_from_slice(DSE7_MAGIC);
    buf.push(DSE7_VERSION);
    buf.push(task_id);
    buf.push(k_eff as u8);
    buf.push(p_eff as u8);
    buf.push(lay.rank_use as u8);
    buf.push(lay.knot_level as u8);
    buf.push(pkt_flags);
    buf.push(log2_nfit_u8);
    buf.extend_from_slice(&(x_cols_raw as u16).to_le_bytes());
    buf.extend_from_slice(&((z_max * 10000.0).round().clamp(1.0, 65000.0) as u16).to_le_bytes());
    buf.extend_from_slice(&((resid_sd * 10000.0).round().clamp(1.0, 65000.0) as u16).to_le_bytes());
    buf.push(p1_u8);
    buf.push(0u8);

    if task_id == 0 {
        buf.extend_from_slice(&card.y_mean.to_le_bytes());
        buf.extend_from_slice(&card.y_sd.to_le_bytes());
    } else if lay.has_custom_classes {
        for &cid in &card.classes {
            buf.extend_from_slice(&cid.to_le_bytes());
        }
    }
    if lay.has_col_map {
        for &idx in &card.col_indices {
            buf.extend_from_slice(&(idx as u16).to_le_bytes());
        }
    }
    for j in 0..p_eff {
        buf.extend_from_slice(&card.center_active[j].to_le_bytes());
        buf.extend_from_slice(&f32_to_bf16_bits(card.scale_active[j]).to_le_bytes());
    }
    if lay.has_dropped_meds {
        for &m in &card.dropped_medians {
            buf.extend_from_slice(&f32_to_bf16_bits(m).to_le_bytes());
        }
    }
    buf.extend_from_slice(&col_flags);
    for j in 0..p_eff {
        buf.extend_from_slice(&f32_to_f16_bits(lo_vals[j]).to_le_bytes());
        buf.extend_from_slice(&f32_to_f16_bits(span_vals[j]).to_le_bytes());
        buf.extend_from_slice(&rel_q[j]);
    }
    for row in &l_i8 {
        for &v in row {
            buf.push(v as u8);
        }
    }
    for c in 0..o_dim {
        buf.extend_from_slice(&w_scales[c].to_le_bytes());
        buf.extend_from_slice(&b_vals[c].to_le_bytes());
        for &v in &w_i8[c] {
            buf.push(v as u8);
        }
    }
    for q4 in &q_nl_i8 {
        for &v in q4 {
            buf.push(v as u8);
        }
    }
    let crc = crc32_ieee(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    if buf.len() > budget_bytes {
        return Err(DopeError::Budget("over_budget".into()));
    }
    Ok(buf)
}

/// Fail-closed unpacker: reconstructs `SingleEncV7Packet`, header, and `ScaleCardV7` strictly from wire bytes.
pub fn unpack_dse7_packet(bytes: &[u8]) -> Result<(SingleEncV7Packet, Dse7Header, ScaleCardV7)> {
    if bytes.len() < DSE7_HEADER_BYTES + 4 {
        return Err(DopeError::Artifact("dse7 packet too short".into()));
    }
    if &bytes[..4] != DSE7_MAGIC || bytes[4] != DSE7_VERSION {
        return Err(DopeError::Artifact("dse7 invalid magic or version".into()));
    }
    let body_len = bytes.len() - 4;
    let exp_crc = u32::from_le_bytes(bytes[body_len..].try_into().unwrap());
    if crc32_ieee(&bytes[..body_len]) != exp_crc {
        return Err(DopeError::Artifact("dse7 crc32 mismatch".into()));
    }
    let task_id = bytes[5];
    let k_eff = bytes[6] as usize;
    let p_eff = bytes[7] as usize;
    let rank_use = bytes[8] as usize;
    let knot_level = bytes[9] as usize;
    let pkt_flags = bytes[10];
    let log2_nfit_norm = bytes[11] as f32 / 255.0;
    let x_cols_raw = u16::from_le_bytes([bytes[12], bytes[13]]) as usize;
    let z_max = u16::from_le_bytes([bytes[14], bytes[15]]) as f32 / 10000.0;
    let resid_sd = u16::from_le_bytes([bytes[16], bytes[17]]) as f32 / 10000.0;
    let p1_prior = bytes[18] as f32 / 255.0;
    if task_id > 2
        || p_eff == 0
        || p_eff > MAX_COLS
        || x_cols_raw < p_eff
        || rank_use > MAX_RANK
        || !matches!(knot_level, 5 | 9 | 17)
        || k_eff == 0
        || k_eff > MAX_CLASSES
    {
        return Err(DopeError::Artifact("dse7 header bounds invalid".into()));
    }
    let lay = Dse7Layout {
        p_eff,
        knot_level,
        rank_use,
        has_q_nl: (pkt_flags & FLAG_HAS_Q_NL) != 0,
        has_col_map: (pkt_flags & FLAG_HAS_COL_MAP) != 0,
        has_custom_classes: (pkt_flags & FLAG_HAS_CUSTOM_CLASSES) != 0,
        has_dropped_meds: (pkt_flags & FLAG_HAS_DROPPED_MEDS) != 0,
    };
    if bytes.len() != dse7_wire_len(x_cols_raw, task_id, k_eff, lay) {
        return Err(DopeError::Artifact("dse7 wire length mismatch".into()));
    }
    let mut cur = DSE7_HEADER_BYTES;
    let (y_mean, y_sd, classes) = if task_id == 0 {
        let ym = f32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap());
        let ys = f32::from_le_bytes(bytes[cur + 4..cur + 8].try_into().unwrap());
        cur += 8;
        (ym, ys, Vec::new())
    } else if lay.has_custom_classes {
        let mut cls = Vec::with_capacity(k_eff);
        for _ in 0..k_eff {
            cls.push(i32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap()));
            cur += 4;
        }
        (0.0, 1.0, cls)
    } else {
        (0.0, 1.0, (0..k_eff as i32).collect())
    };
    let col_indices: Vec<usize> = if lay.has_col_map {
        let mut idx = Vec::with_capacity(p_eff);
        for _ in 0..p_eff {
            let v = u16::from_le_bytes([bytes[cur], bytes[cur + 1]]) as usize;
            cur += 2;
            if v >= x_cols_raw {
                return Err(DopeError::Artifact("dse7 col index out of bounds".into()));
            }
            idx.push(v);
        }
        idx
    } else {
        (0..p_eff).collect()
    };
    let mut center_active = Vec::with_capacity(p_eff);
    let mut scale_active = Vec::with_capacity(p_eff);
    for _ in 0..p_eff {
        center_active.push(f32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap()));
        scale_active.push(bf16_bits_to_f32(u16::from_le_bytes([
            bytes[cur + 4],
            bytes[cur + 5],
        ])));
        cur += 6;
    }
    let mut dropped_medians = Vec::new();
    if lay.has_dropped_meds {
        let n_drop = x_cols_raw - p_eff;
        dropped_medians.reserve(n_drop);
        for _ in 0..n_drop {
            dropped_medians.push(bf16_bits_to_f32(u16::from_le_bytes([
                bytes[cur],
                bytes[cur + 1],
            ])));
            cur += 2;
        }
    }
    let col_flags = &bytes[cur..cur + p_eff];
    cur += p_eff;

    let mut pkt = SingleEncV7Packet {
        col_feats: [[0.0; COL_FEAT_DIM]; MAX_COLS],
        q_nl: [[0.0; 4]; MAX_COLS],
        glob_feats: [0.0; GLOB_FEAT_DIM],
        knots: [[0.0; KNOT_COUNT]; MAX_COLS],
        loadings: [[0.0; MAX_RANK]; MAX_COLS],
        w_base: [[0.0; MAX_COLS]; MAX_CLASSES],
        b_base: [0.0; MAX_CLASSES],
        resid_sd,
        col_mask: [0.0; MAX_COLS],
        is_const: [0.0; MAX_COLS],
    };
    let k_glob = if task_id == 2 { k_eff } else { 1 };
    pkt.glob_feats[task_id.min(3) as usize] = 1.0;
    if task_id == 1 {
        pkt.glob_feats[4] = (1.0 - p1_prior).clamp(0.0, 1.0);
        pkt.glob_feats[5] = p1_prior.clamp(0.0, 1.0);
    }
    pkt.glob_feats[4 + MAX_CLASSES] = log2_nfit_norm;
    pkt.glob_feats[4 + MAX_CLASSES + 1] = (p_eff as f32 / MAX_COLS as f32).clamp(0.0, 1.0);
    pkt.glob_feats[4 + MAX_CLASSES + 2] = resid_sd;
    pkt.glob_feats[4 + MAX_CLASSES + 3] = (k_glob as f32 / MAX_CLASSES as f32).clamp(0.0, 1.0);

    for j in 0..p_eff {
        let lo = f16_bits_to_f32(u16::from_le_bytes([bytes[cur], bytes[cur + 1]]));
        let span = f16_bits_to_f32(u16::from_le_bytes([bytes[cur + 2], bytes[cur + 3]])).max(1e-5);
        cur += 4;
        let q_sl = &bytes[cur..cur + knot_level];
        cur += knot_level;
        let mut knots17 = [0.0f32; KNOT_COUNT];
        let step = (KNOT_COUNT - 1) / (knot_level - 1);
        for ki in 0..KNOT_COUNT {
            let seg = (ki / step).min(knot_level - 2);
            let frac = (ki - seg * step) as f32 / step as f32;
            let r0 = q_sl[seg] as f32 / 255.0;
            let r1 = q_sl[seg + 1] as f32 / 255.0;
            knots17[ki] = lo + span * (r0 + frac * (r1 - r0));
        }
        for ki in 1..KNOT_COUNT {
            if knots17[ki] < knots17[ki - 1] {
                knots17[ki] = knots17[ki - 1];
            }
        }
        pkt.knots[j] = knots17;
        pkt.col_mask[j] = 1.0;
        let is_const = if (col_flags[j] & 1) != 0 { 1.0 } else { 0.0 };
        let is_disc = if (col_flags[j] & 2) != 0 { 1.0 } else { 0.0 };
        pkt.is_const[j] = is_const;
        pkt.col_feats[j][..KNOT_COUNT].copy_from_slice(&knots17);
        pkt.col_feats[j][COL_FEAT_DIM - 5] = is_const;
        pkt.col_feats[j][COL_FEAT_DIM - 4] = is_disc;
        pkt.col_feats[j][COL_FEAT_DIM - 3] = (1.0 + span).ln();
        pkt.col_feats[j][COL_FEAT_DIM - 2] = z_max;
        pkt.col_feats[j][COL_FEAT_DIM - 1] = knot_level as f32 / KNOT_COUNT as f32;
    }
    for j in 0..p_eff {
        for r in 0..rank_use {
            let v = (bytes[cur] as i8) as f32 / 127.0;
            cur += 1;
            pkt.loadings[j][r] = v;
            pkt.col_feats[j][KNOT_COUNT + r] = v;
        }
    }
    let o_dim = if task_id == 2 { k_eff } else { 1 };
    for c in 0..o_dim {
        let w_scale = f32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap());
        let b_val = f32::from_le_bytes(bytes[cur + 4..cur + 8].try_into().unwrap());
        cur += 8;
        pkt.b_base[c] = b_val;
        for j in 0..p_eff {
            let wv = (bytes[cur] as i8) as f32 * w_scale;
            cur += 1;
            pkt.w_base[c][j] = wv;
            pkt.col_feats[j][KNOT_COUNT + MAX_RANK + c] = wv;
        }
    }
    if task_id == 2 {
        let mx = pkt.b_base[..k_eff]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0f32;
        for c in 0..k_eff {
            let e = (pkt.b_base[c] - mx).clamp(-20.0, 0.0).exp();
            pkt.glob_feats[4 + c] = e;
            s += e;
        }
        if s > 0.0 {
            for c in 0..k_eff {
                pkt.glob_feats[4 + c] /= s;
            }
        }
    }
    if lay.has_q_nl {
        for j in 0..p_eff {
            for b in 0..4 {
                pkt.q_nl[j][b] = (bytes[cur] as i8) as f32 * 0.015;
                cur += 1;
            }
        }
    }
    let is_int_active: Vec<bool> = (0..p_eff).map(|j| (col_flags[j] & 4) != 0).collect();
    let card = ScaleCardV7 {
        x_cols_raw,
        col_indices,
        center_active,
        scale_active,
        is_int_active,
        dropped_medians,
        task_id,
        y_mean,
        y_sd,
        classes,
    };
    let hdr = Dse7Header {
        task_id,
        k_eff: k_eff as u8,
        p_eff: p_eff as u8,
        x_cols_raw: x_cols_raw as u16,
        rank_use: rank_use as u8,
        knot_level: knot_level as u8,
        has_q_nl: lay.has_q_nl,
        z_max,
        resid_sd,
    };
    Ok((pkt, hdr, card))
}

//! Nested-prefix CRC32-verified binary wire codec for `DSE8` v1 (`512 B .. 65536 B`).
//!
//! Wire layout:
//! - 24-byte fixed header (`magic = "DSE8"`, `version = 1`, task/dims/flags/rank/knots/scales)
//! - Ordered TLV sections `(tag: u8, len: u16 LE, payload[len])`:
//!   - Mandatory base: `0x01` Scale (omitted if `normalized_only`), `0x02` Marginals,
//!     `0x03` Copula, `0x04` Linear
//!   - Optional nested-prefix enhancements: `0x05` Nominal groups, `0x06` Regimes,
//!     `0x07` Effects (S8), `0x08` Pairs (S10/XOR), `0x09` Trees (prefix-truncatable),
//!     `0x0A` Resid (T03), `0x0B` Code (nested FSQ Tier 1/2/3, prefix-truncatable)
//! - Trailing 4-byte `crc32_ieee` covering every preceding byte.

use super::copula_regimes::{RegimesSectionV8, V8_MAX_RANK};
use super::marginals::MarginalColV8;
use super::nominal::NominalGroupV8;
use super::planner::{DSE8_CRC_BYTES, DSE8_HEADER_BYTES, TLV_HDR_BYTES};
use super::scale_v8::ScaleCardV8;
use super::target_v8::{CodeSectionV8, Effect1DV8, LinearSectionV8, PairIntV8, ResidSectionV8};
use super::trees_v8::{TreesSectionV8, V8_TREE_WIRE_BYTES};
use crate::artifact::format2::crc32_ieee;
use crate::error::{DopeError, Result};
use crate::single_encoding::math::f32_to_f16_bits;
use crate::single_encoding::scale_card::f32_to_bf16_bits;

pub use super::packet_decode::unpack_dse8_packet;

pub const DSE8_MAGIC: &[u8; 4] = b"DSE8";
pub const DSE8_VERSION: u8 = 1;

pub const FLAG_NORMALIZED_ONLY: u8 = 0x01;
pub const FLAG_HAS_COL_MAP: u8 = 0x02;
pub const FLAG_HAS_CUSTOM_CLASSES: u8 = 0x04;
pub const FLAG_HAS_DROPPED_MEDS: u8 = 0x08;
pub const FLAG_HAS_SKEW_T: u8 = 0x10;
pub const FLAG_HAS_NEURAL_ENCODER: u8 = 0x20;
/// Marks packets fitted under `dope_dse8_decoder = "active-rank-v1"` (rejected by `v3.0.0` unpackers).
pub const FLAG_ACTIVE_RANK_V1: u8 = 0x40;
pub const ALL_KNOWN_FLAGS: u8 = FLAG_NORMALIZED_ONLY
    | FLAG_HAS_COL_MAP
    | FLAG_HAS_CUSTOM_CLASSES
    | FLAG_HAS_DROPPED_MEDS
    | FLAG_HAS_SKEW_T
    | FLAG_HAS_NEURAL_ENCODER
    | FLAG_ACTIVE_RANK_V1;

pub const TAG_SCALE: u8 = 0x01;
pub const TAG_MARG: u8 = 0x02;
pub const TAG_COPULA: u8 = 0x03;
pub const TAG_LINEAR: u8 = 0x04;
pub const TAG_NOMINAL: u8 = 0x05;
pub const TAG_REGIMES: u8 = 0x06;
pub const TAG_EFFECTS: u8 = 0x07;
pub const TAG_PAIRS: u8 = 0x08;
pub const TAG_TREES: u8 = 0x09;
pub const TAG_RESID: u8 = 0x0A;
pub const TAG_CODE: u8 = 0x0B;
pub const TAG_WEIGHTS_FP: u8 = 0x0C;

#[derive(Clone, Debug)]
pub struct Dse8Header {
    pub task_id: u8,
    pub k_eff: u8,
    pub p_eff: u8,
    pub x_cols_raw: u16,
    pub flags: u8,
    pub rank_use: u8,
    pub knot_level: u8,
    pub log2_nfit_norm: f32,
    pub z_max: f32,
    pub resid_sd: f32,
    pub p1_prior: f32,
    pub inv_nu: f32,
    pub code_tier: u8,
}

#[derive(Clone, Debug)]
pub struct Dse8Packet {
    pub header: Dse8Header,
    pub card: ScaleCardV8,
    pub marginals: Vec<MarginalColV8>,
    pub loadings: Vec<[f32; V8_MAX_RANK]>,
    pub linear: LinearSectionV8,
    pub nominal_groups: Vec<NominalGroupV8>,
    pub regimes: RegimesSectionV8,
    pub effects: Vec<Effect1DV8>,
    pub pairs: Vec<PairIntV8>,
    pub trees: TreesSectionV8,
    pub resid_model: ResidSectionV8,
    pub code: CodeSectionV8,
    pub weights_fp: Option<[u8; 8]>,
}

fn push_tlv(buf: &mut Vec<u8>, tag: u8, payload: &[u8]) {
    buf.push(tag);
    buf.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    buf.extend_from_slice(payload);
}

fn encode_scale_section(card: &ScaleCardV8, flags: u8) -> Vec<u8> {
    if card.normalized_only {
        return Vec::new();
    }
    let mut b = Vec::with_capacity(card.wire_len());
    if card.task_id == 0 {
        b.extend_from_slice(&card.y_mean.to_le_bytes());
        b.extend_from_slice(&card.y_sd.to_le_bytes());
    } else if (flags & FLAG_HAS_CUSTOM_CLASSES) != 0 {
        for &cid in &card.classes {
            b.extend_from_slice(&cid.to_le_bytes());
        }
    }
    if (flags & FLAG_HAS_COL_MAP) != 0 {
        for &idx in &card.col_indices {
            b.extend_from_slice(&(idx as u16).to_le_bytes());
        }
    }
    for a in 0..card.col_indices.len() {
        let gk = card.group_kinds[a] & 0x0F;
        let bit_int = u8::from(card.is_int_active[a]) << 4;
        b.push(gk | bit_int);
        if !matches!(gk, 3..=5) {
            b.extend_from_slice(&card.center_active[a].to_le_bytes());
            b.extend_from_slice(&f32_to_bf16_bits(card.scale_active[a]).to_le_bytes());
        }
    }
    if (flags & FLAG_HAS_DROPPED_MEDS) != 0 {
        for &m in &card.dropped_medians {
            b.extend_from_slice(&f32_to_bf16_bits(m).to_le_bytes());
        }
    }
    b
}

fn encode_nominal_section(groups: &[NominalGroupV8]) -> Vec<u8> {
    let mut b = Vec::new();
    if groups.is_empty() {
        return b;
    }
    b.push(groups.len().min(32) as u8);
    for g in groups.iter().take(32) {
        b.push(g.cols.len() as u8);
        b.push((g.baseline_prob * 255.0).round().clamp(0.0, 255.0) as u8);
        for (&c, &p) in g.cols.iter().zip(&g.probs) {
            b.push(c);
            b.push((p * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    b
}

fn encode_effects_section(effects: &[Effect1DV8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(effects.len() * 12);
    for ef in effects {
        b.push(ef.col_idx);
        b.push(ef.cls_idx);
        b.extend_from_slice(&f32_to_f16_bits(ef.scale).to_le_bytes());
        for &v in &ef.bins8 {
            let q = (v / ef.scale.max(1e-5)).round().clamp(-127.0, 127.0) as i8;
            b.push(q as u8);
        }
    }
    b
}

fn encode_pairs_section(pairs: &[PairIntV8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(pairs.len() * 5);
    for pr in pairs {
        b.push(pr.col_i);
        b.push(pr.col_j);
        b.push(pr.cls_idx);
        b.push(((pr.w_prod * 100.0).round().clamp(-125.0, 125.0) as i8) as u8);
        b.push(((pr.w_xor * 100.0).round().clamp(-125.0, 125.0) as i8) as u8);
    }
    b
}

fn encode_resid_section(rs: &ResidSectionV8) -> Vec<u8> {
    if rs.cols.is_empty() && rs.q5_resid.iter().all(|v| v.abs() < 0.05) {
        return Vec::new();
    }
    let mut b = Vec::with_capacity(6 + 2 * rs.cols.len());
    b.push(rs.cols.len() as u8);
    for (&c, &w) in rs.cols.iter().zip(&rs.log_sd_w) {
        b.push(c);
        b.push(((w * 100.0).round().clamp(-60.0, 60.0) as i8) as u8);
    }
    for &q in &rs.q5_resid {
        b.push(((q * 100.0).round().clamp(-125.0, 125.0) as i8) as u8);
    }
    b
}

/// Pack a `Dse8Packet` strictly within `budget_bytes`, greedily fitting nested optional sections.
pub fn pack_dse8_packet(pkt: &Dse8Packet, budget_bytes: usize) -> Result<Vec<u8>> {
    pack_with_code_priority(pkt, budget_bytes, false)
}

pub(super) fn pack_with_code_priority(
    pkt: &Dse8Packet,
    budget_bytes: usize,
    reserve_code: bool,
) -> Result<Vec<u8>> {
    let h = &pkt.header;
    let mut sections: Vec<(u8, Vec<u8>)> = Vec::with_capacity(11);

    // Mandatory sections first!
    if !pkt.card.normalized_only {
        sections.push((TAG_SCALE, encode_scale_section(&pkt.card, h.flags)));
    }
    let mut marg_b = Vec::new();
    for m in &pkt.marginals {
        m.encode_bytes(&mut marg_b);
    }
    sections.push((TAG_MARG, marg_b));

    let mut cop_b = Vec::with_capacity(h.p_eff as usize * h.rank_use as usize);
    for row in &pkt.loadings {
        for &v in row.iter().take(h.rank_use as usize) {
            cop_b.push(((v * 127.0).round().clamp(-127.0, 127.0) as i8) as u8);
        }
    }
    sections.push((TAG_COPULA, cop_b));

    let mut lin_b = Vec::new();
    lin_b.extend_from_slice(&f32_to_f16_bits(pkt.linear.temp).to_le_bytes());
    for c in 0..pkt.linear.o_dim {
        lin_b.extend_from_slice(&pkt.linear.w_scales[c].to_le_bytes());
        lin_b.extend_from_slice(&pkt.linear.b[c].to_le_bytes());
        let sc = pkt.linear.w_scales[c].max(1e-6);
        for &wv in &pkt.linear.w[c] {
            lin_b.push(((wv / sc).round().clamp(-127.0, 127.0) as i8) as u8);
        }
    }
    sections.push((TAG_LINEAR, lin_b));

    // Nominal legality is MANDATORY whenever nominal groups exist (never silently emit illegal rows!).
    let nominal_pl = encode_nominal_section(&pkt.nominal_groups);
    if !nominal_pl.is_empty() {
        sections.push((TAG_NOMINAL, nominal_pl));
    }
    let fp_overhead = if pkt.weights_fp.is_some() {
        TLV_HDR_BYTES + 8
    } else {
        0
    };

    let mandatory_bytes: usize = DSE8_HEADER_BYTES
        + DSE8_CRC_BYTES
        + fp_overhead
        + sections
            .iter()
            .map(|(_, p)| TLV_HDR_BYTES + p.len())
            .sum::<usize>();
    if mandatory_bytes > budget_bytes {
        return Err(DopeError::Budget("over_budget".into()));
    }
    let mut rem = budget_bytes - mandatory_bytes;
    let code_reserved = if reserve_code {
        (1..=pkt.code.tier)
            .rev()
            .map(CodeSectionV8::tier_bytes)
            .find(|&need| pkt.code.packed.len() >= need && TLV_HDR_BYTES + need <= rem)
            .map_or(0, |need| TLV_HDR_BYTES + need)
    } else {
        0
    };
    rem -= code_reserved;

    // Optional enhancement sections in strict ascending tag & utility-per-byte order!
    let mut try_push = |tag: u8, payload: Vec<u8>, rem: &mut usize| {
        if !payload.is_empty() && TLV_HDR_BYTES + payload.len() <= *rem {
            *rem -= TLV_HDR_BYTES + payload.len();
            sections.push((tag, payload));
        }
    };

    let mut reg_b = Vec::new();
    pkt.regimes.encode_bytes(&mut reg_b);
    try_push(TAG_REGIMES, reg_b, &mut rem);

    // Prefix-fit effects (12 B each)
    if !pkt.effects.is_empty() && rem >= TLV_HDR_BYTES + 12 {
        let k = ((rem - TLV_HDR_BYTES) / 12).min(pkt.effects.len());
        if k > 0 {
            try_push(
                TAG_EFFECTS,
                encode_effects_section(&pkt.effects[..k]),
                &mut rem,
            );
        }
    }
    // Prefix-fit pairs (5 B each)
    if !pkt.pairs.is_empty() && rem >= TLV_HDR_BYTES + 5 {
        let k = ((rem - TLV_HDR_BYTES) / 5).min(pkt.pairs.len());
        if k > 0 {
            try_push(TAG_PAIRS, encode_pairs_section(&pkt.pairs[..k]), &mut rem);
        }
    }
    // Prefix-fit trees (14 B each) — reserve room for nested FSQ code if requested!
    let code_reserve = if !reserve_code
        && pkt.code.tier > 0
        && rem >= TLV_HDR_BYTES + CodeSectionV8::tier_bytes(1)
    {
        TLV_HDR_BYTES + CodeSectionV8::tier_bytes(pkt.code.tier).min(rem - TLV_HDR_BYTES)
    } else {
        0
    };
    let tree_rem = rem.saturating_sub(code_reserve);
    if !pkt.trees.trees.is_empty() && tree_rem >= TLV_HDR_BYTES + V8_TREE_WIRE_BYTES {
        let n_t = ((tree_rem - TLV_HDR_BYTES) / V8_TREE_WIRE_BYTES).min(pkt.trees.trees.len());
        if n_t > 0 {
            let mut tb = Vec::with_capacity(n_t * V8_TREE_WIRE_BYTES);
            for t in &pkt.trees.trees[..n_t] {
                t.encode_bytes(&mut tb);
            }
            try_push(TAG_TREES, tb, &mut rem);
        }
    }
    // Resid model
    let resid_b = encode_resid_section(&pkt.resid_model);
    if !resid_b.is_empty() && rem.saturating_sub(code_reserve) >= TLV_HDR_BYTES + resid_b.len() {
        try_push(TAG_RESID, resid_b, &mut rem);
    }
    // Nested FSQ neural code (fits highest tier 3 -> 2 -> 1 that fits!)
    rem += code_reserved;
    let mut final_code_tier = 0u8;
    if pkt.code.tier > 0 {
        for cand_t in (1..=pkt.code.tier).rev() {
            let need = CodeSectionV8::tier_bytes(cand_t);
            if pkt.code.packed.len() >= need && TLV_HDR_BYTES + need <= rem {
                let trunc = pkt.code.truncate_to_tier(cand_t);
                final_code_tier = trunc.tier;
                try_push(TAG_CODE, trunc.packed, &mut rem);
                break;
            }
        }
    }
    // Trailing mandatory 8-byte neural decoder weights fingerprint when neural encoder was used.
    let mut final_flags = h.flags & ALL_KNOWN_FLAGS;
    if let Some(fp) = pkt.weights_fp {
        final_flags |= FLAG_HAS_NEURAL_ENCODER;
        sections.push((TAG_WEIGHTS_FP, fp.to_vec()));
    } else {
        final_flags &= !FLAG_HAS_NEURAL_ENCODER;
    }

    let mut buf = Vec::with_capacity(budget_bytes);
    buf.extend_from_slice(DSE8_MAGIC);
    buf.push(DSE8_VERSION);
    buf.push(h.task_id);
    buf.push(h.k_eff);
    buf.push(h.p_eff);
    buf.extend_from_slice(&h.x_cols_raw.to_le_bytes());
    buf.push(final_flags);
    buf.push(h.rank_use);
    buf.push(h.knot_level);
    buf.push((h.log2_nfit_norm * 255.0).round().clamp(0.0, 255.0) as u8);
    buf.extend_from_slice(&((h.z_max * 10000.0).round().clamp(1.0, 65000.0) as u16).to_le_bytes());
    buf.extend_from_slice(
        &((h.resid_sd * 10000.0).round().clamp(1.0, 65000.0) as u16).to_le_bytes(),
    );
    buf.push((h.p1_prior * 255.0).round().clamp(0.0, 255.0) as u8);
    buf.push(((h.inv_nu / 0.45) * 255.0).round().clamp(0.0, 255.0) as u8);
    buf.push(sections.len() as u8);
    buf.push(final_code_tier);
    buf.extend_from_slice(&0u16.to_le_bytes());

    for (tag, payload) in &sections {
        push_tlv(&mut buf, *tag, payload);
    }
    let crc = crc32_ieee(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    debug_assert!(buf.len() <= budget_bytes);
    Ok(buf)
}

/// Nested-prefix truncation: shrink a larger `DSE8` packet (e.g. 4096 B) to `<= target_budget`
/// with zero refitting by dropping trailing optional sections / trees / FSQ tiers & re-signing CRC32!
pub fn truncate_dse8_packet(bytes: &[u8], target_budget: usize) -> Result<Vec<u8>> {
    let pkt = unpack_dse8_packet(bytes)?;
    if bytes.len() <= target_budget {
        return Ok(bytes.to_vec());
    }
    pack_dse8_packet(&pkt, target_budget)
}

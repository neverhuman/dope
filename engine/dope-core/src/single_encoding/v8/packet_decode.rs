//! Fail-closed CRC32-verified binary unpacker for `DSE8` v1 wire packets.
//!
//! Strictly enforces:
//! - CRC32-IEEE covering every preceding byte
//! - Known header flag bits & zero reserved header bytes (`bytes[22..24] == 0`)
//! - Strict ascending, non-duplicate TLV section tags (`tag > last_tag`)
//! - Consistency between `FLAG_NORMALIZED_ONLY` and `TAG_SCALE`
//! - Consistency between `header.code_tier` and `TAG_CODE`
//! - Consistency between `FLAG_HAS_NEURAL_ENCODER` and `TAG_WEIGHTS_FP`

use super::copula_regimes::{RegimesSectionV8, V8_MAX_RANK};
use super::marginals::MarginalColV8;
use super::nominal::NominalGroupV8;
use super::packet::{
    Dse8Header, Dse8Packet, ALL_KNOWN_FLAGS, DSE8_MAGIC, DSE8_VERSION, FLAG_ACTIVE_RANK_V1,
    FLAG_HAS_COL_MAP, FLAG_HAS_CUSTOM_CLASSES, FLAG_HAS_DROPPED_MEDS, FLAG_HAS_NEURAL_ENCODER,
    FLAG_NORMALIZED_ONLY, TAG_CODE, TAG_COPULA, TAG_EFFECTS, TAG_LINEAR, TAG_MARG, TAG_NOMINAL,
    TAG_PAIRS, TAG_REGIMES, TAG_RESID, TAG_SCALE, TAG_TREES, TAG_WEIGHTS_FP,
};
use super::planner::{DSE8_CRC_BYTES, DSE8_HEADER_BYTES, TLV_HDR_BYTES};
use super::scale_v8::{ScaleCardV8, V8_MAX_CLASSES, V8_MAX_COLS};
use super::target_v8::{
    CodeSectionV8, Effect1DV8, LinearSectionV8, PairIntV8, ResidSectionV8, V8_MAX_EFFECTS,
    V8_MAX_PAIRS, V8_MAX_RESID_COLS,
};
use super::trees_v8::TreesSectionV8;
use crate::artifact::format2::crc32_ieee;
use crate::error::{DopeError, Result};
use crate::single_encoding::math::f16_bits_to_f32;
use crate::single_encoding::scale_card::bf16_bits_to_f32;

/// Fail-closed CRC32-verified unpacker for `DSE8` v1 wire packets.
pub fn unpack_dse8_packet(bytes: &[u8]) -> Result<Dse8Packet> {
    if bytes.len() < DSE8_HEADER_BYTES + DSE8_CRC_BYTES {
        return Err(DopeError::Artifact("dse8 packet too short".into()));
    }
    if &bytes[..4] != DSE8_MAGIC || bytes[4] != DSE8_VERSION {
        return Err(DopeError::Artifact("dse8 invalid magic or version".into()));
    }
    let body_len = bytes.len() - DSE8_CRC_BYTES;
    let exp_crc = u32::from_le_bytes(bytes[body_len..].try_into().unwrap());
    if crc32_ieee(&bytes[..body_len]) != exp_crc {
        return Err(DopeError::Artifact("dse8 crc32 mismatch".into()));
    }

    let task_id = bytes[5];
    let k_eff = bytes[6] as usize;
    let p_eff = bytes[7] as usize;
    let x_cols_raw = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let flags = bytes[10];
    let rank_use = bytes[11] as usize;
    let knot_level = bytes[12] as usize;
    let log2_nfit_norm = bytes[13] as f32 / 255.0;
    let z_max = u16::from_le_bytes([bytes[14], bytes[15]]) as f32 / 10000.0;
    let resid_sd = u16::from_le_bytes([bytes[16], bytes[17]]) as f32 / 10000.0;
    let p1_prior = bytes[18] as f32 / 255.0;
    let inv_nu = (bytes[19] as f32 / 255.0) * 0.45;
    let section_count = bytes[20] as usize;
    let code_tier = bytes[21];
    let reserved = u16::from_le_bytes([bytes[22], bytes[23]]);

    if (flags & !ALL_KNOWN_FLAGS) != 0 {
        return Err(DopeError::Artifact("dse8 unknown header flags".into()));
    }
    if (flags & FLAG_ACTIVE_RANK_V1) != 0 && (flags & FLAG_HAS_NEURAL_ENCODER) == 0 {
        return Err(DopeError::Artifact(
            "dse8 active_rank_v1 flag requires neural encoder".into(),
        ));
    }
    if reserved != 0 {
        return Err(DopeError::Artifact(
            "dse8 non-zero reserved header bytes".into(),
        ));
    }
    if code_tier > 3 {
        return Err(DopeError::Artifact("dse8 invalid code_tier".into()));
    }
    if task_id > 2
        || p_eff == 0
        || p_eff > V8_MAX_COLS
        || x_cols_raw < p_eff
        || rank_use > V8_MAX_RANK
        || !matches!(knot_level, 5 | 9 | 17)
        || k_eff == 0
        || k_eff > V8_MAX_CLASSES
    {
        return Err(DopeError::Artifact("dse8 header bounds invalid".into()));
    }

    let normalized_only = (flags & FLAG_NORMALIZED_ONLY) != 0;
    if normalized_only
        && (flags & (FLAG_HAS_COL_MAP | FLAG_HAS_CUSTOM_CLASSES | FLAG_HAS_DROPPED_MEDS)) != 0
    {
        return Err(DopeError::Artifact(
            "dse8 scale flags forbidden in normalized_only mode".into(),
        ));
    }
    if normalized_only && x_cols_raw != p_eff {
        return Err(DopeError::Artifact(
            "dse8 normalized_only forbids projected x_cols_raw != p_eff".into(),
        ));
    }
    let o_dim = if task_id == 2 { k_eff } else { 1 };
    let mut card = ScaleCardV8 {
        normalized_only,
        x_cols_raw,
        col_indices: (0..p_eff).collect(),
        group_kinds: vec![0; p_eff],
        center_active: vec![0.0; p_eff],
        scale_active: vec![1.0; p_eff],
        is_int_active: vec![false; p_eff],
        dropped_medians: Vec::new(),
        task_id,
        y_mean: 0.0,
        y_sd: 1.0,
        classes: if task_id == 0 {
            Vec::new()
        } else {
            (0..k_eff as i32).collect()
        },
    };

    let mut marginals: Option<Vec<MarginalColV8>> = None;
    let mut loadings: Option<Vec<[f32; V8_MAX_RANK]>> = None;
    let mut linear: Option<LinearSectionV8> = None;
    let mut nominal_groups = Vec::new();
    let mut regimes = RegimesSectionV8::default();
    let mut effects = Vec::new();
    let mut pairs = Vec::new();
    let mut trees = TreesSectionV8::default();
    let mut resid_model = ResidSectionV8::default();
    let mut code = CodeSectionV8::default();

    let mut saw_scale = false;
    let mut saw_code = false;
    let mut weights_fp: Option<[u8; 8]> = None;
    let mut last_tag = 0u8;

    let mut cur = DSE8_HEADER_BYTES;
    for _ in 0..section_count {
        if cur + TLV_HDR_BYTES > body_len {
            return Err(DopeError::Artifact("dse8 truncated tlv header".into()));
        }
        let tag = bytes[cur];
        if tag <= last_tag {
            return Err(DopeError::Artifact(
                "dse8 duplicate or out-of-order section tag".into(),
            ));
        }
        last_tag = tag;
        let slen = u16::from_le_bytes([bytes[cur + 1], bytes[cur + 2]]) as usize;
        cur += TLV_HDR_BYTES;
        if cur + slen > body_len {
            return Err(DopeError::Artifact("dse8 truncated tlv payload".into()));
        }
        let pl = &bytes[cur..cur + slen];
        cur += slen;

        match tag {
            TAG_SCALE => {
                if normalized_only {
                    return Err(DopeError::Artifact(
                        "dse8 scale section forbidden in normalized_only mode".into(),
                    ));
                }
                saw_scale = true;
                decode_scale_section(pl, &mut card, flags, k_eff, p_eff, x_cols_raw)?;
            }
            TAG_MARG => {
                let mut mc = 0usize;
                let mut cols = Vec::with_capacity(p_eff);
                for _ in 0..p_eff {
                    cols.push(
                        MarginalColV8::decode_bytes(pl, &mut mc, knot_level).ok_or_else(|| {
                            DopeError::Artifact("dse8 bad marginal section".into())
                        })?,
                    );
                }
                if mc != pl.len() {
                    return Err(DopeError::Artifact("dse8 marginal trailing bytes".into()));
                }
                marginals = Some(cols);
            }
            TAG_COPULA => {
                if pl.len() != p_eff * rank_use {
                    return Err(DopeError::Artifact("dse8 bad copula len".into()));
                }
                let mut l = vec![[0.0f32; V8_MAX_RANK]; p_eff];
                for j in 0..p_eff {
                    for r in 0..rank_use {
                        l[j][r] = (pl[j * rank_use + r] as i8) as f32 / 127.0;
                    }
                }
                loadings = Some(l);
            }
            TAG_LINEAR => {
                if pl.len() != 2 + o_dim * (8 + p_eff) {
                    return Err(DopeError::Artifact("dse8 bad linear len".into()));
                }
                let temp = f16_bits_to_f32(u16::from_le_bytes([pl[0], pl[1]])).clamp(0.2, 5.0);
                let mut lc = 2;
                let mut w = vec![vec![0.0f32; p_eff]; o_dim];
                let mut b = vec![0.0f32; o_dim];
                let mut w_scales = vec![0.0f32; o_dim];
                for c in 0..o_dim {
                    let sc = f32::from_le_bytes(pl[lc..lc + 4].try_into().unwrap());
                    let bv = f32::from_le_bytes(pl[lc + 4..lc + 8].try_into().unwrap());
                    lc += 8;
                    w_scales[c] = sc;
                    b[c] = bv;
                    for j in 0..p_eff {
                        w[c][j] = (pl[lc + j] as i8) as f32 * sc;
                    }
                    lc += p_eff;
                }
                linear = Some(LinearSectionV8 {
                    o_dim,
                    p_eff,
                    w,
                    b,
                    w_scales,
                    temp,
                    resid_sd,
                });
            }
            TAG_NOMINAL => nominal_groups = decode_nominal_section(pl, p_eff)?,
            TAG_REGIMES => {
                regimes = RegimesSectionV8::decode_bytes(pl)
                    .ok_or_else(|| DopeError::Artifact("dse8 bad regimes section".into()))?;
            }
            TAG_EFFECTS => {
                if pl.len() % 12 != 0 || pl.len() / 12 > V8_MAX_EFFECTS {
                    return Err(DopeError::Artifact("dse8 bad effects len".into()));
                }
                for ch in pl.chunks_exact(12) {
                    let sc = f16_bits_to_f32(u16::from_le_bytes([ch[2], ch[3]]));
                    let mut bins8 = [0.0f32; 8];
                    for bi in 0..8 {
                        bins8[bi] = (ch[4 + bi] as i8) as f32 * sc;
                    }
                    effects.push(Effect1DV8 {
                        col_idx: ch[0],
                        cls_idx: ch[1],
                        scale: sc,
                        bins8,
                        utility: 0.0,
                    });
                }
            }
            TAG_PAIRS => {
                if pl.len() % 5 != 0 || pl.len() / 5 > V8_MAX_PAIRS {
                    return Err(DopeError::Artifact("dse8 bad pairs len".into()));
                }
                for ch in pl.chunks_exact(5) {
                    pairs.push(PairIntV8 {
                        col_i: ch[0],
                        col_j: ch[1],
                        cls_idx: ch[2],
                        w_prod: (ch[3] as i8) as f32 / 100.0,
                        w_xor: (ch[4] as i8) as f32 / 100.0,
                        utility: 0.0,
                    });
                }
            }
            TAG_TREES => {
                trees = TreesSectionV8::decode_bytes(pl)
                    .ok_or_else(|| DopeError::Artifact("dse8 bad trees section".into()))?;
            }
            TAG_RESID => resid_model = decode_resid_section(pl, p_eff)?,
            TAG_CODE => {
                let tier = match pl.len() {
                    32 => 1,
                    64 => 2,
                    128 => 3,
                    _ => return Err(DopeError::Artifact("dse8 invalid code section len".into())),
                };
                saw_code = true;
                code = CodeSectionV8 {
                    tier,
                    packed: pl.to_vec(),
                };
            }
            TAG_WEIGHTS_FP => {
                if pl.len() != 8 {
                    return Err(DopeError::Artifact("dse8 invalid weights_fp len".into()));
                }
                let mut fp = [0u8; 8];
                fp.copy_from_slice(pl);
                weights_fp = Some(fp);
            }
            _ => return Err(DopeError::Artifact("dse8 unknown section tag".into())),
        }
    }
    if cur != body_len {
        return Err(DopeError::Artifact("dse8 trailing payload bytes".into()));
    }
    if !normalized_only && !saw_scale {
        return Err(DopeError::Artifact(
            "dse8 missing mandatory scale section".into(),
        ));
    }
    if code_tier == 0 && saw_code {
        return Err(DopeError::Artifact(
            "dse8 unexpected code section for code_tier=0".into(),
        ));
    }
    if code_tier > 0 && (!saw_code || code.tier != code_tier) {
        return Err(DopeError::Artifact(
            "dse8 code section inconsistent with header code_tier".into(),
        ));
    }
    let has_nn_flag = (flags & FLAG_HAS_NEURAL_ENCODER) != 0;
    if has_nn_flag != weights_fp.is_some() {
        return Err(DopeError::Artifact(
            "dse8 neural encoder flag inconsistent with weights_fp".into(),
        ));
    }

    Ok(Dse8Packet {
        header: Dse8Header {
            task_id,
            k_eff: k_eff as u8,
            p_eff: p_eff as u8,
            x_cols_raw: x_cols_raw as u16,
            flags,
            rank_use: rank_use as u8,
            knot_level: knot_level as u8,
            log2_nfit_norm,
            z_max,
            resid_sd,
            p1_prior,
            inv_nu,
            code_tier,
        },
        card,
        marginals: marginals.ok_or_else(|| DopeError::Artifact("dse8 missing marginals".into()))?,
        loadings: loadings.ok_or_else(|| DopeError::Artifact("dse8 missing copula".into()))?,
        linear: linear.ok_or_else(|| DopeError::Artifact("dse8 missing linear".into()))?,
        nominal_groups,
        regimes,
        effects,
        pairs,
        trees,
        resid_model,
        code,
        weights_fp,
    })
}

fn decode_scale_section(
    pl: &[u8],
    card: &mut ScaleCardV8,
    flags: u8,
    k_eff: usize,
    p_eff: usize,
    x_cols_raw: usize,
) -> Result<()> {
    let mut c = 0usize;
    let bad = || DopeError::Artifact("dse8 bad scale section".into());
    if card.task_id == 0 {
        if c + 8 > pl.len() {
            return Err(bad());
        }
        card.y_mean = f32::from_le_bytes(pl[c..c + 4].try_into().unwrap());
        card.y_sd = f32::from_le_bytes(pl[c + 4..c + 8].try_into().unwrap());
        c += 8;
    } else if (flags & FLAG_HAS_CUSTOM_CLASSES) != 0 {
        if c + 4 * k_eff > pl.len() {
            return Err(bad());
        }
        card.classes.clear();
        for _ in 0..k_eff {
            card.classes
                .push(i32::from_le_bytes(pl[c..c + 4].try_into().unwrap()));
            c += 4;
        }
    }
    if (flags & FLAG_HAS_COL_MAP) != 0 {
        if c + 2 * p_eff > pl.len() {
            return Err(bad());
        }
        card.col_indices.clear();
        for _ in 0..p_eff {
            let idx = u16::from_le_bytes([pl[c], pl[c + 1]]) as usize;
            if idx >= x_cols_raw {
                return Err(bad());
            }
            card.col_indices.push(idx);
            c += 2;
        }
    }
    for a in 0..p_eff {
        if c >= pl.len() {
            return Err(bad());
        }
        let b = pl[c];
        c += 1;
        let gk = b & 0x0F;
        card.group_kinds[a] = gk;
        card.is_int_active[a] = (b & 0x10) != 0;
        if !matches!(gk, 3..=5) {
            if c + 6 > pl.len() {
                return Err(bad());
            }
            card.center_active[a] = f32::from_le_bytes(pl[c..c + 4].try_into().unwrap());
            card.scale_active[a] = bf16_bits_to_f32(u16::from_le_bytes([pl[c + 4], pl[c + 5]]));
            c += 6;
        }
    }
    if (flags & FLAG_HAS_DROPPED_MEDS) != 0 {
        let n_drop = x_cols_raw - p_eff;
        if c + 2 * n_drop > pl.len() {
            return Err(bad());
        }
        card.dropped_medians.clear();
        for _ in 0..n_drop {
            card.dropped_medians
                .push(bf16_bits_to_f32(u16::from_le_bytes([pl[c], pl[c + 1]])));
            c += 2;
        }
    }
    if c != pl.len() {
        return Err(bad());
    }
    Ok(())
}

fn decode_nominal_section(pl: &[u8], p_eff: usize) -> Result<Vec<NominalGroupV8>> {
    if pl.is_empty() {
        return Ok(Vec::new());
    }
    let bad = || DopeError::Artifact("dse8 bad nominal section".into());
    let ng = pl[0] as usize;
    let mut c = 1usize;
    let mut out = Vec::with_capacity(ng);
    for _ in 0..ng {
        if c + 2 > pl.len() {
            return Err(bad());
        }
        let w = pl[c] as usize;
        let bp = pl[c + 1] as f32 / 255.0;
        c += 2;
        if w < 2 || c + 2 * w > pl.len() {
            return Err(bad());
        }
        let mut cols = Vec::with_capacity(w);
        let mut probs = Vec::with_capacity(w);
        for _ in 0..w {
            let col = pl[c];
            if col as usize >= p_eff {
                return Err(bad());
            }
            cols.push(col);
            probs.push(pl[c + 1] as f32 / 255.0);
            c += 2;
        }
        out.push(NominalGroupV8 {
            cols,
            probs,
            baseline_prob: bp,
        });
    }
    if c != pl.len() {
        return Err(bad());
    }
    Ok(out)
}

fn decode_resid_section(pl: &[u8], p_eff: usize) -> Result<ResidSectionV8> {
    let bad = || DopeError::Artifact("dse8 bad resid section".into());
    if pl.is_empty() {
        return Ok(ResidSectionV8::default());
    }
    let k = pl[0] as usize;
    if k > V8_MAX_RESID_COLS || pl.len() != 1 + 2 * k + 5 {
        return Err(bad());
    }
    let mut cols = Vec::with_capacity(k);
    let mut log_sd_w = Vec::with_capacity(k);
    for i in 0..k {
        let col = pl[1 + 2 * i];
        if col as usize >= p_eff {
            return Err(bad());
        }
        cols.push(col);
        log_sd_w.push((pl[2 + 2 * i] as i8) as f32 / 100.0);
    }
    let mut q5 = [0.0f32; 5];
    let base = 1 + 2 * k;
    for i in 0..5 {
        q5[i] = (pl[base + i] as i8) as f32 / 100.0;
    }
    Ok(ResidSectionV8 {
        cols,
        log_sd_w,
        q5_resid: q5,
    })
}

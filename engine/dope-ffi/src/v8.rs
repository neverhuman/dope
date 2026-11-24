//! V8 episode (ABI 3): the FIT-only DSE8 packet, the receiver decoder's inputs built from that
//! packet alone, and the FIT-only evidence bank, for the `SingleEncV8` trainer.
//!
//! Everything is recomputed from the episode's FIT rows; query rows are only mapped into the
//! packet's unit space. The layout is sized by `p_cap = min(p, V8_MAX_COLS)` so a 48-column
//! episode does not pay for 64-column edge blocks; columns at or beyond `p_eff` are zero.
//!
//! Column meta is computed in the scale card's own unit space (center, scale). DSE8 uses
//! `lattice_q0_unit` as the lattice step of its marginals, so meta fitted in any other unit
//! would put decoded values off the grid.

use dope_core::evidence::{EvidenceTier, COL_DIM, EDGE_DIM, GLOB_DIM as EV_GLOB_DIM};
use dope_core::marginal::column_meta::META_FIELDS;
use dope_core::single_encoding::v8::{
    build_decoder_inputs, extract_fit_evidence_v8, fit_dse8_packet, fit_unit_meta_v2,
    unpack_dse8_packet, Dse8ColumnMeta, Dse8FitOptions, ScaleCardV8, DEC_CODE_DIM, DEC_COL_DIM,
    DEC_EDGE_DIM, DEC_GLOB_DIM, MARG_CONT, V8_MAX_CLASSES, V8_MAX_COLS, V8_MAX_RANK,
};
use dope_core::single_encoding::KNOT_COUNT;

pub use crate::v8_reserved::{
    dope_ffi_project_fit_midrank, dope_ffi_v8_episode, dope_ffi_v8_episode_prepared,
    dope_ffi_v8_episode_reserved, dope_ffi_v8_schema_wrapper_bytes,
};
use crate::{columns, Code, MAX_BUDGET};

/// Number of `MISC` slots. In order: resid_sd, z_max, p_eff, k_eff, x_cols_raw, task_id,
/// rank_use, knot_level, normalized_only, packet_len, y_mean, y_sd, n_classes (the packet's
/// class list), log2_nfit_norm, p1_prior, inv_nu.
pub const MISC_LEN: usize = 16;

/// Block ids of [`offsets`]; `HEAD` is where the rows start.
pub mod block {
    pub const EV_COL: usize = 0;
    pub const EV_EDGE: usize = 1;
    pub const EV_GLOB: usize = 2;
    pub const DEC_COL: usize = 3;
    pub const DEC_EDGE: usize = 4;
    pub const DEC_GLOB: usize = 5;
    pub const DEC_CODE: usize = 6;
    pub const KNOTS: usize = 7;
    pub const LOADINGS: usize = 8;
    pub const W_BASE: usize = 9;
    pub const B_BASE: usize = 10;
    pub const IS_CONT: usize = 11;
    pub const COL_INDEX: usize = 12;
    pub const COL_META: usize = 13;
    pub const MISC: usize = 14;
    pub const HEAD: usize = 15;
    pub const COUNT: usize = 16;
}

/// Offsets (f32 elements) of every block for a layout of width `p_cap`. Shapes, row-major:
/// EV_COL `[p][128]`, EV_EDGE `[p][p][16]`, EV_GLOB `[160]`, DEC_COL `[p][48]`, DEC_EDGE
/// `[p][p][4]`, DEC_GLOB `[32]`, DEC_CODE `[64]`, KNOTS `[p][17]`, LOADINGS `[p][12]`, W_BASE
/// `[8][p]`, B_BASE `[8]`, IS_CONT `[p]`, COL_INDEX `[p]` (-1 unused), COL_META
/// `[p][META_FIELDS]`, MISC `[MISC_LEN]`. After HEAD: `x_fit [n_fit][p]`, `y_fit [n_fit]`,
/// `x_qry [n_qry][p]`, `y_qry [n_qry]`.
pub const fn offsets(p_cap: usize) -> [usize; block::COUNT] {
    let sizes = [
        p_cap * COL_DIM,
        p_cap * p_cap * EDGE_DIM,
        EV_GLOB_DIM,
        p_cap * DEC_COL_DIM,
        p_cap * p_cap * DEC_EDGE_DIM,
        DEC_GLOB_DIM,
        DEC_CODE_DIM,
        p_cap * KNOT_COUNT,
        p_cap * V8_MAX_RANK,
        V8_MAX_CLASSES * p_cap,
        V8_MAX_CLASSES,
        p_cap,
        p_cap,
        p_cap * META_FIELDS,
        MISC_LEN,
    ];
    let mut out = [0usize; block::COUNT];
    let mut i = 1;
    while i < block::COUNT {
        out[i] = out[i - 1] + sizes[i - 1];
        i += 1;
    }
    out
}

/// Layout width for an input of `p` columns.
pub const fn p_cap(p: usize) -> usize {
    if p < V8_MAX_COLS {
        p
    } else {
        V8_MAX_COLS
    }
}

/// Total f32 elements for `p` input columns and the given row counts.
pub const fn total(p: usize, n_fit: usize, n_qry: usize) -> usize {
    let w = p_cap(p);
    offsets(w)[block::HEAD] + (n_fit + n_qry) * (w + 1)
}

/// One native V8 episode.
#[derive(Clone, Debug, PartialEq)]
pub struct V8Episode {
    pub packet: Vec<u8>,
    /// `total(p, n_fit, n_qry)` floats.
    pub floats: Vec<f32>,
}

fn fit_card(
    x_raw: &[Vec<f64>],
    y_raw: &[f64],
    task: u8,
    p_eff: usize,
    normalized_only: bool,
    meta: &[Dse8ColumnMeta],
) -> Result<ScaleCardV8, Code> {
    // Same two-step choice as `fit_dse8_packet`: a probe at V8_MAX_COLS, refitted at p_eff.
    let probe = ScaleCardV8::fit(x_raw, y_raw, task, V8_MAX_COLS, normalized_only, Some(meta))
        .map_err(|_| Code::Fit)?;
    if probe.col_indices.len() == p_eff {
        return Ok(probe);
    }
    ScaleCardV8::fit(x_raw, y_raw, task, p_eff, normalized_only, Some(meta)).map_err(|_| Code::Fit)
}

/// Fit the DSE8 packet on the FIT rows only, build the receiver decoder's inputs from the
/// unpacked packet, extract the FIT-only evidence bank, and map FIT and query rows into the
/// packet's unit space.
#[allow(clippy::too_many_arguments)]
pub fn v8_episode(
    x_fit: &[f32],
    n_fit: usize,
    p: usize,
    y_fit: &[f32],
    x_qry: &[f32],
    n_qry: usize,
    y_qry: &[f32],
    task: u8,
    budget: usize,
    normalized_only: bool,
    tier: u8,
    seed: u64,
) -> Result<V8Episode, Code> {
    v8_episode_rows(
        x_fit,
        n_fit,
        p,
        y_fit,
        x_qry,
        n_qry,
        y_qry,
        task,
        budget,
        normalized_only,
        tier,
        seed,
        None,
    )
}

/// `v8_episode` with stable physical ids of the FIT rows: the evidence bank is then the same for
/// any order of those rows and cross-fits on duplicate-group folds (`extract_with_rows`).
#[allow(clippy::too_many_arguments)]
pub fn v8_episode_rows(
    x_fit: &[f32],
    n_fit: usize,
    p: usize,
    y_fit: &[f32],
    x_qry: &[f32],
    n_qry: usize,
    y_qry: &[f32],
    task: u8,
    budget: usize,
    normalized_only: bool,
    tier: u8,
    seed: u64,
    fit_row_ids: Option<&[u64]>,
) -> Result<V8Episode, Code> {
    v8_episode_with_reserve(
        x_fit,
        n_fit,
        p,
        y_fit,
        x_qry,
        n_qry,
        y_qry,
        task,
        budget,
        normalized_only,
        tier,
        seed,
        fit_row_ids,
        0,
        false,
    )
}

/// Shared native V8 episode builder with optional exact neural code-tier + DPA wrapper reservation.
#[allow(clippy::too_many_arguments)]
pub fn v8_episode_with_reserve(
    x_fit: &[f32],
    n_fit: usize,
    p: usize,
    y_fit: &[f32],
    x_qry: &[f32],
    n_qry: usize,
    y_qry: &[f32],
    task: u8,
    budget: usize,
    normalized_only: bool,
    tier: u8,
    seed: u64,
    fit_row_ids: Option<&[u64]>,
    code_tier: u8,
    reserve_dpa_wrapper: bool,
) -> Result<V8Episode, Code> {
    v8_episode_prepared(
        x_fit,
        n_fit,
        p,
        y_fit,
        x_qry,
        n_qry,
        y_qry,
        task,
        budget,
        normalized_only,
        tier,
        seed,
        fit_row_ids,
        code_tier,
        reserve_dpa_wrapper,
        0,
    )
}

/// Shared native V8 episode builder with exact neural code-tier + real `V7Schema` wrapper bytes.
#[allow(clippy::too_many_arguments)]
pub fn v8_episode_prepared(
    x_fit: &[f32],
    n_fit: usize,
    p: usize,
    y_fit: &[f32],
    x_qry: &[f32],
    n_qry: usize,
    y_qry: &[f32],
    task: u8,
    budget: usize,
    normalized_only: bool,
    tier: u8,
    seed: u64,
    fit_row_ids: Option<&[u64]>,
    code_tier: u8,
    reserve_dpa_wrapper: bool,
    wrapper_bytes_override: usize,
) -> Result<V8Episode, Code> {
    let shape_ok = p > 0
        && p <= u16::MAX as usize
        && n_fit >= 4
        && task <= 2
        && tier <= 2
        && code_tier <= 3
        && budget > 0
        && budget <= MAX_BUDGET as usize
        && x_fit.len() == n_fit * p
        && y_fit.len() == n_fit
        && x_qry.len() == n_qry * p
        && y_qry.len() == n_qry
        && fit_row_ids.is_none_or(|ids| ids.len() == n_fit);
    if !shape_ok {
        return Err(Code::Shape);
    }
    if y_fit.iter().any(|v| !v.is_finite()) {
        return Err(Code::Target);
    }
    let x_raw = columns(x_fit, n_fit, p);
    let y_raw: Vec<f64> = y_fit.iter().map(|&v| v as f64).collect();

    // FIT-only meta in the packet's unit space, shared with every other DSE8 fit path.
    let meta = fit_unit_meta_v2(&x_raw, &y_raw, task, normalized_only).map_err(|_| Code::Fit)?;
    let dse8_meta: Vec<Dse8ColumnMeta> = meta.iter().map(Dse8ColumnMeta::from).collect();
    let dpa_overhead = if wrapper_bytes_override > 0 {
        wrapper_bytes_override
    } else if reserve_dpa_wrapper {
        dope_core::auto::v8_dpa::estimate_dpa_v8_wrapper_bytes(&x_raw, &y_raw, task)
    } else {
        0
    };
    let eff_budget = budget
        .checked_sub(dpa_overhead)
        .filter(|&b| b >= 128)
        .ok_or(Code::Shape)?;

    let opts = Dse8FitOptions {
        task_id: task,
        budget_bytes: eff_budget,
        normalized_only,
        code_tier,
        reserve_neural_code: code_tier > 0,
        col_meta: Some(&dse8_meta),
        fit_row_ids,
        seed,
        ..Dse8FitOptions::default()
    };
    let packet = fit_dse8_packet(&x_raw, &y_raw, &opts).map_err(|_| Code::Fit)?;
    let pkt = unpack_dse8_packet(&packet).map_err(|_| Code::Unpack)?;
    let p_eff = pkt.header.p_eff as usize;
    let card = fit_card(&x_raw, &y_raw, task, p_eff, normalized_only, &dse8_meta)?;
    // The card must be the packet's own: same active columns in the same order.
    let width = p_cap(p);
    if card.col_indices.len() != p_eff || p_eff > width || p_eff != pkt.marginals.len() {
        return Err(Code::Unpack);
    }
    if !normalized_only && pkt.card.col_indices != card.col_indices {
        return Err(Code::Unpack);
    }

    let x_unit_fit = card.forward_x(&x_raw);
    let (ev_y, keep) = card.forward_y(&y_raw);
    let y_codes_fit: Vec<f32> = y_raw
        .iter()
        .map(|&v| card.forward_y_one(v).map_or(-1.0, |ys| ys as f32))
        .collect();
    if keep.len() < 4 {
        return Err(Code::Target);
    }
    let ev_unit: Vec<Vec<f32>> = x_unit_fit
        .iter()
        .map(|col| keep.iter().map(|&i| col[i]).collect())
        .collect();
    let active_meta: Vec<Dse8ColumnMeta> = card.col_indices.iter().map(|&j| dse8_meta[j]).collect();
    let tier_enum = match tier {
        0 => EvidenceTier::T0,
        1 => EvidenceTier::T1,
        _ => EvidenceTier::T2,
    };
    let ev = extract_fit_evidence_v8(
        &x_raw,
        &ev_unit,
        &ev_y,
        &keep,
        &card,
        Some(&active_meta),
        tier_enum,
        seed,
        fit_row_ids,
        true,
    )
    .map_err(|_| Code::Fit)?;
    if ev.p != p_eff || ev.col.len() != p_eff || ev.edge.len() != p_eff * p_eff {
        return Err(Code::Fit);
    }

    let off = offsets(width);
    let mut out = vec![0.0f32; total(p, n_fit, n_qry)];
    let (dec_col, dec_edge, dec_glob, dec_code) = build_decoder_inputs(&pkt);
    for c in 0..p_eff {
        out[off[block::EV_COL] + c * COL_DIM..][..COL_DIM].copy_from_slice(&ev.col[c]);
        out[off[block::DEC_COL] + c * DEC_COL_DIM..][..DEC_COL_DIM].copy_from_slice(&dec_col[c]);
        for d in 0..p_eff {
            let at = c * width + d;
            out[off[block::EV_EDGE] + at * EDGE_DIM..][..EDGE_DIM]
                .copy_from_slice(&ev.edge[c * p_eff + d]);
            out[off[block::DEC_EDGE] + at * DEC_EDGE_DIM..][..DEC_EDGE_DIM]
                .copy_from_slice(&dec_edge[c * p_eff + d][..DEC_EDGE_DIM]);
        }
        let m = &pkt.marginals[c];
        out[off[block::KNOTS] + c * KNOT_COUNT..][..KNOT_COUNT].copy_from_slice(&m.knots17);
        out[off[block::LOADINGS] + c * V8_MAX_RANK..][..V8_MAX_RANK]
            .copy_from_slice(&pkt.loadings[c]);
        out[off[block::IS_CONT] + c] = if m.kind == MARG_CONT { 1.0 } else { 0.0 };
        let j = card.col_indices[c];
        out[off[block::COL_INDEX] + c] = j as f32;
        out[off[block::COL_META] + c * META_FIELDS..][..META_FIELDS]
            .copy_from_slice(&meta[j].to_f32s());
    }
    for c in p_eff..width {
        out[off[block::COL_INDEX] + c] = -1.0;
    }
    out[off[block::EV_GLOB]..][..EV_GLOB_DIM].copy_from_slice(&ev.glob);
    out[off[block::DEC_GLOB]..][..DEC_GLOB_DIM].copy_from_slice(&dec_glob);
    out[off[block::DEC_CODE]..][..DEC_CODE_DIM].copy_from_slice(&dec_code);
    for k in 0..pkt.linear.o_dim.min(V8_MAX_CLASSES) {
        out[off[block::W_BASE] + k * width..][..p_eff].copy_from_slice(&pkt.linear.w[k][..p_eff]);
        out[off[block::B_BASE] + k] = pkt.linear.b[k];
    }
    let h = &pkt.header;
    let misc: [f32; MISC_LEN] = [
        h.resid_sd,
        h.z_max,
        p_eff as f32,
        h.k_eff as f32,
        h.x_cols_raw as f32,
        task as f32,
        h.rank_use as f32,
        h.knot_level as f32,
        if normalized_only { 1.0 } else { 0.0 },
        packet.len() as f32,
        card.y_mean,
        card.y_sd,
        card.classes.len() as f32,
        h.log2_nfit_norm,
        h.p1_prior,
        h.inv_nu,
    ];
    out[off[block::MISC]..][..MISC_LEN].copy_from_slice(&misc);

    let mut at = off[block::HEAD];
    put_unit_rows(&mut out[at..at + n_fit * width], &x_unit_fit, n_fit, width);
    at += n_fit * width;
    out[at..at + n_fit].copy_from_slice(&y_codes_fit);
    at += n_fit;
    if n_qry > 0 {
        let x_unit_qry = card.forward_x(&columns(x_qry, n_qry, p));
        put_unit_rows(&mut out[at..at + n_qry * width], &x_unit_qry, n_qry, width);
        at += n_qry * width;
        for (i, &v) in y_qry.iter().enumerate() {
            out[at + i] = card.forward_y_one(v as f64).map_or(-1.0, |ys| ys as f32);
        }
    }
    Ok(V8Episode {
        packet,
        floats: out,
    })
}

fn put_unit_rows(dst: &mut [f32], cols: &[Vec<f32>], n: usize, width: usize) {
    for (slot, col) in cols.iter().enumerate().take(width) {
        for (i, &v) in col.iter().enumerate().take(n) {
            dst[i * width + slot] = v;
        }
    }
}

/// Number of f32 elements `dope_ffi_v8_episode` writes.
#[no_mangle]
pub extern "C" fn dope_ffi_v8_len(p: u32, n_fit: u32, n_qry: u32) -> u64 {
    total(p as usize, n_fit as usize, n_qry as usize) as u64
}

/// Offset of block `block` (see [`block`]) for `p` input columns, or u64::MAX.
#[no_mangle]
pub extern "C" fn dope_ffi_v8_offset(block: u32, p: u32) -> u64 {
    let off = offsets(p_cap(p as usize));
    off.get(block as usize).map_or(u64::MAX, |&v| v as u64)
}

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

use dope_core::evidence::{
    extract, ColumnMeta as EvidenceMeta, EvidenceInput, EvidenceTier, COL_DIM, EDGE_DIM,
    GLOB_DIM as EV_GLOB_DIM,
};
use dope_core::marginal::column_meta::META_FIELDS;
use dope_core::single_encoding::v8::{
    build_decoder_inputs, fit_dse8_packet, fit_unit_meta_v2, unpack_dse8_packet, Dse8ColumnMeta,
    Dse8FitOptions, ScaleCardV8, DEC_CODE_DIM, DEC_COL_DIM, DEC_EDGE_DIM, DEC_GLOB_DIM, MARG_CONT,
    V8_MAX_CLASSES, V8_MAX_COLS, V8_MAX_RANK,
};
use dope_core::single_encoding::KNOT_COUNT;

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

/// Target in the packet's coordinates: regression standardized by the card (clamped to +-6;
/// `normalized_only` only clamps), classes as their index in the packet's class list, -1 for a
/// class the packet does not carry (masked, never another class).
fn target_code(v: f32, card: &ScaleCardV8) -> f32 {
    if card.task_id == 0 {
        let z = if card.normalized_only {
            v
        } else {
            (v - card.y_mean) / card.y_sd.max(1e-5)
        };
        z.clamp(-6.0, 6.0)
    } else {
        card.classes
            .binary_search(&(v.round() as i32))
            .map_or(-1.0, |k| k as f32)
    }
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
    let shape_ok = p > 0
        && p <= u16::MAX as usize
        && n_fit >= 4
        && task <= 2
        && tier <= 2
        && budget > 0
        && budget <= MAX_BUDGET as usize
        && x_fit.len() == n_fit * p
        && y_fit.len() == n_fit
        && x_qry.len() == n_qry * p
        && y_qry.len() == n_qry;
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

    let opts = Dse8FitOptions {
        task_id: task,
        budget_bytes: budget,
        normalized_only,
        col_meta: Some(&dse8_meta),
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
    let y_codes_fit: Vec<f32> = y_fit.iter().map(|&v| target_code(v, &card)).collect();

    // Evidence on the packet's active columns and the FIT rows whose target the packet carries.
    let keep: Vec<usize> = (0..n_fit)
        .filter(|&i| y_codes_fit[i] >= 0.0 || task == 0)
        .collect();
    if keep.len() < 4 {
        return Err(Code::Target);
    }
    let ev_unit: Vec<Vec<f32>> = x_unit_fit
        .iter()
        .map(|col| keep.iter().map(|&i| col[i]).collect())
        .collect();
    let ev_raw: Vec<Vec<f64>> = card
        .col_indices
        .iter()
        .map(|&j| keep.iter().map(|&i| x_raw[j][i]).collect())
        .collect();
    let ev_y: Vec<f64> = keep.iter().map(|&i| y_codes_fit[i] as f64).collect();
    let ev_meta: Vec<EvidenceMeta> = card
        .col_indices
        .iter()
        .map(|&j| EvidenceMeta::from(&dse8_meta[j]))
        .collect();
    let tier_enum = match tier {
        0 => EvidenceTier::T0,
        1 => EvidenceTier::T1,
        _ => EvidenceTier::T2,
    };
    let n_classes = match task {
        0 => 1,
        1 => 2,
        _ => card.classes.len().clamp(2, V8_MAX_CLASSES),
    };
    let ev = extract(&EvidenceInput {
        x_unit: &ev_unit,
        x_raw: Some(&ev_raw),
        y: &ev_y,
        task_id: task,
        n_classes,
        col_meta: Some(&ev_meta),
        packet_probe: None,
        tier: tier_enum,
        target_free: false,
        seed,
    })
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
            out[at + i] = if v.is_finite() {
                target_code(v, &card)
            } else {
                -1.0
            };
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

/// # Safety
/// Same buffer contract as `dope_ffi_dse7_episode`; `out_f32_cap` must be at least
/// `dope_ffi_v8_len(p, n_fit, n_qry)`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn dope_ffi_v8_episode(
    x_fit: *const f32,
    n_fit: u32,
    p: u32,
    y_fit: *const f32,
    x_qry: *const f32,
    n_qry: u32,
    y_qry: *const f32,
    task: u8,
    budget: u32,
    normalized_only: u8,
    tier: u8,
    seed: u64,
    out_f32: *mut f32,
    out_f32_cap: u64,
    out_packet: *mut u8,
    out_packet_cap: u32,
    out_packet_len: *mut u32,
) -> i32 {
    let qry_missing = n_qry > 0 && (x_qry.is_null() || y_qry.is_null());
    if x_fit.is_null()
        || y_fit.is_null()
        || qry_missing
        || out_f32.is_null()
        || out_packet.is_null()
        || out_packet_len.is_null()
    {
        return Code::NullPointer as i32;
    }
    let (n_fit, p, n_qry) = (n_fit as usize, p as usize, n_qry as usize);
    let (Some(cells), Some(qry_cells)) = (n_fit.checked_mul(p), n_qry.checked_mul(p)) else {
        return Code::Shape as i32;
    };
    if (out_f32_cap as usize) < total(p, n_fit, n_qry) {
        return Code::Capacity as i32;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: lengths are the caller's contract stated above.
        let xf = unsafe { std::slice::from_raw_parts(x_fit, cells) };
        let yf = unsafe { std::slice::from_raw_parts(y_fit, n_fit) };
        let (xq, yq): (&[f32], &[f32]) = if n_qry == 0 {
            (&[], &[])
        } else {
            unsafe {
                (
                    std::slice::from_raw_parts(x_qry, qry_cells),
                    std::slice::from_raw_parts(y_qry, n_qry),
                )
            }
        };
        v8_episode(
            xf,
            n_fit,
            p,
            yf,
            xq,
            n_qry,
            yq,
            task,
            budget as usize,
            normalized_only != 0,
            tier,
            seed,
        )
    }));
    match result {
        Err(_) => Code::Panic as i32,
        Ok(Err(code)) => code as i32,
        Ok(Ok(ep)) => {
            if ep.packet.len() > out_packet_cap as usize {
                return Code::Capacity as i32;
            }
            // SAFETY: capacities were checked against the written lengths.
            unsafe {
                std::ptr::copy_nonoverlapping(ep.floats.as_ptr(), out_f32, ep.floats.len());
                std::ptr::copy_nonoverlapping(ep.packet.as_ptr(), out_packet, ep.packet.len());
                *out_packet_len = ep.packet.len() as u32;
            }
            Code::Ok as i32
        }
    }
}

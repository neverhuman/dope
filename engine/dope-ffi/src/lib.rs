//! C ABI of the per-episode, FIT-only native stages called by the training episode factory
//! (`research/levels/s3_episode_factory.py`).
//!
//! Every function is a pure function of its arguments. It keeps no global state and no cache,
//! and it is single-threaded apart from whatever `dope-core` does internally (workers set
//! `RAYON_NUM_THREADS=1`). Each episode carries a fresh random train/test split, so every
//! statistic is recomputed from that episode's FIT rows. Query rows are only mapped into the
//! packet's coordinates; they never influence the packet.
//!
//! Errors are integer codes ([`Code`]). They never carry a cell value, header or extremum.
//!
//! Buffers: `x_*` arrays are row-major `[n][p]` f32 (NaN = missing). Targets are f32:
//! regression values, binary 0/1, multiclass class ids. Outputs go to caller-owned buffers,
//! so nothing is allocated across the boundary.

use std::panic::{catch_unwind, AssertUnwindSafe};

use dope_core::marginal::column_meta::{
    fit_columns, ColumnDecl, Nc2ColumnMetaV2, UnitSpace, META_FIELDS,
};
use dope_core::single_encoding::{
    fit_dse7_packet, unpack_dse7_packet, COL_FEAT_DIM, KNOT_COUNT, MAX_CLASSES, MAX_COLS, MAX_RANK,
};

/// Bumped on any change of a signature or of [`layout`].
pub const ABI_VERSION: u32 = 2;
/// Largest packet budget accepted (the 64 KiB development frontier).
pub const MAX_BUDGET: u32 = 65_536;

/// Status codes returned by every `extern "C"` function.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Code {
    Ok = 0,
    NullPointer = 1,
    Shape = 2,
    Fit = 3,
    Capacity = 4,
    Panic = 5,
    Unpack = 6,
    Target = 7,
}

/// Offsets (in f32 elements) of the fixed head written by `dope_ffi_dse7_episode`. After the
/// head come `x_fit [n_fit][MAX_COLS]`, `y_fit [n_fit]`, `x_qry [n_qry][MAX_COLS]` and
/// `y_qry [n_qry]`, all in packet coordinates. Columns at or beyond `p_eff` are zero.
pub mod layout {
    use super::{COL_FEAT_DIM, KNOT_COUNT, MAX_CLASSES, MAX_COLS, MAX_RANK};
    pub const COL_FEATS: usize = 0;
    pub const Q_NL: usize = COL_FEATS + MAX_COLS * COL_FEAT_DIM;
    pub const GLOB: usize = Q_NL + MAX_COLS * 4;
    pub const KNOTS: usize = GLOB + super::GLOB_DIM;
    pub const LOADINGS: usize = KNOTS + MAX_COLS * KNOT_COUNT;
    pub const W_BASE: usize = LOADINGS + MAX_COLS * MAX_RANK;
    pub const B_BASE: usize = W_BASE + MAX_CLASSES * MAX_COLS;
    /// resid_sd, p_eff, k_eff, rank_use, knot_level, x_cols_raw, has_q_nl, z_max.
    pub const MISC: usize = B_BASE + MAX_CLASSES;
    pub const COL_MASK: usize = MISC + 8;
    pub const IS_CONST: usize = COL_MASK + MAX_COLS;
    /// Source column of each packet column, -1 when unused.
    pub const COL_INDEX: usize = IS_CONST + MAX_COLS;
    /// `Nc2ColumnMetaV2::to_f32s` of each packet column, from FIT raw values, in the packet's own
    /// unit space (the scale card's center and scale): `[MAX_COLS][META_FIELDS]`, zero when unused.
    pub const COL_META: usize = COL_INDEX + MAX_COLS;
    pub const HEAD: usize = COL_META + MAX_COLS * super::META_FIELDS;

    /// Total f32 elements for an episode of `n_fit` FIT and `n_qry` query rows.
    pub const fn total(n_fit: usize, n_qry: usize) -> usize {
        HEAD + (n_fit + n_qry) * (MAX_COLS + 1)
    }
}

const GLOB_DIM: usize = dope_core::single_encoding::GLOB_FEAT_DIM;

/// One native DSE7 episode: the wire packet plus every training tensor in packet coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Dse7Episode {
    pub packet: Vec<u8>,
    /// `layout::total(n_fit, n_qry)` floats.
    pub floats: Vec<f32>,
}

fn columns(x: &[f32], n: usize, p: usize) -> Vec<Vec<f64>> {
    (0..p)
        .map(|j| (0..n).map(|i| x[i * p + j] as f64).collect())
        .collect()
}

/// Map raw targets into the packet's target coordinates. Classes outside the packet's class list
/// become -1 (masked), never another class.
fn target_codes(y: &[f32], task: u8, y_mean: f32, y_sd: f32, classes: &[i32]) -> Vec<f32> {
    y.iter()
        .map(|&v| match task {
            0 => ((v - y_mean) / y_sd).clamp(-6.0, 6.0),
            _ => {
                // `classes` is sorted, so its index is the packet's class code (binary: 0/1).
                classes
                    .binary_search(&(v.round() as i32))
                    .map_or(-1.0, |k| k as f32)
            }
        })
        .collect()
}

fn put_rows(dst: &mut [f32], cols: &[Vec<f32>], n: usize) {
    for (slot, col) in cols.iter().enumerate().take(MAX_COLS) {
        for (i, &v) in col.iter().enumerate().take(n) {
            dst[i * MAX_COLS + slot] = v;
        }
    }
}

/// Fit the DSE7 packet on the FIT rows only, unpack it with the serving decoder, and map FIT and
/// query rows into packet coordinates.
#[allow(clippy::too_many_arguments)]
pub fn dse7_episode(
    x_fit: &[f32],
    n_fit: usize,
    p: usize,
    y_fit: &[f32],
    x_qry: &[f32],
    n_qry: usize,
    y_qry: &[f32],
    task: u8,
    budget: usize,
) -> Result<Dse7Episode, Code> {
    let shape_ok = p > 0
        && p <= u16::MAX as usize
        && n_fit >= 2
        && task <= 2
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
    let fit_cols = columns(x_fit, n_fit, p);
    let y_raw: Vec<f64> = y_fit.iter().map(|&v| v as f64).collect();
    let packet = fit_dse7_packet(&fit_cols, &y_raw, task, budget).map_err(|_| Code::Fit)?;
    let (pkt, header, card) = unpack_dse7_packet(&packet).map_err(|_| Code::Unpack)?;

    let mut out = vec![0.0f32; layout::total(n_fit, n_qry)];
    for c in 0..MAX_COLS {
        let row = &mut out[layout::COL_FEATS + c * COL_FEAT_DIM..][..COL_FEAT_DIM];
        row.copy_from_slice(&pkt.col_feats[c]);
        out[layout::Q_NL + c * 4..][..4].copy_from_slice(&pkt.q_nl[c]);
        out[layout::KNOTS + c * KNOT_COUNT..][..KNOT_COUNT].copy_from_slice(&pkt.knots[c]);
        out[layout::LOADINGS + c * MAX_RANK..][..MAX_RANK].copy_from_slice(&pkt.loadings[c]);
        out[layout::COL_MASK + c] = pkt.col_mask[c];
        out[layout::IS_CONST + c] = pkt.is_const[c];
        out[layout::COL_INDEX + c] = card.col_indices.get(c).map_or(-1.0, |&j| j as f32);
        if let Some(&j) = card.col_indices.get(c) {
            let unit = UnitSpace {
                center: card.center_active[c] as f64,
                scale: card.scale_active[c] as f64,
            };
            let meta = Nc2ColumnMetaV2::fit(&fit_cols[j], ColumnDecl::single(j), Some(unit));
            out[layout::COL_META + c * META_FIELDS..][..META_FIELDS]
                .copy_from_slice(&meta.to_f32s());
        }
    }
    out[layout::GLOB..][..GLOB_DIM].copy_from_slice(&pkt.glob_feats);
    for k in 0..MAX_CLASSES {
        out[layout::W_BASE + k * MAX_COLS..][..MAX_COLS].copy_from_slice(&pkt.w_base[k]);
    }
    out[layout::B_BASE..][..MAX_CLASSES].copy_from_slice(&pkt.b_base);
    let misc = [
        pkt.resid_sd,
        header.p_eff as f32,
        header.k_eff as f32,
        header.rank_use as f32,
        header.knot_level as f32,
        header.x_cols_raw as f32,
        if header.has_q_nl { 1.0 } else { 0.0 },
        header.z_max,
    ];
    out[layout::MISC..][..8].copy_from_slice(&misc);

    let classes = &card.classes;
    let mut at = layout::HEAD;
    put_rows(
        &mut out[at..at + n_fit * MAX_COLS],
        &card.forward_x(&fit_cols),
        n_fit,
    );
    at += n_fit * MAX_COLS;
    let y_fit_codes = target_codes(y_fit, task, card.y_mean, card.y_sd, classes);
    out[at..at + n_fit].copy_from_slice(&y_fit_codes);
    at += n_fit;
    if n_qry > 0 {
        let qry_cols = columns(x_qry, n_qry, p);
        put_rows(
            &mut out[at..at + n_qry * MAX_COLS],
            &card.forward_x(&qry_cols),
            n_qry,
        );
        at += n_qry * MAX_COLS;
        let y_codes = target_codes(y_qry, task, card.y_mean, card.y_sd, classes);
        out[at..at + n_qry].copy_from_slice(&y_codes);
    }
    Ok(Dse7Episode {
        packet,
        floats: out,
    })
}

/// ABI version of this library.
#[no_mangle]
pub extern "C" fn dope_ffi_abi_version() -> u32 {
    ABI_VERSION
}

/// Number of f32 elements `dope_ffi_dse7_episode` writes for the given row counts.
#[no_mangle]
pub extern "C" fn dope_ffi_dse7_len(n_fit: u32, n_qry: u32) -> u64 {
    layout::total(n_fit as usize, n_qry as usize) as u64
}

/// Offset of a named head block (0 COL_FEATS, 1 Q_NL, 2 GLOB, 3 KNOTS, 4 LOADINGS, 5 W_BASE,
/// 6 B_BASE, 7 MISC, 8 COL_MASK, 9 IS_CONST, 10 COL_INDEX, 11 COL_META, 12 HEAD), or u64::MAX.
#[no_mangle]
pub extern "C" fn dope_ffi_dse7_offset(block: u32) -> u64 {
    use layout::*;
    let all = [
        COL_FEATS, Q_NL, GLOB, KNOTS, LOADINGS, W_BASE, B_BASE, MISC, COL_MASK, IS_CONST,
        COL_INDEX, COL_META, HEAD,
    ];
    all.get(block as usize).map_or(u64::MAX, |&v| v as u64)
}

/// # Safety
/// Every pointer must be valid for the stated number of elements: `x_fit` `n_fit * p`, `y_fit`
/// `n_fit`, `x_qry` `n_qry * p` and `y_qry` `n_qry` (both may be null when `n_qry == 0`),
/// `out_f32` `out_f32_cap`, `out_packet` `out_packet_cap`, and `out_packet_len` one u32.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn dope_ffi_dse7_episode(
    x_fit: *const f32,
    n_fit: u32,
    p: u32,
    y_fit: *const f32,
    x_qry: *const f32,
    n_qry: u32,
    y_qry: *const f32,
    task: u8,
    budget: u32,
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
    let Some(cells) = n_fit.checked_mul(p) else {
        return Code::Shape as i32;
    };
    let Some(qry_cells) = n_qry.checked_mul(p) else {
        return Code::Shape as i32;
    };
    if (out_f32_cap as usize) < layout::total(n_fit, n_qry) {
        return Code::Capacity as i32;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
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
        dse7_episode(xf, n_fit, p, yf, xq, n_qry, yq, task, budget as usize)
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

/// `Nc2ColumnMetaV2` of every raw column from FIT rows alone, in each column's robust unit space
/// (FIT median and IQR / 1.349). Writes `[p][META_FIELDS]` floats, one row per column.
///
/// # Safety
/// `x_fit` must hold `n_fit * p` floats (row-major) and `out` `out_cap` floats.
#[no_mangle]
pub unsafe extern "C" fn dope_ffi_column_meta(
    x_fit: *const f32,
    n_fit: u32,
    p: u32,
    out: *mut f32,
    out_cap: u64,
) -> i32 {
    if x_fit.is_null() || out.is_null() {
        return Code::NullPointer as i32;
    }
    let (n_fit, p) = (n_fit as usize, p as usize);
    let Some(cells) = n_fit.checked_mul(p) else {
        return Code::Shape as i32;
    };
    if p == 0 || (out_cap as usize) < p * META_FIELDS {
        return Code::Capacity as i32;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: lengths are the caller's contract stated above.
        let xf = unsafe { std::slice::from_raw_parts(x_fit, cells) };
        fit_columns(&columns(xf, n_fit, p), None, None)
    }));
    match result {
        Err(_) => Code::Panic as i32,
        Ok(metas) => {
            let flat: Vec<f32> = metas.iter().flat_map(|m| m.to_f32s()).collect();
            // SAFETY: `out_cap >= p * META_FIELDS` was checked.
            unsafe { std::ptr::copy_nonoverlapping(flat.as_ptr(), out, flat.len()) };
            Code::Ok as i32
        }
    }
}

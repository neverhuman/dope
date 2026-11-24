//! C ABI entry points for V8 episodes (`dope_ffi_v8_episode`, `dope_ffi_v8_episode_reserved`,
//! `dope_ffi_v8_episode_prepared`), exact `V7Schema` wrapper sizing, and native midrank projection.

use crate::v8::{total, v8_episode_prepared};
use crate::{copy_input, Code};
use dope_core::auto::v8_budget::dpa_v8_wrapper_bytes_from_json;
use dope_core::single_encoding::v8::project_fit_midrank;

/// # Safety
/// Same buffer contract as `dope_ffi_dse7_episode`; `out_f32_cap` must be at least
/// `dope_ffi_v8_len(p, n_fit, n_qry)`. Each input is an aligned, initialized range within one
/// live allocation and stays immutable for the call. The output ranges are writable and do not
/// overlap any input or each other. `fit_row_ids` is null or `n_fit` distinct stable physical ids
/// of the FIT rows (ABI 4), see [`crate::v8::v8_episode_rows`].
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
    fit_row_ids: *const u64,
    out_f32: *mut f32,
    out_f32_cap: u64,
    out_packet: *mut u8,
    out_packet_cap: u32,
    out_packet_len: *mut u32,
) -> i32 {
    // SAFETY: identical pointer-validity contract to `dope_ffi_v8_episode_reserved`.
    unsafe {
        dope_ffi_v8_episode_reserved(
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
            0,
            out_f32,
            out_f32_cap,
            out_packet,
            out_packet_cap,
            out_packet_len,
        )
    }
}

/// # Safety
/// Identical pointer-validity contract to [`dope_ffi_v8_episode`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn dope_ffi_v8_episode_reserved(
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
    fit_row_ids: *const u64,
    code_tier: u8,
    reserve_dpa_wrapper: u8,
    out_f32: *mut f32,
    out_f32_cap: u64,
    out_packet: *mut u8,
    out_packet_cap: u32,
    out_packet_len: *mut u32,
) -> i32 {
    // SAFETY: identical pointer-validity contract to `dope_ffi_v8_episode_prepared`.
    unsafe {
        dope_ffi_v8_episode_prepared(
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
            out_f32,
            out_f32_cap,
            out_packet,
            out_packet_cap,
            out_packet_len,
        )
    }
}

/// # Safety
/// Identical pointer-validity contract to [`dope_ffi_v8_episode`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn dope_ffi_v8_episode_prepared(
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
    fit_row_ids: *const u64,
    code_tier: u8,
    reserve_dpa_wrapper: u8,
    wrapper_bytes_override: u32,
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
        // SAFETY: FIT pointers were checked for null; caller guarantees valid ranges.
        let (xf, yf) = unsafe { (copy_input(x_fit, cells)?, copy_input(y_fit, n_fit)?) };
        // SAFETY: query pointers checked when n_qry > 0; copy_input handles len == 0 safely.
        let (xq, yq) = unsafe { (copy_input(x_qry, qry_cells)?, copy_input(y_qry, n_qry)?) };
        // SAFETY: non-null id pointer covers n_fit u64 values by contract.
        let ids = (!fit_row_ids.is_null())
            .then(|| unsafe { copy_input(fit_row_ids, n_fit) })
            .transpose()?;
        v8_episode_prepared(
            &xf,
            n_fit,
            p,
            &yf,
            &xq,
            n_qry,
            &yq,
            task,
            budget as usize,
            normalized_only != 0,
            tier,
            seed,
            ids.as_deref(),
            code_tier,
            reserve_dpa_wrapper != 0,
            wrapper_bytes_override as usize,
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

/// Compute exact `DPA1` header + `V7Schema` wrapper bytes from a UTF-8 `DpaV8SchemaSpec` JSON slice.
/// Returns the wrapper byte count (`>= 13`) on success, or negative `Code` on error.
///
/// # Safety
/// `schema_json` must point to `schema_json_len` initialized immutable bytes.
#[no_mangle]
pub unsafe extern "C" fn dope_ffi_v8_schema_wrapper_bytes(
    schema_json: *const u8,
    schema_json_len: u32,
    anonymize: u8,
) -> i32 {
    if schema_json.is_null() || schema_json_len == 0 {
        return -(Code::NullPointer as i32);
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: caller guarantees `schema_json` covers `schema_json_len` valid bytes.
        let bytes = unsafe { copy_input(schema_json, schema_json_len as usize)? };
        let text = std::str::from_utf8(&bytes).map_err(|_| Code::Shape)?;
        dpa_v8_wrapper_bytes_from_json(text, anonymize != 0).map_err(|_| Code::Shape)
    }));
    match result {
        Err(_) => -(Code::Panic as i32),
        Ok(Err(code)) => -(code as i32),
        Ok(Ok(n)) => i32::try_from(n).unwrap_or(-(Code::Capacity as i32)),
    }
}

/// Project row-major `[n_eval][p]` packet-unit values through uncapped FIT columns `[n_fit][p]`
/// using the exact native `CdfIntervals::build` tie-group midrank normal scores.
///
/// # Safety
/// `x_fit` covers `n_fit * p` floats, `x_eval` and `out_z` cover `n_eval * p` floats (non-overlapping).
#[no_mangle]
pub unsafe extern "C" fn dope_ffi_project_fit_midrank(
    x_fit: *const f32,
    n_fit: u32,
    p: u32,
    x_eval: *const f32,
    n_eval: u32,
    out_z: *mut f32,
) -> i32 {
    if n_eval == 0 {
        return Code::Ok as i32;
    }
    if x_fit.is_null() || x_eval.is_null() || out_z.is_null() {
        return Code::NullPointer as i32;
    }
    let (n_fit, p, n_eval) = (n_fit as usize, p as usize, n_eval as usize);
    let (Some(fit_cells), Some(eval_cells)) = (n_fit.checked_mul(p), n_eval.checked_mul(p)) else {
        return Code::Shape as i32;
    };
    if p == 0 {
        return Code::Shape as i32;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: pointers were checked for null; caller guarantees valid initialized slices.
        let xf = unsafe { copy_input(x_fit, fit_cells)? };
        // SAFETY: pointers were checked for null; caller guarantees valid initialized slices.
        let xe = unsafe { copy_input(x_eval, eval_cells)? };
        let mut out = vec![0.0f32; eval_cells];
        let mut col_f = vec![0.0f32; n_fit];
        let mut col_e = vec![0.0f32; n_eval];
        for j in 0..p {
            for i in 0..n_fit {
                col_f[i] = xf[i * p + j];
            }
            for i in 0..n_eval {
                col_e[i] = xe[i * p + j];
            }
            let z = project_fit_midrank(&col_f, &col_e);
            for i in 0..n_eval {
                out[i * p + j] = z[i];
            }
        }
        Ok::<Vec<f32>, Code>(out)
    }));
    match result {
        Err(_) => Code::Panic as i32,
        Ok(Err(code)) => code as i32,
        Ok(Ok(out)) => {
            // SAFETY: caller guarantees `out_z` has capacity `eval_cells`.
            unsafe {
                std::ptr::copy_nonoverlapping(out.as_ptr(), out_z, out.len());
            }
            Code::Ok as i32
        }
    }
}

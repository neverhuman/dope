//! Shared `DOPEMAT1` binary table I/O helpers used by `dope_v7_emit` and `dope_v8_emit`.

use std::fs;
use std::path::{Path, PathBuf};

pub const DOPEMAT_MAGIC: &[u8; 8] = b"DOPEMAT1";

/// Read a `DOPEMAT1` binary table (`y_cols == 1`, target-last) into column-major `(x, y)`.
pub fn read_dopemat(path: &Path) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    let raw = fs::read(path).map_err(|e| format!("read_mat_failed: {e}"))?;
    if raw.len() < 24 || &raw[..8] != DOPEMAT_MAGIC {
        return Err("invalid_dopemat_header".into());
    }
    let rows = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
    let x_cols = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
    let y_cols = u32::from_le_bytes(raw[16..20].try_into().unwrap()) as usize;
    if y_cols != 1 || x_cols == 0 || rows == 0 {
        return Err("invalid_dopemat_dims".into());
    }
    let stride = x_cols + 1;
    let expected = 24usize
        .checked_add(
            rows.checked_mul(stride)
                .and_then(|v| v.checked_mul(4))
                .ok_or("mat_overflow")?,
        )
        .ok_or("mat_overflow")?;
    if raw.len() != expected {
        return Err("dopemat_length_mismatch".into());
    }
    let mut x = vec![vec![0.0f64; rows]; x_cols];
    let mut y = vec![0.0f64; rows];
    let data = &raw[24..];
    for (r, yr) in y.iter_mut().enumerate().take(rows) {
        let base = r * stride * 4;
        for (c, xc) in x.iter_mut().enumerate().take(x_cols) {
            let off = base + c * 4;
            xc[r] = f32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as f64;
        }
        let y_off = base + x_cols * 4;
        *yr = f32::from_le_bytes(data[y_off..y_off + 4].try_into().unwrap()) as f64;
    }
    Ok((x, y))
}

/// Write `bytes` atomically via a `.part` sibling and rename.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| format!("mkdir_failed: {e}"))?;
        }
    }
    let mut part = path.as_os_str().to_os_string();
    part.push(".part");
    let part_path = PathBuf::from(part);
    fs::write(&part_path, bytes).map_err(|e| format!("write_failed: {e}"))?;
    fs::rename(&part_path, path).map_err(|e| format!("rename_failed: {e}"))?;
    Ok(())
}

/// Write a row-major synthetic table whose final column is the target.
pub fn write_dopemat(path: &Path, rows: usize, x_cols: usize, body: &[f32]) -> Result<(), String> {
    let mut buf = Vec::with_capacity(24 + body.len() * 4);
    buf.extend_from_slice(DOPEMAT_MAGIC);
    buf.extend_from_slice(&(rows as u32).to_le_bytes());
    buf.extend_from_slice(&(x_cols as u32).to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    for &v in body {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    write_atomic(path, &buf)
}

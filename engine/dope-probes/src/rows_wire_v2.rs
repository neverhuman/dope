//! Overflow-free row-subset wire comparator (`RWW2`).
//!
//! Replaces the historical `RWW1` prototype that clipped features to `[-6, 6]`, multiplied by 32,
//! cast to `i8` (wrapping values beyond `±4`), and artificially capped `ultra` at 88% of rows.
//! `RWW2` encodes each column with a checked affine `i16` range (`[-32767, 32767]`, preserving
//! missing values via `i16::MIN`), stores targets in `f64`, and binary-searches the largest row
//! subset whose brotli-11 wire payload fits strictly within `budget` bytes (up to 100% of rows).

use std::io::{Read, Write};

use crate::{Error, Result};

/// Binary magic header for the overflow-free `rows-wire-v2` packet format.
pub const MAGIC: &[u8; 4] = b"RWW2";
const MISSING_I16: i16 = i16::MIN;
const MAX_QUANT_I16: f64 = 32_767.0;
const MAX_PACKET_BYTES: usize = 64 * 1024 * 1024;
const MAX_DECODED_BYTES: usize = 128 * 1024 * 1024;

/// Decoded `RWW2` row-subset payload in caller-normalized coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct RowsWireV2 {
    pub rows: usize,
    pub cols: usize,
    /// Row-major `rows * cols` feature matrix (`NaN` marks missing cells).
    pub x: Vec<f64>,
    /// Length-`rows` target vector.
    pub y: Vec<f64>,
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % bound
    }
}

fn encode_subset(
    x: &[f64],
    y: &[f64],
    cols: usize,
    indices: &[usize],
    lo: &[f64],
    span: &[f64],
) -> Result<Vec<u8>> {
    let k_rows = indices.len();
    let mut raw = Vec::with_capacity(8 + cols * 16 + k_rows * cols * 2 + k_rows * 8);
    raw.extend_from_slice(&(k_rows as u32).to_le_bytes());
    raw.extend_from_slice(&(cols as u32).to_le_bytes());
    for c in 0..cols {
        raw.extend_from_slice(&lo[c].to_le_bytes());
        raw.extend_from_slice(&span[c].to_le_bytes());
    }
    for &r in indices {
        let base = r * cols;
        for c in 0..cols {
            let v = x[base + c];
            let q = if !v.is_finite() {
                MISSING_I16
            } else if span[c] == 0.0 {
                0i16
            } else {
                let norm = ((v - lo[c]) / span[c]).clamp(0.0, 1.0);
                let scaled = (norm * 2.0 - 1.0) * MAX_QUANT_I16;
                (scaled.round() as i32).clamp(-32_767, 32_767) as i16
            };
            raw.extend_from_slice(&q.to_le_bytes());
        }
    }
    for &r in indices {
        raw.extend_from_slice(&y[r].to_le_bytes());
    }
    let mut compressed = Vec::from(*MAGIC);
    {
        let mut enc = brotli::CompressorWriter::new(&mut compressed, 4096, 11, 22);
        enc.write_all(&raw)
            .map_err(|_| Error::msg("rows-wire-v2 brotli compression failed"))?;
    }
    Ok(compressed)
}

/// Pack the largest deterministic row subset (up to 100% of `rows`) that fits strictly within
/// `budget` bytes. Returns an error only if input shapes are invalid or `budget` cannot hold even
/// a single row.
pub fn pack(
    x: &[f64],
    y: &[f64],
    rows: usize,
    cols: usize,
    budget: usize,
    seed: u64,
) -> Result<Vec<u8>> {
    if rows == 0 || cols == 0 || x.len() != rows * cols || y.len() != rows {
        return Err(Error::msg("rows-wire-v2 invalid table dimensions"));
    }
    if !(16..=MAX_PACKET_BYTES).contains(&budget) {
        return Err(Error::msg("rows-wire-v2 invalid byte budget"));
    }
    let mut lo = vec![f64::INFINITY; cols];
    let mut hi = vec![f64::NEG_INFINITY; cols];
    for r in 0..rows {
        let base = r * cols;
        for c in 0..cols {
            let v = x[base + c];
            if v.is_finite() {
                lo[c] = lo[c].min(v);
                hi[c] = hi[c].max(v);
            }
        }
    }
    let mut span = vec![0.0; cols];
    for c in 0..cols {
        if !lo[c].is_finite() || !hi[c].is_finite() {
            lo[c] = 0.0;
            span[c] = 0.0;
        } else {
            span[c] = (hi[c] - lo[c]).max(0.0);
        }
    }
    let mut perm: Vec<usize> = (0..rows).collect();
    let mut rng = SplitMix64::new(seed ^ 0x5257_5732);
    for i in (1..rows).rev() {
        let j = rng.below(i + 1);
        perm.swap(i, j);
    }
    let mut low = 1usize;
    let mut high = rows;
    let mut best: Option<Vec<u8>> = None;
    while low <= high {
        let mid = low + (high - low) / 2;
        let pkt = encode_subset(x, y, cols, &perm[..mid], &lo, &span)?;
        if pkt.len() <= budget {
            best = Some(pkt);
            low = mid + 1;
        } else {
            if mid == 1 {
                break;
            }
            high = mid - 1;
        }
    }
    best.ok_or_else(|| Error::msg("rows-wire-v2 budget cannot fit a single row"))
}

/// Decode an `RWW2` packet into its stored rows without resampling.
pub fn unpack(bytes: &[u8]) -> Result<RowsWireV2> {
    if bytes.len() < 4 || bytes.len() > MAX_PACKET_BYTES || &bytes[..4] != MAGIC {
        return Err(Error::msg("rows-wire-v2 invalid header"));
    }
    let mut raw = Vec::new();
    let mut dec = brotli::Decompressor::new(&bytes[4..], 4096);
    let mut buf = [0u8; 8192];
    loop {
        let n = dec
            .read(&mut buf)
            .map_err(|_| Error::msg("rows-wire-v2 brotli decode failed"))?;
        if n == 0 {
            break;
        }
        if raw.len().saturating_add(n) > MAX_DECODED_BYTES {
            return Err(Error::msg("rows-wire-v2 decoded payload exceeds bound"));
        }
        raw.extend_from_slice(&buf[..n]);
    }
    if raw.len() < 8 {
        return Err(Error::msg("rows-wire-v2 truncated header"));
    }
    let rows = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
    let expected = 8usize
        .checked_add(
            cols.checked_mul(16)
                .ok_or_else(|| Error::msg("rows-wire-v2 overflow"))?,
        )
        .and_then(|v| {
            rows.checked_mul(cols)
                .and_then(|rc| rc.checked_mul(2))
                .and_then(|b| v.checked_add(b))
        })
        .and_then(|v| rows.checked_mul(8).and_then(|b| v.checked_add(b)))
        .ok_or_else(|| Error::msg("rows-wire-v2 size overflow"))?;
    if rows == 0 || cols == 0 || raw.len() != expected {
        return Err(Error::msg("rows-wire-v2 payload length mismatch"));
    }
    let mut off = 8usize;
    let mut lo = Vec::with_capacity(cols);
    let mut span = Vec::with_capacity(cols);
    for _ in 0..cols {
        lo.push(f64::from_le_bytes(raw[off..off + 8].try_into().unwrap()));
        span.push(f64::from_le_bytes(
            raw[off + 8..off + 16].try_into().unwrap(),
        ));
        off += 16;
    }
    let mut x = Vec::with_capacity(rows * cols);
    for _ in 0..rows {
        for c in 0..cols {
            let q = i16::from_le_bytes(raw[off..off + 2].try_into().unwrap());
            off += 2;
            if q == MISSING_I16 {
                x.push(f64::NAN);
            } else if span[c] == 0.0 {
                x.push(lo[c]);
            } else {
                let norm = ((q as f64 / MAX_QUANT_I16) + 1.0) * 0.5;
                x.push(lo[c] + norm * span[c]);
            }
        }
    }
    let mut y = Vec::with_capacity(rows);
    for _ in 0..rows {
        y.push(f64::from_le_bytes(raw[off..off + 8].try_into().unwrap()));
        off += 8;
    }
    Ok(RowsWireV2 { rows, cols, x, y })
}

/// Resample `out_rows` rows deterministically with replacement from an unpacked `RWW2` table.
pub fn sample_rows(table: &RowsWireV2, out_rows: usize, seed: u64) -> RowsWireV2 {
    let mut rng = SplitMix64::new(seed ^ 0x5257_5341);
    let mut x = Vec::with_capacity(out_rows * table.cols);
    let mut y = Vec::with_capacity(out_rows);
    for _ in 0..out_rows {
        let r = rng.below(table.rows);
        x.extend_from_slice(&table.x[r * table.cols..(r + 1) * table.cols]);
        y.push(table.y[r]);
    }
    RowsWireV2 {
        rows: out_rows,
        cols: table.cols,
        x,
        y,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rww2_round_trips_extreme_values_without_int8_wrap_or_88pct_cap() {
        let rows = 100usize;
        let cols = 3usize;
        let mut x = Vec::with_capacity(rows * cols);
        let mut y = Vec::with_capacity(rows);
        for r in 0..rows {
            let v0 = -10.0 + 20.0 * (r as f64 / (rows - 1) as f64);
            let v1 = if r % 11 == 0 {
                f64::NAN
            } else {
                100.0 - 2.0 * r as f64
            };
            let v2 = 42.0;
            x.extend_from_slice(&[v0, v1, v2]);
            y.push(0.5 * v0 - 0.1 * r as f64);
        }
        // Generous budget must store 100% of rows (no artificial 88% cap).
        let pkt = pack(&x, &y, rows, cols, 65_536, 7).unwrap();
        assert!(pkt.len() <= 65_536);
        let decoded = unpack(&pkt).unwrap();
        assert_eq!(decoded.rows, rows);
        assert_eq!(decoded.cols, cols);
        // Verify extremes (-10 and +10, +100 and -98) did not wrap sign.
        let min0 = decoded
            .x
            .iter()
            .step_by(cols)
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max0 = decoded
            .x
            .iter()
            .step_by(cols)
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((min0 - (-10.0)).abs() < 1e-3, "min0={min0}");
        assert!((max0 - 10.0).abs() < 1e-3, "max0={max0}");
        let missing_count = (0..rows)
            .filter(|&r| decoded.x[r * cols + 1].is_nan())
            .count();
        assert_eq!(missing_count, (0..rows).filter(|&r| r % 11 == 0).count());
        let sampled = sample_rows(&decoded, 400, 99);
        assert_eq!(sampled.rows, 400);
        assert_eq!(sampled.x.len(), 1200);
    }
}

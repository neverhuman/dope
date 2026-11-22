//! Portable, counter-based randomness.
//!
//! Every draw is a pure function of `(seed, stream, index)`, so sampling is
//! bit-reproducible across thread counts and identical on every device: the
//! random inputs are produced here and then moved to the compute device.

use crate::stats::norm_ppf;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

/// Default 128-bit namespace for DOPE v2 stateless row-keyed RNG.
pub const DEFAULT_NAMESPACE_128: [u8; 16] = *b"dope.rowkey.v2\0\0";

pub const PURPOSE_COPULA_EPS: u32 = 1;
pub const PURPOSE_REGIME: u32 = 2;
pub const PURPOSE_MISSING: u32 = 3;
pub const PURPOSE_NOISE: u32 = 4;
pub const PURPOSE_LABEL: u32 = 5;
pub const PURPOSE_CATF: u32 = 6;
pub const PURPOSE_RULE: u32 = 7;

/// Stateless cryptographic draw keyed strictly by:
/// `SHA256(domain || namespace_128 || seed || table_id || row_id || group_id || purpose || draw_idx)`.
///
/// Intentionally excludes any packet digest so Common Random Numbers (`T25`) hold across
/// candidate packets during closed-loop compilation and paired evaluations.
#[allow(clippy::too_many_arguments)]
pub fn row_keyed_u64(
    domain: &str,
    namespace_128: &[u8; 16],
    seed: u64,
    table_id: u64,
    row_id: u64,
    group_id: u64,
    purpose: u32,
    draw_idx: u32,
) -> u64 {
    let mut h = Sha256::new();
    h.update((domain.len() as u32).to_le_bytes());
    h.update(domain.as_bytes());
    h.update(namespace_128);
    h.update(seed.to_le_bytes());
    h.update(table_id.to_le_bytes());
    h.update(row_id.to_le_bytes());
    h.update(group_id.to_le_bytes());
    h.update(purpose.to_le_bytes());
    h.update(draw_idx.to_le_bytes());
    let digest = h.finalize();
    u64::from_le_bytes(digest[..8].try_into().unwrap_or([0; 8]))
}

#[allow(clippy::too_many_arguments)]
pub fn row_keyed_uniform(
    namespace_128: &[u8; 16],
    seed: u64,
    table_id: u64,
    row_id: u64,
    group_id: u64,
    purpose: u32,
    draw_idx: u32,
) -> f64 {
    let bits = row_keyed_u64(
        "DOPE_ROW_KEYED_RNG_V2",
        namespace_128,
        seed,
        table_id,
        row_id,
        group_id,
        purpose,
        draw_idx,
    );
    ((bits >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
}

#[allow(clippy::too_many_arguments)]
pub fn row_keyed_normal(
    namespace_128: &[u8; 16],
    seed: u64,
    table_id: u64,
    row_id: u64,
    group_id: u64,
    purpose: u32,
    draw_idx: u32,
) -> f64 {
    norm_ppf(row_keyed_uniform(
        namespace_128,
        seed,
        table_id,
        row_id,
        group_id,
        purpose,
        draw_idx,
    ))
}

#[inline]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[inline]
pub fn hash3(seed: u64, stream: u64, index: u64) -> u64 {
    splitmix64(splitmix64(seed ^ splitmix64(stream.wrapping_add(0x51_7cc1_b727_220a))) ^ index)
}

/// Uniform on the open interval (0, 1).
#[inline]
pub fn uniform(seed: u64, stream: u64, index: u64) -> f64 {
    ((hash3(seed, stream, index) >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
}

/// Standard normal by inversion (exact quantile, no rejection).
#[inline]
pub fn normal(seed: u64, stream: u64, index: u64) -> f64 {
    norm_ppf(uniform(seed, stream, index))
}

/// A row-major `[rows, cols]` block of standard normals.
/// Column `c` uses stream `stream_base + c`; row `r` uses index `row0 + r`.
pub fn normal_block(seed: u64, stream_base: u64, row0: u64, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    out.par_chunks_mut(cols.max(1))
        .enumerate()
        .for_each(|(r, row)| {
            for (c, v) in row.iter_mut().enumerate() {
                *v = normal(seed, stream_base + c as u64, row0 + r as u64) as f32;
            }
        });
    out
}

/// A row-major `[rows, cols]` block of uniforms in (0, 1).
pub fn uniform_block(seed: u64, stream_base: u64, row0: u64, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    out.par_chunks_mut(cols.max(1))
        .enumerate()
        .for_each(|(r, row)| {
            for (c, v) in row.iter_mut().enumerate() {
                *v = uniform(seed, stream_base + c as u64, row0 + r as u64) as f32;
            }
        });
    out
}

/// Small sequential generator for fitting (initialisation, shuffles, splits).
#[derive(Clone, Debug)]
pub struct Rng {
    seed: u64,
    counter: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { seed, counter: 0 }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.counter += 1;
        hash3(self.seed, 0xF17, self.counter)
    }
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }
    pub fn normal(&mut self) -> f64 {
        norm_ppf(self.uniform())
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.uniform() * n as f64) as usize % n.max(1)
    }
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normals_are_standard() {
        let v = normal_block(7, 0, 0, 20_000, 1);
        let n = v.len() as f64;
        let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
        let var = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }

    #[test]
    fn draws_do_not_depend_on_blocking() {
        let whole = normal_block(3, 5, 0, 10, 3);
        let tail = normal_block(3, 5, 4, 6, 3);
        assert_eq!(&whole[12..], &tail[..]);
    }
}

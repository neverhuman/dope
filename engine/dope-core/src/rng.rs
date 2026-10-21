//! Portable, counter-based randomness.
//!
//! Every draw is a pure function of `(seed, stream, index)`, so sampling is
//! bit-reproducible across thread counts and identical on every device: the
//! random inputs are produced here and then moved to the compute device.

use crate::stats::norm_ppf;
use rayon::prelude::*;

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

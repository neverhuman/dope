//! Pure row-local counter-seeded PRNG for `DSE8` v1 (`dope_core::single_encoding::v8`).
//!
//! Guarantees exact `0.0` diff between decoding an isolated row `r` (`N = 1`) and decoding
//! row `r` inside a bulk batch of `N = 100_000` rows, and matches Python `v8_packet.py`
//! bit-for-bit via simple SplitMix64 counter mixing + exact Box-Muller transform.

pub const STREAM_REGIME: u64 = 0;
pub const STREAM_FACTOR: u64 = 1;
pub const STREAM_NOISE: u64 = 2;
pub const STREAM_ATOM: u64 = 3;
pub const STREAM_MISSING: u64 = 4;
pub const STREAM_TARGET: u64 = 5;
pub const STREAM_STUDENT_T: u64 = 6;

/// Stateless SplitMix64 counter bijective hash keyed by `(seed, row_idx, stream_id, dim)`.
#[inline]
pub fn counter_u64(seed: u64, row_idx: u64, stream_id: u64, dim: u64) -> u64 {
    let mut x = seed
        .wrapping_add(row_idx.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(stream_id.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(dim.wrapping_mul(0x94D0_49BB_1331_11EB));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Uniform float in `(0, 1)` (open interval, strictly safe for `ln` and inverse CDF).
#[inline]
pub fn counter_uniform(seed: u64, row_idx: u64, stream_id: u64, dim: u64) -> f64 {
    let bits = counter_u64(seed, row_idx, stream_id, dim) >> 11; // 53 uniform bits
    (bits as f64 + 0.5) * (1.0 / ((1u64 << 53) as f64))
}

/// Standard normal N(0, 1) via exact Box-Muller on two deterministic counter draws.
#[inline]
pub fn counter_normal(seed: u64, row_idx: u64, stream_id: u64, dim: u64) -> f64 {
    let u1 = counter_uniform(seed, row_idx, stream_id, dim.wrapping_mul(2));
    let u2 = counter_uniform(
        seed,
        row_idx,
        stream_id,
        dim.wrapping_mul(2).wrapping_add(1),
    );
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos()
}

/// Optional Student-t heavy-tail radial scale factor `sqrt(nu / chi2_nu)` via Wilson-Hilferty
/// cube-normal transform (`inv_nu = 1 / nu in [0, 0.45]`). Returns `1.0` when `inv_nu <= 1e-4`.
#[inline]
pub fn counter_student_t_scale(seed: u64, row_idx: u64, inv_nu: f32) -> f32 {
    if inv_nu <= 1e-4 {
        return 1.0;
    }
    let nu = (1.0 / inv_nu as f64).clamp(2.2, 500.0);
    let g = counter_normal(seed, row_idx, STREAM_STUDENT_T, 0);
    let term = 1.0 - 2.0 / (9.0 * nu) + g * (2.0 / (9.0 * nu)).sqrt();
    let chi2_over_nu = term.max(0.20).powi(3);
    (1.0 / chi2_over_nu).sqrt().clamp(0.4, 3.5) as f32
}

/// Sample categorical index `0..probs.len()` from normalized/unnormalized positive weights.
#[inline]
pub fn sample_categorical(probs: &[f64], u: f64) -> usize {
    let sum: f64 = probs.iter().map(|&p| p.max(0.0)).sum();
    if sum <= 1e-12 || probs.is_empty() {
        return 0;
    }
    let target = u.clamp(0.0, 1.0 - 1e-15) * sum;
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p.max(0.0);
        if target <= acc {
            return i;
        }
    }
    probs.len() - 1
}

/// Softmax probabilities over `logits[..k]`, followed by categorical draw using `u in (0, 1)`.
#[inline]
pub fn sample_softmax_categorical(logits: &[f32], u: f64) -> usize {
    if logits.is_empty() {
        return 0;
    }
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = [0.0f64; 16];
    let k = logits.len().min(16);
    for (c, slot) in probs.iter_mut().enumerate().take(k) {
        *slot = ((logits[c] - max_l).clamp(-20.0, 0.0) as f64).exp();
    }
    sample_categorical(&probs[..k], u)
}

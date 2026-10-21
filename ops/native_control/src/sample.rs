//! Port of literal_worker._ControlRng and _control_sample, without fast-math.
use crate::{decode::Model, sys::check};

pub struct Rng(pub u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    pub fn index(&mut self, n: u64) -> u64 {
        check(n > 0);
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x % n
    }
}
/// Partial expansion and final half-even correction used by CPython math.fsum.
/// Inputs are at most64 finite products, never a naive running sum/FMA.
pub fn fsum(input: &[f64]) -> f64 {
    check(input.len() <= 64);
    let mut partials = [0.0f64; 64];
    let mut n = 0;
    for &value in input {
        let mut x = value;
        let mut i = 0;
        for j in 0..n {
            let mut y = partials[j];
            if libm::fabs(x) < libm::fabs(y) {
                core::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        n = i;
        check(x.is_finite());
        if x != 0.0 {
            check(n < 64);
            partials[n] = x;
            n += 1;
        }
    }
    let mut hi = 0.0;
    let mut lo = 0.0;
    if n > 0 {
        n -= 1;
        hi = partials[n];
        while n > 0 {
            let x = hi;
            n -= 1;
            let y = partials[n];
            hi = x + y;
            lo = y - (hi - x);
            if lo != 0.0 {
                break;
            }
        }
        if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
            let y = lo * 2.0;
            let x = hi + y;
            if y == x - hi {
                hi = x;
            }
        }
    }
    hi
}
pub fn draw(model: &Model, values: &[f64], rows: usize, seed: u64, out: &mut [u8]) -> usize {
    check(rows > 0 && rows <= 1_000_000 && rows * model.columns <= 1_000_000);
    let size = rows * model.columns * 8;
    check(size <= out.len());
    let mut rng = Rng::new(seed);
    let mut pos = 0;
    for _ in 0..rows {
        let mut sample = [0.0f64; 64];
        if model.method == 1 {
            let row = rng.index(model.rows.rows as u64) as usize;
            let start = model.rows.start + row * model.columns;
            sample[..model.columns].copy_from_slice(&values[start..start + model.columns]);
        } else {
            let mut independent = [0.0f64; 64];
            let mut i = 0;
            while i < model.columns {
                let u = (rng.index((1u64 << 53) - 1) + 1) as f64 / (1u64 << 53) as f64;
                // Preserve Python's left-associative multiply/divide, not a reassociation.
                let angle = (2.0 * core::f64::consts::PI)
                    * (rng.index((1u64 << 53) - 1) + 1) as f64
                    / (1u64 << 53) as f64;
                let radius = libm::sqrt(-2.0 * crate::host_math::log(u));
                independent[i] = radius * crate::host_math::cos(angle);
                if i + 1 < model.columns {
                    independent[i + 1] = radius * crate::host_math::sin(angle);
                }
                i += 2;
            }
            for i in 0..model.columns {
                let mut products = [0.0; 64];
                let start = model.factor.start + i * (i + 1) / 2;
                for j in 0..=i {
                    products[j] = values[start + j] * independent[j];
                }
                let latent = fsum(&products[..i + 1]);
                let position = (0.5 + 0.5 * crate::host_math::erf(latent / libm::sqrt(2.0)))
                    * (model.count - 1) as f64;
                check(position.is_finite() && position >= 0.0 && position <= model.count as f64);
                let lower = (position as usize).min(model.count - 1);
                let upper = (lower + 1).min(model.count - 1);
                let weight = position - lower as f64;
                let q = model.quantiles.start + i * model.count;
                sample[i] = (1.0 - weight) * values[q + lower] + weight * values[q + upper];
            }
        }
        for value in &sample[..model.columns] {
            check(value.is_finite());
            out[pos..pos + 8].copy_from_slice(&value.to_le_bytes());
            pos += 8;
        }
    }
    check(pos == size);
    size
}

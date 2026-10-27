//! Interpolation, symmetric Jacobi eigensolver, and scalar helpers for `prepare::np`.

use super::np::{pairwise_sum, sorted};

/// `np.interp(x, xp, fp)` for a single finite `x` (xp increasing).
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x.is_nan() {
        return x;
    }
    if x < xp[0] {
        return fp[0];
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    if n == 1 {
        return fp[0];
    }
    // largest j with xp[j] <= x
    let mut j = 0usize;
    {
        let (mut lo, mut hi) = (0usize, n);
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if xp[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if xp[lo] <= x {
            j = lo;
        }
    }
    if j == n - 1 {
        return fp[j];
    }
    if xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let mut r = slope * (x - xp[j]) + fp[j];
    if r.is_nan() {
        r = slope * (x - xp[j + 1]) + fp[j + 1];
        if r.is_nan() && fp[j] == fp[j + 1] {
            r = fp[j];
        }
    }
    r
}

/// Eigenvalues (ascending) of a symmetric `p×p` row-major matrix, cyclic Jacobi.
pub fn eigvalsh_sym(a: &[f64], p: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    if p == 0 {
        return Vec::new();
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        let mut diag = 0.0;
        for i in 0..p {
            diag += m[i * p + i] * m[i * p + i];
            for j in 0..p {
                if i != j {
                    off += m[i * p + j] * m[i * p + j];
                }
            }
        }
        if off <= 1e-30 * diag.max(1e-300) || off == 0.0 {
            break;
        }
        for pi in 0..p {
            for qi in (pi + 1)..p {
                let apq = m[pi * p + qi];
                if apq == 0.0 {
                    continue;
                }
                let app = m[pi * p + pi];
                let aqq = m[qi * p + qi];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..p {
                    let akp = m[k * p + pi];
                    let akq = m[k * p + qi];
                    m[k * p + pi] = c * akp - s * akq;
                    m[k * p + qi] = s * akp + c * akq;
                }
                for k in 0..p {
                    let apk = m[pi * p + k];
                    let aqk = m[qi * p + k];
                    m[pi * p + k] = c * apk - s * aqk;
                    m[qi * p + k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut ev: Vec<f64> = (0..p).map(|i| m[i * p + i]).collect();
    ev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ev
}

/// `descriptors.squash`.
#[inline]
pub fn squash(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

/// `np.clip(x, lo, hi)` (NaN propagates).
#[inline]
pub fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    if x.is_nan() {
        x
    } else {
        x.max(lo).min(hi)
    }
}

/// `np.nan_to_num(x, nan=0, posinf=0, neginf=0)`.
#[inline]
pub fn finite_or_zero(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// `np.nan_to_num` for float32 with all non-finite → 0.
#[inline]
pub fn finite_or_zero32(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// `descriptors.gini_coefficient`.
pub fn gini_coefficient(values: &[f64]) -> f64 {
    let arr: Vec<f64> = values
        .iter()
        .map(|v| v.abs())
        .filter(|v| v.is_finite())
        .collect();
    if arr.len() <= 1 {
        return 0.0;
    }
    let total = pairwise_sum(&arr);
    if total <= 1e-12 {
        return 0.0;
    }
    let s = sorted(&arr);
    let n = s.len();
    let prod: Vec<f64> = s
        .iter()
        .enumerate()
        .map(|(i, v)| (i + 1) as f64 * v)
        .collect();
    let gini = (2.0 * pairwise_sum(&prod)) / (n as f64 * total) - (n as f64 + 1.0) / n as f64;
    clip(gini, 0.0, 1.0)
}

/// numpy `max` reduction (NaN propagates).
pub fn np_max(a: &[f64]) -> f64 {
    let mut m = a[0];
    for &v in &a[1..] {
        if v.is_nan() || m.is_nan() {
            m = f64::NAN;
        } else if v > m {
            m = v;
        }
    }
    m
}

/// `np.round(x, 6)`.
#[inline]
pub fn round6(x: f64) -> f64 {
    (x * 1e6).round_ties_even() / 1e6
}

/// `np.linspace(start, stop, num)` (endpoint=True).
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    let div = num - 1;
    let delta = stop - start;
    let mut y: Vec<f64> = (0..num).map(|i| i as f64).collect();
    if div > 0 {
        let step = delta / div as f64;
        if step == 0.0 {
            for v in &mut y {
                *v = (*v / div as f64) * delta;
            }
        } else {
            for v in &mut y {
                *v *= step;
            }
        }
    } else {
        for v in &mut y {
            *v *= delta;
        }
    }
    for v in &mut y {
        *v += start;
    }
    if num > 1 {
        y[num - 1] = stop;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prepare::np::{aheapsort, argsort};

    #[test]
    fn argsort_matches_stable_without_ties() {
        let a: Vec<f64> = (0..100).map(|i| ((i * 37) % 101) as f64).collect();
        let o = argsort(&a);
        for w in o.windows(2) {
            assert!(a[w[0]] <= a[w[1]]);
        }
    }

    #[test]
    fn eig_identity() {
        let ev = eigvalsh_sym(&[2.0, 1.0, 1.0, 2.0], 2);
        assert!((ev[0] - 1.0).abs() < 1e-14 && (ev[1] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn heapsort_sorts() {
        let a = [5.0, 1.0, 4.0, 2.0, 3.0, 0.0];
        let mut idx: Vec<usize> = (0..a.len()).collect();
        aheapsort(&|i: usize, j: usize| a[i] < a[j], &mut idx);
        assert_eq!(idx, vec![5, 1, 3, 4, 2, 0]);
    }
}

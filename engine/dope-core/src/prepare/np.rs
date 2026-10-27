//! numpy 1.26-exact numerical helpers used by the MLC-v3 preparation port.
//!
//! The Python reference (`research/levels/{scaling,descriptors}.py`) is
//! written against numpy. To make the Rust port bit-identical (or within a
//! few ULP before the final `float32` cast) these helpers mirror numpy's
//! summation order, quantile interpolation and sort tie-breaking:
//!
//! * 1-D `np.sum`/`np.mean`/`np.std` use numpy's pairwise summation
//!   (`pairwise_sum_DOUBLE`, unroll 8, block 128).
//! * `axis=0` reductions of a C-contiguous `(n, p)` array accumulate row by
//!   row (sequential) when `p > 1`; with `p == 1` the axis is contiguous and
//!   numpy falls back to pairwise summation.
//! * `np.quantile(..., method="linear")` uses `(n-1)*q` virtual indices and
//!   numpy's two-sided `_lerp`.
//! * `np.argsort` (default `kind="quicksort"`) is numpy's introsort
//!   (`aquicksort_` + `aheapsort_`, insertion sort below 16 elements). Tie
//!   order therefore matches numpy on hosts WITHOUT AVX-512 (where numpy 1.26
//!   dispatches to x86-simd-sort instead and ties may be ordered differently).
//! * `np.linalg.eigvalsh` is replaced by a cyclic Jacobi solver; eigenvalues
//!   agree with LAPACK to ~1e-15 relative, far below the `float32` outputs.

/// numpy `pairwise_sum_DOUBLE`.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.mean` of a 1-D float64 array (NaN for empty input, like numpy).
pub fn mean1d(a: &[f64]) -> f64 {
    pairwise_sum(a) / a.len() as f64
}

/// `np.std` (ddof=0) of a 1-D float64 array.
pub fn std1d(a: &[f64]) -> f64 {
    let m = mean1d(a);
    let sq: Vec<f64> = a
        .iter()
        .map(|&v| {
            let d = v - m;
            d * d
        })
        .collect();
    (pairwise_sum(&sq) / a.len() as f64).sqrt()
}

/// `np.sum(x, axis=0)` for a row-major `(n, p)` array.
pub fn col_sum(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    if p == 1 {
        return vec![pairwise_sum(&x[..n])];
    }
    let mut out = vec![0.0; p];
    for i in 0..n {
        let row = &x[i * p..(i + 1) * p];
        for j in 0..p {
            out[j] += row[j];
        }
    }
    out
}

/// `np.mean(x, axis=0)` for a row-major `(n, p)` array.
pub fn col_mean(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    col_sum(x, n, p).into_iter().map(|s| s / n as f64).collect()
}

/// `np.std(x, axis=0)` (ddof=0) for a row-major `(n, p)` array.
pub fn col_std(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let m = col_mean(x, n, p);
    let mut sq = vec![0.0; n * p];
    for i in 0..n {
        for j in 0..p {
            let d = x[i * p + j] - m[j];
            sq[i * p + j] = d * d;
        }
    }
    col_sum(&sq, n, p)
        .into_iter()
        .map(|s| (s / n as f64).sqrt())
        .collect()
}

/// `np.sum(x, axis=1)` for a row-major `(n, p)` array (contiguous rows → pairwise).
pub fn row_sum(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    (0..n)
        .map(|i| pairwise_sum(&x[i * p..(i + 1) * p]))
        .collect()
}

/// numpy's float `less` (NaNs sort last).
#[inline]
fn np_less(a: f64, b: f64) -> bool {
    a < b || (b.is_nan() && !a.is_nan())
}

/// Sort ascending with numpy's NaN-last ordering (values only).
pub fn sorted(a: &[f64]) -> Vec<f64> {
    let mut v = a.to_vec();
    v.sort_by(|x, y| {
        if np_less(*x, *y) {
            std::cmp::Ordering::Less
        } else if np_less(*y, *x) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    v
}

/// numpy `_lerp(a, b, t)`.
#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// `np.quantile(a, q)` (method="linear") given the ascending-sorted data.
/// Returns NaN if the data contains NaN (numpy propagates the trailing NaN).
pub fn quantile_sorted(s: &[f64], q: f64) -> f64 {
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    if s[n - 1].is_nan() {
        return f64::NAN;
    }
    let vi = (n as f64 - 1.0) * q;
    let (prev, next) = if vi >= n as f64 - 1.0 {
        (n - 1, n - 1)
    } else if vi < 0.0 {
        (0, 0)
    } else {
        let f = vi.floor() as usize;
        (f, f + 1)
    };
    let gamma = vi - vi.floor();
    lerp(s[prev], s[next], gamma)
}

/// `np.quantile(a, qs)` (method="linear").
pub fn quantiles(a: &[f64], qs: &[f64]) -> Vec<f64> {
    let s = sorted(a);
    qs.iter().map(|&q| quantile_sorted(&s, q)).collect()
}

/// `np.median` of a 1-D array.
pub fn median(a: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return f64::NAN;
    }
    let s = sorted(a);
    if s[n - 1].is_nan() {
        return f64::NAN;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

fn msb(mut n: usize) -> i32 {
    let mut depth = 0;
    n >>= 1;
    while n > 0 {
        depth += 1;
        n >>= 1;
    }
    depth
}

fn aheapsort<F: Fn(usize, usize) -> bool>(less: &F, tosort: &mut [usize]) {
    let n0 = tosort.len();
    if n0 < 2 {
        return;
    }
    // 1-based indexing helper: a[k] == tosort[k-1]
    let mut n = n0;
    let mut l = n >> 1;
    while l > 0 {
        let tmp = tosort[l - 1];
        let mut i = l;
        let mut j = l << 1;
        while j <= n {
            if j < n && less(tosort[j - 1], tosort[j]) {
                j += 1;
            }
            if less(tmp, tosort[j - 1]) {
                tosort[i - 1] = tosort[j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[i - 1] = tmp;
        l -= 1;
    }
    while n > 1 {
        let tmp = tosort[n - 1];
        tosort[n - 1] = tosort[0];
        n -= 1;
        let mut i = 1;
        let mut j = 2;
        while j <= n {
            if j < n && less(tosort[j - 1], tosort[j]) {
                j += 1;
            }
            if less(tmp, tosort[j - 1]) {
                tosort[i - 1] = tosort[j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[i - 1] = tmp;
    }
}

/// numpy `aquicksort_` over element indices with a strict-weak `less(i, j)`
/// comparing the VALUES at indices `i`, `j`.
fn aquicksort<F: Fn(usize, usize) -> bool>(less: &F, num: usize) -> Vec<usize> {
    const SMALL_QUICKSORT: isize = 15;
    let mut ts: Vec<usize> = (0..num).collect();
    if num <= 1 {
        return ts;
    }
    let mut pl: isize = 0;
    let mut pr: isize = num as isize - 1;
    let mut stack: Vec<(isize, isize, i32)> = Vec::new();
    let mut cdepth = msb(num) * 2;
    loop {
        let mut do_insertion = true;
        if cdepth < 0 {
            aheapsort(less, &mut ts[pl as usize..=pr as usize]);
            do_insertion = false;
        } else {
            while (pr - pl) > SMALL_QUICKSORT {
                let pm = pl + ((pr - pl) >> 1);
                if less(ts[pm as usize], ts[pl as usize]) {
                    ts.swap(pm as usize, pl as usize);
                }
                if less(ts[pr as usize], ts[pm as usize]) {
                    ts.swap(pr as usize, pm as usize);
                }
                if less(ts[pm as usize], ts[pl as usize]) {
                    ts.swap(pm as usize, pl as usize);
                }
                let vp = ts[pm as usize];
                let mut pi = pl;
                let mut pj = pr - 1;
                ts.swap(pm as usize, pj as usize);
                loop {
                    loop {
                        pi += 1;
                        if !less(ts[pi as usize], vp) {
                            break;
                        }
                    }
                    loop {
                        pj -= 1;
                        if !less(vp, ts[pj as usize]) {
                            break;
                        }
                    }
                    if pi >= pj {
                        break;
                    }
                    ts.swap(pi as usize, pj as usize);
                }
                let pk = pr - 1;
                ts.swap(pi as usize, pk as usize);
                cdepth -= 1;
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth));
                    pl = pi + 1;
                }
            }
        }
        if do_insertion {
            let mut pi = pl + 1;
            while pi <= pr {
                let vi = ts[pi as usize];
                let mut pj = pi;
                let mut pk = pi - 1;
                while pj > pl && less(vi, ts[pk as usize]) {
                    ts[pj as usize] = ts[pk as usize];
                    pj -= 1;
                    pk -= 1;
                }
                ts[pj as usize] = vi;
                pi += 1;
            }
        }
        match stack.pop() {
            None => break,
            Some((l, r, d)) => {
                pl = l;
                pr = r;
                cdepth = d;
            }
        }
    }
    ts
}

/// `np.argsort(a)` for float64 (default kind, numpy 1.26 generic introsort).
pub fn argsort(a: &[f64]) -> Vec<usize> {
    aquicksort(&|i: usize, j: usize| np_less(a[i], a[j]), a.len())
}

/// `np.argsort(a)` for int64.
pub fn argsort_i64(a: &[i64]) -> Vec<usize> {
    aquicksort(&|i: usize, j: usize| a[i] < a[j], a.len())
}

/// `np.argsort(np.argsort(a)).astype(float64)` (ordinal ranks, numpy tie order).
pub fn ranks_double_argsort(a: &[f64]) -> Vec<f64> {
    let order = argsort(a);
    let mut r = vec![0.0; a.len()];
    for (rank, &idx) in order.iter().enumerate() {
        r[idx] = rank as f64;
    }
    r
}

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

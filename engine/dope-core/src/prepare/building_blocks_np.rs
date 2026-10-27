//! numpy-exact reductions and element functions shared by the prepare port.

use std::cmp::Ordering;

/// numpy's buffered reduction chunk (`NPY_BUFSIZE`).
const BUFSIZE: usize = 8192;
const PW_BLOCKSIZE: usize = 128;

fn pairwise(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= PW_BLOCKSIZE {
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
        pairwise(&a[..n2]) + pairwise(&a[n2..])
    }
}

/// `np.sum` of a contiguous 1-D float64 array.
pub fn sum(a: &[f64]) -> f64 {
    if a.len() <= BUFSIZE {
        return pairwise(a);
    }
    let mut res = 0.0;
    for chunk in a.chunks(BUFSIZE) {
        res += pairwise(chunk);
    }
    res
}

/// `np.mean` of a 1-D array (NaN when empty, like numpy).
pub fn mean(a: &[f64]) -> f64 {
    sum(a) / a.len() as f64
}

/// `np.std` (ddof = 0) of a 1-D array.
pub fn std(a: &[f64]) -> f64 {
    let m = mean(a);
    let sq: Vec<f64> = a
        .iter()
        .map(|&v| {
            let d = v - m;
            d * d
        })
        .collect();
    (sum(&sq) / a.len() as f64).sqrt()
}

/// `np.sum(x, axis=0)` of a C-contiguous `(n, k)` matrix.
pub fn sum_axis0(x: &[f64], n: usize, k: usize) -> Vec<f64> {
    debug_assert_eq!(x.len(), n * k);
    if k == 1 {
        return vec![sum(x)];
    }
    let mut out = vec![0.0f64; k];
    for i in 0..n {
        let row = &x[i * k..(i + 1) * k];
        for j in 0..k {
            out[j] += row[j];
        }
    }
    out
}

/// `np.mean(x, axis=0)`.
pub fn mean_axis0(x: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut s = sum_axis0(x, n, k);
    for v in &mut s {
        *v /= n as f64;
    }
    s
}

/// `np.std(x, axis=0)` (ddof = 0).
pub fn std_axis0(x: &[f64], n: usize, k: usize) -> Vec<f64> {
    let m = mean_axis0(x, n, k);
    let mut sq = vec![0.0f64; n * k];
    for i in 0..n {
        for j in 0..k {
            let d = x[i * k + j] - m[j];
            sq[i * k + j] = d * d;
        }
    }
    sum_axis0(&sq, n, k)
        .into_iter()
        .map(|s| (s / n as f64).sqrt())
        .collect()
}

/// `np.sum(x, axis=1)` of a C-contiguous `(n, k)` matrix.
pub fn sum_axis1(x: &[f64], n: usize, k: usize) -> Vec<f64> {
    (0..n).map(|i| sum(&x[i * k..(i + 1) * k])).collect()
}

/// `np.max` of a non-empty 1-D array (NaN propagates).
pub fn max(a: &[f64]) -> f64 {
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

/// `np.min` of a non-empty 1-D array (NaN propagates).
pub fn min(a: &[f64]) -> f64 {
    let mut m = a[0];
    for &v in &a[1..] {
        if v.is_nan() || m.is_nan() {
            m = f64::NAN;
        } else if v < m {
            m = v;
        }
    }
    m
}

/// `np.argmax` (first maximal index; first NaN wins).
pub fn argmax(a: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in a.iter().enumerate() {
        if a[best].is_nan() {
            return best;
        }
        if v.is_nan() || v > a[best] {
            best = i;
        }
    }
    best
}

/// `np.clip` of a scalar (`minimum(maximum(x, lo), hi)`).
pub fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    let a = if x.is_nan() || x >= lo { x } else { lo };
    if a.is_nan() || a <= hi {
        a
    } else {
        hi
    }
}

/// `np.maximum` of two scalars (NaN propagates).
pub fn maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

/// `np.nan_to_num(x, nan=0, posinf=0, neginf=0)`.
pub fn nan_to_zero(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Python builtin `max(a, b)` on floats (keeps `a` on ties / NaN `b`).
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

fn less(a: f64, b: f64) -> bool {
    a < b || (b.is_nan() && !a.is_nan())
}

fn aheapsort(v: &[f64], t: &mut [usize]) {
    let mut n = t.len();
    let a = |t: &[usize], i: usize| t[i - 1];
    let mut l = n >> 1;
    while l > 0 {
        let tmp = a(t, l);
        let (mut i, mut j) = (l, l << 1);
        while j <= n {
            if j < n && less(v[a(t, j)], v[a(t, j + 1)]) {
                j += 1;
            }
            if less(v[tmp], v[a(t, j)]) {
                t[i - 1] = a(t, j);
                i = j;
                j += j;
            } else {
                break;
            }
        }
        t[i - 1] = tmp;
        l -= 1;
    }
    while n > 1 {
        let tmp = a(t, n);
        t[n - 1] = a(t, 1);
        n -= 1;
        let (mut i, mut j) = (1usize, 2usize);
        while j <= n {
            if j < n && less(v[a(t, j)], v[a(t, j + 1)]) {
                j += 1;
            }
            if less(v[tmp], v[a(t, j)]) {
                t[i - 1] = a(t, j);
                i = j;
                j += j;
            } else {
                break;
            }
        }
        t[i - 1] = tmp;
    }
}

/// `np.argsort(v)` with the default (introsort, unstable) kind, including
/// numpy's tie ordering on hosts without AVX-512 dispatch.
pub fn argsort(v: &[f64]) -> Vec<usize> {
    let num = v.len();
    let mut t: Vec<usize> = (0..num).collect();
    if num <= 1 {
        return t;
    }
    let msb = (usize::BITS - 1 - num.leading_zeros()) as i64;
    let mut cdepth: i64 = msb * 2;
    let (mut pl, mut pr): (usize, usize) = (0, num - 1);
    let mut stack: Vec<(usize, usize, i64)> = Vec::new();
    loop {
        if cdepth < 0 {
            aheapsort(v, &mut t[pl..=pr]);
        } else {
            while pr - pl > 15 {
                let pm = pl + ((pr - pl) >> 1);
                if less(v[t[pm]], v[t[pl]]) {
                    t.swap(pm, pl);
                }
                if less(v[t[pr]], v[t[pm]]) {
                    t.swap(pr, pm);
                }
                if less(v[t[pm]], v[t[pl]]) {
                    t.swap(pm, pl);
                }
                let vp = v[t[pm]];
                let mut pi = pl;
                let mut pj = pr - 1;
                t.swap(pm, pj);
                loop {
                    pi += 1;
                    while less(v[t[pi]], vp) {
                        pi += 1;
                    }
                    pj -= 1;
                    while less(vp, v[t[pj]]) {
                        pj -= 1;
                    }
                    if pi >= pj {
                        break;
                    }
                    t.swap(pi, pj);
                }
                let pk = pr - 1;
                t.swap(pi, pk);
                cdepth -= 1;
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth));
                    pl = pi + 1;
                }
            }
            for pi in pl + 1..=pr {
                let vi = t[pi];
                let vp = v[vi];
                let mut pj = pi;
                while pj > pl && less(vp, v[t[pj - 1]]) {
                    t[pj] = t[pj - 1];
                    pj -= 1;
                }
                t[pj] = vi;
            }
        }
        match stack.pop() {
            Some((l, r, d)) => {
                pl = l;
                pr = r;
                cdepth = d;
            }
            None => break,
        }
    }
    t
}

/// `np.sort(v)` values (order of equal values is irrelevant).
pub fn sorted(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(|a, b| {
        if less(*a, *b) {
            Ordering::Less
        } else if less(*b, *a) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
    s
}

/// `np.sort(v)[::-1]`.
pub fn sorted_desc(v: &[f64]) -> Vec<f64> {
    let mut s = sorted(v);
    s.reverse();
    s
}

/// `np.quantile(v, q)` (method "linear") on an already sorted, NaN-free
/// 1-D array, reproducing numpy's `_lerp` rounding.
pub fn quantile_sorted(s: &[f64], q: f64) -> f64 {
    let n = s.len();
    let vi = (n as f64 - 1.0) * q;
    let (prev, next) = if vi >= n as f64 - 1.0 {
        (n - 1, n - 1)
    } else if vi < 0.0 {
        (0, 0)
    } else {
        let f = vi.floor();
        (f as usize, f as usize + 1)
    };
    let gamma = vi - vi.floor().max(0.0).min(n as f64 - 1.0);
    let gamma = if vi >= n as f64 - 1.0 || vi < 0.0 {
        vi - vi.floor()
    } else {
        gamma
    };
    let a = s[prev];
    let b = s[next];
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

/// numpy/glibc `asinh` (fdlibm `s_asinh.c`), built on libm `log`/`log1p`.
pub fn asinh(x: f64) -> f64 {
    const LN2: f64 = 6.93147180559945286227e-01;
    let hx = (x.to_bits() >> 32) as u32;
    let ix = hx & 0x7fff_ffff;
    if ix < 0x3e30_0000 {
        return x;
    }
    let w = if ix > 0x41b0_0000 {
        if ix >= 0x7ff0_0000 {
            return x + x;
        }
        x.abs().ln() + LN2
    } else {
        let xa = x.abs();
        if ix > 0x4000_0000 {
            (2.0 * xa + 1.0 / ((xa * xa + 1.0).sqrt() + xa)).ln()
        } else {
            let t = xa * xa;
            (xa + t / (1.0 + (1.0 + t).sqrt())).ln_1p()
        }
    };
    w.copysign(x)
}

/// `np.sign` of a scalar.
pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x.is_nan() {
        f64::NAN
    } else {
        0.0
    }
}

/// `np.round(x, 5)` (`rint(x * 1e5) / 1e5`).
pub fn round5(x: f64) -> f64 {
    (x * 100000.0).round_ties_even() / 100000.0
}

/// Singular values (descending) of a row-major `k × p` matrix via
/// one-sided Jacobi; matches LAPACK `gesdd` to ~1e-15 relative.
pub fn singular_values(m: &[f64], k: usize, p: usize) -> Vec<f64> {
    let (rows, cols, mut a) = if p <= k {
        (k, p, m.to_vec())
    } else {
        let mut t = vec![0.0; k * p];
        for i in 0..k {
            for j in 0..p {
                t[j * k + i] = m[i * p + j];
            }
        }
        (p, k, t)
    };
    let at = |a: &[f64], i: usize, j: usize| a[i * cols + j];
    for _sweep in 0..100 {
        let mut off = 0.0f64;
        for c1 in 0..cols {
            for c2 in c1 + 1..cols {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for r in 0..rows {
                    let x = at(&a, r, c1);
                    let y = at(&a, r, c2);
                    alpha += x * x;
                    beta += y * y;
                    gamma += x * y;
                }
                if gamma == 0.0 {
                    continue;
                }
                let denom = (alpha * beta).sqrt();
                if denom > 0.0 {
                    off = off.max(gamma.abs() / denom);
                }
                if gamma.abs() <= 1e-300 || gamma.abs() <= f64::EPSILON * denom {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let t = if zeta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for r in 0..rows {
                    let x = a[r * cols + c1];
                    let y = a[r * cols + c2];
                    a[r * cols + c1] = c * x - s * y;
                    a[r * cols + c2] = s * x + c * y;
                }
            }
        }
        if off <= f64::EPSILON {
            break;
        }
    }
    let mut sv: Vec<f64> = (0..cols)
        .map(|c| {
            let mut s = 0.0;
            for r in 0..rows {
                let v = at(&a, r, c);
                s += v * v;
            }
            s.sqrt()
        })
        .collect();
    sv.sort_by(|x, y| y.partial_cmp(x).unwrap_or(Ordering::Equal));
    sv
}

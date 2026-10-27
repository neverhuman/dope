//! Rust port of `research/levels/building_block_landmarks.py` (building-block
//! probes, row summaries and the multiclass/target simplex).
//!
//! Every reduction mirrors the numpy 1.26 kernel the Python reference uses so
//! the results are bit-identical where the libm functions agree (see [`np`]):
//! 1-D sums are numpy's pairwise sum, axis-0 sums of an `(n, k)` matrix are
//! sequential per column for `k >= 2` and pairwise for `k == 1`, `argsort` is
//! numpy's (unstable) introsort, quantiles use numpy's `_lerp`.
//!
//! Matrices are row-major `n × p` `f64` slices.

/// `LEVEL_RANK` of the Python reference; unknown levels rank as `ultra` (3).
pub fn level_rank(level: &str) -> u8 {
    match level {
        "low" | "small" => 0,
        "medium" => 1,
        "high" => 2,
        _ => 3,
    }
}

/// numpy-exact reductions and element functions shared by the prepare port.
pub mod np {
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
        // one-based heap indexing as in numpy's aheapsort_
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
        // numpy computes gamma = virtual - floor(virtual) before clamping
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
            // |x| < 2**-28
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
        // Work on the orientation with fewer columns: columns of A (rows×cols).
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
}

/// `_squash` on a scalar: non-finite → 0, else `v / (1 + |v|)`.
pub fn squash(v: f64) -> f64 {
    if !v.is_finite() {
        0.0
    } else {
        v / (1.0 + v.abs())
    }
}

/// `_squash` on an array (nan_to_num first, then `v / (1 + |v|)`).
pub fn squash_vec(v: &[f64]) -> Vec<f64> {
    v.iter()
        .map(|&x| {
            let a = np::nan_to_zero(x);
            a / (1.0 + a.abs())
        })
        .collect()
}

/// `_erf_vec` (Abramowitz–Stegun 7.1.26) on a scalar.
pub fn erf_as(x: f64) -> f64 {
    let sign = np::sign(x);
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-(ax * ax)).exp())
}

/// `_gini`.
pub fn gini(values: &[f64]) -> f64 {
    let arr: Vec<f64> = values
        .iter()
        .map(|v| v.abs())
        .filter(|v| v.is_finite())
        .collect();
    if arr.len() <= 1 || np::sum(&arr) <= 1e-12 {
        return 0.0;
    }
    let s = np::sorted(&arr);
    let n = s.len();
    let weighted: Vec<f64> = s
        .iter()
        .enumerate()
        .map(|(i, &v)| (i + 1) as f64 * v)
        .collect();
    np::clip(
        (2.0 * np::sum(&weighted)) / (n as f64 * np::sum(&s)) - (n as f64 + 1.0) / n as f64,
        0.0,
        1.0,
    )
}

/// `_safe_corr_vec`: |Pearson| of each column of `a` (`n × k`) with `y_z`.
pub fn safe_corr_vec(a: &[f64], n: usize, k: usize, y_z: &[f64]) -> Vec<f64> {
    if n * k == 0 || y_z.is_empty() {
        return vec![0.0; k.max(if k == 0 { 1 } else { 0 })];
    }
    let mu = np::mean_axis0(a, n, k);
    let mut a_c = vec![0.0f64; n * k];
    for i in 0..n {
        for j in 0..k {
            a_c[i * k + j] = a[i * k + j] - mu[j];
        }
    }
    let a_s = np::std_axis0(&a_c, n, k);
    let valid: Vec<usize> = (0..k).filter(|&j| a_s[j] > 1e-9).collect();
    let mut out = vec![0.0f64; k];
    if !valid.is_empty() {
        let kv = valid.len();
        let mut prod = vec![0.0f64; n * kv];
        for i in 0..n {
            for (c, &j) in valid.iter().enumerate() {
                prod[i * kv + c] = (a_c[i * k + j] / a_s[j]) * y_z[i];
            }
        }
        let m = np::mean_axis0(&prod, n, kv);
        for (c, &j) in valid.iter().enumerate() {
            out[j] = m[c].abs();
        }
    }
    out.into_iter().map(np::nan_to_zero).collect()
}

/// `_safe_pair_corr`: clipped Pearson correlation of two equal-length vectors.
pub fn safe_pair_corr(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || b.len() != a.len() {
        return 0.0;
    }
    let (sa, sb) = (np::std(a), np::std(b));
    if sa <= 1e-9 || sb <= 1e-9 {
        return 0.0;
    }
    let (ma, mb) = (np::mean(a), np::mean(b));
    let prod: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| ((x - ma) / sa) * ((y - mb) / sb))
        .collect();
    let c = np::mean(&prod);
    if c.is_finite() {
        np::clip(c, -1.0, 1.0)
    } else {
        0.0
    }
}

/// `compute_row_summary_features` (no missing mask): row-major `n × 8`
/// `[row_mean, row_std, row_l1, row_l2, row_min, row_max, row_skew, row_missing_frac]`.
pub fn compute_row_summary_features(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * 8];
    if n == 0 || p == 0 {
        return out;
    }
    let mut clean = vec![0.0f64; p];
    let mut tmp = vec![0.0f64; p];
    for i in 0..n {
        let row = &x[i * p..(i + 1) * p];
        let mut finite = 0usize;
        for j in 0..p {
            if row[j].is_finite() {
                finite += 1;
                clean[j] = row[j];
            } else {
                clean[j] = 0.0;
            }
        }
        let miss_frac = 1.0 - finite as f64 / p as f64;
        let r_mean = np::mean(&clean);
        for j in 0..p {
            let c = clean[j] - r_mean;
            tmp[j] = c * c;
        }
        let r_std = (np::mean(&tmp) + 1e-8).sqrt();
        for j in 0..p {
            tmp[j] = clean[j].abs();
        }
        let r_l1 = np::mean(&tmp);
        for j in 0..p {
            tmp[j] = clean[j] * clean[j];
        }
        let r_l2 = (np::mean(&tmp) + 1e-8).sqrt();
        let r_min = np::min(&clean);
        let r_max = np::max(&clean);
        for j in 0..p {
            tmp[j] = ((clean[j] - r_mean) / r_std).powf(3.0);
        }
        let r_skew = np::clip(np::mean(&tmp), -10.0, 10.0);
        out[i * 8..(i + 1) * 8]
            .copy_from_slice(&[r_mean, r_std, r_l1, r_l2, r_min, r_max, r_skew, miss_frac]);
    }
    out
}

/// Output of [`compute_building_block_landmarks`] (float32 like Python).
#[derive(Debug, Clone, PartialEq)]
pub struct BuildingBlockDescriptors {
    /// `_squash(bb_gains16)`.
    pub bb_gains16: [f32; 16],
    /// `_squash(row_corrs)`.
    pub row_summary_corrs8: [f32; 8],
    /// Per-column `[asinh, atan, erf, step, hinge, fract, xi]`, `max(p, 1)` rows.
    pub col_bb_probes7: Vec<[f32; 7]>,
    /// `_squash(het_and_tail8)`.
    pub het_and_tail8: [f32; 8],
    /// Multiclass simplex / target moments.
    pub multiclass_simplex20: [f32; 20],
    /// `[bb16 | row8 | contrasts16 | het_tail8 | mc20]`.
    pub building_block_vec68: [f32; 68],
}

impl BuildingBlockDescriptors {
    /// The all-zero block (Python's `n < 6 or p == 0` branch; also `mode=none`).
    pub fn zeroed(p: usize) -> Self {
        Self {
            bb_gains16: [0.0; 16],
            row_summary_corrs8: [0.0; 8],
            col_bb_probes7: vec![[0.0; 7]; p.max(1)],
            het_and_tail8: [0.0; 8],
            multiclass_simplex20: [0.0; 20],
            building_block_vec68: [0.0; 68],
        }
    }
}

fn to_f32<const N: usize>(v: &[f64]) -> [f32; N] {
    let mut out = [0.0f32; N];
    for (o, &x) in out.iter_mut().zip(v) {
        *o = x as f32;
    }
    out
}

/// Elementwise map of an `n × p` matrix.
fn map_mat(z: &[f64], f: impl Fn(f64) -> f64) -> Vec<f64> {
    z.iter().map(|&v| f(v)).collect()
}

/// Elementwise map with the column index.
fn map_mat_col(z: &[f64], p: usize, f: impl Fn(f64, usize) -> f64) -> Vec<f64> {
    z.iter().enumerate().map(|(i, &v)| f(v, i % p)).collect()
}

/// `compute_building_block_landmarks(X_train_scaled, y_train_scaled, level=level)`.
///
/// `x` is row-major `n × p`, `y` has length `n`.
pub fn compute_building_block_landmarks(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    level: &str,
) -> BuildingBlockDescriptors {
    assert_eq!(x.len(), n * p, "x must be row-major n×p");
    assert_eq!(y.len(), n, "y must have n rows");
    let lvl_r = level_rank(level);
    if n < 6 || p == 0 {
        return BuildingBlockDescriptors::zeroed(p);
    }

    // Standardize columns on D_train
    let c_mean = np::mean_axis0(x, n, p);
    let c_std = np::std_axis0(x, n, p);
    let mut z = vec![0.0f64; n * p];
    for i in 0..n {
        for j in 0..p {
            if c_std[j] > 1e-9 {
                z[i * p + j] = np::clip((x[i * p + j] - c_mean[j]) / c_std[j], -8.0, 8.0);
            }
        }
    }
    let z = z;

    let y_std = np::std(y);
    let y_z: Vec<f64> = if y_std > 1e-9 {
        let ym = np::mean(y);
        y.iter().map(|&v| (v - ym) / y_std).collect()
    } else {
        vec![0.0; n]
    };

    let corr = |a: &[f64], k: usize| safe_corr_vec(a, n, k, &y_z);
    let zeros_p = || vec![0.0f64; p];

    // 1. Per-column building-block probes
    let c_asinh = corr(&map_mat(&z, np::asinh), p);
    let c_atan = corr(&map_mat(&z, f64::atan), p);
    let sqrt2 = 2.0f64.sqrt();
    let c_erf = corr(&map_mat(&z, |v| erf_as(v / sqrt2)), p);
    let c_cbrt = corr(&map_mat(&z, |v| np::sign(v) * v.abs().powf(1.0 / 3.0)), p);
    let c_log1p = corr(&map_mat(&z, |v| np::sign(v) * v.abs().ln_1p()), p);
    let c_tanh = if lvl_r >= 1 {
        corr(&map_mat(&z, |v| (0.75 * v).tanh()), p)
    } else {
        zeros_p()
    };
    let c_laplace = if lvl_r >= 1 {
        corr(&map_mat(&z, |v| (-v.abs()).exp_m1()), p)
    } else {
        zeros_p()
    };

    let mut q = [
        vec![0.0f64; p],
        vec![0.0; p],
        vec![0.0; p],
        vec![0.0; p],
        vec![0.0; p],
    ];
    let mut col = vec![0.0f64; n];
    for j in 0..p {
        for i in 0..n {
            col[i] = z[i * p + j];
        }
        let s = np::sorted(&col);
        for (qi, &qq) in [0.10, 0.25, 0.50, 0.75, 0.90].iter().enumerate() {
            q[qi][j] = np::quantile_sorted(&s, qq);
        }
    }
    let [q10, q25, q50, q75, q90] = q;
    let c_clamp = if lvl_r >= 1 {
        corr(&map_mat_col(&z, p, |v, j| np::clip(v, q10[j], q90[j])), p)
    } else {
        zeros_p()
    };
    let c_step = if lvl_r >= 2 {
        corr(
            &map_mat_col(&z, p, |v, j| if v >= q50[j] { 1.0 } else { 0.0 }),
            p,
        )
    } else {
        zeros_p()
    };
    let c_hinge = if lvl_r >= 2 {
        corr(
            &map_mat_col(&z, p, |v, j| {
                np::maximum(0.0, v - q75[j]) - np::maximum(0.0, q25[j] - v)
            }),
            p,
        )
    } else {
        zeros_p()
    };
    let half_pi = 0.5 * std::f64::consts::PI;
    let c_sincos = if lvl_r >= 2 {
        let s = corr(&map_mat(&z, |v| (half_pi * v).sin()), p);
        let c = corr(&map_mat(&z, |v| (half_pi * v).cos()), p);
        s.iter().zip(&c).map(|(&a, &b)| np::maximum(a, b)).collect()
    } else {
        zeros_p()
    };
    let c_fract = if lvl_r >= 2 {
        corr(&map_mat(&z, |v| v - v.floor()), p)
    } else {
        zeros_p()
    };

    // Chatterjee xi rank-jump proxy
    let mut c_xi = zeros_p();
    if lvl_r >= 2 && y_std > 1e-9 {
        let order_y = np::argsort(y);
        let ranks_idx = np::argsort(&order_y.iter().map(|&v| v as f64).collect::<Vec<_>>());
        let y_ranks: Vec<f64> = ranks_idx.iter().map(|&v| v as f64).collect();
        let mut diffs = vec![0.0f64; (n - 1) * p];
        for j in 0..p {
            for i in 0..n {
                col[i] = z[i * p + j];
            }
            let order = np::argsort(&col);
            for i in 0..n - 1 {
                diffs[i * p + j] = (y_ranks[order[i + 1]] - y_ranks[order[i]]).abs();
            }
        }
        let d = np::mean_axis0(&diffs, n - 1, p);
        let denom = (n + 1).max(2) as f64;
        for j in 0..p {
            c_xi[j] = np::clip(1.0 - (3.0 * d[j]) / denom, 0.0, 1.0);
        }
    }

    // 2. Row-level summary features & their target correlations
    let row_feats = compute_row_summary_features(&z, n, p);
    let row_corrs = corr(&row_feats, 8);
    let row_gain = np::max(&row_corrs);

    // 3. Bivariate polar & normalized-ratio probes on top driver pairs
    let (mut hypot_gain, mut atan2_gain, mut norm_diff_gain) = (0.0f64, 0.0f64, 0.0f64);
    if lvl_r >= 2 && p >= 2 && y_std > 1e-9 {
        let base_score: Vec<f64> = (0..p)
            .map(|j| np::maximum(c_asinh[j], np::maximum(c_step[j], c_xi[j])))
            .collect();
        let mut order = np::argsort(&base_score);
        order.reverse();
        let top: Vec<usize> = order.into_iter().take(p.min(6)).collect();
        let mut pairs = Vec::new();
        for a in 0..top.len() {
            for b in a + 1..top.len() {
                pairs.push((top[a], top[b]));
            }
        }
        if !pairs.is_empty() {
            let m = pairs.len();
            let mut h = vec![0.0f64; n * m];
            let mut at = vec![0.0f64; n * m];
            let mut dd = vec![0.0f64; n * m];
            for i in 0..n {
                for (c, &(ia, ib)) in pairs.iter().enumerate() {
                    let zi = z[i * p + ia];
                    let zj = z[i * p + ib];
                    h[i * m + c] = zi.hypot(zj);
                    at[i * m + c] = zi.atan2(zj);
                    dd[i * m + c] = (zi - zj) / (1.0 + zi.abs() + zj.abs());
                }
            }
            hypot_gain = np::max(&corr(&h, m));
            atan2_gain = np::max(&corr(&at, m));
            norm_diff_gain = np::max(&corr(&dd, m));
        }
    }

    let gate = |ok: bool, v: f64| if ok { v } else { 0.0 };
    let bb_gains16: [f64; 16] = [
        np::max(&c_asinh),
        np::max(&c_atan),
        np::max(&c_erf),
        np::max(&c_cbrt),
        np::max(&c_log1p),
        row_gain,
        gate(lvl_r >= 1, np::max(&c_tanh)),
        gate(lvl_r >= 1, np::max(&c_laplace)),
        gate(lvl_r >= 1, np::max(&c_clamp)),
        gate(lvl_r >= 2, np::max(&c_step)),
        gate(lvl_r >= 2, np::max(&c_hinge)),
        gate(lvl_r >= 2, np::max(&c_sincos)),
        gate(lvl_r >= 2, np::max(&c_fract)),
        gate(lvl_r >= 2, hypot_gain),
        gate(lvl_r >= 3, atan2_gain),
        gate(lvl_r >= 3, norm_diff_gain),
    ];

    // 4. Conditional heteroscedasticity & tail co-dependence
    let mut het_and_tail8 = [0.0f64; 8];
    if lvl_r >= 1 && y_std > 1e-9 {
        let best_j = np::argmax(&c_asinh);
        let res_abs: Vec<f64> = (0..n)
            .map(|i| (y_z[i] - c_asinh[best_j] * z[i * p + best_j]).abs())
            .collect();
        let res_std = np::std(&res_abs);
        let het_corrs = if res_std > 1e-9 {
            let rm = np::mean(&res_abs);
            let den = res_std.max(1e-9);
            let target: Vec<f64> = res_abs.iter().map(|&r| (r - rm) / den).collect();
            safe_corr_vec(&z, n, p, &target)
        } else {
            zeros_p()
        };
        let s_het = np::sorted_desc(&het_corrs);
        het_and_tail8[0] = s_het[0];
        het_and_tail8[1] = np::mean(&s_het[..p.min(3)]);
        het_and_tail8[2] = np::mean(&het_corrs);
        het_and_tail8[3] = gini(&het_corrs);
        if lvl_r >= 2 {
            let ind = |f: &dyn Fn(f64) -> bool, v: &[f64]| -> Vec<f64> {
                v.iter().map(|&x| if f(x) { 1.0 } else { 0.0 }).collect()
            };
            let y_hi = ind(&|v| v > 1.0, &y_z);
            let y_lo = ind(&|v| v < -1.0, &y_z);
            let mh = np::mean(&y_hi);
            let ml = np::mean(&y_lo);
            let t_hi: Vec<f64> = y_hi.iter().map(|&v| v - mh).collect();
            let t_lo: Vec<f64> = y_lo.iter().map(|&v| v - ml).collect();
            let u_corrs = safe_corr_vec(&ind(&|v| v > 1.0, &z), n, p, &t_hi);
            let l_corrs = safe_corr_vec(&ind(&|v| v < -1.0, &z), n, p, &t_lo);
            het_and_tail8[4] = np::max(&u_corrs);
            het_and_tail8[5] = np::mean(&u_corrs);
            het_and_tail8[6] = np::max(&l_corrs);
            het_and_tail8[7] = np::mean(&l_corrs);
        }
    }

    // 5. Multiclass simplex, Fisher discriminant & target modality profile
    let mc_vec20 = compute_multiclass_and_target_simplex(&z, n, p, y, lvl_r);

    // 6. Contrasts among the building-block probes
    let c_lin = corr(&z, p);
    let lin_max = np::max(&c_lin);
    let denom = np::py_max(np::py_max(np::max(&bb_gains16), lin_max), 0.02);
    let g = &bb_gains16;
    let contrasts_raw: [f64; 16] = [
        (g[0] - lin_max) / denom,
        (g[1] - lin_max) / denom,
        (g[2] - lin_max) / denom,
        (g[3] - lin_max) / denom,
        (g[5] - lin_max) / denom,
        np::mean(&row_corrs) / denom,
        gini(&row_corrs),
        gate(lvl_r >= 1, (g[7] - g[6]) / denom),
        gate(lvl_r >= 1, (g[8] - lin_max) / denom),
        gate(lvl_r >= 2, (g[9] - g[2]) / denom),
        gate(lvl_r >= 2, (g[10] - lin_max) / denom),
        gate(lvl_r >= 2, (g[11] - lin_max) / denom),
        gate(lvl_r >= 2, (g[12] - lin_max) / denom),
        gate(lvl_r >= 2, (g[13] - lin_max) / denom),
        gate(lvl_r >= 3, (g[14] - g[13]) / denom),
        gate(lvl_r >= 3, (g[15] - lin_max) / denom),
    ];
    let bb_contrasts16: Vec<f64> = contrasts_raw
        .iter()
        .map(|&v| np::clip(v, -1.0, 1.0))
        .collect();

    let col_bb_probes7: Vec<[f32; 7]> = (0..p)
        .map(|j| {
            [
                c_asinh[j], c_atan[j], c_erf[j], c_step[j], c_hinge[j], c_fract[j], c_xi[j],
            ]
            .map(|v| np::nan_to_zero(v as f32 as f64) as f32)
        })
        .collect();

    let bb_sq = squash_vec(&bb_gains16);
    let row_sq = squash_vec(&row_corrs);
    let het_sq = squash_vec(&het_and_tail8);
    let mut vec68 = Vec::with_capacity(68);
    vec68.extend_from_slice(&bb_sq);
    vec68.extend_from_slice(&row_sq);
    vec68.extend_from_slice(&bb_contrasts16);
    vec68.extend_from_slice(&het_sq);
    vec68.extend_from_slice(&mc_vec20);
    let mut building_block_vec68 = [0.0f32; 68];
    for (o, &v) in building_block_vec68.iter_mut().zip(&vec68) {
        let f = v as f32;
        *o = if f.is_finite() { f } else { 0.0 };
    }

    BuildingBlockDescriptors {
        bb_gains16: to_f32(&bb_sq),
        row_summary_corrs8: to_f32(&row_sq),
        col_bb_probes7,
        het_and_tail8: to_f32(&het_sq),
        multiclass_simplex20: to_f32(&mc_vec20),
        building_block_vec68,
    }
}

/// `_compute_multiclass_and_target_simplex(Z, y, lvl_r)` on standardized `z`.
pub fn compute_multiclass_and_target_simplex(
    z: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    lvl_r: u8,
) -> [f64; 20] {
    let mut out = [0.0f64; 20];
    // np.unique(np.round(y, 5), return_counts=True)
    let rounded = np::sorted(&y.iter().map(|&v| np::round5(v)).collect::<Vec<_>>());
    let mut u_vals: Vec<f64> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    for &v in &rounded {
        // numpy's unique compares adjacent sorted values with `!=`
        if u_vals.last().is_some_and(|&last| last == v) {
            *counts.last_mut().expect("non-empty") += 1;
        } else {
            u_vals.push(v);
            counts.push(1);
        }
    }
    let k = u_vals.len();
    if k <= 1 {
        return out;
    }
    let is_discrete = 2 <= k && k <= 32usize.min((n / 3).max(2));
    out[0] = if is_discrete {
        (k.min(64) as f64).log2() / 6.0
    } else {
        0.0
    };
    if is_discrete {
        let priors: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();
        let terms: Vec<f64> = priors.iter().map(|&q| q * q.max(1e-12).ln()).collect();
        let ent = -np::sum(&terms) / (k as f64).ln().max(1e-9);
        out[1] = np::clip(ent, 0.0, 1.0);
        out[2] = np::min(&priors) / np::max(&priors).max(1e-9);
        out[3] = gini(&priors);
        if lvl_r >= 1 {
            let mut centroids = vec![0.0f64; k * p];
            for (c, &val) in u_vals.iter().enumerate() {
                let rows: Vec<usize> = (0..n).filter(|&i| (y[i] - val).abs() <= 1e-4).collect();
                if !rows.is_empty() {
                    let mut sub = Vec::with_capacity(rows.len() * p);
                    for &i in &rows {
                        sub.extend_from_slice(&z[i * p..(i + 1) * p]);
                    }
                    let m = np::mean_axis0(&sub, rows.len(), p);
                    centroids[c * p..(c + 1) * p].copy_from_slice(&m);
                }
            }
            let mut weighted = vec![0.0f64; k * p];
            for c in 0..k {
                for j in 0..p {
                    let v = centroids[c * p + j];
                    weighted[c * p + j] = priors[c] * (v * v);
                }
            }
            let between = np::sum_axis0(&weighted, k, p);
            let ratio: Vec<f64> = between
                .iter()
                .map(|&b| b / np::maximum(1.0 - b, 1e-4))
                .collect();
            let fisher = squash_vec(&ratio);
            let s_fish = np::sorted_desc(&fisher);
            let n_fish = 4.min(s_fish.len());
            out[4..(4 + n_fish)].copy_from_slice(&s_fish[..n_fish]);
            out[8] = np::mean(&fisher);
            out[9] = gini(&fisher);
            if lvl_r >= 2 && k >= 2 {
                let mut scaled = vec![0.0f64; k * p];
                for c in 0..k {
                    let sp = priors[c].sqrt();
                    for j in 0..p {
                        scaled[c * p + j] = sp * centroids[c * p + j];
                    }
                }
                let sv_sq = squash_vec(&np::singular_values(&scaled, k, p));
                let n_sv = 6.min(sv_sq.len());
                out[10..(10 + n_sv)].copy_from_slice(&sv_sq[..n_sv]);
            }
        }
    }
    // Target distribution moments
    let y_std = np::std(y).max(1e-9);
    let ym = np::mean(y);
    let y_z: Vec<f64> = y.iter().map(|&v| (v - ym) / y_std).collect();
    let m3: Vec<f64> = y_z.iter().map(|&v| v.powf(3.0)).collect();
    let m4: Vec<f64> = y_z.iter().map(|&v| v.powf(4.0)).collect();
    let ash: Vec<f64> = y_z
        .iter()
        .map(|&v| {
            let a = np::asinh(v);
            a * a
        })
        .collect();
    out[16] = squash(np::mean(&m3));
    out[17] = squash((np::mean(&m4) - 3.0) / 4.0);
    out[18] = squash(np::mean(&ash));
    let s = np::sorted(&y_z);
    let q10 = np::quantile_sorted(&s, 0.10);
    let q25 = np::quantile_sorted(&s, 0.25);
    let q75 = np::quantile_sorted(&s, 0.75);
    let q90 = np::quantile_sorted(&s, 0.90);
    out[19] = squash((q90 - q10) / (q75 - q25).max(1e-4) / 3.0);
    out.map(np::nan_to_zero)
}

#[cfg(test)]
mod tests {
    use super::np;

    #[test]
    fn argsort_ties_follow_numpy_introsort() {
        // np.argsort([1,0,1,0,...]) for 20 alternating values (numpy 1.26, no AVX-512)
        let v: Vec<f64> = (0..20).map(|i| (i % 2) as f64).collect();
        let a = np::argsort(&v);
        let mut sorted: Vec<f64> = a.iter().map(|&i| v[i]).collect();
        let check = sorted.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(sorted, check);
    }

    #[test]
    fn asinh_matches_reference_points() {
        for &x in &[0.0, 1e-30, 0.3, -1.7, 2.5, -40.0, 1e10] {
            let r = np::asinh(x);
            assert!((r - x.asinh()).abs() <= 4.0 * f64::EPSILON * r.abs().max(1.0));
        }
    }

    #[test]
    fn quantile_matches_numpy_linear() {
        let s = [1.0, 2.0, 4.0, 8.0];
        assert_eq!(np::quantile_sorted(&s, 0.5), 3.0);
        assert_eq!(np::quantile_sorted(&s, 1.0), 8.0);
        assert_eq!(np::quantile_sorted(&s, 0.0), 1.0);
    }
}

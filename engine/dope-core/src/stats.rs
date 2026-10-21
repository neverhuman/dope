//! Numerical building blocks: normal distribution, ranks, dependence
//! measures and small dense linear algebra (all f64, no external BLAS).

pub const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

#[inline]
pub fn norm_pdf(z: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * z * z).exp()
}

/// Standard normal CDF via the complementary error function (|err| < 1.2e-7).
pub fn norm_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

fn erfc(x: f64) -> f64 {
    // Numerical Recipes erfcc (Chebyshev), relative error < 1.2e-7.
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 {
        r
    } else {
        2.0 - r
    }
}

/// Standard normal quantile, Wichura's AS241 (PPND16, ~1e-16 accuracy).
pub fn norm_ppf(p: f64) -> f64 {
    let p = p.clamp(1e-300, 1.0 - 1e-16);
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180_625 - q * q;
        return q
            * (((((((r * 2_509.080_928_730_122_7 + 33_430.575_583_588_128) * r
                + 67_265.770_927_008_7)
                * r
                + 45_921.953_931_549_87)
                * r
                + 13_731.693_765_509_461)
                * r
                + 1_971.590_950_306_551_3)
                * r
                + 133.141_667_891_784_38)
                * r
                + 3.387_132_872_796_366_5)
            / (((((((r * 5_226.495_278_852_545 + 28_729.085_735_721_943) * r
                + 39_307.895_800_092_71)
                * r
                + 21_213.794_301_586_597)
                * r
                + 5_394.196_021_424_751)
                * r
                + 687.187_007_492_057_9)
                * r
                + 42.313_330_701_600_91)
                * r
                + 1.0);
    }
    let mut r = if q < 0.0 { p } else { 1.0 - p };
    r = (-r.ln()).sqrt();
    let val = if r <= 5.0 {
        r -= 1.6;
        (((((((r * 7.745_450_142_783_414e-4 + 0.022_723_844_989_269_184) * r
            + 0.241_780_725_177_450_6)
            * r
            + 1.270_458_252_452_368_4)
            * r
            + 3.647_848_324_763_204_5)
            * r
            + 5.769_497_221_460_691)
            * r
            + 4.630_337_846_156_546)
            * r
            + 1.423_437_110_749_683_5)
            / (((((((r * 1.050_750_071_644_416_9e-9 + 5.475_938_084_995_345e-4) * r
                + 0.015_198_666_563_616_457)
                * r
                + 0.148_103_976_427_480_08)
                * r
                + 0.689_767_334_985_1)
                * r
                + 1.676_384_830_183_803_8)
                * r
                + 2.053_191_626_637_759)
                * r
                + 1.0)
    } else {
        r -= 5.0;
        (((((((r * 2.010_334_399_292_288_1e-7 + 2.711_555_568_743_487_6e-5) * r
            + 0.001_242_660_947_388_078_4)
            * r
            + 0.026_532_189_526_576_124)
            * r
            + 0.296_560_571_828_504_9)
            * r
            + 1.784_826_539_917_291_3)
            * r
            + 5.463_784_911_164_114)
            * r
            + 6.657_904_643_501_103)
            / (((((((r * 2.044_263_103_389_939_7e-15 + 1.421_511_758_316_446e-7) * r
                + 1.846_318_317_510_054_8e-5)
                * r
                + 7.868_691_311_456_133e-4)
                * r
                + 0.014_875_361_290_850_615)
                * r
                + 0.136_929_880_922_735_8)
                * r
                + 0.599_832_206_555_888)
                * r
                + 1.0)
    };
    if q < 0.0 {
        -val
    } else {
        val
    }
}

/// Linear-interpolation quantile (type 7) of an ascending slice.
pub fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    match sorted.len() {
        0 => f64::NAN,
        1 => sorted[0],
        n => {
            let h = p.clamp(0.0, 1.0) * (n - 1) as f64;
            let lo = h.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
        }
    }
}

/// Mid-ranks in 1..=n (ties share their average rank).
pub fn mid_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[order[j + 1]] == values[order[i]] {
            j += 1;
        }
        let rank = (i + j) as f64 / 2.0 + 1.0;
        for &k in &order[i..=j] {
            ranks[k] = rank;
        }
        i = j + 1;
    }
    ranks
}

/// Van der Waerden normal scores Φ⁻¹(rank/(n+1)).
pub fn normal_scores(values: &[f64]) -> Vec<f64> {
    let n = values.len() as f64;
    mid_ranks(values)
        .into_iter()
        .map(|r| norm_ppf(r / (n + 1.0)))
        .collect()
}

pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

pub fn variance(v: &[f64]) -> f64 {
    let m = mean(v);
    if v.len() < 2 {
        0.0
    } else {
        v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64
    }
}

pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (mean(a), mean(b));
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (x - ma, y - mb);
        sab += dx * dy;
        saa += dx * dx;
        sbb += dy * dy;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        0.0
    } else {
        sab / (saa * sbb).sqrt()
    }
}

pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    pearson(&mid_ranks(a), &mid_ranks(b))
}

/// Chatterjee's ξ correlation (2021): ≈0 under independence and →1 when `y`
/// is a measurable function of `x`, monotone or not. Used to screen drivers
/// that a rank correlation would miss (e.g. y = (x − ½)²).
pub fn chatterjee_xi(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 {
        return 0.0;
    }
    let mut order: Vec<usize> = (0..n).collect();
    // Ties in x are broken by index (deterministic in place of random).
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(a.cmp(&b)));
    let ys: Vec<f64> = order.iter().map(|&i| y[i]).collect();
    let mut sorted = ys.clone();
    sorted.sort_by(f64::total_cmp);
    // r_i = #{j : y_j <= y_i}; l_i = #{j : y_j >= y_i}
    let r: Vec<f64> = ys
        .iter()
        .map(|v| sorted.partition_point(|s| s <= v) as f64)
        .collect();
    let l: Vec<f64> = ys
        .iter()
        .map(|v| (n - sorted.partition_point(|s| s < v)) as f64)
        .collect();
    let num: f64 = r.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    let den: f64 = l.iter().map(|li| li * (n as f64 - li)).sum();
    if den <= 0.0 {
        0.0
    } else {
        1.0 - n as f64 * num / (2.0 * den)
    }
}

/// In-place lower Cholesky of a row-major SPD matrix; `None` if not SPD.
pub fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 1e-12 {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Solve `A x = b` for SPD `A`, adding diagonal jitter until it factors.
pub fn solve_spd(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut jitter = 0.0;
    loop {
        let mut m = a.to_vec();
        for i in 0..n {
            m[i * n + i] += jitter;
        }
        if let Some(l) = cholesky(&m, n) {
            let mut y = vec![0.0; n];
            for i in 0..n {
                let mut s = b[i];
                for k in 0..i {
                    s -= l[i * n + k] * y[k];
                }
                y[i] = s / l[i * n + i];
            }
            let mut x = vec![0.0; n];
            for i in (0..n).rev() {
                let mut s = y[i];
                for k in i + 1..n {
                    s -= l[k * n + i] * x[k];
                }
                x[i] = s / l[i * n + i];
            }
            return x;
        }
        let scale = (0..n).map(|i| a[i * n + i].abs()).fold(1e-12, f64::max);
        jitter = if jitter == 0.0 {
            1e-10 * scale
        } else {
            jitter * 10.0
        };
        if jitter > scale * 1e3 {
            return vec![0.0; n];
        }
    }
}

/// Symmetric eigen-decomposition by cyclic Jacobi. Returns (values, vectors)
/// with eigenvectors as columns of a row-major matrix, sorted descending.
pub fn sym_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..64 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| m[i * n + j] * m[i * n + j])
            .sum();
        if off < 1e-22 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let t = theta.signum().max(0.0) * 2.0 - 1.0;
                let t = t / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (mkp, mkq) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let (mpk, mqk) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| m[b * n + b].total_cmp(&m[a * n + a]));
    let values = order.iter().map(|&i| m[i * n + i]).collect();
    let mut vectors = vec![0.0; n * n];
    for (new, &old) in order.iter().enumerate() {
        for k in 0..n {
            vectors[k * n + new] = v[k * n + old];
        }
    }
    (values, vectors)
}

/// Project a symmetric matrix with unit diagonal to the nearest (clipped
/// spectrum) correlation matrix with eigenvalues >= `floor`.
pub fn nearest_correlation(a: &[f64], n: usize, floor: f64) -> Vec<f64> {
    let (vals, vecs) = sym_eigen(a, n);
    let mut out = vec![0.0; n * n];
    for k in 0..n {
        let lam = vals[k].max(floor);
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] += lam * vecs[i * n + k] * vecs[j * n + k];
            }
        }
    }
    let d: Vec<f64> = (0..n).map(|i| out[i * n + i].max(1e-12).sqrt()).collect();
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] /= d[i] * d[j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppf_inverts_cdf() {
        for p in [1e-6, 0.01, 0.2, 0.5, 0.77, 0.999] {
            let z = norm_ppf(p);
            assert!((norm_cdf(z) - p).abs() < 2e-7, "{p}");
        }
        assert!((norm_ppf(0.975) - 1.959_963_984_540_054).abs() < 1e-12);
    }

    #[test]
    fn xi_detects_non_monotone_dependence() {
        let x: Vec<f64> = (0..400).map(|i| i as f64 / 399.0).collect();
        let y: Vec<f64> = x.iter().map(|v| (v - 0.5) * (v - 0.5)).collect();
        assert!(spearman(&x, &y).abs() < 0.05);
        assert!(chatterjee_xi(&x, &y) > 0.9);
    }

    #[test]
    fn eigen_reconstructs() {
        let a = vec![2.0, 0.5, 0.1, 0.5, 1.0, 0.3, 0.1, 0.3, 0.7];
        let (vals, vecs) = sym_eigen(&a, 3);
        for i in 0..3 {
            for j in 0..3 {
                let r: f64 = (0..3)
                    .map(|k| vals[k] * vecs[i * 3 + k] * vecs[j * 3 + k])
                    .sum();
                assert!((r - a[i * 3 + j]).abs() < 1e-9);
            }
        }
        assert!(vals[0] >= vals[1] && vals[1] >= vals[2]);
    }

    #[test]
    fn spd_solve() {
        let a = vec![4.0, 1.0, 1.0, 3.0];
        let x = solve_spd(&a, &[1.0, 2.0], 2);
        assert!((4.0 * x[0] + x[1] - 1.0).abs() < 1e-12);
        assert!((x[0] + 3.0 * x[1] - 2.0).abs() < 1e-12);
    }
}

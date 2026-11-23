use super::{KpiRng, Rows};
use dope_core::copula::Dependence;
use dope_core::stats::{norm_cdf, norm_ppf};

/// `m` rows (`p` features then the target) from a Gaussian copula fitted on A: exact empirical
/// marginals per column, each X column missing at its A share, dependence of `rank` (or full).
pub(super) fn copula_rows(a: &Rows, rank: Option<usize>, m: usize, rng: &mut KpiRng) -> Vec<f32> {
    let (n, p) = (a.n, a.p);
    let d = p + 1;
    let value = |i: usize, j: usize| {
        if j < p {
            f64::from(a.x[i * p + j])
        } else {
            f64::from(a.y[i])
        }
    };
    let mut sorted: Vec<Vec<f64>> = Vec::with_capacity(d);
    let mut miss = vec![0.0f64; d];
    for (j, share) in miss.iter_mut().enumerate() {
        let mut col: Vec<f64> = (0..n)
            .map(|i| value(i, j))
            .filter(|v| v.is_finite())
            .collect();
        *share = 1.0 - col.len() as f64 / n.max(1) as f64;
        col.sort_by(f64::total_cmp);
        sorted.push(col);
    }
    // Normal scores of A from mid-ranks; a missing cell scores 0.
    let mut z = vec![0.0f64; n * d];
    for j in 0..d {
        let s = &sorted[j];
        for i in 0..n {
            let v = value(i, j);
            if v.is_finite() && !s.is_empty() {
                let lo = s.partition_point(|&w| w < v);
                let hi = s.partition_point(|&w| w <= v);
                z[i * d + j] = norm_ppf((lo + hi) as f64 / (2 * s.len()) as f64);
            }
        }
    }
    let mut r = vec![0.0f64; d * d];
    let sd: Vec<f64> = (0..d)
        .map(|j| ((0..n).map(|i| z[i * d + j].powi(2)).sum::<f64>() / n.max(1) as f64).sqrt())
        .collect();
    for j in 0..d {
        for k in 0..d {
            r[j * d + k] = if j == k {
                1.0
            } else if sd[j] > 0.0 && sd[k] > 0.0 {
                (0..n).map(|i| z[i * d + j] * z[i * d + k]).sum::<f64>()
                    / (n as f64 * sd[j] * sd[k])
            } else {
                0.0
            };
        }
    }
    let dep = match rank {
        Some(rk) => Dependence::fit_factor(&r, d, rk),
        None => Dependence::fit_full(&r, d),
    };
    let (mix, w) = dep.mixing(d);
    let mut out = Vec::with_capacity(m * d);
    let mut e = vec![0.0f64; w];
    for _ in 0..m {
        for v in e.iter_mut() {
            *v = rng.normal();
        }
        for j in 0..d {
            let zj: f64 = (0..w).map(|k| mix[j * w + k] * e[k]).sum();
            let s = &sorted[j];
            let x = if s.is_empty() || (j < p && rng.uniform() < miss[j]) {
                f64::NAN
            } else {
                let u = norm_cdf(zj);
                s[((u * s.len() as f64) as usize).min(s.len() - 1)]
            };
            out.push(x as f32);
        }
    }
    out
}

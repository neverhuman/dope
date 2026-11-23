//! Core statistical primitives for the pure-Rust FIT-only Evidence Bank (`dope_core::evidence`).

use crate::stats::{norm_cdf, norm_pdf, norm_ppf, quantile_sorted};

pub use super::rank_stats::{dcor_unbiased_1d, hartigan_dip_stat, kendall_tau_b};

pub const SUPPORT_FLOOR_K: usize = 10;

#[inline]
pub fn squash(v: f64) -> f32 {
    if !v.is_finite() {
        return 0.0;
    }
    (v / (1.0 + v.abs())).clamp(-1.0, 1.0) as f32
}

#[inline]
pub fn squash_z(z: f64) -> f32 {
    if !z.is_finite() {
        return 0.0;
    }
    let zc = z.clamp(-8.0, 8.0);
    (zc / (3.0 + zc.abs())) as f32
}

#[inline]
pub fn clamp_f32(v: f64, lo: f64, hi: f64) -> f32 {
    if !v.is_finite() {
        return 0.0;
    }
    v.clamp(lo, hi) as f32
}

#[inline]
pub fn set_mask_bit(mask: &mut [u8], idx: usize) {
    let byte = idx / 8;
    let bit = idx % 8;
    if byte < mask.len() {
        mask[byte] |= 1u8 << bit;
    }
}

/// Wilson-Hilferty cube-root normal approximation for chi^2_df -> standard normal z.
pub fn wilson_hilferty_z(chi2: f64, df: f64) -> f64 {
    if !chi2.is_finite() || !df.is_finite() || df <= 0.0 || chi2 <= 0.0 {
        return 0.0;
    }
    let spread = 2.0 / (9.0 * df);
    let cube = (chi2 / df).cbrt();
    ((cube - (1.0 - spread)) / spread.sqrt()).clamp(-8.0, 8.0)
}

/// Privacy-safe tail probability floor: `u_floor = min(0.20, 10 / max(n + 1, 20))`.
#[inline]
pub fn support_u_floor(n_obs: usize) -> f64 {
    (SUPPORT_FLOOR_K as f64 / (n_obs as f64 + 1.0).max(20.0)).min(0.20)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Column-permutation-invariant canonical row ordering from `X` only.
/// Guarantees exact bit-for-bit row-permutation invariance AND column-permutation equivariance
/// without ever reading target labels `y`.
pub(crate) fn canonical_row_order(x_unit: &[Vec<f32>]) -> Vec<usize> {
    let p = x_unit.len();
    let n = x_unit.first().map(|c| c.len()).unwrap_or(0);
    let mut keys: Vec<(u64, u64, usize)> = Vec::with_capacity(n);
    for r in 0..n {
        let mut h1 = 0u64;
        let mut h2 = 0u64;
        for col in x_unit.iter().take(p) {
            let v = col[r];
            let bits = if !v.is_finite() {
                0x7FC0_0000u64
            } else if v == 0.0 {
                0u64 // unify -0.0 and +0.0
            } else {
                v.to_bits() as u64
            };
            let m1 = splitmix64(bits ^ 0xA24B_AED4_963E_E407);
            let m2 = splitmix64(bits ^ 0x9FB2_1C65_1E98_DF25);
            h1 = h1.wrapping_add(m1);
            h2 ^= m2;
        }
        keys.push((h1, h2, r));
    }
    keys.sort_unstable();
    keys.into_iter().map(|(_, _, r)| r).collect()
}

/// Compute tie-aware conditional-mean normal scores and tie-attenuation factor `kappa_j in [0.15, 1.0]`.
pub fn tie_aware_normal_scores(col: &[f64]) -> (Vec<f64>, f64, usize) {
    let n = col.len();
    let mut obs: Vec<(usize, f64)> = col
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (i, v))
        .collect();
    let n_obs = obs.len();
    let mut z = vec![0.0f64; n];
    if n_obs < 4 {
        return (z, 1.0, n_obs);
    }
    obs.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let nf = n_obs as f64;
    let u_eps = 0.5 / nf;

    let mut i = 0usize;
    while i < n_obs {
        let mut j = i + 1;
        while j < n_obs && (obs[j].1 - obs[i].1).abs() <= 1e-12 * obs[i].1.abs().max(1.0) {
            j += 1;
        }
        let m = j - i;
        let score = if m == 1 {
            norm_ppf((i as f64 + 0.5) / nf)
        } else {
            let a = (i as f64 / nf).max(u_eps);
            let b = (j as f64 / nf).min(1.0 - u_eps);
            let span = (b - a).max(1e-9);
            ((norm_pdf(norm_ppf(a)) - norm_pdf(norm_ppf(b))) / span).clamp(-3.5, 3.5)
        };
        for &(row_idx, _) in &obs[i..j] {
            z[row_idx] = score;
        }
        i = j;
    }

    let mean_z = obs.iter().map(|&(r, _)| z[r]).sum::<f64>() / nf;
    let var_z = obs
        .iter()
        .map(|&(r, _)| (z[r] - mean_z).powi(2))
        .sum::<f64>()
        / nf;
    let ref_var = (0..n_obs)
        .map(|r| {
            let t = norm_ppf((r as f64 + 0.5) / nf);
            t * t
        })
        .sum::<f64>()
        / nf;
    let kappa = (var_z.sqrt() / ref_var.sqrt().max(1e-6)).clamp(0.15, 1.0);
    (z, kappa, n_obs)
}

/// Assign each finite observation to `0..n_bins` value-consistent quantile bins (`255` for NaN).
pub fn value_consistent_bins(col: &[f64], n_bins: usize) -> (Vec<u8>, Vec<f64>) {
    let n = col.len();
    let nb = n_bins.clamp(2, 32);
    let mut fin: Vec<f64> = col.iter().copied().filter(|v| v.is_finite()).collect();
    if fin.len() < 4 {
        return (vec![0u8; n], vec![0.0; nb - 1]);
    }
    fin.sort_by(f64::total_cmp);
    let mut edges = Vec::with_capacity(nb - 1);
    for b in 1..nb {
        edges.push(quantile_sorted(&fin, b as f64 / nb as f64));
    }
    let mut bins = vec![255u8; n];
    for (i, &v) in col.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        let mut b = 0usize;
        while b < edges.len() && v > edges[b] {
            b += 1;
        }
        bins[i] = b as u8;
    }
    (bins, edges)
}

/// Exact O(n) first five sample L-moments `[lambda_1, lambda_2, lambda_3, lambda_4, lambda_5]`.
pub fn l_moments_5(sorted: &[f64]) -> [f64; 5] {
    let n = sorted.len();
    if n < 5 {
        return [0.0; 5];
    }
    let nf = n as f64;
    let mut b = [0.0f64; 5];
    for (idx, &x) in sorted.iter().enumerate() {
        let i = idx as f64;
        b[0] += x;
        let w1 = i / (nf - 1.0);
        b[1] += w1 * x;
        let w2 = w1 * (i - 1.0).max(0.0) / (nf - 2.0);
        b[2] += w2 * x;
        let w3 = w2 * (i - 2.0).max(0.0) / (nf - 3.0);
        b[3] += w3 * x;
        let w4 = w3 * (i - 3.0).max(0.0) / (nf - 4.0);
        b[4] += w4 * x;
    }
    for slot in &mut b {
        *slot /= nf;
    }
    let l1 = b[0];
    let l2 = 2.0 * b[1] - b[0];
    let l3 = 6.0 * b[2] - 6.0 * b[1] + b[0];
    let l4 = 20.0 * b[3] - 30.0 * b[2] + 12.0 * b[1] - b[0];
    let l5 = 70.0 * b[4] - 140.0 * b[3] + 90.0 * b[2] - 20.0 * b[1] + b[0];
    [l1, l2, l3, l4, l5]
}

/// Exact O(n) PAVA isotonic regression R^2 (takes better of increasing & decreasing).
pub fn pava_isotonic_r2(x: &[f64], y: &[f64]) -> f64 {
    let mut pairs: Vec<(f64, f64, usize)> = x
        .iter()
        .zip(y)
        .enumerate()
        .filter(|(_, (a, b))| a.is_finite() && b.is_finite())
        .map(|(i, (&a, &b))| (a, b, i))
        .collect();
    let n = pairs.len();
    if n < 8 {
        return 0.0;
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
    let ys: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let mean_y = ys.iter().sum::<f64>() / n as f64;
    let var_y = ys.iter().map(|&v| (v - mean_y).powi(2)).sum::<f64>();
    if var_y <= 1e-12 {
        return 0.0;
    }
    let fit_inc = |vals: &[f64]| -> f64 {
        let mut blocks: Vec<(f64, usize)> = Vec::with_capacity(vals.len());
        for &v in vals {
            blocks.push((v, 1));
            while blocks.len() >= 2 {
                let b2 = blocks[blocks.len() - 1];
                let b1 = blocks[blocks.len() - 2];
                if b1.0 / b1.1 as f64 > b2.0 / b2.1 as f64 {
                    blocks.pop();
                    let last = blocks.last_mut().unwrap();
                    last.0 += b2.0;
                    last.1 += b2.1;
                } else {
                    break;
                }
            }
        }
        let mut sse = 0.0;
        let mut pos = 0usize;
        for (sum, cnt) in blocks {
            let mu = sum / cnt as f64;
            for &v in &vals[pos..pos + cnt] {
                sse += (v - mu).powi(2);
            }
            pos += cnt;
        }
        (1.0 - sse / var_y).clamp(0.0, 1.0)
    };
    let r2_inc = fit_inc(&ys);
    let ys_neg: Vec<f64> = ys.iter().map(|&v| -v).collect();
    let r2_dec = fit_inc(&ys_neg);
    r2_inc.max(r2_dec)
}

/// Miller-Madow bias-corrected mutual information (natural log units) and Wilson-Hilferty G-test Z.
pub fn miller_madow_mi_and_z(bx: &[u8], by: &[u8], nx: usize, ny: usize) -> (f64, f64) {
    let mut counts = vec![0usize; nx * ny];
    let mut cx = vec![0usize; nx];
    let mut cy = vec![0usize; ny];
    let mut n = 0usize;
    for (&ix, &iy) in bx.iter().zip(by) {
        let a = ix as usize;
        let b = iy as usize;
        if a < nx && b < ny {
            counts[a * ny + b] += 1;
            cx[a] += 1;
            cy[b] += 1;
            n += 1;
        }
    }
    if n < SUPPORT_FLOOR_K {
        return (0.0, 0.0);
    }
    let nf = n as f64;
    let non_empty_x = cx.iter().filter(|&&c| c > 0).count();
    let non_empty_y = cy.iter().filter(|&&c| c > 0).count();
    if non_empty_x <= 1 || non_empty_y <= 1 {
        return (0.0, 0.0);
    }
    let mut mi = 0.0;
    for i in 0..nx {
        if cx[i] == 0 {
            continue;
        }
        for j in 0..ny {
            let cij = counts[i * ny + j];
            if cij > 0 {
                let pij = cij as f64 / nf;
                let px = cx[i] as f64 / nf;
                let py = cy[j] as f64 / nf;
                mi += pij * (pij / (px * py)).ln();
            }
        }
    }
    let df = ((non_empty_x - 1) * (non_empty_y - 1)) as f64;
    let g_stat = 2.0 * nf * mi.max(0.0);
    let z = wilson_hilferty_z(g_stat, df);
    let mm_mi = (mi - df / (2.0 * nf)).max(0.0);
    (mm_mi, z)
}

/// Analytic bivariate Gaussian distance correlation for correlation `r` (Szekely-Rizzo-Bakirov 2007).
pub fn gaussian_dcor(r: f64) -> f64 {
    let a = r.abs().clamp(0.0, 0.9999);
    let num =
        a * a.asin() + (1.0 - a * a).sqrt() - a * (0.5 * a).asin() - (4.0 - a * a).sqrt() + 1.0;
    let den = 1.0 + std::f64::consts::FRAC_PI_3 - 3.0_f64.sqrt();
    (num / den).max(0.0).sqrt().clamp(0.0, 1.0)
}

/// Bivariate Gaussian copula 10% tail co-exceedance `P(U1 > 0.9 | U2 > 0.9)`.
pub fn gaussian_copula_tail10_ref(rho: f64) -> f64 {
    let r = rho.clamp(-0.985, 0.985);
    let z0 = norm_ppf(0.90);
    let s = (1.0 - r * r).max(1e-6).sqrt();
    let steps = 24usize;
    let lo = z0;
    let hi = 5.5;
    let h = (hi - lo) / steps as f64;
    let mut joint = 0.0;
    for k in 0..steps {
        let z = lo + (k as f64 + 0.5) * h;
        let cond_p = 1.0 - norm_cdf((z0 - r * z) / s);
        joint += norm_pdf(z) * cond_p * h;
    }
    (joint / 0.10).clamp(0.0, 1.0)
}

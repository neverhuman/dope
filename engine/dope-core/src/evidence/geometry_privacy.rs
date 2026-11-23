//! Tier T1 Global Geometry, Mardia Multivariate Normality, TwoNN Intrinsic Dim,
//! Hopkins Clustering, 2/3-means Regime Variance, and Privacy Risk Indicators.

use super::primitives::{clamp_f32, set_mask_bit, squash, squash_z};
use super::GLOB_DIM;
use crate::rng::Rng;

/// Fill `G_GEOM_PRIV` slots `40..54` and return estimated Student-t tail `inv_nu in [0, 0.45]`.
pub(crate) fn fill_geometry_and_privacy(
    z_sub: &[Vec<f64>],
    precision: &[f64],
    p: usize,
    n_fit: usize,
    seed: u64,
    out_glob: &mut [f32; GLOB_DIM],
    out_glob_mask: &mut [u8; 20],
) -> f64 {
    let m = z_sub.first().map(|c| c.len()).unwrap_or(0);
    if p == 0 || m < 8 {
        return 0.0;
    }
    let m_mard = m.min(192);
    let mut rows = vec![vec![0.0f64; p]; m_mard];
    for r in 0..m_mard {
        for j in 0..p {
            rows[r][j] = z_sub[j][r];
        }
    }
    let mut mah2 = vec![0.0f64; m_mard];
    let mut t_rows = vec![vec![0.0f64; p]; m_mard];
    for r in 0..m_mard {
        for i in 0..p {
            let mut s = 0.0;
            for j in 0..p {
                s += precision[i * p + j] * rows[r][j];
            }
            t_rows[r][i] = s;
        }
        let d2: f64 = rows[r].iter().zip(&t_rows[r]).map(|(a, b)| a * b).sum();
        mah2[r] = d2.max(0.0);
    }
    let b2p = mah2.iter().map(|d2| d2 * d2).sum::<f64>() / m_mard as f64;
    let pf = p as f64;
    let exp_b2 = pf * (pf + 2.0);
    let sd_b2 = (8.0 * exp_b2 / m_mard as f64).sqrt().max(1e-6);
    let mardia_kurt_z = (b2p - exp_b2) / sd_b2;
    let kappa_m = (b2p / exp_b2.max(1.0)) - 1.0;
    let inv_nu = (kappa_m / (2.0 * (1.0 + 2.0 * kappa_m.max(0.0)))).clamp(0.0, 0.45);

    let m_sk = m_mard.min(96);
    let mut sum_cube = 0.0;
    for a in 0..m_sk {
        for b in 0..m_sk {
            let dot: f64 = rows[a].iter().zip(&t_rows[b]).map(|(x, y)| x * y).sum();
            sum_cube += dot * dot * dot;
        }
    }
    let b1p = sum_cube / ((m_sk * m_sk) as f64);
    let df_s = pf * (pf + 1.0) * (pf + 2.0) / 6.0;
    let mardia_skew_z = ((m_sk as f64 * b1p / 6.0) - df_s) / (2.0 * df_s).sqrt().max(1.0);

    let mut d1 = vec![f64::INFINITY; m_mard];
    let mut d2 = vec![f64::INFINITY; m_mard];
    let mut exact_dup = 0usize;
    for i in 0..m_mard {
        for j in 0..m_mard {
            if i == j {
                continue;
            }
            let dist = rows[i]
                .iter()
                .zip(&rows[j])
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            if dist < d1[i] {
                d2[i] = d1[i];
                d1[i] = dist;
            } else if dist < d2[i] {
                d2[i] = dist;
            }
        }
        if d1[i] < 1e-6 {
            exact_dup += 1;
        }
    }
    let mut ratios: Vec<f64> = d1
        .iter()
        .zip(&d2)
        .filter(|(&a, &b)| a > 1e-6 && b.is_finite() && b > a)
        .map(|(&a, &b)| (b / a).ln())
        .collect();
    ratios.sort_by(f64::total_cmp);
    let keep = (ratios.len() * 9 / 10).max(1);
    let twonn_id = if !ratios.is_empty() {
        let sum_log: f64 = ratios[..keep].iter().sum();
        if sum_log > 1e-9 {
            (keep as f64 / sum_log) / pf.max(1.0)
        } else {
            1.0
        }
    } else {
        1.0
    };

    // Column-permutation-invariant Hopkins probes: derive probe coordinate j from column min/max
    // using an order-independent scalar quantile seed so Hopkins is column-permutation invariant.
    let mut rng = Rng::new(seed ^ 0x4094_1A55);
    let mut mins = vec![f64::INFINITY; p];
    let mut maxs = vec![f64::NEG_INFINITY; p];
    for r in &rows {
        for j in 0..p {
            mins[j] = mins[j].min(r[j]);
            maxs[j] = maxs[j].max(r[j]);
        }
    }
    let n_probe = m_mard.min(24);
    let mut sum_u = 0.0;
    let mut sum_w = 0.0;
    for k in 0..n_probe {
        let mut best_u = f64::INFINITY;
        let u_scalar = rng.uniform();
        let probe: Vec<f64> = (0..p)
            .map(|j| {
                // Mix column's own value distribution deterministically without depending on j's index.
                let col_hash =
                    ((mins[j].to_bits().wrapping_add(maxs[j].to_bits())) & 0xFFFF) as f64 / 65536.0;
                let t = (u_scalar + col_hash).fract();
                mins[j] + t * (maxs[j] - mins[j]).max(1e-6)
            })
            .collect();
        for r in &rows {
            let dist = probe
                .iter()
                .zip(r)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            if dist < best_u {
                best_u = dist;
            }
        }
        sum_u += best_u;
        sum_w += d1[k].min(1e6);
    }
    let hopkins = if sum_u + sum_w > 1e-9 {
        sum_u / (sum_u + sum_w)
    } else {
        0.5
    };

    let (km2_r2, km3_r2) = kmeans_r2_2_and_3(&rows);

    let mut d1_sorted: Vec<f64> = d1.iter().copied().filter(|v| v.is_finite()).collect();
    d1_sorted.sort_by(f64::total_cmp);
    let med_d1 = d1_sorted
        .get(d1_sorted.len() / 2)
        .copied()
        .unwrap_or(1.0)
        .max(1e-6);
    let q05_d1 = d1_sorted.get(d1_sorted.len() / 20).copied().unwrap_or(0.0);
    let mean_d1 = d1_sorted.iter().sum::<f64>() / d1_sorted.len().max(1) as f64;
    let sd_d1 = (d1_sorted
        .iter()
        .map(|&v| (v - mean_d1).powi(2))
        .sum::<f64>()
        / d1_sorted.len().max(1) as f64)
        .sqrt();
    let far_outliers = d1_sorted.iter().filter(|&&v| v > 3.0 * med_d1).count() as f64
        / d1_sorted.len().max(1) as f64;
    let high_mah = mah2
        .iter()
        .filter(|&&d| d > pf + 3.0 * (2.0 * pf).sqrt())
        .count() as f64
        / m_mard as f64;

    out_glob[40] = squash_z(mardia_skew_z / 5.0);
    out_glob[41] = squash_z(mardia_kurt_z / 5.0);
    out_glob[42] = clamp_f32(inv_nu, 0.0, 0.45);
    out_glob[43] = clamp_f32(twonn_id, 0.0, 1.5);
    out_glob[44] = clamp_f32(hopkins, 0.0, 1.0);
    out_glob[45] = clamp_f32(km2_r2, 0.0, 1.0);
    out_glob[46] = clamp_f32(km3_r2, 0.0, 1.0);
    out_glob[47] = clamp_f32(exact_dup as f64 / m_mard as f64, 0.0, 1.0);
    out_glob[48] = squash(med_d1 / pf.sqrt().max(1.0));
    out_glob[49] = squash(sd_d1 / mean_d1.max(1e-6));
    out_glob[50] = squash(q05_d1 / pf.sqrt().max(1.0));
    out_glob[51] = clamp_f32(far_outliers, 0.0, 1.0);
    out_glob[52] = clamp_f32(high_mah, 0.0, 1.0);
    out_glob[53] = ((n_fit as f64).max(1.0).log2() / 14.0).clamp(0.0, 1.5) as f32;
    for s in 40..54 {
        set_mask_bit(out_glob_mask, s);
    }
    inv_nu
}

fn kmeans_r2_2_and_3(rows: &[Vec<f64>]) -> (f64, f64) {
    let n = rows.len();
    let p = rows.first().map(|r| r.len()).unwrap_or(0);
    if n < 12 || p == 0 {
        return (0.0, 0.0);
    }
    let mut grand = vec![0.0; p];
    for r in rows {
        for j in 0..p {
            grand[j] += r[j] / n as f64;
        }
    }
    let tot_ss: f64 = rows
        .iter()
        .map(|r| {
            r.iter()
                .zip(&grand)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
        })
        .sum();
    if tot_ss <= 1e-9 {
        return (0.0, 0.0);
    }
    let run_k = |k: usize| -> f64 {
        let mut centers: Vec<Vec<f64>> = (0..k).map(|c| rows[(c * n) / k].clone()).collect();
        let mut assign = vec![0usize; n];
        for _ in 0..8 {
            for (i, r) in rows.iter().enumerate() {
                let mut best_c = 0;
                let mut best_d = f64::INFINITY;
                for (c, cen) in centers.iter().enumerate() {
                    let d: f64 = r.iter().zip(cen).map(|(a, b)| (a - b).powi(2)).sum();
                    if d < best_d {
                        best_d = d;
                        best_c = c;
                    }
                }
                assign[i] = best_c;
            }
            let mut sums = vec![vec![0.0; p]; k];
            let mut cnts = vec![0usize; k];
            for (i, &c) in assign.iter().enumerate() {
                cnts[c] += 1;
                for j in 0..p {
                    sums[c][j] += rows[i][j];
                }
            }
            for c in 0..k {
                if cnts[c] > 0 {
                    for j in 0..p {
                        centers[c][j] = sums[c][j] / cnts[c] as f64;
                    }
                }
            }
        }
        let within: f64 = rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                r.iter()
                    .zip(&centers[assign[i]])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
            })
            .sum();
        (1.0 - within / tot_ss).clamp(0.0, 1.0)
    };
    (run_k(2), run_k(3))
}

//! Tier T1 Evidence: tie-disattenuated latent correlation, PD projection, precision matrix,
//! pairwise edge tensor (16 unambiguous channels), and spectral Marchenko-Pastur diagnostics.

use super::geometry_privacy::fill_geometry_and_privacy;
use super::primitives::{
    clamp_f32, dcor_unbiased_1d, gaussian_copula_tail10_ref, gaussian_dcor, kendall_tau_b,
    miller_madow_mi_and_z, set_mask_bit, squash, squash_z, support_u_floor, wilson_hilferty_z,
    SUPPORT_FLOOR_K,
};
use super::support_marginal::ColT0Summary;
use super::{ColumnMeta, COL_DIM, EDGE_DIM, GLOB_DIM};
use crate::rng::Rng;
use crate::stats::{
    chatterjee_xi, nearest_correlation, norm_cdf, pearson, solve_spd, spearman, sym_eigen,
};

pub(crate) struct DepContext {
    pub r_lat: Vec<f64>,
    pub eig_vals: Vec<f64>,
    pub eig_vecs: Vec<f64>,
    pub loadings12: Vec<[f64; 12]>,
}

/// Compute Tier T1 pairwise dependence, precision, spectrum, geometry, and privacy diagnostics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn extract_dependence_t1(
    x_unit: &[Vec<f32>],
    z_tie: &[Vec<f64>],
    kappa: &[f64],
    bins16: &[Vec<u8>],
    t0: &[ColT0Summary],
    meta: Option<&[ColumnMeta]>,
    seed: u64,
    out_col: &mut [[f32; COL_DIM]],
    out_col_mask: &mut [[u8; 16]],
    out_edge: &mut [[f32; EDGE_DIM]],
    out_edge_mask: &mut [u16],
    out_glob: &mut [f32; GLOB_DIM],
    out_glob_mask: &mut [u8; 20],
) -> DepContext {
    let p = x_unit.len();
    let n = x_unit.first().map(|c| c.len()).unwrap_or(0);
    let mut r_raw = vec![0.0f64; p * p];
    for i in 0..p {
        r_raw[i * p + i] = 1.0;
    }

    let m_sub = n.min(512);
    let sub_rows: Vec<usize> = if n <= m_sub {
        (0..n).collect()
    } else {
        let mut r = Rng::new(seed ^ 0xE71D_50B5_0001);
        let mut idx: Vec<usize> = (0..n).collect();
        for k in 0..m_sub {
            let pick = k + (r.next_u64() as usize % (n - k));
            idx.swap(k, pick);
        }
        idx.truncate(m_sub);
        idx.sort_unstable();
        idx
    };

    let x_sub: Vec<Vec<f64>> = (0..p)
        .map(|j| sub_rows.iter().map(|&r| x_unit[j][r] as f64).collect())
        .collect();
    let z_sub: Vec<Vec<f64>> = (0..p)
        .map(|j| sub_rows.iter().map(|&r| z_tie[j][r]).collect())
        .collect();
    let z2_sub: Vec<Vec<f64>> = (0..p)
        .map(|j| z_sub[j].iter().map(|&v| v * v - 1.0).collect())
        .collect();
    let b4_sub: Vec<Vec<u8>> = (0..p)
        .map(|j| {
            sub_rows
                .iter()
                .map(|&r| (bins16[j][r] / 4).min(3))
                .collect()
        })
        .collect();
    let u_sub: Vec<Vec<f64>> = (0..p)
        .map(|j| z_sub[j].iter().map(|&v| norm_cdf(v)).collect())
        .collect();

    // 1. Disattenuated latent correlation R_lat
    for i in 0..p {
        if t0[i].is_const {
            continue;
        }
        for j in (i + 1)..p {
            if t0[j].is_const {
                continue;
            }
            let r_z = pearson(&z_tie[i], &z_tie[j]);
            let r_lat = (r_z / (kappa[i] * kappa[j]).max(0.0225)).clamp(-0.985, 0.985);
            r_raw[i * p + j] = r_lat;
            r_raw[j * p + i] = r_lat;
        }
    }
    let r_lat = if p >= 2 {
        nearest_correlation(&r_raw, p, 1e-3)
    } else {
        r_raw.clone()
    };

    // 2. Eigendecomposition & rank-12 canonicalized loadings
    let (eig_vals, eig_vecs) = if p >= 1 {
        sym_eigen(&r_lat, p)
    } else {
        (Vec::new(), Vec::new())
    };
    let r_use = p.min(12);
    let mut loadings12 = vec![[0.0f64; 12]; p];
    for k in 0..r_use {
        let s = eig_vals[k].max(0.0).sqrt();
        // Order-invariant sign canonicalization: choose sign so sum of cubes of column loadings >= 0
        // (breaks exact ties with largest-magnitude entry, preserving column-perm equivariance!).
        let sum_cube: f64 = (0..p).map(|i| eig_vecs[i * p + k].powi(3)).sum();
        let sign = if sum_cube.abs() > 1e-12 {
            if sum_cube < 0.0 {
                -1.0
            } else {
                1.0
            }
        } else {
            let sum_v: f64 = (0..p).map(|i| eig_vecs[i * p + k]).sum();
            if sum_v < 0.0 {
                -1.0
            } else {
                1.0
            }
        };
        for i in 0..p {
            loadings12[i][k] = sign * eig_vecs[i * p + k] * s;
        }
    }
    for row in &mut loadings12 {
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.99 {
            let inv = 0.99 / norm;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }

    // 3. Adaptive-ridge precision matrix Theta = (R_lat + lam * I)^-1
    let lam_prec = (p as f64 / (2.0 * n.max(1) as f64)).clamp(0.05, 0.25);
    let mut r_reg = r_lat.clone();
    for i in 0..p {
        r_reg[i * p + i] += lam_prec;
    }
    let mut precision = vec![0.0f64; p * p];
    for col in 0..p {
        let mut e = vec![0.0f64; p];
        e[col] = 1.0;
        let sol = solve_spd(&r_reg, &e, p);
        for row in 0..p {
            precision[row * p + col] = sol[row];
        }
    }

    // 4. Pairwise Edge Tensor (16 channels; channel 15 strictly `collinear_gt_0p70`).
    let q_tail = support_u_floor(m_sub).max(0.10);
    let mut sum_abs_r = 0.0;
    let mut sig_pairs = 0usize;
    let mut high_corr_pairs = 0usize;
    let mut total_pairs = 0usize;

    for i in 0..p {
        let d_idx = i * p + i;
        out_edge[d_idx][0] = 1.0;
        out_edge[d_idx][1] = 1.0;
        out_edge[d_idx][3] = 1.0;
        out_edge[d_idx][14] = 1.0;
        out_edge[d_idx][15] = 1.0;
        out_edge_mask[d_idx] |= (1 << 0) | (1 << 1) | (1 << 3) | (1 << 14) | (1 << 15);
    }

    let mut max_abs_r = vec![0.0f64; p];
    let mut mean_abs_r = vec![0.0f64; p];
    let mut mean_pcorr_sq = vec![0.0f64; p];
    let mut mean_lam_u = vec![0.0f64; p];
    let mut mean_lam_l = vec![0.0f64; p];
    let mut max_nongauss = vec![0.0f64; p];
    let mut max_resid_corr = vec![0.0f64; p];

    for i in 0..p {
        for j in (i + 1)..p {
            let rl = r_lat[i * p + j];
            let pcorr = (-precision[i * p + j]
                / (precision[i * p + i].max(1e-9) * precision[j * p + j].max(1e-9)).sqrt())
            .clamp(-1.0, 1.0);
            let same_grp = meta
                .map(|ms| {
                    if ms[i].group_id == ms[j].group_id {
                        1.0f32
                    } else {
                        0.0
                    }
                })
                .unwrap_or(0.0);
            let collinear = if rl.abs() > 0.70 { 1.0f32 } else { 0.0 };
            if rl.abs() > 0.70 {
                high_corr_pairs += 1;
            }
            total_pairs += 1;
            sum_abs_r += rl.abs();
            if rl.abs() * (n.max(4) as f64 - 3.0).sqrt() > 2.5 {
                sig_pairs += 1;
            }

            let (sp_minus_p, tau_b, xi_ij, xi_ji, ng_excess, lam_u_ex, lam_l_ex, vol_coup) =
                if t0[i].is_const || t0[j].is_const || m_sub < SUPPORT_FLOOR_K {
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                } else {
                    let r_p = pearson(&x_sub[i], &x_sub[j]);
                    let r_s = spearman(&x_sub[i], &x_sub[j]);
                    let tau = if p <= 48 {
                        kendall_tau_b(&x_sub[i], &x_sub[j])
                    } else {
                        (2.0 / std::f64::consts::PI) * rl.asin()
                    };
                    let xi1 = chatterjee_xi(&x_sub[i], &x_sub[j]).clamp(-0.2, 1.0);
                    let xi2 = chatterjee_xi(&x_sub[j], &x_sub[i]).clamp(-0.2, 1.0);
                    let ng = if p <= 48 {
                        (dcor_unbiased_1d(&z_sub[i], &z_sub[j]) - gaussian_dcor(rl))
                            .clamp(-1.0, 1.0)
                    } else {
                        let (_, mi_z) = miller_madow_mi_and_z(&b4_sub[i], &b4_sub[j], 4, 4);
                        (mi_z / 8.0).clamp(-1.0, 1.0)
                    };
                    let mut cnt_u = 0usize;
                    let mut cnt_l = 0usize;
                    for r in 0..m_sub {
                        if u_sub[i][r] > 1.0 - q_tail && u_sub[j][r] > 1.0 - q_tail {
                            cnt_u += 1;
                        }
                        if u_sub[i][r] < q_tail && u_sub[j][r] < q_tail {
                            cnt_l += 1;
                        }
                    }
                    let g_tail = gaussian_copula_tail10_ref(rl);
                    let lu = ((cnt_u as f64 / (m_sub as f64 * q_tail)) - g_tail).clamp(-1.0, 1.0);
                    let ll = ((cnt_l as f64 / (m_sub as f64 * q_tail)) - g_tail).clamp(-1.0, 1.0);
                    let vc = (pearson(&z2_sub[i], &z2_sub[j]) - rl * rl).clamp(-1.0, 1.0);
                    ((r_s - r_p).clamp(-1.0, 1.0), tau, xi1, xi2, ng, lu, ll, vc)
                };

            let ll_dot: f64 = (0..12).map(|k| loadings12[i][k] * loadings12[j][k]).sum();
            let resid_c = (rl - ll_dot).clamp(-1.0, 1.0);

            let co_miss = if t0[i].miss_rate > 0.0 && t0[j].miss_rate > 0.0 {
                let mi: Vec<f64> = x_unit[i]
                    .iter()
                    .map(|v| if v.is_finite() { 0.0 } else { 1.0 })
                    .collect();
                let mj: Vec<f64> = x_unit[j]
                    .iter()
                    .map(|v| if v.is_finite() { 0.0 } else { 1.0 })
                    .collect();
                pearson(&mi, &mj).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            let fill_pair = |slot: &mut [f32; EDGE_DIM], em: &mut u16, xi_dir: f64| {
                slot[0] = rl as f32;
                slot[1] = pcorr as f32;
                slot[2] = sp_minus_p as f32;
                slot[3] = tau_b as f32;
                slot[4] = xi_dir as f32;
                slot[5] = ng_excess as f32;
                slot[6] = lam_u_ex as f32;
                slot[7] = lam_l_ex as f32;
                slot[8] = vol_coup as f32;
                slot[12] = resid_c as f32;
                slot[13] = co_miss as f32;
                slot[14] = same_grp;
                slot[15] = collinear;
                *em |= 0b1111_0001_1111_1111;
            };
            fill_pair(
                &mut out_edge[i * p + j],
                &mut out_edge_mask[i * p + j],
                xi_ij,
            );
            fill_pair(
                &mut out_edge[j * p + i],
                &mut out_edge_mask[j * p + i],
                xi_ji,
            );

            for a in [i, j] {
                max_abs_r[a] = max_abs_r[a].max(rl.abs());
                mean_abs_r[a] += rl.abs();
                mean_pcorr_sq[a] += pcorr * pcorr;
                mean_lam_u[a] += lam_u_ex.max(0.0);
                mean_lam_l[a] += lam_l_ex.max(0.0);
                max_nongauss[a] = max_nongauss[a].max(ng_excess.abs());
                max_resid_corr[a] = max_resid_corr[a].max(resid_c.abs());
            }
        }
    }

    // 5. Fill per-column dependence block D_DEP_PREC (slots 56..78)
    let denom_p = (p.saturating_sub(1)).max(1) as f64;
    for j in 0..p {
        let comm4: f64 = loadings12[j][..4.min(r_use)].iter().map(|v| v * v).sum();
        let comm8: f64 = loadings12[j][..8.min(r_use)].iter().map(|v| v * v).sum();
        let comm12: f64 = loadings12[j][..r_use].iter().map(|v| v * v).sum();
        let psi = (1.0 - comm12).clamp(0.02, 1.0);
        let vif = precision[j * p + j].max(1.0);
        for k in 0..12 {
            out_col[j][56 + k] = loadings12[j][k] as f32;
            set_mask_bit(&mut out_col_mask[j], 56 + k);
        }
        out_col[j][68] = psi as f32;
        out_col[j][69] = ((1.0 + (vif - 1.0).max(0.0)).ln() / 4.0).clamp(0.0, 2.0) as f32;
        out_col[j][70] = (mean_pcorr_sq[j] / denom_p).sqrt() as f32;
        out_col[j][71] = max_abs_r[j] as f32;
        out_col[j][72] = (mean_abs_r[j] / denom_p) as f32;
        out_col[j][73] = (mean_lam_u[j] / denom_p).clamp(0.0, 1.0) as f32;
        out_col[j][74] = (mean_lam_l[j] / denom_p).clamp(0.0, 1.0) as f32;
        out_col[j][75] = max_nongauss[j] as f32;
        out_col[j][76] = max_resid_corr[j] as f32;
        out_col[j][77] = (kappa[j] * 0.5 + 0.5 * (comm4 + comm8 - comm12).clamp(0.0, 1.0)) as f32;
        for s in 68..78 {
            set_mask_bit(&mut out_col_mask[j], s);
        }
    }

    // 6. Global Spectral & Marchenko-Pastur Block G_SPEC_MP (slots 16..40)
    let eig_sum: f64 = eig_vals.iter().map(|v| v.max(0.0)).sum::<f64>().max(1e-9);
    let mut spec_h = 0.0;
    let mut sum_q2 = 0.0;
    let mp_edge = (1.0 + (p as f64 / n.max(1) as f64).sqrt()).powi(2);
    let mut above_mp = 0usize;
    let mut log_det = 0.0;
    for (k, &ev) in eig_vals.iter().enumerate() {
        let q = ev.max(0.0) / eig_sum;
        if q > 1e-12 {
            spec_h -= q * q.ln();
        }
        sum_q2 += q * q;
        if ev > mp_edge {
            above_mp += 1;
        }
        log_det += ev.max(1e-6).ln();
        if k < 12 {
            out_glob[16 + k] = q as f32;
            set_mask_bit(out_glob_mask, 16 + k);
        }
    }
    let erank = spec_h.exp() / p.max(1) as f64;
    let part_ratio = (1.0 / sum_q2.max(1e-9)) / p.max(1) as f64;
    let cond = eig_vals.first().copied().unwrap_or(1.0)
        / eig_vals.last().copied().unwrap_or(1.0).max(1e-6);
    let bartlett_chi2 =
        -((n as f64 - 1.0 - (2.0 * p as f64 + 5.0) / 6.0).max(1.0) * log_det).max(0.0);
    let bartlett_df = (p * p.saturating_sub(1)) as f64 * 0.5;
    let bartlett_z = if bartlett_df > 0.0 {
        wilson_hilferty_z(bartlett_chi2, bartlett_df)
    } else {
        0.0
    };
    let mut sum_r2 = 0.0;
    let mut sum_p2 = 0.0;
    for i in 0..p {
        for j in (i + 1)..p {
            let r = r_lat[i * p + j];
            let pc = -precision[i * p + j]
                / (precision[i * p + i].max(1e-9) * precision[j * p + j].max(1e-9)).sqrt();
            sum_r2 += r * r;
            sum_p2 += pc * pc;
        }
    }
    let kmo = if sum_r2 + sum_p2 > 1e-9 {
        sum_r2 / (sum_r2 + sum_p2)
    } else {
        0.5
    };
    let mean_r_all = if total_pairs > 0 {
        sum_abs_r / total_pairs as f64
    } else {
        0.0
    };
    let frac_sig = if total_pairs > 0 {
        sig_pairs as f64 / total_pairs as f64
    } else {
        0.0
    };
    let frac_collinear = if total_pairs > 0 {
        high_corr_pairs as f64 / total_pairs as f64
    } else {
        0.0
    };
    let eig1_mp = eig_vals.first().copied().unwrap_or(1.0) / mp_edge.max(1.0);

    out_glob[28] = clamp_f32(erank, 0.0, 1.0);
    out_glob[29] = clamp_f32(part_ratio, 0.0, 1.0);
    out_glob[30] = (above_mp as f64 / p.max(1) as f64) as f32;
    out_glob[31] = squash(eig1_mp - 1.0);
    out_glob[32] = squash(cond.log10() / 4.0);
    out_glob[33] = clamp_f32(kmo, 0.0, 1.0);
    out_glob[34] = squash_z(bartlett_z);
    out_glob[35] = clamp_f32(mean_r_all, 0.0, 1.0);
    out_glob[36] = clamp_f32(frac_sig, 0.0, 1.0);
    out_glob[37] = clamp_f32(frac_collinear, 0.0, 1.0);
    out_glob[38] = clamp_f32(lam_prec, 0.0, 1.0);
    out_glob[39] = clamp_f32(
        max_resid_corr.iter().copied().fold(0.0f64, f64::max),
        0.0,
        1.0,
    );
    for s in 28..40 {
        set_mask_bit(out_glob_mask, s);
    }

    let _ = fill_geometry_and_privacy(&z_sub, &precision, p, n, seed, out_glob, out_glob_mask);

    DepContext {
        r_lat,
        eig_vals,
        eig_vecs,
        loadings12,
    }
}

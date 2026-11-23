//! Tier T1 Supervised Evidence: 3-fold OOF Newton-IRLS, class-signed 8-bin `Q_nl8`,
//! S8 chi-square gate, `T03` heteroscedasticity (Breusch-Pagan & Glejser), all-pair signed
//! product & XOR quadrant scan (>=25% exploration quota), and composite screening `S_j`.

use super::dependence_pairs::DepContext;
use super::irls::{fit_irls_subset, predict_irls_rows};
use super::primitives::{
    clamp_f32, dcor_unbiased_1d, kendall_tau_b, miller_madow_mi_and_z, pava_isotonic_r2,
    set_mask_bit, squash, squash_z, value_consistent_bins, SUPPORT_FLOOR_K,
};
use super::support_marginal::ColT0Summary;
use super::target_helpers::{
    best_1d_stump, brown_forsythe_z, eb_8bin_residual, fill_alignment_and_stability,
    fill_class_tokens, grouped_eta2, max_class_ks_z, select_interaction_pairs, xor_quadrant_z,
};
use super::{CLASS_COL_DIM, CLASS_DIM, COL_DIM, EDGE_DIM, GLOB_DIM};
use crate::stats::{chatterjee_xi, pearson, spearman};

/// Output of Tier T1 supervised extraction reused by Tier T2 witness and `DSE8` Step-0.
pub(crate) struct TargetContext {
    pub oof_pred: Vec<Vec<f64>>,
    pub lin_gain: f64,
    pub composite_signal: Vec<f64>,
}

/// Extract Tier T1 supervised target evidence, OOF residuals, signed pair/XOR interactions,
/// class tokens, split-half stability, and composite nonlinear screening score `S_j`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn extract_target_t1(
    x_unit: &[Vec<f32>],
    z_tie: &[Vec<f64>],
    bins16: &[Vec<u8>],
    y: &[f64],
    task_id: u8,
    n_classes: usize,
    t0: &[ColT0Summary],
    dep: &DepContext,
    out_col: &mut [[f32; COL_DIM]],
    out_col_mask: &mut [[u8; 16]],
    out_edge: &mut [[f32; EDGE_DIM]],
    out_edge_mask: &mut [u16],
    out_class: &mut [[f32; CLASS_DIM]],
    out_class_col: &mut [[f32; CLASS_COL_DIM]],
    out_glob: &mut [f32; GLOB_DIM],
    out_glob_mask: &mut [u8; 20],
) -> TargetContext {
    let p = x_unit.len();
    let n = y.len();
    let k_eff = match task_id {
        0 => 1,
        1 => 2,
        _ => n_classes.clamp(2, 8),
    };
    let o_dim = if task_id == 2 { k_eff } else { 1 };

    let n_folds = if n >= 30 { 3 } else { 2 };
    let mut oof_pred = vec![vec![0.0f64; n]; o_dim];
    for f in 0..n_folds {
        let train_rows: Vec<usize> = (0..n).filter(|r| r % n_folds != f).collect();
        let val_rows: Vec<usize> = (0..n).filter(|r| r % n_folds == f).collect();
        let (wf, bf) = fit_irls_subset(x_unit, y, &train_rows, task_id, k_eff);
        let pv = predict_irls_rows(x_unit, &val_rows, &wf, &bf, task_id);
        for c in 0..o_dim {
            for (vi, &r) in val_rows.iter().enumerate() {
                oof_pred[c][r] = pv[c][vi];
            }
        }
    }
    let all_rows: Vec<usize> = (0..n).collect();
    let (lin_w, lin_b) = fit_irls_subset(x_unit, y, &all_rows, task_id, k_eff);

    let mut oof_resid_cls = vec![vec![0.0f64; n]; k_eff];
    let mut oof_resid_scalar = vec![0.0f64; n];
    let (lin_gain, resid_sd) = if task_id == 0 {
        let ym = y.iter().sum::<f64>() / n.max(1) as f64;
        let var_y = (y.iter().map(|v| (v - ym).powi(2)).sum::<f64>() / n.max(1) as f64).max(1e-9);
        let mut raw_e = vec![0.0; n];
        let mut sse = 0.0;
        for r in 0..n {
            let e = y[r] - oof_pred[0][r];
            raw_e[r] = e;
            sse += e * e;
        }
        let sd = (sse / n.max(1) as f64).sqrt().max(1e-4);
        for r in 0..n {
            let es = raw_e[r] / sd;
            oof_resid_scalar[r] = es;
            oof_resid_cls[0][r] = es;
        }
        ((1.0 - (sse / n.max(1) as f64) / var_y).clamp(-1.0, 1.0), sd)
    } else if task_id == 1 {
        let p1 = (y.iter().filter(|&&v| v > 0.5).count() as f64 / n.max(1) as f64)
            .clamp(1e-4, 1.0 - 1e-4);
        let null_ll = -(p1 * p1.ln() + (1.0 - p1) * (1.0 - p1).ln());
        let mut ll = 0.0;
        let mut brier = 0.0;
        for r in 0..n {
            let yr = if y[r] > 0.5 { 1.0 } else { 0.0 };
            let pr = oof_pred[0][r].clamp(1e-5, 1.0 - 1e-5);
            ll -= yr * pr.ln() + (1.0 - yr) * (1.0 - pr).ln();
            brier += (yr - pr).powi(2);
            let e = (yr - pr) / (pr * (1.0 - pr)).max(0.02).sqrt();
            oof_resid_scalar[r] = e;
            oof_resid_cls[1][r] = e;
            oof_resid_cls[0][r] = -e;
        }
        ll /= n.max(1) as f64;
        (
            (1.0 - ll / null_ll.max(1e-6)).clamp(-1.0, 1.0),
            (brier / n.max(1) as f64).sqrt(),
        )
    } else {
        let mut cnts = vec![0.0; k_eff];
        for &v in y {
            cnts[(v.round() as isize).clamp(0, (k_eff - 1) as isize) as usize] += 1.0;
        }
        let null_ll: f64 = cnts
            .iter()
            .map(|&c| {
                let pr = (c / n.max(1) as f64).max(1e-6);
                if c > 0.0 {
                    -pr * pr.ln()
                } else {
                    0.0
                }
            })
            .sum();
        let mut ll = 0.0;
        let mut brier = 0.0;
        for r in 0..n {
            let yi = (y[r].round() as isize).clamp(0, (k_eff - 1) as isize) as usize;
            let pt = oof_pred[yi][r].clamp(1e-5, 1.0 - 1e-5);
            ll -= pt.ln();
            for c in 0..k_eff {
                let yc = if yi == c { 1.0 } else { 0.0 };
                let pc = oof_pred[c][r].clamp(1e-5, 1.0 - 1e-5);
                brier += (yc - pc).powi(2) / k_eff as f64;
                oof_resid_cls[c][r] = (yc - pc) / (pc * (1.0 - pc)).max(0.02).sqrt();
            }
            oof_resid_scalar[r] = oof_resid_cls[yi][r];
        }
        ll /= n.max(1) as f64;
        (
            (1.0 - ll / null_ll.max(1e-6)).clamp(-1.0, 1.0),
            (brier / n.max(1) as f64).sqrt(),
        )
    };

    let (y_bins, ny_bins) = if task_id == 0 {
        let (b, _) = value_consistent_bins(y, 8);
        (b, 8usize)
    } else {
        let b: Vec<u8> = y
            .iter()
            .map(|&v| (v.round() as isize).clamp(0, (k_eff - 1) as isize) as u8)
            .collect();
        (b, k_eff)
    };
    let y_entropy = {
        let mut cnt = vec![0usize; ny_bins];
        for &b in &y_bins {
            if (b as usize) < ny_bins {
                cnt[b as usize] += 1;
            }
        }
        cnt.iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let pr = c as f64 / n.max(1) as f64;
                -pr * pr.ln()
            })
            .sum::<f64>()
            .max(1e-4)
    };

    let mut composite_signal = vec![0.0f64; p];
    let mut s8_passed = 0usize;
    let mut bp_scores = Vec::with_capacity(p);
    let mut lin_w_norms = vec![0.0f64; p];
    let mut tot_w_norm = 0.0f64;
    for j in 0..p {
        let wn = (0..o_dim).map(|c| lin_w[c][j].powi(2)).sum::<f64>().sqrt();
        lin_w_norms[j] = wn;
        tot_w_norm += wn;
    }

    for j in 0..p {
        if t0[j].is_const || t0[j].n_obs < SUPPORT_FLOOR_K {
            continue;
        }
        let xj: Vec<f64> = x_unit[j].iter().map(|&v| v as f64).collect();
        let zj = &z_tie[j];
        let b8: Vec<u8> = bins16[j]
            .iter()
            .map(|&b| if b == 255 { 255 } else { b / 2 })
            .collect();

        let (r_lin, r_sp, tau_b, xi_v, dcor_v, mono_r, eta2_v, bf_z, ks_cls_z, stump_g, stump_q) =
            if task_id != 2 {
                let rp = pearson(&xj, y);
                let rs = spearman(&xj, y);
                let tb = if n <= 1024 {
                    kendall_tau_b(&xj, y)
                } else {
                    (2.0 / std::f64::consts::PI) * rs.asin()
                };
                let xi = chatterjee_xi(&xj, y).clamp(-0.2, 1.0);
                let dc = if n <= 1024 {
                    dcor_unbiased_1d(zj, y)
                } else {
                    rs.abs()
                };
                let r2_iso = pava_isotonic_r2(&xj, y);
                let r2_bin = grouped_eta2(y, &b8, 8);
                let mono = (r2_iso / r2_bin.max(0.02)).clamp(0.0, 1.0);
                let eta2 = grouped_eta2(zj, &y_bins, ny_bins);
                let bf = brown_forsythe_z(zj, &y_bins, ny_bins);
                let ks = max_class_ks_z(zj, &y_bins, ny_bins);
                let (sg, sq) = best_1d_stump(y, &bins16[j]);
                (rp, rs, tb, xi, dc, mono, eta2, bf, ks, sg, sq)
            } else {
                let (mut br, mut bsp, mut btau, mut bxi, mut bdc, mut bsg, mut bsq) =
                    (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.5f64);
                for c in 0..k_eff {
                    let yc: Vec<f64> = y
                        .iter()
                        .map(|&v| {
                            if (v.round() as isize) == c as isize {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    br = br.max(pearson(&xj, &yc).abs());
                    bsp = bsp.max(spearman(&xj, &yc).abs());
                    if n <= 1024 {
                        btau = btau.max(kendall_tau_b(&xj, &yc).abs());
                        bdc = bdc.max(dcor_unbiased_1d(zj, &yc));
                    }
                    bxi = bxi.max(chatterjee_xi(&xj, &yc).clamp(0.0, 1.0));
                    let (sg, sq) = best_1d_stump(&yc, &bins16[j]);
                    if sg > bsg {
                        bsg = sg;
                        bsq = sq;
                    }
                }
                let eta2 = grouped_eta2(zj, &y_bins, k_eff);
                let bf = brown_forsythe_z(zj, &y_bins, k_eff);
                let ks = max_class_ks_z(zj, &y_bins, k_eff);
                (br, bsp, btau, bxi, bdc, 0.5, eta2, bf, ks, bsg, bsq)
            };

        let (mi_log, mi_z) = miller_madow_mi_and_z(&b8, &y_bins, 8, ny_bins);
        let mi_norm = (mi_log / y_entropy).clamp(0.0, 1.0);
        let miss_y = if t0[j].miss_rate > 0.0 {
            let mj: Vec<f64> = x_unit[j]
                .iter()
                .map(|v| if v.is_finite() { 0.0 } else { 1.0 })
                .collect();
            if task_id != 2 {
                pearson(&mj, y).abs()
            } else {
                grouped_eta2(&mj, &y_bins, k_eff).sqrt()
            }
        } else {
            0.0
        };

        composite_signal[j] = r_lin
            .abs()
            .max(r_sp.abs())
            .max(xi_v.max(0.0).sqrt())
            .max(eta2_v.max(0.0).sqrt())
            .max(stump_g.max(0.0).sqrt())
            .max(miss_y);

        let resid_j: &[f64] = if task_id != 2 {
            &oof_resid_scalar
        } else {
            let mut best_c = 0usize;
            let mut best_var = -1.0f64;
            for c in 0..k_eff {
                let (q8, _) = eb_8bin_residual(&oof_resid_cls[c], &b8);
                let mu = q8.iter().sum::<f64>() / 8.0;
                let v = q8.iter().map(|x| (x - mu).powi(2)).sum::<f64>();
                if v > best_var {
                    best_var = v;
                    best_c = c;
                }
            }
            &oof_resid_cls[best_c]
        };

        let (q_nl8, s8_chi2) = eb_8bin_residual(resid_j, &b8);
        if s8_chi2 > 14.067 {
            s8_passed += 1;
        }
        let q_mean = q_nl8.iter().sum::<f64>() / 8.0;
        let ale_curv = q_nl8.iter().map(|x| (x - q_mean).abs()).sum::<f64>() / 8.0;

        let abs_e: Vec<f64> = resid_j.iter().map(|v| v.abs()).collect();
        let e2_m1: Vec<f64> = resid_j.iter().map(|v| (v * v).min(25.0) - 1.0).collect();
        let z2_m1: Vec<f64> = zj.iter().map(|v| v * v - 1.0).collect();
        let glejser = pearson(&abs_e, zj);
        let bp_lin = pearson(&e2_m1, zj);
        let bp_quad = pearson(&e2_m1, &z2_m1);
        let bp_r2 = (bp_lin * bp_lin + bp_quad * bp_quad).clamp(0.0, 1.0);
        bp_scores.push(bp_r2);

        let quad_z = pearson(resid_j, &z2_m1) * (n.max(4) as f64 - 3.0).sqrt();
        let sat_v: Vec<f64> = zj.iter().map(|&z| (2.0 * z).asinh()).collect();
        let sat_r = pearson(resid_j, &sat_v).abs();
        let (res_stump_g, _) = best_1d_stump(resid_j, &bins16[j]);

        let n_sqrt = (n.max(4) as f64 - 3.0).sqrt();
        out_col[j][78] = clamp_f32(r_lin, -1.0, 1.0);
        out_col[j][79] = squash_z(r_lin.clamp(-0.999, 0.999).atanh() * n_sqrt);
        out_col[j][80] = clamp_f32(r_sp, -1.0, 1.0);
        out_col[j][81] = clamp_f32(tau_b, -1.0, 1.0);
        out_col[j][82] = clamp_f32(xi_v, -0.2, 1.0);
        out_col[j][83] = squash_z(xi_v * (2.5 * n as f64).sqrt());
        out_col[j][84] = clamp_f32(dcor_v, 0.0, 1.0);
        out_col[j][85] = clamp_f32(mi_norm, 0.0, 1.0);
        out_col[j][86] = squash_z(mi_z);
        out_col[j][87] = clamp_f32(mono_r, 0.0, 1.0);
        out_col[j][88] = clamp_f32(eta2_v, 0.0, 1.0);
        out_col[j][89] = squash_z(bf_z);
        out_col[j][90] = squash_z(ks_cls_z);
        out_col[j][91] = clamp_f32(stump_g, 0.0, 1.0);
        out_col[j][92] = clamp_f32(stump_q * 2.0 - 1.0, -1.0, 1.0);
        out_col[j][93] = clamp_f32(miss_y, 0.0, 1.0);
        out_col[j][94] = clamp_f32(lin_w_norms[j] / (tot_w_norm + 1e-6), 0.0, 1.0);
        out_col[j][95] = clamp_f32(composite_signal[j], 0.0, 1.0);
        for s in 78..96 {
            set_mask_bit(&mut out_col_mask[j], s);
        }

        for b in 0..8 {
            out_col[j][96 + b] = q_nl8[b].tanh() as f32;
        }
        out_col[j][104] = squash(s8_chi2 / 14.067);
        out_col[j][105] = clamp_f32(ale_curv, 0.0, 2.0);
        out_col[j][106] = clamp_f32(bp_r2, 0.0, 1.0);
        out_col[j][107] = clamp_f32(glejser, -1.0, 1.0);
        out_col[j][108] = squash_z(quad_z);
        out_col[j][109] = clamp_f32(sat_r, 0.0, 1.0);
        out_col[j][110] = clamp_f32(res_stump_g, 0.0, 1.0);
        for c in 0..o_dim.min(5) {
            out_col[j][111 + c] = squash(lin_w[c][j]);
        }
        for s in 96..116 {
            set_mask_bit(&mut out_col_mask[j], s);
        }
    }

    // All-Pair Signed Product & Depth-2 XOR Quadrant Interaction Scan (>=25% exploration quota)
    let mut s10_passed = 0usize;
    let mut total_scanned = 0usize;
    for (i, j) in select_interaction_pairs(p, &composite_signal) {
        if t0[i].is_const || t0[j].is_const || n < 4 * SUPPORT_FLOOR_K {
            continue;
        }
        total_scanned += 1;
        let rl = dep.r_lat[i * p + j];
        let prod: Vec<f64> = (0..n).map(|r| z_tie[i][r] * z_tie[j][r] - rl).collect();
        let mut best_prod_z = 0.0f64;
        let mut best_xor_z = 0.0f64;
        let n_ch = if task_id == 2 { k_eff } else { 1 };
        for c in 0..n_ch {
            let e_c = if task_id == 2 {
                &oof_resid_cls[c]
            } else {
                &oof_resid_scalar
            };
            let pz = pearson(&prod, e_c) * (n.max(4) as f64).sqrt();
            if pz.abs() > best_prod_z.abs() {
                best_prod_z = pz;
            }
            let xz = xor_quadrant_z(&z_tie[i], &z_tie[j], e_c);
            if xz.abs() > best_xor_z.abs() {
                best_xor_z = xz;
            }
        }
        let s_prod = squash(best_prod_z / 3.0);
        let s_xor = squash(best_xor_z / 3.0);
        if s_prod.abs() > 0.55 || s_xor.abs() > 0.55 {
            s10_passed += 1;
        }
        out_edge[i * p + j][9] = s_prod;
        out_edge[j * p + i][9] = s_prod;
        out_edge[i * p + j][10] = s_xor;
        out_edge[j * p + i][10] = s_xor;
        out_edge_mask[i * p + j] |= (1 << 9) | (1 << 10);
        out_edge_mask[j * p + i] |= (1 << 9) | (1 << 10);
    }

    fill_class_tokens(
        z_tie,
        bins16,
        y,
        &oof_pred,
        task_id,
        k_eff,
        dep,
        out_class,
        out_class_col,
    );

    out_glob[task_id.min(2) as usize] = 1.0;
    for c in 0..k_eff.min(8) {
        let cnt = if task_id == 0 {
            n
        } else {
            y.iter()
                .filter(|&&v| (v.round() as isize) == c as isize)
                .count()
        };
        out_glob[4 + c] = cnt as f32 / n.max(1) as f32;
    }
    out_glob[12] = ((n.clamp(16, 16384) as f64).log2() / 14.0) as f32;
    out_glob[13] = (p as f64 / 48.0).clamp(0.0, 2.0) as f32;
    out_glob[14] = ((p as f64 / n.max(1) as f64).log2().clamp(-12.0, 4.0) / 12.0) as f32;
    out_glob[15] = (k_eff as f64 / 8.0) as f32;
    for s in 0..16 {
        set_mask_bit(out_glob_mask, s);
    }

    bp_scores.sort_by(|a, b| b.total_cmp(a));
    let top4_bp = if !bp_scores.is_empty() {
        bp_scores.iter().take(4).sum::<f64>() / bp_scores.len().min(4) as f64
    } else {
        0.0
    };
    out_glob[64] = clamp_f32(lin_gain, -1.0, 1.0);
    out_glob[65] = clamp_f32(resid_sd / 2.0, 0.0, 2.0);
    out_glob[66] = clamp_f32(top4_bp, 0.0, 1.0);
    out_glob[67] = (s8_passed as f64 / p.max(1) as f64) as f32;
    out_glob[68] = (s10_passed as f64 / total_scanned.max(1) as f64) as f32;
    out_glob[69] = clamp_f32(y_entropy / (ny_bins.max(2) as f64).ln(), 0.0, 1.0);
    for s in 64..70 {
        set_mask_bit(out_glob_mask, s);
    }

    fill_alignment_and_stability(
        x_unit,
        y,
        task_id,
        k_eff,
        &lin_w,
        dep,
        out_glob,
        out_glob_mask,
    );

    let _ = lin_b;
    TargetContext {
        oof_pred,
        lin_gain,
        composite_signal,
    }
}

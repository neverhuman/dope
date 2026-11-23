//! Helper diagnostics for Tier T1 supervised evidence: grouped ANOVA eta^2, Brown-Forsythe
//! dispersion Z, class KS Z, 1D decision stump, class tokens + YC, alignment, and split-half stability.

use super::dependence_pairs::DepContext;
use super::irls::fit_irls_subset;
use super::primitives::{set_mask_bit, squash, wilson_hilferty_z, SUPPORT_FLOOR_K};
use super::{CLASS_COL_DIM, CLASS_DIM, GLOB_DIM};

pub(crate) fn grouped_eta2(vals: &[f64], groups: &[u8], n_groups: usize) -> f64 {
    let mut sums = vec![0.0; n_groups];
    let mut cnts = vec![0usize; n_groups];
    let mut tot = 0.0;
    let mut n = 0usize;
    for (&v, &g) in vals.iter().zip(groups) {
        if v.is_finite() && (g as usize) < n_groups {
            sums[g as usize] += v;
            cnts[g as usize] += 1;
            tot += v;
            n += 1;
        }
    }
    if n < SUPPORT_FLOOR_K {
        return 0.0;
    }
    let grand = tot / n as f64;
    let ss_tot: f64 = vals
        .iter()
        .zip(groups)
        .filter(|(v, &g)| v.is_finite() && (g as usize) < n_groups)
        .map(|(&v, _)| (v - grand).powi(2))
        .sum();
    if ss_tot <= 1e-12 {
        return 0.0;
    }
    let mut ss_bet = 0.0;
    for g in 0..n_groups {
        if cnts[g] > 0 {
            let mg = sums[g] / cnts[g] as f64;
            ss_bet += cnts[g] as f64 * (mg - grand).powi(2);
        }
    }
    (ss_bet / ss_tot).clamp(0.0, 1.0)
}

pub(crate) fn brown_forsythe_z(vals: &[f64], groups: &[u8], n_groups: usize) -> f64 {
    let mut by_g: Vec<Vec<f64>> = vec![Vec::new(); n_groups];
    for (&v, &g) in vals.iter().zip(groups) {
        if v.is_finite() && (g as usize) < n_groups {
            by_g[g as usize].push(v);
        }
    }
    let mut devs = Vec::with_capacity(vals.len());
    let mut dev_g = Vec::with_capacity(vals.len());
    let mut active = 0usize;
    for (g, arr) in by_g.iter_mut().enumerate() {
        if arr.len() >= SUPPORT_FLOOR_K {
            active += 1;
            arr.sort_by(f64::total_cmp);
            let med = arr[arr.len() / 2];
            for &v in arr.iter() {
                devs.push((v - med).abs());
                dev_g.push(g as u8);
            }
        }
    }
    if active < 2 || devs.len() <= active + 2 {
        return 0.0;
    }
    let eta2 = grouped_eta2(&devs, &dev_g, n_groups);
    let df1 = (active - 1) as f64;
    let df2 = (devs.len() - active) as f64;
    let f_stat = (eta2 / df1) / ((1.0 - eta2).max(1e-6) / df2);
    wilson_hilferty_z(f_stat * df1, df1)
}

pub(crate) fn max_class_ks_z(vals: &[f64], groups: &[u8], n_groups: usize) -> f64 {
    let mut best_z = 0.0f64;
    for g in 0..n_groups {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for (&v, &gr) in vals.iter().zip(groups) {
            if !v.is_finite() {
                continue;
            }
            if gr as usize == g {
                a.push(v);
            } else {
                b.push(v);
            }
        }
        if a.len() < SUPPORT_FLOOR_K || b.len() < SUPPORT_FLOOR_K {
            continue;
        }
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        let (mut i, mut j, mut ks) = (0usize, 0usize, 0.0f64);
        while i < a.len() && j < b.len() {
            if a[i] <= b[j] {
                i += 1;
            } else {
                j += 1;
            }
            ks = ks.max(((i as f64 / a.len() as f64) - (j as f64 / b.len() as f64)).abs());
        }
        let n_eff = (a.len() as f64 * b.len() as f64) / (a.len() + b.len()) as f64;
        best_z = best_z.max(ks * n_eff.sqrt());
    }
    best_z
}

pub(crate) fn best_1d_stump(y: &[f64], bins16: &[u8]) -> (f64, f64) {
    let mut sums = [0.0f64; 16];
    let mut cnts = [0usize; 16];
    let mut tot_y = 0.0;
    let mut tot_y2 = 0.0;
    let mut n = 0usize;
    for (&yv, &b) in y.iter().zip(bins16) {
        if (b as usize) < 16 && yv.is_finite() {
            sums[b as usize] += yv;
            cnts[b as usize] += 1;
            tot_y += yv;
            tot_y2 += yv * yv;
            n += 1;
        }
    }
    if n < 2 * SUPPORT_FLOOR_K {
        return (0.0, 0.5);
    }
    let nf = n as f64;
    let var_tot = (tot_y2 - tot_y * tot_y / nf).max(1e-12);
    let mut best_gain = 0.0f64;
    let mut best_q = 0.5f64;
    let mut left_s = 0.0;
    let mut left_c = 0usize;
    for b in 0..15 {
        left_s += sums[b];
        left_c += cnts[b];
        let right_c = n - left_c;
        if left_c < SUPPORT_FLOOR_K || right_c < SUPPORT_FLOOR_K {
            continue;
        }
        let right_s = tot_y - left_s;
        let between = left_s * left_s / left_c as f64 + right_s * right_s / right_c as f64
            - tot_y * tot_y / nf;
        let gain = (between / var_tot).clamp(0.0, 1.0);
        if gain > best_gain {
            best_gain = gain;
            best_q = left_c as f64 / nf;
        }
    }
    (best_gain, best_q)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_class_tokens(
    z_tie: &[Vec<f64>],
    bins16: &[Vec<u8>],
    y: &[f64],
    oof_pred: &[Vec<f64>],
    task_id: u8,
    k_eff: usize,
    dep: &DepContext,
    out_class: &mut [[f32; CLASS_DIM]],
    out_class_col: &mut [[f32; CLASS_COL_DIM]],
) {
    let p = z_tie.len();
    let n = y.len();
    if task_id == 0 || n < SUPPORT_FLOOR_K {
        return;
    }
    let mut centroids = vec![vec![0.0f64; p]; k_eff];
    let mut cls_cnt = vec![0usize; k_eff];
    for r in 0..n {
        let c = (y[r].round() as isize).clamp(0, (k_eff - 1) as isize) as usize;
        cls_cnt[c] += 1;
        for j in 0..p {
            centroids[c][j] += z_tie[j][r];
        }
    }
    for c in 0..k_eff {
        if cls_cnt[c] > 0 {
            for j in 0..p {
                centroids[c][j] /= cls_cnt[c] as f64;
            }
        }
        let prior = cls_cnt[c] as f64 / n.max(1) as f64;
        let c_norm =
            centroids[c].iter().map(|v| v * v).sum::<f64>().sqrt() / (p.max(1) as f64).sqrt();
        let mut hits = 0usize;
        let mut margin_sum = 0.0;
        for r in 0..n {
            let yi = (y[r].round() as isize).clamp(0, (k_eff - 1) as isize) as usize;
            if yi == c {
                let pc = if task_id == 1 {
                    if c == 1 {
                        oof_pred[0][r]
                    } else {
                        1.0 - oof_pred[0][r]
                    }
                } else {
                    oof_pred[c][r]
                };
                if pc >= 0.5 / (k_eff as f64 / 2.0).max(1.0) {
                    hits += 1;
                }
                margin_sum += pc;
            }
        }
        let recall = if cls_cnt[c] > 0 {
            hits as f64 / cls_cnt[c] as f64
        } else {
            0.0
        };
        let mean_p = if cls_cnt[c] > 0 {
            margin_sum / cls_cnt[c] as f64
        } else {
            0.0
        };

        out_class[c][0] = prior as f32;
        out_class[c][1] = ((cls_cnt[c].max(1) as f64).log2() / 12.0) as f32;
        out_class[c][2] = if cls_cnt[c] < SUPPORT_FLOOR_K {
            1.0
        } else {
            0.0
        };
        out_class[c][3] = squash(c_norm);
        out_class[c][4] = recall as f32;
        out_class[c][5] = mean_p as f32;
        for f in 0..8.min(p) {
            let dot: f64 = (0..p).map(|j| centroids[c][j] * dep.loadings12[j][f]).sum();
            out_class[c][6 + f] = squash(dot / (p.max(1) as f64).sqrt());
        }
        for j in 0..p {
            let mut b_hits = [0usize; 4];
            let mut b_tot = [0usize; 4];
            for r in 0..n {
                let b = bins16[j][r];
                if b < 16 {
                    let q = (b / 4) as usize;
                    b_tot[q] += 1;
                    if (y[r].round() as isize) == c as isize {
                        b_hits[q] += 1;
                    }
                }
            }
            for q in 0..4 {
                let rate = (b_hits[q] as f64 + 8.0 * prior) / (b_tot[q] as f64 + 8.0);
                out_class_col[c * p + j][q] = (rate - prior).clamp(-1.0, 1.0) as f32;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_alignment_and_stability(
    x_unit: &[Vec<f32>],
    y: &[f64],
    task_id: u8,
    k_eff: usize,
    lin_w: &[Vec<f64>],
    dep: &DepContext,
    out_glob: &mut [f32; GLOB_DIM],
    out_glob_mask: &mut [u8; 20],
) {
    let p = x_unit.len();
    let n = y.len();
    if p == 0 || n < 16 {
        return;
    }
    let mut tot_e = 0.0;
    let mut cum = [0.0f64; 5];
    let cuts = [1usize, 2, 4, 8, 12];
    for k in 0..p {
        let ev = dep.eig_vals[k].max(0.0);
        let mut proj_sq = 0.0;
        for w_c in lin_w {
            let dot: f64 = (0..p).map(|j| w_c[j] * dep.eig_vecs[j * p + k]).sum();
            proj_sq += dot * dot;
        }
        let energy = ev * proj_sq;
        tot_e += energy;
        for (ci, &cut) in cuts.iter().enumerate() {
            if k < cut {
                cum[ci] += energy;
            }
        }
    }
    if tot_e > 1e-12 {
        for ci in 0..5 {
            out_glob[88 + ci] = (cum[ci] / tot_e).clamp(0.0, 1.0) as f32;
            set_mask_bit(out_glob_mask, 88 + ci);
        }
    }

    let half1: Vec<usize> = (0..n).step_by(2).collect();
    let half2: Vec<usize> = (1..n).step_by(2).collect();
    if half1.len() >= SUPPORT_FLOOR_K && half2.len() >= SUPPORT_FLOOR_K {
        let (w1, _) = fit_irls_subset(x_unit, y, &half1, task_id, k_eff);
        let (w2, _) = fit_irls_subset(x_unit, y, &half2, task_id, k_eff);
        let mut dot = 0.0;
        let mut n1 = 0.0;
        let mut n2 = 0.0;
        for c in 0..w1.len() {
            for j in 0..p {
                dot += w1[c][j] * w2[c][j];
                n1 += w1[c][j] * w1[c][j];
                n2 += w2[c][j] * w2[c][j];
            }
        }
        let cos = if n1 > 1e-12 && n2 > 1e-12 {
            (dot / (n1 * n2).sqrt()).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        out_glob[94] = cos as f32;
        set_mask_bit(out_glob_mask, 94);
    }
}

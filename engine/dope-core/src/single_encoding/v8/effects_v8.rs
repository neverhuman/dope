//! Sequential decode-aligned 8-bin main effects (`E`) and gated product + XOR quadrant
//! pairwise interactions (`Pr`) for `DSE8` v1.

use super::calibrate::{centered_working_resid, eval_true_loss, CdfIntervals};
use super::target_v8::{Effect1DV8, PairIntV8, V8_MAX_EFFECTS, V8_MAX_PAIRS};
use crate::evidence::primitives::SUPPORT_FLOOR_K;
use crate::single_encoding::math::{f16_bits_to_f32, f32_to_f16_bits};
use crate::stats::pearson;

/// Fit sequential decode-aligned 8-bin nonlinear main effects (`E`) with inner-val loss gating.
#[allow(clippy::too_many_arguments)]
pub fn fit_effects_v8(
    cdf: &CdfIntervals,
    cur_logits: &mut [Vec<f64>],
    y_sc: &[f64],
    tr_rows: &[usize],
    va_rows: &[usize],
    task_id: u8,
    temp: f32,
    max_effects: usize,
) -> Vec<Effect1DV8> {
    let p = cdf.p;
    let o_dim = cur_logits.len();
    let cap = max_effects.min(V8_MAX_EFFECTS).min(16);
    if p == 0 || tr_rows.len() < 4 * SUPPORT_FLOOR_K || va_rows.len() < 10 || cap == 0 {
        return Vec::new();
    }
    let base_va = eval_true_loss(cur_logits, y_sc, va_rows, task_id, temp);
    let mut best_va = base_va;
    let min_improv = (0.002 * base_va.abs()).max(1e-5);
    let mut used = vec![vec![false; o_dim]; p];
    let mut out = Vec::new();

    for _ in 0..cap {
        let resid = centered_working_resid(cur_logits, y_sc, tr_rows, task_id, temp);
        let mut cands: Vec<Effect1DV8> = Vec::new();
        for j in 0..p {
            for c in 0..o_dim {
                if used[j][c] {
                    continue;
                }
                let e = &resid[c];
                let sd_e = (tr_rows.iter().map(|&r| e[r] * e[r]).sum::<f64>()
                    / tr_rows.len() as f64)
                    .sqrt()
                    .max(1e-4);
                let mut sums = [0.0f64; 8];
                let mut cnts = [0.0f64; 8];
                for &r in tr_rows {
                    if !e[r].is_finite() {
                        continue;
                    }
                    let (b0, b1, w) =
                        CdfIntervals::bin_weights::<8>(cdf.u_lo[j][r], cdf.u_hi[j][r]);
                    for b in b0..=b1 {
                        sums[b] += w[b] * e[r];
                        cnts[b] += w[b];
                    }
                }
                let mut bins = [0.0f64; 8];
                let mut chi2 = 0.0;
                let mut w_tot = 0.0;
                let mut w_sum = 0.0;
                for b in 0..8 {
                    if cnts[b] >= SUPPORT_FLOOR_K as f64 {
                        let mu = sums[b] / cnts[b];
                        chi2 += cnts[b] * (mu / sd_e).powi(2);
                        let eb = (0.55 * sums[b] / (cnts[b] + 10.0)).clamp(-2.0, 2.0);
                        bins[b] = eb;
                        w_sum += cnts[b] * eb;
                        w_tot += cnts[b];
                    }
                }
                if chi2 <= 16.0 || w_tot <= 0.0 {
                    continue;
                }
                let mean_b = w_sum / w_tot;
                let mut bins_f32 = [0.0f32; 8];
                let mut var_red = 0.0;
                for b in 0..8 {
                    if cnts[b] >= SUPPORT_FLOOR_K as f64 {
                        let v = (bins[b] - mean_b).clamp(-2.0, 2.0);
                        bins_f32[b] = v as f32;
                        var_red += cnts[b] * v * v;
                    }
                }
                let max_a = bins_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                if max_a < 1e-4 {
                    continue;
                }
                let scale = f16_bits_to_f32(f32_to_f16_bits(max_a / 127.0)).max(1e-5);
                for slot in &mut bins_f32 {
                    let q = (*slot / scale).round().clamp(-127.0, 127.0) as i8;
                    *slot = q as f32 * scale;
                }
                cands.push(Effect1DV8 {
                    col_idx: j as u8,
                    cls_idx: c as u8,
                    scale,
                    bins8: bins_f32,
                    utility: var_red / tr_rows.len() as f64,
                });
            }
        }
        cands.sort_by(|a, b| b.utility.total_cmp(&a.utility));
        let mut accepted = false;
        for cand in cands.into_iter().take(3) {
            let j = cand.col_idx as usize;
            let c = cand.cls_idx as usize;
            let mut backup = Vec::with_capacity(va_rows.len());
            for &r in va_rows {
                backup.push(cur_logits[c][r]);
                cur_logits[c][r] += expected_effect_val(&cand, cdf.u_lo[j][r], cdf.u_hi[j][r]);
            }
            let va_loss = eval_true_loss(cur_logits, y_sc, va_rows, task_id, temp);
            if va_loss + min_improv < best_va {
                best_va = va_loss;
                for &r in tr_rows {
                    cur_logits[c][r] += expected_effect_val(&cand, cdf.u_lo[j][r], cdf.u_hi[j][r]);
                }
                used[j][c] = true;
                out.push(cand);
                accepted = true;
                break;
            }
            for (&r, old) in va_rows.iter().zip(backup) {
                cur_logits[c][r] = old;
            }
        }
        if !accepted {
            break;
        }
    }
    out
}

#[inline]
fn expected_effect_val(ef: &Effect1DV8, u_lo: f32, u_hi: f32) -> f64 {
    let (b0, b1, w) = CdfIntervals::bin_weights::<8>(u_lo, u_hi);
    let mut s = 0.0;
    for b in b0..=b1 {
        s += w[b] * ef.bins8[b] as f64;
    }
    s
}

/// Fit sequential gated product + XOR quadrant pairs (`Pr`) with inner-val loss gating.
#[allow(clippy::too_many_arguments)]
pub fn fit_pairs_v8(
    cdf: &CdfIntervals,
    cur_logits: &mut [Vec<f64>],
    y_sc: &[f64],
    tr_rows: &[usize],
    va_rows: &[usize],
    task_id: u8,
    temp: f32,
    max_pairs: usize,
) -> Vec<PairIntV8> {
    let p_scan = cdf.p.min(32);
    let o_dim = cur_logits.len();
    let cap = max_pairs.min(V8_MAX_PAIRS).min(12);
    if p_scan < 2 || tr_rows.len() < 4 * SUPPORT_FLOOR_K || va_rows.len() < 10 || cap == 0 {
        return Vec::new();
    }
    let base_va = eval_true_loss(cur_logits, y_sc, va_rows, task_id, temp);
    let mut best_va = base_va;
    let min_improv = (0.002 * base_va.abs()).max(1e-5);
    let mut used = std::collections::BTreeSet::<(usize, usize, usize)>::new();
    let mut out = Vec::new();
    let n_tr = tr_rows.len() as f64;

    for _ in 0..cap {
        let resid = centered_working_resid(cur_logits, y_sc, tr_rows, task_id, temp);
        let mut cands: Vec<(PairIntV8, f64)> = Vec::new();
        for i in 0..p_scan {
            for j in (i + 1)..p_scan {
                let mut prod_tr = Vec::with_capacity(tr_rows.len());
                let mut sum_p = 0.0;
                for &r in tr_rows {
                    let pr = (cdf.z_mid[i][r].clamp(-3.0, 3.0) * cdf.z_mid[j][r].clamp(-3.0, 3.0))
                        .clamp(-6.0, 6.0);
                    prod_tr.push(pr);
                    sum_p += pr;
                }
                let mean_p = sum_p / n_tr;
                let prod_c: Vec<f64> = prod_tr.iter().map(|&v| v - mean_p).collect();
                let prod_var = (prod_c.iter().map(|v| v * v).sum::<f64>() / n_tr).max(1e-4);

                for c in 0..o_dim {
                    if used.contains(&(i, j, c)) {
                        continue;
                    }
                    let e_tr: Vec<f64> = tr_rows.iter().map(|&r| resid[c][r]).collect();
                    let sd_e = (e_tr.iter().map(|v| v * v).sum::<f64>() / n_tr)
                        .sqrt()
                        .max(1e-4);
                    let rp = pearson(&prod_c, &e_tr);
                    let zp = rp * n_tr.sqrt();

                    let mut q_sum = [0.0f64; 4];
                    let mut q_cnt = [0usize; 4];
                    for (idx, &r) in tr_rows.iter().enumerate() {
                        let q = (usize::from(cdf.z_mid[i][r] > 0.0) << 1)
                            | usize::from(cdf.z_mid[j][r] > 0.0);
                        q_sum[q] += e_tr[idx];
                        q_cnt[q] += 1;
                    }
                    let (zx, w_xor_raw) = if q_cnt.iter().all(|&cnt| cnt >= SUPPORT_FLOOR_K) {
                        let m00 = q_sum[0] / q_cnt[0] as f64;
                        let m01 = q_sum[1] / q_cnt[1] as f64;
                        let m10 = q_sum[2] / q_cnt[2] as f64;
                        let m11 = q_sum[3] / q_cnt[3] as f64;
                        let contrast = 0.25 * ((m11 + m00) - (m10 + m01));
                        let se = (sd_e * 0.25)
                            * q_cnt
                                .iter()
                                .map(|&cnt| 1.0 / cnt as f64)
                                .sum::<f64>()
                                .sqrt();
                        (contrast / se.max(1e-6), contrast)
                    } else {
                        (0.0, 0.0)
                    };
                    if zp.abs() <= 3.2 && zx.abs() <= 3.2 {
                        continue;
                    }
                    let w_p = if zp.abs() > 3.2 {
                        let cov = prod_c.iter().zip(&e_tr).map(|(a, b)| a * b).sum::<f64>() / n_tr;
                        (0.45 * cov / prod_var).clamp(-1.25, 1.25)
                    } else {
                        0.0
                    };
                    let w_x = if zx.abs() > 3.2 {
                        (0.50 * w_xor_raw).clamp(-1.25, 1.25)
                    } else {
                        0.0
                    };
                    let qp = (w_p * 100.0).round().clamp(-125.0, 125.0) as i8;
                    let qx = (w_x * 100.0).round().clamp(-125.0, 125.0) as i8;
                    if qp != 0 || qx != 0 {
                        let wp_f = qp as f32 / 100.0;
                        let wx_f = qx as f32 / 100.0;
                        let mean_shift = wp_f as f64 * mean_p;
                        cands.push((
                            PairIntV8 {
                                col_i: i as u8,
                                col_j: j as u8,
                                cls_idx: c as u8,
                                w_prod: wp_f,
                                w_xor: wx_f,
                                utility: zp * zp + zx * zx,
                            },
                            mean_shift,
                        ));
                    }
                }
            }
        }
        cands.sort_by(|a, b| b.0.utility.total_cmp(&a.0.utility));
        let mut accepted = false;
        for (cand, mean_shift) in cands.into_iter().take(3) {
            let i = cand.col_i as usize;
            let j = cand.col_j as usize;
            let c = cand.cls_idx as usize;
            let mut backup = Vec::with_capacity(va_rows.len());
            for &r in va_rows {
                backup.push(cur_logits[c][r]);
                cur_logits[c][r] +=
                    pair_delta_centered(&cand, cdf.z_mid[i][r], cdf.z_mid[j][r], mean_shift);
            }
            let va_loss = eval_true_loss(cur_logits, y_sc, va_rows, task_id, temp);
            if va_loss + min_improv < best_va {
                best_va = va_loss;
                for &r in tr_rows {
                    cur_logits[c][r] +=
                        pair_delta_centered(&cand, cdf.z_mid[i][r], cdf.z_mid[j][r], mean_shift);
                }
                used.insert((i, j, c));
                out.push(cand);
                accepted = true;
                break;
            }
            for (&r, old) in va_rows.iter().zip(backup) {
                cur_logits[c][r] = old;
            }
        }
        if !accepted {
            break;
        }
    }
    out
}

#[inline]
fn pair_delta_centered(pr: &PairIntV8, zi: f64, zj: f64, mean_shift: f64) -> f64 {
    let zic = zi.clamp(-3.0, 3.0);
    let zjc = zj.clamp(-3.0, 3.0);
    let xs = if (zic > 0.0) == (zjc > 0.0) {
        1.0
    } else {
        -1.0
    };
    pr.w_prod as f64 * (zic * zjc).clamp(-6.0, 6.0) + pr.w_xor as f64 * xs - mean_shift
}

/// Evaluate analytic effects `E` + pairs `Pr` delta at a single row `z_lat`.
#[inline]
pub fn eval_effects_and_pairs(
    effects: &[Effect1DV8],
    pairs: &[PairIntV8],
    z_lat: &[f32],
    out_delta: &mut [f32],
) {
    let p = z_lat.len();
    for ef in effects {
        let j = ef.col_idx as usize;
        let c = ef.cls_idx as usize;
        if j < p && c < out_delta.len() {
            let u = crate::stats::norm_cdf(z_lat[j] as f64);
            let b = ((u * 8.0).floor() as usize).min(7);
            out_delta[c] += ef.bins8[b];
        }
    }
    for pr in pairs {
        let i = pr.col_i as usize;
        let j = pr.col_j as usize;
        let c = pr.cls_idx as usize;
        if i < p && j < p && c < out_delta.len() {
            let zi = z_lat[i].clamp(-3.0, 3.0);
            let zj = z_lat[j].clamp(-3.0, 3.0);
            let xor_sign = if (zi > 0.0) == (zj > 0.0) {
                1.0f32
            } else {
                -1.0f32
            };
            out_delta[c] += pr.w_prod * (zi * zj).clamp(-6.0, 6.0) + pr.w_xor * xor_sign;
        }
    }
}

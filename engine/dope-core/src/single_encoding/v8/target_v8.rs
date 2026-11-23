//! Target conditional sections for `DSE8` v1 (`W` Newton-IRLS linear, `E` gated S8 8-bin
//! main effects, `Pr` gated S10 product + XOR quadrant pairs, `Rs` heteroscedastic residual
//! model, and `Z` nested FSQ neural code + JopeDime embedding export).
//!
//! Every packed section (`W`, `E`, `Pr`, `Rs`) is **analytically decoded** in Step-0!

use crate::evidence::irls::{fit_irls_subset, predict_irls_rows};
use crate::evidence::primitives::{value_consistent_bins, SUPPORT_FLOOR_K};
use crate::single_encoding::math::{f16_bits_to_f32, f32_to_f16_bits};
use crate::stats::{pearson, quantile_sorted};

pub const V8_MAX_EFFECTS: usize = 24;
pub const V8_MAX_PAIRS: usize = 24;
pub const V8_MAX_RESID_COLS: usize = 8;

/// Exact regularized Newton-IRLS linear head (`TAG_LINEAR`).
#[derive(Clone, Debug)]
pub struct LinearSectionV8 {
    pub o_dim: usize,
    pub p_eff: usize,
    pub w: Vec<Vec<f32>>,   // [o_dim][p_eff]
    pub b: Vec<f32>,        // [o_dim]
    pub w_scales: Vec<f32>, // [o_dim]
    pub temp: f32,          // OOF temperature calibration for cls (1.0 for reg)
    pub resid_sd: f32,
}

/// One gated 8-bin nonlinear main effect (`12 B` on wire; analytically decoded!).
#[derive(Clone, Debug)]
pub struct Effect1DV8 {
    pub col_idx: u8,
    pub cls_idx: u8,
    pub scale: f32,
    pub bins8: [f32; 8],
    pub utility: f64,
}

/// One gated pairwise product + depth-2 XOR quadrant interaction (`6 B` on wire; analytically decoded!).
#[derive(Clone, Debug)]
pub struct PairIntV8 {
    pub col_i: u8,
    pub col_j: u8,
    pub cls_idx: u8,
    pub w_prod: f32,
    pub w_xor: f32,
    pub utility: f64,
}

/// Heteroscedastic residual log-scale + 5 residual quantile-residual knots (`TAG_RESID`).
#[derive(Clone, Debug, Default)]
pub struct ResidSectionV8 {
    pub cols: Vec<u8>,      // <= 8 active cols
    pub log_sd_w: Vec<f32>, // weight on z_j for log(sd(e|x))
    pub q5_resid: [f32; 5], // residual non-Gaussian quantile shifts at u = [0.1, 0.3, 0.5, 0.7, 0.9]
}

/// Nested FSQ neural code (`TAG_CODE`): Tier 0 (0 B), Tier 1 (32 B = 64 syms),
/// Tier 2 (64 B = 128 syms), Tier 3 (128 B = 256 syms), packed 2 4-bit symbols per byte.
#[derive(Clone, Debug, Default)]
pub struct CodeSectionV8 {
    pub tier: u8,
    pub packed: Vec<u8>,
}

impl CodeSectionV8 {
    pub fn tier_bytes(tier: u8) -> usize {
        match tier {
            1 => 32,
            2 => 64,
            3 => 128,
            _ => 0,
        }
    }

    pub fn from_unit_vec(vals: &[f32], tier: u8) -> Self {
        let n_bytes = Self::tier_bytes(tier);
        if n_bytes == 0 {
            return Self::default();
        }
        let n_syms = n_bytes * 2;
        let mut packed = vec![0u8; n_bytes];
        for b in 0..n_bytes {
            let v0 = vals.get(2 * b).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
            let v1 = vals.get(2 * b + 1).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
            let q0 = ((v0 + 1.0) * 7.5).round().clamp(0.0, 15.0) as u8;
            let q1 = ((v1 + 1.0) * 7.5).round().clamp(0.0, 15.0) as u8;
            packed[b] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
        let _ = n_syms;
        Self { tier, packed }
    }

    /// Dequantize to `[-1, 1]` floats (first 64 dims form the Tier-1 JopeDime embedding prefix).
    pub fn dequantize(&self, max_dims: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; max_dims];
        for (b, &byte) in self.packed.iter().enumerate() {
            let i0 = 2 * b;
            let i1 = i0 + 1;
            if i0 < max_dims {
                out[i0] = ((byte & 0x0F) as f32 / 7.5) - 1.0;
            }
            if i1 < max_dims {
                out[i1] = (((byte >> 4) & 0x0F) as f32 / 7.5) - 1.0;
            }
        }
        out
    }

    pub fn truncate_to_tier(&self, target_tier: u8) -> Self {
        let t = self.tier.min(target_tier);
        let n = Self::tier_bytes(t).min(self.packed.len());
        if n == 0 {
            Self::default()
        } else {
            Self {
                tier: t,
                packed: self.packed[..n].to_vec(),
            }
        }
    }
}

/// Fit `LinearSectionV8` with 3-fold OOF temperature calibration and quantized `i8` weights.
pub fn fit_linear_v8(
    x_unit: &[Vec<f32>],
    y_sc: &[f64],
    task_id: u8,
    k_eff: usize,
) -> (LinearSectionV8, Vec<Vec<f64>>) {
    let p = x_unit.len();
    let n = y_sc.len();
    let o_dim = if task_id == 2 { k_eff.max(2) } else { 1 };
    let all_rows: Vec<usize> = (0..n).collect();
    let (w_f64, b_f64) = fit_irls_subset(x_unit, y_sc, &all_rows, task_id, k_eff);

    // 3-fold OOF predictions for calibrated temperature + unbiased residuals for E / Pr / Tr!
    let n_folds = if n >= 30 { 3 } else { 2 };
    let mut oof_pred = vec![vec![0.0f64; n]; o_dim];
    for f in 0..n_folds {
        let tr: Vec<usize> = (0..n).filter(|r| r % n_folds != f).collect();
        let va: Vec<usize> = (0..n).filter(|r| r % n_folds == f).collect();
        let (wf, bf) = fit_irls_subset(x_unit, y_sc, &tr, task_id, k_eff);
        let pv = predict_irls_rows(x_unit, &va, &wf, &bf, task_id);
        for c in 0..o_dim {
            for (vi, &r) in va.iter().enumerate() {
                oof_pred[c][r] = pv[c][vi];
            }
        }
    }

    // Quantize weights cleanly to i8 so residual fitting sees the exact quantized linear head!
    let mut w_q = vec![vec![0.0f32; p]; o_dim];
    let mut w_scales = vec![0.0f32; o_dim];
    let mut b_f32 = vec![0.0f32; o_dim];
    for c in 0..o_dim {
        let max_w = w_f64[c]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f64, f64::max)
            .max(1e-5);
        let sc = (max_w / 127.0) as f32;
        w_scales[c] = sc;
        b_f32[c] = b_f64[c].clamp(-12.0, 12.0) as f32;
        for j in 0..p {
            let qi = (w_f64[c][j] / sc as f64).round().clamp(-127.0, 127.0) as i8;
            w_q[c][j] = qi as f32 * sc;
        }
    }

    // Compute OOF residuals [o_dim][n] (in prediction/logit units!) and regression resid_sd.
    let mut oof_resid = vec![vec![0.0f64; n]; o_dim];
    let (resid_sd, temp) = if task_id == 0 {
        let mut sse = 0.0;
        for r in 0..n {
            let e = y_sc[r] - oof_pred[0][r];
            oof_resid[0][r] = e;
            sse += e * e;
        }
        (
            ((sse / n.max(1) as f64).sqrt().clamp(0.02, 5.0)) as f32,
            1.0f32,
        )
    } else if task_id == 1 {
        let temp = fit_binary_temp_oof(&oof_pred[0], y_sc);
        for r in 0..n {
            let yr = if y_sc[r] > 0.5 { 1.0 } else { 0.0 };
            let pr = oof_pred[0][r].clamp(0.02, 0.98);
            // Newton-step logit residual (y - p) / max(p(1-p), 0.12), clamped for stability
            oof_resid[0][r] = ((yr - pr) / (pr * (1.0 - pr)).max(0.12)).clamp(-3.0, 3.0);
        }
        (0.5f32, temp)
    } else {
        for r in 0..n {
            let yi = (y_sc[r].round() as isize).clamp(0, (o_dim - 1) as isize) as usize;
            for c in 0..o_dim {
                let yc = if yi == c { 1.0 } else { 0.0 };
                let pc = oof_pred[c][r].clamp(0.02, 0.98);
                oof_resid[c][r] = ((yc - pc) / (pc * (1.0 - pc)).max(0.12)).clamp(-3.0, 3.0);
            }
        }
        (0.5f32, 1.0f32)
    };

    (
        LinearSectionV8 {
            o_dim,
            p_eff: p,
            w: w_q,
            b: b_f32,
            w_scales,
            temp,
            resid_sd,
        },
        oof_resid,
    )
}

fn fit_binary_temp_oof(oof_p: &[f64], y: &[f64]) -> f32 {
    let n = oof_p.len();
    if n < 20 {
        return 1.0;
    }
    let logits: Vec<f64> = oof_p
        .iter()
        .map(|&p| {
            let pc = p.clamp(1e-4, 1.0 - 1e-4);
            (pc / (1.0 - pc)).ln()
        })
        .collect();
    let mut best_t = 1.0f64;
    let mut best_ll = f64::INFINITY;
    for step in 0..17 {
        let t = 0.50 + 0.10 * step as f64; // [0.50 .. 2.10]
        let mut ll = 0.0;
        for (&l, &yv) in logits.iter().zip(y) {
            let pr = 1.0 / (1.0 + (-(l * t).clamp(-12.0, 12.0)).exp());
            let yr = if yv > 0.5 { 1.0 } else { 0.0 };
            ll -= yr * pr.max(1e-6).ln() + (1.0 - yr) * (1.0 - pr).max(1e-6).ln();
        }
        if ll < best_ll {
            best_ll = ll;
            best_t = t;
        }
    }
    best_t as f32
}

/// Fit gated 8-bin nonlinear S8 main effects (`E`) on OOF residuals (`chi^2 > 14.067` & `k >= 10`).
pub fn fit_effects_v8(
    z_tie: &[Vec<f64>],
    oof_resid: &[Vec<f64>],
    max_effects: usize,
) -> Vec<Effect1DV8> {
    let p = z_tie.len();
    let o_dim = oof_resid.len();
    let n = z_tie.first().map(|c| c.len()).unwrap_or(0);
    if p == 0 || n < 4 * SUPPORT_FLOOR_K || max_effects == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for j in 0..p {
        let (b8, _) = value_consistent_bins(&z_tie[j], 8);
        for c in 0..o_dim {
            let e = &oof_resid[c];
            let sd_e = (e.iter().map(|v| v * v).sum::<f64>() / n as f64)
                .sqrt()
                .max(1e-4);
            let mut sums = [0.0f64; 8];
            let mut cnts = [0usize; 8];
            for (&ev, &b) in e.iter().zip(&b8) {
                if (b as usize) < 8 && ev.is_finite() {
                    sums[b as usize] += ev;
                    cnts[b as usize] += 1;
                }
            }
            let mut bins = [0.0f32; 8];
            let mut chi2 = 0.0;
            let mut var_red = 0.0;
            for b in 0..8 {
                if cnts[b] >= SUPPORT_FLOOR_K {
                    let mu = sums[b] / cnts[b] as f64;
                    chi2 += cnts[b] as f64 * (mu / sd_e).powi(2);
                    // Empirical-Bayes shrunk bin mean (exact logit/y scale).
                    let eb = sums[b] / (cnts[b] as f64 + 8.0);
                    bins[b] = eb.clamp(-2.5, 2.5) as f32;
                    var_red += cnts[b] as f64 * eb * eb;
                }
            }
            if chi2 > 14.067 {
                let max_a = bins
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0f32, f32::max)
                    .max(1e-4);
                let scale = f16_bits_to_f32(f32_to_f16_bits(max_a / 127.0)).max(1e-5);
                for slot in &mut bins {
                    let q = (*slot / scale).round().clamp(-127.0, 127.0) as i8;
                    *slot = q as f32 * scale;
                }
                out.push(Effect1DV8 {
                    col_idx: j as u8,
                    cls_idx: c as u8,
                    scale,
                    bins8: bins,
                    utility: var_red / n as f64,
                });
            }
        }
    }
    out.sort_by(|a, b| b.utility.total_cmp(&a.utility));
    out.truncate(max_effects.min(V8_MAX_EFFECTS));
    out
}

/// Fit gated pairwise product + depth-2 XOR quadrant interactions (`Pr`) on OOF residuals (`|z| > 3.0`).
pub fn fit_pairs_v8(
    z_tie: &[Vec<f64>],
    oof_resid: &[Vec<f64>],
    max_pairs: usize,
) -> Vec<PairIntV8> {
    let p = z_tie.len();
    let o_dim = oof_resid.len();
    let n = z_tie.first().map(|c| c.len()).unwrap_or(0);
    if p < 2 || n < 4 * SUPPORT_FLOOR_K || max_pairs == 0 {
        return Vec::new();
    }
    let mut cands = Vec::new();
    let p_scan = p.min(48);
    for i in 0..p_scan {
        for j in (i + 1)..p_scan {
            let r_ij = pearson(&z_tie[i], &z_tie[j]);
            let prod: Vec<f64> = (0..n)
                .map(|r| (z_tie[i][r] * z_tie[j][r] - r_ij).clamp(-6.0, 6.0))
                .collect();
            let prod_var = (prod.iter().map(|v| v * v).sum::<f64>() / n as f64).max(1e-4);
            for c in 0..o_dim {
                let e = &oof_resid[c];
                let sd_e = (e.iter().map(|v| v * v).sum::<f64>() / n as f64)
                    .sqrt()
                    .max(1e-4);
                let rp = pearson(&prod, e);
                let zp = rp * (n as f64).sqrt();

                // Depth-2 XOR quadrant contrast on z_i > 0, z_j > 0 (k >= 10 per quadrant!)
                let mut q_sum = [0.0f64; 4];
                let mut q_cnt = [0usize; 4];
                for r in 0..n {
                    let q = (usize::from(z_tie[i][r] > 0.0) << 1) | usize::from(z_tie[j][r] > 0.0);
                    q_sum[q] += e[r];
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

                if zp.abs() > 3.0 || zx.abs() > 3.0 {
                    let w_p = if zp.abs() > 3.0 {
                        let cov: f64 =
                            prod.iter().zip(e).map(|(a, b)| a * b).sum::<f64>() / n as f64;
                        (0.65 * cov / prod_var).clamp(-1.25, 1.25)
                    } else {
                        0.0
                    };
                    let w_x = if zx.abs() > 3.0 {
                        (0.75 * w_xor_raw).clamp(-1.25, 1.25)
                    } else {
                        0.0
                    };
                    // Quantize to i8 in [-1.25, 1.25] (step 0.01)
                    let qp = (w_p * 100.0).round().clamp(-125.0, 125.0) as i8;
                    let qx = (w_x * 100.0).round().clamp(-125.0, 125.0) as i8;
                    if qp != 0 || qx != 0 {
                        cands.push(PairIntV8 {
                            col_i: i as u8,
                            col_j: j as u8,
                            cls_idx: c as u8,
                            w_prod: qp as f32 / 100.0,
                            w_xor: qx as f32 / 100.0,
                            utility: zp * zp + zx * zx,
                        });
                    }
                }
            }
        }
    }
    cands.sort_by(|a, b| b.utility.total_cmp(&a.utility));
    cands.truncate(max_pairs.min(V8_MAX_PAIRS));
    cands
}

/// Fit heteroscedastic residual log-scale & 5-knot non-Gaussian residual shape (`TAG_RESID`, reg only).
pub fn fit_resid_model_v8(z_tie: &[Vec<f64>], resid: &[f64], resid_sd: f32) -> ResidSectionV8 {
    let p = z_tie.len();
    let n = resid.len();
    if p == 0 || n < 4 * SUPPORT_FLOOR_K || resid_sd <= 1e-5 {
        return ResidSectionV8::default();
    }
    let sd = resid_sd as f64;
    let std_e: Vec<f64> = resid.iter().map(|&e| (e / sd).clamp(-5.0, 5.0)).collect();
    let log_abs: Vec<f64> = std_e.iter().map(|&e| (e.abs() + 0.15).ln()).collect();
    let mean_la = log_abs.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = log_abs.iter().map(|&v| v - mean_la).collect();

    let mut ranked: Vec<(usize, f64)> = (0..p)
        .map(|j| (j, pearson(&z_tie[j], &centered)))
        .filter(|&(_, r)| r.abs() * (n as f64).sqrt() > 2.5)
        .collect();
    ranked.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    ranked.truncate(V8_MAX_RESID_COLS);

    let cols: Vec<u8> = ranked.iter().map(|&(j, _)| j as u8).collect();
    let log_sd_w: Vec<f32> = ranked
        .iter()
        .map(|&(_, r)| {
            let w = (0.45 * r).clamp(-0.60, 0.60);
            ((w * 100.0).round().clamp(-60.0, 60.0) as i8) as f32 / 100.0
        })
        .collect();

    let mut sorted_e = std_e;
    sorted_e.sort_by(f64::total_cmp);
    let us = [0.10, 0.30, 0.50, 0.70, 0.90];
    let z_ref = [-1.28155, -0.52440, 0.0, 0.52440, 1.28155];
    let mut q5 = [0.0f32; 5];
    for k in 0..5 {
        let diff = (quantile_sorted(&sorted_e, us[k]) - z_ref[k]).clamp(-1.25, 1.25);
        let q = (diff * 100.0).round().clamp(-125.0, 125.0) as i8;
        q5[k] = q as f32 / 100.0;
    }
    if cols.is_empty() && q5.iter().all(|v| v.abs() < 0.08) {
        return ResidSectionV8::default();
    }
    ResidSectionV8 {
        cols,
        log_sd_w,
        q5_resid: q5,
    }
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
            // Map latent Gaussian z_lat into 8 equal-probability bins via Phi(z_lat).
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

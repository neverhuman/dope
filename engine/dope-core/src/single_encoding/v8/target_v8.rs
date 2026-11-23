//! Target conditional sections for `DSE8` v1 (`W` OOF-CV Newton-IRLS linear, `Rs`
//! zero-centered variance-preserving residual model, and `Z` nested FSQ neural code).

use super::calibrate::{eval_true_loss, linear_syn_to_fit_sd_ratio};
use crate::evidence::irls::{finite_f64, fit_irls_subset_lam, fit_ridge_grid_subset};
use crate::evidence::primitives::SUPPORT_FLOOR_K;
use crate::stats::{pearson, quantile_sorted};

pub use super::effects_v8::{eval_effects_and_pairs, fit_effects_v8, fit_pairs_v8};

pub const V8_MAX_EFFECTS: usize = 24;
pub const V8_MAX_PAIRS: usize = 24;
pub const V8_MAX_RESID_COLS: usize = 8;

/// Exact regularized Newton-IRLS linear head (`TAG_LINEAR`).
#[derive(Clone, Debug)]
pub struct LinearSectionV8 {
    pub o_dim: usize,
    pub p_eff: usize,
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
    pub w_scales: Vec<f32>,
    pub temp: f32,
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

/// One gated pairwise product + depth-2 XOR quadrant interaction (`5 B` on wire; analytically decoded!).
#[derive(Clone, Debug)]
pub struct PairIntV8 {
    pub col_i: u8,
    pub col_j: u8,
    pub cls_idx: u8,
    pub w_prod: f32,
    pub w_xor: f32,
    pub utility: f64,
}

/// Heteroscedastic residual log-scale + 5 zero-centered residual quantile shifts (`TAG_RESID`).
#[derive(Clone, Debug, Default)]
pub struct ResidSectionV8 {
    pub cols: Vec<u8>,
    pub log_sd_w: Vec<f32>,
    pub q5_resid: [f32; 5],
}

/// Nested FSQ neural code (`TAG_CODE`): Tier 0 (0 B), Tier 1 (32 B), Tier 2 (64 B), Tier 3 (128 B).
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
        let mut packed = vec![0u8; n_bytes];
        for (b, slot) in packed.iter_mut().enumerate() {
            let v0 = vals.get(2 * b).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
            let v1 = vals.get(2 * b + 1).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
            let q0 = ((v0 + 1.0) * 7.5).round().clamp(0.0, 15.0) as u8;
            let q1 = ((v1 + 1.0) * 7.5).round().clamp(0.0, 15.0) as u8;
            *slot = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
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

/// Fit `LinearSectionV8` with 3-fold OOF `lam` selection, collinear synthetic-X stability guard,
/// and OOF temperature calibration. Returns `(LinearSectionV8, oof_logits[o_dim][n])`.
pub fn fit_linear_v8(
    x_unit: &[Vec<f32>],
    y_sc: &[f64],
    task_id: u8,
    k_eff: usize,
    x_syn_probe: &[Vec<f32>],
) -> (LinearSectionV8, Vec<Vec<f64>>) {
    let p = x_unit.len();
    let n = y_sc.len();
    let o_dim = if task_id == 2 { k_eff.max(2) } else { 1 };
    let all_rows: Vec<usize> = (0..n).collect();
    let n_folds = if n >= 30 { 3 } else { 2 };
    let folds: Vec<(Vec<usize>, Vec<usize>)> = (0..n_folds)
        .map(|f| {
            let tr = (0..n).filter(|r| r % n_folds != f).collect();
            let va = (0..n).filter(|r| r % n_folds == f).collect();
            (tr, va)
        })
        .collect();

    let reg_lams: &[f64] = &[1e-3, 1e-2, 0.1, 1.0, 5.0, 20.0];
    let cls_lams: &[f64] = if p <= 24 {
        &[1e-3, 1e-2, 0.1, 1.0, 5.0, 20.0]
    } else {
        &[1e-2, 0.1, 1.0, 5.0, 20.0]
    };

    let quantize_rows_f64 = |w: &[Vec<f64>]| -> Vec<Vec<f64>> {
        w.iter()
            .map(|row| {
                let max_w = row.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1e-5);
                let sc = (max_w / 127.0) as f32 as f64;
                row.iter()
                    .map(|&v| ((v / sc).round().clamp(-127.0, 127.0) as i8) as f64 * sc)
                    .collect()
            })
            .collect()
    };

    let (best_lam, oof_logits) = if task_id == 0 {
        let mut oof_by_lam = vec![vec![vec![0.0f64; n]; 1]; reg_lams.len()];
        for (tr, va) in &folds {
            let sols = fit_ridge_grid_subset(x_unit, y_sc, tr, reg_lams);
            for (li, (wf, bf)) in sols.iter().enumerate() {
                for &r in va {
                    let mut z = *bf;
                    for j in 0..p {
                        z += finite_f64(x_unit[j][r]) * wf[j];
                    }
                    oof_by_lam[li][0][r] = z;
                }
            }
        }
        let full_sols = fit_ridge_grid_subset(x_unit, y_sc, &all_rows, reg_lams);
        let mut best_idx = reg_lams.len() - 1;
        let mut best_loss = f64::INFINITY;
        for (li, lg) in oof_by_lam.iter().enumerate() {
            let wq = quantize_rows_f64(&[full_sols[li].0.clone()]);
            let ratio = linear_syn_to_fit_sd_ratio(&wq, x_unit, x_syn_probe);
            let loss = eval_true_loss(lg, y_sc, &all_rows, 0, 1.0);
            if ratio <= 1.80 && loss < best_loss {
                best_loss = loss;
                best_idx = li;
            }
        }
        if !best_loss.is_finite() {
            best_idx = reg_lams.len() - 1;
        }
        (reg_lams[best_idx], oof_by_lam.swap_remove(best_idx))
    } else {
        let max_it = if task_id == 1 { 12 } else { 10 };
        let w_clip = if task_id == 1 { 32.0 } else { 24.0 };
        let mut best_lam = *cls_lams.last().unwrap();
        let mut best_loss = f64::INFINITY;
        let mut best_oof = vec![vec![0.0f64; n]; o_dim];
        let mut fallback_oof = vec![vec![0.0f64; n]; o_dim];
        for &lam in cls_lams {
            let mut oof_lg = vec![vec![0.0f64; n]; o_dim];
            for (tr, va) in &folds {
                let (wf, bf) = fit_irls_subset_lam(
                    x_unit, y_sc, tr, task_id, k_eff, lam, max_it, w_clip, 1e-4, 1e-3,
                );
                for &r in va {
                    for c in 0..o_dim {
                        let mut z = bf[c];
                        for j in 0..p {
                            z += finite_f64(x_unit[j][r]) * wf[c][j];
                        }
                        oof_lg[c][r] = z.clamp(-12.0, 12.0);
                    }
                }
            }
            let (wf_full, _) = fit_irls_subset_lam(
                x_unit, y_sc, &all_rows, task_id, k_eff, lam, max_it, w_clip, 1e-4, 1e-3,
            );
            // Measure stability on the EXACT i8-quantized weights that will be decoded on X_syn!
            let wq = quantize_rows_f64(&wf_full);
            let ratio = linear_syn_to_fit_sd_ratio(&wq, x_unit, x_syn_probe);
            let loss = eval_true_loss(&oof_lg, y_sc, &all_rows, task_id, 1.0);
            fallback_oof = oof_lg.clone();
            if ratio <= 1.35 && loss < best_loss {
                best_loss = loss;
                best_lam = lam;
                best_oof = oof_lg;
            }
        }
        if !best_loss.is_finite() {
            best_oof = fallback_oof;
        }
        (best_lam, best_oof)
    };

    let (max_it_full, w_clip_full) = match task_id {
        0 => (1, 64.0),
        1 => (12, 64.0),
        _ => (10, 32.0),
    };
    let (w_f64, b_f64) = fit_irls_subset_lam(
        x_unit,
        y_sc,
        &all_rows,
        task_id,
        k_eff,
        best_lam,
        max_it_full,
        w_clip_full,
        1e-4,
        1e-3,
    );

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

    let (resid_sd, temp) = if task_id == 0 {
        // Compute OOF RMSE so near-deterministic fits stay sharp.
        let oof_mse = eval_true_loss(&oof_logits, y_sc, &all_rows, 0, 1.0);
        (oof_mse.sqrt().clamp(0.002, 5.0) as f32, 1.0f32)
    } else {
        let temp = fit_cls_temp_oof(&oof_logits, y_sc, &all_rows, task_id);
        (0.5f32, temp)
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
        oof_logits,
    )
}

fn fit_cls_temp_oof(oof_logits: &[Vec<f64>], y_sc: &[f64], rows: &[usize], task_id: u8) -> f32 {
    if rows.len() < 20 {
        return 1.0;
    }
    let mut best_t = 1.0f32;
    let mut best_ll = f64::INFINITY;
    for step in 0..17 {
        let t = 0.50 + 0.10 * step as f32;
        let ll = eval_true_loss(oof_logits, y_sc, rows, task_id, t);
        if ll < best_ll {
            best_ll = ll;
            best_t = t;
        }
    }
    best_t
}

/// Fit zero-centered residual shape & variance-preserving heteroscedastic log-scale (`TAG_RESID`).
/// Returns `(ResidSectionV8, resid_sd_scale_multiplier)`.
pub fn fit_resid_model_v8(
    z_tie: &[Vec<f64>],
    resid: &[f64],
    resid_sd: f32,
) -> (ResidSectionV8, f32) {
    let p = z_tie.len();
    let n = resid.len();
    if p == 0 || n < 4 * SUPPORT_FLOOR_K || resid_sd < 0.08 {
        return (ResidSectionV8::default(), 1.0);
    }
    let sd = resid_sd as f64;
    let std_e: Vec<f64> = resid.iter().map(|&e| (e / sd).clamp(-5.0, 5.0)).collect();
    let log_abs: Vec<f64> = std_e.iter().map(|&e| (e.abs() + 0.15).ln()).collect();
    let mean_la = log_abs.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = log_abs.iter().map(|&v| v - mean_la).collect();

    let mut ranked: Vec<(usize, f64)> = (0..p)
        .map(|j| (j, pearson(&z_tie[j], &centered)))
        .filter(|&(_, r)| r.abs() * (n as f64).sqrt() > 3.0)
        .collect();
    ranked.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    ranked.truncate(V8_MAX_RESID_COLS.min(4));

    let cols: Vec<u8> = ranked.iter().map(|&(j, _)| j as u8).collect();
    let log_sd_w: Vec<f32> = ranked
        .iter()
        .map(|&(_, r)| {
            let w = (0.35 * r).clamp(-0.45, 0.45);
            ((w * 100.0).round().clamp(-45.0, 45.0) as i8) as f32 / 100.0
        })
        .collect();

    // Variance normalization factor so E[(resid_sd * exp(w^T z))^2] matches target variance!
    let w2_sum: f32 = log_sd_w.iter().map(|w| w * w).sum();
    let sd_mult = (-w2_sum).exp().clamp(0.65, 1.0);

    let mut sorted_e = std_e;
    sorted_e.sort_by(f64::total_cmp);
    let us = [0.10, 0.30, 0.50, 0.70, 0.90];
    let z_ref = [-1.28155, -0.52440, 0.0, 0.52440, 1.28155];
    let mut raw_q5 = [0.0f64; 5];
    for k in 0..5 {
        raw_q5[k] = (quantile_sorted(&sorted_e, us[k]) - z_ref[k]).clamp(-1.0, 1.0);
    }
    // Zero-center q5 so non-Gaussian residual quantiles NEVER shift E[y]!
    let mean_q5 = raw_q5.iter().sum::<f64>() / 5.0;
    let mut q5 = [0.0f32; 5];
    for k in 0..5 {
        let diff = (raw_q5[k] - mean_q5).clamp(-0.90, 0.90);
        let q = (diff * 100.0).round().clamp(-90.0, 90.0) as i8;
        q5[k] = q as f32 / 100.0;
    }
    if cols.is_empty() && q5.iter().all(|v| v.abs() < 0.08) {
        return (ResidSectionV8::default(), 1.0);
    }
    (
        ResidSectionV8 {
            cols,
            log_sd_w,
            q5_resid: q5,
        },
        sd_mult,
    )
}

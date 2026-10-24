//! 100% Pure-Rust synthetic table generator for `DOPE-MLC-v3` (`DML3`/`CSC1` + `.safetensors` weights).

use super::wire::{Dml3Envelope, ObliviousTree, VectorLeafTree};
use super::{MlcV3Model, MLC_V3_MAX_COLS};
use crate::rng::Rng;
use crate::{DopeError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FixedSampleDraws {
    pub reg_idx: Vec<usize>,
    pub eps_chol: Vec<Vec<f64>>,
    pub eps_diag: Vec<Vec<f64>>,
    pub eps_target: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlcV3SampledTable {
    pub headers: Vec<String>,
    pub x_scaled: Vec<Vec<f64>>,
    pub y_scaled: Vec<f64>,
    pub x_raw: Vec<Vec<f64>>,
    pub y_raw: Vec<f64>,
}

pub fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 {
        -1.0
    } else if x > 0.0 {
        1.0
    } else {
        0.0
    };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-(ax * ax)).exp())
}

fn interp_1d(u: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if xp.is_empty() || fp.is_empty() {
        return u;
    }
    if u <= xp[0] || xp.len() == 1 {
        return fp[0];
    }
    let last = xp.len() - 1;
    if u >= xp[last] {
        return fp[last];
    }
    let idx = xp
        .partition_point(|&x| x < u)
        .saturating_sub(1)
        .min(last - 1);
    let dx = (xp[idx + 1] - xp[idx]).max(1e-12);
    fp[idx] + ((u - xp[idx]) / dx).clamp(0.0, 1.0) * (fp[idx + 1] - fp[idx])
}

fn row_summary_first4(row: &[f64]) -> [f64; 4] {
    if row.is_empty() {
        return [0.0; 4];
    }
    let pf = row.len() as f64;
    let r_mean = row.iter().copied().sum::<f64>() / pf;
    let r_std = (row.iter().map(|&v| (v - r_mean).powi(2)).sum::<f64>() / pf + 1e-8).sqrt();
    let r_l1 = row.iter().map(|&v| v.abs()).sum::<f64>() / pf;
    let r_l2 = (row.iter().map(|&v| v * v).sum::<f64>() / pf + 1e-8).sqrt();
    [r_mean, r_std, r_l1, r_l2]
}

pub fn build_enriched_basis(
    x: &[Vec<f64>],
    gates: &BTreeMap<String, f64>,
    pairs: &[(usize, usize)],
    level: &str,
    include_v3_primitives: bool,
) -> Vec<Vec<f64>> {
    let is_high = matches!(level, "high" | "ultra");
    let is_ultra = level == "ultra";
    let mw = gates.get("monotone_w").copied().unwrap_or(0.55);
    let hw = gates
        .get("hinge_w")
        .copied()
        .unwrap_or(if is_high { 0.55 } else { 0.0 });
    let pw = gates
        .get("pair_w")
        .copied()
        .unwrap_or(if is_high { 0.60 } else { 0.0 });
    let tw = gates
        .get("tree_w")
        .copied()
        .unwrap_or(if is_ultra { 1.0 } else { 0.0 });
    let aw = gates.get("asinh_atan_w").copied().unwrap_or(0.45 * mw);
    let rw = gates.get("row_summary_w").copied().unwrap_or(0.40 * mw);
    let fw = gates.get("periodic_fract_w").copied().unwrap_or(0.30 * hw);

    x.iter()
        .map(|row| {
            let mut phi = Vec::with_capacity(1 + row.len() * 14 + pairs.len() + 4);
            phi.push(1.0);
            phi.extend(row.iter().copied());
            phi.extend(row.iter().map(|&v| mw * v.signum() * v.abs().ln_1p()));
            phi.extend(row.iter().map(|&v| mw * (0.75 * v).tanh()));
            phi.extend(row.iter().map(|&v| 0.5 * mw * (v.abs() - 0.8)));
            if is_high {
                phi.extend(row.iter().map(|&v| hw * (v - 0.5).max(0.0)));
                phi.extend(row.iter().map(|&v| hw * (-v - 0.5).max(0.0)));
                phi.extend(row.iter().map(|&v| 0.35 * hw * (1.5 * v).sin()));
                for &(i, j) in pairs {
                    phi.push(if i < row.len() && j < row.len() {
                        pw * row[i] * row[j]
                    } else {
                        0.0
                    });
                }
            }
            if is_ultra {
                phi.extend(row.iter().map(|&v| tw * (3.0 * (v - 1.0)).tanh()));
                phi.extend(row.iter().map(|&v| tw * (3.0 * (v + 1.0)).tanh()));
                phi.extend(row.iter().map(|&v| 0.5 * tw * (-0.5 * v * v).exp()));
            }
            if include_v3_primitives {
                phi.extend(row.iter().map(|&v| aw * v.asinh()));
                phi.extend(row.iter().map(|&v| aw * v.atan()));
                phi.extend(row_summary_first4(row).iter().map(|&v| rw * v));
                if is_high {
                    phi.extend(row.iter().map(|&v| fw * (1.5 * v).cos()));
                }
            }
            phi
        })
        .collect()
}

fn eval_oblivious_trees(x: &[Vec<f64>], trees: &[ObliviousTree]) -> Vec<f64> {
    let mut out = vec![0.0f64; x.len()];
    for t in trees {
        for (i, row) in x.iter().enumerate() {
            let mut idx = 0usize;
            for (&f, &th) in t.features.iter().zip(t.thresholds.iter()) {
                let bit = usize::from(f < row.len() && row[f] >= th);
                idx = (idx << 1) | bit;
            }
            if idx < t.leaf_values.len() {
                out[i] += t.leaf_values[idx];
            }
        }
    }
    out
}

fn eval_vector_leaf_trees(
    x: &[Vec<f64>],
    trees: &[VectorLeafTree],
    k_classes: usize,
) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0f64; k_classes]; x.len()];
    for t in trees {
        for (i, row) in x.iter().enumerate() {
            let mut idx = 0usize;
            for (&f, &th) in t.features.iter().zip(t.thresholds.iter()) {
                let bit = usize::from(f < row.len() && row[f] >= th);
                idx = (idx << 1) | bit;
            }
            if idx < t.leaf_values.len() {
                for (c, &lv) in t.leaf_values[idx].iter().take(k_classes).enumerate() {
                    out[i][c] += lv;
                }
            }
        }
    }
    out
}

pub fn sample_synthetic_table(
    model: &MlcV3Model,
    env: &Dml3Envelope,
    rows: usize,
    seed: u64,
) -> Result<MlcV3SampledTable> {
    sample_synthetic_table_with_draws(model, env, rows, seed, None)
}

pub fn sample_synthetic_table_with_draws(
    model: &MlcV3Model,
    env: &Dml3Envelope,
    rows: usize,
    seed: u64,
    draws: Option<&FixedSampleDraws>,
) -> Result<MlcV3SampledTable> {
    let enc = &env.encoding;
    let n = if rows == 0 { enc.n_train.max(1) } else { rows };
    let p_use = enc.p_use;
    if p_use == 0 || p_use > MLC_V3_MAX_COLS {
        return Err(DopeError::Artifact("invalid mlc_v3 column count".into()));
    }
    let mut rng = Rng::new(seed);
    let rw_sum: f64 = enc.regime_weights.iter().copied().sum::<f64>().max(1e-12);
    let mut rw_cdf = Vec::with_capacity(enc.regime_weights.len());
    let mut acc = 0.0;
    for &w in &enc.regime_weights {
        acc += w / rw_sum;
        rw_cdf.push(acc);
    }
    let rank = enc.chol_factor.first().map_or(0, Vec::len);
    let mut z_syn = vec![vec![0.0f64; p_use]; n];
    let mut z_pad = vec![vec![0.0f32; MLC_V3_MAX_COLS]; n];
    for i in 0..n {
        let reg = if let Some(d) = draws {
            d.reg_idx[i].min(rw_cdf.len().saturating_sub(1))
        } else {
            let u_reg = rng.uniform();
            rw_cdf
                .partition_point(|&c| c < u_reg)
                .min(rw_cdf.len().saturating_sub(1))
        };
        let eps_r: Vec<f64> = if let Some(d) = draws {
            d.eps_chol[i][..rank].to_vec()
        } else {
            (0..rank).map(|_| rng.normal()).collect()
        };
        for j in 0..p_use {
            let dot_chol: f64 = enc.chol_factor[j]
                .iter()
                .zip(eps_r.iter())
                .map(|(&c, &e)| c * e)
                .sum();
            let e_diag = draws.map_or_else(|| rng.normal(), |d| d.eps_diag[i][j]);
            let v = enc.regime_means[reg][j]
                + enc.regime_scales[reg][j] * (dot_chol + e_diag * enc.diag_res_std[j]);
            z_syn[i][j] = v;
            z_pad[i][j] = v as f32;
        }
    }
    let z_ref = model.cov_net.forward(&z_pad, &enc.dist_latent, p_use)?;
    for j in 0..p_use {
        let mut mean = 0.0;
        for i in 0..n {
            z_syn[i][j] = 0.94 * z_syn[i][j] + 0.06 * (z_ref[i][j] as f64);
            mean += z_syn[i][j];
        }
        mean /= n as f64;
        let std = (z_syn.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / (n as f64))
            .sqrt()
            .max(1e-6);
        for row in z_syn.iter_mut().take(n) {
            row[j] = (row[j] - mean) / std;
        }
    }
    let mut x_syn = vec![vec![0.0f64; p_use]; n];
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    for i in 0..n {
        for j in 0..p_use {
            let u = (0.5 * (1.0 + erf_approx(z_syn[i][j] * inv_sqrt2))).clamp(0.001, 0.999);
            x_syn[i][j] = if enc.is_binary_col[j] {
                if u >= 1.0 - enc.binary_probs[j] {
                    enc.binary_low_high[j][1]
                } else {
                    enc.binary_low_high[j][0]
                }
            } else if let Some((u_vals, cum_p)) = enc.discrete_support.get(&j) {
                let idx = cum_p
                    .partition_point(|&cp| cp < u)
                    .min(u_vals.len().saturating_sub(1));
                u_vals[idx]
            } else {
                interp_1d(u, &enc.knot_grid, &enc.col_knots[j])
            };
        }
    }

    let tb = &enc.target_bundle;
    let phi = build_enriched_basis(
        &x_syn,
        &tb.gates,
        &tb.pairs,
        &enc.level,
        tb.include_v3_primitives,
    );
    let is_low_med = matches!(enc.level.as_str(), "low" | "small" | "medium");
    let y_syn = if enc.task == "multiclass" || tb.task_kind == "multiclass" {
        let k_cls = tb.k_classes.max(3);
        let mut priors: Vec<f64> = if tb.class_priors.len() == k_cls {
            tb.class_priors.iter().map(|&p| p.max(1e-4)).collect()
        } else {
            vec![1.0 / (k_cls as f64); k_cls]
        };
        let p_sum: f64 = priors.iter().sum();
        for p in &mut priors {
            *p /= p_sum;
        }
        let m = phi.first().map_or(0, Vec::len).min(tb.u_basis.len());
        let r_rank = tb.u_basis.first().map_or(0, Vec::len);
        let mut w_mat = vec![vec![0.0f64; k_cls]; m];
        for (b_idx, w_row) in w_mat.iter_mut().enumerate().take(m) {
            for (c_idx, w_cell) in w_row
                .iter_mut()
                .enumerate()
                .take(k_cls.min(tb.v_basis.len()))
            {
                *w_cell = (0..r_rank.min(tb.v_basis[c_idx].len()))
                    .map(|r_idx| tb.u_basis[b_idx][r_idx] * tb.v_basis[c_idx][r_idx])
                    .sum();
            }
        }
        let tree_scores = eval_vector_leaf_trees(&x_syn, &tb.mc_trees, k_cls);
        let noise_amp = tb.noise_scale * if is_low_med { 0.08 } else { 0.03 };
        let mut noisy = vec![vec![0.0f64; k_cls]; n];
        let mut y_out = vec![0.0f64; n];
        for i in 0..n {
            let (mut best_c, mut best_v) = (0usize, f64::NEG_INFINITY);
            for c_idx in 0..k_cls {
                let dot: f64 = (0..m).map(|b| phi[i][b] * w_mat[b][c_idx]).sum();
                let eps = draws.map_or_else(|| rng.normal(), |d| d.eps_target[i][c_idx]);
                let sc = priors[c_idx] + dot + tree_scores[i][c_idx] + eps * noise_amp;
                noisy[i][c_idx] = sc;
                if sc > best_v {
                    best_v = sc;
                    best_c = c_idx;
                }
            }
            y_out[i] = best_c as f64;
        }
        if n >= k_cls {
            let mut order_cls: Vec<usize> = (0..k_cls).collect();
            order_cls.sort_by(|&a, &b| priors[a].total_cmp(&priors[b]));
            let mut used = BTreeSet::new();
            for c_idx in order_cls {
                if priors[c_idx] <= 1e-5 {
                    continue;
                }
                if let Some(pos) = y_out.iter().position(|&yv| (yv as usize) == c_idx) {
                    used.insert(pos);
                } else {
                    let mut cand_rows: Vec<usize> = (0..n).collect();
                    cand_rows.sort_by(|&a, &b| noisy[b][c_idx].total_cmp(&noisy[a][c_idx]));
                    for cand in cand_rows {
                        if used.insert(cand) {
                            y_out[cand] = c_idx as f64;
                            break;
                        }
                    }
                }
            }
        }
        y_out
    } else {
        let mut x_pad = vec![vec![0.0f32; MLC_V3_MAX_COLS]; n];
        for i in 0..n {
            for j in 0..p_use {
                x_pad[i][j] = x_syn[i][j] as f32;
            }
        }
        let res_net = model
            .target_net
            .forward(&x_pad, &enc.task_latent, p_use, None)?;
        let tree_pred = eval_oblivious_trees(&x_syn, &tb.trees);
        let m = phi.first().map_or(0, Vec::len).min(tb.w_basis.len());
        let mut pred = Vec::with_capacity(n);
        for i in 0..n {
            let dot: f64 = (0..m).map(|b| phi[i][b] * tb.w_basis[b]).sum();
            pred.push(dot + tb.neural_w * ((res_net[i] as f64) - tb.res_net_mean) + tree_pred[i]);
        }
        if matches!(enc.task.as_str(), "classification" | "binary") {
            let prior = tb.class_prior.clamp(0.01, 0.99);
            let n_mult = if is_low_med {
                0.12
            } else if enc.level == "high" {
                0.06
            } else {
                0.03
            };
            let score: Vec<f64> = (0..n)
                .map(|i| {
                    let eps = draws.map_or_else(|| rng.normal(), |d| d.eps_target[i][0]);
                    pred[i] + eps * tb.noise_scale * n_mult
                })
                .collect();
            let mut sorted = score.clone();
            sorted.sort_by(f64::total_cmp);
            let q_pos = (1.0 - prior) * ((n.saturating_sub(1)) as f64);
            let lo = (q_pos.floor() as usize).min(n - 1);
            let hi = (lo + 1).min(n - 1);
            let frac = q_pos - (lo as f64);
            let thresh = sorted[lo] + frac * (sorted[hi] - sorted[lo]);
            let mut y_out: Vec<f64> = score
                .iter()
                .map(|&s| if s >= thresh { 1.0 } else { 0.0 })
                .collect();
            if n >= 2 && y_out.iter().all(|&v| v == y_out[0]) {
                let (mut imin, mut imax) = (0usize, 0usize);
                for i in 1..n {
                    if score[i] < score[imin] {
                        imin = i;
                    }
                    if score[i] > score[imax] {
                        imax = i;
                    }
                }
                y_out[imax] = 1.0;
                y_out[imin] = 0.0;
            }
            y_out
        } else {
            let n_mult = if is_low_med {
                0.42
            } else if enc.level == "high" {
                0.22
            } else {
                0.10
            };
            let sigma = tb.residual_std * tb.noise_scale * n_mult;
            (0..n)
                .map(|i| {
                    let eps = draws.map_or_else(|| rng.normal(), |d| d.eps_target[i][0]);
                    (pred[i] + eps * sigma).clamp(tb.y_min - 0.5, tb.y_max + 0.5)
                })
                .collect()
        }
    };

    let (x_raw, y_raw) = match &env.scaler {
        Some(sc) => sc.inverse_transform(&x_syn, &y_syn),
        None => (x_syn.clone(), y_syn.clone()),
    };
    let mut headers: Vec<String> = (0..p_use).map(|j| format!("f_{j:03}")).collect();
    headers.push("target".into());
    Ok(MlcV3SampledTable {
        headers,
        x_scaled: x_syn,
        y_scaled: y_syn,
        x_raw,
        y_raw,
    })
}

//! Pure-Rust 196D scale-invariant, extrema-free fitted DOPE packet extractor (`WP2`).
//!
//! Emits zero target extrema (`y_min`/`y_max`), deduplicates `noise_scale`, normalizes shape
//! counts, handles multiclass class-prior entropy cleanly, and pools symmetric per-column tokens.

use super::research_view::{
    basis_at, mean_std_max, resample_knots_9, rms_slice, write_latent_stats,
    ResearchConditioningLevel, TreeScanStats,
};
use super::{bad, MlcV3Encoding, MLC_V3_GATE_KEYS_14, MLC_V3_MAX_COLS};
use crate::Result;
use std::collections::BTreeSet;

pub const PACKET_GLOBAL_V1_SCHEMA: &str = "dope.packet-global/v1";
pub const PACKET_GLOBAL_V1_DIM: usize = 28;
pub const PACKET_FEATURES_V1_SCHEMA: &str = "dope.packet-features/v1";
pub const PACKET_FEATURES_V1_DIM: usize = 196;

pub const PACKET_GLOBAL_V1_FIELDS: [&str; PACKET_GLOBAL_V1_DIM] = [
    "asinh_atan_w",
    "copula_retention",
    "het_scale_w",
    "hinge_w",
    "monotone_w",
    "neural_w",
    "noise_scale",
    "pair_w",
    "periodic_fract_w",
    "regime_sharpness",
    "rff_w",
    "ridge_alpha",
    "row_summary_w",
    "tree_w",
    "residual_std_scaled",
    "neural_w_active",
    "res_net_mean_tanh",
    "is_regression",
    "is_binary",
    "is_multiclass",
    "log2_k_classes_norm",
    "class_entropy_norm",
    "log10_n_train_norm",
    "log2_p_use_norm",
    "aspect_ratio_tanh",
    "n_trees_norm",
    "n_pairs_norm",
    "level_code",
];

/// Extract the 28-D scale-invariant, extrema-free global packet token from a fitted `MlcV3Encoding`.
pub fn packet_global_v1_from_encoding(
    enc: &MlcV3Encoding,
    include_target_bundle: bool,
) -> Result<[f32; PACKET_GLOBAL_V1_DIM]> {
    let full = packet_features_v1_from_encoding(enc, include_target_bundle)?;
    let mut out = [0.0f32; PACKET_GLOBAL_V1_DIM];
    out.copy_from_slice(&full[..PACKET_GLOBAL_V1_DIM]);
    Ok(out)
}

/// Extract the 196-D scale-invariant, column-permutation-invariant fitted packet vector (`WP2`).
pub fn packet_features_v1_from_encoding(
    enc: &MlcV3Encoding,
    include_target_bundle: bool,
) -> Result<[f32; PACKET_FEATURES_V1_DIM]> {
    let lvl = ResearchConditioningLevel::parse(&enc.level)?;
    if enc.n_train == 0 || enc.p_use == 0 || enc.p_use > MLC_V3_MAX_COLS {
        return Err(bad(
            "packet_features_v1 requires 1 <= p_use <= 64 and n_train >= 1",
        ));
    }
    let p_use = enc.p_use;
    if enc.col_knots.len() < p_use
        || enc.is_binary_col.len() < p_use
        || enc.binary_probs.len() < p_use
        || enc.chol_factor.len() < p_use
        || enc.diag_res_std.len() < p_use
    {
        return Err(bad("packet_features_v1 column arrays shorter than p_use"));
    }
    let b = &enc.target_bundle;
    let is_mc = enc.task == "multiclass" || b.task_kind == "multiclass";
    let is_bin = !is_mc && (enc.task == "classification" || enc.task == "binary");
    let is_reg = !is_mc && (enc.task == "regression" || enc.task.is_empty());
    if !is_reg && !is_bin && !is_mc {
        return Err(bad("packet_features_v1 unsupported task"));
    }

    let mut out = [0.0f64; PACKET_FEATURES_V1_DIM];
    for (i, key) in MLC_V3_GATE_KEYS_14.iter().enumerate() {
        let g = *enc
            .gates
            .get(*key)
            .ok_or_else(|| bad("packet_features_v1 requires the 14 canonical gates"))?;
        if !g.is_finite() {
            return Err(bad("packet_features_v1 gate must be finite"));
        }
        out[i] = g;
    }

    let n_trees = if !b.mc_trees.is_empty() {
        b.mc_trees.len()
    } else {
        b.trees.len()
    };
    let k_cls = if is_mc {
        b.k_classes.max(b.class_priors.len()).max(3)
    } else if is_bin {
        2
    } else {
        1
    };

    if include_target_bundle {
        for v in [b.residual_std, b.neural_w, b.res_net_mean, b.class_prior] {
            if !v.is_finite() {
                return Err(bad(
                    "packet_features_v1 target_bundle scalar must be finite",
                ));
            }
        }
        if !is_mc {
            out[14] = b.residual_std.clamp(0.0, 4.0) / 4.0;
            out[15] = b.neural_w.clamp(0.0, 2.0) / 2.0;
            out[16] = b.res_net_mean.tanh();
        }
        out[21] = if is_bin {
            let p = b.class_prior.clamp(1e-6, 1.0 - 1e-6);
            (-p * p.ln() - (1.0 - p) * (1.0 - p).ln()) / 2.0_f64.ln()
        } else if is_mc && !b.class_priors.is_empty() {
            if b.class_priors.iter().any(|p| !p.is_finite()) {
                return Err(bad("packet_features_v1 non-finite class_priors"));
            }
            let raw: Vec<f64> = b.class_priors.iter().map(|p| p.max(1e-6)).collect();
            let s: f64 = raw.iter().sum();
            let ent: f64 = raw
                .iter()
                .map(|p| {
                    let q = *p / s;
                    -q * q.ln()
                })
                .sum();
            (ent / (raw.len().max(2) as f64).ln()).clamp(0.0, 1.0)
        } else {
            0.0
        };
        out[25] = ((n_trees as f64) / 56.0).clamp(0.0, 2.0);
        out[26] = ((b.pairs.len() as f64) / 28.0).clamp(0.0, 2.0);
    }
    out[17] = f64::from(u8::from(is_reg));
    out[18] = f64::from(u8::from(is_bin));
    out[19] = f64::from(u8::from(is_mc));
    out[20] = ((k_cls as f64).log2() / 8.0).clamp(0.0, 2.0);
    out[22] = ((enc.n_train as f64).log10() / 6.0).clamp(0.0, 2.0);
    out[23] = ((p_use as f64).log2() / 6.0).clamp(0.0, 2.0);
    out[24] = ((p_use as f64) / (enc.n_train as f64)).tanh();
    out[27] = lvl.code();

    if !enc.regime_weights.is_empty() {
        let mut rw: Vec<f64> = enc.regime_weights.iter().map(|v| v.max(0.0)).collect();
        let rs: f64 = rw.iter().sum::<f64>().max(1e-8);
        for v in &mut rw {
            *v /= rs;
        }
        rw.sort_by(|a, b| b.total_cmp(a));
        for (k, &v) in rw.iter().take(8).enumerate() {
            out[28 + k] = v;
        }
    }

    let mut chol_sq = vec![0.0f64; p_use];
    let mut chol_top1 = vec![0.0f64; p_use];
    for j in 0..p_use {
        chol_sq[j] = enc.chol_factor[j].iter().map(|v| v * v).sum();
        chol_top1[j] = enc.chol_factor[j]
            .iter()
            .fold(0.0f64, |acc, v| acc.max(v.abs()));
    }
    let (cs_mean, cs_std, cs_max) = mean_std_max(&chol_sq);
    out[36] = cs_mean;
    out[37] = cs_std;
    out[38] = cs_max;
    out[39] = enc.diag_res_std[..p_use].iter().sum::<f64>() / (p_use as f64);

    let basis_rows: Vec<f64> = if !include_target_bundle {
        Vec::new()
    } else if is_mc && !b.u_basis.is_empty() {
        b.u_basis
            .iter()
            .map(|row| row.iter().map(|v| v * v).sum::<f64>().sqrt())
            .collect()
    } else {
        b.w_basis.iter().map(|v| v.abs()).collect()
    };
    if include_target_bundle && !basis_rows.is_empty() {
        if basis_rows.iter().any(|v| !v.is_finite()) {
            return Err(bad("packet_features_v1 non-finite basis weights"));
        }
        let l1: f64 = basis_rows.iter().sum();
        let l2: f64 = basis_rows.iter().map(|v| v * v).sum::<f64>().sqrt();
        let max_w = basis_rows.iter().fold(0.0f64, |acc, &v| acc.max(v));
        let mut sorted_w = basis_rows.clone();
        sorted_w.sort_by(|a, b| b.total_cmp(a));
        let top4: f64 = sorted_w.iter().take(4).sum();
        out[40] = (1.0 + l2).ln().tanh();
        out[41] = (l1 / (basis_rows.len() as f64)).tanh();
        out[42] = max_w / (l2 + 1e-8);
        out[43] = top4 / (l1 + 1e-8);
    }

    let mut pair_deg = vec![0usize; p_use];
    let mut unique_pair_cols = BTreeSet::new();
    if include_target_bundle {
        for &(a, c) in &b.pairs {
            if a < p_use {
                pair_deg[a] += 1;
                unique_pair_cols.insert(a);
            }
            if c < p_use {
                pair_deg[c] += 1;
                unique_pair_cols.insert(c);
            }
        }
    }

    let mut stats = TreeScanStats::new(p_use);
    if include_target_bundle {
        stats.collect(b, p_use);
    }
    let total_splits: usize = stats.split_cnt.iter().sum();
    let total_depth_w: f64 = stats.depth_w.iter().sum();
    let sum_col_basis: f64 = (0..p_use)
        .map(|j| {
            basis_at(&basis_rows, 1 + j)
                + basis_at(&basis_rows, 1 + p_use + j)
                + basis_at(&basis_rows, 1 + 2 * p_use + j)
        })
        .sum();

    let mut col_tokens = vec![[0.0f64; 32]; p_use];
    let mut activities = vec![0.0f64; p_use];
    for j in 0..p_use {
        let knots9 = resample_knots_9(&enc.col_knots[j]);
        let mut tok = [0.0f64; 32];
        tok[..9].copy_from_slice(&knots9);
        let denom = ((knots9[6] - knots9[2]) * 6.0).abs().max(1e-4);
        tok[9] = knots9[6] - knots9[2];
        tok[10] = (((knots9[6] + knots9[2] - 2.0 * knots9[4]) * 6.0) / denom).tanh();
        tok[11] = ((((knots9[8] - knots9[0]) * 6.0) / denom) / 4.0).tanh();
        tok[12] = f64::from(u8::from(enc.is_binary_col[j]));
        tok[13] = if enc.is_binary_col[j] {
            let p = enc.binary_probs[j].clamp(0.0, 1.0);
            4.0 * p * (1.0 - p)
        } else {
            0.0
        };
        if let Some((u_vals, _)) = enc.discrete_support.get(&j) {
            tok[14] = 1.0;
            tok[15] = ((u_vals.len() as f64) / 32.0).clamp(0.0, 2.0);
        }
        tok[16] = chol_sq[j].clamp(0.0, 2.0);
        tok[17] = chol_top1[j].clamp(0.0, 2.0);
        tok[18] = enc.diag_res_std[j].clamp(0.0, 2.0);
        if !enc.regime_means.is_empty() && enc.regime_means.iter().all(|r| r.len() > j) {
            let rm: Vec<f64> = enc.regime_means.iter().map(|r| r[j]).collect();
            tok[19] = mean_std_max(&rm).1.tanh();
        }
        if !enc.regime_scales.is_empty() && enc.regime_scales.iter().all(|r| r.len() > j) {
            let rs: f64 = enc.regime_scales.iter().map(|r| r[j]).sum::<f64>()
                / (enc.regime_scales.len() as f64);
            tok[20] = rs.clamp(0.0, 4.0) / 4.0;
        }
        if include_target_bundle {
            let (w_l, w_q, w_a) = (
                basis_at(&basis_rows, 1 + j),
                basis_at(&basis_rows, 1 + p_use + j),
                basis_at(&basis_rows, 1 + 2 * p_use + j),
            );
            let split_share = (stats.split_cnt[j] as f64) / (total_splits.max(1) as f64);
            let basis_share = (w_l + w_q + w_a) / (sum_col_basis + 1e-8);
            let act = 0.5 * basis_share + 0.5 * split_share;
            tok[21] = (pair_deg[j] as f64) / (b.pairs.len().max(1) as f64);
            tok[22] = split_share;
            tok[23] = stats.depth_w[j] / total_depth_w.max(1e-8);
            tok[24] = (stats.root_cnt[j] as f64) / (n_trees.max(1) as f64);
            if !stats.col_thresh[j].is_empty() {
                let (tm, ts, _) = mean_std_max(&stats.col_thresh[j]);
                tok[25] = (tm / 4.0).tanh();
                tok[26] = if stats.col_thresh[j].len() > 1 {
                    (ts / 4.0).tanh()
                } else {
                    0.0
                };
            }
            tok[27] = w_l.tanh();
            tok[28] = w_q.tanh();
            tok[29] = w_a.tanh();
            tok[30] = basis_share;
            tok[31] = act;
            activities[j] = act;
        }
        col_tokens[j] = tok;
    }

    let max_act = activities.iter().fold(0.0f64, |a, &v| a.max(v));
    let exp_w: Vec<f64> = activities
        .iter()
        .map(|&a| (4.0 * (a - max_act)).exp())
        .collect();
    let exp_sum: f64 = exp_w.iter().sum::<f64>().max(1e-8);
    for d in 0..32 {
        let vals: Vec<f64> = (0..p_use).map(|j| col_tokens[j][d]).collect();
        let (m, s, mx) = mean_std_max(&vals);
        out[44 + d] = m;
        out[76 + d] = s;
        out[108 + d] = mx;
        out[140 + d] = (0..p_use)
            .map(|j| (exp_w[j] / exp_sum) * col_tokens[j][d])
            .sum();
    }

    if include_target_bundle {
        if n_trees > 0 {
            out[172] = ((stats.depth_sum as f64) / (n_trees as f64) / 4.0).clamp(0.0, 2.0);
            if !stats.all_leaves.is_empty() {
                let abs_l: Vec<f64> = stats.all_leaves.iter().map(|v| v.abs()).collect();
                let (lm_abs, _, lmax_abs) = mean_std_max(&abs_l);
                out[173] = lm_abs.tanh();
                out[174] = mean_std_max(&stats.all_leaves).1.tanh();
                out[175] = lmax_abs.tanh();
            }
            out[176] = stats.first_rms / (stats.first_rms + stats.last_rms + 1e-8);
            let uniq_split = stats.split_cnt.iter().filter(|&&c| c > 0).count();
            out[177] = (uniq_split as f64) / (p_use as f64);
            if !stats.all_thresh.is_empty() {
                let abs_th: Vec<f64> = stats.all_thresh.iter().map(|v| v.abs()).collect();
                out[178] = (mean_std_max(&abs_th).0 / 4.0).tanh();
                out[179] = (mean_std_max(&stats.all_thresh).1 / 4.0).tanh();
            }
        }
        out[180] = (unique_pair_cols.len() as f64) / (p_use as f64);
        out[181] = (*pair_deg.iter().max().unwrap_or(&0) as f64) / (b.pairs.len().max(1) as f64);
        let tail_start = (1 + 3 * p_use).min(basis_rows.len());
        let tail = &basis_rows[tail_start..];
        out[182] = rms_slice(tail).tanh();
        let total_l1: f64 = basis_rows.iter().sum::<f64>() + 1e-8;
        let e_lin: f64 = basis_rows
            .iter()
            .take((1 + p_use).min(basis_rows.len()))
            .sum();
        let e_quad: f64 = basis_rows
            .iter()
            .skip((1 + p_use).min(basis_rows.len()))
            .take(p_use)
            .sum();
        let e_abs: f64 = basis_rows
            .iter()
            .skip((1 + 2 * p_use).min(basis_rows.len()))
            .take(p_use)
            .sum();
        let e_tail: f64 = tail.iter().sum();
        out[183] = e_lin / total_l1;
        out[184] = e_quad / total_l1;
        out[185] = e_abs / total_l1;
        out[186] = e_tail / total_l1;
        out[187] = f64::from(u8::from(b.include_v3_primitives));
    }

    let tl: Vec<f64> = enc.task_latent.iter().map(|&v| v as f64).collect();
    let dl: Vec<f64> = enc.dist_latent.iter().map(|&v| v as f64).collect();
    write_latent_stats(&tl, &mut out[188..192]);
    write_latent_stats(&dl, &mut out[192..196]);

    Ok(std::array::from_fn(|i| {
        if out[i].is_finite() {
            out[i] as f32
        } else {
            0.0
        }
    }))
}

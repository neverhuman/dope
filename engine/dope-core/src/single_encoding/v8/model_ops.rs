//! Row-local neural decode operations, `entmax_1.5`, and source-free Receiver Decoder inputs
//! for `SingleEncV8` (`dope_core::single_encoding::v8`).

use super::copula_regimes::V8_MAX_RANK;
use super::model::{
    NeuralPlanV8, BILIN_RANK, DEC_CODE_DIM, DEC_COL_DIM, DEC_EDGE_DIM, DEC_GLOB_DIM, NODE_DEPTH,
    NODE_LEAVES, NODE_TREES,
};
use super::packet::Dse8Packet;
use super::scale_v8::{V8_MAX_CLASSES, V8_MAX_COLS};
use crate::single_encoding::math::silu;
use crate::single_encoding::KNOT_COUNT;

/// Apply 2-layer row-local RealNVP coupling flow to latent Gaussian vector `z_lat[..p]`.
/// Strictly zero-init identity (`flow_s == 0 && flow_t == 0` -> bit-exact no-op!).
pub fn apply_row_local_flow(plan: &NeuralPlanV8, z_lat: &mut [f32]) {
    if plan.is_identity || z_lat.len() < 2 {
        return;
    }
    for layer in 0..2 {
        let mut ctx = 0.0f32;
        let mut cnt = 0usize;
        for (j, &zv) in z_lat.iter().enumerate() {
            if j % 2 == layer {
                ctx += silu(zv);
                cnt += 1;
            }
        }
        let c_bar = (ctx / cnt.max(1) as f32).clamp(-3.0, 3.0);
        for (j, zv) in z_lat.iter_mut().enumerate() {
            if j % 2 != layer {
                let s = plan.flow_s[j][layer];
                let t = plan.flow_t[j][layer];
                *zv = (*zv * (s * c_bar).clamp(-0.6, 0.6).exp() + t * c_bar).clamp(-5.0, 5.0);
            }
        }
    }
}

/// Evaluate Soft-NODE (`T=8, D=3`) + rank-4 bilinear residual gated by `alpha_table`.
pub fn eval_neural_target_residual(
    plan: &NeuralPlanV8,
    x_unit: &[f32],
    o_dim: usize,
    out_delta: &mut [f32],
) {
    if plan.is_identity {
        return;
    }
    let p = x_unit.len().min(V8_MAX_COLS);
    let mut nl = [0.0f32; V8_MAX_CLASSES];

    for t in 0..NODE_TREES {
        let mut p_right = [0.0f32; NODE_DEPTH];
        for d in 0..NODE_DEPTH {
            let mut feat = 0.0f32;
            for j in 0..p {
                feat += plan.node_w[t][d][j] * x_unit[j];
            }
            let z = ((feat - plan.node_thr[t][d]) * 2.0).clamp(-10.0, 10.0);
            p_right[d] = 1.0 / (1.0 + (-z).exp());
        }
        for l in 0..NODE_LEAVES {
            let mut w_l = 1.0f32;
            for d in 0..NODE_DEPTH {
                w_l *= if (l & (1 << d)) != 0 {
                    p_right[d]
                } else {
                    1.0 - p_right[d]
                };
            }
            for c in 0..o_dim {
                nl[c] += w_l * plan.node_leaves[t][l][c];
            }
        }
    }

    for r in 0..BILIN_RANK {
        let mut du = 0.0f32;
        let mut dv = 0.0f32;
        for j in 0..p {
            du += plan.bilin_u[r][j] * x_unit[j];
            dv += plan.bilin_v[r][j] * x_unit[j];
        }
        let prod = du.tanh() * dv.tanh();
        for c in 0..o_dim {
            nl[c] += plan.bilin_w[r][c] * prod;
        }
    }

    for c in 0..o_dim {
        out_delta[c] += plan.alpha_table * nl[c];
    }
}

/// Exact closed-form `entmax_1.5(z)` (Peters et al. 2019) in O(p log p).
pub fn entmax_15(logits: &[f32]) -> Vec<f32> {
    let n = logits.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let mut zs: Vec<f32> = logits.iter().map(|&v| v * 0.5).collect();
    let max_z = zs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in &mut zs {
        *v -= max_z;
    }
    let mut sorted = zs.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));

    let mut sum_z = 0.0f32;
    let mut sum_z2 = 0.0f32;
    let mut tau = sorted[0] - 1.0;
    for (k_idx, &zk) in sorted.iter().enumerate() {
        let k = (k_idx + 1) as f32;
        sum_z += zk;
        sum_z2 += zk * zk;
        let mean = sum_z / k;
        let ss = (sum_z2 - k * mean * mean).max(0.0);
        let delta = (1.0 - ss).max(0.0);
        let tau_k = mean - (delta / k).sqrt();
        if k_idx + 1 == n || sorted[k_idx + 1] <= tau_k {
            tau = tau_k;
            break;
        }
    }
    let mut probs: Vec<f32> = zs.iter().map(|&z| (z - tau).max(0.0).powi(2)).collect();
    let s: f32 = probs.iter().sum::<f32>().max(1e-12);
    for p in &mut probs {
        *p /= s;
    }
    probs
}

/// Source-free Receiver Decoder input tensors `(col_feats, edge_feats, glob_feats, code_feats)`.
pub type DecoderInputsV8 = (
    Vec<[f32; DEC_COL_DIM]>,
    Vec<Vec<f32>>,
    [f32; DEC_GLOB_DIM],
    [f32; DEC_CODE_DIM],
);

/// Construct deterministic source-free Receiver Decoder inputs strictly from `Dse8Packet`.
pub fn build_decoder_inputs(pkt: &Dse8Packet) -> DecoderInputsV8 {
    let p = pkt.header.p_eff as usize;
    let mut col = vec![[0.0f32; DEC_COL_DIM]; p];
    for j in 0..p {
        let m = &pkt.marginals[j];
        col[j][..KNOT_COUNT].copy_from_slice(&m.knots17);
        col[j][17..29].copy_from_slice(&pkt.loadings[j]);
        for c in 0..pkt.linear.o_dim.min(V8_MAX_CLASSES) {
            col[j][29 + c] = pkt.linear.w[c][j];
        }
        col[j][37] = m.kind as f32 / 4.0;
        col[j][38] = m.tail_family as f32 / 2.0;
        col[j][39] = if m.is_int { 1.0 } else { 0.0 };
        col[j][40] = m.miss_rate;
        col[j][41] = (1.0 + m.span).ln();
        col[j][42] = pkt.header.z_max;
        col[j][43] = pkt.header.knot_level as f32 / KNOT_COUNT as f32;
        col[j][44] = m.atom_vals.len() as f32 / 8.0;
        col[j][45] = m.heaping_excess;
    }
    let mut edge = vec![vec![0.0f32; DEC_EDGE_DIM]; p * p];
    for i in 0..p {
        for j in 0..p {
            let dot: f32 = (0..V8_MAX_RANK)
                .map(|r| pkt.loadings[i][r] * pkt.loadings[j][r])
                .sum();
            edge[i * p + j][0] = dot.clamp(-1.0, 1.0);
        }
    }
    for g in &pkt.nominal_groups {
        for &a in &g.cols {
            for &b in &g.cols {
                if (a as usize) < p && (b as usize) < p {
                    edge[a as usize * p + b as usize][1] = 1.0;
                }
            }
        }
    }
    for pr in &pkt.pairs {
        let (i, j) = (pr.col_i as usize, pr.col_j as usize);
        if i < p && j < p {
            edge[i * p + j][2] = pr.w_prod;
            edge[j * p + i][2] = pr.w_prod;
            edge[i * p + j][3] = pr.w_xor;
            edge[j * p + i][3] = pr.w_xor;
        }
    }

    let mut glob = [0.0f32; DEC_GLOB_DIM];
    glob[pkt.header.task_id.min(2) as usize] = 1.0;
    glob[3] = if pkt.card.normalized_only { 1.0 } else { 0.0 };
    if pkt.header.task_id == 1 {
        glob[4] = 1.0 - pkt.header.p1_prior;
        glob[5] = pkt.header.p1_prior;
    } else if pkt.header.task_id == 2 {
        let k = pkt.header.k_eff as usize;
        let mx = pkt.linear.b[..k]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = pkt.linear.b[..k]
            .iter()
            .map(|&v| (v - mx).clamp(-20.0, 0.0).exp())
            .collect();
        let s: f32 = exps.iter().sum::<f32>().max(1e-6);
        for c in 0..k.min(8) {
            glob[4 + c] = exps[c] / s;
        }
    }
    glob[12] = pkt.header.log2_nfit_norm;
    glob[13] = pkt.header.p_eff as f32 / V8_MAX_COLS as f32;
    glob[14] = pkt.header.resid_sd;
    glob[15] = pkt.header.k_eff as f32 / V8_MAX_CLASSES as f32;
    glob[16] = pkt.header.inv_nu;
    glob[17] = pkt.regimes.regimes.len() as f32 / 8.0;
    glob[18] = pkt.effects.len() as f32 / 24.0;
    glob[19] = pkt.pairs.len() as f32 / 24.0;
    glob[20] = pkt.trees.trees.len() as f32 / 40.0;
    glob[21] = pkt.nominal_groups.len() as f32 / 8.0;
    glob[22] = pkt.code.tier as f32 / 3.0;
    glob[23] = pkt.linear.temp;

    let z_vec = pkt.code.dequantize(DEC_CODE_DIM);
    let mut code_arr = [0.0f32; DEC_CODE_DIM];
    code_arr.copy_from_slice(&z_vec[..DEC_CODE_DIM]);
    (col, edge, glob, code_arr)
}

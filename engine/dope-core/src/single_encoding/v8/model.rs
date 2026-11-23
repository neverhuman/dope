//! Pure-Rust F32 `rank <= 2` SafeTensors inference runtime for `SingleEncV8`.
//!
//! Zero Python, zero ONNX, zero LibTorch. Implements exact 1:1 parity with PyTorch
//! `research/levels/v8_single_encoding.py`:
//! - `PairSplitAttnBlock`: Pre-LN split-QKV multi-head attention with pairwise edge bias + GELU FFN.
//! - Optional Owner Evidence Encoder (`enc_*`): consumes `(col [P, 128], edge [P, P, 16], glob [160])`
//!   -> emits nested FSQ code `Z` (`<= 256` dims in `[-1, 1]`) and allocation utility bias.
//! - Source-free Receiver Decoder (`dec_*`): consumes dequantized `DSE8` packet features
//!   `(col [P, 48], edge [P, P, 4], glob [32], code [64])` -> emits zero-init-safe `NeuralPlanV8`:
//!   - Knot & loading shrinkage residuals,
//!   - 2-layer row-local RealNVP affine coupling flow on `z_lat`,
//!   - Soft-NODE (`T = 8`, depth `D = 3`, exact closed-form `entmax_1.5`),
//!   - Rank-4 bilinear interaction residual,
//!   - Learned table trust gate `alpha_table = sigmoid(g_raw - 2.0)`,
//!   - Heteroscedastic residual log-scale modifier.

use super::copula_regimes::V8_MAX_RANK;
use super::packet::Dse8Packet;
use super::scale_v8::{V8_MAX_CLASSES, V8_MAX_COLS};
use crate::error::{DopeError, Result};
use crate::evidence::{
    Evidence, COL_DIM as EV_COL_DIM, EDGE_DIM as EV_EDGE_DIM, GLOB_DIM as EV_GLOB_DIM,
};
use crate::single_encoding::math::gelu_exact;
use crate::single_encoding::safetensors::{
    parse_f32_rank2_safetensors, take_layernorm, take_linear, LayerNorm, Linear, TensorMap,
};
use crate::single_encoding::KNOT_COUNT;
use sha2::{Digest, Sha256};

pub use super::model_ops::{
    apply_row_local_flow, build_decoder_inputs, entmax_15, eval_neural_target_residual,
};

pub const DEC_COL_DIM: usize = 48;
pub const DEC_EDGE_DIM: usize = 4;
pub const DEC_GLOB_DIM: usize = 32;
pub const DEC_CODE_DIM: usize = 64;
pub const NODE_TREES: usize = 8;
pub const NODE_DEPTH: usize = 3;
pub const NODE_LEAVES: usize = 8;
pub const BILIN_RANK: usize = 4;
pub const FLOW_HIDDEN: usize = 16;

#[derive(Clone, Debug)]
struct PairSplitAttnBlock {
    n_heads: usize,
    head_dim: usize,
    norm1: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    edge_proj: Linear, // edge_dim -> n_heads
    norm2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
}

impl PairSplitAttnBlock {
    fn take(
        tensors: &TensorMap,
        prefix: &str,
        d_model: usize,
        n_heads: usize,
        edge_dim: usize,
    ) -> Result<Self> {
        let head_dim = d_model / n_heads;
        let ff1 = take_linear(tensors, &format!("{prefix}.ff1"), Some(d_model), None)?;
        let d_ff = ff1.out_dim;
        Ok(Self {
            n_heads,
            head_dim,
            norm1: take_layernorm(tensors, &format!("{prefix}.norm1"), d_model)?,
            q_proj: take_linear(
                tensors,
                &format!("{prefix}.q_proj"),
                Some(d_model),
                Some(d_model),
            )?,
            k_proj: take_linear(
                tensors,
                &format!("{prefix}.k_proj"),
                Some(d_model),
                Some(d_model),
            )?,
            v_proj: take_linear(
                tensors,
                &format!("{prefix}.v_proj"),
                Some(d_model),
                Some(d_model),
            )?,
            o_proj: take_linear(
                tensors,
                &format!("{prefix}.o_proj"),
                Some(d_model),
                Some(d_model),
            )?,
            edge_proj: take_linear(
                tensors,
                &format!("{prefix}.edge_proj"),
                Some(edge_dim),
                Some(n_heads),
            )?,
            norm2: take_layernorm(tensors, &format!("{prefix}.norm2"), d_model)?,
            ff1,
            ff2: take_linear(tensors, &format!("{prefix}.ff2"), Some(d_ff), Some(d_model))?,
        })
    }

    fn forward(&self, x: &[Vec<f32>], edge: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let p = x.len();
        let d = self.n_heads * self.head_dim;
        let mut q = Vec::with_capacity(p);
        let mut k = Vec::with_capacity(p);
        let mut v = Vec::with_capacity(p);
        for c in 0..p {
            let h = self.norm1.forward_vec(&x[c]);
            q.push(self.q_proj.forward_vec(&h));
            k.push(self.k_proj.forward_vec(&h));
            v.push(self.v_proj.forward_vec(&h));
        }
        // Precompute edge attention bias [p * p][n_heads]
        let mut e_bias = Vec::with_capacity(p * p);
        for e_ij in edge.iter().take(p * p) {
            e_bias.push(self.edge_proj.forward_vec(e_ij));
        }

        let scale = (self.head_dim as f32).sqrt().recip();
        let mut attn_out = vec![vec![0.0f32; d]; p];
        for h in 0..self.n_heads {
            let off = h * self.head_dim;
            for i in 0..p {
                let mut scores = vec![0.0f32; p];
                let mut max_s = f32::NEG_INFINITY;
                for j in 0..p {
                    let mut dot = 0.0f32;
                    for t in 0..self.head_dim {
                        dot += q[i][off + t] * k[j][off + t];
                    }
                    let s = dot * scale + e_bias[i * p + j][h];
                    scores[j] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }
                let mut sum_e = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    sum_e += *s;
                }
                let inv_e = sum_e.max(1e-12).recip();
                for j in 0..p {
                    let w = scores[j] * inv_e;
                    for t in 0..self.head_dim {
                        attn_out[i][off + t] += w * v[j][off + t];
                    }
                }
            }
        }
        let mut out = Vec::with_capacity(p);
        for i in 0..p {
            let op = self.o_proj.forward_vec(&attn_out[i]);
            let h1: Vec<f32> = (0..d).map(|t| x[i][t] + op[t]).collect();
            let n2 = self.norm2.forward_vec(&h1);
            let mut ff = self.ff1.forward_vec(&n2);
            for val in &mut ff {
                *val = gelu_exact(*val);
            }
            let ff2 = self.ff2.forward_vec(&ff);
            out.push((0..d).map(|t| h1[t] + ff2[t]).collect());
        }
        out
    }
}

/// Optional Owner Evidence Encoder (`enc_*` prefix in SafeTensors).
#[derive(Clone, Debug)]
struct OwnerEncoderV8 {
    d_model: usize,
    col_in: Linear,
    glob_in: Linear,
    blocks: Vec<PairSplitAttnBlock>,
    final_norm: LayerNorm,
    code_head: Linear, // d_model -> 64..256
}

/// Pure-Rust `SingleEncV8` model weights (strictly F32, `rank <= 2`, `<= 16 MiB`).
#[derive(Clone, Debug)]
pub struct SingleEncV8Weights {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_dec_layers: usize,
    sha256: [u8; 32],
    owner_enc: Option<OwnerEncoderV8>,
    dec_col_in: Linear,
    dec_glob_in: Linear,
    dec_code_in: Linear,
    dec_blocks: Vec<PairSplitAttnBlock>,
    dec_final_norm: LayerNorm,
    delta_knots: Linear,
    delta_l: Linear,
    delta_w: Linear,
    // Soft-NODE heads: T=8, D=3 -> per-col selector logits [T*D], global thresholds [T*D], leaves [T*8*K]
    node_sel: Linear,  // d_model -> T * D (per column!)
    node_thr: Linear,  // d_model -> T * D (global)
    node_leaf: Linear, // d_model -> T * 8 * V8_MAX_CLASSES (global)
    // Rank-4 bilinear residual: per-col U [4], V [4], global W [4 * K]
    bilin_u: Linear, // d_model -> 4
    bilin_v: Linear, // d_model -> 4
    bilin_w: Linear, // d_model -> 4 * V8_MAX_CLASSES
    // 2-layer row-local RealNVP coupling flow (global hypernet -> 2*(w1, b1, s, t) over split halves)
    flow_scale_head: Linear, // d_model -> 2 (per col: 2 layers of scale/shift modulated by other half)
    flow_shift_head: Linear, // d_model -> 2
    // Trust gate alpha_table = sigmoid(g - 2.0) & residual scale modifier
    gate_head: Linear, // d_model -> 2 ([0]=trust_gate_raw, [1]=log_resid_sd_delta)
}

/// Precomputed per-packet neural residual plan (strictly row-local O(P * R + T * P) decode!).
#[derive(Clone, Debug)]
pub struct NeuralPlanV8 {
    pub alpha_table: f32,
    pub resid_sd_mult: f32,
    pub delta_knots: Vec<[f32; KNOT_COUNT]>,
    pub delta_l: Vec<[f32; V8_MAX_RANK]>,
    pub delta_w: Vec<Vec<f32>>, // [o_dim][p_eff]
    // Soft-NODE [T][D][p_eff] entmax_1.5 weights, thresholds [T][D], leaves [T][8][o_dim]
    pub node_w: [[[f32; V8_MAX_COLS]; NODE_DEPTH]; NODE_TREES],
    pub node_thr: [[f32; NODE_DEPTH]; NODE_TREES],
    pub node_leaves: [[[f32; V8_MAX_CLASSES]; NODE_LEAVES]; NODE_TREES],
    // Rank-4 bilinear U[r][p], V[r][p], W[r][c]
    pub bilin_u: [[f32; V8_MAX_COLS]; BILIN_RANK],
    pub bilin_v: [[f32; V8_MAX_COLS]; BILIN_RANK],
    pub bilin_w: [[f32; V8_MAX_CLASSES]; BILIN_RANK],
    // Lightweight zero-init-safe 2-layer row-local RealNVP coupling weights per col
    pub flow_s: Vec<[f32; 2]>,
    pub flow_t: Vec<[f32; 2]>,
    pub is_identity: bool,
}

impl SingleEncV8Weights {
    /// Fail-closed parser for canonical F32 `rank <= 2` SafeTensors (`<= 16 MiB`).
    pub fn from_safetensors_bytes(bytes: &[u8], n_heads: usize) -> Result<Self> {
        let tensors = parse_f32_rank2_safetensors(bytes)?;
        let dec_col_in = take_linear(&tensors, "dec_col_in", Some(DEC_COL_DIM), None)?;
        let d_model = dec_col_in.out_dim;
        if n_heads == 0 || !d_model.is_multiple_of(n_heads) {
            return Err(DopeError::Artifact("v8 invalid n_heads for d_model".into()));
        }

        // Optional Owner Evidence Encoder (`enc_*`)
        let owner_enc = if tensors.contains_key("enc_col_in.weight") {
            let e_col = take_linear(&tensors, "enc_col_in", Some(EV_COL_DIM), Some(d_model))?;
            let e_glob = take_linear(&tensors, "enc_glob_in", Some(EV_GLOB_DIM), Some(d_model))?;
            let mut e_blocks = Vec::new();
            let mut idx = 0usize;
            while tensors.contains_key(&format!("enc_blocks.{idx}.q_proj.weight")) {
                e_blocks.push(PairSplitAttnBlock::take(
                    &tensors,
                    &format!("enc_blocks.{idx}"),
                    d_model,
                    n_heads,
                    EV_EDGE_DIM,
                )?);
                idx += 1;
            }
            let e_norm = take_layernorm(&tensors, "enc_final_norm", d_model)?;
            let code_head = take_linear(&tensors, "enc_code_head", Some(d_model), None)?;
            Some(OwnerEncoderV8 {
                d_model,
                col_in: e_col,
                glob_in: e_glob,
                blocks: e_blocks,
                final_norm: e_norm,
                code_head,
            })
        } else {
            None
        };

        let dec_glob_in = take_linear(&tensors, "dec_glob_in", Some(DEC_GLOB_DIM), Some(d_model))?;
        let dec_code_in = take_linear(&tensors, "dec_code_in", Some(DEC_CODE_DIM), Some(d_model))?;
        let mut dec_blocks = Vec::new();
        let mut idx = 0usize;
        while tensors.contains_key(&format!("dec_blocks.{idx}.q_proj.weight")) {
            dec_blocks.push(PairSplitAttnBlock::take(
                &tensors,
                &format!("dec_blocks.{idx}"),
                d_model,
                n_heads,
                DEC_EDGE_DIM,
            )?);
            idx += 1;
        }
        let dec_final_norm = take_layernorm(&tensors, "dec_final_norm", d_model)?;
        let delta_knots = take_linear(&tensors, "delta_knots", Some(d_model), Some(KNOT_COUNT))?;
        let delta_l = take_linear(&tensors, "delta_L", Some(d_model), Some(V8_MAX_RANK))?;
        let delta_w = take_linear(&tensors, "delta_W", Some(d_model), Some(V8_MAX_CLASSES))?;
        let node_sel = take_linear(
            &tensors,
            "node_sel",
            Some(d_model),
            Some(NODE_TREES * NODE_DEPTH),
        )?;
        let node_thr = take_linear(
            &tensors,
            "node_thr",
            Some(d_model),
            Some(NODE_TREES * NODE_DEPTH),
        )?;
        let node_leaf = take_linear(
            &tensors,
            "node_leaf",
            Some(d_model),
            Some(NODE_TREES * NODE_LEAVES * V8_MAX_CLASSES),
        )?;
        let bilin_u = take_linear(&tensors, "bilin_u", Some(d_model), Some(BILIN_RANK))?;
        let bilin_v = take_linear(&tensors, "bilin_v", Some(d_model), Some(BILIN_RANK))?;
        let bilin_w = take_linear(
            &tensors,
            "bilin_w",
            Some(d_model),
            Some(BILIN_RANK * V8_MAX_CLASSES),
        )?;
        let flow_scale_head = take_linear(&tensors, "flow_scale_head", Some(d_model), Some(2))?;
        let flow_shift_head = take_linear(&tensors, "flow_shift_head", Some(d_model), Some(2))?;
        let gate_head = take_linear(&tensors, "gate_head", Some(d_model), Some(2))?;

        Ok(Self {
            d_model,
            n_heads,
            n_dec_layers: dec_blocks.len(),
            sha256: Sha256::digest(bytes).into(),
            owner_enc,
            dec_col_in,
            dec_glob_in,
            dec_code_in,
            dec_blocks,
            dec_final_norm,
            delta_knots,
            delta_l,
            delta_w,
            node_sel,
            node_thr,
            node_leaf,
            bilin_u,
            bilin_v,
            bilin_w,
            flow_scale_head,
            flow_shift_head,
            gate_head,
        })
    }

    /// Full SHA-256 of the SafeTensors bytes these weights were loaded from.
    pub fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    /// 8-byte decoder weights fingerprint stored in `TAG_WEIGHTS_FP` for fail-closed identity checks.
    pub fn fingerprint(&self) -> [u8; 8] {
        let mut fp = [0u8; 8];
        fp.copy_from_slice(&self.sha256[..8]);
        fp
    }

    /// Run Owner Evidence Encoder on FIT-only `Evidence` -> continuous code in `[-1, 1]` for FSQ.
    pub fn encode_evidence_code(&self, ev: &Evidence) -> Option<Vec<f32>> {
        let enc = self.owner_enc.as_ref()?;
        let p = ev.p;
        if p == 0 {
            return None;
        }
        let g_tok = enc.glob_in.forward_vec(&ev.glob);
        let mut h = Vec::with_capacity(p);
        for c in 0..p {
            let ci = enc.col_in.forward_vec(&ev.col[c]);
            h.push((0..enc.d_model).map(|t| ci[t] + g_tok[t]).collect());
        }
        let edge_vecs: Vec<Vec<f32>> = ev.edge.iter().map(|e| e.to_vec()).collect();
        for blk in &enc.blocks {
            h = blk.forward(&h, &edge_vecs);
        }
        let mut g_pool = vec![0.0f32; enc.d_model];
        for row in &h {
            let n = enc.final_norm.forward_vec(row);
            for t in 0..enc.d_model {
                g_pool[t] += n[t] / p as f32;
            }
        }
        let raw = enc.code_head.forward_vec(&g_pool);
        Some(raw.into_iter().map(|v| v.tanh()).collect())
    }

    /// Source-free Receiver Decoder: maps `Dse8Packet` -> precomputed row-local `NeuralPlanV8`.
    pub fn build_neural_plan(&self, pkt: &Dse8Packet) -> NeuralPlanV8 {
        let p = pkt.header.p_eff as usize;
        let o_dim = pkt.linear.o_dim;
        let (col_feats, edge_feats, glob_feats, code_feats) = build_decoder_inputs(pkt);

        let g_tok = self.dec_glob_in.forward_vec(&glob_feats);
        let z_tok = self.dec_code_in.forward_vec(&code_feats);
        let mut h = Vec::with_capacity(p);
        for c in 0..p {
            let ci = self.dec_col_in.forward_vec(&col_feats[c]);
            h.push(
                (0..self.d_model)
                    .map(|t| ci[t] + g_tok[t] + z_tok[t])
                    .collect(),
            );
        }
        for blk in &self.dec_blocks {
            h = blk.forward(&h, &edge_feats);
        }
        let mut g_pool = vec![0.0f32; self.d_model];
        for c in 0..p {
            h[c] = self.dec_final_norm.forward_vec(&h[c]);
            for t in 0..self.d_model {
                g_pool[t] += h[c][t] / p.max(1) as f32;
            }
        }

        let mut delta_knots = vec![[0.0f32; KNOT_COUNT]; p];
        let mut delta_l = vec![[0.0f32; V8_MAX_RANK]; p];
        let mut delta_w = vec![vec![0.0f32; p]; o_dim];
        let mut node_logits = [[[0.0f32; V8_MAX_COLS]; NODE_DEPTH]; NODE_TREES];
        let mut bilin_u = [[0.0f32; V8_MAX_COLS]; BILIN_RANK];
        let mut bilin_v = [[0.0f32; V8_MAX_COLS]; BILIN_RANK];
        let mut flow_s = vec![[0.0f32; 2]; p];
        let mut flow_t = vec![[0.0f32; 2]; p];
        let mut max_effect = 0.0f32;

        for c in 0..p {
            let dk = self.delta_knots.forward_vec(&h[c]);
            for k in 0..KNOT_COUNT {
                let v = 0.06 * dk[k].tanh();
                delta_knots[c][k] = v;
                max_effect = max_effect.max(v.abs());
            }
            let dl = self.delta_l.forward_vec(&h[c]);
            for r in 0..V8_MAX_RANK {
                let v = 0.10 * dl[r].tanh();
                delta_l[c][r] = v;
                max_effect = max_effect.max(v.abs());
            }
            let dw = self.delta_w.forward_vec(&h[c]);
            for cls in 0..o_dim {
                let v = 0.50 * dw[cls].tanh();
                delta_w[cls][c] = v;
                max_effect = max_effect.max(v.abs());
            }
            let ns = self.node_sel.forward_vec(&h[c]);
            for t in 0..NODE_TREES {
                for d in 0..NODE_DEPTH {
                    node_logits[t][d][c] = ns[t * NODE_DEPTH + d];
                }
            }
            let bu = self.bilin_u.forward_vec(&h[c]);
            let bv = self.bilin_v.forward_vec(&h[c]);
            for r in 0..BILIN_RANK {
                bilin_u[r][c] = 0.50 * bu[r].tanh();
                bilin_v[r][c] = 0.50 * bv[r].tanh();
            }
            let fs = self.flow_scale_head.forward_vec(&h[c]);
            let ft = self.flow_shift_head.forward_vec(&h[c]);
            for l in 0..2 {
                flow_s[c][l] = 0.35 * fs[l].tanh();
                flow_t[c][l] = 0.35 * ft[l].tanh();
                max_effect = max_effect.max(flow_s[c][l].abs()).max(flow_t[c][l].abs());
            }
        }

        // Exact closed-form entmax_1.5 across columns for each Soft-NODE split!
        let mut node_w = [[[0.0f32; V8_MAX_COLS]; NODE_DEPTH]; NODE_TREES];
        for t in 0..NODE_TREES {
            for d in 0..NODE_DEPTH {
                let probs = entmax_15(&node_logits[t][d][..p]);
                node_w[t][d][..p].copy_from_slice(&probs);
            }
        }

        let nt_raw = self.node_thr.forward_vec(&g_pool);
        let mut node_thr = [[0.0f32; NODE_DEPTH]; NODE_TREES];
        for t in 0..NODE_TREES {
            for d in 0..NODE_DEPTH {
                node_thr[t][d] = 2.0 * nt_raw[t * NODE_DEPTH + d].tanh();
            }
        }
        let nl_raw = self.node_leaf.forward_vec(&g_pool);
        let mut node_leaves = [[[0.0f32; V8_MAX_CLASSES]; NODE_LEAVES]; NODE_TREES];
        for t in 0..NODE_TREES {
            for l in 0..NODE_LEAVES {
                for cls in 0..V8_MAX_CLASSES {
                    let v = 0.75 * nl_raw[(t * NODE_LEAVES + l) * V8_MAX_CLASSES + cls].tanh();
                    node_leaves[t][l][cls] = v;
                    max_effect = max_effect.max(v.abs());
                }
            }
        }
        let bw_raw = self.bilin_w.forward_vec(&g_pool);
        let mut bilin_w = [[0.0f32; V8_MAX_CLASSES]; BILIN_RANK];
        for r in 0..BILIN_RANK {
            for cls in 0..V8_MAX_CLASSES {
                let v = 0.50 * bw_raw[r * V8_MAX_CLASSES + cls].tanh();
                bilin_w[r][cls] = v;
                max_effect = max_effect.max(v.abs());
            }
        }
        let gate_raw = self.gate_head.forward_vec(&g_pool);
        let alpha_table = 1.0 / (1.0 + (-(gate_raw[0] - 2.0)).exp());
        let resid_sd_mult = (0.50 * gate_raw[1].tanh()).exp();
        max_effect = max_effect.max((resid_sd_mult - 1.0).abs());

        NeuralPlanV8 {
            alpha_table,
            resid_sd_mult,
            delta_knots,
            delta_l,
            delta_w,
            node_w,
            node_thr,
            node_leaves,
            bilin_u,
            bilin_v,
            bilin_w,
            flow_s,
            flow_t,
            is_identity: max_effect < 1e-7,
        }
    }
}

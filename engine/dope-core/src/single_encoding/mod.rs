//! Pure-Rust inference runtime for `SingleEncV7` (`<= 2 KiB`, `source_free` packet residual model).
//!
//! Loads canonical F32 HuggingFace SafeTensors checkpoints (`rank <= 2`, `< 8 MiB` target, hard
//! `16 MiB` fail-closed parser cap) with zero Python/libtorch dependencies, refines dequantized
//! `dope.single-encoding/v1`-aligned packet statistics via split-QKV column attention, and decodes
//! synthetic rows strictly row-locally (zero batch statistics, zero cross-row state).

mod math;
pub mod packet;
mod safetensors;
pub mod scale_card;
pub mod v8;

use crate::error::{DopeError, Result};
use crate::rng::Rng;
use math::{gelu_exact, silu};
use safetensors::{parse_f32_rank2_safetensors, take_layernorm, take_linear, LayerNorm, Linear};

pub use packet::{fit_dse7_packet, unpack_dse7_packet, Dse7Header, DSE7_MAGIC};
pub use safetensors::{MAX_HEADER_BYTES, MAX_SAFETENSORS_BYTES};
pub use scale_card::{sample_zero_init_rows, ScaleCardV7};

pub const MAX_COLS: usize = 48;
pub const KNOT_COUNT: usize = 17;
pub const MAX_RANK: usize = 12;
pub const MAX_CLASSES: usize = 8;
pub const COL_FEAT_DIM: usize = KNOT_COUNT + MAX_RANK + MAX_CLASSES + 5; // 42 (V7.0 / V7.1)
pub const COL_FEAT_DIM_V2: usize = COL_FEAT_DIM + 4; // 46 (V7.2 with 4-quartile nonlinear residuals)
pub const GLOB_FEAT_DIM: usize = 4 + MAX_CLASSES + 4; // 16

#[derive(Clone, Debug)]
struct SplitAttnBlock {
    n_heads: usize,
    head_dim: usize,
    norm1: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    norm2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
}

impl SplitAttnBlock {
    fn forward(&self, x: &[Vec<f32>], mask: &[f32; MAX_COLS]) -> Vec<Vec<f32>> {
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
        let scale = (self.head_dim as f32).sqrt().recip();
        let mut attn_out = vec![vec![0.0f32; d]; p];
        for h in 0..self.n_heads {
            let off = h * self.head_dim;
            for i in 0..p {
                let mut scores = [0.0f32; MAX_COLS];
                let mut max_s = f32::NEG_INFINITY;
                for j in 0..p {
                    let s = if mask[j] < 0.5 {
                        -1e4
                    } else {
                        let mut dot = 0.0f32;
                        for t in 0..self.head_dim {
                            dot += q[i][off + t] * k[j][off + t];
                        }
                        dot * scale
                    };
                    scores[j] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }
                let mut sum_e = 0.0f32;
                for j in 0..p {
                    let e = (scores[j] - max_s).exp();
                    scores[j] = e;
                    sum_e += e;
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
            let o_proj = self.o_proj.forward_vec(&attn_out[i]);
            let mut h1 = vec![0.0f32; d];
            for t in 0..d {
                h1[t] = x[i][t] + o_proj[t];
            }
            let n2 = self.norm2.forward_vec(&h1);
            let mut ff = self.ff1.forward_vec(&n2);
            for val in &mut ff {
                *val = gelu_exact(*val);
            }
            let ff2 = self.ff2.forward_vec(&ff);
            let m = mask[i];
            let mut h2 = vec![0.0f32; d];
            for t in 0..d {
                h2[t] = (h1[t] + ff2[t]) * m;
            }
            out.push(h2);
        }
        out
    }
}

/// Dequantized `SingleEncV7` packet input (strictly `source_free`: zero raw rows).
#[derive(Clone, Debug)]
pub struct SingleEncV7Packet {
    pub col_feats: [[f32; COL_FEAT_DIM]; MAX_COLS],
    pub q_nl: [[f32; 4]; MAX_COLS],
    pub glob_feats: [f32; GLOB_FEAT_DIM],
    pub knots: [[f32; KNOT_COUNT]; MAX_COLS],
    pub loadings: [[f32; MAX_RANK]; MAX_COLS],
    pub w_base: [[f32; MAX_COLS]; MAX_CLASSES],
    pub b_base: [f32; MAX_CLASSES],
    pub resid_sd: f32,
    pub col_mask: [f32; MAX_COLS],
    pub is_const: [f32; MAX_COLS],
}

/// Precomputed per-packet refined parameters for O(P * R + P * H) row-local decoding.
#[derive(Clone, Debug)]
pub struct RefinedV7Packet {
    pub l_eff: [[f32; MAX_RANK]; MAX_COLS],
    pub noise_sd: [f32; MAX_COLS],
    pub knots_eff: [[f32; KNOT_COUNT]; MAX_COLS],
    pub knots_base: [[f32; KNOT_COUNT]; MAX_COLS],
    pub is_disc: [bool; MAX_COLS],
    pub is_const: [f32; MAX_COLS],
    pub col_mask: [f32; MAX_COLS],
    pub z_max: f32,
    pub w_eff: [[f32; MAX_COLS]; MAX_CLASSES],
    pub b_base: [f32; MAX_CLASSES],
    pub h_target: usize,
    pub hyper_w1: Vec<[f32; MAX_COLS]>,
    pub hyper_b1: Vec<f32>,
    pub hyper_w2: Vec<[f32; MAX_CLASSES]>,
    pub temp_scale: f32,
    pub resid_sd_eff: f32,
}

/// One decoded synthetic row (features in normalized packet coordinates + target logits/regression).
#[derive(Clone, Debug)]
pub struct DecodedV7Row {
    pub x: [f32; MAX_COLS],
    pub logits: [f32; MAX_CLASSES],
    pub y_reg: f32,
}

/// Pure-Rust `SingleEncV7` model weights loaded from canonical F32 rank-<=2 SafeTensors.
#[derive(Clone, Debug)]
pub struct SingleEncV7Weights {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub h_target: usize,
    col_in: Linear,
    glob_in: Linear,
    blocks: Vec<SplitAttnBlock>,
    final_norm: LayerNorm,
    delta_knots: Linear,
    delta_l: Linear,
    delta_w: Linear,
    hyper_w1: Linear,
    hyper_b1: Linear,
    hyper_w2: Linear,
    hyper_temp: Linear,
}

impl SingleEncV7Weights {
    /// Fail-closed parser for canonical HF SafeTensors (`dtype == "F32"`, `rank <= 2`).
    /// Auto-detects both V7.0/V7.1 (`col_in = 42`) and V7.2 (`col_in = 46` with `q_nl`).
    pub fn from_safetensors_bytes(bytes: &[u8], n_heads: usize) -> Result<Self> {
        let tensors = parse_f32_rank2_safetensors(bytes)?;
        let col_in = take_linear(&tensors, "col_in", None, None)?;
        if col_in.in_dim != COL_FEAT_DIM && col_in.in_dim != COL_FEAT_DIM_V2 {
            return Err(DopeError::Artifact(
                "col_in input dim must be 42 (v7.0/v7.1) or 46 (v7.2)".into(),
            ));
        }
        let d_model = col_in.out_dim;
        if n_heads == 0 || !d_model.is_multiple_of(n_heads) {
            return Err(DopeError::Artifact("invalid n_heads for d_model".into()));
        }
        let head_dim = d_model / n_heads;
        let glob_in = take_linear(&tensors, "glob_in", Some(GLOB_FEAT_DIM), Some(d_model))?;
        let mut blocks = Vec::new();
        let mut idx = 0usize;
        while tensors.contains_key(&format!("blocks.{idx}.q_proj.weight")) {
            let pfx = format!("blocks.{idx}");
            let ff1 = take_linear(&tensors, &format!("{pfx}.ff1"), Some(d_model), None)?;
            let d_ff = ff1.out_dim;
            blocks.push(SplitAttnBlock {
                n_heads,
                head_dim,
                norm1: take_layernorm(&tensors, &format!("{pfx}.norm1"), d_model)?,
                q_proj: take_linear(
                    &tensors,
                    &format!("{pfx}.q_proj"),
                    Some(d_model),
                    Some(d_model),
                )?,
                k_proj: take_linear(
                    &tensors,
                    &format!("{pfx}.k_proj"),
                    Some(d_model),
                    Some(d_model),
                )?,
                v_proj: take_linear(
                    &tensors,
                    &format!("{pfx}.v_proj"),
                    Some(d_model),
                    Some(d_model),
                )?,
                o_proj: take_linear(
                    &tensors,
                    &format!("{pfx}.o_proj"),
                    Some(d_model),
                    Some(d_model),
                )?,
                norm2: take_layernorm(&tensors, &format!("{pfx}.norm2"), d_model)?,
                ff1,
                ff2: take_linear(&tensors, &format!("{pfx}.ff2"), Some(d_ff), Some(d_model))?,
            });
            idx += 1;
        }
        let final_norm = take_layernorm(&tensors, "final_norm", d_model)?;
        let delta_knots = take_linear(&tensors, "delta_knots", Some(d_model), Some(KNOT_COUNT))?;
        let delta_l = take_linear(&tensors, "delta_L", Some(d_model), Some(MAX_RANK))?;
        let delta_w = take_linear(&tensors, "delta_W", Some(d_model), Some(MAX_CLASSES))?;
        let hyper_w1 = take_linear(&tensors, "hyper_w1", Some(d_model), None)?;
        let h_target = hyper_w1.out_dim;
        let hyper_b1 = take_linear(&tensors, "hyper_b1", Some(d_model), Some(h_target))?;
        let hyper_w2 = take_linear(
            &tensors,
            "hyper_w2",
            Some(d_model),
            Some(h_target * MAX_CLASSES),
        )?;
        let hyper_temp = take_linear(&tensors, "hyper_temp", Some(d_model), Some(2))?;
        Ok(Self {
            d_model,
            n_heads,
            n_layers: blocks.len(),
            h_target,
            col_in,
            glob_in,
            blocks,
            final_norm,
            delta_knots,
            delta_l,
            delta_w,
            hyper_w1,
            hyper_b1,
            hyper_w2,
            hyper_temp,
        })
    }

    /// Run column Transformer once per packet to build a fast row-local decoder state.
    pub fn refine_packet(&self, pkt: &SingleEncV7Packet) -> RefinedV7Packet {
        let g_tok = self.glob_in.forward_vec(&pkt.glob_feats);
        let use_v2 = self.col_in.in_dim == COL_FEAT_DIM_V2;
        let mut feat_buf = [0.0f32; COL_FEAT_DIM_V2];
        let mut h = Vec::with_capacity(MAX_COLS);
        for c in 0..MAX_COLS {
            let ci = if use_v2 {
                feat_buf[..COL_FEAT_DIM].copy_from_slice(&pkt.col_feats[c]);
                feat_buf[COL_FEAT_DIM..].copy_from_slice(&pkt.q_nl[c]);
                self.col_in.forward_vec(&feat_buf)
            } else {
                self.col_in.forward_vec(&pkt.col_feats[c])
            };
            let m = pkt.col_mask[c];
            h.push((0..self.d_model).map(|t| (ci[t] + g_tok[t]) * m).collect());
        }
        for blk in &self.blocks {
            h = blk.forward(&h, &pkt.col_mask);
        }
        let mut g_pool = vec![0.0f32; self.d_model];
        let denom: f32 = pkt.col_mask.iter().sum::<f32>().max(1.0);
        for c in 0..MAX_COLS {
            let m = pkt.col_mask[c];
            h[c] = self
                .final_norm
                .forward_vec(&h[c])
                .into_iter()
                .map(|v| v * m)
                .collect();
            for t in 0..self.d_model {
                g_pool[t] += h[c][t] / denom;
            }
        }

        let mut l_eff = [[0.0f32; MAX_RANK]; MAX_COLS];
        let mut noise_sd = [1.0f32; MAX_COLS];
        let mut knots_eff = [[0.0f32; KNOT_COUNT]; MAX_COLS];
        let mut w_eff = [[0.0f32; MAX_COLS]; MAX_CLASSES];
        let mut is_disc = [false; MAX_COLS];
        let mut hyper_w1 = vec![[0.0f32; MAX_COLS]; self.h_target];

        for c in 0..MAX_COLS {
            is_disc[c] = pkt.col_feats[c][COL_FEAT_DIM - 4] > 0.5;
            let dl = self.delta_l.forward_vec(&h[c]);
            let mut l_raw = [0.0f32; MAX_RANK];
            let mut sq = 0.0f32;
            for r in 0..MAX_RANK {
                let v = pkt.loadings[c][r] + 0.10 * dl[r].tanh();
                l_raw[r] = v;
                sq += v * v;
            }
            let scale = (sq.sqrt() / 0.99).max(1.0);
            let mut sq_eff = 0.0f32;
            for r in 0..MAX_RANK {
                l_eff[c][r] = l_raw[r] / scale;
                sq_eff += l_eff[c][r] * l_eff[c][r];
            }
            noise_sd[c] = (1.0 - sq_eff).max(0.02).sqrt();

            let dk = self.delta_knots.forward_vec(&h[c]);
            let mut base_diffs = [0.0f32; KNOT_COUNT - 1];
            let mut w_raw = [0.0f32; KNOT_COUNT - 1];
            let mut span_b = 0.0f32;
            let mut w_sum = 0.0f32;
            for k in 0..(KNOT_COUNT - 1) {
                let bd = (pkt.knots[c][k + 1] - pkt.knots[c][k]).max(0.0);
                let dk_scaled = 0.06 * dk[k + 1].tanh();
                let wr = bd * (0.25 * dk_scaled.tanh()).exp();
                base_diffs[k] = bd;
                w_raw[k] = wr;
                span_b += bd;
                w_sum += wr;
            }
            let k0 = pkt.knots[c][0];
            knots_eff[c][0] = k0;
            let mut acc = k0;
            for k in 0..(KNOT_COUNT - 1) {
                let step = if span_b > 1e-7 {
                    w_raw[k] * (span_b / w_sum.max(1e-7))
                } else {
                    base_diffs[k]
                };
                acc += step;
                knots_eff[c][k + 1] = acc;
            }

            let dw = self.delta_w.forward_vec(&h[c]);
            let cm = pkt.col_mask[c];
            for cls in 0..MAX_CLASSES {
                w_eff[cls][c] = (pkt.w_base[cls][c] + 0.50 * dw[cls].tanh()) * cm;
            }
            let w1_c = self.hyper_w1.forward_vec(&h[c]);
            for hi in 0..self.h_target {
                hyper_w1[hi][c] = 0.50 * w1_c[hi].tanh();
            }
        }

        let hyper_b1: Vec<f32> = self
            .hyper_b1
            .forward_vec(&g_pool)
            .into_iter()
            .map(|v| 0.50 * v.tanh())
            .collect();
        let w2_raw = self.hyper_w2.forward_vec(&g_pool);
        let mut hyper_w2 = vec![[0.0f32; MAX_CLASSES]; self.h_target];
        for hi in 0..self.h_target {
            for cls in 0..MAX_CLASSES {
                hyper_w2[hi][cls] = 0.50 * w2_raw[hi * MAX_CLASSES + cls].tanh();
            }
        }
        let temp_raw = self.hyper_temp.forward_vec(&g_pool);
        let temp_scale = (0.50 * temp_raw[0].tanh()).exp();
        let resid_sd_eff = pkt.resid_sd * (0.50 * temp_raw[1].tanh()).exp();
        let z_max = pkt.col_feats[0][COL_FEAT_DIM - 2].clamp(1.2, 2.6);

        RefinedV7Packet {
            l_eff,
            noise_sd,
            knots_eff,
            knots_base: pkt.knots,
            is_disc,
            is_const: pkt.is_const,
            col_mask: pkt.col_mask,
            z_max,
            w_eff,
            b_base: pkt.b_base,
            h_target: self.h_target,
            hyper_w1,
            hyper_b1,
            hyper_w2,
            temp_scale,
            resid_sd_eff,
        }
    }

    /// Strictly row-local decode of one synthetic row from counter noise `(z_factor, z_noise, y_noise)`.
    pub fn decode_row(
        ref_pkt: &RefinedV7Packet,
        z_factor: &[f32; MAX_RANK],
        z_noise: &[f32; MAX_COLS],
        y_noise: f32,
    ) -> DecodedV7Row {
        let z_max = ref_pkt.z_max;
        let dz = (2.0 * z_max) / (KNOT_COUNT - 1) as f32;
        let mut x = [0.0f32; MAX_COLS];
        for c in 0..MAX_COLS {
            if ref_pkt.col_mask[c] < 0.5 || ref_pkt.is_const[c] > 0.5 {
                x[c] = 0.0;
                continue;
            }
            let mut z_lat = z_noise[c] * ref_pkt.noise_sd[c];
            for r in 0..MAX_RANK {
                z_lat += z_factor[r] * ref_pkt.l_eff[c][r];
            }
            let pos = (z_lat.clamp(-z_max, z_max) + z_max) / dz;
            let i0 = (pos.floor() as isize).clamp(0, (KNOT_COUNT - 2) as isize) as usize;
            let frac = (pos - i0 as f32).clamp(0.0, 1.0);
            let val = if ref_pkt.is_disc[c] {
                if frac < 0.5 {
                    ref_pkt.knots_base[c][i0]
                } else {
                    ref_pkt.knots_base[c][i0 + 1]
                }
            } else {
                let ke = &ref_pkt.knots_eff[c];
                let mut xc = ke[i0] + frac * (ke[i0 + 1] - ke[i0]);
                let slope_lo = 0.25 * (ke[1] - ke[0]) / dz;
                let slope_hi = 0.25 * (ke[KNOT_COUNT - 1] - ke[KNOT_COUNT - 2]) / dz;
                xc += slope_lo * (z_lat + z_max).min(0.0) + slope_hi * (z_lat - z_max).max(0.0);
                xc
            };
            x[c] = val.clamp(-6.0, 6.0) * (1.0 - ref_pkt.is_const[c]) * ref_pkt.col_mask[c];
        }

        let mut hid = vec![0.0f32; ref_pkt.h_target];
        for hi in 0..ref_pkt.h_target {
            let mut acc = ref_pkt.hyper_b1[hi];
            for c in 0..MAX_COLS {
                acc += x[c] * ref_pkt.hyper_w1[hi][c];
            }
            hid[hi] = silu(acc);
        }
        let mut logits = [0.0f32; MAX_CLASSES];
        for cls in 0..MAX_CLASSES {
            let mut lin = ref_pkt.b_base[cls];
            for c in 0..MAX_COLS {
                lin += x[c] * ref_pkt.w_eff[cls][c];
            }
            let mut nl = 0.0f32;
            for hi in 0..ref_pkt.h_target {
                nl += hid[hi] * ref_pkt.hyper_w2[hi][cls];
            }
            logits[cls] = (lin + nl) * ref_pkt.temp_scale;
        }
        let y_reg = logits[0] + ref_pkt.resid_sd_eff * y_noise;
        DecodedV7Row { x, logits, y_reg }
    }

    /// Counter-seeded deterministic row-local sampling (`(seed, row_index)` -> `DecodedV7Row`).
    pub fn sample_counter_rows(
        &self,
        pkt: &SingleEncV7Packet,
        n_rows: usize,
        seed: u64,
    ) -> Vec<DecodedV7Row> {
        let refined = self.refine_packet(pkt);
        let mut out = Vec::with_capacity(n_rows);
        for row_idx in 0..n_rows {
            let row_seed = seed ^ (row_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut rng = Rng::new(row_seed);
            let mut zf = [0.0f32; MAX_RANK];
            for v in &mut zf {
                *v = rng.normal() as f32;
            }
            let mut zn = [0.0f32; MAX_COLS];
            for v in &mut zn {
                *v = rng.normal() as f32;
            }
            let yn = rng.normal() as f32;
            out.push(Self::decode_row(&refined, &zf, &zn, yn));
        }
        out
    }
}

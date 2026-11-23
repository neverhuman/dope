//! Compact distilled depth-3 oblivious residual forest (`TAG_TREES`, `Tr`) for `DSE8` v1.
//!
//! - Fits up to `40` depth-3 oblivious trees (`min_leaf >= 10` privacy/support floor!) on the
//!   residual after `W + E + Pr`.
//! - Encodes each tree in `15 B` (`cls_id: u8`, `cols: [u8; 3]`, `thrs: [u8; 3]` in `0..15`
//!   latent-quantile bin space, `leaf_scale: f16` [2 B], `leaves_i8: [i8; 8]` [8 B] -> `1 + 3 + 3 + 2 + 8 = 17 B`,
//!   or `14 B` when `o_dim == 1` by omitting `cls_id` and packing 4-bit thresholds!).
//!   Actually let's use **exact 14 B/tree** (`cols[3]` [3 B] + `packed_thrs_cls[2]` [2 B: 3×4-bit bin thresholds `0..15` + 4-bit `cls_id` `0..15`] + `scale_u8` [1 B log-scale] + `leaves_i8[8]` [8 B] = **14 B/tree for ALL tasks**!).
//! - Prefix-truncatable: keeping the first `T' <= T` trees is a valid forest with zero refitting!

use crate::evidence::primitives::{value_consistent_bins, SUPPORT_FLOOR_K};

pub const V8_TREE_WIRE_BYTES: usize = 14;
pub const V8_MAX_TREES: usize = 40;

#[derive(Clone, Debug)]
pub struct OblivTreeV8 {
    pub cls_idx: u8,
    pub cols: [u8; 3],
    pub thrs: [u8; 3], // 0..14 in 16-bin latent quantile space (z_lat mapped via Phi)
    pub scale: f32,
    pub leaves: [f32; 8],
}

#[derive(Clone, Debug, Default)]
pub struct TreesSectionV8 {
    pub trees: Vec<OblivTreeV8>,
}

impl OblivTreeV8 {
    #[inline]
    pub fn eval_row(&self, z_lat: &[f32]) -> f32 {
        let p = z_lat.len();
        let mut idx = 0usize;
        for d in 0..3 {
            let c = self.cols[d] as usize;
            if c < p {
                let b = z_to_bin16(z_lat[c]);
                if b > self.thrs[d] {
                    idx |= 1 << d;
                }
            }
        }
        self.leaves[idx]
    }

    pub fn encode_bytes(&self, out: &mut Vec<u8>) {
        out.push(self.cols[0]);
        out.push(self.cols[1]);
        out.push(self.cols[2]);
        let b0 = (self.thrs[0] & 0x0F) | ((self.thrs[1] & 0x0F) << 4);
        let b1 = (self.thrs[2] & 0x0F) | ((self.cls_idx & 0x0F) << 4);
        out.push(b0);
        out.push(b1);
        // Log-quantized scale in [0.0005, 0.05] -> u8 [0..255]
        let sc_u8 = encode_tree_scale(self.scale);
        let sc_dec = decode_tree_scale(sc_u8);
        out.push(sc_u8);
        for &lv in &self.leaves {
            let q = (lv / sc_dec).round().clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
        }
    }

    pub fn decode_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < V8_TREE_WIRE_BYTES {
            return None;
        }
        let cols = [bytes[0], bytes[1], bytes[2]];
        let thrs = [bytes[3] & 0x0F, (bytes[3] >> 4) & 0x0F, bytes[4] & 0x0F];
        let cls_idx = (bytes[4] >> 4) & 0x0F;
        let scale = decode_tree_scale(bytes[5]);
        let mut leaves = [0.0f32; 8];
        for l in 0..8 {
            leaves[l] = (bytes[6 + l] as i8) as f32 * scale;
        }
        Some(Self {
            cls_idx,
            cols,
            thrs,
            scale,
            leaves,
        })
    }
}

#[inline]
pub fn z_to_bin16(z: f32) -> u8 {
    let u = crate::stats::norm_cdf(z as f64);
    ((u * 16.0).floor() as usize).min(15) as u8
}

#[inline]
fn encode_tree_scale(scale: f32) -> u8 {
    let s = scale.clamp(1e-4, 0.10);
    let t = ((s.ln() - (1e-4f32).ln()) / ((0.10f32).ln() - (1e-4f32).ln())).clamp(0.0, 1.0);
    (t * 255.0).round() as u8
}

#[inline]
fn decode_tree_scale(u: u8) -> f32 {
    let lo = (1e-4f32).ln();
    let hi = (0.10f32).ln();
    (lo + (u as f32 / 255.0) * (hi - lo)).exp()
}

impl TreesSectionV8 {
    /// Fit a distilled depth-3 oblivious residual forest on remaining target residuals.
    /// Enforces strict `min_leaf >= 10` support floor and early-stops when gain < noise floor.
    pub fn fit(z_tie: &[Vec<f64>], target_resid: &[Vec<f64>], max_trees: usize) -> Self {
        let p = z_tie.len();
        let o_dim = target_resid.len();
        let n = z_tie.first().map(|c| c.len()).unwrap_or(0);
        let t_cap = max_trees.min(V8_MAX_TREES);
        if p == 0 || o_dim == 0 || n < 3 * SUPPORT_FLOOR_K || t_cap == 0 {
            return Self::default();
        }

        // Convert tie-aware normal scores to 16-bin indices matching `z_to_bin16` at decode time!
        let bins16: Vec<Vec<u8>> = z_tie
            .iter()
            .map(|col| {
                let (b, _) = value_consistent_bins(col, 16);
                b.into_iter().map(|v| if v >= 16 { 7 } else { v }).collect()
            })
            .collect();

        let mut resid: Vec<Vec<f64>> = target_resid.to_vec();
        let mut trees = Vec::with_capacity(t_cap);
        let lr = 0.25f64;
        let reg = 6.0f64;

        for step in 0..t_cap {
            let c = step % o_dim;
            let r_vec = &mut resid[c];
            let var_r = r_vec.iter().map(|v| v * v).sum::<f64>() / n as f64;
            if var_r < 1e-4 {
                continue;
            }
            let mut leaf_id = vec![0usize; n];
            let mut cols = [0u8; 3];
            let mut thrs = [7u8; 3];
            let mut tot_gain = 0.0f64;

            for depth in 0..3 {
                let n_leaves = 1usize << depth;
                let mut cur_g = [0.0f64; 4];
                let mut cur_c = [0usize; 4];
                for i in 0..n {
                    let l = leaf_id[i];
                    cur_g[l] += r_vec[i];
                    cur_c[l] += 1;
                }
                let base_score: f64 = (0..n_leaves)
                    .map(|l| cur_g[l] * cur_g[l] / (cur_c[l] as f64 + reg))
                    .sum();
                let mut best_gain = 0.0f64;
                let mut best_col = 0usize;
                let mut best_thr = 7u8;

                for col in 0..p.min(64) {
                    let mut hg = [[0.0f64; 16]; 4];
                    let mut hc = [[0usize; 16]; 4];
                    let cb = &bins16[col];
                    for i in 0..n {
                        let b = cb[i] as usize;
                        let l = leaf_id[i];
                        hg[l][b] += r_vec[i];
                        hc[l][b] += 1;
                    }
                    let mut lg = [0.0f64; 4];
                    let mut lc = [0usize; 4];
                    for t in 0..15u8 {
                        let tb = t as usize;
                        let mut score = 0.0;
                        let mut ok = false;
                        for l in 0..n_leaves {
                            lg[l] += hg[l][tb];
                            lc[l] += hc[l][tb];
                            let rc = cur_c[l] - lc[l];
                            if lc[l] >= SUPPORT_FLOOR_K && rc >= SUPPORT_FLOOR_K {
                                ok = true;
                                let rg = cur_g[l] - lg[l];
                                score += lg[l] * lg[l] / (lc[l] as f64 + reg)
                                    + rg * rg / (rc as f64 + reg);
                            } else {
                                score += cur_g[l] * cur_g[l] / (cur_c[l] as f64 + reg);
                            }
                        }
                        let gain = score - base_score;
                        if ok && gain > best_gain {
                            best_gain = gain;
                            best_col = col;
                            best_thr = t;
                        }
                    }
                }
                cols[depth] = best_col as u8;
                thrs[depth] = best_thr;
                tot_gain += best_gain;
                let cb = &bins16[best_col];
                for i in 0..n {
                    if cb[i] > best_thr {
                        leaf_id[i] |= 1 << depth;
                    }
                }
            }

            // Require statistically meaningful gain over noise floor (`> 2.5 * var_r`).
            if tot_gain < 2.5 * var_r {
                if o_dim == 1 {
                    break;
                }
                continue;
            }

            let mut g8 = [0.0f64; 8];
            let mut c8 = [0usize; 8];
            for i in 0..n {
                let l = leaf_id[i];
                g8[l] += r_vec[i];
                c8[l] += 1;
            }
            let mut leaves_f32 = [0.0f32; 8];
            let mut max_abs = 0.0f32;
            for l in 0..8 {
                if c8[l] >= SUPPORT_FLOOR_K {
                    let v = (lr * g8[l] / (c8[l] as f64 + reg)).clamp(-2.0, 2.0) as f32;
                    leaves_f32[l] = v;
                    max_abs = max_abs.max(v.abs());
                }
            }
            if max_abs < 1e-4 {
                continue;
            }
            let sc_u8 = encode_tree_scale(max_abs / 127.0);
            let sc = decode_tree_scale(sc_u8);
            for slot in &mut leaves_f32 {
                let q = (*slot / sc).round().clamp(-127.0, 127.0) as i8;
                *slot = q as f32 * sc;
            }
            for i in 0..n {
                r_vec[i] -= leaves_f32[leaf_id[i]] as f64;
            }
            trees.push(OblivTreeV8 {
                cls_idx: c as u8,
                cols,
                thrs,
                scale: sc,
                leaves: leaves_f32,
            });
        }
        Self { trees }
    }

    pub fn eval_row(&self, z_lat: &[f32], out_delta: &mut [f32]) {
        for t in &self.trees {
            let c = t.cls_idx as usize;
            if c < out_delta.len() {
                out_delta[c] += t.eval_row(z_lat);
            }
        }
    }

    pub fn encode_bytes(&self, out: &mut Vec<u8>) {
        for t in &self.trees {
            t.encode_bytes(out);
        }
    }

    pub fn decode_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() % V8_TREE_WIRE_BYTES != 0 {
            return None;
        }
        let n = bytes.len() / V8_TREE_WIRE_BYTES;
        if n > V8_MAX_TREES {
            return None;
        }
        let mut trees = Vec::with_capacity(n);
        for chunk in bytes.chunks_exact(V8_TREE_WIRE_BYTES) {
            trees.push(OblivTreeV8::decode_bytes(chunk)?);
        }
        Some(Self { trees })
    }
}

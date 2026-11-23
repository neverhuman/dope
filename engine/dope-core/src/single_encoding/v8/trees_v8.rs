//! Compact distilled depth-3 oblivious residual forest (`TAG_TREES`, `Tr`) for `DSE8` v1.
//!
//! - Uses exact decode-aligned 16-bin CDF tie-group overlap histograms so discrete/binary/tied
//!   columns split at their true decode `Phi(z_lat)` boundary `(thr + 1) / 16`!
//! - Boosts sequentially on live centered Newton working residuals with inner-val early stopping
//!   and rollback to the best validation prefix (0 spurious trees on clean linear tables!).
//! - Encodes each tree in exact `14 B` (`V8_TREE_WIRE_BYTES`), prefix-truncatable with zero refit.

use super::calibrate::{centered_working_resid, eval_true_loss, CdfIntervals};
use crate::evidence::primitives::SUPPORT_FLOOR_K;

pub const V8_TREE_WIRE_BYTES: usize = 14;
pub const V8_MAX_TREES: usize = 40;

#[derive(Clone, Debug)]
pub struct OblivTreeV8 {
    pub cls_idx: u8,
    pub cols: [u8; 3],
    pub thrs: [u8; 3],
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

    #[inline]
    fn eval_on_cdf_row(&self, cdf: &CdfIntervals, row: usize) -> f64 {
        let mut idx = 0usize;
        for d in 0..3 {
            let c = self.cols[d] as usize;
            if c < cdf.p {
                let u_cut = (self.thrs[d] as f32 + 1.0) / 16.0;
                if cdf.u_mid[c][row] > u_cut {
                    idx |= 1 << d;
                }
            }
        }
        self.leaves[idx] as f64
    }

    pub fn encode_bytes(&self, out: &mut Vec<u8>) {
        out.push(self.cols[0]);
        out.push(self.cols[1]);
        out.push(self.cols[2]);
        let b0 = (self.thrs[0] & 0x0F) | ((self.thrs[1] & 0x0F) << 4);
        let b1 = (self.thrs[2] & 0x0F) | ((self.cls_idx & 0x0F) << 4);
        out.push(b0);
        out.push(b1);
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
        for (l, slot) in leaves.iter_mut().enumerate() {
            *slot = (bytes[6 + l] as i8) as f32 * scale;
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
    /// Fit a decode-aligned depth-3 oblivious residual forest with live working residuals,
    /// strict `min_leaf >= 10` support floor, and inner-val early stopping + rollback!
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        cdf: &CdfIntervals,
        cur_logits: &mut [Vec<f64>],
        y_sc: &[f64],
        tr_rows: &[usize],
        va_rows: &[usize],
        task_id: u8,
        temp: f32,
        max_trees: usize,
    ) -> Self {
        let p = cdf.p.min(64);
        let o_dim = cur_logits.len();
        let n_tr = tr_rows.len();
        let t_cap = max_trees.min(V8_MAX_TREES);
        if p == 0 || o_dim == 0 || n_tr < 3 * SUPPORT_FLOOR_K || va_rows.len() < 10 || t_cap == 0 {
            return Self::default();
        }

        // Precompute exact 16-bin overlap weights on tr_rows once for O(1) histogram accumulation!
        let mut tr_w16: Vec<Vec<(u8, u8, [f32; 16])>> = Vec::with_capacity(p);
        for col in 0..p {
            let mut col_w = Vec::with_capacity(n_tr);
            for &r in tr_rows {
                let (b0, b1, wf) =
                    CdfIntervals::bin_weights::<16>(cdf.u_lo[col][r], cdf.u_hi[col][r]);
                let mut w32 = [0.0f32; 16];
                for b in b0..=b1 {
                    w32[b] = wf[b] as f32;
                }
                col_w.push((b0 as u8, b1 as u8, w32));
            }
            tr_w16.push(col_w);
        }

        let base_va = eval_true_loss(cur_logits, y_sc, va_rows, task_id, temp);
        let mut best_va = base_va;
        let mut best_len = 0usize;
        let mut best_logits = cur_logits.to_vec();
        let min_improv = (0.001 * base_va.abs()).max(1e-5);
        let patience = (3 * o_dim).max(4);
        let mut stale = 0usize;

        let mut trees = Vec::with_capacity(t_cap);
        let lr = 0.22f64;
        let reg = 8.0f64;

        for step in 0..t_cap {
            let c = step % o_dim;
            let resid_all = centered_working_resid(cur_logits, y_sc, tr_rows, task_id, temp);
            let r_tr: Vec<f64> = tr_rows.iter().map(|&r| resid_all[c][r]).collect();
            let var_r = r_tr.iter().map(|v| v * v).sum::<f64>() / n_tr as f64;
            if var_r < 1e-5 {
                stale += 1;
                if stale >= patience {
                    break;
                }
                continue;
            }

            let mut leaf_id = vec![0usize; n_tr];
            let mut cols = [0u8; 3];
            let mut thrs = [7u8; 3];
            let mut tot_gain = 0.0f64;

            for depth in 0..3 {
                let n_leaves = 1usize << depth;
                let mut cur_g = [0.0f64; 4];
                let mut cur_c = [0.0f64; 4];
                for i in 0..n_tr {
                    let l = leaf_id[i];
                    cur_g[l] += r_tr[i];
                    cur_c[l] += 1.0;
                }
                let base_score: f64 = (0..n_leaves)
                    .map(|l| cur_g[l] * cur_g[l] / (cur_c[l] + reg))
                    .sum();
                let mut best_gain = 0.0f64;
                let mut best_col = 0usize;
                let mut best_thr = 7u8;

                for (col, col_w) in tr_w16.iter().enumerate().take(p) {
                    let mut hg = [[0.0f64; 16]; 4];
                    let mut hc = [[0.0f64; 16]; 4];
                    for (i, &(b0, b1, ref w)) in col_w.iter().enumerate() {
                        let l = leaf_id[i];
                        let rv = r_tr[i];
                        for b in (b0 as usize)..=(b1 as usize) {
                            let wb = w[b] as f64;
                            hg[l][b] += wb * rv;
                            hc[l][b] += wb;
                        }
                    }
                    let mut lg = [0.0f64; 4];
                    let mut lc = [0.0f64; 4];
                    for t in 0..15u8 {
                        let tb = t as usize;
                        let mut score = 0.0;
                        let mut ok = false;
                        for l in 0..n_leaves {
                            lg[l] += hg[l][tb];
                            lc[l] += hc[l][tb];
                            let rc = (cur_c[l] - lc[l]).max(0.0);
                            if lc[l] >= SUPPORT_FLOOR_K as f64 && rc >= SUPPORT_FLOOR_K as f64 {
                                ok = true;
                                let rg = cur_g[l] - lg[l];
                                score += lg[l] * lg[l] / (lc[l] + reg) + rg * rg / (rc + reg);
                            } else {
                                score += cur_g[l] * cur_g[l] / (cur_c[l] + reg);
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
                let u_cut = (best_thr as f32 + 1.0) / 16.0;
                for (i, &r) in tr_rows.iter().enumerate() {
                    if cdf.u_mid[best_col][r] > u_cut {
                        leaf_id[i] |= 1 << depth;
                    }
                }
            }

            if tot_gain < 2.5 * var_r {
                stale += 1;
                if stale >= patience {
                    break;
                }
                continue;
            }

            let mut g8 = [0.0f64; 8];
            let mut c8 = [0usize; 8];
            for i in 0..n_tr {
                let l = leaf_id[i];
                g8[l] += r_tr[i];
                c8[l] += 1;
            }
            let mut leaves_f64 = [0.0f64; 8];
            let mut w_sum = 0.0;
            let mut w_tot = 0.0;
            for l in 0..8 {
                if c8[l] >= SUPPORT_FLOOR_K {
                    let v = (lr * g8[l] / (c8[l] as f64 + reg)).clamp(-1.8, 1.8);
                    leaves_f64[l] = v;
                    w_sum += c8[l] as f64 * v;
                    w_tot += c8[l] as f64;
                }
            }
            if w_tot <= 0.0 {
                stale += 1;
                if stale >= patience {
                    break;
                }
                continue;
            }
            let mean_lv = w_sum / w_tot;
            let mut leaves_f32 = [0.0f32; 8];
            let mut max_abs = 0.0f32;
            for l in 0..8 {
                if c8[l] >= SUPPORT_FLOOR_K {
                    let v = (leaves_f64[l] - mean_lv).clamp(-1.8, 1.8) as f32;
                    leaves_f32[l] = v;
                    max_abs = max_abs.max(v.abs());
                }
            }
            if max_abs < 1e-4 {
                stale += 1;
                if stale >= patience {
                    break;
                }
                continue;
            }
            let sc_u8 = encode_tree_scale(max_abs / 127.0);
            let sc = decode_tree_scale(sc_u8);
            for slot in &mut leaves_f32 {
                let q = (*slot / sc).round().clamp(-127.0, 127.0) as i8;
                *slot = q as f32 * sc;
            }
            let tree = OblivTreeV8 {
                cls_idx: c as u8,
                cols,
                thrs,
                scale: sc,
                leaves: leaves_f32,
            };
            for &r in tr_rows {
                cur_logits[c][r] += tree.eval_on_cdf_row(cdf, r);
            }
            for &r in va_rows {
                cur_logits[c][r] += tree.eval_on_cdf_row(cdf, r);
            }
            trees.push(tree);

            let va_loss = eval_true_loss(cur_logits, y_sc, va_rows, task_id, temp);
            if va_loss + min_improv < best_va {
                best_va = va_loss;
                best_len = trees.len();
                best_logits = cur_logits.to_vec();
                stale = 0;
            } else {
                stale += 1;
                if stale >= patience {
                    break;
                }
            }
        }

        // Roll back to the exact best inner-validation prefix!
        trees.truncate(best_len);
        for (dst, src) in cur_logits.iter_mut().zip(best_logits) {
            *dst = src;
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

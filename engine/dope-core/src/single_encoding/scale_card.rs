//! Compact embedded raw-unit scale card (`ScaleCardV7`) and V7.2 quartile residual extractor.
//!
//! Embedded directly inside self-contained `DSE7` v2 wire packets so `len(m.bin) <= budget_bytes`
//! covers 100% of everything required to synthesize raw-unit rows (`source_free`, zero sidecar).

use super::math::percentile_sorted;
use super::{
    DecodedV7Row, RefinedV7Packet, SingleEncV7Packet, SingleEncV7Weights, COL_FEAT_DIM,
    MAX_CLASSES, MAX_COLS, MAX_RANK,
};
use crate::error::{DopeError, Result};
use crate::rng::Rng;
use std::collections::BTreeMap;

#[inline]
pub(crate) fn f32_to_bf16_bits(v: f32) -> u16 {
    if !v.is_finite() {
        return 0;
    }
    let bits = v.to_bits();
    let round = 0x7fffu32 + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

#[inline]
pub(crate) fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[inline]
pub(crate) fn quant_bf16_roundtrip(v: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(v))
}

/// Self-contained raw-unit scale card embedded inside `DSE7` v2 wire packets.
#[derive(Clone, Debug)]
pub struct ScaleCardV7 {
    pub x_cols_raw: usize,
    pub col_indices: Vec<usize>,
    pub center_active: Vec<f32>,
    pub scale_active: Vec<f32>,
    pub is_int_active: Vec<bool>,
    pub dropped_medians: Vec<f32>,
    pub task_id: u8,
    pub y_mean: f32,
    pub y_sd: f32,
    pub classes: Vec<i32>,
}

impl ScaleCardV7 {
    /// Fit robust median + bf16-quantized IQR/std scale card strictly on `A` rows.
    ///
    /// When `x_raw.len() > p_eff`, selects the top-`p_eff` columns by target correlation
    /// `|corr(x_j, y)|` (breaking ties by variance, strictly on `A` rows).
    pub fn fit(x_raw: &[Vec<f64>], y_raw: &[f64], task_id: u8, p_cap: usize) -> Result<Self> {
        let p_raw = x_raw.len();
        let n = y_raw.len();
        if p_raw == 0 || n < 4 || p_cap == 0 {
            return Err(DopeError::Input(
                "dse7 requires >= 1 feature and >= 4 rows".into(),
            ));
        }
        let mut center_all = vec![0.0f32; p_raw];
        let mut scale_all = vec![0.0f32; p_raw];
        for j in 0..p_raw {
            let mut fin: Vec<f64> = x_raw[j].iter().copied().filter(|v| v.is_finite()).collect();
            if fin.len() < 2 {
                center_all[j] = fin.first().copied().unwrap_or(0.0) as f32;
                scale_all[j] = 0.0;
                continue;
            }
            fin.sort_by(f64::total_cmp);
            let med = percentile_sorted(&fin, 0.5);
            let q25 = percentile_sorted(&fin, 0.25);
            let q75 = percentile_sorted(&fin, 0.75);
            let mean = fin.iter().sum::<f64>() / fin.len() as f64;
            let var = fin.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / fin.len() as f64;
            center_all[j] = med as f32;
            let s_raw = ((q75 - q25) / 1.349).max(var.sqrt()).max(1e-6) as f32;
            scale_all[j] = quant_bf16_roundtrip(s_raw).max(1e-6);
        }
        let p_eff = p_raw.min(p_cap).min(MAX_COLS);
        let col_indices: Vec<usize> = if p_eff == p_raw {
            (0..p_raw).collect()
        } else {
            let y_mean = y_raw.iter().sum::<f64>() / n as f64;
            let y_var = y_raw
                .iter()
                .map(|&v| (v - y_mean).powi(2))
                .sum::<f64>()
                .max(1e-12);
            let mut scored: Vec<(f64, usize)> = (0..p_raw)
                .map(|j| {
                    if scale_all[j] <= 0.0 {
                        return (-1.0, j);
                    }
                    let c = center_all[j] as f64;
                    let mut xm = 0.0;
                    for &v in &x_raw[j] {
                        xm += if v.is_finite() { v } else { c };
                    }
                    xm /= n as f64;
                    let (mut cov, mut xv) = (0.0, 0.0);
                    for r in 0..n {
                        let d = (if x_raw[j][r].is_finite() {
                            x_raw[j][r]
                        } else {
                            c
                        }) - xm;
                        cov += d * (y_raw[r] - y_mean);
                        xv += d * d;
                    }
                    let corr = if xv > 1e-12 {
                        (cov / (xv * y_var).sqrt()).abs()
                    } else {
                        0.0
                    };
                    (corr, j)
                })
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let mut kept: Vec<usize> = scored.into_iter().take(p_eff).map(|(_, j)| j).collect();
            kept.sort_unstable();
            kept
        };
        let mut is_active = vec![false; p_raw];
        let mut center_active = Vec::with_capacity(col_indices.len());
        let mut scale_active = Vec::with_capacity(col_indices.len());
        let mut is_int_active = Vec::with_capacity(col_indices.len());
        for &j in &col_indices {
            is_active[j] = true;
            center_active.push(center_all[j]);
            scale_active.push(scale_all[j]);
            let all_int = x_raw[j]
                .iter()
                .filter(|v| v.is_finite())
                .all(|&v| (v - v.round()).abs() < 1e-6);
            is_int_active.push(all_int);
        }
        let mut dropped_medians = Vec::with_capacity(p_raw - col_indices.len());
        for (j, &act) in is_active.iter().enumerate() {
            if !act {
                dropped_medians.push(quant_bf16_roundtrip(center_all[j]));
            }
        }
        let (y_mean, y_sd, classes) = if task_id == 0 {
            let mu = y_raw.iter().sum::<f64>() / n as f64;
            let var = y_raw.iter().map(|&v| (v - mu) * (v - mu)).sum::<f64>() / n as f64;
            (mu as f32, var.sqrt().max(1e-5) as f32, Vec::new())
        } else {
            let mut counts: BTreeMap<i32, usize> = BTreeMap::new();
            for &v in y_raw {
                let id = if task_id == 1 {
                    i32::from(v > 0.5)
                } else {
                    v.round() as i32
                };
                *counts.entry(id).or_insert(0) += 1;
            }
            if counts.len() < 2 {
                return Err(DopeError::Input(
                    "dse7 requires >= 2 classes for classification".into(),
                ));
            }
            let mut by_cnt: Vec<(usize, i32)> = counts.into_iter().map(|(id, c)| (c, id)).collect();
            by_cnt.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            let mut cls: Vec<i32> = by_cnt
                .into_iter()
                .take(MAX_CLASSES)
                .map(|(_, id)| id)
                .collect();
            cls.sort_unstable();
            (0.0, 1.0, cls)
        };
        Ok(Self {
            x_cols_raw: p_raw,
            col_indices,
            center_active,
            scale_active,
            is_int_active,
            dropped_medians,
            task_id,
            y_mean,
            y_sd,
            classes,
        })
    }

    /// Forward-scale active columns to `[-6, 6]` with `asinh` tail compression.
    pub fn forward_x(&self, x_raw: &[Vec<f64>]) -> Vec<Vec<f32>> {
        let n = x_raw[0].len();
        self.col_indices
            .iter()
            .enumerate()
            .map(|(slot, &j)| {
                let c = self.center_active[slot] as f64;
                let s = self.scale_active[slot] as f64;
                if s <= 0.0 {
                    return vec![0.0f32; n];
                }
                x_raw[j]
                    .iter()
                    .map(|&v| {
                        let z = ((if v.is_finite() { v } else { c }) - c) / s;
                        let az = z.abs();
                        let out = if az > 3.0 {
                            z.signum() * (3.0 + (az - 3.0).asinh())
                        } else {
                            z
                        };
                        out.clamp(-6.0, 6.0) as f32
                    })
                    .collect()
            })
            .collect()
    }

    /// Inverse-scale one model row `[MAX_COLS]` back to full raw column width `x_cols_raw`.
    pub fn inverse_row_x(&self, x_model: &[f32; MAX_COLS]) -> Vec<f32> {
        let mut raw = vec![0.0f32; self.x_cols_raw];
        if !self.dropped_medians.is_empty() {
            let mut act_pos = 0usize;
            let mut drp_pos = 0usize;
            for (j, slot) in raw.iter_mut().enumerate().take(self.x_cols_raw) {
                if act_pos < self.col_indices.len() && self.col_indices[act_pos] == j {
                    act_pos += 1;
                } else if drp_pos < self.dropped_medians.len() {
                    *slot = self.dropped_medians[drp_pos];
                    drp_pos += 1;
                }
            }
        }
        for (slot, &j) in self.col_indices.iter().enumerate() {
            let s = (x_model[slot] as f64).clamp(-6.0, 6.0);
            let a = s.abs();
            let z = if a > 3.0 {
                s.signum() * (3.0 + (a - 3.0).sinh())
            } else {
                s
            };
            let mut val = self.center_active[slot] as f64 + (self.scale_active[slot] as f64) * z;
            if self.is_int_active.get(slot).copied().unwrap_or(false) {
                val = val.round();
            }
            raw[j] = val as f32;
        }
        raw
    }
}

/// Empirical-Bayes-shrunken 4-quartile nonlinear target residuals (`[-127, 127]`, scale `0.015`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_q_nl_i8(
    x_sc: &[Vec<f32>],
    y_sc: &[f64],
    z_mat: &[Vec<f64>],
    w_scales: &[f32],
    b_vals: &[f32],
    w_i8: &[Vec<i8>],
    resid_sd: f32,
    task_id: u8,
    k_eff: usize,
) -> Vec<[i8; 4]> {
    let p = x_sc.len();
    let n = y_sc.len();
    let mut out = vec![[0i8; 4]; p];
    if n < 12 || p == 0 {
        return out;
    }
    let mut resid = vec![0.0f64; n];
    if task_id == 0 {
        let denom = (resid_sd as f64).max(0.10);
        for r in 0..n {
            let mut pred = b_vals[0] as f64;
            for j in 0..p {
                pred += (x_sc[j][r] as f64) * ((w_i8[0][j] as f64) * (w_scales[0] as f64));
            }
            resid[r] = ((y_sc[r] - pred) / denom).clamp(-4.0, 4.0);
        }
    } else if task_id == 1 {
        for r in 0..n {
            let mut z = b_vals[0] as f64;
            for j in 0..p {
                z += (x_sc[j][r] as f64) * ((w_i8[0][j] as f64) * (w_scales[0] as f64));
            }
            let pr = 1.0 / (1.0 + (-z.clamp(-8.0, 8.0)).exp());
            resid[r] = (y_sc[r] - pr) * 2.0;
        }
    } else {
        let mut raw_s = vec![0.0f64; n];
        let mut mean_s = 0.0f64;
        for r in 0..n {
            let mut lg = [0.0f64; 8];
            let mut mx = f64::NEG_INFINITY;
            for c in 0..k_eff {
                let mut z = b_vals[c] as f64;
                for j in 0..p {
                    z += (x_sc[j][r] as f64) * ((w_i8[c][j] as f64) * (w_scales[c] as f64));
                }
                lg[c] = z;
                mx = mx.max(z);
            }
            let mut sum_e = 0.0;
            for c in 0..k_eff {
                sum_e += (lg[c] - mx).clamp(-20.0, 0.0).exp();
            }
            let yc = (y_sc[r].round() as usize).min(k_eff - 1);
            let p_true = (lg[yc] - mx).clamp(-20.0, 0.0).exp() / sum_e.max(1e-12);
            raw_s[r] = 1.0 - p_true;
            mean_s += raw_s[r];
        }
        mean_s /= n as f64;
        for r in 0..n {
            resid[r] = (raw_s[r] - mean_s) * 2.5;
        }
    }
    for j in 0..p {
        let mut sums = [0.0f64; 4];
        let mut cnts = [0.0f64; 4];
        for r in 0..n {
            let zv = z_mat[j][r];
            let b = if zv < -0.6745 {
                0
            } else if zv < 0.0 {
                1
            } else if zv < 0.6745 {
                2
            } else {
                3
            };
            sums[b] += resid[r];
            cnts[b] += 1.0;
        }
        for b in 0..4 {
            let eb = sums[b] / (cnts[b] + 8.0);
            out[j][b] = (eb / 0.015).round().clamp(-127.0, 127.0) as i8;
        }
    }
    out
}

pub(crate) const DSE7_HEADER_BYTES: usize = 20;

#[derive(Clone, Copy)]
pub(crate) struct Dse7Layout {
    pub p_eff: usize,
    pub knot_level: usize,
    pub rank_use: usize,
    pub has_q_nl: bool,
    pub has_col_map: bool,
    pub has_custom_classes: bool,
    pub has_dropped_meds: bool,
}

pub(crate) fn dse7_wire_len(x_cols_raw: usize, task_id: u8, k_eff: usize, l: Dse7Layout) -> usize {
    let o_dim = if task_id == 2 { k_eff } else { 1 };
    let target_scale_b = if task_id == 0 {
        8
    } else if l.has_custom_classes {
        4 * k_eff
    } else {
        0
    };
    let col_map_b = if l.has_col_map { 2 * l.p_eff } else { 0 };
    let dropped_b = if l.has_dropped_meds {
        2 * x_cols_raw.saturating_sub(l.p_eff)
    } else {
        0
    };
    let q_nl_b = if l.has_q_nl { 4 * l.p_eff } else { 0 };
    DSE7_HEADER_BYTES
        + target_scale_b
        + col_map_b
        + 6 * l.p_eff
        + dropped_b
        + l.p_eff * (1 + 4 + l.knot_level + l.rank_use)
        + o_dim * (8 + l.p_eff)
        + q_nl_b
        + 4
}

pub(crate) fn choose_dse7_layout(
    x_cols_raw: usize,
    task_id: u8,
    k_eff: usize,
    has_custom_classes: bool,
    budget: usize,
) -> Option<Dse7Layout> {
    let p_max = x_cols_raw.min(MAX_COLS);
    let mut p_cands = vec![p_max];
    for &p in &[32usize, 24, 16, 12, 8, 4, 2, 1] {
        if p < p_max {
            p_cands.push(p);
        }
    }
    for &p_eff in &p_cands {
        let has_col_map = p_eff < x_cols_raw;
        let dropped = x_cols_raw.saturating_sub(p_eff);
        let drop_opts: &[bool] = if has_col_map && 2 * dropped <= budget / 3 {
            &[true, false]
        } else {
            &[false]
        };
        for &has_dropped_meds in drop_opts {
            for &knot_level in &[17usize, 9, 5] {
                for &has_q_nl in &[true, false] {
                    for &rank_use in &[12usize, 8, 6, 4, 2, 1, 0] {
                        let lay = Dse7Layout {
                            p_eff,
                            knot_level,
                            rank_use,
                            has_q_nl,
                            has_col_map,
                            has_custom_classes,
                            has_dropped_meds,
                        };
                        if dse7_wire_len(x_cols_raw, task_id, k_eff, lay) <= budget {
                            return Some(lay);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Analytic zero-init DSE7 decoder (exact zero-training baseline with zero neural weights).
pub fn sample_zero_init_rows(
    pkt: &SingleEncV7Packet,
    n_rows: usize,
    seed: u64,
) -> Vec<DecodedV7Row> {
    let mut noise_sd = [1.0f32; MAX_COLS];
    let mut is_disc = [false; MAX_COLS];
    for c in 0..MAX_COLS {
        is_disc[c] = pkt.col_feats[c][COL_FEAT_DIM - 4] > 0.5;
        let sq: f32 = pkt.loadings[c].iter().map(|&v| v * v).sum();
        noise_sd[c] = (1.0 - sq).max(0.02).sqrt();
    }
    let refined = RefinedV7Packet {
        l_eff: pkt.loadings,
        noise_sd,
        knots_eff: pkt.knots,
        knots_base: pkt.knots,
        is_disc,
        is_const: pkt.is_const,
        col_mask: pkt.col_mask,
        z_max: pkt.col_feats[0][COL_FEAT_DIM - 2].clamp(1.2, 2.6),
        w_eff: pkt.w_base,
        b_base: pkt.b_base,
        h_target: 0,
        hyper_w1: Vec::new(),
        hyper_b1: Vec::new(),
        hyper_w2: Vec::new(),
        temp_scale: 1.0,
        resid_sd_eff: pkt.resid_sd,
    };
    let mut out = Vec::with_capacity(n_rows);
    for row_idx in 0..n_rows {
        let mut rng = Rng::new(seed ^ (row_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut zf = [0.0f32; MAX_RANK];
        for v in &mut zf {
            *v = rng.normal() as f32;
        }
        let mut zn = [0.0f32; MAX_COLS];
        for v in &mut zn {
            *v = rng.normal() as f32;
        }
        out.push(SingleEncV7Weights::decode_row(
            &refined,
            &zf,
            &zn,
            rng.normal() as f32,
        ));
    }
    out
}

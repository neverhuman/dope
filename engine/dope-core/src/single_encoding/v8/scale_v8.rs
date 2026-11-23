//! Scale card, `Dse8ColumnMeta`, and nonlinear composite column screening for `DSE8` v1.
//!
//! Supports BOTH:
//! - `self_contained_raw` (`normalized_only = false`): embeds compact per-column scale card,
//!   automatically skipping 6-byte center/scale entries on boolean (`group_kind == 3`) and
//!   nominal one-hot (`group_kind == 4`) columns!
//! - `normalized_only` (`normalized_only = true`): zero scale bytes on the wire (for NC1/NC2
//!   external-card workflows where normalization lives with the owner).
//!
//! Strictly enforces `k >= 10` class support floor and never silently maps unseen/overflow classes to `0`.

use crate::error::{DopeError, Result};
use crate::evidence::primitives::{value_consistent_bins, SUPPORT_FLOOR_K};
use crate::evidence::ColumnMeta as EvidenceColMeta;
use crate::single_encoding::scale_card::{bf16_bits_to_f32, f32_to_bf16_bits};
use crate::stats::{chatterjee_xi, pearson, quantile_sorted, spearman};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const V8_MAX_COLS: usize = 64;
pub const V8_MAX_CLASSES: usize = 8;

/// Privacy-safe column structural metadata (exact 1:1 mirror of `evidence::ColumnMeta` and `Nc2ColumnMetaV2`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dse8ColumnMeta {
    pub group_id: u16,
    /// 0=numeric_cont, 1=integer, 2=ordinal, 3=boolean, 4=nominal_onehot, 5=missing_ind
    pub group_kind: u8,
    pub group_width: u16,
    pub parent_col_idx: u16,
    /// 0=cont, 1=lattice, 2=atoms, 3=zero_inflated, 4=const
    pub support_class: u8,
    pub skew_guard_active: u8,
    pub lattice_q0_unit: f32,
    pub lattice_origin_unit: f32,
    pub has_lower_bound: u8,
    pub has_upper_bound: u8,
    pub log_pre_tail_asym: f32,
    pub log_unique_bucket: f32,
}

impl Default for Dse8ColumnMeta {
    fn default() -> Self {
        Self {
            group_id: 0,
            group_kind: 0,
            group_width: 1,
            parent_col_idx: u16::MAX,
            support_class: 0,
            skew_guard_active: 0,
            lattice_q0_unit: 0.0,
            lattice_origin_unit: 0.0,
            has_lower_bound: 0,
            has_upper_bound: 0,
            log_pre_tail_asym: 0.0,
            log_unique_bucket: 0.0,
        }
    }
}

impl From<&Dse8ColumnMeta> for EvidenceColMeta {
    fn from(m: &Dse8ColumnMeta) -> Self {
        Self {
            group_id: m.group_id,
            group_kind: m.group_kind,
            group_width: m.group_width,
            parent_col_idx: m.parent_col_idx,
            support_class: m.support_class,
            skew_guard_active: m.skew_guard_active,
            lattice_q0_unit: m.lattice_q0_unit,
            lattice_origin_unit: m.lattice_origin_unit,
            has_lower_bound: m.has_lower_bound,
            has_upper_bound: m.has_upper_bound,
            log_pre_tail_asym: m.log_pre_tail_asym,
            log_unique_bucket: m.log_unique_bucket,
        }
    }
}

impl From<&EvidenceColMeta> for Dse8ColumnMeta {
    fn from(m: &EvidenceColMeta) -> Self {
        Self {
            group_id: m.group_id,
            group_kind: m.group_kind,
            group_width: m.group_width,
            parent_col_idx: m.parent_col_idx,
            support_class: m.support_class,
            skew_guard_active: m.skew_guard_active,
            lattice_q0_unit: m.lattice_q0_unit,
            lattice_origin_unit: m.lattice_origin_unit,
            has_lower_bound: m.has_lower_bound,
            has_upper_bound: m.has_upper_bound,
            log_pre_tail_asym: m.log_pre_tail_asym,
            log_unique_bucket: m.log_unique_bucket,
        }
    }
}

/// Self-contained scale card for `DSE8` v1 (omitted on wire when `normalized_only = true`).
#[derive(Clone, Debug)]
pub struct ScaleCardV8 {
    pub normalized_only: bool,
    pub x_cols_raw: usize,
    pub col_indices: Vec<usize>,
    pub group_kinds: Vec<u8>,
    pub center_active: Vec<f32>,
    pub scale_active: Vec<f32>,
    pub is_int_active: Vec<bool>,
    pub dropped_medians: Vec<f32>,
    pub task_id: u8,
    pub y_mean: f32,
    pub y_sd: f32,
    pub classes: Vec<i32>,
}

impl ScaleCardV8 {
    /// Fit a `DSE8` scale card with nonlinear composite column screening (never Pearson-on-class-ID).
    pub fn fit(
        x_raw: &[Vec<f64>],
        y_raw: &[f64],
        task_id: u8,
        p_cap: usize,
        normalized_only: bool,
        col_meta: Option<&[Dse8ColumnMeta]>,
    ) -> Result<Self> {
        let x_cols_raw = x_raw.len();
        if x_cols_raw == 0 || y_raw.is_empty() {
            return Err(DopeError::Input("dse8_empty_table".into()));
        }
        let n = y_raw.len();
        if x_raw.iter().any(|c| c.len() != n) {
            return Err(DopeError::Input("dse8_row_len_mismatch".into()));
        }

        // Target class table with k >= 10 support floor (fallback to observed distinct if tiny n < 20).
        let (y_mean, y_sd, classes) = if task_id == 0 {
            let mut fin: Vec<f64> = y_raw.iter().copied().filter(|v| v.is_finite()).collect();
            if fin.is_empty() {
                return Err(DopeError::Input("dse8_no_finite_target".into()));
            }
            fin.sort_by(f64::total_cmp);
            let m = fin.iter().sum::<f64>() / fin.len() as f64;
            let var = fin.iter().map(|v| (v - m).powi(2)).sum::<f64>() / fin.len() as f64;
            let sd = var.sqrt().max(1e-5);
            (m as f32, sd as f32, Vec::new())
        } else {
            let mut counts: BTreeMap<i32, usize> = BTreeMap::new();
            for &v in y_raw {
                if v.is_finite() {
                    *counts.entry(v.round() as i32).or_default() += 1;
                }
            }
            let min_c = if n >= 2 * SUPPORT_FLOOR_K {
                SUPPORT_FLOOR_K
            } else {
                1
            };
            let mut kept: Vec<(i32, usize)> =
                counts.into_iter().filter(|&(_, c)| c >= min_c).collect();
            if kept.len() < 2 {
                // Tiny table fallback: keep top-2 classes.
                let mut all: BTreeMap<i32, usize> = BTreeMap::new();
                for &v in y_raw {
                    if v.is_finite() {
                        *all.entry(v.round() as i32).or_default() += 1;
                    }
                }
                kept = all.into_iter().collect();
            }
            // Keep up to V8_MAX_CLASSES most frequent classes, then sort ascending by class ID.
            kept.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            kept.truncate(if task_id == 1 { 2 } else { V8_MAX_CLASSES });
            kept.sort_by_key(|&(cid, _)| cid);
            let cls: Vec<i32> = kept.into_iter().map(|(cid, _)| cid).collect();
            if cls.len() < 2 {
                return Err(DopeError::Input("dse8_single_class_target".into()));
            }
            (0.0, 1.0, cls)
        };

        // Composite nonlinear column screening when x_cols_raw > p_cap.
        let p_use = x_cols_raw.min(p_cap.clamp(1, V8_MAX_COLS));
        let col_indices = if x_cols_raw <= p_use {
            (0..x_cols_raw).collect()
        } else {
            screen_columns_composite(x_raw, y_raw, task_id, &classes, p_use, col_meta)
        };

        let mut active_flag = vec![false; x_cols_raw];
        for &idx in &col_indices {
            active_flag[idx] = true;
        }

        let mut group_kinds = Vec::with_capacity(col_indices.len());
        let mut center_active = Vec::with_capacity(col_indices.len());
        let mut scale_active = Vec::with_capacity(col_indices.len());
        let mut is_int_active = Vec::with_capacity(col_indices.len());

        for &idx in &col_indices {
            let mut fin: Vec<f64> = x_raw[idx]
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();
            fin.sort_by(f64::total_cmp);
            let all_int = !fin.is_empty() && fin.iter().all(|&v| (v - v.round()).abs() <= 1e-6);
            let is_binary_01 = !fin.is_empty()
                && fin
                    .iter()
                    .all(|&v| (v - 0.0).abs() <= 1e-6 || (v - 1.0).abs() <= 1e-6);
            let gk = col_meta
                .and_then(|ms| ms.get(idx))
                .map(|m| m.group_kind)
                .unwrap_or_else(|| {
                    if is_binary_01 {
                        3
                    } else if all_int {
                        1
                    } else {
                        0
                    }
                });
            group_kinds.push(gk);
            is_int_active.push(all_int || gk == 1 || gk == 3 || gk == 4);

            if normalized_only || gk == 3 || gk == 4 || gk == 5 || fin.is_empty() {
                center_active.push(0.0);
                scale_active.push(1.0);
            } else {
                let med = quantile_sorted(&fin, 0.50);
                let q25 = quantile_sorted(&fin, 0.25);
                let q75 = quantile_sorted(&fin, 0.75);
                let mean = fin.iter().sum::<f64>() / fin.len() as f64;
                let sd =
                    (fin.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / fin.len() as f64).sqrt();
                let sc = ((q75 - q25) / 1.349).max(sd).max(1e-5);
                center_active.push(med as f32);
                scale_active.push(bf16_bits_to_f32(f32_to_bf16_bits(sc as f32)).max(1e-5));
            }
        }

        let mut dropped_medians = Vec::new();
        if !normalized_only {
            for (c, col) in x_raw.iter().enumerate() {
                if !active_flag[c] {
                    let mut fin: Vec<f64> = col.iter().copied().filter(|v| v.is_finite()).collect();
                    fin.sort_by(f64::total_cmp);
                    let med = if fin.is_empty() {
                        0.0
                    } else {
                        quantile_sorted(&fin, 0.50) as f32
                    };
                    dropped_medians.push(bf16_bits_to_f32(f32_to_bf16_bits(med)));
                }
            }
        }

        Ok(Self {
            normalized_only,
            x_cols_raw,
            col_indices,
            group_kinds,
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

    /// Transform raw feature columns to active unit-coordinate matrix (preserving NaN!).
    pub fn forward_x(&self, x_raw: &[Vec<f64>]) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(self.col_indices.len());
        for (a, &raw_idx) in self.col_indices.iter().enumerate() {
            let ctr = self.center_active[a] as f64;
            let sc = (self.scale_active[a] as f64).max(1e-5);
            let col: Vec<f32> = x_raw[raw_idx]
                .iter()
                .map(|&v| {
                    if !v.is_finite() {
                        f32::NAN
                    } else if self.normalized_only || matches!(self.group_kinds[a], 3..=5) {
                        v.clamp(-6.0, 6.0) as f32
                    } else {
                        ((v - ctr) / sc).clamp(-6.0, 6.0) as f32
                    }
                })
                .collect();
            out.push(col);
        }
        out
    }

    /// Inverse-transform one decoded active row back to full `x_cols_raw` raw units.
    pub fn inverse_row_x(&self, x_active: &[f32], miss_active: &[bool]) -> Vec<f64> {
        let mut out = vec![0.0f64; self.x_cols_raw];
        let mut is_act = vec![false; self.x_cols_raw];
        for (a, &raw_idx) in self.col_indices.iter().enumerate() {
            is_act[raw_idx] = true;
            if miss_active.get(a).copied().unwrap_or(false) || !x_active[a].is_finite() {
                out[raw_idx] = f64::NAN;
                continue;
            }
            let raw_v = if self.normalized_only || matches!(self.group_kinds[a], 3..=5) {
                x_active[a] as f64
            } else {
                self.center_active[a] as f64 + (self.scale_active[a] as f64) * (x_active[a] as f64)
            };
            out[raw_idx] = if !self.normalized_only && self.is_int_active[a] {
                raw_v.round()
            } else {
                raw_v
            };
        }
        let mut d_cur = 0usize;
        for (c, slot) in out.iter_mut().enumerate() {
            if !is_act[c] {
                *slot = self.dropped_medians.get(d_cur).copied().unwrap_or(0.0) as f64;
                d_cur += 1;
            }
        }
        out
    }

    /// Map raw target values to internal normalized/class-index targets, returning `(y_sc, valid_rows)`.
    /// Rows with rare/unseen classes excluded by `k >= 10` are cleanly filtered out (never mapped to 0!).
    pub fn forward_y(&self, y_raw: &[f64]) -> (Vec<f64>, Vec<usize>) {
        let mut y_sc = Vec::with_capacity(y_raw.len());
        let mut valid = Vec::with_capacity(y_raw.len());
        for (r, &v) in y_raw.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            if self.task_id == 0 {
                let ys = if self.normalized_only {
                    v.clamp(-6.0, 6.0)
                } else {
                    ((v - self.y_mean as f64) / (self.y_sd as f64).max(1e-5)).clamp(-6.0, 6.0)
                };
                y_sc.push(ys);
                valid.push(r);
            } else {
                let iv = v.round() as i32;
                if let Ok(pos) = self.classes.binary_search(&iv) {
                    y_sc.push(pos as f64);
                    valid.push(r);
                }
            }
        }
        (y_sc, valid)
    }

    /// Approximate wire bytes of the `TAG_SCALE` section.
    pub fn wire_len(&self) -> usize {
        if self.normalized_only {
            return 0;
        }
        let mut len = 0usize;
        if self.task_id == 0 {
            len += 8; // y_mean, y_sd
        } else {
            let custom = self
                .classes
                .iter()
                .enumerate()
                .any(|(i, &cid)| cid != i as i32);
            if custom {
                len += 4 * self.classes.len();
            }
        }
        let has_map = self.col_indices.len() != self.x_cols_raw
            || self.col_indices.iter().enumerate().any(|(a, &r)| a != r);
        if has_map {
            len += 2 * self.col_indices.len();
        }
        // Per-active-col kind/int byte + 6 B center/scale only for non-bool/non-onehot cols!
        len += self.col_indices.len();
        for &gk in &self.group_kinds {
            if !matches!(gk, 3..=5) {
                len += 6;
            }
        }
        len += 2 * self.dropped_medians.len();
        len
    }
}

/// Class-relabeling-invariant nonlinear composite screening score S_j when `x_cols_raw > p_cap`.
fn screen_columns_composite(
    x_raw: &[Vec<f64>],
    y_raw: &[f64],
    task_id: u8,
    classes: &[i32],
    p_use: usize,
    col_meta: Option<&[Dse8ColumnMeta]>,
) -> Vec<usize> {
    let p_raw = x_raw.len();
    let mut scores = vec![0.0f64; p_raw];
    for j in 0..p_raw {
        let col = &x_raw[j];
        let mut fin_vals: Vec<f64> = Vec::with_capacity(col.len());
        let mut fin_y: Vec<f64> = Vec::with_capacity(col.len());
        let mut n_miss = 0usize;
        for (&xv, &yv) in col.iter().zip(y_raw) {
            if xv.is_finite() && yv.is_finite() {
                fin_vals.push(xv);
                fin_y.push(yv);
            } else if !xv.is_finite() {
                n_miss += 1;
            }
        }
        if fin_vals.len() < SUPPORT_FLOOR_K {
            continue;
        }
        let s = if task_id == 0 {
            let rp = pearson(&fin_vals, &fin_y).abs();
            let rs = spearman(&fin_vals, &fin_y).abs();
            let xi = chatterjee_xi(&fin_vals, &fin_y).max(0.0).sqrt();
            rp.max(rs).max(xi)
        } else {
            let mut best = 0.0f64;
            for &cid in classes {
                let yc: Vec<f64> = fin_y
                    .iter()
                    .map(|&v| if v.round() as i32 == cid { 1.0 } else { 0.0 })
                    .collect();
                let rp = pearson(&fin_vals, &yc).abs();
                let rs = spearman(&fin_vals, &yc).abs();
                let xi = chatterjee_xi(&fin_vals, &yc).max(0.0).sqrt();
                best = best.max(rp).max(rs).max(xi);
            }
            // Add quick 8-bin mutual separation proxy
            let (b8, _) = value_consistent_bins(&fin_vals, 8);
            let mut hit = [0usize; 8];
            let mut tot = [0usize; 8];
            let c0 = classes[0];
            for (&b, &yv) in b8.iter().zip(&fin_y) {
                if (b as usize) < 8 {
                    tot[b as usize] += 1;
                    if yv.round() as i32 == c0 {
                        hit[b as usize] += 1;
                    }
                }
            }
            let p0 = hit.iter().sum::<usize>() as f64 / tot.iter().sum::<usize>().max(1) as f64;
            let mut tv = 0.0;
            for b in 0..8 {
                if tot[b] >= SUPPORT_FLOOR_K {
                    tv += ((hit[b] as f64 / tot[b] as f64) - p0).abs()
                        * (tot[b] as f64 / fin_vals.len() as f64);
                }
            }
            best.max((2.0 * tv).min(1.0))
        };
        let miss_bonus = if n_miss >= SUPPORT_FLOOR_K { 0.05 } else { 0.0 };
        scores[j] = s + miss_bonus;
    }

    // Group-aware tie preservation: prefer keeping complete small nominal groups together.
    let mut order: Vec<usize> = (0..p_raw).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let mut kept = Vec::with_capacity(p_use);
    let mut used = vec![false; p_raw];
    for &c in &order {
        if kept.len() >= p_use {
            break;
        }
        if used[c] {
            continue;
        }
        if let Some(ms) = col_meta {
            if ms[c].group_kind == 4 && (ms[c].group_width as usize) <= 8 {
                let gid = ms[c].group_id;
                let members: Vec<usize> = (0..p_raw)
                    .filter(|&k| ms[k].group_id == gid && ms[k].group_kind == 4 && !used[k])
                    .collect();
                if kept.len() + members.len() <= p_use {
                    for m in members {
                        used[m] = true;
                        kept.push(m);
                    }
                    continue;
                }
            }
        }
        used[c] = true;
        kept.push(c);
    }
    kept.sort_unstable();
    kept
}

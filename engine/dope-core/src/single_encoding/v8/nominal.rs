//! Exact NC1/NC2 nominal one-hot sibling groups (`TAG_NOMINAL`, `0` illegal multi-hot rows!).

use super::rng_v8::{counter_uniform, sample_categorical, STREAM_ATOM};
use super::scale_v8::Dse8ColumnMeta;
use crate::evidence::primitives::SUPPORT_FLOOR_K;

/// Exact nominal one-hot sibling group (`TAG_NOMINAL`) ensuring 0 illegal multi-hot rows.
#[derive(Clone, Debug)]
pub struct NominalGroupV8 {
    /// Active column indices belonging to this one-hot group (`2..=32` cols).
    pub cols: Vec<u8>,
    /// Smoothed empirical probability for each column (`k >= 10` enforced on rare categories!).
    pub probs: Vec<f32>,
    /// Probability of all-zero baseline state (when dummy-encoded with dropped reference level).
    pub baseline_prob: f32,
}

/// Detect and fit exact NC1/NC2 nominal one-hot groups (`k >= 10` per level).
pub fn fit_nominal_groups(
    x_unit: &[Vec<f32>],
    active_meta: Option<&[Dse8ColumnMeta]>,
) -> Vec<NominalGroupV8> {
    let Some(ms) = active_meta else {
        return Vec::new();
    };
    let p = x_unit.len();
    let n = x_unit.first().map(|c| c.len()).unwrap_or(0);
    if n < SUPPORT_FLOOR_K {
        return Vec::new();
    }
    let mut by_gid = std::collections::BTreeMap::<u16, Vec<usize>>::new();
    for (j, m) in ms.iter().enumerate().take(p) {
        if m.group_kind == 4 {
            by_gid.entry(m.group_id).or_default().push(j);
        }
    }
    let mut groups = Vec::new();
    for cols in by_gid.into_values() {
        if cols.len() < 2 || cols.len() > 32 {
            continue;
        }
        let mut counts = vec![0usize; cols.len()];
        let mut base_cnt = 0usize;
        let mut valid_n = 0usize;
        for r in 0..n {
            let mut active_idx = None;
            let mut sum_hot = 0usize;
            for (ci, &col_j) in cols.iter().enumerate() {
                if x_unit[col_j][r].is_finite() && x_unit[col_j][r] > 0.5 {
                    active_idx = Some(ci);
                    sum_hot += 1;
                }
            }
            if sum_hot <= 1 {
                valid_n += 1;
                if let Some(ci) = active_idx {
                    counts[ci] += 1;
                } else {
                    base_cnt += 1;
                }
            }
        }
        if valid_n < SUPPORT_FLOOR_K {
            continue;
        }
        let mut probs = vec![0.0f32; cols.len()];
        let mut sum_kept = 0usize;
        for (ci, &cnt) in counts.iter().enumerate() {
            if cnt >= SUPPORT_FLOOR_K {
                sum_kept += cnt;
                probs[ci] = cnt as f32;
            }
        }
        let kept_base = if base_cnt >= SUPPORT_FLOOR_K {
            base_cnt
        } else {
            0
        };
        let tot = (sum_kept + kept_base) as f32;
        if tot <= 0.0 {
            continue;
        }
        for p_v in &mut probs {
            *p_v /= tot;
        }
        let baseline_prob = kept_base as f32 / tot;
        groups.push(NominalGroupV8 {
            cols: cols.into_iter().map(|c| c as u8).collect(),
            probs,
            baseline_prob,
        });
    }
    groups
}

/// Apply exact exclusive simplex decoding for a nominal one-hot group (`0` illegal rows!).
pub fn decode_nominal_group(
    grp: &NominalGroupV8,
    z_lat_row: &[f32],
    x_row: &mut [f32],
    miss_row: &mut [bool],
    seed: u64,
    row_idx: u64,
    group_idx: u64,
) {
    let k = grp.cols.len();
    let mut weights = Vec::with_capacity(k + 1);
    for (ci, &col_u8) in grp.cols.iter().enumerate() {
        let j = col_u8 as usize;
        let pr = grp.probs[ci] as f64;
        if pr <= 1e-6 {
            weights.push(0.0);
        } else {
            let z = z_lat_row.get(j).copied().unwrap_or(0.0) as f64;
            weights.push(pr * (0.75 * z.clamp(-3.0, 3.0)).exp());
        }
    }
    if grp.baseline_prob > 1e-6 {
        weights.push(grp.baseline_prob as f64);
    } else {
        weights.push(0.0);
    }
    let u = counter_uniform(seed, row_idx, STREAM_ATOM, 1024 + group_idx);
    let pick = sample_categorical(&weights, u);
    for (ci, &col_u8) in grp.cols.iter().enumerate() {
        let j = col_u8 as usize;
        if j < x_row.len() {
            x_row[j] = if pick == ci { 1.0 } else { 0.0 };
            miss_row[j] = false;
        }
    }
}

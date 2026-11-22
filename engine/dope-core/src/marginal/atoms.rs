//! Exact atom detection with structural minimum support `k = 10` and <= 32 atoms/column.

use crate::stats::norm_ppf;
use serde::{Deserialize, Serialize};

pub const DEFAULT_SUPPORT_K: usize = 10;
pub const MAX_ATOMS_V2: usize = 32;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AtomV2 {
    pub value: f64,
    pub prob: f64,
    pub u_lo: f64,
    pub u_hi: f64,
    pub z_lo: f64,
    pub z_hi: f64,
}

/// Detect exact point-mass atoms on sorted finite values `sorted`.
///
/// A value qualifies as an atom iff:
/// 1. `count >= max(support_k, ceil(0.005 * n))` (privacy support floor `k = 10`, O-3), and
/// 2. Its empirical mass exceeds `3.0x` the local smooth density expectation (or >= 10% mass).
///
/// Returns at most `MAX_ATOMS_V2` atoms ordered by ascending value.
pub fn detect_atoms(sorted: &[f64], support_k: usize) -> Vec<AtomV2> {
    let n = sorted.len();
    if n < support_k.max(2) {
        return Vec::new();
    }
    let min_count = support_k.max(((0.005 * n as f64).ceil() as usize).max(2));
    let mut runs: Vec<(f64, usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        let v = sorted[i];
        let mut j = i + 1;
        while j < n && sorted[j] == v {
            j += 1;
        }
        runs.push((v, i, j));
        i = j;
    }
    if runs.len() <= 1 {
        return Vec::new();
    }
    let nf = n as f64;
    let avg_mass = 1.0 / runs.len() as f64;
    let mut candidates: Vec<(usize, f64, usize, usize)> = Vec::new();
    for (idx, &(v, lo, hi)) in runs.iter().enumerate() {
        let cnt = hi - lo;
        if cnt < min_count {
            continue;
        }
        let share = cnt as f64 / nf;
        // Local neighbor baseline: average share of immediate neighbor runs (or global uniform).
        let left_cnt = if idx > 0 {
            runs[idx - 1].2 - runs[idx - 1].1
        } else {
            0
        };
        let right_cnt = if idx + 1 < runs.len() {
            runs[idx + 1].2 - runs[idx + 1].1
        } else {
            0
        };
        let neigh = match (idx > 0, idx + 1 < runs.len()) {
            (true, true) => 0.5 * (left_cnt + right_cnt) as f64 / nf,
            (true, false) => left_cnt as f64 / nf,
            (false, true) => right_cnt as f64 / nf,
            (false, false) => avg_mass,
        };
        let baseline = neigh.min(avg_mass).max(1.0 / nf);
        if share >= 0.10 || share >= 3.0 * baseline {
            candidates.push((cnt, v, lo, hi));
        }
    }
    // Keep up to MAX_ATOMS_V2 largest-mass atoms, then sort by ascending value.
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.total_cmp(&b.1)));
    candidates.truncate(MAX_ATOMS_V2);
    candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
    candidates
        .into_iter()
        .map(|(cnt, value, lo, hi)| {
            let u_lo = (lo as f64 / nf).clamp(1e-9, 1.0 - 1e-9);
            let u_hi = (hi as f64 / nf).clamp(1e-9, 1.0 - 1e-9);
            AtomV2 {
                value,
                prob: cnt as f64 / nf,
                u_lo,
                u_hi,
                z_lo: norm_ppf(u_lo),
                z_hi: norm_ppf(u_hi),
            }
        })
        .collect()
}

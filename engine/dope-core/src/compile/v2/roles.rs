//! Strict `BUILD / SELECT / CAL` role separation under Rule A-1 duplicate grouping.

use crate::teacher::panel::duplicate_grouped_kfolds;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize)]
pub struct CompilerRoles {
    pub build_rows: Vec<usize>,
    pub select_rows: Vec<usize>,
    pub cal_rows: Vec<usize>,
    pub roles_sha256: String,
}

impl CompilerRoles {
    /// Partition `0..n` into strictly disjoint duplicate-grouped `BUILD (~64%)`,
    /// `SELECT (~16%)`, and `CAL (~20%)` roles.
    pub fn split(x: &[Vec<f64>], y: &[f64], seed: u64) -> Self {
        let folds = duplicate_grouped_kfolds(x, y, 5, seed ^ 0xc01e_5a17);
        // Fold 4 -> CAL (20%); remaining 80% split 4:1 into BUILD (64%) and SELECT (16%).
        let cal_rows = folds.get(4).cloned().unwrap_or_default();
        let rem: Vec<usize> = folds
            .iter()
            .take(4)
            .flat_map(|f| f.iter().copied())
            .collect();
        let rem_x: Vec<Vec<f64>> = x
            .iter()
            .map(|col| rem.iter().map(|&r| col[r]).collect())
            .collect();
        let rem_y: Vec<f64> = rem.iter().map(|&r| y[r]).collect();
        let sub = duplicate_grouped_kfolds(&rem_x, &rem_y, 5, seed ^ 0x005e_1ec7);
        let mut select_rows: Vec<usize> = sub
            .first()
            .map(|f| f.iter().map(|&loc| rem[loc]).collect())
            .unwrap_or_default();
        let mut build_rows: Vec<usize> = sub
            .iter()
            .skip(1)
            .flat_map(|f| f.iter().map(|&loc| rem[loc]))
            .collect();
        build_rows.sort_unstable();
        select_rows.sort_unstable();
        let mut cal_sorted = cal_rows;
        cal_sorted.sort_unstable();

        let mut h = Sha256::new();
        h.update(seed.to_le_bytes());
        for (tag, rows) in [(0u8, &build_rows), (1u8, &select_rows), (2u8, &cal_sorted)] {
            h.update([tag]);
            h.update((rows.len() as u64).to_le_bytes());
            for &r in rows {
                h.update((r as u64).to_le_bytes());
            }
        }
        Self {
            build_rows,
            select_rows,
            cal_rows: cal_sorted,
            roles_sha256: format!("{:x}", h.finalize()),
        }
    }
}

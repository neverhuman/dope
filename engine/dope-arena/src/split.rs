//! Deterministic disjoint `A / B / C / T / P` splitter respecting physical groups,
//! time ordering, and Rule A-1 exact-duplicate grouping.

use crate::{ArenaError, ArenaResult};
use dope_core::RawTable;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SplitRatios {
    pub a_fit: f64,
    pub b_twin: f64,
    pub c_calib: f64,
    pub t_test: f64,
    pub p_privacy: f64,
}

impl Default for SplitRatios {
    fn default() -> Self {
        Self {
            a_fit: 0.35,
            b_twin: 0.25,
            c_calib: 0.15,
            t_test: 0.15,
            p_privacy: 0.10,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct DisjointSplits {
    pub a_fit: Vec<usize>,
    pub b_twin: Vec<usize>,
    pub c_calib: Vec<usize>,
    pub t_test: Vec<usize>,
    pub p_privacy: Vec<usize>,
    pub split_sha256: String,
    pub cluster_count: usize,
}

/// Canonical duplicate-cluster key for Rule A-1: either the declared `group_col` value
/// or the SHA-256 digest of the entire physical row so exact duplicates never straddle splits.
fn row_cluster_key(table: &RawTable, row: usize, group_col: Option<usize>) -> String {
    if let Some(gc) = group_col {
        return format!("grp:{}", table.columns[gc][row].trim());
    }
    let mut h = Sha256::new();
    for col in &table.columns {
        let cell = col[row].trim();
        h.update((cell.len() as u64).to_le_bytes());
        h.update(cell.as_bytes());
    }
    format!("row:{:x}", h.finalize())
}

fn splitmix64_hash(seed: u64, key: &str) -> u64 {
    let mut h = Sha256::new();
    h.update(seed.to_le_bytes());
    h.update(key.as_bytes());
    let d = h.finalize();
    u64::from_le_bytes(d[..8].try_into().unwrap_or([0; 8]))
}

/// Partition `table` into 5 strictly disjoint physical-row roles (`A/B/C/T/P`).
/// When `time_col` is given, splits chronologically (oldest -> A/B, middle -> C/P, newest -> T)
/// while keeping identical group/row clusters strictly indivisible.
pub fn split_table_disjoint(
    table: &RawTable,
    group_col: Option<usize>,
    time_col: Option<usize>,
    ratios: &SplitRatios,
    seed: u64,
) -> ArenaResult<DisjointSplits> {
    let n = table.rows();
    if n < 10 {
        return Err(ArenaError::Contract(
            "table needs at least 10 rows for disjoint A/B/C/T/P splits".into(),
        ));
    }
    let sum = ratios.a_fit + ratios.b_twin + ratios.c_calib + ratios.t_test + ratios.p_privacy;
    if !sum.is_finite() || (sum - 1.0).abs() > 1e-6 {
        return Err(ArenaError::Contract("split ratios must sum to 1.0".into()));
    }

    let mut clusters_map: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for r in 0..n {
        let key = row_cluster_key(table, r, group_col);
        clusters_map.entry(key).or_default().push(r);
    }
    let mut clusters: Vec<(String, Vec<usize>)> = clusters_map.into_iter().collect();
    if clusters.len() < 5 {
        return Err(ArenaError::Contract(
            "fewer than 5 distinct duplicate/group clusters for 5-way split".into(),
        ));
    }

    if let Some(tc) = time_col {
        clusters.sort_by(|a, b| {
            let ta = table.columns[tc][a.1[0]].trim().parse::<f64>().ok();
            let tb = table.columns[tc][b.1[0]].trim().parse::<f64>().ok();
            match (ta, tb) {
                (Some(x), Some(y)) => x.total_cmp(&y).then(a.0.cmp(&b.0)),
                _ => table.columns[tc][a.1[0]]
                    .cmp(&table.columns[tc][b.1[0]])
                    .then(a.0.cmp(&b.0)),
            }
        });
    } else {
        clusters.sort_by_key(|(k, _)| splitmix64_hash(seed, k));
    }

    let cluster_count = clusters.len();
    let targets = [
        (ratios.a_fit * n as f64).round() as usize,
        (ratios.b_twin * n as f64).round() as usize,
        (ratios.c_calib * n as f64).round() as usize,
        (ratios.p_privacy * n as f64).round() as usize,
        (ratios.t_test * n as f64).round() as usize,
    ];
    let mut buckets: [Vec<usize>; 5] = Default::default();
    if time_col.is_some() {
        // Chronological blocks: A, B, C, P, T (newest rows strictly in sealed T).
        let mut b = 0usize;
        for (idx, (_, rows)) in clusters.into_iter().enumerate() {
            let rem_clusters = cluster_count - idx;
            let rem_empty = (b + 1..5).filter(|&k| buckets[k].is_empty()).count();
            if b < 4
                && !buckets[b].is_empty()
                && (buckets[b].len() >= targets[b] || rem_clusters <= rem_empty)
            {
                b += 1;
            }
            buckets[b].extend(rows);
        }
    } else {
        // Greedy deficit assignment guaranteeing every role gets >= 1 cluster.
        for (idx, (_, rows)) in clusters.into_iter().enumerate() {
            let chosen = if idx < 5 {
                idx
            } else {
                (0..5)
                    .max_by(|&i, &j| {
                        let di = targets[i] as isize - buckets[i].len() as isize;
                        let dj = targets[j] as isize - buckets[j].len() as isize;
                        di.cmp(&dj).then(j.cmp(&i))
                    })
                    .unwrap_or(0)
            };
            buckets[chosen].extend(rows);
        }
    }
    for b in &mut buckets {
        b.sort_unstable();
    }
    let mut h = Sha256::new();
    h.update(seed.to_le_bytes());
    for (bi, b) in buckets.iter().enumerate() {
        h.update((bi as u64).to_le_bytes());
        h.update((b.len() as u64).to_le_bytes());
        for &r in b {
            h.update((r as u64).to_le_bytes());
        }
    }
    let [a_fit, b_twin, c_calib, p_privacy, t_test] = buckets;
    Ok(DisjointSplits {
        a_fit,
        b_twin,
        c_calib,
        t_test,
        p_privacy,
        split_sha256: format!("{:x}", h.finalize()),
        cluster_count,
    })
}

/// Extract a row-indexed sub-table without altering headers or target placement.
pub fn slice_table(table: &RawTable, rows: &[usize]) -> RawTable {
    let columns = table
        .columns
        .iter()
        .map(|col| rows.iter().map(|&r| col[r].clone()).collect())
        .collect();
    RawTable {
        headers: table.headers.clone(),
        columns,
    }
}

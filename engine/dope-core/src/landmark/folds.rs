use super::{job_seed, LandmarkFrame};
use crate::{rng::hash3, target::Head, DopeError, Result};
use std::collections::BTreeMap;

/// Stable physical-identity folds, stratified by class for independent rows.
/// Groups/time blocks override stratification and can never cross a boundary.
pub fn assign_folds(frame: &LandmarkFrame, folds: usize, seed: u64) -> Result<Vec<usize>> {
    if !(2..=10).contains(&folds) {
        return Err(DopeError::Input("invalid landmark fold count".into()));
    }
    let mut strata: BTreeMap<usize, Vec<(u64, Vec<usize>)>> = BTreeMap::new();
    if let Some(groups) = &frame.groups {
        let mut blocks: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (r, &g) in groups.iter().enumerate() {
            blocks.entry(g).or_default().push(r);
        }
        if blocks.len() < folds {
            return Err(DopeError::Input(
                "too few independent landmark blocks".into(),
            ));
        }
        strata.insert(0, blocks.into_iter().collect());
    } else {
        for (r, &id) in frame.ids.iter().enumerate() {
            let class = if frame.head == Head::Regression {
                0
            } else {
                frame.y[r] as usize
            };
            strata.entry(class).or_default().push((id, vec![r]));
        }
    }
    let mut assigned = vec![0; frame.rows()];
    let mut offset = 0;
    for blocks in strata.values_mut() {
        blocks.sort_by_key(|(id, _)| (hash3(seed, 0x464f4c44, *id), *id));
        for (i, (_, rows)) in blocks.iter().enumerate() {
            for &r in rows {
                assigned[r] = (i + offset) % folds;
            }
        }
        offset = (offset + blocks.len()) % folds;
    }
    if (0..folds).any(|k| assigned.iter().filter(|&&f| f == k).count() < 2) {
        return Err(DopeError::Input(
            "landmark folds need at least two query rows".into(),
        ));
    }
    Ok(assigned)
}

pub(crate) fn capped(frame: &LandmarkFrame, rows: &[usize], cap: usize, seed: u64) -> Vec<usize> {
    let mut out = rows.to_vec();
    out.sort_by_key(|&r| (hash3(seed, 0x434150, frame.ids[r]), frame.ids[r]));
    out.truncate(cap);
    out
}

pub(crate) fn fold_rows(
    frame: &LandmarkFrame,
    assigned: &[usize],
    k: usize,
    cap: usize,
    seed: u64,
) -> (Vec<usize>, Vec<usize>) {
    let train: Vec<usize> = (0..frame.rows()).filter(|&r| assigned[r] != k).collect();
    let query: Vec<usize> = (0..frame.rows()).filter(|&r| assigned[r] == k).collect();
    (capped(frame, &train, cap, job_seed(seed, 0, k)), query)
}

//! Three-coverage rollup and 10,000-draw family-clustered Holm-Bonferroni qualification (`T30`).

use super::harm::SplitMix64;
use crate::dvs::{ensure, DvsError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Per-dataset paired pass observation (`0.0` or `1.0` pass indicator, or continuous score)
/// keyed by physical source `family_id`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyComparisonRow {
    pub family_id: String,
    pub candidate_score: f64,
    pub comparator_score: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct PairedHolmResult {
    pub comparator_id: String,
    pub family_count: usize,
    pub mean_diff: f64,
    pub ci95_low: f64,
    pub ci95_high: f64,
    pub one_sided_p_value: f64,
    pub holm_adjusted_p_value: f64,
    pub strictly_beats_at_05: bool,
}

/// 10,000-draw family-clustered paired bootstrap with Holm-Bonferroni step-down adjustment (`T30`).
pub fn family_clustered_holm_comparison(
    comparators: &[(String, Vec<FamilyComparisonRow>)],
    draws: usize,
    seed: u64,
) -> Result<Vec<PairedHolmResult>, DvsError> {
    ensure(!comparators.is_empty() && draws >= 200, "invalid_holm_args")?;
    let mut raw_results = Vec::with_capacity(comparators.len());
    for (idx, (comp_id, rows)) in comparators.iter().enumerate() {
        ensure(!rows.is_empty(), "empty_family_comparison")?;
        // Aggregate multiple datasets/views per physical family into one independent cluster mean.
        let mut by_family: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
        for r in rows {
            ensure(
                r.candidate_score.is_finite() && r.comparator_score.is_finite(),
                "nonfinite_comparison_score",
            )?;
            let e = by_family.entry(r.family_id.as_str()).or_insert((0.0, 0));
            e.0 += r.candidate_score - r.comparator_score;
            e.1 += 1;
        }
        let fam_diffs: Vec<f64> = by_family
            .values()
            .map(|&(sum, cnt)| sum / cnt as f64)
            .collect();
        let m = fam_diffs.len();
        let mean_diff = fam_diffs.iter().sum::<f64>() / m as f64;
        let mut rng = SplitMix64::new(seed ^ ((idx as u64 + 1).wrapping_mul(0x9e37_79b9)));
        let mut boot = Vec::with_capacity(draws);
        let mut le_zero = 0usize;
        for _ in 0..draws {
            let mut acc = 0.0;
            for _ in 0..m {
                acc += fam_diffs[rng.gen_range(m)];
            }
            let d = acc / m as f64;
            if d <= 0.0 {
                le_zero += 1;
            }
            boot.push(d);
        }
        boot.sort_by(f64::total_cmp);
        let lo = boot[((0.025 * (draws as f64 - 1.0)).floor() as usize).min(draws - 1)];
        let hi = boot[((0.975 * (draws as f64 - 1.0)).ceil() as usize).min(draws - 1)];
        let p_one_sided = (le_zero as f64 + 1.0) / (draws as f64 + 1.0);
        raw_results.push((comp_id.clone(), m, mean_diff, lo, hi, p_one_sided));
    }

    // Holm-Bonferroni step-down adjustment across all comparators.
    let k = raw_results.len();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| {
        raw_results[a]
            .5
            .total_cmp(&raw_results[b].5)
            .then(a.cmp(&b))
    });
    let mut adj = vec![1.0_f64; k];
    let mut running_max = 0.0_f64;
    for (rank, &idx) in order.iter().enumerate() {
        let mult = (k - rank) as f64;
        let p = (raw_results[idx].5 * mult).min(1.0);
        running_max = running_max.max(p);
        adj[idx] = running_max;
    }

    Ok(raw_results
        .into_iter()
        .enumerate()
        .map(|(i, (comp_id, m, mean_diff, lo, hi, p))| PairedHolmResult {
            comparator_id: comp_id,
            family_count: m,
            mean_diff,
            ci95_low: lo,
            ci95_high: hi,
            one_sided_p_value: p,
            holm_adjusted_p_value: adj[i],
            strictly_beats_at_05: lo > 0.0 && adj[i] < 0.05,
        })
        .collect())
}

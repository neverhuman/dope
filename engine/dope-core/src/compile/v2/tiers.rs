//! Carry-forward tier monotonicity (`4 KiB -> 16 KiB -> 64 KiB -> Max`) preventing inversions (`T21`).

use super::select::CandidateScore;
use serde::Serialize;

pub const TIER_IMPROVEMENT_MARGIN: f64 = 0.005;

#[derive(Clone, Debug, Serialize)]
pub struct TierAssignment {
    pub tier_cap_bytes: usize,
    pub carried_forward: bool,
    pub winner: CandidateScore,
}

/// Enforce carry-forward monotonicity across ascending byte caps:
/// a larger tier replaces the carried-forward smaller tier iff it fits inside `cap` AND
/// either reduces violation buckets (`v_max` / `v_mean`) OR improves `l_pred` by at least
/// `TIER_IMPROVEMENT_MARGIN` (`0.005`), guaranteeing zero tier inversions > 0.005 (`T21`).
pub fn resolve_carry_forward_tiers(
    ascending_tiers: &[(usize, Vec<CandidateScore>)],
) -> Vec<TierAssignment> {
    let mut out: Vec<TierAssignment> = Vec::with_capacity(ascending_tiers.len());
    let mut carried: Option<CandidateScore> = None;

    for (cap, pool) in ascending_tiers {
        let best_in_cap = pool
            .iter()
            .filter(|c| c.bytes <= *cap)
            .min_by(|a, b| a.cmp_lex(b))
            .cloned();

        let (winner, was_carried) = match (&carried, best_in_cap) {
            (None, Some(cand)) => (cand, false),
            (Some(prev), None) => (prev.clone(), true),
            (Some(prev), Some(cand)) => {
                let pb = prev.bucket_key();
                let cb = cand.bucket_key();
                let better_viol = (cb.0, cb.1) < (pb.0, pb.1);
                let same_viol_better_pred = (cb.0, cb.1) == (pb.0, pb.1)
                    && (prev.l_pred - cand.l_pred) >= TIER_IMPROVEMENT_MARGIN;
                if better_viol || same_viol_better_pred {
                    (cand, false)
                } else {
                    (prev.clone(), true)
                }
            }
            (None, None) => continue,
        };
        carried = Some(winner.clone());
        out.push(TierAssignment {
            tier_cap_bytes: *cap,
            carried_forward: was_carried,
            winner,
        });
    }
    out
}

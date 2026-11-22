//! Paired harm, informative-cell qualification, and donor-aware cluster bootstrap (`T29`).

use super::bundle::{mean_slice, PredictionBundle};
use crate::dvs::{ensure, DvsError};
use serde::Serialize;
use std::collections::BTreeMap;

/// Deterministic SplitMix64 PRNG for bootstrap draws (no external RNG dependency).
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub(crate) fn gen_range(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % bound
    }
}

/// Group row indices by donor cluster ID (or singleton rows if `donor_ids` is absent).
pub(crate) fn donor_clusters(n: usize, donor_ids: Option<&[u64]>) -> Vec<Vec<usize>> {
    match donor_ids {
        None => (0..n).map(|i| vec![i]).collect(),
        Some(ids) => {
            let mut map: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
            for (idx, &id) in ids.iter().enumerate() {
                map.entry(id).or_default().push(idx);
            }
            map.into_values().collect()
        }
    }
}

/// One-sided percentile bootstrap bound over donor clusters (`alpha` e.g. 0.05 -> 95th pct UCB).
pub fn donor_bootstrap_bounds(
    deltas: &[f64],
    donor_ids: Option<&[u64]>,
    draws: usize,
    seed: u64,
    ucb_quantile: f64,
) -> Result<(f64, f64, f64), DvsError> {
    ensure(!deltas.is_empty() && draws >= 50, "invalid_bootstrap_args")?;
    ensure(
        ucb_quantile.is_finite() && ucb_quantile > 0.5 && ucb_quantile < 1.0,
        "invalid_bootstrap_quantile",
    )?;
    let clusters = donor_clusters(deltas.len(), donor_ids);
    let m = clusters.len();
    let cluster_sums: Vec<(f64, usize)> = clusters
        .iter()
        .map(|idxs| (idxs.iter().map(|&i| deltas[i]).sum(), idxs.len()))
        .collect();
    let point = mean_slice(deltas);
    let mut rng = SplitMix64::new(seed);
    let mut boot = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut sum = 0.0;
        let mut count = 0usize;
        for _ in 0..m {
            let c = rng.gen_range(m);
            sum += cluster_sums[c].0;
            count += cluster_sums[c].1;
        }
        boot.push(sum / count.max(1) as f64);
    }
    boot.sort_by(f64::total_cmp);
    let lo_idx = (((1.0 - ucb_quantile) * (draws as f64 - 1.0)).floor() as usize).min(draws - 1);
    let hi_idx = ((ucb_quantile * (draws as f64 - 1.0)).ceil() as usize).min(draws - 1);
    Ok((point, boot[lo_idx], boot[hi_idx]))
}

#[derive(Clone, Debug, Serialize)]
pub struct ParityAssessment {
    pub informative: bool,
    pub reason: Option<&'static str>,
    pub real_gain: f64,
    pub real_gain_lcb95: f64,
    pub unclipped_retention_s: Option<f64>,
    pub unclipped_retention_b: Option<f64>,
    pub paired_harm_vs_b: Option<f64>,
    pub paired_harm_ucb95: Option<f64>,
    pub harm_threshold: f64,
    pub parity_pass: bool,
    pub pass_tau_95_vs_a: bool,
    pub pass_tau_98_vs_a: bool,
    pub donor_cluster_count: usize,
}

/// Evaluate informative-cell status and paired harm vs fresh twin `B` (`dope-fidelity/v3`).
pub fn assess_parity(
    bundle: &PredictionBundle,
    draws: usize,
    seed: u64,
) -> Result<ParityAssessment, DvsError> {
    bundle.validate()?;
    let n = bundle.n_query();
    let l0 = bundle.mean_null();
    let la = bundle.mean_real_a();
    let gain = l0 - la;
    let gain_deltas: Vec<f64> = (0..n)
        .map(|i| bundle.null_losses[i] - bundle.real_a_losses[i])
        .collect();
    let (_, gain_lcb95, _) = donor_bootstrap_bounds(
        &gain_deltas,
        bundle.donor_ids.as_deref(),
        draws,
        seed ^ 0x11,
        0.95,
    )?;
    let ret_s = bundle.unclipped_retention_s();
    let ret_b = bundle.unclipped_retention_b();
    let rel_floor = 0.01 * l0.abs().max(1e-12);
    let twin_ok = ret_b.is_none_or(|rb| (0.80..=1.20).contains(&rb));
    let informative = gain > 0.0 && gain >= rel_floor && gain_lcb95 > 0.0 && twin_ok;
    let reason = if gain <= 0.0 || gain < rel_floor {
        Some("uninformative_real_gain_below_1pct")
    } else if gain_lcb95 <= 0.0 {
        Some("uninformative_real_gain_lcb_le_zero")
    } else if !twin_ok {
        Some("uninformative_twin_retention_out_of_band")
    } else {
        None
    };

    // Paired harm is evaluated against twin B when supplied, else against A.
    let ref_losses = bundle
        .real_b_losses
        .as_deref()
        .unwrap_or(&bundle.real_a_losses);
    let harm_deltas: Vec<f64> = (0..n)
        .map(|i| bundle.synth_losses[i] - ref_losses[i])
        .collect();
    let (harm_mean, _, harm_ucb95) = donor_bootstrap_bounds(
        &harm_deltas,
        bundle.donor_ids.as_deref(),
        draws,
        seed ^ 0x22,
        0.95,
    )?;
    let harm_threshold = 0.02 * gain.max(0.0);
    let parity_pass = informative && harm_ucb95 <= harm_threshold;
    let pass_tau_95 = informative && ret_s.is_some_and(|r| r >= 0.95);
    let pass_tau_98 = informative && ret_s.is_some_and(|r| r >= 0.98);
    let clusters = donor_clusters(n, bundle.donor_ids.as_deref()).len();

    Ok(ParityAssessment {
        informative,
        reason,
        real_gain: gain,
        real_gain_lcb95: gain_lcb95,
        unclipped_retention_s: ret_s,
        unclipped_retention_b: ret_b,
        paired_harm_vs_b: Some(harm_mean),
        paired_harm_ucb95: Some(harm_ucb95),
        harm_threshold,
        parity_pass,
        pass_tau_95_vs_a: pass_tau_95,
        pass_tau_98_vs_a: pass_tau_98,
        donor_cluster_count: clusters,
    })
}

//! Closed-loop exact-artifact compiler v2 (`BUILD/SELECT/CAL`, `P+` projection,
//! transitive bucketed lexicographic selection, and carry-forward tier monotonicity).

pub mod backoff;
pub mod loop_driver;
pub mod pipeline;
pub mod project;
pub mod roles;
pub mod select;
pub mod tiers;

pub use backoff::{
    run_privacy_backoff, CapacityBackoffStep, PrivacyBackoffOutcome, PRIVACY_BACKOFF_LADDER,
};
pub use pipeline::{fit_v2, FitV2Options, FitV2Report, FittedV2};

pub use loop_driver::{
    lineage_hash, run_exact_artifact_loop, CompileLedger, CompilerMove, DecodedCandidateArtifact,
};
pub use project::{
    compile_projected_sampler, project_moments_damped_newton, ProjectionFiveGapLadder,
    ProjectionResult, MIN_PROJECTION_ESS,
};
pub use roles::CompilerRoles;
pub use select::{select_best_decoded, CandidateScore};
pub use tiers::{resolve_carry_forward_tiers, TierAssignment, TIER_IMPROVEMENT_MARGIN};

use crate::artifact::format2_codec::FORMAT2_EMBED_SUMMARY_DIM;
use crate::catf::CatfModel;
use crate::distil::DistilledStudent;
use crate::target::Head;

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_embed_summary(
    n_fit: usize,
    budget_bytes: usize,
    catf: &CatfModel,
    student: &DistilledStudent,
    ess: f64,
    gap_raw: f64,
    gap_rew: f64,
    ridge_risk: f64,
) -> [f64; FORMAT2_EMBED_SUMMARY_DIM] {
    let d = catf.marginals.len();
    let n_disc = catf
        .marginals
        .iter()
        .filter(|m| matches!(m, crate::marginal::v2::ColumnMarginalV2::Discrete { .. }))
        .count();
    let catf_trees: usize = catf.conditionals.iter().map(|c| c.trees.len()).sum();
    let mean_sd = if catf.conditionals.is_empty() {
        1.0
    } else {
        catf.conditionals
            .iter()
            .map(|c| c.copula_cond_sd)
            .sum::<f64>()
            / catf.conditionals.len() as f64
    };
    let head_code = match student.head {
        Head::Regression => 0.0,
        Head::Binary => 1.0,
        Head::Multiclass(k) => 2.0 + (k as f64) * 0.01,
    };
    let mut s = [0.0; FORMAT2_EMBED_SUMMARY_DIM];
    s[0] = (n_fit as f64).max(1.0).ln();
    s[1] = (budget_bytes as f64).max(1.0).ln();
    s[2] = d as f64;
    s[3] = (d as f64 + 1.0).ln();
    s[4] = if d > 0 { n_disc as f64 / d as f64 } else { 0.0 };
    s[5] = catf.rules.rules.len() as f64;
    s[6] = catf_trees as f64;
    s[7] = mean_sd;
    s[8] = head_code;
    s[9] = student.main_effects.effects.len() as f64;
    s[10] = student.pair_effects.len() as f64;
    s[11] = student.forest.trees() as f64;
    s[12] = student.temperature;
    s[13] = ess;
    s[14] = gap_raw;
    s[15] = gap_rew;
    s[16] = ridge_risk.min(10.0);
    s
}

/// Prune copula predecessor weights per column to the `max_preds` strongest edges so wide tables
/// fit strictly under tight byte budgets (`2 KiB` / `4 KiB`) while preserving dominant covariance.
pub(crate) fn prune_catf_copula(catf: &mut CatfModel, max_preds: usize, drop_trees: bool) {
    for cond in &mut catf.conditionals {
        if drop_trees {
            cond.trees.clear();
        }
        if cond.predecessors.len() <= max_preds {
            continue;
        }
        let mut idx: Vec<usize> = (0..cond.predecessors.len()).collect();
        idx.sort_by(|&a, &b| {
            cond.copula_weights[b]
                .abs()
                .total_cmp(&cond.copula_weights[a].abs())
                .then(a.cmp(&b))
        });
        idx.truncate(max_preds);
        idx.sort_unstable();
        let kept_p: Vec<usize> = idx.iter().map(|&i| cond.predecessors[i]).collect();
        let kept_w: Vec<f64> = idx.iter().map(|&i| cond.copula_weights[i]).collect();
        let drop_var: f64 = (0..cond.copula_weights.len())
            .filter(|i| !idx.contains(i))
            .map(|i| cond.copula_weights[i].powi(2))
            .sum();
        cond.predecessors = kept_p;
        cond.copula_weights = kept_w;
        cond.copula_cond_sd = (cond.copula_cond_sd.powi(2) + drop_var).min(1.0).sqrt();
    }
}

/// Downsample continuous marginal interior knots to at most `max_k` (keeping endpoints intact).
/// When `trim_extra` is true (only used when strict byte cap is still exceeded), also prune
/// secondary atoms/offgrid/heaping/rare discrete levels and clear auxiliary structural rules.
pub(crate) fn compact_marginals(catf: &mut CatfModel, max_k: usize, trim_extra: bool) {
    if trim_extra {
        catf.rules.rules.clear();
    }
    for (col_idx, m) in catf.marginals.iter_mut().enumerate() {
        match m {
            crate::marginal::v2::ColumnMarginalV2::Continuous {
                atoms,
                heaping,
                smooth_u,
                smooth_x,
                offgrid_u,
                offgrid_x,
                ..
            } => {
                if smooth_u.len() > max_k && max_k >= 2 {
                    let len = smooth_u.len();
                    let mut nu = Vec::with_capacity(max_k);
                    let mut nx = Vec::with_capacity(max_k);
                    for i in 0..max_k {
                        let pos = (i * (len - 1)) / (max_k - 1);
                        nu.push(smooth_u[pos]);
                        nx.push(smooth_x[pos]);
                    }
                    *smooth_u = nu;
                    *smooth_x = nx;
                }
                if trim_extra {
                    *heaping = None;
                    offgrid_u.clear();
                    offgrid_x.clear();
                    let keep_a = if max_k <= 2 { 0 } else { 1 };
                    if atoms.len() > keep_a {
                        atoms.sort_by(|a, b| b.prob.total_cmp(&a.prob));
                        atoms.truncate(keep_a);
                        atoms.sort_by(|a, b| a.value.total_cmp(&b.value));
                    }
                }
            }
            crate::marginal::v2::ColumnMarginalV2::Discrete { values, probs, .. } => {
                let cap = max_k.max(2);
                if trim_extra && values.len() > cap {
                    let mut idx: Vec<usize> = (0..values.len()).collect();
                    idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
                    idx.truncate(cap);
                    idx.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
                    let nv: Vec<f64> = idx.iter().map(|&i| values[i]).collect();
                    let mut np: Vec<f64> = idx.iter().map(|&i| probs[i]).collect();
                    let sum: f64 = np.iter().sum::<f64>().max(1e-12);
                    for p in &mut np {
                        *p /= sum;
                    }
                    let new_k = nv.len();
                    *values = nv;
                    *probs = np;
                    if let Some(cond) = catf.conditionals.iter_mut().find(|c| c.col_idx == col_idx)
                    {
                        cond.k_bins = new_k;
                        cond.trees.clear();
                    }
                }
            }
        }
    }
}

/// Keep only the `max_effects` highest-energy `MainEffect` splines (preserving intercept).
pub(crate) fn prune_main_effects(student: &mut DistilledStudent, max_effects: usize) {
    if student.main_effects.effects.len() <= max_effects {
        return;
    }
    let mut scored: Vec<(f64, usize)> = student
        .main_effects
        .effects
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let energy: f64 = e.values.iter().map(|v| v * v).sum();
            (energy, i)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut keep: Vec<usize> = scored
        .into_iter()
        .take(max_effects)
        .map(|(_, i)| i)
        .collect();
    keep.sort_unstable();
    student.main_effects.effects = keep
        .into_iter()
        .map(|i| student.main_effects.effects[i].clone())
        .collect();
}

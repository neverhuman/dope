//! Copula-Anchored Autoregressive Tree Flow (`CATF`) public API.

pub mod fit;
pub mod order;
pub mod sample;
mod split;

use crate::copula::calibrate_latent_correlation_matrix;
use crate::distil::DistilledStudent;
use crate::marginal::atoms::DEFAULT_SUPPORT_K;
use crate::marginal::v2::ColumnMarginalV2;
use crate::sample::Samples;
use crate::stats::norm_ppf;
use crate::structure::{RuleMode, RuleSet};
use fit::{copula_conditional_params, fit_catf_trees, standard_z_bins, CatfConditional, CATF_BINS};
use order::least_predictable_order;
use sample::{sample_catf_covariates, sample_post_rule_targets};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct CatfConfig {
    pub max_trees_per_col: usize,
    pub max_depth: usize,
    pub support_k: usize,
    pub alpha_shrink: f64,
    pub eta: f64,
    pub max_knots: usize,
    pub max_discrete: usize,
    pub rule_mode: RuleMode,
    pub forced_first: Vec<usize>,
}

impl Default for CatfConfig {
    fn default() -> Self {
        Self {
            max_trees_per_col: 12,
            max_depth: 3,
            support_k: DEFAULT_SUPPORT_K,
            alpha_shrink: 5.0,
            eta: 0.35,
            max_knots: 24,
            max_discrete: 64,
            rule_mode: RuleMode::Structural,
            forced_first: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatfModel {
    pub order: Vec<usize>,
    pub conditionals: Vec<CatfConditional>,
    pub marginals: Vec<ColumnMarginalV2>,
    pub is_nominal: Vec<bool>,
    pub nom_levels: Vec<Vec<f64>>,
    pub rules: RuleSet,
}

impl CatfModel {
    /// Fit full Phase P5 covariate model: `ColumnMarginalV2` + zero-byte calibrated copula
    /// + least-predictable-first CATF trees + `BUILD/SELECT`-confirmed structural rules.
    pub fn fit(
        build_x: &[Vec<f64>],
        select_x: &[Vec<f64>],
        is_nominal: &[bool],
        cfg: &CatfConfig,
    ) -> Self {
        let d = build_x.len();
        let n = build_x.first().map_or(0, Vec::len);
        let support_k = cfg.support_k.max(DEFAULT_SUPPORT_K);
        let marginals: Vec<ColumnMarginalV2> = (0..d)
            .map(|j| ColumnMarginalV2::fit(&build_x[j], cfg.max_knots, cfg.max_discrete, support_k))
            .collect();
        let is_discrete: Vec<bool> = marginals
            .iter()
            .map(|m| matches!(m, ColumnMarginalV2::Discrete { .. }))
            .collect();
        let nom_levels: Vec<Vec<f64>> = (0..d)
            .map(|j| match &marginals[j] {
                ColumnMarginalV2::Discrete { values, .. } => values.clone(),
                _ => Vec::new(),
            })
            .collect();
        let nom_codes: Vec<Vec<Option<u8>>> = (0..d)
            .map(|j| {
                if !is_nominal[j] || nom_levels[j].is_empty() {
                    vec![None; n]
                } else {
                    (0..n)
                        .map(|r| {
                            let v = build_x[j][r];
                            v.is_finite().then(|| {
                                nom_levels[j]
                                    .iter()
                                    .enumerate()
                                    .min_by(|a, b| (a.1 - v).abs().total_cmp(&(b.1 - v).abs()))
                                    .map_or(0, |(i, _)| i as u8)
                            })
                        })
                        .collect()
                }
            })
            .collect();

        // Mine structural rules BEFORE CATF so deterministically derived columns skip CATF tree fitting.
        let rules = RuleSet::mine(build_x, select_x, &is_discrete, cfg.rule_mode, support_k);
        let derived_cols = rules.derived_columns();

        // Empirical Pearson correlation matrix + zero-byte copula latent calibration!
        let emp_corr = empirical_corr_matrix(build_x);
        let latent_corr = calibrate_latent_correlation_matrix(&marginals, &emp_corr);
        let z_cols: Vec<Vec<f64>> = (0..d)
            .map(|j| stratified_latent_z(&marginals[j], &build_x[j]))
            .collect();
        let order = least_predictable_order(&z_cols, &cfg.forced_first);
        let min_leaf = support_k.max(((0.005 * n as f64).ceil() as usize).max(4));

        let mut conditionals = Vec::with_capacity(d);
        for (pos, &col) in order.iter().enumerate() {
            let preds = order[..pos].to_vec();
            let (copula_weights, copula_cond_sd) =
                copula_conditional_params(&latent_corr, d, col, &preds);
            let (k_bins, z_edges, z_centers, base_missing_prob, target_states) =
                match &marginals[col] {
                    ColumnMarginalV2::Discrete {
                        values,
                        probs,
                        p_missing,
                        ..
                    } => {
                        let l = values.len();
                        let mut edges = Vec::with_capacity(l + 1);
                        edges.push(f64::NEG_INFINITY);
                        let mut cum = 0.0;
                        for &p in &probs[..l.saturating_sub(1)] {
                            cum += p;
                            edges.push(norm_ppf(cum.clamp(1e-9, 1.0 - 1e-9)));
                        }
                        edges.push(f64::INFINITY);
                        let centers: Vec<f64> = (0..l)
                            .map(|i| {
                                let lo = if edges[i].is_finite() { edges[i] } else { -2.5 };
                                let hi = if edges[i + 1].is_finite() {
                                    edges[i + 1]
                                } else {
                                    2.5
                                };
                                0.5 * (lo + hi)
                            })
                            .collect();
                        let states: Vec<usize> = (0..n)
                            .map(|r| {
                                let v = build_x[col][r];
                                if !v.is_finite() {
                                    l
                                } else {
                                    values
                                        .iter()
                                        .enumerate()
                                        .min_by(|a, b| (a.1 - v).abs().total_cmp(&(b.1 - v).abs()))
                                        .map_or(0, |(i, _)| i)
                                }
                            })
                            .collect();
                        (l, edges, centers, *p_missing, states)
                    }
                    ColumnMarginalV2::Continuous { p_missing, .. } => {
                        let (edges, centers) = standard_z_bins(CATF_BINS);
                        let states: Vec<usize> = (0..n)
                            .map(|r| {
                                let zv = z_cols[col][r];
                                if !zv.is_finite() {
                                    CATF_BINS
                                } else {
                                    edges[1..CATF_BINS]
                                        .partition_point(|&e| e < zv)
                                        .min(CATF_BINS - 1)
                                }
                            })
                            .collect();
                        (CATF_BINS, edges, centers, *p_missing, states)
                    }
                };
            let mut cond = CatfConditional {
                col_idx: col,
                predecessors: preds,
                copula_weights,
                copula_cond_sd,
                base_missing_prob,
                is_discrete: is_discrete[col],
                k_bins,
                z_edges,
                z_centers,
                trees: Vec::new(),
            };
            if !derived_cols.contains(&col) {
                fit_catf_trees(
                    build_x,
                    &z_cols,
                    is_nominal,
                    &nom_codes,
                    &mut cond,
                    &target_states,
                    cfg.max_trees_per_col,
                    cfg.max_depth,
                    min_leaf,
                    cfg.alpha_shrink,
                    cfg.eta,
                );
            }
            conditionals.push(cond);
        }

        Self {
            order,
            conditionals,
            marginals,
            is_nominal: is_nominal.to_vec(),
            nom_levels,
            rules,
        }
    }

    pub fn sample_x(&self, n_rows: usize, seed: u64, table_id: u64) -> Vec<Vec<f64>> {
        sample_catf_covariates(
            &self.order,
            &self.conditionals,
            &self.marginals,
            &self.is_nominal,
            &self.nom_levels,
            &self.rules,
            n_rows,
            seed,
            table_id,
        )
    }

    /// Sample full `(X, Y)` guaranteeing `Y` is evaluated strictly at post-rule `X` (`T32`).
    pub fn sample_with_student(
        &self,
        student: &DistilledStudent,
        n_rows: usize,
        seed: u64,
        table_id: u64,
    ) -> Samples {
        let post_rule_x = self.sample_x(n_rows, seed, table_id);
        sample_post_rule_targets(student, &post_rule_x, seed, table_id)
    }
}

/// Map observed column values to latent standard-normal scores while spreading tied point-mass
/// atom rows evenly across their `[u_lo, u_hi]` interval so `z_cols` matches `N(0, 1)` across
/// all 32 CATF bins instead of collapsing an entire atom into one artificial bin spike.
fn stratified_latent_z(marginal: &ColumnMarginalV2, col: &[f64]) -> Vec<f64> {
    let n = col.len();
    let ColumnMarginalV2::Continuous { atoms, .. } = marginal else {
        return col.iter().map(|&v| marginal.z_score(v)).collect();
    };
    if atoms.is_empty() {
        return col.iter().map(|&v| marginal.z_score(v)).collect();
    }
    let mut atom_counts = vec![0usize; atoms.len()];
    let mut row_atom: Vec<Option<usize>> = vec![None; n];
    for (r, &v) in col.iter().enumerate() {
        if v.is_finite() {
            for (ai, a) in atoms.iter().enumerate() {
                if (v - a.value).abs() <= 1e-12 * a.value.abs().max(1.0) {
                    row_atom[r] = Some(ai);
                    atom_counts[ai] += 1;
                    break;
                }
            }
        }
    }
    let mut seen = vec![0usize; atoms.len()];
    (0..n)
        .map(|r| {
            if let Some(ai) = row_atom[r] {
                let a = &atoms[ai];
                let k = seen[ai];
                seen[ai] += 1;
                let frac = (k as f64 + 0.5) / atom_counts[ai].max(1) as f64;
                let u = (a.u_lo + frac * (a.u_hi - a.u_lo)).clamp(1e-6, 1.0 - 1e-6);
                norm_ppf(u)
            } else {
                marginal.z_score(col[r])
            }
        })
        .collect()
}

fn empirical_corr_matrix(x_cols: &[Vec<f64>]) -> Vec<f64> {
    let d = x_cols.len();
    let n = x_cols.first().map_or(0, Vec::len);
    let mut r = vec![0.0; d * d];
    for i in 0..d {
        r[i * d + i] = 1.0;
        for j in (i + 1)..d {
            let (mut sx, mut sy, mut cnt) = (0.0, 0.0, 0usize);
            for row in 0..n {
                let (a, b) = (x_cols[i][row], x_cols[j][row]);
                if a.is_finite() && b.is_finite() {
                    sx += a;
                    sy += b;
                    cnt += 1;
                }
            }
            let corr = if cnt < 4 {
                0.0
            } else {
                let (mx, my) = (sx / cnt as f64, sy / cnt as f64);
                let (mut num, mut vx, mut vy) = (0.0, 0.0, 0.0);
                for row in 0..n {
                    let (a, b) = (x_cols[i][row], x_cols[j][row]);
                    if a.is_finite() && b.is_finite() {
                        let (dx, dy) = (a - mx, b - my);
                        num += dx * dy;
                        vx += dx * dx;
                        vy += dy * dy;
                    }
                }
                if vx <= 1e-12 || vy <= 1e-12 {
                    0.0
                } else {
                    (num / (vx * vy).sqrt()).clamp(-0.995, 0.995)
                }
            };
            r[i * d + j] = corr;
            r[j * d + i] = corr;
        }
    }
    r
}

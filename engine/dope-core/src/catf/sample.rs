//! Stateless direct one-pass CATF truncated-normal sampler + post-rule target evaluation (`T32`).

use super::fit::{copula_base_logits, CatfConditional};
use crate::distil::DistilledStudent;
use crate::marginal::v2::ColumnMarginalV2;
use crate::rng::{
    row_keyed_normal, row_keyed_uniform, DEFAULT_NAMESPACE_128, PURPOSE_CATF, PURPOSE_LABEL,
    PURPOSE_NOISE, PURPOSE_RULE,
};
use crate::sample::Samples;
use crate::stats::{norm_cdf, norm_ppf};
use crate::structure::RuleSet;
use crate::target::{log_sum_exp, Head};
use rayon::prelude::*;

/// Sample `n_rows` synthetic covariates `X` in one stateless pass via CATF + ColumnMarginalV2
/// + direct structural rule enforcement (zero rejection loop!).
#[allow(clippy::too_many_arguments)]
pub fn sample_catf_covariates(
    order: &[usize],
    conditionals: &[CatfConditional],
    marginals: &[ColumnMarginalV2],
    is_nominal: &[bool],
    nom_levels: &[Vec<f64>],
    rules: &RuleSet,
    n_rows: usize,
    seed: u64,
    table_id: u64,
) -> Vec<Vec<f64>> {
    let d = marginals.len();
    let mut flat = vec![0.0; n_rows * d];
    flat.par_chunks_mut(d).enumerate().for_each(|(r, x_row)| {
        let rid = r as u64;
        let mut z_row = vec![0.0_f64; d];
        let mut nom_codes: Vec<Option<u8>> = vec![None; d];
        for (ord_pos, &col) in order.iter().enumerate() {
            let cond = &conditionals[ord_pos];
            let n_states = cond.k_bins + 1;
            let mut logits = vec![0.0; n_states];
            let mut mu = 0.0;
            for (p_i, &pf) in cond.predecessors.iter().enumerate() {
                let zv = z_row[pf];
                if zv.is_finite() {
                    mu += cond.copula_weights[p_i] * zv;
                }
            }
            copula_base_logits(
                &cond.z_edges,
                mu,
                cond.copula_cond_sd,
                cond.base_missing_prob,
                &mut logits,
            );
            for tree in &cond.trees {
                tree.add_logits(x_row, is_nominal, &nom_codes, &cond.z_centers, &mut logits);
            }
            let lse = log_sum_exp(&logits);
            let u_bin = row_keyed_uniform(
                &DEFAULT_NAMESPACE_128,
                seed,
                table_id,
                rid,
                0,
                PURPOSE_CATF,
                (col * 3) as u32,
            );
            let mut cum = 0.0;
            let mut picked = n_states - 1;
            for (s, &lg) in logits.iter().enumerate() {
                cum += (lg - lse).exp();
                if u_bin <= cum {
                    picked = s;
                    break;
                }
            }
            if picked == cond.k_bins {
                x_row[col] = f64::NAN;
                z_row[col] = f64::NAN;
                nom_codes[col] = None;
                continue;
            }
            let u_in = row_keyed_uniform(
                &DEFAULT_NAMESPACE_128,
                seed,
                table_id,
                rid,
                0,
                PURPOSE_CATF,
                (col * 3 + 1) as u32,
            );
            let u_mix = row_keyed_uniform(
                &DEFAULT_NAMESPACE_128,
                seed,
                table_id,
                rid,
                0,
                PURPOSE_CATF,
                (col * 3 + 2) as u32,
            );
            // Direct truncated-normal inversion inside bin `[z_edges[picked], z_edges[picked+1]]`.
            let p_lo = if cond.z_edges[picked].is_finite() {
                norm_cdf(cond.z_edges[picked])
            } else {
                1e-6
            };
            let p_hi = if cond.z_edges[picked + 1].is_finite() {
                norm_cdf(cond.z_edges[picked + 1])
            } else {
                1.0 - 1e-6
            };
            let u_draw = (p_lo + u_in * (p_hi - p_lo).max(1e-9)).clamp(1e-6, 1.0 - 1e-6);
            z_row[col] = norm_ppf(u_draw);
            x_row[col] = marginals[col].sample_from_uniform(u_draw, u_mix, 1.0);
            if is_nominal[col] {
                let code = nom_levels[col]
                    .iter()
                    .enumerate()
                    .min_by(|a, b| {
                        (a.1 - x_row[col])
                            .abs()
                            .total_cmp(&(b.1 - x_row[col]).abs())
                    })
                    .map(|(i, _)| i as u8);
                nom_codes[col] = code;
            }
        }
        // Apply mined structural rules strictly AFTER covariate transport (`T32`).
        let u_exc = row_keyed_uniform(
            &DEFAULT_NAMESPACE_128,
            seed,
            table_id,
            rid,
            0,
            PURPOSE_RULE,
            0,
        );
        let u_renorm = row_keyed_uniform(
            &DEFAULT_NAMESPACE_128,
            seed,
            table_id,
            rid,
            0,
            PURPOSE_RULE,
            1,
        );
        rules.apply_row(x_row, u_exc, u_renorm);
    });

    (0..d)
        .into_par_iter()
        .map(|j| (0..n_rows).map(|r| flat[r * d + j]).collect())
        .collect()
}

/// Sample full `(X, Y)` evaluating the distilled conditional student strictly at the
/// **post-transport, post-rule** covariates `X` (`T32`).
pub fn sample_post_rule_targets(
    student: &DistilledStudent,
    post_rule_x: &[Vec<f64>],
    seed: u64,
    table_id: u64,
) -> Samples {
    let n = post_rule_x.first().map_or(0, Vec::len);
    let logits = student.predict_logits(post_rule_x);
    let o = student.head.outputs();
    let y: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|r| {
            let rid = r as u64;
            match student.head {
                Head::Regression => {
                    let mu = logits[r];
                    let z = row_keyed_normal(
                        &DEFAULT_NAMESPACE_128,
                        seed,
                        table_id,
                        rid,
                        0,
                        PURPOSE_NOISE,
                        0,
                    );
                    let eps = student
                        .residual_model
                        .as_ref()
                        .map_or(0.0, |rm| rm.sample_residual(mu, z));
                    mu + eps
                }
                Head::Binary => {
                    let prob = 1.0 / (1.0 + (-logits[r] / student.temperature).exp());
                    let u = row_keyed_uniform(
                        &DEFAULT_NAMESPACE_128,
                        seed,
                        table_id,
                        rid,
                        0,
                        PURPOSE_LABEL,
                        0,
                    );
                    (u < prob) as u8 as f64
                }
                Head::Multiclass(k) => {
                    let tau = student.temperature;
                    (0..k)
                        .map(|c| {
                            let u = row_keyed_uniform(
                                &DEFAULT_NAMESPACE_128,
                                seed,
                                table_id,
                                rid,
                                0,
                                PURPOSE_LABEL,
                                c as u32,
                            );
                            logits[r * o + c] / tau - (-u.ln()).ln()
                        })
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(&b.1))
                        .map_or(0.0, |(c, _)| c as f64)
                }
            }
        })
        .collect();
    Samples {
        x: post_rule_x.to_vec(),
        y,
    }
}

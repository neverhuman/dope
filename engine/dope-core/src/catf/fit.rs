//! CATF fitter: Copula-Anchored Autoregressive Tree Flow with 32-bin + missing states,
//! Hermite basis (`He_1..3`) + full-K residual, typed predicates (`NominalIn` for `T05`),
//! and depth-2 lookahead for XOR / 3-bit parity (`T16`).

use super::split::{
    apply_tree_leaf_row, best_lookahead_pair, best_split_gain, build_candidate_predicates,
    compute_leaf_tilts, eval_tree_on_select, mean_state_nll,
};
use crate::stats::{cholesky, norm_cdf, norm_ppf};
use crate::target::log_sum_exp;
use serde::{Deserialize, Serialize};

pub const CATF_BINS: usize = 32;

/// Typed oblivious tree predicate ensuring nominal permutation invariance (`T05`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CatfPredicate {
    NumericLess {
        feat: usize,
        threshold: f64,
    },
    /// Set-membership bitmask over discrete/nominal code indices `0..64` (`T05`).
    NominalIn {
        feat: usize,
        mask: u64,
    },
    IsMissing {
        feat: usize,
    },
}

impl CatfPredicate {
    #[inline]
    pub fn eval(&self, x_row: &[f64], is_nom: &[bool], nom_codes: &[Option<u8>]) -> bool {
        match *self {
            Self::NumericLess { feat, threshold } => {
                let v = x_row[feat];
                v.is_finite() && v < threshold
            }
            Self::NominalIn { feat, mask } => {
                if is_nom[feat] {
                    nom_codes[feat].is_some_and(|c| c < 64 && ((mask >> c) & 1) == 1)
                } else {
                    false
                }
            }
            Self::IsMissing { feat } => !x_row[feat].is_finite(),
        }
    }
}

/// One oblivious tree tilting `K` state log-probabilities via Hermite (`He_1..3`) or full-K
/// leaf vectors plus an explicit missingness logit tilt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatfTree {
    pub predicates: Vec<CatfPredicate>,
    /// True if leaves store `[He_1, He_2, He_3, miss]` (4 floats/leaf); false for full `[K + 1]`.
    pub hermite_mode: bool,
    pub leaf_values: Vec<f64>,
    pub n_states: usize,
}

impl CatfTree {
    #[inline]
    pub fn leaf_index(&self, x_row: &[f64], is_nom: &[bool], nom_codes: &[Option<u8>]) -> usize {
        let mut idx = 0usize;
        for (l, pred) in self.predicates.iter().enumerate() {
            if pred.eval(x_row, is_nom, nom_codes) {
                idx |= 1 << l;
            }
        }
        idx
    }

    pub fn add_logits(
        &self,
        x_row: &[f64],
        is_nom: &[bool],
        nom_codes: &[Option<u8>],
        bin_centers_z: &[f64],
        logits: &mut [f64],
    ) {
        let lf = self.leaf_index(x_row, is_nom, nom_codes);
        let k = self.n_states - 1;
        if self.hermite_mode {
            let base = lf * 4;
            let (c1, c2, c3, cm) = (
                self.leaf_values[base],
                self.leaf_values[base + 1],
                self.leaf_values[base + 2],
                self.leaf_values[base + 3],
            );
            for (b, &z) in bin_centers_z.iter().enumerate().take(k) {
                let he1 = z;
                let he2 = z * z - 1.0;
                let he3 = z * z * z - 3.0 * z;
                logits[b] += c1 * he1 + c2 * he2 + c3 * he3;
            }
            logits[k] += cm;
        } else {
            let base = lf * self.n_states;
            for s in 0..self.n_states {
                logits[s] += self.leaf_values[base + s];
            }
        }
    }
}

/// Conditional specification for one column `col_idx` given predecessors in CATF order.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatfConditional {
    pub col_idx: usize,
    pub predecessors: Vec<usize>,
    /// Linear Gaussian copula conditional weights `mu_j = sum_p copula_weights[p] * z[pred[p]]`.
    pub copula_weights: Vec<f64>,
    pub copula_cond_sd: f64,
    pub base_missing_prob: f64,
    pub is_discrete: bool,
    /// Number of value states (`CATF_BINS` for continuous; `L` for discrete).
    pub k_bins: usize,
    /// Normal-score bin edges `[k_bins + 1]` (continuous) or discrete thresholds `[L + 1]`.
    pub z_edges: Vec<f64>,
    pub z_centers: Vec<f64>,
    pub trees: Vec<CatfTree>,
}

/// Compute exact copula conditional regression `(weights, cond_sd)` of `col` on `preds`
/// from the full calibrated correlation matrix `corr` (`d x d`).
pub fn copula_conditional_params(
    corr: &[f64],
    d: usize,
    col: usize,
    preds: &[usize],
) -> (Vec<f64>, f64) {
    let m = preds.len();
    if m == 0 {
        return (Vec::new(), 1.0);
    }
    let mut sub = vec![0.0; m * m];
    let mut rhs = vec![0.0; m];
    for a in 0..m {
        rhs[a] = corr[preds[a] * d + col];
        for b in 0..m {
            sub[a * m + b] = corr[preds[a] * d + preds[b]] + if a == b { 1e-4 } else { 0.0 };
        }
    }
    let Some(l) = cholesky(&sub, m) else {
        return (vec![0.0; m], 1.0);
    };
    // Solve L y = rhs, then L^T w = y.
    let mut y = vec![0.0; m];
    for i in 0..m {
        let mut s = rhs[i];
        for k in 0..i {
            s -= l[i * m + k] * y[k];
        }
        y[i] = s / l[i * m + i].max(1e-9);
    }
    let mut w = vec![0.0; m];
    for i in (0..m).rev() {
        let mut s = y[i];
        for k in (i + 1)..m {
            s -= l[k * m + i] * w[k];
        }
        w[i] = s / l[i * m + i].max(1e-9);
    }
    let r2: f64 = w.iter().zip(&rhs).map(|(&wi, &ri)| wi * ri).sum();
    let sd = (1.0 - r2).clamp(1e-3, 1.0).sqrt();
    (w, sd)
}

/// Base log-probabilities `[k_bins + 1]` from the Gaussian copula conditional `(mu, sd)` + missing prob.
pub fn copula_base_logits(z_edges: &[f64], mu: f64, sd: f64, p_miss: f64, out_logits: &mut [f64]) {
    let k = z_edges.len() - 1;
    let p_m = p_miss.clamp(1e-6, 1.0 - 1e-6);
    let p_obs = 1.0 - p_m;
    let inv_sd = 1.0 / sd.max(1e-3);
    for b in 0..k {
        let p_lo = if z_edges[b].is_finite() {
            norm_cdf((z_edges[b] - mu) * inv_sd)
        } else {
            0.0
        };
        let p_hi = if z_edges[b + 1].is_finite() {
            norm_cdf((z_edges[b + 1] - mu) * inv_sd)
        } else {
            1.0
        };
        let mass = (p_hi - p_lo).max(1e-7);
        out_logits[b] = (p_obs * mass).ln();
    }
    out_logits[k] = p_m.ln();
}

/// Standard equal-probability `CATF_BINS` normal-score bin edges and centers.
pub fn standard_z_bins(k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut edges = Vec::with_capacity(k + 1);
    edges.push(f64::NEG_INFINITY);
    for b in 1..k {
        edges.push(norm_ppf(b as f64 / k as f64));
    }
    edges.push(f64::INFINITY);
    let centers: Vec<f64> = (0..k)
        .map(|b| norm_ppf((b as f64 + 0.5) / k as f64).clamp(-2.5, 2.5))
        .collect();
    (edges, centers)
}

pub type SelectEval<'a> = (&'a [Vec<f64>], &'a [Vec<f64>], &'a [Vec<Option<u8>>]);

/// Fit CATF oblivious residual trees on top of the Gaussian copula anchor for column `col_idx`.
#[allow(clippy::too_many_arguments)]
pub fn fit_catf_trees(
    x_cols: &[Vec<f64>],
    z_cols: &[Vec<f64>],
    is_nominal: &[bool],
    nom_codes: &[Vec<Option<u8>>],
    select_eval: Option<SelectEval<'_>>,
    cond: &mut CatfConditional,
    target_states: &[usize],
    max_trees: usize,
    max_depth: usize,
    min_leaf: usize,
    alpha_shrink: f64,
    eta: f64,
) {
    let n = target_states.len();
    let n_states = cond.k_bins + 1;
    if cond.predecessors.is_empty() || max_trees == 0 || n < 2 * min_leaf {
        return;
    }
    let depth = max_depth.clamp(1, 4);
    let use_hermite = !cond.is_discrete && cond.k_bins > 8;

    // Precompute candidate typed predicates over predecessors ONLY.
    let preds = build_candidate_predicates(x_cols, is_nominal, nom_codes, &cond.predecessors, n);
    if preds.is_empty() {
        return;
    }
    let pred_bits: Vec<Vec<bool>> = preds
        .iter()
        .map(|pr| {
            (0..n)
                .map(|r| match *pr {
                    CatfPredicate::NumericLess { feat, threshold } => {
                        let v = x_cols[feat][r];
                        v.is_finite() && v < threshold
                    }
                    CatfPredicate::NominalIn { feat, mask } => {
                        nom_codes[feat][r].is_some_and(|c| c < 64 && ((mask >> c) & 1) == 1)
                    }
                    CatfPredicate::IsMissing { feat } => !x_cols[feat][r].is_finite(),
                })
                .collect()
        })
        .collect();

    // Initialize per-row logits to the exact Gaussian copula conditional anchor!
    let mut logits = vec![0.0; n * n_states];
    for r in 0..n {
        let mut mu = 0.0;
        for (p_i, &pf) in cond.predecessors.iter().enumerate() {
            let zv = z_cols[pf][r];
            if zv.is_finite() {
                mu += cond.copula_weights[p_i] * zv;
            }
        }
        copula_base_logits(
            &cond.z_edges,
            mu,
            cond.copula_cond_sd,
            cond.base_missing_prob,
            &mut logits[r * n_states..(r + 1) * n_states],
        );
    }

    let k_bins = cond.k_bins;
    let mut grad = vec![0.0; n * n_states];
    let mut hess = vec![0.0; n * n_states];
    let base_nll = mean_state_nll(&logits, target_states, n_states);
    let mut cur_nll = base_nll;

    // Optional held-out SELECT logits & states so trees are accepted ONLY on OOS NLL gain (`BUG-R17`).
    let (mut sel_logits, sel_states, mut cur_sel_nll) = if let Some((sx, sz, snc)) = select_eval {
        let ns = sx.first().map_or(0, Vec::len);
        let mut sl = vec![0.0; ns * n_states];
        let mut ss = vec![k_bins; ns];
        for r in 0..ns {
            let mut mu = 0.0;
            for (p_i, &pf) in cond.predecessors.iter().enumerate() {
                let zv = sz[pf][r];
                if zv.is_finite() {
                    mu += cond.copula_weights[p_i] * zv;
                }
            }
            copula_base_logits(
                &cond.z_edges,
                mu,
                cond.copula_cond_sd,
                cond.base_missing_prob,
                &mut sl[r * n_states..(r + 1) * n_states],
            );
            let zc = sz[cond.col_idx][r];
            if zc.is_finite() {
                ss[r] = cond.z_edges[1..k_bins]
                    .partition_point(|&e| e < zc)
                    .min(k_bins - 1);
            }
        }
        let s0 = mean_state_nll(&sl, &ss, n_states);
        (Some((sx, snc, sl)), ss, s0)
    } else {
        (None, Vec::new(), f64::INFINITY)
    };

    for _t in 0..max_trees {
        for r in 0..n {
            let row = &logits[r * n_states..(r + 1) * n_states];
            let lse = log_sum_exp(row);
            let ts = target_states[r];
            for s in 0..n_states {
                let p = (row[s] - lse).exp();
                grad[r * n_states + s] = p - if s == ts { 1.0 } else { 0.0 };
                hess[r * n_states + s] = (p * (1.0 - p)).max(1e-5);
            }
        }

        let mut chosen_preds: Vec<CatfPredicate> = Vec::with_capacity(depth);
        let mut row_leaf = vec![0usize; n];
        let mut l = 0usize;
        while l < depth {
            let leaves = 1usize << l;
            let best_single = best_split_gain(
                &pred_bits,
                &row_leaf,
                leaves,
                &grad,
                &hess,
                n_states,
                min_leaf,
                alpha_shrink,
            );
            // Depth-2 lookahead for XOR / 3-bit parity (`T16`) when a single split has tiny gain!
            let use_lookahead =
                l + 1 < depth && best_single.as_ref().is_none_or(|&(g, _)| g < 0.05);
            if use_lookahead {
                if let Some((g2, p1, p2)) = best_lookahead_pair(
                    &pred_bits,
                    &row_leaf,
                    leaves,
                    &grad,
                    &hess,
                    n_states,
                    min_leaf,
                    alpha_shrink,
                ) {
                    if g2 > best_single.map_or(0.0, |(g, _)| g) + 1e-4 {
                        for r in 0..n {
                            if pred_bits[p1][r] {
                                row_leaf[r] |= 1 << l;
                            }
                            if pred_bits[p2][r] {
                                row_leaf[r] |= 1 << (l + 1);
                            }
                        }
                        chosen_preds.push(preds[p1].clone());
                        chosen_preds.push(preds[p2].clone());
                        l += 2;
                        continue;
                    }
                }
            }
            let Some((gain, pi)) = best_single else {
                break;
            };
            if gain <= 1e-5 {
                break;
            }
            for r in 0..n {
                if pred_bits[pi][r] {
                    row_leaf[r] |= 1 << l;
                }
            }
            chosen_preds.push(preds[pi].clone());
            l += 1;
        }
        if chosen_preds.is_empty() {
            break;
        }
        let nl = 1usize << chosen_preds.len();
        let leaf_values = compute_leaf_tilts(
            &row_leaf,
            nl,
            &grad,
            &hess,
            n_states,
            &cond.z_centers,
            use_hermite,
            alpha_shrink,
            eta,
        );
        let tree = CatfTree {
            predicates: chosen_preds,
            hermite_mode: use_hermite,
            leaf_values,
            n_states,
        };
        let mut next_logits = logits.clone();
        for (r, &lf) in row_leaf.iter().enumerate().take(n) {
            apply_tree_leaf_row(
                &tree,
                lf,
                &cond.z_centers,
                k_bins,
                n_states,
                &mut next_logits[r * n_states..(r + 1) * n_states],
            );
        }
        let next_nll = mean_state_nll(&next_logits, target_states, n_states);
        let min_gain = if use_hermite {
            ((nl * 2) as f64 / (4.0 * n.max(1) as f64)).clamp(0.0005, 0.015)
        } else {
            1e-4
        };
        let mut accept = next_nll < cur_nll - min_gain;
        let mut next_sel_opt: Option<(Vec<f64>, f64)> = None;
        if accept {
            if let Some((sx, snc, ref sl)) = sel_logits {
                let (nsl, sn) = eval_tree_on_select(
                    &tree,
                    sx,
                    snc,
                    sl,
                    &sel_states,
                    &cond.z_centers,
                    k_bins,
                    n_states,
                );
                if sn < cur_sel_nll - 1e-4 {
                    next_sel_opt = Some((nsl, sn));
                } else {
                    accept = false;
                }
            }
        }
        if accept {
            logits = next_logits;
            cur_nll = next_nll;
            if let Some((nsl, sn)) = next_sel_opt {
                if let Some((_, _, ref mut sl)) = sel_logits {
                    *sl = nsl;
                }
                cur_sel_nll = sn;
            }
            cond.trees.push(tree);
        } else {
            break;
        }
    }
}

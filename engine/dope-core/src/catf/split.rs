//! CATF split scoring, typed predicates (`NominalIn` for `T05`), depth-2 lookahead (`T16`),
//! and Hermite/full-K leaf tilt computation.

use super::fit::{CatfPredicate, CatfTree};
use crate::target::log_sum_exp;

pub(crate) fn mean_state_nll(logits: &[f64], states: &[usize], n_states: usize) -> f64 {
    let n = states.len().max(1);
    let mut acc = 0.0;
    for (r, &s) in states.iter().enumerate() {
        let row = &logits[r * n_states..(r + 1) * n_states];
        acc += log_sum_exp(row) - row[s];
    }
    acc / n as f64
}

pub(crate) fn build_candidate_predicates(
    x_cols: &[Vec<f64>],
    is_nominal: &[bool],
    nom_codes: &[Vec<Option<u8>>],
    predecessors: &[usize],
    n: usize,
) -> Vec<CatfPredicate> {
    let mut out = Vec::new();
    for &f in predecessors {
        let miss_cnt = x_cols[f].iter().filter(|v| !v.is_finite()).count();
        if miss_cnt >= 5 && miss_cnt + 5 <= n {
            out.push(CatfPredicate::IsMissing { feat: f });
        }
        if is_nominal[f] {
            let mut present = 0u64;
            for c in nom_codes[f].iter().flatten() {
                if *c < 64 {
                    present |= 1u64 << c;
                }
            }
            for c in 0..64u8 {
                if ((present >> c) & 1) == 1 {
                    out.push(CatfPredicate::NominalIn {
                        feat: f,
                        mask: 1u64 << c,
                    });
                }
            }
        } else {
            let mut vals: Vec<f64> = x_cols[f]
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();
            vals.sort_by(f64::total_cmp);
            if vals.len() < 4 {
                continue;
            }
            let mut cuts = Vec::new();
            for q in 1..16 {
                let cut = vals[(q * (vals.len() - 1)) / 16];
                if cuts
                    .last()
                    .is_none_or(|&prev: &f64| (cut - prev).abs() > 1e-12)
                {
                    cuts.push(cut);
                }
            }
            for threshold in cuts {
                out.push(CatfPredicate::NumericLess { feat: f, threshold });
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn split_score_from_bits(
    bits: &[bool],
    row_leaf: &[usize],
    leaves: usize,
    grad: &[f64],
    hess: &[f64],
    n_states: usize,
    min_leaf: usize,
    lambda: f64,
) -> Option<f64> {
    let mut sg = vec![0.0; leaves * 2 * n_states];
    let mut sh = vec![0.0; leaves * 2 * n_states];
    let mut cnt = vec![0usize; leaves * 2];
    for (r, &lf) in row_leaf.iter().enumerate() {
        let slot = lf * 2 + usize::from(bits[r]);
        cnt[slot] += 1;
        for s in 0..n_states {
            sg[slot * n_states + s] += grad[r * n_states + s];
            sh[slot * n_states + s] += hess[r * n_states + s];
        }
    }
    let mut gain = 0.0;
    let mut valid_any = false;
    for lf in 0..leaves {
        let (c0, c1) = (cnt[lf * 2], cnt[lf * 2 + 1]);
        if c0 < min_leaf || c1 < min_leaf {
            continue;
        }
        valid_any = true;
        for s in 0..n_states {
            let (g0, h0) = (sg[lf * 2 * n_states + s], sh[lf * 2 * n_states + s]);
            let (g1, h1) = (
                sg[(lf * 2 + 1) * n_states + s],
                sh[(lf * 2 + 1) * n_states + s],
            );
            let gt = g0 + g1;
            let ht = h0 + h1;
            gain += g0 * g0 / (h0 + lambda) + g1 * g1 / (h1 + lambda) - gt * gt / (ht + lambda);
        }
    }
    valid_any.then_some(gain)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn best_split_gain(
    pred_bits: &[Vec<bool>],
    row_leaf: &[usize],
    leaves: usize,
    grad: &[f64],
    hess: &[f64],
    n_states: usize,
    min_leaf: usize,
    lambda: f64,
) -> Option<(f64, usize)> {
    let mut best: Option<(f64, usize)> = None;
    for (pi, bits) in pred_bits.iter().enumerate() {
        if let Some(g) = split_score_from_bits(
            bits, row_leaf, leaves, grad, hess, n_states, min_leaf, lambda,
        ) {
            if best.is_none_or(|(bg, _)| g > bg + 1e-12) {
                best = Some((g, pi));
            }
        }
    }
    best
}

/// Depth-2 simultaneous split lookahead cracking pure XOR and 3-bit parity (`T16`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn best_lookahead_pair(
    pred_bits: &[Vec<bool>],
    row_leaf: &[usize],
    leaves: usize,
    grad: &[f64],
    hess: &[f64],
    n_states: usize,
    min_leaf: usize,
    lambda: f64,
) -> Option<(f64, usize, usize)> {
    let m = pred_bits.len().min(24);
    let mut best: Option<(f64, usize, usize)> = None;
    for p1 in 0..m {
        for p2 in (p1 + 1)..m {
            let mut sg = vec![0.0; leaves * 4 * n_states];
            let mut sh = vec![0.0; leaves * 4 * n_states];
            let mut cnt = vec![0usize; leaves * 4];
            for (r, &lf) in row_leaf.iter().enumerate() {
                let q = usize::from(pred_bits[p1][r]) | (usize::from(pred_bits[p2][r]) << 1);
                let slot = lf * 4 + q;
                cnt[slot] += 1;
                for s in 0..n_states {
                    sg[slot * n_states + s] += grad[r * n_states + s];
                    sh[slot * n_states + s] += hess[r * n_states + s];
                }
            }
            let mut gain = 0.0;
            let mut ok = false;
            for lf in 0..leaves {
                if (0..4).any(|q| cnt[lf * 4 + q] < min_leaf) {
                    continue;
                }
                ok = true;
                for s in 0..n_states {
                    let mut gt = 0.0;
                    let mut ht = 0.0;
                    let mut sum_q = 0.0;
                    for q in 0..4 {
                        let gq = sg[(lf * 4 + q) * n_states + s];
                        let hq = sh[(lf * 4 + q) * n_states + s];
                        gt += gq;
                        ht += hq;
                        sum_q += gq * gq / (hq + lambda);
                    }
                    gain += sum_q - gt * gt / (ht + lambda);
                }
            }
            if ok && best.is_none_or(|(bg, _, _)| gain > bg + 1e-12) {
                best = Some((gain, p1, p2));
            }
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_leaf_tilts(
    row_leaf: &[usize],
    nl: usize,
    grad: &[f64],
    hess: &[f64],
    n_states: usize,
    z_centers: &[f64],
    hermite_mode: bool,
    alpha: f64,
    eta: f64,
) -> Vec<f64> {
    let mut sg = vec![0.0; nl * n_states];
    let mut sh = vec![0.0; nl * n_states];
    let mut cnt = vec![0usize; nl];
    for (r, &lf) in row_leaf.iter().enumerate() {
        cnt[lf] += 1;
        for s in 0..n_states {
            sg[lf * n_states + s] += grad[r * n_states + s];
            sh[lf * n_states + s] += hess[r * n_states + s];
        }
    }
    let k_bins = n_states - 1;
    if !hermite_mode {
        let mut vals = vec![0.0; nl * n_states];
        for lf in 0..nl {
            let shrink = cnt[lf] as f64 / (cnt[lf] as f64 + alpha);
            for s in 0..n_states {
                let raw = -eta * sg[lf * n_states + s] / (sh[lf * n_states + s] + alpha);
                vals[lf * n_states + s] = (raw * shrink).clamp(-3.0, 3.0);
            }
        }
        return vals;
    }
    let mut vals = vec![0.0; nl * 4];
    let basis = |b: usize| -> [f64; 3] {
        let z = z_centers[b];
        [z, z * z - 1.0, z * z * z - 3.0 * z]
    };
    let mut norm2 = [1e-6; 3];
    for b in 0..k_bins {
        let he = basis(b);
        for q in 0..3 {
            norm2[q] += he[q] * he[q];
        }
    }
    for lf in 0..nl {
        let shrink = cnt[lf] as f64 / (cnt[lf] as f64 + alpha);
        let mut step_k = vec![0.0; k_bins];
        for b in 0..k_bins {
            step_k[b] = (-eta * sg[lf * n_states + b] / (sh[lf * n_states + b] + alpha) * shrink)
                .clamp(-2.5, 2.5);
        }
        for q in 0..3 {
            let proj: f64 = (0..k_bins).map(|b| step_k[b] * basis(b)[q]).sum::<f64>() / norm2[q];
            vals[lf * 4 + q] = proj.clamp(-1.5, 1.5);
        }
        let miss_step =
            -eta * sg[lf * n_states + k_bins] / (sh[lf * n_states + k_bins] + alpha) * shrink;
        vals[lf * 4 + 3] = miss_step.clamp(-2.5, 2.5);
    }
    vals
}

pub(crate) fn apply_tree_leaf_row(
    tree: &CatfTree,
    lf: usize,
    z_centers: &[f64],
    k_bins: usize,
    n_states: usize,
    row_logits: &mut [f64],
) {
    if tree.hermite_mode {
        let base = lf * 4;
        let (c1, c2, c3, cm) = (
            tree.leaf_values[base],
            tree.leaf_values[base + 1],
            tree.leaf_values[base + 2],
            tree.leaf_values[base + 3],
        );
        for (b, &z) in z_centers.iter().enumerate().take(k_bins) {
            row_logits[b] += c1 * z + c2 * (z * z - 1.0) + c3 * (z * z * z - 3.0 * z);
        }
        row_logits[k_bins] += cm;
    } else {
        let base = lf * n_states;
        for s in 0..n_states {
            row_logits[s] += tree.leaf_values[base + s];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_tree_on_select(
    tree: &CatfTree,
    sx: &[Vec<f64>],
    snc: &[Vec<Option<u8>>],
    prev_logits: &[f64],
    sel_states: &[usize],
    z_centers: &[f64],
    k_bins: usize,
    n_states: usize,
) -> (Vec<f64>, f64) {
    let mut nsl = prev_logits.to_vec();
    for r in 0..sel_states.len() {
        let mut lf = 0usize;
        for (li, pr) in tree.predicates.iter().enumerate() {
            let bit = match *pr {
                CatfPredicate::NumericLess { feat, threshold } => {
                    sx[feat][r].is_finite() && sx[feat][r] < threshold
                }
                CatfPredicate::NominalIn { feat, mask } => {
                    snc[feat][r].is_some_and(|c| c < 64 && ((mask >> c) & 1) == 1)
                }
                CatfPredicate::IsMissing { feat } => !sx[feat][r].is_finite(),
            };
            if bit {
                lf |= 1 << li;
            }
        }
        apply_tree_leaf_row(
            tree,
            lf,
            z_centers,
            k_bins,
            n_states,
            &mut nsl[r * n_states..(r + 1) * n_states],
        );
    }
    let nll = mean_state_nll(&nsl, sel_states, n_states);
    (nsl, nll)
}

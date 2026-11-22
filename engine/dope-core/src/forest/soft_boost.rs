//! Weighted soft-target Newton boosting for teacher-distilled oblivious vector-leaf forests.
//!
//! Fits directly to calibrated continuous teacher expectations or full class-probability
//! vectors (`grad = w * (p - t)`, `hess = w * p * (1 - p)`), stopping on the forest byte cap
//! or probe-set plateau rather than truncating on noisy real validation labels (`G5`, `T09`, `T10`).

use super::{Forest, LEVELS};
use crate::rng::Rng;
use crate::target::{log_sum_exp, Head};
use half::f16;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct SoftBoostOptions {
    pub depth: usize,
    pub bits: u8,
    pub eta: f64,
    pub lambda: f64,
    pub max_trees: usize,
    /// Optional byte cap for the serialized forest payload (`Forest::byte_len`).
    pub forest_byte_cap: Option<usize>,
    /// Stop if relative probe loss improvement over `patience` trees is `< rel_tol`.
    pub patience: usize,
    pub rel_tol: f64,
    pub min_weight: f64,
    pub sample_rows: usize,
    pub seed: u64,
}

impl Default for SoftBoostOptions {
    fn default() -> Self {
        Self {
            depth: 6,
            bits: 8,
            eta: 0.15,
            lambda: 1.0,
            max_trees: 2000,
            forest_byte_cap: None,
            patience: 30,
            rel_tol: 1e-4,
            min_weight: 5.0,
            sample_rows: 24_000,
            seed: 0,
        }
    }
}

/// Column-major input codes with continuous/probability soft targets `[n * o]` and row weights `[n]`.
pub struct SoftSplit<'a> {
    pub codes: &'a [u8],
    pub base: &'a [f64],
    pub soft_targets: &'a [f64],
    pub weights: &'a [f64],
}

pub struct SoftBoosted {
    pub forest: Forest,
    pub probe_path: Vec<f64>,
}

/// Weighted soft-target loss (MSE for regression; cross-entropy for binary/multiclass).
pub fn soft_loss(head: Head, f: &[f64], targets: &[f64], weights: &[f64]) -> f64 {
    let n = weights.len();
    if n == 0 {
        return 0.0;
    }
    let o = head.outputs();
    let mut sum_w = 0.0;
    let mut acc = 0.0;
    match head {
        Head::Regression => {
            for r in 0..n {
                let w = weights[r].max(0.0);
                let d = f[r] - targets[r];
                acc += w * d * d;
                sum_w += w;
            }
        }
        Head::Binary => {
            for r in 0..n {
                let w = weights[r].max(0.0);
                let t = targets[r].clamp(0.0, 1.0);
                let z = f[r];
                // Stable soft binary cross-entropy: log(1 + exp(z)) - t * z.
                let ce = if z >= 0.0 {
                    z + (-z).exp().ln_1p()
                } else {
                    z.exp().ln_1p()
                } - t * z;
                acc += w * ce;
                sum_w += w;
            }
        }
        Head::Multiclass(k) => {
            for r in 0..n {
                let w = weights[r].max(0.0);
                let row = &f[r * o..(r + 1) * o];
                let lse = log_sum_exp(row);
                let mut ce = 0.0;
                for c in 0..k {
                    let tc = targets[r * o + c].clamp(0.0, 1.0);
                    if tc > 0.0 {
                        ce -= tc * (row[c] - lse);
                    }
                }
                acc += w * ce;
                sum_w += w;
            }
        }
    }
    if sum_w > 0.0 {
        acc / sum_w
    } else {
        0.0
    }
}

fn soft_gradients(
    head: Head,
    f: &[f64],
    targets: &[f64],
    weights: &[f64],
    g: &mut [f64],
    h: &mut [f64],
) {
    let n = weights.len();
    let o = head.outputs();
    match head {
        Head::Regression => {
            for r in 0..n {
                let w = weights[r].max(0.0);
                g[r] = w * (f[r] - targets[r]);
                h[r] = w;
            }
        }
        Head::Binary => {
            for r in 0..n {
                let w = weights[r].max(0.0);
                let p = 1.0 / (1.0 + (-f[r]).exp());
                g[r] = w * (p - targets[r].clamp(0.0, 1.0));
                h[r] = w * (p * (1.0 - p)).max(1e-6);
            }
        }
        Head::Multiclass(k) => {
            for r in 0..n {
                let w = weights[r].max(0.0);
                let row = &f[r * o..(r + 1) * o];
                let lse = log_sum_exp(row);
                for c in 0..k {
                    let p = (row[c] - lse).exp();
                    let tc = targets[r * o + c].clamp(0.0, 1.0);
                    g[r * o + c] = w * (p - tc);
                    h[r * o + c] = w * (p * (1.0 - p)).max(1e-6);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn best_weighted_threshold(
    col: &[u8],
    rows: &[u32],
    leaf: &[u16],
    leaves: usize,
    g: &[f64],
    h: &[f64],
    weights: &[f64],
    o: usize,
    lambda: f64,
    min_weight: f64,
) -> Option<(f64, u8)> {
    let w_stride = 2 * o + 1;
    let mut hist = vec![0.0; leaves * LEVELS * w_stride];
    for (i, &r) in rows.iter().enumerate() {
        let r = r as usize;
        let base = (leaf[i] as usize * LEVELS + col[r] as usize) * w_stride;
        for k in 0..o {
            hist[base + k] += g[r * o + k];
            hist[base + o + k] += h[r * o + k];
        }
        hist[base + 2 * o] += weights[r].max(0.0);
    }
    let mut gain = vec![0.0; LEVELS - 1];
    let mut left_w = vec![0.0; LEVELS - 1];
    let mut total_w = 0.0;
    let mut acc = vec![0.0; w_stride];
    for l in 0..leaves {
        let slab = &hist[l * LEVELS * w_stride..(l + 1) * LEVELS * w_stride];
        let mut tot = vec![0.0; w_stride];
        for b in 0..LEVELS {
            for j in 0..w_stride {
                tot[j] += slab[b * w_stride + j];
            }
        }
        total_w += tot[2 * o];
        acc.fill(0.0);
        for c in 0..LEVELS - 1 {
            for j in 0..w_stride {
                acc[j] += slab[c * w_stride + j];
            }
            let mut s = 0.0;
            for k in 0..o {
                let (gl, hl) = (acc[k], acc[o + k]);
                let (gr, hr) = (tot[k] - gl, tot[o + k] - hl);
                s += gl * gl / (hl + lambda) + gr * gr / (hr + lambda);
            }
            gain[c] += s;
            left_w[c] += acc[2 * o];
        }
    }
    (0..LEVELS - 1)
        .filter(|&c| left_w[c] >= min_weight && total_w - left_w[c] >= min_weight)
        .map(|c| (gain[c], c as u8))
        .max_by(|a, b| a.0.total_cmp(&b.0))
}

/// Fit an oblivious vector-leaf forest to weighted soft teacher targets (`T09`, `T10`).
pub fn boost_soft(
    head: Head,
    d: usize,
    train: &SoftSplit,
    probe: &SoftSplit,
    opts: &SoftBoostOptions,
) -> SoftBoosted {
    let o = head.outputs();
    let n = train.weights.len();
    let np = probe.weights.len();
    let depth = opts.depth.clamp(1, 8);
    let bits = opts.bits.clamp(2, 8);
    let nl = 1usize << depth;
    let mut forest = Forest {
        depth,
        outputs: o,
        bits,
        ..Forest::empty(o)
    };
    let mut f = train.base.to_vec();
    let mut fp = probe.base.to_vec();
    let mut probe_path = vec![soft_loss(head, &fp, probe.soft_targets, probe.weights)];
    if d == 0 || n < 2 {
        return SoftBoosted { forest, probe_path };
    }
    let max_trees = match opts.forest_byte_cap {
        Some(cap) => {
            let mut fit_t = 0usize;
            while fit_t < opts.max_trees && Forest::byte_len(fit_t + 1, depth, o, bits, d) <= cap {
                fit_t += 1;
            }
            fit_t
        }
        None => opts.max_trees,
    };
    let (mut g, mut h) = (vec![0.0; n * o], vec![0.0; n * o]);
    let mut rng = Rng::new(opts.seed ^ 0x50f7_b005);
    let mut all: Vec<u32> = (0..n as u32).collect();
    let levels = ((1i32 << (bits - 1)) - 1).max(1);
    let mut best = probe_path[0];
    let mut best_t = 0usize;
    let mut row_leaf = vec![0usize; n];

    for t in 0..max_trees {
        soft_gradients(head, &f, train.soft_targets, train.weights, &mut g, &mut h);
        let rows: Vec<u32> = if n > opts.sample_rows {
            rng.shuffle(&mut all);
            let mut s = all[..opts.sample_rows].to_vec();
            s.sort_unstable();
            s
        } else {
            all.clone()
        };
        let mut leaf = vec![0u16; rows.len()];
        for l in 0..depth {
            let leaves = 1usize << l;
            let pick = (0..d)
                .into_par_iter()
                .filter_map(|feat| {
                    best_weighted_threshold(
                        &train.codes[feat * n..(feat + 1) * n],
                        &rows,
                        &leaf,
                        leaves,
                        &g,
                        &h,
                        train.weights,
                        o,
                        opts.lambda,
                        opts.min_weight,
                    )
                    .map(|(gain, c)| (gain, feat, c))
                })
                .max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)));
            let (feat, c) = pick.map_or((0, 255u8), |(_, feat, c)| (feat, c));
            forest.features.push(feat);
            forest.thresholds.push(c);
            let col = &train.codes[feat * n..(feat + 1) * n];
            for (i, &r) in rows.iter().enumerate() {
                if col[r as usize] > c {
                    leaf[i] |= 1 << l;
                }
            }
        }
        let k0 = t * depth;
        let leaf_of = |codes: &[u8], rows_n: usize, r: usize| -> usize {
            let mut idx = 0;
            for l in 0..depth {
                if codes[forest.features[k0 + l] * rows_n + r] > forest.thresholds[k0 + l] {
                    idx |= 1 << l;
                }
            }
            idx
        };
        let mut sg = vec![0.0; nl * o];
        let mut sh = vec![0.0; nl * o];
        for r in 0..n {
            let lf = leaf_of(train.codes, n, r);
            row_leaf[r] = lf;
            for k in 0..o {
                sg[lf * o + k] += g[r * o + k];
                sh[lf * o + k] += h[r * o + k];
            }
        }
        let values: Vec<f64> = (0..nl * o)
            .map(|i| (-opts.eta * sg[i] / (sh[i] + opts.lambda)).clamp(-2.0, 2.0))
            .collect();
        let maxabs = values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let scale = f16::from_f64((maxabs / levels as f64).max(1e-7)).to_f64();
        let qcodes: Vec<i32> = values
            .iter()
            .map(|v| ((v / scale).round() as i32).clamp(-levels, levels))
            .collect();
        for r in 0..n {
            for k in 0..o {
                f[r * o + k] += qcodes[row_leaf[r] * o + k] as f64 * scale;
            }
        }
        for r in 0..np {
            let lf = leaf_of(probe.codes, np, r);
            for k in 0..o {
                fp[r * o + k] += qcodes[lf * o + k] as f64 * scale;
            }
        }
        forest.scales.push(scale);
        forest.codes.extend(qcodes);
        let pl = soft_loss(head, &fp, probe.soft_targets, probe.weights);
        probe_path.push(pl);
        if pl < best * (1.0 - opts.rel_tol) - 1e-12 {
            best = pl;
            best_t = t + 1;
        } else if t + 1 - best_t >= opts.patience {
            break;
        }
    }
    // Truncate back to `best_t` so post-plateau diverging trees are never retained (BUG-R1).
    let forest = forest.truncate(best_t);
    SoftBoosted { forest, probe_path }
}

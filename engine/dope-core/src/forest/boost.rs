//! Newton boosting of oblivious trees on byte-coded inputs, quantizing each
//! tree as it is fitted (error feedback).

use crate::rng::Rng;
use crate::target::{log_sum_exp, loss, Head};
use half::f16;
use rayon::prelude::*;

use super::{Forest, LEVELS};

#[derive(Clone, Debug)]
pub struct BoostOptions {
    pub depth: usize,
    pub bits: u8,
    /// Shrinkage applied to every Newton step.
    pub eta: f64,
    /// L2 regularization of leaf values (Hessian units).
    pub lambda: f64,
    pub max_trees: usize,
    /// Stop after this many trees without a validation improvement.
    pub patience: usize,
    /// Rows required on each side of a split (summed over leaves).
    pub min_rows: usize,
    /// Rows used to choose each tree's splits (leaf values use every row).
    pub sample_rows: usize,
    pub seed: u64,
}

impl Default for BoostOptions {
    fn default() -> Self {
        Self {
            depth: 4,
            bits: 6,
            eta: 0.2,
            lambda: 1.0,
            max_trees: 400,
            patience: 40,
            min_rows: 20,
            sample_rows: 20_000,
            seed: 0,
        }
    }
}

/// A boosted forest and its validation loss after 0, 1, …, T trees.
pub struct Boosted {
    pub forest: Forest,
    pub val_path: Vec<f64>,
}

/// Rows of column-major input codes (`codes[f * n + r]`) with base outputs.
pub struct Split<'a> {
    pub codes: &'a [u8],
    pub base: &'a [f64],
    pub y: &'a [f64],
}

fn gradients(head: Head, f: &[f64], y: &[f64], g: &mut [f64], h: &mut [f64]) {
    let o = head.outputs();
    match head {
        Head::Regression => {
            for r in 0..y.len() {
                g[r] = f[r] - y[r];
                h[r] = 1.0;
            }
        }
        Head::Binary => {
            for r in 0..y.len() {
                let p = 1.0 / (1.0 + (-f[r]).exp());
                g[r] = p - y[r];
                h[r] = (p * (1.0 - p)).max(1e-6);
            }
        }
        Head::Multiclass(k) => {
            for r in 0..y.len() {
                let row = &f[r * o..(r + 1) * o];
                let lse = log_sum_exp(row);
                let t = (y[r] as usize).min(k - 1);
                for c in 0..k {
                    let p = (row[c] - lse).exp();
                    g[r * o + c] = p - if c == t { 1.0 } else { 0.0 };
                    h[r * o + c] = (p * (1.0 - p)).max(1e-6);
                }
            }
        }
    }
}

/// Best oblivious split of the current leaves on one feature: (gain, threshold).
#[allow(clippy::too_many_arguments)]
fn best_threshold(
    col: &[u8],
    rows: &[u32],
    leaf: &[u16],
    leaves: usize,
    g: &[f64],
    h: &[f64],
    o: usize,
    lambda: f64,
    min_rows: usize,
) -> Option<(f64, u8)> {
    let w = 2 * o + 1;
    let mut hist = vec![0.0; leaves * LEVELS * w];
    for (i, &r) in rows.iter().enumerate() {
        let r = r as usize;
        let base = (leaf[i] as usize * LEVELS + col[r] as usize) * w;
        for k in 0..o {
            hist[base + k] += g[r * o + k];
            hist[base + o + k] += h[r * o + k];
        }
        hist[base + 2 * o] += 1.0;
    }
    let mut gain = vec![0.0; LEVELS - 1];
    let mut left_rows = vec![0.0; LEVELS - 1];
    let mut total_rows = 0.0;
    let mut acc = vec![0.0; w];
    for l in 0..leaves {
        let slab = &hist[l * LEVELS * w..(l + 1) * LEVELS * w];
        let mut tot = vec![0.0; w];
        for b in 0..LEVELS {
            for j in 0..w {
                tot[j] += slab[b * w + j];
            }
        }
        total_rows += tot[2 * o];
        acc.iter_mut().for_each(|v| *v = 0.0);
        for c in 0..LEVELS - 1 {
            for j in 0..w {
                acc[j] += slab[c * w + j];
            }
            let mut s = 0.0;
            for k in 0..o {
                let (gl, hl) = (acc[k], acc[o + k]);
                let (gr, hr) = (tot[k] - gl, tot[o + k] - hl);
                s += gl * gl / (hl + lambda) + gr * gr / (hr + lambda);
            }
            gain[c] += s;
            left_rows[c] += acc[2 * o];
        }
    }
    let min = min_rows as f64;
    (0..LEVELS - 1)
        .filter(|&c| left_rows[c] >= min && total_rows - left_rows[c] >= min)
        .map(|c| (gain[c], c as u8))
        .max_by(|a, b| a.0.total_cmp(&b.0))
}

/// Boost a forest on top of base outputs, quantizing each tree as it is fitted.
pub fn boost(head: Head, d: usize, fit: &Split, val: &Split, opts: &BoostOptions) -> Boosted {
    let o = head.outputs();
    let (n, nv) = (fit.y.len(), val.y.len());
    let depth = opts.depth.clamp(1, 8);
    let nl = 1usize << depth;
    let mut forest = Forest {
        depth,
        outputs: o,
        bits: opts.bits,
        ..Forest::empty(o)
    };
    let (mut f, mut fv) = (fit.base.to_vec(), val.base.to_vec());
    let mut val_path = vec![loss(head, &fv, val.y)];
    if d == 0 || n < 2 * opts.min_rows {
        return Boosted { forest, val_path };
    }
    let (mut g, mut h) = (vec![0.0; n * o], vec![0.0; n * o]);
    let mut rng = Rng::new(opts.seed ^ 0xf0_2e57);
    let mut all: Vec<u32> = (0..n as u32).collect();
    let levels = ((1i32 << (opts.bits - 1)) - 1).max(1);
    let (mut best, mut best_t) = (val_path[0], 0usize);
    let mut row_leaf = vec![0usize; n];
    let mut val_leaf = vec![0usize; nv];
    for t in 0..opts.max_trees {
        gradients(head, &f, fit.y, &mut g, &mut h);
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
                    best_threshold(
                        &fit.codes[feat * n..(feat + 1) * n],
                        &rows,
                        &leaf,
                        leaves,
                        &g,
                        &h,
                        o,
                        opts.lambda,
                        opts.min_rows,
                    )
                    .map(|(gain, c)| (gain, feat, c))
                })
                .max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)));
            // No admissible split: a split that sends every row left.
            let (feat, c) = pick.map_or((0, 255u8), |(_, feat, c)| (feat, c));
            forest.features.push(feat);
            forest.thresholds.push(c);
            let col = &fit.codes[feat * n..(feat + 1) * n];
            for (i, &r) in rows.iter().enumerate() {
                if col[r as usize] > c {
                    leaf[i] |= 1 << l;
                }
            }
        }
        // Newton leaf values on every fitting row.
        let k0 = t * depth;
        let leaf_of = |codes: &[u8], rows: usize, r: usize| -> usize {
            let mut i = 0;
            for l in 0..depth {
                if codes[forest.features[k0 + l] * rows + r] > forest.thresholds[k0 + l] {
                    i |= 1 << l;
                }
            }
            i
        };
        let mut sg = vec![0.0; nl * o];
        let mut sh = vec![0.0; nl * o];
        for r in 0..n {
            let lf = leaf_of(fit.codes, n, r);
            row_leaf[r] = lf;
            for k in 0..o {
                sg[lf * o + k] += g[r * o + k];
                sh[lf * o + k] += h[r * o + k];
            }
        }
        let values: Vec<f64> = (0..nl * o)
            .map(|i| -opts.eta * sg[i] / (sh[i] + opts.lambda))
            .collect();
        let maxabs = values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let scale = f16::from_f64((maxabs / levels as f64).max(1e-7)).to_f64();
        let codes: Vec<i32> = values
            .iter()
            .map(|v| ((v / scale).round() as i32).clamp(-levels, levels))
            .collect();
        for r in 0..n {
            for k in 0..o {
                f[r * o + k] += codes[row_leaf[r] * o + k] as f64 * scale;
            }
        }
        for r in 0..nv {
            val_leaf[r] = leaf_of(val.codes, nv, r);
            for k in 0..o {
                fv[r * o + k] += codes[val_leaf[r] * o + k] as f64 * scale;
            }
        }
        forest.scales.push(scale);
        forest.codes.extend(codes);
        let vl = loss(head, &fv, val.y);
        val_path.push(vl);
        if vl < best - 1e-12 {
            best = vl;
            best_t = t + 1;
        } else if t + 1 - best_t >= opts.patience {
            break;
        }
    }
    Boosted { forest, val_path }
}

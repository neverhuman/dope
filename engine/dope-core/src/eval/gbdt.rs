//! Histogram gradient-boosted trees (the boosted-tree auditor family).
//!
//! Second-order boosting with depth-wise trees, 64 quantile bins per
//! feature, L2 leaf regularization and early stopping on a 10% split of the
//! auditor's own training rows. Squared loss, logistic or softmax.

use super::Matrix;
use crate::rng::Rng;
use crate::target::Head;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Params {
    pub rounds: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_leaf: usize,
    pub lambda: f64,
    pub bins: usize,
    pub patience: usize,
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            rounds: 400,
            learning_rate: 0.08,
            max_depth: 6,
            min_leaf: 20,
            lambda: 1.0,
            bins: 64,
            patience: 30,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug)]
enum Node {
    Split {
        feature: usize,
        bin: u8,
        left: usize,
        right: usize,
    },
    Leaf(f64),
}

struct Binned {
    rows: usize,
    cols: usize,
    data: Vec<u8>, // row-major
}

fn edges(x: &Matrix, rows: &[usize], bins: usize) -> Vec<Vec<f64>> {
    (0..x.cols)
        .into_par_iter()
        .map(|c| {
            let mut v: Vec<f64> = rows.iter().map(|&r| x.at(r, c)).collect();
            v.sort_by(f64::total_cmp);
            v.dedup();
            if v.len() <= bins {
                return v.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
            }
            let mut e: Vec<f64> = (1..bins).map(|k| v[k * v.len() / bins]).collect();
            e.dedup();
            e
        })
        .collect()
}

fn bin(x: &Matrix, rows: &[usize], edges: &[Vec<f64>]) -> Binned {
    let cols = x.cols;
    let mut data = vec![0u8; rows.len() * cols];
    data.par_chunks_mut(cols.max(1))
        .enumerate()
        .for_each(|(i, out)| {
            let r = rows[i];
            for c in 0..cols {
                out[c] = edges[c].partition_point(|e| *e < x.at(r, c)) as u8;
            }
        });
    Binned {
        rows: rows.len(),
        cols,
        data,
    }
}

fn grow(b: &Binned, g: &[f64], h: &[f64], p: &Params) -> Vec<Node> {
    let mut nodes = vec![Node::Leaf(0.0)];
    let mut stack = vec![(0usize, (0..b.rows).collect::<Vec<usize>>(), 0usize)];
    while let Some((id, rows, depth)) = stack.pop() {
        let (gs, hs): (f64, f64) = rows
            .iter()
            .fold((0.0, 0.0), |a, &r| (a.0 + g[r], a.1 + h[r]));
        let leaf = -gs / (hs + p.lambda) * p.learning_rate;
        if depth >= p.max_depth || rows.len() < 2 * p.min_leaf {
            nodes[id] = Node::Leaf(leaf);
            continue;
        }
        let parent = gs * gs / (hs + p.lambda);
        let best = (0..b.cols)
            .into_par_iter()
            .map(|c| {
                let mut hist = vec![(0.0f64, 0.0f64, 0usize); 256];
                for &r in &rows {
                    let e = &mut hist[b.data[r * b.cols + c] as usize];
                    e.0 += g[r];
                    e.1 += h[r];
                    e.2 += 1;
                }
                let (mut gl, mut hl, mut nl) = (0.0, 0.0, 0usize);
                let mut best = (0.0f64, 0u8);
                for k in 0..255usize {
                    gl += hist[k].0;
                    hl += hist[k].1;
                    nl += hist[k].2;
                    let nr = rows.len() - nl;
                    if nl < p.min_leaf || nr < p.min_leaf {
                        continue;
                    }
                    let (gr, hr) = (gs - gl, hs - hl);
                    let gain = gl * gl / (hl + p.lambda) + gr * gr / (hr + p.lambda) - parent;
                    if gain > best.0 {
                        best = (gain, k as u8);
                    }
                }
                (best.0, c, best.1)
            })
            .reduce(
                || (0.0, 0, 0),
                |a, b| {
                    if b.0 > a.0 || (b.0 == a.0 && b.1 < a.1 && b.0 > 0.0) {
                        b
                    } else {
                        a
                    }
                },
            );
        if best.0 <= 1e-12 {
            nodes[id] = Node::Leaf(leaf);
            continue;
        }
        let (_, feature, bin) = best;
        let (left, right): (Vec<usize>, Vec<usize>) = rows
            .iter()
            .partition(|&&r| b.data[r * b.cols + feature] <= bin);
        let (li, ri) = (nodes.len(), nodes.len() + 1);
        nodes.push(Node::Leaf(0.0));
        nodes.push(Node::Leaf(0.0));
        nodes[id] = Node::Split {
            feature,
            bin,
            left: li,
            right: ri,
        };
        stack.push((li, left, depth + 1));
        stack.push((ri, right, depth + 1));
    }
    nodes
}

fn predict_tree(nodes: &[Node], row: &[u8]) -> f64 {
    let mut i = 0;
    loop {
        match &nodes[i] {
            Node::Leaf(v) => return *v,
            Node::Split {
                feature,
                bin,
                left,
                right,
            } => i = if row[*feature] <= *bin { *left } else { *right },
        }
    }
}

fn softmax_row(z: &[f64]) -> Vec<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// Fit on `(x, y)` and return predictions for `test`: values (regression),
/// P(class 1) (binary) or row-major class probabilities (multiclass).
pub fn fit_predict(x: &Matrix, y: &[f64], head: Head, test: &Matrix, p: &Params) -> Vec<f64> {
    let o = head.outputs();
    let mut idx: Vec<usize> = (0..x.rows).collect();
    Rng::new(p.seed ^ 0x6bd7).shuffle(&mut idx);
    let n_val = if x.rows >= 200 { x.rows / 10 } else { 0 };
    let (val, fit) = idx.split_at(n_val);
    let e = edges(x, fit, p.bins);
    let bf = bin(x, fit, &e);
    let bv = bin(x, val, &e);
    let all_test: Vec<usize> = (0..test.rows).collect();
    let bt = bin(test, &all_test, &e);
    let yf: Vec<f64> = fit.iter().map(|&r| y[r]).collect();
    let yv: Vec<f64> = val.iter().map(|&r| y[r]).collect();
    let n = fit.len();
    let init: Vec<f64> = match head {
        Head::Regression => vec![yf.iter().sum::<f64>() / n.max(1) as f64],
        Head::Binary => {
            let q = (yf.iter().sum::<f64>() / n.max(1) as f64).clamp(1e-4, 1.0 - 1e-4);
            vec![(q / (1.0 - q)).ln()]
        }
        Head::Multiclass(k) => {
            let mut c = vec![1.0; k];
            yf.iter().for_each(|t| c[(*t as usize).min(k - 1)] += 1.0);
            let s: f64 = c.iter().sum();
            c.into_iter().map(|v| (v / s).ln()).collect()
        }
    };
    let mut sf: Vec<f64> = (0..n * o).map(|i| init[i % o]).collect();
    let mut sv: Vec<f64> = (0..val.len() * o).map(|i| init[i % o]).collect();
    let mut trees: Vec<Vec<Vec<Node>>> = Vec::new();
    let (mut best_loss, mut best_round, mut since) = (f64::INFINITY, 0usize, 0usize);
    let val_loss = |s: &[f64]| -> f64 { super::loss_from_scores(head, s, &yv) };
    for round in 0..p.rounds {
        let mut round_trees = Vec::with_capacity(o);
        let probs: Vec<Vec<f64>> = match head {
            Head::Multiclass(_) => (0..n)
                .map(|r| softmax_row(&sf[r * o..(r + 1) * o]))
                .collect(),
            _ => Vec::new(),
        };
        for k in 0..o {
            let (g, h): (Vec<f64>, Vec<f64>) = (0..n)
                .map(|r| match head {
                    Head::Regression => (sf[r] - yf[r], 1.0),
                    Head::Binary => {
                        let q = 1.0 / (1.0 + (-sf[r]).exp());
                        (q - yf[r], (q * (1.0 - q)).max(1e-6))
                    }
                    Head::Multiclass(_) => {
                        let q = probs[r][k];
                        let t = ((yf[r] as usize) == k) as u8 as f64;
                        (q - t, (q * (1.0 - q)).max(1e-6))
                    }
                })
                .unzip();
            let tree = grow(&bf, &g, &h, p);
            for r in 0..n {
                sf[r * o + k] += predict_tree(&tree, &bf.data[r * bf.cols..(r + 1) * bf.cols]);
            }
            for r in 0..val.len() {
                sv[r * o + k] += predict_tree(&tree, &bv.data[r * bv.cols..(r + 1) * bv.cols]);
            }
            round_trees.push(tree);
        }
        trees.push(round_trees);
        if !val.is_empty() {
            let l = val_loss(&sv);
            if l < best_loss - 1e-9 {
                best_loss = l;
                best_round = round + 1;
                since = 0;
            } else {
                since += 1;
                if since >= p.patience {
                    break;
                }
            }
        } else {
            best_round = round + 1;
        }
    }
    trees.truncate(best_round.max(1));
    let mut out = Vec::with_capacity(test.rows * o);
    for r in 0..test.rows {
        let row = &bt.data[r * bt.cols..(r + 1) * bt.cols];
        let mut z: Vec<f64> = init.clone();
        for rt in &trees {
            for (k, t) in rt.iter().enumerate() {
                z[k] += predict_tree(t, row);
            }
        }
        match head {
            Head::Regression => out.push(z[0]),
            Head::Binary => out.push(1.0 / (1.0 + (-z[0]).exp())),
            Head::Multiclass(_) => out.extend(softmax_row(&z)),
        }
    }
    out
}

//! Boosted oblivious trees: the sharp part of the conditional model.
//!
//! The residual network captures smooth structure in a few hundred bytes.
//! What it misses (thresholds, interactions among categorical levels) is
//! exactly what tree learners exploit first, so DOPE adds a forest of
//! *oblivious* trees (one feature and one threshold per level, as in
//! CatBoost) boosted by Newton steps on the loss of the quantized network:
//!
//! ```text
//! out(x) = net(x) + Σ_t leaf_t[ Σ_l 2^l · 1{code(x_{f_tl}) > c_tl} ]
//! ```
//!
//! Inputs are model-space values coded on 256 levels (`code`), so a split is
//! one feature index and one byte. Each tree is quantized as soon as it is
//! fitted (b-bit leaf codes, one half-precision scale), and the next tree is
//! fitted to the gradient of the *quantized* ensemble, so rounding error is
//! corrected by later trees (error feedback). Trees form a prefix that is cut
//! to the byte budget, which is how larger budgets buy more retention.

use crate::quant::{pack, packed_len, unpack};
use half::f16;

/// Number of input code levels (one byte per threshold).
pub const LEVELS: usize = 256;

/// Model-space value → input code. Callers impute missing values first.
#[inline]
pub fn code(x: f64) -> u8 {
    (x * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8
}

#[derive(Clone, Debug, PartialEq)]
pub struct Forest {
    pub depth: usize,
    pub outputs: usize,
    /// Leaf code width (symmetric codes).
    pub bits: u8,
    /// Split feature per (tree, level).
    pub features: Vec<usize>,
    /// Split threshold code per (tree, level): go right when `code > threshold`.
    pub thresholds: Vec<u8>,
    /// Per-tree leaf scale (exactly representable in f16).
    pub scales: Vec<f64>,
    /// Leaf codes, `[tree][leaf][output]`.
    pub codes: Vec<i32>,
}

impl Forest {
    pub fn empty(outputs: usize) -> Self {
        Self {
            depth: 0,
            outputs,
            bits: 8,
            features: Vec::new(),
            thresholds: Vec::new(),
            scales: Vec::new(),
            codes: Vec::new(),
        }
    }

    pub fn trees(&self) -> usize {
        self.scales.len()
    }

    pub fn leaves(&self) -> usize {
        1 << self.depth
    }

    /// Leaf of tree `t` for one row of input codes (indexed by feature).
    #[inline]
    pub fn leaf(&self, t: usize, codes: &[u8]) -> usize {
        let mut i = 0;
        for l in 0..self.depth {
            let k = t * self.depth + l;
            if codes[self.features[k]] > self.thresholds[k] {
                i |= 1 << l;
            }
        }
        i
    }

    #[inline]
    pub fn value(&self, t: usize, leaf: usize, k: usize) -> f64 {
        self.codes[(t * self.leaves() + leaf) * self.outputs + k] as f64 * self.scales[t]
    }

    /// Add the forest's output for one row of input codes to `out` (`o` values).
    #[inline]
    pub fn add(&self, codes: &[u8], out: &mut [f64]) {
        for t in 0..self.trees() {
            let leaf = self.leaf(t, codes);
            for (k, o) in out.iter_mut().enumerate().take(self.outputs) {
                *o += self.value(t, leaf, k);
            }
        }
    }

    /// The first `trees` trees.
    pub fn truncate(&self, trees: usize) -> Self {
        let trees = trees.min(self.trees());
        if trees == 0 {
            return Self::empty(self.outputs);
        }
        let per = self.leaves() * self.outputs;
        Self {
            depth: self.depth,
            outputs: self.outputs,
            bits: self.bits,
            features: self.features[..trees * self.depth].to_vec(),
            thresholds: self.thresholds[..trees * self.depth].to_vec(),
            scales: self.scales[..trees].to_vec(),
            codes: self.codes[..trees * per].to_vec(),
        }
    }

    /// Payload of `to_bytes` for a forest of this shape over `d` features.
    pub fn byte_len(trees: usize, depth: usize, outputs: usize, bits: u8, d: usize) -> usize {
        let idx = if d <= 256 { 1 } else { 2 };
        trees * depth * (idx + 1) + 2 * trees + packed_len(trees * (1 << depth) * outputs, bits)
    }

    /// One `U8` blob: feature indices (`u8`, or `u16` LE when d > 256),
    /// threshold codes, f16 scales, packed leaf codes.
    pub fn to_bytes(&self, d: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::byte_len(
            self.trees(),
            self.depth,
            self.outputs,
            self.bits,
            d,
        ));
        for &f in &self.features {
            if d <= 256 {
                out.push(f as u8);
            } else {
                out.extend((f as u16).to_le_bytes());
            }
        }
        out.extend(&self.thresholds);
        for &s in &self.scales {
            out.extend(f16::from_f64(s).to_le_bytes());
        }
        let offset = 1i32 << (self.bits - 1);
        let unsigned: Vec<u32> = self.codes.iter().map(|&c| (c + offset) as u32).collect();
        out.extend(pack(&unsigned, self.bits));
        out
    }

    pub fn from_bytes(
        b: &[u8],
        d: usize,
        trees: usize,
        depth: usize,
        outputs: usize,
        bits: u8,
    ) -> Option<Self> {
        if trees == 0 {
            return Some(Self::empty(outputs));
        }
        if b.len() != Self::byte_len(trees, depth, outputs, bits, d) || !(2..=8).contains(&bits) {
            return None;
        }
        let n = trees * depth;
        let (features, mut at): (Vec<usize>, usize) = if d <= 256 {
            (b[..n].iter().map(|&v| v as usize).collect(), n)
        } else {
            (
                b[..2 * n]
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]) as usize)
                    .collect(),
                2 * n,
            )
        };
        if features.iter().any(|&f| f >= d) {
            return None;
        }
        let thresholds = b[at..at + n].to_vec();
        at += n;
        let scales = b[at..at + 2 * trees]
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f64())
            .collect();
        at += 2 * trees;
        let offset = 1i32 << (bits - 1);
        let codes = unpack(&b[at..], trees * (1 << depth) * outputs, bits)
            .into_iter()
            .map(|u| u as i32 - offset)
            .collect();
        Some(Self {
            depth,
            outputs,
            bits,
            features,
            thresholds,
            scales,
            codes,
        })
    }
}

mod boost;
pub use boost::{boost, BoostOptions, Boosted, Split};

/// Column-major input codes of selected rows (`x[feature][row]`), with
/// missing values replaced by `impute[feature]`.
pub fn column_codes(x: &[Vec<f64>], rows: &[usize], impute: &[f64]) -> Vec<u8> {
    let n = rows.len();
    let mut out = vec![0u8; x.len() * n];
    for (f, col) in x.iter().enumerate() {
        for (i, &r) in rows.iter().enumerate() {
            let v = col[r];
            out[f * n + i] = code(if v.is_finite() { v } else { impute[f] });
        }
    }
    out
}

/// Forest outputs `[n, o]` for column-major codes.
pub fn predict_columns(forest: &Forest, codes: &[u8], d: usize, n: usize) -> Vec<f64> {
    let o = forest.outputs;
    let mut out = vec![0.0; n * o];
    if forest.trees() == 0 {
        return out;
    }
    let mut row = vec![0u8; d];
    for r in 0..n {
        for f in 0..d {
            row[f] = codes[f * n + r];
        }
        forest.add(&row, &mut out[r * o..(r + 1) * o]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::target::{loss, Head};

    fn toy(n: usize, seed: u64) -> (Vec<u8>, Vec<f64>) {
        // y depends on an interaction a step function cannot express additively.
        let mut rng = Rng::new(seed);
        let d = 3;
        let mut codes = vec![0u8; d * n];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let x: Vec<f64> = (0..d).map(|_| rng.uniform()).collect();
            for f in 0..d {
                codes[f * n + r] = code(x[f]);
            }
            y[r] = if (x[0] > 0.5) ^ (x[1] > 0.3) {
                1.0
            } else {
                0.0
            } + 0.05 * rng.normal();
        }
        (codes, y)
    }

    #[test]
    fn boosting_learns_an_interaction_and_serializes_exactly() {
        let (cf, yf) = toy(3000, 1);
        let (cv, yv) = toy(1000, 2);
        let base_f = vec![0.5; yf.len()];
        let base_v = vec![0.5; yv.len()];
        let b = boost(
            Head::Regression,
            3,
            &Split {
                codes: &cf,
                base: &base_f,
                y: &yf,
            },
            &Split {
                codes: &cv,
                base: &base_v,
                y: &yv,
            },
            &BoostOptions {
                max_trees: 60,
                ..Default::default()
            },
        );
        let last = *b.val_path.last().unwrap();
        assert!(last < 0.2 * b.val_path[0], "{:?}", &b.val_path[..5]);
        let bytes = b.forest.to_bytes(3);
        assert_eq!(bytes.len(), Forest::byte_len(b.forest.trees(), 4, 1, 6, 3));
        let back = Forest::from_bytes(&bytes, 3, b.forest.trees(), 4, 1, 6).unwrap();
        assert_eq!(back, b.forest);
        // Prediction through the serialized forest reproduces the validation path.
        let mut pv = predict_columns(&back, &cv, 3, yv.len());
        pv.iter_mut().zip(&base_v).for_each(|(p, b)| *p += b);
        assert!((loss(Head::Regression, &pv, &yv) - last).abs() < 1e-12);
    }

    #[test]
    fn multiclass_forest_reduces_cross_entropy() {
        let (cf, yf) = toy(2000, 3);
        let (cv, yv) = toy(800, 4);
        let lab = |y: &[f64], c: &[u8], n: usize| -> Vec<f64> {
            (0..n)
                .map(|r| {
                    if y[r] > 0.5 {
                        2.0
                    } else if c[2 * n + r] > 128 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        let (lf, lv) = (lab(&yf, &cf, 2000), lab(&yv, &cv, 800));
        let head = Head::Multiclass(3);
        let (bf, bv) = (vec![0.0; 3 * 2000], vec![0.0; 3 * 800]);
        let b = boost(
            head,
            3,
            &Split {
                codes: &cf,
                base: &bf,
                y: &lf,
            },
            &Split {
                codes: &cv,
                base: &bv,
                y: &lv,
            },
            &BoostOptions {
                max_trees: 40,
                ..Default::default()
            },
        );
        assert!(*b.val_path.last().unwrap() < 0.5 * b.val_path[0]);
    }
}

//! Oblivious trees on the tensor path: one gather of split features, one
//! comparison, a bit-weighted sum and one leaf gather per group of trees.

use crate::error::Result;
use crate::program::Program;
use candle_core::{DType, Device, Tensor};

use super::t2;

/// The forest as tensors: per (tree, level) the layout position of its split
/// feature, the threshold and the bit weight; leaves as a `[T·L, o]` table.
pub(super) struct DeviceForest {
    pos: Tensor,
    thr: Tensor,
    pow2: Tensor,
    offs: Tensor,
    leaves: Tensor,
    impute: Tensor,
    trees: usize,
    depth: usize,
    outputs: usize,
}

impl DeviceForest {
    pub(super) fn new(
        program: &Program,
        pos_of: &[usize],
        layout: &[usize],
        dev: &Device,
    ) -> Result<Self> {
        let f = &program.forest;
        let (t, dp, o, nl) = (f.trees(), f.depth, f.outputs, f.leaves());
        let pos: Vec<u32> = f.features.iter().map(|&j| pos_of[j] as u32).collect();
        let n = pos.len();
        let mut leaves = Vec::with_capacity(t * nl * o);
        for tr in 0..t {
            for l in 0..nl {
                for k in 0..o {
                    leaves.push(f.value(tr, l, k) as f32);
                }
            }
        }
        Ok(Self {
            pos: Tensor::from_vec(pos, n, dev)?,
            thr: t2(f.thresholds.iter().map(|&c| c as f32).collect(), 1, n, dev)?,
            pow2: t2(
                (0..n).map(|i| (1u32 << (i % dp)) as f32).collect(),
                1,
                n,
                dev,
            )?,
            offs: t2((0..t).map(|tr| (tr * nl) as f32).collect(), 1, t, dev)?,
            leaves: t2(leaves, t * nl, o, dev)?,
            impute: t2(
                layout.iter().map(|&j| program.impute(j) as f32).collect(),
                1,
                layout.len(),
                dev,
            )?,
            trees: t,
            depth: dp,
            outputs: o,
        })
    }

    /// Forest outputs `[n, o]` for model-space rows `x` (layout order) with
    /// missing mask `miss`; trees are evaluated in groups to bound memory.
    pub(super) fn outputs(&self, x: &Tensor, miss: &Tensor, n: usize) -> Result<Tensor> {
        let d = x.dim(1)?;
        let full = miss.where_cond(&self.impute.broadcast_as((n, d))?.contiguous()?, x)?;
        let codes = full.affine(255.0, 0.5)?.floor()?.clamp(0f32, 255f32)?;
        let group = ((1usize << 22) / (n * self.depth).max(1)).clamp(1, self.trees);
        let mut acc = Tensor::zeros((n, self.outputs), DType::F32, x.device())?;
        let mut t0 = 0;
        while t0 < self.trees {
            let g = group.min(self.trees - t0);
            let (a, len) = (t0 * self.depth, g * self.depth);
            let sel = codes.index_select(&self.pos.narrow(0, a, len)?, 1)?;
            let leaf = sel
                .broadcast_gt(&self.thr.narrow(1, a, len)?)?
                .to_dtype(DType::F32)?
                .broadcast_mul(&self.pow2.narrow(1, a, len)?)?
                .reshape((n, g, self.depth))?
                .sum(2)?
                .broadcast_add(&self.offs.narrow(1, t0, g)?)?;
            let ids = leaf.to_dtype(DType::U32)?.flatten_all()?;
            let vals = self
                .leaves
                .index_select(&ids, 0)?
                .reshape((n, g, self.outputs))?
                .sum(1)?;
            acc = (acc + vals)?;
            t0 += g;
        }
        Ok(acc)
    }
}

//! The declared attention, normalization and SiLU operators, CPU reference only.
use super::{manifest::Graph, weights::Weights};
use crate::Result;
use candle_core::{Module, Tensor};
use candle_nn::{LayerNorm, Linear};

pub struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    width: usize,
}
impl Attention {
    pub fn load(weights: &mut Weights, prefix: &str, graph: &Graph) -> Result<Self> {
        let mut linear =
            |name: &str| weights.linear(&format!("{prefix}.{name}"), graph.width, graph.width);
        Ok(Self {
            q: linear("q")?,
            k: linear("k")?,
            v: linear("v")?,
            out: linear("out")?,
            heads: graph.heads,
            width: graph.width,
        })
    }
    pub fn forward(&self, query: &Tensor, context: &Tensor) -> Result<Tensor> {
        let is_matrix = query.rank() == 2;
        let query = if is_matrix {
            query.unsqueeze(0)?
        } else {
            query.clone()
        };
        let (batch, qlen, _) = query.dims3()?;
        let context = if context.rank() == 2 {
            context
                .unsqueeze(0)?
                .broadcast_as((batch, context.dim(0)?, self.width))?
                .contiguous()?
        } else {
            context.clone()
        };
        let klen = context.dim(1)?;
        let dim = self.width / self.heads;
        let project = |linear: &Linear, value: &Tensor, len| -> Result<Tensor> {
            Ok(linear
                .forward(value)?
                .reshape((batch, len, self.heads, dim))?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch * self.heads, len, dim))?)
        };
        let q = project(&self.q, &query, qlen)?;
        let k = project(&self.k, &context, klen)?;
        let v = project(&self.v, &context, klen)?;
        let scores = q
            .matmul(&k.transpose(1, 2)?.contiguous()?)?
            .affine(1. / (dim as f64).sqrt(), 0.)?;
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        let mixed = probabilities
            .matmul(&v)?
            .reshape((batch, self.heads, qlen, dim))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, qlen, self.width))?;
        let out = self.out.forward(&mixed)?;
        Ok(if is_matrix { out.squeeze(0)? } else { out })
    }
}

pub struct Block {
    nq: LayerNorm,
    nk: LayerNorm,
    nf: LayerNorm,
    attention: Attention,
    up: Linear,
    down: Linear,
}
impl Block {
    pub fn load(weights: &mut Weights, prefix: &str, graph: &Graph) -> Result<Self> {
        Ok(Self {
            nq: weights.norm(&format!("{prefix}.norm_q"), graph.width)?,
            nk: weights.norm(&format!("{prefix}.norm_kv"), graph.width)?,
            nf: weights.norm(&format!("{prefix}.norm_ff"), graph.width)?,
            attention: Attention::load(weights, &format!("{prefix}.attention"), graph)?,
            up: weights.linear(&format!("{prefix}.up"), graph.width, graph.width * 4)?,
            down: weights.linear(&format!("{prefix}.down"), graph.width * 4, graph.width)?,
        })
    }
    pub fn forward(&self, query: &Tensor, context: &Tensor) -> Result<Tensor> {
        let value = (query
            + self
                .attention
                .forward(&self.nq.forward(query)?, &self.nk.forward(context)?)?)?;
        Ok((&value
            + self
                .down
                .forward(&self.up.forward(&self.nf.forward(&value)?)?.silu()?)?)?)
    }
}

pub struct Induced {
    slots: Tensor,
    read: Block,
    write: Block,
}
impl Induced {
    pub fn load(weights: &mut Weights, prefix: &str, graph: &Graph) -> Result<Self> {
        Ok(Self {
            slots: weights.get(
                &format!("{prefix}.slots"),
                &[graph.inducing_slots, graph.width],
            )?,
            read: Block::load(weights, &format!("{prefix}.read"), graph)?,
            write: Block::load(weights, &format!("{prefix}.write"), graph)?,
        })
    }
    pub fn forward(&self, rows: &Tensor) -> Result<Tensor> {
        self.write
            .forward(rows, &self.read.forward(&self.slots, rows)?)
    }
}

//! Only selected code, positional schema, noisy row state and time enter this graph.
use super::{
    blocks::Block,
    manifest::{check, Graph, Schema},
    tokenizer::Tokenizer,
    weights::Weights,
};
use crate::Result;
use candle_core::{Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear};

pub struct Decoder {
    pub graph: Graph,
    tokenizer: Tokenizer,
    conditions: Vec<Linear>,
    time: Linear,
    own: Vec<Block>,
    cross: Vec<Block>,
    norm: LayerNorm,
    numeric: Linear,
    categorical: Linear,
    missing: Linear,
}
impl Decoder {
    pub fn load(weights: &mut Weights, graph: &Graph) -> Result<Self> {
        let tokenizer = Tokenizer::load(weights, graph)?;
        let conditions = graph
            .blocks()
            .iter()
            .enumerate()
            .map(|(i, b)| {
                weights.linear(
                    &format!("conditions.{i}"),
                    *b,
                    graph.condition_slots * graph.width,
                )
            })
            .collect::<Result<_>>()?;
        let time = weights.linear("time", 8, graph.width)?;
        let own = (0..graph.decoder_blocks)
            .map(|i| Block::load(weights, &format!("self_blocks.{i}"), graph))
            .collect::<Result<_>>()?;
        let cross = (0..graph.decoder_blocks)
            .map(|i| Block::load(weights, &format!("cross_blocks.{i}"), graph))
            .collect::<Result<_>>()?;
        let norm = weights.norm("norm", graph.width)?;
        let numeric = weights.linear("numeric", graph.width, 1)?;
        let categorical = weights.linear("categorical", graph.width, graph.max_categories)?;
        let missing = weights.linear("missing", graph.width, 1)?;
        Ok(Self {
            graph: graph.clone(),
            tokenizer,
            conditions,
            time,
            own,
            cross,
            norm,
            numeric,
            categorical,
            missing,
        })
    }
    pub fn forward(
        &self,
        code: &Tensor,
        schema: &Schema,
        noisy: &Tensor,
        observed: &Tensor,
        time: f32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        schema.validate(&self.graph)?;
        let width = code.dim(0)?;
        let (rows, columns) = noisy.dims2()?;
        check(
            code.rank() == 1
                && self.graph.widths().contains(&width)
                && time.is_finite()
                && (0.0..=1.0).contains(&time)
                && rows > 0
                && rows <= 16384
                && columns == schema.columns.len()
                && rows.checked_mul(columns).is_some_and(|n| n <= 262144)
                && rows
                    .checked_mul(columns * columns)
                    .and_then(|n| n.checked_mul(self.graph.heads))
                    .is_some_and(|n| n <= 33554432)
                && rows
                    .checked_mul(columns)
                    .and_then(|n| n.checked_mul(self.graph.width))
                    .is_some_and(|n| n <= 33554432),
            "learned decoder envelope exceeded",
        )?;
        let mut conditions = Vec::new();
        let mut start = 0;
        for (block, projection) in self.graph.blocks().iter().zip(&self.conditions) {
            if start + block > width {
                break;
            }
            conditions.push(
                projection
                    .forward(&code.narrow(0, start, *block)?.unsqueeze(0)?)?
                    .reshape((self.graph.condition_slots, self.graph.width))?,
            );
            start += block;
        }
        let condition = Tensor::cat(&conditions, 0)?;
        let mut phase = Vec::new();
        for f in [1., 10., 100., 1000.] {
            phase.push((time * f).sin());
        }
        for f in [1., 10., 100., 1000.] {
            phase.push((time * f).cos());
        }
        let phase = Tensor::from_vec(phase, (1, 8), &Device::Cpu)?;
        let mut token = self
            .tokenizer
            .forward(noisy, observed, schema)?
            .broadcast_add(&self.time.forward(&phase)?)?;
        for (own, cross) in self.own.iter().zip(&self.cross) {
            token = own.forward(&token, &token)?;
            token = cross.forward(&token, &condition)?;
        }
        let token = self.norm.forward(&token)?;
        let mut logits = self.categorical.forward(&token)?.to_vec3::<f32>()?;
        for row in &mut logits {
            for (j, column) in schema.columns.iter().enumerate() {
                if column.kind == "categorical" {
                    for (k, value) in row[j].iter_mut().enumerate() {
                        if k < 2 || k >= column.categories {
                            *value = -1e9;
                        }
                    }
                }
            }
        }
        Ok((
            self.numeric.forward(&token)?.squeeze(2)?,
            Tensor::from_vec(
                logits.into_iter().flatten().flatten().collect::<Vec<_>>(),
                (rows, columns, self.graph.max_categories),
                &Device::Cpu,
            )?,
            self.missing.forward(&token)?.squeeze(2)?,
        ))
    }
}

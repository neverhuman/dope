//! Native target-free trunk and task branch with trained nested readouts.
use super::{
    blocks::{Block, Induced},
    manifest::{check, Graph, Schema},
    tokenizer::Tokenizer,
    weights::Weights,
};
use crate::Result;
use candle_core::{DType, Module, Tensor};
use candle_nn::{LayerNorm, Linear};

pub struct TableEncoder {
    pub graph: Graph,
    tokenizer: Tokenizer,
    features: Vec<Block>,
    rows: Vec<Induced>,
    summaries: Tensor,
    residual_summaries: Option<Tensor>,
    pool: Vec<Block>,
    target: Vec<Block>,
    distribution_norm: LayerNorm,
    task_norm: LayerNorm,
    distribution_heads: Vec<Linear>,
    task_heads: Vec<Linear>,
}
impl TableEncoder {
    pub fn load(weights: &mut Weights, graph: &Graph) -> Result<Self> {
        let tokenizer = Tokenizer::load(weights, graph)?;
        let features = (0..graph.feature_blocks)
            .map(|i| Block::load(weights, &format!("features.{i}"), graph))
            .collect::<Result<_>>()?;
        let rows = (0..graph.inducing_blocks)
            .map(|i| Induced::load(weights, &format!("rows.{i}"), graph))
            .collect::<Result<_>>()?;
        let slots = if graph.slot_readout() {
            graph.readout_slots()
        } else {
            graph.summary_slots
        };
        let summaries = weights.get("summaries", &[slots, graph.width])?;
        let residual_summaries = if graph.slot_readout() {
            Some(weights.get("residual_summaries", &[slots, graph.width])?)
        } else {
            None
        };
        let pool = (0..2)
            .map(|i| Block::load(weights, &format!("pool.{i}"), graph))
            .collect::<Result<_>>()?;
        let target = (0..graph.target_blocks)
            .map(|i| Block::load(weights, &format!("target.{i}"), graph))
            .collect::<Result<_>>()?;
        let distribution_norm = weights.norm("distribution_norm", graph.width)?;
        let task_norm = weights.norm("task_norm", graph.width)?;
        let distribution_heads = graph
            .blocks()
            .iter()
            .enumerate()
            .map(|(i, b)| {
                weights.linear(
                    &format!("distribution_heads.{i}"),
                    graph.width * graph.readout_slots(),
                    b / 2,
                )
            })
            .collect::<Result<_>>()?;
        let task_heads = graph
            .blocks()
            .iter()
            .enumerate()
            .map(|(i, b)| {
                weights.linear(
                    &format!("task_heads.{i}"),
                    graph.width * graph.readout_slots(),
                    b / 2,
                )
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            graph: graph.clone(),
            tokenizer,
            features,
            rows,
            summaries,
            residual_summaries,
            pool,
            target,
            distribution_norm,
            task_norm,
            distribution_heads,
            task_heads,
        })
    }

    pub fn encode(
        &self,
        values: &Tensor,
        observed: &Tensor,
        schema: &Schema,
        target_included: bool,
        width: usize,
    ) -> Result<Tensor> {
        schema.validate(&self.graph)?;
        let (count, columns) = values.dims2()?;
        check(
            schema.target_index == Some(columns.saturating_sub(1))
                && columns >= 2
                && columns == schema.columns.len()
                && observed.dims() == values.dims()
                && count > 0
                && count <= self.graph.max_context
                && count
                    .checked_mul(columns - 1)
                    .is_some_and(|n| n <= self.graph.cell_cap)
                && count
                    .checked_mul((columns - 1).pow(2))
                    .and_then(|n| n.checked_mul(self.graph.heads))
                    .is_some_and(|n| n <= 33554432)
                && count
                    .checked_mul(columns)
                    .and_then(|n| n.checked_mul(self.graph.width))
                    .is_some_and(|n| n <= 33554432)
                && self.graph.widths().contains(&width),
            "learned encoder envelope exceeded",
        )?;
        let feature_schema = Schema {
            columns: schema.columns[..columns - 1].to_vec(),
            task: None,
            target_index: None,
            ..schema.clone()
        };
        let mut feature = self.tokenizer.forward(
            &values.narrow(1, 0, columns - 1)?,
            &observed.narrow(1, 0, columns - 1)?,
            &feature_schema,
        )?;
        let mut next = 0;
        for (i, block) in self.features.iter().enumerate() {
            feature = block.forward(&feature, &feature)?;
            if i % 2 == 1 && next < self.rows.len() {
                let previous = feature.mean(1)?;
                let mixed = self.rows[next].forward(&previous)?;
                feature = feature.broadcast_add(&(mixed - previous)?.unsqueeze(1)?)?;
                next += 1;
            }
        }
        let mut rows = feature.mean(1)?;
        for block in &self.rows[next..] {
            rows = block.forward(&rows)?;
        }
        let summary = self.pool_summary(&self.summaries, &rows)?;
        let mut distribution = self.readout(&summary, &self.distribution_norm)?;
        let target_rows = if target_included {
            let target_schema = Schema {
                columns: schema.columns[columns - 1..].to_vec(),
                ..schema.clone()
            };
            let mut target = (&rows
                + self
                    .tokenizer
                    .forward(
                        &values.narrow(1, columns - 1, 1)?,
                        &observed.narrow(1, columns - 1, 1)?,
                        &target_schema,
                    )?
                    .squeeze(1)?)?;
            for block in &self.target {
                target = block.forward(&target, &rows)?;
            }
            Some(target)
        } else {
            None
        };
        let mut task = if let Some(target) = &target_rows {
            Some(if self.graph.slot_readout() {
                self.readout(
                    &self.pool_summary(&self.summaries, target)?,
                    &self.task_norm,
                )?
            } else {
                self.task_norm.forward(&target.mean(0)?)?
            })
        } else {
            None
        };
        let mut blocks = Vec::new();
        let widths = self.graph.blocks();
        for (i, block) in widths.iter().enumerate() {
            if i == 3 {
                if let Some(queries) = &self.residual_summaries {
                    distribution =
                        self.readout(&self.pool_summary(queries, &rows)?, &self.distribution_norm)?;
                    if let Some(target) = &target_rows {
                        task = Some(
                            self.readout(&self.pool_summary(queries, target)?, &self.task_norm)?,
                        );
                    }
                }
            }
            let d = self.distribution_heads[i]
                .forward(&distribution.unsqueeze(0)?)?
                .squeeze(0)?;
            let t = if let Some(task) = &task {
                self.task_heads[i]
                    .forward(&task.unsqueeze(0)?)?
                    .squeeze(0)?
            } else {
                d.zeros_like()?
            };
            let (d, t) = if self.graph.bounded_code() {
                (d.tanh()?, t.tanh()?)
            } else {
                (d, t)
            };
            blocks.push(Tensor::cat(&[d, t], 0)?);
            if widths[..i + 1].iter().sum::<usize>() == width {
                break;
            }
            check(*block > 0, "invalid learned readout")?;
        }
        Ok(Tensor::cat(&blocks, 0)?
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?)
    }
    fn pool_summary(&self, queries: &Tensor, rows: &Tensor) -> Result<Tensor> {
        let mut summary = queries.clone();
        for block in &self.pool {
            summary = block.forward(&summary, rows)?;
        }
        Ok(summary)
    }
    fn readout(&self, summary: &Tensor, norm: &LayerNorm) -> Result<Tensor> {
        if !self.graph.slot_readout() {
            return Ok(norm.forward(&summary.mean(0)?)?);
        }
        let result = norm.forward(summary)?;
        let result = if self.graph.pooled_control() {
            result.mean_keepdim(0)?.broadcast_as(result.shape())?
        } else {
            result
        };
        Ok(result.flatten_all()?)
    }
}

//! Positional, typed tokens. Source names and private column states are not inputs.
use super::{
    manifest::{check, Graph, Schema},
    weights::Weights,
};
use crate::Result;
use candle_core::{Device, Module, Tensor};
use candle_nn::{Embedding, Linear};

pub struct Tokenizer {
    values: Linear,
    types: Embedding,
    categories: Embedding,
    bindings: Linear,
}
impl Tokenizer {
    pub fn load(weights: &mut Weights, graph: &Graph) -> Result<Self> {
        Ok(Self {
            values: weights.linear("tokenizer.values", 8, graph.width)?,
            types: weights.embedding("tokenizer.types", 2, graph.width)?,
            categories: weights.embedding(
                "tokenizer.categories",
                graph.max_categories,
                graph.width,
            )?,
            bindings: weights.linear("tokenizer.bindings", 8, graph.width)?,
        })
    }
    pub fn forward(&self, values: &Tensor, observed: &Tensor, schema: &Schema) -> Result<Tensor> {
        let (rows, columns) = values.dims2()?;
        check(
            columns == schema.columns.len() && observed.dims() == values.dims(),
            "learned token shape",
        )?;
        let raw = values.to_vec2::<f32>()?;
        let present = observed.to_vec2::<f32>()?;
        let mut channels = Vec::with_capacity(rows * columns * 8);
        let mut indices = Vec::with_capacity(rows * columns);
        let mut kinds = Vec::with_capacity(columns);
        let mut bindings = Vec::with_capacity(columns * 8);
        for c in &schema.columns {
            kinds.push(u32::from(c.kind == "categorical"));
            let phase: Vec<f32> = [1., 0.1, 0.01, 0.001]
                .iter()
                .map(|f| c.slot as f32 * f)
                .collect();
            bindings.extend(phase.iter().map(|p| p.sin()));
            bindings.extend(phase.iter().map(|p| p.cos()));
        }
        for r in 0..rows {
            for j in 0..columns {
                let value = raw[r][j];
                let obs = present[r][j];
                check(
                    value.is_finite() && (obs == 0. || obs == 1.),
                    "nonfinite learned token",
                )?;
                let categorical = kinds[j] == 1;
                check(
                    !categorical
                        || (value >= 0.
                            && value.fract() == 0.
                            && value < schema.columns[j].categories as f32),
                    "learned category outside schema",
                )?;
                channels.extend([
                    value.clamp(-8., 8.),
                    (value.signum() * value.abs().ln_1p()).clamp(-8., 8.),
                    obs,
                    1. - obs,
                    u8::from(value == 0.) as f32,
                    u8::from(value.fract() == 0.) as f32,
                    u8::from(categorical) as f32,
                    u8::from(value == 1.) as f32,
                ]);
                indices.push(if categorical { value as u32 } else { 0 });
            }
        }
        let channels = Tensor::from_vec(channels, (rows, columns, 8), &Device::Cpu)?;
        let indices = Tensor::from_vec(indices, (rows, columns), &Device::Cpu)?;
        let kind_tensor = Tensor::from_vec(kinds.clone(), columns, &Device::Cpu)?;
        let bindings = Tensor::from_vec(bindings, (columns, 8), &Device::Cpu)?;
        let kind_mask = Tensor::from_vec(
            kinds.iter().map(|k| *k as f32).collect::<Vec<_>>(),
            (columns, 1),
            &Device::Cpu,
        )?;
        Ok(self
            .values
            .forward(&channels)?
            .broadcast_add(&self.types.forward(&kind_tensor)?)?
            .broadcast_add(&self.bindings.forward(&bindings)?)?
            .broadcast_add(
                &self
                    .categories
                    .forward(&indices)?
                    .broadcast_mul(&kind_mask)?,
            )?)
    }
}

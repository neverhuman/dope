//! Seeded code-only DDIM sampling. This path has no raw-table or encoder-state argument.
use super::{
    decoder::Decoder,
    manifest::{check, Schema},
};
use crate::{rng, Result};
use candle_core::{Device, Tensor};

pub struct LearnedSamples {
    pub schema: Schema,
    pub rows: Vec<Vec<Option<f64>>>,
}
fn alpha(time: f32) -> f32 {
    (((time + 0.008) / 1.008 * std::f32::consts::FRAC_PI_2)
        .cos()
        .powi(2))
    .clamp(0.001, 0.999)
}

pub fn sample(
    decoder: &Decoder,
    code: &Tensor,
    schema: &Schema,
    rows: usize,
    seed: u64,
    steps: usize,
) -> Result<LearnedSamples> {
    schema.validate(&decoder.graph)?;
    let columns = schema.columns.len();
    check(
        rows > 0
            && rows <= 1000000
            && [8, 16, 32, 64].contains(&steps)
            && rows.checked_mul(columns).is_some_and(|n| n <= 4000000),
        "learned sample allocation exceeded",
    )?;
    let mut output = Vec::with_capacity(rows);
    let chunk = 256
        .min(33554432 / (columns * columns * decoder.graph.heads))
        .min(33554432 / (columns * decoder.graph.width));
    check(chunk > 0, "learned sampler graph allocation exceeded")?;
    for row0 in (0..rows).step_by(chunk) {
        let count = (rows - row0).min(chunk);
        let mut values = vec![vec![0f32; columns]; count];
        let mut observed = vec![vec![0f32; columns]; count];
        for (r, row) in values.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                if schema.columns[j].kind == "numeric" {
                    *value = rng::normal(seed, 0x4c4541524e + j as u64, (row0 + r) as u64) as f32;
                }
            }
        }
        for step in (1..=steps).rev() {
            let time = step as f32 / steps as f32;
            let a = alpha(time);
            let previous = alpha((step - 1) as f32 / steps as f32);
            let noisy = Tensor::from_vec(
                values.iter().flatten().copied().collect::<Vec<_>>(),
                (count, columns),
                &Device::Cpu,
            )?;
            let present = Tensor::from_vec(
                observed.iter().flatten().copied().collect::<Vec<_>>(),
                (count, columns),
                &Device::Cpu,
            )?;
            let (epsilon, logits, missing) =
                decoder.forward(code, schema, &noisy, &present, time)?;
            let epsilon = epsilon.to_vec2::<f32>()?;
            let logits = logits.to_vec3::<f32>()?;
            let missing = missing.to_vec2::<f32>()?;
            for r in 0..count {
                for j in 0..columns {
                    check(
                        epsilon[r][j].is_finite()
                            && missing[r][j].is_finite()
                            && logits[r][j].iter().all(|v| v.is_finite()),
                        "nonfinite learned sampler prediction",
                    )?;
                    if schema.columns[j].kind == "numeric" {
                        let x0 = (values[r][j] - (1. - a).sqrt() * epsilon[r][j]) / a.sqrt();
                        check(x0.is_finite(), "nonfinite learned diffusion state")?;
                        let x0 = x0.clamp(-16., 16.);
                        values[r][j] = if step > 1 {
                            previous.sqrt() * x0 + (1. - previous).sqrt() * epsilon[r][j]
                        } else {
                            x0
                        };
                    } else {
                        let maximum = logits[r][j]
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, f32::max);
                        let probabilities: Vec<f32> =
                            logits[r][j].iter().map(|v| (v - maximum).exp()).collect();
                        let total: f32 = probabilities.iter().sum();
                        let random = rng::uniform(
                            seed,
                            0x434154 + (step * columns + j) as u64,
                            (row0 + r) as u64,
                        ) as f32
                            * total;
                        let mut sum = 0.;
                        let mut chosen = 2;
                        for (k, p) in probabilities
                            .iter()
                            .enumerate()
                            .take(schema.columns[j].categories)
                            .skip(2)
                        {
                            sum += p;
                            chosen = k;
                            if random < sum {
                                break;
                            }
                        }
                        values[r][j] = chosen as f32;
                    }
                    let probability = 1. / (1. + (-missing[r][j]).exp());
                    observed[r][j] = u8::from(
                        rng::uniform(
                            seed,
                            0x4d4953 + (step * columns + j) as u64,
                            (row0 + r) as u64,
                        ) > probability as f64,
                    ) as f32;
                }
            }
        }
        for (r, row) in values.into_iter().enumerate() {
            let mut converted = Vec::with_capacity(columns);
            for (j, value) in row.into_iter().enumerate() {
                let column = &schema.columns[j];
                if observed[r][j] == 0. && schema.target_index != Some(j) {
                    converted.push(None);
                    continue;
                }
                let raw = if column.kind == "numeric" {
                    value as f64 * column.scale + column.center
                } else {
                    value as f64
                };
                check(raw.is_finite(), "learned inverse transform overflow")?;
                converted.push(Some(raw));
            }
            output.push(converted);
        }
    }
    Ok(LearnedSamples {
        schema: schema.clone(),
        rows: output,
    })
}

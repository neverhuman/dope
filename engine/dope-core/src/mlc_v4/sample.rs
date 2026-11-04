//! Source-free reference group sampler, fixed draws and SplitMix64 replay.
use super::{basis, wire, GroupModel, Packet, Result, TargetModel, WireError};
use dope_norm::UnitSchema;
use serde::{Deserialize, Serialize};
/// Synthetic rows are explicitly distinguished from private real UnitTable rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticTable {
    pub schema: UnitSchema,
    pub packet_sha256: String,
    pub seed: Option<u64>,
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
}
/// Draw schedule: two uniforms per group plus one target uniform per row.
/// Numeric value uses first, missing indicator second. Categorical uses first.
/// Regression noise is symmetric uniform with variance one, not Gaussian.
pub fn sample_with_draws(packet: &Packet, draws: &[Vec<f64>]) -> Result<SyntheticTable> {
    packet.validate()?;
    let packet_bytes = wire::encode(packet)?;
    let n = draws.len();
    let cols = packet.schema.columns;
    if n == 0 || n > 100000 || n.checked_mul(cols).is_none_or(|v| v > 16_000_000) {
        return Err(WireError::new("sample_shape_limit"));
    }
    let draw_width = packet.groups.len() * 2 + 1;
    if draws.iter().any(|row| {
        row.len() != draw_width || row.iter().any(|v| !v.is_finite() || *v < 0.0 || *v >= 1.0)
    }) {
        return Err(WireError::new("invalid_draws"));
    }
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for draws in draws {
        let mut row = Vec::with_capacity(cols);
        for (i, (group, model)) in packet.schema.groups.iter().zip(&packet.groups).enumerate() {
            let u = draws[2 * i];
            let miss = draws[2 * i + 1];
            match model {
                GroupModel::Numeric {
                    knots,
                    missing_probability,
                    imputation,
                } => {
                    let missing = miss < *missing_probability;
                    row.push(if missing {
                        *imputation
                    } else {
                        quantile(knots, u)
                    });
                    if group.has_missing {
                        row.push(f64::from(missing));
                    }
                }
                GroupModel::Discrete {
                    values,
                    probabilities,
                    missing_probability,
                    imputation,
                } => {
                    let missing = miss < *missing_probability;
                    row.push(if missing {
                        *imputation
                    } else {
                        values[categorical(probabilities, u)]
                    });
                    if group.has_missing {
                        row.push(f64::from(missing));
                    }
                }
                GroupModel::Categorical { probabilities } => {
                    let chosen = categorical(probabilities, u);
                    row.extend((0..probabilities.len()).map(|k| f64::from(k == chosen)));
                }
            }
        }
        let phi = basis::features(&row, &packet.gates)?;
        let u = draws[draw_width - 1];
        let target = match &packet.target {
            TargetModel::Regression {
                intercept,
                coefficients,
                noise_scale,
            } => {
                let predicted = dot(coefficients, &phi) + intercept;
                predicted + noise_scale * (2.0 * u - 1.0) * 3.0f64.sqrt()
            }
            TargetModel::Classification {
                intercepts,
                coefficients,
            } => {
                let logits: Vec<_> = intercepts
                    .iter()
                    .zip(coefficients)
                    .map(|(b, w)| b + dot(w, &phi))
                    .collect();
                let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let probabilities: Vec<_> = logits.iter().map(|v| (v - max).exp()).collect();
                categorical(&probabilities, u) as f64
            }
        };
        if !target.is_finite() {
            return Err(WireError::new("nonfinite_sample_target"));
        }
        x.push(row);
        y.push(target);
    }
    Ok(SyntheticTable {
        schema: packet.schema.clone(),
        packet_sha256: wire::sha256(&packet_bytes),
        seed: None,
        x,
        y,
    })
}
/// Deterministic source-free draws. No private card or source table is accepted.
pub fn sample(packet: &Packet, rows: usize, seed: u64) -> Result<SyntheticTable> {
    packet.validate()?;
    wire::encode(packet)?;
    if rows == 0
        || rows > 100000
        || rows
            .checked_mul(packet.schema.columns)
            .is_none_or(|v| v > 16_000_000)
    {
        return Err(WireError::new("sample_shape_limit"));
    }
    let width = packet.groups.len() * 2 + 1;
    let mut state = seed;
    let mut draws = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut row = Vec::with_capacity(width);
        for _ in 0..width {
            row.push(uniform(&mut state));
        }
        draws.push(row);
    }
    let mut out = sample_with_draws(packet, &draws)?;
    out.seed = Some(seed);
    Ok(out)
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn quantile(knots: &[f64], u: f64) -> f64 {
    let index = u * (knots.len() - 1) as f64;
    let low = (index.floor() as usize).min(knots.len() - 2);
    let fraction = index - low as f64;
    knots[low] + fraction * (knots[low + 1] - knots[low])
}
fn categorical(probabilities: &[f64], u: f64) -> usize {
    let threshold = u * probabilities.iter().sum::<f64>();
    let mut sum = 0.0;
    for (i, p) in probabilities.iter().enumerate() {
        sum += p;
        if threshold < sum {
            return i;
        }
    }
    probabilities.len() - 1
}
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 11) as f64 / 9007199254740992.0
}

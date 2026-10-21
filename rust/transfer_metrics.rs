//! Original-unit transfer diagnostics, independent of historical certification.
//!
//! Shared primitives for the authoritative metric core in `crate::metrics`.
//! Query losses are evaluator-private. Neither a score nor these diagnostics
//! authorize release. Bytes remain an independent feasibility constraint.

pub mod drivers;
pub mod frontier;
pub mod inference;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MetricError {
    InvalidEvidence,
    Nonfinite,
    DegenerateScale,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LossSpace {
    OriginalMse,
    DiscreteLogLoss,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrainingScale {
    Positive(f64),
    Degenerate,
}

fn finite(value: f64) -> Result<f64, MetricError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MetricError::Nonfinite)
    }
}

fn mean(values: &[f64]) -> Result<f64, MetricError> {
    if values.is_empty() {
        return Err(MetricError::InvalidEvidence);
    }
    finite(values.iter().map(|v| v / values.len() as f64).sum())
}

pub fn training_variance(
    fit_targets: &[f64],
    minimum_variance: f64,
) -> Result<TrainingScale, MetricError> {
    if finite(minimum_variance)? < 0.0 {
        return Err(MetricError::InvalidEvidence);
    }
    for value in fit_targets {
        finite(*value)?;
    }
    let center = mean(fit_targets)?;
    let variance = mean(
        &fit_targets
            .iter()
            .map(|value| (value - center).powi(2))
            .collect::<Vec<_>>(),
    )?;
    Ok(if variance <= minimum_variance {
        TrainingScale::Degenerate
    } else {
        TrainingScale::Positive(variance)
    })
}

include!("transfer_metrics/tdl.rs");

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GuardedRatio {
    Finite(f64),
    BothZero,
    Unbounded,
}

pub fn loss_ratio(real: f64, synthetic: f64) -> Result<GuardedRatio, MetricError> {
    if finite(real)? < 0.0 || finite(synthetic)? < 0.0 {
        return Err(MetricError::InvalidEvidence);
    }
    Ok(if real == 0.0 {
        if synthetic == 0.0 {
            GuardedRatio::BothZero
        } else {
            GuardedRatio::Unbounded
        }
    } else if (synthetic / real).is_finite() {
        GuardedRatio::Finite(synthetic / real)
    } else {
        GuardedRatio::Unbounded
    })
}

#[derive(Clone, Copy, Debug)]
pub struct LossSummary {
    pub real_loss: f64,
    pub synthetic_loss: f64,
    pub null_loss: f64,
    pub gap: f64,
    pub normalized_gap: Option<f64>,
    pub null_normalized_gap: Option<f64>,
    pub retention_numerator: f64,
    pub retention_denominator: f64,
    pub retention: Option<f64>,
    pub ratio: GuardedRatio,
}

pub fn loss_summary(
    pairs: &PairedLosses,
    scale: TrainingScale,
) -> Result<LossSummary, MetricError> {
    let (real, synthetic, null) = pairs.means()?;
    let gap = finite(synthetic - real)?;
    let numerator = null - synthetic;
    let denominator = null - real;
    let normalized_gap = match scale {
        TrainingScale::Degenerate => None,
        TrainingScale::Positive(value) if finite(value)? > 0.0 => Some(finite(gap / value)?),
        TrainingScale::Positive(_) => return Err(MetricError::InvalidEvidence),
    };
    Ok(LossSummary {
        real_loss: real,
        synthetic_loss: synthetic,
        null_loss: null,
        gap,
        normalized_gap,
        null_normalized_gap: if null > 0.0 {
            Some(finite(gap / null)?)
        } else {
            None
        },
        retention_numerator: numerator,
        retention_denominator: denominator,
        retention: if denominator > 0.0 && denominator >= 0.01 * null.abs() {
            Some(finite(numerator / denominator)?)
        } else {
            None
        },
        ratio: loss_ratio(real, synthetic)?,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct Margin {
    pub relative: f64,
    pub absolute_scaled: f64,
    pub absolute_raw: Option<f64>,
}

impl Margin {
    pub fn tolerance(self, real: f64, scale: TrainingScale) -> Result<f64, MetricError> {
        if finite(real)? < 0.0
            || finite(self.relative)? < 0.0
            || finite(self.absolute_scaled)? < 0.0
        {
            return Err(MetricError::InvalidEvidence);
        }
        if let Some(raw) = self.absolute_raw
            && finite(raw)? < 0.0
        {
            return Err(MetricError::InvalidEvidence);
        }
        let tolerance = match scale {
            TrainingScale::Positive(value) if finite(value)? > 0.0 => {
                finite(self.relative * real + self.absolute_scaled * value)?
            }
            TrainingScale::Positive(_) => return Err(MetricError::InvalidEvidence),
            TrainingScale::Degenerate => {
                finite(self.absolute_raw.ok_or(MetricError::DegenerateScale)?)?
            }
        };
        if tolerance < 0.0 {
            return Err(MetricError::InvalidEvidence);
        }
        Ok(tolerance)
    }
}

pub fn weighted_cvar(values: &[f64], weights: &[f64], alpha: f64) -> Result<f64, MetricError> {
    if values.is_empty() || values.len() != weights.len() || !(0.0..1.0).contains(&finite(alpha)?) {
        return Err(MetricError::InvalidEvidence);
    }
    let mut rows = Vec::new();
    for (value, weight) in values.iter().zip(weights) {
        if finite(*weight)? < 0.0 {
            return Err(MetricError::InvalidEvidence);
        }
        rows.push((finite(*value)?, *weight));
    }
    let total = finite(weights.iter().sum())?;
    if total <= 0.0 {
        return Err(MetricError::InvalidEvidence);
    }
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut remaining = 1.0 - alpha;
    let mut result = 0.0;
    for (value, weight) in rows {
        let take = (weight / total).min(remaining);
        result += value * take;
        remaining -= take;
        if remaining <= 0.0 {
            break;
        }
    }
    finite(result / (1.0 - alpha))
}

include!("transfer_metrics/mfs_v4.rs");

#[cfg(test)]
mod tests;

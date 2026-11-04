//! Grouped permutation evidence from a frozen prediction callback.
use super::{ensure, DvsError};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub enum Task {
    Regression,
    Binary,
    Multiclass(usize),
}
#[derive(Clone, Copy, Debug)]
pub struct PermutationOptions {
    pub repeats: usize,
    pub seed: u64,
    pub max_predict_calls: usize,
    pub task: Task,
}

#[derive(Clone, Debug, Serialize)]
pub struct GroupImportance {
    pub group_id: usize,
    pub mean_loss_increase: f64,
}

pub fn prediction_losses(
    task: Task,
    predictions: &[Vec<f64>],
    y: &[f64],
) -> Result<Vec<f64>, DvsError> {
    ensure(
        !y.is_empty() && predictions.len() == y.len(),
        "prediction_shape_mismatch",
    )?;
    let mut out = Vec::with_capacity(y.len());
    for (p, &target) in predictions.iter().zip(y) {
        ensure(
            target.is_finite() && p.iter().all(|v| v.is_finite()),
            "invalid_prediction",
        )?;
        let value = match task {
            Task::Regression => {
                ensure(p.len() == 1, "prediction_shape_mismatch")?;
                (p[0] - target).powi(2)
            }
            Task::Binary => {
                ensure(
                    p.len() == 1 && (0.0..=1.0).contains(&p[0]) && (target == 0.0 || target == 1.0),
                    "invalid_prediction",
                )?;
                let v = p[0].clamp(1e-7, 1.0 - 1e-7);
                -(target * v.ln() + (1.0 - target) * (1.0 - v).ln())
            }
            Task::Multiclass(k) => {
                ensure(
                    k >= 2
                        && p.len() == k
                        && p.iter().all(|v| (0.0..=1.0).contains(v))
                        && (p.iter().sum::<f64>() - 1.0).abs() <= 1e-8
                        && target >= 0.0
                        && target < k as f64
                        && target == target.round(),
                    "invalid_prediction",
                )?;
                -p[target as usize].clamp(1e-7, 1.0).ln()
            }
        };
        ensure(value.is_finite(), "nonfinite_prediction_loss")?;
        out.push(value);
    }
    Ok(out)
}

/// Caller supplies a frozen model's prediction-only callback, never a fitter.
/// The declared call budget is checked before even the baseline prediction.
pub fn grouped_permutation_importance<F>(
    x: &[Vec<f64>],
    y: &[f64],
    groups: &[usize],
    options: PermutationOptions,
    mut predict: F,
) -> Result<Vec<GroupImportance>, DvsError>
where
    F: FnMut(&[Vec<f64>]) -> Result<Vec<Vec<f64>>, DvsError>,
{
    let PermutationOptions {
        repeats,
        seed,
        max_predict_calls,
        task,
    } = options;
    ensure(
        x.len() >= 2 && x.len() == y.len() && !groups.is_empty() && repeats > 0,
        "invalid_permutation_shape",
    )?;
    for row in x {
        ensure(
            row.len() == groups.len() && row.iter().all(|v| v.is_finite()),
            "invalid_permutation_shape",
        )?;
    }
    let mut grouped = BTreeMap::<usize, Vec<usize>>::new();
    for (column, &group) in groups.iter().enumerate() {
        grouped.entry(group).or_default().push(column);
    }
    let calls = grouped
        .len()
        .checked_mul(repeats)
        .and_then(|n| n.checked_add(1))
        .ok_or(DvsError("permutation_budget_exceeded"))?;
    ensure(calls <= max_predict_calls, "permutation_budget_exceeded")?;
    let mean = |v: &[f64]| -> f64 { v.iter().map(|x| x / v.len() as f64).sum() };
    let base = mean(&prediction_losses(task, &predict(x)?, y)?);
    let mut state = if seed == 0 { 1 } else { seed };
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut out = Vec::new();
    for (group_id, columns) in grouped {
        let mut importance = 0.0;
        for _ in 0..repeats {
            let mut permutation: Vec<usize> = (0..x.len()).collect();
            for i in (1..x.len()).rev() {
                let j = (next() % (i as u64 + 1)) as usize;
                permutation.swap(i, j);
            }
            let mut changed = x.to_vec();
            for (i, row) in changed.iter_mut().enumerate() {
                for &column in &columns {
                    row[column] = x[permutation[i]][column];
                }
            }
            let loss = mean(&prediction_losses(task, &predict(&changed)?, y)?);
            importance += (loss - base) / repeats as f64;
        }
        ensure(importance.is_finite(), "nonfinite_permutation_importance")?;
        out.push(GroupImportance {
            group_id,
            mean_loss_increase: importance,
        });
    }
    Ok(out)
}

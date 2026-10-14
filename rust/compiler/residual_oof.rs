// Residual fitting uses only a bounded view of permitted outer-fit rows. These
// fixed row folds never depend on a candidate, target value, or final outcome.
// Subset/full-fit discrepancy and Gaussian-reference assumptions remain explicit.

fn oof_residual_view(
    table: &Table,
    completed: &[Vec<f32>],
    target: &Target,
) -> Option<(Table, Vec<f32>)> {
    oof_residual_view_capacity(table, completed, target, None)
}

fn oof_residual_view_capacity(
    table: &Table,
    completed: &[Vec<f32>],
    target: &Target,
    capacity: Option<CapacityRefinement>,
) -> Option<(Table, Vec<f32>)> {
    let (view, residuals) = oof_signed_residual_view_capacity(table, completed, target, capacity)?;
    Some((
        view,
        residuals.into_iter().map(|v| v.abs().max(1e-5)).collect(),
    ))
}

fn oof_signed_residual_view_capacity(
    table: &Table,
    completed: &[Vec<f32>],
    target: &Target,
    capacity: Option<CapacityRefinement>,
) -> Option<(Table, Vec<f32>)> {
    if table.rows < 24 {
        return None;
    }
    #[cfg(feature = "gpu-training")]
    if matches!(target, Target::CompactNeuralResidual { .. }) {
        return None;
    }
    let count = table.rows.min(256);
    let ids = (0..count)
        .map(|at| at * table.rows / count)
        .collect::<Vec<_>>();
    let folds = ids
        .iter()
        .map(|id| (splitmix64(*id as u64 ^ 0x6f6f_665f_726f_7773) % 3) as usize)
        .collect::<Vec<_>>();
    let mut residuals = vec![0.0; count];
    for fold in 0..3 {
        let train = ids
            .iter()
            .enumerate()
            .filter(|(at, _)| folds[*at] != fold)
            .map(|(_, id)| *id)
            .collect::<Vec<_>>();
        if train.is_empty() || !folds.contains(&fold) {
            return None;
        }
        let subset = Table {
            rows: train.len(),
            features: table.features,
            columns: completed
                .iter()
                .map(|column| train.iter().map(|id| column[*id]).collect())
                .collect(),
            target: train.iter().map(|id| table.target[*id]).collect(),
        };
        let screen = target_screen(&subset.columns, &subset.target);
        let fitted = if let Some(capacity) = capacity {
            let family = match target {
                Target::SparseLinear { .. } => 0,
                Target::SparseGam { .. } => 1,
                Target::Ga2m { .. } => 2,
                Target::Mars { .. } => 3,
                Target::ObliviousTree { .. } => 4,
                Target::CompactNeuralResidual { .. } => 5,
                _ => return None,
            };
            refined_target_model(
                &subset,
                &subset.columns,
                &screen,
                Task::Regression,
                family,
                capacity,
            )
        } else {
            match target {
                Target::SparseLinear { .. } => {
                    fit_linear_target(&subset, &subset.columns, &screen).0
                }
                Target::SparseGam { .. } => {
                    fit_gam_target(&subset, &subset.columns, &screen, Task::Regression)
                }
                Target::Ga2m { .. } => {
                    fit_ga2m_target(&subset, &subset.columns, &screen, Task::Regression)
                }
                Target::Mars { .. } => {
                    fit_mars_target(&subset, &subset.columns, &screen, Task::Regression)
                }
                Target::ObliviousTree { .. } => {
                    fit_oblivious_tree_target(&subset, &subset.columns, &screen, Task::Regression)
                }
                #[cfg(not(feature = "gpu-training"))]
                Target::CompactNeuralResidual { .. } => fit_compact_neural_target_native(
                    &subset,
                    &subset.columns,
                    &screen,
                    Task::Regression,
                ),
                _ => return None,
            }
        };
        for (at, &id) in ids.iter().enumerate().filter(|(at, _)| folds[*at] == fold) {
            let row = completed
                .iter()
                .map(|column| column[id])
                .collect::<Vec<_>>();
            residuals[at] = table.target[id] - evaluate_target(&fitted, &row).0;
        }
    }
    Some((
        Table {
            rows: count,
            features: table.features,
            columns: completed
                .iter()
                .map(|column| ids.iter().map(|id| column[*id]).collect())
                .collect(),
            target: ids.iter().map(|id| table.target[*id]).collect(),
        },
        residuals,
    ))
}

fn residual_rule(target: &Target, rows: usize, used_oof: bool) -> &'static str {
    if used_oof {
        return "outer_fit_subset256_three_fixed_row_folds_gaussian_log_abs_unqualified";
    }
    if rows < 24 {
        return "full_fit_small_sample_unqualified";
    }
    if cfg!(feature = "gpu-training") && matches!(target, Target::CompactNeuralResidual { .. }) {
        return "full_fit_gpu_oof_unavailable_unqualified";
    }
    "full_fit_oof_unavailable_unqualified"
}

#[cfg(test)]
mod residual_tests {
    use super::*;
    #[test]
    fn oof_tree_residuals_do_not_collapse_to_interpolating_training_fit() {
        let x = (0..96).map(|row| row as f32 / 95.0).collect::<Vec<_>>();
        let y = (0..96)
            .map(|row| (splitmix64(row) % 1000) as f32 / 1000.0)
            .collect::<Vec<_>>();
        let table = Table::from_arrays(&x, &y, 96, 1, Task::Regression).unwrap();
        let screen = target_screen(&table.columns, &table.target);
        let target = fit_oblivious_tree_target(&table, &table.columns, &screen, Task::Regression);
        let (view, residuals) = oof_residual_view(&table, &table.columns, &target).unwrap();
        assert_eq!(view.rows, 96);
        assert_eq!(residuals.len(), 96);
        assert!(residuals.iter().all(|r| r.is_finite() && *r > 0.0));
        assert!(residuals.iter().sum::<f32>() / 96.0 > 0.15);
        assert!(residual_rule(&target, 96, true).contains("three_fixed_row_folds"));
        assert_eq!(
            residual_rule(&target, 96, false),
            "full_fit_oof_unavailable_unqualified"
        );
    }
}

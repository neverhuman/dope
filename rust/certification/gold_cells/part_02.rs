

fn selected_rows(table: &Table, indices: &[usize]) -> Table {
    Table {
        rows: indices.len(),
        features: table.features,
        columns: table
            .columns
            .iter()
            .map(|column| indices.iter().map(|&row| column[row]).collect())
            .collect(),
        target: indices.iter().map(|&row| table.target[row]).collect(),
    }
}

fn rare_slice_indices(table: &Table, task: Task, minority: f32) -> Vec<usize> {
    table
        .target
        .iter()
        .enumerate()
        .filter_map(|(row, &value)| {
            let rare = if task == Task::Binary {
                value == minority
            } else {
                value <= 0.1 || value >= 0.9
            };
            rare.then_some(row)
        })
        .collect()
}

fn calibration_error(task: Task, target: &[f32], prediction: &[f32]) -> f64 {
    match task {
        Task::Regression => {
            target
                .iter()
                .zip(prediction)
                .map(|(actual, predicted)| f64::from(actual - predicted))
                .sum::<f64>()
                .abs()
                / target.len().max(1) as f64
        }
        Task::Binary => {
            let mut sums = [(0.0f64, 0.0f64, 0usize); 10];
            for (actual, predicted) in target.iter().zip(prediction) {
                let slot = (f64::from(*predicted).clamp(0.0, 1.0) * 10.0)
                    .floor()
                    .min(9.0) as usize;
                sums[slot].0 += f64::from(*actual);
                sums[slot].1 += f64::from(*predicted);
                sums[slot].2 += 1;
            }
            sums.into_iter()
                .filter(|(_, _, count)| *count > 0)
                .map(|(actual, predicted, count)| {
                    (actual / count as f64 - predicted / count as f64).abs() * count as f64
                })
                .sum::<f64>()
                / target.len().max(1) as f64
        }
    }
}

fn indexed_loss(task: Task, target: &[f32], prediction: &[f32], indices: &[usize]) -> f64 {
    let selected_target: Vec<_> = indices.iter().map(|index| target[*index]).collect();
    let selected_prediction: Vec<_> = indices.iter().map(|index| prediction[*index]).collect();
    loss(task, &selected_target, &selected_prediction)
}

fn subset_retention(
    task: Task,
    target: &[f32],
    null: &[f32],
    trtr: &[f32],
    tstr: &[f32],
    indices: &[usize],
) -> Option<f64> {
    if indices.len() < 10 {
        return None;
    }
    let null_loss = indexed_loss(task, target, null, indices);
    let real_loss = indexed_loss(task, target, trtr, indices);
    let synthetic_loss = indexed_loss(task, target, tstr, indices);
    (null_loss - real_loss >= 0.01 * null_loss.abs())
        .then(|| null_normalized_excess_loss_retention(null_loss, real_loss, synthetic_loss))
}

fn rare_or_tail_retention(
    task: Task,
    target: &[f32],
    null: &[f32],
    trtr: &[f32],
    tstr: &[f32],
) -> Option<f64> {
    let indices: Vec<usize> = match task {
        Task::Binary => {
            let ones = target.iter().filter(|value| **value >= 0.5).count();
            let rare_one = ones <= target.len().saturating_sub(ones);
            target
                .iter()
                .enumerate()
                .filter(|(_, value)| (**value >= 0.5) == rare_one)
                .map(|(index, _)| index)
                .collect()
        }
        Task::Regression => {
            let mut sorted = target.to_vec();
            sorted.sort_by(f32::total_cmp);
            let low = quantile(&sorted, 1, 10);
            let high = quantile(&sorted, 9, 10);
            target
                .iter()
                .enumerate()
                .filter(|(_, value)| **value <= low || **value >= high)
                .map(|(index, _)| index)
                .collect()
        }
    };
    subset_retention(task, target, null, trtr, tstr, &indices)
}

fn subgroup_retention(
    task: Task,
    test: &Table,
    null: &[f32],
    trtr: &[f32],
    tstr: &[f32],
) -> Option<f64> {
    let mut retained = Vec::new();
    for column in 0..test.features.min(32) {
        let values = completed(test, column);
        let mut sorted = values.clone();
        sorted.sort_by(f32::total_cmp);
        for (threshold, lower) in [
            (quantile(&sorted, 1, 5), true),
            (quantile(&sorted, 4, 5), false),
        ] {
            let indices: Vec<_> = values
                .iter()
                .enumerate()
                .filter(|(_, value)| {
                    if lower {
                        **value <= threshold
                    } else {
                        **value >= threshold
                    }
                })
                .map(|(index, _)| index)
                .collect();
            if let Some(value) = subset_retention(task, &test.target, null, trtr, tstr, &indices) {
                retained.push(value);
            }
        }
    }
    retained.into_iter().reduce(f64::min)
}

fn nominal_coverage(
    task: Task,
    training_target: &[f32],
    training_prediction: &[f32],
    test_target: &[f32],
    test_prediction: &[f32],
) -> Option<f64> {
    if test_target.is_empty() {
        return None;
    }
    match task {
        Task::Regression => {
            let sigma = (training_target
                .iter()
                .zip(training_prediction)
                .map(|(actual, predicted)| f64::from(actual - predicted).powi(2))
                .sum::<f64>()
                / training_target.len().max(1) as f64)
                .sqrt();
            Some(
                test_target
                    .iter()
                    .zip(test_prediction)
                    .filter(|(actual, predicted)| {
                        f64::from((**actual - **predicted).abs()) <= 1.96 * sigma
                    })
                    .count() as f64
                    / test_target.len() as f64,
            )
        }
        Task::Binary => {
            let mut bins = [(0.0f64, 0.0f64, 0usize); 10];
            for (actual, predicted) in test_target.iter().zip(test_prediction) {
                let slot = (f64::from(*predicted).clamp(0.0, 1.0) * 10.0)
                    .floor()
                    .min(9.0) as usize;
                bins[slot].0 += f64::from(*actual);
                bins[slot].1 += f64::from(*predicted);
                bins[slot].2 += 1;
            }
            let supported: Vec<_> = bins
                .into_iter()
                .filter(|(_, _, count)| *count >= 5)
                .collect();
            if supported.is_empty() {
                None
            } else {
                Some(
                    supported
                        .iter()
                        .filter(|(actual, predicted, count)| {
                            let observed = actual / *count as f64;
                            let expected = predicted / *count as f64;
                            let radius = 1.96
                                * (expected * (1.0 - expected) / *count as f64)
                                    .max(1e-9)
                                    .sqrt();
                            (observed - expected).abs() <= radius
                        })
                        .count() as f64
                        / supported.len() as f64,
                )
            }
        }
    }
}

fn query_p95_normalized_error(real: &Table, synthetic: &Table) -> f64 {
    let mut errors = Vec::new();
    for column in 0..real.features.min(64) {
        let real_observed: Vec<_> = real.columns[column]
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        let synthetic_observed: Vec<_> = synthetic.columns[column]
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        errors.push(
            (real_observed.len() as f64 / real.rows.max(1) as f64
                - synthetic_observed.len() as f64 / synthetic.rows.max(1) as f64)
                .abs(),
        );
        for threshold in (1..10).map(|index| index as f32 / 10.0) {
            let real_rate = real_observed
                .iter()
                .filter(|value| **value <= threshold)
                .count() as f64
                / real_observed.len().max(1) as f64;
            let synthetic_rate = synthetic_observed
                .iter()
                .filter(|value| **value <= threshold)
                .count() as f64
                / synthetic_observed.len().max(1) as f64;
            errors.push((real_rate - synthetic_rate).abs() / real_rate.max(0.05));
        }
    }
    errors.sort_by(f64::total_cmp);
    errors
        .get((errors.len().saturating_sub(1) * 95).div_ceil(100))
        .copied()
        .unwrap_or(0.0)
}

fn type_i_error(real: &Table, synthetic: &Table) -> f64 {
    let mut supported_nulls = 0usize;
    let mut false_rejections = 0usize;
    for column in 0..real.features.min(64) {
        let real_correlation = correlation(&completed(real, column), &real.target).abs();
        let synthetic_correlation =
            correlation(&completed(synthetic, column), &synthetic.target).abs();
        let real_null_radius = 1.0 / (real.rows.saturating_sub(3).max(1) as f64).sqrt();
        if real_correlation <= real_null_radius {
            supported_nulls += 1;
            let synthetic_rejection_radius =
                1.96 / (synthetic.rows.saturating_sub(3).max(1) as f64).sqrt();
            false_rejections += (synthetic_correlation > synthetic_rejection_radius) as usize;
        }
    }
    false_rejections as f64 / supported_nulls.max(1) as f64
}

fn row_key(table: &Table, row: usize) -> Vec<u32> {
    (0..=table.features)
        .map(|column| {
            let value = if column == table.features {
                table.target[row]
            } else {
                table.columns[column][row]
            };
            if value.is_nan() {
                f32::NAN.to_bits()
            } else {
                value.to_bits()
            }
        })
        .collect()
}

fn exact_copy_count(reference: &Table, synthetic: &Table) -> usize {
    let rows: HashSet<Vec<u32>> = (0..reference.rows)
        .map(|row| row_key(reference, row))
        .collect();
    (0..synthetic.rows)
        .filter(|row| rows.contains(&row_key(synthetic, *row)))
        .count()
}

fn missingness_key(table: &Table, row: usize) -> Vec<u64> {
    let mut key = vec![0u64; table.features.div_ceil(64)];
    for column in 0..table.features {
        if table.columns[column][row].is_nan() {
            key[column / 64] |= 1 << (column % 64);
        }
    }
    key
}

fn row_value(table: &Table, row: usize, column: usize) -> f32 {
    if column == table.features {
        table.target[row]
    } else {
        table.columns[column][row]
    }
}

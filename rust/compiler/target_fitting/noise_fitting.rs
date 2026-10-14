fn fit_noise(
    table: &Table,
    completed: &[Vec<f32>],
    _screen: &[(usize, f32)],
    target: &Target,
    task: Task,
) -> (Noise, &'static str) {
    let mut linear = Vec::with_capacity(table.rows);
    let mut probabilities = Vec::with_capacity(table.rows);
    for row in 0..table.rows {
        let values: Vec<_> = completed.iter().map(|column| column[row]).collect();
        let (prediction, logistic) = evaluate_target(target, &values);
        linear.push(prediction);
        probabilities.push(if logistic {
            sigmoid(f64::from(prediction)) as f32
        } else {
            prediction.clamp(0.0, 1.0)
        });
    }
    if task == Task::Binary {
        let mut order: Vec<_> = (0..table.rows).collect();
        order.sort_by(|left, right| {
            probabilities[*left]
                .total_cmp(&probabilities[*right])
                .then_with(|| left.cmp(right))
        });
        let bins = 9.min(table.rows.max(2));
        let mut knots = Vec::with_capacity(bins);
        let mut calibrated = Vec::with_capacity(bins);
        for bin in 0..bins {
            let start = bin * table.rows / bins;
            let end = ((bin + 1) * table.rows / bins)
                .max(start + 1)
                .min(table.rows);
            let rows = &order[start..end];
            knots.push(rows.iter().map(|row| probabilities[*row]).sum::<f32>() / rows.len() as f32);
            calibrated
                .push(rows.iter().map(|row| table.target[*row]).sum::<f32>() / rows.len() as f32);
        }
        let epsilon = 1e-4;
        for index in 0..bins {
            let remaining = bins - index - 1;
            knots[index] =
                knots[index].clamp(index as f32 * epsilon, 1.0 - remaining as f32 * epsilon);
            calibrated[index] =
                calibrated[index].clamp(index as f32 * epsilon, 1.0 - remaining as f32 * epsilon);
            if index > 0 {
                knots[index] = knots[index].max(knots[index - 1] + epsilon);
                calibrated[index] = calibrated[index].max(calibrated[index - 1] + epsilon);
            }
        }
        return (
            Noise::IsotonicCalibration {
                knots,
                probabilities: calibrated,
            },
            "full_fit_binary_calibration_unqualified",
        );
    }
    let oof = oof_residual_view(table, completed, target);
    let fitting_rule = residual_rule(target, table.rows, oof.is_some());
    let (residual_table, residuals) = oof.unwrap_or_else(|| {
        (
            table.clone(),
            table
                .target
                .iter()
                .zip(&linear)
                .map(|(actual, prediction)| (actual - prediction).abs().max(1e-5))
                .collect(),
        )
    });
    let mut sorted = residuals.clone();
    sorted.sort_by(f32::total_cmp);
    let minimum_sigma = quantile(&sorted, 0.1).clamp(1e-4, 1.0);
    let maximum_sigma = quantile(&sorted, 0.9)
        .clamp(minimum_sigma, 1.0)
        .max(minimum_sigma);
    let residual_screen = target_screen(&residual_table.columns, &residuals);
    let selected: Vec<_> = residual_screen
        .iter()
        .take(4)
        .map(|(feature, _)| *feature)
        .collect();
    let basis: Vec<_> = selected
        .iter()
        .map(|feature| residual_table.columns[*feature].clone())
        .collect();
    // E[log |N(0,1)|] = -(EulerGamma + log 2)/2. This correction
    // names a Gaussian reference; it is not a calibration certificate.
    let log_residuals: Vec<_> = residuals
        .iter()
        .map(|value| value.ln() + 0.635_181_4)
        .collect();
    let (intercept, coefficients) = fit_basis_coefficients(&log_residuals, &basis, false);
    (
        Noise::Heteroscedastic {
            intercept,
            terms: selected
                .into_iter()
                .zip(coefficients)
                .filter(|(_, coefficient)| coefficient.abs() >= 1e-5)
                .map(|(feature, coefficient)| LinearTerm {
                    feature: feature as u32,
                    coefficient,
                })
                .collect(),
            minimum_sigma,
            maximum_sigma,
        },
        fitting_rule,
    )
}

#[cfg(test)]
mod declared_task_tests {
    use super::*;

    #[test]
    fn binary_valued_regression_keeps_regression_heads_and_noise() {
        let x = (0..64).map(|row| row as f32 / 63.0).collect::<Vec<_>>();
        let y = x
            .iter()
            .map(|v| u8::from(*v >= 0.5) as f32)
            .collect::<Vec<_>>();
        let table = Table::from_arrays(&x, &y, 64, 1, Task::Regression).unwrap();
        let screen = vec![(0, 1.0)];
        for task in [Task::Regression, Task::Binary] {
            let heads = [
                fit_gam_target(&table, &table.columns, &screen, task),
                fit_ga2m_target(&table, &table.columns, &screen, task),
                fit_mars_target(&table, &table.columns, &screen, task),
                fit_oblivious_tree_target(&table, &table.columns, &screen, task),
            ];
            for head in heads {
                let (_, logistic) = evaluate_target(&head, &[0.5]);
                assert_eq!(logistic, task == Task::Binary);
                let (noise, _) = fit_noise(&table, &table.columns, &screen, &head, task);
                assert_eq!(
                    matches!(noise, Noise::IsotonicCalibration { .. }),
                    task == Task::Binary
                );
            }
            #[cfg(not(feature = "gpu-training"))]
            {
                let head = fit_compact_neural_target_native(&table, &table.columns, &screen, task);
                assert_eq!(evaluate_target(&head, &[0.5]).1, task == Task::Binary);
            }
        }
    }
}

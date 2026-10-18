#[cfg(test)]
mod joint_ablation_tests {
    use super::*;

    fn ridge_xtx_xty_oracle(columns: &[Vec<f32>], target: &[f32], lambda: f64) -> (f64, Vec<f64>) {
        let rows = target.len();
        let width = columns.len();
        let target_mean = target.iter().map(|value| f64::from(*value)).sum::<f64>() / rows as f64;
        let means: Vec<f64> = columns
            .iter()
            .map(|column| column.iter().map(|value| f64::from(*value)).sum::<f64>() / rows as f64)
            .collect();
        let mut xtx = vec![0.0; width * width];
        let mut xty = vec![0.0; width];
        for left in 0..width {
            for (row, actual) in target.iter().enumerate() {
                let x_left = f64::from(columns[left][row]) - means[left];
                xty[left] += x_left * (f64::from(*actual) - target_mean);
                for right in 0..width {
                    xtx[left * width + right] +=
                        x_left * (f64::from(columns[right][row]) - means[right]);
                }
            }
            xtx[left * width + left] += lambda;
        }
        let coefficients = solve_system(xtx, xty, width);
        let intercept = target_mean
            - coefficients.iter().zip(&means).map(|(coefficient, mean)| coefficient * mean).sum::<f64>();
        (intercept, coefficients)
    }

    #[test]
    fn exact_ridge_normal_equations_match_the_unquantized_linear_fit() {
        let columns = vec![
            vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            vec![1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
        ];
        let target: Vec<f32> = columns[0]
            .iter()
            .zip(&columns[1])
            .map(|(left, right)| 0.75 + 1.25 * left - 0.5 * right)
            .collect();
        let lambda = 1.0;
        let (intercept, coefficients) = ridge_xtx_xty_oracle(&columns, &target, lambda);
        let rows = target.len();
        let residuals: Vec<f64> = (0..rows)
            .map(|row| {
                f64::from(target[row]) - intercept
                    - columns.iter().enumerate().map(|(feature, column)| {
                        coefficients[feature] * f64::from(column[row])
                    }).sum::<f64>()
            })
            .collect();
        assert!(residuals.iter().sum::<f64>().abs() < 1e-12, "intercept is unpenalized");
        for feature in 0..columns.len() {
            let gradient = (0..rows)
                .map(|row| f64::from(columns[feature][row]) * residuals[row])
                .sum::<f64>() - lambda * coefficients[feature];
            assert!(gradient.abs() < 1e-11, "ridge normal equation residual");
        }

        let table = Table { rows, features: columns.len(), columns: columns.clone(), target: target.clone() };
        let fitted = fit_linear_target(&table, &columns, &[(0, 1.0), (1, 1.0)]).0;
        let Target::SparseLinear { intercept: fitted_intercept, terms } = fitted else {
            panic!("linear fit control")
        };
        assert!((f64::from(fitted_intercept) - intercept).abs() < 1e-6);
        for (feature, coefficient) in coefficients.iter().enumerate() {
            let fitted = terms.iter().find(|term| term.feature == feature as u32).unwrap();
            assert!((f64::from(fitted.coefficient) - coefficient).abs() < 1e-6);
        }
    }

    #[test]
    fn joint_ablation_design_freezes_controls_and_keeps_unsupported_cells_visible() {
        let plan: serde_json::Value = serde_json::from_str(include_str!(
            "../../research/transfer/ablations/joint.json"
        ))
        .unwrap();
        assert_eq!(plan["status"], "pilot_complete_development_blocked");
        assert_eq!(plan["design"]["precision"], "float64");
        assert_eq!(plan["design"]["byte_budget"], 128);
        assert_eq!(plan["design"]["seed_schedule"].as_array().unwrap().len(), 3);
        assert_eq!(plan["candidates"].as_array().unwrap().len(), 8);
        assert!(plan["candidates"].as_array().unwrap().iter().any(|item| {
            item["id"] == "circuit" && item["status"] == "unsupported_preserved"
        }));
        assert!(plan["candidates"].as_array().unwrap().iter().any(|item| {
            item["id"] == "low_rank_residual" && item["status"] == "unsupported_preserved"
        }));
        assert_eq!(plan["claims"]["privacy"], false);
        assert_eq!(plan["claims"]["generalization"], false);
    }
}

fn selected_noise(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
    target: &Target,
    task: Task,
    controls: (Option<CapacityRefinement>, Option<ResidualProfile>),
) -> Result<(Noise, &'static str)> {
    let (capacity, profile) = controls;
    let Some(profile) = profile else {
        return Ok(match capacity {
            None => fit_noise(table, completed, screen, target, task),
            Some(c) => fit_noise_bounded(
                table,
                completed,
                screen,
                target,
                task,
                c.residual_support,
                capacity,
            ),
        });
    };
    if profile == ResidualProfile::NoNoise {
        return Ok((
            Noise::ResidualV1 {
                distribution: ResidualV1::NoNoise,
            },
            "explicit_no_noise_development_unqualified_v1",
        ));
    }
    if table.rows < profile.minimum_rows() {
        return Err(DopeError::Unsupported(
            "development_residual_minimum_rows_not_met".into(),
        ));
    }
    if profile == ResidualProfile::GaussianHomoscedastic {
        let (noise, _) = fit_noise_bounded(table, completed, screen, target, task, 0, capacity);
        return Ok((
            noise,
            "outer_fit_subset256_three_fixed_row_folds_homoscedastic_gaussian_unqualified_v1",
        ));
    }
    if profile == ResidualProfile::GaussianHeteroscedastic {
        let (noise, _) = match capacity {
            None => fit_noise(table, completed, screen, target, task),
            Some(c) => fit_noise_bounded(
                table,
                completed,
                screen,
                target,
                task,
                c.residual_support,
                capacity,
            ),
        };
        return Ok((
            noise,
            "outer_fit_subset256_three_fixed_row_folds_heteroscedastic_gaussian_unqualified_v1",
        ));
    }
    let (_, residuals) = oof_signed_residual_view_capacity(table, completed, target, capacity)
        .ok_or_else(|| DopeError::Unsupported("development_residual_oof_unavailable".into()))?;
    if residuals.iter().any(|v| !v.is_finite()) {
        return Err(DopeError::Data("development_residual_fit_invalid".into()));
    }
    let moments = |values: &[f32]| {
        let mean = values.iter().map(|v| f64::from(*v)).sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|v| (f64::from(*v) - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        (mean as f32, variance.sqrt() as f32)
    };
    let distribution = match profile {
        ResidualProfile::NoNoise => unreachable!("handled explicit no-noise profile"),
        ResidualProfile::GaussianHomoscedastic | ResidualProfile::GaussianHeteroscedastic => {
            unreachable!("handled explicit Gaussian profiles")
        }
        ResidualProfile::StudentT4 => {
            let (location, sigma) = moments(&residuals);
            // A t4 variable has variance2: match the OOF second moment. Fixed
            // degrees of freedom are an operator choice, not fitted tail evidence.
            ResidualV1::StudentT4 {
                location,
                scale: sigma / 2.0_f32.sqrt(),
            }
        }
        ResidualProfile::GaussianMixture2 => {
            let mut sorted = residuals;
            sorted.sort_by(f32::total_cmp);
            let split = sorted.len() / 2;
            let (left, right) = sorted.split_at(split);
            let (a, sa) = moments(left);
            let (b, sb) = moments(right);
            // Deterministic lower/upper half moment fit, not an EM likelihood
            // optimum. This bounded approximation is named in the fitting rule.
            ResidualV1::GaussianMixture2 {
                weight: split as f32 / sorted.len() as f32,
                locations: [a, b],
                scales: [sa, sb],
            }
        }
        ResidualProfile::MonotoneQuantile17 => {
            let mut sorted = residuals;
            sorted.sort_by(f32::total_cmp);
            let values = (0..17)
                .map(|at| {
                    let position = at as f64 * (sorted.len() - 1) as f64 / 16.0;
                    let lower = position.floor() as usize;
                    let upper = position.ceil() as usize;
                    sorted[lower]
                        + (sorted[upper] - sorted[lower]) * (position - lower as f64) as f32
                })
                .collect();
            ResidualV1::MonotoneQuantile17 { values }
        }
    };
    distribution.validate().map_err(DopeError::Data)?;
    let rule = match profile {
        ResidualProfile::GaussianMixture2 => {
            "outer_fit_subset256_three_fixed_row_folds_half_moment_mixture2_unqualified_v1"
        }
        ResidualProfile::StudentT4 => {
            "outer_fit_subset256_three_fixed_row_folds_fixed_t4_second_moment_unqualified_v1"
        }
        ResidualProfile::MonotoneQuantile17 => {
            "outer_fit_subset256_three_fixed_row_folds_bounded_quantile17_unqualified_v1"
        }
        ResidualProfile::NoNoise => unreachable!("handled explicit no-noise profile"),
        ResidualProfile::GaussianHomoscedastic | ResidualProfile::GaussianHeteroscedastic => {
            unreachable!("handled explicit Gaussian profiles")
        }
    };
    Ok((Noise::ResidualV1 { distribution }, rule))
}

#[cfg(test)]
mod residual_study_fixtures {
    use super::*;

    type Fixture = (Table, Vec<Vec<f32>>, Vec<(usize, f32)>, Target);

    fn rare_tail_fixture(rows: usize) -> Fixture {
        let x = (0..rows)
            .map(|row| row as f32 / (rows - 1).max(1) as f32)
            .collect::<Vec<_>>();
        let y = x
            .iter()
            .enumerate()
            .map(|(row, value)| {
                let signed = if row % 2 == 0 { 1.0 } else { -1.0 };
                let scale = if *value >= 0.9 { 1.25 } else { 0.08 };
                0.2 + 1.7 * value + signed * scale
            })
            .collect::<Vec<_>>();
        let table = Table::from_arrays(&x, &y, rows, 1, Task::Regression).unwrap();
        let screen = target_screen(&table.columns, &table.target);
        let target = fit_linear_target(&table, &table.columns, &screen).0;
        (table, vec![x], screen, target)
    }

    #[test]
    fn one_priced_fixture_constructs_every_preregistered_candidate_unqualified() {
        let (table, completed, screen, target) = rare_tail_fixture(96);
        let candidates = [
            ResidualProfile::NoNoise,
            ResidualProfile::GaussianHomoscedastic,
            ResidualProfile::GaussianHeteroscedastic,
            ResidualProfile::StudentT4,
            ResidualProfile::GaussianMixture2,
            ResidualProfile::MonotoneQuantile17,
        ];
        for profile in candidates {
            let (noise, rule) = selected_noise(
                &table,
                &completed,
                &screen,
                &target,
                Task::Regression,
                (None, Some(profile)),
            )
            .unwrap();
            assert!(rule.contains("unqualified"), "{}", profile.as_str());
            match &noise {
                Noise::ResidualV1 { distribution } => distribution.validate().unwrap(),
                Noise::Homoscedastic { sigma } => assert!(sigma.is_finite() && *sigma >= 0.0),
                Noise::Heteroscedastic {
                    intercept,
                    minimum_sigma,
                    maximum_sigma,
                    ..
                } => {
                    assert!(intercept.is_finite());
                    assert!(0.0 <= *minimum_sigma && minimum_sigma <= maximum_sigma);
                }
                _ => panic!("unexpected candidate noise for {}", profile.as_str()),
            }
        }
    }

    #[test]
    fn residual_scale_candidates_refuse_fewer_than_three_fold_support_rows() {
        let (table, completed, screen, target) = rare_tail_fixture(23);
        for profile in [
            ResidualProfile::GaussianHomoscedastic,
            ResidualProfile::GaussianHeteroscedastic,
            ResidualProfile::StudentT4,
            ResidualProfile::GaussianMixture2,
            ResidualProfile::MonotoneQuantile17,
        ] {
            let error = selected_noise(
                &table,
                &completed,
                &screen,
                &target,
                Task::Regression,
                (None, Some(profile)),
            )
            .unwrap_err();
            assert!(matches!(
                error,
                DopeError::Unsupported(reason)
                    if reason == "development_residual_minimum_rows_not_met"
            ));
        }
    }

    #[test]
    fn critical_tail_stop_rule_overrides_an_improved_global_loss() {
        #[derive(Clone, Copy)]
        struct Evidence {
            global_mse_ratio: f64,
            transfer_lcb: f64,
            driver_lcb: f64,
            tail_transfer_lcb: f64,
            tail_loss_ratio: f64,
            coverage_error: f64,
            rare_slice_support: usize,
            artifact_bytes: usize,
        }
        fn passes(e: Evidence) -> bool {
            e.global_mse_ratio <= 1.0
                && e.transfer_lcb >= 0.9
                && e.driver_lcb >= 0.8
                && e.tail_transfer_lcb >= 0.8
                && e.tail_loss_ratio <= 1.1
                && e.coverage_error <= 0.05
                && e.rare_slice_support >= 100
                && e.artifact_bytes <= 10_240
        }
        let globally_better_but_tail_failed = Evidence {
            global_mse_ratio: 0.90,
            transfer_lcb: 0.95,
            driver_lcb: 0.85,
            tail_transfer_lcb: 0.79,
            tail_loss_ratio: 1.0,
            coverage_error: 0.01,
            rare_slice_support: 120,
            artifact_bytes: 2_000,
        };
        assert!(!passes(globally_better_but_tail_failed));
    }
}

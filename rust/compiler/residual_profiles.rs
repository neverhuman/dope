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
            "explicit_no_noise_development_v1",
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
    };
    Ok((Noise::ResidualV1 { distribution }, rule))
}

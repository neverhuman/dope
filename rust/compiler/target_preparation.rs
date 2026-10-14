type PreparedTargets = Vec<(&'static str, Target, (Noise, &'static str))>;

fn prepare_symbolic_targets(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
    task: Task,
    options: &CompileOptions,
    seed: u64,
    restricted_profile: Option<(usize, u8, usize, u8)>,
) -> Result<PreparedTargets> {
    let (target_model, noise, _fitted_utility) = match task {
        Task::Regression => fit_linear_target(table, completed, screen),
        Task::Binary => fit_logistic_target(table, completed, screen),
    };
    let baseline_name = if task == Task::Regression {
        "sparse_linear"
    } else {
        "sparse_logistic"
    };
    let baseline_rule = if task == Task::Regression {
        "full_fit_homoscedastic_baseline_unqualified"
    } else {
        "full_fit_binary_calibration_unqualified"
    };
    let (noise, baseline_rule) = if options.development_residual.is_some() {
        selected_noise(
            table,
            completed,
            screen,
            &target_model,
            task,
            (None, options.development_residual),
        )?
    } else {
        (noise, baseline_rule)
    };
    let mut target_models = vec![(
        baseline_name,
        target_model.clone(),
        (noise.clone(), baseline_rule),
    )];
    let gam = fit_gam_target(table, completed, screen, task);
    target_models.push((
        "sparse_gam",
        gam.clone(),
        selected_noise(
            table,
            completed,
            screen,
            &gam,
            task,
            (None, options.development_residual),
        )?,
    ));
    let ga2m = fit_ga2m_target(table, completed, screen, task);
    target_models.push((
        "ga2m",
        ga2m.clone(),
        selected_noise(
            table,
            completed,
            screen,
            &ga2m,
            task,
            (None, options.development_residual),
        )?,
    ));
    let mars = fit_mars_target(table, completed, screen, task);
    target_models.push((
        "mars",
        mars.clone(),
        selected_noise(
            table,
            completed,
            screen,
            &mars,
            task,
            (None, options.development_residual),
        )?,
    ));
    let oblivious_tree = fit_oblivious_tree_target(table, completed, screen, task);
    target_models.push((
        "oblivious_tree",
        oblivious_tree.clone(),
        selected_noise(
            table,
            completed,
            screen,
            &oblivious_tree,
            task,
            (None, options.development_residual),
        )?,
    ));
    let compact_neural = if restricted_profile.is_some_and(|profile| profile.2 != 5) {
        target_model.clone()
    } else {
        fit_compact_neural_target(table, completed, screen, seed, task)?
    };
    target_models.push((
        "compact_neural_residual",
        compact_neural.clone(),
        selected_noise(
            table,
            completed,
            screen,
            &compact_neural,
            task,
            (None, options.development_residual),
        )?,
    ));
    Ok(target_models)
}

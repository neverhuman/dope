#[cfg(feature = "gpu-training")]
fn fit_compact_neural_target(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
    seed: u64,
    task: Task,
) -> Result<Target> {
    use tch::nn::{Module, OptimizerConfig};

    let training_device = compact_neural_training_device()?;
    let device = match training_device {
        ResearchTargetDevice::Cuda => {
            if !tch::Cuda::is_available() {
                return Err(DopeError::Unsupported(
                    "frozen neural candidate requires CUDA libtorch training".into(),
                ));
            }
            if std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref() != Ok(":4096:8") {
                return Err(DopeError::Unsupported(
                    "frozen neural candidate requires CUBLAS_WORKSPACE_CONFIG=:4096:8".into(),
                ));
            }
            tch::Device::Cuda(0)
        }
        ResearchTargetDevice::Cpu => {
            if std::env::var("DOPE_RESEARCH_TARGET_PROFILE").as_deref()
                != Ok("features12_steps2048")
            {
                return Err(DopeError::Unsupported(
                    "CPU research training requires features12_steps2048".into(),
                ));
            }
            tch::Device::Cpu
        }
    };
    let train = || {
        #[cfg(feature = "gpu-research-training")]
        let ablation = match std::env::var("DOPE_RESEARCH_TARGET_PROFILE") {
            Ok(profile) if profile.starts_with("grid_") => {
                Some(bounded_research_ablation_settings(&profile)?)
            }
            _ => None,
        };
        #[cfg(feature = "gpu-research-training")]
        let (feature_limit, hidden_width, optimizer_steps) = match ablation {
            Some(settings) => (
                settings.feature_limit,
                settings.hidden_width,
                settings.optimizer_steps,
            ),
            None => compact_neural_training_settings(screen.len())?,
        };
        #[cfg(not(feature = "gpu-research-training"))]
        let (feature_limit, hidden_width, optimizer_steps) =
            compact_neural_training_settings(screen.len())?;
        let selected = screen
            .iter()
            .take(feature_limit)
            .map(|(feature, _)| *feature)
            .collect::<Vec<_>>();
        let mut rows = Vec::with_capacity(table.rows * selected.len());
        for row in 0..table.rows {
            for &feature in &selected {
                rows.push(completed[feature][row]);
            }
        }
        let x = tch::Tensor::from_slice(&rows)
            .view([table.rows as i64, selected.len() as i64])
            .to_device(device);
        let y = tch::Tensor::from_slice(&table.target)
            .view([table.rows as i64, 1])
            .to_device(device);
        #[cfg(feature = "gpu-research-training")]
        if let Some(settings) = ablation {
            if settings.hidden_width == 0 {
                return fit_linear_gpu_ablation(
                    table,
                    completed,
                    &selected,
                    &x,
                    &y,
                    settings.optimizer_steps,
                    settings.readout_refit,
                    task,
                );
            }
        }
        let store = tch::nn::VarStore::new(device);
        let root = store.root();
        let hidden = tch::nn::linear(
            &root / "hidden",
            selected.len() as i64,
            hidden_width as i64,
            Default::default(),
        );
        let residual = tch::nn::linear(
            &root / "residual",
            hidden_width as i64,
            1,
            Default::default(),
        );
        let skip = tch::nn::linear(&root / "skip", selected.len() as i64, 1, Default::default());
        let forward = |input: &tch::Tensor| {
            skip.forward(input) + residual.forward(&hidden.forward(input).tanh())
        };
        let logistic = task == Task::Binary;
        let mut optimizer = tch::nn::AdamW::default()
            .build(&store, 2e-3)
            .map_err(|error| DopeError::Data(format!("neural candidate optimizer: {error}")))?;
        let mut loss_log = open_research_loss_log(table, logistic, &selected, completed, device)?;
        for step in 0..optimizer_steps {
            let prediction = forward(&x);
            let loss = if logistic {
                prediction.binary_cross_entropy_with_logits::<&tch::Tensor>(
                    &y,
                    None,
                    None,
                    tch::Reduction::Mean,
                )
            } else {
                prediction.mse_loss(&y, tch::Reduction::Mean)
            };
            let train_loss = loss_log.as_ref().map(|_| loss.detach().double_value(&[]));
            optimizer.backward_step_clip(&loss, 5.0);
            if let Some((file, vx, vy)) = loss_log.as_mut() {
                let validation_loss = tch::no_grad(|| {
                    let prediction = forward(vx);
                    let held = if logistic {
                        prediction.binary_cross_entropy_with_logits::<&tch::Tensor>(
                            vy,
                            None,
                            None,
                            tch::Reduction::Mean,
                        )
                    } else {
                        prediction.mse_loss(vy, tch::Reduction::Mean)
                    };
                    held.double_value(&[])
                });
                use std::io::Write;
                file.write_all(
                    research_loss_record(step, train_loss.expect("train loss"), validation_loss)
                        .as_bytes(),
                )
                .map_err(|error| DopeError::Data(format!("research loss log: {error}")))?;
            }
        }
        if let Some((file, _, _)) = loss_log.as_mut() {
            use std::io::Write;
            file.flush()
                .map_err(|error| DopeError::Data(format!("research loss log: {error}")))?;
        }
        let hidden_biases = tensor_values(
            hidden.bs.as_ref().expect("linear bias").shallow_clone(),
            "neural hidden bias export",
        )?;
        let input_weights = tensor_values(hidden.ws.shallow_clone(), "neural input export")?;
        #[cfg(feature = "gpu-research-training")]
        if ablation.is_some_and(|settings| !settings.readout_refit) {
            let skip_bias = tensor_values(
                skip.bs.as_ref().expect("linear bias").shallow_clone(),
                "neural skip bias export",
            )?[0];
            let residual_bias = tensor_values(
                residual.bs.as_ref().expect("linear bias").shallow_clone(),
                "neural residual bias export",
            )?[0];
            let linear_weights =
                tensor_values(skip.ws.shallow_clone(), "neural skip weight export")?;
            let output_weights =
                tensor_values(residual.ws.shallow_clone(), "neural output weight export")?;
            return Ok(Target::CompactNeuralResidual {
                intercept: skip_bias + residual_bias,
                logistic,
                linear_terms: selected
                    .iter()
                    .copied()
                    .zip(linear_weights)
                    .map(|(feature, coefficient)| LinearTerm {
                        feature: feature as u32,
                        coefficient,
                    })
                    .collect(),
                hidden_features: selected.into_iter().map(|feature| feature as u32).collect(),
                hidden_width: hidden_width as u8,
                input_weights,
                hidden_biases,
                output_weights,
            });
        }
        // The short GPU fit learns the hidden basis. Refit its readout with the
        // same regularized solver as the native candidate: an unfinished Adam
        // intercept can otherwise dominate targets with small variance.
        let mut basis: Vec<Vec<f32>> = selected
            .iter()
            .map(|feature| completed[*feature].clone())
            .collect();
        basis.extend((0..hidden_width).map(|unit| {
            (0..table.rows)
                .map(|row| {
                    (hidden_biases[unit]
                        + selected
                            .iter()
                            .enumerate()
                            .map(|(input, feature)| {
                                input_weights[unit * selected.len() + input]
                                    * completed[*feature][row]
                            })
                            .sum::<f32>())
                    .tanh()
                })
                .collect()
        }));
        let (intercept, coefficients) = fit_basis_coefficients(&table.target, &basis, logistic);
        let (linear_weights, output_weights) = coefficients.split_at(selected.len());
        Ok(Target::CompactNeuralResidual {
            intercept,
            logistic,
            linear_terms: selected
                .iter()
                .copied()
                .zip(linear_weights.iter().copied())
                .filter(|(_, coefficient)| coefficient.abs() >= 1e-5)
                .map(|(feature, coefficient)| LinearTerm {
                    feature: feature as u32,
                    coefficient,
                })
                .collect(),
            hidden_features: selected.into_iter().map(|feature| feature as u32).collect(),
            hidden_width: hidden_width as u8,
            input_weights,
            hidden_biases,
            output_weights: output_weights.to_vec(),
        })
    };
    match training_device {
        ResearchTargetDevice::Cuda => crate::libtorch::with_seeded_libtorch(seed, train),
        #[cfg(feature = "gpu-research-training")]
        ResearchTargetDevice::Cpu => crate::libtorch::with_seeded_cpu_libtorch(seed, train),
        #[cfg(not(feature = "gpu-research-training"))]
        ResearchTargetDevice::Cpu => Err(DopeError::Unsupported(
            "CPU research training requires gpu-research-training".into(),
        )),
    }
}

#[cfg(not(feature = "gpu-training"))]
fn fit_compact_neural_target(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
    _seed: u64,
    task: Task,
) -> Result<Target> {
    if std::env::var_os("DOPE_RESEARCH_TARGET_DEVICE").is_some() {
        return Err(DopeError::Unsupported(
            "research device selection requires gpu-research-training".into(),
        ));
    }
    Ok(fit_compact_neural_target_native(
        table, completed, screen, task,
    ))
}

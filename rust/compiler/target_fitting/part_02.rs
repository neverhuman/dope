#[cfg(any(test, feature = "gpu-research-training"))]
fn bounded_research_training_settings(profile: &str) -> Result<(usize, usize, usize)> {
    let (features, width, steps) = match profile {
        "features12_steps512" => (12, 16, 512),
        "features12_steps2048" => (12, 16, 2048),
        "features24_steps512" => (24, 16, 512),
        "features24_steps2048" => (24, 16, 2048),
        "features12_steps8192" => (12, 16, 8192),
        "features16_steps8192" => (16, 16, 8192),
        "features12_width8_steps8192" => (12, 8, 8192),
        "features24_width8_steps8192" => (24, 8, 8192),
        _ => return Err(DopeError::Unsupported("unknown bounded GPU research profile".into())),
    };
    Ok((features, width, steps))
}

#[cfg(any(test, feature = "gpu-research-training"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResearchAblationSettings {
    feature_limit: usize,
    hidden_width: usize,
    optimizer_steps: usize,
    readout_refit: bool,
}

#[cfg(any(test, feature = "gpu-research-training"))]
fn bounded_research_ablation_settings(profile: &str) -> Result<ResearchAblationSettings> {
    let invalid = || DopeError::Unsupported("unknown bounded GPU research ablation".into());
    let fields = profile.split('_').collect::<Vec<_>>();
    let ["grid", features, width, steps, readout] = fields.as_slice() else {
        return Err(invalid());
    };
    let feature_limit = match *features {
        "features6" => 6,
        "features12" => 12,
        "features24" => 24,
        _ => return Err(invalid()),
    };
    let hidden_width = match *width {
        "widthlinear" => 0,
        "width8" => 8,
        "width16" => 16,
        _ => return Err(invalid()),
    };
    let optimizer_steps = match *steps {
        "steps512" => 512,
        "steps2048" => 2048,
        "steps8192" => 8192,
        _ => return Err(invalid()),
    };
    let readout_refit = match *readout {
        "readouton" => true,
        "readoutoff" => false,
        _ => return Err(invalid()),
    };
    Ok(ResearchAblationSettings {
        feature_limit,
        hidden_width,
        optimizer_steps,
        readout_refit,
    })
}

#[cfg(feature = "gpu-research-training")]
fn fit_linear_gpu_ablation(
    table: &Table,
    completed: &[Vec<f32>],
    selected: &[usize],
    x: &tch::Tensor,
    y: &tch::Tensor,
    optimizer_steps: usize,
    readout_refit: bool,
) -> Result<Target> {
    use tch::nn::{Module, OptimizerConfig};

    let store = tch::nn::VarStore::new(x.device());
    let skip = tch::nn::linear(
        &store.root() / "skip",
        selected.len() as i64,
        1,
        Default::default(),
    );
    let logistic = table.target.iter().all(|value| *value == 0.0 || *value == 1.0);
    let mut optimizer = tch::nn::AdamW::default()
        .build(&store, 2e-3)
        .map_err(|error| DopeError::Data(format!("linear ablation optimizer: {error}")))?;
    let mut loss_log = open_research_loss_log(table, logistic, selected, completed, x.device())?;
    for step in 0..optimizer_steps {
        let prediction = skip.forward(x);
        let loss = if logistic {
            prediction.binary_cross_entropy_with_logits::<&tch::Tensor>(
                y,
                None,
                None,
                tch::Reduction::Mean,
            )
        } else {
            prediction.mse_loss(y, tch::Reduction::Mean)
        };
        let train_loss = loss_log.as_ref().map(|_| loss.detach().double_value(&[]));
        optimizer.backward_step_clip(&loss, 5.0);
        if let Some((file, vx, vy)) = loss_log.as_mut() {
            let validation_loss = tch::no_grad(|| {
                let prediction = skip.forward(vx);
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
    let (intercept, coefficients) = if readout_refit {
        let basis = selected.iter().map(|feature| completed[*feature].clone()).collect::<Vec<_>>();
        fit_basis_coefficients(&table.target, &basis, logistic)
    } else {
        let intercept = tensor_values(
            skip.bs.as_ref().expect("linear bias").shallow_clone(),
            "linear ablation bias export",
        )?[0];
        (intercept, tensor_values(skip.ws.shallow_clone(), "linear ablation weight export")?)
    };
    let terms = selected
        .iter()
        .copied()
        .zip(coefficients)
        .filter(|(_, coefficient)| !readout_refit || coefficient.abs() >= 1e-5)
        .map(|(feature, coefficient)| LinearTerm { feature: feature as u32, coefficient })
        .collect();
    Ok(if logistic {
        Target::SparseLogistic { intercept, terms }
    } else {
        Target::SparseLinear { intercept, terms }
    })
}

#[cfg(feature = "gpu-training")]
fn compact_neural_training_settings(feature_count: usize) -> Result<(usize, usize, usize)> {
    #[cfg(feature = "gpu-research-training")]
    match std::env::var("DOPE_RESEARCH_TARGET_PROFILE") {
        Ok(profile) => return bounded_research_training_settings(&profile),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(DopeError::Unsupported("unknown bounded GPU research profile".into()));
        }
        Err(std::env::VarError::NotPresent) => {}
    }
    Ok((12, feature_count.min(12).clamp(4, 12), 96))
}

#[cfg(test)]
mod bounded_research_training_tests {
    use super::*;

    #[test]
    fn profiles_preserve_operator_dimensions_and_reject_unknown_requests() {
        for (profile, features, steps) in [
            ("features12_steps512", 12, 512),
            ("features12_steps2048", 12, 2048),
            ("features24_steps512", 24, 512),
            ("features24_steps2048", 24, 2048),
        ] {
            let (limit, width, iterations) = bounded_research_training_settings(profile).unwrap();
            assert_eq!((limit, width, iterations), (features, 16, steps));
            assert!(limit <= 24 && width <= 16);
        }
        assert!(bounded_research_training_settings("features48_steps99999").is_err());
    }

    #[test]
    fn refinement_profiles_fit_existing_target_codec_dimensions() {
        for (profile, expected) in [
            ("features12_steps8192", (12, 16, 8192)),
            ("features16_steps8192", (16, 16, 8192)),
            ("features12_width8_steps8192", (12, 8, 8192)),
            ("features24_width8_steps8192", (24, 8, 8192)),
        ] {
            let observed = bounded_research_training_settings(profile).unwrap();
            assert_eq!(observed, expected);
            assert!(observed.0 <= 24 && observed.1 <= 16 && observed.2 <= 8192);
        }
        for profile in [
            "features12_width32_steps8192",
            "features12_width64_steps8192",
            "features12_steps8192 ",
        ] {
            assert!(bounded_research_training_settings(profile).is_err());
        }
    }
    #[test]
    fn ablation_grid_is_exactly_bounded_and_preserves_legacy_parser() {
        let mut count = 0;
        for features in [6, 12, 24] {
            for (width_name, width) in [("linear", 0), ("8", 8), ("16", 16)] {
                for steps in [512, 2048, 8192] {
                    for readout_refit in [false, true] {
                        let readout = if readout_refit { "on" } else { "off" };
                        let name = format!("grid_features{features}_width{width_name}_steps{steps}_readout{readout}");
                        let observed = bounded_research_ablation_settings(&name).unwrap();
                        assert_eq!(observed, ResearchAblationSettings {
                            feature_limit: features,
                            hidden_width: width,
                            optimizer_steps: steps,
                            readout_refit,
                        });
                        assert!(bounded_research_training_settings(&name).is_err());
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 54);
        for profile in [
            "grid_features06_width8_steps512_readouton",
            "grid_features6_width0_steps512_readouton",
            "grid_features6_width32_steps512_readouton",
            "grid_features6_width8_steps513_readouton",
            "grid_features6_width8_steps512_readouttrue",
            "grid_features6_width8_steps512_readouton_extra",
            "grid_features6_width8_steps512_readouton ",
            "features12_steps2048",
        ] {
            assert!(bounded_research_ablation_settings(profile).is_err());
        }
    }
}


#[cfg(any(test, feature = "gpu-training"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResearchTargetDevice {
    Cuda,
    Cpu,
}

#[cfg(any(test, feature = "gpu-research-training"))]
fn parse_research_target_device(
    requested: Option<&std::ffi::OsStr>,
    cuda_visible: Option<&std::ffi::OsStr>,
) -> Result<ResearchTargetDevice> {
    match requested.and_then(std::ffi::OsStr::to_str) {
        None if requested.is_none() => Ok(ResearchTargetDevice::Cuda),
        Some("cuda") => Ok(ResearchTargetDevice::Cuda),
        Some("cpu") if cuda_visible == Some(std::ffi::OsStr::new("")) => {
            Ok(ResearchTargetDevice::Cpu)
        }
        Some("cpu") => Err(DopeError::Unsupported(
            "CPU research training requires empty CUDA_VISIBLE_DEVICES".into(),
        )),
        _ => Err(DopeError::Unsupported("unknown research target device".into())),
    }
}

#[cfg(test)]
mod research_target_device_tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn cuda_default_and_cpu_visibility_are_explicit() {
        assert_eq!(parse_research_target_device(None, None).unwrap(), ResearchTargetDevice::Cuda);
        assert_eq!(parse_research_target_device(Some(OsStr::new("cuda")), None).unwrap(), ResearchTargetDevice::Cuda);
        assert_eq!(parse_research_target_device(Some(OsStr::new("cpu")), Some(OsStr::new(""))).unwrap(), ResearchTargetDevice::Cpu);
        for visible in [None, Some(OsStr::new("0")), Some(OsStr::new("-1"))] {
            assert!(parse_research_target_device(Some(OsStr::new("cpu")), visible).is_err());
        }
        for requested in ["", "CPU", "cpu ", "auto"] {
            assert!(parse_research_target_device(Some(OsStr::new(requested)), Some(OsStr::new(""))).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn malformed_device_or_visibility_is_rejected() {
        use std::os::unix::ffi::OsStrExt;
        let invalid = OsStr::from_bytes(&[0xff]);
        assert!(parse_research_target_device(Some(invalid), Some(OsStr::new(""))).is_err());
        assert!(parse_research_target_device(Some(OsStr::new("cpu")), Some(invalid)).is_err());
    }
}

#[cfg(feature = "gpu-training")]
fn compact_neural_training_device() -> Result<ResearchTargetDevice> {
    #[cfg(feature = "gpu-research-training")]
    {
        let requested = std::env::var_os("DOPE_RESEARCH_TARGET_DEVICE");
        let visible = std::env::var_os("CUDA_VISIBLE_DEVICES");
        parse_research_target_device(requested.as_deref(), visible.as_deref())
    }
    #[cfg(not(feature = "gpu-research-training"))]
    {
        if std::env::var_os("DOPE_RESEARCH_TARGET_DEVICE").is_some() {
            return Err(DopeError::Unsupported(
                "research device selection requires gpu-research-training".into(),
            ));
        }
        Ok(ResearchTargetDevice::Cuda)
    }
}

#[cfg(feature = "gpu-training")]
fn fit_compact_neural_target(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
    seed: u64,
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
            Some(settings) => (settings.feature_limit, settings.hidden_width, settings.optimizer_steps),
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
                    table, completed, &selected, &x, &y,
                    settings.optimizer_steps, settings.readout_refit,
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
        let logistic = table
            .target
            .iter()
            .all(|value| *value == 0.0 || *value == 1.0);
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
            let linear_weights = tensor_values(skip.ws.shallow_clone(), "neural skip weight export")?;
            let output_weights = tensor_values(residual.ws.shallow_clone(), "neural output weight export")?;
            return Ok(Target::CompactNeuralResidual {
                intercept: skip_bias + residual_bias,
                logistic,
                linear_terms: selected
                    .iter()
                    .copied()
                    .zip(linear_weights)
                    .map(|(feature, coefficient)| LinearTerm { feature: feature as u32, coefficient })
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
) -> Result<Target> {
    if std::env::var_os("DOPE_RESEARCH_TARGET_DEVICE").is_some() {
        return Err(DopeError::Unsupported(
            "research device selection requires gpu-research-training".into(),
        ));
    }
    Ok(fit_compact_neural_target_native(table, completed, screen))
}

fn fit_noise(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
    target: &Target,
) -> Noise {
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
    if table
        .target
        .iter()
        .all(|value| *value == 0.0 || *value == 1.0)
    {
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
        return Noise::IsotonicCalibration {
            knots,
            probabilities: calibrated,
        };
    }
    let residuals: Vec<f32> = table
        .target
        .iter()
        .zip(&linear)
        .map(|(actual, prediction)| (actual - prediction).abs().max(1e-5))
        .collect();
    let mut sorted = residuals.clone();
    sorted.sort_by(f32::total_cmp);
    let minimum_sigma = quantile(&sorted, 0.1).clamp(1e-4, 1.0);
    let maximum_sigma = quantile(&sorted, 0.9)
        .clamp(minimum_sigma, 1.0)
        .max(minimum_sigma);
    let selected: Vec<_> = screen.iter().take(4).map(|(feature, _)| *feature).collect();
    let basis: Vec<_> = selected
        .iter()
        .map(|feature| completed[*feature].clone())
        .collect();
    let log_residuals: Vec<_> = residuals.iter().map(|value| value.ln()).collect();
    let (intercept, coefficients) = fit_basis_coefficients(&log_residuals, &basis, false);
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
    }
}

fn quantize(value: f32, bits: u8, unit: bool) -> f32 {
    let levels = ((1u32 << bits.min(16)) - 1) as f32;
    if unit {
        (value.clamp(0.0, 1.0) * levels).round() / levels
    } else {
        let step = 2.0f32.powi(-(i32::from(bits) - 2));
        (value / step).round() * step
    }
}

fn quantize_marginal(marginal: &mut Marginal, bits: u8) {
    match marginal {
        Marginal::Constant { value } => *value = quantize(*value, bits, true),
        Marginal::Bernoulli { probability } => *probability = quantize(*probability, bits, true),
        Marginal::Grid {
            values,
            probabilities,
        } => {
            values
                .iter_mut()
                .for_each(|value| *value = quantize(*value, bits, true));
            probabilities.iter_mut().for_each(|value| {
                let positive = *value > 0.0;
                *value = quantize(*value, bits, true);
                if positive && *value == 0.0 {
                    *value = 1.0 / ((1u32 << bits.min(16)) - 1) as f32;
                }
            });
        }
        Marginal::QuantileSpline { values } => values
            .iter_mut()
            .for_each(|value| *value = quantize(*value, bits, true)),
        Marginal::Beta { alpha, beta } => {
            *alpha = quantize(*alpha, bits, false).max(1e-4);
            *beta = quantize(*beta, bits, false).max(1e-4);
        }
        Marginal::Gaussian { mean, sigma } => {
            *mean = quantize(*mean, bits, true);
            *sigma = quantize(*sigma, bits, false).max(1e-4);
        }
        Marginal::Histogram {
            edges,
            probabilities,
        } => {
            edges
                .iter_mut()
                .for_each(|value| *value = quantize(*value, bits, true));
            probabilities.iter_mut().for_each(|value| {
                let positive = *value > 0.0;
                *value = quantize(*value, bits, true);
                if positive && *value == 0.0 {
                    *value = 1.0 / ((1u32 << bits.min(16)) - 1) as f32;
                }
            });
        }
        Marginal::ZeroInflated {
            point,
            probability,
            base,
        } => {
            *point = quantize(*point, bits, true);
            *probability = quantize(*probability, bits, true);
            quantize_marginal(base, bits);
        }
    }
}

fn quantize_edges(edges: &mut [CopulaEdge], bits: u8) {
    edges.iter_mut().for_each(|edge| {
        edge.correlation = quantize(edge.correlation, bits, false).clamp(-0.995, 0.995)
    });
}

fn quantize_strict_unit(values: &mut [f32], bits: u8) {
    let maximum = (1u32 << bits.min(16)) - 1;
    debug_assert!(values.len() <= maximum as usize + 1);
    let length = values.len();
    for (index, value) in values.iter_mut().enumerate() {
        let lower = index as u32;
        let upper = maximum - (length - index - 1) as u32;
        let level = (*value * maximum as f32).round() as u32;
        *value = level.clamp(lower, upper) as f32 / maximum as f32;
    }
    for index in 1..values.len() {
        let minimum = values[index - 1] + 1.0 / maximum as f32;
        values[index] = values[index].max(minimum);
    }
}

fn quantize_dependence(dependence: &mut Dependence, bits: u8) {
    match dependence {
        Dependence::Independent => {}
        Dependence::ChowLiu { edges, .. } | Dependence::SparseGraph { edges } => {
            quantize_edges(edges, bits)
        }
        Dependence::GaussianCopula { cholesky } => cholesky
            .iter_mut()
            .for_each(|value| *value = quantize(*value, bits, false).clamp(-1.0, 1.0)),
        Dependence::Poet {
            loadings,
            residual_edges,
            ..
        } => {
            loadings.iter_mut().for_each(|loading| {
                loading.loading = quantize(loading.loading, bits, false).clamp(-1.0, 1.0)
            });
            quantize_edges(residual_edges, bits);
        }
        Dependence::Vine { edges } => edges.iter_mut().for_each(|edge| {
            edge.parameter = quantize(edge.parameter, bits, false).clamp(-0.995, 0.995)
        }),
        Dependence::Mixture {
            weights,
            components,
        } => {
            weights
                .iter_mut()
                .for_each(|weight| *weight = quantize(*weight, bits, true));
            components
                .iter_mut()
                .for_each(|component| quantize_dependence(component, bits));
        }
        Dependence::Triangular { terms } => terms.iter_mut().for_each(|term| {
            term.coefficient = quantize(term.coefficient, bits, false).clamp(-0.995, 0.995)
        }),
    }
}

fn quantize_additive(terms: &mut [crate::model::AdditiveTerm], bits: u8) {
    for term in terms {
        quantize_strict_unit(&mut term.knots, bits);
        term.values
            .iter_mut()
            .for_each(|value| *value = quantize(*value, bits, false));
    }
}

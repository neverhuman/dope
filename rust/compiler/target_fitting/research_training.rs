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
        _ => {
            return Err(DopeError::Unsupported(
                "unknown bounded GPU research profile".into(),
            ));
        }
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
    task: Task,
) -> Result<Target> {
    use tch::nn::{Module, OptimizerConfig};

    let store = tch::nn::VarStore::new(x.device());
    let skip = tch::nn::linear(
        &store.root() / "skip",
        selected.len() as i64,
        1,
        Default::default(),
    );
    let logistic = task == Task::Binary;
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
        let basis = selected
            .iter()
            .map(|feature| completed[*feature].clone())
            .collect::<Vec<_>>();
        fit_basis_coefficients(&table.target, &basis, logistic)
    } else {
        let intercept = tensor_values(
            skip.bs.as_ref().expect("linear bias").shallow_clone(),
            "linear ablation bias export",
        )?[0];
        (
            intercept,
            tensor_values(skip.ws.shallow_clone(), "linear ablation weight export")?,
        )
    };
    let terms = selected
        .iter()
        .copied()
        .zip(coefficients)
        .filter(|(_, coefficient)| !readout_refit || coefficient.abs() >= 1e-5)
        .map(|(feature, coefficient)| LinearTerm {
            feature: feature as u32,
            coefficient,
        })
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
            return Err(DopeError::Unsupported(
                "unknown bounded GPU research profile".into(),
            ));
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
                        let name = format!(
                            "grid_features{features}_width{width_name}_steps{steps}_readout{readout}"
                        );
                        let observed = bounded_research_ablation_settings(&name).unwrap();
                        assert_eq!(
                            observed,
                            ResearchAblationSettings {
                                feature_limit: features,
                                hidden_width: width,
                                optimizer_steps: steps,
                                readout_refit,
                            }
                        );
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
        _ => Err(DopeError::Unsupported(
            "unknown research target device".into(),
        )),
    }
}

#[cfg(test)]
mod research_target_device_tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn cuda_default_and_cpu_visibility_are_explicit() {
        assert_eq!(
            parse_research_target_device(None, None).unwrap(),
            ResearchTargetDevice::Cuda
        );
        assert_eq!(
            parse_research_target_device(Some(OsStr::new("cuda")), None).unwrap(),
            ResearchTargetDevice::Cuda
        );
        assert_eq!(
            parse_research_target_device(Some(OsStr::new("cpu")), Some(OsStr::new(""))).unwrap(),
            ResearchTargetDevice::Cpu
        );
        for visible in [None, Some(OsStr::new("0")), Some(OsStr::new("-1"))] {
            assert!(parse_research_target_device(Some(OsStr::new("cpu")), visible).is_err());
        }
        for requested in ["", "CPU", "cpu ", "auto"] {
            assert!(parse_research_target_device(
                Some(OsStr::new(requested)),
                Some(OsStr::new(""))
            )
            .is_err());
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

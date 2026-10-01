

#[cfg(feature = "gpu-training")]
fn fit_compact_neural_target(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
    seed: u64,
) -> Result<Target> {
    use tch::nn::{Module, OptimizerConfig};

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
    crate::libtorch::with_seeded_libtorch(seed, || {
        let selected = screen
            .iter()
            .take(12)
            .map(|(feature, _)| *feature)
            .collect::<Vec<_>>();
        let hidden_width = selected.len().clamp(4, 12);
        let mut rows = Vec::with_capacity(table.rows * selected.len());
        for row in 0..table.rows {
            for &feature in &selected {
                rows.push(completed[feature][row]);
            }
        }
        let device = tch::Device::Cuda(0);
        let x = tch::Tensor::from_slice(&rows)
            .view([table.rows as i64, selected.len() as i64])
            .to_device(device);
        let y = tch::Tensor::from_slice(&table.target)
            .view([table.rows as i64, 1])
            .to_device(device);
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
        for _ in 0..96 {
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
            optimizer.backward_step_clip(&loss, 5.0);
        }
        let hidden_biases = tensor_values(
            hidden.bs.as_ref().expect("linear bias").shallow_clone(),
            "neural hidden bias export",
        )?;
        let input_weights = tensor_values(hidden.ws.shallow_clone(), "neural input export")?;
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
    })
}

#[cfg(not(feature = "gpu-training"))]
fn fit_compact_neural_target(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
    _seed: u64,
) -> Result<Target> {
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

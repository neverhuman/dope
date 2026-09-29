
fn quantized_kernel(kernel: &Kernel, bits: u8) -> Kernel {
    let mut output = kernel.clone();
    output.quantization_bits = bits;
    for schema in &mut output.schema {
        schema.missing_probability = quantize(schema.missing_probability, bits, true);
        schema.impute = quantize(schema.impute, bits, true);
        match &mut schema.transform {
            crate::model::Transform::Identity => {}
            crate::model::Transform::Log1p { scale } => {
                *scale = quantize(*scale, bits, false).max(1e-4)
            }
            crate::model::Transform::Power { exponent } => {
                *exponent = quantize(*exponent, bits, false).max(1e-4)
            }
            crate::model::Transform::Logit { epsilon } => {
                *epsilon = quantize(*epsilon, bits, true).clamp(1e-4, 0.499)
            }
            crate::model::Transform::Winsorize { lower, upper } => {
                *lower = quantize(*lower, bits, true);
                *upper = quantize(*upper, bits, true).max(*lower + 1e-4);
            }
        }
    }
    for marginal in &mut output.marginals {
        quantize_marginal(marginal, bits);
    }
    let (dependence, target, noise) = output
        .symbolic_mut()
        .expect("symbolic tournament only quantizes symbolic kernels");
    quantize_dependence(dependence, bits);
    match target {
        Target::SparseLinear { intercept, terms } | Target::SparseLogistic { intercept, terms } => {
            *intercept = quantize(*intercept, bits, false);
            terms
                .iter_mut()
                .for_each(|term| term.coefficient = quantize(term.coefficient, bits, false));
            terms.sort_by_key(|term| term.feature);
        }
        Target::SparseGam {
            intercept, terms, ..
        } => {
            *intercept = quantize(*intercept, bits, false);
            quantize_additive(terms, bits);
        }
        Target::Ga2m {
            intercept,
            main_terms,
            interactions,
            ..
        } => {
            *intercept = quantize(*intercept, bits, false);
            quantize_additive(main_terms, bits);
            for term in interactions {
                quantize_strict_unit(&mut term.left_knots, bits);
                quantize_strict_unit(&mut term.right_knots, bits);
                term.values
                    .iter_mut()
                    .for_each(|value| *value = quantize(*value, bits, false));
            }
        }
        Target::Mars {
            intercept, terms, ..
        } => {
            *intercept = quantize(*intercept, bits, false);
            for term in terms {
                term.coefficient = quantize(term.coefficient, bits, false);
                term.factors
                    .iter_mut()
                    .for_each(|factor| factor.knot = quantize(factor.knot, bits, true));
            }
        }
        Target::ObliviousTree {
            thresholds, leaves, ..
        } => {
            thresholds
                .iter_mut()
                .for_each(|value| *value = quantize(*value, bits, true));
            leaves
                .iter_mut()
                .for_each(|value| *value = quantize(*value, bits, false));
        }
        Target::CompactNeuralResidual {
            intercept,
            linear_terms,
            input_weights,
            hidden_biases,
            output_weights,
            ..
        } => {
            *intercept = quantize(*intercept, bits, false);
            linear_terms
                .iter_mut()
                .for_each(|term| term.coefficient = quantize(term.coefficient, bits, false));
            input_weights
                .iter_mut()
                .for_each(|value| *value = quantize(*value, bits, false));
            hidden_biases
                .iter_mut()
                .for_each(|value| *value = quantize(*value, bits, false));
            output_weights
                .iter_mut()
                .for_each(|value| *value = quantize(*value, bits, false));
        }
    }
    match noise {
        Noise::Homoscedastic { sigma } => *sigma = quantize(*sigma, bits, true),
        Noise::BernoulliCalibration { base_rate } => *base_rate = quantize(*base_rate, bits, true),
        Noise::Heteroscedastic {
            intercept,
            terms,
            minimum_sigma,
            maximum_sigma,
        } => {
            *intercept = quantize(*intercept, bits, false);
            terms
                .iter_mut()
                .for_each(|term| term.coefficient = quantize(term.coefficient, bits, false));
            *minimum_sigma = quantize(*minimum_sigma, bits, true);
            *maximum_sigma = quantize(*maximum_sigma, bits, true).max(*minimum_sigma);
        }
        Noise::IsotonicCalibration {
            knots,
            probabilities,
        } => {
            quantize_strict_unit(knots, bits);
            quantize_strict_unit(probabilities, bits);
        }
    }
    output
}

fn quantized_target_utility(kernel: &Kernel, table: &Table, completed: &[Vec<f32>]) -> f64 {
    let (_, target, _) = kernel
        .symbolic()
        .expect("symbolic utility only evaluates symbolic kernels");
    let mean = table.target.iter().sum::<f32>() / table.rows.max(1) as f32;
    match table
        .target
        .iter()
        .enumerate()
        .fold((0.0f64, 0.0f64), |(loss, null), (row, actual)| {
            let row_values: Vec<f32> = completed.iter().map(|column| column[row]).collect();
            let (linear, logistic) = evaluate_target(target, &row_values);
            if logistic {
                let probability = f64::from(1.0 / (1.0 + (-linear.clamp(-30.0, 30.0)).exp()))
                    .clamp(1e-9, 1.0 - 1e-9);
                let base = f64::from(mean).clamp(1e-9, 1.0 - 1e-9);
                let y = f64::from(*actual);
                (
                    loss - (y * probability.ln() + (1.0 - y) * (1.0 - probability).ln()),
                    null - (y * base.ln() + (1.0 - y) * (1.0 - base).ln()),
                )
            } else {
                (
                    loss + f64::from(*actual - linear.clamp(0.0, 1.0)).powi(2),
                    null + f64::from(*actual - mean).powi(2),
                )
            }
        }) {
        (loss, null_loss) if null_loss <= 1e-12 => {
            (1.0 - loss / table.rows.max(1) as f64).clamp(0.0, 1.0)
        }
        (loss, null_loss) => (1.0 - loss / null_loss).clamp(0.0, 1.0),
    }
}

fn quantized_driver_agreement(kernel: &Kernel, screen: &[(usize, f32)]) -> f64 {
    let (_, target, _) = kernel
        .symbolic()
        .expect("symbolic driver metric only evaluates symbolic kernels");
    let mut encoded: Vec<LinearTerm> = match target {
        Target::SparseLinear { terms, .. } | Target::SparseLogistic { terms, .. } => terms.clone(),
        Target::SparseGam { terms, .. } => terms
            .iter()
            .map(|term| LinearTerm {
                feature: term.feature,
                coefficient: term
                    .values
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0, f32::max),
            })
            .collect(),
        Target::Ga2m {
            main_terms,
            interactions,
            ..
        } => main_terms
            .iter()
            .map(|term| LinearTerm {
                feature: term.feature,
                coefficient: term
                    .values
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0, f32::max),
            })
            .chain(interactions.iter().flat_map(|term| {
                let coefficient = term
                    .values
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0, f32::max);
                [
                    LinearTerm {
                        feature: term.left,
                        coefficient,
                    },
                    LinearTerm {
                        feature: term.right,
                        coefficient,
                    },
                ]
            }))
            .collect(),
        Target::Mars { terms, .. } => terms
            .iter()
            .flat_map(|term| {
                term.factors.iter().map(|factor| LinearTerm {
                    feature: factor.feature,
                    coefficient: term.coefficient.abs(),
                })
            })
            .collect(),
        Target::ObliviousTree { features, .. } => features
            .iter()
            .map(|feature| LinearTerm {
                feature: *feature,
                coefficient: 1.0,
            })
            .collect(),
        Target::CompactNeuralResidual {
            linear_terms,
            hidden_features,
            hidden_width,
            input_weights,
            output_weights,
            ..
        } => {
            let mut influence: BTreeMap<u32, f32> = linear_terms
                .iter()
                .map(|term| (term.feature, term.coefficient.abs()))
                .collect();
            for (input, feature) in hidden_features.iter().enumerate() {
                let neural = (0..usize::from(*hidden_width))
                    .map(|unit| {
                        (output_weights[unit] * input_weights[unit * hidden_features.len() + input])
                            .abs()
                    })
                    .sum::<f32>();
                *influence.entry(*feature).or_default() += neural;
            }
            influence
                .into_iter()
                .map(|(feature, coefficient)| LinearTerm {
                    feature,
                    coefficient,
                })
                .collect()
        }
    };
    if screen.is_empty() {
        return 1.0;
    }
    let k = 20
        .min(5.max((screen.len() as f64 * 0.1).ceil() as usize))
        .min(screen.len())
        .max(1);
    let actual: std::collections::HashSet<usize> =
        screen.iter().take(k).map(|(feature, _)| *feature).collect();
    encoded.sort_by(|left, right| {
        right
            .coefficient
            .abs()
            .total_cmp(&left.coefficient.abs())
            .then_with(|| left.feature.cmp(&right.feature))
    });
    let overlap = encoded
        .iter()
        .take(k)
        .filter(|term| actual.contains(&(term.feature as usize)))
        .count() as f64
        / k as f64;
    let gains: BTreeMap<usize, f64> = screen
        .iter()
        .map(|(feature, score)| (*feature, f64::from(*score)))
        .collect();
    let ideal = screen
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, (_, score))| f64::from(*score) / ((rank + 2) as f64).log2())
        .sum::<f64>();
    let dcg = encoded
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, term)| {
            gains.get(&(term.feature as usize)).copied().unwrap_or(0.0) / ((rank + 2) as f64).log2()
        })
        .sum::<f64>();
    0.5 * overlap
        + 0.5
            * if ideal <= 1e-12 {
                1.0
            } else {
                (dcg / ideal).clamp(0.0, 1.0)
            }
}

fn stable_seed(table: &Table) -> u64 {
    let mut signatures: Vec<[u8; 32]> = table
        .columns
        .iter()
        .map(|column| {
            let mut values = column.clone();
            values.sort_by(f32::total_cmp);
            let mut hasher = blake3::Hasher::new();
            for value in values {
                hasher.update(&value.to_bits().to_le_bytes());
            }
            *hasher.finalize().as_bytes()
        })
        .collect();
    signatures.sort_unstable();
    let mut hasher = blake3::Hasher::new();
    for signature in signatures {
        hasher.update(&signature);
    }
    let mut target = table.target.clone();
    target.sort_by(f32::total_cmp);
    for value in target {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    u64::from_le_bytes(hasher.finalize().as_bytes()[..8].try_into().unwrap())
}

fn pareto(candidates: &[CandidateReport]) -> Vec<CandidateReport> {
    let mut frontier: Vec<_> = candidates
        .iter()
        .filter(|candidate| {
            !candidates.iter().any(|other| {
                other.candidate_id != candidate.candidate_id
                    && other.artifact_bytes <= candidate.artifact_bytes
                    && other.score >= candidate.score
                    && (other.artifact_bytes < candidate.artifact_bytes
                        || other.score > candidate.score)
            })
        })
        .cloned()
        .collect();
    frontier.sort_by(|left, right| {
        left.artifact_bytes
            .cmp(&right.artifact_bytes)
            .then_with(|| right.score.total_cmp(&left.score))
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    frontier
}

fn maximum_normalized_gate_shortfall(candidate: &CandidateReport) -> f64 {
    [
        ((0.99 - candidate.utility_retention) / 0.99).max(0.0),
        ((0.95 - candidate.driver_agreement) / 0.95).max(0.0),
        ((0.90 - candidate.proxy_joint_fidelity) / 0.90).max(0.0),
        ((candidate.proxy_membership_auc - 0.60) / 0.40).max(0.0),
    ]
    .into_iter()
    .fold(0.0, f64::max)
}

fn target_parameter_bytes(target: &Target) -> usize {
    match target {
        Target::SparseLinear { terms, .. } | Target::SparseLogistic { terms, .. } => {
            8 + terms.len() * 8
        }
        Target::SparseGam { terms, .. } => {
            9 + terms
                .iter()
                .map(|term| 8 + 8 * term.knots.len())
                .sum::<usize>()
        }
        Target::Ga2m {
            main_terms,
            interactions,
            ..
        } => {
            9 + main_terms
                .iter()
                .map(|term| 8 + 8 * term.knots.len())
                .sum::<usize>()
                + interactions
                    .iter()
                    .map(|term| {
                        8 + 4 * (term.left_knots.len() + term.right_knots.len() + term.values.len())
                    })
                    .sum::<usize>()
        }
        Target::Mars { terms, .. } => {
            9 + terms
                .iter()
                .map(|term| 8 + term.factors.len() * 9)
                .sum::<usize>()
        }
        Target::ObliviousTree {
            features, leaves, ..
        } => 5 + features.len() * 8 + leaves.len() * 4,
        Target::CompactNeuralResidual {
            linear_terms,
            hidden_features,
            input_weights,
            hidden_biases,
            output_weights,
            ..
        } => {
            12 + linear_terms.len() * 8
                + hidden_features.len() * 4
                + 4 * (input_weights.len() + hidden_biases.len() + output_weights.len())
        }
    }
}
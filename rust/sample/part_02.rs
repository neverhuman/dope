

pub(crate) fn evaluate_target(target: &Target, row: &[f32]) -> (f32, bool) {
    match target {
        Target::SparseLinear { intercept, terms } => (
            *intercept
                + terms
                    .iter()
                    .map(|term| term.coefficient * row[term.feature as usize])
                    .sum::<f32>(),
            false,
        ),
        Target::SparseLogistic { intercept, terms } => (
            *intercept
                + terms
                    .iter()
                    .map(|term| term.coefficient * row[term.feature as usize])
                    .sum::<f32>(),
            true,
        ),
        Target::SparseGam {
            intercept,
            logistic,
            terms,
        } => (
            *intercept + terms.iter().map(|term| additive(term, row)).sum::<f32>(),
            *logistic,
        ),
        Target::Ga2m {
            intercept,
            logistic,
            main_terms,
            interactions,
        } => (
            *intercept
                + main_terms
                    .iter()
                    .map(|term| additive(term, row))
                    .sum::<f32>()
                + interactions
                    .iter()
                    .map(|term| interaction(term, row))
                    .sum::<f32>(),
            *logistic,
        ),
        Target::Mars {
            intercept,
            logistic,
            terms,
        } => (
            *intercept
                + terms
                    .iter()
                    .map(|term| {
                        term.coefficient
                            * term
                                .factors
                                .iter()
                                .map(|factor| {
                                    if factor.direction > 0 {
                                        (row[factor.feature as usize] - factor.knot).max(0.0)
                                    } else {
                                        (factor.knot - row[factor.feature as usize]).max(0.0)
                                    }
                                })
                                .product::<f32>()
                    })
                    .sum::<f32>(),
            *logistic,
        ),
        Target::ObliviousTree {
            logistic,
            features,
            thresholds,
            leaves,
        } => {
            let leaf = features.iter().zip(thresholds).enumerate().fold(
                0usize,
                |index, (depth, (feature, threshold))| {
                    index | (((row[*feature as usize] > *threshold) as usize) << depth)
                },
            );
            (leaves[leaf], *logistic)
        }
        Target::CompactNeuralResidual {
            intercept,
            logistic,
            linear_terms,
            hidden_features,
            hidden_width,
            input_weights,
            hidden_biases,
            output_weights,
        } => {
            let inputs = hidden_features.len();
            let hidden = (0..usize::from(*hidden_width))
                .map(|unit| {
                    let linear = hidden_biases[unit]
                        + hidden_features
                            .iter()
                            .enumerate()
                            .map(|(input, feature)| {
                                input_weights[unit * inputs + input] * row[*feature as usize]
                            })
                            .sum::<f32>();
                    linear.tanh()
                })
                .collect::<Vec<_>>();
            (
                *intercept
                    + linear_terms
                        .iter()
                        .map(|term| term.coefficient * row[term.feature as usize])
                        .sum::<f32>()
                    + output_weights
                        .iter()
                        .zip(hidden)
                        .map(|(weight, activation)| weight * activation)
                        .sum::<f32>(),
                *logistic,
            )
        }
    }
}

fn sample_v2(kernel: &Kernel, options: SampleOptions) -> Result<Vec<f32>> {
    kernel.validate().map_err(DopeError::Codec)?;
    let (dependence, target, noise) = match &kernel.program {
        KernelProgram::Symbolic {
            dependence,
            target,
            noise,
        } => (dependence, target, noise),
        KernelProgram::NeuralJoint(generator) => {
            return sample_neural(kernel, generator, options);
        }
    };
    let rows = options.rows;
    let width = kernel.features as usize;
    let stride = width + 1;
    let seed = options.seed.unwrap_or(kernel.seed);
    let mut output = vec![0.0f32; rows.saturating_mul(stride)];
    output
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(row_index, row)| {
            sample_latent(
                dependence,
                seed,
                row_index,
                width,
                0x1234_5678,
                &mut row[..width],
            );
            for (feature, marginal) in kernel.marginals.iter().enumerate() {
                row[feature] = inverse_transform(
                    &kernel.schema[feature].transform,
                    inverse_marginal(marginal, normal_cdf(f64::from(row[feature])) as f32),
                );
            }
            let (linear, logistic) = evaluate_target(target, row);
            row[width] = match (&kernel.task, noise) {
                (Task::Regression, Noise::Homoscedastic { sigma }) => (linear
                    + *sigma * inverse_normal(uniform(seed, row_index, width, 0x9abc_def0)) as f32)
                    .clamp(0.0, 1.0),
                (
                    Task::Regression,
                    Noise::Heteroscedastic {
                        intercept,
                        terms,
                        minimum_sigma,
                        maximum_sigma,
                    },
                ) => {
                    let log_sigma = *intercept
                        + terms
                            .iter()
                            .map(|term| term.coefficient * row[term.feature as usize])
                            .sum::<f32>();
                    let sigma = log_sigma.exp().clamp(*minimum_sigma, *maximum_sigma);
                    (linear
                        + sigma
                            * inverse_normal(uniform(seed, row_index, width, 0x9abc_def0)) as f32)
                        .clamp(0.0, 1.0)
                }
                (Task::Binary, _) => {
                    let mut probability = if logistic {
                        1.0 / (1.0 + (-linear.clamp(-30.0, 30.0)).exp())
                    } else {
                        linear.clamp(0.0, 1.0)
                    };
                    if let Noise::IsotonicCalibration {
                        knots,
                        probabilities,
                    } = noise
                    {
                        probability = piecewise(knots, probabilities, probability);
                    }
                    (uniform(seed, row_index, width, 0x9abc_def0) as f32 <= probability) as u8
                        as f32
                }
                _ => linear.clamp(0.0, 1.0),
            };
            for (feature, schema) in kernel.schema.iter().enumerate() {
                if schema.missing_probability > 0.0
                    && uniform(seed, row_index, feature, 0xfeed_beef) as f32
                        <= schema.missing_probability
                {
                    row[feature] = f32::NAN;
                }
            }
        });
    Ok(output)
}

fn sample_neural(
    kernel: &Kernel,
    generator: &JointGenerator,
    options: SampleOptions,
) -> Result<Vec<f32>> {
    let width = kernel.features as usize;
    let generation_seed = options.seed.unwrap_or(kernel.seed);
    let mut lineage_seed = blake3::Hasher::new();
    lineage_seed.update(b"neural-joint-generation-v1");
    lineage_seed.update(&kernel.seed.to_le_bytes());
    lineage_seed.update(&generation_seed.to_le_bytes());
    lineage_seed.update(generator.training_hash.as_bytes());
    lineage_seed.update(generator.implementation_hash.as_bytes());
    let seed = u64::from_le_bytes(
        lineage_seed.finalize().as_bytes()[..8]
            .try_into()
            .expect("eight hash bytes"),
    );
    let rows = (0..options.rows)
        .into_par_iter()
        .map(|row_index| {
            let row_seed =
                splitmix64(seed ^ (row_index as u64).wrapping_mul(0xd6e8_feb8_6659_fd93));
            let (ranks, missing) = crate::neural::sample_joint(generator, width, row_seed)?;
            let mut row = Vec::with_capacity(width + 1);
            for feature in 0..width {
                let probability = crate::neural::normal_cdf(ranks[feature]).clamp(1e-7, 1.0 - 1e-7);
                let value = inverse_transform(
                    &kernel.schema[feature].transform,
                    inverse_marginal(&kernel.marginals[feature], probability),
                );
                row.push(if missing[feature] { f32::NAN } else { value });
            }
            let probability = crate::neural::normal_cdf(ranks[width]).clamp(1e-7, 1.0 - 1e-7);
            row.push(inverse_marginal(&generator.target_marginal, probability));
            if row.iter().any(|value| value.is_infinite()) {
                return Err(DopeError::Codec("neural sampling produced infinity".into()));
            }
            Ok(row)
        })
        .collect::<Result<Vec<Vec<f32>>>>()?;
    Ok(rows.into_iter().flatten().collect())
}

#[derive(Clone)]
enum V1Marginal {
    Bernoulli(f64),
    Grid(Vec<f64>, Vec<f64>),
    Beta(Beta),
    ZeroInflatedBeta(f64, Beta),
    Quantile(Vec<f64>, Vec<f64>),
}

enum V1Dependence {
    Independent,
    Gaussian { cholesky: Vec<f64>, width: usize },
}

enum V1Target {
    Linear {
        intercept: f32,
        terms: Vec<(usize, f32)>,
        logistic: bool,
    },
    Tree {
        left: Vec<i64>,
        right: Vec<i64>,
        feature: Vec<i64>,
        threshold: Vec<f32>,
        value: Vec<f32>,
        leaf_sigma: Vec<f32>,
    },
    Constant(f32),
}

fn number(value: Option<&Value>, name: &str) -> Result<f64> {
    value
        .and_then(Value::as_f64)
        .ok_or_else(|| DopeError::Codec(format!("V1 field {name} is missing or non-numeric")))
}

include!("v1_sampling.rs");

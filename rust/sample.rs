use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use rayon::prelude::*;
use serde_json::Value;
use statrs::distribution::{Beta, ContinuousCDF};

use crate::codec::LoadedKernel;
use crate::error::{DopeError, Result, io_error};
use crate::model::{
    AdditiveTerm, Dependence, InteractionTerm, JointGenerator, Kernel, KernelProgram, Marginal,
    Noise, Target, Task, Transform,
};

#[derive(Clone, Copy, Debug)]
pub struct SampleOptions {
    pub rows: usize,
    pub seed: Option<u64>,
}

#[inline]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[inline]
fn uniform(seed: u64, row: usize, column: usize, stream: u64) -> f64 {
    let counter = (row as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
        ^ (column as u64).wrapping_mul(0xa076_1d64_78bd_642f)
        ^ stream;
    let bits = splitmix64(seed ^ counter) >> 11;
    ((bits as f64) + 0.5) * (1.0 / ((1u64 << 53) as f64))
}

#[inline]
fn inverse_normal(probability: f64) -> f64 {
    const A: [f64; 6] = [
        -39.696_830_286_653_76,
        220.946_098_424_520_5,
        -275.928_510_446_968_7,
        138.357_751_867_269,
        -30.664_798_066_147_16,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -54.476_098_798_224_06,
        161.585_836_858_040_9,
        -155.698_979_859_886_6,
        66.801_311_887_719_72,
        -13.280_681_552_885_72,
    ];
    const C: [f64; 6] = [
        -0.007_784_894_002_430_293,
        -0.322_396_458_041_136_5,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        0.007_784_695_709_041_462,
        0.322_467_129_070_039_8,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p = probability.clamp(1e-15, 1.0 - 1e-15);
    if p < 0.02425 {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    if p > 0.97575 {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    let q = p - 0.5;
    let r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
}

#[inline]
fn normal_cdf(value: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26, maximum absolute error below 7.5e-8.
    let absolute = value.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * absolute);
    let polynomial = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let tail = 0.398_942_280_401_432_7 * (-0.5 * absolute * absolute).exp() * polynomial;
    if value >= 0.0 { 1.0 - tail } else { tail }
}

#[inline]
fn inverse_marginal(marginal: &Marginal, probability: f32) -> f32 {
    match marginal {
        Marginal::Constant { value } => *value,
        Marginal::Bernoulli {
            probability: threshold,
        } => (probability <= *threshold) as u8 as f32,
        Marginal::Grid {
            values,
            probabilities,
        } => {
            let total = probabilities.iter().sum::<f32>().max(f32::EPSILON);
            let threshold = probability * total;
            let mut cumulative = 0.0;
            for (value, mass) in values.iter().zip(probabilities) {
                cumulative += mass;
                if threshold <= cumulative {
                    return *value;
                }
            }
            *values.last().unwrap_or(&0.0)
        }
        Marginal::QuantileSpline { values } => {
            let position = probability.clamp(0.0, 1.0) * (values.len() - 1) as f32;
            let lower = position as usize;
            let upper = (lower + 1).min(values.len() - 1);
            values[lower] + (position - lower as f32) * (values[upper] - values[lower])
        }
        Marginal::Beta { alpha, beta } => Beta::new(f64::from(*alpha), f64::from(*beta))
            .map(|distribution| {
                distribution.inverse_cdf(f64::from(probability).clamp(1e-9, 1.0 - 1e-9)) as f32
            })
            .unwrap_or(probability),
        Marginal::Gaussian { mean, sigma } => (*mean
            + *sigma * inverse_normal(f64::from(probability).clamp(1e-9, 1.0 - 1e-9)) as f32)
            .clamp(0.0, 1.0),
        Marginal::Histogram {
            edges,
            probabilities,
        } => {
            let total = probabilities.iter().sum::<f32>().max(f32::EPSILON);
            let threshold = probability * total;
            let mut cumulative = 0.0;
            for (index, mass) in probabilities.iter().enumerate() {
                let prior = cumulative;
                cumulative += mass;
                if threshold <= cumulative {
                    let within = ((threshold - prior) / mass.max(f32::EPSILON)).clamp(0.0, 1.0);
                    return edges[index] + within * (edges[index + 1] - edges[index]);
                }
            }
            *edges.last().unwrap_or(&0.0)
        }
        Marginal::ZeroInflated {
            point,
            probability: mass,
            base,
        } => {
            if probability <= *mass {
                *point
            } else {
                inverse_marginal(base, (probability - mass) / (1.0 - mass).max(f32::EPSILON))
            }
        }
    }
}

fn inverse_transform(transform: &Transform, value: f32) -> f32 {
    match transform {
        Transform::Identity => value,
        Transform::Log1p { scale } => ((value * scale.ln_1p()).exp() - 1.0) / scale,
        Transform::Power { exponent } => value.clamp(0.0, 1.0).powf(1.0 / exponent),
        Transform::Logit { epsilon } => {
            let low = (epsilon / (1.0 - epsilon)).ln();
            let high = ((1.0 - epsilon) / epsilon).ln();
            1.0 / (1.0 + (-(low + value * (high - low))).exp())
        }
        Transform::Winsorize { lower, upper } => lower + value * (upper - lower),
    }
    .clamp(0.0, 1.0)
}

fn apply_edge(latent: &mut [f32], parent: usize, child: usize, correlation: f32) {
    let rho = correlation.clamp(-0.995, 0.995);
    latent[child] = rho * latent[parent] + (1.0 - rho * rho).max(1e-8).sqrt() * latent[child];
}

fn sample_latent(
    dependence: &Dependence,
    seed: u64,
    row: usize,
    width: usize,
    stream: u64,
    latent: &mut [f32],
) {
    let initialize = |values: &mut [f32], stream: u64| {
        for (feature, value) in values.iter_mut().enumerate() {
            *value = inverse_normal(uniform(seed, row, feature, stream)) as f32;
        }
    };
    match dependence {
        Dependence::Independent => initialize(latent, stream),
        Dependence::ChowLiu { edges, .. } => {
            initialize(latent, stream);
            for edge in edges {
                apply_edge(
                    latent,
                    edge.parent as usize,
                    edge.child as usize,
                    edge.correlation,
                );
            }
        }
        Dependence::GaussianCopula { cholesky } => {
            let mut base = vec![0.0; width];
            initialize(&mut base, stream);
            for feature in 0..width {
                latent[feature] = (0..=feature)
                    .map(|source| cholesky[feature * width + source] * base[source])
                    .sum();
            }
        }
        Dependence::SparseGraph { edges } => {
            initialize(latent, stream);
            for edge in edges {
                apply_edge(
                    latent,
                    edge.parent as usize,
                    edge.child as usize,
                    edge.correlation,
                );
            }
        }
        Dependence::Poet {
            rank,
            loadings,
            residual_edges,
        } => {
            initialize(latent, stream);
            let factors: Vec<f32> = (0..usize::from(*rank))
                .map(|factor| {
                    inverse_normal(uniform(seed, row, width + factor, stream ^ 0x5a5a)) as f32
                })
                .collect();
            for loading in loadings {
                latent[loading.feature as usize] +=
                    loading.loading * factors[loading.factor as usize];
            }
            for edge in residual_edges {
                apply_edge(
                    latent,
                    edge.parent as usize,
                    edge.child as usize,
                    edge.correlation,
                );
            }
        }
        Dependence::Vine { edges } => {
            initialize(latent, stream);
            for edge in edges {
                apply_edge(
                    latent,
                    edge.left as usize,
                    edge.right as usize,
                    edge.parameter,
                );
            }
        }
        Dependence::Mixture {
            weights,
            components,
        } => {
            let threshold =
                uniform(seed, row, width, stream ^ 0x7777) as f32 * weights.iter().sum::<f32>();
            let mut cumulative = 0.0;
            let mut selected = components.len() - 1;
            for (index, weight) in weights.iter().enumerate() {
                cumulative += weight;
                if threshold <= cumulative {
                    selected = index;
                    break;
                }
            }
            sample_latent(
                &components[selected],
                seed,
                row,
                width,
                stream ^ selected as u64,
                latent,
            );
        }
        Dependence::Triangular { terms } => {
            initialize(latent, stream);
            for term in terms {
                apply_edge(
                    latent,
                    term.parent as usize,
                    term.child as usize,
                    term.coefficient,
                );
            }
        }
    }
}

fn piecewise(knots: &[f32], values: &[f32], value: f32) -> f32 {
    let upper = knots
        .partition_point(|knot| *knot < value)
        .min(knots.len() - 1);
    if upper == 0 {
        return values[0];
    }
    let lower = upper - 1;
    let fraction = (value - knots[lower]) / (knots[upper] - knots[lower]).max(f32::EPSILON);
    values[lower] + fraction * (values[upper] - values[lower])
}

fn additive(term: &AdditiveTerm, row: &[f32]) -> f32 {
    piecewise(&term.knots, &term.values, row[term.feature as usize])
}

fn interaction(term: &InteractionTerm, row: &[f32]) -> f32 {
    let left_upper = term
        .left_knots
        .partition_point(|knot| *knot < row[term.left as usize])
        .min(term.left_knots.len() - 1);
    let right_upper = term
        .right_knots
        .partition_point(|knot| *knot < row[term.right as usize])
        .min(term.right_knots.len() - 1);
    let left_lower = left_upper.saturating_sub(1);
    let right_lower = right_upper.saturating_sub(1);
    let left_fraction = if left_upper == left_lower {
        0.0
    } else {
        (row[term.left as usize] - term.left_knots[left_lower])
            / (term.left_knots[left_upper] - term.left_knots[left_lower]).max(f32::EPSILON)
    };
    let right_fraction = if right_upper == right_lower {
        0.0
    } else {
        (row[term.right as usize] - term.right_knots[right_lower])
            / (term.right_knots[right_upper] - term.right_knots[right_lower]).max(f32::EPSILON)
    };
    let width = term.right_knots.len();
    let at = |left: usize, right: usize| term.values[left * width + right];
    let low = at(left_lower, right_lower)
        + right_fraction * (at(left_lower, right_upper) - at(left_lower, right_lower));
    let high = at(left_upper, right_lower)
        + right_fraction * (at(left_upper, right_upper) - at(left_upper, right_lower));
    low + left_fraction * (high - low)
}

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

include!("sample/v1_sampling.rs");

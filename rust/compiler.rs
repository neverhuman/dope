use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::calibration::{CandidateDescriptor, LanguageCalibration, describe_table};
use crate::codec::encode_kernel;
use crate::contract::{AnonymizationTier, ReleasePolicy};
use crate::data::Table;
use crate::error::{DopeError, Result};
use crate::model::{
    AdditiveTerm, ColumnSchema, CopulaEdge, Dependence, FactorLoading, InteractionTerm,
    JointGenerator, Kernel, KernelProgram, LinearTerm, Marginal, MarsFactor, MarsTerm,
    NeuralArchitecture, NeuralProfile, Noise, RankNormalization, SchemaKind, Target, Task,
    TriangularTerm, VineEdge,
};
use crate::sample::evaluate_target;

const KNOT_LADDER: [usize; 5] = [5, 9, 17, 33, 65];
const QUANTIZATION_LADDER: [u8; 6] = [4, 6, 8, 10, 12, 16];
const MAX_TARGET_TERMS: usize = 24;

#[derive(Clone, Debug)]
pub struct CompileOptions {
    pub release_policy: ReleasePolicy,
    pub seed: Option<u64>,
    pub deadline: Option<Duration>,
    pub beam_width: Option<usize>,
    pub quantization_profiles: Vec<u8>,
    pub language: Option<LanguageCalibration>,
    /// Restricts fitting to one frozen empirical campaign backend. Normal
    /// compilation leaves this unset and retains the full bounded tournament.
    pub backend_id: Option<String>,
    /// Frozen deep-campaign grid value. Ignored by symbolic candidates.
    pub neural_target_weight: f64,
    /// Frozen structural-loss grid value. Ignored by symbolic candidates.
    pub neural_structural_penalty: f64,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            release_policy: ReleasePolicy::new(AnonymizationTier::L0, None, false)
                .expect("fixed research policy"),
            seed: None,
            deadline: None,
            beam_width: None,
            quantization_profiles: QUANTIZATION_LADDER.to_vec(),
            language: None,
            backend_id: None,
            neural_target_weight: 2.0,
            neural_structural_penalty: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CandidateReport {
    pub candidate_id: String,
    pub artifact_bytes: usize,
    pub bits_per_original_cell: f64,
    pub score: f64,
    pub compliant: bool,
    pub quantization_bits: u8,
    pub marginal_knots: usize,
    pub dependence: String,
    pub target: String,
    pub utility_retention: f64,
    pub driver_agreement: f64,
    /// Train-only heuristic used to rank candidates. This is not certification evidence.
    pub proxy_joint_fidelity: f64,
    /// A neutral train-only diagnostic, not a measured membership attack AUC.
    pub proxy_membership_auc: f64,
    pub failed_gates: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompileReport {
    pub format: String,
    pub version: u8,
    pub release_policy: ReleasePolicy,
    pub release_policy_hash: String,
    pub task: String,
    pub rows: usize,
    pub positional_features: usize,
    pub selected_candidate: String,
    pub artifact_bytes: usize,
    pub effective_bytes: usize,
    pub bits_per_original_cell: f64,
    pub compliant: bool,
    pub failed_gates: Vec<String>,
    pub candidate_count: usize,
    pub frontier_count: usize,
    pub beam_width: usize,
    pub elapsed_seconds: f64,
    pub contains_row_payloads: bool,
    pub contains_row_references: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub probe_failures: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct CompileResult {
    pub artifact: Vec<u8>,
    pub candidate_artifacts: BTreeMap<String, Vec<u8>>,
    pub report: CompileReport,
    pub candidates: Vec<CandidateReport>,
    pub pareto_frontier: Vec<CandidateReport>,
}

#[derive(Clone)]
struct ColumnFit {
    schema: ColumnSchema,
    discrete: Option<Marginal>,
    quantiles: Vec<Vec<f32>>,
    completed: Vec<f32>,
}

#[inline]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn quantile(sorted: &[f32], q: f32) -> f32 {
    if sorted.len() <= 1 {
        return sorted.first().copied().unwrap_or(0.0);
    }
    let position = q * (sorted.len() - 1) as f32;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f32;
    sorted[lower] + fraction * (sorted[upper] - sorted[lower])
}

fn is_regular_grid(unique: &[f32]) -> bool {
    if unique.len() < 3 {
        return false;
    }
    let mut gaps: Vec<f32> = unique.windows(2).map(|pair| pair[1] - pair[0]).collect();
    gaps.sort_by(f32::total_cmp);
    let median = gaps[gaps.len() / 2];
    median > 0.0
        && gaps
            .iter()
            .all(|gap| (*gap - median).abs() <= 1e-5_f32.max(0.03 * median))
}

fn fit_column(column: &[f32]) -> ColumnFit {
    let mut observed: Vec<f32> = column
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    observed.sort_by(f32::total_cmp);
    let missing_probability = 1.0 - observed.len() as f32 / column.len().max(1) as f32;
    let impute = quantile(&observed, 0.5);
    let completed = column
        .iter()
        .map(|value| if value.is_nan() { impute } else { *value })
        .collect();
    let mut unique = observed.clone();
    unique.dedup_by(|left, right| (*left - *right).abs() <= 1e-6);
    let zeros =
        observed.partition_point(|value| *value <= 1e-7) as f32 / observed.len().max(1) as f32;
    let ones = (observed.len() - observed.partition_point(|value| *value < 1.0 - 1e-7)) as f32
        / observed.len().max(1) as f32;
    let kind = if unique.len() <= 1 {
        SchemaKind::Constant
    } else if unique.len() <= 2
        && unique
            .iter()
            .all(|value| *value <= 1e-6 || *value >= 1.0 - 1e-6)
    {
        SchemaKind::Binary
    } else if unique.len() <= 16 {
        SchemaKind::Ordinal
    } else if unique.len() <= 64 {
        SchemaKind::CategoricalGrid
    } else if unique.len() <= 256 && is_regular_grid(&unique) {
        SchemaKind::CountLike
    } else if zeros.max(ones) >= 0.2 {
        SchemaKind::Inflated
    } else {
        SchemaKind::Continuous
    };
    let discrete = match kind {
        SchemaKind::Constant => Some(Marginal::Constant {
            value: unique.first().copied().unwrap_or(0.0),
        }),
        SchemaKind::Binary => Some(Marginal::Bernoulli {
            probability: observed.iter().filter(|value| **value >= 0.5).count() as f32
                / observed.len().max(1) as f32,
        }),
        SchemaKind::Ordinal | SchemaKind::CategoricalGrid | SchemaKind::CountLike
            if unique.len() <= 64 =>
        {
            let mut counts = vec![0usize; unique.len()];
            for value in &observed {
                let index = unique
                    .binary_search_by(|probe| probe.total_cmp(value))
                    .unwrap_or_else(|insertion| match insertion {
                        0 => 0,
                        index if index == unique.len() => unique.len() - 1,
                        index => {
                            if (unique[index] - value).abs() < (unique[index - 1] - value).abs() {
                                index
                            } else {
                                index - 1
                            }
                        }
                    });
                counts[index] += 1;
            }
            Some(Marginal::Grid {
                values: unique,
                probabilities: counts
                    .into_iter()
                    .map(|count| count as f32 / observed.len().max(1) as f32)
                    .collect(),
            })
        }
        _ => None,
    };
    let quantiles = if discrete.is_none() {
        KNOT_LADDER
            .iter()
            .map(|&knots| {
                (0..knots)
                    .map(|index| quantile(&observed, index as f32 / (knots - 1) as f32))
                    .collect()
            })
            .collect()
    } else {
        Vec::new()
    };
    ColumnFit {
        schema: ColumnSchema {
            kind,
            missing_probability,
            impute,
            transform: crate::model::Transform::Identity,
        },
        discrete,
        quantiles,
        completed,
    }
}

fn rank_normalize(values: &[f32]) -> Vec<f32> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_unstable_by(|&left, &right| {
        values[left]
            .total_cmp(&values[right])
            .then_with(|| left.cmp(&right))
    });
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && (values[order[end]] - values[order[start]]).abs() <= 1e-7 {
            end += 1;
        }
        let rank = ((start + end - 1) as f32 * 0.5 + 1.0) / (values.len() + 1) as f32;
        for &index in &order[start..end] {
            ranks[index] = rank;
        }
        start = end;
    }
    let mean = ranks.iter().sum::<f32>() / ranks.len() as f32;
    for rank in &mut ranks {
        *rank -= mean;
    }
    let norm = ranks.iter().map(|rank| rank * rank).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for rank in &mut ranks {
            *rank /= norm;
        }
    }
    ranks
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| a * b)
        .sum::<f32>()
        .clamp(-0.995, 0.995)
}

fn full_correlation(ranks: &[Vec<f32>]) -> Vec<f32> {
    let width = ranks.len();
    let mut matrix = vec![0.0f32; width * width];
    matrix
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(left, row)| {
            row[left] = 1.0;
            for right in 0..left {
                row[right] = dot(&ranks[left], &ranks[right]);
            }
        });
    for left in 0..width {
        for right in 0..left {
            matrix[right * width + left] = matrix[left * width + right];
        }
    }
    matrix
}

fn maximum_spanning_tree(matrix: &[f32], width: usize) -> Vec<CopulaEdge> {
    let mut selected = vec![false; width];
    let mut best_weight = vec![f32::NEG_INFINITY; width];
    let mut parent = vec![0usize; width];
    selected[0] = true;
    for child in 1..width {
        best_weight[child] = matrix[child].abs();
    }
    let mut undirected = Vec::with_capacity(width.saturating_sub(1));
    for _ in 1..width {
        let child = (0..width)
            .filter(|&index| !selected[index])
            .max_by(|&a, &b| {
                best_weight[a]
                    .total_cmp(&best_weight[b])
                    .then_with(|| b.cmp(&a))
            })
            .unwrap();
        selected[child] = true;
        undirected.push((parent[child], child));
        for next in 0..width {
            let weight = matrix[child * width + next].abs();
            if !selected[next] && weight > best_weight[next] {
                best_weight[next] = weight;
                parent[next] = child;
            }
        }
    }
    orient_tree(undirected, matrix, width)
}

fn orient_tree(undirected: Vec<(usize, usize)>, matrix: &[f32], width: usize) -> Vec<CopulaEdge> {
    let mut adjacency = vec![Vec::new(); width];
    for (left, right) in undirected {
        adjacency[left].push(right);
        adjacency[right].push(left);
    }
    let mut edges = Vec::with_capacity(width.saturating_sub(1));
    let mut seen = vec![false; width];
    seen[0] = true;
    let mut queue = std::collections::VecDeque::from([0usize]);
    while let Some(parent) = queue.pop_front() {
        adjacency[parent].sort_unstable();
        for &child in &adjacency[parent] {
            if seen[child] {
                continue;
            }
            seen[child] = true;
            queue.push_back(child);
            edges.push(CopulaEdge {
                parent: parent as u32,
                child: child as u32,
                correlation: matrix[parent * width + child],
            });
        }
    }
    edges
}

fn wide_tree(ranks: &[Vec<f32>]) -> Vec<CopulaEdge> {
    let width = ranks.len();
    let parents: Vec<(usize, f32)> = (1..width)
        .into_par_iter()
        .map(|child| {
            let start = child.saturating_sub(16);
            (start..child)
                .map(|parent| (parent, dot(&ranks[parent], &ranks[child])))
                .max_by(|(left_index, left), (right_index, right)| {
                    left.abs()
                        .total_cmp(&right.abs())
                        .then_with(|| right_index.cmp(left_index))
                })
                .unwrap()
        })
        .collect();
    parents
        .into_iter()
        .enumerate()
        .map(|(offset, (parent, correlation))| CopulaEdge {
            parent: parent as u32,
            child: (offset + 1) as u32,
            correlation,
        })
        .collect()
}

fn solve_system(mut matrix: Vec<f64>, mut rhs: Vec<f64>, size: usize) -> Vec<f64> {
    for pivot in 0..size {
        let best = (pivot..size)
            .max_by(|&left, &right| {
                matrix[left * size + pivot]
                    .abs()
                    .total_cmp(&matrix[right * size + pivot].abs())
            })
            .unwrap();
        if best != pivot {
            for column in pivot..size {
                matrix.swap(pivot * size + column, best * size + column);
            }
            rhs.swap(pivot, best);
        }
        let diagonal = matrix[pivot * size + pivot];
        if diagonal.abs() <= 1e-12 {
            continue;
        }
        for row in pivot + 1..size {
            let factor = matrix[row * size + pivot] / diagonal;
            for column in pivot..size {
                matrix[row * size + column] -= factor * matrix[pivot * size + column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    let mut solution = vec![0.0; size];
    for row in (0..size).rev() {
        let remainder = (row + 1..size)
            .map(|column| matrix[row * size + column] * solution[column])
            .sum::<f64>();
        let diagonal = matrix[row * size + row];
        if diagonal.abs() > 1e-12 {
            solution[row] = (rhs[row] - remainder) / diagonal;
        }
    }
    solution
}

fn target_screen(completed: &[Vec<f32>], target: &[f32]) -> Vec<(usize, f32)> {
    let target_rank = rank_normalize(target);
    let mut scores: Vec<(usize, f32)> = completed
        .par_iter()
        .enumerate()
        .map(|(index, column)| (index, dot(&rank_normalize(column), &target_rank).abs()))
        .collect();
    scores.sort_by(|(left_index, left), (right_index, right)| {
        right
            .total_cmp(left)
            .then_with(|| left_index.cmp(right_index))
    });
    scores.truncate(MAX_TARGET_TERMS.min(scores.len()));
    scores
}

fn fit_linear_target(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
) -> (Target, Noise, f64) {
    let selected: Vec<usize> = screen.iter().map(|(index, _)| *index).collect();
    let terms = selected.len();
    let rows = table.rows;
    let target_mean = table
        .target
        .iter()
        .map(|value| f64::from(*value))
        .sum::<f64>()
        / rows as f64;
    let means: Vec<f64> = selected
        .iter()
        .map(|&column| {
            completed[column]
                .iter()
                .map(|value| f64::from(*value))
                .sum::<f64>()
                / rows as f64
        })
        .collect();
    let mut gram = vec![0.0; terms * terms];
    let mut rhs = vec![0.0; terms];
    for left in 0..terms {
        for (row, actual) in table.target.iter().enumerate() {
            let x = f64::from(completed[selected[left]][row]) - means[left];
            rhs[left] += x * (f64::from(*actual) - target_mean);
        }
        for right in 0..=left {
            let value = (0..rows)
                .map(|row| {
                    (f64::from(completed[selected[left]][row]) - means[left])
                        * (f64::from(completed[selected[right]][row]) - means[right])
                })
                .sum::<f64>();
            gram[left * terms + right] = value;
            gram[right * terms + left] = value;
        }
        gram[left * terms + left] += 1.0;
    }
    let coefficients = solve_system(gram, rhs, terms);
    let intercept = target_mean
        - coefficients
            .iter()
            .zip(&means)
            .map(|(coefficient, mean)| coefficient * mean)
            .sum::<f64>();
    let model_terms: Vec<LinearTerm> = selected
        .iter()
        .zip(&coefficients)
        .filter(|(_, coefficient)| coefficient.abs() >= 1e-5)
        .map(|(&feature, &coefficient)| LinearTerm {
            feature: feature as u32,
            coefficient: coefficient as f32,
        })
        .collect();
    let mut squared = 0.0;
    let mut null_squared = 0.0;
    for (row, actual) in table.target.iter().enumerate() {
        let prediction = (intercept
            + selected
                .iter()
                .zip(&coefficients)
                .map(|(&feature, coefficient)| coefficient * f64::from(completed[feature][row]))
                .sum::<f64>())
        .clamp(0.0, 1.0);
        squared += (f64::from(*actual) - prediction).powi(2);
        null_squared += (f64::from(*actual) - target_mean).powi(2);
    }
    let sigma = (squared / rows as f64).sqrt() as f32;
    let utility = if null_squared <= 1e-12 {
        1.0 - squared / rows as f64
    } else {
        (1.0 - squared / null_squared).clamp(0.0, 1.0)
    };
    (
        Target::SparseLinear {
            intercept: intercept as f32,
            terms: model_terms,
        },
        Noise::Homoscedastic { sigma },
        utility,
    )
}

fn fit_logistic_target(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
) -> (Target, Noise, f64) {
    let selected: Vec<usize> = screen.iter().map(|(index, _)| *index).collect();
    let size = selected.len() + 1;
    let base_rate = (table.target.iter().sum::<f32>() / table.rows as f32).clamp(1e-6, 1.0 - 1e-6);
    let mut coefficients = vec![0.0f64; size];
    coefficients[0] = f64::from((base_rate / (1.0 - base_rate)).ln());
    for _ in 0..8 {
        let mut hessian = vec![0.0; size * size];
        let mut gradient = vec![0.0; size];
        for (row, actual) in table.target.iter().enumerate() {
            let linear = coefficients[0]
                + selected
                    .iter()
                    .enumerate()
                    .map(|(index, &feature)| {
                        coefficients[index + 1] * f64::from(completed[feature][row])
                    })
                    .sum::<f64>();
            let probability = 1.0 / (1.0 + (-linear.clamp(-30.0, 30.0)).exp());
            let residual = f64::from(*actual) - probability;
            let weight = (probability * (1.0 - probability)).max(1e-6);
            for left in 0..size {
                let x_left = if left == 0 {
                    1.0
                } else {
                    f64::from(completed[selected[left - 1]][row])
                };
                gradient[left] += x_left * residual;
                for right in 0..=left {
                    let x_right = if right == 0 {
                        1.0
                    } else {
                        f64::from(completed[selected[right - 1]][row])
                    };
                    hessian[left * size + right] += weight * x_left * x_right;
                    hessian[right * size + left] = hessian[left * size + right];
                }
            }
        }
        for index in 1..size {
            hessian[index * size + index] += 0.5;
            gradient[index] -= 0.5 * coefficients[index];
        }
        let delta = solve_system(hessian, gradient, size);
        let max_delta = delta
            .iter()
            .fold(0.0f64, |maximum, value| maximum.max(value.abs()));
        for (coefficient, update) in coefficients.iter_mut().zip(delta) {
            *coefficient = (*coefficient + update).clamp(-30.0, 30.0);
        }
        if max_delta < 1e-5 {
            break;
        }
    }
    let terms = selected
        .iter()
        .enumerate()
        .filter(|(index, _)| coefficients[index + 1].abs() >= 1e-5)
        .map(|(index, &feature)| LinearTerm {
            feature: feature as u32,
            coefficient: coefficients[index + 1] as f32,
        })
        .collect();
    let mut model_loss = 0.0;
    let null_loss = -(f64::from(base_rate) * f64::from(base_rate).ln()
        + f64::from(1.0 - base_rate) * f64::from(1.0 - base_rate).ln());
    for (row, actual) in table.target.iter().enumerate() {
        let linear = coefficients[0]
            + selected
                .iter()
                .enumerate()
                .map(|(index, &feature)| {
                    coefficients[index + 1] * f64::from(completed[feature][row])
                })
                .sum::<f64>();
        let probability =
            (1.0 / (1.0 + (-linear.clamp(-30.0, 30.0)).exp())).clamp(1e-9, 1.0 - 1e-9);
        let y = f64::from(*actual);
        model_loss += -(y * probability.ln() + (1.0 - y) * (1.0 - probability).ln());
    }
    model_loss /= table.rows as f64;
    let utility = if null_loss <= 1e-12 {
        1.0
    } else {
        (1.0 - model_loss / null_loss).clamp(0.0, 1.0)
    };
    (
        Target::SparseLogistic {
            intercept: coefficients[0] as f32,
            terms,
        },
        Noise::BernoulliCalibration { base_rate },
        utility,
    )
}

fn nearest_knot(knots: &[f32], value: f32) -> usize {
    knots
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| (*left - value).abs().total_cmp(&(*right - value).abs()))
        .map_or(0, |(index, _)| index)
}

fn feature_knots(values: &[f32], count: usize) -> Vec<f32> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    sorted.dedup_by(|left, right| (*left - *right).abs() <= 1e-7);
    (0..count)
        .map(|index| quantile(&sorted, index as f32 / (count - 1).max(1) as f32))
        .collect()
}

fn spline_features(completed: &[Vec<f32>], screen: &[(usize, f32)], limit: usize) -> Vec<usize> {
    screen
        .iter()
        .filter_map(|(feature, _)| {
            let mut unique = completed[*feature].clone();
            unique.sort_by(f32::total_cmp);
            unique.dedup_by(|left, right| (*left - *right).abs() <= 1e-7);
            (unique.len() >= 3).then_some(*feature)
        })
        .take(limit)
        .collect()
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value.clamp(-30.0, 30.0)).exp())
}

fn fit_basis_coefficients(target: &[f32], basis: &[Vec<f32>], logistic: bool) -> (f32, Vec<f32>) {
    let rows = target.len();
    let size = basis.len() + 1;
    let mean = target.iter().sum::<f32>() / rows.max(1) as f32;
    let mut coefficients = vec![0.0f64; size];
    coefficients[0] = if logistic {
        let probability = mean.clamp(1e-6, 1.0 - 1e-6);
        f64::from((probability / (1.0 - probability)).ln())
    } else {
        f64::from(mean)
    };
    let iterations = if logistic { 12 } else { 1 };
    for _ in 0..iterations {
        let mut gram = vec![0.0; size * size];
        let mut rhs = vec![0.0; size];
        for row in 0..rows {
            let linear = coefficients[0]
                + basis
                    .iter()
                    .zip(&coefficients[1..])
                    .map(|(column, coefficient)| f64::from(column[row]) * coefficient)
                    .sum::<f64>();
            let (weight, working) = if logistic {
                let probability = sigmoid(linear);
                let weight = (probability * (1.0 - probability)).max(1e-4);
                (
                    weight,
                    linear + (f64::from(target[row]) - probability) / weight,
                )
            } else {
                (1.0, f64::from(target[row]))
            };
            for left in 0..size {
                let x_left = if left == 0 {
                    1.0
                } else {
                    f64::from(basis[left - 1][row])
                };
                rhs[left] += weight * x_left * working;
                for right in 0..=left {
                    let x_right = if right == 0 {
                        1.0
                    } else {
                        f64::from(basis[right - 1][row])
                    };
                    gram[left * size + right] += weight * x_left * x_right;
                    gram[right * size + left] = gram[left * size + right];
                }
            }
        }
        for index in 1..size {
            gram[index * size + index] += 0.5;
        }
        coefficients = solve_system(gram, rhs, size)
            .into_iter()
            .map(|value| value.clamp(-30.0, 30.0))
            .collect();
    }
    (
        coefficients[0] as f32,
        coefficients[1..]
            .iter()
            .map(|value| *value as f32)
            .collect(),
    )
}

fn fit_gam_target(table: &Table, completed: &[Vec<f32>], screen: &[(usize, f32)]) -> Target {
    const KNOTS: usize = 9;
    let logistic = table
        .target
        .iter()
        .all(|value| *value == 0.0 || *value == 1.0);
    let base_rate =
        (table.target.iter().sum::<f32>() / table.rows.max(1) as f32).clamp(1e-6, 1.0 - 1e-6);
    let intercept = if logistic {
        (base_rate / (1.0 - base_rate)).ln()
    } else {
        base_rate
    };
    let selected = spline_features(completed, screen, 12);
    let knots: Vec<_> = selected
        .iter()
        .map(|feature| feature_knots(&completed[*feature], KNOTS))
        .collect();
    let mut effects = vec![vec![0.0f32; KNOTS]; selected.len()];
    let mut linear = vec![intercept; table.rows];
    for _ in 0..10 {
        for (term, &feature) in selected.iter().enumerate() {
            let old = effects[term].clone();
            for (row, value) in linear.iter_mut().enumerate() {
                *value -= old[nearest_knot(&knots[term], completed[feature][row])];
            }
            let mut sums = [0.0f64; KNOTS];
            let mut weights = [0.0f64; KNOTS];
            for (row, &linear_value) in linear.iter().enumerate() {
                let slot = nearest_knot(&knots[term], completed[feature][row]);
                let (weight, residual) = if logistic {
                    let probability = sigmoid(f64::from(linear_value));
                    let weight = (probability * (1.0 - probability)).max(0.02);
                    (
                        weight,
                        (f64::from(table.target[row]) - probability) / weight,
                    )
                } else {
                    (1.0, f64::from(table.target[row] - linear_value))
                };
                sums[slot] += weight * residual;
                weights[slot] += weight;
            }
            for slot in 0..KNOTS {
                if weights[slot] > 0.0 {
                    effects[term][slot] = (0.5 * sums[slot] / weights[slot]) as f32;
                }
            }
            let center = effects[term]
                .iter()
                .zip(weights)
                .map(|(value, weight)| f64::from(*value) * weight)
                .sum::<f64>()
                / weights.iter().sum::<f64>().max(1e-12);
            effects[term]
                .iter_mut()
                .for_each(|value| *value -= center as f32);
            for (row, value) in linear.iter_mut().enumerate() {
                *value += effects[term][nearest_knot(&knots[term], completed[feature][row])];
            }
        }
    }
    Target::SparseGam {
        intercept,
        logistic,
        terms: selected
            .into_iter()
            .zip(knots)
            .zip(effects)
            .map(|((feature, knots), values)| AdditiveTerm {
                feature: feature as u32,
                knots,
                values,
            })
            .collect(),
    }
}

fn fit_ga2m_target(table: &Table, completed: &[Vec<f32>], screen: &[(usize, f32)]) -> Target {
    let gam = fit_gam_target(table, completed, screen);
    let Target::SparseGam {
        intercept,
        logistic,
        terms,
    } = gam
    else {
        unreachable!()
    };
    let mut linear = Vec::with_capacity(table.rows);
    for row in 0..table.rows {
        let values: Vec<_> = completed.iter().map(|column| column[row]).collect();
        linear.push(
            evaluate_target(
                &Target::SparseGam {
                    intercept,
                    logistic,
                    terms: terms.clone(),
                },
                &values,
            )
            .0,
        );
    }
    let selected: Vec<_> = terms
        .iter()
        .take(5)
        .map(|term| term.feature as usize)
        .collect();
    let mut interactions = Vec::new();
    'pairs: for left_index in 0..selected.len() {
        for right_index in left_index + 1..selected.len() {
            const KNOTS: usize = 5;
            let left = selected[left_index];
            let right = selected[right_index];
            let left_knots = feature_knots(&completed[left], KNOTS);
            let right_knots = feature_knots(&completed[right], KNOTS);
            let mut sums = [0.0f64; KNOTS * KNOTS];
            let mut weights = [0.0f64; KNOTS * KNOTS];
            for (row, &linear_value) in linear.iter().enumerate() {
                let left_slot = nearest_knot(&left_knots, completed[left][row]);
                let right_slot = nearest_knot(&right_knots, completed[right][row]);
                let slot = left_slot * KNOTS + right_slot;
                let (weight, residual) = if logistic {
                    let probability = sigmoid(f64::from(linear_value));
                    let weight = (probability * (1.0 - probability)).max(0.02);
                    (
                        weight,
                        (f64::from(table.target[row]) - probability) / weight,
                    )
                } else {
                    (1.0, f64::from(table.target[row] - linear_value))
                };
                sums[slot] += weight * residual;
                weights[slot] += weight;
            }
            let mut values: Vec<f32> = sums
                .iter()
                .zip(&weights)
                .map(|(sum, weight)| {
                    if *weight > 0.0 {
                        (0.5 * sum / weight) as f32
                    } else {
                        0.0
                    }
                })
                .collect();
            let center = values
                .iter()
                .zip(&weights)
                .map(|(value, weight)| f64::from(*value) * weight)
                .sum::<f64>()
                / weights.iter().sum::<f64>().max(1e-12);
            values.iter_mut().for_each(|value| *value -= center as f32);
            for (row, linear_value) in linear.iter_mut().enumerate() {
                let slot = nearest_knot(&left_knots, completed[left][row]) * KNOTS
                    + nearest_knot(&right_knots, completed[right][row]);
                *linear_value += values[slot];
            }
            let (left, right, left_knots, right_knots, values) = if left < right {
                (left, right, left_knots, right_knots, values)
            } else {
                let transposed = (0..KNOTS)
                    .flat_map(|new_left| {
                        let values = &values;
                        (0..KNOTS).map(move |new_right| values[new_right * KNOTS + new_left])
                    })
                    .collect();
                (right, left, right_knots, left_knots, transposed)
            };
            interactions.push(InteractionTerm {
                left: left as u32,
                right: right as u32,
                left_knots,
                right_knots,
                values,
            });
            if interactions.len() == 4 {
                break 'pairs;
            }
        }
    }
    Target::Ga2m {
        intercept,
        logistic,
        main_terms: terms,
        interactions,
    }
}

fn fit_mars_target(table: &Table, completed: &[Vec<f32>], screen: &[(usize, f32)]) -> Target {
    let logistic = table
        .target
        .iter()
        .all(|value| *value == 0.0 || *value == 1.0);
    let mut factors = Vec::new();
    let mut basis = Vec::new();
    for &(feature, _) in screen.iter().take(8) {
        let knot = feature_knots(&completed[feature], 3)[1];
        for direction in [-1, 1] {
            factors.push(MarsFactor {
                feature: feature as u32,
                knot,
                direction,
            });
            basis.push(
                completed[feature]
                    .iter()
                    .map(|value| {
                        if direction > 0 {
                            (*value - knot).max(0.0)
                        } else {
                            (knot - *value).max(0.0)
                        }
                    })
                    .collect(),
            );
        }
    }
    let (intercept, coefficients) = fit_basis_coefficients(&table.target, &basis, logistic);
    Target::Mars {
        intercept,
        logistic,
        terms: factors
            .into_iter()
            .zip(coefficients)
            .filter(|(_, coefficient)| coefficient.abs() >= 1e-5)
            .map(|(factor, coefficient)| MarsTerm {
                coefficient,
                factors: vec![factor],
            })
            .collect(),
    }
}

fn fit_oblivious_tree_target(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
) -> Target {
    let logistic = table
        .target
        .iter()
        .all(|value| *value == 0.0 || *value == 1.0);
    let features: Vec<_> = screen.iter().take(4).map(|(feature, _)| *feature).collect();
    let thresholds: Vec<_> = features
        .iter()
        .map(|feature| feature_knots(&completed[*feature], 3)[1])
        .collect();
    let mut sums = vec![0.0f64; 1usize << features.len()];
    let mut counts = vec![0usize; sums.len()];
    for (row, actual) in table.target.iter().enumerate() {
        let leaf = features.iter().zip(&thresholds).enumerate().fold(
            0usize,
            |leaf, (depth, (feature, threshold))| {
                leaf | (((completed[*feature][row] > *threshold) as usize) << depth)
            },
        );
        sums[leaf] += f64::from(*actual);
        counts[leaf] += 1;
    }
    let base = f64::from(table.target.iter().sum::<f32>() / table.rows.max(1) as f32);
    let leaves = sums
        .into_iter()
        .zip(counts)
        .map(|(sum, count)| {
            let value = if count == 0 { base } else { sum / count as f64 };
            if logistic {
                let probability = value.clamp(1e-4, 1.0 - 1e-4);
                (probability / (1.0 - probability)).ln() as f32
            } else {
                value as f32
            }
        })
        .collect();
    Target::ObliviousTree {
        logistic,
        features: features.into_iter().map(|feature| feature as u32).collect(),
        thresholds,
        leaves,
    }
}

#[cfg(not(feature = "gpu-training"))]
fn compact_neural_weight(unit: usize, input: usize) -> f32 {
    let mut value = (unit as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (input as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 30;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    let magnitude = 1.25 + ((value >> 8) & 3) as f32 * 0.5;
    if value & 1 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

#[cfg(not(feature = "gpu-training"))]
fn fit_compact_neural_target_native(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
) -> Target {
    let logistic = table
        .target
        .iter()
        .all(|value| *value == 0.0 || *value == 1.0);
    let selected: Vec<_> = screen
        .iter()
        .take(12)
        .map(|(feature, _)| *feature)
        .collect();
    let hidden_width = selected.len().clamp(4, 12);
    let input_weights: Vec<_> = (0..hidden_width)
        .flat_map(|unit| (0..selected.len()).map(move |input| compact_neural_weight(unit, input)))
        .collect();
    let hidden_biases: Vec<_> = (0..hidden_width)
        .map(|unit| {
            let center = input_weights[unit * selected.len()..(unit + 1) * selected.len()]
                .iter()
                .sum::<f32>();
            let offset = match unit % 3 {
                0 => -0.5,
                1 => 0.0,
                _ => 0.5,
            };
            -0.5 * center + offset
        })
        .collect();
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
                            input_weights[unit * selected.len() + input] * completed[*feature][row]
                        })
                        .sum::<f32>())
                .tanh()
            })
            .collect()
    }));
    let (intercept, coefficients) = fit_basis_coefficients(&table.target, &basis, logistic);
    let (linear_coefficients, output_weights) = coefficients.split_at(selected.len());
    Target::CompactNeuralResidual {
        intercept,
        logistic,
        linear_terms: selected
            .iter()
            .zip(linear_coefficients)
            .filter(|(_, coefficient)| coefficient.abs() >= 1e-5)
            .map(|(&feature, &coefficient)| LinearTerm {
                feature: feature as u32,
                coefficient,
            })
            .collect(),
        hidden_features: selected.into_iter().map(|feature| feature as u32).collect(),
        hidden_width: hidden_width as u8,
        input_weights,
        hidden_biases,
        output_weights: output_weights.to_vec(),
    }
}

#[cfg(feature = "gpu-training")]
fn tensor_values(tensor: tch::Tensor, context: &str) -> Result<Vec<f32>> {
    Vec::<f32>::try_from(tensor.to_device(tch::Device::Cpu).view([-1]))
        .map_err(|error| DopeError::Data(format!("{context}: {error}")))
}

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
        let output_weights = tensor_values(residual.ws.shallow_clone(), "neural output export")?;
        let residual_bias = residual
            .bs
            .as_ref()
            .expect("linear bias")
            .double_value(&[0]) as f32;
        let skip_bias = skip.bs.as_ref().expect("linear bias").double_value(&[0]) as f32;
        let linear_weights = tensor_values(skip.ws.shallow_clone(), "neural skip export")?;
        Ok(Target::CompactNeuralResidual {
            intercept: residual_bias + skip_bias,
            logistic,
            linear_terms: selected
                .iter()
                .copied()
                .zip(linear_weights)
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
            output_weights,
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

pub fn compile_kernel_from_arrays(
    features: &[f32],
    target: &[f32],
    rows: usize,
    columns: usize,
    task: Task,
    options: &CompileOptions,
) -> Result<CompileResult> {
    let table = Table::from_arrays_with_policy(
        features,
        target,
        rows,
        columns,
        task,
        &options.release_policy,
    )?;
    compile_table(&table, task, options)
}

pub fn compile_kernel_from_dir(
    path: &Path,
    task: Task,
    options: &CompileOptions,
) -> Result<CompileResult> {
    let table = Table::read_dataset_dir_with_policy(path, task, &options.release_policy)?;
    compile_table(&table, task, options)
}

fn backend_profile(id: &str) -> Option<(usize, u8, usize, u8)> {
    Some(match id {
        "independent_quantile" => (2, 0, 0, 8),
        "chow_liu" => (2, 1, 0, 8),
        "triangular_autoregressive" => (2, 2, 0, 8),
        "sparse_gaussian_copula" => (3, 3, 1, 10),
        "poet_factor" => (3, 4, 1, 10),
        "truncated_c_vine" => (3, 5, 2, 10),
        "two_component_copula_mixture" => (3, 6, 2, 10),
        "compact_neural_residual_symbolic" => (2, 2, 5, 8),
        "tabsds_rank" => (4, 1, 2, 12),
        "adversarial_random_forest" => (3, 6, 4, 10),
        "conditional_density_forest" => (4, 3, 4, 12),
        "forest_diffusion_vp" => (4, 4, 4, 12),
        "forest_diffusion_flow_matching" => (4, 5, 4, 12),
        "tvae" => (2, 4, 5, 8),
        "ctgan" => (3, 6, 5, 10),
        "taegan" => (3, 5, 5, 10),
        "tabddpm" => (4, 3, 5, 12),
        "tabsyn" => (4, 1, 5, 12),
        "single_table_autoregressive_transformer" => (3, 2, 5, 12),
        "masked_diffusion_transformer" => (4, 5, 5, 16),
        "legacy_tvae" => (2, 4, 5, 8),
        "legacy_ctgan" => (3, 6, 5, 10),
        "legacy_taegan" => (3, 5, 5, 10),
        "legacy_tabddpm" => (4, 3, 5, 12),
        "legacy_tabsyn" => (4, 1, 5, 12),
        "legacy_single_table_autoregressive_transformer" => (3, 2, 5, 12),
        "legacy_masked_diffusion_transformer" => (4, 5, 5, 16),
        "synthetic_pretrained_cross_table" => (3, 1, 1, 10),
        "per_dataset_adapter" => (3, 4, 1, 10),
        "symbolic_latent_diffusion_residual" => (4, 3, 5, 12),
        "symbolic_autoregressive_residual" => (2, 2, 5, 10),
        _ => return None,
    })
}

fn dependence_name(family: u8) -> &'static str {
    match family {
        0 => "independence",
        1 => "chow_liu",
        2 => "triangular_autoregressive",
        3 => "sparse_gaussian_copula",
        4 => "poet_factor",
        5 => "truncated_c_vine",
        6 => "two_component_copula_mixture",
        _ => unreachable!("bounded dependence family"),
    }
}

fn strongest_sparse_edges(
    correlation: Option<&Vec<f32>>,
    fallback: &[CopulaEdge],
    width: usize,
    limit: usize,
) -> Vec<CopulaEdge> {
    let Some(matrix) = correlation else {
        return fallback.iter().take(limit).cloned().collect();
    };
    let mut pairs = Vec::with_capacity(width.saturating_mul(width.saturating_sub(1)) / 2);
    for left in 0..width {
        for right in left + 1..width {
            pairs.push(CopulaEdge {
                parent: left as u32,
                child: right as u32,
                correlation: matrix[left * width + right],
            });
        }
    }
    pairs.sort_by(|left, right| {
        right
            .correlation
            .abs()
            .total_cmp(&left.correlation.abs())
            .then_with(|| left.parent.cmp(&right.parent))
            .then_with(|| left.child.cmp(&right.child))
    });
    pairs.truncate(limit);
    pairs
}

fn poet_dependence(
    correlation: Option<&Vec<f32>>,
    fallback: &[CopulaEdge],
    width: usize,
) -> Dependence {
    let rank = width.clamp(1, 4);
    let mut loadings = Vec::with_capacity(width * rank);
    for feature in 0..width {
        for factor in 0..rank {
            let loading = correlation.map_or_else(
                || f32::from(feature == factor),
                |matrix| matrix[feature * width + factor] / (rank as f32).sqrt(),
            );
            loadings.push(FactorLoading {
                feature: feature as u32,
                factor: factor as u8,
                loading: loading.clamp(-1.0, 1.0),
            });
        }
    }
    Dependence::Poet {
        rank: rank as u8,
        loadings,
        residual_edges: strongest_sparse_edges(
            correlation,
            fallback,
            width,
            width.saturating_mul(2),
        ),
    }
}

fn vine_dependence(
    correlation: Option<&Vec<f32>>,
    fallback: &[CopulaEdge],
    width: usize,
) -> Dependence {
    let sparse = strongest_sparse_edges(correlation, fallback, width, width.saturating_mul(3));
    Dependence::Vine {
        edges: sparse
            .into_iter()
            .enumerate()
            .map(|(index, edge)| VineEdge {
                left: edge.parent,
                right: edge.child,
                conditioning_depth: (index % 4) as u8,
                parameter: edge.correlation,
            })
            .collect(),
    }
}

fn neural_architecture(id: &str) -> Option<NeuralArchitecture> {
    match id {
        "tvae" | "micro_tvae_4_16" | "micro_tvae_8_24" | "micro_tvae_12_32" => {
            Some(NeuralArchitecture::Tvae)
        }
        "single_table_autoregressive_transformer" | "tiny_mat_16_2_32" | "tiny_mat_24_3_48" => {
            Some(NeuralArchitecture::MaskedAutoregressiveTransformer)
        }
        "tabsyn" => Some(NeuralArchitecture::TabSyn),
        "tabddpm_direct_rank" => Some(NeuralArchitecture::TabDdpm),
        _ => None,
    }
}

fn neural_profile(id: &str) -> NeuralProfile {
    match id {
        "micro_tvae_4_16" => NeuralProfile::MicroTvae4,
        "micro_tvae_8_24" => NeuralProfile::MicroTvae8,
        "micro_tvae_12_32" => NeuralProfile::MicroTvae12,
        "tiny_mat_16_2_32" => NeuralProfile::TinyMat16,
        "tiny_mat_24_3_48" => NeuralProfile::TinyMat24,
        _ => NeuralProfile::Full,
    }
}

fn randomized_gaussian_ranks(values: &[f32], seed: u64, column: usize) -> (Vec<f32>, Vec<f32>) {
    let mut observed = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    observed.sort_by(f32::total_cmp);
    let normal = Normal::new(0.0, 1.0).expect("standard normal");
    let mut ranks = Vec::with_capacity(values.len());
    let mut missing = Vec::with_capacity(values.len());
    for (row, value) in values.iter().enumerate() {
        if !value.is_finite() || observed.is_empty() {
            ranks.push(0.0);
            missing.push(1.0);
            continue;
        }
        let lower = observed.partition_point(|candidate| *candidate < *value - 1e-7);
        let upper = observed.partition_point(|candidate| *candidate <= *value + 1e-7);
        let counter = seed
            ^ (row as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
            ^ (column as u64).wrapping_mul(0xa076_1d64_78bd_642f);
        let jitter = (splitmix64(counter) >> 11) as f64 / (1u64 << 53) as f64;
        let probability = (lower as f64 + jitter * (upper - lower).max(1) as f64 + 0.5)
            / (observed.len() + 1) as f64;
        ranks.push(normal.inverse_cdf(probability.clamp(1e-6, 1.0 - 1e-6)) as f32);
        missing.push(0.0);
    }
    (ranks, missing)
}

fn correlation_tree_permutation(feature_ranks: &[Vec<f32>], target_ranks: &[f32]) -> Vec<u32> {
    let width = feature_ranks.len();
    if width <= 1 {
        return (0..width as u32).collect();
    }
    let root = (0..width)
        .max_by(|&left, &right| {
            dot(&feature_ranks[left], target_ranks)
                .abs()
                .total_cmp(&dot(&feature_ranks[right], target_ranks).abs())
                .then_with(|| right.cmp(&left))
        })
        .unwrap_or(0);
    let mut selected = vec![false; width];
    let mut parent = vec![root; width];
    let mut best = vec![f32::NEG_INFINITY; width];
    selected[root] = true;
    for feature in 0..width {
        best[feature] = dot(&feature_ranks[root], &feature_ranks[feature]).abs();
    }
    for _ in 1..width {
        let child = (0..width)
            .filter(|&feature| !selected[feature])
            .max_by(|&left, &right| {
                best[left]
                    .total_cmp(&best[right])
                    .then_with(|| right.cmp(&left))
            })
            .expect("unselected feature");
        selected[child] = true;
        for feature in 0..width {
            let score = dot(&feature_ranks[child], &feature_ranks[feature]).abs();
            if !selected[feature] && score > best[feature] {
                best[feature] = score;
                parent[feature] = child;
            }
        }
    }
    let mut children = vec![Vec::new(); width];
    for child in 0..width {
        if child != root {
            children[parent[child]].push(child);
        }
    }
    for entries in &mut children {
        entries.sort_unstable();
    }
    let mut permutation = Vec::with_capacity(width);
    let mut stack = vec![root];
    while let Some(feature) = stack.pop() {
        permutation.push(feature as u32);
        stack.extend(children[feature].iter().rev());
    }
    permutation
}

fn neural_training_data(
    table: &Table,
    fits: &[ColumnFit],
    seed: u64,
) -> crate::neural_train::NeuralTrainingData {
    let tokens = table.features + 1;
    let mut rank_columns = Vec::with_capacity(tokens);
    let mut missing_columns = Vec::with_capacity(tokens);
    for (column, values) in table
        .columns
        .iter()
        .chain(std::iter::once(&table.target))
        .enumerate()
    {
        let (ranks, missing) = randomized_gaussian_ranks(values, seed, column);
        rank_columns.push(ranks);
        missing_columns.push(missing);
    }
    let feature_permutation = correlation_tree_permutation(
        &rank_columns[..table.features],
        &rank_columns[table.features],
    );
    let mut ranks = Vec::with_capacity(table.rows * tokens);
    let mut missing = Vec::with_capacity(table.rows * tokens);
    for row in 0..table.rows {
        for column in feature_permutation
            .iter()
            .map(|value| *value as usize)
            .chain(std::iter::once(table.features))
        {
            ranks.push(rank_columns[column][row]);
            missing.push(missing_columns[column][row]);
        }
    }
    let target_fit = fit_column(&table.target);
    let target_marginal = target_fit
        .discrete
        .unwrap_or_else(|| Marginal::QuantileSpline {
            values: target_fit.quantiles[4].clone(),
        });
    let mut sorted_target = table.target.clone();
    sorted_target.sort_by(f32::total_cmp);
    let lower_tail = quantile(&sorted_target, 0.1);
    let upper_tail = quantile(&sorted_target, 0.9);
    let binary = table.target.iter().all(|value| matches!(*value, 0.0 | 1.0));
    let ones = table.target.iter().filter(|value| **value >= 0.5).count();
    let zeroes = table.rows.saturating_sub(ones);
    let rare_class = if ones < zeroes {
        Some(true)
    } else if zeroes < ones {
        Some(false)
    } else {
        None
    };
    let row_weights = table
        .target
        .iter()
        .map(|target| {
            if (binary && rare_class == Some(*target >= 0.5))
                || (!binary && (*target <= lower_tail || *target >= upper_tail))
            {
                2.0
            } else {
                1.0
            }
        })
        .collect::<Vec<_>>();
    let normalization = fits
        .iter()
        .map(|fit| RankNormalization {
            location: 0.0,
            scale: 1.0,
            missing_probability: fit.schema.missing_probability,
            randomized_discrete: !matches!(
                fit.schema.kind,
                SchemaKind::Continuous | SchemaKind::Inflated
            ),
        })
        .chain(std::iter::once(RankNormalization {
            location: 0.0,
            scale: 1.0,
            missing_probability: 0.0,
            randomized_discrete: binary,
        }))
        .collect::<Vec<_>>();
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"joint-rank-training-v1");
    for value in &ranks {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    for value in &missing {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    crate::neural_train::NeuralTrainingData {
        rows: table.rows,
        features: table.features,
        ranks,
        missing,
        row_weights,
        feature_permutation,
        target_marginal,
        normalization,
        training_hash: hasher.finalize().to_hex().to_string(),
    }
}

fn compile_neural_table(
    table: &Table,
    task: Task,
    options: &CompileOptions,
    architecture: NeuralArchitecture,
    seed: u64,
    started: Instant,
) -> Result<CompileResult> {
    if architecture == NeuralArchitecture::TabDdpm
        && !matches!(
            options.release_policy.tier,
            AnonymizationTier::L0 | AnonymizationTier::L1
        )
    {
        return Err(DopeError::Data(
            "direct rank TabDDPM is eligible only for L1 or research L0".into(),
        ));
    }
    if !matches!(options.neural_target_weight, 2.0 | 4.0)
        || !matches!(options.neural_structural_penalty, 0.0 | 0.1)
    {
        return Err(DopeError::Data(
            "neural loss configuration is outside the frozen 2x2 grid".into(),
        ));
    }
    let fits = table
        .columns
        .iter()
        .map(|column| fit_column(column))
        .collect::<Vec<_>>();
    let data = neural_training_data(table, &fits, seed);
    let candidate_id = options
        .backend_id
        .clone()
        .expect("restricted neural candidate");
    let candidate_seed = {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&seed.to_le_bytes());
        hasher.update(candidate_id.as_bytes());
        u64::from_le_bytes(hasher.finalize().as_bytes()[..8].try_into().unwrap())
    };
    let generator: JointGenerator = crate::neural_train::fit_joint_generator(
        &data,
        crate::neural_train::NeuralTrainingConfig {
            architecture,
            profile: neural_profile(&candidate_id),
            target_weight: options.neural_target_weight,
            structural_penalty: options.neural_structural_penalty,
            seed: candidate_seed,
            deadline: started
                + options
                    .deadline
                    .unwrap_or(Duration::from_secs(600))
                    .min(Duration::from_secs(600)),
        },
    )?;
    let marginals = fits
        .iter()
        .map(|fit| {
            fit.discrete
                .clone()
                .unwrap_or_else(|| Marginal::QuantileSpline {
                    values: fit.quantiles[4].clone(),
                })
        })
        .collect::<Vec<_>>();
    let kernel = Kernel {
        task,
        rows_fitted: table.rows as u64,
        features: table.features as u32,
        seed,
        seed_policy: u8::from(options.seed.is_none()),
        quantization_bits: 8,
        compliant: false,
        schema: fits.iter().map(|fit| fit.schema.clone()).collect(),
        marginals,
        program: KernelProgram::NeuralJoint(generator),
    };
    let artifact = encode_kernel(&kernel)?;
    if !options
        .release_policy
        .accepts_artifact_bytes(artifact.len())
    {
        return Err(DopeError::Data(format!(
            "no encoded neural candidate fits the release policy; smallest observed size: {} bytes",
            artifact.len()
        )));
    }
    let report = CandidateReport {
        candidate_id: candidate_id.clone(),
        artifact_bytes: artifact.len(),
        bits_per_original_cell: 8.0 * artifact.len() as f64
            / (table.rows * (table.features + 1)).max(1) as f64,
        score: 0.0,
        compliant: false,
        quantization_bits: 8,
        marginal_knots: 65,
        dependence: "neural_joint".into(),
        target: match architecture {
            NeuralArchitecture::Tvae => "tvae",
            NeuralArchitecture::MaskedAutoregressiveTransformer => {
                "masked_autoregressive_transformer"
            }
            NeuralArchitecture::TabSyn => "tabsyn",
            NeuralArchitecture::TabDdpm => "tabddpm",
        }
        .into(),
        utility_retention: 0.0,
        driver_agreement: 0.0,
        proxy_joint_fidelity: 0.0,
        proxy_membership_auc: 0.5,
        failed_gates: vec!["production_evidence_unmeasured".into()],
    };
    let compile_report = CompileReport {
        format: "dope-kernel-compile-report".into(),
        version: 3,
        release_policy: options.release_policy.clone(),
        release_policy_hash: options.release_policy.hash(),
        task: task.as_str().into(),
        rows: table.rows,
        positional_features: table.features,
        selected_candidate: candidate_id.clone(),
        artifact_bytes: artifact.len(),
        effective_bytes: artifact.len(),
        bits_per_original_cell: report.bits_per_original_cell,
        compliant: false,
        failed_gates: report.failed_gates.clone(),
        candidate_count: 1,
        frontier_count: 1,
        beam_width: 1,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        contains_row_payloads: false,
        contains_row_references: false,
        probe_failures: BTreeMap::new(),
    };
    Ok(CompileResult {
        candidate_artifacts: BTreeMap::from([(candidate_id, artifact.clone())]),
        artifact,
        report: compile_report,
        candidates: vec![report.clone()],
        pareto_frontier: vec![report],
    })
}

fn compile_table(table: &Table, task: Task, options: &CompileOptions) -> Result<CompileResult> {
    if let Some(architecture) = options.backend_id.as_deref().and_then(neural_architecture) {
        let started = Instant::now();
        return compile_neural_table(
            table,
            task,
            options,
            architecture,
            options.seed.unwrap_or_else(|| stable_seed(table)),
            started,
        );
    }
    let symbolic = compile_symbolic_table(table, task, options)?;
    #[cfg(feature = "gpu-training")]
    {
        probe_compact_neural(table, task, options, symbolic)
    }
    #[cfg(not(feature = "gpu-training"))]
    {
        Ok(symbolic)
    }
}

#[cfg(feature = "gpu-training")]
fn probe_compact_neural(
    table: &Table,
    task: Task,
    options: &CompileOptions,
    mut result: CompileResult,
) -> Result<CompileResult> {
    if options.backend_id.is_some() {
        return Ok(result);
    }
    let probes: &[&str] = match options.release_policy.tier {
        AnonymizationTier::L3 => &["micro_tvae_4_16", "micro_tvae_8_24"],
        AnonymizationTier::L2 => &["micro_tvae_4_16", "micro_tvae_8_24", "tiny_mat_16_2_32"],
        AnonymizationTier::L0 | AnonymizationTier::L1 => &[],
    };
    if probes.is_empty() {
        return Ok(result);
    }
    let started = Instant::now();
    let seed = options.seed.unwrap_or_else(|| stable_seed(table));
    for &id in probes {
        if options
            .deadline
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            break;
        }
        let mut restricted = options.clone();
        restricted.backend_id = Some(id.into());
        if let Some(limit) = options.deadline {
            restricted.deadline = Some(limit.saturating_sub(started.elapsed()));
        }
        let architecture = neural_architecture(id).expect("frozen compact neural candidate");
        // Training and proxy predictions never establish size eligibility. Only
        // the final encoded DPK passes the policy byte cap in this function.
        match compile_neural_table(table, task, &restricted, architecture, seed, Instant::now()) {
            Ok(neural) => {
                result
                    .candidate_artifacts
                    .extend(neural.candidate_artifacts);
                result.candidates.extend(neural.candidates);
            }
            Err(error) => {
                result
                    .report
                    .probe_failures
                    .insert(id.into(), error.to_string());
            }
        }
    }
    result.pareto_frontier = pareto(&result.candidates);
    let selected = result
        .candidates
        .iter()
        .min_by(|left, right| {
            maximum_normalized_gate_shortfall(left)
                .total_cmp(&maximum_normalized_gate_shortfall(right))
                .then_with(|| left.artifact_bytes.cmp(&right.artifact_bytes))
                .then_with(|| left.candidate_id.cmp(&right.candidate_id))
        })
        .expect("symbolic baseline exists");
    result.artifact = result.candidate_artifacts[&selected.candidate_id].clone();
    result.report.selected_candidate = selected.candidate_id.clone();
    result.report.artifact_bytes = result.artifact.len();
    result.report.effective_bytes = result.artifact.len();
    result.report.bits_per_original_cell = selected.bits_per_original_cell;
    result.report.compliant = selected.compliant;
    result.report.failed_gates = selected.failed_gates.clone();
    result.report.candidate_count = result.candidates.len();
    result.report.frontier_count = result.pareto_frontier.len();
    result.report.elapsed_seconds += started.elapsed().as_secs_f64();
    Ok(result)
}

fn compile_symbolic_table(
    table: &Table,
    task: Task,
    options: &CompileOptions,
) -> Result<CompileResult> {
    let started = Instant::now();
    let seed = options.seed.unwrap_or_else(|| stable_seed(table));
    let restricted_profile = options
        .backend_id
        .as_deref()
        .map(|id| {
            backend_profile(id)
                .ok_or_else(|| DopeError::Data(format!("unknown frozen empirical backend {id}")))
        })
        .transpose()?;
    if options.quantization_profiles.is_empty()
        || options
            .quantization_profiles
            .iter()
            .any(|profile| !QUANTIZATION_LADDER.contains(profile))
    {
        return Err(DopeError::Data(
            "invalid quantization profile selection".into(),
        ));
    }
    let seed_policy = options.seed.is_none() as u8;
    let width = table.features;
    let beam_width = if restricted_profile.is_some() {
        1
    } else {
        options
            .beam_width
            .unwrap_or(if width > 256 { 72 } else { 108 })
            .max(1)
    };
    let deadline = options.deadline.map(|duration| started + duration);
    let fits: Vec<ColumnFit> = table
        .columns
        .par_iter()
        .map(|column| fit_column(column))
        .collect();
    let schema: Vec<_> = fits.iter().map(|fit| fit.schema.clone()).collect();
    let completed: Vec<_> = fits.iter().map(|fit| fit.completed.clone()).collect();
    let ranks: Vec<_> = completed
        .par_iter()
        .map(|column| rank_normalize(column))
        .collect();
    let correlation = (width <= 512).then(|| full_correlation(&ranks));
    let edges = correlation.as_ref().map_or_else(
        || wide_tree(&ranks),
        |matrix| maximum_spanning_tree(matrix, width),
    );
    let triangular_terms = wide_tree(&ranks)
        .into_iter()
        .map(|edge| TriangularTerm {
            parent: edge.parent,
            child: edge.child,
            coefficient: edge.correlation,
        })
        .collect::<Vec<_>>();
    let screen = target_screen(&completed, &table.target);
    let (target_model, noise, _fitted_utility) = match task {
        Task::Regression => fit_linear_target(table, &completed, &screen),
        Task::Binary => fit_logistic_target(table, &completed, &screen),
    };
    let baseline_name = if task == Task::Regression {
        "sparse_linear"
    } else {
        "sparse_logistic"
    };
    let mut target_models = vec![(baseline_name, target_model.clone(), noise.clone())];
    let gam = fit_gam_target(table, &completed, &screen);
    target_models.push((
        "sparse_gam",
        gam.clone(),
        fit_noise(table, &completed, &screen, &gam),
    ));
    let ga2m = fit_ga2m_target(table, &completed, &screen);
    target_models.push((
        "ga2m",
        ga2m.clone(),
        fit_noise(table, &completed, &screen, &ga2m),
    ));
    let mars = fit_mars_target(table, &completed, &screen);
    target_models.push((
        "mars",
        mars.clone(),
        fit_noise(table, &completed, &screen, &mars),
    ));
    let oblivious_tree = fit_oblivious_tree_target(table, &completed, &screen);
    target_models.push((
        "oblivious_tree",
        oblivious_tree.clone(),
        fit_noise(table, &completed, &screen, &oblivious_tree),
    ));
    let compact_neural = if restricted_profile.is_some_and(|profile| profile.2 != 5) {
        target_model.clone()
    } else {
        fit_compact_neural_target(table, &completed, &screen, seed)?
    };
    target_models.push((
        "compact_neural_residual",
        compact_neural.clone(),
        fit_noise(table, &completed, &screen, &compact_neural),
    ));
    let seed_kernel = Kernel {
        task,
        rows_fitted: table.rows as u64,
        features: width as u32,
        seed,
        seed_policy,
        quantization_bits: 8,
        compliant: false,
        schema,
        marginals: Vec::new(),
        program: KernelProgram::Symbolic {
            dependence: Dependence::Independent,
            target: target_model,
            noise,
        },
    };

    let mut artifacts = BTreeMap::<String, Vec<u8>>::new();
    let mut candidates = Vec::<CandidateReport>::new();
    let dataset_descriptor = describe_table(table, task);
    let mut specs: Vec<(usize, u8, usize, u8, f64)> = Vec::new();
    for (knot_index, &knots) in KNOT_LADDER.iter().enumerate() {
        for dependence_family in 0..7 {
            for (target_index, (target_name, target_model, _)) in target_models.iter().enumerate() {
                for &bits in &options.quantization_profiles {
                    if restricted_profile.is_some_and(|profile| {
                        profile != (knot_index, dependence_family, target_index, bits)
                    }) {
                        continue;
                    }
                    let dependence_name = dependence_name(dependence_family);
                    let descriptor = CandidateDescriptor {
                        quantization_bits: bits,
                        marginal_knots: knots,
                        dependence: dependence_name.into(),
                        target: (*target_name).into(),
                        effective_bytes: 56
                            + width * (knots * 2 + 8)
                            + usize::from(dependence_family != 0) * width.saturating_sub(1) * 12
                            + target_parameter_bytes(target_model),
                    };
                    let priority = options.language.as_ref().map_or(0.0, |language| {
                        language.candidate_priority(&dataset_descriptor, &descriptor)
                    });
                    specs.push((knot_index, dependence_family, target_index, bits, priority));
                }
            }
        }
    }
    specs.sort_by(|left, right| {
        right
            .4
            .total_cmp(&left.4)
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.3.cmp(&right.3))
    });
    let mut selected_specs: Vec<_> = specs.into_iter().take(beam_width).collect();
    if restricted_profile.is_none() {
        let baseline = (0, 0, 0, options.quantization_profiles[0]);
        if !selected_specs
            .iter()
            .any(|spec| (spec.0, spec.1, spec.2, spec.3) == baseline)
        {
            selected_specs.push((baseline.0, baseline.1, baseline.2, baseline.3, 0.0));
        }
    }
    let mut smallest_observed = None::<usize>;
    for (knot_index, dependence_family, target_index, bits, _) in selected_specs {
        if deadline.is_some_and(|limit| Instant::now() > limit) && !candidates.is_empty() {
            break;
        }
        let knots = KNOT_LADDER[knot_index];
        let marginals: Vec<Marginal> = fits
            .iter()
            .map(|fit| {
                fit.discrete
                    .clone()
                    .unwrap_or_else(|| Marginal::QuantileSpline {
                        values: fit.quantiles[knot_index].clone(),
                    })
            })
            .collect();
        let marginal_fidelity = 1.0 - 0.035 / (knots as f64).sqrt();
        let (dependence, dependence_name, dependence_fidelity) = match dependence_family {
            0 => {
                let fidelity = correlation.as_ref().map_or(0.85, |matrix| {
                    let mean = matrix
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| index / width != index % width)
                        .map(|(_, value)| f64::from(value.abs()))
                        .sum::<f64>()
                        / (width * width - width).max(1) as f64;
                    (1.0 - mean / 0.5).clamp(0.0, 1.0)
                });
                (Dependence::Independent, "independence", fidelity)
            }
            1 => (
                Dependence::ChowLiu {
                    root: 0,
                    edges: edges.clone(),
                },
                "chow_liu",
                0.97,
            ),
            2 => (
                Dependence::Triangular {
                    terms: triangular_terms.clone(),
                },
                "triangular_autoregressive",
                0.98,
            ),
            3 => (
                Dependence::SparseGraph {
                    edges: strongest_sparse_edges(
                        correlation.as_ref(),
                        &edges,
                        width,
                        width.saturating_mul(4),
                    ),
                },
                "sparse_gaussian_copula",
                0.985,
            ),
            4 => (
                poet_dependence(correlation.as_ref(), &edges, width),
                "poet_factor",
                0.985,
            ),
            5 => (
                vine_dependence(correlation.as_ref(), &edges, width),
                "truncated_c_vine",
                0.99,
            ),
            6 => (
                Dependence::Mixture {
                    weights: vec![0.5, 0.5],
                    components: vec![
                        Dependence::ChowLiu {
                            root: 0,
                            edges: edges.clone(),
                        },
                        Dependence::Triangular {
                            terms: triangular_terms.clone(),
                        },
                    ],
                },
                "two_component_copula_mixture",
                0.99,
            ),
            _ => unreachable!("bounded dependence family"),
        };
        let (target_name, target_model, noise) = &target_models[target_index];
        let mut kernel = seed_kernel.clone();
        kernel.marginals = marginals.clone();
        kernel.program = KernelProgram::Symbolic {
            dependence: dependence.clone(),
            target: target_model.clone(),
            noise: noise.clone(),
        };
        kernel = quantized_kernel(&kernel, bits);
        let utility = quantized_target_utility(&kernel, table, &completed);
        let driver_agreement = quantized_driver_agreement(&kernel, &screen);
        let quantization_penalty = 0.5 / ((1u32 << bits) - 1) as f64;
        let proxy_joint_fidelity = 0.55
            * (marginal_fidelity - quantization_penalty).clamp(0.0, 1.0)
            + 0.45 * dependence_fidelity;
        let proxy_membership_auc = 0.5;
        let mut failed_gates = Vec::new();
        if utility < 0.99 {
            failed_gates.push("utility".to_string());
        }
        if driver_agreement < 0.95 {
            failed_gates.push("driver".to_string());
        }
        if proxy_joint_fidelity < 0.90 {
            failed_gates.push("proxy_joint_fidelity".to_string());
        }
        if proxy_membership_auc > 0.60 {
            failed_gates.push("proxy_anti_memorization".to_string());
        }
        failed_gates.push("production_evidence_unmeasured".to_string());
        // Compilation sees train.csv only. No train-only heuristic may certify an artifact.
        kernel.compliant = false;
        let artifact = encode_kernel(&kernel)?;
        smallest_observed =
            Some(smallest_observed.map_or(artifact.len(), |size| size.min(artifact.len())));
        if !options
            .release_policy
            .accepts_artifact_bytes(artifact.len())
        {
            continue;
        }
        let candidate_id = options
            .backend_id
            .clone()
            .unwrap_or_else(|| format!("k{knots}-{dependence_name}-{target_name}-q{bits}"));
        let score = 0.45 * utility + 0.25 * driver_agreement + 0.30 * proxy_joint_fidelity;
        let record = CandidateReport {
            candidate_id: candidate_id.clone(),
            artifact_bytes: artifact.len(),
            bits_per_original_cell: 8.0 * artifact.len() as f64 / (table.rows * (width + 1)) as f64,
            score,
            compliant: kernel.compliant,
            quantization_bits: bits,
            marginal_knots: knots,
            dependence: dependence_name.into(),
            target: (*target_name).into(),
            utility_retention: utility,
            driver_agreement,
            proxy_joint_fidelity,
            proxy_membership_auc,
            failed_gates,
        };
        artifacts.insert(candidate_id, artifact);
        candidates.push(record);
    }
    if candidates.is_empty() {
        return Err(DopeError::Data(match smallest_observed {
            Some(size) => format!(
                "no encoded candidate fits the release policy; smallest observed size: {size} bytes"
            ),
            None => "deadline expired before baseline encoding".into(),
        }));
    }
    let frontier = pareto(&candidates);
    let selected = candidates
        .iter()
        .min_by(|left, right| {
            maximum_normalized_gate_shortfall(left)
                .total_cmp(&maximum_normalized_gate_shortfall(right))
                .then_with(|| left.artifact_bytes.cmp(&right.artifact_bytes))
                .then_with(|| left.candidate_id.cmp(&right.candidate_id))
        })
        .unwrap()
        .clone();
    let artifact = artifacts.get(&selected.candidate_id).unwrap().clone();
    let report = CompileReport {
        format: "dope-kernel-compile-report".into(),
        version: 3,
        release_policy: options.release_policy.clone(),
        release_policy_hash: options.release_policy.hash(),
        task: task.as_str().into(),
        rows: table.rows,
        positional_features: width,
        selected_candidate: selected.candidate_id.clone(),
        artifact_bytes: artifact.len(),
        effective_bytes: artifact.len(),
        bits_per_original_cell: selected.bits_per_original_cell,
        compliant: selected.compliant,
        failed_gates: selected.failed_gates.clone(),
        candidate_count: candidates.len(),
        frontier_count: frontier.len(),
        beam_width,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        contains_row_payloads: false,
        contains_row_references: false,
        probe_failures: BTreeMap::new(),
    };
    Ok(CompileResult {
        artifact,
        candidate_artifacts: artifacts,
        report,
        candidates,
        pareto_frontier: frontier,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_small_cap_reports_smallest_observed_encoded_candidate() {
        let options = CompileOptions {
            release_policy: ReleasePolicy::new(AnonymizationTier::L3, Some(1), false).unwrap(),
            beam_width: Some(1),
            ..Default::default()
        };
        let error = compile_kernel_from_arrays(
            &[0.0, 0.25, 0.75, 1.0],
            &[0.0, 0.25, 0.75, 1.0],
            4,
            1,
            Task::Regression,
            &options,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("smallest observed size:"));
    }

    #[test]
    fn near_equal_discrete_values_are_counted_without_exact_search_panics() {
        let fit = fit_column(&[0.1, 0.100_000_05, 0.2, 0.2]);
        assert!(fit.discrete.is_some());
    }
    use crate::codec::decode_kernel;
    use std::collections::BTreeSet;

    #[test]
    fn compiles_deterministically_and_keeps_positions() {
        let rows = 100;
        let features: Vec<f32> = (0..rows)
            .flat_map(|row| [0.1, (row % 2) as f32, row as f32 / rows as f32])
            .collect();
        let target: Vec<f32> = (0..rows)
            .map(|row| 0.2 + 0.6 * row as f32 / rows as f32)
            .collect();
        let options = CompileOptions {
            seed: Some(7),
            ..Default::default()
        };
        let first =
            compile_kernel_from_arrays(&features, &target, rows, 3, Task::Regression, &options)
                .unwrap();
        let second =
            compile_kernel_from_arrays(&features, &target, rows, 3, Task::Regression, &options)
                .unwrap();
        assert_eq!(first.artifact, second.artifact);
        let kernel = decode_kernel(&first.artifact).unwrap();
        assert!(matches!(kernel.schema[0].kind, SchemaKind::Constant));
        assert!(matches!(kernel.schema[1].kind, SchemaKind::Binary));
    }

    #[test]
    fn tournaments_all_fitted_targets_on_nonlinear_data() {
        let rows = 256;
        let features: Vec<f32> = (0..rows)
            .flat_map(|row| {
                let left = (row % 16) as f32 / 15.0;
                let right = (row / 16) as f32 / 15.0;
                [left, right]
            })
            .collect();
        let target: Vec<f32> = (0..rows)
            .map(|row| {
                let left = row % 16 >= 8;
                let right = row / 16 >= 8;
                if left == right { 0.1 } else { 0.9 }
            })
            .collect();
        let options = CompileOptions {
            seed: Some(1729),
            beam_width: Some(18),
            quantization_profiles: vec![8],
            language: None,
            deadline: None,
            backend_id: None,
            neural_target_weight: 2.0,
            neural_structural_penalty: 0.0,
            release_policy: ReleasePolicy::new(AnonymizationTier::L0, None, false).unwrap(),
        };
        let result =
            compile_kernel_from_arrays(&features, &target, rows, 2, Task::Regression, &options)
                .unwrap();
        let target_families: BTreeSet<_> = result
            .candidates
            .iter()
            .map(|candidate| candidate.target.as_str())
            .collect();
        assert_eq!(
            target_families,
            BTreeSet::from([
                "compact_neural_residual",
                "ga2m",
                "mars",
                "oblivious_tree",
                "sparse_gam",
                "sparse_linear",
            ])
        );
        let dependence_families: BTreeSet<_> = result
            .candidates
            .iter()
            .map(|candidate| candidate.dependence.as_str())
            .collect();
        assert_eq!(
            dependence_families,
            BTreeSet::from(["chow_liu", "independence", "triangular_autoregressive",])
        );
        let baseline = result
            .candidates
            .iter()
            .find(|candidate| candidate.target == "sparse_linear")
            .unwrap();
        let tree = result
            .candidates
            .iter()
            .find(|candidate| candidate.target == "oblivious_tree")
            .unwrap();
        assert!(tree.utility_retention > baseline.utility_retention + 0.5);
        assert_eq!(result.report.candidate_count, 18);
        decode_kernel(&result.artifact).unwrap();
    }
}

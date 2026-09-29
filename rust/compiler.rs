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

include!("compiler/target_fitting.rs");
include!("compiler/candidate_search.rs");
include!("compiler/neural_candidates.rs");

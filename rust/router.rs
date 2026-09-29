use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "gpu-training")]
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::data::Table;
use crate::error::{DopeError, Result};
use crate::model::Task;

pub const MAX_ROUTER_BUNDLE_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_PAIRWISE_FEATURES: usize = 64;
pub const ROUTER_OUTPUTS: usize = 10;
pub const DATASET_SKETCH_VERSION: u8 = 2;
pub const ROUTER_ACTION_NAMES: [&str; ROUTER_OUTPUTS] = [
    "retention_mean",
    "retention_lower",
    "kpi_failure_probability",
    "runtime_p50_ms",
    "runtime_p95_ms",
    "memory_p95_bytes",
    "artifact_bytes",
    "timeout_probability",
    "uncertainty",
    "expected_regret",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DatasetSketch {
    pub version: u8,
    pub values: Vec<f32>,
}

#[derive(Clone)]
struct FeatureSummary {
    values: Vec<f32>,
    canonical_values: Vec<f32>,
    observed: Vec<f32>,
}

fn quantile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let position = q * (sorted.len() - 1) as f32;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * (position - low as f32)
}

fn moments(sorted: &[f32]) -> (f32, f32, f32) {
    if sorted.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    // Values are sorted first so row permutations have identical accumulation order.
    let mean = sorted.iter().map(|&v| f64::from(v)).sum::<f64>() / sorted.len() as f64;
    let variance = sorted
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / sorted.len() as f64;
    let deviation = variance.sqrt();
    let skew = if deviation > 0.0 {
        sorted
            .iter()
            .map(|&v| ((f64::from(v) - mean) / deviation).powi(3))
            .sum::<f64>()
            / sorted.len() as f64
    } else {
        0.0
    };
    (mean as f32, variance as f32, skew as f32)
}

fn correlation(mut pairs: Vec<(f32, f32)>) -> f32 {
    if pairs.len() < 2 {
        return 0.0;
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
    let n = pairs.len() as f64;
    let mean_x = pairs.iter().map(|p| f64::from(p.0)).sum::<f64>() / n;
    let mean_y = pairs.iter().map(|p| f64::from(p.1)).sum::<f64>() / n;
    let covariance = pairs
        .iter()
        .map(|p| (f64::from(p.0) - mean_x) * (f64::from(p.1) - mean_y))
        .sum::<f64>();
    let var_x = pairs
        .iter()
        .map(|p| (f64::from(p.0) - mean_x).powi(2))
        .sum::<f64>();
    let var_y = pairs
        .iter()
        .map(|p| (f64::from(p.1) - mean_y).powi(2))
        .sum::<f64>();
    if var_x == 0.0 || var_y == 0.0 {
        0.0
    } else {
        (covariance / (var_x * var_y).sqrt()).clamp(-1.0, 1.0) as f32
    }
}

fn distribution(values: &mut [f32]) -> [f32; 6] {
    values.sort_by(f32::total_cmp);
    [
        values.first().copied().unwrap_or(0.0),
        quantile(values, 0.25),
        quantile(values, 0.5),
        quantile(values, 0.75),
        values.last().copied().unwrap_or(0.0),
        if values.is_empty() {
            0.0
        } else {
            values.iter().map(|&v| f64::from(v)).sum::<f64>() as f32 / values.len() as f32
        },
    ]
}

impl DatasetSketch {
    /// Constructs a name-, path-, row-, and column-order-independent sketch from a loaded
    /// train table. Callers are responsible for loading only `train.csv`; this API accepts no
    /// validation path and retains no row payloads.
    pub fn from_train(table: &Table, task: Task) -> Self {
        let mut target = table.target.clone();
        target.sort_by(f32::total_cmp);
        let (target_mean, target_variance, target_skew) = moments(&target);
        let target_rare = if task == Task::Binary {
            target_mean.min(1.0 - target_mean)
        } else {
            let q25 = quantile(&target, 0.25);
            let q75 = quantile(&target, 0.75);
            let iqr = q75 - q25;
            let low = q25 - 1.5 * iqr;
            let high = q75 + 1.5 * iqr;
            target.iter().filter(|&&v| v < low || v > high).count() as f32
                / target.len().max(1) as f32
        };

        let mut features = Vec::with_capacity(table.features);
        for column in &table.columns {
            let mut observed: Vec<f32> = column.iter().copied().filter(|v| v.is_finite()).collect();
            observed.sort_by(f32::total_cmp);
            let (mean, variance, skew) = moments(&observed);
            let canonical_values = observed.clone();
            let unique = observed.windows(2).filter(|w| w[0] != w[1]).count()
                + usize::from(!observed.is_empty());
            let q10 = quantile(&observed, 0.10);
            let q25 = quantile(&observed, 0.25);
            let q50 = quantile(&observed, 0.50);
            let q75 = quantile(&observed, 0.75);
            let q90 = quantile(&observed, 0.90);
            let tail = (q90 - q10) / (q75 - q25).abs().max(1e-6);
            let association = correlation(
                column
                    .iter()
                    .copied()
                    .zip(table.target.iter().copied())
                    .filter(|(x, y)| x.is_finite() && y.is_finite())
                    .collect(),
            )
            .abs();
            features.push(FeatureSummary {
                values: vec![
                    1.0 - observed.len() as f32 / table.rows.max(1) as f32,
                    mean,
                    variance,
                    skew,
                    q10,
                    q25,
                    q50,
                    q75,
                    q90,
                    unique as f32 / observed.len().max(1) as f32,
                    tail,
                    association,
                ],
                canonical_values,
                observed: column.clone(),
            });
        }
        // Canonical feature order is derived solely from contents, never original position.
        features.sort_by(|a, b| {
            b.values[2]
                .total_cmp(&a.values[2])
                .then_with(|| {
                    a.values
                        .iter()
                        .zip(&b.values)
                        .find_map(|(x, y)| (x != y).then(|| x.total_cmp(y)))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    a.canonical_values
                        .iter()
                        .zip(&b.canonical_values)
                        .find_map(|(x, y)| (x != y).then(|| x.total_cmp(y)))
                        .unwrap_or_else(|| a.canonical_values.len().cmp(&b.canonical_values.len()))
                })
        });

        let mut values = vec![
            if task == Task::Binary { 1.0 } else { 0.0 },
            (table.rows as f32).ln_1p(),
            (table.features as f32).ln_1p(),
            target_mean,
            target_variance,
            target_skew,
            quantile(&target, 0.10),
            quantile(&target, 0.50),
            quantile(&target, 0.90),
            target_rare,
        ];
        for index in 0..12 {
            let mut across: Vec<f32> = features.iter().map(|f| f.values[index]).collect();
            values.extend(distribution(&mut across));
        }

        let selected = &features[..features.len().min(MAX_PAIRWISE_FEATURES)];
        // Preserve the canonical top-64 feature set itself. Padding keeps the
        // router input shape stable without encoding original column position.
        for index in 0..MAX_PAIRWISE_FEATURES {
            if let Some(feature) = selected.get(index) {
                values.extend_from_slice(&feature.values);
            } else {
                values.extend([0.0; 12]);
            }
        }
        let mut pairwise = Vec::new();
        for left in 0..selected.len() {
            for right in left + 1..selected.len() {
                pairwise.push(
                    correlation(
                        selected[left]
                            .observed
                            .iter()
                            .copied()
                            .zip(selected[right].observed.iter().copied())
                            .filter(|(x, y)| x.is_finite() && y.is_finite())
                            .collect(),
                    )
                    .abs(),
                );
            }
        }
        values.extend(distribution(&mut pairwise));
        Self {
            version: DATASET_SKETCH_VERSION,
            values,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct QuantizedLayer {
    pub input: usize,
    pub output: usize,
    pub weights: Vec<i8>,
    pub biases: Vec<i32>,
    pub multiplier: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RouterBundle {
    pub format: String,
    pub version: u8,
    pub sketch_mean: Vec<f32>,
    pub sketch_std: Vec<f32>,
    pub input_scale: f32,
    pub candidate_ids: Vec<String>,
    pub candidate_embeddings: Vec<Vec<f32>>,
    pub layers: Vec<QuantizedLayer>,
    pub output_scale: Vec<f32>,
    pub output_bias: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ga2m_router: Option<Ga2mRouterBundle>,
    pub trained: bool,
    pub training_evidence_sha256: Option<String>,
}

impl RouterBundle {
    pub fn validate(&self) -> Result<()> {
        let encoded = serde_json::to_vec(self)?;
        let embedding = self.candidate_embeddings.first().map_or(0, Vec::len);
        let mut expected = self.sketch_mean.len() + embedding;
        let valid_layers = self.layers.len() == 3
            && self.layers.iter().enumerate().all(|(index, layer)| {
                let valid = layer.input == expected
                    && layer.output > 0
                    && layer.weights.len() == layer.input * layer.output
                    && layer.biases.len() == layer.output
                    && layer.multiplier.is_finite()
                    && layer.multiplier > 0.0;
                expected = layer.output;
                valid && (index == 2 || layer.output > 0)
            });
        let digest_valid = self.training_evidence_sha256.as_ref().is_none_or(|hash| {
            hash.len() == 64
                && hash
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        });
        if self.format != "dope-distilled-router"
            || self.version != 1
            || encoded.len() > MAX_ROUTER_BUNDLE_BYTES
            || self.sketch_mean.is_empty()
            || self.sketch_mean.len() != self.sketch_std.len()
            || !self.sketch_mean.iter().all(|v| v.is_finite())
            || !self.sketch_std.iter().all(|v| v.is_finite() && *v > 0.0)
            || !self.input_scale.is_finite()
            || self.input_scale <= 0.0
            || self.candidate_ids.is_empty()
            || self.candidate_ids.iter().any(String::is_empty)
            || self.candidate_ids.len() != self.candidate_embeddings.len()
            || self.candidate_ids.iter().collect::<BTreeSet<_>>().len() != self.candidate_ids.len()
            || !self
                .candidate_embeddings
                .iter()
                .all(|v| v.len() == embedding && v.iter().all(|value| value.is_finite()))
            || !valid_layers
            || expected != ROUTER_OUTPUTS
            || self.output_scale.len() != ROUTER_OUTPUTS
            || self.output_bias.len() != ROUTER_OUTPUTS
            || !self
                .output_scale
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            || !self.output_bias.iter().all(|value| value.is_finite())
            || self.ga2m_router.as_ref().is_some_and(|ga2m| {
                ga2m.candidate_ids != self.candidate_ids
                    || ga2m.validate_for_width(self.sketch_mean.len()).is_err()
            })
            || !digest_valid
            || self.trained != self.training_evidence_sha256.is_some()
        {
            return Err(DopeError::Data("invalid distilled router bundle".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RouterPrediction {
    pub candidate_id: String,
    pub retention_mean: f32,
    pub retention_lower: f32,
    pub kpi_failure_probability: f32,
    pub runtime_p50_ms: f32,
    pub runtime_p95_ms: f32,
    pub memory_p95_bytes: f32,
    pub artifact_bytes: f32,
    pub timeout_probability: f32,
    pub uncertainty: f32,
    pub expected_regret: f32,
}

impl RouterPrediction {
    pub fn action_values(&self) -> [f32; ROUTER_OUTPUTS] {
        [
            self.retention_mean,
            self.retention_lower,
            self.kpi_failure_probability,
            self.runtime_p50_ms,
            self.runtime_p95_ms,
            self.memory_p95_bytes,
            self.artifact_bytes,
            self.timeout_probability,
            self.uncertainty,
            self.expected_regret,
        ]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CandidateActionEmbedding {
    pub candidate_id: String,
    pub hidden_state: Vec<f32>,
    pub standardized_actions: Vec<f32>,
    pub prediction: RouterPrediction,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RouterActionEmbedding {
    pub standardized_sketch: Vec<f32>,
    pub candidates: Vec<CandidateActionEmbedding>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RouterLabel {
    pub lineage_group_id: String,
    pub structural_profile: String,
    pub candidate_id: String,
    pub sketch: DatasetSketch,
    pub retention: f32,
    pub runtime_ms: f32,
    pub peak_memory_bytes: f32,
    pub artifact_bytes: f32,
    pub failed: bool,
}

const GA2M_BINS: usize = 8;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Ga2mRouterBundle {
    pub format: String,
    pub version: u8,
    pub candidate_ids: Vec<String>,
    pub feature_indices: Vec<usize>,
    pub feature_mean: Vec<f32>,
    pub feature_std: Vec<f32>,
    pub pair_indices: Vec<(usize, usize)>,
    pub intercepts: Vec<f32>,
    pub univariate_effects: Vec<f32>,
    pub pair_effects: Vec<f32>,
}

impl Ga2mRouterBundle {
    fn bin(value: f32) -> usize {
        ((((value.clamp(-4.0, 4.0) + 4.0) / 8.0) * GA2M_BINS as f32).floor() as usize)
            .min(GA2M_BINS - 1)
    }

    fn validate_for_width(&self, sketch_width: usize) -> Result<()> {
        if self.format != "dope-ga2m-router"
            || self.version != 1
            || self.candidate_ids.is_empty()
            || self.candidate_ids.iter().any(String::is_empty)
            || self.candidate_ids.iter().collect::<BTreeSet<_>>().len() != self.candidate_ids.len()
            || self.feature_indices.len() != self.feature_mean.len()
            || self.feature_indices.len() != self.feature_std.len()
            || self.intercepts.len() != self.candidate_ids.len()
            || self.univariate_effects.len()
                != self.candidate_ids.len() * self.feature_indices.len() * GA2M_BINS
            || self.pair_effects.len()
                != self.candidate_ids.len() * self.pair_indices.len() * GA2M_BINS * GA2M_BINS
            || self
                .feature_indices
                .iter()
                .any(|&index| index >= sketch_width)
            || self.pair_indices.iter().any(|&(left, right)| {
                left >= self.feature_indices.len() || right >= self.feature_indices.len()
            })
            || !self.feature_mean.iter().all(|value| value.is_finite())
            || !self
                .feature_std
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            || !self.intercepts.iter().all(|value| value.is_finite())
            || !self
                .univariate_effects
                .iter()
                .all(|value| value.is_finite())
            || !self.pair_effects.iter().all(|value| value.is_finite())
        {
            return Err(DopeError::Data("invalid GA2M router bundle".into()));
        }
        Ok(())
    }

    pub fn predict(&self, sketch: &DatasetSketch) -> Result<Vec<(String, f32)>> {
        self.validate_for_width(sketch.values.len())?;
        let bins = self
            .feature_indices
            .iter()
            .zip(&self.feature_mean)
            .zip(&self.feature_std)
            .map(|((&index, &mean), &std)| Self::bin((sketch.values[index] - mean) / std))
            .collect::<Vec<_>>();
        Ok(self
            .candidate_ids
            .iter()
            .enumerate()
            .map(|(candidate, id)| {
                let mut prediction = self.intercepts[candidate];
                for (feature, &bin) in bins.iter().enumerate() {
                    let offset =
                        (candidate * self.feature_indices.len() + feature) * GA2M_BINS + bin;
                    prediction += self.univariate_effects[offset];
                }
                for (pair, &(left, right)) in self.pair_indices.iter().enumerate() {
                    let offset = ((candidate * self.pair_indices.len() + pair) * GA2M_BINS
                        + bins[left])
                        * GA2M_BINS
                        + bins[right];
                    prediction += self.pair_effects[offset];
                }
                (id.clone(), prediction)
            })
            .collect())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RouterTrainingReport {
    pub format: String,
    pub version: u8,
    #[serde(default)]
    pub input_cells: usize,
    #[serde(default)]
    pub replicates_per_candidate: usize,
    #[serde(default)]
    pub complete_lineage_profile_outcomes: usize,
    #[serde(default)]
    pub excluded_incomplete_outcomes: usize,
    pub training_lineages: usize,
    pub validation_lineages: usize,
    pub labels: usize,
    pub teacher_top_four_oracle_recall: f64,
    pub teacher_maximum_regret: f64,
    pub teacher_mean_regret: f64,
    pub student_top_four_oracle_recall: f64,
    pub student_maximum_regret: f64,
    pub student_maximum_profile_regret_upper: f64,
    pub student_mean_regret: f64,
    pub random_mean_regret: f64,
    #[serde(default)]
    pub best_fixed_candidate_id: String,
    pub best_fixed_mean_regret: f64,
    pub ga2m_mean_regret: f64,
    pub ga2m_top_four_oracle_recall: f64,
    pub ga2m_maximum_profile_regret_upper: f64,
    pub paired_hypervolume_improvement: f64,
    pub paired_hypervolume_ci_lower: f64,
    pub student_max_abs_difference: f64,
    pub student_max_abs_difference_by_output: Vec<f64>,
    pub top_choice_agreement: f64,
    pub bundle_bytes: u64,
    pub teacher_training_seconds: f64,
    pub ga2m_router: Ga2mRouterBundle,
}

#[cfg(feature = "gpu-training")]
fn train_ga2m_router(
    labels: &[RouterLabel],
    candidate_ids: &[String],
    candidate_index: &BTreeMap<&str, usize>,
    train_indices: &[usize],
    sketch_mean: &[f32],
    sketch_std: &[f32],
) -> Ga2mRouterBundle {
    let width = sketch_mean.len();
    let candidates = candidate_ids.len();
    let mut candidate_sums = vec![0.0f64; candidates];
    let mut candidate_counts = vec![0usize; candidates];
    for &index in train_indices {
        let candidate = candidate_index[labels[index].candidate_id.as_str()];
        candidate_sums[candidate] += f64::from(labels[index].retention.clamp(-2.0, 2.0));
        candidate_counts[candidate] += 1;
    }
    let intercepts = candidate_sums
        .iter()
        .zip(&candidate_counts)
        .map(|(&sum, &count)| (sum / count.max(1) as f64) as f32)
        .collect::<Vec<_>>();
    let mut feature_scores = vec![0.0f64; width];
    for &index in train_indices {
        let label = &labels[index];
        let candidate = candidate_index[label.candidate_id.as_str()];
        let residual = f64::from(label.retention.clamp(-2.0, 2.0) - intercepts[candidate]);
        for (feature, score) in feature_scores.iter_mut().enumerate() {
            let normalized =
                (label.sketch.values[feature] - sketch_mean[feature]) / sketch_std[feature];
            *score += f64::from(normalized.clamp(-4.0, 4.0)) * residual;
        }
    }
    let mut feature_indices = (0..width).collect::<Vec<_>>();
    feature_indices.sort_by(|&left, &right| {
        feature_scores[right]
            .abs()
            .total_cmp(&feature_scores[left].abs())
            .then_with(|| left.cmp(&right))
    });
    feature_indices.truncate(width.min(32));
    let feature_mean = feature_indices
        .iter()
        .map(|&index| sketch_mean[index])
        .collect::<Vec<_>>();
    let feature_std = feature_indices
        .iter()
        .map(|&index| sketch_std[index])
        .collect::<Vec<_>>();
    let pair_width = feature_indices.len().min(8);
    let pair_indices = (0..pair_width)
        .flat_map(|left| (left + 1..pair_width).map(move |right| (left, right)))
        .collect::<Vec<_>>();
    let bins_for = |label: &RouterLabel| {
        feature_indices
            .iter()
            .zip(&feature_mean)
            .zip(&feature_std)
            .map(|((&index, &mean), &std)| {
                Ga2mRouterBundle::bin((label.sketch.values[index] - mean) / std)
            })
            .collect::<Vec<_>>()
    };
    let training_bins = train_indices
        .iter()
        .map(|&index| bins_for(&labels[index]))
        .collect::<Vec<_>>();
    let mut predictions = train_indices
        .iter()
        .map(|&index| intercepts[candidate_index[labels[index].candidate_id.as_str()]])
        .collect::<Vec<_>>();
    let mut univariate_effects = vec![0.0f32; candidates * feature_indices.len() * GA2M_BINS];
    for feature in 0..feature_indices.len() {
        let mut sums = vec![0.0f64; candidates * GA2M_BINS];
        let mut counts = vec![0usize; candidates * GA2M_BINS];
        for (row, &index) in train_indices.iter().enumerate() {
            let candidate = candidate_index[labels[index].candidate_id.as_str()];
            let slot = candidate * GA2M_BINS + training_bins[row][feature];
            sums[slot] += f64::from(labels[index].retention.clamp(-2.0, 2.0) - predictions[row]);
            counts[slot] += 1;
        }
        for candidate in 0..candidates {
            for bin in 0..GA2M_BINS {
                let slot = candidate * GA2M_BINS + bin;
                let effect = (0.35 * sums[slot] / (counts[slot] + 8) as f64) as f32;
                univariate_effects
                    [(candidate * feature_indices.len() + feature) * GA2M_BINS + bin] = effect;
            }
        }
        for (row, &index) in train_indices.iter().enumerate() {
            let candidate = candidate_index[labels[index].candidate_id.as_str()];
            predictions[row] += univariate_effects[(candidate * feature_indices.len() + feature)
                * GA2M_BINS
                + training_bins[row][feature]];
        }
    }
    let mut pair_effects = vec![0.0f32; candidates * pair_indices.len() * GA2M_BINS * GA2M_BINS];
    for (pair, &(left, right)) in pair_indices.iter().enumerate() {
        let mut sums = vec![0.0f64; candidates * GA2M_BINS * GA2M_BINS];
        let mut counts = vec![0usize; candidates * GA2M_BINS * GA2M_BINS];
        for (row, &index) in train_indices.iter().enumerate() {
            let candidate = candidate_index[labels[index].candidate_id.as_str()];
            let slot = (candidate * GA2M_BINS + training_bins[row][left]) * GA2M_BINS
                + training_bins[row][right];
            sums[slot] += f64::from(labels[index].retention.clamp(-2.0, 2.0) - predictions[row]);
            counts[slot] += 1;
        }
        for candidate in 0..candidates {
            for left_bin in 0..GA2M_BINS {
                for right_bin in 0..GA2M_BINS {
                    let slot = (candidate * GA2M_BINS + left_bin) * GA2M_BINS + right_bin;
                    let effect = (0.2 * sums[slot] / (counts[slot] + 12) as f64) as f32;
                    pair_effects[((candidate * pair_indices.len() + pair) * GA2M_BINS
                        + left_bin)
                        * GA2M_BINS
                        + right_bin] = effect;
                }
            }
        }
        for (row, &index) in train_indices.iter().enumerate() {
            let candidate = candidate_index[labels[index].candidate_id.as_str()];
            predictions[row] += pair_effects[((candidate * pair_indices.len() + pair) * GA2M_BINS
                + training_bins[row][left])
                * GA2M_BINS
                + training_bins[row][right]];
        }
    }
    Ga2mRouterBundle {
        format: "dope-ga2m-router".into(),
        version: 1,
        candidate_ids: candidate_ids.to_vec(),
        feature_indices,
        feature_mean,
        feature_std,
        pair_indices,
        intercepts,
        univariate_effects,
        pair_effects,
    }
}

#[cfg(feature = "gpu-training")]
fn tensor_vector(tensor: tch::Tensor, context: &str) -> Result<Vec<f32>> {
    Vec::<f32>::try_from(tensor.to_device(tch::Device::Cpu).view([-1]))
        .map_err(|error| DopeError::Data(format!("{context}: {error}")))
}

#[cfg(feature = "gpu-training")]
fn quantized_layer(
    input: usize,
    output: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
    input_scale: f32,
    output_scale: f32,
) -> QuantizedLayer {
    let weight_scale = weights
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max)
        .max(1e-8)
        / 127.0;
    let weights = weights
        .into_iter()
        .map(|value| (value / weight_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    let bias_scale = input_scale * weight_scale;
    let biases = biases
        .into_iter()
        .map(|value| {
            (value / bias_scale)
                .round()
                .clamp(i32::MIN as f32, i32::MAX as f32) as i32
        })
        .collect();
    QuantizedLayer {
        input,
        output,
        weights,
        biases,
        multiplier: bias_scale / output_scale.max(1e-8),
    }
}

/// Evaluates the frozen int8 student with libtorch operations. This is the
/// parity reference for the Rust integer runtime; it is intentionally not the
/// floating-point teacher used to train the student.
#[cfg(feature = "gpu-training")]
fn libtorch_student_outputs(
    bundle: &RouterBundle,
    labels: &[RouterLabel],
) -> Result<Vec<Vec<f32>>> {
    let candidate_index = bundle
        .candidate_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut ga2m_prediction_cache = BTreeMap::<(String, String), BTreeMap<String, f32>>::new();
    let mut ga2m_scores = Vec::with_capacity(labels.len());
    for label in labels {
        let key = (
            label.lineage_group_id.clone(),
            label.structural_profile.clone(),
        );
        if !ga2m_prediction_cache.contains_key(&key) {
            let predictions = bundle
                .ga2m_router
                .as_ref()
                .map(|ga2m| ga2m.predict(&label.sketch))
                .transpose()?
                .unwrap_or_default()
                .into_iter()
                .collect();
            ga2m_prediction_cache.insert(key.clone(), predictions);
        }
        ga2m_scores.push(
            ga2m_prediction_cache[&key]
                .get(&label.candidate_id)
                .copied()
                .unwrap_or(0.0),
        );
    }
    let mut quantized_input = Vec::<f64>::new();
    for label in labels {
        let embedding = &bundle.candidate_embeddings[candidate_index[label.candidate_id.as_str()]];
        quantized_input.extend(
            label
                .sketch
                .values
                .iter()
                .zip(&bundle.sketch_mean)
                .zip(&bundle.sketch_std)
                .map(|((&value, &mean), &std)| ((value - mean) / std).clamp(-8.0, 8.0))
                .chain(embedding.iter().copied())
                .map(|value| f64::from((value / bundle.input_scale).round().clamp(-127.0, 127.0))),
        );
    }
    let mut activation = tch::Tensor::from_slice(&quantized_input)
        .view([labels.len() as i64, bundle.layers[0].input as i64]);
    for (layer_index, layer) in bundle.layers.iter().enumerate() {
        let weights = tch::Tensor::from_slice(
            &layer
                .weights
                .iter()
                .map(|&value| f64::from(value))
                .collect::<Vec<_>>(),
        )
        .view([layer.output as i64, layer.input as i64]);
        let biases = tch::Tensor::from_slice(
            &layer
                .biases
                .iter()
                .map(|&value| f64::from(value))
                .collect::<Vec<_>>(),
        );
        activation =
            (activation.matmul(&weights.transpose(0, 1)) + biases) * f64::from(layer.multiplier);
        // Rust's f64::round uses ties away from zero; libtorch's round uses
        // ties-to-even. Spell out the frozen Rust rule for bitwise parity.
        activation = activation.sign() * (activation.abs() + 0.5).floor();
        activation = if layer_index + 1 == bundle.layers.len() {
            activation.clamp(-127.0, 127.0)
        } else {
            activation.clamp(0.0, 127.0)
        };
    }
    let quantized = Vec::<f64>::try_from(activation.view([-1]))
        .map_err(|error| DopeError::Data(format!("libtorch student output: {error}")))?;
    Ok(quantized
        .chunks_exact(ROUTER_OUTPUTS)
        .enumerate()
        .map(|(row_index, row)| {
            let mut output = row
                .iter()
                .enumerate()
                .map(|(index, &value)| {
                    value as f32 * bundle.output_scale[index] + bundle.output_bias[index]
                })
                .collect::<Vec<_>>();
            output[0] += ga2m_scores[row_index];
            output[1] += ga2m_scores[row_index];
            output
        })
        .collect())
}

#[cfg(feature = "gpu-training")]
pub fn train_distilled_router(
    labels: &[RouterLabel],
    candidate_ids: &[String],
    training_evidence_sha256: &str,
    validation_lineage_ids: Option<&BTreeSet<String>>,
) -> Result<(RouterBundle, RouterTrainingReport)> {
    use tch::nn::{Module, OptimizerConfig};

    if !tch::Cuda::is_available() {
        return Err(DopeError::Unsupported(
            "router teacher training requires CUDA libtorch".into(),
        ));
    }
    if std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref() != Ok(":4096:8") {
        return Err(DopeError::Unsupported(
            "router teacher requires CUBLAS_WORKSPACE_CONFIG=:4096:8".into(),
        ));
    }
    if labels.is_empty() || candidate_ids.is_empty() {
        return Err(DopeError::Data("router training labels are empty".into()));
    }
    let candidate_index = candidate_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let sketch_width = labels[0].sketch.values.len();
    if labels.iter().any(|label| {
        label.sketch.version != DATASET_SKETCH_VERSION
            || label.sketch.values.len() != sketch_width
            || !candidate_index.contains_key(label.candidate_id.as_str())
    }) {
        return Err(DopeError::Data(
            "router labels have inconsistent sketches or candidates".into(),
        ));
    }
    let mut by_outcome = BTreeMap::<(&str, &str), Vec<usize>>::new();
    for (index, label) in labels.iter().enumerate() {
        by_outcome
            .entry((
                label.lineage_group_id.as_str(),
                label.structural_profile.as_str(),
            ))
            .or_default()
            .push(index);
    }
    if by_outcome.values().any(|indices| {
        indices.len() != candidate_ids.len()
            || indices
                .iter()
                .map(|&index| labels[index].candidate_id.as_str())
                .collect::<BTreeSet<_>>()
                .len()
                != candidate_ids.len()
    }) {
        return Err(DopeError::Data(
            "every router lineage/profile outcome must have every frozen candidate label".into(),
        ));
    }
    let all_lineages = labels
        .iter()
        .map(|label| label.lineage_group_id.as_str())
        .collect::<BTreeSet<_>>();
    let validation_lineages = if let Some(explicit) = validation_lineage_ids {
        let selected = all_lineages
            .iter()
            .filter(|lineage| explicit.contains(**lineage))
            .copied()
            .collect::<BTreeSet<_>>();
        if selected.len() != explicit.len()
            || selected.is_empty()
            || selected.len() == all_lineages.len()
        {
            return Err(DopeError::Data(
                "explicit router validation lineages are absent, empty, or consume training".into(),
            ));
        }
        selected
    } else {
        let selected = all_lineages
            .iter()
            .filter(|lineage| blake3::hash(lineage.as_bytes()).as_bytes()[0] % 5 == 0)
            .copied()
            .collect::<BTreeSet<_>>();
        if selected.is_empty() {
            BTreeSet::from([*all_lineages.iter().next().expect("nonempty lineages")])
        } else {
            selected
        }
    };
    let train_indices = labels
        .iter()
        .enumerate()
        .filter(|(_, label)| !validation_lineages.contains(label.lineage_group_id.as_str()))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if train_indices.is_empty() {
        return Err(DopeError::Data(
            "router training split contains no training lineages".into(),
        ));
    }
    let mut sketch_mean = vec![0.0f32; sketch_width];
    for &index in &train_indices {
        for (slot, value) in sketch_mean.iter_mut().zip(&labels[index].sketch.values) {
            *slot += *value / train_indices.len() as f32;
        }
    }
    let mut sketch_std = vec![0.0f32; sketch_width];
    for &index in &train_indices {
        for ((slot, value), mean) in sketch_std
            .iter_mut()
            .zip(&labels[index].sketch.values)
            .zip(&sketch_mean)
        {
            *slot += (*value - *mean).powi(2) / train_indices.len() as f32;
        }
    }
    sketch_std
        .iter_mut()
        .for_each(|value| *value = value.sqrt().max(1e-5));
    let ga2m_router = train_ga2m_router(
        labels,
        candidate_ids,
        &candidate_index,
        &train_indices,
        &sketch_mean,
        &sketch_std,
    );
    let mut ga2m_scores = vec![0.0f32; labels.len()];
    for indices in by_outcome.values() {
        let predictions = ga2m_router
            .predict(&labels[indices[0]].sketch)?
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        for &index in indices {
            ga2m_scores[index] = predictions[labels[index].candidate_id.as_str()];
        }
    }

    let oracle = by_outcome
        .iter()
        .map(|(&outcome, indices)| {
            let best = indices
                .iter()
                .map(|&index| labels[index].retention)
                .reduce(f32::max)
                .unwrap_or(0.0);
            (outcome, best)
        })
        .collect::<BTreeMap<_, _>>();
    let targets = labels
        .iter()
        .zip(&ga2m_scores)
        .map(|(label, ga2m_score)| {
            // Retention is deliberately unclamped in KPI aggregation. The
            // teacher uses a robust bounded target so a single near-zero TRTR
            // denominator cannot dominate every candidate ranking update.
            let retention = label.retention.clamp(-2.0, 4.0);
            let retention_residual = retention - ga2m_score;
            let regret = (oracle[&(
                label.lineage_group_id.as_str(),
                label.structural_profile.as_str(),
            )] - label.retention)
                .max(0.0)
                .clamp(0.0, 4.0);
            [
                retention_residual,
                retention_residual,
                f32::from(label.failed || label.retention < 0.0),
                label.runtime_ms,
                label.runtime_ms,
                label.peak_memory_bytes,
                label.artifact_bytes,
                f32::from(label.failed),
                regret,
                regret,
            ]
        })
        .collect::<Vec<_>>();
    let mut target_mean = [0.0f32; ROUTER_OUTPUTS];
    for &index in &train_indices {
        for (slot, value) in target_mean.iter_mut().zip(targets[index]) {
            *slot += value / train_indices.len() as f32;
        }
    }
    let mut target_std = [0.0f32; ROUTER_OUTPUTS];
    for &index in &train_indices {
        for ((slot, value), mean) in target_std.iter_mut().zip(targets[index]).zip(target_mean) {
            *slot += (value - mean).powi(2) / train_indices.len() as f32;
        }
    }
    target_std
        .iter_mut()
        .for_each(|value| *value = value.sqrt().max(1e-5));
    let flat_sketches = labels
        .iter()
        .flat_map(|label| {
            label
                .sketch
                .values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    ((*value - sketch_mean[index]) / sketch_std[index]).clamp(-8.0, 8.0)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let flat_targets = targets
        .iter()
        .flat_map(|target| {
            (0..ROUTER_OUTPUTS)
                .map(|index| (target[index] - target_mean[index]) / target_std[index])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let candidate_indices = labels
        .iter()
        .map(|label| candidate_index[label.candidate_id.as_str()] as i64)
        .collect::<Vec<_>>();
    crate::libtorch::with_seeded_libtorch(1729, || {
        let device = tch::Device::Cuda(0);
        let sketches = tch::Tensor::from_slice(&flat_sketches)
            .view([labels.len() as i64, sketch_width as i64])
            .to_device(device);
        let target_tensor = tch::Tensor::from_slice(&flat_targets)
            .view([labels.len() as i64, ROUTER_OUTPUTS as i64])
            .to_device(device);
        let actual_retention = tch::Tensor::from_slice(
            &labels
                .iter()
                .map(|label| label.retention)
                .collect::<Vec<_>>(),
        )
        .view([labels.len() as i64, 1])
        .to_device(device);
        let ga2m_tensor = tch::Tensor::from_slice(&ga2m_scores)
            .view([labels.len() as i64, 1])
            .to_device(device);
        let indices = tch::Tensor::from_slice(&candidate_indices).to_device(device);
        let train_tensor = tch::Tensor::from_slice(
            &train_indices
                .iter()
                .map(|&index| index as i64)
                .collect::<Vec<_>>(),
        )
        .to_device(device);
        let store = tch::nn::VarStore::new(device);
        let root = store.root();
        let embedding_width = 32usize;
        let teacher_embeddings = root.var(
            "teacher_candidate_embeddings",
            &[candidate_ids.len() as i64, embedding_width as i64],
            tch::nn::Init::Randn {
                mean: 0.0,
                stdev: 0.05,
            },
        );
        let token_projection = tch::nn::linear(&root / "teacher_token", 12, 32, Default::default());
        let query_projection = tch::nn::linear(
            &root / "teacher_query",
            embedding_width as i64,
            32,
            Default::default(),
        );
        let teacher_layer1 = tch::nn::linear(
            &root / "teacher_layer1",
            (88 + 32 + embedding_width) as i64,
            256,
            Default::default(),
        );
        let teacher_layer2 =
            tch::nn::linear(&root / "teacher_layer2", 256, 128, Default::default());
        let teacher_layer3 = tch::nn::linear(
            &root / "teacher_layer3",
            128,
            ROUTER_OUTPUTS as i64,
            Default::default(),
        );
        let teacher_forward = |sketch: &tch::Tensor, candidate: &tch::Tensor| {
            let embedding = teacher_embeddings.index_select(0, candidate);
            let tokens = sketch.narrow(1, 82, 64 * 12).view([-1, 64, 12]);
            let encoded_tokens = token_projection.forward(&tokens).tanh();
            let query = query_projection.forward(&embedding).unsqueeze(2);
            let attention = (encoded_tokens.matmul(&query).squeeze_dim(-1) / 32.0f64.sqrt())
                .softmax(-1, tch::Kind::Float);
            let pooled = (encoded_tokens * attention.unsqueeze(2)).sum_dim_intlist(
                &[1i64][..],
                false,
                tch::Kind::Float,
            );
            let global = tch::Tensor::cat(&[sketch.narrow(1, 0, 82), sketch.narrow(1, 850, 6)], 1);
            let input = tch::Tensor::cat(&[global, pooled, embedding], 1);
            teacher_layer3.forward(
                &teacher_layer2
                    .forward(&teacher_layer1.forward(&input).gelu("none"))
                    .gelu("none"),
            )
        };

        let embeddings = root.var(
            "student_candidate_embeddings",
            &[candidate_ids.len() as i64, embedding_width as i64],
            tch::nn::Init::Randn {
                mean: 0.0,
                stdev: 0.05,
            },
        );
        let input_width = sketch_width + embedding_width;
        let layer1 = tch::nn::linear(
            &root / "layer1",
            input_width as i64,
            256,
            Default::default(),
        );
        let layer2 = tch::nn::linear(&root / "layer2", 256, 128, Default::default());
        let layer3 = tch::nn::linear(
            &root / "layer3",
            128,
            ROUTER_OUTPUTS as i64,
            Default::default(),
        );
        let forward = |sketch: &tch::Tensor, candidate: &tch::Tensor| {
            let embedding = embeddings.index_select(0, candidate);
            let input = tch::Tensor::cat(&[sketch, &embedding], 1);
            let hidden1 = layer1.forward(&input).relu();
            let hidden2 = layer2.forward(&hidden1).relu();
            (
                input,
                hidden1,
                hidden2.shallow_clone(),
                layer3.forward(&hidden2),
            )
        };
        let mut teacher_optimizer = tch::nn::AdamW::default()
            .build(&store, 1e-3)
            .map_err(|error| DopeError::Data(format!("router teacher optimizer: {error}")))?;
        let started = Instant::now();
        let training_sketches = sketches.index_select(0, &train_tensor);
        let training_candidates = indices.index_select(0, &train_tensor);
        let training_targets = target_tensor.index_select(0, &train_tensor);
        let training_actual_retention = actual_retention.index_select(0, &train_tensor);
        let training_ga2m = ga2m_tensor.index_select(0, &train_tensor);
        for _ in 0..1200 {
            let prediction = teacher_forward(&training_sketches, &training_candidates);
            let predicted_rank = (prediction.narrow(1, 1, 1) * f64::from(target_std[1])
                + f64::from(target_mean[1])
                + &training_ga2m)
                .view([-1, candidate_ids.len() as i64]);
            let target_rank = training_actual_retention.view([-1, candidate_ids.len() as i64]);
            let target_distribution = (target_rank * 8.0).softmax(-1, tch::Kind::Float);
            let rank_loss = -(target_distribution
                * (predicted_rank * 8.0).log_softmax(-1, tch::Kind::Float))
            .sum_dim_intlist(&[1i64][..], false, tch::Kind::Float)
            .mean(tch::Kind::Float);
            let loss = prediction.smooth_l1_loss(&training_targets, tch::Reduction::Mean, 0.5)
                + rank_loss * 2.0;
            teacher_optimizer.backward_step_clip(&loss, 5.0);
        }
        let teacher_normalized_output = teacher_forward(&sketches, &indices);
        let teacher_training_output =
            teacher_forward(&training_sketches, &training_candidates).detach();
        let mut student_optimizer = tch::nn::AdamW::default()
            .build(&store, 1e-3)
            .map_err(|error| DopeError::Data(format!("router student optimizer: {error}")))?;
        for _ in 0..1200 {
            let (_, _, _, prediction) = forward(&training_sketches, &training_candidates);
            let distilled_target = &training_targets * 0.8 + &teacher_training_output * 0.2;
            let predicted_rank = (prediction.narrow(1, 1, 1) * f64::from(target_std[1])
                + f64::from(target_mean[1])
                + &training_ga2m)
                .view([-1, candidate_ids.len() as i64]);
            let target_rank = training_actual_retention.view([-1, candidate_ids.len() as i64]);
            let target_distribution = (target_rank * 8.0).softmax(-1, tch::Kind::Float);
            let rank_loss = -(target_distribution
                * (predicted_rank * 8.0).log_softmax(-1, tch::Kind::Float))
            .sum_dim_intlist(&[1i64][..], false, tch::Kind::Float)
            .mean(tch::Kind::Float);
            let loss = prediction.smooth_l1_loss(&distilled_target, tch::Reduction::Mean, 0.5)
                + rank_loss * 2.0;
            student_optimizer.backward_step_clip(&loss, 5.0);
        }
        let (float_input, hidden1, hidden2, normalized_output) = forward(&sketches, &indices);
        let teacher_float_outputs =
            tensor_vector(teacher_normalized_output, "router teacher output")?;
        let float_outputs =
            tensor_vector(normalized_output.shallow_clone(), "router student output")?;
        let candidate_embeddings =
            tensor_vector(embeddings.shallow_clone(), "candidate embeddings")?
                .chunks_exact(embedding_width)
                .map(<[f32]>::to_vec)
                .collect::<Vec<_>>();
        let input_values = tensor_vector(float_input, "router input activations")?;
        let hidden1_values = tensor_vector(hidden1, "router hidden1 activations")?;
        let hidden2_values = tensor_vector(hidden2, "router hidden2 activations")?;
        let input_scale = input_values
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max)
            .max(1e-8)
            / 127.0;
        let hidden1_scale = hidden1_values
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
            .max(1e-8)
            / 127.0;
        let hidden2_scale = hidden2_values
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
            .max(1e-8)
            / 127.0;
        let final_scale = float_outputs
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max)
            .max(1e-8)
            / 127.0;
        let layers = vec![
            quantized_layer(
                input_width,
                256,
                tensor_vector(layer1.ws.shallow_clone(), "router layer1 weights")?,
                tensor_vector(
                    layer1.bs.as_ref().expect("linear bias").shallow_clone(),
                    "router layer1 biases",
                )?,
                input_scale,
                hidden1_scale,
            ),
            quantized_layer(
                256,
                128,
                tensor_vector(layer2.ws.shallow_clone(), "router layer2 weights")?,
                tensor_vector(
                    layer2.bs.as_ref().expect("linear bias").shallow_clone(),
                    "router layer2 biases",
                )?,
                hidden1_scale,
                hidden2_scale,
            ),
            quantized_layer(
                128,
                ROUTER_OUTPUTS,
                tensor_vector(layer3.ws.shallow_clone(), "router layer3 weights")?,
                tensor_vector(
                    layer3.bs.as_ref().expect("linear bias").shallow_clone(),
                    "router layer3 biases",
                )?,
                hidden2_scale,
                final_scale,
            ),
        ];
        let bundle = RouterBundle {
            format: "dope-distilled-router".into(),
            version: 1,
            sketch_mean,
            sketch_std,
            input_scale,
            candidate_ids: candidate_ids.to_vec(),
            candidate_embeddings,
            layers,
            output_scale: target_std
                .iter()
                .map(|value| final_scale * *value)
                .collect(),
            output_bias: target_mean.to_vec(),
            ga2m_router: Some(ga2m_router.clone()),
            trained: true,
            training_evidence_sha256: Some(training_evidence_sha256.into()),
        };
        bundle.validate()?;
        let router = QuantizedRouter::new(bundle.clone())?;
        let libtorch_student = libtorch_student_outputs(&bundle, labels)?;
        let teacher = labels
            .iter()
            .enumerate()
            .map(|(row, label)| {
                let start = row * ROUTER_OUTPUTS;
                let mut output = (0..ROUTER_OUTPUTS)
                    .map(|index| {
                        teacher_float_outputs[start + index] * target_std[index]
                            + target_mean[index]
                    })
                    .collect::<Vec<_>>();
                output[0] += ga2m_scores[row];
                output[1] += ga2m_scores[row];
                (
                    label.lineage_group_id.as_str(),
                    label.candidate_id.as_str(),
                    output,
                )
            })
            .collect::<Vec<_>>();
        let mut student_max_abs_difference = 0.0f64;
        let mut student_max_abs_difference_by_output = vec![0.0f64; ROUTER_OUTPUTS];
        let mut top_choice_agreement_count = 0usize;
        let mut top_four_recall_count = 0usize;
        let mut student_top_four_recall_count = 0usize;
        let mut regrets = Vec::new();
        let mut student_regrets = Vec::new();
        let mut profile_regrets = BTreeMap::<&str, Vec<f64>>::new();
        let mut random_regrets = Vec::new();
        let mut ga2m_regrets = Vec::new();
        let mut ga2m_top_four_recall_count = 0usize;
        let mut ga2m_profile_regrets = BTreeMap::<&str, Vec<f64>>::new();
        let mut fixed_sums = BTreeMap::<&str, (f64, usize)>::new();
        for label in labels
            .iter()
            .filter(|label| validation_lineages.contains(label.lineage_group_id.as_str()))
        {
            let entry = fixed_sums.entry(label.candidate_id.as_str()).or_default();
            entry.0 += f64::from(label.retention);
            entry.1 += 1;
        }
        let best_fixed = fixed_sums
            .iter()
            .max_by(|left, right| {
                (left.1.0 / left.1.1.max(1) as f64)
                    .total_cmp(&(right.1.0 / right.1.1.max(1) as f64))
            })
            .map(|(&id, _)| id)
            .unwrap_or(candidate_ids[0].as_str());
        let mut fixed_regrets = Vec::new();
        let mut hypervolume_pairs = Vec::new();
        let validation_outcomes = by_outcome
            .iter()
            .filter(|((lineage, _), _)| validation_lineages.contains(lineage))
            .collect::<Vec<_>>();
        for entry in &validation_outcomes {
            let lineage = entry.0.0;
            let profile = entry.0.1;
            let indices_for_outcome = entry.1;
            let mut teacher_rank = indices_for_outcome
                .iter()
                .map(|&index| {
                    let row = &teacher[index];
                    (row.1, row.2[1], index)
                })
                .collect::<Vec<_>>();
            teacher_rank.sort_by(|left, right| {
                right.1.total_cmp(&left.1).then_with(|| left.0.cmp(right.0))
            });
            let actual_best = indices_for_outcome
                .iter()
                .copied()
                .max_by(|&left, &right| labels[left].retention.total_cmp(&labels[right].retention))
                .expect("complete lineage");
            let top_four_best = teacher_rank
                .iter()
                .take(4)
                .map(|(_, _, index)| labels[*index].retention)
                .reduce(f32::max)
                .unwrap_or(f32::NEG_INFINITY);
            top_four_recall_count +=
                usize::from(labels[actual_best].retention - top_four_best <= 1e-6);
            let teacher_selected = teacher_rank[0].2;
            regrets.push(f64::from(
                labels[actual_best].retention - labels[teacher_selected].retention,
            ));
            let mut random_hasher = blake3::Hasher::new();
            random_hasher.update(lineage.as_bytes());
            random_hasher.update(profile.as_bytes());
            let random_index =
                usize::from(random_hasher.finalize().as_bytes()[1]) % indices_for_outcome.len();
            random_regrets.push(f64::from(
                labels[actual_best].retention - labels[indices_for_outcome[random_index]].retention,
            ));
            let ga2m = ga2m_router.predict(&labels[actual_best].sketch)?;
            let mut ga2m_rank = ga2m.iter().collect::<Vec<_>>();
            ga2m_rank.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let ga2m_top_four_best = ga2m_rank
                .iter()
                .take(4)
                .filter_map(|(id, _)| {
                    indices_for_outcome
                        .iter()
                        .copied()
                        .find(|&index| labels[index].candidate_id == *id)
                })
                .map(|index| labels[index].retention)
                .reduce(f32::max)
                .unwrap_or(f32::NEG_INFINITY);
            ga2m_top_four_recall_count +=
                usize::from(labels[actual_best].retention - ga2m_top_four_best <= 1e-6);
            let ga2m_candidate = &ga2m_rank[0].0;
            let ga2m_selected = indices_for_outcome
                .iter()
                .copied()
                .find(|&index| labels[index].candidate_id == *ga2m_candidate)
                .expect("complete GA2M lineage");
            let ga2m_regret =
                f64::from(labels[actual_best].retention - labels[ga2m_selected].retention);
            ga2m_regrets.push(ga2m_regret);
            ga2m_profile_regrets
                .entry(profile)
                .or_default()
                .push(ga2m_regret);
            let fixed = indices_for_outcome
                .iter()
                .copied()
                .find(|&index| labels[index].candidate_id == best_fixed)
                .expect("complete fixed candidate");
            fixed_regrets.push(f64::from(
                labels[actual_best].retention - labels[fixed].retention,
            ));
            let max_runtime = indices_for_outcome
                .iter()
                .map(|&index| labels[index].runtime_ms)
                .reduce(f32::max)
                .unwrap_or(0.0)
                + 1.0;
            let hypervolume = |index: usize| {
                f64::from(labels[index].retention.max(0.0))
                    * f64::from((max_runtime - labels[index].runtime_ms).max(0.0))
            };
            let student = router.predict(&labels[actual_best].sketch)?;
            let mut student_rank = student
                .iter()
                .map(|prediction| {
                    let index = indices_for_outcome
                        .iter()
                        .copied()
                        .find(|&index| labels[index].candidate_id == prediction.candidate_id)
                        .expect("complete student lineage/profile outcome");
                    (prediction, index)
                })
                .collect::<Vec<_>>();
            student_rank.sort_by(|left, right| {
                right
                    .0
                    .retention_lower
                    .total_cmp(&left.0.retention_lower)
                    .then_with(|| left.0.candidate_id.cmp(&right.0.candidate_id))
            });
            let student_top_four_best = student_rank
                .iter()
                .take(4)
                .map(|(_, index)| labels[*index].retention)
                .reduce(f32::max)
                .unwrap_or(f32::NEG_INFINITY);
            student_top_four_recall_count +=
                usize::from(labels[actual_best].retention - student_top_four_best <= 1e-6);
            let student_selected = student_rank[0].1;
            let student_regret =
                f64::from(labels[actual_best].retention - labels[student_selected].retention);
            student_regrets.push(student_regret);
            profile_regrets
                .entry(profile)
                .or_default()
                .push(student_regret);
            hypervolume_pairs.push((hypervolume(student_selected), hypervolume(fixed)));
            let student_best = student
                .iter()
                .max_by(|left, right| left.retention_lower.total_cmp(&right.retention_lower))
                .expect("router predictions");
            let reference_best = indices_for_outcome
                .iter()
                .copied()
                .max_by(|&left, &right| {
                    libtorch_student[left][1]
                        .total_cmp(&libtorch_student[right][1])
                        .then_with(|| {
                            candidate_index[labels[left].candidate_id.as_str()]
                                .cmp(&candidate_index[labels[right].candidate_id.as_str()])
                        })
                })
                .expect("complete libtorch student lineage");
            top_choice_agreement_count +=
                usize::from(student_best.candidate_id == labels[reference_best].candidate_id);
            for prediction in student {
                let reference_index = indices_for_outcome
                    .iter()
                    .copied()
                    .find(|&index| labels[index].candidate_id == prediction.candidate_id)
                    .expect("complete libtorch student prediction");
                for (output_index, (actual, distilled)) in libtorch_student[reference_index]
                    .iter()
                    .zip([
                        prediction.retention_mean,
                        prediction.retention_lower,
                        prediction.kpi_failure_probability,
                        prediction.runtime_p50_ms,
                        prediction.runtime_p95_ms,
                        prediction.memory_p95_bytes,
                        prediction.artifact_bytes,
                        prediction.timeout_probability,
                        prediction.uncertainty,
                        prediction.expected_regret,
                    ])
                    .enumerate()
                {
                    let actual = match output_index {
                        2 | 7 => actual.clamp(0.0, 1.0),
                        3..=6 | 8 | 9 => actual.max(0.0),
                        _ => *actual,
                    };
                    let difference = f64::from((actual - distilled).abs());
                    student_max_abs_difference = student_max_abs_difference.max(difference);
                    student_max_abs_difference_by_output[output_index] =
                        student_max_abs_difference_by_output[output_index].max(difference);
                }
            }
        }
        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len().max(1) as f64;
        let reference_hypervolume = mean(
            &hypervolume_pairs
                .iter()
                .map(|(_, fixed)| fixed.abs())
                .collect::<Vec<_>>(),
        )
        .max(1e-9);
        let improvements = hypervolume_pairs
            .iter()
            .map(|(routed, fixed)| (routed - fixed) / reference_hypervolume)
            .collect::<Vec<_>>();
        let improvement_mean = mean(&improvements);
        let improvement_std = if improvements.len() > 1 {
            (improvements
                .iter()
                .map(|value| (value - improvement_mean).powi(2))
                .sum::<f64>()
                / (improvements.len() - 1) as f64)
                .sqrt()
        } else {
            0.0
        };
        let validation_count = validation_lineages.len();
        let validation_outcome_count = validation_outcomes.len();
        let profile_regret_upper = profile_regrets
            .values()
            .map(|values| {
                let profile_mean = mean(values);
                let standard_deviation = if values.len() > 1 {
                    (values
                        .iter()
                        .map(|value| (value - profile_mean).powi(2))
                        .sum::<f64>()
                        / (values.len() - 1) as f64)
                        .sqrt()
                } else {
                    0.0
                };
                profile_mean + 1.645 * standard_deviation / (values.len().max(1) as f64).sqrt()
            })
            .reduce(f64::max)
            .unwrap_or(f64::INFINITY);
        let profile_upper = |groups: &BTreeMap<&str, Vec<f64>>| {
            groups
                .values()
                .map(|values| {
                    let group_mean = mean(values);
                    let deviation = if values.len() > 1 {
                        (values
                            .iter()
                            .map(|value| (value - group_mean).powi(2))
                            .sum::<f64>()
                            / (values.len() - 1) as f64)
                            .sqrt()
                    } else {
                        0.0
                    };
                    group_mean + 1.645 * deviation / (values.len().max(1) as f64).sqrt()
                })
                .reduce(f64::max)
                .unwrap_or(f64::INFINITY)
        };
        let report = RouterTrainingReport {
            format: "dope-router-training-report".into(),
            version: 1,
            input_cells: labels.len(),
            replicates_per_candidate: 1,
            complete_lineage_profile_outcomes: by_outcome.len(),
            excluded_incomplete_outcomes: 0,
            training_lineages: all_lineages.len() - validation_count,
            validation_lineages: validation_count,
            labels: labels.len(),
            teacher_top_four_oracle_recall: top_four_recall_count as f64
                / validation_outcome_count.max(1) as f64,
            teacher_maximum_regret: regrets.iter().copied().reduce(f64::max).unwrap_or(0.0),
            teacher_mean_regret: mean(&regrets),
            student_top_four_oracle_recall: student_top_four_recall_count as f64
                / validation_outcome_count.max(1) as f64,
            student_maximum_regret: student_regrets
                .iter()
                .copied()
                .reduce(f64::max)
                .unwrap_or(0.0),
            student_maximum_profile_regret_upper: profile_regret_upper,
            student_mean_regret: mean(&student_regrets),
            random_mean_regret: mean(&random_regrets),
            best_fixed_candidate_id: best_fixed.into(),
            best_fixed_mean_regret: mean(&fixed_regrets),
            ga2m_mean_regret: mean(&ga2m_regrets),
            ga2m_top_four_oracle_recall: ga2m_top_four_recall_count as f64
                / validation_outcome_count.max(1) as f64,
            ga2m_maximum_profile_regret_upper: profile_upper(&ga2m_profile_regrets),
            paired_hypervolume_improvement: improvement_mean,
            paired_hypervolume_ci_lower: improvement_mean
                - 1.96 * improvement_std / (improvements.len().max(1) as f64).sqrt(),
            student_max_abs_difference,
            student_max_abs_difference_by_output,
            top_choice_agreement: top_choice_agreement_count as f64
                / validation_outcome_count.max(1) as f64,
            bundle_bytes: serde_json::to_vec(&bundle)?.len() as u64,
            teacher_training_seconds: started.elapsed().as_secs_f64(),
            ga2m_router,
        };
        Ok((bundle, report))
    })
}

#[cfg(not(feature = "gpu-training"))]
pub fn train_distilled_router(
    _labels: &[RouterLabel],
    _candidate_ids: &[String],
    _training_evidence_sha256: &str,
    _validation_lineage_ids: Option<&BTreeSet<String>>,
) -> Result<(RouterBundle, RouterTrainingReport)> {
    Err(DopeError::Unsupported(
        "router teacher training requires the gpu-training build".into(),
    ))
}

pub struct QuantizedRouter {
    bundle: RouterBundle,
}

struct CandidateForwardPass {
    candidate_id: String,
    penultimate_activation: Vec<i8>,
    decoded_output: Vec<f32>,
}

struct RouterForwardPass {
    standardized_sketch: Vec<f32>,
    candidates: Vec<CandidateForwardPass>,
}

fn prediction_from_output(candidate_id: String, output: &[f32]) -> RouterPrediction {
    RouterPrediction {
        candidate_id,
        retention_mean: output[0],
        retention_lower: output[1],
        kpi_failure_probability: output[2].clamp(0.0, 1.0),
        runtime_p50_ms: output[3].max(0.0),
        runtime_p95_ms: output[4].max(0.0),
        memory_p95_bytes: output[5].max(0.0),
        artifact_bytes: output[6].max(0.0),
        timeout_probability: output[7].clamp(0.0, 1.0),
        uncertainty: output[8].max(0.0),
        expected_regret: output[9].max(0.0),
    }
}

impl QuantizedRouter {
    pub fn new(bundle: RouterBundle) -> Result<Self> {
        bundle.validate()?;
        if !bundle.trained {
            return Err(DopeError::Data(
                "untrained router bundle cannot drive candidate selection".into(),
            ));
        }
        Ok(Self { bundle })
    }

    fn forward(&self, sketch: &DatasetSketch) -> Result<RouterForwardPass> {
        if sketch.version != DATASET_SKETCH_VERSION
            || sketch.values.len() != self.bundle.sketch_mean.len()
        {
            return Err(DopeError::Data("router sketch shape mismatch".into()));
        }
        let first = &self.bundle.layers[0];
        let sketch_width = sketch.values.len();
        let standardized_sketch = sketch
            .values
            .iter()
            .zip(&self.bundle.sketch_mean)
            .zip(&self.bundle.sketch_std)
            .map(|((&value, &mean), &std)| ((value - mean) / std).clamp(-8.0, 8.0))
            .collect::<Vec<_>>();
        if !standardized_sketch.iter().all(|value| value.is_finite()) {
            return Err(DopeError::Data(
                "router sketch contains non-finite values".into(),
            ));
        }
        let quantized_sketch = standardized_sketch
            .iter()
            .map(|&value| {
                (value / self.bundle.input_scale)
                    .round()
                    .clamp(-127.0, 127.0) as i8
            })
            .collect::<Vec<_>>();
        let first_base = (0..first.output)
            .map(|output| {
                let mut sum = i64::from(first.biases[output]);
                for (input, &value) in quantized_sketch.iter().enumerate() {
                    sum +=
                        i64::from(value) * i64::from(first.weights[output * first.input + input]);
                }
                sum
            })
            .collect::<Vec<_>>();
        let ga2m_predictions = self
            .bundle
            .ga2m_router
            .as_ref()
            .map(|ga2m| ga2m.predict(sketch))
            .transpose()?
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        let candidates = self
            .bundle
            .candidate_ids
            .iter()
            .zip(&self.bundle.candidate_embeddings)
            .map(|(candidate_id, embedding)| {
                let quantized_embedding = embedding
                    .iter()
                    .map(|value| {
                        (value / self.bundle.input_scale)
                            .round()
                            .clamp(-127.0, 127.0) as i8
                    })
                    .collect::<Vec<_>>();
                let mut activation = (0..first.output)
                    .map(|output| {
                        let mut sum = first_base[output];
                        for (offset, &value) in quantized_embedding.iter().enumerate() {
                            let input = sketch_width + offset;
                            sum += i64::from(value)
                                * i64::from(first.weights[output * first.input + input]);
                        }
                        (sum as f64 * f64::from(first.multiplier))
                            .round()
                            .clamp(0.0, 127.0) as i8
                    })
                    .collect::<Vec<_>>();
                let mut penultimate_activation = Vec::new();
                for (layer_index, layer) in self.bundle.layers.iter().enumerate().skip(1) {
                    if layer_index + 1 == self.bundle.layers.len() {
                        penultimate_activation.clone_from(&activation);
                    }
                    let mut next = Vec::with_capacity(layer.output);
                    for output in 0..layer.output {
                        let mut sum = i64::from(layer.biases[output]);
                        for (input, &value) in activation.iter().enumerate().take(layer.input) {
                            sum += i64::from(value)
                                * i64::from(layer.weights[output * layer.input + input]);
                        }
                        let value = (sum as f64 * f64::from(layer.multiplier)).round();
                        let value = if layer_index + 1 == self.bundle.layers.len() {
                            value.clamp(-127.0, 127.0)
                        } else {
                            value.clamp(0.0, 127.0)
                        };
                        next.push(value as i8);
                    }
                    activation = next;
                }
                let mut output: Vec<f32> = activation
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        f32::from(v) * self.bundle.output_scale[i] + self.bundle.output_bias[i]
                    })
                    .collect();
                if let Some(score) = ga2m_predictions.get(candidate_id) {
                    output[0] += score;
                    output[1] += score;
                }
                CandidateForwardPass {
                    candidate_id: candidate_id.clone(),
                    penultimate_activation,
                    decoded_output: output,
                }
            })
            .collect();
        Ok(RouterForwardPass {
            standardized_sketch,
            candidates,
        })
    }

    pub fn predict(&self, sketch: &DatasetSketch) -> Result<Vec<RouterPrediction>> {
        Ok(self
            .forward(sketch)?
            .candidates
            .into_iter()
            .map(|candidate| {
                prediction_from_output(candidate.candidate_id, &candidate.decoded_output)
            })
            .collect())
    }

    pub fn action_embedding(&self, sketch: &DatasetSketch) -> Result<RouterActionEmbedding> {
        let forward = self.forward(sketch)?;
        let candidates = forward
            .candidates
            .into_iter()
            .map(|candidate| {
                let prediction = prediction_from_output(
                    candidate.candidate_id.clone(),
                    &candidate.decoded_output,
                );
                let action_values = prediction.action_values();
                if !action_values.iter().all(|value| value.is_finite()) {
                    return Err(DopeError::Data(
                        "router produced non-finite action predictions".into(),
                    ));
                }
                let standardized_actions = action_values
                    .into_iter()
                    .zip(&self.bundle.output_bias)
                    .zip(&self.bundle.output_scale)
                    .map(|((value, &bias), &scale)| (value - bias) / scale)
                    .collect::<Vec<_>>();
                if !standardized_actions.iter().all(|value| value.is_finite()) {
                    return Err(DopeError::Data(
                        "router produced non-finite standardized actions".into(),
                    ));
                }
                Ok(CandidateActionEmbedding {
                    candidate_id: candidate.candidate_id,
                    hidden_state: candidate
                        .penultimate_activation
                        .into_iter()
                        .map(|value| f32::from(value) / 127.0)
                        .collect(),
                    standardized_actions,
                    prediction,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RouterActionEmbedding {
            standardized_sketch: forward.standardized_sketch,
            candidates,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SelectionPolicy {
    pub time_budget_ms: Option<f32>,
}

impl SelectionPolicy {
    pub fn select(
        &self,
        predictions: &[RouterPrediction],
        cheapest_deterministic_baseline: &str,
    ) -> Vec<String> {
        let mut ranked = predictions.to_vec();
        ranked.sort_by(|a, b| {
            b.retention_lower
                .total_cmp(&a.retention_lower)
                .then_with(|| a.expected_regret.total_cmp(&b.expected_regret))
                .then_with(|| a.candidate_id.cmp(&b.candidate_id))
        });
        let ambiguous = ranked.first().is_some_and(|best| best.uncertainty > 0.02)
            || ranked
                .first()
                .zip(ranked.get(1))
                .is_some_and(|(a, b)| (a.retention_lower - b.retention_lower).abs() < 0.01);
        let highly_uncertain = ranked.first().is_some_and(|best| best.uncertainty > 0.05);
        let budget_permits_all = self
            .time_budget_ms
            .is_some_and(|budget| ranked.iter().map(|p| p.runtime_p95_ms).sum::<f32>() <= budget);
        let count = if highly_uncertain && budget_permits_all {
            ranked.len()
        } else if ambiguous {
            ranked.len().min(8)
        } else {
            ranked.len().min(4)
        };
        let mut selected: Vec<_> = ranked.into_iter().take(count).collect();
        if !selected
            .iter()
            .any(|p| p.candidate_id == cheapest_deterministic_baseline)
            && let Some(baseline) = predictions
                .iter()
                .find(|p| p.candidate_id == cheapest_deterministic_baseline)
        {
            selected.push(baseline.clone());
        }
        // Cost orders work only after evidence-preserving membership is fixed.
        selected.sort_by(|a, b| {
            a.runtime_p95_ms
                .total_cmp(&b.runtime_p95_ms)
                .then_with(|| a.candidate_id.cmp(&b.candidate_id))
        });
        selected.into_iter().map(|p| p.candidate_id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(columns: Vec<Vec<f32>>) -> Table {
        Table {
            rows: columns[0].len(),
            features: columns.len(),
            columns,
            target: vec![0.0, 1.0, 0.0, 1.0],
        }
    }

    #[test]
    fn sketch_is_row_and_column_permutation_invariant() {
        let original = table(vec![
            vec![0.1, 0.4, f32::NAN, 0.8],
            vec![0.9, 0.2, 0.3, 0.7],
        ]);
        let permuted = Table {
            rows: 4,
            features: 2,
            columns: vec![vec![0.3, 0.9, 0.7, 0.2], vec![f32::NAN, 0.1, 0.8, 0.4]],
            target: vec![0.0, 0.0, 1.0, 1.0],
        };
        assert_eq!(
            DatasetSketch::from_train(&original, Task::Binary),
            DatasetSketch::from_train(&permuted, Task::Binary)
        );
    }

    #[test]
    fn selection_expands_and_keeps_baseline() {
        let predictions: Vec<_> = (0..10)
            .map(|index| RouterPrediction {
                candidate_id: format!("c{index}"),
                retention_mean: 1.0,
                retention_lower: 1.0 - index as f32 * 0.02,
                kpi_failure_probability: 0.0,
                runtime_p50_ms: 1.0,
                runtime_p95_ms: (10 - index) as f32,
                memory_p95_bytes: 1.0,
                artifact_bytes: 1.0,
                timeout_probability: 0.0,
                uncertainty: if index == 0 { 0.03 } else { 0.0 },
                expected_regret: index as f32,
            })
            .collect();
        let selected = SelectionPolicy::default().select(&predictions, "c9");
        assert_eq!(selected.len(), 9);
        assert!(selected.iter().any(|id| id == "c9"));
    }

    #[test]
    fn int8_inference_is_deterministic_across_three_repeats() {
        let sketch =
            DatasetSketch::from_train(&table(vec![vec![0.1, 0.4, 0.2, 0.8]]), Task::Binary);
        let input = sketch.values.len() + 1;
        let bundle = RouterBundle {
            format: "dope-distilled-router".into(),
            version: 1,
            sketch_mean: vec![0.0; sketch.values.len()],
            sketch_std: vec![1.0; sketch.values.len()],
            input_scale: 0.01,
            candidate_ids: vec!["baseline".into()],
            candidate_embeddings: vec![vec![0.0]],
            layers: vec![
                QuantizedLayer {
                    input,
                    output: 2,
                    weights: vec![1; input * 2],
                    biases: vec![0; 2],
                    multiplier: 0.01,
                },
                QuantizedLayer {
                    input: 2,
                    output: 2,
                    weights: vec![1; 4],
                    biases: vec![0; 2],
                    multiplier: 0.5,
                },
                QuantizedLayer {
                    input: 2,
                    output: ROUTER_OUTPUTS,
                    weights: vec![1; 2 * ROUTER_OUTPUTS],
                    biases: vec![0; ROUTER_OUTPUTS],
                    multiplier: 0.5,
                },
            ],
            output_scale: vec![0.01; ROUTER_OUTPUTS],
            output_bias: vec![0.0; ROUTER_OUTPUTS],
            ga2m_router: None,
            trained: true,
            training_evidence_sha256: Some("a".repeat(64)),
        };
        let router = QuantizedRouter::new(bundle).unwrap();
        let first = router.predict(&sketch).unwrap();
        assert_eq!(first, router.predict(&sketch).unwrap());
        assert_eq!(first, router.predict(&sketch).unwrap());
        let embedding = router.action_embedding(&sketch).unwrap();
        assert_eq!(embedding.candidates[0].prediction, first[0]);
        assert_eq!(embedding.candidates[0].hidden_state.len(), 2);
        assert!(
            embedding.candidates[0]
                .hidden_state
                .iter()
                .all(|value| (-1.0..=1.0).contains(value))
        );
        let expected_actions = first[0]
            .action_values()
            .into_iter()
            .map(|value| value / 0.01)
            .collect::<Vec<_>>();
        assert_eq!(
            embedding.candidates[0].standardized_actions,
            expected_actions
        );
    }
}

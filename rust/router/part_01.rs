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
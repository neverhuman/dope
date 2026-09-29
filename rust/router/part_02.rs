

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

include!("student_training.rs");
include!("router_selection.rs");


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
    fn documented_sketch_decomposition_matches_emitted_width() {
        // Paper §2.2: target/shape, across-column distributions, padded
        // per-feature blocks, and global pairwise distribution.
        let target_and_shape = 10;
        let across_columns = 12 * 6;
        let per_feature = 64 * 12;
        let global_pairwise = 6;
        let documented_width = target_and_shape + across_columns + per_feature + global_pairwise;
        assert_eq!(documented_width, 856);
        for width in [1, 12, 64, 65] {
            let train = table(vec![vec![0.1, 0.4, 0.6, 0.8]; width]);
            let sketch = DatasetSketch::from_train(&train, Task::Binary);
            assert_eq!(sketch.values.len(), documented_width);
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
    fn sketch_cutoff_tie_changes_columns_but_preserves_row_permutations() {
        let mut columns = vec![vec![0.0, 0.0, 1.0, 1.0]; MAX_PAIRWISE_FEATURES];
        columns.push(vec![0.0, 1.0, 0.0, 1.0]);
        let mut original = table(columns);
        original.target.fill(0.0);
        let mut permuted = original.clone();
        permuted.columns.swap(0, MAX_PAIRWISE_FEATURES);
        let first = DatasetSketch::from_train(&original, Task::Regression);
        let second = DatasetSketch::from_train(&permuted, Task::Regression);
        let per_column_start = 10 + 12 * 6;
        let pairwise_start = per_column_start + MAX_PAIRWISE_FEATURES * 12;
        let expected_summary = [
            0.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 1.0, 0.0,
        ];
        // Both A and B tie on all twelve summaries, including target association.
        for summary in second.values[per_column_start..pairwise_start].chunks_exact(12) {
            assert_eq!(summary, &expected_summary[..]);
        }
        assert_eq!(first.values.len(), 856);
        assert_eq!(second.values.len(), 856);
        assert_eq!(first.values[..pairwise_start], second.values[..pairwise_start]);
        assert_eq!(first.values[pairwise_start..], [1.0; 6]);
        assert_eq!(
            second.values[pairwise_start..],
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.96875]
        );
        // A synchronous row permutation preserves alignment and the retained set.
        for column in &mut permuted.columns {
            column.reverse();
        }
        permuted.target.reverse();
        assert_eq!(
            second,
            DatasetSketch::from_train(&permuted, Task::Regression)
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

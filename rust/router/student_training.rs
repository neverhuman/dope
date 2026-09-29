
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

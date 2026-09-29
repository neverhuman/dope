|| {
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
    include!("validation_report.rs")()
}
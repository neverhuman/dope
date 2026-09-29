

    fn train_transformer(
        data: &NeuralTrainingData,
        config: NeuralTrainingConfig,
    ) -> Result<JointNetwork> {
        let device = Device::Cuda(0);
        let tokens = data.features + 1;
        let (width, heads, ff_width) =
            config.profile.transformer_dimensions().ok_or_else(|| {
                DopeError::Data("transformer profile is incompatible with training".into())
            })?;
        let binary_target_probability = match &data.target_marginal {
            Marginal::Bernoulli { probability } => Some(f64::from(*probability)),
            _ => None,
        };
        let (ranks, missing, weights, _, train_indices, validation_indices) =
            tensors(data, config, device);
        let zeros = Tensor::zeros([data.rows as i64, 1], (Kind::Float, device));
        let shifted_rank = Tensor::cat(
            &[zeros.shallow_clone(), ranks.narrow(1, 0, tokens as i64 - 1)],
            1,
        );
        let shifted_missing = Tensor::cat(&[zeros, missing.narrow(1, 0, tokens as i64 - 1)], 1);
        let input = Tensor::stack(&[shifted_rank, shifted_missing], 2);
        let mut store = nn::VarStore::new(device);
        let transformer = build_transformer(&store.root(), tokens, width, ff_width);
        let mut optimizer = nn::AdamW::default()
            .build(&store, 2e-3)
            .map_err(|error| DopeError::Data(format!("transformer AdamW optimizer: {error}")))?;
        let mut best = f64::INFINITY;
        let mut best_checkpoint = None;
        let mut plateau_checks = 0usize;
        let mut row_batches = DeterministicRowBatches::new(
            train_indices.clone(),
            config.seed ^ 0x71a4_5f09,
            TRANSFORMER_ROW_BATCH,
        );
        for step in 0..2_000 {
            enforce_deadline(config.deadline)?;
            if step == 1_600 {
                if let Some(checkpoint) = best_checkpoint.take() {
                    restore(&mut store, checkpoint)?;
                }
                best = f64::INFINITY;
                plateau_checks = 0;
            }
            let qat = step >= 1_600;
            let batch_indices = row_batches.next(device);
            let (parameters, logits) = transformer_forward(
                &transformer,
                &input.index_select(0, &batch_indices),
                qat,
                width,
                heads,
            );
            let loss = reconstruction_loss(
                parameters,
                logits,
                &ranks.index_select(0, &batch_indices),
                &missing.index_select(0, &batch_indices),
                &weights.index_select(0, &batch_indices),
                config.structural_penalty,
                binary_target_probability,
            );
            optimizer.backward_step_clip(&loss, 5.0);
            if (step + 1) % 50 == 0 && !validation_indices.is_empty() {
                let mut validation_sum = 0.0;
                for validation_indices in
                    validation_batches(&validation_indices, device, TRANSFORMER_ROW_BATCH)
                {
                    let (parameters, logits) = transformer_forward(
                        &transformer,
                        &input.index_select(0, &validation_indices),
                        qat,
                        width,
                        heads,
                    );
                    validation_sum += reconstruction_loss(
                        parameters,
                        logits,
                        &ranks.index_select(0, &validation_indices),
                        &missing.index_select(0, &validation_indices),
                        &weights.index_select(0, &validation_indices),
                        config.structural_penalty,
                        binary_target_probability,
                    )
                    .double_value(&[])
                        * validation_indices.size()[0] as f64;
                }
                let validation = validation_sum / validation_indices.len() as f64;
                if validation + 1e-7 < best {
                    best = validation;
                    best_checkpoint = Some(snapshot(&store)?);
                    plateau_checks = 0;
                } else if step + 1 >= 400 {
                    plateau_checks += 1;
                    if qat && plateau_checks >= 8 {
                        break;
                    }
                }
            }
        }
        if let Some(checkpoint) = best_checkpoint {
            restore(&mut store, checkpoint)?;
        }
        let blocks = transformer
            .blocks
            .iter()
            .enumerate()
            .map(|(index, block)| {
                Ok(TransformerBlock {
                    query: export_linear(&block.query, &format!("block {index} query"))?,
                    key: export_linear(&block.key, &format!("block {index} key"))?,
                    value: export_linear(&block.value, &format!("block {index} value"))?,
                    attention_output: export_linear(
                        &block.attention_output,
                        &format!("block {index} attention output"),
                    )?,
                    attention_norm: export_norm(
                        &block.attention_norm,
                        &format!("block {index} attention norm"),
                    )?,
                    feed_forward_1: export_linear(
                        &block.feed_forward_1,
                        &format!("block {index} feed forward 1"),
                    )?,
                    feed_forward_2: export_linear(
                        &block.feed_forward_2,
                        &format!("block {index} feed forward 2"),
                    )?,
                    feed_forward_norm: export_norm(
                        &block.feed_forward_norm,
                        &format!("block {index} feed forward norm"),
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(JointNetwork::MaskedAutoregressiveTransformer {
            transformer: Box::new(AutoregressiveTransformer {
                input_projection: export_linear(
                    &transformer.input_projection,
                    "transformer input projection",
                )?,
                positional_embeddings: tensor_values(
                    transformer.positional_embeddings,
                    "transformer positional embeddings",
                )?,
                blocks,
                value_head: export_linear(&transformer.value_head, "transformer value head")?,
                missing_head: export_linear(&transformer.missing_head, "transformer missing head")?,
            }),
        })
    }

    pub(super) fn fit(
        data: &NeuralTrainingData,
        config: NeuralTrainingConfig,
    ) -> Result<JointGenerator> {
        let tokens = data.features + 1;
        if data.rows < 2
            || data.features == 0
            || data.ranks.len() != data.rows * tokens
            || data.missing.len() != data.ranks.len()
            || data.row_weights.len() != data.rows
            || data.feature_permutation.len() != data.features
            || data.normalization.len() != tokens
        {
            return Err(DopeError::Data(
                "invalid joint neural training matrix".into(),
            ));
        }
        if match config.architecture {
            NeuralArchitecture::Tvae => config.profile.tvae_dimensions().is_none(),
            NeuralArchitecture::MaskedAutoregressiveTransformer => {
                config.profile.transformer_dimensions().is_none()
            }
            NeuralArchitecture::TabSyn | NeuralArchitecture::TabDdpm => {
                config.profile != NeuralProfile::Full
            }
        } {
            return Err(DopeError::Data(
                "neural profile is incompatible with its architecture".into(),
            ));
        }
        if estimated_gpu_bytes(data, config.architecture) > GPU_MEMORY_LIMIT_BYTES {
            return Err(DopeError::Data(
                "GPU OOM: estimated neural training footprint exceeds the frozen 16 GiB limit"
                    .into(),
            ));
        }
        if !tch::Cuda::is_available() {
            return Err(DopeError::Unsupported(
                "joint neural candidate requires CUDA libtorch training".into(),
            ));
        }
        enforce_deadline(config.deadline)?;
        let network =
            crate::libtorch::with_seeded_libtorch(config.seed, || match config.architecture {
                NeuralArchitecture::Tvae => {
                    let (_, _, _, _, _, decoder) = train_autoencoder(data, config, true, 2_000)?;
                    Ok(JointNetwork::Tvae {
                        decoder: Box::new(export_decoder(&decoder)?),
                    })
                }
                NeuralArchitecture::MaskedAutoregressiveTransformer => {
                    train_transformer(data, config)
                }
                NeuralArchitecture::TabSyn => train_diffusion(data, config, false),
                NeuralArchitecture::TabDdpm => train_diffusion(data, config, true),
            })?;
        let candidate_id = match (config.architecture, config.profile) {
            (NeuralArchitecture::Tvae, NeuralProfile::Full) => "tvae",
            (NeuralArchitecture::Tvae, NeuralProfile::MicroTvae4) => "micro_tvae_4_16",
            (NeuralArchitecture::Tvae, NeuralProfile::MicroTvae8) => "micro_tvae_8_24",
            (NeuralArchitecture::Tvae, NeuralProfile::MicroTvae12) => "micro_tvae_12_32",
            (NeuralArchitecture::MaskedAutoregressiveTransformer, NeuralProfile::Full) => {
                "single_table_autoregressive_transformer"
            }
            (NeuralArchitecture::MaskedAutoregressiveTransformer, NeuralProfile::TinyMat16) => {
                "tiny_mat_16_2_32"
            }
            (NeuralArchitecture::MaskedAutoregressiveTransformer, NeuralProfile::TinyMat24) => {
                "tiny_mat_24_3_48"
            }
            (NeuralArchitecture::TabSyn, NeuralProfile::Full) => "tabsyn",
            (NeuralArchitecture::TabDdpm, NeuralProfile::Full) => "tabddpm_direct_rank",
            _ => unreachable!("validated neural profile"),
        };
        let mut training_hasher = blake3::Hasher::new();
        training_hasher.update(b"joint-training-run-v1");
        training_hasher.update(data.training_hash.as_bytes());
        training_hasher.update(candidate_id.as_bytes());
        training_hasher.update(&config.seed.to_le_bytes());
        training_hasher.update(&config.target_weight.to_bits().to_le_bytes());
        training_hasher.update(&config.structural_penalty.to_bits().to_le_bytes());
        let generator = JointGenerator {
            architecture: config.architecture,
            profile: config.profile,
            feature_permutation: data.feature_permutation.clone(),
            target_marginal: data.target_marginal.clone(),
            normalization: data.normalization.clone(),
            network,
            training_hash: training_hasher.finalize().to_hex().to_string(),
            implementation_hash: crate::contract::candidate_implementation_hash(candidate_id),
        };
        // Task-specific validation is repeated by Kernel::validate after export.
        generator
            .validate(data.features, crate::model::Task::Regression)
            .or_else(|_| generator.validate(data.features, crate::model::Task::Binary))
            .map_err(DopeError::Data)?;
        Ok(generator)
    }

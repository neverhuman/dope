

    fn train_autoencoder(
        data: &NeuralTrainingData,
        config: NeuralTrainingConfig,
        variational: bool,
        max_steps: usize,
    ) -> Result<(
        nn::VarStore,
        nn::Linear,
        nn::Linear,
        nn::Linear,
        nn::Linear,
        DecoderLayers,
    )> {
        let device = Device::Cuda(0);
        let tokens = data.features + 1;
        let (latent_width, hidden_width) = config.profile.tvae_dimensions().ok_or_else(|| {
            DopeError::Data("TVAE profile is incompatible with autoencoder training".into())
        })?;
        let binary_target_probability = match &data.target_marginal {
            Marginal::Bernoulli { probability } => Some(f64::from(*probability)),
            _ => None,
        };
        let (ranks, missing, weights, input, train_indices, validation_indices) =
            tensors(data, config, device);
        let mut store = nn::VarStore::new(device);
        let root = store.root();
        let encoder_1 = nn::linear(
            &root / "encoder_1",
            (tokens * 2) as i64,
            hidden_width as i64,
            Default::default(),
        );
        let encoder_2 = nn::linear(
            &root / "encoder_2",
            hidden_width as i64,
            hidden_width as i64,
            Default::default(),
        );
        let latent_location = nn::linear(
            &root / "latent_location",
            hidden_width as i64,
            latent_width as i64,
            Default::default(),
        );
        let latent_log_variance = nn::linear(
            &root / "latent_log_variance",
            hidden_width as i64,
            latent_width as i64,
            Default::default(),
        );
        let decoder = build_decoder(&(&root / "decoder"), tokens, latent_width, hidden_width);
        let mut optimizer = nn::AdamW::default()
            .build(&store, 2e-3)
            .map_err(|error| DopeError::Data(format!("joint AdamW optimizer: {error}")))?;
        let mut best_validation = f64::INFINITY;
        let mut best_checkpoint = None;
        let mut plateau_checks = 0usize;
        let qat_start = max_steps * 4 / 5;
        let mut row_batches = DeterministicRowBatches::new(
            train_indices.clone(),
            config.seed ^ 0xa810_ec01,
            TRAIN_ROW_BATCH,
        );
        for step in 0..max_steps {
            enforce_deadline(config.deadline)?;
            if step == qat_start {
                if let Some(checkpoint) = best_checkpoint.take() {
                    restore(&mut store, checkpoint)?;
                }
                best_validation = f64::INFINITY;
                plateau_checks = 0;
            }
            let qat = step >= qat_start;
            let batch_indices = row_batches.next(device);
            let batch = input.index_select(0, &batch_indices);
            let hidden = encoder_1.forward(&batch).gelu("tanh").dropout(0.1, true);
            let hidden = encoder_2.forward(&hidden).gelu("tanh").dropout(0.1, true);
            let location = latent_location.forward(&hidden);
            let log_variance = latent_log_variance.forward(&hidden).clamp(-10.0, 5.0);
            let latent = if variational {
                &location + (&log_variance * 0.5).exp() * Tensor::randn_like(&location)
            } else {
                location.shallow_clone()
            };
            let (parameters, missing_logits) = decoder_forward(&decoder, &latent, qat);
            let mut loss = reconstruction_loss(
                parameters,
                missing_logits,
                &ranks.index_select(0, &batch_indices),
                &missing.index_select(0, &batch_indices),
                &weights.index_select(0, &batch_indices),
                config.structural_penalty,
                binary_target_probability,
            );
            if variational {
                let kl = -0.5
                    * (1.0f64 + &log_variance
                        - location.pow_tensor_scalar(2.0)
                        - log_variance.exp())
                    .mean(Kind::Float);
                let beta = 0.01 * ((step + 1) as f64 / 400.0).min(1.0);
                loss += beta * kl;
            }
            optimizer.backward_step_clip(&loss, 5.0);
            if (step + 1) % 50 == 0 && !validation_indices.is_empty() {
                let mut validation_sum = 0.0;
                for validation_indices in
                    validation_batches(&validation_indices, device, VALIDATION_ROW_BATCH)
                {
                    let batch = input.index_select(0, &validation_indices);
                    let hidden = encoder_2
                        .forward(&encoder_1.forward(&batch).gelu("tanh"))
                        .gelu("tanh");
                    let latent = latent_location.forward(&hidden);
                    let (parameters, logits) = decoder_forward(&decoder, &latent, qat);
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
                if validation + 1e-7 < best_validation {
                    best_validation = validation;
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
        Ok((
            store,
            encoder_1,
            encoder_2,
            latent_location,
            latent_log_variance,
            decoder,
        ))
    }

    include!("../gpu_diffusion.rs");

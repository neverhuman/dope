    fn cosine_schedule() -> Vec<f32> {
        let offset = 0.008f64;
        let base = (offset / (1.0 + offset) * std::f64::consts::FRAC_PI_2)
            .cos()
            .powi(2);
        (0..DIFFUSION_TRAIN_STEPS)
            .map(|step| {
                let time = (step + 1) as f64 / DIFFUSION_TRAIN_STEPS as f64;
                (((time + offset) / (1.0 + offset) * std::f64::consts::FRAC_PI_2)
                    .cos()
                    .powi(2)
                    / base) as f32
            })
            .collect()
    }

    fn inference_timesteps() -> Vec<u8> {
        (0..DIFFUSION_INFERENCE_STEPS)
            .map(|index| {
                ((DIFFUSION_TRAIN_STEPS - 1)
                    - index * (DIFFUSION_TRAIN_STEPS - 1) / (DIFFUSION_INFERENCE_STEPS - 1))
                    as u8
            })
            .collect()
    }

    fn time_embedding(rows: i64, timestep: usize, device: Device) -> Tensor {
        let mut values = Vec::with_capacity(NEURAL_LATENT_WIDTH);
        for index in 0..NEURAL_LATENT_WIDTH / 2 {
            let frequency = 10_000.0f32.powf(-2.0 * index as f32 / NEURAL_LATENT_WIDTH as f32);
            values.push((timestep as f32 * frequency).sin());
            values.push((timestep as f32 * frequency).cos());
        }
        Tensor::from_slice(&values)
            .to_device(device)
            .view([1, NEURAL_LATENT_WIDTH as i64])
            .repeat([rows, 1])
    }

    fn denoiser_forward(denoiser: &DenoiserLayers, input: &Tensor, qat: bool) -> Tensor {
        let hidden = linear(&denoiser.hidden_1, input, qat).gelu("tanh");
        let hidden = linear(&denoiser.hidden_2, &hidden, qat).gelu("tanh");
        let hidden = linear(&denoiser.hidden_3, &hidden, qat).gelu("tanh");
        linear(&denoiser.output, &hidden, qat)
    }

    fn train_diffusion(
        data: &NeuralTrainingData,
        config: NeuralTrainingConfig,
        direct_ranks: bool,
    ) -> Result<JointNetwork> {
        let device = Device::Cuda(0);
        let tokens = data.features + 1;
        let ranks = Tensor::from_slice(&data.ranks)
            .view([data.rows as i64, tokens as i64])
            .to_device(device);
        let missing = Tensor::from_slice(&data.missing)
            .view([data.rows as i64, tokens as i64])
            .to_device(device);
        let autoencoder = if direct_ranks {
            None
        } else {
            Some(train_autoencoder(data, config, false, 1_000)?)
        };
        let latent = if let Some((_, encoder_1, encoder_2, latent_location, _, _)) = &autoencoder {
            let input = Tensor::cat(&[ranks, missing], 1);
            transform_row_batches(&input, |batch| {
                latent_location.forward(
                    &encoder_2
                        .forward(&encoder_1.forward(batch).gelu("tanh"))
                        .gelu("tanh"),
                )
            })
            .detach()
        } else {
            // TabDDPM learns the Gaussian rank vector itself. No autoencoder or
            // decoder participates in this training or Rust inference path.
            ranks.detach()
        };
        let latent_width = if direct_ranks {
            tokens
        } else {
            NEURAL_LATENT_WIDTH
        };
        let (train_indices, validation_indices) = split_indices(data);
        let validation_indices_tensor = Tensor::from_slice(&validation_indices).to_device(device);
        let validation_latent = latent.index_select(0, &validation_indices_tensor);
        let validation_noise = Tensor::randn_like(&validation_latent);
        let schedule = cosine_schedule();
        let mut store = nn::VarStore::new(device);
        let root = store.root();
        let denoiser = DenoiserLayers {
            hidden_1: nn::linear(
                &root / "hidden_1",
                (latent_width + NEURAL_LATENT_WIDTH) as i64,
                NEURAL_HIDDEN_WIDTH as i64,
                Default::default(),
            ),
            hidden_2: nn::linear(
                &root / "hidden_2",
                NEURAL_HIDDEN_WIDTH as i64,
                NEURAL_HIDDEN_WIDTH as i64,
                Default::default(),
            ),
            hidden_3: nn::linear(
                &root / "hidden_3",
                NEURAL_HIDDEN_WIDTH as i64,
                NEURAL_HIDDEN_WIDTH as i64,
                Default::default(),
            ),
            output: nn::linear(
                &root / "output",
                NEURAL_HIDDEN_WIDTH as i64,
                latent_width as i64,
                Default::default(),
            ),
        };
        let mut optimizer = nn::AdamW::default()
            .build(&store, 2e-3)
            .map_err(|error| DopeError::Data(format!("diffusion AdamW optimizer: {error}")))?;
        let mut best = f64::INFINITY;
        let mut best_checkpoint = None;
        let mut plateau_checks = 0usize;
        let mut row_batches =
            DeterministicRowBatches::new(train_indices, config.seed ^ 0x7ab5_91d3, TRAIN_ROW_BATCH);
        for step in 0..1_500 {
            enforce_deadline(config.deadline)?;
            if step == 1_200 {
                if let Some(checkpoint) = best_checkpoint.take() {
                    restore(&mut store, checkpoint)?;
                }
                best = f64::INFINITY;
                plateau_checks = 0;
            }
            let timestep = (step * 37 + 17) % DIFFUSION_TRAIN_STEPS;
            let alpha = f64::from(schedule[timestep]);
            let batch_indices = row_batches.next(device);
            let training_latent = latent.index_select(0, &batch_indices);
            let noise = Tensor::randn_like(&training_latent);
            let noisy = alpha.sqrt() * &training_latent + (1.0 - alpha).sqrt() * &noise;
            let denoiser_input = Tensor::cat(
                &[
                    noisy,
                    time_embedding(training_latent.size()[0], timestep, device),
                ],
                1,
            );
            let prediction = denoiser_forward(&denoiser, &denoiser_input, step >= 1_200);
            let loss = prediction.mse_loss(&noise, tch::Reduction::Mean);
            optimizer.backward_step_clip(&loss, 5.0);
            if (step + 1) % 50 == 0 {
                let validation_timestep = DIFFUSION_TRAIN_STEPS / 2;
                let validation_alpha = f64::from(schedule[validation_timestep]);
                let mut validation_sum = 0.0;
                let mut validation_rows = 0usize;
                for offset in (0..validation_latent.size()[0]).step_by(VALIDATION_ROW_BATCH) {
                    let rows =
                        (validation_latent.size()[0] - offset).min(VALIDATION_ROW_BATCH as i64);
                    let latent_batch = validation_latent.narrow(0, offset, rows);
                    let noise_batch = validation_noise.narrow(0, offset, rows);
                    let validation_noisy = validation_alpha.sqrt() * &latent_batch
                        + (1.0 - validation_alpha).sqrt() * &noise_batch;
                    let validation_input = Tensor::cat(
                        &[
                            validation_noisy,
                            time_embedding(rows, validation_timestep, device),
                        ],
                        1,
                    );
                    validation_sum += denoiser_forward(&denoiser, &validation_input, step >= 1_200)
                        .mse_loss(&noise_batch, tch::Reduction::Mean)
                        .double_value(&[])
                        * rows as f64;
                    validation_rows += rows as usize;
                }
                let value = validation_sum / validation_rows.max(1) as f64;
                if value + 1e-7 < best {
                    best = value;
                    best_checkpoint = Some(snapshot(&store)?);
                    plateau_checks = 0;
                } else if step + 1 >= 400 {
                    plateau_checks += 1;
                    if step >= 1_200 && plateau_checks >= 8 {
                        break;
                    }
                }
            }
        }
        if let Some(checkpoint) = best_checkpoint {
            restore(&mut store, checkpoint)?;
        }
        let exported = Box::new(LatentDenoiser {
            hidden_1: export_linear(&denoiser.hidden_1, "diffusion denoiser hidden 1")?,
            hidden_2: export_linear(&denoiser.hidden_2, "diffusion denoiser hidden 2")?,
            hidden_3: export_linear(&denoiser.hidden_3, "diffusion denoiser hidden 3")?,
            output: export_linear(&denoiser.output, "diffusion denoiser output")?,
        });
        if direct_ranks {
            Ok(JointNetwork::TabDdpm {
                denoiser: exported,
                alpha_cumprod: schedule,
                inference_timesteps: inference_timesteps(),
            })
        } else {
            let (_, _, _, _, _, decoder) = autoencoder.expect("latent diffusion autoencoder");
            Ok(JointNetwork::TabSyn {
                decoder: Box::new(export_decoder(&decoder)?),
                denoiser: exported,
                alpha_cumprod: schedule,
                inference_timesteps: inference_timesteps(),
            })
        }
    }

    fn build_transformer(
        path: &nn::Path<'_>,
        tokens: usize,
        width: usize,
        ff_width: usize,
    ) -> TransformerLayers {
        let linear_width = || nn::LinearConfig::default();
        let blocks = (0..2)
            .map(|index| {
                let block = path / format!("block_{index}");
                TransformerTrainingBlock {
                    query: nn::linear(&block / "query", width as i64, width as i64, linear_width()),
                    key: nn::linear(&block / "key", width as i64, width as i64, linear_width()),
                    value: nn::linear(&block / "value", width as i64, width as i64, linear_width()),
                    attention_output: nn::linear(
                        &block / "attention_output",
                        width as i64,
                        width as i64,
                        linear_width(),
                    ),
                    attention_norm: nn::layer_norm(
                        &block / "attention_norm",
                        vec![width as i64],
                        Default::default(),
                    ),
                    feed_forward_1: nn::linear(
                        &block / "feed_forward_1",
                        width as i64,
                        ff_width as i64,
                        linear_width(),
                    ),
                    feed_forward_2: nn::linear(
                        &block / "feed_forward_2",
                        ff_width as i64,
                        width as i64,
                        linear_width(),
                    ),
                    feed_forward_norm: nn::layer_norm(
                        &block / "feed_forward_norm",
                        vec![width as i64],
                        Default::default(),
                    ),
                }
            })
            .collect();
        TransformerLayers {
            input_projection: nn::linear(
                path / "input_projection",
                2,
                width as i64,
                Default::default(),
            ),
            positional_embeddings: path.var(
                "positional_embeddings",
                &[tokens as i64, width as i64],
                nn::Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
            ),
            blocks,
            value_head: nn::linear(path / "value_head", width as i64, 2, Default::default()),
            missing_head: nn::linear(path / "missing_head", width as i64, 1, Default::default()),
        }
    }

    fn transformer_forward(
        transformer: &TransformerLayers,
        input: &Tensor,
        qat: bool,
        width: usize,
        heads: usize,
    ) -> (Tensor, Tensor) {
        let sizes = input.size();
        let rows = sizes[0];
        let tokens = sizes[1];
        let mut state = linear(&transformer.input_projection, input, qat)
            + transformer.positional_embeddings.unsqueeze(0);
        let head_width = (width / heads) as i64;
        let mask = Tensor::ones([tokens, tokens], (Kind::Float, input.device())).triu(1) * -1e9;
        for block in &transformer.blocks {
            let reshape = |value: Tensor| {
                value
                    .view([rows, tokens, heads as i64, head_width])
                    .transpose(1, 2)
            };
            let query = reshape(linear(&block.query, &state, qat));
            let key = reshape(linear(&block.key, &state, qat));
            let value = reshape(linear(&block.value, &state, qat));
            let attention = (query.matmul(&key.transpose(-2, -1)) / (head_width as f64).sqrt()
                + &mask)
                .softmax(-1, Kind::Float)
                .matmul(&value)
                .transpose(1, 2)
                .contiguous()
                .view([rows, tokens, width as i64]);
            state = block
                .attention_norm
                .forward(&(state + linear(&block.attention_output, &attention, qat)));
            let feed_forward = linear(
                &block.feed_forward_2,
                &linear(&block.feed_forward_1, &state, qat).gelu("tanh"),
                qat,
            );
            state = block.feed_forward_norm.forward(&(state + feed_forward));
        }
        (
            linear(&transformer.value_head, &state, qat),
            linear(&transformer.missing_head, &state, qat).squeeze_dim(-1),
        )
    }
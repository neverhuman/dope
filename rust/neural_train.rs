use std::time::Instant;

use crate::error::{DopeError, Result};
use crate::model::{
    JointGenerator, Marginal, NeuralArchitecture, NeuralProfile, RankNormalization,
};

#[derive(Clone, Debug)]
pub struct NeuralTrainingData {
    pub rows: usize,
    pub features: usize,
    /// Row-major Gaussian rank values, including the target in the final slot.
    pub ranks: Vec<f32>,
    /// Row-major explicit missingness channels, including the target.
    pub missing: Vec<f32>,
    pub row_weights: Vec<f32>,
    pub feature_permutation: Vec<u32>,
    pub target_marginal: Marginal,
    pub normalization: Vec<RankNormalization>,
    pub training_hash: String,
}

#[derive(Clone, Copy, Debug)]
pub struct NeuralTrainingConfig {
    pub architecture: NeuralArchitecture,
    pub profile: NeuralProfile,
    pub target_weight: f64,
    pub structural_penalty: f64,
    pub seed: u64,
    pub deadline: Instant,
}

#[cfg(not(feature = "gpu-training"))]
pub fn fit_joint_generator(
    _data: &NeuralTrainingData,
    _config: NeuralTrainingConfig,
) -> Result<JointGenerator> {
    Err(DopeError::Unsupported(
        "joint neural training requires the gpu-training build".into(),
    ))
}

#[cfg(feature = "gpu-training")]
pub fn fit_joint_generator(
    data: &NeuralTrainingData,
    config: NeuralTrainingConfig,
) -> Result<JointGenerator> {
    gpu::fit(data, config)
}

#[cfg(feature = "gpu-training")]
mod gpu {
    use super::*;
    use tch::nn::{Module, OptimizerConfig};
    use tch::{Device, Kind, Tensor, nn};

    use crate::model::{
        AutoregressiveTransformer, DIFFUSION_INFERENCE_STEPS, DIFFUSION_TRAIN_STEPS, JointDecoder,
        JointNetwork, LatentDenoiser, LayerNormParameters, NEURAL_HIDDEN_WIDTH,
        NEURAL_LATENT_WIDTH, TRANSFORMER_HEADS, TRANSFORMER_WIDTH, TransformerBlock,
    };
    use crate::neural::quantize_linear;

    const GPU_MEMORY_LIMIT_BYTES: u128 = 16 * 1024 * 1024 * 1024;
    const TRAIN_ROW_BATCH: usize = 128;
    const TRANSFORMER_ROW_BATCH: usize = 32;
    const VALIDATION_ROW_BATCH: usize = 256;

    fn estimated_gpu_bytes(data: &NeuralTrainingData, architecture: NeuralArchitecture) -> u128 {
        let rows = data.rows as u128;
        let tokens = (data.features + 1) as u128;
        let batch_rows = rows.min(match architecture {
            NeuralArchitecture::MaskedAutoregressiveTransformer => TRANSFORMER_ROW_BATCH as u128,
            _ => TRAIN_ROW_BATCH as u128,
        });
        let activations = match architecture {
            NeuralArchitecture::Tvae | NeuralArchitecture::TabSyn | NeuralArchitecture::TabDdpm => {
                batch_rows
                    * (tokens * 5
                        + NEURAL_HIDDEN_WIDTH as u128 * 12
                        + NEURAL_LATENT_WIDTH as u128 * 10)
            }
            NeuralArchitecture::MaskedAutoregressiveTransformer => {
                batch_rows
                    * (tokens * (5 + TRANSFORMER_WIDTH as u128 * 20)
                        + TRANSFORMER_HEADS as u128 * tokens * tokens * 2)
            }
        };
        // Ranks, missingness, weights, and shifted/input channels stay resident;
        // only autograd activations scale with the bounded row batch.
        let resident = rows.saturating_mul(tokens).saturating_mul(8);
        activations
            .saturating_add(resident)
            .saturating_mul(std::mem::size_of::<f32>() as u128)
            .saturating_add(64 * 1024 * 1024)
    }

    #[inline]
    fn splitmix64(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    struct DeterministicRowBatches {
        base: Vec<i64>,
        order: Vec<i64>,
        cursor: usize,
        epoch: u64,
        seed: u64,
        batch_size: usize,
    }

    impl DeterministicRowBatches {
        fn new(indices: Vec<i64>, seed: u64, batch_size: usize) -> Self {
            debug_assert!(batch_size > 0);
            let mut batches = Self {
                base: indices.clone(),
                order: indices,
                cursor: 0,
                epoch: 0,
                seed,
                batch_size,
            };
            batches.shuffle_epoch();
            batches
        }

        fn shuffle_epoch(&mut self) {
            let epoch_seed = splitmix64(self.seed ^ self.epoch);
            self.order.clone_from(&self.base);
            self.order.sort_by_key(|row| {
                (
                    splitmix64(epoch_seed ^ (*row as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)),
                    *row,
                )
            });
            self.cursor = 0;
        }

        fn next(&mut self, device: Device) -> Tensor {
            if self.cursor == self.order.len() {
                self.epoch = self.epoch.wrapping_add(1);
                self.shuffle_epoch();
            }
            let end = (self.cursor + self.batch_size).min(self.order.len());
            let indices = Tensor::from_slice(&self.order[self.cursor..end]).to_device(device);
            self.cursor = end;
            indices
        }
    }

    fn validation_batches(
        indices: &[i64],
        device: Device,
        batch_size: usize,
    ) -> impl Iterator<Item = Tensor> + '_ {
        indices
            .chunks(batch_size)
            .map(move |chunk| Tensor::from_slice(chunk).to_device(device))
    }

    fn transform_row_batches(rows: &Tensor, transform: impl Fn(&Tensor) -> Tensor) -> Tensor {
        let transformed = rows
            .split(TRAIN_ROW_BATCH as i64, 0)
            .into_iter()
            .map(|batch| transform(&batch))
            .collect::<Vec<_>>();
        Tensor::cat(&transformed, 0)
    }

    fn enforce_deadline(deadline: Instant) -> Result<()> {
        if Instant::now() >= deadline {
            return Err(DopeError::Data(
                "neural fitting timeout exceeded the frozen 600 second limit".into(),
            ));
        }
        Ok(())
    }

    fn snapshot(store: &nn::VarStore) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        store
            .save_to_stream(&mut bytes)
            .map_err(|error| DopeError::Data(format!("neural checkpoint save: {error}")))?;
        Ok(bytes)
    }

    fn restore(store: &mut nn::VarStore, bytes: Vec<u8>) -> Result<()> {
        store
            .load_from_stream(std::io::Cursor::new(bytes))
            .map_err(|error| DopeError::Data(format!("neural checkpoint restore: {error}")))
    }

    struct DecoderLayers {
        hidden_1: nn::Linear,
        hidden_2: nn::Linear,
        value_head: nn::Linear,
        missing_head: nn::Linear,
    }

    struct TransformerLayers {
        input_projection: nn::Linear,
        positional_embeddings: Tensor,
        blocks: Vec<TransformerTrainingBlock>,
        value_head: nn::Linear,
        missing_head: nn::Linear,
    }

    struct TransformerTrainingBlock {
        query: nn::Linear,
        key: nn::Linear,
        value: nn::Linear,
        attention_output: nn::Linear,
        attention_norm: nn::LayerNorm,
        feed_forward_1: nn::Linear,
        feed_forward_2: nn::Linear,
        feed_forward_norm: nn::LayerNorm,
    }

    struct DenoiserLayers {
        hidden_1: nn::Linear,
        hidden_2: nn::Linear,
        hidden_3: nn::Linear,
        output: nn::Linear,
    }

    fn tensor_values(tensor: Tensor, context: &str) -> Result<Vec<f32>> {
        Vec::<f32>::try_from(tensor.to_device(Device::Cpu).view([-1]))
            .map_err(|error| DopeError::Data(format!("{context}: {error}")))
    }

    fn export_linear(linear: &nn::Linear, context: &str) -> Result<crate::model::QuantizedLinear> {
        let size = linear.ws.size();
        let output = usize::try_from(size[0]).map_err(|_| DopeError::Data(context.into()))?;
        let input = usize::try_from(size[1]).map_err(|_| DopeError::Data(context.into()))?;
        let weights = tensor_values(linear.ws.shallow_clone(), context)?;
        let biases = linear
            .bs
            .as_ref()
            .map(|bias| tensor_values(bias.shallow_clone(), context))
            .transpose()?
            .unwrap_or_else(|| vec![0.0; output]);
        quantize_linear(input, output, &weights, &biases)
    }

    fn export_norm(norm: &nn::LayerNorm, context: &str) -> Result<LayerNormParameters> {
        let weight = norm
            .ws
            .as_ref()
            .ok_or_else(|| DopeError::Data(format!("{context} has no weight")))?;
        let bias = norm
            .bs
            .as_ref()
            .ok_or_else(|| DopeError::Data(format!("{context} has no bias")))?;
        Ok(LayerNormParameters {
            weight: tensor_values(weight.shallow_clone(), context)?,
            bias: tensor_values(bias.shallow_clone(), context)?,
        })
    }

    fn fake_int8_weight(weight: &Tensor) -> Tensor {
        let scale = (weight.abs().amax([1], true) / 127.0).clamp_min(1e-8);
        let quantized = (weight / &scale).round().clamp(-128.0, 127.0) * scale;
        weight + (quantized - weight).detach()
    }

    fn linear(linear: &nn::Linear, input: &Tensor, qat: bool) -> Tensor {
        if qat {
            input.linear(&fake_int8_weight(&linear.ws), linear.bs.as_ref())
        } else {
            linear.forward(input)
        }
    }

    fn decoder_forward(decoder: &DecoderLayers, latent: &Tensor, qat: bool) -> (Tensor, Tensor) {
        let hidden = linear(&decoder.hidden_1, latent, qat).gelu("tanh");
        let hidden = linear(&decoder.hidden_2, &hidden, qat).gelu("tanh");
        (
            linear(&decoder.value_head, &hidden, qat),
            linear(&decoder.missing_head, &hidden, qat),
        )
    }

    fn build_decoder(
        path: &nn::Path<'_>,
        tokens: usize,
        latent_width: usize,
        hidden_width: usize,
    ) -> DecoderLayers {
        DecoderLayers {
            hidden_1: nn::linear(
                path / "hidden_1",
                latent_width as i64,
                hidden_width as i64,
                Default::default(),
            ),
            hidden_2: nn::linear(
                path / "hidden_2",
                hidden_width as i64,
                hidden_width as i64,
                Default::default(),
            ),
            value_head: nn::linear(
                path / "value_head",
                hidden_width as i64,
                (tokens * 2) as i64,
                Default::default(),
            ),
            missing_head: nn::linear(
                path / "missing_head",
                hidden_width as i64,
                tokens as i64,
                Default::default(),
            ),
        }
    }

    fn export_decoder(decoder: &DecoderLayers) -> Result<JointDecoder> {
        Ok(JointDecoder {
            hidden_1: export_linear(&decoder.hidden_1, "decoder hidden 1")?,
            hidden_2: export_linear(&decoder.hidden_2, "decoder hidden 2")?,
            value_head: export_linear(&decoder.value_head, "decoder value head")?,
            missing_head: export_linear(&decoder.missing_head, "decoder missing head")?,
            variance_floor: 0.01,
        })
    }

    fn reconstruction_loss(
        parameters: Tensor,
        missing_logits: Tensor,
        ranks: &Tensor,
        missing: &Tensor,
        weights: &Tensor,
        structural_penalty: f64,
        binary_target_probability: Option<f64>,
    ) -> Tensor {
        let tokens = ranks.size()[1];
        let parameters = parameters.view([-1, tokens, 2]);
        let location = parameters.select(2, 0);
        let log_scale = parameters.select(2, 1).clamp(-8.0, 2.0);
        let observed_weights = weights * (1.0f64 - missing);
        let gaussian = ((ranks - &location).pow_tensor_scalar(2.0) * (-2.0f64 * &log_scale).exp()
            + 2.0f64 * &log_scale)
            * observed_weights;
        let missing_loss = missing_logits.binary_cross_entropy_with_logits::<&Tensor>(
            missing,
            Some(weights),
            None,
            tch::Reduction::Mean,
        );
        let mut loss = gaussian.mean(Kind::Float) + missing_loss;
        if let Some(positive_probability) = binary_target_probability {
            let threshold = statrs::distribution::ContinuousCDF::inverse_cdf(
                &statrs::distribution::Normal::new(0.0, 1.0).expect("standard normal"),
                (1.0 - positive_probability).clamp(1e-6, 1.0 - 1e-6),
            );
            let target_location = location.select(1, tokens - 1);
            let target_scale = log_scale.select(1, tokens - 1).exp().clamp_min(1e-4);
            let target_probability = 0.5f64
                * (1.0f64
                    + ((target_location - threshold) / (target_scale * std::f64::consts::SQRT_2))
                        .erf());
            let target = ranks
                .select(1, tokens - 1)
                .ge(threshold)
                .to_kind(Kind::Float);
            loss += (target_probability - target)
                .pow_tensor_scalar(2.0)
                .mean(Kind::Float)
                * 0.1f64;
        }
        if structural_penalty > 0.0 {
            let structural_width = tokens.min(64);
            let actual_values = ranks.narrow(1, 0, structural_width);
            let generated_values = location.narrow(1, 0, structural_width);
            let actual = &actual_values - actual_values.mean_dim(&[0i64][..], true, Kind::Float);
            let generated =
                &generated_values - generated_values.mean_dim(&[0i64][..], true, Kind::Float);
            let denominator = ranks.size()[0].max(1) as f64;
            let actual_correlation = actual.transpose(0, 1).matmul(&actual) / denominator;
            let generated_correlation = generated.transpose(0, 1).matmul(&generated) / denominator;
            let correlation = (actual_correlation - generated_correlation)
                .pow_tensor_scalar(2.0)
                .mean(Kind::Float);
            let sample = ranks.size()[0].min(256);
            let actual_sample = actual_values.narrow(0, 0, sample);
            let generated_sample = generated_values.narrow(0, 0, sample);
            let squared_distance = |left: &Tensor, right: &Tensor| {
                (left.unsqueeze(1) - right.unsqueeze(0))
                    .pow_tensor_scalar(2.0)
                    .mean_dim(&[2i64][..], false, Kind::Float)
            };
            let xx = squared_distance(&actual_sample, &actual_sample);
            let yy = squared_distance(&generated_sample, &generated_sample);
            let xy = squared_distance(&actual_sample, &generated_sample);
            let mmd = [0.1f64, 1.0, 10.0]
                .into_iter()
                .map(|bandwidth| {
                    (-&xx / bandwidth).exp().mean(Kind::Float)
                        + (-&yy / bandwidth).exp().mean(Kind::Float)
                        - 2.0 * (-&xy / bandwidth).exp().mean(Kind::Float)
                })
                .reduce(|left, right| left + right)
                .expect("three MMD bandwidths")
                / 3.0;
            // Generated rows are compared with every training row, including
            // the source row for an autoencoder pass, so exact reconstruction
            // receives the intended copy penalty.
            let nearest = xy.min_dim(1, false).0;
            let copy = (-nearest / 0.01).exp().mean(Kind::Float);
            loss += structural_penalty * (0.5 * correlation + 0.3 * mmd + 0.2 * copy);
        }
        loss
    }

    fn split_indices(data: &NeuralTrainingData) -> (Vec<i64>, Vec<i64>) {
        let tokens = data.features + 1;
        let mut train = Vec::new();
        let mut validation = Vec::new();
        for row in 0..data.rows {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"joint-early-stop-split-v1");
            for value in &data.ranks[row * tokens..(row + 1) * tokens] {
                hasher.update(&value.to_bits().to_le_bytes());
            }
            if hasher.finalize().as_bytes()[0] < 26 {
                validation.push(row as i64);
            } else {
                train.push(row as i64);
            }
        }
        if validation.is_empty() && train.len() > 1 {
            validation.push(train.pop().unwrap());
        }
        if train.is_empty() {
            train.append(&mut validation.clone());
        }
        (train, validation)
    }

    fn tensors(
        data: &NeuralTrainingData,
        config: NeuralTrainingConfig,
        device: Device,
    ) -> (Tensor, Tensor, Tensor, Tensor, Vec<i64>, Vec<i64>) {
        let tokens = data.features + 1;
        let ranks = Tensor::from_slice(&data.ranks)
            .view([data.rows as i64, tokens as i64])
            .to_device(device);
        let missing = Tensor::from_slice(&data.missing)
            .view([data.rows as i64, tokens as i64])
            .to_device(device);
        let mut loss_weights = Vec::with_capacity(data.rows * tokens);
        for row in 0..data.rows {
            loss_weights.extend((0..tokens).map(|token| {
                if token + 1 == tokens {
                    data.row_weights[row] * config.target_weight as f32
                } else {
                    1.0
                }
            }));
        }
        let weights = Tensor::from_slice(&loss_weights)
            .view([data.rows as i64, tokens as i64])
            .to_device(device);
        let (train, validation) = split_indices(data);
        let input = Tensor::cat(&[ranks.shallow_clone(), missing.shallow_clone()], 1);
        (ranks, missing, weights, input, train, validation)
    }

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
        let mut stale = 0usize;
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
                stale = 0;
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
                    stale = 0;
                } else if step + 1 >= 400 {
                    stale += 1;
                    if qat && stale >= 8 {
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
        let mut stale = 0usize;
        let mut row_batches =
            DeterministicRowBatches::new(train_indices, config.seed ^ 0x7ab5_91d3, TRAIN_ROW_BATCH);
        for step in 0..1_500 {
            enforce_deadline(config.deadline)?;
            if step == 1_200 {
                if let Some(checkpoint) = best_checkpoint.take() {
                    restore(&mut store, checkpoint)?;
                }
                best = f64::INFINITY;
                stale = 0;
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
                    stale = 0;
                } else if step + 1 >= 400 {
                    stale += 1;
                    if step >= 1_200 && stale >= 8 {
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
        let mut stale = 0usize;
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
                stale = 0;
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
                    stale = 0;
                } else if step + 1 >= 400 {
                    stale += 1;
                    if qat && stale >= 8 {
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
}

#[cfg(all(test, feature = "gpu-training"))]
mod tests {
    use super::*;
    use crate::codec::encode_kernel;
    use crate::model::{
        ColumnSchema, Kernel, KernelProgram, Marginal, QuantizedLinear, RankNormalization,
        SchemaKind, Task, Transform,
    };
    use std::time::Duration;
    use tch::{Device, Tensor};

    fn fixture(rows: usize, features: usize) -> NeuralTrainingData {
        let tokens = features + 1;
        NeuralTrainingData {
            rows,
            features,
            ranks: (0..rows * tokens)
                .map(|index| {
                    let row = index / tokens;
                    let column = index % tokens;
                    (((row * 37 + column * 101) % 997) as f32 / 997.0 - 0.5) * 4.0
                })
                .collect(),
            missing: vec![0.0; rows * tokens],
            row_weights: vec![1.0; rows],
            feature_permutation: (0..features as u32).collect(),
            target_marginal: Marginal::Gaussian {
                mean: 0.5,
                sigma: 0.2,
            },
            normalization: vec![
                RankNormalization {
                    location: 0.0,
                    scale: 1.0,
                    missing_probability: 0.0,
                    randomized_discrete: false,
                };
                tokens
            ],
            training_hash: "a".repeat(64),
        }
    }

    #[test]
    fn seeded_tvae_training_repeats_exactly() {
        let data = fixture(32, 4);
        let train = || {
            let started = Instant::now();
            fit_joint_generator(
                &data,
                NeuralTrainingConfig {
                    architecture: NeuralArchitecture::Tvae,
                    profile: NeuralProfile::Full,
                    target_weight: 2.0,
                    structural_penalty: 0.0,
                    seed: 17_729,
                    deadline: started + Duration::from_secs(60),
                },
            )
            .unwrap()
        };
        assert_eq!(train(), train());
    }

    #[test]
    fn micro_tvae_and_direct_diffusion_repeat_and_export() {
        let data = fixture(32, 2);
        for (architecture, profile) in [
            (NeuralArchitecture::Tvae, NeuralProfile::MicroTvae4),
            (NeuralArchitecture::TabDdpm, NeuralProfile::Full),
        ] {
            let train = || {
                fit_joint_generator(
                    &data,
                    NeuralTrainingConfig {
                        architecture,
                        profile,
                        target_weight: 2.0,
                        structural_penalty: 0.0,
                        seed: 17_729,
                        deadline: Instant::now() + Duration::from_secs(120),
                    },
                )
                .unwrap()
            };
            let first = train();
            let second = train();
            assert_eq!(first, second);
            let kernel = Kernel {
                task: Task::Regression,
                rows_fitted: data.rows as u64,
                features: data.features as u32,
                seed: 17_729,
                seed_policy: 0,
                quantization_bits: 8,
                compliant: false,
                schema: vec![
                    ColumnSchema {
                        kind: SchemaKind::Continuous,
                        missing_probability: 0.0,
                        impute: 0.5,
                        transform: Transform::Identity,
                    };
                    data.features
                ],
                marginals: vec![
                    Marginal::Gaussian {
                        mean: 0.5,
                        sigma: 0.2,
                    };
                    data.features
                ],
                program: KernelProgram::NeuralJoint(first),
            };
            let bytes = encode_kernel(&kernel).unwrap();
            assert_eq!(&bytes[..6], b"DPK3\x03\x03");
            if architecture == NeuralArchitecture::Tvae {
                assert!(bytes.len() < 10_240, "micro TVAE is {} bytes", bytes.len());
            }
        }
    }

    fn parity_layer(input: usize, output: usize, salt: usize) -> QuantizedLinear {
        QuantizedLinear {
            input_dim: input as u32,
            output_dim: output as u32,
            weights: (0..input * output)
                .map(|index| ((index * 17 + salt * 31) % 31) as i8 - 15)
                .collect(),
            scales: (0..output)
                .map(|channel| 0.0005 + (channel + salt) as f32 * 0.00001)
                .collect(),
            biases: (0..output)
                .map(|channel| (channel as f32 - output as f32 * 0.5) * 0.001)
                .collect(),
        }
    }

    fn parity_splitmix64(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn libtorch_linear(input: &Tensor, layer: &QuantizedLinear) -> Tensor {
        let device = input.device();
        let weights = Tensor::from_slice(
            &layer
                .weights
                .iter()
                .map(|&weight| f32::from(weight))
                .collect::<Vec<_>>(),
        )
        .view([i64::from(layer.output_dim), i64::from(layer.input_dim)])
        .to_device(device);
        let scales = Tensor::from_slice(&layer.scales).to_device(device);
        let biases = Tensor::from_slice(&layer.biases).to_device(device);
        input.matmul(&weights.transpose(0, 1)) * scales + biases
    }

    #[test]
    fn rust_libtorch_quantized_golden_matches_for_100000_draws() {
        const DRAWS: usize = 100_000;
        const BATCH: usize = 2_048;
        let first = parity_layer(32, 8, 3);
        let second = parity_layer(8, 2, 7);
        first.validate().unwrap();
        second.validate().unwrap();
        crate::libtorch::with_seeded_libtorch(91_773, || {
            let device = Device::Cuda(0);
            let mut maximum_difference = 0.0f64;
            let mut decisions_agree = 0usize;
            let mut decisions = 0usize;
            for start in (0..DRAWS).step_by(BATCH) {
                let rows = (DRAWS - start).min(BATCH);
                let inputs = (0..rows * 32)
                    .map(|offset| {
                        let bits =
                            parity_splitmix64((start * 32 + offset) as u64 ^ 0xa076_1d64_78bd_642f);
                        ((bits >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
                    })
                    .collect::<Vec<_>>();
                let reference_input = Tensor::from_slice(&inputs)
                    .view([rows as i64, 32])
                    .to_device(device);
                let reference = libtorch_linear(
                    &libtorch_linear(&reference_input, &first).gelu("tanh"),
                    &second,
                );
                let reference =
                    Vec::<f32>::try_from(reference.to_device(Device::Cpu).view([-1]))
                        .map_err(|error| DopeError::Data(format!("parity output: {error}")))?;
                for (row, expected) in inputs.chunks_exact(32).zip(reference.chunks_exact(2)) {
                    let hidden = crate::neural::int8_linear(&first, row)?
                        .into_iter()
                        .map(crate::neural::gelu)
                        .collect::<Vec<_>>();
                    let actual = crate::neural::int8_linear(&second, &hidden)?;
                    for (&actual, &expected) in actual.iter().zip(expected) {
                        maximum_difference =
                            maximum_difference.max(f64::from((actual - expected).abs()));
                        decisions_agree += usize::from((actual >= 0.0) == (expected >= 0.0));
                        decisions += 1;
                    }
                }
            }
            assert!(maximum_difference <= 1e-3, "{maximum_difference}");
            assert!(decisions_agree as f64 / decisions as f64 >= 0.999);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn transformer_486_by_273_completes_with_bounded_batches() {
        let rows = 486usize;
        let features = 273usize;
        let data = fixture(rows, features);
        let started = Instant::now();
        let generator = fit_joint_generator(
            &data,
            NeuralTrainingConfig {
                architecture: NeuralArchitecture::MaskedAutoregressiveTransformer,
                profile: NeuralProfile::Full,
                target_weight: 2.0,
                structural_penalty: 0.0,
                seed: 17_729,
                deadline: started + Duration::from_secs(600),
            },
        )
        .unwrap();
        assert!(started.elapsed() < Duration::from_secs(600));
        generator
            .validate(features, crate::model::Task::Regression)
            .unwrap();
    }
}

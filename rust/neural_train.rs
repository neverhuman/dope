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

    include!("neural_train/gpu_diffusion.rs");
}
#[cfg(all(test, feature = "gpu-training"))]
include!("neural_train/tests.rs");

use crate::error::{DopeError, Result};
use crate::model::{
    AutoregressiveTransformer, JointDecoder, JointGenerator, JointNetwork, LatentDenoiser,
    LayerNormParameters, NEURAL_LATENT_WIDTH, QuantizedLinear, TRANSFORMER_HEADS,
    TRANSFORMER_WIDTH,
};

#[inline]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[inline]
fn uniform(seed: u64, token: usize, stream: u64) -> f32 {
    let counter = (token as u64).wrapping_mul(0xd6e8_feb8_6659_fd93) ^ stream;
    let bits = splitmix64(seed ^ counter) >> 40;
    (bits as f32 + 0.5) * (1.0 / (1u32 << 24) as f32)
}

#[inline]
fn normal(seed: u64, token: usize, stream: u64) -> f32 {
    let left = uniform(seed, token, stream).max(1e-7);
    let right = uniform(seed, token, stream ^ 0xa076_1d64_78bd_642f);
    (-2.0 * left.ln()).sqrt() * (std::f32::consts::TAU * right).cos()
}

#[inline]
pub fn gelu(value: f32) -> f32 {
    0.5 * value
        * (1.0
            + (std::f32::consts::FRAC_2_SQRT_PI * (value + 0.044_715 * value * value * value))
                .tanh())
}

pub fn softmax(values: &mut [f32]) -> Result<()> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(DopeError::Codec(
            "softmax input is empty or non-finite".into(),
        ));
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        total += *value;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(DopeError::Codec("softmax normalization is invalid".into()));
    }
    for value in values {
        *value /= total;
    }
    Ok(())
}

pub fn layer_normalize(values: &mut [f32], parameters: &LayerNormParameters) -> Result<()> {
    if values.is_empty()
        || parameters.weight.len() != values.len()
        || parameters.bias.len() != values.len()
    {
        return Err(DopeError::Codec(
            "layer-normalization width mismatch".into(),
        ));
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / values.len() as f32;
    let inverse = (variance + 1e-5).sqrt().recip();
    for ((value, weight), bias) in values
        .iter_mut()
        .zip(&parameters.weight)
        .zip(&parameters.bias)
    {
        *value = (*value - mean) * inverse * *weight + *bias;
    }
    Ok(())
}

pub fn int8_linear(layer: &QuantizedLinear, input: &[f32]) -> Result<Vec<f32>> {
    layer.validate().map_err(DopeError::Codec)?;
    let width = layer.input_dim as usize;
    if input.len() != width || input.iter().any(|value| !value.is_finite()) {
        return Err(DopeError::Codec(
            "int8 linear input width or value is invalid".into(),
        ));
    }
    let mut output = Vec::with_capacity(layer.output_dim as usize);
    for channel in 0..layer.output_dim as usize {
        let row = &layer.weights[channel * width..(channel + 1) * width];
        let accumulator = row
            .iter()
            .zip(input)
            .map(|(weight, value)| f32::from(*weight) * *value)
            .sum::<f32>();
        output.push(accumulator * layer.scales[channel] + layer.biases[channel]);
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(DopeError::Codec("int8 linear output is non-finite".into()));
    }
    Ok(output)
}

pub fn quantize_linear(
    input_dim: usize,
    output_dim: usize,
    weights: &[f32],
    biases: &[f32],
) -> Result<QuantizedLinear> {
    if weights.len() != input_dim.saturating_mul(output_dim) || biases.len() != output_dim {
        return Err(DopeError::Data("linear export dimensions disagree".into()));
    }
    let mut quantized = Vec::with_capacity(weights.len());
    let mut scales = Vec::with_capacity(output_dim);
    for row in weights.chunks_exact(input_dim) {
        if row.iter().any(|value| !value.is_finite()) {
            return Err(DopeError::Data(
                "linear export contains non-finite weights".into(),
            ));
        }
        let maximum = row.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        let scale = (maximum / 127.0).max(1e-8);
        scales.push(scale);
        quantized.extend(row.iter().map(|value| {
            (*value / scale)
                .round()
                .clamp(f32::from(i8::MIN), f32::from(i8::MAX)) as i8
        }));
    }
    let layer = QuantizedLinear {
        input_dim: input_dim as u32,
        output_dim: output_dim as u32,
        weights: quantized,
        scales,
        biases: biases.to_vec(),
    };
    layer.validate().map_err(DopeError::Data)?;
    Ok(layer)
}

fn decoder_forward(decoder: &JointDecoder, latent: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
    let hidden = int8_linear(&decoder.hidden_1, latent)?
        .into_iter()
        .map(gelu)
        .collect::<Vec<_>>();
    let hidden = int8_linear(&decoder.hidden_2, &hidden)?
        .into_iter()
        .map(gelu)
        .collect::<Vec<_>>();
    Ok((
        int8_linear(&decoder.value_head, &hidden)?,
        int8_linear(&decoder.missing_head, &hidden)?,
    ))
}

fn denoiser_forward(
    denoiser: &LatentDenoiser,
    latent: &[f32],
    timestep: usize,
) -> Result<Vec<f32>> {
    let mut input = Vec::with_capacity(NEURAL_LATENT_WIDTH * 2);
    input.extend_from_slice(latent);
    for index in 0..NEURAL_LATENT_WIDTH / 2 {
        let frequency = 10_000.0f32.powf(-2.0 * index as f32 / NEURAL_LATENT_WIDTH as f32);
        input.push((timestep as f32 * frequency).sin());
        input.push((timestep as f32 * frequency).cos());
    }
    let hidden = int8_linear(&denoiser.hidden_1, &input)?
        .into_iter()
        .map(gelu)
        .collect::<Vec<_>>();
    let hidden = int8_linear(&denoiser.hidden_2, &hidden)?
        .into_iter()
        .map(gelu)
        .collect::<Vec<_>>();
    let hidden = int8_linear(&denoiser.hidden_3, &hidden)?
        .into_iter()
        .map(gelu)
        .collect::<Vec<_>>();
    int8_linear(&denoiser.output, &hidden)
}

fn decode_rank_values(
    decoder: &JointDecoder,
    latent: &[f32],
    seed: u64,
) -> Result<(Vec<f32>, Vec<bool>)> {
    let (parameters, missing_logits) = decoder_forward(decoder, latent)?;
    let tokens = missing_logits.len();
    if parameters.len() != tokens * 2 {
        return Err(DopeError::Codec(
            "joint decoder head widths disagree".into(),
        ));
    }
    let values = (0..tokens)
        .map(|token| {
            let location = parameters[token * 2];
            let scale = parameters[token * 2 + 1]
                .clamp(-8.0, 2.0)
                .exp()
                .max(decoder.variance_floor);
            location + scale * normal(seed, token, 0x31c3_84d2)
        })
        .collect();
    let missing = missing_logits
        .iter()
        .enumerate()
        .map(|(token, logit)| {
            let probability = 1.0 / (1.0 + (-logit.clamp(-20.0, 20.0)).exp());
            uniform(seed, token, 0xe922_7a11) <= probability
        })
        .collect();
    Ok((values, missing))
}

#[derive(Default)]
struct AttentionCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

fn cached_attention(query: &[f32], cache: &AttentionCache) -> Result<Vec<f32>> {
    if query.len() != TRANSFORMER_WIDTH
        || cache.keys.is_empty()
        || cache.keys.len() != cache.values.len()
    {
        return Err(DopeError::Codec(
            "invalid autoregressive attention cache".into(),
        ));
    }
    let head_width = TRANSFORMER_WIDTH / TRANSFORMER_HEADS;
    let mut output = vec![0.0; TRANSFORMER_WIDTH];
    for head in 0..TRANSFORMER_HEADS {
        let range = head * head_width..(head + 1) * head_width;
        let mut scores = cache
            .keys
            .iter()
            .map(|key| {
                query[range.clone()]
                    .iter()
                    .zip(&key[range.clone()])
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
                    / (head_width as f32).sqrt()
            })
            .collect::<Vec<_>>();
        softmax(&mut scores)?;
        for (score, value) in scores.iter().zip(&cache.values) {
            for index in range.clone() {
                output[index] += *score * value[index];
            }
        }
    }
    Ok(output)
}

fn transformer_sample(
    transformer: &AutoregressiveTransformer,
    tokens: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<bool>)> {
    let mut caches = (0..transformer.blocks.len())
        .map(|_| AttentionCache::default())
        .collect::<Vec<_>>();
    let mut ranks = Vec::with_capacity(tokens);
    let mut missing = Vec::with_capacity(tokens);
    let mut prior = [0.0f32, 0.0];
    for token in 0..tokens {
        let mut state = int8_linear(&transformer.input_projection, &prior)?;
        for (value, position) in state.iter_mut().zip(
            &transformer.positional_embeddings
                [token * TRANSFORMER_WIDTH..(token + 1) * TRANSFORMER_WIDTH],
        ) {
            *value += *position;
        }
        for (block, cache) in transformer.blocks.iter().zip(&mut caches) {
            let query = int8_linear(&block.query, &state)?;
            cache.keys.push(int8_linear(&block.key, &state)?);
            cache.values.push(int8_linear(&block.value, &state)?);
            let attention =
                int8_linear(&block.attention_output, &cached_attention(&query, cache)?)?;
            for (value, residual) in state.iter_mut().zip(attention) {
                *value += residual;
            }
            layer_normalize(&mut state, &block.attention_norm)?;
            let feed_forward = int8_linear(
                &block.feed_forward_2,
                &int8_linear(&block.feed_forward_1, &state)?
                    .into_iter()
                    .map(gelu)
                    .collect::<Vec<_>>(),
            )?;
            for (value, residual) in state.iter_mut().zip(feed_forward) {
                *value += residual;
            }
            layer_normalize(&mut state, &block.feed_forward_norm)?;
        }
        let parameters = int8_linear(&transformer.value_head, &state)?;
        let standard_deviation = parameters[1].clamp(-8.0, 2.0).exp().max(1e-3);
        let rank = parameters[0] + standard_deviation * normal(seed, token, 0x819a_02e7);
        let logit = int8_linear(&transformer.missing_head, &state)?[0];
        let probability = 1.0 / (1.0 + (-logit.clamp(-20.0, 20.0)).exp());
        let is_missing = uniform(seed, token, 0x4423_a715) <= probability;
        ranks.push(rank);
        missing.push(is_missing);
        // Training shifts the preceding Gaussian rank and missingness channel
        // into this token position, so inference must preserve that same
        // representation rather than converting the rank back to a CDF.
        prior = [rank, f32::from(is_missing)];
    }
    Ok((ranks, missing))
}

fn tabsyn_sample(
    decoder: &JointDecoder,
    denoiser: &LatentDenoiser,
    alpha_cumprod: &[f32],
    inference_timesteps: &[u8],
    seed: u64,
) -> Result<(Vec<f32>, Vec<bool>)> {
    let mut latent = (0..NEURAL_LATENT_WIDTH)
        .map(|index| normal(seed, index, 0x52d1_8b6c))
        .collect::<Vec<_>>();
    for (index, &encoded_timestep) in inference_timesteps.iter().enumerate() {
        let timestep = usize::from(encoded_timestep);
        let prediction = denoiser_forward(denoiser, &latent, timestep)?;
        let alpha = alpha_cumprod[timestep].clamp(1e-6, 1.0);
        let previous_alpha = inference_timesteps.get(index + 1).map_or(1.0, |step| {
            alpha_cumprod[usize::from(*step)].clamp(1e-6, 1.0)
        });
        for (value, noise) in latent.iter_mut().zip(prediction) {
            let clean = (*value - (1.0 - alpha).sqrt() * noise) / alpha.sqrt();
            *value = previous_alpha.sqrt() * clean + (1.0 - previous_alpha).sqrt() * noise;
        }
    }
    decode_rank_values(decoder, &latent, seed ^ 0xced1_159b)
}

#[inline]
pub fn normal_cdf(value: f32) -> f32 {
    let absolute = value.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * absolute);
    let polynomial = t
        * (0.319_381_53
            + t * (-0.356_563_78 + t * (1.781_478 + t * (-1.821_256 + t * 1.330_274_5))));
    let tail = 0.398_942_3 * (-0.5 * absolute * absolute).exp() * polynomial;
    if value >= 0.0 { 1.0 - tail } else { tail }
}

/// Samples all feature ranks plus the target rank. Returned feature positions
/// are restored to schema order even when the transformer uses a learned order.
pub fn sample_joint(
    generator: &JointGenerator,
    features: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<bool>)> {
    generator
        .validate(features, crate::model::Task::Regression)
        .or_else(|_| generator.validate(features, crate::model::Task::Binary))
        .map_err(DopeError::Codec)?;
    let tokens = features + 1;
    let (ordered_ranks, ordered_missing) = match &generator.network {
        JointNetwork::Tvae { decoder } => {
            let latent = (0..NEURAL_LATENT_WIDTH)
                .map(|index| normal(seed, index, 0x1cc5_19a7))
                .collect::<Vec<_>>();
            decode_rank_values(decoder, &latent, seed ^ 0xd8e4_915c)?
        }
        JointNetwork::MaskedAutoregressiveTransformer { transformer } => {
            transformer_sample(transformer, tokens, seed)?
        }
        JointNetwork::TabSyn {
            decoder,
            denoiser,
            alpha_cumprod,
            inference_timesteps,
        } => tabsyn_sample(decoder, denoiser, alpha_cumprod, inference_timesteps, seed)?,
    };
    if ordered_ranks.len() != tokens || ordered_missing.len() != tokens {
        return Err(DopeError::Codec(
            "neural sampler returned the wrong width".into(),
        ));
    }
    let mut ranks = vec![0.0; tokens];
    let mut missing = vec![false; tokens];
    for (ordered, &natural) in generator.feature_permutation.iter().enumerate() {
        ranks[natural as usize] = ordered_ranks[ordered];
        missing[natural as usize] = ordered_missing[ordered];
    }
    ranks[features] = ordered_ranks[features];
    // Targets are never emitted as missing, even if a corrupt training source
    // had a nonzero missingness head logit.
    missing[features] = false;
    Ok((ranks, missing))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{LoadedKernel, decode_kernel, encode_kernel};
    use crate::model::{
        AutoregressiveTransformer, ColumnSchema, JointGenerator, JointNetwork, Kernel,
        KernelProgram, LatentDenoiser, LayerNormParameters, Marginal, NeuralArchitecture,
        RankNormalization, SchemaKind, Task, Transform, TransformerBlock,
    };
    use crate::sample::{SampleOptions, sample_kernel};

    fn zero_linear(input: usize, output: usize, biases: Vec<f32>) -> QuantizedLinear {
        QuantizedLinear {
            input_dim: input as u32,
            output_dim: output as u32,
            weights: vec![0; input * output],
            scales: vec![1.0; output],
            biases,
        }
    }

    fn zero_decoder(tokens: usize) -> Box<JointDecoder> {
        let mut value_biases = Vec::with_capacity(tokens * 2);
        for _ in 0..tokens {
            value_biases.extend([0.0, -2.0]);
        }
        Box::new(JointDecoder {
            hidden_1: zero_linear(NEURAL_LATENT_WIDTH, 128, vec![0.0; 128]),
            hidden_2: zero_linear(128, 128, vec![0.0; 128]),
            value_head: zero_linear(128, tokens * 2, value_biases),
            missing_head: zero_linear(128, tokens, vec![-20.0; tokens]),
            variance_floor: 0.01,
        })
    }

    fn normalization(tokens: usize) -> Vec<RankNormalization> {
        vec![
            RankNormalization {
                location: 0.0,
                scale: 1.0,
                missing_probability: 0.0,
                randomized_discrete: true,
            };
            tokens
        ]
    }

    fn tvae_generator(features: usize) -> JointGenerator {
        let tokens = features + 1;
        JointGenerator {
            architecture: NeuralArchitecture::Tvae,
            feature_permutation: (0..features as u32).collect(),
            target_marginal: Marginal::Bernoulli { probability: 0.4 },
            normalization: normalization(tokens),
            network: JointNetwork::Tvae {
                decoder: zero_decoder(tokens),
            },
            training_hash: "a".repeat(64),
            implementation_hash: "b".repeat(64),
        }
    }

    #[test]
    fn int8_linear_uses_per_output_scales() {
        let layer = QuantizedLinear {
            input_dim: 2,
            output_dim: 2,
            weights: vec![2, -1, 3, 4],
            scales: vec![0.5, 0.25],
            biases: vec![1.0, -1.0],
        };
        assert_eq!(int8_linear(&layer, &[2.0, 4.0]).unwrap(), vec![1.0, 4.5]);
    }

    #[test]
    fn softmax_is_stable_and_normalized() {
        let mut values = vec![1_000.0, 1_001.0, 999.0];
        softmax(&mut values).unwrap();
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(values[1] > values[0] && values[0] > values[2]);
    }

    #[test]
    fn layer_normalization_is_bounded() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        layer_normalize(
            &mut values,
            &LayerNormParameters {
                weight: vec![1.0; 4],
                bias: vec![0.0; 4],
            },
        )
        .unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().sum::<f32>().abs() < 1e-5);
    }

    #[test]
    fn neural_section_round_trips_and_sampling_never_falls_back() {
        let kernel = Kernel {
            task: Task::Binary,
            rows_fitted: 64,
            features: 1,
            seed: 11,
            seed_policy: 0,
            quantization_bits: 8,
            compliant: false,
            schema: vec![ColumnSchema {
                kind: SchemaKind::Binary,
                missing_probability: 0.0,
                impute: 0.0,
                transform: Transform::Identity,
            }],
            marginals: vec![Marginal::Bernoulli { probability: 0.6 }],
            program: KernelProgram::NeuralJoint(tvae_generator(1)),
        };
        let encoded = encode_kernel(&kernel).unwrap();
        assert_eq!(&encoded[..6], b"DPK3\x03\x02");
        let decoded = decode_kernel(&encoded).unwrap();
        assert_eq!(decoded, kernel);
        let loaded = LoadedKernel::V3(Box::new(decoded));
        let first = sample_kernel(
            &loaded,
            SampleOptions {
                rows: 128,
                seed: Some(9),
            },
        )
        .unwrap();
        let second = sample_kernel(
            &loaded,
            SampleOptions {
                rows: 128,
                seed: Some(9),
            },
        )
        .unwrap();
        assert_eq!(first, second);
        assert!(first.iter().all(|value| matches!(*value, 0.0 | 1.0)));
    }

    #[test]
    fn architecture_payload_mismatch_is_rejected() {
        let mut generator = tvae_generator(2);
        generator.architecture = NeuralArchitecture::TabSyn;
        assert!(generator.validate(2, Task::Binary).is_err());
        generator.architecture = NeuralArchitecture::Tvae;
        generator.feature_permutation = vec![0, 0];
        assert!(generator.validate(2, Task::Binary).is_err());
    }

    #[test]
    fn transformer_and_ddim_sampling_are_deterministic_and_finite() {
        let features = 2;
        let tokens = features + 1;
        let norm = LayerNormParameters {
            weight: vec![1.0; TRANSFORMER_WIDTH],
            bias: vec![0.0; TRANSFORMER_WIDTH],
        };
        let block = || TransformerBlock {
            query: zero_linear(64, 64, vec![0.0; 64]),
            key: zero_linear(64, 64, vec![0.0; 64]),
            value: zero_linear(64, 64, vec![0.0; 64]),
            attention_output: zero_linear(64, 64, vec![0.0; 64]),
            attention_norm: norm.clone(),
            feed_forward_1: zero_linear(64, 128, vec![0.0; 128]),
            feed_forward_2: zero_linear(128, 64, vec![0.0; 64]),
            feed_forward_norm: norm.clone(),
        };
        let transformer = JointGenerator {
            architecture: NeuralArchitecture::MaskedAutoregressiveTransformer,
            feature_permutation: vec![1, 0],
            target_marginal: Marginal::Bernoulli { probability: 0.5 },
            normalization: normalization(tokens),
            network: JointNetwork::MaskedAutoregressiveTransformer {
                transformer: Box::new(AutoregressiveTransformer {
                    input_projection: zero_linear(2, 64, vec![0.0; 64]),
                    positional_embeddings: vec![0.0; tokens * 64],
                    blocks: vec![block(), block()],
                    value_head: zero_linear(64, 2, vec![0.0, -2.0]),
                    missing_head: zero_linear(64, 1, vec![-20.0]),
                }),
            },
            training_hash: "c".repeat(64),
            implementation_hash: "d".repeat(64),
        };
        let schedule = (0..100)
            .map(|step| 1.0 - (step + 1) as f32 / 101.0)
            .collect::<Vec<_>>();
        let tabsyn = JointGenerator {
            architecture: NeuralArchitecture::TabSyn,
            feature_permutation: vec![0, 1],
            target_marginal: Marginal::Bernoulli { probability: 0.5 },
            normalization: normalization(tokens),
            network: JointNetwork::TabSyn {
                decoder: zero_decoder(tokens),
                denoiser: Box::new(LatentDenoiser {
                    hidden_1: zero_linear(64, 128, vec![0.0; 128]),
                    hidden_2: zero_linear(128, 128, vec![0.0; 128]),
                    hidden_3: zero_linear(128, 128, vec![0.0; 128]),
                    output: zero_linear(128, 32, vec![0.0; 32]),
                }),
                alpha_cumprod: schedule,
                inference_timesteps: (0..32).map(|index| (99 - index * 99 / 31) as u8).collect(),
            },
            training_hash: "e".repeat(64),
            implementation_hash: "f".repeat(64),
        };
        for generator in [transformer, tabsyn] {
            let first = sample_joint(&generator, features, 123).unwrap();
            let second = sample_joint(&generator, features, 123).unwrap();
            assert_eq!(first, second);
            assert!(first.0.iter().all(|value| value.is_finite()));
            assert!(first.1.iter().all(|missing| !missing));
        }
    }
}

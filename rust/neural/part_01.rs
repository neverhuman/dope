use crate::error::{DopeError, Result};
use crate::model::{
    AutoregressiveTransformer, JointDecoder, JointGenerator, JointNetwork, LatentDenoiser,
    LayerNormParameters, NEURAL_LATENT_WIDTH, QuantizedLinear,
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
    let mut input = Vec::with_capacity(latent.len() + NEURAL_LATENT_WIDTH);
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

fn cached_attention(
    query: &[f32],
    cache: &AttentionCache,
    width: usize,
    heads: usize,
) -> Result<Vec<f32>> {
    if query.len() != width || cache.keys.is_empty() || cache.keys.len() != cache.values.len() {
        return Err(DopeError::Codec(
            "invalid autoregressive attention cache".into(),
        ));
    }
    let head_width = width / heads;
    let mut output = vec![0.0; width];
    for head in 0..heads {
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
    width: usize,
    heads: usize,
) -> Result<(Vec<f32>, Vec<bool>)> {
    let mut caches = (0..transformer.blocks.len())
        .map(|_| AttentionCache::default())
        .collect::<Vec<_>>();
    let mut ranks = Vec::with_capacity(tokens);
    let mut missing = Vec::with_capacity(tokens);
    let mut prior = [0.0f32, 0.0];
    for token in 0..tokens {
        let mut state = int8_linear(&transformer.input_projection, &prior)?;
        for (value, position) in state
            .iter_mut()
            .zip(&transformer.positional_embeddings[token * width..(token + 1) * width])
        {
            *value += *position;
        }
        for (block, cache) in transformer.blocks.iter().zip(&mut caches) {
            let query = int8_linear(&block.query, &state)?;
            cache.keys.push(int8_linear(&block.key, &state)?);
            cache.values.push(int8_linear(&block.value, &state)?);
            let attention = int8_linear(
                &block.attention_output,
                &cached_attention(&query, cache, width, heads)?,
            )?;
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
    let latent = diffusion_sample(
        denoiser,
        alpha_cumprod,
        inference_timesteps,
        NEURAL_LATENT_WIDTH,
        seed,
    )?;
    decode_rank_values(decoder, &latent, seed ^ 0xced1_159b)
}

fn diffusion_sample(
    denoiser: &LatentDenoiser,
    alpha_cumprod: &[f32],
    inference_timesteps: &[u8],
    width: usize,
    seed: u64,
) -> Result<Vec<f32>> {
    let mut latent = (0..width)
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
    Ok(latent)
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
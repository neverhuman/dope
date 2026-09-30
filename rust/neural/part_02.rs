

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
            let latent_width = generator
                .profile
                .tvae_dimensions()
                .ok_or_else(|| DopeError::Codec("invalid TVAE profile".into()))?
                .0;
            let latent = (0..latent_width)
                .map(|index| normal(seed, index, 0x1cc5_19a7))
                .collect::<Vec<_>>();
            decode_rank_values(decoder, &latent, seed ^ 0xd8e4_915c)?
        }
        JointNetwork::MaskedAutoregressiveTransformer { transformer } => {
            let (width, heads, _) = generator
                .profile
                .transformer_dimensions()
                .ok_or_else(|| DopeError::Codec("invalid transformer profile".into()))?;
            transformer_sample(transformer, tokens, seed, width, heads)?
        }
        JointNetwork::TabSyn {
            decoder,
            denoiser,
            alpha_cumprod,
            inference_timesteps,
        } => tabsyn_sample(decoder, denoiser, alpha_cumprod, inference_timesteps, seed)?,
        JointNetwork::TabDdpm {
            denoiser,
            alpha_cumprod,
            inference_timesteps,
        } => {
            let ranks =
                diffusion_sample(denoiser, alpha_cumprod, inference_timesteps, tokens, seed)?;
            let missing = generator
                .normalization
                .iter()
                .enumerate()
                .map(|(index, norm)| uniform(seed, index, 0xa737_812e) < norm.missing_probability)
                .collect();
            (ranks, missing)
        }
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
        let observed_rate = generator.normalization[natural as usize].missing_probability;
        missing[natural as usize] = if observed_rate == 0.0 {
            false
        } else if observed_rate == 1.0 {
            true
        } else {
            ordered_missing[ordered]
        };
    }
    ranks[features] = ordered_ranks[features];
    // Targets are never emitted as missing, even if a corrupt training source
    // had a nonzero missingness head logit.
    missing[features] = false;
    Ok((ranks, missing))
}

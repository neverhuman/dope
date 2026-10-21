//! Additive deployment seam. Historical `sample_joint` and GELU stay unchanged.
//! Tensor runtimes supply decoder heads; native code owns randomness and support.
use super::*;

/// PyTorch's `gelu("tanh")` approximation, distinct from historical DPK behavior.
pub fn gelu_tanh(value: f32) -> f32 {
    0.5 * value
        * (1.0
            + (std::f32::consts::FRAC_2_PI.sqrt() * (value + 0.044_715 * value * value * value))
                .tanh())
}

fn tvae(generator: &JointGenerator, features: usize) -> Result<&JointDecoder> {
    generator
        .validate(features, crate::model::Task::Regression)
        .or_else(|_| generator.validate(features, crate::model::Task::Binary))
        .map_err(DopeError::Codec)?;
    match &generator.network {
        JointNetwork::Tvae { decoder } => Ok(decoder),
        _ => Err(DopeError::Unsupported("deployment_requires_tvae".into())),
    }
}

/// One deterministic latent row. Row seeds come from `sample::joint_row_seeds`.
pub fn tvae_latent(generator: &JointGenerator, features: usize, seed: u64) -> Result<Vec<f32>> {
    let decoder = tvae(generator, features)?;
    Ok((0..decoder.hidden_1.input_dim as usize)
        .map(|index| normal(seed, index, 0x1cc5_19a7))
        .collect())
}

/// Finish externally computed decoder heads using the original random streams,
/// feature permutation, observed support and target-missingness rules.
pub fn finish_tvae(
    generator: &JointGenerator,
    features: usize,
    seed: u64,
    parameters: &[f32],
    missing_logits: &[f32],
) -> Result<(Vec<f32>, Vec<bool>)> {
    let decoder = tvae(generator, features)?;
    if parameters.len() != 2 * (features + 1)
        || missing_logits.len() != features + 1
        || parameters
            .iter()
            .chain(missing_logits)
            .any(|x| !x.is_finite())
    {
        return Err(DopeError::Codec("deployment_head_invalid".into()));
    }
    let (ordered_ranks, ordered_missing) = decode_rank_parameters(
        decoder.variance_floor,
        parameters,
        missing_logits,
        seed ^ 0xd8e4_915c,
    )?;
    restore_joint_order(generator, features, ordered_ranks, ordered_missing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tanh_gelu_matches_independent_training_values_and_preserves_legacy() {
        // Independently calculated with Python math, sqrt(2/pi), tanh.
        for (x, expected) in [
            (-2.0, -0.045_402_307),
            (-1.0, -0.158_808),
            (0.0, 0.0),
            (1.0, 0.841_192),
            (2.0, 1.954_597_7),
        ] {
            assert!((gelu_tanh(x) - expected).abs() < 3e-7);
        }
        assert!((gelu(1.0) - gelu_tanh(1.0)).abs() > 0.05);
    }
}

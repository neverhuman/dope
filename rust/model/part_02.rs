

impl TransformerBlock {
    fn validate(&self, width: usize, ff_width: usize) -> std::result::Result<(), String> {
        for linear in [
            &self.query,
            &self.key,
            &self.value,
            &self.attention_output,
            &self.feed_forward_1,
            &self.feed_forward_2,
        ] {
            linear.validate()?;
        }
        if [&self.query, &self.key, &self.value, &self.attention_output]
            .iter()
            .any(|linear| linear.input_dim as usize != width || linear.output_dim as usize != width)
            || self.feed_forward_1.input_dim as usize != width
            || self.feed_forward_1.output_dim as usize != ff_width
            || self.feed_forward_2.input_dim as usize != ff_width
            || self.feed_forward_2.output_dim as usize != width
            || !self.attention_norm.validate(width)
            || !self.feed_forward_norm.validate(width)
        {
            return Err("transformer block violates the frozen architecture".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutoregressiveTransformer {
    /// Projects the previously sampled rank and missingness indicator.
    pub input_projection: QuantizedLinear,
    pub positional_embeddings: Vec<f32>,
    pub blocks: Vec<TransformerBlock>,
    pub value_head: QuantizedLinear,
    pub missing_head: QuantizedLinear,
}

impl AutoregressiveTransformer {
    fn validate(
        &self,
        tokens: usize,
        width: usize,
        ff_width: usize,
    ) -> std::result::Result<(), String> {
        self.input_projection.validate()?;
        self.value_head.validate()?;
        self.missing_head.validate()?;
        for block in &self.blocks {
            block.validate(width, ff_width)?;
        }
        if self.input_projection.input_dim != 2
            || self.input_projection.output_dim as usize != width
            || self.positional_embeddings.len() != tokens * width
            || self
                .positional_embeddings
                .iter()
                .any(|value| !value.is_finite())
            || self.blocks.len() != 2
            || self.value_head.input_dim as usize != width
            || self.value_head.output_dim != 2
            || self.missing_head.input_dim as usize != width
            || self.missing_head.output_dim != 1
        {
            return Err("autoregressive transformer violates the frozen architecture".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LatentDenoiser {
    pub hidden_1: QuantizedLinear,
    pub hidden_2: QuantizedLinear,
    pub hidden_3: QuantizedLinear,
    pub output: QuantizedLinear,
}

impl LatentDenoiser {
    fn validate(&self) -> std::result::Result<(), String> {
        self.validate_dimensions(NEURAL_LATENT_WIDTH * 2, NEURAL_LATENT_WIDTH)
    }

    fn validate_dimensions(&self, input: usize, output: usize) -> std::result::Result<(), String> {
        for layer in [&self.hidden_1, &self.hidden_2, &self.hidden_3, &self.output] {
            layer.validate()?;
        }
        if self.hidden_1.input_dim as usize != input
            || self.hidden_1.output_dim as usize != NEURAL_HIDDEN_WIDTH
            || self.hidden_2.input_dim as usize != NEURAL_HIDDEN_WIDTH
            || self.hidden_2.output_dim as usize != NEURAL_HIDDEN_WIDTH
            || self.hidden_3.input_dim as usize != NEURAL_HIDDEN_WIDTH
            || self.hidden_3.output_dim as usize != NEURAL_HIDDEN_WIDTH
            || self.output.input_dim as usize != NEURAL_HIDDEN_WIDTH
            || self.output.output_dim as usize != output
        {
            return Err("latent denoiser violates the frozen architecture".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JointNetwork {
    Tvae {
        decoder: Box<JointDecoder>,
    },
    MaskedAutoregressiveTransformer {
        transformer: Box<AutoregressiveTransformer>,
    },
    TabSyn {
        decoder: Box<JointDecoder>,
        denoiser: Box<LatentDenoiser>,
        alpha_cumprod: Vec<f32>,
        inference_timesteps: Vec<u8>,
    },
    TabDdpm {
        denoiser: Box<LatentDenoiser>,
        alpha_cumprod: Vec<f32>,
        inference_timesteps: Vec<u8>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankNormalization {
    pub location: f32,
    pub scale: f32,
    pub missing_probability: f32,
    pub randomized_discrete: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointGenerator {
    pub architecture: NeuralArchitecture,
    #[serde(default, skip_serializing_if = "NeuralProfile::is_full")]
    pub profile: NeuralProfile,
    /// Feature positions in generation order. The target is implicit and last.
    pub feature_permutation: Vec<u32>,
    pub target_marginal: Marginal,
    /// Feature entries followed by the target entry.
    pub normalization: Vec<RankNormalization>,
    pub network: JointNetwork,
    pub training_hash: String,
    pub implementation_hash: String,
}

include!("generator_validation.rs");

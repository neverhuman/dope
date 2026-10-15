

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

impl QuantizedLinear {
    pub fn encoded_number_count(&self) -> usize {
        self.weights.len() + self.scales.len() + self.biases.len()
    }
}

impl LayerNormParameters {
    pub fn encoded_number_count(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}

impl JointDecoder {
    pub fn encoded_number_count(&self) -> usize {
        self.hidden_1.encoded_number_count()
            + self.hidden_2.encoded_number_count()
            + self.value_head.encoded_number_count()
            + self.missing_head.encoded_number_count()
            + 1
    }
}

impl TransformerBlock {
    pub fn encoded_number_count(&self) -> usize {
        self.query.encoded_number_count()
            + self.key.encoded_number_count()
            + self.value.encoded_number_count()
            + self.attention_output.encoded_number_count()
            + self.attention_norm.encoded_number_count()
            + self.feed_forward_1.encoded_number_count()
            + self.feed_forward_2.encoded_number_count()
            + self.feed_forward_norm.encoded_number_count()
    }
}

impl AutoregressiveTransformer {
    pub fn encoded_number_count(&self) -> usize {
        self.input_projection.encoded_number_count()
            + self.positional_embeddings.len()
            + self
                .blocks
                .iter()
                .map(TransformerBlock::encoded_number_count)
                .sum::<usize>()
            + self.value_head.encoded_number_count()
            + self.missing_head.encoded_number_count()
    }
}

impl LatentDenoiser {
    pub fn encoded_number_count(&self) -> usize {
        self.hidden_1.encoded_number_count()
            + self.hidden_2.encoded_number_count()
            + self.hidden_3.encoded_number_count()
            + self.output.encoded_number_count()
    }
}

impl JointNetwork {
    pub fn encoded_number_count(&self) -> usize {
        match self {
            Self::Tvae { decoder } => decoder.encoded_number_count(),
            Self::MaskedAutoregressiveTransformer { transformer } => {
                transformer.encoded_number_count()
            }
            Self::TabSyn {
                decoder,
                denoiser,
                alpha_cumprod,
                inference_timesteps,
            } => {
                decoder.encoded_number_count()
                    + denoiser.encoded_number_count()
                    + alpha_cumprod.len()
                    + inference_timesteps.len()
            }
            Self::TabDdpm {
                denoiser,
                alpha_cumprod,
                inference_timesteps,
            } => {
                denoiser.encoded_number_count()
                    + alpha_cumprod.len()
                    + inference_timesteps.len()
            }
        }
    }
}

impl Marginal {
    pub fn encoded_number_count(&self) -> usize {
        match self {
            Self::Constant { .. } | Self::Bernoulli { .. } => 1,
            Self::Beta { .. } | Self::Gaussian { .. } => 2,
            Self::Grid {
                values,
                probabilities,
            } => values.len() + probabilities.len(),
            Self::QuantileSpline { values } => values.len(),
            Self::Histogram {
                edges,
                probabilities,
            } => edges.len() + probabilities.len(),
            Self::ZeroInflated { base, .. } => 2 + base.encoded_number_count(),
        }
    }
}

impl Dependence {
    pub fn encoded_number_count(&self) -> usize {
        match self {
            Self::Independent => 0,
            Self::ChowLiu { edges, .. } => 1 + edges.len() * 3,
            Self::GaussianCopula { cholesky } => cholesky.len(),
            Self::SparseGraph { edges } => edges.len() * 3,
            Self::Poet {
                loadings,
                residual_edges,
                ..
            } => 1 + loadings.len() * 3 + residual_edges.len() * 3,
            Self::Vine { edges } => edges.len() * 4,
            Self::Mixture {
                weights,
                components,
            } => {
                weights.len()
                    + components
                        .iter()
                        .map(Self::encoded_number_count)
                        .sum::<usize>()
            }
            Self::Triangular { terms } => terms.len() * 3,
        }
    }
}

impl Target {
    pub fn encoded_number_count(&self) -> usize {
        match self {
            Self::SparseLinear { terms, .. } | Self::SparseLogistic { terms, .. } => {
                1 + terms.len() * 2
            }
            Self::SparseGam { terms, .. } => {
                1 + terms
                    .iter()
                    .map(|term| 1 + term.knots.len() + term.values.len())
                    .sum::<usize>()
            }
            Self::Ga2m {
                main_terms,
                interactions,
                ..
            } => {
                1 + main_terms
                    .iter()
                    .map(|term| 1 + term.knots.len() + term.values.len())
                    .sum::<usize>()
                    + interactions
                        .iter()
                        .map(|term| {
                            2 + term.left_knots.len()
                                + term.right_knots.len()
                                + term.values.len()
                        })
                        .sum::<usize>()
            }
            Self::Mars { terms, .. } => {
                1 + terms
                    .iter()
                    .map(|term| 1 + term.factors.len() * 3)
                    .sum::<usize>()
            }
            Self::ObliviousTree {
                features,
                thresholds,
                leaves,
                ..
            } => features.len() + thresholds.len() + leaves.len(),
            Self::CompactNeuralResidual {
                linear_terms,
                hidden_features,
                input_weights,
                hidden_biases,
                output_weights,
                ..
            } => {
                1 + linear_terms.len() * 2
                    + hidden_features.len()
                    + input_weights.len()
                    + hidden_biases.len()
                    + output_weights.len()
            }
        }
    }
}

impl Noise {
    pub fn encoded_number_count(&self) -> usize {
        match self {
            Self::Homoscedastic { .. } | Self::BernoulliCalibration { .. } => 1,
            Self::Heteroscedastic { terms, .. } => 3 + terms.len() * 2,
            Self::IsotonicCalibration {
                knots,
                probabilities,
            } => knots.len() + probabilities.len(),
            Self::ResidualV1 { distribution } => match distribution {
                ResidualV1::NoNoise => 0,
                ResidualV1::StudentT4 { .. } => 2,
                ResidualV1::GaussianMixture2 { .. } => 5,
                ResidualV1::MonotoneQuantile17 { values } => values.len(),
            },
        }
    }
}

impl Kernel {
    pub fn encoded_number_count(&self) -> usize {
        let schema_numbers: usize = self
            .schema
            .iter()
            .map(|col| {
                2 + match col.transform {
                    Transform::Identity => 0,
                    Transform::Log1p { .. }
                    | Transform::Power { .. }
                    | Transform::Logit { .. } => 1,
                    Transform::Winsorize { .. } => 2,
                }
            })
            .sum();
        let marginal_numbers: usize = self
            .marginals
            .iter()
            .map(Marginal::encoded_number_count)
            .sum();
        let program_numbers = match &self.program {
            KernelProgram::Symbolic {
                dependence,
                target,
                noise,
            } => {
                dependence.encoded_number_count()
                    + target.encoded_number_count()
                    + noise.encoded_number_count()
            }
            KernelProgram::NeuralJoint(joint) => {
                joint.feature_permutation.len()
                    + joint.target_marginal.encoded_number_count()
                    + joint.normalization.len() * 3
                    + joint.network.encoded_number_count()
            }
        };
        (schema_numbers + marginal_numbers + program_numbers).max(1)
    }
}

include!("generator_validation.rs");

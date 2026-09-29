use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Regression,
    Binary,
}

impl Task {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "regression" | "reg" => Some(Self::Regression),
            "binary" | "bin" | "classification" | "clf" => Some(Self::Binary),
            _ => None,
        }
    }

    pub const fn code(self) -> u8 {
        match self {
            Self::Regression => 0,
            Self::Binary => 1,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Regression => "regression",
            Self::Binary => "binary",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemaKind {
    Constant,
    Binary,
    Ordinal,
    CategoricalGrid,
    CountLike,
    Continuous,
    Inflated,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Transform {
    Identity,
    Log1p { scale: f32 },
    Power { exponent: f32 },
    Logit { epsilon: f32 },
    Winsorize { lower: f32, upper: f32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColumnSchema {
    pub kind: SchemaKind,
    pub missing_probability: f32,
    pub impute: f32,
    pub transform: Transform,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Marginal {
    Constant {
        value: f32,
    },
    Bernoulli {
        probability: f32,
    },
    Grid {
        values: Vec<f32>,
        probabilities: Vec<f32>,
    },
    QuantileSpline {
        values: Vec<f32>,
    },
    Beta {
        alpha: f32,
        beta: f32,
    },
    Gaussian {
        mean: f32,
        sigma: f32,
    },
    Histogram {
        edges: Vec<f32>,
        probabilities: Vec<f32>,
    },
    ZeroInflated {
        point: f32,
        probability: f32,
        base: Box<Marginal>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CopulaEdge {
    pub parent: u32,
    pub child: u32,
    pub correlation: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FactorLoading {
    pub feature: u32,
    pub factor: u8,
    pub loading: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VineEdge {
    pub left: u32,
    pub right: u32,
    pub conditioning_depth: u8,
    pub parameter: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriangularTerm {
    pub parent: u32,
    pub child: u32,
    pub coefficient: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Dependence {
    Independent,
    ChowLiu {
        root: u32,
        edges: Vec<CopulaEdge>,
    },
    GaussianCopula {
        cholesky: Vec<f32>,
    },
    SparseGraph {
        edges: Vec<CopulaEdge>,
    },
    Poet {
        rank: u8,
        loadings: Vec<FactorLoading>,
        residual_edges: Vec<CopulaEdge>,
    },
    Vine {
        edges: Vec<VineEdge>,
    },
    Mixture {
        weights: Vec<f32>,
        components: Vec<Dependence>,
    },
    Triangular {
        terms: Vec<TriangularTerm>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearTerm {
    pub feature: u32,
    pub coefficient: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdditiveTerm {
    pub feature: u32,
    pub knots: Vec<f32>,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InteractionTerm {
    pub left: u32,
    pub right: u32,
    pub left_knots: Vec<f32>,
    pub right_knots: Vec<f32>,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarsFactor {
    pub feature: u32,
    pub knot: f32,
    pub direction: i8,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarsTerm {
    pub coefficient: f32,
    pub factors: Vec<MarsFactor>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Target {
    SparseLinear {
        intercept: f32,
        terms: Vec<LinearTerm>,
    },
    SparseLogistic {
        intercept: f32,
        terms: Vec<LinearTerm>,
    },
    SparseGam {
        intercept: f32,
        logistic: bool,
        terms: Vec<AdditiveTerm>,
    },
    Ga2m {
        intercept: f32,
        logistic: bool,
        main_terms: Vec<AdditiveTerm>,
        interactions: Vec<InteractionTerm>,
    },
    Mars {
        intercept: f32,
        logistic: bool,
        terms: Vec<MarsTerm>,
    },
    ObliviousTree {
        logistic: bool,
        features: Vec<u32>,
        thresholds: Vec<f32>,
        leaves: Vec<f32>,
    },
    CompactNeuralResidual {
        intercept: f32,
        logistic: bool,
        linear_terms: Vec<LinearTerm>,
        hidden_features: Vec<u32>,
        hidden_width: u8,
        input_weights: Vec<f32>,
        hidden_biases: Vec<f32>,
        output_weights: Vec<f32>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Noise {
    Homoscedastic {
        sigma: f32,
    },
    BernoulliCalibration {
        base_rate: f32,
    },
    Heteroscedastic {
        intercept: f32,
        terms: Vec<LinearTerm>,
        minimum_sigma: f32,
        maximum_sigma: f32,
    },
    IsotonicCalibration {
        knots: Vec<f32>,
        probabilities: Vec<f32>,
    },
}

pub const NEURAL_LATENT_WIDTH: usize = 32;
pub const NEURAL_HIDDEN_WIDTH: usize = 128;
pub const TRANSFORMER_WIDTH: usize = 64;
pub const TRANSFORMER_HEADS: usize = 4;
pub const TRANSFORMER_FF_WIDTH: usize = 128;
pub const DIFFUSION_TRAIN_STEPS: usize = 100;
pub const DIFFUSION_INFERENCE_STEPS: usize = 32;
pub const MAX_NEURAL_TENSOR_ELEMENTS: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuralArchitecture {
    Tvae,
    MaskedAutoregressiveTransformer,
    TabSyn,
    TabDdpm,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuralProfile {
    #[default]
    Full,
    MicroTvae4,
    MicroTvae8,
    MicroTvae12,
    TinyMat16,
    TinyMat24,
}

impl NeuralProfile {
    pub fn tvae_dimensions(self) -> Option<(usize, usize)> {
        match self {
            Self::Full => Some((NEURAL_LATENT_WIDTH, NEURAL_HIDDEN_WIDTH)),
            Self::MicroTvae4 => Some((4, 16)),
            Self::MicroTvae8 => Some((8, 24)),
            Self::MicroTvae12 => Some((12, 32)),
            Self::TinyMat16 | Self::TinyMat24 => None,
        }
    }

    pub fn transformer_dimensions(self) -> Option<(usize, usize, usize)> {
        match self {
            Self::Full => Some((TRANSFORMER_WIDTH, TRANSFORMER_HEADS, TRANSFORMER_FF_WIDTH)),
            Self::TinyMat16 => Some((16, 2, 32)),
            Self::TinyMat24 => Some((24, 3, 48)),
            _ => None,
        }
    }

    pub fn is_full(&self) -> bool {
        *self == Self::Full
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantizedLinear {
    pub input_dim: u32,
    pub output_dim: u32,
    /// Row-major, per-output-channel symmetric int8 weights.
    pub weights: Vec<i8>,
    pub scales: Vec<f32>,
    pub biases: Vec<f32>,
}

impl QuantizedLinear {
    pub fn validate(&self) -> std::result::Result<(), String> {
        let input = self.input_dim as usize;
        let output = self.output_dim as usize;
        if input == 0
            || output == 0
            || input.saturating_mul(output) > MAX_NEURAL_TENSOR_ELEMENTS
            || self.weights.len() != input * output
            || self.scales.len() != output
            || self.biases.len() != output
            || self
                .scales
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || self.biases.iter().any(|value| !value.is_finite())
        {
            return Err("invalid quantized linear tensor".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerNormParameters {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl LayerNormParameters {
    fn validate(&self, width: usize) -> bool {
        self.weight.len() == width
            && self.bias.len() == width
            && self.weight.iter().all(|value| value.is_finite())
            && self.bias.iter().all(|value| value.is_finite())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointDecoder {
    pub hidden_1: QuantizedLinear,
    pub hidden_2: QuantizedLinear,
    /// Two outputs per token: rank-space location and log scale.
    pub value_head: QuantizedLinear,
    pub missing_head: QuantizedLinear,
    pub variance_floor: f32,
}

impl JointDecoder {
    fn validate(
        &self,
        tokens: usize,
        latent: usize,
        hidden: usize,
    ) -> std::result::Result<(), String> {
        self.hidden_1.validate()?;
        self.hidden_2.validate()?;
        self.value_head.validate()?;
        self.missing_head.validate()?;
        if self.hidden_1.input_dim as usize != latent
            || self.hidden_1.output_dim as usize != hidden
            || self.hidden_2.input_dim as usize != hidden
            || self.hidden_2.output_dim as usize != hidden
            || self.value_head.input_dim as usize != hidden
            || self.value_head.output_dim as usize != tokens * 2
            || self.missing_head.input_dim as usize != hidden
            || self.missing_head.output_dim as usize != tokens
            || !self.variance_floor.is_finite()
            || !(1e-4..=0.5).contains(&self.variance_floor)
        {
            return Err("joint decoder violates the frozen architecture".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransformerBlock {
    pub query: QuantizedLinear,
    pub key: QuantizedLinear,
    pub value: QuantizedLinear,
    pub attention_output: QuantizedLinear,
    pub attention_norm: LayerNormParameters,
    pub feed_forward_1: QuantizedLinear,
    pub feed_forward_2: QuantizedLinear,
    pub feed_forward_norm: LayerNormParameters,
}

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

impl JointGenerator {
    pub fn validate(&self, features: usize, task: Task) -> std::result::Result<(), String> {
        let tokens = features + 1;
        if self.feature_permutation.len() != features {
            return Err("neural feature permutation width mismatch".into());
        }
        let mut permutation = self.feature_permutation.clone();
        permutation.sort_unstable();
        if permutation != (0..features as u32).collect::<Vec<_>>() {
            return Err("neural feature permutation is not bijective".into());
        }
        validate_marginal(&self.target_marginal, 0)?;
        if task == Task::Binary && !matches!(self.target_marginal, Marginal::Bernoulli { .. }) {
            return Err("binary neural target requires a Bernoulli marginal".into());
        }
        if self.normalization.len() != tokens
            || self.normalization.iter().any(|normalization| {
                !normalization.location.is_finite()
                    || !normalization.scale.is_finite()
                    || normalization.scale <= 0.0
                    || !valid_probability(normalization.missing_probability)
            })
            || !valid_hash(&self.training_hash)
            || !valid_hash(&self.implementation_hash)
        {
            return Err("invalid neural normalization or provenance".into());
        }
        match (&self.architecture, &self.network) {
            (NeuralArchitecture::Tvae, JointNetwork::Tvae { decoder }) => {
                let (latent, hidden) = self
                    .profile
                    .tvae_dimensions()
                    .ok_or("TVAE profile is incompatible with its architecture")?;
                decoder.validate(tokens, latent, hidden)
            }
            (
                NeuralArchitecture::MaskedAutoregressiveTransformer,
                JointNetwork::MaskedAutoregressiveTransformer { transformer },
            ) => {
                let (width, _, ff_width) = self
                    .profile
                    .transformer_dimensions()
                    .ok_or("transformer profile is incompatible with its architecture")?;
                transformer.validate(tokens, width, ff_width)
            }
            (
                NeuralArchitecture::TabSyn,
                JointNetwork::TabSyn {
                    decoder,
                    denoiser,
                    alpha_cumprod,
                    inference_timesteps,
                },
            ) => {
                if self.profile != NeuralProfile::Full {
                    return Err("TabSyn uses the full profile".into());
                }
                decoder.validate(tokens, NEURAL_LATENT_WIDTH, NEURAL_HIDDEN_WIDTH)?;
                denoiser.validate()?;
                if alpha_cumprod.len() != DIFFUSION_TRAIN_STEPS
                    || alpha_cumprod
                        .iter()
                        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
                    || !alpha_cumprod.windows(2).all(|pair| pair[0] >= pair[1])
                    || inference_timesteps.len() != DIFFUSION_INFERENCE_STEPS
                    || !inference_timesteps.windows(2).all(|pair| pair[0] > pair[1])
                    || inference_timesteps
                        .first()
                        .is_none_or(|step| usize::from(*step) >= DIFFUSION_TRAIN_STEPS)
                {
                    return Err("invalid TabSyn diffusion schedule".into());
                }
                Ok(())
            }
            (
                NeuralArchitecture::TabDdpm,
                JointNetwork::TabDdpm {
                    denoiser,
                    alpha_cumprod,
                    inference_timesteps,
                },
            ) => {
                if self.profile != NeuralProfile::Full {
                    return Err("TabDDPM uses the full profile".into());
                }
                denoiser.validate_dimensions(tokens + NEURAL_LATENT_WIDTH, tokens)?;
                validate_diffusion_schedule(alpha_cumprod, inference_timesteps)
            }
            _ => Err("neural architecture tag and tensor payload disagree".into()),
        }
    }
}

fn validate_diffusion_schedule(
    alpha_cumprod: &[f32],
    inference_timesteps: &[u8],
) -> std::result::Result<(), String> {
    if alpha_cumprod.len() != DIFFUSION_TRAIN_STEPS
        || alpha_cumprod
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || !alpha_cumprod.windows(2).all(|pair| pair[0] >= pair[1])
        || inference_timesteps.len() != DIFFUSION_INFERENCE_STEPS
        || !inference_timesteps.windows(2).all(|pair| pair[0] > pair[1])
        || inference_timesteps
            .first()
            .is_none_or(|step| usize::from(*step) >= DIFFUSION_TRAIN_STEPS)
    {
        return Err("invalid diffusion schedule".into());
    }
    Ok(())
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KernelProgram {
    Symbolic {
        dependence: Dependence,
        target: Target,
        noise: Noise,
    },
    NeuralJoint(JointGenerator),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Kernel {
    pub task: Task,
    pub rows_fitted: u64,
    pub features: u32,
    pub seed: u64,
    pub seed_policy: u8,
    pub quantization_bits: u8,
    pub compliant: bool,
    pub schema: Vec<ColumnSchema>,
    pub marginals: Vec<Marginal>,
    pub program: KernelProgram,
}

impl Kernel {
    pub fn validate(&self) -> std::result::Result<(), String> {
        let width = self.features as usize;
        if width == 0 {
            return Err("a data kernel requires at least one positional feature".into());
        }
        if self.schema.len() != width || self.marginals.len() != width {
            return Err("schema and marginal counts must equal positional feature count".into());
        }
        if !matches!(self.quantization_bits, 4 | 6 | 8 | 10 | 12 | 16) {
            return Err("unsupported quantization profile".into());
        }
        for schema in &self.schema {
            if !(0.0..=1.0).contains(&schema.missing_probability) || !schema.impute.is_finite() {
                return Err("invalid schema parameter".into());
            }
            match schema.transform {
                Transform::Identity => {}
                Transform::Log1p { scale } if scale.is_finite() && scale > 0.0 => {}
                Transform::Power { exponent } if exponent.is_finite() && exponent > 0.0 => {}
                Transform::Logit { epsilon } if (0.0..0.5).contains(&epsilon) => {}
                Transform::Winsorize { lower, upper }
                    if lower.is_finite()
                        && upper.is_finite()
                        && 0.0 <= lower
                        && lower < upper
                        && upper <= 1.0 => {}
                _ => return Err("invalid bounded transform".into()),
            }
        }
        for marginal in &self.marginals {
            validate_marginal(marginal, 0)?;
        }
        match &self.program {
            KernelProgram::Symbolic {
                dependence,
                target,
                noise,
            } => {
                validate_dependence(dependence, width, 0)?;
                validate_target(target, width)?;
                validate_noise(noise, width)?;
            }
            KernelProgram::NeuralJoint(generator) => generator.validate(width, self.task)?,
        }
        Ok(())
    }

    pub fn symbolic(&self) -> Option<(&Dependence, &Target, &Noise)> {
        match &self.program {
            KernelProgram::Symbolic {
                dependence,
                target,
                noise,
            } => Some((dependence, target, noise)),
            KernelProgram::NeuralJoint(_) => None,
        }
    }

    pub fn symbolic_mut(&mut self) -> Option<(&mut Dependence, &mut Target, &mut Noise)> {
        match &mut self.program {
            KernelProgram::Symbolic {
                dependence,
                target,
                noise,
            } => Some((dependence, target, noise)),
            KernelProgram::NeuralJoint(_) => None,
        }
    }
}

fn valid_probability(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn increasing(values: &[f32]) -> bool {
    values
        .windows(2)
        .all(|pair| pair[0].is_finite() && pair[0] <= pair[1])
        && values.last().is_none_or(|value| value.is_finite())
}

fn validate_marginal(marginal: &Marginal, depth: usize) -> std::result::Result<(), String> {
    if depth > 2 {
        return Err("marginal nesting exceeds its bound".into());
    }
    match marginal {
        Marginal::Constant { value } if valid_probability(*value) => Ok(()),
        Marginal::Bernoulli { probability } if valid_probability(*probability) => Ok(()),
        Marginal::Grid {
            values,
            probabilities,
        } if !values.is_empty()
            && values.len() == probabilities.len()
            && values.len() <= 65_536
            && increasing(values)
            && probabilities.iter().all(|value| valid_probability(*value))
            && probabilities.iter().sum::<f32>() > 0.0 =>
        {
            Ok(())
        }
        Marginal::QuantileSpline { values }
            if matches!(values.len(), 5 | 9 | 17 | 33 | 65) && increasing(values) =>
        {
            Ok(())
        }
        Marginal::Beta { alpha, beta }
            if alpha.is_finite() && *alpha > 0.0 && beta.is_finite() && *beta > 0.0 =>
        {
            Ok(())
        }
        Marginal::Gaussian { mean, sigma }
            if valid_probability(*mean) && sigma.is_finite() && *sigma > 0.0 =>
        {
            Ok(())
        }
        Marginal::Histogram {
            edges,
            probabilities,
        } if (2..=257).contains(&edges.len())
            && probabilities.len() + 1 == edges.len()
            && increasing(edges)
            && probabilities.iter().all(|value| valid_probability(*value))
            && probabilities.iter().sum::<f32>() > 0.0 =>
        {
            Ok(())
        }
        Marginal::ZeroInflated {
            point,
            probability,
            base,
        } if valid_probability(*point) && valid_probability(*probability) => {
            validate_marginal(base, depth + 1)
        }
        _ => Err("invalid bounded marginal".into()),
    }
}

fn valid_edge(edge: &CopulaEdge, width: usize) -> bool {
    (edge.parent as usize) < width
        && (edge.child as usize) < width
        && edge.parent != edge.child
        && (-0.995..=0.995).contains(&edge.correlation)
}

fn validate_dependence(
    dependence: &Dependence,
    width: usize,
    depth: usize,
) -> std::result::Result<(), String> {
    if depth > 2 {
        return Err("dependence mixture nesting exceeds its bound".into());
    }
    match dependence {
        Dependence::Independent => Ok(()),
        Dependence::ChowLiu { root, edges } => {
            if *root as usize >= width || edges.len() != width.saturating_sub(1) {
                return Err("invalid Chow-Liu tree shape".into());
            }
            let mut seen = vec![false; width];
            seen[*root as usize] = true;
            for edge in edges {
                let parent = edge.parent as usize;
                let child = edge.child as usize;
                if !valid_edge(edge, width) || !seen[parent] || seen[child] {
                    return Err("Chow-Liu edges must be bounded and topologically ordered".into());
                }
                seen[child] = true;
            }
            Ok(())
        }
        Dependence::GaussianCopula { cholesky }
            if cholesky.len() == width * width
                && cholesky
                    .iter()
                    .all(|value| value.is_finite() && value.abs() <= 1.0) =>
        {
            Ok(())
        }
        Dependence::SparseGraph { edges }
            if edges.len() <= width.saturating_mul(8)
                && edges.iter().all(|edge| valid_edge(edge, width)) =>
        {
            Ok(())
        }
        Dependence::Poet {
            rank,
            loadings,
            residual_edges,
        } if (1..=16).contains(rank)
            && loadings.len() <= width * usize::from(*rank)
            && loadings.iter().all(|loading| {
                (loading.feature as usize) < width
                    && loading.factor < *rank
                    && loading.loading.is_finite()
                    && loading.loading.abs() <= 1.0
            })
            && residual_edges.len() <= width.saturating_mul(4)
            && residual_edges.iter().all(|edge| valid_edge(edge, width)) =>
        {
            Ok(())
        }
        Dependence::Vine { edges }
            if edges.len() <= width.saturating_mul(8)
                && edges.iter().all(|edge| {
                    (edge.left as usize) < width
                        && (edge.right as usize) < width
                        && edge.left != edge.right
                        && edge.conditioning_depth <= 8
                        && (-0.995..=0.995).contains(&edge.parameter)
                }) =>
        {
            Ok(())
        }
        Dependence::Mixture {
            weights,
            components,
        } if !components.is_empty()
            && components.len() <= 8
            && weights.len() == components.len()
            && weights.iter().all(|weight| valid_probability(*weight))
            && weights.iter().sum::<f32>() > 0.0 =>
        {
            components
                .iter()
                .try_for_each(|component| validate_dependence(component, width, depth + 1))
        }
        Dependence::Triangular { terms }
            if terms.len() <= width.saturating_mul(8)
                && terms.iter().all(|term| {
                    term.parent < term.child
                        && (term.child as usize) < width
                        && term.coefficient.is_finite()
                        && term.coefficient.abs() <= 0.995
                }) =>
        {
            Ok(())
        }
        _ => Err("invalid bounded dependence operator".into()),
    }
}

fn validate_additive(term: &AdditiveTerm, width: usize) -> bool {
    (term.feature as usize) < width
        && (3..=17).contains(&term.knots.len())
        && term.knots.len() == term.values.len()
        && increasing(&term.knots)
        && term.values.iter().all(|value| value.is_finite())
}

fn validate_target(target: &Target, width: usize) -> std::result::Result<(), String> {
    let linear_terms_valid = |terms: &[LinearTerm]| {
        terms.len() <= 64
            && terms
                .iter()
                .all(|term| (term.feature as usize) < width && term.coefficient.is_finite())
    };
    match target {
        Target::SparseLinear { intercept, terms } | Target::SparseLogistic { intercept, terms }
            if intercept.is_finite() && linear_terms_valid(terms) =>
        {
            Ok(())
        }
        Target::SparseGam {
            intercept, terms, ..
        } if intercept.is_finite()
            && terms.len() <= 64
            && terms.iter().all(|term| validate_additive(term, width)) =>
        {
            Ok(())
        }
        Target::Ga2m {
            intercept,
            main_terms,
            interactions,
            ..
        } if intercept.is_finite()
            && main_terms.len() <= 64
            && main_terms.iter().all(|term| validate_additive(term, width))
            && interactions.len() <= 64
            && interactions.iter().all(|term| {
                (term.left as usize) < width
                    && (term.right as usize) < width
                    && term.left < term.right
                    && (3..=9).contains(&term.left_knots.len())
                    && (3..=9).contains(&term.right_knots.len())
                    && increasing(&term.left_knots)
                    && increasing(&term.right_knots)
                    && term.values.len() == term.left_knots.len() * term.right_knots.len()
                    && term.values.iter().all(|value| value.is_finite())
            }) =>
        {
            Ok(())
        }
        Target::Mars {
            intercept, terms, ..
        } if intercept.is_finite()
            && terms.len() <= 128
            && terms.iter().all(|term| {
                term.coefficient.is_finite()
                    && (1..=3).contains(&term.factors.len())
                    && term.factors.iter().all(|factor| {
                        (factor.feature as usize) < width
                            && factor.knot.is_finite()
                            && matches!(factor.direction, -1 | 1)
                    })
            }) =>
        {
            Ok(())
        }
        Target::ObliviousTree {
            features,
            thresholds,
            leaves,
            ..
        } if features.len() <= 8
            && features.len() == thresholds.len()
            && leaves.len() == 1usize << features.len()
            && features.iter().all(|feature| (*feature as usize) < width)
            && thresholds.iter().all(|value| value.is_finite())
            && leaves.iter().all(|value| value.is_finite()) =>
        {
            Ok(())
        }
        Target::CompactNeuralResidual {
            intercept,
            linear_terms,
            hidden_features,
            hidden_width,
            input_weights,
            hidden_biases,
            output_weights,
            ..
        } if intercept.is_finite()
            && linear_terms_valid(linear_terms)
            && (1..=24).contains(&hidden_features.len())
            && (1..=16).contains(hidden_width)
            && hidden_features
                .iter()
                .all(|feature| (*feature as usize) < width)
            && {
                let mut unique = hidden_features.clone();
                unique.sort_unstable();
                unique.dedup();
                unique.len() == hidden_features.len()
            }
            && input_weights.len() == hidden_features.len() * usize::from(*hidden_width)
            && hidden_biases.len() == usize::from(*hidden_width)
            && output_weights.len() == usize::from(*hidden_width)
            && input_weights.iter().all(|value| value.is_finite())
            && hidden_biases.iter().all(|value| value.is_finite())
            && output_weights.iter().all(|value| value.is_finite()) =>
        {
            Ok(())
        }
        _ => Err("invalid bounded target operator".into()),
    }
}

fn validate_noise(noise: &Noise, width: usize) -> std::result::Result<(), String> {
    match noise {
        Noise::Homoscedastic { sigma } if valid_probability(*sigma) => Ok(()),
        Noise::BernoulliCalibration { base_rate } if valid_probability(*base_rate) => Ok(()),
        Noise::Heteroscedastic {
            intercept,
            terms,
            minimum_sigma,
            maximum_sigma,
        } if intercept.is_finite()
            && valid_probability(*minimum_sigma)
            && valid_probability(*maximum_sigma)
            && minimum_sigma <= maximum_sigma
            && terms.len() <= 64
            && terms
                .iter()
                .all(|term| (term.feature as usize) < width && term.coefficient.is_finite()) =>
        {
            Ok(())
        }
        Noise::IsotonicCalibration {
            knots,
            probabilities,
        } if (2..=33).contains(&knots.len())
            && knots.len() == probabilities.len()
            && increasing(knots)
            && increasing(probabilities)
            && probabilities.iter().all(|value| valid_probability(*value)) =>
        {
            Ok(())
        }
        _ => Err("invalid bounded noise or calibration operator".into()),
    }
}

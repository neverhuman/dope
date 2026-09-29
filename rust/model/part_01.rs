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
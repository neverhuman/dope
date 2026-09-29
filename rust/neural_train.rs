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
    include!("neural_train/gpu/part_01.rs");
    include!("neural_train/gpu/part_02.rs");
}
#[cfg(all(test, feature = "gpu-training"))]
include!("neural_train/tests.rs");

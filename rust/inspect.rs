use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::codec::{ArtifactAccounting, LoadedKernel, account_artifact, load_kernel};
use crate::error::{Result, io_error};
use crate::model::{Dependence, KernelProgram, Marginal, NeuralArchitecture, Noise, Target};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Inspection {
    pub format: String,
    pub version: u8,
    pub artifact_bytes: usize,
    pub byte_accounting: Option<ArtifactAccounting>,
    pub task: String,
    pub rows_fitted: u64,
    pub positional_features: u32,
    pub target_position: String,
    pub decoder_id: Option<u8>,
    pub quantization_bits: Option<u8>,
    pub compliant: Option<bool>,
    pub schema: Vec<String>,
    pub marginals: Vec<String>,
    pub dependence: String,
    pub target: String,
    pub noise: String,
    pub sexpr: String,
}

pub fn inspect_kernel(path: &Path) -> Result<Inspection> {
    let artifact_bytes = fs::metadata(path)
        .map_err(|error| io_error(path, error))?
        .len() as usize;
    let loaded = load_kernel(path)?;
    let native_version = if matches!(loaded, LoadedKernel::V3(_)) {
        3
    } else {
        2
    };
    let byte_accounting = if native_version == 3 {
        Some(account_artifact(
            &fs::read(path).map_err(|error| io_error(path, error))?,
        )?)
    } else {
        None
    };
    match loaded {
        LoadedKernel::V3(kernel) | LoadedKernel::V2(kernel) => {
            let schema: Vec<String> = kernel
                .schema
                .iter()
                .map(|entry| format!("{:?}", entry.kind).to_ascii_lowercase())
                .collect();
            let marginals: Vec<String> = kernel
                .marginals
                .iter()
                .map(|entry| {
                    match entry {
                        Marginal::Constant { .. } => "constant",
                        Marginal::Bernoulli { .. } => "bernoulli",
                        Marginal::Grid { .. } => "categorical_grid",
                        Marginal::QuantileSpline { .. } => "quantile_spline",
                        Marginal::Beta { .. } => "beta",
                        Marginal::Gaussian { .. } => "gaussian",
                        Marginal::Histogram { .. } => "histogram",
                        Marginal::ZeroInflated { .. } => "zero_inflated",
                    }
                    .to_string()
                })
                .collect();
            let (dependence, target, noise) = match &kernel.program {
                KernelProgram::Symbolic {
                    dependence,
                    target,
                    noise,
                } => (
                    match dependence {
                        Dependence::Independent => "independence",
                        Dependence::ChowLiu { .. } => "chow_liu",
                        Dependence::GaussianCopula { .. } => "gaussian_copula",
                        Dependence::SparseGraph { .. } => "sparse_graph",
                        Dependence::Poet { .. } => "poet",
                        Dependence::Vine { .. } => "vine",
                        Dependence::Mixture { .. } => "mixture",
                        Dependence::Triangular { .. } => "triangular",
                    }
                    .to_string(),
                    match target {
                        Target::SparseLinear { .. } => "sparse_linear",
                        Target::SparseLogistic { .. } => "sparse_logistic",
                        Target::SparseGam { .. } => "sparse_gam",
                        Target::Ga2m { .. } => "ga2m",
                        Target::Mars { .. } => "mars",
                        Target::ObliviousTree { .. } => "oblivious_tree",
                        Target::CompactNeuralResidual { .. } => "compact_neural_residual",
                    }
                    .to_string(),
                    match noise {
                        Noise::Homoscedastic { .. } => "homoscedastic",
                        Noise::BernoulliCalibration { .. } => "bernoulli_calibration",
                        Noise::Heteroscedastic { .. } => "heteroscedastic",
                        Noise::IsotonicCalibration { .. } => "isotonic_calibration",
                    }
                    .to_string(),
                ),
                KernelProgram::NeuralJoint(generator) => (
                    "neural_joint".into(),
                    match generator.architecture {
                        NeuralArchitecture::Tvae => "tvae",
                        NeuralArchitecture::MaskedAutoregressiveTransformer => {
                            "masked_autoregressive_transformer"
                        }
                        NeuralArchitecture::TabSyn => "tabsyn",
                    }
                    .into(),
                    "joint_missingness_and_rank_variance".into(),
                ),
            };
            let sexpr = format!(
                "(dpk v={} task={} n={} p={} seed={} (schema {}) (marginals {}) (dependence kind={}) (target kind={}) (noise kind={}))",
                native_version,
                kernel.task.as_str(),
                kernel.rows_fitted,
                kernel.features,
                kernel.seed,
                schema
                    .iter()
                    .enumerate()
                    .map(|(index, kind)| format!("({kind} i={index})"))
                    .collect::<Vec<_>>()
                    .join(" "),
                marginals
                    .iter()
                    .enumerate()
                    .map(|(index, kind)| format!("({kind} i={index})"))
                    .collect::<Vec<_>>()
                    .join(" "),
                dependence,
                target,
                noise,
            );
            Ok(Inspection {
                format: format!("dope-kernel-v{native_version}"),
                version: native_version,
                artifact_bytes,
                byte_accounting,
                task: kernel.task.as_str().into(),
                rows_fitted: kernel.rows_fitted,
                positional_features: kernel.features,
                target_position: "last".into(),
                decoder_id: Some(1),
                quantization_bits: Some(kernel.quantization_bits),
                compliant: Some(kernel.compliant),
                schema,
                marginals,
                dependence,
                target,
                noise,
                sexpr,
            })
        }
        LoadedKernel::V1(value) => {
            let shell = &value["shell"];
            Ok(Inspection {
                format: "dope-kernel".into(),
                version: 1,
                artifact_bytes,
                byte_accounting,
                task: shell["task"].as_str().unwrap_or("unknown").into(),
                rows_fitted: shell["n"].as_u64().unwrap_or(0),
                positional_features: shell["p"].as_u64().unwrap_or(0) as u32,
                target_position: "last".into(),
                decoder_id: None,
                quantization_bits: None,
                compliant: None,
                schema: Vec::new(),
                marginals: value["marginals"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|entry| entry["kind"].as_str().unwrap_or("unknown").into())
                    .collect(),
                dependence: value["dependence"]["kind"]
                    .as_str()
                    .unwrap_or("unknown")
                    .into(),
                target: value["target"]["kind"].as_str().unwrap_or("unknown").into(),
                noise: value["residual"]["kind"]
                    .as_str()
                    .unwrap_or("unknown")
                    .into(),
                sexpr: value["program"].as_str().unwrap_or("").into(),
            })
        }
    }
}

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::codec::{ArtifactAccounting, LoadedKernel, account_artifact, load_kernel};
use crate::error::{DopeError, Result, io_error};
use crate::metrics::{METRIC_NAME, artifact_charge, strict_parse_json};
use crate::model::{Dependence, KernelProgram, Marginal, NeuralArchitecture, Noise, Target};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InspectionByteScopes {
    pub payload_bytes: usize,
    pub sampling_closure_bytes: Option<usize>,
    pub section_bytes: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InspectionMetricStatus {
    pub metric_name: String,
    pub status: String,
    pub protocol_frozen: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Inspection {
    pub format: String,
    pub version: u8,
    pub artifact_bytes: usize,
    pub byte_scopes: InspectionByteScopes,
    pub byte_accounting: Option<ArtifactAccounting>,
    pub metric_status: InspectionMetricStatus,
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

impl Inspection {
    /// Total framing bytes across the container header and section headers.
    pub fn framing_bytes(&self) -> Option<usize> {
        self.byte_accounting.as_ref().map(|acc| {
            acc.header_bytes
                + acc
                    .sections
                    .iter()
                    .map(|section| section.framing_bytes)
                    .sum::<usize>()
        })
    }
}

fn inspect_tdl1_bytes(bytes: &[u8], sampling_closure_bytes: Option<usize>) -> Result<Inspection> {
    let accounting = artifact_charge::account_tdl1_container(bytes)
        .map_err(|err| DopeError::Codec(err.reason.into()))?;
    let payload_bytes = bytes.len();
    let section_bytes = accounting
        .sections
        .iter()
        .map(|s| s.encoded_bytes)
        .collect();
    let section_count = accounting.sections.len();
    Ok(Inspection {
        format: "tdl-container-v1".into(),
        version: 1,
        artifact_bytes: payload_bytes,
        byte_scopes: InspectionByteScopes {
            payload_bytes,
            sampling_closure_bytes,
            section_bytes,
        },
        byte_accounting: Some(accounting),
        metric_status: InspectionMetricStatus {
            metric_name: METRIC_NAME.to_owned(),
            status: "unmeasured".to_owned(),
            protocol_frozen: false,
        },
        task: "tdl_container".into(),
        rows_fitted: 0,
        positional_features: 0,
        target_position: "none".into(),
        decoder_id: None,
        quantization_bits: None,
        compliant: None,
        schema: Vec::new(),
        marginals: Vec::new(),
        dependence: "tdl1".into(),
        target: "tdl1".into(),
        noise: "tdl1".into(),
        sexpr: format!("(tdl1 v=1 sections={section_count} bytes={payload_bytes})"),
    })
}

fn extract_charge_manifest_input(value: &Value) -> Option<&Value> {
    let obj = value.as_object()?;
    if obj.contains_key("sections") && obj.contains_key("closure_files") {
        return Some(value);
    }
    if obj.get("component").and_then(Value::as_str) == Some("charge") {
        let input = obj.get("input")?;
        let in_obj = input.as_object()?;
        if in_obj.contains_key("sections") && in_obj.contains_key("closure_files") {
            return Some(input);
        }
    }
    None
}

fn parse_inspection_json(bytes: &[u8]) -> Result<Option<Value>> {
    if bytes.starts_with(crate::codec::MAGIC)
        || bytes.starts_with(crate::codec::V2_MAGIC)
        || bytes.starts_with(artifact_charge::MAGIC)
    {
        return Ok(None);
    }
    let text = std::str::from_utf8(bytes).map_err(|_| DopeError::Codec("invalid_json".into()))?;
    strict_parse_json(text)
        .map(Some)
        .map_err(|error| DopeError::Codec(error.reason.into()))
}

pub fn inspect_kernel(path: &Path) -> Result<Inspection> {
    let raw_bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    if raw_bytes.starts_with(artifact_charge::MAGIC) {
        return inspect_tdl1_bytes(&raw_bytes, None);
    }
    if let Some(json_val) = parse_inspection_json(&raw_bytes)?
        && let Some(charge_input) = extract_charge_manifest_input(&json_val)
    {
        let evaluated = artifact_charge::evaluate(charge_input);
        if evaluated.get("status").and_then(Value::as_str) != Some("complete") {
            let reason = evaluated
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("invalid_charge_manifest");
            return Err(DopeError::Codec(reason.into()));
        }
        let container = artifact_charge::canonical_container(&charge_input["sections"])
            .map_err(|err| DopeError::Codec(err.reason.into()))?;
        let closure_total = evaluated
            .get("sampling_closure_bytes")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        return inspect_tdl1_bytes(&container, closure_total);
    }

    let artifact_bytes = raw_bytes.len();
    let loaded = load_kernel(path)?;
    let native_version = if matches!(loaded, LoadedKernel::V3(_)) {
        3
    } else {
        2
    };
    let byte_accounting = if native_version == 3 {
        Some(account_artifact(&raw_bytes)?)
    } else {
        None
    };
    let section_bytes = byte_accounting
        .as_ref()
        .map(|acc| acc.sections.iter().map(|s| s.encoded_bytes).collect())
        .unwrap_or_default();
    let byte_scopes = InspectionByteScopes {
        payload_bytes: artifact_bytes,
        sampling_closure_bytes: None,
        section_bytes,
    };
    let metric_status = InspectionMetricStatus {
        metric_name: METRIC_NAME.to_owned(),
        status: "unmeasured".to_owned(),
        protocol_frozen: false,
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
                        Noise::ResidualV1 { .. } => "development_residual_v1",
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
                        NeuralArchitecture::TabDdpm => "tabddpm_direct_rank",
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
                byte_scopes,
                byte_accounting,
                metric_status,
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
                byte_scopes,
                byte_accounting,
                metric_status,
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

/// Reconcile `inspect_kernel` with `artifact_charge` for a native `DPK3` artifact,
/// binary `TDL1` container, or JSON `TDL1` charge manifest, returning the exact
/// `artifact_charge` JSON record after verifying equality with `Inspection`.
/// Bare artifacts have no closure declaration and return an incomplete charge.
/// Bundle directories return the conservative recipient inventory. Their observed
/// manifest identity is not a trusted publication pin.
pub fn inspect_charge(path: &Path) -> Result<Value> {
    if path.is_dir() {
        let limits = crate::bundle::BundleLimits::default();
        let digest = crate::bundle::manifest_fingerprint(path, &limits)
            .map_err(|error| DopeError::Codec(error.code().into()))?;
        return inspect_bundle(path, &digest, &limits);
    }
    let inspection = inspect_kernel(path)?;
    let raw_bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    let charge = if let Some(json_val) = parse_inspection_json(&raw_bytes)?
        && let Some(charge_input) = extract_charge_manifest_input(&json_val)
    {
        artifact_charge::evaluate(charge_input)
    } else {
        artifact_charge::evaluate_artifact_bytes(&raw_bytes, &Value::Null, None)
    };
    if charge.get("status").and_then(Value::as_str) == Some("incomplete") {
        return Ok(charge);
    }
    if charge.get("status").and_then(Value::as_str) != Some("complete") {
        let reason = charge
            .get("reason")
            .and_then(Value::as_str)
            .unwrap_or("invalid_container");
        return Err(DopeError::Codec(reason.into()));
    }
    let charge_payload = charge["payload_bytes"].as_u64().unwrap() as usize;
    let charge_framing = charge["framing_bytes"].as_u64().unwrap() as usize;
    let charge_closure = charge["sampling_closure_bytes"].as_u64().unwrap() as usize;
    let charge_sections: Vec<usize> = charge["section_bytes"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    if inspection.artifact_bytes != charge_payload
        || inspection.byte_scopes.payload_bytes != charge_payload
        || inspection.framing_bytes() != Some(charge_framing)
        || inspection.byte_scopes.section_bytes != charge_sections
        || inspection.byte_scopes.sampling_closure_bytes != Some(charge_closure)
    {
        return Err(DopeError::Codec(
            "inspect byte accounting disagrees with TDL1 byte charge".into(),
        ));
    }
    Ok(charge)
}

#[cfg(test)]
mod tests;

mod bundle_inspection;
pub use bundle_inspection::inspect_bundle;

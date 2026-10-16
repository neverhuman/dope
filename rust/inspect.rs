use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::codec::{ArtifactAccounting, LoadedKernel, account_artifact, load_kernel};
use crate::error::{DopeError, Result, io_error};
use crate::metrics::{METRIC_NAME, artifact_charge};
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
            sampling_closure_bytes: Some(sampling_closure_bytes.unwrap_or(payload_bytes)),
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

pub fn inspect_kernel(path: &Path) -> Result<Inspection> {
    let raw_bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    if raw_bytes.starts_with(artifact_charge::MAGIC) {
        return inspect_tdl1_bytes(&raw_bytes, None);
    }
    if !raw_bytes.starts_with(crate::codec::MAGIC)
        && !raw_bytes.starts_with(crate::codec::V2_MAGIC)
        && let Ok(json_val) = serde_json::from_slice::<Value>(&raw_bytes)
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
pub fn inspect_charge(path: &Path) -> Result<Value> {
    let inspection = inspect_kernel(path)?;
    let raw_bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    let charge = if !raw_bytes.starts_with(crate::codec::MAGIC)
        && !raw_bytes.starts_with(crate::codec::V2_MAGIC)
        && !raw_bytes.starts_with(artifact_charge::MAGIC)
        && let Ok(json_val) = serde_json::from_slice::<Value>(&raw_bytes)
        && let Some(charge_input) = extract_charge_manifest_input(&json_val)
    {
        artifact_charge::evaluate(charge_input)
    } else {
        artifact_charge::evaluate_artifact_bytes(&raw_bytes, &json!([]), None)
    };
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
        || inspection
            .byte_scopes
            .sampling_closure_bytes
            .unwrap_or(inspection.byte_scopes.payload_bytes)
            != charge_closure
    {
        return Err(DopeError::Codec(
            "inspect byte accounting disagrees with TDL1 byte charge".into(),
        ));
    }
    Ok(charge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{CompileOptions, compile_kernel_from_dir};
    use crate::model::Task;
    use std::path::PathBuf;

    #[test]
    fn inspect_and_tdl1_charge_report_identical_numbers_on_toy_and_golden_cases() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let tmp_dir =
            std::env::temp_dir().join(format!("dope-inspect-tdl1-{}", std::process::id()));
        fs::create_dir_all(&tmp_dir).unwrap();

        // 1. Toy case: compile examples/readme-regression -> toy.dpk
        let toy_dir = manifest_dir.join("examples/readme-regression");
        let compiled =
            compile_kernel_from_dir(&toy_dir, Task::Regression, &CompileOptions::default())
                .expect("toy regression must compile");
        let toy_dpk_path = tmp_dir.join("toy.dpk");
        fs::write(&toy_dpk_path, &compiled.artifact).unwrap();

        let committed_inspect: Value = serde_json::from_str(
            &fs::read_to_string(toy_dir.join("INSPECT.txt")).expect("INSPECT.txt must exist"),
        )
        .unwrap();
        let toy_inspection = inspect_kernel(&toy_dpk_path).expect("toy.dpk must inspect");
        let toy_inspection_json = serde_json::to_value(&toy_inspection).unwrap();
        assert_eq!(
            toy_inspection_json, committed_inspect,
            "toy.dpk inspection must match committed INSPECT.txt"
        );

        let toy_native_charge = inspect_charge(&toy_dpk_path).expect("toy.dpk charge must succeed");
        let direct_dpk_charge =
            artifact_charge::evaluate_artifact_bytes(&compiled.artifact, &json!([]), None);
        assert_eq!(toy_native_charge, direct_dpk_charge);
        assert_eq!(
            toy_inspection.artifact_bytes as u64,
            toy_native_charge["payload_bytes"].as_u64().unwrap()
        );
        assert_eq!(
            toy_inspection.byte_scopes.payload_bytes as u64,
            toy_native_charge["payload_bytes"].as_u64().unwrap()
        );
        assert_eq!(
            toy_inspection.framing_bytes().unwrap() as u64,
            toy_native_charge["framing_bytes"].as_u64().unwrap()
        );
        let native_section_vec: Vec<usize> = toy_native_charge["section_bytes"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(toy_inspection.byte_scopes.section_bytes, native_section_vec);

        // Also package the toy DPK sections into a canonical TDL1 container and verify
        // inspect_kernel and artifact_charge::evaluate report identical numbers on it.
        let toy_tdl1_sections = artifact_charge::dpk_to_tdl1_sections(&compiled.artifact)
            .expect("toy DPK sections must convert to TDL1 sections");
        let toy_tdl1_bytes = artifact_charge::canonical_container(&toy_tdl1_sections)
            .expect("toy TDL1 container must encode");
        let toy_tdl1_path = tmp_dir.join("toy.tdl1");
        fs::write(&toy_tdl1_path, &toy_tdl1_bytes).unwrap();
        let toy_tdl1_inspect = inspect_kernel(&toy_tdl1_path).expect("toy.tdl1 must inspect");
        let toy_tdl1_charge = artifact_charge::evaluate(&json!({
            "sections": toy_tdl1_sections,
            "closure_files": [],
        }));
        assert_eq!(toy_tdl1_charge["status"], "complete");
        assert_eq!(
            toy_tdl1_inspect.artifact_bytes as u64,
            toy_tdl1_charge["payload_bytes"].as_u64().unwrap()
        );
        assert_eq!(
            toy_tdl1_inspect.byte_scopes.payload_bytes as u64,
            toy_tdl1_charge["payload_bytes"].as_u64().unwrap()
        );
        assert_eq!(
            toy_tdl1_inspect.byte_scopes.sampling_closure_bytes.unwrap() as u64,
            toy_tdl1_charge["sampling_closure_bytes"].as_u64().unwrap()
        );
        assert_eq!(
            toy_tdl1_inspect.framing_bytes().unwrap() as u64,
            toy_tdl1_charge["framing_bytes"].as_u64().unwrap()
        );
        assert_eq!(
            toy_tdl1_inspect.byte_scopes.section_bytes,
            toy_inspection.byte_scopes.section_bytes
        );
        assert_eq!(inspect_charge(&toy_tdl1_path).unwrap(), toy_tdl1_charge);

        // 2. Golden charge cases: tests/golden/charge/*.json
        let golden_charge_dir = manifest_dir.join("tests/golden/charge");
        let mut checked_goldens = 0usize;
        let mut entries: Vec<_> = fs::read_dir(&golden_charge_dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|entry| entry.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        entries.sort();

        for golden_path in entries {
            let case: Value =
                serde_json::from_str(&fs::read_to_string(&golden_path).unwrap()).unwrap();
            let expected = &case["expected"];
            let evaluated = artifact_charge::evaluate(&case["input"]);
            assert_eq!(&evaluated, expected, "golden charge evaluation must match");

            if expected["status"].as_str() != Some("complete") {
                assert!(
                    inspect_kernel(&golden_path).is_err(),
                    "non-complete golden charge case {} must fail inspect_kernel",
                    golden_path.display()
                );
                continue;
            }
            checked_goldens += 1;

            // Inspect the golden JSON manifest directly
            let manifest_inspect =
                inspect_kernel(&golden_path).expect("complete golden charge manifest must inspect");
            let manifest_charge = inspect_charge(&golden_path)
                .expect("complete golden charge manifest must produce charge");
            assert_eq!(&manifest_charge, expected);

            let exp_payload = expected["payload_bytes"].as_u64().unwrap() as usize;
            let exp_framing = expected["framing_bytes"].as_u64().unwrap() as usize;
            let exp_closure = expected["sampling_closure_bytes"].as_u64().unwrap() as usize;
            let exp_sections: Vec<usize> = expected["section_bytes"]
                .as_object()
                .unwrap()
                .values()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();

            assert_eq!(manifest_inspect.artifact_bytes, exp_payload);
            assert_eq!(manifest_inspect.byte_scopes.payload_bytes, exp_payload);
            assert_eq!(
                manifest_inspect.byte_scopes.sampling_closure_bytes,
                Some(exp_closure)
            );
            assert_eq!(manifest_inspect.framing_bytes(), Some(exp_framing));
            assert_eq!(manifest_inspect.byte_scopes.section_bytes, exp_sections);

            // Also write the binary TDL1 container and inspect it directly
            let container = artifact_charge::canonical_container(&case["input"]["sections"])
                .expect("golden sections must encode to TDL1 container");
            let container_path =
                tmp_dir.join(format!("{}.tdl1", case["case_id"].as_str().unwrap()));
            fs::write(&container_path, &container).unwrap();
            let container_inspect =
                inspect_kernel(&container_path).expect("binary TDL1 container must inspect");
            assert_eq!(container_inspect.artifact_bytes, exp_payload);
            assert_eq!(container_inspect.byte_scopes.payload_bytes, exp_payload);
            assert_eq!(container_inspect.framing_bytes(), Some(exp_framing));
            assert_eq!(container_inspect.byte_scopes.section_bytes, exp_sections);
        }
        assert!(
            checked_goldens >= 6,
            "expected at least 6 complete golden charge cases, got {checked_goldens}"
        );
        let _ = fs::remove_dir_all(&tmp_dir);
    }
}

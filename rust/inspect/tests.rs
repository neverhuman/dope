use super::*;
use crate::compiler::{CompileOptions, compile_kernel_from_dir};
use crate::model::Task;
use serde_json::json;
use std::path::PathBuf;

fn bare_tdl1_fixture(name: &str) -> (PathBuf, Vec<u8>) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/inspect-closure-tests")
        .join(std::process::id().to_string());
    fs::create_dir_all(&root).unwrap();
    let bytes = artifact_charge::canonical_container(&json!([
        {"name": "weights", "hex": "010203"}
    ]))
    .unwrap();
    let path = root.join(name);
    fs::write(&path, &bytes).unwrap();
    (path, bytes)
}

#[test]
fn bare_tdl1_inspection_keeps_sampling_closure_unmeasured() {
    let (path, bytes) = bare_tdl1_fixture("inspection.tdl1");
    let inspection = inspect_kernel(&path).unwrap();
    assert_eq!(inspection.byte_scopes.payload_bytes, bytes.len());
    assert_eq!(inspection.byte_scopes.sampling_closure_bytes, None);
    assert_eq!(inspection.byte_scopes.section_bytes, vec![3]);
    assert_eq!(inspection.framing_bytes(), Some(bytes.len() - 3));
    let serialized = serde_json::to_value(inspection).unwrap();
    assert!(serialized["byte_scopes"]["sampling_closure_bytes"].is_null());
    assert_eq!(serialized["metric_status"]["status"], "unmeasured");
}

#[test]
fn bare_tdl1_charge_is_incomplete_without_a_closure_manifest() {
    let (path, bytes) = bare_tdl1_fixture("charge.tdl1");
    let charge = inspect_charge(&path).unwrap();
    assert_eq!(charge["status"], "incomplete");
    assert_eq!(charge["reason"], "missing_charge_manifest");
    assert_eq!(charge, artifact_charge::evaluate(&json!({})));
    assert_eq!(
        charge,
        artifact_charge::evaluate_artifact_bytes(&bytes, &Value::Null, None)
    );
    let declared_empty = artifact_charge::evaluate_artifact_bytes(&bytes, &json!([]), None);
    assert_eq!(declared_empty["status"], "complete");
    assert_eq!(declared_empty["sampling_closure_bytes"], bytes.len());
}

#[test]
fn inspect_rejects_duplicate_charge_manifest_keys() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/inspect-strict-json")
        .join(std::process::id().to_string());
    fs::create_dir_all(&root).unwrap();
    let duplicate = r#"{"sections":[{"name":"weights","hex":"010203"}],"closure_files":[{"name":"forbidden","hex":"aabbccdd","scope":"runtime","dataset_specific":true}],"closure_files":[]}"#;
    let valid = r#"{"sections":[{"name":"weights","hex":"010203"}],"closure_files":[]}"#;
    let nested = r#"{"sections":[{"name":"weights","hex":"010203"}],"closure_files":[{"name":"forbidden","hex":"aabbccdd","scope":"runtime","dataset_specific":true,"dataset_specific":false}]}"#;
    let cases = [
        duplicate.to_owned(),
        format!(r#"{{"component":"charge","input":{duplicate}}}"#),
        nested.to_owned(),
        format!(r#"{{"component":"charge","input":{valid},"input":{valid}}}"#),
    ];
    for (index, input) in cases.iter().enumerate() {
        let path = root.join(format!("duplicate-{index}.json"));
        fs::write(&path, input).unwrap();
        for result in [
            inspect_kernel(&path).map(|_| ()),
            inspect_charge(&path).map(|_| ()),
        ] {
            assert!(matches!(
                result,
                Err(DopeError::Codec(reason)) if reason == "duplicate_json_key"
            ));
        }
    }
}

#[test]
fn inspect_and_tdl1_charge_report_identical_numbers_on_toy_and_golden_cases() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let tmp_dir = std::env::temp_dir().join(format!("dope-inspect-tdl1-{}", std::process::id()));
    fs::create_dir_all(&tmp_dir).unwrap();

    // 1. Toy case: compile examples/readme-regression -> toy.dpk
    let toy_dir = manifest_dir.join("examples/readme-regression");
    let compiled = compile_kernel_from_dir(&toy_dir, Task::Regression, &CompileOptions::default())
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

    let missing_charge = artifact_charge::evaluate(&json!({}));
    assert_eq!(toy_inspection.byte_scopes.sampling_closure_bytes, None);
    assert_eq!(inspect_charge(&toy_dpk_path).unwrap(), missing_charge);
    let toy_native_charge =
        artifact_charge::evaluate_artifact_bytes(&compiled.artifact, &json!([]), None);
    assert_eq!(toy_native_charge["status"], "complete");
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
    assert_eq!(toy_tdl1_inspect.byte_scopes.sampling_closure_bytes, None);
    assert_eq!(
        toy_tdl1_inspect.framing_bytes().unwrap() as u64,
        toy_tdl1_charge["framing_bytes"].as_u64().unwrap()
    );
    assert_eq!(
        toy_tdl1_inspect.byte_scopes.section_bytes,
        toy_inspection.byte_scopes.section_bytes
    );
    assert_eq!(inspect_charge(&toy_tdl1_path).unwrap(), missing_charge);

    let toy_manifest_path = tmp_dir.join("toy-charge.json");
    fs::write(
        &toy_manifest_path,
        serde_json::to_vec(&json!({
            "sections": toy_tdl1_sections,
            "closure_files": [],
        }))
        .unwrap(),
    )
    .unwrap();
    let declared_toy = inspect_kernel(&toy_manifest_path).unwrap();
    assert_eq!(
        declared_toy.byte_scopes.sampling_closure_bytes,
        Some(toy_tdl1_bytes.len())
    );
    assert_eq!(inspect_charge(&toy_manifest_path).unwrap(), toy_tdl1_charge);

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
        let case: Value = serde_json::from_str(&fs::read_to_string(&golden_path).unwrap()).unwrap();
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
        let container_path = tmp_dir.join(format!("{}.tdl1", case["case_id"].as_str().unwrap()));
        fs::write(&container_path, &container).unwrap();
        let container_inspect =
            inspect_kernel(&container_path).expect("binary TDL1 container must inspect");
        assert_eq!(container_inspect.artifact_bytes, exp_payload);
        assert_eq!(container_inspect.byte_scopes.payload_bytes, exp_payload);
        assert_eq!(container_inspect.framing_bytes(), Some(exp_framing));
        assert_eq!(container_inspect.byte_scopes.section_bytes, exp_sections);
        assert_eq!(container_inspect.byte_scopes.sampling_closure_bytes, None);
        assert_eq!(inspect_charge(&container_path).unwrap(), missing_charge);
    }
    assert!(
        checked_goldens >= 6,
        "expected at least 6 complete golden charge cases, got {checked_goldens}"
    );
    let _ = fs::remove_dir_all(&tmp_dir);
}

use dope_core::dvs::{roles::DeclaredEvidenceRoles, DataPins};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);
const PACKET: &[u8] = br#"{"artifact":"toy","v":1}"#;
const SCHEMA: &[u8] = br#"{"schema":"toy","v":1}"#;
const SIDECAR: &[u8] = br#"{"sidecar":"toy"}"#;
const DECODER: &[u8] = br#"{"decoder":"toy"}"#;
const PRIVATE_DEPENDENCY: &[u8] = br#"{"dependency":"CANARY_PRIVATE_VALUE"}"#;

fn directory() -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/dvs-contract")
        .join(format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn run(args: &[&str]) -> Output {
    assert!(Path::new(env!("CARGO_BIN_EXE_dope")).is_absolute());
    Command::new(env!("CARGO_BIN_EXE_dope"))
        .env("PATH", "")
        .env_remove("PYTHON")
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONPATH")
        .env_remove("VIRTUAL_ENV")
        .args(["--threads", "1"])
        .args(args)
        .output()
        .unwrap()
}

fn write(path: &Path, value: &Value) {
    std::fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

fn pins() -> Value {
    json!({
        "fit_scope_sha256":"a".repeat(64),
        "real_test_sha256":"b".repeat(64),
        "prepared_view_sha256":"c".repeat(64),
        "groups_sha256":"d".repeat(64),
        "roster_sha256":"e".repeat(64)
    })
}

fn roles(pins: &Value) -> Value {
    let common: DataPins = serde_json::from_value(pins.clone()).unwrap();
    serde_json::to_value(DeclaredEvidenceRoles::new(common, "f".repeat(64)).unwrap()).unwrap()
}

fn driver(pins: &Value) -> Value {
    let case: Value = serde_json::from_str(include_str!(
        "../../../tests/golden/drivers/v2/v2_additive.json"
    ))
    .unwrap();
    let mut evidence = case["input"].clone();
    evidence["protocol_version"] = json!("driver-rank-informative-v1");
    evidence["importance_unit"] = json!("declared_group");
    evidence["group_ids"] = json!([0, 1]);
    evidence["query_sha256"] = pins["real_test_sha256"].clone();
    evidence["groups_sha256"] = pins["groups_sha256"].clone();
    evidence["prepared_view_sha256"] = pins["prepared_view_sha256"].clone();
    evidence
}

fn request(complete: bool, include_roles: bool) -> Value {
    let pins = pins();
    let declared_roles = roles(&pins);
    let mut value = json!({
        "protocol":"dope-dvs-v1",
        "preparation_recipe":"nc1-minmax-onehot-v1",
        "pins":pins.clone(),
        "scheduled":[{"cell_sha256":"1".repeat(64),"family_id":0}],
        "required_families":[0],
        "cells":[],
        "selection":if complete {
            json!({
                "synthetic_validation_losses":[1.0,2.0,3.0],
                "real_fit_test_losses":[3.0,1.0,2.0],
                "synthetic_fit_test_losses":[3.0,1.0,2.0]
            })
        } else { Value::Null },
        "artifacts":{
            "packet":PACKET.to_vec(),
            "public_schema":SCHEMA.to_vec(),
            "mandatory_sidecars":[SIDECAR.to_vec()],
            "shared_decoder_parts":[DECODER.to_vec()],
            "owner_private_parts":[PRIVATE_DEPENDENCY.to_vec()]
        },
        "privacy":if complete {
            json!({
                "pins":pins.clone(),
                "real_fit":[[0.0],[1.0]],
                "synthetic":[[0.25],[0.75]],
                "real_holdout":[[0.1],[0.9]],
                "max_distance_pairs":8
            })
        } else { Value::Null }
    });
    if complete {
        value["cells"] = json!([{
            "cell_sha256":"1".repeat(64),
            "family_id":0,
            "pins":pins.clone(),
            "outcome":{
                "status":"complete",
                "null_losses":[4.0,4.0],
                "real_losses":[2.0,2.0],
                "synthetic_losses":[1.5,1.5],
                "real_gain_lcb":1.0,
                "absolute_gain_floor":0.01,
                "relative_gain_floor":0.01,
                "driver_evidence":driver(&pins),
                "wall_seconds":null,
                "fit_count":null
            }
        }]);
    } else {
        value["cells"] = json!([{
            "cell_sha256":"1".repeat(64),
            "family_id":0,
            "pins":pins.clone(),
            "outcome":{
                "status":"failed",
                "reason":"native_failure",
                "wall_seconds":null,
                "fit_count":null
            }
        }]);
    }
    if include_roles {
        value["declared_roles"] = declared_roles;
    }
    value
}

#[test]
fn transfer_report_serializes_actual_parts_and_card_without_python() {
    let root = directory();
    let request_path = root.join("request.json");
    let out = root.join("report");
    write(&request_path, &request(true, true));
    let result = run(&[
        "transfer-report",
        "--request",
        request_path.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(out.join("receipt.json")).unwrap()).unwrap();
    let card = std::fs::read_to_string(out.join("AuditCard.md")).unwrap();
    assert_eq!(receipt["status"], "complete");
    assert_eq!(receipt["robust_retention"], 1.25);
    assert_eq!(
        receipt["role_binding"]["state"],
        "caller_declared_hash_checked"
    );
    assert_eq!(
        receipt["role_binding"]["schema"],
        "dope.dvs.declared-evidence-roles/v1"
    );
    assert_eq!(receipt["role_binding"]["real_fit_usage"], "fit_only");
    assert_eq!(
        receipt["role_binding"]["synthetic_validation_usage"],
        "model_choice"
    );
    assert_eq!(
        receipt["role_binding"]["real_test_usage"],
        "assessment_only"
    );
    assert_eq!(receipt["role_binding"]["common_pins"], receipt["pins"]);
    assert_eq!(receipt["bytes"]["payload_bytes"], 63);
    assert_eq!(receipt["bytes"]["shared_bytes"], 17);
    assert_eq!(receipt["bytes"]["sampling_closure_bytes"], 80);
    assert_eq!(receipt["bytes"]["owner_private_dependency_bytes"], 37);
    let parts = receipt["bytes"]["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 5);
    assert_eq!(parts[0]["scope"], "packet");
    assert_eq!(parts[0]["bytes"], 24);
    assert_eq!(
        parts[0]["sha256"],
        "1f3d77e1cbdc128ff021df6c4a204410dd45b1e34d97d1641d6b4abbd0447945"
    );
    assert_eq!(parts[1]["scope"], "public_schema");
    assert_eq!(parts[1]["bytes"], 22);
    assert_eq!(
        parts[1]["sha256"],
        "fd74daf8038274a3dc813b688ccc78b3fbbc27666e9180e3a8a62bd2e5b2a8da"
    );
    assert_eq!(parts[2]["scope"], "mandatory_sidecar");
    assert_eq!(parts[2]["bytes"], 17);
    assert_eq!(
        parts[2]["sha256"],
        "48184d66fd6356be12b39cf74c779ed5f83854b894c998064883b43b63182ae1"
    );
    assert_eq!(parts[3]["scope"], "shared_decoder");
    assert_eq!(parts[3]["bytes"], 17);
    assert_eq!(
        parts[3]["sha256"],
        "81d90c956b7609ca5b168fab2417f13be621e78f3bc62559af2040a058775cc0"
    );
    assert_eq!(parts[4]["scope"], "owner_private");
    assert_eq!(parts[4]["bytes"], 37);
    assert_eq!(
        parts[4]["sha256"],
        "cf1a88efe1d44fbd10fd9f17fc6d1590992dc8378ae598f024b1bac3fa824aa5"
    );

    let status = receipt["status"].as_str().unwrap();
    let digest = receipt["role_binding"]["binding_sha256"].as_str().unwrap();
    let selection = receipt["selection"].to_string();
    let privacy = receipt["privacy"].to_string();
    let state_line = format!("State: {status}");
    assert!(card.contains(state_line.as_str()));
    assert!(card.contains("Robust retention: 1.250000"));
    assert!(card.contains("Payload bytes: 63"));
    assert!(card.contains("Sampling closure bytes: 80"));
    assert!(card.contains("Owner-private dependency bytes: 37"));
    assert!(card.contains("Evidence role binding: caller_declared_hash_checked"));
    assert!(card.contains(digest));
    assert!(card.contains(selection.as_str()));
    assert!(card.contains(privacy.as_str()));
    let public_outputs = format!(
        "{}\n{}\n{}\n{}",
        receipt,
        card,
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!public_outputs.contains("CANARY_PRIVATE_VALUE"));
}

#[test]
fn transfer_report_incomplete_keeps_failed_cell_and_unknown_costs() {
    let root = directory();
    let request_path = root.join("request.json");
    let out = root.join("report");
    write(&request_path, &request(false, true));
    let result = run(&[
        "transfer-report",
        "--request",
        request_path.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(out.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt["status"], "incomplete");
    assert_eq!(receipt["scheduled_cells"], 1);
    assert_eq!(receipt["failed_or_missing_cells"], 1);
    assert_eq!(receipt["cells"][0]["status"], "failed");
    assert!(receipt["cells"][0]["wall_seconds"].is_null());
    assert!(receipt["cells"][0]["fit_count"].is_null());
    assert!(std::fs::read_to_string(out.join("AuditCard.md"))
        .unwrap()
        .contains("Failed or missing: 1"));
}

#[test]
fn transfer_report_refuses_absent_roles_without_partial_output() {
    let root = directory();
    let request_path = root.join("request.json");
    let out = root.join("report");
    write(&request_path, &request(true, false));
    let result = run(&[
        "transfer-report",
        "--request",
        request_path.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(!result.status.success());
    assert!(!out.exists());
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(error.contains("missing_dvs_role_binding"));
    assert!(!error.contains("CANARY_PRIVATE_VALUE"));
}

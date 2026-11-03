//! Actual native CLI journeys with bounded public synthetic fixtures.
use dope_core::dvs::{roles::DeclaredEvidenceRoles, DataPins};
use dope_core::nc1::{
    Cell, DeclaredSchema, FeatureKind, FeatureSpec, InputColumn, InputTable, TargetSpec, Task,
};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
fn directory() -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/next-version-cli")
        .join(format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
    std::fs::create_dir_all(&p).unwrap();
    p
}
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_dope"))
        .args(["--threads", "1"])
        .args(args)
        .output()
        .unwrap()
}
fn declared_roles(pins: &Value) -> Value {
    let common: DataPins = serde_json::from_value(pins.clone()).unwrap();
    serde_json::to_value(DeclaredEvidenceRoles::new(common, "f".repeat(64)).unwrap()).unwrap()
}
fn write(path: &Path, value: &Value) {
    std::fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
fn path(p: &Path) -> &str {
    p.to_str().unwrap()
}
#[test]
fn nc1_owner_fit_apply_inverse_is_private_and_never_clips_query_tails() {
    let root = directory();
    let table = InputTable::new(
        "a".repeat(64),
        (0..20).collect(),
        vec![
            InputColumn {
                name: "CANARY_PRIVATE_X".into(),
                values: (0..20).map(|i| Cell::Number(i as f64)).collect(),
            },
            InputColumn {
                name: "CANARY_PRIVATE_Y".into(),
                values: (0..20).map(|i| Cell::Number((i * i) as f64)).collect(),
            },
        ],
    )
    .unwrap();
    let declared = DeclaredSchema {
        features: vec![FeatureSpec {
            column: 0,
            kind: FeatureKind::Continuous,
        }],
        target: TargetSpec {
            column: 1,
            task: Task::Regression,
        },
    };
    let request = root.join("fit.json");
    let out = root.join("private-fit");
    write(
        &request,
        &json!({"table":table,"declared":declared,"fit_rows":(0..16).collect::<Vec<_>>()}),
    );
    let first = run(&[
        "normalize",
        "fit",
        "--request",
        path(&request),
        "--out",
        path(&out),
    ]);
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(!String::from_utf8_lossy(&first.stdout).contains("CANARY"));
    let card = out.join("ScaleCard.private.json");
    let original = std::fs::read(&card).unwrap();
    let public = std::fs::read_to_string(out.join("schema.public.json")).unwrap();
    assert!(!public.contains("CANARY"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&card).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    let apply = root.join("apply.json");
    let applied = root.join("applied");
    write(
        &apply,
        &json!({"table":table,"rows":[19],"features_only":false}),
    );
    let result = run(&[
        "normalize",
        "apply",
        "--card",
        path(&card),
        "--request",
        path(&apply),
        "--out",
        path(&applied),
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let matrix: Value =
        serde_json::from_slice(&std::fs::read(applied.join("matrix.unit.private.json")).unwrap())
            .unwrap();
    assert!(matrix["x"][0][0].as_f64().unwrap() > 1.);
    assert!(matrix["y"][0].as_f64().unwrap() > 1.);
    let inverse = root.join("inverse.json");
    let raw = root.join("prediction.private.json");
    write(&inverse, &json!({"values":[3.]}));
    assert!(run(&[
        "normalize",
        "inverse-target",
        "--card",
        path(&card),
        "--request",
        path(&inverse),
        "--out",
        path(&raw)
    ])
    .status
    .success());
    let values: Value = serde_json::from_slice(&std::fs::read(raw).unwrap()).unwrap();
    assert_eq!(values["values"][0], 675.);
    assert!(!run(&[
        "normalize",
        "fit",
        "--request",
        path(&request),
        "--out",
        path(&out)
    ])
    .status
    .success());
    assert_eq!(std::fs::read(card).unwrap(), original);
}
#[test]
fn dvs_actual_handler_retains_missing_denominator_and_redacts_parse_errors() {
    let root = directory();
    let request = root.join("request.json");
    let out = root.join("report");
    let pins = json!({"fit_scope_sha256":"a".repeat(64),"real_test_sha256":"b".repeat(64),"prepared_view_sha256":"c".repeat(64),"groups_sha256":"d".repeat(64),"roster_sha256":"e".repeat(64)});
    let declared_roles = declared_roles(&pins);
    write(
        &request,
        &json!({"protocol":"dope-dvs-v1","preparation_recipe":"nc1-minmax-onehot-v1","pins":pins.clone(),"declared_roles":declared_roles,
        "scheduled":[{"cell_sha256":"1".repeat(64),"family_id":0}],"required_families":[0],"cells":[],"selection":null,"privacy":null,
        "artifacts":{"packet":[1,2],"public_schema":[3],"mandatory_sidecars":[],"shared_decoder_parts":[],"owner_private_parts":[]}}),
    );
    let result = run(&[
        "transfer-report",
        "--request",
        path(&request),
        "--out",
        path(&out),
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(out.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt["scheduled_cells"], 1);
    assert_eq!(receipt["failed_or_missing_cells"], 1);
    assert_eq!(
        receipt["role_binding"]["state"],
        "caller_declared_hash_checked"
    );
    assert_eq!(receipt["qualification_state"], "not_run");
    assert!(receipt["robust_retention"].is_null());
    assert!(out.join("AuditCard.md").is_file());
    write(
        &request,
        &json!({"CANARY_PRIVATE_HEADER":"CANARY_PRIVATE_VALUE"}),
    );
    let bad = root.join("bad");
    let result = run(&[
        "transfer-report",
        "--request",
        path(&request),
        "--out",
        path(&bad),
    ]);
    assert!(!result.status.success());
    assert!(!bad.exists());
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(error.contains("invalid_dvs_request"));
    assert!(!error.contains("CANARY"));
}
#[test]
fn dml4_pack_inspect_and_sample_need_no_encoder_or_source() {
    let root = directory();
    let input = root.join("packet.json");
    let wire = root.join("packet.dml4");
    let sample = root.join("samples.json");
    std::fs::write(
        &input,
        include_bytes!("../../dope-core/tests/fixtures/v4_producer/regression.packet.json"),
    )
    .unwrap();
    for args in [
        vec![
            "mlc-v4",
            "pack",
            "--input",
            path(&input),
            "--out",
            path(&wire),
        ],
        vec!["mlc-v4", "inspect", "--packet", path(&wire)],
        vec![
            "mlc-v4",
            "sample",
            "--packet",
            path(&wire),
            "--rows",
            "4",
            "--seed",
            "42",
            "--out",
            path(&sample),
        ],
    ] {
        let result = run(&args);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    let sampled: Value = serde_json::from_slice(&std::fs::read(&sample).unwrap()).unwrap();
    assert_eq!(sampled["x"].as_array().unwrap().len(), 4);
    let before = std::fs::read(&sample).unwrap();
    assert!(!run(&[
        "mlc-v4",
        "sample",
        "--packet",
        path(&wire),
        "--rows",
        "4",
        "--out",
        path(&sample)
    ])
    .status
    .success());
    assert_eq!(std::fs::read(sample).unwrap(), before);
}

#[test]
fn dvs_cli_requires_role_binding_before_creating_outputs() {
    let root = directory();
    let request = root.join("request.json");
    let out = root.join("report");
    write(
        &request,
        &json!({"protocol":"dope-dvs-v1","preparation_recipe":"nc1-minmax-onehot-v1",
        "pins":{"fit_scope_sha256":"a".repeat(64),"real_test_sha256":"b".repeat(64),"prepared_view_sha256":"c".repeat(64),"groups_sha256":"d".repeat(64),"roster_sha256":"e".repeat(64)},
        "scheduled":[{"cell_sha256":"1".repeat(64),"family_id":0}],"required_families":[0],"cells":[],"selection":null,"privacy":null,
        "artifacts":{"packet":[1],"public_schema":[2],"mandatory_sidecars":[],"shared_decoder_parts":[],"owner_private_parts":[]}}),
    );
    let result = run(&[
        "transfer-report",
        "--request",
        path(&request),
        "--out",
        path(&out),
    ]);
    assert!(!result.status.success());
    assert!(!out.exists());
    assert!(String::from_utf8_lossy(&result.stderr).contains("missing_dvs_role_binding"));
}

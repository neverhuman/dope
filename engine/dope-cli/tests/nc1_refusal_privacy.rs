//! Native CLI privacy/refusal journeys for NC1; literals are public synthetic canaries.
use dope_core::nc1::{
    Cell, DeclaredSchema, FeatureKind, FeatureSpec, InputColumn, InputTable, TargetSpec, Task,
};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn directory() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "dope-nc1-refusal-privacy-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn path(path: &Path) -> &str {
    path.to_str().unwrap()
}

fn write_json(path: &Path, value: &Value) {
    std::fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_dope"))
        .args(["--threads", "1"])
        .args(args)
        .env("PATH", "")
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME")
        .env_remove("PYTHON")
        .env_remove("VIRTUAL_ENV")
        .output()
        .unwrap()
}

fn regression_request() -> Value {
    let table = InputTable::new(
        "a".repeat(64),
        vec![13, 33, 73, 103],
        vec![
            InputColumn {
                name: "CANARY_NC1_HEADER_X".into(),
                values: [0.0, 2.0, 4.0, 1.0e300]
                    .into_iter()
                    .map(Cell::Number)
                    .collect(),
            },
            InputColumn {
                name: "CANARY_NC1_TARGET_HEADER".into(),
                values: [10.0, 20.0, 40.0, 1.0e300]
                    .into_iter()
                    .map(Cell::Number)
                    .collect(),
            },
            InputColumn {
                name: "CANARY_NC1_CATEGORY_HEADER".into(),
                values: [
                    "CANARY_NC1_VOCAB_A",
                    "CANARY_NC1_VOCAB_B",
                    "CANARY_NC1_VOCAB_A",
                    "CANARY_NC1_VOCAB_QUERY",
                ]
                .into_iter()
                .map(|value| Cell::Category(value.into()))
                .collect(),
            },
        ],
    )
    .unwrap();
    let declared = DeclaredSchema {
        features: vec![
            FeatureSpec {
                column: 0,
                kind: FeatureKind::Continuous,
            },
            FeatureSpec {
                column: 2,
                kind: FeatureKind::Nominal,
            },
        ],
        target: TargetSpec {
            column: 1,
            task: Task::Regression,
        },
    };
    json!({"table": table, "declared": declared, "fit_rows": [13, 33, 73, 103]})
}

fn fit_cli(root: &Path, request: &Value, output: &Path) -> Output {
    let request_path = root.join("fit-request.json");
    write_json(&request_path, request);
    run(&[
        "normalize",
        "fit",
        "--request",
        path(&request_path),
        "--out",
        path(output),
    ])
}

fn canaries() -> [&'static str; 16] {
    [
        "CANARY_NC1_HEADER_X",
        "CANARY_NC1_TARGET_HEADER",
        "CANARY_NC1_CATEGORY_HEADER",
        "CANARY_NC1_CLASS_FEATURE",
        "CANARY_NC1_CLASS_TARGET",
        "CANARY_NC1_VOCAB_A",
        "CANARY_NC1_VOCAB_B",
        "CANARY_NC1_VOCAB_QUERY",
        "CANARY_NC1_CLASS_POS",
        "CANARY_NC1_CLASS_NEG",
        "CANARY_NC1_PARSE_HEADER",
        "CANARY_NC1_PARSE_VALUE",
        "CANARY_NC1_UNKNOWN_FIELD",
        "CANARY_NC1_UNKNOWN_VALUE",
        "1e300",
        "1e+300",
    ]
}

fn assert_redacted(text: &str) {
    for canary in canaries() {
        assert!(
            !text.contains(canary),
            "CLI output disclosed planted private canary {canary}"
        );
    }
}

fn assert_refusal(
    output: Output,
    output_path: &Path,
    expected_error: &str,
    invocations: &mut usize,
) {
    *invocations += 1;
    assert!(!output.status.success(), "refusal unexpectedly succeeded");
    assert!(output.stdout.is_empty(), "refusal wrote to stdout");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert_eq!(stderr, format!("error: {expected_error}\n"));
    assert_redacted(&stderr);
    assert!(
        !output_path.exists(),
        "refusal left a partial public artifact"
    );
}

#[test]
fn successful_fit_redacts_public_schema_and_normal_output() {
    let root = directory();
    let out = root.join("regression-out");
    let result = fit_cli(&root, &regression_request(), &out);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stderr.is_empty());
    assert_redacted(&String::from_utf8(result.stdout).unwrap());

    let private = std::fs::read_to_string(out.join("ScaleCard.private.json")).unwrap();
    assert!(private.contains("CANARY_NC1_HEADER_X"));
    assert!(private.contains("CANARY_NC1_TARGET_HEADER"));
    assert!(private.contains("CANARY_NC1_VOCAB_A"));
    let card: Value = serde_json::from_str(&private).unwrap();
    assert_eq!(
        card["features"][0]["state"]["numeric"]["hi"].as_f64(),
        Some(1.0e300)
    );

    let public_bytes = std::fs::read(out.join("schema.public.json")).unwrap();
    let public_text = String::from_utf8(public_bytes.clone()).unwrap();
    assert_redacted(&public_text);
    let public: Value = serde_json::from_slice(&public_bytes).unwrap();
    assert_eq!(public["schema"], "dope.nc1-public/v1");
    assert_eq!(public["columns"], 5); // numeric + three fit levels + OTHER
    assert_eq!(public["target"], "regression");
    for private_key in ["names", "features", "declared", "lo", "hi", "median"] {
        assert!(public.get(private_key).is_none());
    }

    let binary_table = InputTable::new(
        "b".repeat(64),
        vec![17, 37, 77, 117],
        vec![
            InputColumn {
                name: "CANARY_NC1_CLASS_FEATURE".into(),
                values: [0.0, 1.0, 2.0, 3.0].into_iter().map(Cell::Number).collect(),
            },
            InputColumn {
                name: "CANARY_NC1_CLASS_TARGET".into(),
                values: [
                    "CANARY_NC1_CLASS_NEG",
                    "CANARY_NC1_CLASS_POS",
                    "CANARY_NC1_CLASS_NEG",
                    "CANARY_NC1_CLASS_POS",
                ]
                .into_iter()
                .map(|value| Cell::Category(value.into()))
                .collect(),
            },
        ],
    )
    .unwrap();
    let binary = json!({
        "table": binary_table,
        "declared": DeclaredSchema {
            features: vec![FeatureSpec { column: 0, kind: FeatureKind::Continuous }],
            target: TargetSpec {
                column: 1,
                task: Task::Binary { positive_label: "CANARY_NC1_CLASS_POS".into() },
            },
        },
        "fit_rows": [17, 37, 77, 117]
    });
    let class_out = root.join("binary-out");
    let class_result = fit_cli(&root, &binary, &class_out);
    assert!(
        class_result.status.success(),
        "{}",
        String::from_utf8_lossy(&class_result.stderr)
    );
    assert!(class_result.stderr.is_empty());
    let class_private = std::fs::read_to_string(class_out.join("ScaleCard.private.json")).unwrap();
    assert!(class_private.contains("CANARY_NC1_CLASS_POS"));
    assert!(class_private.contains("CANARY_NC1_CLASS_NEG"));
    let class_public = std::fs::read_to_string(class_out.join("schema.public.json")).unwrap();
    assert_redacted(&class_public);
    assert_redacted(&String::from_utf8(class_result.stdout).unwrap());
}

#[test]
fn malformed_and_unknown_request_refusals_are_redacted_and_atomic() {
    let root = directory();
    let malformed = root.join("malformed.json");
    std::fs::write(
        &malformed,
        br#"{"CANARY_NC1_PARSE_HEADER":"CANARY_NC1_PARSE_VALUE""#,
    )
    .unwrap();
    let malformed_out = root.join("malformed-out");
    let malformed_result = run(&[
        "normalize",
        "fit",
        "--request",
        path(&malformed),
        "--out",
        path(&malformed_out),
    ]);

    let mut unknown = regression_request();
    unknown["CANARY_NC1_UNKNOWN_FIELD"] = json!("CANARY_NC1_UNKNOWN_VALUE");
    let unknown_path = root.join("unknown.json");
    write_json(&unknown_path, &unknown);
    let unknown_out = root.join("unknown-out");
    let unknown_result = run(&[
        "normalize",
        "fit",
        "--request",
        path(&unknown_path),
        "--out",
        path(&unknown_out),
    ]);

    let mut invocations = 0;
    assert_refusal(
        malformed_result,
        &malformed_out,
        "invalid_normalization_request",
        &mut invocations,
    );
    assert_refusal(
        unknown_result,
        &unknown_out,
        "invalid_normalization_request",
        &mut invocations,
    );
    assert_eq!(
        invocations, 2,
        "every parser refusal must invoke the native CLI"
    );
}

#[test]
fn invalid_role_and_public_schema_refusals_are_redacted_and_atomic() {
    let root = directory();
    let mut invalid_role = regression_request();
    invalid_role["declared"]["features"][0]["column"] = json!(1);
    let role_path = root.join("invalid-role.json");
    write_json(&role_path, &invalid_role);
    let role_out = root.join("invalid-role-out");
    let role_result = run(&[
        "normalize",
        "fit",
        "--request",
        path(&role_path),
        "--out",
        path(&role_out),
    ]);

    let seed_out = root.join("seed-out");
    let seed_result = fit_cli(&root, &regression_request(), &seed_out);
    assert!(
        seed_result.status.success(),
        "{}",
        String::from_utf8_lossy(&seed_result.stderr)
    );
    let mut card: Value =
        serde_json::from_slice(&std::fs::read(seed_out.join("ScaleCard.private.json")).unwrap())
            .unwrap();
    card["public"]["columns"] = json!(0);
    let bad_card_path = root.join("invalid-public-schema.private.json");
    write_json(&bad_card_path, &card);
    let apply_request = root.join("apply.json");
    write_json(
        &apply_request,
        &json!({"table": regression_request()["table"], "rows": [13]}),
    );
    let schema_out = root.join("invalid-schema-out");
    let schema_result = run(&[
        "normalize",
        "apply",
        "--card",
        path(&bad_card_path),
        "--request",
        path(&apply_request),
        "--out",
        path(&schema_out),
    ]);

    let mut invocations = 0;
    assert_refusal(
        role_result,
        &role_out,
        "invalid_feature_role:column=1",
        &mut invocations,
    );
    assert_refusal(
        schema_result,
        &schema_out,
        "invalid_public_schema",
        &mut invocations,
    );
    assert_eq!(
        invocations, 2,
        "each semantic refusal must invoke the native CLI"
    );
}

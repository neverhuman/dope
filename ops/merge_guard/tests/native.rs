//! Language and OS boundary cases beyond the original Python scenarios.
use super::super::*;
use github::{Api, Transport};
use serde_json::json;
use std::os::unix::fs::PermissionsExt;
use std::time::{Duration, Instant};

fn scratch(label: &str) -> std::path::PathBuf {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target/vnext/20261014/guard-native-port/process-fixtures");
    let dir = root.join(format!(
        "{label}-{}-{}",
        std::process::id(),
        timestamp::now().unwrap().replace(':', "_")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
fn executable(dir: &std::path::Path, source: &str) -> std::path::PathBuf {
    let path = dir.join("fake gh with spaces");
    std::fs::write(&path, source).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
    path
}
#[test]
fn version_one_canonical_digests_match_python_unicode_and_numbers() {
    let rows: Value = serde_json::from_str(include_str!("../fixtures/canonical-v1.json")).unwrap();
    for (i, row) in array(&rows).unwrap().iter().enumerate() {
        let input: Value = serde_json::from_str(text(&row["input_json"]).unwrap()).unwrap();
        assert_eq!(
            json::canonical(&input).unwrap(),
            row["canonical"].as_str().unwrap(),
            "case{i}"
        );
        assert_eq!(
            json::digest(&input).unwrap(),
            row["sha256"].as_str().unwrap(),
            "case{i}"
        );
    }
}
#[test]
fn actual_transport_preserves_literal_argv_without_a_shell_command() {
    let dir = scratch("argv");
    let fake = executable(&dir, "#!/bin/sh\nprintf '%s\\n' \"$@\"\n");
    let endpoint = "repos/fixture/pulls?value=literal; quoted ' é";
    let argv = vec![
        "rtk".into(),
        "proxy".into(),
        fake.to_string_lossy().into_owned(),
        "api".into(),
        "--method".into(),
        "GET".into(),
        endpoint.into(),
        "--paginate".into(),
        "--slurp".into(),
    ];
    let response = github::ProcessTransport::default().run(&argv).unwrap();
    assert_eq!(response.code, 0);
    assert_eq!(
        response.stdout.lines().collect::<Vec<_>>(),
        ["api", "--method", "GET", endpoint, "--paginate", "--slurp"]
    );
}
#[test]
fn request_timeout_terminates_only_its_owned_child_group() {
    let dir = scratch("timeout");
    let fake = executable(&dir, "#!/bin/sh\nexec sleep 2\n");
    let mut unrelated = std::process::Command::new("sleep")
        .arg("2")
        .spawn()
        .unwrap();
    let mut transport = github::ProcessTransport {
        limit: Duration::from_millis(40),
    };
    let start = Instant::now();
    let result = transport.run(&[
        "rtk".into(),
        "proxy".into(),
        fake.to_string_lossy().into_owned(),
        "api".into(),
        "--method".into(),
        "GET".into(),
        "user".into(),
    ]);
    assert_eq!(result.err(), Some(Refused("api_unavailable")));
    assert!(start.elapsed() < Duration::from_secs(1));
    assert!(unrelated.try_wait().unwrap().is_none());
    unrelated.kill().unwrap();
    unrelated.wait().unwrap();
    assert_eq!(
        github::ProcessTransport::default().limit,
        Duration::from_secs(45)
    );
}
#[test]
fn actual_malformed_response_never_qualifies_or_echoes_payload() {
    let dir = scratch("malformed");
    let fake = executable(&dir, "#!/bin/sh\nprintf '%s' 'malformed payload marker'\n");
    let mut api = github::GitHub::new(
        fake.to_string_lossy().into_owned(),
        github::ProcessTransport::default(),
    );
    assert_eq!(
        api.get("user", false, false),
        Err(Refused("api_unavailable"))
    );
    assert_eq!(api.calls()[0]["method"], "GET");
    assert!(!api.calls().to_string().contains("payload marker"));
}
#[test]
fn malformed_receipt_file_is_a_safe_json_refusal() {
    use clap::Parser;
    let dir = scratch("receipt");
    let path = dir.join("malformed.json");
    std::fs::write(&path, "private malformed payload marker").unwrap();
    let args = crate::Cli::try_parse_from([
        "guard",
        "--base",
        "b",
        "--pr",
        "1",
        "--source",
        "a",
        "--review-receipt",
        path.to_str().unwrap(),
    ])
    .unwrap();
    let result = crate::execute(args);
    assert_eq!(result["reason"], "receipt_unavailable");
    assert!(!result.to_string().contains("payload marker"));
}
fn baseline() -> (Value, Value, Value) {
    let cases: Value = serde_json::from_str(include_str!(
        "../fixtures/test_qualified_classic_and_ruleset_controls.json"
    ))
    .unwrap();
    (
        cases[0]["args"][0].clone(),
        cases[0]["args"][1].clone(),
        cases[0]["args"][2].clone(),
    )
}
#[test]
fn missing_or_mistyped_remote_fields_fail_closed_without_panics() {
    let (original, pins, receipt) = baseline();
    for replacement in [Value::Null, json!([]), json!("bad"), json!(false), json!(1)] {
        for key in [
            "before",
            "source_commit",
            "reviews",
            "runs",
            "commits",
            "workflow_definitions",
        ] {
            let mut evidence = original.clone();
            evidence[key] = replacement.clone();
            assert!(
                evidence::evaluate_preflight(&evidence, &pins, &receipt).is_err(),
                "{key}: {replacement}"
            );
        }
    }
    for app in [json!(true), json!(15368.0), json!("15368"), Value::Null] {
        let mut evidence = original.clone();
        evidence["before"]["controls"]["protection"]["required_status_checks"]["checks"][0]["app_id"] =
            app;
        evidence["after"] = evidence["before"].clone();
        assert_eq!(
            evidence::evaluate_preflight(&evidence, &pins, &receipt).err(),
            Some(Refused("control_unavailable"))
        );
    }
}
#[test]
fn review_body_hash_uses_utf8_and_unicode_word_boundaries() {
    let (mut evidence, pins, mut receipt) = baseline();
    let body = format!(
        "{}\nVerified café 雪 🚀; source {}.",
        evidence["reviews"][0]["body"].as_str().unwrap(),
        pins["source_sha"].as_str().unwrap().to_uppercase()
    );
    receipt["body_sha256"] = json!(json::hash_bytes(body.as_bytes()));
    evidence["reviews"][0]["body"] = json!(body);
    assert!(evidence::evaluate_preflight(&evidence, &pins, &receipt).is_ok());
    let body = format!(
        "{}\nOld source BBBBBBBBBBBB.",
        evidence["reviews"][0]["body"].as_str().unwrap()
    );
    receipt["body_sha256"] = json!(json::hash_bytes(body.as_bytes()));
    evidence["reviews"][0]["body"] = json!(body);
    assert_eq!(
        evidence::evaluate_preflight(&evidence, &pins, &receipt).err(),
        Some(Refused("approval_body_mismatch"))
    );
}

use super::{badge, focus};
use serde_json::{Value, json};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
fn folder() -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(format!(
            "target/audit-proof-tests/{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
    fs::create_dir_all(&path).unwrap();
    path
}
fn report() -> Value {
    json!({"auditor_version":"1.7.1","scope":{"mode":"full","paths":[]},
    "git":{"head":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","mode":"full","dirty_worktree":false},
    "dirty_worktree":false,"generated_at":"10000","score":88,"caps_applied":[],"findings":[],
    "decision":{"status":"pass","passed":true,"minimum_score":85}})
}
const HEAD: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
#[test]
fn deterministic_fresh_badge_and_failed_states() {
    let good = report();
    let (body, status) = badge::render(&good, HEAD, 10001).unwrap();
    assert_eq!(status, "pass");
    assert_eq!(body, badge::render(&good, HEAD, 10002).unwrap().0);
    for (field, value) in [
        ("caps_applied", json!(["cap"])),
        ("score", json!(84)),
        ("findings", json!([{"severity":"high"}])),
        (
            "decision",
            json!({"status":"fail","passed":false,"minimum_score":85}),
        ),
    ] {
        let mut bad = good.clone();
        bad[field] = value;
        let (body, status) = badge::render(&bad, HEAD, 10001).unwrap();
        assert_eq!(status, "fail");
        assert!(!String::from_utf8(body).unwrap().contains("#22863a"));
    }
}
#[test]
fn dirty_report_is_gray_not_passing_head() {
    let mut value = report();
    value["dirty_worktree"] = json!(true);
    value["git"]["dirty_worktree"] = json!(true);
    let (body, status) = badge::render(&value, HEAD, 10001).unwrap();
    assert_eq!(status, "local");
    assert!(!String::from_utf8(body).unwrap().contains("#22863a"));
}
#[test]
fn malformed_partial_wrong_head_and_stale_reports_refuse() {
    for (field, value) in [
        ("scope", json!({"mode":"full","paths":["rust/"]})),
        ("score", json!(true)),
        ("score", json!(101)),
        ("score", json!("<script>")),
        ("findings", json!([{}])),
        ("decision", json!({})),
        ("caps_applied", Value::Null),
        ("auditor_version", json!("old")),
    ] {
        let mut bad = report();
        bad[field] = value;
        assert!(badge::render(&bad, HEAD, 10001).is_none());
    }
    for (head, now) in [
        (HEAD, 18000),
        (HEAD, 9999),
        ("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", 10001),
    ] {
        assert!(badge::render(&report(), head, now).is_none());
    }
}
#[test]
fn missing_report_writes_gray_and_snapshot_drift_fails() {
    let root = folder();
    let input = root.join("input.json");
    let output = root.join("score.svg");
    let check = root.join("snapshot.svg");
    assert_eq!(
        badge::write_at(&input, HEAD, &output, None, 10001).unwrap(),
        1
    );
    assert!(fs::read_to_string(&output).unwrap().contains("unverified"));
    fs::write(&input, serde_json::to_vec(&report()).unwrap()).unwrap();
    fs::write(&check, b"stale").unwrap();
    assert_eq!(
        badge::write_at(&input, HEAD, &output, Some(&check), 10001).unwrap(),
        1
    );
    fs::copy(&output, &check).unwrap();
    assert_eq!(
        badge::write_at(&input, HEAD, &output, Some(&check), 10001).unwrap(),
        0
    );
}
fn fixture() -> PathBuf {
    let root = folder();
    for name in [
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "rust-toolchain.toml",
        "rust/lib.rs",
        "schemas/golden.json",
    ] {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"fixture").unwrap();
    }
    root
}
fn exercise(outputs: &[(i32, &str)], mutation: Option<&str>) -> (i32, Value, usize) {
    let root = fixture();
    let mut calls = 0;
    let (code, path) = focus::run_with(&root, "metrics::tests", |argv, log| {
        assert_eq!(argv[5], "metrics::tests");
        let (code, text) = outputs[calls];
        calls += 1;
        fs::write(log, text)?;
        if calls == outputs.len() {
            match mutation {
                Some("source") => fs::write(root.join("rust/lib.rs"), b"changed")?,
                Some("golden") => fs::write(root.join("schemas/golden.json"), b"changed")?,
                Some("missing") => fs::remove_file(root.join("Cargo.lock"))?,
                _ => (),
            }
        }
        Ok(code)
    })
    .unwrap();
    (
        code,
        serde_json::from_slice(&fs::read(path.join("receipt.json")).unwrap()).unwrap(),
        calls,
    )
}
#[test]
fn empty_and_ignored_tests_cannot_pass() {
    let (code, _, calls) = exercise(&[(0, "0 tests, 0 benchmarks\n")], None);
    assert_ne!(code, 0);
    assert_eq!(calls, 1);
    let (code, _, _) = exercise(
        &[
            (0, "metrics::tests::one: test\n"),
            (0, "test result: ok. 0 passed; 0 failed; 1 ignored;\n"),
        ],
        None,
    );
    assert_ne!(code, 0);
}
#[test]
fn focused_pass_has_raw_logs_and_source_binding() {
    let (code, receipt, calls) = exercise(
        &[
            (0, "metrics::tests::one: test\n"),
            (0, "test result: ok. 1 passed; 0 failed;\n"),
        ],
        None,
    );
    assert_eq!(code, 0);
    assert_eq!(calls, 2);
    assert_eq!(receipt["status"], "passed");
    assert!(receipt["source_inputs"]["schemas/golden.json"].is_string());
    assert_eq!(receipt["commands"][1]["sha256"].as_str().unwrap().len(), 64);
}
#[test]
fn failed_cargo_exit_is_preserved() {
    let (code, receipt, _) = exercise(
        &[(0, "metrics::tests::one: test\n"), (101, "FAILED\n")],
        None,
    );
    assert_eq!(code, 101);
    assert_eq!(receipt["status"], "failed");
}
#[test]
fn changed_or_missing_source_or_golden_invalidates_pass() {
    for change in ["source", "golden", "missing"] {
        let (code, receipt, _) = exercise(
            &[
                (0, "metrics::tests::one: test\n"),
                (0, "test result: ok. 1 passed; 0 failed;\n"),
            ],
            Some(change),
        );
        assert_ne!(code, 0);
        assert_eq!(receipt["status"], "failed");
    }
}
#[test]
fn warnings_are_not_test_summaries_and_invalid_filters_refuse() {
    assert!(!focus::executed(
        "warning: test result: ok. 1 passed; 0 failed;\n"
    ));
    assert!(focus::selected("warning: unit: test\n").is_empty());
    for filter in ["", "--ignored", "x\ny"] {
        assert!(focus::run_with(&fixture(), filter, |_, _| panic!("must not execute")).is_err());
    }
}

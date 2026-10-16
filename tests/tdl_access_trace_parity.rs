//! FOLLOW-L1a: compare real Python native-worker charges and access IDs with Rust.

use std::process::Command;

use dope_kernel::metrics::artifact_charge::{evaluate, verify_sampler_access_trace};
use serde_json::Value;

fn strings(value: &Value) -> Vec<&str> {
    value
        .as_array()
        .expect("inventory list")
        .iter()
        .map(|item| item.as_str().expect("opaque inventory ID"))
        .collect()
}

#[test]
fn native_python_trace_and_exact_runtime_closure_match_rust() {
    let output = Command::new("python3")
        .args([
            "-B",
            "-c",
            "import json;from research.transfer.tests.test_tdl_access_trace import parity_evidence;print(json.dumps(parity_evidence(),allow_nan=False))",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run Python native trace fixture");
    assert!(
        output.status.success(),
        "Python native trace fixture failed"
    );
    let evidence: Value = serde_json::from_slice(&output.stdout).expect("trace evidence JSON");
    let receipt = &evidence["receipt"];
    assert_eq!(receipt["status"], "passed");
    assert_eq!(receipt["trace_kind"], "linux_strace");
    assert_eq!(receipt["coverage"], "all_threads_and_native_file_opens");
    assert_eq!(receipt["citable"], false);
    assert_eq!(evaluate(&evidence["manifest"]), receipt["byte_accounting"]);

    let sections = strings(&receipt["declared_payload"]);
    let closure = strings(&receipt["declared_closure"]);
    let reads: Vec<&str> = receipt["events"]
        .as_array()
        .expect("observed events")
        .iter()
        .filter(|event| event["operation"] == "read")
        .map(|event| event["inventory_id"].as_str().expect("read ID"))
        .collect();
    assert!(reads.contains(&"payload") && reads.contains(&"prior"));
    verify_sampler_access_trace(&sections, &closure, &reads).expect("all actual reads charged");

    let cases = evidence["admission"].as_array().expect("admission cases");
    assert_eq!(cases.len(), 6);
    for case in cases {
        let observed = strings(&case["observed"]);
        match verify_sampler_access_trace(&sections, &closure, &observed) {
            Ok(()) => assert!(case["reason"].is_null(), "Python rejected Rust admission"),
            Err(error) => assert_eq!(case["reason"], error.reason, "trace reason parity"),
        }
    }
}

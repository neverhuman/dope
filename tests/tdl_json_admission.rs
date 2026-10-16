//! T-122 blocker 8: share the strict JSON admission corpus with Python.

use std::collections::BTreeSet;

use dope_kernel::metrics::strict_parse_json;
use serde_json::Value;

#[test]
fn strict_parser_matches_all_nine_json_admission_goldens() {
    // Parse the fixture envelope independently of the parser under test.
    let suite: Value = serde_json::from_str(include_str!("golden/t122-json-admission.json"))
        .expect("JSON admission fixture must be valid JSON");
    assert_eq!(suite["schema_version"], 1);
    assert_eq!(suite["citable"], false);
    let cases = suite["cases"].as_array().expect("cases must be an array");
    assert_eq!(cases.len(), 9, "all nine shared parser cases are required");

    let mut ids = BTreeSet::new();
    let mut mismatches = Vec::new();
    for case in cases {
        let id = case["case_id"].as_str().expect("case_id must be a string");
        assert!(ids.insert(id), "duplicate parser case_id");
        let input = case["input_json"]
            .as_str()
            .expect("input_json must be a string");
        let actual = strict_parse_json(input);
        match (case.get("expected_error"), case.get("expected_value")) {
            (Some(expected), None) => {
                let expected = expected.as_str().expect("expected_error must be a string");
                match actual {
                    Err(error) if error.reason == expected && error.status == "failed" => {}
                    Err(error) => mismatches.push(format!(
                        "{id}: expected failed/{expected}, got {}/{}",
                        error.status, error.reason
                    )),
                    Ok(_) => mismatches.push(format!(
                        "{id}: expected {expected}, but the parser accepted the input"
                    )),
                }
            }
            (None, Some(expected)) => match actual {
                Ok(value) if value == *expected => {}
                Ok(_) => mismatches.push(format!("{id}: value differs from expected_value")),
                Err(error) => mismatches.push(format!(
                    "{id}: expected acceptance, got {}/{}",
                    error.status, error.reason
                )),
            },
            _ => panic!("each parser case requires exactly one expectation"),
        }
    }

    // Collect failures so every case runs even if the first rejection differs.
    assert!(
        mismatches.is_empty(),
        "checked all {} parser cases; {} mismatches:\n{}",
        cases.len(),
        mismatches.len(),
        mismatches.join("\n")
    );
}

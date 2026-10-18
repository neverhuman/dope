//! Golden parity runner across `tests/golden/{transfer,aggregate,drivers,frontier,charge}/*.json` (T-037).

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{
    GOLDEN_TOLERANCE, MetricError, MetricResult, aggregate, artifact_charge, drivers, frontier,
    require, strict_parse_json, transfer, values_close,
};

pub const COMPONENTS: &[&str] = &["transfer", "aggregate", "drivers", "frontier", "charge"];

/// Historical replay must never dispatch candidate T90 semantics.
pub fn run_case(case: &Value) -> MetricResult<Value> {
    require(case["schema_version"] == 1, "invalid_golden_schema")?;
    run_versioned_case(case)
}

pub fn run_versioned_case(case: &Value) -> MetricResult<Value> {
    let obj = case
        .as_object()
        .ok_or(MetricError::new("invalid_golden_schema"))?;
    require(
        matches!(
            obj.get("schema_version").and_then(Value::as_i64),
            Some(1 | 2)
        ) && !obj["schema_version"].is_f64(),
        "invalid_golden_schema",
    )?;
    require(
        obj.get("citable").and_then(Value::as_bool) == Some(false),
        "golden_must_be_noncitable",
    )?;
    let case_id = obj
        .get("case_id")
        .and_then(Value::as_str)
        .ok_or(MetricError::new("invalid_case_id"))?;
    require(!case_id.is_empty(), "invalid_case_id")?;
    let component = obj
        .get("component")
        .and_then(Value::as_str)
        .ok_or(MetricError::new("unknown_component"))?;
    let input = obj.get("input").unwrap_or(&Value::Null);
    if obj["schema_version"] == 2 {
        require(component == "transfer", "invalid_golden_schema")?;
        return Ok(transfer::evaluate_tdl(input));
    }
    match component {
        "transfer" => Ok(transfer::evaluate(input)),
        "aggregate" => Ok(aggregate::evaluate(input)),
        "drivers" => Ok(drivers::evaluate(input)),
        "frontier" => Ok(frontier::evaluate(input)),
        "charge" => Ok(artifact_charge::evaluate(input)),
        _ => Err(MetricError::new("unknown_component")),
    }
}

pub fn discover_cases(root: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for component in COMPONENTS {
        let dir = root.join(component);
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|s| s.to_str()) == Some("json") {
                    paths.push(p);
                }
            }
        }
    }
    paths.sort();
    paths
}

pub fn verify_golden_corpus(root: &Path) -> MetricResult<usize> {
    let paths = discover_cases(root);
    require(!paths.is_empty(), "empty_golden_corpus")?;
    let mut ids = BTreeSet::new();
    for path in &paths {
        let text = fs::read_to_string(path).map_err(|_| MetricError::new("unreadable_golden"))?;
        let case = strict_parse_json(&text)?;
        let case_id = case["case_id"]
            .as_str()
            .ok_or(MetricError::new("invalid_case_id"))?
            .to_owned();
        require(ids.insert(case_id), "duplicate_case_id")?;
        let actual = run_versioned_case(&case)?;
        let expected = case
            .get("expected")
            .ok_or(MetricError::new("invalid_golden_schema"))?;
        if !values_close(&actual, expected, GOLDEN_TOLERANCE) {
            return Err(MetricError::new("golden_mismatch"));
        }
    }
    Ok(paths.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_replay_refuses_t90_cases() {
        let case = strict_parse_json(include_str!(
            "../../tests/golden/transfer/v2_t90_boundary.json"
        ))
        .unwrap();
        assert_eq!(run_case(&case).unwrap_err().reason, "invalid_golden_schema");
        assert_eq!(run_versioned_case(&case).unwrap()["harm_margin"], 0.0);
    }

    #[test]
    fn rust_matches_golden_parity_corpus() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let root = manifest_dir.join("tests/golden");
        let count = verify_golden_corpus(&root).expect("golden parity corpus must pass in Rust");
        assert!(count >= 40, "expected full golden corpus, got {count}");
    }

    #[test]
    fn rust_matches_json_admission_corpus() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = manifest_dir.join("tests/golden/t122-json-admission.json");
        let text = fs::read_to_string(&path).expect("t122-json-admission.json must be readable");
        let fixture = strict_parse_json(&text).expect("fixture outer JSON must parse");
        let cases = fixture["cases"].as_array().expect("cases must be an array");
        assert!(!cases.is_empty(), "expected non-empty admission cases");
        for case in cases {
            let case_id = case["case_id"].as_str().expect("case_id");
            let input_json = case["input_json"].as_str().expect("input_json");
            let res = strict_parse_json(input_json);
            if let Some(expected_err) = case.get("expected_error").and_then(Value::as_str) {
                let err = res
                    .err()
                    .unwrap_or_else(|| panic!("case `{case_id}` should fail"));
                assert_eq!(err.reason, expected_err, "case `{case_id}` reason mismatch");
            } else if let Some(expected_val) = case.get("expected_value") {
                let val = res.unwrap_or_else(|_| panic!("case `{case_id}` should parse"));
                assert_eq!(&val, expected_val, "case `{case_id}` value mismatch");
            } else {
                panic!("case `{case_id}` missing expected_error or expected_value");
            }
        }
    }
}

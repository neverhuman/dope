//! Source event admission preserves exact checks and receiving-main proof.
use super::super::*;
use serde_json::json;

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
fn live_and_collected_definitions_reject_out_of_scope_jobs() {
    let definitions = json!({
        ".github/workflows/ci.yml": include_str!("../../../.github/workflows/ci.yml"),
        ".github/workflows/jankurai.yml": include_str!("../../../.github/workflows/jankurai.yml")
    });
    assert!(runs::validate_collected_definitions(&definitions).is_ok());
    let mut unexpected = definitions.clone();
    let ci = runs::WORKFLOWS[0].0;
    let source = unexpected[ci].as_str().unwrap().to_owned();
    unexpected[ci] = json!(source + "  arm64-portability:\n    runs-on: fixture\n");
    assert_eq!(
        runs::validate_definitions(&unexpected, false).err(),
        Some(Refused("workflow_definition_invalid"))
    );
    assert_eq!(
        runs::validate_collected_definitions(&unexpected).err(),
        Some(Refused("workflow_definition_invalid"))
    );
}

fn main_push(evidence: &mut Value) {
    for (path, _) in runs::WORKFLOWS {
        let definition = evidence["workflow_definitions"][path]
            .as_str()
            .unwrap()
            .replace("  push:\n", "  push:\n    branches:\n      - main\n");
        evidence["workflow_definitions"][path] = json!(definition);
    }
}

fn pr_only(evidence: &mut Value, receipt: &mut Value) {
    evidence["runs"]
        .as_array_mut()
        .unwrap()
        .retain(|run| run["event"] == "pull_request");
    let mut ids: Vec<_> = evidence["runs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|run| run["id"].as_u64().unwrap())
        .collect();
    ids.sort();
    let ids_text = ids.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
    let body = format!(
        "Reviewed source {}; receipt {}; checks {}.",
        receipt["source_sha"].as_str().unwrap(),
        receipt["receipt_locator"].as_str().unwrap(),
        ids_text
    );
    receipt["check_run_ids"] = json!(ids);
    receipt["body_sha256"] = json!(json::hash_bytes(body.as_bytes()));
    evidence["reviews"][0]["body"] = json!(body);
}

#[test]
fn main_push_profile_requires_both_source_pr_workflows_and_exact_jobs() {
    let (mut original, pins, mut receipt) = baseline();
    main_push(&mut original);
    pr_only(&mut original, &mut receipt);
    assert!(evidence::evaluate_preflight(&original, &pins, &receipt).is_ok());
    for name in ["cpu", "paper-clean", "security", "jankurai-audit"] {
        for condition in ["missing", "cancelled", "wrong-source"] {
            let mut candidate = original.clone();
            for run in candidate["runs"].as_array_mut().unwrap() {
                let jobs = run["jobs"].as_array_mut().unwrap();
                if condition == "missing" {
                    jobs.retain(|job| job["name"] != name);
                } else if let Some(job) = jobs.iter_mut().find(|job| job["name"] == name) {
                    if condition == "cancelled" {
                        job["conclusion"] = json!("cancelled");
                    } else {
                        job["head_sha"] = json!("ffffffffffffffffffffffffffffffffffffffff");
                    }
                }
            }
            assert!(
                evidence::evaluate_preflight(&candidate, &pins, &receipt).is_err(),
                "{name} {condition}"
            );
        }
    }
    for index in 0..2 {
        let mut candidate = original.clone();
        candidate["runs"].as_array_mut().unwrap().remove(index);
        assert_eq!(
            evidence::evaluate_preflight(&candidate, &pins, &receipt).err(),
            Some(Refused("missing_workflow"))
        );
    }
}

#[test]
fn historical_unrestricted_profile_still_requires_source_push_proof() {
    let (mut candidate, pins, mut receipt) = baseline();
    pr_only(&mut candidate, &mut receipt);
    assert_eq!(
        evidence::evaluate_preflight(&candidate, &pins, &receipt).err(),
        Some(Refused("missing_workflow"))
    );
}

#[test]
fn unsupported_or_inconsistent_event_filters_fail_closed() {
    let (original, _, _) = baseline();
    for event in ["push", "pull_request"] {
        for body in [
            "    branches:\n      - release\n",
            "    branches-ignore:\n      - main\n",
            "    paths:\n      - unrelated/**\n",
            "    branches:\n      - main\n    paths-ignore:\n      - ops/**\n",
            "    branches:\n      - '${{ inputs.branch }}'\n",
            "   branches:\n     - release\n",
        ] {
            let mut candidate = original.clone();
            let path = runs::WORKFLOWS[0].0;
            candidate["workflow_definitions"][path] = json!(
                candidate["workflow_definitions"][path]
                    .as_str()
                    .unwrap()
                    .replace(&format!("  {event}:\n"), &format!("  {event}:\n{body}"))
            );
            assert_eq!(
                runs::validate_definitions(&candidate["workflow_definitions"], false).err(),
                Some(Refused("workflow_definition_invalid"))
            );
        }
    }
    let mut mixed = original.clone();
    let path = runs::WORKFLOWS[0].0;
    mixed["workflow_definitions"][path] = json!(
        mixed["workflow_definitions"][path]
            .as_str()
            .unwrap()
            .replace("  push:\n", "  push:\n    branches: [main]\n")
    );
    assert_eq!(
        runs::validate_definitions(&mixed["workflow_definitions"], false).err(),
        Some(Refused("workflow_definition_invalid"))
    );
}

#[test]
fn main_push_profile_keeps_receiving_main_checks_and_app_bindings() {
    let cases: Value = serde_json::from_str(include_str!(
        "../fixtures/test_real_merge_commit_tree_base_and_receiving_checks_qualify.json"
    ))
    .unwrap();
    let mut receiving = cases[0]["args"][0].clone();
    let mut prior = cases[0]["args"][1].clone();
    let mut source = prior["evidence"].clone();
    let mut receipt = prior["review_receipt"].clone();
    main_push(&mut source);
    pr_only(&mut source, &mut receipt);
    prior["evidence"] = source;
    prior["review_receipt"] = receipt;
    assert_eq!(
        evidence::evaluate_merge(&receiving, &prior).err(),
        Some(Refused("approval_body_mismatch"))
    );
    receiving["reviews"] = prior["evidence"]["reviews"].clone();
    receiving["reviews_after"] = receiving["reviews"].clone();
    assert_eq!(
        evidence::evaluate_merge(&receiving, &prior),
        Ok("merged_verified")
    );
    let mut missing_cpu = receiving.clone();
    missing_cpu["receiving_runs"][0]["jobs"]
        .as_array_mut()
        .unwrap()
        .retain(|job| job["name"] != "cpu");
    assert_eq!(
        evidence::evaluate_merge(&missing_cpu, &prior).err(),
        Some(Refused("missing_or_duplicate_jobs"))
    );
    let mut foreign_cpu = receiving.clone();
    for job in foreign_cpu["receiving_runs"][0]["jobs"]
        .as_array_mut()
        .unwrap()
    {
        if job["name"] == "cpu" {
            job["head_sha"] = json!("ffffffffffffffffffffffffffffffffffffffff");
        }
    }
    assert_eq!(
        evidence::evaluate_merge(&foreign_cpu, &prior).err(),
        Some(Refused("check_source_mismatch"))
    );
    let mut missing = receiving.clone();
    missing["receiving_runs"].as_array_mut().unwrap().pop();
    assert_eq!(
        evidence::evaluate_merge(&missing, &prior).err(),
        Some(Refused("missing_workflow"))
    );
    let mut canceled = receiving.clone();
    canceled["receiving_runs"][0]["jobs"][0]["conclusion"] = json!("cancelled");
    assert_eq!(
        evidence::evaluate_merge(&canceled, &prior).err(),
        Some(Refused("checks_not_successful"))
    );
    for app in [Value::Null, json!(-1), json!(1)] {
        let mut wrong = prior.clone();
        wrong["evidence"]["before"]["controls"]["protection"]["required_status_checks"]["checks"]
            [0]["app_id"] = app;
        wrong["evidence"]["after"] = wrong["evidence"]["before"].clone();
        assert_eq!(
            evidence::evaluate_merge(&receiving, &wrong).err(),
            Some(Refused("control_unavailable"))
        );
    }
}

#[test]
fn actual_workflows_use_pr_source_and_main_push_with_current_job_inventory() {
    let definitions = json!({
        ".github/workflows/ci.yml": include_str!("../../../.github/workflows/ci.yml"),
        ".github/workflows/jankurai.yml": include_str!("../../../.github/workflows/jankurai.yml")
    });
    assert_eq!(
        runs::validate_definitions(&definitions, false).unwrap(),
        &["pull_request"]
    );
    assert_eq!(
        runs::validate_definitions(&definitions, true).err(),
        Some(Refused("control_unavailable"))
    );
}

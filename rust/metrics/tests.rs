use super::*;

fn candidate_fixture(name: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tdl-v2")
        .join(name);
    strict_parse_json(&std::fs::read_to_string(path).expect("candidate fixture"))
        .expect("strict candidate fixture")
}

#[test]
fn candidate_hash_goldens_and_parent_are_bound() {
    let protocol = candidate::parse_protocol(include_str!(
        "../../contracts/tdl-contract-v2-candidate.json"
    ))
    .unwrap();
    for golden in candidate_fixture("canonical-hashes.json")
        .as_array()
        .unwrap()
    {
        assert_eq!(
            candidate::identity(golden["kind"].as_str().unwrap(), &golden["payload"]).unwrap(),
            golden["identity"]
        );
        assert_eq!(
            strict_parse_json(golden["canonical"].as_str().unwrap()).unwrap(),
            golden["payload"]
        );
    }
    assert_eq!(
        candidate::protocol_id(&protocol).unwrap(),
        protocol["protocol_id"]
    );
    let receipt = candidate_fixture("candidate-result.json");
    assert_eq!(
        candidate::admit_receipt(&protocol, &receipt, false).unwrap(),
        candidate::ReceiptMode::CandidateAdmission
    );
}

#[test]
fn candidate_admission_counterexamples_fail_without_echoing_values() {
    let protocol = candidate::parse_protocol(include_str!(
        "../../contracts/tdl-contract-v2-candidate.json"
    ))
    .unwrap();
    let receipt = candidate_fixture("candidate-result.json");
    for case in candidate_fixture("admission-cases.json")
        .as_array()
        .unwrap()
    {
        let is_protocol = case["target"] == "protocol";
        let mut value = if is_protocol {
            protocol.clone()
        } else {
            receipt.clone()
        };
        if let Some(panel) = case.get("setup_panel") {
            value["panel"] = panel.clone();
        }
        let path = case["path"].as_str().unwrap();
        if case["remove"] == true {
            let (parent, key) = path.rsplit_once('/').unwrap();
            value
                .pointer_mut(parent)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .remove(key);
        } else {
            *value.pointer_mut(path).unwrap() = case["value"].clone();
        }
        if case["resign_identity"] == true {
            let entity = &mut value["identities"]["artifact"];
            let mut body = entity.as_object().unwrap().clone();
            body.remove("id");
            entity["id"] = json!(candidate::identity("artifact", &Value::Object(body)).unwrap());
        }
        let outcome = if is_protocol {
            if case["resign"] == true {
                match candidate::protocol_id(&value) {
                    Ok(digest) => value["protocol_id"] = json!(digest),
                    Err(error) => {
                        assert_eq!(error.reason, case["reason"].as_str().unwrap());
                        continue;
                    }
                }
            }
            candidate::validate_protocol(&value)
        } else {
            candidate::admit_receipt(&protocol, &value, false).map(|_| ())
        };
        let error = outcome.expect_err("candidate counterexample must fail");
        assert_eq!(error.reason, case["reason"].as_str().unwrap());
        assert!(!format!("{error:?}").contains("CANARY_PRIVATE"));
    }
}

#[test]
fn candidate_strict_ingress_and_explicit_legacy_replay_fence() {
    let protocol = candidate::parse_protocol(include_str!(
        "../../contracts/tdl-contract-v2-candidate.json"
    ))
    .unwrap();
    let legacy = candidate_fixture("legacy-replay.json");
    for case in candidate_fixture("invalid-ingress.json")
        .as_array()
        .unwrap()
    {
        let text = case["text"].as_str().unwrap();
        for error in [
            candidate::parse_protocol(text).unwrap_err(),
            candidate::parse_receipt(&protocol, text, false).unwrap_err(),
        ] {
            assert_eq!(error.reason, case["reason"].as_str().unwrap());
            assert!(!format!("{error:?}").contains("CANARY_PRIVATE"));
        }
    }
    assert_eq!(
        candidate::admit_receipt(&protocol, &legacy, false),
        Err(MetricError::new("legacy_receipt_requires_replay"))
    );
    assert_eq!(
        candidate::admit_receipt(&protocol, &legacy, true).unwrap(),
        candidate::ReceiptMode::LegacyReplay
    );
    assert_eq!(
        golden::run_case(&legacy["case"]).unwrap(),
        legacy["case"]["expected"]
    );
    let receipt = candidate_fixture("candidate-result.json");
    assert_eq!(
        candidate::admit_receipt(&protocol, &receipt, true),
        Err(MetricError::new("candidate_not_legacy_receipt"))
    );
}

#[test]
fn tdl_contract_v1_validates_and_rejects_freeze_while_tbd_remains() {
    let raw = include_str!("../../contracts/tdl-contract-v1.json");
    let contract = strict_parse_json(raw).expect("tdl-contract-v1.json must be strict JSON");
    validate_tdl_contract(&contract).expect("draft tdl-contract-v1.json must validate");
    assert_eq!(contract["metric_name"], METRIC_NAME);
    assert_eq!(contract["protocol_frozen"], false);
    assert_eq!(contract["headline_operating_point"], "T90/D80");

    let mut tbds = BTreeMap::new();
    collect_tbd_fields(&contract, "", &mut tbds);
    assert!(
        !tbds.is_empty(),
        "draft contract must record pending TBD decisions"
    );

    let mut prematurely_frozen = contract.clone();
    prematurely_frozen["protocol_frozen"] = json!(true);
    prematurely_frozen["status"] = json!("frozen");
    assert_eq!(
        validate_tdl_contract(&prematurely_frozen),
        Err(MetricError::new("cannot_freeze_contract_with_tbd"))
    );
}

#[test]
fn tdl_report_aggregate_is_recomputable_from_auditor_reports() {
    let rows = json!([
        {"source_family": "s1", "dataset": "d1", "seed": 101, "family": "tree", "auditor": "catboost", "transfer": 0.94},
        {"source_family": "s1", "dataset": "d1", "seed": 101, "family": "linear", "auditor": "elastic_net", "transfer": 0.92},
        {"source_family": "s1", "dataset": "d1", "seed": 211, "family": "tree", "auditor": "catboost", "transfer": 0.93},
        {"source_family": "s1", "dataset": "d1", "seed": 211, "family": "linear", "auditor": "elastic_net", "transfer": 0.91}
    ]);
    let agg = aggregate::evaluate(&json!({
        "required_families": ["tree", "linear"],
        "required_auditors": {"tree": ["catboost"], "linear": ["elastic_net"]},
        "required_seeds": [101, 211],
        "rows": rows,
        "bootstrap_replicates": 200,
        "bootstrap_seed": 1729,
        "alpha": 0.05
    }));
    assert_eq!(agg["status"], "complete");
    assert!((agg["robust_transfer"].as_f64().unwrap() - 0.915).abs() <= 1e-12);
}

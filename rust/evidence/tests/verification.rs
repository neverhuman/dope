#[test]
fn executor_receipt_pins_and_confirmation_fail_closed() {
    let f = Fixture::new();
    let mut policy = f.policy();
    policy.receipt_hash = "a".repeat(64);
    assert!(verify_receipt(UnverifiedReceipt(f.receipt.clone()), &f.root, &policy).is_err());
    policy = f.policy();
    policy.executor_hash = "a".repeat(64);
    assert!(verify_receipt(UnverifiedReceipt(f.receipt.clone()), &f.root, &policy).is_err());
    let mut f = Fixture::new();
    f.receipt["identity"]["partition_role"] = "confirmation".into();
    assert!(f.verify().is_err());
}

#[test]
fn forged_booleans_versions_byte_sums_losses_and_rows_fail() {
    for (field, value) in [
        ("version", json!(true)),
        ("version", json!(2)),
        ("version", json!(1.0)),
        ("sampling_payload_bytes", json!(true)),
        ("sampling_payload_bytes", json!(999)),
        ("loss_unit", json!("target_units")),
        ("projection", json!("sanitized_public")),
        ("release", json!("eligible")),
        ("artifact_sampling_verified", json!(true)),
    ] {
        let mut f = Fixture::new();
        f.receipt[field] = value;
        assert!(f.verify().is_err(), "{field}");
    }
    let mut f = Fixture::new();
    f.receipt["losses"]["synthetic"] = json!(0);
    assert!(f.verify().is_err());
    f = Fixture::new();
    f.receipt["losses"]["real"] = json!(true);
    assert!(f.verify().is_err());
    for field in [
        "produced_rows",
        "consumed_rows",
        "context_rows",
        "query_rows",
    ] {
        let mut f = Fixture::new();
        f.receipt["identity"][field] = json!(999);
        assert!(f.verify().is_err());
    }
}

#[test]
fn dag_cycles_missing_aliases_and_inconsistent_metadata_fail() {
    for mutation in ["cycle", "missing", "alias", "bytes", "kind"] {
        let mut f = Fixture::new();
        let node = f.receipt["dependencies"][0].clone();
        match mutation {
            "cycle" => f.receipt["dependencies"][0]["requires"] = json!([node["sha256"]]),
            "missing" => {
                f.receipt["dependencies"].as_array_mut().unwrap().remove(0);
            }
            "alias" => f.receipt["dependencies"].as_array_mut().unwrap().push(node),
            "bytes" => f.receipt["dependencies"][0]["bytes"] = json!(999),
            _ => f.receipt["dependencies"][0]["kind"] = json!("policy"),
        }
        assert!(f.verify().is_err(), "{mutation}");
    }
}

#[test]
fn unsafe_paths_symlinks_and_diagnostics_do_not_expose_values() {
    for path in [
        "../private-source-value",
        "/private-source-value",
        "./private-source-value",
        "private-source-value\\secret",
    ] {
        let mut f = Fixture::new();
        f.receipt["dependencies"][0]["path"] = path.into();
        let message = f.verify().unwrap_err().to_string();
        assert!(!message.contains("private-source-value"));
    }
    let mut f = Fixture::new();
    let target = f
        .root
        .join(text(&f.receipt["dependencies"][0]["path"]).unwrap());
    std::os::unix::fs::symlink(target, f.root.join("link")).unwrap();
    f.receipt["dependencies"][0]["path"] = "link".into();
    assert!(f.verify().is_err());
}

#[test]
fn partial_cache_and_failure_dispositions_cannot_be_successes() {
    let f = Fixture::new();
    let cache = f.root.join("cache");
    let v = f.verify().unwrap();
    commit_completed_receipt(&cache, &v).unwrap();
    let key = scientific_cache_key(v.identity()).unwrap();
    fs::rename(
        cache.join(format!("{key}.json")),
        cache.join(".staging-interrupted"),
    )
    .unwrap();
    assert!(load_completed_receipt(&cache, &f.root, &f.policy()).is_err());
    for state in [
        "unsupported",
        "timeout",
        "oom",
        "invalid_output",
        "byte_infeasible",
        "protocol_violation",
        "unresolved",
    ] {
        let mut f = Fixture::new();
        f.receipt["execution"] = state.into();
        f.receipt["losses"] = Value::Null;
        f.receipt["paired_evidence_hash"] = Value::Null;
        assert!(f.verify().is_err());
    }
}

#[test]
fn confirmation_verifies_frozen_commitments_and_rejects_substitution() {
    let mut f = Fixture::with_role(true);
    let v = f.verify().unwrap();
    let cache = f.root.join("cache");
    commit_completed_receipt(&cache, &v).unwrap();
    load_completed_receipt(&cache, &f.root, &f.policy()).unwrap();
    let mut policy = f.policy();
    let mut identity = policy.identity.value().clone();
    identity["partition_role"] = "selection".into();
    policy.identity = EvaluationIdentity::new(identity).unwrap();
    assert!(load_completed_receipt(&cache, &f.root, &policy).is_err());
    f.receipt["freeze_hash"] = Value::Null;
    assert!(f.verify().is_err());
}

#[test]
fn draft_contract_preserves_historical_bytes_and_rejects_false_freeze() {
    let c = parse_json(include_bytes!(
        "../../../production/behavior-contract-v1.json"
    ))
    .unwrap();
    validate_behavior_contract(&c).unwrap();
    for (path, expected) in c["historical_contract_sha256"].as_object().unwrap() {
        assert_eq!(
            hash(&fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join(path)).unwrap()),
            expected.as_str().unwrap()
        );
    }
    let mut invalid = c.clone();
    invalid["status"] = "frozen".into();
    invalid["protocol_frozen"] = true.into();
    invalid["confirmation_allowed"] = true.into();
    assert!(validate_behavior_contract(&invalid).is_err());

    let tbds: Vec<(&str, &str)> = c["freeze_requirements"]
        .as_object()
        .unwrap()
        .iter()
        .filter_map(|(k, v)| v.as_str().filter(|s| s.starts_with("TBD-")).map(|s| (k.as_str(), s)))
        .collect();
    assert_eq!(tbds.len(), 13);
    assert!(tbds.iter().all(|(_, s)| matches!(*s, "TBD-D02" | "TBD-D06" | "TBD-D07" | "TBD-D08")));
}

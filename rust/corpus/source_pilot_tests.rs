use super::*;

fn fixture() -> Value {
    parse_json(include_bytes!("../../tests/fixtures/n19-pilot-v2.json")).unwrap()
}
fn bytes(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}
fn reconcile(plan: &Value, attempts: &Value, pin: &str) -> Result<Value> {
    reconcile_source_pilot_v2(&bytes(plan), &bytes(attempts), pin)
}

#[test]
fn exact_p_successor_integer_and_float_fixtures_match_both_canonical_digests() {
    let fixture = fixture();
    for sample in [&fixture, &fixture["floating_fixture"]] {
        let plan = &sample["plan"];
        let pin = sample["expected_plan_digest"].as_str().unwrap();
        assert_eq!(digest(plan).unwrap(), pin);
        let result = reconcile(plan, &sample["attempts"], pin).unwrap();
        assert_eq!(result, sample["expected_summary"]);
        assert_eq!(digest(&result).unwrap(), sample["expected_summary_digest"]);
        assert_eq!(result["failure_denominator"], 6);
        assert_eq!(result["non_success_numerator"], 5);
        assert_eq!(result["execution_error_numerator"], 2);
        assert_eq!(result["planned_independent_sources"], 2);
        for flag in [
            "execution_authorized",
            "scientific_admission",
            "confirmation_allowed",
            "signed_admission",
        ] {
            assert_eq!(result[flag], false);
        }
    }
    assert_eq!(fixture["plan"]["source_groups"][0]["views"], 2);
    assert_eq!(fixture["plan"]["planned_independent_sources"], 2);
}

#[test]
fn missing_and_unavailable_stay_in_denominator_without_becoming_execution_errors() {
    let fixture = fixture();
    let plan = &fixture["plan"];
    let pin = fixture["expected_plan_digest"].as_str().unwrap();
    let missing = reconcile(plan, &json!([]), pin).unwrap();
    assert_eq!(missing["non_success_numerator"], 6);
    assert_eq!(missing["execution_error_numerator"], 0);
    assert_eq!(missing["status_counts"]["not_attempted"], 6);
    let rows = plan["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|job| {
            json!({
        "group":job["group"], "stage":job["stage"], "attempt":job["attempt"],
        "status":"unavailable", "seconds":0 })
        })
        .collect::<Vec<_>>();
    let unavailable = reconcile(plan, &json!(rows), pin).unwrap();
    assert_eq!(unavailable["failure_denominator"], 6);
    assert_eq!(unavailable["non_success_numerator"], 6);
    assert_eq!(unavailable["execution_error_numerator"], 0);
    assert_eq!(unavailable["status_counts"]["unavailable"], 6);
}

#[test]
fn declared_unavailable_sources_and_actual_over_budget_metadata_are_retained() {
    let fixture = fixture();
    let mut plan = fixture["plan"].clone();
    let group = plan["source_groups"][1]["group"].clone();
    plan["source_groups"][1]["availability"] = json!("unavailable");
    plan["source_groups"][1]["partition"] = Value::Null;
    plan["source_groups"][1]["reasons"] = json!(["rights_unknown"]);
    plan["jobs"]
        .as_array_mut()
        .unwrap()
        .retain(|job| job["group"] != group);
    plan["planned_independent_sources"] = json!(1);
    plan["unavailable_sources"] = json!(1);
    plan["planned_attempts"] = json!(3);
    plan["estimated_seconds"] = json!(3);
    let result = reconcile(&plan, &json!([]), &digest(&plan).unwrap()).unwrap();
    assert_eq!(result["unavailable_sources"], 1);
    assert_eq!(result["planned_independent_sources"], 1);
    assert_eq!(result["failure_denominator"], 3);
    assert_eq!(result["non_success_numerator"], 3);
    let mut attempts = fixture["attempts"].clone();
    attempts[0]["seconds"] = json!(21);
    let result = reconcile(
        &fixture["plan"],
        &attempts,
        fixture["expected_plan_digest"].as_str().unwrap(),
    )
    .unwrap();
    assert_eq!(result["budget_exceeded"], true);
    assert_eq!(result["original_seconds_cap"], 20);
    assert_eq!(result["execution_authorized"], false);
}

#[test]
fn trusted_pin_original_cap_and_false_admission_cannot_be_replaced_by_caller() {
    let fixture = fixture();
    let pin = fixture["expected_plan_digest"].as_str().unwrap();
    let mut cap = fixture["plan"].clone();
    cap["total_seconds_cap"] = json!(40);
    assert!(
        reconcile(&cap, &json!([]), pin)
            .unwrap_err()
            .to_string()
            .contains("pilot_plan_digest_mismatch")
    );
    for flag in [
        "execution_authorized",
        "scientific_admission",
        "confirmation_allowed",
        "signed_admission",
    ] {
        let mut plan = fixture["plan"].clone();
        plan[flag] = json!(true);
        assert!(reconcile(&plan, &json!([]), &digest(&plan).unwrap()).is_err());
    }
    let mut plan = fixture["plan"].clone();
    plan["private_source_name"] = json!("sentinel-secret-source");
    let error = reconcile(&plan, &json!([]), &digest(&plan).unwrap())
        .unwrap_err()
        .to_string();
    assert!(error.contains("pilot_plan_invalid") && !error.contains("sentinel-secret-source"));
    plan = fixture["plan"].clone();
    plan["version"] = json!(1);
    assert!(reconcile(&plan, &json!([]), &digest(&plan).unwrap()).is_err());
}

#[test]
fn whole_group_split_unknown_rights_and_seed_types_refuse_forged_shape() {
    let fixture = fixture();
    for (field, value) in [
        ("partition", json!("final_query")),
        ("reasons", json!(["rights_unknown"])),
    ] {
        let mut plan = fixture["plan"].clone();
        plan["source_groups"][0][field] = value;
        assert!(reconcile(&plan, &json!([]), &digest(&plan).unwrap()).is_err());
    }
    for seed in [json!(true), json!(0.0), json!(9007199254740992u64)] {
        let mut plan = fixture["plan"].clone();
        plan["seed"] = seed;
        assert!(
            reconcile(
                &plan,
                &json!([]),
                fixture["expected_plan_digest"].as_str().unwrap()
            )
            .is_err()
        );
    }
}

#[test]
fn bounded_job_and_observation_admission_rejects_before_native_collection() {
    let fixture = fixture();
    let pin = fixture["expected_plan_digest"].as_str().unwrap();
    let mut plan = fixture["plan"].clone();
    plan["jobs"] = json!(vec![plan["jobs"][0].clone(); 4097]);
    assert!(
        reconcile(&plan, &json!([]), pin)
            .unwrap_err()
            .to_string()
            .contains("pilot_job_budget_exceeded")
    );
    let attempts = json!(vec![fixture["attempts"][0].clone(); 4097]);
    assert!(
        reconcile(&fixture["plan"], &attempts, pin)
            .unwrap_err()
            .to_string()
            .contains("pilot_attempts_invalid")
    );
    for (field, value) in [
        ("attempt", json!(true)),
        ("status", json!("not_attempted")),
        ("seconds", json!(-1)),
        ("group", json!("0".repeat(64))),
    ] {
        let mut attempts = fixture["attempts"].clone();
        attempts[0][field] = value;
        assert!(reconcile(&fixture["plan"], &attempts, pin).is_err());
    }
    let mut attempts = fixture["attempts"].clone();
    let repeated = attempts[0].clone();
    attempts.as_array_mut().unwrap().push(repeated);
    assert!(reconcile(&fixture["plan"], &attempts, pin).is_err());
}

#[test]
fn duplicate_keys_bytes_and_numeric_overflow_have_static_refusals() {
    assert!(
        reconcile_source_pilot_v2(b"{\"format\":1,\"format\":2}", b"[]", &"0".repeat(64)).is_err()
    );
    let bytes = vec![b' '; MAX_PILOT_DOCUMENT_BYTES + 1];
    assert!(
        reconcile_source_pilot_v2(&bytes, b"[]", &"0".repeat(64))
            .unwrap_err()
            .to_string()
            .contains("pilot_document_bytes_exceeded")
    );
    let fixture = fixture();
    let mut attempts = fixture["attempts"].clone();
    for row in attempts.as_array_mut().unwrap() {
        row["seconds"] = json!(f64::MAX);
    }
    assert!(
        reconcile(
            &fixture["plan"],
            &attempts,
            fixture["expected_plan_digest"].as_str().unwrap()
        )
        .unwrap_err()
        .to_string()
        .contains("pilot_attempt_cost_invalid")
    );
}

#[test]
fn read_only_file_entry_preserves_counts_and_refuses_symlink_or_oversize() {
    let fixture = fixture();
    let pin = fixture["expected_plan_digest"].as_str().unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/native-pilot-v2")
        .join(std::process::id().to_string());
    std::fs::create_dir_all(&root).unwrap();
    let plan = root.join("plan.json");
    std::fs::write(&plan, bytes(&fixture["plan"])).unwrap();
    let result = inspect_source_pilot_v2(&plan, None, pin).unwrap();
    assert_eq!(result["summary"]["non_success_numerator"], 6);
    let alias = root.join("symlink.json");
    std::os::unix::fs::symlink(&plan, &alias).unwrap();
    assert!(inspect_source_pilot_v2(&alias, None, pin).is_err());
    let oversized = root.join("oversized.json");
    let file = std::fs::File::create(&oversized).unwrap();
    file.set_len(MAX_PILOT_DOCUMENT_BYTES as u64 + 1).unwrap();
    assert!(inspect_source_pilot_v2(&oversized, None, pin).is_err());
}

use dope_core::dvs::{
    self, roles::DeclaredEvidenceRoles, CellOutcome, DataPins, DvsRequest, FailureReason,
};
use serde_json::{json, Value};

fn pins() -> DataPins {
    serde_json::from_value(json!({
        "fit_scope_sha256":"a".repeat(64),
        "real_test_sha256":"b".repeat(64),
        "prepared_view_sha256":"c".repeat(64),
        "groups_sha256":"d".repeat(64),
        "roster_sha256":"e".repeat(64)
    }))
    .unwrap()
}

fn driver(pins: &DataPins, perturbation: &str) -> Value {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/golden/drivers/v2/v2_additive.json"
    ))
    .unwrap();
    let mut evidence = fixture["input"].clone();
    evidence["protocol_version"] = json!("driver-rank-informative-v1");
    evidence["importance_unit"] = json!("declared_group");
    evidence["group_ids"] = json!([0, 1]);
    evidence["query_sha256"] = json!(pins.real_test_sha256);
    evidence["groups_sha256"] = json!(pins.groups_sha256);
    evidence["prepared_view_sha256"] = json!(pins.prepared_view_sha256);
    evidence["perturbation_sha256"] = json!(perturbation);
    evidence
}

fn request(perturbation: &str) -> DvsRequest {
    let pins = pins();
    let roles = DeclaredEvidenceRoles::new(pins.clone(), "f".repeat(64)).unwrap();
    serde_json::from_value(json!({
        "protocol":"dope-dvs-v1",
        "preparation_recipe":"nc1-minmax-onehot-v1",
        "pins":pins.clone(),
        "declared_roles":roles,
        "scheduled":[{"cell_sha256":"1".repeat(64),"family_id":0}],
        "required_families":[0],
        "cells":[{"cell_sha256":"1".repeat(64),"family_id":0,"pins":pins.clone(),"outcome":{
            "status":"complete","null_losses":[10.0,10.0],"real_losses":[2.0,2.0],
            "synthetic_losses":[4.0,4.0],"real_gain_lcb":1.0,"absolute_gain_floor":0.01,
            "relative_gain_floor":0.01,"driver_evidence":driver(&pins,perturbation),
            "wall_seconds":null,"fit_count":null
        }}],
        "selection":null,
        "artifacts":{"packet":[1],"public_schema":[2],"mandatory_sidecars":[],
            "shared_decoder_parts":[],"owner_private_parts":[]},
        "privacy":null
    }))
    .unwrap()
}

#[test]
fn dvs_cell_identity_binding_captures_all_four_validated_pins_per_cell() {
    let first_input = request(&"9".repeat(64));
    let first = dvs::evaluate(&first_input).unwrap();
    let first_binding = first.cells[0].driver_identity_binding.as_ref().unwrap();
    assert_eq!(
        first_binding.state,
        "declared_identity_hashes_shape_and_scope_checked"
    );
    assert_eq!(first_binding.required_policy, first.driver_policy_binding);
    assert_eq!(first_binding.query_sha256, first.pins.real_test_sha256);
    assert_eq!(first_binding.groups_sha256, first.pins.groups_sha256);
    assert_eq!(
        first_binding.prepared_view_sha256,
        first.pins.prepared_view_sha256
    );
    assert_eq!(first_binding.perturbation_sha256, "9".repeat(64));

    let second = dvs::evaluate(&request(&"8".repeat(64))).unwrap();
    assert_eq!(
        second.cells[0]
            .driver_identity_binding
            .as_ref()
            .unwrap()
            .perturbation_sha256,
        "8".repeat(64)
    );
    assert_ne!(
        first_binding.perturbation_sha256,
        second.cells[0]
            .driver_identity_binding
            .as_ref()
            .unwrap()
            .perturbation_sha256
    );

    let mut wrong_policy = request(&"9".repeat(64));
    if let CellOutcome::Complete {
        driver_evidence, ..
    } = &mut wrong_policy.cells[0].outcome
    {
        driver_evidence.as_mut().unwrap()["inference"]["bound_method"] = json!("ratio_only");
    }
    let wrong_policy = dvs::evaluate(&wrong_policy).unwrap();
    assert_eq!(wrong_policy.cells[0].status, "incomplete");
    assert!(wrong_policy.cells[0].driver_identity_binding.is_none());

    let mut missing = request(&"9".repeat(64));
    if let CellOutcome::Complete {
        driver_evidence, ..
    } = &mut missing.cells[0].outcome
    {
        driver_evidence.as_mut().unwrap()["perturbation_sha256"] = Value::Null;
    }
    let absent = dvs::evaluate(&missing).unwrap();
    assert_eq!(absent.cells[0].status, "incomplete");
    assert!(absent.cells[0].driver_identity_binding.is_none());

    let mut malformed = request(&"9".repeat(64));
    if let CellOutcome::Complete {
        driver_evidence, ..
    } = &mut malformed.cells[0].outcome
    {
        driver_evidence.as_mut().unwrap()["perturbation_sha256"] = json!("CANARY_PRIVATE");
    }
    let rejected = dvs::evaluate(&malformed).unwrap();
    assert_eq!(rejected.cells[0].status, "incomplete");
    assert!(rejected.cells[0].driver_identity_binding.is_none());
    assert!(!serde_json::to_string(&rejected)
        .unwrap()
        .contains("CANARY_PRIVATE"));

    let mut no_driver = request(&"9".repeat(64));
    if let CellOutcome::Complete {
        driver_evidence, ..
    } = &mut no_driver.cells[0].outcome
    {
        *driver_evidence = None;
    }
    let absent_driver = dvs::evaluate(&no_driver).unwrap();
    assert_eq!(absent_driver.cells[0].status, "incomplete");
    assert!(absent_driver.cells[0].driver_identity_binding.is_none());

    let mut failed = request(&"9".repeat(64));
    failed.cells[0].outcome = CellOutcome::Failed {
        reason: FailureReason::NativeFailure,
        wall_seconds: None,
        fit_count: None,
    };
    let failed = dvs::evaluate(&failed).unwrap();
    assert_eq!(failed.cells[0].status, "failed");
    assert!(failed.cells[0].driver_identity_binding.is_none());

    let wire = serde_json::to_value(first).unwrap();
    let binding = &wire["cells"][0]["driver_identity_binding"];
    assert_eq!(
        binding["required_policy"]["driver_protocol"],
        "driver-rank-informative-v1"
    );
    assert_eq!(
        binding["required_policy"]["inference_bound_method"],
        "full_family_max_t"
    );
    let schema: Value = serde_json::from_str(include_str!(
        "../../../schemas/dope-dvs-receipt-v1.schema.json"
    ))
    .unwrap();
    let binding_schema =
        &schema["properties"]["cells"]["items"]["properties"]["driver_identity_binding"];
    assert_eq!(
        schema["properties"]["cells"]["items"]["additionalProperties"],
        false
    );
    assert_eq!(binding_schema["anyOf"][1]["additionalProperties"], false);
    assert_eq!(
        binding_schema["anyOf"][1]["required"]
            .as_array()
            .unwrap()
            .len(),
        6
    );
    assert!(schema["properties"]["cells"]["items"]["required"]
        .as_array()
        .unwrap()
        .iter()
        .all(|key| key != "driver_identity_binding"));
}

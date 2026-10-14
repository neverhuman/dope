use dope_kernel::evidence::{
    canonical_json, parse_json, select_development_learner_v2, validate_learner_roles_v2,
    verify_calibration_receipt_v1,
};
use serde_json::{Value, json};
use sha2::Digest;

fn roles() -> Value {
    json!({"format":"dope-learner-roles","version":2,"selection_families":["linear","tree"],"evaluation_required_families":["linear","tree","kumo"],"held_out_families":["kumo"]})
}

fn bytes(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}

fn rebound(value: &mut Value) {
    let object = value.as_object_mut().unwrap();
    object.remove("binding_digest");
    let hash = format!("{:x}", sha2::Sha256::digest(canonical_json(value).unwrap()));
    value["binding_digest"] = json!(hash);
}

#[test]
fn successor_roles_reserve_held_out_families_before_reading_losses() {
    let record = roles();
    let candidates = json!([
        {"learner_id":"linear-a","family_id":"linear","development_loss":0.2},
        {"learner_id":"kumo-best","family_id":"kumo","development_loss":0.01},
        {"learner_id":"kumo-no-loss","family_id":"kumo"},
        {"learner_id":"not-selected","family_id":"other","development_loss":"must_not_read"}
    ]);
    let decision = select_development_learner_v2(&bytes(&record), &bytes(&candidates)).unwrap();
    assert_eq!(decision["selected_learner"], "linear-a");
    assert_eq!(decision["held_out_excluded_from_selection"], true);
    for key in [
        "protocol_frozen",
        "confirmation_allowed",
        "native_certification",
        "qualification",
    ] {
        assert_eq!(decision[key], false);
    }
    let mut selection_not_evaluated = record.clone();
    selection_not_evaluated["evaluation_required_families"] = json!(["kumo"]);
    assert_eq!(
        validate_learner_roles_v2(&bytes(&selection_not_evaluated)).unwrap()["selection_not_evaluated"],
        json!(["linear", "tree"])
    );
}

#[test]
fn roles_refuse_legacy_overlap_unknown_fields_and_duplicate_json_without_changing_v1() {
    for (mutation, reason) in [
        (json!({"version":1}), "legacy_role_identity"),
        (
            json!({"selection_families":["linear","kumo"]}),
            "held_out_selected",
        ),
        (
            json!({"evaluation_required_families":["linear"]}),
            "held_out_not_evaluated",
        ),
        (json!({"protocol_frozen":true}), "invalid_learner_roles"),
        (json!({"version":true}), "invalid_learner_roles"),
    ] {
        let mut record = roles();
        for (k, v) in mutation.as_object().unwrap() {
            record[k] = v.clone();
        }
        assert!(
            validate_learner_roles_v2(&bytes(&record))
                .unwrap_err()
                .to_string()
                .contains(reason)
        );
    }
    let duplicate = String::from_utf8(bytes(&roles())).unwrap().replacen(
        "\"version\":2",
        "\"version\":2,\"version\":1",
        1,
    );
    assert!(validate_learner_roles_v2(duplicate.as_bytes()).is_err());
    let historical = parse_json(include_bytes!("../production/behavior-contract-v1.json")).unwrap();
    dope_kernel::evidence::validate_behavior_contract(&historical).unwrap();
}

#[test]
fn matching_development_calibration_authenticates_without_qualifying_any_claim() {
    let receipt = include_bytes!("fixtures/vnext-successors/development-calibration-v1.json");
    let decision = verify_calibration_receipt_v1(receipt).unwrap();
    assert_eq!(decision["authenticates"], true);
    assert_eq!(decision["reason"], "development_protocol_is_not_n13b");
    for key in ["qualifies", "n13b", "release"] {
        assert_eq!(decision[key], false);
    }
    assert_eq!(decision["calibration"], "unmeasured");
    let mut tampered = parse_json(receipt).unwrap();
    tampered["repetitions"] = json!(121);
    assert_eq!(
        verify_calibration_receipt_v1(&bytes(&tampered)).unwrap()["authenticates"],
        false
    );
    let mut unknown = parse_json(receipt).unwrap();
    unknown["protocol_id"] = json!("unknown_protocol");
    rebound(&mut unknown);
    assert_eq!(
        verify_calibration_receipt_v1(&bytes(&unknown)).unwrap()["reason"],
        "unknown_protocol"
    );
    let mut wrong_digest = parse_json(receipt).unwrap();
    wrong_digest["protocol_digest"] = json!("0".repeat(64));
    rebound(&mut wrong_digest);
    assert_eq!(
        verify_calibration_receipt_v1(&bytes(&wrong_digest)).unwrap()["reason"],
        "protocol_digest_mismatch"
    );
}

#[test]
fn calibration_refuses_caller_assertions_boolean_versions_and_mistyped_counts() {
    let original = parse_json(include_bytes!(
        "fixtures/vnext-successors/development-calibration-v1.json"
    ))
    .unwrap();
    for mutation in [
        json!({"qualifies":true}),
        json!({"calibration":"measured"}),
        json!({"version":true}),
        json!({"version":2}),
        json!({"repetitions":true}),
        json!({"bootstrap_draws":1.0}),
    ] {
        let mut record = original.clone();
        for (k, v) in mutation.as_object().unwrap() {
            record[k] = v.clone();
        }
        rebound(&mut record);
        assert!(verify_calibration_receipt_v1(&bytes(&record)).is_err());
    }
}

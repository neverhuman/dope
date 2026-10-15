use dope_kernel::evidence::{compare_frozen_geometry_v1, frozen_geometry_state_v1};
use serde_json::{Value, json};

fn fixture() -> Value {
    serde_json::from_slice(include_bytes!(
        "fixtures/vnext-successors/frozen-geometry-v1.json"
    ))
    .unwrap()
}

#[test]
fn geometry_native_digest_matches_python_unicode_signed_zero_and_float_golden() {
    let fixture = fixture();
    let bytes = serde_json::to_vec(&fixture["body"]).unwrap();
    let result = frozen_geometry_state_v1(&bytes).unwrap();
    assert_eq!(result["state_digest"], fixture["expected_state_digest"]);
    assert_eq!(result["label_verified"], false);
    assert_eq!(result["qualifies_transfer"], false);
    // Whitespace/key order and spelling of the same floating value preserve
    // the digest. Float/integer scalar types and signed zero stay distinct.
    let spelling = "{\"columns\":[{\"maximum\":1.2500,\"minimum\":-0.0000,\"kind\":\"continuous\"},{\"levels\":[\"α\",\"β\"],\"kind\":\"nominal\"}],\"schema_id\":\"fixture-schema\",\"encoder_id\":\"fixture-encoder\",\"version\":1,\"format\":\"dope-frozen-geometry-state\"}";
    assert_eq!(
        result,
        frozen_geometry_state_v1(spelling.as_bytes()).unwrap()
    );
    let mut positive = fixture["body"].clone();
    positive["columns"][0]["minimum"] = json!(0.0);
    let positive = frozen_geometry_state_v1(&serde_json::to_vec(&positive).unwrap()).unwrap();
    assert_eq!(
        positive["state_digest"],
        fixture["expected_positive_zero_digest"]
    );
    assert_ne!(positive["state_digest"], result["state_digest"]);
}

#[test]
fn geometry_recomputes_actual_order_encoder_scale_and_schema_content() {
    let body = fixture()["body"].clone();
    let original = serde_json::to_vec(&body).unwrap();
    assert_eq!(
        compare_frozen_geometry_v1(&original, &original).unwrap()["shared_geometry"],
        true
    );
    let mut changed = Vec::new();
    let mut v = body.clone();
    v["columns"].as_array_mut().unwrap().reverse();
    changed.push(v);
    let mut v = body.clone();
    v["columns"][1]["levels"].as_array_mut().unwrap().reverse();
    changed.push(v);
    let mut v = body.clone();
    v["columns"][0]["maximum"] = json!(2.0);
    changed.push(v);
    let mut v = body.clone();
    v["encoder_id"] = json!("other-encoder");
    changed.push(v);
    let mut v = body.clone();
    v["schema_id"] = json!("other-schema");
    changed.push(v);
    for v in changed {
        let result =
            compare_frozen_geometry_v1(&original, &serde_json::to_vec(&v).unwrap()).unwrap();
        assert_eq!(result["shared_geometry"], false);
        assert_eq!(result["privacy_result"], false);
        assert_eq!(result["qualifies_transfer"], false);
    }
}

#[test]
fn geometry_refuses_labels_digest_only_claims_duplicates_and_invalid_columns() {
    let body = fixture()["body"].clone();
    let mut changes = Vec::new();
    let mut v = body.clone();
    v["frozen_id"] = json!("same-label");
    changes.push(v);
    let mut v = body.clone();
    v["state_digest"] = fixture()["expected_state_digest"].clone();
    changes.push(v);
    let mut v = body.clone();
    v["version"] = json!(true);
    changes.push(v);
    let mut v = body.clone();
    v["version"] = json!(1.0);
    changes.push(v);
    let mut v = body.clone();
    v["columns"][0]["minimum"] = json!(0);
    changes.push(v);
    let mut v = body.clone();
    v["columns"][0]["minimum"] = json!(1.25);
    changes.push(v);
    let mut v = body.clone();
    v["columns"][1]["levels"] = json!(["α", "α"]);
    changes.push(v);
    let mut v = body;
    v["columns"][0]["minimum"] = json!(false);
    changes.push(v);
    for v in changes {
        assert!(frozen_geometry_state_v1(&serde_json::to_vec(&v).unwrap()).is_err());
    }
    assert!(frozen_geometry_state_v1(b"{\"version\":1,\"version\":1}").is_err());
    assert!(frozen_geometry_state_v1(b"{\"minimum\":NaN}").is_err());
}

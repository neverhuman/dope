use super::*;
use std::process::Command;

#[test]
fn python_wire_fixtures_have_exact_rust_admission_and_receipt_parity() {
    // This is a public, deterministic fixture producer, never a model worker.
    // No vendor runtime, checkpoint or GPU qualification is inferred from it.
    let output = Command::new("python3")
        .args([
            "-B",
            "-m",
            "research.transfer.tests.test_worker_abi",
            "--emit-fixtures",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env("PYTHONPATH", ".")
        .output()
        .expect("python fixture producer");
    assert!(output.status.success(), "fixture producer failed");
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).expect("fixture corpus");
    assert!(cases.len() >= 40);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let request_text = case["request"].as_str().unwrap().as_bytes();
        let untrusted: Value = serde_json::from_slice(request_text).unwrap();
        // Fixed independent custodian fixture pins, never copied from the request.
        let mut pins = serde_json::Map::new();
        for key in [
            "model_sha256",
            "panel_sha256",
            "worker_image_sha256",
            "checkpoint_sha256",
            "prepared_sha256",
            "split_sha256",
            "dependencies_sha256",
            "preprocessing_sha256",
            "recipe_sha256",
        ] {
            pins.insert(key.into(), Value::String(sha(key.as_bytes())));
        }
        pins.insert(
            "runtime_version".into(),
            Value::String("1.2.3+fixture".into()),
        );
        pins.insert("precision".into(), Value::String("float64".into()));
        pins.insert("device".into(), Value::String("cpu".into()));
        let buffers: Buffers = serde_json::from_value(case["buffers"].clone()).unwrap();
        let result = validate_request(request_text, &buffers, &Value::Object(pins));
        if case["expected_request"] == false {
            assert!(result.is_err(), "request accepted: {name}");
            assert!(
                matches!(result.err(), Some(DopeError::Data(code)) if code == "worker_abi_invalid")
            );
            continue;
        }
        let admitted = match result {
            Ok(v) => v,
            Err(_) => panic!("valid request rejected: {name}"),
        };
        assert_eq!(
            admitted.identity(),
            untrusted["request_sha256"].as_str().unwrap()
        );
        assert_eq!(request_bytes(&admitted).unwrap(), request_text);
        let response_buffers: Buffers =
            serde_json::from_value(case["response_buffers"].clone()).unwrap();
        let result = validate_response(
            case["response"].as_str().unwrap().as_bytes(),
            &response_buffers,
            &admitted,
        );
        if case["expected_response"] == true {
            assert_eq!(
                match result {
                    Ok(v) => v,
                    Err(_) => panic!("valid response rejected: {name}"),
                },
                case["expected_receipt"],
                "{name}"
            );
        } else {
            assert!(result.is_err(), "response accepted: {name}");
            assert!(
                matches!(result.err(), Some(DopeError::Data(code)) if code == "worker_abi_invalid")
            );
        }
    }
}

#[test]
fn parser_rejects_duplicates_nonfinite_oversize_and_private_diagnostics() {
    for bytes in [
        br#"{"type":"request","type":"response"}"#.as_slice(),
        br#"{"private_header":"/private/checkpoint","value":NaN}"#,
        b"null",
        b"[]",
        b"{}",
    ] {
        assert!(
            matches!(validate_request(bytes, &Buffers::new(), &Value::Null).err(),
            Some(DopeError::Data(code)) if code == "worker_abi_invalid")
        );
    }
    assert!(
        validate_request(&vec![b' '; MAX_ENVELOPE + 1], &Buffers::new(), &Value::Null).is_err()
    );
}

#[test]
fn schema_is_closed_and_qualification_is_fixture_only() {
    let schema = parse_json(include_bytes!(
        "../../../schemas/model-worker-v1.schema.json"
    ))
    .unwrap();
    for shape in array(&schema["anyOf"]) {
        assert_eq!(shape["additionalProperties"], false);
        assert_eq!(
            array(&shape["required"]).len(),
            shape["properties"].as_object().unwrap().len()
        );
    }
    let response = &schema["anyOf"][1]["properties"];
    assert_eq!(response["qualified"]["const"], false);
    assert_eq!(response["qualification"]["const"], "fixture_only");
}

#[test]
fn runtime_versions_must_be_exact_even_if_owner_pins_match() {
    for version in ["latest", "1.2", "1.2.*", ">=1.2.3", "1.2.3 ", "v1.2.3"] {
        assert!(!runtime_version(&Value::String(version.into())));
    }
    assert!(runtime_version(&Value::String("1.2.3+fixture".into())));
    assert!(!runtime_version(&Value::String("1.2.3".repeat(1024))));
}

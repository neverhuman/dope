#[test]
fn retained_adversarial_fixtures_refuse_with_fixed_codes() {
    let cases: Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/prepared-view-v1/adversarial.json"
    ))
    .unwrap();
    for case in cases.as_array().unwrap() {
        let fixture = match case["fixture"].as_str().unwrap() {
            "regression.json" => {
                include_str!("../../tests/fixtures/prepared-view-v1/regression.json")
            }
            "binary.json" => include_str!("../../tests/fixtures/prepared-view-v1/binary.json"),
            "multiclass.json" => {
                include_str!("../../tests/fixtures/prepared-view-v1/multiclass.json")
            }
            _ => panic!("unknown fixture"),
        };
        let mut base: Value = serde_json::from_str(fixture).unwrap();
        base["metadata"]["provenance"]["implementation_hash"] = json!(implementation_hash());
        let view = PreparedView::from_private_json(&serde_json::to_vec(&base).unwrap()).unwrap();
        let contract = view.freeze().unwrap();
        *base.pointer_mut(case["pointer"].as_str().unwrap()).unwrap() = case["value"].clone();
        let e = validate_input(&serde_json::to_vec(&base).unwrap(), &contract).unwrap_err();
        assert_eq!(e.code, case["reason"].as_str().unwrap());
        assert!(!format!("{e} {e:?}").contains("private_"));
    }
}

#[test]
fn declared_support_boolean_and_category_numeric_types_are_checked() {
    let mut decl = declaration(false);
    decl["columns"][0]["domain"] = json!({"kind":"numeric","lower":-1.0,"upper":2.0});
    decl["columns"][1]["dtype"] = json!("boolean");
    let input = table("-1,0,100\n2,1,200\n", &decl);
    let fit = prepare(&input, &[0, 1], spec(TaskKind::Regression)).unwrap();
    fit.validate().unwrap();
    for (pointer, replacement, code) in [
        ("/rows/0/0", json!(-0.1), "prepared_support_mismatch"),
        ("/rows/0/1", json!(0.5), "prepared_type_mismatch"),
    ] {
        let mut changed = value(&fit);
        *changed.pointer_mut(pointer).unwrap() = replacement;
        assert_eq!(
            PreparedView::from_private_json(&serde_json::to_vec(&changed).unwrap())
                .unwrap_err()
                .code,
            code
        );
    }
    let fit = prepare(
        &table("10,known,100\n20,other,200\n", &declaration(true)),
        &[0, 1],
        spec(TaskKind::Regression),
    )
    .unwrap();
    for code in [-1., 0.5, 2.] {
        let mut changed = value(&fit);
        changed["rows"][0][1] = json!(code);
        assert_eq!(
            PreparedView::from_private_json(&serde_json::to_vec(&changed).unwrap())
                .unwrap_err()
                .code,
            "prepared_oov_rejected"
        );
    }
    for categories in [
        json!(["same", "same"]),
        json!(["é".repeat(2049)]),
        json!([]),
    ] {
        let mut changed = value(&fit);
        changed["metadata"]["columns"][1]["transform"]["categories"] = categories;
        assert_eq!(
            PreparedView::from_private_json(&serde_json::to_vec(&changed).unwrap())
                .unwrap_err()
                .code,
            "invalid_prepared_contract"
        );
    }
}

#[test]
fn payload_metadata_and_numeric_limits_are_bounded() {
    let fit = numeric();
    assert_eq!(
        PreparedView::from_private_json(&vec![b' '; InputLimits::default().max_bytes + 1])
            .unwrap_err()
            .code,
        "prepared_bytes_exceeded"
    );
    let mut invalid = fit.clone();
    invalid.rows[0][0] = f64::NAN;
    assert_eq!(
        invalid.validate().unwrap_err().code,
        "prepared_nonfinite_value"
    );
    invalid.rows[0][0] = f64::INFINITY;
    assert_eq!(
        invalid.validate().unwrap_err().code,
        "prepared_nonfinite_value"
    );
    invalid.rows[0][0] = f64::MAX;
    assert_eq!(
        invalid.validate().unwrap_err().code,
        "prepared_nonfinite_value"
    );
    invalid.rows = vec![vec![0., 0., 0.]; InputLimits::default().max_rows + 1];
    assert_eq!(
        invalid.validate().unwrap_err().code,
        "prepared_dimensions_exceeded"
    );
    let mut huge = fit.metadata.clone();
    huge.columns[0].transform = PreparedTransform::CategoryIndex {
        categories: vec!["x".repeat(MAX_SCHEMA_BYTES)],
    };
    assert_eq!(
        huge.validate().unwrap_err().code,
        "prepared_metadata_bytes_exceeded"
    );
    let mut old_implementation = value(&fit);
    old_implementation["metadata"]["provenance"]["implementation_hash"] = json!("0".repeat(64));
    let old =
        PreparedView::from_private_json(&serde_json::to_vec(&old_implementation).unwrap()).unwrap();
    let contract = old.freeze().unwrap();
    assert_eq!(
        contract.compile_input(&old).unwrap_err().code,
        "prepared_implementation_mismatch"
    );
}

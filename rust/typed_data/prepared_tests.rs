use super::*;
use serde_json::{Value, json};

include!("prepared_adversarial_tests.rs");

fn declaration(category: bool) -> Value {
    json!({"format":"dope-table-schema","version":1,"columns":[
        {"id":"c0001","source_name":"private_measurement","role":"feature","dtype":"continuous"},
        {"id":"c0002","source_name":"private_kind","role":"feature","dtype":if category {"nominal"} else {"continuous"}},
        {"id":"c0003","source_name":"private_outcome","role":"target","dtype":"continuous","units":"private_unit"}
    ]})
}

fn table(body: &str, declaration: &Value) -> TypedTable {
    let schema =
        TableSchemaV1::from_private_json(&serde_json::to_vec(declaration).unwrap()).unwrap();
    let csv = format!("private_measurement,private_kind,private_outcome\n{body}");
    TypedTable::read_csv(csv.as_bytes(), schema, InputLimits::default()).unwrap()
}

fn spec(task: TaskKind) -> PrepareSpec {
    PrepareSpec {
        task,
        horizon: Horizon {
            steps: 1,
            unit: HorizonUnit::Rows,
        },
        group_map: vec![
            PreparedGroup {
                id: 1,
                column_ids: vec![1],
            },
            PreparedGroup {
                id: 2,
                column_ids: vec![2],
            },
        ],
        target: if task == TaskKind::Regression {
            PreparedTarget::Regression {
                units: "private_unit".into(),
            }
        } else {
            PreparedTarget::Discrete {
                class_order: if task == TaskKind::Binary {
                    vec![0, 1]
                } else {
                    vec![2, 0, 1]
                },
            }
        },
        entity_digest: "a".repeat(64),
        split_digest: "b".repeat(64),
        rights: PreparedRights {
            state: RightsState::ResearchUnassessed,
            evidence_digest: None,
        },
    }
}

fn numeric() -> PreparedView {
    prepare(
        &table("10,20,100\n20,40,200\n1000,2000,300\n", &declaration(false)),
        &[0, 1],
        spec(TaskKind::Regression),
    )
    .unwrap()
}

fn value(view: &PreparedView) -> Value {
    serde_json::from_slice(&view.private_json().unwrap()).unwrap()
}

#[test]
fn fit_only_and_frozen_adapter_preserve_original_units_without_clipping() {
    let input = table("10,20,100\n20,40,200\n1000,2000,300\n", &declaration(false));
    let fit = prepare(&input, &[0, 1], spec(TaskKind::Regression)).unwrap();
    assert_eq!(fit.rows(), &[vec![0., 0., 100.], vec![1., 1., 200.]]);
    let contract = fit.freeze().unwrap();
    let query = contract.apply(&input, &[2]).unwrap();
    assert_eq!(query.rows(), &[vec![99., 99., 300.]]);
    assert!(query.metadata() == fit.metadata());
    assert_eq!(contract.regression_prediction(2.).unwrap(), 300.);
    assert_eq!(contract.regression_prediction(-2.).unwrap(), -100.);
    assert!(contract.regression_prediction(f64::INFINITY).is_err());
    assert_eq!(
        contract.compile_input(&query).unwrap().view().rows(),
        query.rows()
    );
    let mutated_query = table(
        "10,20,100\n20,40,200\n9999,7777,8888\n",
        &declaration(false),
    );
    let refit = prepare(&mutated_query, &[0, 1], spec(TaskKind::Regression)).unwrap();
    assert_eq!(value(&fit), value(&refit));
    let independently_fitted = prepare(&input, &[2], spec(TaskKind::Regression)).unwrap();
    assert_eq!(
        contract
            .validate_view(&independently_fitted)
            .unwrap_err()
            .code,
        "prepared_column_mismatch"
    );
}

#[test]
fn discrete_tasks_validate_with_multiclass_qualification_blocked() {
    for task in [TaskKind::Binary, TaskKind::Multiclass] {
        let body = if task == TaskKind::Binary {
            "10,20,0\n20,40,1\n"
        } else {
            "10,20,2\n20,40,0\n30,60,1\n"
        };
        let input = table(body, &declaration(false));
        let rows: Vec<_> = (0..input.rows()).collect();
        let fit = prepare(&input, &rows, spec(task)).unwrap();
        let contract = fit.freeze().unwrap();
        validate_input(&fit.private_json().unwrap(), &contract).unwrap();
        assert!(fit.metadata.capabilities.regression.qualified);
        assert!(fit.metadata.capabilities.binary.qualified);
        assert!(!fit.metadata.capabilities.multiclass.qualified);
        if task == TaskKind::Multiclass {
            assert_eq!(
                contract.compile_input(&fit).unwrap_err().code,
                "prepared_task_not_qualified"
            );
        } else {
            contract.compile_input(&fit).unwrap();
        }
        let mut swapped = value(&fit);
        swapped["metadata"]["target"]["class_order"]
            .as_array_mut()
            .unwrap()
            .swap(0, 1);
        assert_eq!(
            validate_input(&serde_json::to_vec(&swapped).unwrap(), &contract)
                .unwrap_err()
                .code,
            "prepared_class_mismatch"
        );
        let invalid = table("10,20,42\n", &declaration(false));
        assert_eq!(
            contract.apply(&invalid, &[0]).unwrap_err().code,
            "prepared_class_mismatch"
        );
    }
}

#[test]
fn column_group_and_contract_order_are_bound_to_the_frozen_map() {
    let fit = numeric();
    let contract = fit.freeze().unwrap();
    for (path, code) in [
        ("columns", "invalid_prepared_contract"),
        ("group_map", "prepared_group_mismatch"),
    ] {
        let mut changed = value(&fit);
        changed["metadata"][path].as_array_mut().unwrap().swap(0, 1);
        assert_eq!(
            validate_input(&serde_json::to_vec(&changed).unwrap(), &contract)
                .unwrap_err()
                .code,
            code
        );
    }
    let mut changed = value(&fit);
    changed["metadata"]["group_map"] = json!([{"id":1,"column_ids":[2,1]}]);
    assert_eq!(
        validate_input(&serde_json::to_vec(&changed).unwrap(), &contract)
            .unwrap_err()
            .code,
        "prepared_group_mismatch"
    );
    for path in [
        "fit_digest",
        "entity_digest",
        "split_digest",
        "column_digest",
        "implementation_hash",
    ] {
        let mut changed = value(&fit);
        changed["metadata"]["provenance"][path] = json!("c".repeat(64));
        assert_eq!(
            validate_input(&serde_json::to_vec(&changed).unwrap(), &contract)
                .unwrap_err()
                .code,
            "prepared_contract_mismatch"
        );
    }
    let mut changed = value(&fit);
    changed["metadata"]["horizon"]["steps"] = json!(2);
    assert_eq!(
        validate_input(&serde_json::to_vec(&changed).unwrap(), &contract)
            .unwrap_err()
            .code,
        "prepared_contract_mismatch"
    );
}

fn object_paths(value: &Value, path: &str, paths: &mut Vec<String>) {
    match value {
        Value::Object(object) => {
            paths.push(path.into());
            for (key, value) in object {
                object_paths(value, &format!("{path}/{key}"), paths);
            }
        }
        Value::Array(array) => {
            for (i, value) in array.iter().enumerate() {
                object_paths(value, &format!("{path}/{i}"), paths);
            }
        }
        _ => {}
    }
}

#[test]
fn unknown_missing_and_duplicate_fields_reject_without_source_echo() {
    for category in [false, true] {
        let input = table(
            if category {
                "10,private_category,100\n20,other_category,200\n"
            } else {
                "10,20,100\n20,40,200\n"
            },
            &declaration(category),
        );
        let fit = prepare(&input, &[0, 1], spec(TaskKind::Regression)).unwrap();
        let base = value(&fit);
        let mut paths = Vec::new();
        object_paths(&base, "", &mut paths);
        for path in paths {
            let object = base.pointer(&path).unwrap().as_object().unwrap();
            for key in object.keys() {
                let mut missing = base.clone();
                missing
                    .pointer_mut(&path)
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .remove(key);
                let e = PreparedView::from_private_json(&serde_json::to_vec(&missing).unwrap())
                    .unwrap_err();
                assert_eq!(e.code, "invalid_prepared_json", "{path}/{key}");
            }
            let mut unknown = base.clone();
            unknown
                .pointer_mut(&path)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert("private_header".into(), json!("private_value_987"));
            let e = PreparedView::from_private_json(&serde_json::to_vec(&unknown).unwrap())
                .unwrap_err();
            assert_eq!(e.code, "invalid_prepared_json");
            assert!(!format!("{e} {e:?}").contains("private_"));
        }
        let json = String::from_utf8(fit.private_json().unwrap()).unwrap();
        for needle in ["\"version\":1", "\"state\":\"private\"", "\"scale\":10.0"] {
            assert!(json.contains(needle));
            let doubled = json.replacen(needle, &format!("{needle},{needle}"), 1);
            assert_eq!(
                PreparedView::from_private_json(doubled.as_bytes())
                    .unwrap_err()
                    .code,
                "invalid_prepared_json"
            );
        }
    }
}

#[test]
fn raw_inputs_and_public_projections_cannot_enter_prepared_boundaries() {
    let fit = numeric();
    let contract = fit.freeze().unwrap();
    for bytes in [
        b"private_measurement,private_outcome\n987,456\n".to_vec(),
        serde_json::to_vec(&declaration(false)).unwrap(),
        serde_json::to_vec(&fit.public_projection().unwrap()).unwrap(),
    ] {
        let e = validate_input(&bytes, &contract).unwrap_err();
        assert_eq!(e.code, "invalid_prepared_json");
        assert!(!format!("{e} {e:?}").contains("private_"));
    }
    let public = serde_json::to_string(&fit.public_projection().unwrap()).unwrap();
    for private in [
        "private_",
        "c0001",
        "offset",
        "scale",
        "bounds",
        "digest",
        "100.0",
        "200.0",
        "horizon",
        "class_order",
    ] {
        assert!(!public.contains(private), "{private}");
    }
    assert!(!format!("{fit:?} {contract:?}").contains("private_"));
    let mut alias = value(&fit);
    alias["format"] = json!("PreparedTable");
    assert_eq!(
        PreparedView::from_private_json(&serde_json::to_vec(&alias).unwrap())
            .unwrap_err()
            .code,
        "prepared_input_required"
    );
}

#[test]
fn categories_are_fit_only_and_missing_and_oov_are_explicit_rejections() {
    let input = table(
        "10,private_category,100\n20,other_category,200\n30,unseen_category,300\n",
        &declaration(true),
    );
    let fit = prepare(&input, &[0, 1], spec(TaskKind::Regression)).unwrap();
    let contract = fit.freeze().unwrap();
    assert_eq!(fit.rows(), &[vec![0., 0., 100.], vec![1., 1., 200.]]);
    assert_eq!(
        contract.apply(&input, &[2]).unwrap_err().code,
        "prepared_oov_rejected"
    );
    let public = serde_json::to_string(&fit.public_projection().unwrap()).unwrap();
    for source in ["private_category", "other_category", "unseen_category"] {
        assert!(!public.contains(source));
    }
    for dtype in ["continuous", "nominal"] {
        let mut decl = declaration(dtype == "nominal");
        decl["columns"][0]["mask_tokens"] = json!([{"token":"MASK","state":"missing"}]);
        let data = table(
            if dtype == "nominal" {
                "10,known,100\nMASK,known,200\n"
            } else {
                "10,20,100\nMASK,40,200\n"
            },
            &decl,
        );
        let fit = prepare(&data, &[0], spec(TaskKind::Regression)).unwrap();
        assert_eq!(
            fit.freeze().unwrap().apply(&data, &[1]).unwrap_err().code,
            "prepared_missing_rejected"
        );
    }
}

#[test]
fn semantic_guards_refuse_invalid_contracts_and_rows() {
    let fit = numeric();
    for (pointer, replacement) in [
        ("/version", json!(2)),
        ("/version", json!(1.0)),
        ("/metadata/horizon/steps", json!(0)),
        ("/metadata/columns/0/transform/scale", json!(0)),
        ("/metadata/columns/0/source_id", json!("private_header")),
        ("/metadata/columns/1/source_id", json!("c0001")),
        ("/metadata/group_map/1/column_ids", json!([1])),
        ("/metadata/target/units", json!("  ")),
        ("/metadata/policies/saturation", json!("clip")),
        ("/metadata/projection/state", json!("public")),
        ("/metadata/capabilities/multiclass/qualified", json!(true)),
        (
            "/metadata/provenance/entity_digest",
            json!("source_extrema"),
        ),
        ("/metadata/rights/state", json!("approved")),
        ("/rows/0", json!([1])),
        ("/rows", json!([])),
    ] {
        let mut changed = value(&fit);
        *changed.pointer_mut(pointer).unwrap() = replacement;
        let e =
            PreparedView::from_private_json(&serde_json::to_vec(&changed).unwrap()).unwrap_err();
        assert!(!format!("{e} {e:?}").contains("private_"));
    }
    let input = table("1,2,100\n", &declaration(false));
    for rows in [vec![], vec![0, 0], vec![1]] {
        assert!(prepare(&input, &rows, spec(TaskKind::Regression)).is_err());
    }
    let mut blocked = spec(TaskKind::Regression);
    blocked.rights.state = RightsState::Blocked;
    let fit = prepare(&input, &[0], blocked).unwrap();
    assert_eq!(
        fit.freeze().unwrap().compile_input(&fit).unwrap_err().code,
        "prepared_rights_blocked"
    );
    let mut units = spec(TaskKind::Regression);
    units.target = PreparedTarget::Regression {
        units: "different".into(),
    };
    assert_eq!(
        prepare(&input, &[0], units).unwrap_err().code,
        "prepared_units_mismatch"
    );
}

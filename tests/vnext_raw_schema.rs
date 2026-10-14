use dope_kernel::{schema, typed_data};

use schema::{MaskState, MetadataVisibility, TableSchemaV1};
use serde_json::{Value, json};
use typed_data::{Cell, FittedTransforms, InputLimits, TypedTable, fit_transform};

fn declaration() -> Value {
    json!({
        "format":"dope-table-schema", "version":1,
        "columns":[
            {"id":"c0001", "source_name":"private_measurement", "role":"feature", "dtype":"continuous", "units":"cm", "semantic_label":"length", "mask_tokens":[
                {"token":"", "state":"missing"}, {"token":"ABSENT", "state":"structural_absence"},
                {"token":"PAD", "state":"padding"}, {"token":"CENSOR", "state":"censored"}
            ]},
            {"id":"c0002", "source_name":"private_outcome", "role":"target", "dtype":"continuous", "units":"ms", "semantic_label":"duration"}
        ]
    })
}

fn parse(value: &Value) -> schema::Result<TableSchemaV1> {
    TableSchemaV1::from_private_json(&serde_json::to_vec(value).unwrap())
}

fn table(csv: &str, value: &Value) -> schema::Result<TypedTable> {
    TypedTable::read_csv(csv.as_bytes(), parse(value)?, InputLimits::default())
}

fn numeric_table() -> TypedTable {
    table(
        "private_outcome,private_measurement\n100,10\n300,20\n9999,1000\n",
        &declaration(),
    )
    .unwrap()
}

#[test]
fn named_target_and_header_order_preserve_original_units() {
    let table = numeric_table();
    assert_eq!(table.target_id().unwrap(), "c0002");
    assert_eq!(table.cell(0, "c0001").unwrap(), &Cell::Number(10.0));
    assert_eq!(table.cell(2, "c0002").unwrap(), &Cell::Number(9999.0));
    let state = fit_transform(&table, &[0, 1]).unwrap();
    assert_eq!(
        state
            .transform_numeric("c0001", &[10.0, 20.0, 1000.0])
            .unwrap(),
        [0.0, 1.0, 99.0]
    );
    assert_eq!(
        state
            .inverse_transform("c0002", &[0.0, 1.0, 2.0, -1.0])
            .unwrap(),
        [100.0, 300.0, 500.0, -100.0]
    );
    assert_eq!(state.column_units("c0002").unwrap(), Some("ms"));
    // Original query truth never passes through the affine target map.
    assert_eq!(table.cell(2, "c0002").unwrap(), &Cell::Number(9999.0));
}

#[test]
fn changes_outside_fit_fold_do_not_change_required_state() {
    let original = numeric_table();
    let changed = table(
        "private_measurement,private_outcome\n10,100\n20,300\n-8000,-7000\n",
        &declaration(),
    )
    .unwrap();
    assert_eq!(
        fit_transform(&original, &[0, 1])
            .unwrap()
            .sampling_state_bytes()
            .unwrap(),
        fit_transform(&changed, &[0, 1])
            .unwrap()
            .sampling_state_bytes()
            .unwrap()
    );
    for rows in [vec![], vec![0, 0], vec![3]] {
        assert_eq!(
            fit_transform(&original, &rows).unwrap_err().code,
            "invalid_fit_rows"
        );
    }
}

#[test]
fn private_state_reloads_without_rows_or_source_name_map() {
    let table = numeric_table();
    let state = fit_transform(&table, &[0, 1]).unwrap();
    let bytes = state.sampling_state_bytes().unwrap();
    assert_eq!(state.required_sampling_bytes().unwrap(), bytes.len());
    let encoded = String::from_utf8(bytes.clone()).unwrap();
    for name in [
        "private_measurement",
        "private_outcome",
        "source_name",
        "9999",
    ] {
        assert!(!encoded.contains(name));
    }
    drop(table);
    drop(state);
    let loaded = FittedTransforms::reload(&bytes).unwrap();
    assert_eq!(loaded.inverse_transform("c0002", &[2.0]).unwrap(), [500.0]);
    assert_eq!(loaded.column_units("c0001").unwrap(), Some("cm"));
}

#[test]
fn public_metadata_has_no_source_names_values_domains_or_extrema() {
    let mut schema = declaration();
    schema["columns"][0]["domain"] = json!({"kind":"numeric", "lower":12345.67, "upper":76543.21});
    let parsed = parse(&schema).unwrap();
    let public = serde_json::to_string(&parsed.public_projection().unwrap()).unwrap();
    for secret in [
        "private_measurement",
        "private_outcome",
        "12345.67",
        "76543.21",
        "length",
        "duration",
        "cm",
        "ms",
        "domain",
    ] {
        assert!(!public.contains(secret));
    }
    let table = numeric_table();
    let state = fit_transform(&table, &[0, 1]).unwrap();
    let public = serde_json::to_string(&state.public_projection().unwrap()).unwrap();
    assert!(!public.contains("offset"));
    assert!(!public.contains("scale"));
    assert!(!format!("{table:?}{state:?}").contains("private_"));
}

#[test]
fn approved_semantic_labels_are_separate_from_private_headers() {
    let mut value = declaration();
    value["metadata_visibility"] = json!("approved_semantic_labels");
    let parsed = parse(&value).unwrap();
    assert_eq!(
        parsed.metadata_visibility,
        MetadataVisibility::ApprovedSemanticLabels
    );
    let public = serde_json::to_string(&parsed.public_projection().unwrap()).unwrap();
    assert!(public.contains("length") && public.contains("cm"));
    assert!(!public.contains("private_measurement"));
}

#[test]
fn zero_and_distinct_masks_keep_their_meanings() {
    let data = table(
        "private_measurement,private_outcome\n0,1\n,2\nABSENT,3\nPAD,4\nCENSOR,5\n",
        &declaration(),
    )
    .unwrap();
    assert_eq!(data.cell(0, "c0001").unwrap(), &Cell::Number(0.0));
    let state = fit_transform(&data, &[0]).unwrap();
    for (row, mask) in [
        (1, MaskState::Missing),
        (2, MaskState::StructuralAbsence),
        (3, MaskState::Padding),
        (4, MaskState::Censored),
    ] {
        let cell = data.cell(row, "c0001").unwrap();
        assert_eq!(cell, &Cell::Mask(mask));
        assert_eq!(state.transform_cell("c0001", cell).unwrap(), *cell);
    }
    let restored = FittedTransforms::reload(&state.sampling_state_bytes().unwrap()).unwrap();
    assert_eq!(
        restored
            .transform_cell("c0001", &Cell::Mask(MaskState::Missing))
            .unwrap(),
        Cell::Mask(MaskState::Missing)
    );
}

#[test]
fn nominal_equality_unknown_categories_and_excluded_keys_are_explicit() {
    let mut value = declaration();
    value["columns"][0]["dtype"] = json!("nominal");
    value["columns"][0]["domain"] = json!({"kind":"categorical", "values":["red","blue"]});
    value["columns"].as_array_mut().unwrap().push(
        json!({"id":"c0003","source_name":"private_key","role":"entity_key","dtype":"integer"}),
    );
    let data = table(
        "private_measurement,private_outcome,private_key\nred,1,100\nblue,2,200\ngreen,3,300\n",
        &value,
    )
    .unwrap();
    let state = fit_transform(&data, &[0, 1]).unwrap();
    assert_eq!(
        data.cell(0, "c0001").unwrap(),
        &Cell::Category("red".into())
    );
    assert_eq!(
        data.cell(2, "c0001").unwrap(),
        &Cell::Mask(MaskState::UnknownCategory)
    );
    assert_eq!(data.feature_cells(0).unwrap().len(), 1);
    assert_eq!(
        state.transform_numeric("c0001", &[1.0]).unwrap_err().code,
        "numeric_transform_unavailable"
    );
    assert_eq!(state.transform_numeric("c0003", &[400.0]).unwrap(), [400.0]);
}

#[test]
fn reloaded_transforms_apply_declared_unknown_category_semantics() {
    let mut value = declaration();
    value["columns"][0]["dtype"] = json!("nominal");
    value["columns"][0]["domain"] = json!({"kind":"categorical", "values":["red","blue"]});
    let data = table("private_measurement,private_outcome\nred,1\n", &value).unwrap();
    let state = fit_transform(&data, &[0]).unwrap();
    let retained = state.sampling_state_bytes().unwrap();
    drop(data);
    drop(state);
    let loaded = FittedTransforms::reload(&retained).unwrap();
    assert_eq!(
        loaded
            .transform_cell("c0001", &Cell::Category("green".into()))
            .unwrap(),
        Cell::Mask(MaskState::UnknownCategory)
    );
    assert_eq!(
        loaded
            .transform_cell("c0001", &Cell::Category("red".into()))
            .unwrap(),
        Cell::Category("red".into())
    );
}

#[test]
fn fit_errors_follow_csv_column_positions_after_header_reordering() {
    for (csv, code) in [
        (
            "private_outcome,private_measurement\n1,\n2,ABSENT\n",
            "fit_column_not_observed",
        ),
        (
            "private_outcome,private_measurement\n1,-1.7e308\n2,1.7e308\n",
            "affine_range_not_representable",
        ),
    ] {
        let data = table(csv, &declaration()).unwrap();
        let error = fit_transform(&data, &[0, 1]).unwrap_err();
        assert_eq!((error.code, error.row, error.column), (code, None, Some(2)));
    }
    let mut value = declaration();
    value["columns"][1]["dtype"] = json!("nominal");
    let data = table("private_outcome,private_measurement\nlabel,10\n", &value).unwrap();
    let error = fit_transform(&data, &[0]).unwrap_err();
    assert_eq!(
        (error.code, error.row, error.column),
        ("fit_target_not_numeric", None, Some(1))
    );
}

#[test]
fn retained_raw_cells_enforce_numeric_types_and_declared_domains() {
    for dtype in ["continuous", "integer", "ordinal"] {
        let mut value = declaration();
        value["columns"][0]["dtype"] = json!(dtype);
        value["columns"][0]["domain"] = json!({"kind":"numeric", "lower":0, "upper":10});
        let data = table("private_measurement,private_outcome\n1,1\n3,3\n", &value).unwrap();
        let retained = fit_transform(&data, &[0, 1])
            .unwrap()
            .sampling_state_bytes()
            .unwrap();
        drop(data);
        let loaded = FittedTransforms::reload(&retained).unwrap();
        for number in [-1.0, 11.0] {
            assert_eq!(
                loaded
                    .transform_cell("c0001", &Cell::Number(number))
                    .unwrap_err()
                    .code,
                "outside_declared_domain"
            );
            assert_eq!(
                loaded
                    .transform_numeric("c0001", &[number])
                    .unwrap_err()
                    .code,
                "outside_declared_domain"
            );
        }
        for number in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                loaded
                    .transform_cell("c0001", &Cell::Number(number))
                    .unwrap_err()
                    .code,
                "nonfinite_numeric_value"
            );
            assert_eq!(
                loaded
                    .transform_numeric("c0001", &[number])
                    .unwrap_err()
                    .code,
                "nonfinite_numeric_value"
            );
        }
        if dtype != "continuous" {
            for number in [1.5, 9_007_199_254_740_992.0, -9_007_199_254_740_992.0] {
                assert_eq!(
                    loaded
                        .transform_cell("c0001", &Cell::Number(number))
                        .unwrap_err()
                        .code,
                    "integer_precision_or_type_mismatch"
                );
                assert_eq!(
                    loaded
                        .transform_numeric("c0001", &[number])
                        .unwrap_err()
                        .code,
                    "integer_precision_or_type_mismatch"
                );
            }
        }
        assert_eq!(
            loaded.transform_cell("c0001", &Cell::Number(9.0)).unwrap(),
            Cell::Number(4.0)
        );
        assert_eq!(loaded.transform_numeric("c0001", &[9.0]).unwrap(), [4.0]);
        assert_eq!(loaded.inverse_transform("c0001", &[6.0]).unwrap(), [13.0]);
    }
    let mut value = declaration();
    value["columns"][0]["dtype"] = json!("boolean");
    value["columns"][0]["domain"] = json!({"kind":"numeric", "lower":0, "upper":0});
    let data = table("private_measurement,private_outcome\nfalse,1\n", &value).unwrap();
    let loaded = FittedTransforms::reload(
        &fit_transform(&data, &[0])
            .unwrap()
            .sampling_state_bytes()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        loaded
            .transform_cell("c0001", &Cell::Boolean(true))
            .unwrap_err()
            .code,
        "outside_declared_domain"
    );
    assert_eq!(
        loaded
            .transform_cell("c0001", &Cell::Boolean(false))
            .unwrap(),
        Cell::Boolean(false)
    );
    assert_eq!(loaded.transform_numeric("c0001", &[0.0]).unwrap(), [0.0]);
    assert_eq!(
        loaded.transform_numeric("c0001", &[1.0]).unwrap_err().code,
        "outside_declared_domain"
    );
    for number in [-1.0, 0.5, 2.0] {
        assert_eq!(
            loaded
                .transform_numeric("c0001", &[number])
                .unwrap_err()
                .code,
            "invalid_boolean_value"
        );
    }
}

#[test]
fn raw_json_duplicate_fields_and_duplicate_column_identities_are_refused() {
    let schema_bytes = serde_json::to_string(&declaration()).unwrap();
    for (field, repeated) in [
        (
            "\"role\":\"feature\"",
            "\"role\":\"excluded_sensitive_identifier\",\"role\":\"feature\"",
        ),
        ("\"id\":\"c0001\"", "\"id\":\"c9999\",\"id\":\"c0001\""),
        (
            "\"source_name\":\"private_measurement\"",
            "\"source_name\":\"other\",\"source_name\":\"private_measurement\"",
        ),
        ("\"version\":1", "\"version\":2,\"version\":1"),
    ] {
        assert!(schema_bytes.contains(field));
        let duplicated = schema_bytes.replacen(field, repeated, 1);
        let error = TableSchemaV1::from_private_json(duplicated.as_bytes())
            .err()
            .unwrap();
        assert_eq!(error.code, "invalid_schema_json");
    }
    for (field, code) in [
        ("id", "ambiguous_column_id"),
        ("source_name", "ambiguous_source_name"),
    ] {
        let mut value = declaration();
        value["columns"][1][field] = value["columns"][0][field].clone();
        assert_eq!(parse(&value).err().unwrap().code, code);
    }
    let state_bytes = String::from_utf8(
        fit_transform(&numeric_table(), &[0, 1])
            .unwrap()
            .sampling_state_bytes()
            .unwrap(),
    )
    .unwrap();
    for (field, repeated) in [
        ("\"scale\":10.0", "\"scale\":20.0,\"scale\":10.0"),
        ("\"offset\":10.0", "\"offset\":20.0,\"offset\":10.0"),
        ("\"version\":1", "\"version\":2,\"version\":1"),
    ] {
        assert!(state_bytes.contains(field));
        let duplicated = state_bytes.replacen(field, repeated, 1);
        assert_eq!(
            FittedTransforms::reload(duplicated.as_bytes())
                .unwrap_err()
                .code,
            "invalid_transform_state"
        );
    }
}

#[test]
fn type_errors_have_positions_without_echoing_input() {
    for (dtype, field, code) in [
        ("continuous", "private_value_987", "invalid_numeric_value"),
        ("continuous", "NaN", "nonfinite_numeric_value"),
        ("integer", "1.5", "integer_precision_or_type_mismatch"),
        (
            "integer",
            "9007199254740993",
            "integer_precision_or_type_mismatch",
        ),
        ("boolean", "maybe", "invalid_boolean_value"),
    ] {
        let mut value = declaration();
        value["columns"][0]["dtype"] = json!(dtype);
        let csv = format!("private_measurement,private_outcome\n{field},1\n");
        let error = table(&csv, &value).unwrap_err();
        assert_eq!(
            (error.code, error.row, error.column),
            (code, Some(2), Some(1))
        );
        assert!(!error.to_string().contains(field));
        assert!(!format!("{error:?}").contains("private_measurement"));
    }
}

#[test]
fn schema_and_headers_fail_closed() {
    let base = declaration();
    for version in [json!(true), json!(1.0), json!(2)] {
        let mut value = base.clone();
        value["version"] = version;
        assert!(parse(&value).is_err());
    }
    let mut unknown = base.clone();
    unknown["private_extra"] = json!(1);
    assert!(parse(&unknown).is_err());
    for csv in [
        "private_measurement,private_measurement\n1,2\n",
        "private_measurement,other_header\n1,2\n",
        "private_measurement,private_outcome\n1,2,3\n",
    ] {
        assert!(table(csv, &base).is_err());
    }
    let mut unsupported = base;
    unsupported["columns"][0]["dtype"] = json!("date_time");
    assert_eq!(
        parse(&unsupported).err().unwrap().code,
        "unsupported_logical_type"
    );
}

#[test]
fn reload_rejects_whitespace_units_and_labels_like_schema_intake() {
    let mut value = declaration();
    value["metadata_visibility"] = json!("approved_semantic_labels");
    let data = table("private_measurement,private_outcome\n1,2\n", &value).unwrap();
    let retained = fit_transform(&data, &[0])
        .unwrap()
        .sampling_state_bytes()
        .unwrap();
    for (schema_field, retained_field) in [("units", "units"), ("semantic_label", "label")] {
        let mut bad_schema = value.clone();
        bad_schema["columns"][0][schema_field] = json!(" \t\n");
        assert_eq!(
            parse(&bad_schema).err().unwrap().code,
            "invalid_metadata_label"
        );
        let mut bad_state: Value = serde_json::from_slice(&retained).unwrap();
        bad_state["columns"][0][retained_field] = json!(" \t\n");
        assert_eq!(
            FittedTransforms::reload(&serde_json::to_vec(&bad_state).unwrap())
                .unwrap_err()
                .code,
            "invalid_transform_state"
        );
    }
}

#[test]
fn retained_nominal_cells_use_the_csv_field_byte_limit() {
    let mut value = declaration();
    value["columns"][0]["dtype"] = json!("nominal");
    for field_limit in [8192, 4] {
        let category = "n".repeat(field_limit);
        let csv = format!("private_measurement,private_outcome\n{category},1\n");
        let limits = InputLimits {
            max_field_bytes: field_limit,
            ..InputLimits::default()
        };
        let data = TypedTable::read_csv(csv.as_bytes(), parse(&value).unwrap(), limits).unwrap();
        let retained = fit_transform(&data, &[0])
            .unwrap()
            .sampling_state_bytes()
            .unwrap();
        drop(data);
        let loaded = FittedTransforms::reload(&retained).unwrap();
        let cell = Cell::Category(category.clone());
        assert_eq!(loaded.transform_cell("c0001", &cell).unwrap(), cell);
        let too_long = format!("{category}n");
        let csv = format!("private_measurement,private_outcome\n{too_long},1\n");
        assert_eq!(
            TypedTable::read_csv(csv.as_bytes(), parse(&value).unwrap(), limits)
                .unwrap_err()
                .code,
            "field_bytes_exceeded"
        );
        assert_eq!(
            loaded
                .transform_cell("c0001", &Cell::Category(too_long))
                .unwrap_err()
                .code,
            "field_bytes_exceeded"
        );
    }
}

#[test]
fn retained_field_limit_defaults_for_prior_v1_states_and_refuses_zero() {
    let state = fit_transform(&numeric_table(), &[0, 1]).unwrap();
    let mut retained: Value =
        serde_json::from_slice(&state.sampling_state_bytes().unwrap()).unwrap();
    retained.as_object_mut().unwrap().remove("max_field_bytes");
    let loaded = FittedTransforms::reload(&serde_json::to_vec(&retained).unwrap()).unwrap();
    assert_eq!(loaded.inverse_transform("c0002", &[2.0]).unwrap(), [500.0]);
    let bytes = loaded.sampling_state_bytes().unwrap();
    assert_eq!(loaded.required_sampling_bytes().unwrap(), bytes.len());
    assert_eq!(
        serde_json::from_slice::<Value>(&bytes).unwrap()["max_field_bytes"],
        json!(4096)
    );
    retained["max_field_bytes"] = json!(0);
    assert_eq!(
        FittedTransforms::reload(&serde_json::to_vec(&retained).unwrap())
            .unwrap_err()
            .code,
        "invalid_transform_state"
    );
}

#[test]
fn byte_row_field_and_transform_bounds_are_enforced() {
    let csv = "private_measurement,private_outcome\n1,2\n3,4\n";
    for limits in [
        InputLimits {
            max_bytes: 8,
            ..InputLimits::default()
        },
        InputLimits {
            max_rows: 1,
            ..InputLimits::default()
        },
        InputLimits {
            max_cells: 2,
            ..InputLimits::default()
        },
    ] {
        assert!(
            TypedTable::read_csv(csv.as_bytes(), parse(&declaration()).unwrap(), limits).is_err()
        );
    }
    let state = fit_transform(&numeric_table(), &[0, 1]).unwrap();
    assert_eq!(
        state
            .inverse_transform("c0002", &[f64::INFINITY])
            .unwrap_err()
            .code,
        "nonfinite_transform_value"
    );
    let mut bytes: Value = serde_json::from_slice(&state.sampling_state_bytes().unwrap()).unwrap();
    bytes["columns"][0]["scale"] = json!(0.0);
    assert!(FittedTransforms::reload(&serde_json::to_vec(&bytes).unwrap()).is_err());
}

#[cfg(unix)]
#[test]
fn source_name_map_is_restricted_and_never_overwrites() {
    use std::os::unix::fs::PermissionsExt;
    let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/vnext/fixtures")
        .join(format!(
            "raw-schema-map-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("names.json");
    let data = numeric_table();
    data.write_private_source_map(&path).unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert!(
        std::fs::read_to_string(&path)
            .unwrap()
            .contains("private_measurement")
    );
    assert_eq!(
        data.write_private_source_map(&path).unwrap_err().code,
        "private_map_create_failed"
    );
}

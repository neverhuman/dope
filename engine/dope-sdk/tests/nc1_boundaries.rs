use dope_sdk::nc1::{
    Cell, DeclaredSchema, FeatureKind, FeatureSpec, InputColumn, InputTable, NormOptions,
    ScaleCard, TargetLastViewExt, TargetSpec, Task, UnitSchema, UnitTable,
};

fn n(v: f64) -> Cell {
    Cell::Number(v)
}
fn c(v: &str) -> Cell {
    Cell::Category(v.into())
}
fn regression_table() -> InputTable {
    InputTable::new(
        "a".repeat(64),
        vec![11, 31, 71, 101],
        vec![
            InputColumn {
                name: "CANARY_PRIVATE_HEADER_X".into(),
                values: vec![n(0.), n(2.), n(4.), n(1.0e300)],
            },
            InputColumn {
                name: "CANARY_PRIVATE_TARGET".into(),
                values: vec![n(10.), n(20.), n(40.), n(1.0e300)],
            },
            InputColumn {
                name: "CANARY_PRIVATE_CATEGORY".into(),
                values: vec![
                    c("CANARY_PRIVATE_CAT_A"),
                    c("CANARY_PRIVATE_CAT_B"),
                    c("CANARY_PRIVATE_CAT_A"),
                    c("CANARY_PRIVATE_CAT_QUERY"),
                ],
            },
        ],
    )
    .unwrap()
}
fn regression_declared() -> DeclaredSchema {
    DeclaredSchema {
        features: vec![
            FeatureSpec {
                column: 0,
                kind: FeatureKind::Continuous,
            },
            FeatureSpec {
                column: 2,
                kind: FeatureKind::Nominal,
            },
        ],
        target: TargetSpec {
            column: 1,
            task: Task::Regression,
        },
    }
}

#[test]
fn regression_roundtrip_reordered_feature_query_keeps_physical_order_and_target_tail() {
    let source = regression_table();
    let card = ScaleCard::fit(
        &source,
        &regression_declared(),
        &[71, 11, 31],
        NormOptions::default(),
    )
    .unwrap();
    let card_bytes = card.to_private_json().unwrap();
    let decoded = ScaleCard::from_private_json(&card_bytes).unwrap();
    assert_eq!(decoded.to_private_json().unwrap(), card_bytes);
    assert_eq!(decoded.source_sha256(), source.source_sha256);
    assert_eq!(decoded.fit_scope_sha256(), card.fit_scope_sha256());
    assert_eq!(decoded.transform_sha256(), card.transform_sha256());

    let query = InputTable::new(
        "b".repeat(64),
        vec![901, 401],
        vec![
            InputColumn {
                name: "CANARY_PRIVATE_CATEGORY".into(),
                values: vec![c("CANARY_PRIVATE_CAT_QUERY"), c("CANARY_PRIVATE_CAT_A")],
            },
            InputColumn {
                name: "CANARY_PRIVATE_HEADER_X".into(),
                values: vec![n(6.), n(2.)],
            },
        ],
    )
    .unwrap();
    let (x_only, receipt) = decoded.apply_features(&query, &[401, 901]).unwrap();
    assert_eq!(x_only.row_ids(), &[401, 901]);
    assert_eq!(x_only.y(), None);
    assert_eq!(
        x_only.to_target_last_view().unwrap_err().code,
        "target_last_target_absent"
    );
    assert_eq!(receipt.source_sha256, query.source_sha256);
    assert_eq!(receipt.fit_scope_sha256, decoded.fit_scope_sha256());
    assert_eq!(receipt.transform_sha256, decoded.transform_sha256());
    assert_eq!(
        receipt.schema_sha256,
        decoded.public_schema().sha256().unwrap()
    );
    assert!(decoded.apply(&query, &[901]).is_err()); // declared target absent; no role inference

    let (fit, _) = decoded.apply(&source, &[101, 11, 31]).unwrap();
    assert_eq!(fit.row_ids(), &[101, 11, 31]);
    assert_eq!(fit.y().unwrap().len(), 3);
    assert!(fit.y().unwrap()[0] > 1.0); // target tails remain unclipped
    assert_native_target_last(&fit);
    assert_eq!(fit.x()[0].len(), fit.schema().columns); // target never enters x

    let public = decoded.public_schema().to_json().unwrap();
    let public_text = String::from_utf8(public.clone()).unwrap();
    for private in [
        "CANARY_PRIVATE_HEADER_X",
        "CANARY_PRIVATE_TARGET",
        "CANARY_PRIVATE_CAT_A",
        "1e300",
        "1e+300",
    ] {
        assert!(
            !public_text.contains(private),
            "public schema leaked planted private canary"
        );
    }
    let schema = UnitSchema::from_json(&public).unwrap();
    assert_eq!(
        schema.sha256().unwrap(),
        decoded.public_schema().sha256().unwrap()
    );
    let matrix = fit.to_private_json().unwrap();
    assert_eq!(
        dope_sdk::nc1::UnitTable::from_private_json(&matrix)
            .unwrap()
            .row_ids(),
        &[101, 11, 31]
    );
    assert!(decoded.apply(&source, &[101]).unwrap().0.y().unwrap()[0] > 1.0); // target tail is not clipped
    let mut altered = source.clone();
    altered.source_sha256 = "f".repeat(64);
    altered.columns[0].values[..3].clone_from_slice(&[n(-2.), n(3.), n(9.)]);
    altered.columns[1].values[..3].clone_from_slice(&[n(15.), n(25.), n(45.)]);
    let different_extrema = ScaleCard::fit(
        &altered,
        &regression_declared(),
        &[71, 11, 31],
        NormOptions::default(),
    )
    .unwrap();
    assert_ne!(
        different_extrema.transform_sha256(),
        decoded.transform_sha256()
    );
    assert_eq!(different_extrema.public_schema().to_json().unwrap(), public);
}

fn assert_native_target_last(table: &UnitTable) {
    let view = table.to_target_last_view().unwrap();
    assert_eq!(view.row_ids(), table.row_ids());
    assert_eq!(view.schema(), table.schema());
    assert_eq!(view.source_sha256(), table.source_sha256());
    assert_eq!(view.fit_scope_sha256(), table.fit_scope_sha256());
    assert_eq!(view.transform_sha256(), table.transform_sha256());
    assert_eq!(view.target_column(), table.schema().columns);
    assert_eq!(view.rows().len(), table.x().len());
    for (i, row) in view.rows().iter().enumerate() {
        assert_eq!(&row[..view.target_column()], table.x()[i].as_slice());
        assert_eq!(row.len(), table.schema().columns + 1);
        assert_eq!(row[view.target_column()], table.y().unwrap()[i]);
    }
    assert_eq!(format!("{view:?}"), "TargetLastView(<private>)");
}

#[test]
fn binary_and_multiclass_target_semantics_are_declared_and_stable() {
    let binary = InputTable::new(
        "c".repeat(64),
        vec![5, 25, 65, 105],
        vec![
            InputColumn {
                name: "feature".into(),
                values: vec![n(0.), n(1.), n(2.), n(3.)],
            },
            InputColumn {
                name: "class".into(),
                values: vec![
                    c("CANARY_TARGET_NEG"),
                    c("CANARY_TARGET_POS"),
                    c("CANARY_TARGET_NEG"),
                    c("CANARY_TARGET_POS"),
                ],
            },
            InputColumn {
                name: "unused-after-target".into(),
                values: vec![c("u"); 4],
            },
        ],
    )
    .unwrap();
    let bin = ScaleCard::fit(
        &binary,
        &DeclaredSchema {
            features: vec![FeatureSpec {
                column: 0,
                kind: FeatureKind::Continuous,
            }],
            target: TargetSpec {
                column: 1,
                task: Task::Binary {
                    positive_label: "CANARY_TARGET_POS".into(),
                },
            },
        },
        &[5, 25, 65, 105],
        NormOptions::default(),
    )
    .unwrap();
    let (bt, _) = bin.apply(&binary, &[105, 5]).unwrap();
    assert_eq!(bt.row_ids(), &[105, 5]);
    assert_eq!(bt.y().unwrap(), &[1., 0.]);
    assert_native_target_last(&bt);
    assert_eq!(bin.public_schema().classes, 2);
    let binary_private = bin.to_private_json().unwrap();
    assert!(String::from_utf8_lossy(&binary_private).contains("CANARY_TARGET_POS"));
    let binary_public = String::from_utf8(bin.public_schema().to_json().unwrap()).unwrap();
    assert!(!binary_public.contains("CANARY_TARGET_POS"));
    assert!(!binary_public.contains("CANARY_TARGET_NEG"));

    let multi = InputTable::new(
        "d".repeat(64),
        vec![7, 27, 67, 107],
        vec![
            InputColumn {
                name: "numeric".into(),
                values: vec![n(1.), n(2.), n(3.), n(4.)],
            },
            InputColumn {
                name: "label".into(),
                values: vec![c("red"), c("blue"), c("green"), c("red")],
            },
            InputColumn {
                name: "unused-after-label".into(),
                values: vec![c("u"); 4],
            },
        ],
    )
    .unwrap();
    let labels = vec!["green".into(), "red".into(), "blue".into()];
    let mc = ScaleCard::fit(
        &multi,
        &DeclaredSchema {
            features: vec![FeatureSpec {
                column: 0,
                kind: FeatureKind::Continuous,
            }],
            target: TargetSpec {
                column: 1,
                task: Task::Multiclass {
                    labels: labels.clone(),
                },
            },
        },
        &[7, 27, 67, 107],
        NormOptions::default(),
    )
    .unwrap();
    let (mt, _) = mc.apply(&multi, &[67, 7]).unwrap();
    assert_eq!(mt.y().unwrap(), &[0., 1.]);
    assert_eq!(mt.schema().classes, 3);
    assert_native_target_last(&mt);
    assert_eq!(
        mc.to_private_json().unwrap(),
        ScaleCard::from_private_json(&mc.to_private_json().unwrap())
            .unwrap()
            .to_private_json()
            .unwrap()
    );
}

#[test]
fn malformed_row_selection_and_target_role_refuse() {
    let source = regression_table();
    assert_eq!(
        ScaleCard::fit(
            &source,
            &regression_declared(),
            &[11, 11],
            NormOptions::default()
        )
        .unwrap_err()
        .code,
        "duplicate_selected_row"
    );
    let mut wrong = regression_declared();
    wrong.features[0].column = 1; // declared target is not eligible as a feature
    assert_eq!(
        ScaleCard::fit(&source, &wrong, &[11, 31], NormOptions::default())
            .unwrap_err()
            .code,
        "invalid_feature_role"
    );
}

use dope_norm::*;

fn n(v: f64) -> Cell {
    Cell::Number(v)
}
fn c(v: &str) -> Cell {
    Cell::Category(v.into())
}
fn fixture() -> InputTable {
    InputTable::new(
        "a".repeat(64),
        vec![10, 20, 30, 40],
        vec![
            InputColumn {
                name: "CANARY_HEADER".into(),
                values: vec![n(10.0), n(20.0), Cell::Missing, n(100.0)],
            },
            InputColumn {
                name: "CANARY_CATEGORY".into(),
                values: vec![
                    c("__other__"),
                    c("CANARY_LEVEL"),
                    Cell::Missing,
                    c("unseen"),
                ],
            },
            InputColumn {
                name: "target".into(),
                values: vec![n(100.0), n(200.0), n(300.0), n(500.0)],
            },
        ],
    )
    .unwrap()
}
fn declared() -> DeclaredSchema {
    DeclaredSchema {
        features: vec![
            FeatureSpec {
                column: 0,
                kind: FeatureKind::Continuous,
            },
            FeatureSpec {
                column: 1,
                kind: FeatureKind::Nominal,
            },
        ],
        target: TargetSpec {
            column: 2,
            task: Task::Regression,
        },
    }
}
fn card() -> ScaleCard {
    ScaleCard::fit(
        &fixture(),
        &declared(),
        &[10, 20, 30],
        NormOptions {
            min_level_count: 1,
            max_levels: 31,
        },
    )
    .unwrap()
}
#[test]
fn fit_only_query_poisoning_and_canonical_rows() {
    let original = card().to_private_json().unwrap();
    let mut changed = fixture();
    changed.columns[0].values[3] = n(-987654321.0);
    changed.columns[1].values[3] = c("poison");
    changed.columns[2].values[3] = n(-8e99);
    let fitted = ScaleCard::fit(
        &changed,
        &declared(),
        &[30, 10, 20],
        NormOptions {
            min_level_count: 1,
            max_levels: 31,
        },
    )
    .unwrap();
    assert_eq!(original, fitted.to_private_json().unwrap());
}
#[test]
fn feature_parameters_never_use_target_information() {
    let mut changed = fixture();
    changed.columns[2].values.swap(0, 2);
    let a = card().apply_features(&fixture(), &[10, 20, 30]).unwrap().0;
    let b = ScaleCard::fit(
        &changed,
        &declared(),
        &[10, 20, 30],
        NormOptions {
            min_level_count: 1,
            max_levels: 31,
        },
    )
    .unwrap()
    .apply_features(&changed, &[10, 20, 30])
    .unwrap()
    .0;
    assert_eq!(a.x(), b.x());
}
#[test]
fn tails_and_targets_are_not_clipped() {
    let (table, receipt) = card().apply(&fixture(), &[40]).unwrap();
    assert_eq!(table.x()[0][0], 9.0);
    assert_eq!(table.y().unwrap(), &[2.0]);
    assert_eq!(receipt.out_of_range[0], 1);
    assert_eq!(card().inverse_target(2.0).unwrap(), 500.0);
}
#[test]
fn query_source_is_separate_and_target_can_be_absent() {
    let mut query = fixture();
    query.source_sha256 = "b".repeat(64);
    query.columns.pop();
    let (table, _) = card().apply_features(&query, &[40]).unwrap();
    assert!(table.y().is_none());
    assert_eq!(table.source_sha256(), "b".repeat(64));
    assert_eq!(card().source_sha256(), "a".repeat(64));
}
#[test]
fn nominal_reserved_strings_are_ordinary_distinct_levels() {
    let (table, receipt) = card().apply(&fixture(), &[10, 20, 30, 40]).unwrap();
    let group = &table.schema().groups[1];
    for row in table.x() {
        assert_eq!(
            row[group.start..group.start + group.width]
                .iter()
                .filter(|x| **x == 1.0)
                .count(),
            1
        );
    }
    assert_ne!(table.x()[0], table.x()[3]);
    assert_eq!(receipt.unseen_levels[1], 1);
    assert_eq!(group.width, 4);
}
#[test]
fn public_projection_and_debug_do_not_disclose_private_data() {
    let card = card();
    let public = String::from_utf8(card.public_schema().to_json().unwrap()).unwrap();
    for secret in [
        "CANARY_HEADER",
        "CANARY_CATEGORY",
        "CANARY_LEVEL",
        "100.0",
        "200.0",
    ] {
        assert!(!public.contains(secret), "{secret}");
    }
    let private = String::from_utf8(card.to_private_json().unwrap()).unwrap();
    assert!(private.contains("CANARY_HEADER"));
    assert!(!format!("{card:?}").contains("CANARY"));
    let table = card.apply(&fixture(), &[10]).unwrap().0;
    assert!(!format!("{table:?}").contains("CANARY"));
    let mut query = fixture();
    query.columns[0].name = "LEAK_ME".into();
    let e = card.apply_features(&query, &[10]).unwrap_err();
    assert!(!e.to_string().contains("LEAK_ME"));
}
#[test]
fn private_roundtrip_and_tamper_refusal() {
    let card = card();
    let bytes = card.to_private_json().unwrap();
    let loaded = ScaleCard::from_private_json(&bytes).unwrap();
    assert_eq!(
        loaded.apply(&fixture(), &[40]).unwrap().0.x(),
        card.apply(&fixture(), &[40]).unwrap().0.x()
    );
    let mut json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    json["features"][0]["state"]["numeric"]["lo"] = serde_json::json!(999.0);
    assert!(ScaleCard::from_private_json(&serde_json::to_vec(&json).unwrap()).is_err());
    json["unknown_canary"] = serde_json::json!("secret");
    let error = ScaleCard::from_private_json(&serde_json::to_vec(&json).unwrap()).unwrap_err();
    assert_eq!(error.code, "invalid_json");
}
#[test]
fn private_matrix_roundtrip_and_malformed_shape_refusal() {
    let table = card().apply(&fixture(), &[10, 40]).unwrap().0;
    let bytes = table.to_private_json().unwrap();
    let loaded = UnitTable::from_private_json(&bytes).unwrap();
    assert_eq!(table.sha256().unwrap(), loaded.sha256().unwrap());
    let mut json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    json["x"][0] = serde_json::json!([]);
    assert!(UnitTable::from_private_json(&serde_json::to_vec(&json).unwrap()).is_err());
}
#[test]
fn full_f64_span_and_tiny_span_do_not_overflow() {
    for (lo, hi) in [(-f64::MAX, f64::MAX), (1e-300, 2e-300)] {
        let t = InputTable::new(
            "a".repeat(64),
            vec![0, 1, 2],
            vec![
                InputColumn {
                    name: "x".into(),
                    values: vec![n(lo), n(hi), n(0.0)],
                },
                InputColumn {
                    name: "y".into(),
                    values: vec![n(lo), n(hi), n(0.0)],
                },
            ],
        )
        .unwrap();
        let d = DeclaredSchema {
            features: vec![FeatureSpec {
                column: 0,
                kind: FeatureKind::Continuous,
            }],
            target: TargetSpec {
                column: 1,
                task: Task::Regression,
            },
        };
        let fitted = ScaleCard::fit(&t, &d, &[0, 1], NormOptions::default()).unwrap();
        let out = fitted.apply(&t, &[0, 1]).unwrap().0;
        assert_eq!(out.x(), &[vec![0.0], vec![1.0]]);
        assert_eq!(out.y().unwrap(), &[0.0, 1.0]);
        assert!(fitted.inverse_target(0.5).unwrap().is_finite());
    }
}
#[test]
fn constants_and_all_missing_keep_zero_channels_and_indicator() {
    let t = InputTable::new(
        "a".repeat(64),
        vec![0, 1],
        vec![
            InputColumn {
                name: "constant".into(),
                values: vec![n(7.0), n(7.0)],
            },
            InputColumn {
                name: "missing".into(),
                values: vec![Cell::Missing, Cell::Missing],
            },
            InputColumn {
                name: "y".into(),
                values: vec![n(0.0), n(1.0)],
            },
        ],
    )
    .unwrap();
    let d = DeclaredSchema {
        features: vec![
            FeatureSpec {
                column: 0,
                kind: FeatureKind::Continuous,
            },
            FeatureSpec {
                column: 1,
                kind: FeatureKind::Continuous,
            },
        ],
        target: TargetSpec {
            column: 2,
            task: Task::Regression,
        },
    };
    let fitted = ScaleCard::fit(&t, &d, &[0, 1], NormOptions::default()).unwrap();
    assert_eq!(
        fitted.apply(&t, &[0]).unwrap().0.x(),
        &[vec![0.0, 0.0, 1.0]]
    );
}
#[test]
fn declared_multiclass_order_and_absent_fit_class_are_preserved() {
    let t = InputTable::new(
        "a".repeat(64),
        vec![0, 1, 2],
        vec![
            InputColumn {
                name: "x".into(),
                values: vec![n(0.0), n(1.0), n(2.0)],
            },
            InputColumn {
                name: "y".into(),
                values: vec![c("z"), c("a"), c("future")],
            },
        ],
    )
    .unwrap();
    let d = DeclaredSchema {
        features: vec![FeatureSpec {
            column: 0,
            kind: FeatureKind::Continuous,
        }],
        target: TargetSpec {
            column: 1,
            task: Task::Multiclass {
                labels: vec!["z".into(), "future".into(), "a".into()],
            },
        },
    };
    let f = ScaleCard::fit(&t, &d, &[0, 1], NormOptions::default()).unwrap();
    assert_eq!(f.public_schema().classes, 3);
    assert_eq!(f.apply(&t, &[2]).unwrap().1.unobserved_fit_classes, vec![1]);
    assert_eq!(
        f.apply(&t, &[2, 0, 1]).unwrap().0.y().unwrap(),
        &[1.0, 0.0, 2.0]
    );
}
#[test]
fn explicit_binary_positive_not_minority_or_lexical() {
    let t = InputTable::new(
        "a".repeat(64),
        vec![0, 1, 2],
        vec![
            InputColumn {
                name: "x".into(),
                values: vec![n(0.0), n(1.0), n(2.0)],
            },
            InputColumn {
                name: "y".into(),
                values: vec![c("a"), c("z"), c("a")],
            },
        ],
    )
    .unwrap();
    let d = DeclaredSchema {
        features: vec![FeatureSpec {
            column: 0,
            kind: FeatureKind::Continuous,
        }],
        target: TargetSpec {
            column: 1,
            task: Task::Binary {
                positive_label: "a".into(),
            },
        },
    };
    assert_eq!(
        ScaleCard::fit(&t, &d, &[0, 1, 2], NormOptions::default())
            .unwrap()
            .apply(&t, &[0, 1, 2])
            .unwrap()
            .0
            .y()
            .unwrap(),
        &[1.0, 0.0, 1.0]
    );
    assert!(ScaleCard::fit(&t, &d, &[0, 2], NormOptions::default()).is_err());
}
#[test]
fn explicit_boolean_ordinal_and_integer_semantics() {
    let t = InputTable::new(
        "a".repeat(64),
        vec![0, 1],
        vec![
            InputColumn {
                name: "b".into(),
                values: vec![Cell::Boolean(false), Cell::Boolean(true)],
            },
            InputColumn {
                name: "o".into(),
                values: vec![c("high"), c("low")],
            },
            InputColumn {
                name: "y".into(),
                values: vec![n(0.0), n(1.0)],
            },
        ],
    )
    .unwrap();
    let d = DeclaredSchema {
        features: vec![
            FeatureSpec {
                column: 0,
                kind: FeatureKind::Boolean,
            },
            FeatureSpec {
                column: 1,
                kind: FeatureKind::Ordinal {
                    levels: vec!["low".into(), "high".into()],
                },
            },
        ],
        target: TargetSpec {
            column: 2,
            task: Task::Regression,
        },
    };
    assert_eq!(
        ScaleCard::fit(&t, &d, &[0, 1], NormOptions::default())
            .unwrap()
            .apply(&t, &[0, 1])
            .unwrap()
            .0
            .x(),
        &[vec![0.0, 1.0], vec![1.0, 0.0]]
    );
}
#[test]
fn malformed_rows_roles_nonfinite_and_cardinality_refuse() {
    assert!(ScaleCard::fit(&fixture(), &declared(), &[10, 10], NormOptions::default()).is_err());
    assert!(ScaleCard::fit(&fixture(), &declared(), &[999], NormOptions::default()).is_err());
    let mut d = declared();
    d.features[0].column = 2;
    assert!(ScaleCard::fit(&fixture(), &d, &[10, 20], NormOptions::default()).is_err());
    let mut t = fixture();
    t.columns[0].values[0] = n(f64::NAN);
    assert!(ScaleCard::fit(&t, &declared(), &[10, 20], NormOptions::default()).is_err());
    assert!(ScaleCard::fit(
        &fixture(),
        &declared(),
        &[10, 20],
        NormOptions {
            min_level_count: 0,
            max_levels: 31
        }
    )
    .is_err());
}
#[test]
fn expansion_cap_refuses_instead_of_truncation() {
    let mut t = fixture();
    t.columns[1].values = vec![c("a"), c("b"), c("c"), c("d")];
    let mut d = declared();
    for i in 0..86 {
        let col = t.columns.len();
        t.columns.push(InputColumn {
            name: format!("n{i}"),
            values: vec![c("a"), c("b"), c("c"), c("d")],
        });
        d.features.push(FeatureSpec {
            column: col,
            kind: FeatureKind::Nominal,
        });
    }
    assert_eq!(
        ScaleCard::fit(
            &t,
            &d,
            &[10, 20, 30],
            NormOptions {
                min_level_count: 1,
                max_levels: 31
            }
        )
        .unwrap_err()
        .code,
        "expanded_width_limit"
    );
}
#[test]
fn no_implicit_already_normalized_detection() {
    let mut t = fixture();
    t.columns[0].values = vec![n(0.0), n(1.0), n(0.5), n(2.0)];
    let f = ScaleCard::fit(&t, &declared(), &[10, 20, 30], NormOptions::default()).unwrap();
    assert_eq!(f.apply(&t, &[40]).unwrap().0.x()[0][0], 2.0);
    assert!(InputTable::new(
        "not-a-digest".into(),
        vec![1],
        vec![InputColumn {
            name: "x".into(),
            values: vec![n(0.0)]
        }]
    )
    .is_err());
}

#[test]
fn python_reference_export_when_explicitly_requested() {
    if let Ok(path) = std::env::var("DOPE_NC1_CONFORMANCE_OUT") {
        let dir = std::path::Path::new(&path);
        std::fs::create_dir_all(dir).unwrap();
        let card = card();
        std::fs::write(dir.join("card.json"), card.to_private_json().unwrap()).unwrap();
        std::fs::write(
            dir.join("matrix.json"),
            card.apply(&fixture(), &[10, 20, 30, 40])
                .unwrap()
                .0
                .to_private_json()
                .unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn absent_middle_target_and_reordered_query_columns_are_safe() {
    let t = InputTable::new(
        "a".repeat(64),
        vec![0, 1],
        vec![
            InputColumn {
                name: "a".into(),
                values: vec![n(0.0), n(1.0)],
            },
            InputColumn {
                name: "y".into(),
                values: vec![n(3.0), n(7.0)],
            },
            InputColumn {
                name: "b".into(),
                values: vec![n(2.0), n(4.0)],
            },
        ],
    )
    .unwrap();
    let d = DeclaredSchema {
        features: vec![
            FeatureSpec {
                column: 0,
                kind: FeatureKind::Continuous,
            },
            FeatureSpec {
                column: 2,
                kind: FeatureKind::Continuous,
            },
        ],
        target: TargetSpec {
            column: 1,
            task: Task::Regression,
        },
    };
    let f = ScaleCard::fit(&t, &d, &[0, 1], NormOptions::default()).unwrap();
    let q = InputTable::new(
        "b".repeat(64),
        vec![5],
        vec![
            InputColumn {
                name: "b".into(),
                values: vec![n(3.0)],
            },
            InputColumn {
                name: "a".into(),
                values: vec![n(2.0)],
            },
        ],
    )
    .unwrap();
    assert_eq!(f.apply_features(&q, &[5]).unwrap().0.x(), &[vec![2.0, 0.5]]);
    assert!(f.apply(&q, &[5]).is_err());
}
#[test]
fn unknown_public_fields_and_invalid_group_layout_refuse() {
    let mut public: serde_json::Value =
        serde_json::from_slice(&card().public_schema().to_json().unwrap()).unwrap();
    public["private_name"] = serde_json::json!("CANARY");
    assert_eq!(
        UnitSchema::from_json(&serde_json::to_vec(&public).unwrap())
            .unwrap_err()
            .code,
        "invalid_json"
    );
    public.as_object_mut().unwrap().remove("private_name");
    public["groups"][1]["start"] = serde_json::json!(0);
    assert!(UnitSchema::from_json(&serde_json::to_vec(&public).unwrap()).is_err());
}
#[test]
fn fit_table_physical_presentation_order_does_not_change_card() {
    let mut t = fixture();
    t.row_ids.reverse();
    for col in &mut t.columns {
        col.values.reverse();
    }
    let f = ScaleCard::fit(
        &t,
        &declared(),
        &[30, 20, 10],
        NormOptions {
            min_level_count: 1,
            max_levels: 31,
        },
    )
    .unwrap();
    assert_eq!(
        f.to_private_json().unwrap(),
        card().to_private_json().unwrap()
    );
}

#[test]
fn strict_fit_vocabulary_never_pools_or_silently_truncates() {
    let error = ScaleCard::fit(
        &fixture(),
        &declared(),
        &[10, 20],
        NormOptions {
            min_level_count: 1,
            max_levels: 1,
        },
    )
    .unwrap_err();
    assert_eq!(error.code, "categorical_width_limit");
    assert_eq!(NormOptions::default().min_level_count, 1);
    assert_eq!(NormOptions::default().max_levels, 255);
    assert!(ScaleCard::fit(
        &fixture(),
        &declared(),
        &[10, 20],
        NormOptions {
            min_level_count: 5,
            max_levels: 31
        }
    )
    .is_err());
}

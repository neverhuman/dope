//! Tests for `nc2-bulk-tails-v1` scaling: skew guard, unclipped TEST tails,
//! strictly affine target, k=10 rare-level pooling, and metadata-hidden public schema.

use dope_norm::nc2::{Nc2ScaleCard, NC2_SUPPORT_K, RECIPE_NC2};
use dope_norm::{
    Cell, DeclaredSchema, FeatureKind, FeatureSpec, InputColumn, InputTable, TargetSpec, Task,
};

#[test]
fn nc2_skew_guard_unclipped_tails_affine_target_and_private_hiding() {
    let n = 200usize;
    let row_ids: Vec<u64> = (0..n as u64).collect();
    // Col 0: heavy right-skewed income-like feature (triggers asinh skew guard).
    let heavy: Vec<Cell> = (0..n)
        .map(|i| Cell::Number(((i as f64 / n as f64) * 6.0).exp()))
        .collect();
    // Col 1: symmetric uniform feature (no asinh).
    let sym: Vec<Cell> = (0..n).map(|i| Cell::Number(i as f64)).collect();
    // Col 2: nominal with 2 frequent levels (>= 10) and 1 rare secret level (< 10).
    let nom: Vec<Cell> = (0..n)
        .map(|i| {
            if i < 120 {
                Cell::Category("SECRET_ALPHA".into())
            } else if i < 195 {
                Cell::Category("SECRET_BETA".into())
            } else {
                Cell::Category("PRIVATE_RARE_VIP".into())
            }
        })
        .collect();
    // Col 3: regression target.
    let target: Vec<Cell> = (0..n)
        .map(|i| Cell::Number(10.0 + 2.5 * i as f64))
        .collect();

    let table = InputTable::new(
        "a".repeat(64),
        row_ids,
        vec![
            InputColumn {
                name: "SECRET_COL_INCOME".into(),
                values: heavy,
            },
            InputColumn {
                name: "SECRET_COL_SYM".into(),
                values: sym,
            },
            InputColumn {
                name: "SECRET_COL_CAT".into(),
                values: nom,
            },
            InputColumn {
                name: "SECRET_TARGET_Y".into(),
                values: target,
            },
        ],
    )
    .unwrap();
    let schema = DeclaredSchema {
        features: vec![
            FeatureSpec {
                column: 0,
                kind: FeatureKind::Continuous,
            },
            FeatureSpec {
                column: 1,
                kind: FeatureKind::Continuous,
            },
            FeatureSpec {
                column: 2,
                kind: FeatureKind::Nominal,
            },
        ],
        target: TargetSpec {
            column: 3,
            task: Task::Regression,
        },
    };

    let card = Nc2ScaleCard::fit(&table, &schema, NC2_SUPPORT_K).unwrap();
    assert_eq!(card.recipe, RECIPE_NC2);

    // Public schema MUST NOT leak any column names, category strings, or raw extrema!
    let pub_json = serde_json::to_string(&card.public_schema).unwrap();
    for secret in [
        "SECRET_COL_INCOME",
        "SECRET_COL_SYM",
        "SECRET_COL_CAT",
        "SECRET_TARGET_Y",
        "SECRET_ALPHA",
        "SECRET_BETA",
        "PRIVATE_RARE_VIP",
    ] {
        assert!(!pub_json.contains(secret), "public schema leaked {secret}");
    }
    // Nominal width = 1 (unknown) + 2 (kept >= k=10) + 1 (missing slot) = 4!
    assert_eq!(card.public_schema.feature_groups[2].width, 4);

    // Apply on out-of-range TEST rows: tails MUST NOT clip and MUST invert exactly!
    let test_tbl = InputTable::new(
        "b".repeat(64),
        vec![1000, 1001],
        vec![
            InputColumn {
                name: "SECRET_COL_INCOME".into(),
                values: vec![Cell::Number(5000.0), Cell::Number(0.01)],
            },
            InputColumn {
                name: "SECRET_COL_SYM".into(),
                values: vec![Cell::Number(500.0), Cell::Number(-100.0)],
            },
            InputColumn {
                name: "SECRET_COL_CAT".into(),
                values: vec![Cell::Category("UNSEEN_CAT".into()), Cell::Missing],
            },
            InputColumn {
                name: "SECRET_TARGET_Y".into(),
                values: vec![Cell::Number(999.0), Cell::Number(-50.0)],
            },
        ],
    )
    .unwrap();
    let unit = card.apply(&test_tbl).unwrap();
    // Out-of-range numeric values extend strictly past [0, 1] without clipping!
    assert!(unit.x_cols[0][0] > 1.0);
    assert!(unit.x_cols[2][0] > 1.0);
    assert!(unit.x_cols[2][1] < 0.0);
    // Exact inverse round-trip on unclipped out-of-range TEST values & affine target!
    let rec_heavy = card.inverse_numeric(0, unit.x_cols[0][0]).unwrap();
    let rec_sym = card.inverse_numeric(1, unit.x_cols[2][1]).unwrap();
    let rec_y = card.inverse_target(unit.y[0]).unwrap();
    assert!(
        (rec_heavy - 5000.0).abs() < 1e-4,
        "heavy inverse drift: {rec_heavy}"
    );
    assert!(
        (rec_sym - (-100.0)).abs() < 1e-6,
        "sym inverse drift: {rec_sym}"
    );
    assert!((rec_y - 999.0).abs() < 1e-6, "affine target drift: {rec_y}");
    // Unseen category maps to unknown slot (index 4 in x_cols); Missing maps to missing slot (index 7).
    assert_eq!(unit.x_cols[4][0], 1.0);
    assert_eq!(unit.x_cols[7][1], 1.0);
}

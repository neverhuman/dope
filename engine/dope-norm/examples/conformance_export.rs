//! Export new, synthetic-only reference fixtures. Never reads owner data.
use dope_norm::*;
use serde_json::json;
use std::path::Path;
fn n(v: f64) -> Cell {
    Cell::Number(v)
}
fn c(v: &str) -> Cell {
    Cell::Category(v.into())
}
fn write_case(
    root: &Path,
    table: InputTable,
    schema: DeclaredSchema,
    input: serde_json::Value,
) -> Result<()> {
    let card = ScaleCard::fit(&table, &schema, &[10, 20], NormOptions::default())?;
    let matrix = card.apply(&table, &table.row_ids)?.0;
    std::fs::write(root.join("card.json"), card.to_private_json()?).map_err(|_| NormError {
        code: "fixture_write",
        column: None,
    })?;
    std::fs::write(root.join("matrix.json"), matrix.to_private_json()?).map_err(|_| NormError {
        code: "fixture_write",
        column: None,
    })?;
    std::fs::write(
        root.join("input.json"),
        serde_json::to_vec(&input).map_err(|_| NormError {
            code: "fixture_write",
            column: None,
        })?,
    )
    .map_err(|_| NormError {
        code: "fixture_write",
        column: None,
    })?;
    Ok(())
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("--out") {
        return Err("expected --out NEW_DIRECTORY".into());
    }
    let path = args.next().ok_or("expected output directory")?;
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let root = Path::new(&path);
    std::fs::create_dir(root)?;
    let baseline = InputTable::new(
        "a".repeat(64),
        vec![10, 20, 30],
        vec![
            InputColumn {
                name: "CANARY_HEADER".into(),
                values: vec![n(10.0), n(20.0), n(100.0)],
            },
            InputColumn {
                name: "CANARY_CATEGORY".into(),
                values: vec![c("__other__"), c("CANARY_LEVEL"), c("unseen")],
            },
            InputColumn {
                name: "target".into(),
                values: vec![n(100.0), n(200.0), n(500.0)],
            },
        ],
    )?;
    let declared = DeclaredSchema {
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
    };
    write_case(
        root,
        baseline,
        declared,
        json!({"CANARY_HEADER":[10.0,20.0,100.0],"CANARY_CATEGORY":["__other__","CANARY_LEVEL","unseen"],"target":[100.0,200.0,500.0]}),
    )?;
    let sub = root.join("integer-allmissing");
    std::fs::create_dir(&sub)?;
    let integers = InputTable::new(
        "b".repeat(64),
        vec![10, 20, 30],
        vec![
            InputColumn {
                name: "integer".into(),
                values: vec![n(0.0), n(2.0), Cell::Missing],
            },
            InputColumn {
                name: "empty_nominal".into(),
                values: vec![Cell::Missing, Cell::Missing, Cell::Missing],
            },
            InputColumn {
                name: "y".into(),
                values: vec![n(0.0), n(1.0), n(2.0)],
            },
        ],
    )?;
    let declared = DeclaredSchema {
        features: vec![
            FeatureSpec {
                column: 0,
                kind: FeatureKind::Integer,
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
    };
    write_case(
        &sub,
        integers,
        declared,
        json!({"integer":[0.0,2.0,null],"empty_nominal":[null,null,null],"y":[0.0,1.0,2.0]}),
    )?;
    let sub = root.join("declared-classes");
    std::fs::create_dir(&sub)?;
    let classification = InputTable::new(
        "c".repeat(64),
        vec![10, 20, 30],
        vec![
            InputColumn {
                name: "boolean".into(),
                values: vec![Cell::Boolean(false), Cell::Boolean(true), Cell::Missing],
            },
            InputColumn {
                name: "ordered".into(),
                values: vec![c("high"), c("low"), Cell::Missing],
            },
            InputColumn {
                name: "class".into(),
                values: vec![c("z"), c("a"), c("future")],
            },
        ],
    )?;
    let declared = DeclaredSchema {
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
            task: Task::Multiclass {
                labels: vec!["z".into(), "future".into(), "a".into()],
            },
        },
    };
    write_case(
        &sub,
        classification,
        declared,
        json!({"boolean":[false,true,null],"ordered":["high","low",null],"class":["z","a","future"]}),
    )?;
    println!("exported 3 synthetic NC1 cases");
    Ok(())
}

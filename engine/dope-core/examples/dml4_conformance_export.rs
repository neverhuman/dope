//! Export new, source-free synthetic DML4 reference fixtures.
use dope_core::mlc_v4::{
    sample, wire, GateValues, GroupModel, Level, Packet, PacketIdentity, TargetModel,
};
use dope_norm::{GroupKind, GroupSpec, TargetKind, UnitSchema};
use std::path::Path;
fn fixture() -> Packet {
    let schema = UnitSchema {
        schema: "dope.nc1-public/v1".into(),
        recipe: dope_norm::RECIPE.into(),
        columns: 1,
        groups: vec![GroupSpec {
            id: 0,
            kind: GroupKind::Numeric,
            start: 0,
            width: 1,
            has_missing: false,
        }],
        target: TargetKind::Regression,
        classes: 0,
    };
    Packet {
        contract: "dope.dml4-reference/v1".into(),
        level: Level::L2k,
        schema_sha256: schema.sha256().unwrap(),
        schema,
        identity: PacketIdentity {
            weights_sha256: "1".repeat(64),
            encoder_recipe_sha256: "2".repeat(64),
            conditioning_sha256: "3".repeat(64),
            fit_scope_sha256: "4".repeat(64),
        },
        gates: GateValues {
            linear: 0.3,
            asinh: 0.4,
            periodic: 0.5,
            hinge: 0.6,
            pair: 0.7,
            row_mean: 0.8,
        },
        groups: vec![GroupModel::Numeric {
            knots: vec![0.0, 1.0],
            missing_probability: 0.0,
            imputation: 0.5,
        }],
        target: TargetModel::Regression {
            intercept: 0.0,
            coefficients: vec![0.1; 5],
            noise_scale: 0.0,
        },
    }
}
fn write(root: &Path, p: &Packet, rows: usize) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = wire::encode(p)?;
    let out = sample(p, rows, 42)?;
    std::fs::write(root.join("packet.dml4"), bytes)?;
    std::fs::write(root.join("packet.json"), serde_json::to_vec(p)?)?;
    std::fs::write(root.join("samples.json"), serde_json::to_vec(&out)?)?;
    std::fs::write(
        root.join("input.json"),
        serde_json::to_vec(&serde_json::json!({"rows":rows,"seed":42}))?,
    )?;
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let mut p = fixture();
    write(root, &p, 4)?;
    let sub = root.join("grouped-classification");
    std::fs::create_dir(&sub)?;
    p.level = Level::L16k;
    p.schema.columns = 4;
    p.schema.groups = vec![
        GroupSpec {
            id: 0,
            kind: GroupKind::Numeric,
            start: 0,
            width: 2,
            has_missing: true,
        },
        GroupSpec {
            id: 1,
            kind: GroupKind::Nominal,
            start: 2,
            width: 2,
            has_missing: false,
        },
    ];
    p.schema.target = TargetKind::Multiclass;
    p.schema.classes = 3;
    p.schema_sha256 = p.schema.sha256()?;
    p.groups = vec![
        GroupModel::Numeric {
            knots: vec![0.0, 0.5, 1.0],
            missing_probability: 0.25,
            imputation: 0.5,
        },
        GroupModel::Categorical {
            probabilities: vec![0.2, 0.8],
        },
    ];
    p.target = TargetModel::Classification {
        intercepts: vec![-0.3, 0.1, 0.2],
        coefficients: (0..3)
            .map(|k| {
                (0..20)
                    .map(|j| 0.01 * (k as f64 + 1.0) * (j as f64 + 1.0))
                    .collect()
            })
            .collect(),
    };
    write(&sub, &p, 8)?;
    println!("exported 2 synthetic DML4 cases");
    Ok(())
}

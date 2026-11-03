use dope_core::mlc_v4::{
    basis, sample, sample_with_draws, wire, GateValues, GroupModel, Level, Packet, PacketIdentity,
    TargetModel,
};
use dope_norm::{GroupKind, GroupSpec, TargetKind, UnitSchema};
fn packet() -> Packet {
    let schema = UnitSchema {
        schema: "dope.nc1-public/v1".into(),
        recipe: dope_norm::RECIPE.into(),
        columns: 5,
        groups: vec![
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
            GroupSpec {
                id: 2,
                kind: GroupKind::Boolean,
                start: 4,
                width: 1,
                has_missing: false,
            },
        ],
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
            linear: 1.0,
            asinh: 0.0,
            periodic: 0.0,
            hinge: 0.0,
            pair: 0.0,
            row_mean: 0.0,
        },
        groups: vec![
            GroupModel::Numeric {
                knots: vec![0.0, 1.0],
                missing_probability: 0.25,
                imputation: 0.5,
            },
            GroupModel::Categorical {
                probabilities: vec![0.3, 0.7],
            },
            GroupModel::Discrete {
                values: vec![0.0, 1.0],
                probabilities: vec![0.5, 0.5],
                missing_probability: 0.0,
                imputation: 0.0,
            },
        ],
        target: TargetModel::Regression {
            intercept: 2.0,
            coefficients: (0..25).map(|i| f64::from(i == 0)).collect(),
            noise_scale: 0.0,
        },
    }
}
#[test]
fn fixed_draws_apply_missing_onehot_and_target_exactly() {
    let p = packet();
    let draws = vec![
        vec![0.8, 0.1, 0.2, 0.0, 0.9, 0.0, 0.5],
        vec![0.8, 0.9, 0.8, 0.0, 0.1, 0.0, 0.5],
    ];
    let out = sample_with_draws(&p, &draws).unwrap();
    assert_eq!(
        out.x,
        vec![vec![0.5, 1.0, 1.0, 0.0, 1.0], vec![0.8, 0.0, 0.0, 1.0, 0.0]]
    );
    assert_eq!(out.y, vec![2.5, 2.8]);
}
#[test]
fn source_free_replay_and_group_validity() {
    let p = packet();
    let bytes = wire::encode(&p).unwrap();
    drop(p);
    let decoded = wire::decode(&bytes).unwrap();
    let a = sample(&decoded, 100, 42).unwrap();
    let b = sample(&decoded, 100, 42).unwrap();
    assert_eq!(a, b);
    for row in a.x {
        assert_eq!(row[2] + row[3], 1.0);
        assert!(row[0] >= 0.0 && row[0] <= 1.0);
        assert!(row[1] == 0.0 || row[1] == 1.0);
    }
}
#[test]
fn classification_is_class_specific_and_not_common_shift() {
    let mut p = packet();
    p.schema.target = TargetKind::Binary;
    p.schema.classes = 2;
    p.schema_sha256 = p.schema.sha256().unwrap();
    p.target = TargetModel::Classification {
        intercepts: vec![-20.0, 20.0],
        coefficients: vec![vec![0.0; basis::width(5).unwrap()]; 2],
    };
    let out = sample(&p, 8, 9).unwrap();
    assert!(out.y.iter().all(|v| *v == 1.0));
}
#[test]
fn invalid_uniform_shape_and_allocation_requests_refuse() {
    let p = packet();
    assert!(sample(&p, usize::MAX, 0).is_err());
    assert!(sample(&p, 0, 0).is_err());
    assert!(sample_with_draws(&p, &[vec![0.0]]).is_err());
    assert!(sample_with_draws(&p, &[vec![1.0; 7]]).is_err());
    assert!(sample_with_draws(&p, &[vec![f64::NAN; 7]]).is_err());
}
#[test]
fn each_active_gate_changes_its_declared_basis_coordinate() {
    let mut p = packet();
    let x = [0.25, 0.75];
    p.gates = GateValues {
        linear: 0.1,
        asinh: 0.2,
        periodic: 0.3,
        hinge: 0.4,
        pair: 0.5,
        row_mean: 0.6,
    };
    let expected = vec![
        0.025,
        0.075,
        0.2 * 0.25f64.asinh(),
        0.2 * 0.75f64.asinh(),
        0.3,
        -0.3,
        0.0,
        0.1,
        0.5 * 0.25 * 0.75,
        0.3,
    ];
    let actual = basis::features(&x, &p.gates).unwrap();
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-14);
    }
}

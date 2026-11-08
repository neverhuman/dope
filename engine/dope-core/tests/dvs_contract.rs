use dope_core::dvs::{
    self, drivers, roles::DeclaredEvidenceRoles, AuditorCell, CellOutcome, DataPins, DvsRequest,
    FailureReason, ScheduledCell,
};
use serde_json::{json, Value};

fn pins() -> DataPins {
    serde_json::from_value(json!({
        "fit_scope_sha256": "a".repeat(64),
        "real_test_sha256": "b".repeat(64),
        "prepared_view_sha256": "c".repeat(64),
        "groups_sha256": "d".repeat(64),
        "roster_sha256": "e".repeat(64)
    }))
    .unwrap()
}

fn driver_evidence(pins: &DataPins) -> Value {
    let case: Value = serde_json::from_str(include_str!(
        "../../../tests/golden/drivers/v2/v2_additive.json"
    ))
    .unwrap();
    let mut evidence = case["input"].clone();
    evidence["protocol_version"] = json!("driver-rank-informative-v1");
    evidence["importance_unit"] = json!("declared_group");
    evidence["group_ids"] = json!([0, 1]);
    evidence["query_sha256"] = json!(pins.real_test_sha256.clone());
    evidence["groups_sha256"] = json!(pins.groups_sha256.clone());
    evidence["prepared_view_sha256"] = json!(pins.prepared_view_sha256.clone());
    evidence
}

fn request() -> DvsRequest {
    let pins = pins();
    let declared_roles = DeclaredEvidenceRoles::new(pins.clone(), "f".repeat(64)).unwrap();
    serde_json::from_value(json!({
        "protocol":"dope-dvs-v1",
        "preparation_recipe":"nc1-minmax-onehot-v1",
        "pins":pins.clone(),
        "declared_roles":declared_roles,
        "scheduled":[{"cell_sha256":"1".repeat(64),"family_id":0}],
        "required_families":[0],
        "cells":[{
            "cell_sha256":"1".repeat(64),
            "family_id":0,
            "pins":pins.clone(),
            "outcome":{
                "status":"complete",
                "null_losses":[10.0,10.0],
                "real_losses":[2.0,2.0],
                "synthetic_losses":[4.0,4.0],
                "real_gain_lcb":1.0,
                "absolute_gain_floor":0.01,
                "relative_gain_floor":0.01,
                "driver_evidence":driver_evidence(&pins),
                "wall_seconds":null,
                "fit_count":null
            }
        }],
        "selection":null,
        "artifacts":{
            "packet":[1,2],
            "public_schema":[3,4],
            "mandatory_sidecars":[],
            "shared_decoder_parts":[],
            "owner_private_parts":[]
        },
        "privacy":null
    }))
    .unwrap()
}

fn set_losses(input: &mut DvsRequest, null: f64, real: f64, synthetic: f64) {
    let CellOutcome::Complete {
        null_losses,
        real_losses,
        synthetic_losses,
        ..
    } = &mut input.cells[0].outcome
    else {
        panic!("fixture must start with one complete cell");
    };
    *null_losses = vec![null, null];
    *real_losses = vec![real, real];
    *synthetic_losses = vec![synthetic, synthetic];
}

#[test]
fn dvs_retention_is_unclipped_and_matches_hand_computation() {
    let mut above_one = request();
    set_losses(&mut above_one, 4.0, 2.0, 1.5);
    let receipt = dvs::evaluate(&above_one).unwrap();
    assert_eq!(receipt.cells[0].status, "complete");
    assert_eq!(receipt.cells[0].raw_retention, Some(1.25));

    let mut negative = request();
    set_losses(&mut negative, 4.0, 2.0, 4.5);
    let receipt = dvs::evaluate(&negative).unwrap();
    assert_eq!(receipt.cells[0].status, "complete");
    assert_eq!(receipt.cells[0].raw_retention, Some(-0.25));
}

fn grouped_driver(real: [f64; 2], synthetic: [f64; 2], control: [f64; 2]) -> Value {
    let mut evidence = driver_evidence(&pins());
    evidence["real_importances"] = json!(real);
    evidence["synthetic_importances"] = json!(synthetic);
    evidence["real_control_importances"] = json!(control);
    evidence["inference"]["bounds"] = json!({
        "mass.margin":{"estimate":-0.3,"lower":-0.3},
        "mass.stability":{"estimate":1.0,"lower":1.0},
        "mass.real_strength":{"estimate":4.0,"lower":4.0},
        "mass.control_strength":{"estimate":4.0,"lower":4.0},
        "mass.synthetic_strength":{"estimate":1.0,"lower":1.0}
    });
    evidence
}

#[test]
fn dvs_grouped_driver_has_hand_expected_mass_of_one_half() {
    // Group 0 has real/synthetic/control masses 3/1/3; group 1 has 1/3/1.
    // Their normalized masses are (3/4,1/4) and (1/4,3/4), so real-vs-synthetic
    // L1 agreement is 1/2 and real-vs-control agreement is 1.
    let base = drivers::evaluate(&grouped_driver([3.0, 1.0], [1.0, 3.0], [3.0, 1.0]));
    assert_eq!(base["status"], "complete");
    assert!((base["driver_fidelity"].as_f64().unwrap() - 0.5).abs() < 1e-12);
    assert_eq!(base["components"]["mass"]["a_rs"], 0.5);
    assert_eq!(base["components"]["mass"]["a_rr"], 1.0);
}

#[test]
fn dvs_production_grouped_permutation_is_invariant_to_same_group_one_hot_member_duplication() {
    let category = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let companion = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let y: Vec<f64> = category
        .iter()
        .zip(companion)
        .map(|(&a, b)| 2.0 * a + b)
        .collect();
    let original: Vec<Vec<f64>> = category
        .iter()
        .zip(companion)
        .map(|(&a, b)| vec![a, 1.0 - a, b])
        .collect();
    let duplicated: Vec<Vec<f64>> = category
        .iter()
        .zip(companion)
        .map(|(&a, b)| vec![a, 1.0 - a, a, b])
        .collect();
    let options = || dvs::permutation::PermutationOptions {
        repeats: 8,
        seed: 17,
        max_predict_calls: 25,
        task: dvs::permutation::Task::Regression,
    };
    let original_importance = dvs::permutation::grouped_permutation_importance(
        &original,
        &y,
        &[0, 0, 1],
        options(),
        |rows| {
            assert!(rows.iter().all(|r| r[0] + r[1] == 1.0));
            Ok(rows.iter().map(|r| vec![2.0 * r[0] + r[2]]).collect())
        },
    )
    .unwrap();
    let duplicated_importance = dvs::permutation::grouped_permutation_importance(
        &duplicated,
        &y,
        &[0, 0, 0, 1],
        options(),
        |rows| {
            assert!(rows.iter().all(|r| r[0] + r[1] == 1.0 && r[0] == r[2]));
            Ok(rows.iter().map(|r| vec![2.0 * r[0] + r[3]]).collect())
        },
    )
    .unwrap();

    assert_eq!(original_importance.len(), 2);
    assert_eq!(duplicated_importance.len(), 2);
    assert_eq!(
        original_importance
            .iter()
            .map(|g| g.group_id)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        duplicated_importance
            .iter()
            .map(|g| g.group_id)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    for (original, duplicated) in original_importance.iter().zip(&duplicated_importance) {
        assert_eq!(original.group_id, duplicated.group_id);
        assert!((original.mean_loss_increase - duplicated.mean_loss_increase).abs() < 1e-12);
    }
    let real: Vec<f64> = original_importance
        .iter()
        .map(|g| g.mean_loss_increase.max(0.0))
        .collect();
    let split: Vec<f64> = duplicated_importance
        .iter()
        .map(|g| g.mean_loss_increase.max(0.0))
        .collect();
    assert_eq!(drivers::mass_agreement(&real, &split).unwrap(), 1.0);
    assert!(real.iter().all(|v| *v > 0.0));

    let successor = drivers::rank_agreement_informative(&real, &split).unwrap();
    let identical = drivers::rank_agreement_informative(&real, &real).unwrap();
    assert_eq!(successor, identical);
    assert_eq!(successor["status"], "complete");
    assert_eq!(successor["value"], 1.0);
    assert_eq!(successor["informative_pairs"], 1);

    let tied = drivers::rank_agreement_informative(&[1.0, 1.0], &real).unwrap();
    assert_eq!(tied["status"], "uninformative");
    assert_eq!(tied["reason"], "no_informative_rank_pairs");
}

fn cell(id: char, pins: &DataPins, outcome: CellOutcome) -> AuditorCell {
    AuditorCell {
        cell_sha256: id.to_string().repeat(64),
        family_id: 0,
        pins: pins.clone(),
        outcome,
    }
}

#[test]
fn dvs_keeps_every_scheduled_failure_and_unknown_cost_as_null() {
    let mut input = request();
    let pins = input.pins.clone();
    input.cells[0].outcome = CellOutcome::Failed {
        reason: FailureReason::NativeFailure,
        wall_seconds: None,
        fit_count: None,
    };
    for id in ['2', '3', '4'] {
        input.scheduled.push(ScheduledCell {
            cell_sha256: id.to_string().repeat(64),
            family_id: 0,
        });
    }
    input.cells.push(cell(
        '2',
        &pins,
        CellOutcome::TimedOut {
            wall_seconds: None,
            fit_count: None,
        },
    ));
    input.cells.push(cell('3', &pins, CellOutcome::Unsupported));
    // The fourth scheduled identity is intentionally absent from cells.
    let receipt = dvs::evaluate(&input).unwrap();
    assert_eq!(receipt.scheduled_cells, 4);
    assert_eq!(receipt.failed_or_missing_cells, 4);
    assert_eq!(
        receipt.cells.iter().map(|c| c.status).collect::<Vec<_>>(),
        vec!["failed", "timed_out", "unsupported", "missing"]
    );
    let value = serde_json::to_value(receipt).unwrap();
    for cell in value["cells"].as_array().unwrap() {
        assert!(cell["wall_seconds"].is_null());
        assert!(cell["fit_count"].is_null());
    }
}

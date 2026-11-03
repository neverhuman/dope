use dope_core::dvs::{self, drivers, permutation, receipt, roles, selection, DataPins, DvsRequest};
use serde_json::{json, Value};

fn pins() -> Value {
    json!({"fit_scope_sha256":"a".repeat(64), "real_test_sha256":"b".repeat(64), "prepared_view_sha256":"c".repeat(64), "groups_sha256":"d".repeat(64), "roster_sha256":"e".repeat(64)})
}
fn declared_roles(pins: &Value) -> Value {
    let common: DataPins = serde_json::from_value(pins.clone()).unwrap();
    serde_json::to_value(roles::DeclaredEvidenceRoles::new(common, "f".repeat(64)).unwrap())
        .unwrap()
}
fn driver() -> Value {
    let case: Value = serde_json::from_str(include_str!(
        "../../../tests/golden/drivers/v2/v2_additive.json"
    ))
    .unwrap();
    let mut e = case["input"].clone();
    e["protocol_version"] = json!("driver-rank-informative-v1");
    e["importance_unit"] = json!("declared_group");
    e["group_ids"] = json!([0, 1]);
    e["query_sha256"] = pins()["real_test_sha256"].clone();
    e["groups_sha256"] = pins()["groups_sha256"].clone();
    e["prepared_view_sha256"] = pins()["prepared_view_sha256"].clone();
    e
}
fn request() -> DvsRequest {
    let common_pins = pins();
    let declared_roles = declared_roles(&common_pins);
    serde_json::from_value(json!({"protocol":"dope-dvs-v1", "preparation_recipe":"nc1-minmax-onehot-v1", "pins":common_pins.clone(), "declared_roles":declared_roles,
        "scheduled":[{"cell_sha256":"1".repeat(64),"family_id":0}], "required_families":[0],
        "cells":[{"cell_sha256":"1".repeat(64),"family_id":0,"pins":common_pins.clone(),"outcome":{"status":"complete",
            "null_losses":[10.,10.],"real_losses":[2.,2.],"synthetic_losses":[4.,4.],
            "real_gain_lcb":1.,"absolute_gain_floor":0.01,"relative_gain_floor":0.01,
            "driver_evidence":driver(),"wall_seconds":null,"fit_count":2}}],
        "selection":{"synthetic_validation_losses":[1.,2.,3.],"real_fit_test_losses":[3.,1.,2.],"synthetic_fit_test_losses":[3.,1.,2.]},
        "artifacts":{"packet":[1,2,3],"public_schema":[4,5],"mandatory_sidecars":[[6,7]],"shared_decoder_parts":[[8,9,10]],"owner_private_parts":[[99,98,97,96]]},
        "privacy":{"pins":common_pins,"real_fit":[[0.],[1.]],"synthetic":[[0.],[0.5]],"real_holdout":[[0.25],[0.75]],"max_distance_pairs":8}
    })).unwrap()
}

#[test]
fn dvs_analytic_receipt_preserves_unclipped_r_bytes_and_no_qualification() {
    let r = dvs::evaluate(&request()).unwrap();
    assert_eq!(r.status, "complete");
    assert_eq!(r.robust_retention, Some(0.75));
    assert_eq!(r.bytes.payload_bytes, 7);
    assert_eq!(r.bytes.shared_bytes, 3);
    assert_eq!(r.bytes.sampling_closure_bytes, 10);
    assert_eq!(r.bytes.owner_private_dependency_bytes, 4);
    assert_eq!(r.selection["selected_index"], 0);
    assert_eq!(r.selection["selection_regret"], 2.);
    assert_eq!(r.selection["transfer_rank"]["value"], 1.);
    assert_eq!(r.privacy["exact_copies"], 1);
    assert_eq!(r.privacy["dcr_ratio"], 1.);
    assert_eq!(r.role_binding.state, "caller_declared_hash_checked");
    assert_eq!(
        r.role_binding.binding_sha256,
        "1c687aa2f84640405a667696ef1483f2324402588a4e6dcbcdd2b5afe2ad5763"
    );
    assert_eq!(r.qualification_state, "not_run");
    assert_eq!(r.formal_dp_state, "not_run");
    let card = receipt::audit_card(&r);
    assert!(card.contains("Payload bytes: 7"));
    assert!(card.contains("Robust retention: 0.750000"));
    assert!(card.contains("Evidence role binding: caller_declared_hash_checked"));
    assert!(card.contains(r.role_binding.binding_sha256.as_str()));
    assert!(!card.contains("CANARY_PRIVATE"));
}

#[test]
fn dvs_schedule_missing_failure_and_uninformative_cells_stay_in_denominator() {
    let mut input = request();
    input
        .scheduled
        .push(serde_json::from_value(json!({"cell_sha256":"2".repeat(64),"family_id":0})).unwrap());
    let r = dvs::evaluate(&input).unwrap();
    assert_eq!(r.scheduled_cells, 2);
    assert_eq!(r.failed_or_missing_cells, 1);
    assert_eq!(r.status, "incomplete");
    assert_eq!(r.robust_retention, None);
    assert_eq!(r.cells[1].status, "missing");
    input.cells[0].outcome = dvs::CellOutcome::Complete {
        null_losses: vec![1.],
        real_losses: vec![1.],
        synthetic_losses: vec![0.],
        real_gain_lcb: 0.,
        absolute_gain_floor: 0.01,
        relative_gain_floor: 0.01,
        driver_evidence: None,
        wall_seconds: None,
        fit_count: None,
    };
    let r = dvs::evaluate(&input).unwrap();
    assert_eq!(r.uninformative_cells, 1);
    assert_eq!(r.cells[0].raw_retention, None);
}

#[test]
fn dvs_identity_and_strict_input_refusals_are_source_free() {
    let mut input = request();
    input.cells[0].pins.real_test_sha256 = "f".repeat(64);
    assert_eq!(
        dvs::evaluate(&input).unwrap_err().0,
        "dvs_cell_identity_mismatch"
    );
    let mut input = request();
    input.cells.push(input.cells[0].clone());
    assert_eq!(
        dvs::evaluate(&input).unwrap_err().0,
        "unexpected_or_duplicate_cell"
    );
    let mut input = request();
    input.preparation_recipe = "CANARY_PRIVATE_HEADER".into();
    let error = dvs::evaluate(&input).unwrap_err().to_string();
    assert_eq!(error, "unsupported_preparation_recipe");
    assert!(!error.contains("CANARY"));
    let mut input = request();
    input.scheduled.clear();
    assert_eq!(dvs::evaluate(&input).unwrap_err().0, "empty_dvs_schedule");
}

#[test]
fn dvs_driver_c02_port_replays_legacy_and_new_rank_fixtures() {
    let case: Value = serde_json::from_str(include_str!(
        "../../../tests/golden/drivers/v2/v2_additive.json"
    ))
    .unwrap();
    assert_eq!(
        drivers::evaluate_c02_legacy_v2(&case["input"]),
        case["expected"]
    );
    let cases: Value = serde_json::from_str(include_str!(
        "../../../tests/golden/drivers/informative-v1/rank-cases.json"
    ))
    .unwrap();
    for c in cases.as_array().unwrap() {
        let a: Vec<f64> = serde_json::from_value(c["real"].clone()).unwrap();
        let b: Vec<f64> = serde_json::from_value(c["synthetic"].clone()).unwrap();
        assert_eq!(
            drivers::rank_agreement_informative(&a, &b).unwrap(),
            c["expected"]
        );
    }
    let mut e = driver();
    e["inference"]["bound_method"] = json!("ratio_only");
    assert_eq!(
        drivers::evaluate(&e)["reason"],
        "non_simultaneous_driver_bounds"
    );
    let mut e = driver();
    e["inference"] = Value::Null;
    assert_eq!(drivers::evaluate(&e)["status"], "incomplete");
}

#[test]
fn dvs_selection_never_uses_real_test_to_choose_and_ties_do_not_win() {
    let mut e = selection::SelectionEvidence {
        synthetic_validation_losses: vec![1., 2., 3.],
        real_fit_test_losses: vec![3., 1., 2.],
        synthetic_fit_test_losses: vec![3., 1., 2.],
    };
    let a = selection::evaluate(&e).unwrap();
    assert_eq!(a.selected_index, Some(0));
    assert_eq!(a.selection_regret, Some(2.));
    e.real_fit_test_losses = vec![10., 20., 1.];
    let b = selection::evaluate(&e).unwrap();
    assert_eq!(b.selected_index, Some(0));
    e.synthetic_validation_losses = vec![1., 1., 2.];
    let c = selection::evaluate(&e).unwrap();
    assert_eq!(c.selected_index, None);
    assert_eq!(c.selection_regret, None);
    assert_eq!(c.status, "uninformative");
}

#[test]
fn dvs_privacy_budget_and_zero_reference_distance_are_explicit() {
    let mut p = request().privacy.unwrap();
    p.max_distance_pairs = 7;
    assert_eq!(
        dvs::privacy_diagnostics(&p).unwrap_err().0,
        "privacy_budget_exceeded"
    );
    p.max_distance_pairs = 8;
    p.real_holdout = vec![vec![0.], vec![1.]];
    let d = dvs::privacy_diagnostics(&p).unwrap();
    assert_eq!(d["status"], "uninformative");
    assert!(d["dcr_ratio"].is_null());
}

#[test]
fn dvs_grouped_permutation_preserves_one_hot_groups_and_replays() {
    let x: Vec<Vec<f64>> = (0..16)
        .map(|i| vec![(i % 2) as f64, 1. - (i % 2) as f64, i as f64])
        .collect();
    let y: Vec<f64> = x.iter().map(|r| r[0]).collect();
    let run = || {
        permutation::grouped_permutation_importance(
            &x,
            &y,
            &[0, 0, 1],
            permutation::PermutationOptions {
                repeats: 4,
                seed: 17,
                max_predict_calls: 9,
                task: permutation::Task::Regression,
            },
            |rows| {
                assert!(rows.iter().all(|r| r[0] + r[1] == 1.));
                Ok(rows.iter().map(|r| vec![r[0]]).collect())
            },
        )
        .unwrap()
    };
    let a = run();
    let b = run();
    assert!(a[0].mean_loss_increase > 0.);
    assert_eq!(a[1].mean_loss_increase, 0.);
    assert_eq!(a[0].mean_loss_increase, b[0].mean_loss_increase);
    let mut calls = 0;
    let e = permutation::grouped_permutation_importance(
        &x,
        &y,
        &[0, 0, 1],
        permutation::PermutationOptions {
            repeats: 4,
            seed: 17,
            max_predict_calls: 8,
            task: permutation::Task::Regression,
        },
        |_| {
            calls += 1;
            Ok(vec![])
        },
    );
    assert_eq!(e.unwrap_err().0, "permutation_budget_exceeded");
    assert_eq!(calls, 0);
}

#[test]
fn dvs_entire_legacy_v2_c02_corpus_is_unchanged() {
    let cases = [
        include_str!("../../../tests/golden/drivers/v2/v2_additive.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_boolean_importance.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_control_strength_uncertain.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_correlated_proxy_swap.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_d80_boundary.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_duplicate_component.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_invalid_pin.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_mismatched_bound.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_missing_bound_coordinate.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_missing_bounds.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_missing_perturbation.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_missing_pin.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_negative_noise.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_nominal_no_sign.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_paired_control.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_perturbation_failed.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_perturbation_timed_out.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_perturbation_unsupported.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_point_ratio_insufficient.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_pointwise_bounds.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_real_strength_uncertain.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_shape_mismatch.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_sign_reversal.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_signed_direction.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_stability_uncertain.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_synthetic_strength_uncertain.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_u_shape.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_undeclared_components.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_ungrouped_proxy_swap.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_unstable_real.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_weak_synthetic_strength.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_xor_pair.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_xor_pair_loss.json"),
        include_str!("../../../tests/golden/drivers/v2/v2_zero_importance.json"),
    ];
    // Existing corpus uses the canonical 1e-12 Rust/Python golden tolerance;
    // archived decimal literals need not be identical binary64 encodings.
    fn close(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Number(x), Value::Number(y)) => {
                (x.as_f64().unwrap() - y.as_f64().unwrap()).abs() <= 1e-12
            }
            (Value::Array(x), Value::Array(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(a, b)| close(a, b))
            }
            (Value::Object(x), Value::Object(y)) => {
                x.len() == y.len() && x.iter().all(|(k, a)| y.get(k).is_some_and(|b| close(a, b)))
            }
            _ => a == b,
        }
    }
    for source in cases {
        let case: Value = serde_json::from_str(source).unwrap();
        assert!(
            close(
                &drivers::evaluate_c02_legacy_v2(&case["input"]),
                &case["expected"]
            ),
            "{}",
            case["case_id"]
        );
    }
}

#[test]
fn dvs_role_binding_refuses_missing_wrong_or_reused_views_before_selection() {
    let mut input = request();
    input
        .selection
        .as_mut()
        .unwrap()
        .synthetic_validation_losses
        .clear();
    input
        .declared_roles
        .as_mut()
        .unwrap()
        .synthetic_validation_usage = "assessment_only".into();
    assert_eq!(
        dvs::evaluate(&input).unwrap_err().0,
        "invalid_dvs_role_binding"
    );

    let mut input = request();
    input
        .selection
        .as_mut()
        .unwrap()
        .synthetic_validation_losses
        .clear();
    let reused_pin = input.pins.real_test_sha256.clone();
    input.pins.fit_scope_sha256 = reused_pin.clone();
    for cell in &mut input.cells {
        cell.pins.fit_scope_sha256 = reused_pin.clone();
    }
    input.privacy.as_mut().unwrap().pins.fit_scope_sha256 = reused_pin.clone();
    let roles = input.declared_roles.as_mut().unwrap();
    roles.common_pins = input.pins.clone();
    roles.real_fit_sha256 = reused_pin;
    roles.binding_sha256 = roles.recompute_binding_sha256().unwrap();
    assert_eq!(
        dvs::evaluate(&input).unwrap_err().0,
        "invalid_dvs_role_binding"
    );

    let mut input = request();
    input.declared_roles = None;
    assert_eq!(
        dvs::evaluate(&input).unwrap_err().0,
        "missing_dvs_role_binding"
    );

    let mut input = request();
    let roles = input.declared_roles.as_mut().unwrap();
    roles.synthetic_validation_sha256 = roles.real_test_sha256.clone();
    roles.binding_sha256 = roles.recompute_binding_sha256().unwrap();
    assert_eq!(
        dvs::evaluate(&input).unwrap_err().0,
        "invalid_dvs_role_binding"
    );

    let mut input = request();
    let roles = input.declared_roles.as_mut().unwrap();
    roles.common_pins.real_test_sha256 = "c".repeat(64);
    roles.binding_sha256 = roles.recompute_binding_sha256().unwrap();
    assert_eq!(
        dvs::evaluate(&input).unwrap_err().0,
        "invalid_dvs_role_binding"
    );

    let mut input = request();
    input.declared_roles.as_mut().unwrap().binding_sha256 = "0".repeat(64);
    assert_eq!(
        dvs::evaluate(&input).unwrap_err().0,
        "invalid_dvs_role_binding"
    );
}

#[test]
fn dvs_real_test_poison_changes_assessment_but_not_synthetic_choice() {
    let mut input = request();
    let first = dvs::evaluate(&input).unwrap();
    assert_eq!(first.selection["selected_index"], 0);
    let new_test_pin = "9".repeat(64);
    input.pins.real_test_sha256 = new_test_pin.clone();
    for cell in &mut input.cells {
        cell.pins.real_test_sha256 = new_test_pin.clone();
    }
    input.privacy.as_mut().unwrap().pins.real_test_sha256 = new_test_pin.clone();
    let roles = input.declared_roles.as_mut().unwrap();
    roles.common_pins = input.pins.clone();
    roles.real_test_sha256 = new_test_pin.clone();
    roles.binding_sha256 = roles.recompute_binding_sha256().unwrap();
    if let dvs::CellOutcome::Complete {
        driver_evidence: Some(evidence),
        ..
    } = &mut input.cells[0].outcome
    {
        evidence["query_sha256"] = json!(new_test_pin);
    }
    input.selection.as_mut().unwrap().real_fit_test_losses = vec![100., 0.5, 0.25];
    let poisoned = dvs::evaluate(&input).unwrap();
    assert_eq!(poisoned.selection["selected_index"], 0);
    assert_ne!(
        poisoned.selection["real_best_index"],
        first.selection["real_best_index"]
    );
}

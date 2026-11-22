//! Behavioral tests for `dope-fidelity/v3` (`T17`, `T18`, `T22`, `T29`, `T30`,
//! `S_deploy`, `C2ST`/`MargAUC`, empirical privacy, and correlated driver groups).

use dope_core::dvs::v3::bundle::{PredictionBundle, TaskKind};
use dope_core::dvs::v3::deploy::{assess_deploy_selection, RosterConfigScore};
use dope_core::dvs::v3::equiv::{DriverEquivalenceInput, KappaThresholds};
use dope_core::dvs::v3::groups::correlated_driver_groups;
use dope_core::dvs::v3::harm::donor_bootstrap_bounds;
use dope_core::dvs::v3::privacy::{assess_empirical_privacy, EmpiricalPrivacyInput};
use dope_core::dvs::v3::qualify::{family_clustered_holm_comparison, FamilyComparisonRow};
use dope_core::dvs::v3::realism::{column_marginal_auc, RealismInput};
use dope_core::dvs::v3::{
    evaluate_v3, FidelityCellInput, FidelityCellRequest, FidelityV3Request, PROTOCOL_V3,
};

fn regression_bundle(n: usize, synth_bias: f64, donor_ids: Option<Vec<u64>>) -> PredictionBundle {
    let mut y_true = Vec::with_capacity(n);
    let mut pred_a = Vec::with_capacity(n);
    let mut pred_b = Vec::with_capacity(n);
    let mut pred_s = Vec::with_capacity(n);
    let mut null_losses = Vec::with_capacity(n);
    let mut real_a_losses = Vec::with_capacity(n);
    let mut real_b_losses = Vec::with_capacity(n);
    let mut synth_losses = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64) / (n as f64) * 4.0 - 2.0;
        let y = 1.5 * x + 0.1 * ((i % 7) as f64 - 3.0);
        let pa = 1.5 * x;
        let pb = 1.48 * x + 0.02;
        let ps = 1.5 * x + synth_bias;
        y_true.push(y);
        pred_a.push(pa);
        pred_b.push(pb);
        pred_s.push(ps);
        null_losses.push(y * y);
        real_a_losses.push((y - pa).powi(2));
        real_b_losses.push((y - pb).powi(2));
        synth_losses.push((y - ps).powi(2));
    }
    PredictionBundle {
        task: TaskKind::Regression,
        null_losses,
        real_a_losses,
        real_b_losses: Some(real_b_losses),
        synth_losses,
        donor_ids,
        y_true,
        pred_a,
        pred_b: Some(pred_b),
        pred_s,
        num_outputs: 1,
    }
}

#[test]
fn t17_refuses_degenerate_constant_query_target() {
    let mut b = regression_bundle(40, 0.0, None);
    b.y_true = vec![1.0; 40];
    let err = b.validate().unwrap_err();
    assert_eq!(err.0, "degenerate_constant_query_target");
}

#[test]
fn t18_and_t22_unclipped_retention_and_complete_failure_denominator() {
    let good = regression_bundle(80, 0.01, None);
    // Synthetic beats real A slightly -> unclipped retention > 1.0 (`T22`).
    let mut super_b = regression_bundle(80, 0.0, None);
    super_b.synth_losses = super_b.real_a_losses.iter().map(|v| v * 0.85).collect();
    assert!(super_b.unclipped_retention_s().unwrap() > 1.0);

    let req = FidelityV3Request {
        protocol: PROTOCOL_V3.to_string(),
        mode: "faithful".to_string(),
        seed: 42,
        bootstrap_draws: 400,
        kappa_thresholds: KappaThresholds::default(),
        cells: vec![
            FidelityCellRequest {
                cell_id: "c1".into(),
                family_id: "fam_1".into(),
                auditor_id: "gbdt".into(),
                input: FidelityCellInput::Complete {
                    bundle: Box::new(good),
                    driver: Some(Box::new(DriverEquivalenceInput {
                        perm_a: vec![0.6, 0.3, 0.1],
                        perm_b: Some(vec![0.58, 0.31, 0.11]),
                        perm_s: vec![0.59, 0.30, 0.11],
                        ale_a: Some(vec![0.1, 0.5, 0.9]),
                        ale_b: Some(vec![0.12, 0.49, 0.88]),
                        ale_s: Some(vec![0.11, 0.50, 0.89]),
                        coalition_a: None,
                        coalition_b: None,
                        coalition_s: None,
                    })),
                    deploy_roster: Some(vec![
                        RosterConfigScore {
                            config_id: "best".into(),
                            true_real_loss: 0.10,
                            synth_select_loss: 0.11,
                            twin_b_select_loss: Some(0.11),
                        },
                        RosterConfigScore {
                            config_id: "worse".into(),
                            true_real_loss: 0.25,
                            synth_select_loss: 0.24,
                            twin_b_select_loss: Some(0.26),
                        },
                    ]),
                    realism: Some(Box::new(RealismInput {
                        c2st_labels: vec![0, 0, 0, 0, 1, 1, 1, 1],
                        c2st_scores: vec![0.45, 0.55, 0.48, 0.52, 0.46, 0.54, 0.50, 0.51],
                        marginal_aucs: vec![0.53, 0.55],
                        synth_rule_violation_rate: Some(0.01),
                        real_rule_violation_rate: Some(0.01),
                        illegal_rows: 0,
                    })),
                    privacy: None,
                },
            },
            FidelityCellRequest {
                cell_id: "c2_failed".into(),
                family_id: "fam_2".into(),
                auditor_id: "gbdt".into(),
                input: FidelityCellInput::TimedOut,
            },
        ],
    };
    let receipt = evaluate_v3(&req).unwrap();
    assert_eq!(receipt.composite_score, "forbidden");
    assert_eq!(receipt.formal_dp_state, "not_claimed");
    assert_eq!(receipt.coverage.scheduled_cells, 2);
    assert_eq!(receipt.coverage.eligible_denominator, 2);
    assert_eq!(receipt.coverage.failed_or_missing_cells, 1);
    assert!((receipt.coverage.overall_pass_coverage - 0.5).abs() < 1e-9);
}

#[test]
fn t29_donor_aware_bootstrap_widens_interval_under_clustered_donors() {
    // 10 clusters of 10 identical-sign rows -> clustered uncertainty >> naive i.i.d.
    let mut deltas = Vec::with_capacity(100);
    let mut donors = Vec::with_capacity(100);
    for c in 0..10u64 {
        let val = if c % 2 == 0 { 1.0 } else { -0.8 };
        for _ in 0..10 {
            deltas.push(val);
            donors.push(c);
        }
    }
    let (_, lo_iid, hi_iid) = donor_bootstrap_bounds(&deltas, None, 1000, 7, 0.95).unwrap();
    let (_, lo_cl, hi_cl) = donor_bootstrap_bounds(&deltas, Some(&donors), 1000, 7, 0.95).unwrap();
    assert!(
        (hi_cl - lo_cl) > 2.0 * (hi_iid - lo_iid),
        "clustered width {} must exceed 2x iid width {}",
        hi_cl - lo_cl,
        hi_iid - lo_iid
    );
}

#[test]
fn t30_family_clustered_holm_ci_detects_winner_and_adjusts_multiplicity() {
    let mut strong = Vec::new();
    let mut tie = Vec::new();
    for f in 0..90 {
        let fam = format!("fam_{f:03}");
        strong.push(FamilyComparisonRow {
            family_id: fam.clone(),
            candidate_score: 0.92,
            comparator_score: 0.80 + (f % 3) as f64 * 0.01,
        });
        tie.push(FamilyComparisonRow {
            family_id: fam,
            candidate_score: 0.85,
            comparator_score: 0.85 + if f % 2 == 0 { 0.01 } else { -0.01 },
        });
    }
    let res = family_clustered_holm_comparison(
        &[("rival_a".into(), strong), ("rival_tie".into(), tie)],
        2000,
        123,
    )
    .unwrap();
    assert!(res[0].strictly_beats_at_05);
    assert!(res[0].ci95_low > 0.0);
    assert!(!res[1].strictly_beats_at_05);
    assert!(res[0].holm_adjusted_p_value >= res[0].one_sided_p_value);
}

#[test]
fn privacy_v3_deploy_groups_and_marg_auc_work_end_to_end() {
    // Correlated columns 0 & 1 plus parent-linked column 2 form one group; column 3 is separate.
    let c0: Vec<f64> = (0..60).map(|i| i as f64).collect();
    let c1: Vec<f64> = (0..60).map(|i| (i as f64) * 2.0 + 1.0).collect();
    let c2: Vec<f64> = (0..60).map(|i| ((i * 17) % 13) as f64).collect();
    let c3: Vec<f64> = (0..60).map(|i| ((i * 29) % 11) as f64).collect();
    let groups =
        correlated_driver_groups(&[c0.clone(), c1, c2, c3], Some(&[0, 1, 0, 3]), 0.70).unwrap();
    assert_eq!(groups, vec![vec![0, 1, 2], vec![3]]);

    // Same distribution has low MargAUC; shifted distribution has high MargAUC.
    let same_auc = column_marginal_auc(&c0, &c0).unwrap();
    let shifted: Vec<f64> = c0.iter().map(|v| v + 40.0).collect();
    let shift_auc = column_marginal_auc(&c0, &shifted).unwrap();
    assert!(same_auc <= 0.55);
    assert!(shift_auc >= 0.85);

    // Empirical privacy: synthetic symmetric between A and B passes; copying A fails.
    let a: Vec<Vec<f64>> = (0..30).map(|i| vec![(i as f64) * 0.1, 0.0]).collect();
    let b: Vec<Vec<f64>> = (0..30)
        .map(|i| vec![(i as f64) * 0.1 + 0.05, 0.0])
        .collect();
    let s_good: Vec<Vec<f64>> = (0..30)
        .map(|i| vec![(i as f64) * 0.1 + 0.025, 0.02])
        .collect();
    let good_p = assess_empirical_privacy(&EmpiricalPrivacyInput {
        real_fit_a: a.clone(),
        real_holdout_b: b.clone(),
        synthetic: s_good,
        min_emitted_support: 10,
        canary_rows: vec![vec![99.0, 99.0]],
        max_distance_pairs: 100_000,
    })
    .unwrap();
    assert!(good_p.pass);
    assert_eq!(good_p.formal_dp_state, "not_claimed");

    let bad_p = assess_empirical_privacy(&EmpiricalPrivacyInput {
        real_fit_a: a.clone(),
        real_holdout_b: b,
        synthetic: a,
        min_emitted_support: 10,
        canary_rows: vec![],
        max_distance_pairs: 100_000,
    })
    .unwrap();
    assert!(!bad_p.pass);
    assert_eq!(bad_p.exact_copies, 30);

    // Selection transfer S_deploy check.
    let dep = assess_deploy_selection(
        &[
            RosterConfigScore {
                config_id: "c1".into(),
                true_real_loss: 0.10,
                synth_select_loss: 0.12,
                twin_b_select_loss: Some(0.11),
            },
            RosterConfigScore {
                config_id: "c2".into(),
                true_real_loss: 0.20,
                synth_select_loss: 0.10,
                twin_b_select_loss: Some(0.19),
            },
        ],
        0.01,
    )
    .unwrap();
    assert!(!dep.pass);
    assert!((dep.synth_regret - 0.10).abs() < 1e-9);
}

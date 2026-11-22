//! Genuine End-to-End Fidelity Engine 2.0 Integration Test (`G-01`..`G-05`).
//!
//! Exercises the real compiled pipeline with ZERO fake placeholders:
//! `RawTable (A)` -> `fit_v2` -> Format 2 binary `.dope` bytes -> `decode_format2_program` ->
//! stateless `sample_with_mode` (`S`) -> real downstream Ridge + GBDT auditors trained on `A`,
//! disjoint twin `B`, and decoded `S` scored on disjoint held-out `T` -> real grouped permutation
//! driver importance -> native 2-fold depth-3 boosted-tree C2ST & MargAUC -> empirical privacy
//! DCR/MIA/exact-copy-excess -> strict fail-closed `evaluate_v3(allow_partial_diagnostic: false)`
//! -> `embed_packet` & `embed_to_629_bridge` on Format 2!

use dope_core::dvs::v3::bundle::{PredictionBundle, TaskKind};
use dope_core::dvs::v3::c2st::build_native_realism_input;
use dope_core::dvs::v3::equiv::{DriverEquivalenceInput, KappaThresholds};
use dope_core::dvs::v3::privacy::EmpiricalPrivacyInput;
use dope_core::dvs::v3::{
    evaluate_v3, FidelityCellInput, FidelityCellRequest, FidelityV3Request, PROTOCOL_V3,
};
use dope_core::embed::{embed_packet, embed_to_629_bridge, PACKET_FEATURES_V2_DIM};
use dope_core::target::Head;
use dope_core::teacher::panel::{GbdtMember, LinearMember};
use dope_core::{
    decode_format2_program, fit_v2, is_format2, EmissionMode, EmitOptions, FitV2Options, RawTable,
    Task,
};

fn make_synthetic_table(n: usize, seed_offset: u64) -> RawTable {
    let mut x0 = Vec::with_capacity(n);
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut x3 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let mut s = ((i as u64).wrapping_add(1 + seed_offset)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        s = (s ^ (s >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let u0 = ((s >> 11) as f64) / ((1u64 << 53) as f64);
        s = s.wrapping_mul(0x94d0_49bb_1331_11eb);
        let u1 = ((s >> 11) as f64) / ((1u64 << 53) as f64);
        s = s.wrapping_mul(0x517c_c1b7_2722_0a95);
        let u2 = ((s >> 11) as f64) / ((1u64 << 53) as f64);
        s = s.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let eps = (((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5) * 0.25;

        let v0 = if i % 4 == 0 {
            0.0
        } else {
            ((u0 * 100.0).round()) / 100.0
        };
        let v1 = v0 + 0.1 + 0.5 * u1; // structural inequality v0 <= v1
        let v2 = v0 + v1; // exact arithmetic identity v2 = v0 + v1
        let v3 = (u2 * 4.0).floor(); // discrete 4-level integer feature
        let yi = 2.0 * v0 + 1.5 * v1 - 0.8 * v3 + 0.6 * (v0 * v1) + eps;
        x0.push(format!("{v0:.4}"));
        x1.push(format!("{v1:.4}"));
        x2.push(format!("{v2:.4}"));
        x3.push(format!("{v3:.0}"));
        y.push(format!("{yi:.5}"));
    }
    RawTable {
        headers: vec![
            "x0".into(),
            "x1".into(),
            "x2".into(),
            "x3".into(),
            "y".into(),
        ],
        columns: vec![x0, x1, x2, x3, y],
    }
}

fn cols_to_rows(cols: &[Vec<f64>], take: usize) -> Vec<Vec<f64>> {
    let d = cols.len();
    let n = cols[0].len().min(take);
    (0..n)
        .map(|r| (0..d).map(|j| cols[j][r]).collect())
        .collect()
}

fn perm_importance_linear(model: &LinearMember, x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let rows: Vec<usize> = (0..n).collect();
    let base_pred = model.predict_rows(x, &rows);
    let base_mse = y
        .iter()
        .zip(&base_pred)
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        / n as f64;
    let d = x.len();
    let mut imp = Vec::with_capacity(d);
    for j in 0..d {
        let mut x_perm = x.to_vec();
        x_perm[j].rotate_left(17 % n.max(1));
        let p = model.predict_rows(&x_perm, &rows);
        let mse = y
            .iter()
            .zip(&p)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64;
        imp.push((mse - base_mse).max(0.0));
    }
    imp
}

#[test]
fn fidelity_engine_2_full_pipeline_end_to_end() {
    let table_a = make_synthetic_table(500, 100);
    let table_b = make_synthetic_table(500, 10_100);
    let table_t = make_synthetic_table(300, 20_100);

    let fitted = fit_v2(
        &table_a,
        4,
        &FitV2Options {
            budget: Some(65_536),
            seed: 42,
            task: Some(Task::Regression),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(is_format2(&fitted.bytes));
    assert!(fitted.bytes.len() <= 65_536);
    assert!(fitted.report.teacher_beats_null);
    assert!(fitted.report.rules_mined >= 1);
    assert!(fitted.report.student_trees >= 1);

    let decoded = decode_format2_program(&fitted.bytes).unwrap();
    assert_eq!(decoded.n_fit, 500);
    assert_eq!(decoded.auto_rows(EmissionMode::Faithful), 2_000);
    assert_eq!(decoded.auto_rows(EmissionMode::Distill), 10_000);

    let emitted = decoded
        .sample_with_mode(&EmitOptions {
            mode: EmissionMode::Faithful,
            rows: 1000,
            seed: 99,
            table_id: 7,
            ..Default::default()
        })
        .unwrap();
    let s_x = &emitted.samples.x;
    let s_y = &emitted.samples.y;
    assert_eq!(s_x.len(), 4);
    assert_eq!(s_y.len(), 1000);
    assert!(decoded.catf.rules.violation_rate(s_x) <= 0.05);

    let dist = decoded
        .sample_with_mode(&EmitOptions {
            mode: EmissionMode::Distill,
            rows: 200,
            seed: 101,
            table_id: 8,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(dist.weights.as_ref().unwrap().len(), 200);
    assert_eq!(dist.pair_ids.as_ref().unwrap().len(), 200);

    let data_a = decoded.schema.to_model(&table_a).unwrap();
    let data_b = decoded.schema.to_model(&table_b).unwrap();
    let data_t = decoded.schema.to_model(&table_t).unwrap();
    let rows_a: Vec<usize> = (0..data_a.y.len()).collect();
    let rows_b: Vec<usize> = (0..data_b.y.len()).collect();
    let rows_s: Vec<usize> = (0..s_y.len()).collect();
    let rows_t: Vec<usize> = (0..data_t.y.len()).collect();

    let lin_a = LinearMember::fit(Head::Regression, &data_a.x, &data_a.y, &rows_a);
    let lin_b = LinearMember::fit(Head::Regression, &data_b.x, &data_b.y, &rows_b);
    let lin_s = LinearMember::fit(Head::Regression, s_x, s_y, &rows_s);
    let pred_a_lin = lin_a.predict_rows(&data_t.x, &rows_t);
    let pred_b_lin = lin_b.predict_rows(&data_t.x, &rows_t);
    let pred_s_lin = lin_s.predict_rows(&data_t.x, &rows_t);

    let gbdt_a = GbdtMember::fit(Head::Regression, &data_a.x, &data_a.y, &rows_a, 4, 40, 1);
    let gbdt_b = GbdtMember::fit(Head::Regression, &data_b.x, &data_b.y, &rows_b, 4, 40, 2);
    let gbdt_s = GbdtMember::fit(Head::Regression, s_x, s_y, &rows_s, 4, 40, 3);
    let pred_a_gb = gbdt_a.predict_rows(&data_t.x, &rows_t);
    let pred_b_gb = gbdt_b.predict_rows(&data_t.x, &rows_t);
    let pred_s_gb = gbdt_s.predict_rows(&data_t.x, &rows_t);

    let y_mean_a = data_a.y.iter().sum::<f64>() / data_a.y.len() as f64;
    let sq_err = |pred: &[f64]| -> Vec<f64> {
        data_t
            .y
            .iter()
            .zip(pred)
            .map(|(&yt, &yp)| (yt - yp).powi(2))
            .collect()
    };
    let null_losses: Vec<f64> = data_t.y.iter().map(|&yt| (yt - y_mean_a).powi(2)).collect();

    let rows_real_t = cols_to_rows(&data_t.x, 120);
    let rows_syn_s = cols_to_rows(s_x, 120);
    let realism_in = build_native_realism_input(
        &rows_real_t,
        &rows_syn_s,
        Some(decoded.catf.rules.violation_rate(s_x)),
        Some(decoded.catf.rules.violation_rate(&data_t.x)),
        0,
        12,
    )
    .unwrap();
    let marg_debug = realism_in.marginal_aucs.clone();
    let privacy_in = EmpiricalPrivacyInput {
        real_fit_a: cols_to_rows(&data_a.x, 200),
        real_holdout_b: cols_to_rows(&data_b.x, 200),
        synthetic: cols_to_rows(s_x, 200),
        min_emitted_support: 10,
        canary_rows: vec![],
        max_distance_pairs: 400_000,
    };
    let driver_in = DriverEquivalenceInput {
        perm_a: perm_importance_linear(&lin_a, &data_t.x, &data_t.y),
        perm_b: Some(perm_importance_linear(&lin_b, &data_t.x, &data_t.y)),
        perm_s: perm_importance_linear(&lin_s, &data_t.x, &data_t.y),
        ale_a: None,
        ale_b: None,
        ale_s: None,
        coalition_a: None,
        coalition_b: None,
        coalition_s: None,
    };

    let receipt = evaluate_v3(&FidelityV3Request {
        protocol: PROTOCOL_V3.into(),
        mode: "faithful".into(),
        seed: 7,
        bootstrap_draws: 300,
        kappa_thresholds: KappaThresholds::default(),
        allow_partial_diagnostic: false,
        cells: vec![
            FidelityCellRequest {
                cell_id: "e2e_lin".into(),
                family_id: "fam_e2e".into(),
                auditor_id: "ridge".into(),
                input: FidelityCellInput::Complete {
                    bundle: Box::new(PredictionBundle {
                        task: TaskKind::Regression,
                        null_losses: null_losses.clone(),
                        real_a_losses: sq_err(&pred_a_lin),
                        real_b_losses: Some(sq_err(&pred_b_lin)),
                        synth_losses: sq_err(&pred_s_lin),
                        donor_ids: None,
                        y_true: data_t.y.clone(),
                        pred_a: pred_a_lin,
                        pred_b: Some(pred_b_lin),
                        pred_s: pred_s_lin,
                        num_outputs: 1,
                    }),
                    driver: Some(Box::new(driver_in.clone())),
                    deploy_roster: None,
                    realism: Some(Box::new(realism_in.clone())),
                    privacy: Some(Box::new(privacy_in.clone())),
                },
            },
            FidelityCellRequest {
                cell_id: "e2e_gbdt".into(),
                family_id: "fam_e2e".into(),
                auditor_id: "gbdt".into(),
                input: FidelityCellInput::Complete {
                    bundle: Box::new(PredictionBundle {
                        task: TaskKind::Regression,
                        null_losses,
                        real_a_losses: sq_err(&pred_a_gb),
                        real_b_losses: Some(sq_err(&pred_b_gb)),
                        synth_losses: sq_err(&pred_s_gb),
                        donor_ids: None,
                        y_true: data_t.y.clone(),
                        pred_a: pred_a_gb,
                        pred_b: Some(pred_b_gb),
                        pred_s: pred_s_gb,
                        num_outputs: 1,
                    }),
                    driver: Some(Box::new(driver_in)),
                    deploy_roster: None,
                    realism: Some(Box::new(realism_in)),
                    privacy: Some(Box::new(privacy_in)),
                },
            },
        ],
    })
    .unwrap();

    assert_eq!(receipt.composite_score, "forbidden");
    assert_eq!(receipt.formal_dp_state, "not_claimed");
    assert_eq!(receipt.coverage.informative_cells, 2);
    assert!(
        receipt.coverage.median_unclipped_retention.unwrap() >= 0.85,
        "E2E decoded Format-2 unclipped retention too low: {:?}",
        receipt.coverage
    );
    assert!(
        receipt.coverage.privacy_pass_coverage >= 0.99,
        "E2E empirical privacy failed: {:?}",
        receipt.cells[0].privacy
    );
    assert!(
        receipt.cells[0].realism.as_ref().unwrap().max_marginal_auc <= 0.65,
        "E2E MargAUC too high: marg_aucs={:?}, order={:?}, rules={:?}, realism={:?}",
        marg_debug,
        decoded.catf.order,
        decoded.catf.rules.rules.len(),
        receipt.cells[0].realism
    );

    let emb = embed_packet(&fitted.bytes).unwrap();
    assert_eq!(emb.format_version, 2);
    assert_eq!(emb.features.len(), PACKET_FEATURES_V2_DIM);
    assert!(emb.features[4] > 0.0);
    let bridge = embed_to_629_bridge(&fitted.bytes, 11).unwrap();
    assert_eq!(bridge.synthetic_rows, 4096);
    assert_eq!(bridge.feature_count, 4);
}

//! Full end-to-end DOPE Fidelity Engine 2.0 pipeline integration test:
//! `NC2 -> BUILD/SELECT/CAL -> CATF + MarginalsV2 + Rules -> K=5 Teacher -> DistilledStudent`
//! `-> P+ Damped-Newton Projection -> Format 2 Binary Packet -> dope embed -> dope-fidelity/v3`.

use dope_core::artifact::format2::{
    decode_format2, encode_format2, quantize_f64_i16, Format2Packet, Format2Section,
    TierPrefixEntry, CODEC_Q16, CODEC_RAW, TAG_CATF, TAG_MARGINALS_V2, TAG_RULES,
    TAG_STUDENT_FOREST, TAG_UNIT_SCHEMA,
};
use dope_core::audit::{ridge_certificate, LearningSignatureV1};
use dope_core::catf::{CatfConfig, CatfModel};
use dope_core::compile::v2::{project_moments_damped_newton, CompilerRoles, MIN_PROJECTION_ESS};
use dope_core::distil::{DistilConfig, DistilledStudent};
use dope_core::dvs::v3::bundle::{PredictionBundle, TaskKind};
use dope_core::dvs::v3::equiv::KappaThresholds;
use dope_core::dvs::v3::realism::{column_marginal_auc, RealismInput};
use dope_core::dvs::v3::{
    evaluate_v3, FidelityCellInput, FidelityCellRequest, FidelityV3Request, PROTOCOL_V3,
};
use dope_core::embed::{embed_packet, PACKET_FEATURES_V2_DIM};
use dope_core::rng::Rng;
use dope_core::structure::RuleMode;
use dope_core::target::Head;
use dope_core::teacher::{CrossFittedTeacher, TeacherConfig};

fn slice_cols(cols: &[Vec<f64>], rows: &[usize]) -> Vec<Vec<f64>> {
    cols.iter()
        .map(|c| rows.iter().map(|&r| c[r]).collect())
        .collect()
}

fn slice_vec(v: &[f64], rows: &[usize]) -> Vec<f64> {
    rows.iter().map(|&r| v[r]).collect()
}

#[test]
fn fidelity_engine_2_full_pipeline_end_to_end() {
    // Generate mixed structured dataset: heavy-skew col, decimal lattice col, ARITH sum col,
    // binary col, and nonlinear interaction regression target.
    let mut rng = Rng::new(2026);
    let n = 600usize;
    let mut x: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(n)).collect();
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let x0 = ((rng.uniform() * 100.0).round()) * 0.01; // exact 0.01 decimal lattice
        let x1 = (rng.normal() * 0.6).exp(); // heavy right-skewed
        let x2 = x0 + 0.5 * x1; // ARITH structural rule
        let x3 = rng.below(2) as f64; // discrete binary
        let target = 1.8 * x0 - 0.6 * x1.ln() + 0.9 * x3 * (x0 - 0.5) + 0.1 * rng.normal();
        x[0].push(x0);
        x[1].push(x1);
        x[2].push(x2);
        x[3].push(x3);
        y.push(target);
    }

    // 1. Rule A-1 disjoint BUILD / SELECT / CAL roles (and separate test slice).
    let roles = CompilerRoles::split(&x, &y, 42);
    let build_x = slice_cols(&x, &roles.build_rows);
    let build_y = slice_vec(&y, &roles.build_rows);
    let select_x = slice_cols(&x, &roles.select_rows);
    let cal_x = slice_cols(&x, &roles.cal_rows);
    let cal_y = slice_vec(&y, &roles.cal_rows);

    // 2. ColumnMarginalV2 + Zero-byte copula calibration + CATF + Structural Rules.
    let catf = CatfModel::fit(
        &build_x,
        &select_x,
        &[false, false, false, true],
        &CatfConfig {
            max_trees_per_col: 8,
            max_depth: 3,
            rule_mode: RuleMode::Structural,
            ..Default::default()
        },
    );

    // 3. Whole-Stack K=5 Cross-Fitted Teacher + Distilled Student on BUILD + 4n Q_X probes.
    let teacher = CrossFittedTeacher::fit(
        Head::Regression,
        &build_x,
        &build_y,
        &TeacherConfig {
            folds: 5,
            max_trees_d4: 40,
            max_trees_d6: 40,
            seed: 42,
        },
    );
    let probes_x = catf.sample_x(4 * build_y.len(), 42, 1);
    let student = DistilledStudent::fit(
        &teacher,
        &build_x,
        &probes_x,
        &cal_x,
        &cal_y,
        &DistilConfig {
            max_pairs: 4,
            depth: 5,
            bits: 8,
            max_trees: 120,
            forest_byte_cap: Some(32_000),
            seed: 42,
            ..Default::default()
        },
    );

    // 4. Stateless post-rule synthetic emission (`faithful` 4n) + Owner-side P+ Projection.
    let synth = catf.sample_with_student(&student, 4 * build_y.len(), 99, 0);
    assert!(catf.rules.violation_rate(&synth.x) <= 0.02);

    let sig = LearningSignatureV1::fit(&build_x, &build_y);
    let proj = project_moments_damped_newton(&sig, &synth.x, &synth.y, None);
    assert!(proj.ladder.ess >= MIN_PROJECTION_ESS);
    assert!(proj.ladder.gap_reweighted <= proj.ladder.gap_raw_q + 1e-9);

    let rc = ridge_certificate(&build_x, &build_y, &synth.x, &synth.y, &cal_x, 0.1);
    assert!(
        rc.pred_rmse_diff < 0.30,
        "ridge cert pred diff too high: {}",
        rc.pred_rmse_diff
    );

    // 5. Pack into `.dope` Format 2 binary section container + CRC32 decode + `dope embed`.
    let q_main = quantize_f64_i16(&student.main_effects.intercept).unwrap();
    let forest_bytes = student.forest.to_bytes(4);
    let fmt2 = Format2Packet {
        flags: 1,
        tiers: vec![TierPrefixEntry {
            tier_budget_bytes: 65_536,
            section_prefix_count: 5,
            forest_trees_prefix: student.forest_trees as u32,
        }],
        sections: vec![
            Format2Section {
                tag: TAG_UNIT_SCHEMA,
                codec: CODEC_RAW,
                data: vec![2, 4, 0, 1],
            },
            Format2Section {
                tag: TAG_MARGINALS_V2,
                codec: CODEC_Q16,
                data: q_main,
            },
            Format2Section {
                tag: TAG_CATF,
                codec: CODEC_RAW,
                data: vec![0u8; 16],
            },
            Format2Section {
                tag: TAG_RULES,
                codec: CODEC_RAW,
                data: vec![catf.rules.rules.len() as u8],
            },
            Format2Section {
                tag: TAG_STUDENT_FOREST,
                codec: CODEC_RAW,
                data: forest_bytes,
            },
        ],
    };
    let wire = encode_format2(&fmt2).unwrap();
    assert!(
        wire.len() <= 65_536,
        "Format 2 packet {} exceeded 64 KiB cap",
        wire.len()
    );
    let decoded = decode_format2(&wire).unwrap();
    assert_eq!(decoded, fmt2);

    let emb = embed_packet(&wire).unwrap();
    assert_eq!(emb.format_version, 2);
    assert_eq!(emb.features.len(), PACKET_FEATURES_V2_DIM);

    // 6. Evaluate hard-gate `dope-fidelity/v3` coverage on held-out `cal` slice.
    let (teacher_cal, _) = teacher.predict_probes(&cal_x);
    let student_cal = student.predict_expectations(&cal_x);
    let y_mean = build_y.iter().sum::<f64>() / build_y.len() as f64;
    let null_losses: Vec<f64> = cal_y.iter().map(|&v| (v - y_mean).powi(2)).collect();
    let real_losses: Vec<f64> = cal_y
        .iter()
        .zip(&teacher_cal)
        .map(|(&a, &b)| (a - b).powi(2))
        .collect();
    let synth_losses: Vec<f64> = cal_y
        .iter()
        .zip(&student_cal)
        .map(|(&a, &b)| (a - b).powi(2))
        .collect();
    let marg_aucs: Vec<f64> = (0..4)
        .map(|j| column_marginal_auc(&build_x[j], &synth.x[j]).unwrap())
        .collect();

    let receipt = evaluate_v3(&FidelityV3Request {
        protocol: PROTOCOL_V3.into(),
        mode: "faithful".into(),
        seed: 7,
        bootstrap_draws: 300,
        kappa_thresholds: KappaThresholds::default(),
        allow_partial_diagnostic: true,
        cells: vec![FidelityCellRequest {
            cell_id: "e2e_v2".into(),
            family_id: "fam_e2e".into(),
            auditor_id: "student_vs_teacher".into(),
            input: FidelityCellInput::Complete {
                bundle: Box::new(PredictionBundle {
                    task: TaskKind::Regression,
                    null_losses,
                    real_a_losses: real_losses.clone(),
                    real_b_losses: Some(real_losses),
                    synth_losses,
                    donor_ids: None,
                    y_true: cal_y.clone(),
                    pred_a: teacher_cal.clone(),
                    pred_b: Some(teacher_cal),
                    pred_s: student_cal,
                    num_outputs: 1,
                }),
                driver: None,
                deploy_roster: None,
                realism: Some(Box::new(RealismInput {
                    c2st_labels: vec![0, 0, 0, 0, 1, 1, 1, 1],
                    c2st_scores: vec![0.47, 0.53, 0.49, 0.51, 0.48, 0.52, 0.50, 0.50],
                    marginal_aucs: marg_aucs.clone(),
                    synth_rule_violation_rate: Some(catf.rules.violation_rate(&synth.x)),
                    real_rule_violation_rate: Some(catf.rules.violation_rate(&build_x)),
                    illegal_rows: 0,
                })),
                privacy: None,
            },
        }],
    })
    .unwrap();
    assert_eq!(receipt.composite_score, "forbidden");
    assert_eq!(receipt.formal_dp_state, "not_claimed");
    assert!(
        receipt.coverage.median_unclipped_retention.unwrap() >= 0.85,
        "E2E retention {:?}",
        receipt.coverage.median_unclipped_retention
    );
    assert!(
        marg_aucs.iter().all(|&a| a <= 0.62),
        "E2E marginal AUCs too high: {marg_aucs:?}"
    );
}

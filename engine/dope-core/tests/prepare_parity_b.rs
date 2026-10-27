use std::path::PathBuf;

use dope_core::prepare::building_blocks::{
    compute_building_block_landmarks, BuildingBlockDescriptors,
};
use dope_core::prepare::conditioning::{
    assemble_observed_conditioning, assemble_observed_conditioning_cached,
    assemble_observed_conditioning_with_target_stats, assemble_v3_conditioning,
    build_conditioning_with_cached_table_stats, build_conditioning_with_descriptors,
    build_conditioning_with_full_table_stats, conditioning_digest, observed_report_descriptors,
    observed_target_stats, pad_col_landmarks, report_complete_landmarks, report_observed_landmarks,
    subsample_scaled_rows, table_classes, BundleConditioningInput, LegacyReportConditioningInput,
    FULL_REPORT_RECIPE, OBSERVED_RECIPE,
};
use dope_core::prepare::inter_landmark::{
    compute_inter_landmark_summary_stats, compute_observed_inter_landmark_stats, ReportLandmarkView,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct GoldenFile {
    schema: String,
    cases: Vec<GoldenCase>,
}

#[derive(Deserialize)]
struct GoldenCase {
    name: String,
    level: String,
    task: String,
    recipe: String,
    n: usize,
    p: usize,
    x: Vec<f64>,
    y: Vec<f64>,
    raw_target_stats2: [f64; 2],
    bb: GoldenBb,
    rep: GoldenRep,
    tab: GoldenTab,
    inter48: Vec<f32>,
    vec256: Vec<f32>,
    col_landmarks24: Vec<f32>,
    col_landmarks17x64: Vec<f32>,
    v3_driver_weights: Vec<f32>,
    conditioning_digest: String,
    observed: Vec<GoldenObserved>,
}

#[derive(Deserialize)]
struct GoldenBb {
    bb_gains16: Vec<f32>,
    row_summary_corrs8: Vec<f32>,
    col_bb_probes7: Vec<f32>,
    het_and_tail8: Vec<f32>,
    multiclass_simplex20: Vec<f32>,
    building_block_vec68: Vec<f32>,
}

#[derive(Deserialize)]
struct GoldenRep {
    gains23: [f64; 23],
    mask23: [bool; 23],
    ses23: [f64; 23],
    lambdas23: [f64; 23],
    column_spectrum: Vec<f32>,
    column_prior_importance: Vec<f32>,
    vector116: Vec<f32>,
    complete_values23: [f32; 23],
    complete_mask23: [bool; 23],
    observed_values23: [f32; 23],
    observed_mask23: [bool; 23],
}

#[derive(Deserialize)]
struct GoldenTab {
    table_vec24: [f32; 24],
    column_desc: Vec<f32>,
    empirical_driver_weights: Vec<f32>,
}

#[derive(Deserialize)]
struct GoldenObserved {
    variant: String,
    mode: String,
    recipe: String,
    level: String,
    task: String,
    bundle_values23: [f32; 23],
    bundle_mask23: [bool; 23],
    rep_vector116: Vec<f32>,
    inter48: Vec<f32>,
    vec256: Vec<f32>,
    vec256_with_raw_target_stats: Vec<f32>,
    col_landmarks24: Vec<f32>,
    col_landmarks17x64: Vec<f32>,
    v3_driver_weights: Vec<f32>,
    conditioning_digest: String,
    conditioning_digest_with_raw_target_stats: String,
}

fn load_goldens() -> GoldenFile {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/prepare-v1/b/goldens.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let data: GoldenFile = serde_json::from_str(&raw).expect("valid goldens.json");
    assert_eq!(data.schema, "dope.prepare-goldens-b/v1");
    data
}

fn assert_close(label: &str, got: &[f32], want: &[f32], tol: f32) -> usize {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let mut bit_diff = 0usize;
    let mut max_diff = 0.0f32;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.is_finite() && w.is_finite(),
            "{label}[{i}]: non-finite value got={g} want={w}"
        );
        if g.to_bits() != w.to_bits() {
            bit_diff += 1;
        }
        let d = (g - w).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d <= tol,
            "{label}[{i}]: got={g:.9} want={w:.9} diff={d:.3e} > tol={tol:.1e}"
        );
    }
    bit_diff
}

fn unflatten<const K: usize>(flat: &[f32]) -> Vec<[f32; K]> {
    assert_eq!(flat.len() % K, 0);
    flat.chunks_exact(K)
        .map(|c| {
            let mut row = [0.0f32; K];
            row.copy_from_slice(c);
            row
        })
        .collect()
}

fn flatten<const K: usize>(rows: &[[f32; K]]) -> Vec<f32> {
    rows.iter().flat_map(|r| r.iter().copied()).collect()
}

#[test]
fn full_report_building_blocks_inter_and_conditioning_parity() {
    let goldens = load_goldens();
    let mut exact_digest_matches = 0usize;

    for c in &goldens.cases {
        let bb: BuildingBlockDescriptors =
            compute_building_block_landmarks(&c.x, c.n, c.p, &c.y, &c.level);
        assert_close(
            &format!("{}:bb_gains16", c.name),
            &bb.bb_gains16,
            &c.bb.bb_gains16,
            1e-6,
        );
        assert_close(
            &format!("{}:row_summary_corrs8", c.name),
            &bb.row_summary_corrs8,
            &c.bb.row_summary_corrs8,
            1e-6,
        );
        assert_close(
            &format!("{}:col_bb_probes7", c.name),
            &flatten(&bb.col_bb_probes7),
            &c.bb.col_bb_probes7,
            1e-6,
        );
        assert_close(
            &format!("{}:het_and_tail8", c.name),
            &bb.het_and_tail8,
            &c.bb.het_and_tail8,
            1e-6,
        );
        assert_close(
            &format!("{}:multiclass_simplex20", c.name),
            &bb.multiclass_simplex20,
            &c.bb.multiclass_simplex20,
            1e-6,
        );
        assert_close(
            &format!("{}:building_block_vec68", c.name),
            &bb.building_block_vec68,
            &c.bb.building_block_vec68,
            1e-6,
        );

        let col_spec: Vec<[f32; 9]> = unflatten(&c.rep.column_spectrum);
        let rep_view = ReportLandmarkView {
            gains23: &c.rep.gains23,
            mask23: &c.rep.mask23,
            ses23: &c.rep.ses23,
            lambdas23: &c.rep.lambdas23,
            column_spectrum: &col_spec,
        };
        let inter48 = compute_inter_landmark_summary_stats(&rep_view, &bb, &c.level);
        assert_close(&format!("{}:inter48", c.name), &inter48, &c.inter48, 1e-6);

        let col_desc: Vec<[f32; 8]> = unflatten(&c.tab.column_desc);
        let rep_vec116: [f32; 116] = c.rep.vector116.as_slice().try_into().unwrap();
        let v3 = assemble_v3_conditioning(
            &rep_vec116,
            &c.tab.table_vec24,
            &col_desc,
            &c.tab.empirical_driver_weights,
            &rep_view,
            &c.rep.column_prior_importance,
            &bb,
            &c.level,
        );
        let v_bit_diff = assert_close(&format!("{}:vec256", c.name), &v3.vec256, &c.vec256, 1e-6);
        assert_close(
            &format!("{}:col_landmarks24", c.name),
            &flatten(&v3.col_landmarks24),
            &c.col_landmarks24,
            1e-6,
        );
        assert_close(
            &format!("{}:v3_driver_weights", c.name),
            &v3.v3_driver_weights,
            &c.v3_driver_weights,
            1e-6,
        );
        let cols17x64 = pad_col_landmarks(&v3.col_landmarks24);
        let c_bit_diff = assert_close(
            &format!("{}:col_landmarks17x64", c.name),
            &flatten(&cols17x64),
            &c.col_landmarks17x64,
            1e-6,
        );

        let raw_in = LegacyReportConditioningInput {
            vector116: &rep_vec116,
            rep: rep_view,
            column_prior_importance: &c.rep.column_prior_importance,
            complete_values23: c.rep.complete_values23,
            complete_mask23: c.rep.complete_mask23,
            observed_values23: c.rep.observed_values23,
            observed_mask23: c.rep.observed_mask23,
            full_table_target_stats: None,
            evidence_sample: None,
        };

        // Both complete-only and observed (complete | uninformative) bundles must be accepted in Full mode.
        for (b_vals, b_mask) in [
            (&c.rep.complete_values23, &c.rep.complete_mask23),
            (&c.rep.observed_values23, &c.rep.observed_mask23),
        ] {
            let cond = build_conditioning_with_descriptors(
                &c.x,
                c.n,
                c.p,
                &c.y,
                &c.task,
                &c.tab.table_vec24,
                &col_desc,
                &c.tab.empirical_driver_weights,
                Some(BundleConditioningInput {
                    values: b_vals,
                    mask: b_mask,
                    subsample_n: None,
                }),
                Some(raw_in),
                "full",
                &c.level,
            )
            .expect("full conditioning succeeds");
            assert_eq!(cond.recipe, FULL_REPORT_RECIPE);
            assert_eq!(cond.vec256, v3.vec256);
            assert_eq!(cond.col_landmarks, cols17x64);
            if v_bit_diff == 0 && c_bit_diff == 0 {
                assert_eq!(
                    cond.digest(),
                    c.conditioning_digest,
                    "{}: exact conditioning digest mismatch",
                    c.name
                );
                exact_digest_matches += 1;
            }
        }

        // Also verify conditioning_digest directly on the golden vec256 + col_landmarks17x64 is bit-identical.
        let golden_vec256: [f32; 256] = c.vec256.clone().try_into().unwrap();
        let golden_cols17x64: Vec<[f32; 17]> = unflatten(&c.col_landmarks17x64);
        assert_eq!(
            conditioning_digest(&c.recipe, &c.level, &golden_vec256, &golden_cols17x64),
            c.conditioning_digest,
            "{}: digest formula mismatch on golden tensors",
            c.name
        );
    }
    assert!(
        exact_digest_matches > 0,
        "expected at least one bit-exact legacy full case"
    );
}

#[test]
fn observed_only_none_partial_and_full_as_observed_parity() {
    let goldens = load_goldens();

    for c in &goldens.cases {
        let col_desc: Vec<[f32; 8]> = unflatten(&c.tab.column_desc);

        for obs in &c.observed {
            let classes = table_classes(&c.y, &obs.task).expect("valid task");
            let t_stats = observed_target_stats(&c.y, &obs.task).expect("valid task");
            let rep_desc = observed_report_descriptors(
                &obs.bundle_values23,
                &obs.bundle_mask23,
                c.n,
                c.p,
                classes,
                &obs.task,
                &t_stats,
                &obs.level,
            )
            .expect("valid observed report descriptors");
            assert_close(
                &format!("{}:{}:rep_vector116", c.name, obs.variant),
                &rep_desc.vector116,
                &obs.rep_vector116,
                1e-6,
            );

            let inter48 = compute_observed_inter_landmark_stats(
                &rep_desc.gains23,
                &rep_desc.mask23,
                &obs.level,
            );
            assert_close(
                &format!("{}:{}:inter48", c.name, obs.variant),
                &inter48,
                &obs.inter48,
                1e-6,
            );

            let v3 = assemble_observed_conditioning(
                &obs.bundle_values23,
                &obs.bundle_mask23,
                c.n,
                c.p,
                &c.y,
                &obs.task,
                &c.tab.table_vec24,
                &col_desc,
                &c.tab.empirical_driver_weights,
                &obs.level,
            )
            .expect("assemble observed conditioning");
            let v_bit_diff = assert_close(
                &format!("{}:{}:vec256", c.name, obs.variant),
                &v3.vec256,
                &obs.vec256,
                1e-6,
            );
            assert_close(
                &format!("{}:{}:col_landmarks24", c.name, obs.variant),
                &flatten(&v3.col_landmarks24),
                &obs.col_landmarks24,
                1e-6,
            );
            assert_close(
                &format!("{}:{}:v3_driver_weights", c.name, obs.variant),
                &v3.v3_driver_weights,
                &obs.v3_driver_weights,
                1e-6,
            );

            let cond = build_conditioning_with_descriptors(
                &c.x,
                c.n,
                c.p,
                &c.y,
                &obs.task,
                &c.tab.table_vec24,
                &col_desc,
                &c.tab.empirical_driver_weights,
                Some(BundleConditioningInput {
                    values: &obs.bundle_values23,
                    mask: &obs.bundle_mask23,
                    subsample_n: Some(300),
                }),
                None,
                &obs.mode,
                &obs.level,
            )
            .expect("observed conditioning succeeds");
            assert_eq!(cond.recipe, OBSERVED_RECIPE);
            assert_eq!(cond.recipe, obs.recipe);
            let c_bit_diff = assert_close(
                &format!("{}:{}:col_landmarks17x64", c.name, obs.variant),
                &flatten(&cond.col_landmarks),
                &obs.col_landmarks17x64,
                1e-6,
            );
            if v_bit_diff == 0 && c_bit_diff == 0 {
                assert_eq!(
                    cond.digest(),
                    obs.conditioning_digest,
                    "{}:{}: exact observed conditioning digest mismatch",
                    c.name,
                    obs.variant
                );
            }

            // Now verify with full-table raw_target_stats (PreparedTable.target_stats).
            let cond_raw_ts = build_conditioning_with_full_table_stats(
                &c.x,
                c.n,
                c.p,
                &c.y,
                &obs.task,
                Some(c.raw_target_stats2),
                &c.tab.table_vec24,
                &col_desc,
                &c.tab.empirical_driver_weights,
                Some(BundleConditioningInput {
                    values: &obs.bundle_values23,
                    mask: &obs.bundle_mask23,
                    subsample_n: Some(300),
                }),
                None,
                &obs.mode,
                &obs.level,
            )
            .expect("observed conditioning with raw target stats succeeds");
            let v_raw_bit_diff = assert_close(
                &format!("{}:{}:vec256_with_raw_target_stats", c.name, obs.variant),
                &cond_raw_ts.vec256,
                &obs.vec256_with_raw_target_stats,
                1e-6,
            );
            if v_raw_bit_diff == 0 && c_bit_diff == 0 {
                assert_eq!(
                    cond_raw_ts.digest(),
                    obs.conditioning_digest_with_raw_target_stats,
                    "{}:{}: exact raw-target-stats conditioning digest mismatch",
                    c.name,
                    obs.variant
                );
            }

            // Skewed regression cases must retain non-zero target_stats[85..87] when raw_target_stats are supplied.
            if c.task == "regression" && c.name != "const_y_high" {
                assert!(
                    cond_raw_ts.vec256[85].abs() > 1e-4 || cond_raw_ts.vec256[86].abs() > 1e-4,
                    "{}:{}: expected non-zero regression target_stats[85:87], got {:?}",
                    c.name,
                    obs.variant,
                    &cond_raw_ts.vec256[85..87]
                );
            }

            // Zero bundle (`none` / `fast`): every landmark-derived coordinate must be 0.0.
            if obs.variant == "none" {
                assert!(
                    cond_raw_ts.vec256[..85].iter().all(|&v| v == 0.0),
                    "{}: none mode vec256[..85] not all zero",
                    c.name
                );
                assert!(
                    cond_raw_ts.vec256[91..116].iter().all(|&v| v == 0.0),
                    "{}: none mode vec256[91..116] not all zero",
                    c.name
                );
                assert!(
                    cond_raw_ts.vec256[140..256].iter().all(|&v| v == 0.0),
                    "{}: none mode vec256[140..256] (inter48 + bb68) not all zero",
                    c.name
                );
                for row in &cond_raw_ts.col_landmarks24 {
                    assert!(row[..9].iter().all(|&v| v == 0.0));
                    assert!(row[17..24].iter().all(|&v| v == 0.0));
                }

                // `fast` is an exact alias of `none`.
                let cond_fast = build_conditioning_with_full_table_stats(
                    &c.x,
                    c.n,
                    c.p,
                    &c.y,
                    &obs.task,
                    Some(c.raw_target_stats2),
                    &c.tab.table_vec24,
                    &col_desc,
                    &c.tab.empirical_driver_weights,
                    None,
                    None,
                    "fast",
                    &obs.level,
                )
                .expect("fast conditioning succeeds");
                assert_eq!(cond_fast.vec256, cond_raw_ts.vec256);
                assert_eq!(cond_fast.col_landmarks, cond_raw_ts.col_landmarks);
                assert_eq!(cond_fast.digest(), cond_raw_ts.digest());
            }
        }
    }
}

#[test]
fn base_target_and_shape_invariant_across_none_partial_and_sampled_full() {
    let goldens = load_goldens();

    for c in &goldens.cases {
        let col_desc: Vec<[f32; 8]> = unflatten(&c.tab.column_desc);
        let col_spec: Vec<[f32; 9]> = unflatten(&c.rep.column_spectrum);
        let rep_view = ReportLandmarkView {
            gains23: &c.rep.gains23,
            mask23: &c.rep.mask23,
            ses23: &c.rep.ses23,
            lambdas23: &c.rep.lambdas23,
            column_spectrum: &col_spec,
        };

        // Simulate a subsampled report whose internal vector116[85..91] had different subsample N stats:
        let mut subsampled_rep_vec116: [f32; 116] = c.rep.vector116.as_slice().try_into().unwrap();
        subsampled_rep_vec116[85] = -0.777;
        subsampled_rep_vec116[86] = 0.888;
        subsampled_rep_vec116[87] = 0.0;
        subsampled_rep_vec116[88] = 0.123;
        subsampled_rep_vec116[89] = 0.456;
        subsampled_rep_vec116[90] = -0.321;

        let raw_in = LegacyReportConditioningInput {
            vector116: &subsampled_rep_vec116,
            rep: rep_view,
            column_prior_importance: &c.rep.column_prior_importance,
            complete_values23: c.rep.complete_values23,
            complete_mask23: c.rep.complete_mask23,
            observed_values23: c.rep.observed_values23,
            observed_mask23: c.rep.observed_mask23,
            full_table_target_stats: None,
            evidence_sample: None,
        };

        let cond_none = build_conditioning_with_full_table_stats(
            &c.x,
            c.n,
            c.p,
            &c.y,
            &c.task,
            Some(c.raw_target_stats2),
            &c.tab.table_vec24,
            &col_desc,
            &c.tab.empirical_driver_weights,
            None,
            None,
            "none",
            &c.level,
        )
        .unwrap();

        let obs_partial = &c.observed[1]; // partial_macro
        let cond_partial = build_conditioning_with_full_table_stats(
            &c.x,
            c.n,
            c.p,
            &c.y,
            &c.task,
            Some(c.raw_target_stats2),
            &c.tab.table_vec24,
            &col_desc,
            &c.tab.empirical_driver_weights,
            Some(BundleConditioningInput {
                values: &obs_partial.bundle_values23,
                mask: &obs_partial.bundle_mask23,
                subsample_n: Some(64),
            }),
            None,
            "progressive",
            &c.level,
        )
        .unwrap();

        for sub_n in [32usize, 64, 300] {
            let take_n = sub_n.min(c.n);
            let positions: Vec<usize> = (0..take_n).collect();
            let (sx, s_rows, sy) =
                subsample_scaled_rows(&c.x, c.n, c.p, &c.y, &positions).expect("subsample rows");
            let mut raw_sampled = raw_in;
            raw_sampled.evidence_sample = Some((&sx, s_rows, &sy));

            let cond_sampled_full = build_conditioning_with_full_table_stats(
                &c.x,
                c.n,
                c.p,
                &c.y,
                &c.task,
                Some(c.raw_target_stats2),
                &c.tab.table_vec24,
                &col_desc,
                &c.tab.empirical_driver_weights,
                Some(BundleConditioningInput {
                    values: &c.rep.observed_values23,
                    mask: &c.rep.observed_mask23,
                    subsample_n: Some(sub_n),
                }),
                Some(raw_sampled),
                "full",
                &c.level,
            )
            .unwrap();

            assert!(cond_sampled_full.vec256.iter().all(|v| v.is_finite()));
            assert!(cond_sampled_full
                .col_landmarks
                .iter()
                .all(|r| r.iter().all(|v| v.is_finite())));
            assert_eq!(
                &cond_sampled_full.vec256[85..91],
                &cond_none.vec256[85..91],
                "{}: sampled full (N={sub_n}) vec256[85..91] must match none mode",
                c.name
            );
            assert_eq!(
                &cond_sampled_full.vec256[85..91],
                &cond_partial.vec256[85..91],
                "{}: sampled full (N={sub_n}) vec256[85..91] must match partial progressive mode",
                c.name
            );
            // Table descriptors [116..140] stay on the complete training table:
            assert_eq!(&cond_sampled_full.vec256[116..140], &c.tab.table_vec24);
            // Evidence fields outside [85..91] remain lossless from the native report:
            assert_eq!(
                &cond_sampled_full.vec256[..85],
                &subsampled_rep_vec116[..85]
            );
            assert_eq!(
                &cond_sampled_full.vec256[91..116],
                &subsampled_rep_vec116[91..116]
            );
        }
    }
}

#[test]
fn mode_validation_and_recipe_selection_rules() {
    let goldens = load_goldens();
    let c = &goldens.cases[0];
    let col_desc: Vec<[f32; 8]> = unflatten(&c.tab.column_desc);
    let col_spec: Vec<[f32; 9]> = unflatten(&c.rep.column_spectrum);
    let rep_view = ReportLandmarkView {
        gains23: &c.rep.gains23,
        mask23: &c.rep.mask23,
        ses23: &c.rep.ses23,
        lambdas23: &c.rep.lambdas23,
        column_spectrum: &col_spec,
    };
    let rep_vec116: [f32; 116] = c.rep.vector116.as_slice().try_into().unwrap();
    let raw_in = LegacyReportConditioningInput {
        vector116: &rep_vec116,
        rep: rep_view,
        column_prior_importance: &c.rep.column_prior_importance,
        complete_values23: c.rep.complete_values23,
        complete_mask23: c.rep.complete_mask23,
        observed_values23: c.rep.observed_values23,
        observed_mask23: c.rep.observed_mask23,
        full_table_target_stats: None,
        evidence_sample: None,
    };

    // 1. Full mode without raw_report -> error `landmark_full_requires_native_report`
    let err = build_conditioning_with_descriptors(
        &c.x,
        c.n,
        c.p,
        &c.y,
        &c.task,
        &c.tab.table_vec24,
        &col_desc,
        &c.tab.empirical_driver_weights,
        None,
        None,
        "full",
        &c.level,
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("landmark_full_requires_native_report"));

    // 2. Full mode with mismatched bundle -> error `landmark_bundle_report_mismatch`
    let mut bad_vals = c.rep.complete_values23;
    bad_vals[0] += 0.1;
    let err = build_conditioning_with_descriptors(
        &c.x,
        c.n,
        c.p,
        &c.y,
        &c.task,
        &c.tab.table_vec24,
        &col_desc,
        &c.tab.empirical_driver_weights,
        Some(BundleConditioningInput {
            values: &bad_vals,
            mask: &c.rep.complete_mask23,
            subsample_n: None,
        }),
        Some(raw_in),
        "full",
        &c.level,
    )
    .unwrap_err();
    assert!(err.to_string().contains("landmark_bundle_report_mismatch"));

    // 3. None/Fast mode with non-empty bundle -> error `landmark_mode_none_rejects_bundle`
    for mode in ["none", "fast"] {
        let err = build_conditioning_with_descriptors(
            &c.x,
            c.n,
            c.p,
            &c.y,
            &c.task,
            &c.tab.table_vec24,
            &col_desc,
            &c.tab.empirical_driver_weights,
            Some(BundleConditioningInput {
                values: &c.rep.complete_values23,
                mask: &c.rep.complete_mask23,
                subsample_n: None,
            }),
            None,
            mode,
            &c.level,
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("landmark_mode_none_rejects_bundle"));
    }

    // 4. Uninformative cells (gain = 0.0) are preserved as present in report_observed_landmarks
    //    and absent in report_complete_landmarks.
    let cells = [
        (1i64, "complete", Some(0.25)),
        (2i64, "uninformative", Some(0.0)),
        (3i64, "over_budget", None),
    ];
    let (comp_v, comp_m) = report_complete_landmarks(cells);
    let (obs_v, obs_m) = report_observed_landmarks(cells);
    assert!(comp_m[0] && !comp_m[1] && !comp_m[2]);
    assert!(obs_m[0] && obs_m[1] && !obs_m[2]);
    assert_eq!(comp_v[1], 0.0);
    assert_eq!(obs_v[1].to_bits(), 0.0f32.to_bits());

    // 5. Progressive mode with all 23 mask bits true still uses observed-only-v1 (never selected by mask.all()).
    let full_obs = &c.observed[3];
    assert!(full_obs.bundle_mask23.iter().all(|&m| m));
    let cond_prog = build_conditioning_with_descriptors(
        &c.x,
        c.n,
        c.p,
        &c.y,
        &c.task,
        &c.tab.table_vec24,
        &col_desc,
        &c.tab.empirical_driver_weights,
        Some(BundleConditioningInput {
            values: &full_obs.bundle_values23,
            mask: &full_obs.bundle_mask23,
            subsample_n: Some(300),
        }),
        Some(raw_in),
        "progressive",
        &c.level,
    )
    .unwrap();
    assert_eq!(cond_prog.recipe, OBSERVED_RECIPE);
    assert_eq!(cond_prog.building_block_vec68, [0.0f32; 68]);

    // Ensure assemble_observed_conditioning_with_target_stats is directly callable too.
    let v3_obs = assemble_observed_conditioning_with_target_stats(
        &full_obs.bundle_values23,
        &full_obs.bundle_mask23,
        c.n,
        c.p,
        &c.y,
        &c.task,
        Some(c.raw_target_stats2),
        &c.tab.table_vec24,
        &col_desc,
        &c.tab.empirical_driver_weights,
        &c.level,
    )
    .unwrap();
    assert_close(
        "direct_assemble_observed_with_target_stats",
        &v3_obs.vec256,
        &full_obs.vec256_with_raw_target_stats,
        1e-6,
    );

    // 6. Multiclass progressive updates with cached `classes_override` and `raw_target_stats`
    //    never scan `y_scaled` (even if `y_scaled` is empty `&[]`).
    let mc = goldens
        .cases
        .iter()
        .find(|case| case.task == "multiclass")
        .unwrap();
    let mc_col_desc: Vec<[f32; 8]> = unflatten(&mc.tab.column_desc);
    let mc_classes = table_classes(&mc.y, &mc.task).unwrap();
    assert!(mc_classes >= 3);
    let mc_obs = &mc.observed[1];
    let v3_mc_cached = assemble_observed_conditioning_cached(
        &mc_obs.bundle_values23,
        &mc_obs.bundle_mask23,
        mc.n,
        mc.p,
        &[],
        &mc.task,
        Some(mc_classes),
        Some(mc.raw_target_stats2),
        &mc.tab.table_vec24,
        &mc_col_desc,
        &mc.tab.empirical_driver_weights,
        &mc.level,
    )
    .unwrap();
    assert_close(
        "mc_cached_assemble_observed",
        &v3_mc_cached.vec256,
        &mc_obs.vec256_with_raw_target_stats,
        1e-6,
    );
    let cond_mc_cached = build_conditioning_with_cached_table_stats(
        &[],
        mc.n,
        mc.p,
        &[],
        &mc.task,
        Some(mc_classes),
        Some(mc.raw_target_stats2),
        &mc.tab.table_vec24,
        &mc_col_desc,
        &mc.tab.empirical_driver_weights,
        Some(BundleConditioningInput {
            values: &mc_obs.bundle_values23,
            mask: &mc_obs.bundle_mask23,
            subsample_n: Some(128),
        }),
        None,
        "progressive",
        &mc.level,
    )
    .unwrap();
    assert_eq!(
        cond_mc_cached.digest(),
        mc_obs.conditioning_digest_with_raw_target_stats
    );
}

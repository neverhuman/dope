//! Native fixture proof for explicit High-only execution. No quality/speed claim.
use dope_core::{
    landmark::{
        bundle::{FINE_SLOTS, LARGER_SLOTS, MACRO_SLOTS},
        signature, CellState, ExecutionPlan, LandmarkGroup, LandmarkMode, LandmarkOptions,
        ProgressiveLandmarks, SubsampleSpec,
    },
    prepare::{
        build_conditioning_at_level, prepare_table, Level, NativeReportView, PrepTask,
        PreparedTable,
    },
};

fn table() -> PreparedTable {
    let rows: Vec<Vec<f64>> = (0..40)
        .map(|i| {
            let x = ((i * 17) % 41) as f64 / 41.0;
            vec![x, (i as f64 * 0.7).sin()]
        })
        .collect();
    let y: Vec<f64> = rows
        .iter()
        .enumerate()
        .map(|(i, x)| 2.0 * x[0] + 0.3 * x[1] + 0.01 * (i as f64).cos())
        .collect();
    prepare_table(&rows, &y, PrepTask::Regression).unwrap()
}
fn options() -> LandmarkOptions {
    LandmarkOptions {
        folds: 2,
        ridge_rows: 40,
        small_rows: 40,
        knn_reference: 8,
        spectrum_features: 2,
        seed: 17,
        work_budget: u64::MAX,
        ..Default::default()
    }
}
fn high_equal(table: &PreparedTable, full: &ProgressiveLandmarks, high: &ProgressiveLandmarks) {
    let full_view = NativeReportView::from_signature(full.report());
    let high_view = NativeReportView::from_signature(high.report());
    let a = build_conditioning_at_level(
        table,
        full.bundle(),
        Some(&full_view),
        LandmarkMode::Full,
        Level::High,
        None,
    )
    .unwrap();
    let b = build_conditioning_at_level(
        table,
        high.bundle(),
        Some(&high_view),
        LandmarkMode::Full,
        Level::High,
        None,
    )
    .unwrap();
    assert_eq!(a.vec256.map(f32::to_bits), b.vec256.map(f32::to_bits));
    let a_cols: Vec<_> = a
        .col_landmarks
        .iter()
        .map(|r| r.map(f32::to_bits))
        .collect();
    let b_cols: Vec<_> = b
        .col_landmarks
        .iter()
        .map(|r| r.map(f32::to_bits))
        .collect();
    assert_eq!(a_cols, b_cols);
    assert_eq!(a.digest(), b.digest());
    assert_eq!(
        serde_json::to_vec(&full.report().spectrum).unwrap(),
        serde_json::to_vec(&high.report().spectrum).unwrap(),
    );
}
#[test]
fn constants_have_one_definition_and_legacy_aliases_keep_coordinates() {
    let mut counts = [0; 23];
    for slot in MACRO_SLOTS
        .into_iter()
        .chain(LARGER_SLOTS)
        .chain(FINE_SLOTS)
    {
        counts[slot] += 1;
    }
    assert_eq!(counts, [1; 23]);
    assert_eq!(dope_core::prepare::conditioning::MACRO_INDICES, MACRO_SLOTS);
    assert_eq!(dope_core::prepare::descriptors::MACRO_INDICES, MACRO_SLOTS);
    assert_eq!(
        dope_core::prepare::conditioning::LARGER_ADDED_INDICES,
        LARGER_SLOTS
    );
    assert_eq!(
        dope_core::prepare::descriptors::FINE_ADDED_INDICES,
        FINE_SLOTS
    );
}
#[test]
fn default_full_and_explicit_full_reproduce_legacy_private_report() {
    let table = table();
    let frame = table.landmark_frame().unwrap();
    let opts = options();
    let expected = serde_json::to_vec(&signature(&frame, &opts).unwrap().private_report()).unwrap();
    let mut legacy = ProgressiveLandmarks::new(&frame, &opts, None).unwrap();
    let mut explicit =
        ProgressiveLandmarks::new_with_plan(&frame, &opts, None, ExecutionPlan::FullPanelV1)
            .unwrap();
    for group in [
        LandmarkGroup::Fine,
        LandmarkGroup::Macro,
        LandmarkGroup::Larger,
    ] {
        legacy.next_group(group).unwrap();
    }
    explicit.run_required().unwrap();
    assert_eq!(legacy.execution_plan(), ExecutionPlan::FullPanelV1);
    assert_eq!(legacy.scope(), explicit.scope());
    assert!(legacy.is_complete() && legacy.is_execution_complete());
    assert_eq!(
        serde_json::to_vec(&legacy.report().private_report()).unwrap(),
        expected
    );
    assert_eq!(
        serde_json::to_vec(&explicit.report().private_report()).unwrap(),
        expected
    );
    assert_eq!(legacy.bundle(), explicit.bundle());
}
#[test]
fn high_omits_fine_marks_readiness_and_preserves_conditioning() {
    let table = table();
    let frame = table.landmark_frame().unwrap();
    let opts = options();
    let mut full = ProgressiveLandmarks::new(&frame, &opts, None).unwrap();
    full.run_required().unwrap();
    let mut high =
        ProgressiveLandmarks::new_with_plan(&frame, &opts, None, ExecutionPlan::HighRequiredV1)
            .unwrap();
    assert!(!high.is_execution_complete());
    let zero = serde_json::to_vec(&high.report().private_report()).unwrap();
    high.next_group(LandmarkGroup::Fine).unwrap();
    assert_eq!(
        serde_json::to_vec(&high.report().private_report()).unwrap(),
        zero
    );
    high.next_group(LandmarkGroup::Larger).unwrap();
    assert!(!high.is_execution_complete());
    high.next_group(LandmarkGroup::Macro).unwrap();
    assert!(high.is_execution_complete());
    assert!(
        !high.is_complete(),
        "High readiness is not full-panel measurement"
    );
    for slot in FINE_SLOTS {
        assert_eq!(high.report().cells[slot].state, CellState::NotRequested);
        assert!(high.report().cells[slot].gain.is_none());
        assert!(!high.bundle().mask[slot]);
    }
    high_equal(&table, &full, &high);
    let accounting = high.execution_work();
    assert_eq!(
        accounting.reserved_panel_units,
        full.execution_work().reserved_panel_units
    );
    assert!(accounting.computed_cell_units < accounting.reserved_panel_units);
    assert_eq!(accounting.computed_cells, 18);
    assert_eq!(
        high.report().work.estimated_units,
        accounting.computed_cell_units + accounting.spectrum_units,
    );
    assert_ne!(high.execution_plan().id(), full.execution_plan().id());
    let before = serde_json::to_vec(&high.report().private_report()).unwrap();
    high.run_ids(&(1..=23).collect::<Vec<_>>()).unwrap();
    high.run_required().unwrap();
    assert_eq!(
        serde_json::to_vec(&high.report().private_report()).unwrap(),
        before
    );
    // Explicit upgrade reuses the exact reservation, folds and admitted spectrum.
    high.upgrade_to_full_panel();
    assert!(!high.is_execution_complete());
    high.next_group(LandmarkGroup::Fine).unwrap();
    assert!(high.is_complete() && high.is_execution_complete());
    assert_eq!(
        serde_json::to_vec(&high.report().private_report()).unwrap(),
        serde_json::to_vec(&full.report().private_report()).unwrap(),
    );
    assert_eq!(high.bundle(), full.bundle());
}
#[test]
fn tight_spectrum_boundaries_reserve_unused_full_panel_cost() {
    let table = table();
    let frame = table.landmark_frame().unwrap();
    let generous = options();
    let mut full = ProgressiveLandmarks::new(&frame, &generous, None).unwrap();
    full.run_required().unwrap();
    let reserve = full.execution_work().reserved_panel_units;
    let spectral = full.execution_work();
    assert_eq!(
        spectral.spectrum_cells, 2,
        "fixture must exercise spectrum admission"
    );
    let price = spectral.spectrum_units / spectral.spectrum_cells as u64;
    assert!(price > 0);
    for (extra, expected_spectrum) in [(price - 1, 0), (price, 1), (2 * price, 2)] {
        let opts = LandmarkOptions {
            work_budget: reserve + extra,
            ..generous.clone()
        };
        let mut full = ProgressiveLandmarks::new(&frame, &opts, None).unwrap();
        let mut high =
            ProgressiveLandmarks::new_with_plan(&frame, &opts, None, ExecutionPlan::HighRequiredV1)
                .unwrap();
        full.run_required().unwrap();
        high.run_required().unwrap();
        assert_eq!(full.report().work.spectrum_run, expected_spectrum);
        assert_eq!(high.report().work.spectrum_run, expected_spectrum);
        assert_eq!(
            high.execution_work().spectrum_units,
            full.execution_work().spectrum_units
        );
        high_equal(&table, &full, &high);
    }
}
#[test]
fn budget_dropped_cells_and_subsampling_keep_baseline_admission() {
    let table = table();
    let frame = table.landmark_frame().unwrap();
    let opts = LandmarkOptions {
        work_budget: 500_000,
        ..options()
    };
    let sample = Some(SubsampleSpec::new(32, 7));
    let mut full = ProgressiveLandmarks::new(&frame, &opts, sample).unwrap();
    let mut high =
        ProgressiveLandmarks::new_with_plan(&frame, &opts, sample, ExecutionPlan::HighRequiredV1)
            .unwrap();
    full.run_required().unwrap();
    high.run_required().unwrap();
    assert_eq!(
        full.selection().unwrap().row_ids,
        high.selection().unwrap().row_ids
    );
    assert_eq!(full.report().work.dropped, high.report().work.dropped);
    assert_eq!(
        full.execution_work().spectrum_units,
        high.execution_work().spectrum_units
    );
    for i in 0..23 {
        if !FINE_SLOTS.contains(&i) {
            assert_eq!(
                serde_json::to_value(&full.report().cells[i]).unwrap(),
                serde_json::to_value(&high.report().cells[i]).unwrap(),
            );
        }
    }
    assert!(high.is_execution_complete());
}
#[test]
fn invalid_ids_options_and_subsample_refuse_without_fabricating_progress() {
    let table = table();
    let frame = table.landmark_frame().unwrap();
    let opts = options();
    let mut high =
        ProgressiveLandmarks::new_with_plan(&frame, &opts, None, ExecutionPlan::HighRequiredV1)
            .unwrap();
    let before = serde_json::to_vec(&high.report().private_report()).unwrap();
    for ids in [vec![], vec![0], vec![24], vec![1, 1]] {
        assert!(high.run_ids(&ids).is_err());
    }
    assert_eq!(
        serde_json::to_vec(&high.report().private_report()).unwrap(),
        before
    );
    let bad = LandmarkOptions {
        folds: 1,
        ..opts.clone()
    };
    assert!(
        ProgressiveLandmarks::new_with_plan(&frame, &bad, None, ExecutionPlan::HighRequiredV1)
            .is_err()
    );
    assert!(ProgressiveLandmarks::new_with_plan(
        &frame,
        &opts,
        Some(SubsampleSpec::new(10, 0)),
        ExecutionPlan::HighRequiredV1,
    )
    .is_err());
}

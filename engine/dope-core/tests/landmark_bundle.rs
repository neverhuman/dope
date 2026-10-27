//! LandmarkBundle v1 contract: lattice merge, serde, digests, strict-path
//! bit-identity, requested signatures and deterministic subsamples.
use candle_core::Device;
use dope_core::{
    landmark::{
        bundle::{
            options_digest, row_selection_digest, source_rows_digest, squash_gain, squash_raw_gain,
            FINE_SLOTS, LARGER_SLOTS, MACRO_SLOTS, REPORT_RECIPE_COMPLETE, REPORT_RECIPE_OBSERVED,
        },
        parse_landmark_ids, signature, signature_requested, stratified_subsample, BundleScope,
        CellState, LandmarkBundle, LandmarkFrame, LandmarkGroup, LandmarkMode, LandmarkOptions,
        LandmarkScope, LandmarkStatus, ProgressiveLandmarks, SubsampleSpec, LANDMARK_SLOTS,
        PROGRESSIVE_ORDER,
    },
    learned::{landmark_features, manifest::Schema},
    levels::digest,
    schema::Task,
    target::Head,
};
use std::path::PathBuf;

/// Shared cross-language golden digests (also posted in comms.md).
const ZERO_DIGEST: &str = "3b80a6af433fd746af93fd3c65fe0cfd85f5b887a1a9acecdb3d2748c6658ed5";
const PARTIAL_DIGEST: &str = "4eaedd68a5239fdf3542e93cec1d7bffa4a7143e7c23f53be5a3c8716a53806b";

fn scope(digest: &str) -> BundleScope {
    BundleScope::new(digest)
}
fn gains(pairs: &[(usize, f64)]) -> [Option<f64>; LANDMARK_SLOTS] {
    let mut g = [None; LANDMARK_SLOTS];
    for &(i, v) in pairs {
        g[i] = Some(v);
    }
    g
}
fn golden_partial() -> LandmarkBundle {
    let mut g = [None; LANDMARK_SLOTS];
    for i in MACRO_SLOTS {
        g[i] = Some(0.05 * (i + 1) as f64);
    }
    LandmarkBundle::from_raw_gains(&g, scope("golden").with_subsample(Some(300)))
}

#[test]
fn zero_bundle_is_empty_valid_and_has_the_golden_digest() {
    let z = LandmarkBundle::zero();
    assert!(z.is_zero());
    z.validate().unwrap();
    assert_eq!(z.status, LandmarkStatus::None);
    assert_eq!(z.features46(), [0f32; 46]);
    assert!(z.values.iter().all(|v| v.to_bits() == 0));
    assert_eq!(z.source_digest, "zero");
    assert_eq!(z.conditioning_digest(), ZERO_DIGEST);
    let json = serde_json::to_value(&z).unwrap();
    assert_eq!(json["status"], "none");
    assert_eq!(json["subsample_n"], serde_json::Value::Null);
}

#[test]
fn partial_bundle_squashes_in_f32_and_has_the_golden_digest() {
    let b = golden_partial();
    b.validate().unwrap();
    assert_eq!(b.status, LandmarkStatus::Partial(9));
    for i in 0..LANDMARK_SLOTS {
        if MACRO_SLOTS.contains(&i) {
            let g32 = (0.05 * (i + 1) as f64) as f32;
            assert_eq!(b.values[i].to_bits(), (g32 / (1. + g32.abs())).to_bits());
            assert!(b.mask[i]);
        } else {
            assert_eq!(b.values[i].to_bits(), 0);
            assert!(!b.mask[i]);
        }
    }
    assert_eq!(b.conditioning_digest(), PARTIAL_DIGEST);
    // The digest binds values, mask and subsample only (not provenance/source).
    let mut other = b.clone();
    other.provenance = Some("x".into());
    other.source_digest = "other".into();
    assert_eq!(other.conditioning_digest(), PARTIAL_DIGEST);
    let mut n = b.clone();
    n.subsample_n = None;
    assert_ne!(n.conditioning_digest(), PARTIAL_DIGEST);
    let json = serde_json::to_value(&b).unwrap();
    assert_eq!(json["status"], serde_json::json!({"partial": 9}));
}

#[test]
fn nonfinite_inputs_become_positive_zero_and_missing_while_finite_overflow_stays_present() {
    let b = LandmarkBundle::from_raw_gains(
        &gains(&[
            (0, f64::NAN),
            (1, f64::INFINITY),
            (2, f64::NEG_INFINITY),
            (3, 1e300),
            (4, 0.5),
            (5, -1e300),
            (6, 0.0),
            (7, -0.0),
        ]),
        scope("s"),
    );
    // NaN/±Inf become +0.0/missing; finite 1e300, -1e300, 0.5, 0.0, -0.0 stay present.
    assert_eq!(b.mask.iter().filter(|m| **m).count(), 5);
    assert!(b.values[..3].iter().all(|v| v.to_bits() == 0));
    assert!(!b.mask[0] && !b.mask[1] && !b.mask[2]);
    assert!(b.mask[3] && b.values[3] == 1.0);
    assert!(b.mask[4] && b.values[4] == squash_gain(0.5));
    assert!(b.mask[5] && b.values[5] == -1.0);
    assert!(b.mask[6] && b.values[6] == 0.0);
    assert!(b.mask[7] && b.values[7] == 0.0);
    assert_eq!(squash_raw_gain(f64::NAN), None);
    assert_eq!(squash_raw_gain(f64::INFINITY), None);
    assert_eq!(squash_raw_gain(1e300), Some(1.0f32));
    assert_eq!(squash_raw_gain(-1e300), Some(-1.0f32));
    assert_eq!(b.status, LandmarkStatus::Partial(5));
    b.validate().unwrap();
    let mut values = [0f32; 23];
    let mut mask = [false; 23];
    values[0] = f32::NAN;
    values[1] = 2.0;
    values[2] = -0.25;
    values[3] = 0.7; // masked out → +0.0
    mask[..3].copy_from_slice(&[true, true, true]);
    let s = LandmarkBundle::from_squashed(&values, &mask, scope("s"));
    assert_eq!(s.mask.iter().filter(|m| **m).count(), 1);
    assert_eq!(s.values[2], -0.25);
    assert!(s.values[3].to_bits() == 0 && !s.mask[3]);
    let full = LandmarkBundle::from_raw_gains(&[Some(0.0); 23], scope("s"));
    assert_eq!(full.status, LandmarkStatus::Full);
    let none = LandmarkBundle::from_raw_gains(&[None; 23], scope("s"));
    assert_eq!(none.status, LandmarkStatus::None);
    assert!(!none.is_zero());
    // Fake zero with subsample_n = Some(300) is NOT a valid zero sentinel.
    let mut fake_zero = LandmarkBundle::zero();
    fake_zero.subsample_n = Some(300);
    assert!(!fake_zero.is_zero());
    assert!(fake_zero.validate().is_err());
}

#[test]
fn merge_is_a_monotone_idempotent_order_independent_lattice() {
    let ss = || scope("src").with_subsample(Some(300));
    let macro_b = LandmarkBundle::from_raw_gains(
        &gains(&MACRO_SLOTS.map(|s| (s, 0.1 * s as f64))),
        ss().with_provenance("macro"),
    );
    let larger_b = LandmarkBundle::from_raw_gains(
        &gains(&LARGER_SLOTS.map(|s| (s, -0.05 * s as f64))),
        ss().with_provenance("larger"),
    );
    let fine_b = LandmarkBundle::from_raw_gains(
        &gains(&FINE_SLOTS.map(|s| (s, 0.01 * s as f64))),
        ss().with_provenance("fine"),
    );
    let z = LandmarkBundle::zero();
    // Zero is the identity on both sides and adopts the incoming scope.
    assert_eq!(z.merge(&macro_b).unwrap(), macro_b);
    assert_eq!(macro_b.merge(&z).unwrap(), macro_b);
    // Scoped empty result is NOT a universal zero sentinel and must enforce scope.
    let scoped_empty = LandmarkBundle::from_raw_gains(&[None; 23], ss());
    assert!(!scoped_empty.is_zero());
    assert_eq!(scoped_empty.merge(&macro_b).unwrap(), macro_b);
    let other_scoped_empty =
        LandmarkBundle::from_raw_gains(&[None; 23], scope("other").with_subsample(Some(300)));
    assert!(other_scoped_empty.merge(&macro_b).is_err());
    assert!(macro_b.merge(&other_scoped_empty).is_err());
    // merge sanitizes public fields of BOTH operands first (even against zero).
    let mut dirty_in = macro_b.clone();
    dirty_in.values[0] = f32::NAN;
    dirty_in.mask[0] = true;
    dirty_in.status = LandmarkStatus::Full;
    let cleaned = z.merge(&dirty_in).unwrap();
    cleaned.validate().unwrap();
    assert!(!cleaned.mask[0] && cleaned.values[0].to_bits() == 0);
    assert_eq!(cleaned.status, LandmarkStatus::Partial(8));
    let mut dirty_self = macro_b.clone();
    dirty_self.values[1] = f32::INFINITY;
    dirty_self.mask[1] = true;
    let cleaned2 = dirty_self.merge(&larger_b).unwrap();
    cleaned2.validate().unwrap();
    assert!(!cleaned2.mask[1] && cleaned2.values[1].to_bits() == 0);
    assert_eq!(cleaned2.status, LandmarkStatus::Partial(17));
    // Idempotent.
    assert_eq!(macro_b.merge(&macro_b).unwrap(), macro_b);
    // Monotone status progression.
    let a = z.merge(&macro_b).unwrap();
    let b = a.merge(&larger_b).unwrap();
    let c = b.merge(&fine_b).unwrap();
    assert_eq!(a.status, LandmarkStatus::Partial(9));
    assert_eq!(b.status, LandmarkStatus::Partial(18));
    assert_eq!(c.status, LandmarkStatus::Full);
    c.validate().unwrap();
    // Order independence for disjoint observations (incl. provenance).
    let d = fine_b
        .merge(&larger_b)
        .unwrap()
        .merge(&z)
        .unwrap()
        .merge(&macro_b)
        .unwrap();
    assert_eq!(c, d);
    assert_eq!(c.conditioning_digest(), d.conditioning_digest());
    assert_eq!(c.provenance.as_deref(), Some("fine+larger+macro"));
    // Never downgrades: merging a sparser bundle keeps every observation.
    assert_eq!(c.merge(&macro_b).unwrap().mask, [true; 23]);
    assert_eq!(c.merge(&c).unwrap(), c);
    // Incoming wins on overlap.
    let newer = LandmarkBundle::from_raw_gains(&gains(&[(0, 0.9)]), ss());
    assert_eq!(c.merge(&newer).unwrap().values[0], squash_gain(0.9));
    // Incompatible scopes are rejected.
    let other_n = LandmarkBundle::from_raw_gains(&gains(&[(0, 0.9)]), scope("src"));
    assert!(c.merge(&other_n).is_err());
    let other_src =
        LandmarkBundle::from_raw_gains(&gains(&[(0, 0.9)]), scope("x").with_subsample(Some(300)));
    assert!(c.merge(&other_src).is_err());
}

#[test]
fn serde_round_trips_sanitizes_fields_and_rejects_invalid_scopes() {
    for b in [LandmarkBundle::zero(), golden_partial(), {
        LandmarkBundle::from_raw_gains(&[Some(-0.37); 23], scope("s").with_provenance("native-v1"))
    }] {
        let text = serde_json::to_string(&b).unwrap();
        let back: LandmarkBundle = serde_json::from_str(&text).unwrap();
        assert_eq!(back, b);
        assert_eq!(
            back.values.map(f32::to_bits),
            b.values.map(f32::to_bits),
            "bit-exact values"
        );
        assert_eq!(back.conditioning_digest(), b.conditioning_digest());
    }
    let full =
        serde_json::to_value(LandmarkBundle::from_raw_gains(&[Some(1.0); 23], scope("s"))).unwrap();
    assert_eq!(full["status"], "full");
    // Deserialization sanitizes incoming fields and DERIVES status from the sanitized mask.
    let mut stale_status = serde_json::to_value(golden_partial()).unwrap();
    stale_status["status"] = "full".into();
    let from_stale: LandmarkBundle = serde_json::from_value(stale_status).unwrap();
    assert_eq!(from_stale, golden_partial());
    assert_eq!(from_stale.status, LandmarkStatus::Partial(9));

    let mut unmasked_val = serde_json::to_value(golden_partial()).unwrap();
    unmasked_val["values"][10] = 0.5.into();
    let from_unmasked: LandmarkBundle = serde_json::from_value(unmasked_val).unwrap();
    assert_eq!(from_unmasked, golden_partial());

    let mut out_of_range = serde_json::to_value(golden_partial()).unwrap();
    out_of_range["values"][0] = 1.5.into();
    let from_oor: LandmarkBundle = serde_json::from_value(out_of_range).unwrap();
    assert!(!from_oor.mask[0] && from_oor.values[0].to_bits() == 0);
    assert_eq!(from_oor.status, LandmarkStatus::Partial(8));

    let bad = |edit: &dyn Fn(&mut serde_json::Value)| {
        let mut v = serde_json::to_value(golden_partial()).unwrap();
        edit(&mut v);
        serde_json::from_value::<LandmarkBundle>(v).is_err()
    };
    assert!(bad(&|v| v["source_digest"] = "".into()));
    assert!(bad(&|v| v["source_digest"] = "zero".into()));
    assert!(bad(&|v| v["extra"] = 1.into()));
    assert!(bad(&|v| v["subsample_n"] = 0.into()));
    assert!(!bad(&|_| {}));
}

#[test]
fn modes_groups_and_id_parsing_match_the_descriptor_tiers() {
    for m in LandmarkMode::ALL {
        assert_eq!(m.as_str().parse::<LandmarkMode>().unwrap(), m);
        assert_eq!(serde_json::to_value(m).unwrap(), m.as_str());
    }
    assert!("partial".parse::<LandmarkMode>().is_err());
    assert_eq!(
        PROGRESSIVE_ORDER,
        [
            LandmarkGroup::Macro,
            LandmarkGroup::Larger,
            LandmarkGroup::Fine
        ]
    );
    let mut all: Vec<usize> = PROGRESSIVE_ORDER
        .iter()
        .flat_map(|g| g.slots().iter().copied())
        .collect();
    all.sort_unstable();
    assert_eq!(all, (0..23).collect::<Vec<_>>());
    assert_eq!(
        LandmarkGroup::Macro.ids(),
        vec![1, 2, 3, 4, 5, 6, 14, 15, 22]
    );
    assert_eq!(
        LandmarkGroup::Larger.ids(),
        vec![7, 8, 9, 10, 11, 12, 13, 16, 17]
    );
    assert_eq!(LandmarkGroup::Fine.ids(), vec![18, 19, 20, 21, 23]);
    assert_eq!(
        parse_landmark_ids("fine").unwrap(),
        vec![18, 19, 20, 21, 23]
    );
    assert_eq!(parse_landmark_ids("23,lin, 2,lin").unwrap(), vec![1, 2, 23]);
    assert_eq!(
        parse_landmark_ids("macro,larger,fine").unwrap(),
        (1..=23).collect::<Vec<u8>>()
    );
    for bad in ["", "0", "24", "ridge", "macro,"] {
        assert!(parse_landmark_ids(bad).is_err(), "{bad}");
    }
}

#[test]
fn stratified_subsamples_are_deterministic_sorted_and_balanced() {
    let y: Vec<f64> = (0..1000).map(|i| ((i * 7919) % 1000) as f64).collect();
    let spec = SubsampleSpec::new(300, 7);
    let a = spec.stratified_indices(&y, Task::Regression).unwrap();
    assert_eq!(
        a,
        stratified_subsample(&y, Task::Regression, 300, 7).unwrap()
    );
    assert_eq!(a.len(), 300);
    assert!(a.windows(2).all(|w| w[0] < w[1]));
    // Each y-decile contributes exactly 30 rows.
    for b in 0..10 {
        let lo = (b * 100) as f64;
        assert_eq!(
            a.iter()
                .filter(|&&i| y[i] >= lo && y[i] < lo + 100.)
                .count(),
            30
        );
    }
    assert_ne!(
        a,
        SubsampleSpec::new(300, 8)
            .stratified_indices(&y, Task::Regression)
            .unwrap()
    );
    assert_eq!(
        SubsampleSpec::new(5000, 0)
            .stratified_indices(&y, Task::Regression)
            .unwrap(),
        (0..1000).collect::<Vec<_>>()
    );
    assert!(SubsampleSpec::new(0, 0)
        .stratified_indices(&y, Task::Regression)
        .is_err());
    // Classification: proportional with at least two rows per class.
    let labels: Vec<f64> = (0..1000)
        .map(|i| {
            if i < 900 {
                0.
            } else if i < 997 {
                1.
            } else {
                2.
            }
        })
        .collect();
    let c = SubsampleSpec::new(100, 1)
        .stratified_indices(&labels, Task::Classification)
        .unwrap();
    assert_eq!(c.len(), 100);
    let count = |k: f64| c.iter().filter(|&&i| labels[i] == k).count();
    assert_eq!((count(0.), count(1.), count(2.)), (87, 11, 2));
    assert!(SubsampleSpec::new(5, 1)
        .stratified_indices(&labels, Task::Classification)
        .is_err());
    // Pinned head of a fixed draw, so mirrors (Python) can check determinism.
    assert_eq!(&a[..5], &PINNED_HEAD);
}

const PINNED_HEAD: [usize; 5] = [1, 7, 8, 12, 15];

fn planted(n: usize) -> LandmarkFrame {
    let x: Vec<f64> = (0..n)
        .map(|i| ((i * 73) % n) as f64 / n as f64 * 2.0 - 1.0)
        .collect();
    let z: Vec<f64> = (0..n)
        .map(|i| ((i * 137 + 11) % n) as f64 / n as f64 * 2.0 - 1.0)
        .collect();
    let y = x
        .iter()
        .zip(&z)
        .map(|(x, z)| 2.0 * x + 0.3 * z + 0.4 * (x * z))
        .collect();
    LandmarkFrame::from_columns(
        vec![x, z],
        vec![None, None],
        y,
        Head::Regression,
        (0..n as u64).collect(),
        None,
    )
    .unwrap()
}
fn small_options() -> LandmarkOptions {
    LandmarkOptions {
        folds: 3,
        ridge_rows: 256,
        small_rows: 256,
        spectrum_features: 2,
        work_budget: u64::MAX,
        ..Default::default()
    }
}

#[test]
fn signature_requested_none_is_byte_identical_to_signature() {
    let frame = planted(160);
    let opts = small_options();
    let a = serde_json::to_vec(&signature(&frame, &opts).unwrap().private_report()).unwrap();
    let b = serde_json::to_vec(
        &signature_requested(&frame, &opts, None)
            .unwrap()
            .private_report(),
    )
    .unwrap();
    assert_eq!(a, b);
    let all: Vec<u8> = (1..=23).collect();
    let c = serde_json::to_vec(
        &signature_requested(&frame, &opts, Some(&all))
            .unwrap()
            .private_report(),
    )
    .unwrap();
    assert_eq!(a, c);
}

#[test]
fn requested_cells_match_the_full_panel_and_others_are_not_requested() {
    let frame = planted(160);
    let opts = small_options();
    let full = signature(&frame, &opts).unwrap();
    let ids = LandmarkGroup::Macro.ids();
    let part = signature_requested(&frame, &opts, Some(&ids)).unwrap();
    let report = part.private_report();
    for (i, cell) in part.cells.iter().enumerate() {
        if ids.contains(&(i as u8 + 1)) {
            assert_ne!(cell.state, CellState::NotRequested);
            assert_eq!(
                serde_json::to_value(cell).unwrap(),
                serde_json::to_value(&full.cells[i]).unwrap()
            );
        } else {
            assert_eq!(cell.state, CellState::NotRequested);
            assert!(cell.gain.is_none());
            assert_eq!(report["cells"][i]["state"], "not_requested");
        }
    }
    assert_eq!(part.work.cells_run, ids.len());
    // Scoped bundle of the requested report = full bundle restricted to the group.
    let frame_scope = LandmarkScope::for_frame(&frame, &opts, None).bundle_scope();
    let pb = LandmarkBundle::from_report_with_scope(
        &report,
        frame_scope.clone(),
        REPORT_RECIPE_OBSERVED,
    )
    .unwrap();
    let fb = LandmarkBundle::from_report_with_scope(
        &full.private_report(),
        frame_scope.clone(),
        REPORT_RECIPE_OBSERVED,
    )
    .unwrap();
    assert_eq!(pb.source_digest, fb.source_digest);
    for i in 0..23 {
        if MACRO_SLOTS.contains(&i) {
            assert_eq!(pb.mask[i], fb.mask[i]);
            assert_eq!(pb.values[i].to_bits(), fb.values[i].to_bits());
        } else {
            assert!(!pb.mask[i]);
        }
    }
    // Groups merge back into the full-panel bundle when bound to the same frame scope.
    let mut merged = LandmarkBundle::zero();
    for g in PROGRESSIVE_ORDER {
        let r = signature_requested(&frame, &opts, Some(&g.ids()))
            .unwrap()
            .private_report();
        merged = merged
            .merge(
                &LandmarkBundle::from_report_with_scope(
                    &r,
                    frame_scope.clone(),
                    REPORT_RECIPE_OBSERVED,
                )
                .unwrap(),
            )
            .unwrap();
    }
    assert_eq!(merged.values.map(f32::to_bits), fb.values.map(f32::to_bits));
    assert_eq!(merged.mask, fb.mask);
    // Unbound from_report does not advertise source-bound provenance and does not
    // collide across different same-shaped tables.
    let unbound_a = LandmarkBundle::from_report(&full.private_report(), None).unwrap();
    assert!(unbound_a.provenance.is_none());
    let alt_frame = LandmarkFrame::from_columns(
        vec![vec![1.0; 160], vec![2.0; 160]],
        vec![None, None],
        (0..160).map(|i| (i as f64).sin()).collect(),
        Head::Regression,
        (0..160).collect(),
        None,
    )
    .unwrap();
    let alt_full = signature(&alt_frame, &opts).unwrap();
    let unbound_b = LandmarkBundle::from_report(&alt_full.private_report(), None).unwrap();
    assert_ne!(unbound_a.source_digest, unbound_b.source_digest);
    assert!(unbound_a.merge(&unbound_b).is_err());
    for bad in [&[][..], &[0][..], &[24][..], &[1, 1][..]] {
        assert!(signature_requested(&frame, &opts, Some(bad)).is_err());
    }
}

#[test]
fn subsampled_frame_keeps_physical_ids_and_runs() {
    let frame = planted(400);
    let spec = SubsampleSpec::new(120, 3);
    let (sub, selection) = frame.subsample(&spec).unwrap();
    assert_eq!(sub.rows(), 120);
    assert_eq!(selection.positions.len(), 120);
    // planted() ids are 0..n, so ids == positions here.
    assert_eq!(
        selection.row_ids,
        selection
            .positions
            .iter()
            .map(|p| *p as u64)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        selection.digest,
        row_selection_digest(&selection.spec, &selection.row_ids)
    );
    let opts = small_options();
    let s = signature_requested(&sub, &opts, Some(&[1, 2])).unwrap();
    assert_eq!(s.rows, 120);
    let unbound = LandmarkBundle::from_report(&s.private_report(), Some(120)).unwrap();
    assert_eq!(unbound.subsample_n, Some(120));
    assert!(unbound.provenance.is_none());
    let bound_scope = LandmarkScope::for_frame(&frame, &opts, Some(spec)).bundle_scope();
    let b = LandmarkBundle::from_report_with_scope(
        &s.private_report(),
        bound_scope,
        REPORT_RECIPE_COMPLETE,
    )
    .unwrap();
    assert_eq!(b.subsample_n, Some(120));
    assert_eq!(b.provenance.as_deref(), Some("subsample-120-stratified"));
    assert!(frame.subsample(&SubsampleSpec::new(10, 3)).is_err()); // < 20 rows
}

#[test]
fn selection_is_keyed_by_physical_ids_not_row_order() {
    let n = 500usize;
    let y: Vec<f64> = (0..n).map(|i| ((i * 31) % 97) as f64).collect();
    let ids: Vec<u64> = (0..n as u64).map(|i| 1_000_000 + i * 7).collect();
    let spec = SubsampleSpec::new(150, 11);
    let a = spec.select(&y, Task::Regression, Some(&ids)).unwrap();
    // Reverse the physical order: the same physical rows are selected.
    let ry: Vec<f64> = y.iter().rev().copied().collect();
    let rids: Vec<u64> = ids.iter().rev().copied().collect();
    let b = spec.select(&ry, Task::Regression, Some(&rids)).unwrap();
    let mut ai = a.row_ids.clone();
    let mut bi = b.row_ids.clone();
    ai.sort_unstable();
    bi.sort_unstable();
    assert_eq!(ai, bi);
    assert_eq!(a.digest, b.digest);
    assert!(spec.select(&y, Task::Regression, Some(&ids[..10])).is_err());
    let mut dup = ids.clone();
    dup[1] = dup[0];
    assert!(spec.select(&y, Task::Regression, Some(&dup)).is_err());
}

#[test]
fn progressive_driver_accumulates_groups_and_ends_byte_identical_to_full() {
    let frame = planted(160);
    let opts = small_options();
    let full = signature(&frame, &opts).unwrap();
    let full_json = serde_json::to_vec(&full.private_report()).unwrap();
    let mut p = ProgressiveLandmarks::new(&frame, &opts, None).unwrap();
    assert!(p.bundle().is_zero());
    let scope_digest = p.scope().digest();
    let mut last = 0;
    for (k, g) in PROGRESSIVE_ORDER.into_iter().enumerate() {
        let (bundle, report) = p.next_group(g).unwrap();
        let present = report
            .cells
            .iter()
            .filter(|c| c.state != CellState::NotRequested)
            .count();
        assert_eq!(present, [9, 18, 23][k], "cells computed after {g:?}");
        assert!(bundle.status.present() >= last);
        last = bundle.status.present();
        assert_eq!(bundle.source_digest, scope_digest);
        if k < 2 {
            assert!(report.spectrum.is_empty());
        }
    }
    assert!(p.is_complete());
    assert_eq!(
        serde_json::to_vec(&p.report().private_report()).unwrap(),
        full_json
    );
    let fb = LandmarkBundle::from_report(&full.private_report(), None).unwrap();
    assert_eq!(
        p.bundle().features46().map(f32::to_bits),
        fb.features46().map(f32::to_bits)
    );
    // Re-running a finished group or ids at terminal state is a no-op and does not
    // recompute spectrum or mutate work units.
    let before = p.bundle().clone();
    let (again, again_rep) = p.next_group(LandmarkGroup::Macro).unwrap();
    assert_eq!(again, before);
    assert_eq!(
        serde_json::to_vec(&again_rep.private_report()).unwrap(),
        full_json
    );
    let (again_ids, again_ids_rep) = p.run_ids(&[1, 18, 23]).unwrap();
    assert_eq!(again_ids, before);
    assert_eq!(
        serde_json::to_vec(&again_ids_rep.private_report()).unwrap(),
        full_json
    );
    // Order of groups does not change the end state.
    let mut q = ProgressiveLandmarks::new(&frame, &opts, None).unwrap();
    for g in [
        LandmarkGroup::Fine,
        LandmarkGroup::Macro,
        LandmarkGroup::Larger,
    ] {
        q.next_group(g).unwrap();
    }
    assert_eq!(
        serde_json::to_vec(&q.report().private_report()).unwrap(),
        full_json
    );
    assert_eq!(q.bundle(), p.bundle());
}

#[test]
fn progressive_driver_matches_full_under_bounded_work_budget() {
    let frame = planted(160);
    let mut opts = small_options();
    // Tight budget: drops expensive cells across groups (e.g. 23, 22, 21, ...) while
    // keeping linear baseline and cheaper probes.
    opts.work_budget = 500_000;
    let full = signature(&frame, &opts).unwrap();
    assert!(
        !full.work.dropped.is_empty(),
        "tight budget must drop some cells"
    );
    assert!(
        full.cells.iter().any(|c| c.state == CellState::OverBudget)
            && full.cells.iter().any(|c| c.state == CellState::Complete),
        "tight budget must mix Complete and OverBudget cells"
    );
    let full_json = serde_json::to_vec(&full.private_report()).unwrap();
    let mut p = ProgressiveLandmarks::new(&frame, &opts, None).unwrap();
    for g in PROGRESSIVE_ORDER {
        p.next_group(g).unwrap();
    }
    assert!(p.is_complete());
    assert_eq!(
        serde_json::to_vec(&p.report().private_report()).unwrap(),
        full_json
    );
    let fb = LandmarkBundle::from_report_with_scope(
        &full.private_report(),
        p.scope().bundle_scope(),
        REPORT_RECIPE_OBSERVED,
    )
    .unwrap();
    assert_eq!(p.bundle(), &fb);
    // Repeated terminal call under tight budget stays byte-identical.
    let (b2, r2) = p.next_group(LandmarkGroup::Fine).unwrap();
    assert_eq!(b2, fb);
    assert_eq!(serde_json::to_vec(&r2.private_report()).unwrap(), full_json);
}

#[test]
fn progressive_subsample_scope_binds_n_seed_and_source() {
    let frame = planted(400);
    let opts = small_options();
    let spec = SubsampleSpec::new(150, 2);
    let mut p = ProgressiveLandmarks::new(&frame, &opts, Some(spec)).unwrap();
    assert_eq!(p.selection().unwrap().row_ids.len(), 150);
    let (b, r) = p.next_group(LandmarkGroup::Macro).unwrap();
    assert_eq!(r.rows, 150);
    assert_eq!(b.subsample_n, Some(150));
    // Equivalent to a requested signature on the explicitly subsampled frame.
    let (sub, _) = frame.subsample(&spec).unwrap();
    let direct = signature_requested(&sub, &opts, Some(&LandmarkGroup::Macro.ids())).unwrap();
    for i in MACRO_SLOTS {
        assert_eq!(
            serde_json::to_value(&r.cells[i]).unwrap(),
            serde_json::to_value(&direct.cells[i]).unwrap()
        );
    }
    let other_seed = ProgressiveLandmarks::new(&frame, &opts, Some(SubsampleSpec::new(150, 3)))
        .unwrap()
        .scope()
        .digest();
    assert_ne!(other_seed, p.scope().digest());
    let mut q = ProgressiveLandmarks::new(&frame, &opts, Some(SubsampleSpec::new(150, 3))).unwrap();
    let (other, _) = q.next_group(LandmarkGroup::Larger).unwrap();
    assert!(b.merge(&other).is_err(), "different seed must not merge");
    let unsampled = ProgressiveLandmarks::new(&frame, &opts, None).unwrap();
    assert_ne!(unsampled.scope().digest(), p.scope().digest());

    // Explicit source_rows_digest (e.g. PreparedTable::source_rows_digest() with
    // categorical/target vocabulary or raw training cells) distinguishes two tables
    // that share the same numeric LandmarkFrame codes and physical row IDs.
    let mut vocab_a = ProgressiveLandmarks::new_with_source_digest(
        &frame,
        &opts,
        Some(spec),
        source_rows_digest("vocab-cat-a-target-pos-yes", &(0..400).collect::<Vec<_>>()),
    )
    .unwrap();
    let mut vocab_b = ProgressiveLandmarks::new_with_source_digest(
        &frame,
        &opts,
        Some(spec),
        source_rows_digest("vocab-cat-b-target-pos-no", &(0..400).collect::<Vec<_>>()),
    )
    .unwrap();
    assert_ne!(vocab_a.scope().digest(), vocab_b.scope().digest());
    assert_ne!(vocab_a.scope().digest(), p.scope().digest());
    let (ba, _) = vocab_a.next_group(LandmarkGroup::Macro).unwrap();
    let (bb, _) = vocab_b.next_group(LandmarkGroup::Larger).unwrap();
    assert!(
        ba.merge(&bb).is_err(),
        "renamed vocabulary/target mapping with identical numeric frame codes must not merge"
    );
    // Reject empty or "zero" source_rows_digest.
    assert!(ProgressiveLandmarks::new_with_source_digest(&frame, &opts, None, "").is_err());
    assert!(ProgressiveLandmarks::new_with_source_digest(&frame, &opts, None, "zero").is_err());
}

// ---- strict-path bit-identity --------------------------------------------

fn write_input(
    root: &std::path::Path,
    report: &[u8],
    recipe: &str,
    rows: u64,
) -> (PathBuf, String, PathBuf, Schema) {
    std::fs::create_dir_all(root).unwrap();
    let csv = root.join("fit.csv");
    let text = "a,b,target\n1,2,3\n";
    std::fs::write(&csv, text).unwrap();
    let report_value: serde_json::Value = serde_json::from_slice(report).unwrap();
    let v2 = recipe == landmark_features::OBSERVED_RECIPE;
    let manifest = serde_json::json!({
        "format": if v2 {"dope-fit-cv-landmark-input-v2"} else {"dope-fit-cv-landmark-input-v1"},
        "feature_recipe": recipe, "options": report_value["options"],
        "engine_sha256": digest(b"engine"), "report_sha256": digest(report),
        "producer": {"purpose": "bundle bit-identity test"},
        "scope": {"allowed_roles": ["fit"], "split_policy": "iid",
            "training_input_sha256": digest(text.as_bytes()),
            "column_kinds": ["numeric", "numeric", "numeric"], "task": "regression", "rows": rows}});
    let raw = serde_json::to_vec(&manifest).unwrap();
    std::fs::write(root.join("manifest.json"), &raw).unwrap();
    std::fs::write(root.join("report.json"), report).unwrap();
    let schema: Schema = serde_json::from_value(serde_json::json!({
        "preprocessing": "fit-only-positional-mixed-v1", "task": "regression", "target_index": 2,
        "output": "positional_numeric_and_category_ids",
        "columns": [
            {"slot": 1, "kind": "numeric", "center": 0.0, "scale": 1.0, "categories": 0},
            {"slot": 2, "kind": "numeric", "center": 0.0, "scale": 1.0, "categories": 0},
            {"slot": 0, "kind": "numeric", "center": 0.0, "scale": 1.0, "categories": 0}]}))
    .unwrap();
    (root.to_path_buf(), digest(&raw), csv, schema)
}

fn assert_bit_identical(report: &[u8], rows: u64, tag: &str) -> usize {
    let value: serde_json::Value = serde_json::from_slice(report).unwrap();
    let mut present = 0;
    for (recipe, observed) in [
        (landmark_features::RECIPE, false),
        (landmark_features::OBSERVED_RECIPE, true),
    ] {
        let root = std::env::temp_dir().join(format!(
            "dope-bundle-identity-{tag}-{observed}-{}",
            std::process::id()
        ));
        let (dir, expected, csv, schema) = write_input(&root, report, recipe, rows);
        let strict =
            landmark_features::load_for_recipe(&dir, &expected, &csv, &schema, recipe).unwrap();
        let bundle = if observed {
            LandmarkBundle::from_report(&value, None).unwrap()
        } else {
            LandmarkBundle::from_report_complete(&value, None).unwrap()
        };
        assert_eq!(
            bundle.features46().map(f32::to_bits),
            strict.map(f32::to_bits),
            "{tag} {recipe}"
        );
        present = present.max(bundle.status.present());
        std::fs::remove_dir_all(&root).ok();
    }
    present
}

#[test]
fn full_bundle_from_report_is_bit_identical_to_the_strict_loader() {
    // Synthetic full report: 23 complete cells with gains that stress the
    // f64→f32 cast and the f32 squash; native-shaped evidence fields.
    let rows = 100u64;
    let tricky = [
        0.1,
        0.2,
        0.30000000000000004,
        1e-8,
        -0.999_999_9,
        12345.678,
        -3.5,
        0.333_333_333_333_333_3,
        0.123_456_789_012_345_67,
        2.0f64.powi(-30),
        -1e-40,
        0.999_999_999,
        7.0,
        1.0 / 3.0,
        -0.5,
        0.05,
        0.987_654_321,
        0.0,
        -0.0,
        3.4e38,
        0.4,
        0.6,
        0.011,
    ];
    let cells: Vec<serde_json::Value> = tricky
        .iter()
        .enumerate()
        .map(|(i, g)| {
            serde_json::json!({"id": i + 1, "state": "complete", "gain": g, "se": 0.01,
                "stable": true, "train_rows": [80, 80, 80, 80, 80], "query_rows": rows})
        })
        .collect();
    let report = serde_json::json!({"panel": "landmarks-native-v1", "kind": "real",
        "privacy": "private", "task": "regression", "classes": 0, "rows": rows, "features": 2,
        "options": LandmarkOptions::default(), "cells": cells});
    let raw = serde_json::to_vec_pretty(&report).unwrap();
    assert_eq!(assert_bit_identical(&raw, rows, "synthetic"), 23);
    let b = LandmarkBundle::from_report(&report, None).unwrap();
    assert_eq!(b.status, LandmarkStatus::Full);
    for (i, g) in tricky.iter().enumerate() {
        assert_eq!(b.values[i].to_bits(), squash_gain(*g).to_bits());
    }
}

#[test]
fn native_report_bundle_is_bit_identical_to_the_strict_loader() {
    let frame = planted(100);
    let s = signature(&frame, &LandmarkOptions::default()).unwrap();
    let raw = serde_json::to_vec_pretty(&s.private_report()).unwrap();
    assert!(assert_bit_identical(&raw, 100, "native") > 0);
}

#[test]
fn raw_gain_bundles_reproduce_the_python_fixture_landmark_inputs() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for (version, raw) in [
        (
            "v3",
            (0..23).map(|i| 0.1 * (i + 1) as f64).collect::<Vec<_>>(),
        ),
        (
            "v4",
            [-1., 0., 0.005]
                .into_iter()
                .chain(std::iter::repeat_n(0.1, 20))
                .collect(),
        ),
    ] {
        for name in ["input", "masked"] {
            let t = candle_core::safetensors::load(
                root.join(format!("learned-{version}/{name}/parity.safetensors")),
                &Device::Cpu,
            )
            .unwrap();
            let expected: Vec<f32> = t["landmark_inputs"].to_vec1().unwrap();
            let gains: [Option<f64>; 23] = std::array::from_fn(|i| Some(raw[i]));
            let b = LandmarkBundle::from_raw_gains(&gains, scope("fixture"));
            assert_eq!(b.status, LandmarkStatus::Full);
            assert_eq!(
                b.features46().map(f32::to_bits).to_vec(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{version}/{name}"
            );
        }
    }
}

#[test]
fn scope_digest_binds_panel_options_subsample_and_rows() {
    let o = LandmarkOptions::default();
    let rows = source_rows_digest("abc", &[3, 1, 2]);
    assert_eq!(rows, source_rows_digest("abc", &[1, 2, 3]));
    assert_ne!(rows, source_rows_digest("abd", &[1, 2, 3]));
    assert_ne!(rows, source_rows_digest("abc", &[1, 2, 4]));
    let a = LandmarkScope::new(&o, None, rows.clone());
    assert_eq!(a.panel, "landmarks-native-v1");
    assert_eq!(a.options_digest, options_digest(&o));
    assert_eq!(a.digest().len(), 64);
    let other_opts = LandmarkOptions {
        seed: 1,
        ..Default::default()
    };
    assert_ne!(
        a.digest(),
        LandmarkScope::new(&other_opts, None, rows.clone()).digest()
    );
    let s0 = LandmarkScope::new(&o, Some(SubsampleSpec::new(300, 0)), rows.clone());
    let s1 = LandmarkScope::new(&o, Some(SubsampleSpec::new(300, 1)), rows.clone());
    assert_ne!(a.digest(), s0.digest());
    assert_ne!(s0.digest(), s1.digest());
    assert_eq!(s0.bundle_scope().subsample_n, Some(300));
    assert_eq!(s0.bundle_scope().source_digest, s0.digest());
    let json = serde_json::to_string(&s0).unwrap();
    assert_eq!(serde_json::from_str::<LandmarkScope>(&json).unwrap(), s0);
    assert_eq!(planted(50).content_digest(), planted(50).content_digest());
    assert_ne!(planted(50).content_digest(), planted(51).content_digest());
    assert!(LandmarkMode::Fast.is_landmark_free() && LandmarkMode::None.is_landmark_free());
    assert!(!LandmarkMode::Progressive.is_landmark_free());
}

#[test]
fn uninformative_finite_zeros_stay_present_in_optional_bundles() {
    let cells: Vec<serde_json::Value> = (0..23)
        .map(|i| match i {
            0 => serde_json::json!({"id": 1, "state": "uninformative", "gain": 0.0, "se": 0.01}),
            1 => serde_json::json!({"id": 2, "state": "uninformative", "gain": -0.005, "se": 0.01}),
            2 => serde_json::json!({"id": 3, "state": "uninformative", "gain": null, "se": null}),
            3 => serde_json::json!({"id": 4, "state": "complete", "gain": 0.25, "se": 0.01}),
            _ => {
                serde_json::json!({"id": i + 1, "state": "not_requested", "gain": null, "se": null})
            }
        })
        .collect();
    let report = serde_json::json!({
        "panel": "landmarks-native-v1",
        "kind": "real",
        "privacy": "private",
        "task": "regression",
        "classes": 0,
        "rows": 100,
        "features": 2,
        "options": LandmarkOptions::default(),
        "cells": cells,
    });
    let observed = LandmarkBundle::from_report(&report, None).unwrap();
    assert_eq!(observed.status, LandmarkStatus::Partial(3));
    assert!(observed.mask[0] && observed.values[0].to_bits() == 0);
    assert!(observed.mask[1] && observed.values[1].to_bits() == squash_gain(-0.005).to_bits());
    assert!(!observed.mask[2] && observed.values[2].to_bits() == 0);
    assert!(observed.mask[3] && observed.values[3].to_bits() == squash_gain(0.25).to_bits());

    let strict_v1 = LandmarkBundle::from_report_complete(&report, None).unwrap();
    assert_eq!(strict_v1.status, LandmarkStatus::Partial(1));
    assert!(!strict_v1.mask[0] && !strict_v1.mask[1] && strict_v1.mask[3]);
}

#[test]
fn shared_cross_repo_fixture_landmark_bundle_v1_matches_digests_and_roundtrips() {
    let raw = include_str!("fixtures/landmark_bundle_v1.json");
    let v: serde_json::Value = serde_json::from_str(raw).unwrap();
    assert_eq!(v["schema"], "dope.landmark-bundle-fixtures/v1");
    let cases = v["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let expected_digest = case["conditioning_digest"].as_str().unwrap();
        let bundle: LandmarkBundle = serde_json::from_value(case["bundle"].clone()).unwrap();
        bundle.validate().unwrap();
        assert_eq!(
            bundle.conditioning_digest(),
            expected_digest,
            "case {name} conditioning_digest"
        );
        let round_trip: LandmarkBundle =
            serde_json::from_str(&serde_json::to_string(&bundle).unwrap()).unwrap();
        assert_eq!(round_trip, bundle);
        assert_eq!(round_trip.conditioning_digest(), expected_digest);
    }
}

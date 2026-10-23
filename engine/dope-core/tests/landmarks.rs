use dope_core::{
    landmark::{
        assign_folds, signature, CellState, LandmarkFrame, LandmarkOptions, TransferReference,
        PANEL_V1,
    },
    levels::{ensure_disjoint, fit_levels, Level, LevelOptions},
    schema::{infer, IntakeOptions},
    target::Head,
    FitOptions, RawTable, Samples,
};

fn options() -> LandmarkOptions {
    LandmarkOptions {
        folds: 3,
        ridge_rows: 256,
        small_rows: 256,
        spectrum_features: 0,
        work_budget: u64::MAX,
        ..Default::default()
    }
}
fn planted(n: usize) -> LandmarkFrame {
    let x: Vec<f64> = (0..n)
        .map(|i| ((i * 73) % n) as f64 / n as f64 * 2.0 - 1.0)
        .collect();
    let z: Vec<f64> = (0..n)
        .map(|i| ((i * 137 + 11) % n) as f64 / n as f64 * 2.0 - 1.0)
        .collect();
    let y = x.iter().zip(&z).map(|(x, z)| 2.0 * x + 0.3 * z).collect();
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
fn table() -> RawTable {
    let mut text = "first,second,target\n".to_string();
    for i in 0..180 {
        text += &format!(
            "{},{},{}\n",
            i % 67,
            (i * 37) % 89,
            2 * (i % 67) + (i * 37) % 89
        );
    }
    RawTable::from_reader(text.as_bytes()).unwrap()
}

#[test]
fn all_landmarks_are_ordered_finite_and_linear_signal_is_recovered() {
    let s = signature(&planted(180), &options()).unwrap();
    assert_eq!(s.panel, PANEL_V1);
    assert_eq!(s.cells.len(), 23);
    assert!(s.cells[0].gain.unwrap() > 0.99);
    for (i, c) in s.cells.iter().enumerate() {
        assert_eq!(c.id.0 as usize, i + 1);
        assert!(c.gain.is_none_or(f64::is_finite));
        assert!(c.se.is_none_or(f64::is_finite));
    }
    assert_eq!(s.cells[13].state, CellState::Unsupported);
    let g = s.global_coordinates();
    assert_eq!(g.values.len(), 96);
    assert_eq!(g.mask.len(), 96);
    assert!(g.values.iter().all(|v| v.is_finite()));
    assert!(!g.mask[89]); // T03 is explicitly unimplemented in this panel version
}

#[test]
fn reproducible_across_one_two_and_eight_threads() {
    let frame = planted(120);
    let mut results = Vec::new();
    for n in [1, 2, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap();
        results.push(pool.install(|| {
            serde_json::to_vec(&signature(&frame, &options()).unwrap().private_report()).unwrap()
        }));
    }
    assert_eq!(results[0], results[1]);
    assert_eq!(results[1], results[2]);
}

#[test]
fn groups_never_cross_folds_and_too_few_blocks_refuse() {
    let n = 120;
    let groups: Vec<u64> = (0..n).map(|i| (i / 4) as u64).collect();
    let frame = LandmarkFrame::from_columns(
        vec![(0..n).map(|i| i as f64).collect()],
        vec![None],
        (0..n).map(|i| (i % 2) as f64).collect(),
        Head::Binary,
        (0..n as u64).collect(),
        Some(groups.clone()),
    )
    .unwrap();
    let f = assign_folds(&frame, 5, 13).unwrap();
    for i in 0..n {
        for j in 0..n {
            if groups[i] == groups[j] {
                assert_eq!(f[i], f[j]);
            }
        }
    }
    let bad = LandmarkFrame::from_columns(
        vec![vec![0.0; n]],
        vec![None],
        vec![0.0; n],
        Head::Regression,
        (0..n as u64).collect(),
        Some(vec![1; n]),
    )
    .unwrap();
    assert!(assign_folds(&bad, 5, 0).is_err());
}

#[test]
fn private_role_mutation_cannot_change_fit_signature() {
    let raw = table();
    let schema = infer(&raw, 2, &IntakeOptions::default()).unwrap();
    let mut data = schema.to_model(&raw).unwrap();
    let rows: Vec<usize> = (0..100).collect();
    let mut o = options();
    o.work_budget = 100_000;
    let before = signature(
        &LandmarkFrame::real(&schema, &data, &rows, None).unwrap(),
        &o,
    )
    .unwrap()
    .private_report();
    for y in &mut data.y[100..] {
        *y = 9999.0;
    }
    let after = signature(
        &LandmarkFrame::real(&schema, &data, &rows, None).unwrap(),
        &o,
    )
    .unwrap()
    .private_report();
    assert_eq!(before, after);
}

#[test]
fn constant_targets_are_masked_and_budget_is_explicit() {
    let n = 100;
    let f = LandmarkFrame::from_columns(
        vec![vec![1.0; n]],
        vec![None],
        vec![7.0; n],
        Head::Regression,
        (0..n as u64).collect(),
        None,
    )
    .unwrap();
    let s = signature(&f, &options()).unwrap();
    assert!(s.contrasts.iter().all(Option::is_none));
    assert_eq!(s.cells[0].state, CellState::Uninformative);
    let mut o = options();
    o.work_budget = 1;
    assert!(signature(&f, &o).is_err());
    o.work_budget = 100_000;
    let s = signature(&planted(100), &o).unwrap();
    assert_eq!(s.cells[22].state, CellState::OverBudget);
    assert!(s.work.estimated_units <= o.work_budget);
    assert!(s.cells[0].gain.is_some());
    o.panel = "landmarks-v1".into();
    assert!(signature(&f, &o).is_err());
}

#[test]
fn binary_and_multiclass_brier_scores_preserve_class_columns() {
    for k in [2, 3] {
        let n = 150;
        let x: Vec<f64> = (0..n).map(|i| (i % k) as f64).collect();
        let head = if k == 2 {
            Head::Binary
        } else {
            Head::Multiclass(k)
        };
        let frame = LandmarkFrame::from_columns(
            vec![x.clone()],
            vec![Some(k)],
            x,
            head,
            (0..n as u64).collect(),
            None,
        )
        .unwrap();
        let s = signature(&frame, &options()).unwrap();
        assert!(s.cells[0].gain.unwrap() > 0.95);
        assert_eq!(s.classes, k);
    }
}

#[test]
fn invalid_shapes_ids_categories_and_options_do_not_panic_or_echo_canaries() {
    assert!(LandmarkFrame::from_columns(
        vec![vec![1.0; 20]],
        vec![None],
        vec![0.0; 20],
        Head::Regression,
        vec![0; 20],
        None
    )
    .is_err());
    assert!(LandmarkFrame::from_columns(
        vec![vec![3.0; 20]],
        vec![Some(2)],
        vec![0.0; 20],
        Head::Binary,
        (0..20).collect(),
        None
    )
    .is_err());
    let mut o = options();
    o.folds = 0;
    assert!(signature(&planted(100), &o).is_err());
    o = options();
    o.knn_reference = 0;
    assert!(signature(&planted(100), &o).is_err());
}

#[test]
fn transfer_reuses_reference_refuses_overlap_and_paired_self_difference_is_zero() {
    let raw = table();
    let schema = infer(
        &raw.select_rows(&(0..120).collect::<Vec<_>>()),
        2,
        &IntakeOptions::default(),
    )
    .unwrap();
    let p =
        dope_core::eval::schema_program(&schema, &raw.select_rows(&(0..120).collect::<Vec<_>>()))
            .unwrap();
    let data = schema.to_model(&raw).unwrap();
    let train: Vec<usize> = (0..120).collect();
    let query: Vec<usize> = (120..160).collect();
    let r = TransferReference::new(&p, &data, &train, &query, &options(), false).unwrap();
    assert!(TransferReference::new(&p, &data, &train, &[1, 2], &options(), false).is_err());
    let s = Samples {
        x: data.x.iter().map(|c| c[..120].to_vec()).collect(),
        y: data.y[..120].to_vec(),
    };
    let a = r.score(&s).unwrap();
    let report = a.private_report();
    assert!(report.get("per_row_losses").is_none());
    assert!((a.mirror_retention.unwrap() - 1.0).abs() < 1e-10);
    let paired = r.compare(&a, &a, 100, 99).unwrap();
    assert_eq!(paired.mirror_delta_95, Some([0.0, 0.0]));
    assert!(r.compare(&a, &a, 0, 99).is_err());
    let mut changed = data.clone();
    for y in &mut changed.y[160..] {
        *y = 900.0;
    }
    let rr = TransferReference::new(&p, &changed, &train, &query, &options(), false).unwrap();
    assert_eq!(a.private_report(), rr.score(&s).unwrap().private_report());
}

#[test]
fn overlap_guard_normalizes_numbers_and_header_order() {
    let a = RawTable::from_reader("x,y\n1,2\n3,4\n".as_bytes()).unwrap();
    let b = RawTable::from_reader("y,x\n2.0,1.00\n".as_bytes()).unwrap();
    assert!(ensure_disjoint(&a, &b).is_err());
    let b = RawTable::from_reader("x,y\n5,6\n".as_bytes()).unwrap();
    assert!(ensure_disjoint(&a, &b).is_ok());
}

#[test]
fn low_release_is_identical_when_computing_higher_levels_and_allowlist_is_exact() {
    let mut o = LevelOptions {
        compute_through: Level::Low,
        export_levels: vec![Level::Low],
        fit: FitOptions {
            trees: false,
            audit: false,
            max_steps: 0,
            plan: Some((3, 2, 0)),
            ..Default::default()
        },
        landmarks: options(),
        ..Default::default()
    };
    let a = fit_levels(&table(), 2, &o).unwrap();
    o.compute_through = Level::High;
    let b = fit_levels(&table(), 2, &o).unwrap();
    assert_eq!(a.outputs[0].bytes, b.outputs[0].bytes);
    assert_eq!(b.outputs.len(), 1);
    assert!(b.outputs[0].bytes.len() <= Level::Low.cap());
    o.export_levels = vec![Level::Medium];
    let c = fit_levels(&table(), 2, &o).unwrap();
    assert_eq!(c.outputs.len(), 1);
    assert_eq!(c.outputs[0].level, Level::Medium);
    assert_eq!(
        Level::parse_list("all").unwrap(),
        vec![Level::Low, Level::Medium, Level::High]
    );
    assert!(Level::parse_list("low,small").is_err());
    o.compute_through = Level::Low;
    assert!(fit_levels(&table(), 2, &o).is_err());
}

#[test]
fn frozen_layout_has_contiguous_blocks_and_exact_width() {
    let layout: serde_json::Value = serde_json::from_str(dope_core::landmark::LAYOUT_JSON).unwrap();
    assert_eq!(layout["panel"], PANEL_V1);
    let mut end = 0;
    for b in layout["blocks"].as_array().unwrap() {
        assert_eq!(b["offset"].as_u64().unwrap(), end);
        end += b["length"].as_u64().unwrap();
    }
    assert_eq!(end, 96);
}

#[test]
fn coordinate_reader_refuses_recipe_or_panel_drift() {
    let opts = options();
    let s = signature(&planted(100), &opts).unwrap();
    let bytes = serde_json::to_vec(&s.global_coordinates()).unwrap();
    assert!(dope_core::landmark::GlobalCoordinates::decode(&bytes, &opts).is_ok());
    let mut v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    v["panel"] = serde_json::json!("landmarks-native-v2");
    assert!(dope_core::landmark::GlobalCoordinates::decode(
        &serde_json::to_vec(&v).unwrap(),
        &opts
    )
    .is_err());
    let mut other = opts.clone();
    other.seed += 1;
    assert!(dope_core::landmark::GlobalCoordinates::decode(&bytes, &other).is_err());
}

#[test]
fn sample_remapping_preserves_sub_micro_units_without_csv_rounding() {
    let raw = table();
    let schema = infer(&raw, 2, &IntakeOptions::default()).unwrap();
    let mut source = dope_core::eval::schema_program(&schema, &raw).unwrap();
    let reference = source.clone();
    if let dope_core::schema::Kind::Numeric(s) = &mut source.schema.target.kind {
        s.lo = 0.0;
        s.hi = 1e-8;
        s.log = false;
        s.integer = false;
    }
    let samples = Samples {
        x: vec![vec![0.25], vec![0.75]],
        y: vec![0.123456789],
    };
    let mapped = dope_core::levels::remap_samples(&source, &samples, &reference).unwrap();
    let target = match &reference.schema.target.kind {
        dope_core::schema::Kind::Numeric(s) => s,
        _ => unreachable!(),
    };
    let expected = target.to_model(0.123456789e-8);
    assert!((mapped.y[0] - expected).abs() < 1e-15);
    let roundtrip = match &reference.schema.target.kind {
        dope_core::schema::Kind::Numeric(s) => s.lo + mapped.y[0] * (s.hi - s.lo),
        _ => unreachable!(),
    };
    assert!((roundtrip - 0.123456789e-8).abs() < 1e-14);
}

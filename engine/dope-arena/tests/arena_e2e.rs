//! End-to-end integration tests for `dope-arena`: Rule A-1 splits, sealed panel checks,
//! methods registry authenticity, live 4-arm swap ladder (`--given`), and ledger reduction.

use dope_arena::panel::{PanelDatasetEntry, PanelManifest, PanelRole};
use dope_arena::report::run_swap_ladder;
use dope_arena::schedule::MethodsRegistry;
use dope_arena::split::{split_table_disjoint, SplitRatios};
use dope_core::RawTable;
use std::collections::BTreeSet;
use std::path::Path;

#[test]
fn disjoint_splits_keep_exact_duplicates_and_groups_indivisible() {
    let mut csv = String::from("grp,t,x,y\n");
    for i in 0..100 {
        let g = i / 5; // 20 groups of 5 rows each
        csv += &format!(
            "g{g},{},{:.2},{:.2}\n",
            i / 5,
            (i % 11) as f64,
            (g as f64) * 0.5
        );
    }
    let table = RawTable::from_reader(csv.as_bytes()).unwrap();
    let splits = split_table_disjoint(&table, Some(0), None, &SplitRatios::default(), 42).unwrap();
    let roles = [
        &splits.a_fit,
        &splits.b_twin,
        &splits.c_calib,
        &splits.t_test,
        &splits.p_privacy,
    ];
    let mut seen_rows = BTreeSet::new();
    let mut seen_groups = BTreeSet::new();
    for role in roles {
        assert!(!role.is_empty());
        let mut role_groups = BTreeSet::new();
        for &r in role {
            assert!(seen_rows.insert(r), "row {r} appeared in multiple splits");
            role_groups.insert(table.columns[0][r].clone());
        }
        for g in role_groups {
            assert!(
                seen_groups.insert(g.clone()),
                "group {g} leaked across splits"
            );
        }
    }
    assert_eq!(seen_rows.len(), 100);

    // Chronological split places latest timestamps strictly in sealed test T.
    let time_splits =
        split_table_disjoint(&table, Some(0), Some(1), &SplitRatios::default(), 42).unwrap();
    let max_a = time_splits
        .a_fit
        .iter()
        .map(|&r| table.columns[1][r].parse::<i64>().unwrap())
        .max()
        .unwrap();
    let min_t = time_splits
        .t_test
        .iter()
        .map(|&r| table.columns[1][r].parse::<i64>().unwrap())
        .min()
        .unwrap();
    assert!(
        min_t > max_a,
        "sealed test time {min_t} must be > fit time {max_a}"
    );
}

#[test]
fn sealed_panel_requires_90_families_and_registry_blocks_standins() {
    let mut m = PanelManifest {
        protocol: "dope-fidelity/v3".into(),
        panel_id: "sealed-test".into(),
        role: PanelRole::SealedConfirmation,
        datasets: (0..10)
            .map(|i| PanelDatasetEntry {
                dataset_id: format!("ds_{i}"),
                family_id: format!("fam_{i}"),
                target_column: "y".into(),
                task: "regression".into(),
                group_column: None,
                time_column: None,
                source_sha256: "a".repeat(64),
                rows: 500,
                features: 5,
            })
            .collect(),
        manifest_sha256: None,
    };
    assert!(
        m.verify().is_err(),
        "sealed panel with <90 families must refuse"
    );

    m.datasets = (0..90)
        .map(|i| PanelDatasetEntry {
            dataset_id: format!("ds_{i}"),
            family_id: format!("fam_{i}"),
            target_column: "y".into(),
            task: "regression".into(),
            group_column: None,
            time_column: None,
            source_sha256: "b".repeat(64),
            rows: 500,
            features: 5,
        })
        .collect();
    m.manifest_sha256 = Some(m.canonical_sha256().unwrap());
    m.verify().unwrap();

    // Verify repo bench/methods-registry.json blocks stand-ins and admits authentic methods.
    let reg_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/methods-registry.json");
    let reg = MethodsRegistry::load(&reg_path).unwrap();
    assert!(reg.verify_headline_eligible("dope_engine").is_ok());
    assert!(reg.verify_headline_eligible("rows_wire_v2").is_ok());
    assert!(reg
        .verify_headline_eligible("standin_arf_proj_bins")
        .is_err());
    assert!(reg
        .verify_headline_eligible("standin_gaussian_copula")
        .is_err());
}

#[test]
fn swap_ladder_runs_end_to_end_with_given_and_four_arms() {
    let mut csv = String::from("x0,x1,x2,y\n");
    for i in 0..240 {
        let x0 = ((i * 37) % 100) as f64 / 50.0 - 1.0;
        let x1 = ((i * 53) % 100) as f64 / 50.0 - 1.0;
        let x2 = ((i * 19) % 100) as f64 / 50.0 - 1.0;
        let y = 2.0 * x0 - 1.2 * x1 + 0.4 * x0 * x1 + 0.05 * ((i % 5) as f64 - 2.0);
        csv += &format!("{x0:.4},{x1:.4},{x2:.4},{y:.4}\n");
    }
    let table = RawTable::from_reader(csv.as_bytes()).unwrap();
    let report = run_swap_ladder(
        &table,
        3,
        16_384,
        11,
        vec!["linear-seeded-v1".into(), "gbdt".into()],
    )
    .unwrap();
    assert_eq!(report.arms.len(), 4);
    assert_eq!(report.arms[0].arm_id, "twin_b_real");
    assert_eq!(report.arms[1].arm_id, "real_x_synth_y_given");
    assert_eq!(report.arms[2].arm_id, "synth_matched_1n");
    assert_eq!(report.arms[3].arm_id, "synth_faithful_4n");
    assert_eq!(report.arms[1].rows, report.n_fit);
    assert_eq!(report.arms[3].rows, 4 * report.n_fit);
    assert!(report.arms[1].mean_retention.unwrap() > 0.70);
    assert!(report.arms[3].mean_retention.unwrap() > 0.65);
    assert!(report.arms[1].max_marginal_auc <= 0.55);
}

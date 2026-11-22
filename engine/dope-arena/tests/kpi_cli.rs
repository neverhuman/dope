//! Black-box tests of `dope_kpi build-panels` and `dope_kpi plan` on a
//! synthetic 43-table arena (see `kpi_support`).

mod kpi_support;

use kpi_support::fixture::{write_arena, FixtureTable};
use kpi_support::{
    by_hash, corpus, hash_of, json_file, json_lines, kpi_ok, path_str, scratch, usizes,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const PANEL_ID: &str = "dev-fixture";

struct Built {
    tables: Vec<FixtureTable>,
    arena: PathBuf,
    out: PathBuf,
    summary: String,
}

impl Built {
    fn panel(&self) -> PathBuf {
        self.out.join(format!("{PANEL_ID}.panel.json"))
    }

    fn families(&self) -> PathBuf {
        self.out.join("families.jsonl")
    }

    /// Run `plan` into `<out>/<name>` and return (schedule path, stdout).
    fn plan(&self, name: &str, extra: &[&str]) -> (PathBuf, String) {
        let schedule = self.out.join(name);
        let (panel, families) = (self.panel(), self.families());
        let mut args = vec![
            "plan",
            "--arena",
            path_str(&self.arena),
            "--panel",
            path_str(&panel),
            "--families",
            path_str(&families),
            "--out",
            path_str(&schedule),
        ];
        args.extend_from_slice(extra);
        let stdout = kpi_ok(&args);
        (schedule, stdout)
    }
}

fn build(name: &str, threads: &str) -> Built {
    let dir = scratch(name);
    let (arena, out) = (dir.join("arena"), dir.join("out"));
    let tables = corpus();
    write_arena(&arena, &tables).unwrap();
    let summary = kpi_ok(&[
        "build-panels",
        "--arena",
        path_str(&arena),
        "--out",
        path_str(&out),
        "--panel-id",
        PANEL_ID,
        "--threads",
        threads,
        "--verify-checksums",
    ]);
    Built {
        tables,
        arena,
        out,
        summary,
    }
}

fn text(value: &Value) -> &str {
    value.as_str().expect("string field")
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap()
}

#[test]
fn panel_excludes_every_family_with_a_training_table() {
    let built = build("panel", "2");
    let t = &built.tables;
    assert_eq!(t.len(), 43);
    let families = json_lines(&built.families());
    assert_eq!(families.len(), 43);
    let family_of: BTreeMap<&str, &str> = families
        .iter()
        .map(|row| (text(&row["dataset_hash"]), text(&row["family_id"])))
        .collect();
    let evidence_of: BTreeMap<&str, &str> = families
        .iter()
        .map(|row| (text(&row["dataset_hash"]), text(&row["evidence"])))
        .collect();
    // One row per table, in arena order, with the arena's split and task.
    for (row, table) in families.iter().zip(t) {
        assert_eq!(text(&row["dataset_hash"]), table.dataset_hash);
        assert_eq!(text(&row["split"]), table.split);
        assert_eq!(text(&row["task"]), table.task);
        assert!(text(&row["family_id"]).starts_with("lf-"));
    }

    let panel = json_file(&built.panel());
    assert_eq!(text(&panel["protocol"]), "dope-fidelity/v3");
    assert_eq!(text(&panel["panel_id"]), PANEL_ID);
    assert_eq!(text(&panel["role"]), "development");
    let datasets = panel["datasets"].as_array().unwrap();
    let panel_hashes: BTreeSet<&str> = datasets.iter().map(|d| text(&d["dataset_id"])).collect();
    let tables = by_hash(t);
    for entry in datasets {
        let (_, table) = tables[text(&entry["dataset_id"])];
        assert_eq!(table.split, "val");
        assert_eq!(
            text(&entry["family_id"]),
            family_of[table.dataset_hash.as_str()]
        );
        assert_eq!(text(&entry["target_column"]), "__target__");
        assert_eq!(text(&entry["source_sha256"]), table.canonical_sha256);
        assert_eq!(text(&entry["task"]), table.task);
        assert_eq!(entry["rows"].as_u64().unwrap() as usize, table.n_rows());
        assert_eq!(entry["features"].as_u64().unwrap() as usize, table.n_cols);
    }

    // Zero families shared between the training split and the panel.
    let train_families: BTreeSet<&str> = t
        .iter()
        .filter(|table| table.split == "train")
        .map(|table| family_of[table.dataset_hash.as_str()])
        .collect();
    let panel_families: BTreeSet<&str> = panel_hashes.iter().map(|h| family_of[h]).collect();
    assert_eq!(train_families.intersection(&panel_families).count(), 0);

    // Exactly the six validation tables with a training sibling are left out.
    let val = |stem: &str, nth: usize| hash_of(t, stem, "val", nth);
    let train = |stem: &str| hash_of(t, stem, "train", 0);
    let mut expected_out = vec![
        val("alpha_target_b", 0),
        val("beta_seed_7", 0),
        val("beta_2", 0),
        val("kappa_test", 0),
        val("renamed_copy", 0),
        val("rows", 0),
    ];
    expected_out.sort();
    let all_val: BTreeSet<&str> = t
        .iter()
        .filter(|table| table.split == "val")
        .map(|table| table.dataset_hash.as_str())
        .collect();
    let left_out: Vec<&str> = all_val.difference(&panel_hashes).copied().collect();
    assert_eq!(left_out, expected_out);
    assert_eq!((all_val.len(), panel_hashes.len()), (28, 22));
    assert_eq!(panel_families.len(), 20);

    // Name siblings, content siblings and clean generic stems land where expected.
    let same = |a: &str, b: &str| assert_eq!(family_of[a], family_of[b]);
    same(&train("alpha_target_a"), &val("alpha_target_b", 0));
    same(&train("beta_seed_3"), &val("beta_seed_7", 0));
    same(&train("beta_seed_3"), &val("beta_2", 0));
    same(&train("kappa_train"), &val("kappa_test", 0));
    same(&train("gamma_source"), &val("renamed_copy", 0));
    same(&train("export"), &val("rows", 0));
    same(&val("delta_fold1", 0), &val("delta_fold2", 0));
    same(&val("omega_full", 0), &val("sigma_part", 0));
    assert_eq!(evidence_of[val("beta_2", 0).as_str()], "name");
    assert_eq!(evidence_of[val("renamed_copy", 0).as_str()], "content");
    assert_eq!(evidence_of[val("rows", 0).as_str()], "content");
    assert_eq!(evidence_of[val("sigma_part", 0).as_str()], "content");
    for generic in [
        val("rows", 1),
        val("export", 0),
        val("data", 0),
        train("rows"),
        train("unknown"),
    ] {
        assert_eq!(evidence_of[generic.as_str()], "singleton");
        assert_eq!(family_of[generic.as_str()], format!("lf-{generic}"));
    }
    let singletons = evidence_of.values().filter(|e| **e == "singleton").count();
    assert_eq!(singletons, 43 - (2 + 3 + 2 + 2 + 2 + 2 + 2));

    // The overlap report states the same counts and the excluded hashes.
    let report = json_file(&built.out.join("overlap_report.json"));
    assert_eq!(text(&report["schema"]), "dope.kpi-overlap/v1");
    assert_eq!(report["validation"]["tables"], 28);
    assert_eq!(report["panel"]["tables"], 22);
    assert_eq!(report["panel"]["families"], 20);
    assert_eq!(report["excluded"]["tables"], 6);
    assert_eq!(report["excluded"]["families"], 5);
    assert_eq!(report["excluded"]["tables_by_evidence"]["name"], 4);
    assert_eq!(report["excluded"]["tables_by_evidence"]["content"], 2);
    assert_eq!(report["excluded"]["families_by_evidence"]["name"], 3);
    assert_eq!(report["excluded"]["families_by_evidence"]["content"], 2);
    assert_eq!(report["excluded"]["tables_by_task"]["binary"], 2);
    assert_eq!(report["excluded"]["tables_by_task"]["regression"], 3);
    assert_eq!(report["excluded"]["tables_by_task"]["multiclass"], 1);
    let listed: Vec<&str> = report["excluded"]["dataset_hashes"]
        .as_array()
        .unwrap()
        .iter()
        .map(text)
        .collect();
    assert_eq!(listed, expected_out);
    assert_eq!(report["tables_by_split"]["train"], 15);
    assert_eq!(report["arena"]["checksums_verified"], true);
    assert_eq!(report["content"]["linked_table_pairs"], 3);

    let summary: Value = serde_json::from_str(&built.summary).unwrap();
    assert_eq!(summary["panel_tables"], 22);
    assert_eq!(summary["excluded_tables"], 6);
    assert_eq!(summary["panel_sha256"], panel["manifest_sha256"]);
    assert_eq!(summary["panel_sha256"], report["panel_sha256"]);
}

/// Structural checks of one supported cell against the fixture's duplicate groups.
fn check_cell(cell: &Value, table: &FixtureTable) -> (usize, usize, usize) {
    let roles: Vec<Vec<usize>> = ["a_idx", "b_idx", "c_idx", "t_idx"]
        .iter()
        .map(|key| usizes(&cell[*key]))
        .collect();
    let mut owner: BTreeMap<i32, usize> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for (role, rows) in roles.iter().enumerate() {
        assert!(rows.windows(2).all(|w| w[0] < w[1]), "rows must ascend");
        for &row in rows {
            assert!(row < table.n_rows(), "row index outside the table");
            assert!(seen.insert(row), "row {row} appears in two roles");
            let held = *owner.entry(table.dup[row]).or_insert(role);
            assert_eq!(held, role, "a duplicate group straddles two roles");
        }
    }
    assert_eq!(roles[0].len(), roles[1].len(), "|A| must equal |B|");
    assert!(roles[0].len() >= 16 && roles[3].len() >= 16);
    assert!(roles[0].len() <= 1024 && roles[2].len() <= roles[0].len());
    let groups = |rows: &[usize]| {
        rows.iter()
            .map(|&r| table.dup[r])
            .collect::<BTreeSet<_>>()
            .len()
    };
    let headline = groups(&roles[0]).min(groups(&roles[1])) >= 200 && groups(&roles[3]) >= 200;
    assert_eq!(text(&cell["stratum"]) == "headline", headline);
    (roles[0].len(), roles[2].len(), roles[3].len())
}

#[test]
fn schedule_cells_are_complete_disjoint_and_group_safe() {
    let built = build("plan", "1");
    let t = &built.tables;
    let tables = by_hash(t);
    let panel = json_file(&built.panel());
    let panel_sha = text(&panel["manifest_sha256"]).to_string();
    let family_of: BTreeMap<String, String> = panel["datasets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| {
            (
                text(&d["dataset_id"]).to_string(),
                text(&d["family_id"]).to_string(),
            )
        })
        .collect();

    let (schedule, stdout) = built.plan("full.jsonl", &["--tier", "full"]);
    let cells = json_lines(&schedule);
    let summary: Value = serde_json::from_str(&stdout).unwrap();

    // 22 panel tables x 2 data seeds, plus a p48 view for the two wide tables.
    assert_eq!(cells.len(), 22 * 2 + 2 * 2);
    let ids: Vec<&str> = cells.iter().map(|c| text(&c["cell_id"])).collect();
    assert!(
        ids.windows(2).all(|w| w[0] < w[1]),
        "cells must be sorted by cell_id"
    );

    let (mut supported, mut unsupported, mut headline) = (0, 0, 0);
    let mut sizes: BTreeMap<(String, u64), (usize, usize, usize)> = BTreeMap::new();
    for cell in &cells {
        let hash = text(&cell["dataset_hash"]);
        let (table_id, table) = tables[hash];
        let (view, data_seed) = (text(&cell["view"]), cell["data_seed"].as_u64().unwrap());
        assert_eq!(
            text(&cell["cell_id"]),
            format!("{PANEL_ID}:{hash}:{view}:d{data_seed}")
        );
        assert_eq!(text(&cell["panel_sha256"]), panel_sha);
        assert_eq!(cell["table_id"].as_u64().unwrap() as usize, table_id);
        assert_eq!(text(&cell["family_id"]), family_of[hash]);
        assert_eq!(text(&cell["task"]), table.task);
        assert_eq!(cell["n_classes"].as_u64().unwrap() as u32, table.n_classes);
        assert_eq!(usizes(&cell["sample_seeds"]), [0, 1]);
        assert_eq!(usizes(&cell["multipliers"]), [4, 1]);
        assert!(data_seed < 2);

        let cols = usizes(&cell["cols"]);
        assert!(cols.windows(2).all(|w| w[0] < w[1]) && cols.iter().all(|&c| c < table.n_cols));
        match view {
            "full" => assert_eq!(cols, (0..table.n_cols).collect::<Vec<_>>()),
            "p48" => {
                assert!(table.n_cols > 48, "p48 only exists for wider tables");
                assert_eq!(cols.len(), 48);
            }
            other => panic!("unexpected view {other}"),
        }

        match text(&cell["status"]) {
            "ok" => {
                supported += 1;
                headline += usize::from(text(&cell["stratum"]) == "headline");
                let size = check_cell(cell, table);
                // Both views of a table share the row split of their data seed.
                let first = *sizes.entry((hash.to_string(), data_seed)).or_insert(size);
                assert_eq!(first, size);
            }
            "unsupported" => {
                unsupported += 1;
                assert!(table.n_rows() <= 40, "only the tiny tables are unsupported");
                assert_eq!(text(&cell["stratum"]), "small_n");
                for key in ["a_idx", "b_idx", "c_idx", "t_idx"] {
                    assert!(usizes(&cell[key]).is_empty());
                }
            }
            other => panic!("unexpected status {other}"),
        }
    }
    // Denominator: scheduled cells = supported + unsupported.
    assert_eq!(cells.len(), supported + unsupported);
    assert_eq!((supported, unsupported), (44, 4));
    assert_eq!(summary["cells"], 48);
    assert_eq!(summary["supported"], 44);
    assert_eq!(summary["unsupported"], 4);
    assert_eq!(
        summary["supported_headline"].as_u64().unwrap() as usize,
        headline
    );
    assert_eq!(
        summary["supported_small_n"].as_u64().unwrap() as usize,
        supported - headline
    );
    assert_eq!(summary["cells_by_view"]["p48"], 4);
    assert_eq!(
        (summary["tables"].clone(), summary["families"].clone()),
        (22.into(), 20.into())
    );

    // Role sizes follow formats.md section 4 (rows == groups without duplicates).
    let size_of = |stem: &str, seed: u64| sizes[&(hash_of(t, stem, "val", 0), seed)];
    for seed in [0, 1] {
        assert_eq!(size_of("big_table", seed), (1024, 52, 900));
        assert_eq!(size_of("headline_a", seed), (280, 0, 240));
        assert_eq!(size_of("headline_b", seed), (220, 0, 200));
        assert_eq!(size_of("wide_big", seed), (245, 0, 210));
        assert_eq!(size_of("small_c", seed), (21, 0, 18));
        assert_eq!(size_of("willow", seed), (18, 0, 16));
        assert_eq!(size_of("wide_sensor", seed), (52, 1, 45));
    }
    // big_table, headline_a, headline_b and wide_big (two views) are headline cells.
    assert_eq!(headline, (3 + 2) * 2);
    // Different data seeds give different splits and different column views.
    let cell = |id: String| cells.iter().find(|c| text(&c["cell_id"]) == id).unwrap();
    let wide = hash_of(t, "wide_big", "val", 0);
    let (d0, d1) = (
        cell(format!("{PANEL_ID}:{wide}:p48:d0")),
        cell(format!("{PANEL_ID}:{wide}:p48:d1")),
    );
    assert_ne!(d0["t_idx"], d1["t_idx"]);
    assert_ne!(d0["cols"], d1["cols"]);
    assert_eq!(
        d0["t_idx"],
        cell(format!("{PANEL_ID}:{wide}:full:d0"))["t_idx"]
    );

    // Changing the sample seed list leaves every split and view unchanged.
    let (reseeded, _) = built.plan(
        "reseeded.jsonl",
        &["--tier", "full", "--sample-seeds", "11,12,13"],
    );
    let other = json_lines(&reseeded);
    assert_eq!(other.len(), cells.len());
    for (a, b) in cells.iter().zip(&other) {
        assert_eq!(usizes(&b["sample_seeds"]), [11, 12, 13]);
        let (mut a, mut b) = (a.clone(), b.clone());
        a.as_object_mut().unwrap().remove("sample_seeds");
        b.as_object_mut().unwrap().remove("sample_seeds");
        assert_eq!(a, b);
    }

    // The lite tier scores one table per family (most rows) with one seed each.
    let (lite, lite_out) = built.plan("lite.jsonl", &["--tier", "lite"]);
    let lite_cells = json_lines(&lite);
    assert_eq!(lite_cells.len(), 20 + 2);
    let lite_tables: BTreeSet<&str> = lite_cells
        .iter()
        .map(|c| text(&c["dataset_hash"]))
        .collect();
    assert_eq!(lite_tables.len(), 20);
    assert!(lite_tables.contains(hash_of(t, "delta_fold2", "val", 0).as_str()));
    assert!(!lite_tables.contains(hash_of(t, "delta_fold1", "val", 0).as_str()));
    assert!(lite_tables.contains(hash_of(t, "omega_full", "val", 0).as_str()));
    assert!(lite_cells
        .iter()
        .all(|c| c["data_seed"] == 0 && usizes(&c["sample_seeds"]) == [0]));
    // A lite cell is the same cell as in the full tier, apart from its sample seeds.
    for lite_cell in &lite_cells {
        let full_cell = cell(text(&lite_cell["cell_id"]).to_string());
        assert_eq!(lite_cell["a_idx"], full_cell["a_idx"]);
        assert_eq!(lite_cell["t_idx"], full_cell["t_idx"]);
        assert_eq!(lite_cell["cols"], full_cell["cols"]);
    }
    let lite_summary: Value = serde_json::from_str(&lite_out).unwrap();
    assert_eq!(lite_summary["tier"], "lite");
    assert_eq!(lite_summary["unsupported"], 2);

    for (tier, cells) in [("confirm", 72), ("sealed", 72)] {
        let (_, out) = built.plan(&format!("{tier}.jsonl"), &["--tier", tier]);
        let summary: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(summary["cells"], cells);
        assert_eq!(summary["data_seeds"], serde_json::json!([0, 1, 2]));
        let samples = if tier == "sealed" { 3 } else { 2 };
        assert_eq!(summary["sample_seeds"].as_array().unwrap().len(), samples);
    }
}

#[test]
fn outputs_are_byte_identical_across_runs_and_thread_counts() {
    let first = build("repeat-a", "1");
    let second = build("repeat-b", "1");
    let threaded = build("repeat-c", "4");
    assert_eq!(first.summary, second.summary);
    assert_eq!(first.summary, threaded.summary);
    for name in [
        "families.jsonl",
        "overlap_report.json",
        &format!("{PANEL_ID}.panel.json"),
    ] {
        let bytes = read(&first.out.join(name));
        assert!(!bytes.is_empty());
        assert_eq!(
            bytes,
            read(&second.out.join(name)),
            "{name} differs between runs"
        );
        assert_eq!(
            bytes,
            read(&threaded.out.join(name)),
            "{name} differs with --threads 4"
        );
    }
    for tier in ["lite", "full", "sealed"] {
        let (a, a_out) = first.plan("schedule.jsonl", &["--tier", tier, "--threads", "1"]);
        let (b, b_out) = second.plan("schedule.jsonl", &["--tier", tier, "--threads", "1"]);
        let (c, c_out) = threaded.plan("schedule.jsonl", &["--tier", tier, "--threads", "4"]);
        assert_eq!(read(&a), read(&b), "{tier} schedule differs between runs");
        assert_eq!(
            read(&a),
            read(&c),
            "{tier} schedule differs with --threads 4"
        );
        assert_eq!(a_out, b_out);
        assert_eq!(a_out, c_out);
        assert!(!a.with_file_name("schedule.jsonl.staging").exists());
    }
}

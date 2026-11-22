//! Black-box failure paths of `dope_kpi`: inputs that would leak training
//! lineage into a panel, corrupt arenas and malformed arguments are refused
//! with a non-zero exit and no summary.

mod kpi_support;

use kpi_support::fixture::{write_arena, FixtureTable};
use kpi_support::{corpus, hash_of, json_file, json_lines, kpi_err, kpi_ok, path_str, scratch};
use std::path::{Path, PathBuf};

const PANEL_ID: &str = "dev-fixture";

struct Paths {
    arena: PathBuf,
    out: PathBuf,
    panel: PathBuf,
    families: PathBuf,
    schedule: PathBuf,
}

fn built(name: &str, tables: &[FixtureTable]) -> Paths {
    let dir = scratch(name);
    let paths = Paths {
        arena: dir.join("arena"),
        out: dir.join("out"),
        panel: dir.join("out").join(format!("{PANEL_ID}.panel.json")),
        families: dir.join("out").join("families.jsonl"),
        schedule: dir.join("out").join("schedule.jsonl"),
    };
    write_arena(&paths.arena, tables).unwrap();
    paths
}

fn build_args(paths: &Paths) -> Vec<&str> {
    vec![
        "build-panels",
        "--arena",
        path_str(&paths.arena),
        "--out",
        path_str(&paths.out),
        "--panel-id",
        PANEL_ID,
    ]
}

fn plan_args<'a>(paths: &'a Paths, families: &'a Path, panel: &'a Path) -> Vec<&'a str> {
    vec![
        "plan",
        "--arena",
        path_str(&paths.arena),
        "--panel",
        path_str(panel),
        "--families",
        path_str(families),
        "--tier",
        "lite",
        "--out",
        path_str(&paths.schedule),
    ]
}

#[test]
fn plan_refuses_lineage_that_reaches_the_training_split() {
    let tables = corpus();
    let paths = built("guard-lineage", &tables);
    kpi_ok(&build_args(&paths));
    kpi_ok(&plan_args(&paths, &paths.families, &paths.panel));
    std::fs::remove_file(&paths.schedule).unwrap();

    // Re-label a training table into the family of a panel table.
    let panel = json_file(&paths.panel);
    let panel_family = panel["datasets"][0]["family_id"].as_str().unwrap();
    let train_hash = hash_of(&tables, "orchard", "train", 0);
    let edited: String = json_lines(&paths.families)
        .into_iter()
        .map(|mut row| {
            if row["dataset_hash"] == train_hash.as_str() {
                row["family_id"] = panel_family.into();
            }
            format!("{row}\n")
        })
        .collect();
    let leaky = paths.out.join("families-leaky.jsonl");
    std::fs::write(&leaky, edited).unwrap();
    let err = kpi_err(&plan_args(&paths, &leaky, &paths.panel));
    assert!(err.contains("spans the training split"), "{err}");
    assert!(
        !paths.schedule.exists(),
        "a refused plan must not write a schedule"
    );

    // A families file from another arena (one row short) is refused too.
    let short: String = std::fs::read_to_string(&paths.families)
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| format!("{line}\n"))
        .collect();
    let partial = paths.out.join("families-partial.jsonl");
    std::fs::write(&partial, short).unwrap();
    let err = kpi_err(&plan_args(&paths, &partial, &paths.panel));
    assert!(err.contains("does not cover the arena"), "{err}");

    // A panel edited after it was committed no longer matches its SHA-256.
    let text = std::fs::read_to_string(&paths.panel).unwrap();
    let train_canonical = &tables
        .iter()
        .find(|t| t.dataset_hash == train_hash)
        .unwrap()
        .canonical_sha256;
    let first_pin = panel["datasets"][0]["source_sha256"].as_str().unwrap();
    let forged = paths.out.join("forged.panel.json");
    std::fs::write(&forged, text.replace(first_pin, train_canonical)).unwrap();
    let err = kpi_err(&plan_args(&paths, &paths.families, &forged));
    assert!(err.contains("commitment mismatch"), "{err}");
    assert!(!paths.schedule.exists());
}

#[test]
fn corrupt_arenas_and_bad_arguments_are_refused() {
    let tables = corpus();
    let paths = built("guard-arena", &tables);
    kpi_ok(&build_args(&paths));

    let mut bad_seeds = plan_args(&paths, &paths.families, &paths.panel);
    bad_seeds.extend_from_slice(&["--data-seeds", "3,3"]);
    assert!(kpi_err(&bad_seeds).contains("--data-seeds"));

    let mut bad_id = build_args(&paths);
    bad_id[6] = "bad:id";
    assert!(kpi_err(&bad_id).contains("panel id"));

    // A flipped payload byte is found by the checksum pass and only by it.
    let dup_path = paths.arena.join("dup.i32");
    let mut bytes = std::fs::read(&dup_path).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&dup_path, &bytes).unwrap();
    let mut checked = plan_args(&paths, &paths.families, &paths.panel);
    checked.push("--verify-checksums");
    let err = kpi_err(&checked);
    assert!(err.contains("dup.i32: checksum mismatch"), "{err}");
    let mut checked_build = build_args(&paths);
    checked_build.push("--verify-checksums");
    assert!(kpi_err(&checked_build).contains("checksum mismatch"));

    // A truncated payload fails the cheap open of every subcommand.
    std::fs::write(&dup_path, &bytes[..bytes.len() - 4]).unwrap();
    let err = kpi_err(&build_args(&paths));
    assert!(err.contains("dup.i32: file size disagrees"), "{err}");
    let err = kpi_err(&plan_args(&paths, &paths.families, &paths.panel));
    assert!(err.contains("dup.i32: file size disagrees"), "{err}");

    let missing = scratch("guard-missing");
    let gone = Paths {
        arena: missing.join("no-arena"),
        ..built("guard-missing-out", &tables[..2])
    };
    let err = kpi_err(&build_args(&gone));
    assert!(
        err.contains("META.json") && err.contains("not found"),
        "{err}"
    );
}

#[test]
fn a_panel_cannot_be_built_when_every_validation_table_has_a_training_sibling() {
    let tables: Vec<FixtureTable> = corpus()
        .into_iter()
        .filter(|t| t.family_stem.starts_with("alpha") || t.family_stem.starts_with("beta"))
        .collect();
    assert_eq!(tables.len(), 5);
    let paths = built("guard-empty", &tables);
    let err = kpi_err(&build_args(&paths));
    assert!(err.contains("the panel would be empty"), "{err}");
    assert!(!paths.families.exists() && !paths.panel.exists());
    // Error text never repeats table values or stems.
    assert!(!err.contains("alpha") && !err.contains("beta"), "{err}");
}

#[test]
fn content_share_decides_whether_a_wide_table_claims_a_narrow_one() {
    // A narrow validation table holding three of the 61 eligible columns of a
    // wide training table: all of the smaller table, a sliver of the wider one.
    let all = corpus();
    let pick = |stem: &str| all.iter().find(|t| t.family_stem == stem).unwrap().clone();
    let mut wide = pick("wide_sensor");
    wide.split = "train".into();
    let rows = wide.n_rows();
    let x: Vec<f32> = (0..rows)
        .flat_map(|r| wide.x[r * wide.n_cols..r * wide.n_cols + 3].iter().copied())
        .collect();
    let narrow = FixtureTable::new("00000000000000aa", "narrow_extract", "binary", "val", 3)
        .with_rows(x, (0..rows).map(|r| (r % 2) as f32).collect());
    let paths = built("guard-share", &[wide, narrow, pick("walnut")]);

    let summary = |extra: &[&str]| -> serde_json::Value {
        let mut args = build_args(&paths);
        args.extend_from_slice(extra);
        serde_json::from_str(&kpi_ok(&args)).unwrap()
    };
    let by_default = summary(&[]);
    assert_eq!(by_default, summary(&["--content-share", "smaller"]));
    assert_eq!(
        (
            by_default["panel_tables"].clone(),
            by_default["excluded_tables"].clone()
        ),
        (1.into(), 1.into())
    );
    let report = json_file(&paths.out.join("overlap_report.json"));
    assert_eq!(report["rule"]["content_share"], "smaller");
    assert_eq!(report["excluded"]["tables_by_evidence"]["content"], 1);

    let each = summary(&["--content-share", "each"]);
    assert_eq!(
        (
            each["panel_tables"].clone(),
            each["excluded_tables"].clone()
        ),
        (2.into(), 0.into())
    );
    let report = json_file(&paths.out.join("overlap_report.json"));
    assert_eq!(report["rule"]["content_share"], "each");
    assert_eq!(report["content"]["linked_table_pairs"], 0);
    assert_ne!(each["panel_sha256"], by_default["panel_sha256"]);
}

//! Shared fixture for the `dope_kpi` black-box tests: a 43-table synthetic
//! arena with training siblings, generic stems, renamed row subsets, wide
//! tables, duplicate groups and tables too small to score.
#![allow(dead_code)]

#[path = "../../src/kpi/fixture.rs"]
pub mod fixture;

use fixture::FixtureTable;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// SplitMix64 finaliser, used only to make fixture values and hashes.
pub fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// A table whose values sit on a grid private to `index`, so two tables share
/// almost no values. `levels` distinct grid points per column are available.
fn table(
    index: u64,
    stem: &str,
    task: &str,
    split: &str,
    rows: usize,
    cols: usize,
    levels: u64,
) -> FixtureTable {
    let step = 0.5 + 0.013 * index as f32;
    let x = (0..rows * cols)
        .map(|i| ((mix(index * 1_000_003 + i as u64) % levels) as f32 - 100.0) * step)
        .collect();
    let y = (0..rows as u64)
        .map(|r| match task {
            "regression" => (mix(index * 7919 + r) % 1000) as f32 * (0.25 + 0.01 * index as f32),
            "binary" => (r % 2) as f32,
            _ => (r % 3) as f32,
        })
        .collect();
    FixtureTable::new(&format!("{:016x}", mix(index + 1)), stem, task, split, cols).with_rows(x, y)
}

/// A renamed row subset of `source` (the rows for which `keep` holds).
fn subset(
    index: u64,
    stem: &str,
    split: &str,
    source: &FixtureTable,
    keep: fn(usize) -> bool,
) -> FixtureTable {
    let rows: Vec<usize> = (0..source.n_rows()).filter(|r| keep(*r)).collect();
    let cols = source.n_cols;
    let x = rows
        .iter()
        .flat_map(|&r| source.x[r * cols..(r + 1) * cols].iter().copied())
        .collect();
    let y = rows.iter().map(|&r| source.y[r]).collect();
    FixtureTable::new(
        &format!("{:016x}", mix(index + 1)),
        stem,
        &source.task,
        split,
        cols,
    )
    .with_rows(x, y)
}

/// The fixture corpus, in arena order.
pub fn corpus() -> Vec<FixtureTable> {
    let wide = 2001;
    let gamma = table(3, "gamma_source", "regression", "train", 600, 6, 201);
    let export = table(6, "export", "regression", "train", 400, 6, 201);
    let omega = table(35, "omega_full", "regression", "val", 400, 5, 201);
    let mut dup_heavy = table(26, "dup_heavy", "binary", "val", 900, 4, wide);
    // 300 duplicate groups of one to five rows, interleaved through the table.
    dup_heavy.dup = (0..5)
        .flat_map(|round| (0..300).filter(move |g| g % 5 >= round).map(|g| g * 3 - 50))
        .collect();
    vec![
        // Training split.
        table(1, "alpha_target_a", "regression", "train", 300, 5, wide),
        table(2, "beta_seed_3", "binary", "train", 300, 5, wide),
        gamma.clone(),
        table(4, "rows", "regression", "train", 200, 5, wide),
        table(5, "unknown", "binary", "train", 200, 5, wide),
        export.clone(),
        table(7, "kappa_train", "multiclass", "train", 300, 5, wide),
        table(8, "orchard", "regression", "train", 220, 5, wide),
        table(9, "harbor", "binary", "train", 220, 5, wide),
        table(10, "meadow", "multiclass", "train", 220, 5, wide),
        table(11, "quarry", "regression", "train", 220, 5, wide),
        table(12, "lagoon", "binary", "train", 220, 5, wide),
        table(13, "tundra", "regression", "train", 220, 5, wide),
        table(14, "canyon", "regression", "train", 220, 5, wide),
        table(15, "prairie", "binary", "train", 220, 5, wide),
        // Validation tables with a training sibling by name.
        table(16, "alpha_target_b", "regression", "val", 300, 5, wide),
        table(17, "beta_seed_7", "binary", "val", 300, 5, wide),
        table(18, "beta_2", "binary", "val", 300, 5, wide),
        table(19, "kappa_test", "multiclass", "val", 300, 5, wide),
        // Validation tables with a training sibling by content only.
        subset(20, "renamed_copy", "val", &gamma, |r| r % 2 == 0),
        subset(21, "rows", "val", &export, |r| r % 4 != 0),
        // Clean validation tables: generic stems with unrelated content.
        table(22, "rows", "regression", "val", 250, 5, wide),
        table(23, "export", "binary", "val", 250, 5, wide),
        table(24, "data", "multiclass", "val", 250, 5, wide),
        // Clean: wide tables, duplicate groups, tiny and boundary sizes.
        table(25, "wide_sensor", "regression", "val", 150, 60, wide),
        dup_heavy,
        table(27, "tiny_a", "regression", "val", 10, 4, wide),
        table(28, "tiny_b", "binary", "val", 40, 4, wide),
        table(29, "small_c", "regression", "val", 60, 4, wide),
        table(30, "headline_a", "regression", "val", 800, 5, wide),
        table(31, "headline_b", "binary", "val", 640, 5, wide),
        table(32, "big_table", "multiclass", "val", 3000, 4, wide),
        // Clean families with two validation tables: by name, then by content.
        table(33, "delta_fold1", "regression", "val", 300, 5, wide),
        table(34, "delta_fold2", "regression", "val", 500, 5, wide),
        omega.clone(),
        subset(36, "sigma_part", "val", &omega, |r| r % 5 < 3),
        table(37, "wide_big", "regression", "val", 700, 64, wide),
        table(38, "walnut", "regression", "val", 320, 5, wide),
        table(39, "cedar", "binary", "val", 280, 5, wide),
        table(40, "birch", "multiclass", "val", 260, 5, wide),
        table(41, "maple", "regression", "val", 90, 5, wide),
        table(42, "aspen", "binary", "val", 55, 5, wide),
        table(43, "willow", "multiclass", "val", 52, 5, wide),
    ]
}

/// `dataset_hash` of the first table with `stem` in `split`.
pub fn hash_of(tables: &[FixtureTable], stem: &str, split: &str, nth: usize) -> String {
    tables
        .iter()
        .filter(|t| t.family_stem == stem && t.split == split)
        .nth(nth)
        .unwrap_or_else(|| panic!("no fixture table {stem}/{split}/{nth}"))
        .dataset_hash
        .clone()
}

pub fn by_hash(tables: &[FixtureTable]) -> BTreeMap<String, (usize, &FixtureTable)> {
    tables
        .iter()
        .enumerate()
        .map(|(id, t)| (t.dataset_hash.clone(), (id, t)))
        .collect()
}

/// An empty scratch directory under the build tree.
pub fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("kpi-cli")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn run_kpi(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_dope_kpi"))
        .args(args)
        .output()
        .expect("run dope_kpi")
}

/// Run and require success; returns stdout.
pub fn kpi_ok(args: &[&str]) -> String {
    let output = run_kpi(args);
    assert!(
        output.status.success(),
        "dope_kpi {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

/// Run and require failure; returns stderr.
pub fn kpi_err(args: &[&str]) -> String {
    let output = run_kpi(args);
    assert!(
        !output.status.success(),
        "dope_kpi {args:?} unexpectedly succeeded"
    );
    assert!(
        output.stdout.is_empty(),
        "a failed run must not print a summary"
    );
    String::from_utf8(output.stderr).unwrap()
}

pub fn path_str(path: &Path) -> &str {
    path.to_str().expect("utf-8 path")
}

pub fn json_lines(path: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

pub fn json_file(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

pub fn usizes(value: &serde_json::Value) -> Vec<usize> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_u64().expect("unsigned integer") as usize)
        .collect()
}

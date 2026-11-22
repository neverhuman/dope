//! Shared fixture of the `dope_kpi emit-cli` black-box tests: a five-table
//! arena scheduled with the real `build-panels` and `plan`, and fake
//! generators written as shell scripts (only `sh` and coreutils are used).
#![allow(dead_code)]

use crate::kpi_support::fixture::{write_arena, FixtureTable};
use crate::kpi_support::{json_lines, kpi_ok, mix, path_str, scratch};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub const PANEL_ID: &str = "ext-fixture";

/// Feature values that stress the text form: missing cells, integer-valued
/// floats, negative zero, tiny and huge magnitudes and ordinary fractions,
/// on ranges private to each table so that no two tables look related.
fn feature(index: u64, position: u64) -> f32 {
    let hash = mix(index * 1_000_003 + position);
    let pick = (hash >> 8) % 20_000;
    match hash % 11 {
        0 => f32::NAN,
        1 => (pick + 50_000 * index) as f32,
        2 => -0.0,
        3 => (pick + 50_000 * index) as f32 * 1.0e-9,
        4 => (pick + 50_000 * index) as f32 * 3.0e7 + 0.5,
        _ => (pick as f32 - 10_000.0) * (0.013 + 0.0017 * index as f32),
    }
}

fn table(index: u64, stem: &str, task: &str, rows: usize, cols: usize) -> FixtureTable {
    let x = (0..(rows * cols) as u64)
        .map(|i| feature(index, i))
        .collect();
    let y = (0..rows as u64)
        .map(|r| match task {
            "regression" => (mix(index * 7919 + r) % 1000) as f32 * 0.25 - 100.0,
            "binary" => (mix(index * 7919 + r) % 2) as f32,
            _ => (mix(index * 7919 + r) % 3) as f32,
        })
        .collect();
    FixtureTable::new(&format!("00000000000000a{index}"), stem, task, "val", cols).with_rows(x, y)
}

/// Schedule order follows the dataset hash: binary, regression, multiclass,
/// a 60-column table (a `full` and a `p48` cell) and a table too small to score.
pub fn tables() -> Vec<FixtureTable> {
    vec![
        table(1, "harbor", "binary", 100, 4),
        table(2, "orchard", "regression", 120, 5),
        table(3, "meadow", "multiclass", 90, 6),
        table(4, "sensor", "regression", 80, 60),
        table(5, "pebble", "regression", 10, 4),
    ]
}

pub struct Fixture {
    pub dir: PathBuf,
    pub arena: PathBuf,
    pub schedule: PathBuf,
    pub tables: Vec<FixtureTable>,
    /// The schedule, one JSON object per cell, in file order.
    pub cells: Vec<Value>,
}

/// Write the arena under a fresh scratch directory and schedule the lite tier.
pub fn fixture(name: &str, sample_seeds: &str) -> Fixture {
    let dir = scratch(name);
    let (arena, panel) = (dir.join("arena"), dir.join("panel"));
    let tables = tables();
    write_arena(&arena, &tables).unwrap();
    kpi_ok(&[
        "build-panels",
        "--arena",
        path_str(&arena),
        "--out",
        path_str(&panel),
        "--panel-id",
        PANEL_ID,
    ]);
    let schedule = panel.join("schedule.jsonl");
    kpi_ok(&[
        "plan",
        "--arena",
        path_str(&arena),
        "--panel",
        path_str(&panel.join(format!("{PANEL_ID}.panel.json"))),
        "--families",
        path_str(&panel.join("families.jsonl")),
        "--tier",
        "lite",
        "--sample-seeds",
        sample_seeds,
        "--out",
        path_str(&schedule),
    ]);
    let cells = json_lines(&schedule);
    assert_eq!(cells.len(), 6, "five tables, one of them with two views");
    Fixture {
        dir,
        arena,
        schedule,
        tables,
        cells,
    }
}

impl Fixture {
    pub fn table_of(&self, cell: &Value) -> &FixtureTable {
        let hash = cell["dataset_hash"].as_str().unwrap();
        self.tables.iter().find(|t| t.dataset_hash == hash).unwrap()
    }

    /// Run `emit-cli` for `arm` into `<dir>/<out>`; the process environment
    /// carries a variable that must never reach a generator.
    pub fn emit(&self, arms: &Path, arm: &str, out: &str, extra: &[&str]) -> Output {
        let out = self.dir.join(out);
        Command::new(env!("CARGO_BIN_EXE_dope_kpi"))
            .args(["emit-cli", "--arena", path_str(&self.arena)])
            .args(["--schedule", path_str(&self.schedule)])
            .args(["--out", path_str(&out)])
            .args(["--arm-file", path_str(arms), "--arm", arm])
            .args(extra)
            .env("DOPE_KPI_TEST_CANARY", "must-not-leak")
            .output()
            .expect("run dope_kpi")
    }

    /// Run and require success; returns the stdout summary.
    pub fn emit_ok(&self, arms: &Path, arm: &str, out: &str, extra: &[&str]) -> Value {
        let output = self.emit(arms, arm, out, extra);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "emit-cli {arm} failed: {stderr}");
        assert!(
            stderr.is_empty(),
            "emit-cli {arm} wrote to stderr: {stderr}"
        );
        serde_json::from_slice(&output.stdout).expect("summary is one JSON object")
    }

    /// Run and require a refusal; returns stderr.
    pub fn emit_err(&self, arms: &Path, arm: &str, out: &str, extra: &[&str]) -> String {
        let output = self.emit(arms, arm, out, extra);
        assert!(
            !output.status.success(),
            "emit-cli {arm} unexpectedly succeeded"
        );
        assert!(
            output.stdout.is_empty(),
            "a refused run must not print a summary"
        );
        String::from_utf8(output.stderr).unwrap()
    }
}

/// Text generator. `fit <mode> <fit_rows> <packet> <card> <fit_seed> <task> <task2> <classes>`
/// and `sample <mode> <packet> <card> <out> <rows> <seed> <task> <classes>`.
pub const TEXT_GENERATOR: &str = r#"set -eu
phase=$1; mode=$2
if [ "$phase" = fit ]; then
  rows=$3; packet=$4; card=$5
  echo "fit rows=$rows seed=$6 task=$7/$8 classes=$9"
  env
  if read -r line; then echo "stdin is open"; exit 9; fi
  case "$mode" in
    copy) cp "$rows" "$packet" ;;
    stats) head -n 1 "$rows" > "$packet"; wc -l < "$rows" > "$card" ;;
    exit3) cp "$rows" "$packet"; cp "$rows" "$card"; cp "$rows" leak.tmp; exit 3 ;;
    sleep) cp "$rows" "$packet"; cp "$rows" leak.tmp; sleep 60 ;;
    nopacket) cp "$rows" leak.tmp ;;
    *) exit 64 ;;
  esac
  exit 0
fi
packet=$3; out=$5; rows=$6; classes=$9
echo "sample seed=$7 rows=$rows task=$8"
features=$(head -n 1 "$packet" | tr -cd , | wc -c)
body() {
  n=$1; have=$(($(wc -l < "$packet") - 1))
  while [ "$n" -gt 0 ]; do
    take=$n; if [ "$take" -gt "$have" ]; then take=$have; fi
    tail -n +2 "$packet" | head -n "$take"
    n=$((n - take))
  done
}
plain() { head -n 1 "$packet"; body "$1"; }
retarget() {
  plain "$rows" | cut -d, -f1-"$features" > features.csv
  { echo target; yes "$1" | head -n "$rows"; } > target.col
  paste -d, features.csv target.col > "$out"
}
case "$mode" in
  copy) plain "$rows" > "$out" ;;
  stats) zeros=$(head -n 1 "$packet" | tr -c ',\n' 0)
         { head -n 1 "$packet"; yes "$zeros" | head -n "$rows"; } > "$out" ;;
  shuffle) plain "$rows" > plain.csv
           cut -d, -f"$((features + 1))" plain.csv > target.col
           cut -d, -f2-"$features" plain.csv > middle.col
           cut -d, -f1 plain.csv > first.col
           paste -d, target.col middle.col first.col > "$out" ;;
  exit4) plain "$rows" > "$out"; exit 4 ;;
  sleep) plain "$rows" > "$out"; sleep 60 ;;
  noout) plain "$rows" > elsewhere.csv ;;
  dropcol) plain "$rows" | cut -d, -f2- > "$out" ;;
  extracol) plain "$rows" > plain.csv
            { echo extra; yes 1 | head -n "$rows"; } > extra.col
            paste -d, plain.csv extra.col > "$out" ;;
  text) { plain "$((rows - 1))"; body 1 | tr -c ',\n' x; } > "$out" ;;
  badlabel) retarget "$classes" ;;
  notarget) retarget "" ;;
  short) plain "$((rows - 1))" > "$out" ;;
  long) plain "$((rows + 1))" > "$out" ;;
  *) exit 64 ;;
esac
"#;

/// Binary generator over `DOPEMAT1` with the same argument layout; modes
/// `copy` and (fit only) `exit3`. `od` prints host-endian words, which is
/// little-endian on every supported test host.
pub const MAT_GENERATOR: &str = r#"set -eu
phase=$1; mode=$2
if [ "$phase" = fit ]; then
  cp "$3" "$4"
  if [ "$mode" = exit3 ]; then cp "$3" leak.tmp; exit 3; fi
  exit 0
fi
packet=$3; out=$5; rows=$6
have=$(od -An -tu4 -j8 -N4 "$packet" | tr -d ' ')
cols=$(od -An -tu4 -j12 -N4 "$packet" | tr -d ' ')
row_bytes=$(((cols + 1) * 4))
le32() {
  v=$1; i=0
  while [ "$i" -lt 4 ]; do
    printf "\\$(printf '%03o' $((v & 255)))"; v=$((v >> 8)); i=$((i + 1))
  done
}
{
  head -c 8 "$packet"
  le32 "$rows"
  tail -c +13 "$packet" | head -c 12
  n=$rows
  while [ "$n" -gt 0 ]; do
    take=$n; if [ "$take" -gt "$have" ]; then take=$have; fi
    tail -c +25 "$packet" | head -c $((take * row_bytes))
    n=$((n - take))
  done
} > "$out"
"#;

/// One arm of a test arm file: the script is run through `sh`, so it needs no
/// execute bit and the test cannot race its own `exec` against the write.
pub struct ArmDef<'a> {
    pub arm_id: &'a str,
    pub class: &'a str,
    pub io: &'a str,
    pub script: &'a Path,
    pub fit_mode: &'a str,
    pub sample_mode: &'a str,
}

impl ArmDef<'_> {
    fn json(&self) -> Value {
        json!({
            "arm_id": self.arm_id,
            "artifact_class": self.class,
            "io": self.io,
            "binary": path_str(self.script),
            "fit": ["sh", "{bin}", "fit", self.fit_mode, "{fit_rows}", "{packet}", "{card}",
                    "{fit_seed}", "{task}", "{task2}", "{classes}"],
            "sample": ["sh", "{bin}", "sample", self.sample_mode, "{packet}", "{card}", "{out}",
                       "{rows}", "{seed}", "{task}", "{classes}"],
        })
    }
}

pub fn write_script(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, body).unwrap();
    path
}

pub fn write_arm_file(dir: &Path, name: &str, arms: &[ArmDef]) -> PathBuf {
    let arms: Vec<Value> = arms.iter().map(ArmDef::json).collect();
    let file = json!({ "schema": "dope.kpi-ext-arms/v1", "arms": arms });
    write_script(dir, name, &file.to_string())
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// The 31-bit command seed of formats.md section 4, recomputed independently.
pub fn command_seed(cell: &Value, sample_seed: u64, purpose: &str) -> u64 {
    let key = format!(
        "dope-kpi-v2|{}|{}|{}|{sample_seed}|{purpose}",
        cell["panel_sha256"].as_str().unwrap(),
        cell["dataset_hash"].as_str().unwrap(),
        cell["data_seed"]
    );
    let digest = Sha256::digest(key.as_bytes());
    let low = u64::from_le_bytes(digest[..8].try_into().unwrap());
    low % (1 << 31)
}

/// A `DOPEMAT1` file parsed without the code under test: (rows, x_cols, values).
pub fn read_mat(path: &Path) -> (usize, usize, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&bytes[..8], b"DOPEMAT1");
    let word = |i: usize| u32::from_le_bytes(bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap());
    assert_eq!((word(2), word(3)), (1, 0), "y_cols and the reserved word");
    let (rows, cols) = (word(0) as usize, word(1) as usize);
    let values: Vec<f32> = bytes[24..]
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    assert_eq!(values.len(), rows * (cols + 1));
    (rows, cols, values)
}

/// Fit row `position` of a cell (features of the view, then the target), as stored in the arena.
pub fn fit_row(table: &FixtureTable, cell: &Value, position: usize) -> Vec<f32> {
    let row = cell["a_idx"][position].as_u64().unwrap() as usize;
    let cols = cell["cols"].as_array().unwrap();
    let mut values: Vec<f32> = cols
        .iter()
        .map(|c| table.x[row * table.n_cols + c.as_u64().unwrap() as usize])
        .collect();
    values.push(table.y[row]);
    values
}

pub fn stem(cell: &Value) -> String {
    cell["cell_id"].as_str().unwrap().replace(':', "__")
}

/// Every regular file under `dir`, sorted.
pub fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in std::fs::read_dir(&next).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

pub fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

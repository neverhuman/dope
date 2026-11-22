//! Shared fixture for the scoring tests: a small arena whose tables carry a
//! known linear signal for each task, and runners for the scoring pipeline.
#![allow(dead_code)]

use crate::kpi_support::fixture::{write_arena, FixtureTable};
use crate::kpi_support::{json_lines, kpi_ok, mix, path_str, scratch};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const PANEL_ID: &str = "sig";
pub const AUDITORS: [&str; 3] = ["linear-seeded-v1", "gbdt", "mlp-seeded-v1"];

fn uniform(key: u64) -> f64 {
    ((mix(key) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// Standard normal by Box-Muller from two hashed uniforms.
fn normal(index: u64, row: usize, stream: u64) -> f64 {
    let key = index * 1_000_003 + (row as u64) * 64 + stream * 2;
    (-2.0 * uniform(key).ln()).sqrt() * (std::f64::consts::TAU * uniform(key + 1)).cos()
}

/// `rows` rows with four features in very different raw units and a target
/// driven by `1.6 x0 - 1.2 x1 + 0.8 x2`; the fourth feature is noise. A few
/// cells of the last two features are missing.
pub fn signal_table(index: u64, stem: &str, task: &str, rows: usize) -> FixtureTable {
    let (mut x, mut y) = (Vec::with_capacity(rows * 4), Vec::with_capacity(rows));
    for r in 0..rows {
        let z: Vec<f64> = (0..5).map(|s| normal(index, r, s)).collect();
        let score = 1.6 * z[0] - 1.2 * z[1] + 0.8 * z[2];
        x.push((10.0 + 3.0 * z[0]) as f32);
        x.push((250.0 * z[1]) as f32);
        x.push(if r % 23 == 0 {
            f32::NAN
        } else {
            (0.01 * z[2]) as f32
        });
        x.push(if r % 17 == 0 { f32::NAN } else { z[3] as f32 });
        y.push(match task {
            "regression" => (50.0 + 10.0 * (score + 0.6 * z[4])) as f32,
            "binary" => f32::from(score + 0.9 * z[4] > 0.2),
            _ => {
                let noisy = score + 0.8 * z[4];
                f32::from(noisy > -1.0) + f32::from(noisy > 1.0)
            }
        });
    }
    FixtureTable::new(&format!("{:016x}", mix(index + 77)), stem, task, "val", 4).with_rows(x, y)
}

/// Three signal tables of 2,930 duplicate groups (A and B of 1,024 rows, T of
/// 879 groups, no usable C), a
/// 4,400-row table whose cell also has a full real twin C, a table too small
/// to score, and one training table.
pub fn corpus() -> Vec<FixtureTable> {
    let mut binary = signal_table(2, "sig_binary", "binary", 3030);
    // 100 duplicate groups of two rows (2,930 groups in all), so T carries donor clusters.
    binary.dup = (0..3030).map(|r| if r < 200 { r / 2 } else { r }).collect();
    let mut train = signal_table(6, "sig_train", "regression", 300);
    train.split = "train".to_string();
    vec![
        signal_table(1, "sig_regression", "regression", 2930),
        binary,
        signal_table(3, "sig_multiclass", "multiclass", 2930),
        signal_table(4, "sig_twin", "binary", 4400),
        signal_table(5, "sig_tiny", "regression", 10),
        train,
    ]
}

pub struct Signal {
    pub tables: Vec<FixtureTable>,
    pub dir: PathBuf,
    pub arena: PathBuf,
    /// Lite schedule of every panel table (one data seed).
    pub schedule: PathBuf,
}

impl Signal {
    /// Write the arena and plan it (lite tier plus `plan_extra`) under a
    /// scratch directory named `name`.
    pub fn build(name: &str, plan_extra: &[&str]) -> Self {
        let dir = scratch(name);
        let (arena, out) = (dir.join("arena"), dir.join("plan"));
        let tables = corpus();
        write_arena(&arena, &tables).unwrap();
        kpi_ok(&[
            "build-panels",
            "--arena",
            path_str(&arena),
            "--out",
            path_str(&out),
            "--panel-id",
            PANEL_ID,
        ]);
        let schedule = out.join("schedule.jsonl");
        let panel = out.join(format!("{PANEL_ID}.panel.json"));
        let families = out.join("families.jsonl");
        let mut plan = vec![
            "plan",
            "--arena",
            path_str(&arena),
            "--panel",
            path_str(&panel),
            "--families",
            path_str(&families),
            "--tier",
            "lite",
            "--out",
            path_str(&schedule),
        ];
        plan.extend_from_slice(plan_extra);
        kpi_ok(&plan);
        Self {
            tables,
            dir,
            arena,
            schedule,
        }
    }

    pub fn hash(&self, stem: &str) -> &str {
        let table = self.tables.iter().find(|t| t.family_stem == stem);
        &table.expect("fixture stem").dataset_hash
    }

    pub fn cell_id(&self, stem: &str) -> String {
        format!("{PANEL_ID}:{}:full:d0", self.hash(stem))
    }

    /// A schedule holding only the cells of `stems`, written beside the full one.
    pub fn subset(&self, name: &str, stems: &[&str]) -> PathBuf {
        let keep: Vec<String> = stems.iter().map(|s| self.cell_id(s)).collect();
        let text = std::fs::read_to_string(&self.schedule).unwrap();
        let lines: Vec<&str> = text
            .lines()
            .filter(|line| keep.iter().any(|id| line.contains(id.as_str())))
            .collect();
        assert_eq!(lines.len(), stems.len(), "one cell per requested stem");
        let path = self.dir.join(name);
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        path
    }

    /// Run `real-ref`; returns the output path and the summary.
    pub fn real_ref(&self, schedule: &Path, name: &str, extra: &[&str]) -> (PathBuf, Value) {
        let out = self.dir.join(name);
        let mut args = vec![
            "real-ref",
            "--arena",
            path_str(&self.arena),
            "--schedule",
            path_str(schedule),
            "--out",
            path_str(&out),
        ];
        args.extend_from_slice(extra);
        let summary = serde_json::from_str(&kpi_ok(&args)).unwrap();
        (out, summary)
    }

    /// Run `score` for `arm` into `<dir>/<out_name>`; returns that directory and the summary.
    pub fn score(
        &self,
        schedule: &Path,
        realref: &Path,
        emit_dir: &Path,
        arm: &str,
        out_name: &str,
        extra: &[&str],
    ) -> (PathBuf, Value) {
        let out = self.dir.join(out_name);
        let mut args = score_args(&self.arena, schedule, realref, emit_dir, arm, &out);
        args.extend_from_slice(extra);
        let summary = serde_json::from_str(&kpi_ok(&args)).unwrap();
        (out, summary)
    }
}

pub fn score_args<'a>(
    arena: &'a Path,
    schedule: &'a Path,
    realref: &'a Path,
    emit_dir: &'a Path,
    arm: &'a str,
    out: &'a Path,
) -> Vec<&'a str> {
    vec![
        "score",
        "--arena",
        path_str(arena),
        "--schedule",
        path_str(schedule),
        "--realref",
        path_str(realref),
        "--emit-dir",
        path_str(emit_dir),
        "--arm",
        arm,
        "--out",
        path_str(out),
    ]
}

/// `rows.jsonl` of a score directory.
pub fn score_rows(out: &Path) -> Vec<Value> {
    json_lines(&out.join("rows.jsonl"))
}

pub fn text(value: &Value) -> &str {
    value.as_str().expect("string field")
}

pub fn number(value: &Value) -> f64 {
    value.as_f64().expect("numeric field")
}

/// Rows keyed by `(cell_id, auditor, multiplier, sample_seed)`.
pub fn by_key(rows: &[Value]) -> BTreeMap<(String, String, u64, u64), &Value> {
    rows.iter()
        .map(|row| {
            let key = (
                text(&row["cell_id"]).to_string(),
                text(&row["auditor"]).to_string(),
                row["multiplier"].as_u64().unwrap(),
                row["sample_seed"].as_u64().unwrap(),
            );
            (key, row)
        })
        .collect()
}

/// Write a `DOPEMAT1` file from row-major values (features first, target last).
pub fn write_mat(path: &Path, rows: usize, x_cols: usize, data: &[f32]) {
    assert_eq!(data.len(), rows * (x_cols + 1), "matrix shape");
    let mut bytes = b"DOPEMAT1".to_vec();
    for field in [rows as u32, x_cols as u32, 1, 0] {
        bytes.extend_from_slice(&field.to_le_bytes());
    }
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

/// The A rows of a scheduled cell, repeated `times` times, as matrix values.
pub fn repeated_a_rows(signal: &Signal, cell: &Value, times: usize) -> (usize, usize, Vec<f32>) {
    let hash = text(&cell["dataset_hash"]);
    let table = signal.tables.iter().find(|t| t.dataset_hash == hash);
    let table = table.expect("scheduled table");
    let cols = crate::kpi_support::usizes(&cell["cols"]);
    let a_idx = crate::kpi_support::usizes(&cell["a_idx"]);
    let mut data = Vec::new();
    for _ in 0..times {
        for &row in &a_idx {
            data.extend(cols.iter().map(|&c| table.x[row * table.n_cols + c]));
            data.push(table.y[row]);
        }
    }
    (a_idx.len() * times, cols.len(), data)
}

/// Relative path of an emitted file (formats.md section 5).
pub fn mat_name(arm: &str, cell_id: &str, seed: u64, multiplier: u64) -> String {
    format!(
        "{arm}/{}.s{seed}.m{multiplier}.mat",
        cell_id.replace(':', "__")
    )
}

/// One `emit.jsonl` row as the Python emitter writes it.
pub fn emit_row(
    arm: &str,
    cell_id: &str,
    (seed, multiplier): (u64, u64),
    status: &str,
    reason: &str,
    rows: usize,
    packet_bytes: u64,
) -> Value {
    let file = if status == "ok" {
        mat_name(arm, cell_id, seed, multiplier)
    } else {
        String::new()
    };
    serde_json::json!({
        "arch_id": "fixture_v1", "arm_id": arm, "artifact_class": "source_free",
        "cell_id": cell_id, "cold_bytes": 0, "decode_ms": 3.25, "encode_ms": 41.5,
        "file": file, "multiplier": multiplier, "packet_bytes": packet_bytes,
        "packet_sha256": "ab".repeat(32), "reason": reason, "rows": rows,
        "sample_seed": seed, "status": status, "weights_sha256": "cd".repeat(32),
    })
}

/// Append rows to the shared `emit.jsonl` of an emit directory.
pub fn append_emit_log(emit_dir: &Path, rows: &[Value]) {
    use std::io::Write;
    std::fs::create_dir_all(emit_dir).unwrap();
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(emit_dir.join("emit.jsonl"))
        .unwrap();
    for row in rows {
        writeln!(log, "{row}").unwrap();
    }
}

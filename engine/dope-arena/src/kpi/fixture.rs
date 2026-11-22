//! Test support: write a flat corpus arena (formats.md section 1) from in-memory tables.
//!
//! Deliberately independent of the reader in `io.rs` (no shared types), so the
//! reader is exercised against a second implementation of the layout. Unit
//! tests reach it as `kpi::fixture`; CLI tests include this file by path.

use sha2::{Digest, Sha256};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct FixtureTable {
    pub dataset_hash: String,
    pub canonical_sha256: String,
    pub family_stem: String,
    pub task: String,
    pub n_classes: u32,
    pub split: String,
    pub n_cols: usize,
    /// Row-major features, `rows * n_cols` values.
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    /// Duplicate-group id per row; defaults to one group per row.
    pub dup: Vec<i32>,
}

impl FixtureTable {
    /// An empty table; `canonical_sha256` is derived from the dataset hash.
    pub fn new(
        dataset_hash: &str,
        family_stem: &str,
        task: &str,
        split: &str,
        n_cols: usize,
    ) -> Self {
        let n_classes = match task {
            "binary" => 2,
            "multiclass" => 3,
            _ => 0,
        };
        Self {
            dataset_hash: dataset_hash.to_string(),
            canonical_sha256: format!("{:x}", Sha256::digest(dataset_hash.as_bytes())),
            family_stem: family_stem.to_string(),
            task: task.to_string(),
            n_classes,
            split: split.to_string(),
            n_cols,
            x: Vec::new(),
            y: Vec::new(),
            dup: Vec::new(),
        }
    }

    /// Set the rows; every row becomes its own duplicate group.
    pub fn with_rows(mut self, x: Vec<f32>, y: Vec<f32>) -> Self {
        assert_eq!(x.len(), y.len() * self.n_cols, "fixture shape");
        self.dup = (0..y.len() as i32).collect();
        self.x = x;
        self.y = y;
        self
    }

    pub fn n_rows(&self) -> usize {
        self.y.len()
    }
}

fn le_bytes<T: Copy>(values: &[T], encode: fn(T) -> [u8; 4]) -> Vec<u8> {
    values.iter().flat_map(|v| encode(*v)).collect()
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Write `META.json`, `index.jsonl`, `X.f32`, `y.f32` and `dup.i32` into `dir`.
pub fn write_arena(dir: &Path, tables: &[FixtureTable]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let (mut x, mut y, mut dup, mut index) = (Vec::new(), Vec::new(), Vec::new(), String::new());
    let (mut x_off, mut row_off) = (0usize, 0usize);
    for (table_id, table) in tables.iter().enumerate() {
        let rows = table.n_rows();
        assert_eq!(table.x.len(), rows * table.n_cols, "fixture shape");
        assert_eq!(table.dup.len(), rows, "fixture duplicate groups");
        index.push_str(&format!(
            "{{\"table_id\":{table_id},\"dataset_hash\":{},\"canonical_sha256\":{},\
             \"family_stem\":{},\"task\":{},\"n_classes\":{},\"n_rows\":{rows},\"n_cols\":{},\
             \"split\":{},\"x_off\":{x_off},\"row_off\":{row_off}}}\n",
            serde_json::json!(table.dataset_hash),
            serde_json::json!(table.canonical_sha256),
            serde_json::json!(table.family_stem),
            serde_json::json!(table.task),
            table.n_classes,
            table.n_cols,
            serde_json::json!(table.split),
        ));
        x.extend(le_bytes(&table.x, f32::to_le_bytes));
        y.extend(le_bytes(&table.y, f32::to_le_bytes));
        dup.extend(le_bytes(&table.dup, i32::to_le_bytes));
        x_off += rows * table.n_cols;
        row_off += rows;
    }
    let meta = format!(
        "{{\"schema\":\"dope.flat-corpus/v1\",\"source_manifest_sha256\":\"{}\",\"tables\":{},\
         \"rows\":{row_off},\"cells\":{x_off},\"x_sha256\":\"{}\",\"y_sha256\":\"{}\",\
         \"dup_sha256\":\"{}\",\"index_sha256\":\"{}\"}}\n",
        sha(b"fixture-manifest"),
        tables.len(),
        sha(&x),
        sha(&y),
        sha(&dup),
        sha(index.as_bytes()),
    );
    std::fs::write(dir.join("X.f32"), x)?;
    std::fs::write(dir.join("y.f32"), y)?;
    std::fs::write(dir.join("dup.i32"), dup)?;
    std::fs::write(dir.join("index.jsonl"), index)?;
    std::fs::write(dir.join("META.json"), meta)
}

//! Reader for the flat corpus arena `dope.flat-corpus/v1` (formats.md section 1).
//!
//! `open` checks the identity file, the index digest and every shape and
//! offset against the three binary files before any table is read. Bulk
//! checksums are verified on demand. Errors name files, table ids and field
//! names only, never cell values.

use super::jsonl::{file_error, file_sha256, parse_json, parse_jsonl, read_file, sha256_hex};
use dope_arena::{ArenaError, ArenaResult};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub const ARENA_SCHEMA: &str = "dope.flat-corpus/v1";
pub const SPLIT_TRAIN: &str = "train";
pub const SPLIT_VAL: &str = "val";

#[derive(Clone, Debug, Deserialize)]
pub struct ArenaMeta {
    pub schema: String,
    pub source_manifest_sha256: String,
    pub tables: u64,
    pub rows: u64,
    pub cells: u64,
    pub x_sha256: String,
    pub y_sha256: String,
    pub dup_sha256: String,
    pub index_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct IndexRow {
    pub table_id: usize,
    pub dataset_hash: String,
    pub canonical_sha256: String,
    pub family_stem: String,
    pub task: String,
    pub n_classes: u32,
    pub n_rows: usize,
    pub n_cols: usize,
    pub split: String,
    pub x_off: u64,
    pub row_off: u64,
}

/// One table held in memory: row-major features and the target (see `read_dup` for groups).
#[derive(Clone, Debug)]
pub struct TableData {
    pub n_rows: usize,
    pub n_cols: usize,
    x: Vec<f32>,
    pub y: Vec<f32>,
}

impl TableData {
    fn check(&self, row: usize, col: usize) -> ArenaResult<()> {
        if row >= self.n_rows || col >= self.n_cols {
            return Err(contract(format!(
                "cell ({row}, {col}) is outside a {} x {} table",
                self.n_rows, self.n_cols
            )));
        }
        Ok(())
    }

    /// Feature value at (`row`, `col`); NaN marks a missing cell.
    // Row-wise access is used by the scoring stages; `build-panels` reads columns.
    #[allow(dead_code)]
    pub fn cell(&self, row: usize, col: usize) -> ArenaResult<f32> {
        self.check(row, col)?;
        Ok(self.x[row * self.n_cols + col])
    }

    /// All features of one row, in column order.
    #[allow(dead_code)]
    pub fn row(&self, row: usize) -> ArenaResult<&[f32]> {
        self.check(row, 0)?;
        Ok(&self.x[row * self.n_cols..(row + 1) * self.n_cols])
    }

    /// Copy one feature column into `out` (cleared first).
    pub fn column_into(&self, col: usize, out: &mut Vec<f32>) -> ArenaResult<()> {
        if col >= self.n_cols {
            return Err(contract(format!(
                "column {col} is outside a {}-column table",
                self.n_cols
            )));
        }
        out.clear();
        out.extend(self.x.iter().skip(col).step_by(self.n_cols).copied());
        Ok(())
    }
}

#[derive(Debug)]
pub struct Arena {
    dir: PathBuf,
    meta: ArenaMeta,
    index: Vec<IndexRow>,
    by_hash: BTreeMap<String, usize>,
}

fn contract(message: String) -> ArenaError {
    ArenaError::Contract(message)
}

fn is_sha256_hex(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn check_row(position: usize, row: &IndexRow, x_off: u64, row_off: u64) -> ArenaResult<()> {
    let fail = |field: &str| {
        Err(contract(format!(
            "index.jsonl: table {position}: invalid {field}"
        )))
    };
    if row.table_id != position {
        return fail("table_id");
    }
    if row.dataset_hash.is_empty() || row.dataset_hash.contains([':', '/', '|']) {
        return fail("dataset_hash");
    }
    if !is_sha256_hex(&row.canonical_sha256) {
        return fail("canonical_sha256");
    }
    let classes_ok = match row.task.as_str() {
        "regression" => row.n_classes == 0,
        "binary" => row.n_classes == 2,
        "multiclass" => row.n_classes >= 2,
        _ => return fail("task"),
    };
    if !classes_ok {
        return fail("n_classes");
    }
    if row.n_rows == 0 || row.n_cols == 0 {
        return fail("shape");
    }
    if row.split.is_empty() {
        return fail("split");
    }
    if row.x_off != x_off {
        return fail("x_off");
    }
    if row.row_off != row_off {
        return fail("row_off");
    }
    Ok(())
}

fn read_block<T>(
    path: &Path,
    offset: u64,
    count: usize,
    decode: fn([u8; 4]) -> T,
) -> ArenaResult<Vec<T>> {
    let mut file = File::open(path).map_err(|e| file_error(path, &e))?;
    file.seek(SeekFrom::Start(offset * 4))
        .map_err(|e| file_error(path, &e))?;
    let mut bytes = vec![0u8; count * 4];
    file.read_exact(&mut bytes)
        .map_err(|e| file_error(path, &e))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|word| decode([word[0], word[1], word[2], word[3]]))
        .collect())
}

impl Arena {
    pub fn open(dir: &Path) -> ArenaResult<Self> {
        let meta: ArenaMeta = parse_json("META.json", &read_file(&dir.join("META.json"))?)?;
        if meta.schema != ARENA_SCHEMA {
            return Err(contract("META.json: unsupported arena schema".into()));
        }
        let index_bytes = read_file(&dir.join("index.jsonl"))?;
        if sha256_hex(&index_bytes) != meta.index_sha256 {
            return Err(contract(
                "index.jsonl does not match META.json index_sha256".into(),
            ));
        }
        let index: Vec<IndexRow> = parse_jsonl("index.jsonl", &index_bytes)?;
        let mut by_hash = BTreeMap::new();
        let (mut cells, mut rows) = (0u64, 0u64);
        for (position, row) in index.iter().enumerate() {
            check_row(position, row, cells, rows)?;
            if by_hash.insert(row.dataset_hash.clone(), position).is_some() {
                return Err(contract(format!(
                    "index.jsonl: table {position}: duplicate dataset_hash"
                )));
            }
            let table_cells = (row.n_rows as u64)
                .checked_mul(row.n_cols as u64)
                .and_then(|c| cells.checked_add(c));
            cells = table_cells.ok_or_else(|| {
                contract(format!("index.jsonl: table {position}: shape overflow"))
            })?;
            rows += row.n_rows as u64;
        }
        if index.len() as u64 != meta.tables || rows != meta.rows || cells != meta.cells {
            return Err(contract(
                "META.json totals disagree with index.jsonl".into(),
            ));
        }
        let arena = Self {
            dir: dir.to_path_buf(),
            meta,
            index,
            by_hash,
        };
        for (name, elements) in [("X.f32", cells), ("y.f32", rows), ("dup.i32", rows)] {
            let path = arena.dir.join(name);
            let size = std::fs::metadata(&path)
                .map_err(|e| file_error(&path, &e))?
                .len();
            if Some(size) != elements.checked_mul(4) {
                return Err(contract(format!(
                    "{name}: file size disagrees with index.jsonl"
                )));
            }
        }
        Ok(arena)
    }

    pub fn meta(&self) -> &ArenaMeta {
        &self.meta
    }

    pub fn tables(&self) -> &[IndexRow] {
        &self.index
    }

    pub fn table(&self, table_id: usize) -> ArenaResult<&IndexRow> {
        self.index
            .get(table_id)
            .ok_or_else(|| contract(format!("table {table_id} is outside the arena")))
    }

    pub fn table_id_of(&self, dataset_hash: &str) -> Option<usize> {
        self.by_hash.get(dataset_hash).copied()
    }

    /// Recompute the SHA-256 of the three binary files and compare with `META.json`.
    pub fn verify_checksums(&self) -> ArenaResult<()> {
        let pins = [
            ("X.f32", &self.meta.x_sha256),
            ("y.f32", &self.meta.y_sha256),
            ("dup.i32", &self.meta.dup_sha256),
        ];
        for (name, pinned) in pins {
            if &file_sha256(&self.dir.join(name))? != pinned {
                return Err(contract(format!("{name}: checksum mismatch")));
            }
        }
        Ok(())
    }

    /// Duplicate-group id of every row of one table.
    pub fn read_dup(&self, table_id: usize) -> ArenaResult<Vec<i32>> {
        let row = self.table(table_id)?;
        read_block(
            &self.dir.join("dup.i32"),
            row.row_off,
            row.n_rows,
            i32::from_le_bytes,
        )
    }

    /// Features and target of one table.
    pub fn read_table(&self, table_id: usize) -> ArenaResult<TableData> {
        let row = self.table(table_id)?;
        let cells = row.n_rows * row.n_cols;
        Ok(TableData {
            n_rows: row.n_rows,
            n_cols: row.n_cols,
            x: read_block(
                &self.dir.join("X.f32"),
                row.x_off,
                cells,
                f32::from_le_bytes,
            )?,
            y: read_block(
                &self.dir.join("y.f32"),
                row.row_off,
                row.n_rows,
                f32::from_le_bytes,
            )?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kpi::fixture::{write_arena, FixtureTable};

    fn two_tables() -> Vec<FixtureTable> {
        let first = FixtureTable::new("aaaaaaaaaaaaaaa1", "first", "regression", "train", 2)
            .with_rows((0..6).map(|i| i as f32).collect(), vec![0.5, 1.5, 2.5]);
        let mut second = FixtureTable::new("aaaaaaaaaaaaaaa2", "second", "multiclass", "val", 3)
            .with_rows(vec![10.0, f32::NAN, 12.0, 13.0, 14.0, 15.0], vec![0.0, 2.0]);
        second.n_classes = 3;
        second.dup = vec![7, 7];
        vec![first, second]
    }

    #[test]
    fn reads_tables_back_with_bounds_checks() {
        let dir = crate::kpi::unit_test_dir("io-read");
        write_arena(&dir, &two_tables()).unwrap();
        let arena = Arena::open(&dir).unwrap();
        arena.verify_checksums().unwrap();
        assert_eq!(arena.tables().len(), 2);
        assert_eq!(arena.meta().rows, 5);
        assert_eq!(arena.meta().cells, 12);
        assert_eq!(arena.table_id_of("aaaaaaaaaaaaaaa2"), Some(1));
        assert_eq!(arena.table_id_of("missing"), None);

        let first = arena.read_table(0).unwrap();
        assert_eq!((first.n_rows, first.n_cols), (3, 2));
        assert_eq!(first.cell(2, 1).unwrap(), 5.0);
        assert_eq!(first.row(1).unwrap(), &[2.0, 3.0]);
        assert_eq!(first.y, vec![0.5, 1.5, 2.5]);
        assert_eq!(arena.read_dup(0).unwrap(), vec![0, 1, 2]);
        let mut column = vec![99.0];
        first.column_into(1, &mut column).unwrap();
        assert_eq!(column, vec![1.0, 3.0, 5.0]);

        let second = arena.read_table(1).unwrap();
        assert_eq!(second.cell(0, 0).unwrap(), 10.0);
        assert!(second.cell(0, 1).unwrap().is_nan());
        assert_eq!(second.cell(1, 2).unwrap(), 15.0);
        assert_eq!(arena.read_dup(1).unwrap(), vec![7, 7]);

        assert!(second.cell(2, 0).is_err());
        assert!(second.cell(0, 3).is_err());
        assert!(second.row(2).is_err());
        assert!(second.column_into(3, &mut column).is_err());
        assert!(arena.read_table(2).is_err());
        assert!(arena.read_dup(2).is_err());
        assert!(arena.table(2).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_corrupt_or_inconsistent_arenas() {
        let dir = crate::kpi::unit_test_dir("io-corrupt");
        write_arena(&dir, &two_tables()).unwrap();

        // A flipped payload byte passes the cheap open but fails the checksum pass.
        let x_path = dir.join("X.f32");
        let mut x = std::fs::read(&x_path).unwrap();
        x[5] ^= 0x40;
        std::fs::write(&x_path, &x).unwrap();
        let err = Arena::open(&dir).unwrap().verify_checksums().unwrap_err();
        assert!(err.to_string().contains("X.f32: checksum mismatch"));

        // A truncated payload is caught at open.
        std::fs::write(&x_path, &x[..x.len() - 4]).unwrap();
        let err = Arena::open(&dir).unwrap_err().to_string();
        assert!(err.contains("X.f32: file size disagrees"), "{err}");

        // An index edited after packing no longer matches its digest.
        write_arena(&dir, &two_tables()).unwrap();
        let index_path = dir.join("index.jsonl");
        let edited = std::fs::read_to_string(&index_path)
            .unwrap()
            .replace("\"split\":\"val\"", "\"split\":\"train\"");
        std::fs::write(&index_path, edited).unwrap();
        let err = Arena::open(&dir).unwrap_err().to_string();
        assert!(err.contains("index_sha256"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_rows_that_break_the_layout_contract() {
        let good = IndexRow {
            table_id: 3,
            dataset_hash: "abcd".into(),
            canonical_sha256: "0".repeat(64),
            family_stem: "stem".into(),
            task: "binary".into(),
            n_classes: 2,
            n_rows: 4,
            n_cols: 2,
            split: "val".into(),
            x_off: 40,
            row_off: 9,
        };
        check_row(3, &good, 40, 9).unwrap();
        type Mutation = fn(&mut IndexRow);
        let cases: [(&str, Mutation); 9] = [
            ("table_id", |r| r.table_id = 2),
            ("dataset_hash", |r| r.dataset_hash = "ab:cd".into()),
            ("canonical_sha256", |r| r.canonical_sha256 = "xyz".into()),
            ("task", |r| r.task = "ranking".into()),
            ("n_classes", |r| r.n_classes = 3),
            ("shape", |r| r.n_cols = 0),
            ("split", |r| r.split.clear()),
            ("x_off", |r| r.x_off = 41),
            ("row_off", |r| r.row_off = 8),
        ];
        for (field, mutate) in cases {
            let mut bad = good.clone();
            mutate(&mut bad);
            let err = check_row(3, &bad, 40, 9).unwrap_err().to_string();
            assert!(err.ends_with(&format!("invalid {field}")), "{field}: {err}");
        }
    }
}

//! Shared support of the `dope_statpacket` end-to-end tests: the command-line
//! wrapper, `DOPEMAT1` files, a small generator and the synthetic tables.
#![allow(dead_code)]

pub mod eval;

use eval::Table;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub const BIN: &str = env!("CARGO_BIN_EXE_dope_statpacket");
pub const FIT_ROWS: usize = 600;
pub const TEST_ROWS: usize = 2000;
pub const LEVELS: [&str; 5] = ["A", "B", "C", "D", "E"];

pub fn dir(name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("statpacket_e2e_{name}"));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

pub fn run(args: &[&str]) -> Output {
    Command::new(BIN).args(args).output().unwrap()
}

pub fn ok_json(args: &[&str]) -> serde_json::Value {
    let out = run(args);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

pub fn text(path: &Path) -> &str {
    path.to_str().unwrap()
}

pub fn write_mat(path: &Path, table: &Table) {
    let mut bytes = b"DOPEMAT1".to_vec();
    for v in [table.rows(), table.p, 1, 0] {
        bytes.extend((v as u32).to_le_bytes());
    }
    for (row, y) in table.x.chunks(table.p).zip(&table.y) {
        for v in row.iter().chain([y]) {
            bytes.extend((*v as f32).to_le_bytes());
        }
    }
    std::fs::write(path, bytes).unwrap();
}

/// Reads a DOPEMAT1 file with exactly one target column.
pub fn read_mat(path: &Path) -> Table {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"DOPEMAT1");
    let word =
        |i: usize| u32::from_le_bytes(bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap()) as usize;
    let (rows, p, y_cols) = (word(0), word(1), word(2));
    assert_eq!((y_cols, word(3)), (1, 0));
    assert_eq!(bytes.len(), 24 + 4 * rows * (p + 1));
    let cells: Vec<f64> = bytes[24..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64)
        .collect();
    let mut table = Table {
        x: Vec::with_capacity(rows * p),
        y: Vec::with_capacity(rows),
        p,
    };
    for row in cells.chunks(p + 1) {
        table.x.extend(&row[..p]);
        table.y.push(row[p]);
    }
    table
}

/// Deterministic generator (splitmix64 with Box-Muller normals).
pub struct Rng(pub u64);

impl Rng {
    pub fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// A table split into FIT rows and a disjoint real test set.
pub struct Fixture {
    pub name: &'static str,
    pub task: &'static str,
    /// 0 for regression.
    pub classes: usize,
    pub fit: Table,
    pub test: Table,
}

fn split(name: &'static str, task: &'static str, classes: usize, all: Table) -> Fixture {
    Fixture {
        name,
        task,
        classes,
        fit: all.slice(0, FIT_ROWS),
        test: all.slice(FIT_ROWS, FIT_ROWS + TEST_ROWS),
    }
}

/// Regression, 16 columns (three latent factors; skewed, integer, binary and
/// five-level columns): a linear signal, one interaction `x0 * x4` and noise.
pub fn regression_fixture() -> Fixture {
    let (mut rng, p) = (Rng(0x5e9_0001), 16);
    let mut all = Table {
        x: Vec::new(),
        y: Vec::new(),
        p,
    };
    for _ in 0..FIT_ROWS + TEST_ROWS {
        let g = [rng.normal(), rng.normal(), rng.normal()];
        let mut x = vec![0.0; p];
        x[0] = g[0] + 0.6 * rng.normal();
        x[1] = g[0] - 0.5 * g[1] + 0.6 * rng.normal();
        x[2] = g[1] + 0.7 * rng.normal();
        x[3] = (0.5 * (g[1] + 0.5 * rng.normal())).exp();
        x[4] = g[2] + 0.5 * rng.normal();
        x[5] = (3.0 * g[2] + 2.0 * rng.normal()).round();
        x[6] = (g[0] + rng.normal() > 0.3) as u8 as f64;
        x[7] = (g[1] + rng.normal() + 2.0).floor().clamp(0.0, 4.0);
        for (j, slot) in x.iter_mut().enumerate().skip(8) {
            *slot = 0.3 * g[j % 3] + rng.normal();
        }
        let signal = x[0] - 0.8 * x[2] + 0.5 * x[4] + 0.6 * x[6] + 0.3 * x[8] + 0.7 * x[0] * x[4];
        all.y.push(40.0 + 12.0 * (signal + 0.5 * rng.normal()));
        // Raw units differ by column.
        x[2] = 100.0 + 15.0 * x[2];
        x[9] *= 0.01;
        all.x.extend(x);
    }
    split("regression", "regression", 0, all)
}

/// Binary, 12 columns, about 15 % positives, logistic-linear signal.
pub fn binary_fixture() -> Fixture {
    let (mut rng, p) = (Rng(0xb1_0002), 12);
    let mut all = Table {
        x: Vec::new(),
        y: Vec::new(),
        p,
    };
    for _ in 0..FIT_ROWS + TEST_ROWS {
        let g = [rng.normal(), rng.normal()];
        let mut x = vec![0.0; p];
        x[0] = g[0] + 0.5 * rng.normal();
        x[1] = 0.7 * g[0] + 0.7 * rng.normal();
        x[2] = g[1] + 0.6 * rng.normal();
        x[3] = (g[1] + rng.normal() > 0.0) as u8 as f64;
        x[4] = (0.6 * rng.normal()).exp();
        x[5] = (2.0 * g[0] + 2.0 * rng.normal()).round();
        for (j, slot) in x.iter_mut().enumerate().skip(6) {
            *slot = 0.4 * g[j % 2] + rng.normal();
        }
        let logit = -3.0 + 1.1 * x[0] - 0.8 * x[2] + 0.9 * x[3] + 0.5 * x[6] - 0.4 * x[7];
        all.y
            .push((rng.unit() < 1.0 / (1.0 + (-logit).exp())) as u8 as f64);
        x[1] = 50.0 + 8.0 * x[1];
        all.x.extend(x);
    }
    split("binary", "binary", 2, all)
}

/// Five classes with separated means on `p` columns that share latent
/// factors; class shares 0.30 / 0.25 / 0.20 / 0.15 / 0.10.
pub fn multiclass_table(p: usize, rows: usize, seed: u64) -> Table {
    let mut rng = Rng(seed);
    let mut all = Table {
        x: Vec::new(),
        y: Vec::new(),
        p,
    };
    for _ in 0..rows {
        let u = rng.unit();
        let class = [0.30, 0.55, 0.75, 0.90]
            .iter()
            .filter(|&&edge| u >= edge)
            .count();
        let g = [rng.normal(), rng.normal(), rng.normal()];
        for j in 0..p {
            // Half of the columns carry a class shift; the rest are correlated noise.
            let shift = if j < p / 2 && j % 5 == class {
                1.3
            } else {
                0.0
            };
            let value = shift + 0.6 * g[j % 3] + rng.normal();
            all.x.push(if j == 1 {
                (0.4 * value).exp()
            } else if j == 2 {
                value.round()
            } else {
                value
            });
        }
        all.y.push(class as f64);
    }
    all
}

/// Five classes that differ in the spread of every column (column `j` is
/// wider for class `j mod 5`), so the class signal sits in non-linear main effects.
pub fn spread_table(p: usize, rows: usize, seed: u64) -> Table {
    let mut rng = Rng(seed);
    let mut all = Table {
        x: Vec::new(),
        y: Vec::new(),
        p,
    };
    for _ in 0..rows {
        let class = (5.0 * rng.unit()) as usize;
        for j in 0..p {
            let wide = if j % 5 == class { 2.6 } else { 1.0 };
            all.x.push(wide * rng.normal());
        }
        all.y.push(class as f64);
    }
    all
}

pub fn multiclass_fixture() -> Fixture {
    let all = multiclass_table(20, FIT_ROWS + TEST_ROWS, 0x3c_0003);
    split("multiclass", "multiclass", 5, all)
}

pub fn fixtures() -> Vec<Fixture> {
    vec![regression_fixture(), binary_fixture(), multiclass_fixture()]
}

/// `fit` through the command line; returns the JSON report.
pub fn fit(
    table: &Table,
    task: &str,
    classes: usize,
    budget: usize,
    packet: &Path,
    card: &Path,
) -> serde_json::Value {
    let input = packet.with_extension("fit.mat");
    write_mat(&input, table);
    ok_json(&[
        "fit",
        "--in",
        text(&input),
        "--task",
        task,
        "--classes",
        &classes.to_string(),
        "--budget",
        &budget.to_string(),
        "--out",
        text(packet),
        "--card",
        text(card),
    ])
}

/// `sample` through the command line: rows `row0 .. row0 + rows` in raw units.
pub fn sample(
    packet: &Path,
    card: &Path,
    rows: usize,
    row0: usize,
    seed: u64,
    labels: &str,
) -> Table {
    let out = packet.with_extension(format!("s{seed}.r{row0}.{labels}.mat"));
    ok_json(&[
        "sample",
        "--packet",
        text(packet),
        "--card",
        text(card),
        "--rows",
        &rows.to_string(),
        "--row0",
        &row0.to_string(),
        "--seed",
        &seed.to_string(),
        "--labels",
        labels,
        "--out",
        text(&out),
    ]);
    read_mat(&out)
}

/// `truncate` through the command line; returns the JSON report.
pub fn truncate(packet: &Path, level: &str, out: &Path) -> serde_json::Value {
    ok_json(&[
        "truncate",
        "--packet",
        text(packet),
        "--level",
        level,
        "--out",
        text(out),
    ])
}

/// The 64-D and 128-D embeddings of a packet through the command line.
pub fn embed(packet: &Path) -> (Vec<i8>, Vec<i8>) {
    let report = ok_json(&["embed", "--packet", text(packet)]);
    let list = |key: &str| -> Vec<i8> {
        report[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i8)
            .collect()
    };
    (list("embedding_64"), list("embedding_128"))
}

pub fn cosine(a: &[i8], b: &[i8]) -> f64 {
    let dot =
        |x: &[i8], y: &[i8]| -> f64 { x.iter().zip(y).map(|(&u, &v)| u as f64 * v as f64).sum() };
    dot(a, b) / (dot(a, a) * dot(b, b)).sqrt()
}

/// Share of rows with every class label.
pub fn class_shares(table: &Table, classes: usize) -> Vec<f64> {
    let mut share = vec![0.0; classes];
    table.y.iter().for_each(|&y| share[y as usize] += 1.0);
    share.iter_mut().for_each(|s| *s /= table.rows() as f64);
    share
}

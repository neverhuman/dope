//! Levels probe: receipt format, replay summaries and the driver that produces receipts.
//!
//! For each table the driver fits DOPE independently at 2/4/16/64 KiB plus one program at the
//! summed low+med+high file bytes, and records bytes, the marginal compression of the level
//! chain (`|brotli(A‖B)| − |brotli(A)|` on inflated containers, an estimate rather than a framed
//! codec), robust retention on the held-out file, and retention of pooled low/med/high samples.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::stats::median;
use crate::{fmt_int, Result};

mod run;

pub use run::{brotli_len, byte_columns, inflate, probe_table, ByteColumns};

pub const LEVELS: [&str; 4] = ["low", "med", "high", "ultra"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level {
    pub bytes: u64,
    pub inflated: u64,
    pub metadata: u64,
    pub brotli_alone: u64,
    pub trees: serde_json::Value,
    pub dependence: serde_json::Value,
    pub regimes: serde_json::Value,
    pub plan: serde_json::Value,
    pub self_audit: Option<f64>,
    pub robust: f64,
    pub per_auditor: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerLevel<T> {
    pub low: T,
    pub med: T,
    pub high: T,
    pub ultra: T,
}

impl<T> PerLevel<T> {
    pub fn get(&self, level: &str) -> &T {
        match level {
            "low" => &self.low,
            "med" => &self.med,
            "high" => &self.high,
            _ => &self.ultra,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleAtSum {
    pub budget: u64,
    pub bytes: u64,
    pub robust: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pooled {
    pub robust: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelsRow {
    pub table: String,
    pub levels: PerLevel<Level>,
    pub conditional_bytes: PerLevel<i64>,
    pub independent_sum_lmh: i64,
    pub conditional_sum_lmh: i64,
    pub single_at_sum: SingleAtSum,
    pub pooled_lmh: Pooled,
}

pub fn load(path: &Path) -> Result<Vec<LevelsRow>> {
    crate::load_jsonl(path)
}

/// Carry-forward by self-audit: move up only when the new level's self-audit beats the
/// incumbent's by `margin`. Returns the robust retention released at each level.
pub fn policy(r: &LevelsRow, margin: f64) -> [f64; 4] {
    let mut cur: Option<&str> = None;
    let mut out = [0.0; 4];
    for (k, l) in LEVELS.iter().enumerate() {
        let v = r.levels.get(l).self_audit.unwrap_or(0.0);
        let take = match cur {
            None => true,
            Some(c) => v >= r.levels.get(c).self_audit.unwrap_or(0.0) + margin,
        };
        if take {
            cur = Some(l);
        }
        out[k] = r.levels.get(cur.unwrap_or(l)).robust;
    }
    out
}

pub fn independent(r: &LevelsRow) -> [f64; 4] {
    LEVELS.map(|l| r.levels.get(l).robust)
}

/// Adjacent steps where retention drops by more than `tol`.
pub fn violations(seq: &[f64], tol: f64) -> usize {
    seq.windows(2).filter(|w| w[1] < w[0] - tol).count()
}

#[derive(Debug, Clone, Copy)]
pub struct Scheme {
    pub median: [f64; 4],
    pub mean: [f64; 4],
    pub violations: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Summary {
    pub independent: Scheme,
    pub carry_0: Scheme,
    pub carry_005: Scheme,
    pub chain_saving_median: f64,
    pub chain_saving_min: f64,
    pub chain_saving_max: f64,
    pub pool_median: f64,
    pub pool_beats_ultra: usize,
    pub pool_beats_best_single: usize,
    pub tables: usize,
}

fn scheme(rows: &[LevelsRow], f: impl Fn(&LevelsRow) -> [f64; 4]) -> Scheme {
    let per: Vec<[f64; 4]> = rows.iter().map(f).collect();
    let col = |k: usize| per.iter().map(|d| d[k]).collect::<Vec<_>>();
    Scheme {
        median: [0, 1, 2, 3].map(|k| median(&col(k))),
        mean: [0, 1, 2, 3].map(|k| col(k).iter().sum::<f64>() / per.len() as f64),
        violations: per.iter().map(|d| violations(d, 0.02)).sum(),
    }
}

pub fn summary(rows: &[LevelsRow]) -> Summary {
    let savings: Vec<f64> = rows
        .iter()
        .map(|r| 1.0 - r.conditional_sum_lmh as f64 / r.independent_sum_lmh as f64)
        .collect();
    let pool: Vec<f64> = rows.iter().map(|r| r.pooled_lmh.robust).collect();
    let best_lower = |r: &LevelsRow| {
        ["low", "med", "high"]
            .iter()
            .map(|l| r.levels.get(l).robust)
            .fold(f64::NEG_INFINITY, f64::max)
    };
    Summary {
        independent: scheme(rows, independent),
        carry_0: scheme(rows, |r| policy(r, 0.0)),
        carry_005: scheme(rows, |r| policy(r, 0.005)),
        chain_saving_median: median(&savings),
        chain_saving_min: savings.iter().copied().fold(f64::INFINITY, f64::min),
        chain_saving_max: savings.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        pool_median: median(&pool),
        pool_beats_ultra: rows
            .iter()
            .filter(|r| r.pooled_lmh.robust > r.levels.ultra.robust)
            .count(),
        pool_beats_best_single: rows
            .iter()
            .filter(|r| r.pooled_lmh.robust > best_lower(r))
            .count(),
        tables: rows.len(),
    }
}

fn join3(d: [f64; 4]) -> String {
    d.map(|x| format!("{x:.3}")).join(" / ")
}

/// The levels tables in Markdown.
pub fn render(rows: &[LevelsRow]) -> String {
    let mut s = String::new();
    s += "| Table | low | med | high | ultra | carry (no margin) low/med/high/ultra | carry (margin 0.005) |\n";
    s += "|---|---:|---:|---:|---:|---|---|\n";
    for r in rows {
        let cells: Vec<String> = LEVELS
            .iter()
            .map(|l| {
                let v = r.levels.get(l);
                format!("{:.3} ({} B)", v.robust, fmt_int(v.bytes as i64))
            })
            .collect();
        s += &format!(
            "| {} | {} | {} | {} |\n",
            r.table,
            cells.join(" | "),
            join3(policy(r, 0.0)),
            join3(policy(r, 0.005))
        );
    }
    let sm = summary(rows);
    s += "\n| Scheme | median low/med/high/ultra | mean low/med/high/ultra | drops > 0.02 |\n";
    s += "|---|---|---|---:|\n";
    for (name, sc) in [
        ("independent", sm.independent),
        ("carry, no margin", sm.carry_0),
        ("carry, margin 0.005", sm.carry_005),
    ] {
        s += &format!(
            "| {name} | {} | {} | {} |\n",
            join3(sc.median),
            join3(sc.mean),
            sc.violations
        );
    }
    s += "\n| Table | low | med | high | ultra | indep. low+med+high | marginal chain | saving |\n";
    s += "|---|---:|---:|---:|---:|---:|---:|---:|\n";
    for r in rows {
        let cells: Vec<String> = LEVELS
            .iter()
            .map(|l| {
                format!(
                    "{} → {}",
                    fmt_int(r.levels.get(l).brotli_alone as i64),
                    fmt_int(*r.conditional_bytes.get(l))
                )
            })
            .collect();
        s += &format!(
            "| {} | {} | {} | {} | {:.0}% |\n",
            r.table,
            cells.join(" | "),
            fmt_int(r.independent_sum_lmh),
            fmt_int(r.conditional_sum_lmh),
            100.0 * (1.0 - r.conditional_sum_lmh as f64 / r.independent_sum_lmh as f64)
        );
    }
    s += "\n| Table | high (16 KiB) | ultra (64 KiB) | single fit at low+med+high bytes | pooled low/med/high samples |\n";
    s += "|---|---:|---:|---:|---:|\n";
    for r in rows {
        s += &format!(
            "| {} | {:.3} | {:.3} | {:.3} ({} B cap) | {:.3} |\n",
            r.table,
            r.levels.high.robust,
            r.levels.ultra.robust,
            r.single_at_sum.robust,
            fmt_int(r.single_at_sum.budget as i64),
            r.pooled_lmh.robust
        );
    }
    s += &format!(
        "\npool > ultra: {}/{} | pool > best single level: {} | chain saving median {:.0}% ({:.0}%–{:.0}%)\n",
        sm.pool_beats_ultra,
        sm.tables,
        sm.pool_beats_best_single,
        100.0 * sm.chain_saving_median,
        100.0 * sm.chain_saving_min,
        100.0 * sm.chain_saving_max
    );
    s
}

/// Compare two levels runs table by table: identical programs (file bytes and self-audit) and
/// the change in robust retention, for the four levels and the single fit at the summed bytes.
pub fn render_compare(a: &[LevelsRow], b: &[LevelsRow]) -> String {
    let mut s = String::from(
        "| Table | identical programs (low/med/high/ultra) | robust retention, first run | robust retention, second run | largest change |\n|---|---|---|---|---:|\n",
    );
    let (mut same, mut total, mut worst) = (0, 0, 0.0f64);
    for ra in a {
        let Some(rb) = b.iter().find(|r| r.table == ra.table) else {
            continue;
        };
        let flags: Vec<&str> = LEVELS
            .iter()
            .map(|l| {
                let (x, y) = (ra.levels.get(l), rb.levels.get(l));
                total += 1;
                if x.bytes == y.bytes && x.self_audit == y.self_audit && x.plan == y.plan {
                    same += 1;
                    "yes"
                } else {
                    "no"
                }
            })
            .collect();
        let (ia, ib) = (independent(ra), independent(rb));
        let change = ia
            .iter()
            .zip(&ib)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        worst = worst.max(change);
        s += &format!(
            "| {} | {} | {} | {} | {change:.3} |\n",
            ra.table,
            flags.join("/"),
            join3(ia),
            join3(ib)
        );
    }
    let common = |x: &[LevelsRow], y: &[LevelsRow]| -> Vec<LevelsRow> {
        x.iter()
            .filter(|r| y.iter().any(|o| o.table == r.table))
            .cloned()
            .collect()
    };
    let (sa, sb) = (summary(&common(a, b)), summary(&common(b, a)));
    s += &format!(
        "\nidentical programs: {same}/{total}; largest per-cell change {worst:.3}\nover the {} common tables, median robust retention low/med/high/ultra: first {}, second {}\n",
        sa.tables,
        join3(sa.independent.median),
        join3(sb.independent.median)
    );
    s += &format!(
        "carry (margin 0.005) mean: first {}, second {}; drops > 0.02: independent {} vs {}, carry {} vs {}\n",
        join3(sa.carry_005.mean),
        join3(sb.carry_005.mean),
        sa.independent.violations,
        sb.independent.violations,
        sa.carry_005.violations,
        sb.carry_005.violations
    );
    s
}

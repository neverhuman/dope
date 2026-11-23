//! Guard fields merged into an arm's rows when `rundir` loads it.
//!
//! When `score/<arm>/guards.jsonl` exists (written by `dope_kpi guards`),
//! every row takes `realism_pass` and `privacy_pass` from the guards line
//! with its `ledger_id`, and `exact_copy_excess` from `guards-summary.json`:
//! the excess of the row's synthetic file, absent when no complete entry of
//! the file has exact-copy counts. The summary must carry the SHA-256 of this
//! rows file and of the guards file, every row must find its line (same cell
//! and status) and every line its row, and a row that already carries a guard
//! field is refused. Errors name files and lines, never a value. Without
//! `guards.jsonl` the rows stay as they are.

use super::jsonl::{parse_json, parse_jsonl_numbered, read_file, sha256_hex};
use super::rundir::{contract, KpiRow, ROWS_FILE, SCORE_DIR};
use dope_arena::ArenaResult;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

pub const GUARDS_FILE: &str = "guards.jsonl";
pub const GUARDS_SUMMARY_FILE: &str = "guards-summary.json";
pub const GUARDS_SCHEMA: &str = "dope.kpi-guards/v2";

/// The id of a synthetic file: its ledger entry ids without the auditor.
pub fn file_id(cell_id: &str, sample_seed: u64, multiplier: u32) -> String {
    format!("{cell_id}:s{sample_seed}:m{multiplier}")
}

/// The guard files merged into one arm.
#[derive(Clone, Debug, PartialEq)]
pub struct GuardsRef {
    pub guards_sha256: String,
    pub summary_sha256: String,
}

/// The fields of a guards line that the rows take.
#[derive(Deserialize)]
struct LineFields {
    ledger_id: String,
    cell_id: String,
    status: String,
    realism_pass: bool,
    privacy_pass: bool,
}

/// The fields of `guards-summary.json` the merge reads.
#[derive(Deserialize)]
struct SummaryFields {
    schema: String,
    arm_id: String,
    rows_sha256: String,
    guards_sha256: String,
    file_exact_copy_excess: BTreeMap<String, f64>,
}

/// Merge the guard files of `arm_id` into `rows` (`lines[i]` is the line of
/// `rows[i]` in the rows file whose SHA-256 is `rows_sha256`).
pub fn merge(
    dir: &Path,
    arm_id: &str,
    rows_sha256: &str,
    lines: &[usize],
    rows: &mut [KpiRow],
) -> ArenaResult<Option<GuardsRef>> {
    let arm_dir = dir.join(SCORE_DIR).join(arm_id);
    let path = arm_dir.join(GUARDS_FILE);
    if !path.is_file() {
        return Ok(None);
    }
    let at = |file: &str| format!("{SCORE_DIR}/{arm_id}/{file}");
    let (guards, summary_file, rows_file) =
        (at(GUARDS_FILE), at(GUARDS_SUMMARY_FILE), at(ROWS_FILE));
    let bytes = read_file(&path)?;
    let summary_bytes = read_file(&arm_dir.join(GUARDS_SUMMARY_FILE))?;
    let summary: SummaryFields = parse_json(&summary_file, &summary_bytes)?;
    if summary.schema != GUARDS_SCHEMA || summary.arm_id != arm_id {
        return contract(format!(
            "{summary_file} is not a {GUARDS_SCHEMA} summary of this arm"
        ));
    }
    let guards_sha256 = sha256_hex(&bytes);
    if summary.guards_sha256 != guards_sha256 {
        return contract(format!(
            "{summary_file} does not describe {guards}; rerun dope_kpi guards"
        ));
    }
    if summary.rows_sha256 != rows_sha256 {
        return contract(format!(
            "{guards} was written for another {rows_file}; rerun dope_kpi guards"
        ));
    }
    let mut by_id: BTreeMap<String, (usize, LineFields, bool)> = BTreeMap::new();
    for (number, line) in parse_jsonl_numbered::<LineFields>(&guards, &bytes)? {
        if by_id
            .insert(line.ledger_id.clone(), (number, line, false))
            .is_some()
        {
            return contract(format!("{guards}: line {number} repeats an entry"));
        }
    }
    for (row, &number) in rows.iter_mut().zip(lines) {
        let Some(id) = &row.ledger_id else {
            return contract(format!("{rows_file}: line {number} has no ledger_id"));
        };
        let file = file_id(&row.cell_id, row.sample_seed, row.multiplier);
        if *id != format!("{file}:{}", row.auditor) {
            return contract(format!(
                "{rows_file}: line {number} has a ledger_id that is not its cell, seed, multiplier and auditor"
            ));
        }
        let Some((line_number, line, used)) = by_id.get_mut(id) else {
            return contract(format!(
                "{rows_file}: line {number} has no line in {guards}"
            ));
        };
        if *used || line.cell_id != row.cell_id || line.status != row.status {
            return contract(format!(
                "{guards}: line {line_number} does not match {rows_file} line {number}"
            ));
        }
        if row.realism_pass.is_some()
            || row.privacy_pass.is_some()
            || row.exact_copy_excess.is_some()
        {
            return contract(format!(
                "{rows_file}: line {number} already carries guard fields beside {guards}"
            ));
        }
        *used = true;
        row.realism_pass = Some(line.realism_pass);
        row.privacy_pass = Some(line.privacy_pass);
        row.exact_copy_excess = summary.file_exact_copy_excess.get(&file).copied();
    }
    let mut unused: Vec<usize> = by_id
        .values()
        .filter(|(_, _, used)| !used)
        .map(|(n, ..)| *n)
        .collect();
    unused.sort_unstable();
    if let Some(number) = unused.first() {
        return contract(format!(
            "{guards}: line {number} matches no row of {rows_file}"
        ));
    }
    Ok(Some(GuardsRef {
        guards_sha256,
        summary_sha256: sha256_hex(&summary_bytes),
    }))
}

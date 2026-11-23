//! `dope_kpi guards`: the realism and privacy gates and the exact-copy excess
//! of scored arms, read from `score/<arm>/ledger.jsonl` (`guard_eval.rs`).
//!
//! Per arm, in `<run-dir>/score/<arm>/`:
//! - `guards.jsonl`: one line per ledger entry, in ledger order
//!   (`guard_eval::GuardLine`: `ledger_id`, `cell_id`, `status`, both pass
//!   flags, the exact-copy counts and dope-core's assessments);
//! - `guards-summary.json` (`dope.kpi-guards/v2`): counts, pass shares, the
//!   exact-copy excess `max(0, rate(S in A) - rate(B in A))` of every
//!   synthetic file (cell, sample seed, multiplier), the rates pooled over
//!   the file's complete entries with an assessed privacy input, the same
//!   pooled over the whole arm (descriptive only), and the SHA-256 of the
//!   ledger, the rows and `guards.jsonl` and the tool.
//!
//! The ledger is streamed one synthetic file at a time, `--threads` files in
//! parallel, so memory stays bounded and the bytes do not depend on the
//! thread count. `rows.jsonl` must hold exactly the ledger's entries with
//! the same cells and statuses. Scored rows are never rewritten: `rundir`
//! merges the guard fields when it loads the arm (`guard_merge.rs`).

use super::emitlog::check_arm_id;
use super::guard_eval::{assess_group, GroupResult, GuardLine};
use super::guard_merge::{file_id, GUARDS_FILE, GUARDS_SCHEMA, GUARDS_SUMMARY_FILE};
use super::jsonl::{
    canonical_json, canonical_pretty, file_error, parse_jsonl_numbered, parse_line, read_file,
    sha256_hex, write_atomic, write_lines,
};
use super::par::par_map;
use super::receipt::{tool, Tool};
use super::rundir::{contract, ROWS_FILE, SCORE_DIR, STATUS_OK};
use super::score::{parse_ledger_id, LEDGER_FILE};
use dope_arena::ArenaResult;
use dope_core::dvs::v3::FidelityCellRequest;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(clap::Args, Debug)]
pub struct GuardsArgs {
    /// Run directory holding `score/<arm>/ledger.jsonl` and `rows.jsonl`.
    #[arg(long)]
    pub run_dir: PathBuf,
    /// Arm to assess (repeat for several).
    #[arg(long = "arm", required = true)]
    pub arms: Vec<String>,
    /// Threads assessing synthetic files in parallel.
    #[arg(long, default_value_t = 1)]
    pub threads: usize,
}

/// One gate over the entries of an arm.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct GateCounts {
    pub passed: usize,
    /// Complete entries whose input dope-core assessed.
    pub assessed: usize,
    /// Complete entries without the input (they fail the gate).
    pub absent: usize,
    /// Complete entries whose input dope-core refused, by code (they fail the gate).
    pub refused: BTreeMap<String, usize>,
    /// Distinct inputs assessed: the entries of one file share one assessment.
    pub assessments: usize,
    /// `passed` over every entry (an entry that is not complete fails).
    pub pass_share: Option<f64>,
    /// `passed` over the complete entries.
    pub pass_share_complete: Option<f64>,
}

impl GateCounts {
    fn add(&mut self, pass: bool, assessed: bool, refusal: Option<&str>) {
        self.passed += usize::from(pass);
        self.assessed += usize::from(assessed);
        match refusal {
            Some(code) => *self.refused.entry(code.to_string()).or_default() += 1,
            None => self.absent += usize::from(!assessed),
        }
    }

    fn finish(&mut self, entries: usize, complete: usize) {
        let share = |n: usize| (n > 0).then(|| self.passed as f64 / n as f64);
        (self.pass_share, self.pass_share_complete) = (share(entries), share(complete));
    }
}

/// Exact-copy counts pooled over complete entries with an assessed privacy input.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ExactCopyPool {
    pub entries: usize,
    pub synthetic_rows: u64,
    pub exact_copies: u64,
    pub twin_rows: u64,
    pub twin_exact_copies: u64,
    pub rate_s_in_a: Option<f64>,
    pub rate_b_in_a: Option<f64>,
}

impl ExactCopyPool {
    fn add(&mut self, line: &GuardLine) {
        let counts = (
            line.synthetic_rows,
            line.exact_copies,
            line.twin_rows,
            line.twin_exact_copies,
        );
        if let (Some(rows), Some(copies), Some(twins), Some(twin_copies)) = counts {
            self.entries += 1;
            self.synthetic_rows += rows;
            self.exact_copies += copies;
            self.twin_rows += twins;
            self.twin_exact_copies += twin_copies;
        }
    }

    /// `max(0, rate(S in A) - rate(B in A))`, absent without counts.
    fn finish(&mut self) -> Option<f64> {
        let rate = |hits: u64, rows: u64| (rows > 0).then(|| hits as f64 / rows as f64);
        self.rate_s_in_a = rate(self.exact_copies, self.synthetic_rows);
        self.rate_b_in_a = rate(self.twin_exact_copies, self.twin_rows);
        Some((self.rate_s_in_a? - self.rate_b_in_a?).max(0.0))
    }
}

/// `guards-summary.json`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GuardsSummary {
    pub schema: String,
    pub arm_id: String,
    pub tool: Tool,
    /// How the flags and the excess were decided.
    pub rule: BTreeMap<String, String>,
    pub ledger_sha256: String,
    pub rows_sha256: String,
    pub guards_sha256: String,
    pub entries: usize,
    pub by_status: BTreeMap<String, usize>,
    pub realism: GateCounts,
    pub privacy: GateCounts,
    /// Excess of every synthetic file `<cell>:s<seed>:m<multiplier>` with
    /// counts; each row takes its file's value.
    pub file_exact_copy_excess: BTreeMap<String, f64>,
    /// Counts pooled over every complete entry of the arm (descriptive).
    pub exact_copy: ExactCopyPool,
    pub pooled_exact_copy_excess: Option<f64>,
}

fn rule() -> BTreeMap<String, String> {
    [
        ("realism", "dope_core::dvs::v3::realism::assess_realism pass_standard"),
        ("privacy", "dope_core::dvs::v3::privacy::assess_empirical_privacy pass (empirical, not formal DP or HIPAA)"),
        ("absent_or_refused_input", "fails the gate (evaluate_v3 without partial evidence)"),
        ("entry_not_complete", "fails both gates, no exact-copy counts"),
        ("file_exact_copy_excess", "per synthetic file, max(0, sum exact_copies / sum synthetic_rows - sum twin_exact_copies / sum twin_rows) over its complete entries"),
        ("pooled_exact_copy_excess", "the same over every complete entry of the arm (descriptive, read by no guard)"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// One row of `rows.jsonl`, as the join with the ledger reads it.
#[derive(Deserialize)]
struct RowKey {
    #[serde(default)]
    ledger_id: Option<String>,
    cell_id: String,
    status: String,
}

/// A synthetic file: (cell, sample seed, multiplier).
type FileKey = (String, u64, u32);

/// One ledger entry: its line number, the request and its file.
type LedgerEntry = (usize, FidelityCellRequest, FileKey);

/// Consecutive ledger entries of one synthetic file, with their line numbers.
struct Group {
    key: FileKey,
    entries: Vec<(usize, FidelityCellRequest)>,
}

impl Group {
    fn assess(&self) -> GroupResult {
        let requests: Vec<&FidelityCellRequest> = self.entries.iter().map(|(_, r)| r).collect();
        assess_group(&self.key.0, &requests)
    }
}

/// The ledger read line by line, hashed as it is read.
struct LedgerReader {
    reader: BufReader<std::fs::File>,
    path: PathBuf,
    label: String,
    hasher: Sha256,
    line: usize,
    buf: Vec<u8>,
}

impl LedgerReader {
    fn open(path: &Path, label: String) -> ArenaResult<LedgerReader> {
        let file = std::fs::File::open(path).map_err(|e| file_error(path, &e))?;
        Ok(LedgerReader {
            reader: BufReader::with_capacity(1 << 20, file),
            path: path.to_path_buf(),
            label,
            hasher: Sha256::new(),
            line: 0,
            buf: Vec::new(),
        })
    }

    fn next(&mut self) -> ArenaResult<Option<LedgerEntry>> {
        loop {
            self.buf.clear();
            let got = self.reader.read_until(b'\n', &mut self.buf);
            if got.map_err(|e| file_error(&self.path, &e))? == 0 {
                return Ok(None);
            }
            self.hasher.update(&self.buf);
            self.line += 1;
            if self.buf.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let request: FidelityCellRequest = parse_line(&self.label, self.line, &self.buf)?;
            let Some(key) = parse_ledger_id(&request.cell_id, &request.auditor_id) else {
                return contract(format!(
                    "{}: line {}: the entry id is not <cell>:s<seed>:m<multiplier>:<auditor>",
                    self.label, self.line
                ));
            };
            return Ok(Some((self.line, request, key)));
        }
    }
}

/// Every entry of the ledger, `threads` files at a time; `take` sees each group's lines in order.
fn stream_ledger(
    reader: &mut LedgerReader,
    threads: usize,
    take: &mut dyn FnMut(&Group, GroupResult) -> ArenaResult<()>,
) -> ArenaResult<()> {
    let batch = threads.max(1) * 2;
    let mut groups: Vec<Group> = Vec::new();
    let mut flush = |groups: &mut Vec<Group>| -> ArenaResult<()> {
        let shared: &[Group] = groups;
        let results = par_map(threads, shared.len(), |i| shared[i].assess());
        for (group, result) in shared.iter().zip(results) {
            take(group, result)?;
        }
        groups.clear();
        Ok(())
    };
    while let Some((line, request, key)) = reader.next()? {
        match groups.last_mut() {
            Some(group) if group.key == key => group.entries.push((line, request)),
            _ => {
                if groups.len() >= batch {
                    flush(&mut groups)?;
                }
                groups.push(Group {
                    key,
                    entries: vec![(line, request)],
                });
            }
        }
    }
    flush(&mut groups)
}

/// Assess one arm and write its two guard files; returns the summary and its SHA-256.
pub fn guard_arm(
    run_dir: &Path,
    arm_id: &str,
    threads: usize,
) -> ArenaResult<(GuardsSummary, String)> {
    check_arm_id(arm_id)?;
    let arm_dir = run_dir.join(SCORE_DIR).join(arm_id);
    let at = |file: &str| format!("{SCORE_DIR}/{arm_id}/{file}");
    let (rows_file, ledger_file) = (at(ROWS_FILE), at(LEDGER_FILE));
    let rows_bytes = read_file(&arm_dir.join(ROWS_FILE))?;
    let mut rows: BTreeMap<String, (usize, RowKey)> = BTreeMap::new();
    for (number, row) in parse_jsonl_numbered::<RowKey>(&rows_file, &rows_bytes)? {
        let Some(id) = row.ledger_id.clone() else {
            return contract(format!("{rows_file}: line {number} has no ledger_id"));
        };
        if rows.insert(id, (number, row)).is_some() {
            return contract(format!("{rows_file}: line {number} repeats an entry"));
        }
    }
    let mut summary = GuardsSummary {
        schema: GUARDS_SCHEMA.to_string(),
        arm_id: arm_id.to_string(),
        tool: tool(),
        rule: rule(),
        ledger_sha256: String::new(),
        rows_sha256: sha256_hex(&rows_bytes),
        guards_sha256: String::new(),
        entries: 0,
        by_status: BTreeMap::new(),
        realism: GateCounts::default(),
        privacy: GateCounts::default(),
        file_exact_copy_excess: BTreeMap::new(),
        exact_copy: ExactCopyPool::default(),
        pooled_exact_copy_excess: None,
    };
    let mut rendered: Vec<String> = Vec::new();
    let mut files: BTreeMap<String, ExactCopyPool> = BTreeMap::new();
    let mut reader = LedgerReader::open(&arm_dir.join(LEDGER_FILE), ledger_file.clone())?;
    stream_ledger(&mut reader, threads, &mut |group, result| {
        summary.realism.assessments += result.realism_assessments;
        summary.privacy.assessments += result.privacy_assessments;
        let (cell, seed, multiplier) = &group.key;
        let file = file_id(cell, *seed, *multiplier);
        for ((number, _), line) in group.entries.iter().zip(result.lines) {
            let Some((row_number, row)) = rows.remove(&line.ledger_id) else {
                return contract(format!(
                    "{ledger_file}: line {number} has no row in {rows_file}"
                ));
            };
            if row.cell_id != line.cell_id || row.status != line.status {
                return contract(format!(
                    "{ledger_file}: line {number} does not match {rows_file} line {row_number}"
                ));
            }
            summary.entries += 1;
            *summary
                .by_status
                .entry(line.status.to_string())
                .or_default() += 1;
            if line.status == STATUS_OK {
                summary.realism.add(
                    line.realism_pass,
                    line.realism.is_some(),
                    line.realism_refusal,
                );
                summary.privacy.add(
                    line.privacy_pass,
                    line.privacy.is_some(),
                    line.privacy_refusal,
                );
                summary.exact_copy.add(&line);
                files.entry(file.clone()).or_default().add(&line);
            }
            rendered.push(canonical_json(&line)?);
        }
        Ok(())
    })?;
    if let Some(number) = rows.values().map(|(n, _)| *n).min() {
        return contract(format!(
            "{rows_file}: line {number} has no entry in {ledger_file}"
        ));
    }
    summary.ledger_sha256 = format!("{:x}", reader.hasher.finalize());
    let complete = summary.by_status.get(STATUS_OK).copied().unwrap_or(0);
    summary.realism.finish(summary.entries, complete);
    summary.privacy.finish(summary.entries, complete);
    summary.file_exact_copy_excess = files
        .into_iter()
        .filter_map(|(file, mut pool)| pool.finish().map(|excess| (file, excess)))
        .collect();
    summary.pooled_exact_copy_excess = summary.exact_copy.finish();
    summary.guards_sha256 = write_lines(
        &arm_dir.join(GUARDS_FILE),
        rendered.iter().map(String::as_str),
    )?;
    let text = canonical_pretty(&summary)?;
    write_atomic(&arm_dir.join(GUARDS_SUMMARY_FILE), text.as_bytes())?;
    Ok((summary, sha256_hex(text.as_bytes())))
}

#[derive(Serialize)]
struct ArmLine {
    entries: usize,
    realism_pass_share_complete: Option<f64>,
    privacy_pass_share_complete: Option<f64>,
    pooled_exact_copy_excess: Option<f64>,
    guards_sha256: String,
    summary_sha256: String,
}

#[derive(Serialize)]
struct Summary {
    command: &'static str,
    arms: BTreeMap<String, ArmLine>,
}

pub fn run(args: &GuardsArgs) -> ArenaResult<String> {
    for arm in &args.arms {
        check_arm_id(arm)?;
    }
    let mut arms = BTreeMap::new();
    for arm in &args.arms {
        if arms.contains_key(arm) {
            continue;
        }
        let (summary, summary_sha256) = guard_arm(&args.run_dir, arm, args.threads)?;
        arms.insert(
            arm.clone(),
            ArmLine {
                entries: summary.entries,
                realism_pass_share_complete: summary.realism.pass_share_complete,
                privacy_pass_share_complete: summary.privacy.pass_share_complete,
                pooled_exact_copy_excess: summary.pooled_exact_copy_excess,
                guards_sha256: summary.guards_sha256,
                summary_sha256,
            },
        );
    }
    canonical_json(&Summary {
        command: "guards",
        arms,
    })
}

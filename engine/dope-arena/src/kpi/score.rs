//! `dope_kpi score`: score one arm on every scheduled cell (formats.md section 7).
//!
//! Outputs in `--out`:
//! - `ledger.jsonl`: one `FidelityCellRequest` per cell x sample seed x
//!   multiplier x auditor. The existing ledger reader refuses a repeated
//!   `cell_id`, so the entry id is `<cell_id>:s<seed>:m<multiplier>:<auditor>`
//!   and the file is sorted by that id.
//! - `rows.jsonl`: one compact row per ledger entry, in the same order.
//!
//! The privacy input holds the A, B and synthetic rows in unit space. Its
//! `min_emitted_support` cannot be measured from rows: it is the value the
//! emitter declares, else the contract floor, so the privacy gate of an arm
//! that declares nothing rests on the distance tests alone.

use super::cellplan::Cell;
use super::controls;
use super::emitlog::{check_arm_id, index, load_arm, write_arm_log, EmitRow, EMIT_PART};
use super::io::Arena;
use super::jsonl::{canonical_json, read_file, read_jsonl, write_jsonl, write_lines};
use super::realref::{cell_sha256, RealRef};
use super::scorecell::{score_cell, Entry, ScoreContext};
use super::shard::{owned, run_workers, worker_count, Shard};
use dope_arena::{ArenaError, ArenaResult};
use dope_core::dvs::v3::{FidelityCellInput, FidelityCellRequest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

pub const LEDGER_FILE: &str = "ledger.jsonl";
pub const ROWS_FILE: &str = "rows.jsonl";

#[derive(clap::Args, Debug)]
pub struct ScoreArgs {
    /// Flat corpus arena directory (`dope.flat-corpus/v1`).
    #[arg(long)]
    pub arena: PathBuf,
    /// `schedule.jsonl` written by `plan`.
    #[arg(long)]
    pub schedule: PathBuf,
    /// `realref.jsonl` written by `real-ref` for the same schedule.
    #[arg(long)]
    pub realref: PathBuf,
    /// Run directory holding `emit.jsonl` and `<arm>/*.mat`; control arms write theirs here.
    #[arg(long)]
    pub emit_dir: PathBuf,
    /// Arm to score. `twin_c`, `copy_a`, `bootstrap_a`, `indep_marginals`,
    /// `label_shuffle` and `rust_format2[_b<bytes>]` are generated natively first.
    #[arg(long)]
    pub arm: String,
    /// Output directory for `ledger.jsonl` and `rows.jsonl`.
    #[arg(long)]
    pub out: PathBuf,
    /// Packets longer than this many bytes become over-budget entries.
    #[arg(long)]
    pub budget_bytes: Option<u64>,
    /// Rows of A, B and S kept in each privacy input (0 writes none).
    #[arg(long, default_value_t = 4096)]
    pub privacy_rows: usize,
    /// Worker processes (each computes on one thread).
    #[arg(long, default_value_t = 1)]
    pub threads: usize,
    /// Internal: score one shard of the schedule (set by the parent process).
    #[arg(long, hide = true)]
    pub shard: Option<Shard>,
}

/// One row of `rows.jsonl`. `rows`, `reason` and `ledger_id` extend formats.md section 7.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ScoreRow {
    pub cell_id: String,
    pub family_id: String,
    pub task: String,
    pub view: String,
    pub stratum: String,
    pub auditor: String,
    pub data_seed: u64,
    pub sample_seed: u64,
    pub multiplier: u32,
    pub l0: Option<f64>,
    pub l_a: Option<f64>,
    pub l_b: Option<f64>,
    pub l_s: Option<f64>,
    pub r_s: Option<f64>,
    pub r_b: Option<f64>,
    pub informative: bool,
    pub status: String,
    pub packet_bytes: u64,
    pub artifact_class: String,
    pub c2st_auc: Option<f64>,
    pub c2st_real_ref: Option<f64>,
    /// Synthetic rows read from the file (0 when none was read).
    pub rows: u64,
    pub reason: String,
    pub ledger_id: String,
}

#[derive(Debug, Serialize)]
struct Summary {
    command: &'static str,
    arm_id: String,
    cells: usize,
    auditors: Vec<String>,
    entries: usize,
    by_status: BTreeMap<String, usize>,
    by_reason: BTreeMap<String, usize>,
    informative_ok: usize,
    ledger_sha256: String,
    rows_sha256: String,
}

pub fn ledger_id(cell_id: &str, sample_seed: u64, multiplier: u32, auditor: &str) -> String {
    format!("{cell_id}:s{sample_seed}:m{multiplier}:{auditor}")
}

/// `(cell_id, sample_seed, multiplier)` of a ledger entry id.
pub fn parse_ledger_id(id: &str, auditor: &str) -> Option<(String, u64, u32)> {
    let rest = id.strip_suffix(auditor)?.strip_suffix(':')?;
    let (rest, multiplier) = rest.rsplit_once(":m")?;
    let (cell_id, seed) = rest.rsplit_once(":s")?;
    Some((
        cell_id.to_string(),
        seed.parse().ok()?,
        multiplier.parse().ok()?,
    ))
}

pub fn status_of(input: &FidelityCellInput) -> &'static str {
    match input {
        FidelityCellInput::Complete { .. } => "ok",
        FidelityCellInput::OverBudget { .. } => "over_budget",
        FidelityCellInput::Unsupported => "unsupported",
        _ => "failed",
    }
}

/// Reference rows of every cell, keyed by `cell_id`.
type ByCell<'a> = BTreeMap<&'a str, Vec<&'a RealRef>>;

/// Real references grouped by cell, checked against the schedule they must
/// match, and the auditor roster they share.
fn references_by_cell<'a>(
    cells: &[Cell],
    realref: &'a [RealRef],
) -> ArenaResult<(ByCell<'a>, Vec<String>)> {
    let mut by_cell: BTreeMap<&str, Vec<&RealRef>> = BTreeMap::new();
    for row in realref {
        by_cell.entry(&row.cell_id).or_default().push(row);
    }
    let mut roster: Option<Vec<String>> = None;
    for cell in cells {
        let sha = cell_sha256(cell)?;
        let rows = by_cell.get(cell.cell_id.as_str());
        let names: Vec<String> = rows
            .map(|rows| rows.iter().map(|r| r.auditor.clone()).collect())
            .unwrap_or_default();
        let pinned = rows.is_some_and(|rows| rows.iter().all(|r| r.cell_sha256 == sha));
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        let same_roster = roster.as_ref().is_none_or(|known| *known == names);
        if !pinned || names.is_empty() || sorted.len() != names.len() || !same_roster {
            return Err(ArenaError::Contract(format!(
                "realref does not match the schedule at cell {}",
                cell.cell_id
            )));
        }
        roster = Some(names);
    }
    if by_cell.len() != cells.len() {
        return Err(ArenaError::Contract(
            "realref holds cells that are not in the schedule".into(),
        ));
    }
    Ok((by_cell, roster.unwrap_or_default()))
}

/// One ledger entry as written: its id, its ledger line and its compact row line.
struct Line {
    id: String,
    ledger: String,
    row: String,
}

/// Summary fields of one row line.
#[derive(Deserialize)]
struct RowHead {
    status: String,
    reason: String,
    informative: bool,
}

fn render(cell: &Cell, reference: &RealRef, entry: Entry) -> ArenaResult<Line> {
    let id = ledger_id(
        &cell.cell_id,
        entry.sample_seed,
        entry.multiplier,
        &entry.auditor,
    );
    let complete = matches!(entry.input, FidelityCellInput::Complete { .. });
    let row = ScoreRow {
        cell_id: cell.cell_id.clone(),
        family_id: cell.family_id.clone(),
        task: cell.task.clone(),
        view: cell.view.clone(),
        stratum: cell.stratum.clone(),
        auditor: entry.auditor.clone(),
        data_seed: cell.data_seed,
        sample_seed: entry.sample_seed,
        multiplier: entry.multiplier,
        l0: reference.l0,
        l_a: reference.l_a,
        l_b: reference.l_b,
        l_s: entry.l_s,
        r_s: entry.r_s,
        r_b: reference.r_b,
        informative: reference.informative,
        status: status_of(&entry.input).to_string(),
        packet_bytes: entry.packet_bytes,
        artifact_class: entry.artifact_class,
        c2st_auc: entry.c2st_auc,
        c2st_real_ref: reference.c2st_real_ref,
        rows: entry.synthetic_rows,
        reason: if complete {
            String::new()
        } else {
            entry.reason
        },
        ledger_id: id.clone(),
    };
    let request = FidelityCellRequest {
        cell_id: id.clone(),
        family_id: cell.family_id.clone(),
        auditor_id: entry.auditor,
        input: entry.input,
    };
    Ok(Line {
        ledger: canonical_json(&request)?,
        row: canonical_json(&row)?,
        id,
    })
}

/// Lines of the cells at `positions`, scored in this process.
fn compute(
    ctx: &ScoreContext,
    cells: &[Cell],
    positions: &[usize],
    references: &ByCell,
) -> ArenaResult<Vec<Line>> {
    let mut lines = Vec::new();
    for &position in positions {
        let cell = &cells[position];
        let refs = &references[cell.cell_id.as_str()];
        let by_auditor: BTreeMap<&str, &RealRef> =
            refs.iter().map(|r| (r.auditor.as_str(), *r)).collect();
        for entry in score_cell(ctx, cell, refs)? {
            let reference = by_auditor[entry.auditor.as_str()];
            lines.push(render(cell, reference, entry)?);
        }
    }
    Ok(lines)
}

fn text_lines(path: &Path) -> ArenaResult<Vec<String>> {
    let text = String::from_utf8(read_file(path)?)
        .map_err(|_| ArenaError::Contract("worker output is not UTF-8".into()))?;
    Ok(text.lines().map(str::to_string).collect())
}

/// The same lines from `workers` processes, each writing one part directory.
fn compute_in_workers(args: &ScoreArgs, workers: usize) -> ArenaResult<Vec<Line>> {
    let part = |index: usize| args.out.join(format!(".shard-{index}"));
    // Parts left behind by an interrupted run are never read.
    if let Ok(entries) = std::fs::read_dir(&args.out) {
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy().starts_with(".shard-") {
                std::fs::remove_dir_all(entry.path())?;
            }
        }
    }
    run_workers(workers, |index| {
        let mut argv: Vec<OsString> = vec!["score".into()];
        let paths = [
            ("--arena", &args.arena),
            ("--schedule", &args.schedule),
            ("--realref", &args.realref),
            ("--emit-dir", &args.emit_dir),
            ("--out", &part(index)),
        ];
        for (flag, path) in paths {
            argv.push(flag.into());
            argv.push(path.clone().into());
        }
        argv.extend(["--arm".into(), args.arm.clone().into()]);
        argv.extend([
            "--privacy-rows".into(),
            args.privacy_rows.to_string().into(),
        ]);
        if let Some(cap) = args.budget_bytes {
            argv.extend(["--budget-bytes".into(), cap.to_string().into()]);
        }
        argv
    })?;
    // Workers of a native arm generate their own cells; their log rows are merged here.
    let native = controls::is_native(&args.arm);
    let (mut lines, mut emitted) = (Vec::new(), Vec::new());
    for index in 0..workers {
        let dir = part(index);
        let ledger = text_lines(&dir.join(LEDGER_FILE))?;
        let rows = text_lines(&dir.join(ROWS_FILE))?;
        if ledger.len() != rows.len() {
            return Err(ArenaError::Contract(
                "worker ledger and rows disagree".into(),
            ));
        }
        for (ledger, row) in ledger.into_iter().zip(rows) {
            let id = serde_json::from_str::<ScoreRow>(&row)?.ledger_id;
            lines.push(Line { id, ledger, row });
        }
        if native {
            emitted.extend(read_jsonl::<EmitRow>(&dir.join(EMIT_PART))?);
        }
        std::fs::remove_dir_all(&dir)?;
    }
    if native {
        write_arm_log(&args.emit_dir, &args.arm, &mut emitted)?;
    }
    Ok(lines)
}

pub fn run(args: &ScoreArgs) -> ArenaResult<String> {
    check_arm_id(&args.arm)?;
    let arena = Arena::open(&args.arena)?;
    let cells: Vec<Cell> = read_jsonl(&args.schedule)?;
    let realref: Vec<RealRef> = read_jsonl(&args.realref)?;
    let (references, roster) = references_by_cell(&cells, &realref)?;
    let workers = worker_count(args.threads, cells.len());
    let mut lines = if args.shard.is_none() && workers > 1 {
        compute_in_workers(args, workers)?
    } else {
        let positions = owned(args.shard, cells.len());
        let emits = if controls::is_native(&args.arm) {
            // A native arm is generated by the process that scores it.
            let (arm, dir) = (&args.arm, &args.emit_dir);
            let mut rows = controls::emit_cells(&arena, &cells, &positions, arm, dir)?;
            if args.shard.is_some() {
                write_jsonl(&args.out.join(EMIT_PART), &rows)?;
            } else {
                write_arm_log(dir, arm, &mut rows)?;
            }
            index(rows)?
        } else {
            load_arm(&args.emit_dir, &args.arm)?
        };
        let ctx = ScoreContext {
            arena: &arena,
            emit_dir: &args.emit_dir,
            arm_id: &args.arm,
            emits: &emits,
            budget_bytes: args.budget_bytes,
            privacy_rows: args.privacy_rows,
        };
        compute(&ctx, &cells, &positions, &references)?
    };
    lines.sort_by(|a, b| a.id.cmp(&b.id));

    let ledger_sha256 = write_lines(
        &args.out.join(LEDGER_FILE),
        lines.iter().map(|line| line.ledger.as_str()),
    )?;
    let rows_sha256 = write_lines(
        &args.out.join(ROWS_FILE),
        lines.iter().map(|line| line.row.as_str()),
    )?;
    let (mut by_status, mut by_reason) = (BTreeMap::new(), BTreeMap::new());
    let mut informative_ok = 0;
    for line in &lines {
        let head: RowHead = serde_json::from_str(&line.row)?;
        informative_ok += usize::from(head.informative && head.status == "ok");
        *by_status.entry(head.status).or_default() += 1;
        if !head.reason.is_empty() {
            *by_reason.entry(head.reason).or_default() += 1;
        }
    }
    canonical_json(&Summary {
        command: "score",
        arm_id: args.arm.clone(),
        cells: cells.len(),
        auditors: roster,
        entries: lines.len(),
        by_status,
        by_reason,
        informative_ok,
        ledger_sha256,
        rows_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_ids_are_unique_per_entry_and_parse_back() {
        let id = ledger_id("dev:0123:p48:d2", 7, 4, "linear-seeded-v1");
        assert_eq!(id, "dev:0123:p48:d2:s7:m4:linear-seeded-v1");
        assert_eq!(
            parse_ledger_id(&id, "linear-seeded-v1"),
            Some(("dev:0123:p48:d2".to_string(), 7, 4))
        );
        assert_eq!(parse_ledger_id(&id, "gbdt"), None);
        assert_eq!(parse_ledger_id("dev:0123:full:d0", "gbdt"), None);
        assert_eq!(parse_ledger_id("x:sA:m4:gbdt", "gbdt"), None);
        assert_ne!(id, ledger_id("dev:0123:p48:d2", 7, 1, "linear-seeded-v1"));
    }

    #[test]
    fn statuses_cover_every_ledger_input() {
        let failed = FidelityCellInput::Failed {
            reason: "missing_file".into(),
        };
        let over = FidelityCellInput::OverBudget {
            actual_bytes: 3000,
            cap_bytes: 2048,
        };
        assert_eq!(status_of(&failed), "failed");
        assert_eq!(status_of(&over), "over_budget");
        assert_eq!(status_of(&FidelityCellInput::Unsupported), "unsupported");
        assert_eq!(status_of(&FidelityCellInput::Missing), "failed");
        // Terminal inputs serialise to a tagged object the ledger reader accepts.
        let request = FidelityCellRequest {
            cell_id: "c:s0:m4:gbdt".into(),
            family_id: "lf".into(),
            auditor_id: "gbdt".into(),
            input: over,
        };
        let line = canonical_json(&request).unwrap();
        assert_eq!(
            line,
            "{\"auditor_id\":\"gbdt\",\"cell_id\":\"c:s0:m4:gbdt\",\"family_id\":\"lf\",\
             \"input\":{\"actual_bytes\":3000,\"cap_bytes\":2048,\"status\":\"over_budget\"}}"
        );
        serde_json::from_str::<FidelityCellRequest>(&line).unwrap();
    }
}

//! `dope_kpi emit-cli`: run an external generator on every scheduled cell
//! (`docs/plans/v2-gap-closure/ext-arms.md`).
//!
//! Writes `<out>/<arm_id>/emit.jsonl` with exactly one row per scheduled
//! (cell, sample seed, multiplier), sorted by that key, and
//! `<out>/<arm_id>/emit-summary.json`. A generator failure is a row, never a
//! skip and never an error of the run; the run fails only when its own
//! inputs or the disk do. Apart from `encode_ms` and `decode_ms`, the log and
//! every `DOPEMAT1` file are a pure function of the inputs and the bytes the
//! generator produced, for any `--threads`.
//!
//! The log is complete at every moment: before the first command it is
//! written with the cells that still have to run marked `run_interrupted`,
//! and the finished log replaces it in one rename. A log on disk therefore
//! never describes files that a later, unfinished run has replaced.

use super::cellplan::{Cell, STATUS_OK, STATUS_UNSUPPORTED};
use super::ext_arm::{load_arm, valid_arm_id, ARCH_ID};
use super::ext_cell::{
    cell_keys, file_stem, requested_rows, CellJob, EmitRow, LOG_DIR, PACKET_DIR, WORK_DIR,
};
use super::ext_reason::Reason;
use super::ext_scratch::remove_path;
use super::ext_summary::Summary;
use super::io::Arena;
use super::jsonl::{
    canonical_json, canonical_pretty, file_error, read_jsonl, write_atomic, write_jsonl,
};
use super::par::par_map;
use dope_arena::{ArenaError, ArenaResult};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub const EMIT_FILE: &str = "emit.jsonl";
pub const SUMMARY_FILE: &str = "emit-summary.json";

#[derive(clap::Args, Debug)]
pub struct EmitCliArgs {
    /// Flat corpus arena directory (`dope.flat-corpus/v1`).
    #[arg(long)]
    pub arena: PathBuf,
    /// `schedule.jsonl` written by `plan`.
    #[arg(long)]
    pub schedule: PathBuf,
    /// Run directory; everything is written under `<out>/<arm>/`.
    #[arg(long)]
    pub out: PathBuf,
    /// Arm file (`dope.kpi-ext-arms/v1`).
    #[arg(long)]
    pub arm_file: PathBuf,
    /// `arm_id` to run from the arm file.
    #[arg(long)]
    pub arm: String,
    /// Cells run in parallel; each cell uses one CPU thread.
    #[arg(long, default_value_t = 1)]
    pub threads: usize,
    /// Wall-clock limit of each fit and each sample command, in seconds.
    #[arg(long, default_value_t = 300.0)]
    pub timeout_s: f64,
    /// Keep cells whose rows are all `ok` in the existing log and re-run the others.
    #[arg(long)]
    pub resume: bool,
    /// Run only the first N cells of the schedule (a partial run, flagged in the summary).
    #[arg(long)]
    pub limit_cells: Option<usize>,
}

fn contract<T>(message: String) -> ArenaResult<T> {
    Err(ArenaError::Contract(message))
}

fn strictly_ascending(values: &[usize]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn distinct<T: Ord>(values: &[T]) -> bool {
    values.iter().collect::<BTreeSet<_>>().len() == values.len()
}

/// A cell must describe a table of this arena and be safe to turn into file names.
fn check_cell(arena: &Arena, position: usize, cell: &Cell) -> ArenaResult<()> {
    let fail = |field: &str| contract(format!("schedule: cell {position}: invalid {field}"));
    let Ok(table) = arena.table(cell.table_id) else {
        return fail("table_id");
    };
    if table.dataset_hash != cell.dataset_hash {
        return fail("dataset_hash");
    }
    if table.task != cell.task || table.n_classes != cell.n_classes {
        return fail("task");
    }
    let stem = file_stem(&cell.cell_id);
    let name_safe = stem
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    if stem.is_empty() || stem.starts_with('.') || !name_safe {
        return fail("cell_id");
    }
    let cols = &cell.cols;
    if cols.is_empty() || !strictly_ascending(cols) || cols.iter().any(|&c| c >= table.n_cols) {
        return fail("cols");
    }
    if cell.sample_seeds.is_empty() || !distinct(&cell.sample_seeds) {
        return fail("sample_seeds");
    }
    let multipliers = &cell.multipliers;
    if multipliers.is_empty() || !distinct(multipliers) || multipliers.contains(&0) {
        return fail("multipliers");
    }
    match cell.status.as_str() {
        STATUS_UNSUPPORTED => Ok(()),
        STATUS_OK => {
            let rows = &cell.a_idx;
            let in_table = strictly_ascending(rows) && rows.iter().all(|&r| r < table.n_rows);
            let fits = multipliers
                .iter()
                .all(|&m| requested_rows(cell, m) <= u64::from(u32::MAX));
            if rows.is_empty() || !in_table || !fits {
                return fail("a_idx");
            }
            Ok(())
        }
        _ => fail("status"),
    }
}

/// Whether the earlier rows of a cell are all `ok`, were made by this arm and
/// binary, and still have their files; only then is the cell not run again.
fn kept(job: &CellJob, cell: &Cell, earlier: &[EmitRow]) -> bool {
    let mut keys = cell_keys(cell);
    keys.sort_unstable();
    let width = cell.cols.len() as u64 + 1;
    earlier.len() == keys.len()
        && keys.iter().zip(earlier).all(|(&key, row)| {
            let rows = requested_rows(cell, key.1);
            let path = job.mat_path(cell, key);
            let on_disk = std::fs::metadata(&path).is_ok_and(|m| m.len() == 24 + rows * width * 4);
            (row.sample_seed, row.multiplier) == key
                && row.status == STATUS_OK
                && row.arm_id == job.arm.arm_id
                && row.arch_id == ARCH_ID
                && row.weights_sha256 == job.arm.weights_sha256
                && row.artifact_class == job.arm.artifact_class.name()
                && row.rows == rows
                && job.run_dir.join(&row.file) == path
                && on_disk
        })
}

fn sort_key(row: &EmitRow) -> (&str, u64, u32) {
    (&row.cell_id, row.sample_seed, row.multiplier)
}

/// Sort the rows of a run, require one row per scheduled file and write the log.
fn write_log(path: &Path, mut rows: Vec<EmitRow>, expected: usize) -> ArenaResult<Vec<EmitRow>> {
    rows.sort_by(|a, b| sort_key(a).cmp(&sort_key(b)));
    if rows.len() != expected {
        return contract("emit rows do not cover the schedule".into());
    }
    write_jsonl(path, &rows)?;
    Ok(rows)
}

pub fn run(args: &EmitCliArgs) -> ArenaResult<String> {
    if !args.timeout_s.is_finite() || args.timeout_s <= 0.0 || args.timeout_s > 1.0e9 {
        return contract("--timeout-s must be a positive number of seconds".into());
    }
    if args.limit_cells == Some(0) {
        return contract("--limit-cells must be at least 1".into());
    }
    if !valid_arm_id(&args.arm) {
        return contract("--arm is not a valid arm id".into());
    }
    let arena = Arena::open(&args.arena)?;
    let arm = load_arm(&args.arm_file, &args.arm)?;
    let mut cells: Vec<Cell> = read_jsonl(&args.schedule)?;
    let schedule_cells = cells.len();
    for (position, cell) in cells.iter().enumerate() {
        check_cell(&arena, position, cell)?;
    }
    if !distinct(&cells.iter().map(|cell| &cell.cell_id).collect::<Vec<_>>()) {
        return contract("schedule: duplicate cell_id".into());
    }
    cells.truncate(args.limit_cells.unwrap_or(usize::MAX));
    let expected_rows: usize = cells.iter().map(|cell| cell_keys(cell).len()).sum();

    // Everything above is checked before the run directory is touched.
    let arm_dir = args.out.join(&arm.arm_id);
    std::fs::create_dir_all(&arm_dir).map_err(|e| file_error(&arm_dir, &e))?;
    let run_dir = std::fs::canonicalize(&args.out).map_err(|e| file_error(&args.out, &e))?;
    if run_dir.to_str().is_none() {
        return contract("--out must be a UTF-8 path".into());
    }
    let arm_dir = run_dir.join(&arm.arm_id);
    let emit_path = arm_dir.join(EMIT_FILE);
    let mut earlier: BTreeMap<String, Vec<EmitRow>> = BTreeMap::new();
    if emit_path.exists() {
        if !args.resume {
            return contract(format!(
                "{}: this arm already has a log here; pass --resume or use a new --out",
                emit_path.display()
            ));
        }
        let mut rows: Vec<EmitRow> = read_jsonl(&emit_path)?;
        rows.sort_by(|a, b| sort_key(a).cmp(&sort_key(b)));
        for row in rows {
            earlier.entry(row.cell_id.clone()).or_default().push(row);
        }
    }
    // Scratch of an interrupted run may still hold rows: remove it first.
    remove_path(&arm_dir.join(WORK_DIR))?;
    for name in [LOG_DIR, PACKET_DIR, WORK_DIR] {
        let dir = arm_dir.join(name);
        std::fs::create_dir_all(&dir).map_err(|e| file_error(&dir, &e))?;
    }

    let job = CellJob {
        arena: &arena,
        arm: &arm,
        run_dir: &run_dir,
        timeout: Duration::from_secs_f64(args.timeout_s),
    };
    // Cells that need no command: unsupported ones, and those kept by --resume.
    let settled: Vec<Option<Vec<EmitRow>>> = cells
        .iter()
        .map(|cell| {
            if cell.status != STATUS_OK {
                return Some(job.rows_without_samples(cell, Reason::ScheduleUnsupported, 0.0));
            }
            let rows = earlier.get(&cell.cell_id);
            rows.filter(|rows| kept(&job, cell, rows)).cloned()
        })
        .collect();
    let unsupported = cells.iter().filter(|c| c.status != STATUS_OK).count();
    let cells_kept = settled.iter().flatten().count() - unsupported;
    // Until the finished log replaces it, the log says which cells are not done.
    let unfinished = cells.iter().zip(&settled).flat_map(|(cell, rows)| {
        let pending = || job.rows_without_samples(cell, Reason::RunInterrupted, 0.0);
        rows.clone().unwrap_or_else(pending)
    });
    write_log(&emit_path, unfinished.collect(), expected_rows)?;
    remove_path(&arm_dir.join(SUMMARY_FILE))?;

    let broken = AtomicBool::new(false);
    let results = par_map(args.threads, cells.len(), |index| {
        if let Some(rows) = &settled[index] {
            return Ok(rows.clone());
        }
        // Cells are claimed in order, so this placeholder always sorts after the real error.
        if broken.load(Ordering::Relaxed) {
            return contract("not run: an earlier cell stopped the run".into());
        }
        let rows = job.run(&cells[index]);
        if rows.is_err() {
            broken.store(true, Ordering::Relaxed);
        }
        rows
    });
    let mut rows = Vec::with_capacity(expected_rows);
    for result in results {
        rows.extend(result?);
    }
    remove_path(&arm_dir.join(WORK_DIR))?;
    let rows = write_log(&emit_path, rows, expected_rows)?;

    let mut summary = Summary {
        command: "emit-cli",
        arm_id: arm.arm_id.clone(),
        arch_id: ARCH_ID,
        artifact_class: arm.artifact_class.name(),
        io: arm.io.name(),
        weights_sha256: arm.weights_sha256.clone(),
        cold_bytes: arm.cold_bytes,
        budget_bytes: arm.budget_bytes,
        emit_file: format!("{}/{EMIT_FILE}", arm.arm_id),
        schedule_cells,
        limit_cells: args.limit_cells,
        cells: cells.len(),
        cells_kept,
        cells_run: cells.len() - cells_kept - unsupported,
        ..Summary::default()
    };
    summary.tally(&rows);
    write_atomic(
        &arm_dir.join(SUMMARY_FILE),
        canonical_pretty(&summary)?.as_bytes(),
    )?;
    canonical_json(&summary)
}

//! `dope_kpi real-ref`: everything that depends on real rows only, computed
//! once per (cell, auditor) and reused by every arm.
//!
//! Per row of T: the null loss, the losses and predictions of the auditor
//! fitted on A and of the one fitted on the twin B, and the target in scoring
//! units. Per (cell, auditor): the informative flag of the contract (real gain
//! at least 1% of the null loss with a positive bootstrap lower bound, twin
//! retention within [0.8, 1.2]), taken from `dvs::v3::harm::assess_parity`.
//! Per cell: the real-vs-real C2ST reference, B (label 0) against A (label 1),
//! built exactly like an arm's B-against-S test.

use super::auditors::{self, Recipe};
use super::celldata::{task_kind, CellData};
use super::cellplan::Cell;
use super::io::Arena;
use super::jsonl::{canonical_json, read_file, read_jsonl, sha256_hex, write_lines};
use super::seeds::{crn_seed, STRUCTURAL_SAMPLE_SEED};
use super::shard::{owned, run_workers, worker_count, Shard};
use dope_arena::{ArenaError, ArenaResult};
use dope_core::dvs::v3::bundle::PredictionBundle;
use dope_core::dvs::v3::c2st::boosted_tree_c2st;
use dope_core::dvs::v3::harm::assess_parity;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::path::PathBuf;

pub const REF_OK: &str = "ok";
pub const REF_UNSUPPORTED: &str = "unsupported";
/// Trees of the native C2ST discriminator (its own upper limit).
pub const C2ST_TREES: usize = 32;

#[derive(clap::Args, Debug)]
pub struct RealRefArgs {
    /// Flat corpus arena directory (`dope.flat-corpus/v1`).
    #[arg(long)]
    pub arena: PathBuf,
    /// `schedule.jsonl` written by `plan`.
    #[arg(long)]
    pub schedule: PathBuf,
    /// Output `realref.jsonl`.
    #[arg(long)]
    pub out: PathBuf,
    /// Auditors to fit (comma separated); default: the whole native roster.
    #[arg(long, value_delimiter = ',')]
    pub auditors: Option<Vec<String>>,
    /// Bootstrap draws behind the informative flag.
    #[arg(long, default_value_t = 2000)]
    pub bootstrap_draws: usize,
    /// Worker processes (each computes on one thread).
    #[arg(long, default_value_t = 1)]
    pub threads: usize,
    /// Internal: compute one shard of the schedule (set by the parent process).
    #[arg(long, hide = true)]
    pub shard: Option<Shard>,
}

/// Real-only reference of one (cell, auditor).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RealRef {
    pub cell_id: String,
    /// SHA-256 of the canonical schedule line this row was computed from.
    pub cell_sha256: String,
    pub auditor: String,
    pub status: String,
    pub reason: String,
    pub num_outputs: usize,
    pub informative: bool,
    pub informative_reason: String,
    pub l0: Option<f64>,
    pub l_a: Option<f64>,
    pub l_b: Option<f64>,
    pub r_b: Option<f64>,
    pub real_gain_lcb95: Option<f64>,
    pub c2st_real_ref: Option<f64>,
    pub y_true: Vec<f64>,
    pub null_losses: Vec<f64>,
    pub real_a_losses: Vec<f64>,
    pub real_b_losses: Vec<f64>,
    pub pred_a: Vec<f64>,
    pub pred_b: Vec<f64>,
}

/// Sort key and summary fields of one reference line.
#[derive(Deserialize)]
struct RefHead {
    cell_id: String,
    auditor: String,
    status: String,
    reason: String,
    informative: bool,
}

#[derive(Debug, Serialize)]
struct Summary {
    command: &'static str,
    auditors: Vec<String>,
    cells: usize,
    rows: usize,
    ok: usize,
    unsupported: usize,
    unsupported_by_reason: BTreeMap<String, usize>,
    informative: usize,
    informative_share_of_ok: f64,
    informative_by_auditor: BTreeMap<String, usize>,
    realref_sha256: String,
}

pub fn cell_sha256(cell: &Cell) -> ArenaResult<String> {
    Ok(sha256_hex(canonical_json(cell)?.as_bytes()))
}

/// One auditor seed per cell, shared by the A, B and synthetic fits (common random numbers).
pub fn auditor_seed(cell: &Cell) -> u64 {
    crn_seed(
        &cell.panel_sha256,
        &cell.dataset_hash,
        cell.data_seed,
        STRUCTURAL_SAMPLE_SEED,
        "auditor",
    )
}

/// The largest synthetic table an arm may deliver for this cell.
pub fn max_train_rows(cell: &Cell) -> usize {
    let largest = cell.multipliers.iter().copied().max().unwrap_or(1).max(1);
    cell.a_idx.len() * largest as usize
}

/// Roster to fit: the default, or a duplicate-free list of known names in the order given.
pub fn auditor_list(given: &Option<Vec<String>>) -> ArenaResult<Vec<String>> {
    let Some(names) = given else {
        return Ok(auditors::ROSTER.iter().map(|a| a.to_string()).collect());
    };
    let distinct: BTreeSet<&String> = names.iter().collect();
    if names.is_empty()
        || distinct.len() != names.len()
        || !names.iter().all(|a| auditors::is_known(a))
    {
        return Err(ArenaError::Contract(format!(
            "--auditors must list distinct names from {}",
            auditors::ROSTER.join(", ")
        )));
    }
    Ok(names.clone())
}

fn unsupported(cell: &Cell, sha: &str, auditor: &str, reason: &str) -> RealRef {
    RealRef {
        cell_id: cell.cell_id.clone(),
        cell_sha256: sha.to_string(),
        auditor: auditor.to_string(),
        status: REF_UNSUPPORTED.to_string(),
        reason: reason.to_string(),
        num_outputs: 0,
        informative: false,
        informative_reason: REF_UNSUPPORTED.to_string(),
        l0: None,
        l_a: None,
        l_b: None,
        r_b: None,
        real_gain_lcb95: None,
        c2st_real_ref: None,
        y_true: Vec::new(),
        null_losses: Vec::new(),
        real_a_losses: Vec::new(),
        real_b_losses: Vec::new(),
        pred_a: Vec::new(),
        pred_b: Vec::new(),
    }
}

/// Fit one auditor on A and on B and assemble its reference row.
fn auditor_row(
    cell: &Cell,
    sha: &str,
    data: &CellData,
    auditor: &str,
    c2st: Option<f64>,
    draws: usize,
) -> RealRef {
    let head = data.head;
    let cols = cell.cols.len();
    if !auditors::supports(auditor, head, cols, max_train_rows(cell), data.t.n) {
        return unsupported(cell, sha, auditor, auditors::REASON_SHAPE);
    }
    let (xa, ya) = (data.card.features(&data.a), data.card.targets(&data.a));
    let (xb, yb) = (data.card.features(&data.b), data.card.targets(&data.b));
    let (xt, yt) = (data.card.features(&data.t), data.card.targets(&data.t));
    let seed = auditor_seed(cell);
    let fit = |x, y: &[f64], rows| {
        auditors::fit_predict(auditor, head, x, y, &xt, Recipe::new(seed, rows, data.a.n))
    };
    let (Ok(raw_a), Ok(raw_b)) = (fit(&xa, &ya, data.a.n), fit(&xb, &yb, data.b.n)) else {
        return unsupported(cell, sha, auditor, auditors::REASON_FAILED);
    };
    let null = auditors::null_predictions(head, &ya, data.t.n);
    let (null_losses, _) = auditors::losses(head, &null, &yt);
    let (real_a_losses, pred_a) = auditors::losses(head, &raw_a, &yt);
    let (real_b_losses, pred_b) = auditors::losses(head, &raw_b, &yt);
    // The contract's informative rule reads real losses only; the synthetic
    // slots are filled with the A model so the bundle is well formed.
    let bundle = PredictionBundle {
        task: task_kind(head),
        null_losses,
        real_a_losses: real_a_losses.clone(),
        real_b_losses: Some(real_b_losses),
        synth_losses: real_a_losses,
        donor_ids: data.t_donors.clone(),
        y_true: yt,
        pred_a: pred_a.clone(),
        pred_b: Some(pred_b),
        pred_s: pred_a,
        num_outputs: head.outputs(),
    };
    let purpose = format!("informative|{}|{auditor}", cell.view);
    let boot_seed = crn_seed(
        &cell.panel_sha256,
        &cell.dataset_hash,
        cell.data_seed,
        STRUCTURAL_SAMPLE_SEED,
        &purpose,
    );
    let parity = match assess_parity(&bundle, draws, boot_seed) {
        Ok(parity) => parity,
        Err(err) => return unsupported(cell, sha, auditor, err.0),
    };
    RealRef {
        cell_id: cell.cell_id.clone(),
        cell_sha256: sha.to_string(),
        auditor: auditor.to_string(),
        status: REF_OK.to_string(),
        reason: String::new(),
        num_outputs: bundle.num_outputs,
        informative: parity.informative,
        informative_reason: parity.reason.unwrap_or("").to_string(),
        l0: Some(bundle.mean_null()),
        l_a: Some(bundle.mean_real_a()),
        l_b: bundle.mean_real_b(),
        r_b: parity.unclipped_retention_b,
        real_gain_lcb95: Some(parity.real_gain_lcb95),
        c2st_real_ref: c2st,
        y_true: bundle.y_true,
        null_losses: bundle.null_losses,
        real_a_losses: bundle.synth_losses,
        real_b_losses: bundle.real_b_losses.unwrap_or_default(),
        pred_a: bundle.pred_s,
        pred_b: bundle.pred_b.unwrap_or_default(),
    }
}

/// Reference rows of one cell, one per auditor in roster order.
pub fn cell_rows(
    arena: &Arena,
    cell: &Cell,
    roster: &[String],
    draws: usize,
) -> ArenaResult<Vec<RealRef>> {
    let sha = cell_sha256(cell)?;
    let data = match CellData::load(arena, cell)? {
        Ok(data) => data,
        Err(reason) => {
            return Ok(roster
                .iter()
                .map(|a| unsupported(cell, &sha, a, reason))
                .collect())
        }
    };
    let (unit_a, unit_b) = (data.card.unit_rows(&data.a), data.card.unit_rows(&data.b));
    let c2st = boosted_tree_c2st(&unit_b, &unit_a, C2ST_TREES)
        .ok()
        .map(|(_, _, auc)| auc);
    Ok(roster
        .iter()
        .map(|a| auditor_row(cell, &sha, &data, a, c2st, draws))
        .collect())
}

/// Reference lines of the cells at `positions`, rendered in this process.
fn compute(
    arena: &Arena,
    cells: &[Cell],
    positions: &[usize],
    roster: &[String],
    draws: usize,
) -> ArenaResult<Vec<String>> {
    let mut lines = Vec::with_capacity(positions.len() * roster.len());
    for &position in positions {
        for row in cell_rows(arena, &cells[position], roster, draws)? {
            lines.push(canonical_json(&row)?);
        }
    }
    Ok(lines)
}

/// The same lines from `workers` processes, each writing one part file.
fn compute_in_workers(
    args: &RealRefArgs,
    roster: &[String],
    workers: usize,
) -> ArenaResult<Vec<String>> {
    let part = |index: usize| {
        let mut name = args.out.clone().into_os_string();
        name.push(format!(".shard-{index}"));
        PathBuf::from(name)
    };
    run_workers(workers, |index| {
        let mut argv: Vec<OsString> = vec!["real-ref".into(), "--arena".into()];
        argv.push(args.arena.clone().into());
        argv.push("--schedule".into());
        argv.push(args.schedule.clone().into());
        argv.push("--out".into());
        argv.push(part(index).into());
        argv.push("--auditors".into());
        argv.push(roster.join(",").into());
        argv.push("--bootstrap-draws".into());
        argv.push(args.bootstrap_draws.to_string().into());
        argv
    })?;
    let mut lines = Vec::new();
    for index in 0..workers {
        let path = part(index);
        let text = String::from_utf8(read_file(&path)?)
            .map_err(|_| ArenaError::Contract("worker output is not UTF-8".into()))?;
        lines.extend(text.lines().map(str::to_string));
        std::fs::remove_file(&path)?;
    }
    Ok(lines)
}

pub fn run(args: &RealRefArgs) -> ArenaResult<String> {
    let roster = auditor_list(&args.auditors)?;
    if args.bootstrap_draws < 100 {
        return Err(ArenaError::Contract(
            "--bootstrap-draws must be at least 100".into(),
        ));
    }
    let arena = Arena::open(&args.arena)?;
    let cells: Vec<Cell> = read_jsonl(&args.schedule)?;
    let workers = worker_count(args.threads, cells.len());
    let lines = if args.shard.is_none() && workers > 1 {
        compute_in_workers(args, &roster, workers)?
    } else {
        let positions = owned(args.shard, cells.len());
        compute(&arena, &cells, &positions, &roster, args.bootstrap_draws)?
    };
    let mut rows = Vec::with_capacity(lines.len());
    for line in lines {
        let head: RefHead = serde_json::from_str(&line)?;
        rows.push((head, line));
    }
    rows.sort_by(|a, b| (&a.0.cell_id, &a.0.auditor).cmp(&(&b.0.cell_id, &b.0.auditor)));
    let realref_sha256 = write_lines(&args.out, rows.iter().map(|row| row.1.as_str()))?;

    let ok = rows.iter().filter(|r| r.0.status == REF_OK).count();
    let informative = rows.iter().filter(|r| r.0.informative).count();
    let mut unsupported_by_reason = BTreeMap::new();
    let mut informative_by_auditor: BTreeMap<String, usize> =
        roster.iter().map(|a| (a.clone(), 0)).collect();
    for (head, _) in &rows {
        if head.status != REF_OK {
            *unsupported_by_reason
                .entry(head.reason.clone())
                .or_default() += 1;
        }
        if head.informative {
            *informative_by_auditor
                .entry(head.auditor.clone())
                .or_default() += 1;
        }
    }
    canonical_json(&Summary {
        command: "real-ref",
        auditors: roster,
        cells: rows
            .iter()
            .map(|r| &r.0.cell_id)
            .collect::<BTreeSet<_>>()
            .len(),
        rows: rows.len(),
        ok,
        unsupported: rows.len() - ok,
        unsupported_by_reason,
        informative,
        informative_share_of_ok: informative as f64 / ok.max(1) as f64,
        informative_by_auditor,
        realref_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auditor_lists_are_validated() {
        assert_eq!(auditor_list(&None).unwrap(), auditors::ROSTER);
        let two = Some(vec!["gbdt".to_string(), "linear-seeded-v1".to_string()]);
        assert_eq!(auditor_list(&two).unwrap(), ["gbdt", "linear-seeded-v1"]);
        for bad in [
            vec![],
            vec!["gbdt".to_string(), "gbdt".to_string()],
            vec!["mlp".to_string()],
        ] {
            let err = auditor_list(&Some(bad)).unwrap_err().to_string();
            assert!(err.contains("--auditors"), "{err}");
        }
    }

    #[test]
    fn unsupported_rows_round_trip_with_null_statistics() {
        let cell: Cell = serde_json::from_str(
            "{\"a_idx\":[1,2],\"b_idx\":[3,4],\"c_idx\":[],\"cell_id\":\"dev:h1:full:d0\",\"cols\":[0],\
             \"data_seed\":3,\"dataset_hash\":\"h1\",\"family_id\":\"lf-h1\",\"multipliers\":[4,1],\
             \"n_classes\":0,\"panel_sha256\":\"p\",\"sample_seeds\":[0],\"stratum\":\"small_n\",\
             \"t_idx\":[0],\"table_id\":0,\"task\":\"regression\",\"view\":\"full\"}",
        )
        .unwrap();
        let sha = cell_sha256(&cell).unwrap();
        assert_eq!(sha.len(), 64);
        assert_eq!(max_train_rows(&cell), 8);
        assert_eq!(auditor_seed(&cell), crn_seed("p", "h1", 3, 0, "auditor"));
        let row = unsupported(&cell, &sha, "gbdt", "schedule_unsupported");
        let line = canonical_json(&row).unwrap();
        assert!(line.contains("\"l0\":null") && line.contains("\"status\":\"unsupported\""));
        assert_eq!(serde_json::from_str::<RealRef>(&line).unwrap(), row);
        // Any change to the scheduled rows changes the pin the scorer checks.
        let mut moved = cell.clone();
        moved.t_idx = vec![5];
        assert_ne!(cell_sha256(&moved).unwrap(), sha);
    }
}

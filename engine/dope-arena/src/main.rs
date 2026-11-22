//! `dope-arena` CLI: unified Rust benchmark orchestrator for `dope-fidelity/v3`.

use clap::{Args, Parser, Subcommand};
use dope_arena::ledger::read_ledger;
use dope_arena::panel::PanelManifest;
use dope_arena::report::{reduce_cells, run_swap_ladder};
use dope_arena::schedule::MethodsRegistry;
use dope_arena::split::{split_table_disjoint, SplitRatios};
use dope_core::RawTable;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "dope-arena",
    version,
    about = "Rust benchmark Arena for dope-fidelity/v3"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Verify a dev or sealed confirmation panel manifest and its SHA-256 commitment.
    VerifyPanel(VerifyPanelArgs),
    /// Produce deterministic disjoint A/B/C/T/P splits respecting groups, time, and Rule A-1.
    Split(SplitArgs),
    /// Run the 4-arm native swap ladder (Twin-B, Real-X+Synth-Y --given, 1n, 4n) on a CSV.
    SwapLadder(SwapLadderArgs),
    /// Reduce an Arena JSONL ledger into a canonical dope-fidelity/v3 coverage receipt.
    Reduce(ReduceArgs),
    /// Verify that a method ID is authentic and eligible for headline claims.
    CheckMethod(CheckMethodArgs),
}

#[derive(Args)]
struct VerifyPanelArgs {
    manifest: PathBuf,
}

#[derive(Args)]
struct SplitArgs {
    csv: PathBuf,
    #[arg(long)]
    group: Option<String>,
    #[arg(long)]
    time: Option<String>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

#[derive(Args)]
struct SwapLadderArgs {
    csv: PathBuf,
    #[arg(short, long)]
    target: String,
    #[arg(short, long, default_value_t = 65536)]
    budget: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value = "linear-seeded-v1,gbdt")]
    auditors: String,
}

#[derive(Args)]
struct ReduceArgs {
    ledger: PathBuf,
    #[arg(long, default_value = "faithful")]
    mode: String,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 1000)]
    draws: usize,
}

#[derive(Args)]
struct CheckMethodArgs {
    registry: PathBuf,
    method_id: String,
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::VerifyPanel(a) => {
            let m = PanelManifest::load(&a.manifest).map_err(|e| e.to_string())?;
            let sha = m.canonical_sha256().map_err(|e| e.to_string())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "panel_id": m.panel_id,
                    "role": m.role,
                    "datasets": m.datasets.len(),
                    "unique_families": m.unique_families(),
                    "canonical_sha256": sha,
                }))
                .unwrap_or_default()
            );
        }
        Command::Split(a) => {
            let table = RawTable::read(&a.csv).map_err(|e| e.to_string())?;
            let gc = a
                .group
                .as_deref()
                .map(|g| table.column_index(g))
                .transpose()
                .map_err(|e| e.to_string())?;
            let tc = a
                .time
                .as_deref()
                .map(|t| table.column_index(t))
                .transpose()
                .map_err(|e| e.to_string())?;
            let splits = split_table_disjoint(&table, gc, tc, &SplitRatios::default(), a.seed)
                .map_err(|e| e.to_string())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&splits).unwrap_or_default()
            );
        }
        Command::SwapLadder(a) => {
            let table = RawTable::read(&a.csv).map_err(|e| e.to_string())?;
            let target = table.column_index(&a.target).map_err(|e| e.to_string())?;
            let auditors: Vec<String> = a
                .auditors
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let report = run_swap_ladder(&table, target, a.budget, a.seed, auditors)
                .map_err(|e| e.to_string())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&report).unwrap_or_default()
            );
        }
        Command::Reduce(a) => {
            let (cells, sha) = read_ledger(&a.ledger).map_err(|e| e.to_string())?;
            let receipt =
                reduce_cells(cells, a.mode, a.seed, a.draws).map_err(|e| e.to_string())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "ledger_sha256": sha,
                    "receipt": receipt,
                }))
                .unwrap_or_default()
            );
        }
        Command::CheckMethod(a) => {
            let reg = MethodsRegistry::load(&a.registry).map_err(|e| e.to_string())?;
            reg.verify_headline_eligible(&a.method_id)
                .map_err(|e| e.to_string())?;
            println!(
                "{{\"method_id\": \"{}\", \"headline_eligible\": true}}",
                a.method_id
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

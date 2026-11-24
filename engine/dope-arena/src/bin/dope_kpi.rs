//! `dope_kpi`: the `dope-kpi/v2` scoreboard CLI (`docs/plans/v2-gap-closure/formats.md`).
//!
//! Each subcommand lives in its own module under `src/kpi/`, owns its
//! argument struct and returns a one-line JSON summary for stdout. Adding a
//! subcommand is one `Command` variant and one arm in `run`.

use clap::{Parser, Subcommand};
use std::process::ExitCode;

#[path = "../kpi/mod.rs"]
mod kpi;

#[derive(Parser)]
#[command(
    name = "dope_kpi",
    version,
    about = "DOPE scoreboard data plane (dope-kpi/v2)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Cluster tables into lineage families and write the panel free of training siblings.
    BuildPanels(kpi::panels_build::BuildPanelsArgs),
    /// Write the frozen cell schedule (row roles and column views) for a panel and tier.
    Plan(kpi::cellplan::PlanArgs),
    /// Run an external generator command on every scheduled cell and log one row per file.
    EmitCli(kpi::ext_emit::EmitCliArgs),
    /// Fit the auditors on real rows once: per-row losses, informative flags, C2ST reference.
    RealRef(kpi::realref::RealRefArgs),
    /// Score one arm (external files or a native control) into a ledger and compact rows.
    Score(kpi::score::ScoreArgs),
    /// Reduce a score ledger with `evaluate_v3` (kappa file, partial evidence, filters).
    ReduceLedger(kpi::reduce::ReduceArgs),
    /// Realism and privacy gates and the exact-copy excess of scored arms (`guards.jsonl`).
    Guards(kpi::guards::GuardsArgs),
    /// TRG of each arm with its stratified family bootstrap interval (dope.kpi-receipt/v2).
    Primary(kpi::primary::PrimaryArgs),
    /// Paired comparison of candidates with one baseline under the promotion rule.
    Compare(kpi::compare::CompareArgs),
    /// Audit a primary or compare receipt's input identities, denominators and point estimates.
    Audit(kpi::audit::AuditArgs),
    /// Sign-flip null of one comparison: the false-PROMOTE rate of the full rule.
    Null(kpi::null::NullArgs),
    /// README scoreboard block from primary, compare and null receipts.
    RenderReadme(kpi::render::RenderArgs),
    /// README hook graphic (SVG): TRG or Transfer Score against artifact bytes from receipts.
    RenderHero(kpi::hero::HeroArgs),
    /// White paper LaTeX tables (`v3-ranking-table.tex`, `v3-paired-table.tex`) from receipts.
    RenderPaperTables(kpi::render_paper::PaperTablesArgs),
}

fn run(cli: &Cli) -> dope_arena::ArenaResult<String> {
    match &cli.command {
        Command::BuildPanels(args) => kpi::panels_build::run(args),
        Command::Plan(args) => kpi::cellplan::run(args),
        Command::EmitCli(args) => kpi::ext_emit::run(args),
        Command::RealRef(args) => kpi::realref::run(args),
        Command::Score(args) => kpi::score::run(args),
        Command::ReduceLedger(args) => kpi::reduce::run(args),
        Command::Guards(args) => kpi::guards::run(args),
        Command::Primary(args) => kpi::primary::run(args),
        Command::Compare(args) => kpi::compare::run(args),
        Command::Audit(args) => kpi::audit::run(args),
        Command::Null(args) => kpi::null::run(args),
        Command::RenderReadme(args) => kpi::render::run(args),
        Command::RenderHero(args) => kpi::hero::run(args),
        Command::RenderPaperTables(args) => kpi::render_paper::run(args),
    }
}

fn main() -> ExitCode {
    // Before any thread exists: auditor fits compute on their calling thread.
    kpi::shard::single_threaded_inner_pools();
    match run(&Cli::parse()) {
        Ok(summary) => {
            println!("{summary}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

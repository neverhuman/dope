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
}

fn run(cli: &Cli) -> dope_arena::ArenaResult<String> {
    match &cli.command {
        Command::BuildPanels(args) => kpi::panels_build::run(args),
        Command::Plan(args) => kpi::cellplan::run(args),
    }
}

fn main() -> ExitCode {
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

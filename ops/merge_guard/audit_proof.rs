//! Local proof tooling; not a product or scientific acceptance authority.
use clap::{Parser, Subcommand};
use std::path::PathBuf;
#[path = "audit_proof/badge.rs"]
mod badge;
#[path = "audit_proof/focus.rs"]
mod focus;
#[cfg(test)]
#[path = "audit_proof/tests.rs"]
mod tests;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    Focus {
        filter: String,
    },
    Badge {
        #[arg(long)]
        report: PathBuf,
        #[arg(long)]
        expected_head: String,
        #[arg(long, default_value = "target/jankurai/score.svg")]
        output: PathBuf,
        #[arg(long)]
        check: Option<PathBuf>,
    },
}
fn main() {
    let result = match Args::parse().command {
        Action::Focus { filter } => focus::run(&filter),
        Action::Badge {
            report,
            expected_head,
            output,
            check,
        } => badge::write(&report, &expected_head, &output, check.as_deref()),
    };
    match result {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("proof tooling failed: {error}");
            std::process::exit(1);
        }
    }
}

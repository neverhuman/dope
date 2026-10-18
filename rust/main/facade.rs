//! Public CLI boundary. Legacy certification stays in the dope-kernel binary.
#[path = "facade/artifacts.rs"]
mod artifacts;
#[path = "facade/dispatch.rs"]
mod dispatch;
#[path = "facade/exit_codes.rs"]
mod exit_codes;
#[path = "facade/io.rs"]
mod io;
#[path = "facade/registry.rs"]
mod registry;
#[path = "facade/workflow.rs"]
mod workflow;

use clap::Parser;
use exit_codes::Failure;
use registry::Cli;
use std::process::ExitCode;

pub fn main() -> ExitCode {
    let args = std::env::args_os().collect::<Vec<_>>();
    let migrated_panel = args.iter().any(|arg| {
        arg == "--panel-tag" || arg.to_str().is_some_and(|s| s.starts_with("--panel-tag="))
    });
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(error) => {
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) {
                if error.print().is_err() {
                    return fail(Failure::new(
                        exit_codes::Code::Internal,
                        "output_write_failed",
                    ));
                }
                return ExitCode::SUCCESS;
            }
            // Clap diagnostics may contain raw arguments, values and private paths.
            return fail(Failure::usage("invalid_arguments_use_command_help"));
        }
    };
    if migrated_panel {
        eprintln!("migration: --panel-tag is retired; use --panel");
    }
    match dispatch::run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => fail(error),
    }
}

fn fail(error: Failure) -> ExitCode {
    eprintln!(
        "{}",
        serde_json::json!({"error":error.reason,"exit_code":error.code as u8,"docs":"docs/cli.md"})
    );
    ExitCode::from(error.code as u8)
}

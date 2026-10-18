//! The single argument registry for public help, aliases, parsing and dispatch.
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "dope",
    version,
    about = "Offline native DOPE development workflows; qualification requires measured gates"
)]
pub struct Cli {
    #[command(flatten)]
    pub limits: Limits,
    /// Offline operation is mandatory; --offline=false is refused.
    #[arg(long, global = true, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true", require_equals = true)]
    pub offline: bool,
    /// Explicit network requests are refused before any file is opened.
    #[arg(long, global = true)]
    pub network: bool,
    /// Development panel only. --panel-tag is a deprecated migration alias.
    #[arg(long, global = true, visible_alias = "panel-tag", default_value = "development", value_parser = ["development", "tdl-candidate"])]
    pub panel: String,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Args, Clone)]
pub struct Limits {
    #[arg(long, global = true, default_value_t = 16_777_216, value_parser = clap::value_parser!(u64).range(1..=16_777_216))]
    pub max_input_bytes: u64,
    #[arg(long, global = true, default_value_t = 16_777_216, value_parser = clap::value_parser!(u64).range(1..=16_777_216))]
    pub max_output_bytes: u64,
    #[arg(long, global = true, default_value_t = 100_000, value_parser = clap::value_parser!(u64).range(1..=100_000))]
    pub max_rows: u64,
    #[arg(long, global = true, default_value_t = 1_000_000, value_parser = clap::value_parser!(u64).range(1..=1_000_000))]
    pub max_cells: u64,
    #[arg(long, global = true, default_value_t = 4096, value_parser = clap::value_parser!(u64).range(1..=4096))]
    pub max_field_bytes: u64,
}

#[derive(Subcommand)]
pub enum Command {
    /// Validate input and write an opaque intake receipt; no fitted replay claim.
    Prepare {
        #[command(flatten)]
        input: Input,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Validate declared headered input without fitting or evaluation.
    #[command(visible_alias = "validate")]
    ValidateInput {
        #[command(flatten)]
        input: Input,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Fit a development artifact from declared CSV or a legacy numeric directory.
    #[command(visible_alias = "fit")]
    Compile(Compile),
    /// Inspect bytes and unmeasured status, without exposing private metadata.
    Inspect {
        #[arg(long, visible_alias = "kernel")]
        artifact: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Sample original units from a retained artifact; writes private synthetic data.
    Sample {
        #[arg(long, visible_alias = "kernel")]
        artifact: PathBuf,
        #[arg(long, value_parser = clap::value_parser!(u64).range(1..=100_000))]
        rows: u64,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Report built-in capabilities; no installs, network probes or imports.
    Doctor {
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Candidate evaluation is unavailable; explicit --real-dir replays legacy certification.
    #[command(visible_alias = "certify", visible_alias = "verify")]
    Evaluate {
        #[arg(long, visible_alias = "kernel")]
        artifact: PathBuf,
        #[arg(long)]
        real_dir: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        report: Option<PathBuf>,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
        #[arg(long, default_value = "l3", value_parser = ["l0", "l1", "l2", "l3"])]
        tier: String,
    },
    /// Project an existing legacy receipt with explicit independent release states.
    Report {
        #[arg(long)]
        report: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Run a bounded built-in synthetic engineering fixture; no scientific claim.
    Demo {
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        report: Option<PathBuf>,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
    },
    /// Print the package version, optionally as JSON.
    Version {
        #[arg(long)]
        json: bool,
    },
}

#[derive(Args)]
pub struct Input {
    #[arg(long)]
    pub csv: PathBuf,
    #[arg(long)]
    pub schema: PathBuf,
}

#[derive(Args)]
pub struct Compile {
    #[arg(long, conflicts_with = "dataset_dir", requires = "schema")]
    pub csv: Option<PathBuf>,
    #[arg(long, requires = "csv")]
    pub schema: Option<PathBuf>,
    /// Legacy headerless numeric train.csv directory; explicit compatibility mode.
    #[arg(long, conflicts_with = "csv")]
    pub dataset_dir: Option<PathBuf>,
    #[arg(long, default_value = "regression", value_parser = ["regression", "binary"])]
    pub task: String,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long)]
    pub report: Option<PathBuf>,
    #[arg(long, visible_alias = "fit-seed")]
    pub seed: Option<u64>,
    #[arg(long, default_value = "chow_liu", value_parser = ["chow_liu"])]
    pub candidate: String,
    #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u64).range(1..=600))]
    pub deadline_seconds: u64,
    #[arg(long, visible_alias = "max-artifact-bytes", default_value_t = 32768, value_parser = clap::value_parser!(u64).range(1..=16_777_216))]
    pub max_complete_bytes: u64,
    #[arg(long, default_value = "l3", value_parser = ["l0", "l1", "l2", "l3"])]
    pub tier: String,
    /// A formal-DP request fails closed: no verified mechanism/accountant is wired.
    #[arg(long)]
    pub require_formal_dp: bool,
}

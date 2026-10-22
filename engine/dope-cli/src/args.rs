//! Command-line arguments of `dope`.

use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "dope",
    version,
    about = "Compile a table into a tiny generative program that preserves what models learn from it."
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand)]
pub(crate) enum Command {
    /// Compile a CSV into a .dope artifact (a safetensors file).
    Fit(FitArgs),
    /// Compile small/medium/large artifacts and a portable catalog for one table.
    FitTiers(TierArgs),
    /// Draw synthetic rows from an artifact.
    Sample(SampleArgs),
    /// Use the stored conditional model as a predictor.
    Predict(PredictArgs),
    /// Show what an artifact contains and where its bytes go.
    Inspect(InspectArgs),
    /// Write an ML-ready version of a CSV (one-hot categoricals, numerics on [0,1]).
    Encode(EncodeArgs),
    /// Measure retention: train auditors on synthetic vs real rows, test on held-out real rows.
    #[command(name = "eval")]
    Retention(EvalArgs),
    /// Sweep byte budgets and report the transfer-vs-bytes frontier and TDL.
    Frontier(FrontierArgs),
    /// Measure sampling throughput (rows per second, no output written).
    Throughput(ThroughputArgs),
    /// Report the compute backends available in this build.
    Doctor,
}

#[derive(Args)]
pub(crate) struct FitArgs {
    /// Training CSV with a header row.
    pub(crate) csv: PathBuf,
    /// Target column (header name or zero-based index).
    #[arg(short, long)]
    pub(crate) target: String,
    /// Output artifact.
    #[arg(short, long, default_value = "model.dope")]
    pub(crate) out: PathBuf,
    /// Whole-file cap: 2048 bytes, 4k/4KiB (binary), 4KB (decimal).
    #[arg(short, long)]
    pub(crate) budget: Option<String>,
    /// small (2 KiB), medium (4 KiB), large (16 KiB); --budget overrides this cap.
    #[arg(long)]
    pub(crate) size: Option<dope_core::ModelSize>,
    /// Hard cap on complete-generator scalar parameters, including marginals and trees.
    #[arg(long)]
    pub(crate) max_parameters: Option<usize>,
    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,
    /// regression | classification (inferred by default).
    #[arg(long)]
    pub(crate) task: Option<String>,
    /// Replace column names and labels with positional codes.
    #[arg(long)]
    pub(crate) anonymize: bool,
    /// auto | cpu | cuda[:N] | metal[:N]
    #[arg(long, default_value = "auto")]
    pub(crate) device: String,
    /// Write the fit report as JSON.
    #[arg(long)]
    pub(crate) report: Option<PathBuf>,
    /// Force a plan "knots,inputs,hidden" (ablations).
    #[arg(long)]
    pub(crate) plan: Option<String>,
    /// Force a dependence family: independent | tree | factorN | full (ablations).
    #[arg(long)]
    pub(crate) dependence: Option<String>,
    /// Force the number of regimes (ablations; 1 = plain Gaussian copula).
    #[arg(long)]
    pub(crate) regimes: Option<usize>,
    /// Network only: no boosted trees in the conditional model (ablations).
    #[arg(long)]
    pub(crate) no_trees: bool,
    /// Skip the self-audit and use the middle dependence share (faster fits).
    #[arg(long)]
    pub(crate) no_audit: bool,
    /// Share of the stored gain quantization may give up (0 = keep 8-bit weights).
    #[arg(long, default_value_t = 0.005)]
    pub(crate) tolerance: f64,
    #[arg(short, long)]
    pub(crate) quiet: bool,
}

#[derive(Args)]
pub(crate) struct TierArgs {
    pub(crate) csv: PathBuf,
    #[arg(short, long)]
    pub(crate) target: String,
    /// A new directory; existing directories are refused to prevent overwrite.
    #[arg(short, long)]
    pub(crate) out: PathBuf,
    /// Dataset identifier for this release (no raw column values).
    #[arg(long)]
    pub(crate) dataset_id: String,
    /// Public source URL or dataset DOI, retained in the catalog.
    #[arg(long)]
    pub(crate) source: String,
    /// SPDX license of the training data (artifact release rights must be reviewed).
    #[arg(long)]
    pub(crate) data_license: String,
    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,
    #[arg(long)]
    pub(crate) task: Option<String>,
    #[arg(long, default_value = "auto")]
    pub(crate) device: String,
}

#[derive(Args)]
pub(crate) struct SampleArgs {
    pub(crate) model: PathBuf,
    #[arg(short = 'n', long, default_value_t = 1000)]
    pub(crate) rows: usize,
    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,
    #[arg(short, long)]
    pub(crate) out: Option<PathBuf>,
    /// raw (original units and labels) or ml (one-hot, [0,1]).
    #[arg(long, default_value = "raw")]
    pub(crate) space: String,
    /// For --space ml: unit (robust 0-1) or quantile.
    #[arg(long, default_value = "unit")]
    pub(crate) scaling: String,
    #[arg(long, default_value = "auto")]
    pub(crate) device: String,
}

#[derive(Args)]
pub(crate) struct ThroughputArgs {
    pub(crate) model: PathBuf,
    #[arg(short = 'n', long, default_value_t = 1_000_000)]
    pub(crate) rows: usize,
    #[arg(long, default_value_t = 3)]
    pub(crate) repeats: usize,
    #[arg(long, default_value_t = 65_536)]
    pub(crate) chunk: usize,
    #[arg(long, default_value = "auto")]
    pub(crate) device: String,
}

#[derive(Args)]
pub(crate) struct PredictArgs {
    pub(crate) model: PathBuf,
    pub(crate) csv: PathBuf,
    #[arg(short, long)]
    pub(crate) out: Option<PathBuf>,
}

#[derive(Args)]
pub(crate) struct InspectArgs {
    pub(crate) model: PathBuf,
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Args)]
pub(crate) struct EncodeArgs {
    pub(crate) csv: PathBuf,
    #[arg(short, long)]
    pub(crate) target: Option<String>,
    /// Reuse the schema of an artifact (so real and synthetic encode identically).
    #[arg(short, long)]
    pub(crate) model: Option<PathBuf>,
    #[arg(long, default_value = "unit")]
    pub(crate) scaling: String,
    #[arg(short, long)]
    pub(crate) out: Option<PathBuf>,
}

#[derive(Args)]
pub(crate) struct EvalArgs {
    #[arg(short, long)]
    pub(crate) model: PathBuf,
    /// Real rows the model was fitted on (trains the real-data reference).
    #[arg(long)]
    pub(crate) train: PathBuf,
    /// Held-out real rows (never seen by fit).
    #[arg(long)]
    pub(crate) test: PathBuf,
    /// Synthetic CSV; by default draws `--factor` × |train| rows from the model.
    #[arg(long)]
    pub(crate) synth: Option<PathBuf>,
    #[arg(long, default_value_t = 4.0)]
    pub(crate) factor: f64,
    /// Comma list of auditors: linear,gbdt,mlp
    #[arg(long, default_value = "linear,gbdt,mlp")]
    pub(crate) auditors: String,
    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,
    #[arg(long)]
    pub(crate) json: Option<PathBuf>,
}

#[derive(Args)]
pub(crate) struct FrontierArgs {
    #[arg(long)]
    pub(crate) train: PathBuf,
    #[arg(long)]
    pub(crate) test: PathBuf,
    #[arg(short, long)]
    pub(crate) target: String,
    /// Comma list of budgets, e.g. 1k,2k,4k,8k,16k,64k
    #[arg(long, default_value = "1k,2k,4k,8k,16k,64k")]
    pub(crate) budgets: String,
    /// Retention threshold for TDL.
    #[arg(long, default_value_t = 0.9)]
    pub(crate) tau: f64,
    #[arg(long, default_value = "linear,gbdt,mlp")]
    pub(crate) auditors: String,
    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,
    #[arg(long)]
    pub(crate) json: Option<PathBuf>,
}

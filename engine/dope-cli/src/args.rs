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
    /// Number of CPU worker threads (1–1024); otherwise use Rayon's default.
    #[arg(long, global = true, value_parser = clap::value_parser!(u16).range(1..=1024))]
    pub(crate) threads: Option<u16>,
    /// Terminal color output: auto (TTY-detected, honors NO_COLOR), always, or never.
    #[arg(long, global = true, default_value = "auto", value_parser = ["auto", "always", "never"])]
    pub(crate) color: String,
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand)]
pub(crate) enum Command {
    /// Experimental frozen shared encoder and code-only CPU decoder.
    #[command(subcommand)]
    Learned(crate::learned::LearnedCommand),
    /// 100% pure-Rust DOPE-MLC-v3 SafeTensors inference and DML3/CSC1 wire sampling.
    #[command(subcommand)]
    MlcV3(crate::mlc_v3::MlcV3Command),
    /// Experimental portable levels with explicit release permissions.
    FitLevels(LevelArgs),
    /// Native dataset structure signatures (real measurements stay private).
    #[command(subcommand)]
    Landmarks(LandmarkCommand),
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
    Doctor(DoctorArgs),
}

#[derive(Subcommand)]
pub(crate) enum LandmarkCommand {
    /// Describe authorized fit rows. Output is a private measurement.
    Real(LandmarkRealArgs),
    /// Describe samples of an artifact. Opens no real dataset.
    Synthetic(LandmarkSyntheticArgs),
    /// Train on synthetic/fit rows, score on disjoint real query rows.
    Compare(LandmarkCompareArgs),
}

#[derive(Args)]
pub(crate) struct LandmarkCommon {
    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,
    #[arg(long, default_value = "landmarks-native-v1")]
    pub(crate) panel: String,
    #[arg(long, default_value_t = 5)]
    pub(crate) folds: usize,
    #[arg(long, default_value_t = 2_000_000_000)]
    pub(crate) work_budget: u64,
    /// At most 64 private feature spectra; 0 disables them.
    #[arg(long, default_value_t = 64)]
    pub(crate) spectrum_features: usize,
    /// Training rows per ridge fold (maximum 50000).
    #[arg(long, default_value_t = 8192)]
    pub(crate) ridge_rows: usize,
    /// Training rows per tree/network fold (maximum 4096).
    #[arg(long, default_value_t = 4096)]
    pub(crate) small_rows: usize,
    #[arg(long, default_value_t = 2048)]
    pub(crate) knn_reference: usize,
    /// Write JSON instead of a human-readable report.
    #[arg(long)]
    pub(crate) json: bool,
    #[arg(short, long)]
    pub(crate) out: Option<PathBuf>,
}
#[derive(Args)]
pub(crate) struct LandmarkRealArgs {
    pub(crate) csv: PathBuf,
    #[arg(short, long)]
    pub(crate) target: String,
    #[arg(long, value_parser=["regression","classification"])]
    pub(crate) task: Option<String>,
    /// Optional grouping/time-block column; excluded from predictor features.
    #[arg(long)]
    pub(crate) group: Option<String>,
    /// Optional subset of landmark ids/names or group (macro|larger|fine|1,2,...).
    #[arg(long)]
    pub(crate) ids: Option<String>,
    /// Optional deterministic stratified row subsample size before CV.
    #[arg(long)]
    pub(crate) subsample: Option<usize>,
    /// Seed for deterministic stratified row subsampling (defaults to --seed).
    #[arg(long)]
    pub(crate) subsample_seed: Option<u64>,
    /// Write the LandmarkBundle v1 JSON for this measurement.
    #[arg(long)]
    pub(crate) emit_bundle: Option<PathBuf>,
    #[command(flatten)]
    pub(crate) common: LandmarkCommon,
}
#[derive(Args)]
pub(crate) struct LandmarkSyntheticArgs {
    pub(crate) model: PathBuf,
    #[arg(long, default_value_t = 4096)]
    pub(crate) rows: usize,
    #[command(flatten)]
    pub(crate) common: LandmarkCommon,
}
#[derive(Args)]
pub(crate) struct LandmarkCompareArgs {
    pub(crate) model: PathBuf,
    #[arg(long)]
    pub(crate) train: PathBuf,
    #[arg(long)]
    pub(crate) query: PathBuf,
    /// Diagnostic broad panel; core mirror always remains the selection score.
    #[arg(long)]
    pub(crate) broad: bool,
    #[command(flatten)]
    pub(crate) common: LandmarkCommon,
}
#[derive(Args)]
pub(crate) struct LevelArgs {
    pub(crate) csv: PathBuf,
    #[arg(short, long)]
    pub(crate) target: String,
    /// Highest level computed locally, independent of release permissions.
    #[arg(long, default_value = "high")]
    pub(crate) compute_through: String,
    /// Explicit release allowlist. all = low,medium,high; ultra is explicit.
    #[arg(long)]
    pub(crate) export_levels: String,
    /// New directory for authorized public objects and inventory.
    #[arg(short, long)]
    pub(crate) out: PathBuf,
    /// Generate deterministic artifact-derived 96-coordinate descriptors.
    #[arg(long)]
    pub(crate) descriptors: bool,
    /// Concatenate only authorized descriptors, with level boundaries and masks.
    #[arg(long, requires = "descriptors")]
    pub(crate) combine: bool,
    /// Optional PRIVATE report outside the release directory.
    #[arg(long)]
    pub(crate) private_report: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,
    #[arg(long,value_parser=["regression","classification"])]
    pub(crate) task: Option<String>,
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
    /// Print the fit report as one JSON object on stdout.
    #[arg(long, conflicts_with = "quiet")]
    pub(crate) json: bool,
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
pub(crate) struct DoctorArgs {
    /// Print runtime identity, available backends, thread count and executable digest.
    #[arg(long)]
    pub(crate) json: bool,
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
    /// Replace column names and labels with positional codes.
    #[arg(long)]
    pub(crate) anonymize: bool,
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
    /// Auditors: linear,gbdt,mlp; explicit CPU controls: linear-seeded-v1,mlp-seeded-v1.
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

#[derive(Parser)]
#[command(
    name = "dope-kernel",
    version,
    about = "Native DOPE Data Kernel V3 compiler and runtime"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Report this native executable and supported runtime profiles without data,
    /// optional imports, network access, or automatic weight installation.
    Doctor,
    /// Compile a headered CSV (numbers, categories, dates, missing cells) into one
    /// table artifact: an invertible column plan plus a packed, natively trained program.
    FitTable {
        #[arg(long)]
        csv: PathBuf,
        /// Target column name (required; never inferred from column order).
        #[arg(long)]
        target: String,
        #[arg(long)]
        out: PathBuf,
        /// Column to leave out (repeatable), for example identifiers.
        #[arg(long)]
        drop: Vec<String>,
        /// auto, regression or binary.
        #[arg(long, default_value = "auto")]
        task: String,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
        #[arg(long, default_value = "compact_neural_residual_symbolic")]
        candidate: String,
        #[arg(long, default_value = "features12_steps2048")]
        native_target_profile: String,
        /// Categories seen fewer times are pooled into one rare level.
        #[arg(long, default_value_t = 5)]
        min_level_count: usize,
    },
    /// Sample rows from a table artifact with the source header, categories and units.
    SampleTable {
        #[arg(long, visible_alias = "artifact")]
        model: PathBuf,
        #[arg(long)]
        rows: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Report a table artifact's bytes and column kinds without names or values.
    InspectTable {
        #[arg(long, visible_alias = "artifact")]
        model: PathBuf,
    },
    /// Explicit development mixed feature/mask carrier; old raw defaults stay fixed.
    CompileMixed {
        #[arg(long)]
        csv: PathBuf,
        #[arg(long)]
        schema: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_parser = ["independent_v1", "numeric_split_v1"])]
        profile: String,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
        #[arg(long, default_value_t = 32768)]
        max_complete_bytes: usize,
        #[arg(long, default_value_t = 60.0)]
        deadline_seconds: f64,
    },
    /// Inspect complete charged mixed state, without private dictionaries/probabilities.
    InspectMixed {
        #[arg(long)]
        artifact: PathBuf,
    },
    /// Write tagged JSON cells, or emit only a digest in a fresh sealed worker.
    SampleMixed {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        rows: usize,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
        #[arg(long)]
        isolated: bool,
    },
    /// Inspect a hash-bound development pilot plan without execution or admission.
    InspectDevelopmentPilot {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        expected_plan_digest: String,
        #[arg(long)]
        attempts: Option<PathBuf>,
    },
    /// Compile a headered CSV directly into a row-free DOPE artifact by inferring
    /// numeric/nominal column types from fit rows with opaque public IDs by default.
    CompileCsv {
        #[arg(long)]
        csv: PathBuf,
        /// Target column name or 0-based index (defaults to the last column).
        #[arg(long)]
        target: Option<String>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
        #[arg(long, default_value = "chow_liu")]
        candidate: String,
        #[arg(long, default_value_t = 10240)]
        max_complete_bytes: usize,
        #[arg(long, default_value_t = 60.0)]
        deadline_seconds: f64,
    },
    /// Fit a headered, declared continuous schema using fit rows only.
    CompileRaw {
        #[arg(long)]
        csv: PathBuf,
        #[arg(long)]
        schema: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        report: Option<PathBuf>,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
        #[arg(long, default_value = "chow_liu")]
        candidate: String,
        /// Explicit residual-v1 development comparison. Requires --candidate development_symbolic.
        #[arg(long)]
        residual_profile: Option<String>,
        #[arg(long, default_value_t = 32768)]
        max_complete_bytes: usize,
        #[arg(long, default_value_t = 60.0)]
        deadline_seconds: f64,
    },
    /// Sample original-unit values from a complete retained raw-state artifact.
    SampleRaw {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        rows: usize,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Inspect complete sampling bytes and explicit raw profile limitations.
    InspectRaw {
        #[arg(long)]
        artifact: PathBuf,
    },
    /// Fresh Linux worker: verified bytes only, closed inherited FDs, no new
    /// filesystem/network/process access. Redirect stdout and stderr to pipes/files.
    SampleRawIsolated {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        rows: usize,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
        /// Emit private row-major f64 little-endian bytes on stdout and the
        /// redacted receipt on stderr. Accept only on successful process exit.
        #[arg(long)]
        private_samples_f64le: bool,
    },
    /// Print the immutable release identity and contract digests.
    Version {
        #[arg(long)]
        json: bool,
    },
    /// Validate the embedded production KPI contract without opening data.
    FreezeContract,
    /// Construct an RC manifest after validating every frozen evidence gate.
    BuildRc {
        #[arg(long)]
        release_dir: PathBuf,
        #[arg(long)]
        gate_evidence: PathBuf,
        #[arg(long)]
        coverage: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        source_commit: String,
        #[arg(long)]
        source_tag_object_sha256: String,
        #[arg(long)]
        signature_kind: String,
        #[arg(long)]
        public_key_sha256: String,
    },
    /// Inspect or operate on a production campaign.
    Campaign {
        #[command(subcommand)]
        command: CampaignCommand,
    },
    /// Embed a supervised numeric training CSV with a fully qualified router.
    #[command(group(
        clap::ArgGroup::new("target_selector")
            .required(true)
            .multiple(false)
            .args(["target_column", "target_index"])
    ))]
    EmbedDataset {
        #[arg(long)]
        csv: PathBuf,
        #[arg(long)]
        target_column: Option<String>,
        #[arg(long)]
        target_index: Option<usize>,
        #[arg(long, value_parser = ["binary", "regression"])]
        task: String,
        #[arg(long)]
        router_bundle: PathBuf,
        #[arg(long)]
        router_evidence: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        vector_out: Option<PathBuf>,
        /// Include source headers and raw extrema in a restricted research artifact.
        #[arg(long, default_value_t = false)]
        restricted_metadata: bool,
    },
    /// Validate a router/evidence pair for production dataset embeddings.
    QualifyEmbeddingRouter {
        #[arg(long)]
        router_bundle: PathBuf,
        #[arg(long)]
        router_evidence: PathBuf,
    },
    /// Inspect real regression datasets and their reconstructable CSV headers.
    InspectRegressionCorpus {
        #[arg(long)]
        dataset_root: PathBuf,
    },
    /// Embed one deterministic shard of a real regression corpus.
    EmbedRegressionCorpus {
        #[arg(long)]
        dataset_root: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long)]
        router_bundle: PathBuf,
        #[arg(long)]
        router_evidence: PathBuf,
        #[arg(long)]
        method: String,
        #[arg(long)]
        expected_dimension: usize,
        #[arg(long)]
        shard_index: usize,
        #[arg(long)]
        shard_count: usize,
        #[arg(long, default_value_t = 32)]
        jobs: usize,
        #[arg(long, default_value_t = 1)]
        determinism_checks: usize,
        #[arg(long)]
        manifest_out: PathBuf,
    },
    /// Verify every consolidated regression embedding and its router lineage.
    VerifyRegressionEmbeddings {
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long)]
        router_bundle: PathBuf,
        #[arg(long)]
        router_evidence: PathBuf,
        #[arg(long)]
        method: String,
        #[arg(long)]
        expected_dimension: usize,
        #[arg(long)]
        expected_count: usize,
    },
    Convert {
        #[arg(long)]
        dataset_dir: PathBuf,
        #[arg(long)]
        real_holdout_dir: PathBuf,
        #[arg(long)]
        task: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
        #[arg(long, default_value_t = 0)]
        runtime_dictionary_bytes: usize,
        #[arg(long, default_value_t = 1)]
        supported_datasets: usize,
        #[arg(long)]
        synthetic_seed: Vec<u64>,
        #[arg(long, default_value = "l3")]
        tier: String,
        #[arg(long)]
        max_artifact_bytes: Option<usize>,
        #[arg(long)]
        require_formal_dp: bool,
    },
    #[command(visible_alias = "fit")]
    Compile {
        #[arg(long)]
        dataset_dir: PathBuf,
        #[arg(long)]
        task: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        report: Option<PathBuf>,
        #[arg(long)]
        pareto_frontier: Option<PathBuf>,
        /// Fit seed, also stored in the encoded artifact; sampling uses its own seed.
        #[arg(long, visible_alias = "fit-seed")]
        seed: Option<u64>,
        #[arg(long)]
        deadline_seconds: Option<f64>,
        #[arg(long)]
        language: Option<PathBuf>,
        #[arg(long)]
        candidate: Option<String>,
        #[arg(long, default_value_t = 2.0)]
        neural_target_weight: f64,
        #[arg(long, default_value_t = 0.0)]
        neural_structural_penalty: f64,
        /// Development-only encoded-feedback allocation; does not certify improvement.
        #[arg(long)]
        adaptive_allocation: bool,
        /// Train the compact neural residual in pure Rust with this profile,
        /// for example `features12_steps2048` (the published headline shape).
        #[arg(long)]
        native_target_profile: Option<String>,
        /// Store values as quantization-grid codes (packed DPK3 values).
        #[arg(long)]
        packed_codec: bool,
        #[arg(long, default_value = "l3")]
        tier: String,
        #[arg(long)]
        max_artifact_bytes: Option<usize>,
        #[arg(long)]
        require_formal_dp: bool,
    },
    Inspect {
        #[arg(long, visible_alias = "artifact")]
        kernel: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    Sample {
        #[arg(long, visible_alias = "artifact")]
        kernel: PathBuf,
        #[arg(long)]
        rows: usize,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        seed: Option<u64>,
    },
    PackCorpus {
        #[arg(long, required = true)]
        corpus: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
        #[arg(long, default_value_t = 256.0)]
        shard_mib: f64,
        #[arg(long)]
        archive_zstd: bool,
    },
    InventoryCorpus {
        #[arg(long, required = true)]
        corpus: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
    },
    FinalizeInventory {
        #[arg(long)]
        inventory: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
    },
    /// Create the path-free 60/40 validation-select/validation-cert assignment.
    SplitValidation {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
    },
    CalibrateLanguage {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
    },
    #[command(visible_alias = "evaluate", visible_alias = "verify")]
    Certify {
        #[arg(long)]
        real_dir: PathBuf,
        #[arg(long, visible_alias = "artifact")]
        kernel: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
        #[arg(long, default_value_t = 0)]
        runtime_dictionary_bytes: usize,
        #[arg(long, default_value_t = 1)]
        supported_datasets: usize,
        #[arg(long, default_value = "l3")]
        tier: String,
        #[arg(long)]
        max_artifact_bytes: Option<usize>,
        #[arg(long)]
        require_formal_dp: bool,
    },
}

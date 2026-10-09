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
        #[arg(long, default_value = "l3")]
        tier: String,
        #[arg(long)]
        max_artifact_bytes: Option<usize>,
        #[arg(long)]
        require_formal_dp: bool,
    },
    Inspect {
        #[arg(long)]
        kernel: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    Sample {
        #[arg(long)]
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
    Certify {
        #[arg(long)]
        real_dir: PathBuf,
        #[arg(long)]
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

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
        #[arg(long)]
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

#[derive(Subcommand)]
enum CampaignCommand {
    /// Freeze deep source, binary, evidence, and implementation hashes.
    FreezeDeepImplementation {
        #[arg(long)]
        source_tree: PathBuf,
        #[arg(long)]
        binary: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Create a new mode-0600 campaign receipt key from OS randomness.
    CreateReceiptKey {
        #[arg(long)]
        out: PathBuf,
    },
    /// Freeze immutable campaign inputs and initialize the durable ledger.
    Freeze {
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        specs: PathBuf,
        #[arg(long)]
        source_commit: String,
        #[arg(long)]
        source_tree: PathBuf,
        #[arg(long)]
        environment_lock: PathBuf,
        #[arg(long)]
        host_inventory: PathBuf,
        #[arg(long)]
        source_hashes: PathBuf,
        #[arg(long)]
        sbom: PathBuf,
        #[arg(long)]
        notices: PathBuf,
        #[arg(long)]
        checkpoint_manifest: PathBuf,
        #[arg(long)]
        corpus_manifest: PathBuf,
        #[arg(long)]
        validation_manifest: PathBuf,
        #[arg(long)]
        receipt_key: PathBuf,
    },
    /// Serve the networked lease/heartbeat/receipt controller.
    Serve {
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long, default_value = "0.0.0.0:9797")]
        bind: String,
        #[arg(long)]
        receipt_key: PathBuf,
        #[arg(long, hide = true)]
        max_requests: Option<usize>,
    },
    /// Run immutable jobs leased from the campaign controller.
    Worker {
        #[arg(long)]
        controller: String,
        #[arg(long)]
        worker: String,
        #[arg(long)]
        work_dir: PathBuf,
        #[arg(long)]
        receipt_key: PathBuf,
        #[arg(long)]
        once: bool,
    },
    /// Verify the deterministic path-free 1% corpus sample.
    VerifyCorpus {
        #[arg(long)]
        manifest: PathBuf,
    },
    /// Export the exact deterministic corpus sample paths for staging.
    ExportCorpusSample {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Select a deterministic baseline, training-gold, or validation-select cohort.
    PlanCohort {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        validation_manifest: PathBuf,
        #[arg(long)]
        kind: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Split training-gold into frozen, disjoint deep discovery and confirmation cohorts.
    PlanDeepCohorts {
        #[arg(long)]
        training_gold: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Materialize one frozen deep cohort stage for the gold-matrix runner.
    MaterializeDeepCohort {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        stage: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Rebuild measured and failed deep outcomes from durable gold blocks.
    BuildDeepEvidence {
        #[arg(long)]
        cohort: PathBuf,
        #[arg(long, required = true)]
        block: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Select one safe frozen loss configuration per joint architecture.
    SelectDeepDiscovery {
        #[arg(long)]
        evidence: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Apply the paired confirmation promotion gates to one joint architecture.
    QualifyDeepConfirmation {
        #[arg(long)]
        candidate: String,
        #[arg(long)]
        evidence: PathBuf,
        #[arg(long)]
        scheduled_cells: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Select the deterministic final family from passing validation-select decisions.
    SelectDeepFinal {
        #[arg(long)]
        evidence: PathBuf,
        #[arg(long, required = true)]
        qualification: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Build a content-addressed per-dataset DPK3.2 artifact manifest.
    BuildDeepArtifactManifest {
        #[arg(long)]
        selection: PathBuf,
        /// Dataset ID and artifact path as DATASET_ID=PATH.
        #[arg(long, required = true)]
        artifact: Vec<String>,
        #[arg(long)]
        representative_dataset_id: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Render the measured deep-joint model card.
    BuildDeepModelCard {
        #[arg(long)]
        selection: PathBuf,
        #[arg(long)]
        report: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Publish measured deep evidence slices and explicit failed/missing states.
    ReportDeepCampaign {
        #[arg(long)]
        evidence: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Materialize the guarded validation-cert cohort after its one authorization event.
    PlanValidationCert {
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        validation_manifest: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Run the restart-safe first baseline matrix over a frozen cohort shard.
    RunBaseline {
        #[arg(long)]
        cohort: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, required = true)]
        candidate: Vec<String>,
        #[arg(long, required = true)]
        auditor: Vec<String>,
        #[arg(long, default_value_t = 0)]
        shard_index: usize,
        #[arg(long, default_value_t = 1)]
        shard_count: usize,
        #[arg(long)]
        primary_only: bool,
    },
    /// Aggregate measured JSONL cells without inventing missing evidence.
    SummarizeEvidence {
        #[arg(long, required = true)]
        metrics: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Run an exhaustive, restart-safe training/select gold matrix shard.
    RunGoldMatrix {
        #[arg(long)]
        cohort: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, required = true)]
        candidate: Vec<String>,
        #[arg(long, required = true)]
        auditor: Vec<String>,
        #[arg(long, required = true)]
        size_multiplier: Vec<usize>,
        #[arg(long, required = true)]
        generation_seed: Vec<u64>,
        #[arg(long, required = true)]
        auditor_seed: Vec<u64>,
        #[arg(long, default_value_t = 0)]
        shard_index: usize,
        #[arg(long, default_value_t = 1)]
        shard_count: usize,
        #[arg(long)]
        primary_only: bool,
        #[arg(long)]
        blocks_only: bool,
        #[arg(long, default_value_t = 2.0)]
        neural_target_weight: f64,
        #[arg(long, default_value_t = 0.0)]
        neural_structural_penalty: f64,
    },
    /// Run frozen routed validation-cert plus exhaustive shadow evidence.
    RunRoutedCert {
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        cohort: PathBuf,
        #[arg(long)]
        router_bundle: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 0)]
        shard_index: usize,
        #[arg(long, default_value_t = 1)]
        shard_count: usize,
    },
    /// Validate and sign content-addressed exhaustive gold record blocks.
    SignGoldBlocks {
        #[arg(long)]
        cohort: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        worker: String,
        #[arg(long)]
        source_sha256: String,
        #[arg(long)]
        evaluator_binary_sha256: String,
        #[arg(long)]
        router_bundle: Option<PathBuf>,
        #[arg(long)]
        receipt_key: PathBuf,
    },
    /// Train the GPU teacher and distill an int8 Rust router from measured labels.
    TrainRouter {
        #[arg(long, required = true)]
        cohort: Vec<PathBuf>,
        #[arg(long, required = true)]
        metrics: Vec<PathBuf>,
        #[arg(long)]
        sketch_cache: Option<PathBuf>,
        #[arg(long)]
        bundle_out: PathBuf,
        #[arg(long)]
        report_out: PathBuf,
        #[arg(long)]
        evidence_out: PathBuf,
    },
    /// Benchmark the invariant sketch and optional frozen int8 router.
    BenchmarkRouter {
        #[arg(long)]
        bundle: Option<PathBuf>,
        #[arg(long, default_value_t = 10_000)]
        rows: usize,
        #[arg(long, default_value_t = 500)]
        features: usize,
        #[arg(long, default_value_t = 5)]
        repeats: usize,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Execute one frozen candidate/auditor gold cell and emit typed evidence.
    EvaluateGoldCell {
        #[arg(long)]
        dataset_dir: PathBuf,
        #[arg(long)]
        task: String,
        #[arg(long)]
        candidate: String,
        #[arg(long)]
        auditor: String,
        #[arg(long)]
        lineage: String,
        #[arg(long)]
        profile: String,
        #[arg(long)]
        phase: String,
        #[arg(long)]
        size_multiplier: usize,
        #[arg(long)]
        generation_seed: u64,
        #[arg(long)]
        auditor_seed: u64,
        #[arg(long)]
        routed: bool,
        #[arg(long)]
        primary_only: bool,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 2.0)]
        neural_target_weight: f64,
        #[arg(long, default_value_t = 0.0)]
        neural_structural_penalty: f64,
    },
    /// Inspect a resumable campaign ledger.
    Status {
        #[arg(long)]
        ledger: Option<PathBuf>,
        #[arg(long)]
        state_dir: Option<PathBuf>,
        #[arg(long)]
        gold_out: Option<PathBuf>,
        #[arg(long)]
        cohort: Option<PathBuf>,
        #[arg(long)]
        router_bundle: Option<PathBuf>,
        #[arg(long)]
        metrics: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Reconstruct canonical metrics exclusively from signed receipts.
    ExportMetrics {
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        receipt_key: PathBuf,
    },
    /// Reconcile xbabe3 block receipts and publish final validation-cert metrics.
    ExportValidationCertMetrics {
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        cert_out: PathBuf,
        #[arg(long)]
        cohort: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        receipt_key: PathBuf,
    },
    /// Freeze an int8 Rust router; failed evidence requires an explicit measured-frontier flag.
    FreezeRouter {
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        evidence: PathBuf,
        #[arg(long)]
        allow_failed_frontier: bool,
    },
    /// Record the one allowed validation-cert authorization event.
    OpenValidationCert {
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        authorization: PathBuf,
    },
    /// Publish a measured failed frontier when RC gates are not satisfied.
    BuildRc {
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        metrics: PathBuf,
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
}

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand};
use dope_kernel::calibration::{TrainingOptions, load_language, train_language};
use dope_kernel::campaign::{
    CohortPlan, FreezeOptions, GoldMatrixOptions, RoutedCertOptions, benchmark_router,
    corpus_sample_plan, create_receipt_key, export_metrics, export_validation_cert_metrics,
    failed_release_decision, freeze, freeze_router, gold_block_status, ledger_path,
    open_validation_cert, plan_cohort, plan_validation_cert_cohort, run_baseline, run_gold_matrix,
    run_routed_cert_matrix, serve, sign_gold_blocks, summarize_evidence, train_router_from_metrics,
    worker,
};
use dope_kernel::certification::certify_kernel;
use dope_kernel::certification::{GoldCellOptions, evaluate_gold_cell};
use dope_kernel::codec::load_kernel;
use dope_kernel::compiler::{CompileOptions, compile_kernel_from_dir};
use dope_kernel::corpus::{
    build_split_manifest_multi, load_inventory, refresh_inventory_lineages, split_inventory,
    split_validation_manifest, write_exclusions, write_inventory, write_split_manifest,
};
use dope_kernel::deep_campaign::{
    DeepCampaignReport, DeepCohortPlan, DeepOutcome, FinalFamilySelection, Qualification,
    build_deep_artifact_manifest, build_deep_campaign_report, build_deep_model_card,
    freeze_deep_implementation, plan_deep_cohorts, qualify_confirmation_candidate,
    rebuild_deep_outcomes, select_discovery_configurations, select_final_family,
};
use dope_kernel::embedding::{
    TargetSelector, embed_dataset, qualify_embedding_router, write_embedding_csv,
};
use dope_kernel::inspect_kernel;
use dope_kernel::model::Task;
use dope_kernel::production::{
    BuildRcOptions, CoverageReport, DEVELOPMENT_VERSION, GateEvidence, KPI_CONTRACT_BLAKE3,
    KPI_CONTRACT_SHA256, KpiContract, build_rc_manifest,
};
use dope_kernel::regression_embeddings::{
    RegressionEmbeddingBatchOptions, embed_regression_corpus, inspect_regression_corpus,
    verify_regression_embeddings,
};
use dope_kernel::release::{ConversionOptions, convert_release};
use dope_kernel::sample::{SampleOptions, sample_kernel_to_csv};
use dope_kernel::{DopeError, Result};

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

fn parse_task(value: &str) -> Result<Task> {
    Task::parse(value).ok_or_else(|| DopeError::Data("task must be regression or binary".into()))
}

fn write_json(path: &PathBuf, value: &impl serde::Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(path, bytes).map_err(|error| dope_kernel::error::io_error(path, error))
}

fn run(command: Command) -> Result<()> {
    match command {
        Command::Version { json } => {
            let value = serde_json::json!({
                "name": "dope-kernel",
                "version": DEVELOPMENT_VERSION,
                "kpi_contract": {"sha256": KPI_CONTRACT_SHA256, "blake3": KPI_CONTRACT_BLAKE3}
            });
            if json {
                println!("{}", serde_json::to_string(&value)?);
            } else {
                println!("dope-kernel {DEVELOPMENT_VERSION}");
            }
        }
        Command::FreezeContract => {
            let contract = KpiContract::embedded()?;
            contract.validate()?;
            println!(
                "{}",
                serde_json::json!({"frozen": true, "release_identity": DEVELOPMENT_VERSION, "kpi_contract_sha256": KPI_CONTRACT_SHA256, "kpi_contract_blake3": KPI_CONTRACT_BLAKE3})
            );
        }
        Command::BuildRc {
            release_dir,
            gate_evidence,
            coverage,
            out,
            source_commit,
            source_tag_object_sha256,
            signature_kind,
            public_key_sha256,
        } => {
            let evidence: GateEvidence = serde_json::from_slice(
                &fs::read(&gate_evidence)
                    .map_err(|error| dope_kernel::error::io_error(&gate_evidence, error))?,
            )?;
            let coverage: CoverageReport = serde_json::from_slice(
                &fs::read(&coverage)
                    .map_err(|error| dope_kernel::error::io_error(&coverage, error))?,
            )?;
            let manifest = build_rc_manifest(
                &release_dir,
                &evidence,
                &coverage,
                &BuildRcOptions {
                    source_commit,
                    source_tag_object_sha256,
                    signature_kind,
                    public_key_sha256,
                },
            )?;
            write_json(&out, &manifest)?;
            println!("{}", serde_json::to_string(&manifest)?);
        }
        Command::Campaign { command } => match command {
            CampaignCommand::FreezeDeepImplementation {
                source_tree,
                binary,
                out,
            } => {
                let manifest = freeze_deep_implementation(&source_tree, &binary)?;
                dope_kernel::production::write_canonical(&out, &manifest)?;
                println!("{}", serde_json::to_string(&manifest)?);
            }
            CampaignCommand::CreateReceiptKey { out } => {
                let hashes = create_receipt_key(&out)?;
                println!("{}", serde_json::to_string(&hashes)?);
            }
            CampaignCommand::Freeze {
                state_dir,
                specs,
                source_commit,
                source_tree,
                environment_lock,
                host_inventory,
                source_hashes,
                sbom,
                notices,
                checkpoint_manifest,
                corpus_manifest,
                validation_manifest,
                receipt_key,
            } => {
                let state = freeze(&FreezeOptions {
                    state_dir: &state_dir,
                    specs: &specs,
                    source_commit: &source_commit,
                    source_tree: &source_tree,
                    environment_lock: &environment_lock,
                    host_inventory: &host_inventory,
                    source_hashes: &source_hashes,
                    sbom: &sbom,
                    notices: &notices,
                    checkpoint_manifest: &checkpoint_manifest,
                    corpus_manifest: &corpus_manifest,
                    validation_manifest: &validation_manifest,
                    receipt_key: &receipt_key,
                })?;
                println!("{}", serde_json::to_string(&state)?);
            }
            CampaignCommand::Serve {
                state_dir,
                bind,
                receipt_key,
                max_requests,
            } => {
                serve(&state_dir, &bind, &receipt_key, max_requests)?;
            }
            CampaignCommand::Worker {
                controller,
                worker: worker_id,
                work_dir,
                receipt_key,
                once,
            } => {
                let completed = worker(&controller, &worker_id, &work_dir, &receipt_key, once)?;
                println!(
                    "{}",
                    serde_json::json!({"completed": completed, "worker": worker_id})
                );
            }
            CampaignCommand::VerifyCorpus { manifest } => {
                let verification = dope_kernel::campaign::verify_corpus_sample(&manifest)?;
                println!("{}", serde_json::to_string(&verification)?);
            }
            CampaignCommand::ExportCorpusSample { manifest, out } => {
                let plan = corpus_sample_plan(&manifest)?;
                dope_kernel::production::write_canonical(&out, &plan)?;
                println!(
                    "{}",
                    serde_json::json!({"paths": plan.paths.len(), "out": out})
                );
            }
            CampaignCommand::PlanCohort {
                manifest,
                validation_manifest,
                kind,
                out,
            } => {
                let cohort = plan_cohort(&manifest, &validation_manifest, &kind)?;
                dope_kernel::production::write_canonical(&out, &cohort)?;
                println!(
                    "{}",
                    serde_json::json!({"kind": cohort.kind, "records": cohort.records.len(), "exclusions": cohort.exclusions.len(), "out": out})
                );
            }
            CampaignCommand::PlanDeepCohorts { training_gold, out } => {
                let cohort: CohortPlan = serde_json::from_slice(
                    &fs::read(&training_gold)
                        .map_err(|error| dope_kernel::error::io_error(&training_gold, error))?,
                )?;
                let plan = plan_deep_cohorts(&cohort)?;
                dope_kernel::production::write_canonical(&out, &plan)?;
                println!(
                    "{}",
                    serde_json::json!({"discovery": plan.discovery.len(), "confirmation": plan.confirmation.len(), "out": out})
                );
            }
            CampaignCommand::MaterializeDeepCohort { plan, stage, out } => {
                let plan: DeepCohortPlan = serde_json::from_slice(
                    &fs::read(&plan).map_err(|error| dope_kernel::error::io_error(&plan, error))?,
                )?;
                let cohort = plan.stage_cohort(&stage)?;
                dope_kernel::production::write_canonical(&out, &cohort)?;
                println!(
                    "{}",
                    serde_json::json!({"stage": stage, "records": cohort.records.len(), "out": out})
                );
            }
            CampaignCommand::BuildDeepEvidence { cohort, block, out } => {
                let cohort_plan: CohortPlan = serde_json::from_slice(
                    &fs::read(&cohort)
                        .map_err(|error| dope_kernel::error::io_error(&cohort, error))?,
                )?;
                let outcomes = rebuild_deep_outcomes(&cohort_plan, &block)?;
                dope_kernel::production::write_canonical(&out, &outcomes)?;
                println!(
                    "{}",
                    serde_json::json!({"outcomes": outcomes.len(), "out": out})
                );
            }
            CampaignCommand::SelectDeepDiscovery { evidence, out } => {
                let outcomes: Vec<DeepOutcome> = serde_json::from_slice(
                    &fs::read(&evidence)
                        .map_err(|error| dope_kernel::error::io_error(&evidence, error))?,
                )?;
                let selection = select_discovery_configurations(&outcomes)?;
                dope_kernel::production::write_canonical(&out, &selection)?;
                println!("{}", serde_json::to_string(&selection)?);
            }
            CampaignCommand::QualifyDeepConfirmation {
                candidate,
                evidence,
                scheduled_cells,
                out,
            } => {
                let outcomes: Vec<DeepOutcome> = serde_json::from_slice(
                    &fs::read(&evidence)
                        .map_err(|error| dope_kernel::error::io_error(&evidence, error))?,
                )?;
                let qualification =
                    qualify_confirmation_candidate(&candidate, &outcomes, scheduled_cells)?;
                dope_kernel::production::write_canonical(&out, &qualification)?;
                println!("{}", serde_json::to_string(&qualification)?);
            }
            CampaignCommand::SelectDeepFinal {
                evidence,
                qualification,
                out,
            } => {
                let outcomes: Vec<DeepOutcome> = serde_json::from_slice(
                    &fs::read(&evidence)
                        .map_err(|error| dope_kernel::error::io_error(&evidence, error))?,
                )?;
                let qualifications = qualification
                    .iter()
                    .map(|path| {
                        fs::read(path)
                            .map_err(|error| dope_kernel::error::io_error(path, error))
                            .and_then(|bytes| {
                                serde_json::from_slice::<Qualification>(&bytes).map_err(Into::into)
                            })
                    })
                    .collect::<dope_kernel::Result<Vec<_>>>()?;
                let selection = select_final_family(&outcomes, &qualifications)?;
                dope_kernel::production::write_canonical(&out, &selection)?;
                println!("{}", serde_json::to_string(&selection)?);
            }
            CampaignCommand::BuildDeepArtifactManifest {
                selection,
                artifact,
                representative_dataset_id,
                out,
            } => {
                let selection: FinalFamilySelection = serde_json::from_slice(
                    &fs::read(&selection)
                        .map_err(|error| dope_kernel::error::io_error(&selection, error))?,
                )?;
                let artifacts = artifact
                    .iter()
                    .map(|entry| {
                        let (dataset_id, path) = entry.split_once('=').ok_or_else(|| {
                            dope_kernel::DopeError::Data(
                                "deep artifact must use DATASET_ID=PATH".into(),
                            )
                        })?;
                        Ok((dataset_id.to_string(), PathBuf::from(path)))
                    })
                    .collect::<dope_kernel::Result<Vec<_>>>()?;
                let manifest = build_deep_artifact_manifest(
                    &selection,
                    &artifacts,
                    &representative_dataset_id,
                )?;
                dope_kernel::production::write_canonical(&out, &manifest)?;
                println!("{}", serde_json::to_string(&manifest)?);
            }
            CampaignCommand::BuildDeepModelCard {
                selection,
                report,
                out,
            } => {
                let selection: FinalFamilySelection = serde_json::from_slice(
                    &fs::read(&selection)
                        .map_err(|error| dope_kernel::error::io_error(&selection, error))?,
                )?;
                let report: DeepCampaignReport = serde_json::from_slice(
                    &fs::read(&report)
                        .map_err(|error| dope_kernel::error::io_error(&report, error))?,
                )?;
                let card = build_deep_model_card(&selection, &report)?;
                fs::write(&out, card).map_err(|error| dope_kernel::error::io_error(&out, error))?;
                println!("{}", serde_json::json!({"out": out}));
            }
            CampaignCommand::ReportDeepCampaign { evidence, out } => {
                let outcomes: Vec<DeepOutcome> = serde_json::from_slice(
                    &fs::read(&evidence)
                        .map_err(|error| dope_kernel::error::io_error(&evidence, error))?,
                )?;
                let report = build_deep_campaign_report(&outcomes)?;
                dope_kernel::production::write_canonical(&out, &report)?;
                println!(
                    "{}",
                    serde_json::json!({"slices": report.slices.len(), "failures": report.failures.len(), "out": out})
                );
            }
            CampaignCommand::PlanValidationCert {
                state_dir,
                manifest,
                validation_manifest,
                out,
            } => {
                let cohort =
                    plan_validation_cert_cohort(&state_dir, &manifest, &validation_manifest)?;
                dope_kernel::production::write_canonical(&out, &cohort)?;
                println!(
                    "{}",
                    serde_json::json!({"kind": cohort.kind, "records": cohort.records.len(), "out": out})
                );
            }
            CampaignCommand::RunBaseline {
                cohort,
                out,
                candidate,
                auditor,
                shard_index,
                shard_count,
                primary_only,
            } => {
                let summary = run_baseline(
                    &cohort,
                    &out,
                    &candidate,
                    &auditor,
                    shard_index,
                    shard_count,
                    primary_only,
                )?;
                println!("{}", serde_json::to_string(&summary)?);
            }
            CampaignCommand::SummarizeEvidence { metrics, out } => {
                let summary = summarize_evidence(&metrics, &out)?;
                println!("{}", serde_json::to_string(&summary)?);
            }
            CampaignCommand::RunGoldMatrix {
                cohort,
                out,
                candidate,
                auditor,
                size_multiplier,
                generation_seed,
                auditor_seed,
                shard_index,
                shard_count,
                primary_only,
                blocks_only,
                neural_target_weight,
                neural_structural_penalty,
            } => {
                let summary = run_gold_matrix(&GoldMatrixOptions {
                    cohort_path: &cohort,
                    out: &out,
                    candidate_ids: &candidate,
                    auditor_ids: &auditor,
                    size_multipliers: &size_multiplier,
                    generation_seeds: &generation_seed,
                    auditor_seeds: &auditor_seed,
                    shard_index,
                    shard_count,
                    primary_only,
                    blocks_only,
                    neural_target_weight,
                    neural_structural_penalty,
                })?;
                println!("{}", serde_json::to_string(&summary)?);
            }
            CampaignCommand::RunRoutedCert {
                state_dir,
                cohort,
                router_bundle,
                out,
                shard_index,
                shard_count,
            } => {
                let summary = run_routed_cert_matrix(&RoutedCertOptions {
                    state_dir: &state_dir,
                    cohort_path: &cohort,
                    router_bundle: &router_bundle,
                    out: &out,
                    shard_index,
                    shard_count,
                })?;
                println!("{}", serde_json::to_string(&summary)?);
            }
            CampaignCommand::SignGoldBlocks {
                cohort,
                out,
                worker,
                source_sha256,
                evaluator_binary_sha256,
                router_bundle,
                receipt_key,
            } => {
                let summary = sign_gold_blocks(
                    &out,
                    &cohort,
                    &worker,
                    &source_sha256,
                    &evaluator_binary_sha256,
                    router_bundle.as_deref(),
                    &receipt_key,
                )?;
                println!("{}", serde_json::to_string(&summary)?);
            }
            CampaignCommand::TrainRouter {
                cohort,
                metrics,
                sketch_cache,
                bundle_out,
                report_out,
                evidence_out,
            } => {
                let (report, evidence) = train_router_from_metrics(
                    &cohort,
                    &metrics,
                    sketch_cache.as_deref(),
                    &bundle_out,
                    &report_out,
                    &evidence_out,
                )?;
                println!(
                    "{}",
                    serde_json::json!({
                        "report": {
                            "training_lineages": report.training_lineages,
                            "validation_lineages": report.validation_lineages,
                            "labels": report.labels,
                            "teacher_top_four_oracle_recall": report.teacher_top_four_oracle_recall,
                            "teacher_maximum_regret": report.teacher_maximum_regret,
                            "teacher_mean_regret": report.teacher_mean_regret,
                            "student_top_four_oracle_recall": report.student_top_four_oracle_recall,
                            "student_maximum_regret": report.student_maximum_regret,
                            "student_maximum_profile_regret_upper": report.student_maximum_profile_regret_upper,
                            "student_mean_regret": report.student_mean_regret,
                            "random_mean_regret": report.random_mean_regret,
                            "best_fixed_candidate_id": report.best_fixed_candidate_id,
                            "best_fixed_mean_regret": report.best_fixed_mean_regret,
                            "ga2m_mean_regret": report.ga2m_mean_regret,
                            "ga2m_top_four_oracle_recall": report.ga2m_top_four_oracle_recall,
                            "ga2m_maximum_profile_regret_upper": report.ga2m_maximum_profile_regret_upper,
                            "paired_hypervolume_improvement": report.paired_hypervolume_improvement,
                            "paired_hypervolume_ci_lower": report.paired_hypervolume_ci_lower,
                            "student_max_abs_difference": report.student_max_abs_difference,
                            "student_max_abs_difference_by_output": report.student_max_abs_difference_by_output,
                            "top_choice_agreement": report.top_choice_agreement,
                            "bundle_bytes": report.bundle_bytes,
                            "teacher_training_seconds": report.teacher_training_seconds,
                        },
                        "promotion_failed_gates": evidence.failed_gates(),
                        "bundle": bundle_out,
                        "evidence": evidence_out
                    })
                );
            }
            CampaignCommand::BenchmarkRouter {
                bundle,
                rows,
                features,
                repeats,
                out,
            } => {
                let report = benchmark_router(bundle.as_deref(), rows, features, repeats)?;
                if let Some(out) = out {
                    dope_kernel::production::write_canonical(&out, &report)?;
                }
                println!("{}", serde_json::to_string(&report)?);
            }
            CampaignCommand::EvaluateGoldCell {
                dataset_dir,
                task,
                candidate,
                auditor,
                lineage,
                profile,
                phase,
                size_multiplier,
                generation_seed,
                auditor_seed,
                routed,
                primary_only,
                cache_dir,
                out,
                neural_target_weight,
                neural_structural_penalty,
            } => {
                let evidence = evaluate_gold_cell(&GoldCellOptions {
                    dataset_dir: &dataset_dir,
                    task: parse_task(&task)?,
                    candidate_id: &candidate,
                    auditor_id: &auditor,
                    lineage_group_id: &lineage,
                    structural_profile: &profile,
                    phase: &phase,
                    size_multiplier,
                    generation_seed,
                    auditor_seed,
                    routed,
                    ancillary: !primary_only,
                    cache_dir: cache_dir.as_deref(),
                    neural_target_weight,
                    neural_structural_penalty,
                })?;
                dope_kernel::production::write_canonical(&out, &evidence)?;
                println!("{}", serde_json::to_string(&evidence)?);
            }
            CampaignCommand::Status {
                ledger,
                state_dir,
                gold_out,
                cohort,
                router_bundle,
                metrics,
                json,
            } => {
                if let (Some(gold_out), Some(cohort)) = (gold_out.as_ref(), cohort.as_ref()) {
                    if ledger.is_some() || state_dir.is_some() {
                        return Err(DopeError::Data(
                            "gold block status cannot be combined with ledger status".into(),
                        ));
                    }
                    let status = gold_block_status(
                        cohort,
                        gold_out,
                        router_bundle.as_deref(),
                        metrics.as_deref(),
                    )?;
                    if json {
                        println!("{}", serde_json::to_string(&status)?);
                    } else {
                        println!(
                            "gold lineages: {}/{}; cells: {}/{}; failures: {}; throughput: {}; coverage: {:.6}; router regret: {}; PTF-v1: {}",
                            status.completed_lineages,
                            status.eligible_lineages,
                            status.completed_cells,
                            status.attempted_cells,
                            status.failed_cells,
                            status.throughput_lineages_per_hour.map_or_else(
                                || "unmeasured".into(),
                                |value| format!("{value:.2}/hour")
                            ),
                            status.coverage,
                            status
                                .router_regret
                                .map_or_else(|| "unmeasured".into(), |value| format!("{value:.6}")),
                            status
                                .ptf_v1
                                .map_or_else(|| "unmeasured".into(), |value| format!("{value:.6}")),
                        );
                    }
                    return Ok(());
                }
                if gold_out.is_some()
                    || cohort.is_some()
                    || router_bundle.is_some()
                    || metrics.is_some()
                {
                    return Err(DopeError::Data(
                        "gold block status requires both --gold-out and --cohort".into(),
                    ));
                }
                let path = match (ledger, state_dir) {
                    (Some(path), None) => path,
                    (None, Some(state_dir)) => ledger_path(&state_dir),
                    _ => {
                        return Err(DopeError::Data(
                            "campaign status requires exactly one of --ledger or --state-dir"
                                .into(),
                        ));
                    }
                };
                let status = dope_kernel::CampaignLedger::open(&path)?.status()?;
                if json {
                    println!("{}", serde_json::to_string(&status)?);
                } else {
                    println!(
                        "gold labels: {}/{}; failures: {}; timeouts: {}; throughput: {}; coverage: {}; router regret: {}; PTF-v1: {}",
                        status.completed_gold_labels,
                        status.total,
                        status.failed_jobs,
                        status.timed_out_jobs,
                        status
                            .throughput_jobs_per_hour
                            .map_or_else(|| "unmeasured".into(), |v| format!("{v:.2}/hour")),
                        status
                            .coverage
                            .map_or_else(|| "unmeasured".into(), |v| format!("{v:.6}")),
                        status
                            .current_router_validation_regret
                            .map_or_else(|| "unmeasured".into(), |v| format!("{v:.6}")),
                        status
                            .ptf_v1
                            .map_or_else(|| "unmeasured".into(), |v| format!("{v:.6}")),
                    );
                }
            }
            CampaignCommand::ExportMetrics {
                state_dir,
                out,
                receipt_key,
            } => {
                let metrics = export_metrics(&state_dir, &out, &receipt_key)?;
                println!("{}", serde_json::to_string(&metrics)?);
            }
            CampaignCommand::ExportValidationCertMetrics {
                state_dir,
                cert_out,
                cohort,
                out,
                receipt_key,
            } => {
                let metrics = export_validation_cert_metrics(
                    &state_dir,
                    &cert_out,
                    &cohort,
                    &out,
                    &receipt_key,
                )?;
                println!("{}", serde_json::to_string(&metrics)?);
            }
            CampaignCommand::FreezeRouter {
                state_dir,
                bundle,
                evidence,
                allow_failed_frontier,
            } => {
                let state = freeze_router(&state_dir, &bundle, &evidence, allow_failed_frontier)?;
                println!("{}", serde_json::to_string(&state)?);
            }
            CampaignCommand::OpenValidationCert {
                state_dir,
                authorization,
            } => {
                let state = open_validation_cert(&state_dir, &authorization)?;
                println!("{}", serde_json::to_string(&state)?);
            }
            CampaignCommand::BuildRc {
                state_dir,
                metrics,
                release_dir,
                gate_evidence,
                coverage,
                out,
                source_commit,
                source_tag_object_sha256,
                signature_kind,
                public_key_sha256,
            } => {
                let metrics: dope_kernel::campaign::MetricsBundle = serde_json::from_slice(
                    &fs::read(&metrics)
                        .map_err(|error| dope_kernel::error::io_error(&metrics, error))?,
                )?;
                if !metrics.failed_gates.is_empty() {
                    let decision = failed_release_decision(&state_dir, &metrics)?;
                    dope_kernel::production::write_canonical(&out, &decision)?;
                    println!("{}", serde_json::to_string(&decision)?);
                } else {
                    let evidence: GateEvidence =
                        serde_json::from_slice(&fs::read(&gate_evidence).map_err(|error| {
                            dope_kernel::error::io_error(&gate_evidence, error)
                        })?)?;
                    let coverage: CoverageReport = serde_json::from_slice(
                        &fs::read(&coverage)
                            .map_err(|error| dope_kernel::error::io_error(&coverage, error))?,
                    )?;
                    let manifest = build_rc_manifest(
                        &release_dir,
                        &evidence,
                        &coverage,
                        &BuildRcOptions {
                            source_commit,
                            source_tag_object_sha256,
                            signature_kind,
                            public_key_sha256,
                        },
                    )?;
                    dope_kernel::production::write_canonical(&out, &manifest)?;
                    println!("{}", serde_json::to_string(&manifest)?);
                }
            }
        },
        Command::EmbedDataset {
            csv,
            target_column,
            target_index,
            task,
            router_bundle,
            router_evidence,
            out,
            vector_out,
        } => {
            let selector = match (target_column.as_deref(), target_index) {
                (Some(name), None) => TargetSelector::Name(name),
                (None, Some(index)) => TargetSelector::Index(index),
                _ => {
                    return Err(DopeError::Data(
                        "exactly one target column selector is required".into(),
                    ));
                }
            };
            let embedding = embed_dataset(
                &csv,
                selector,
                parse_task(&task)?,
                &router_bundle,
                &router_evidence,
            )?;
            dope_kernel::production::write_canonical(&out, &embedding)?;
            if let Some(path) = &vector_out {
                write_embedding_csv(path, &embedding)?;
            }
            println!(
                "{}",
                serde_json::json!({
                    "dimension": embedding.dimension,
                    "candidates": embedding.candidate_order,
                    "schema_sha256": embedding.schema_sha256,
                    "out": out,
                    "vector_out": vector_out,
                })
            );
        }
        Command::QualifyEmbeddingRouter {
            router_bundle,
            router_evidence,
        } => {
            let qualification = qualify_embedding_router(&router_bundle, &router_evidence)?;
            println!("{}", serde_json::to_string(&qualification)?);
        }
        Command::InspectRegressionCorpus { dataset_root } => {
            let inventory = inspect_regression_corpus(&dataset_root)?;
            println!("{}", serde_json::to_string(&inventory)?);
        }
        Command::EmbedRegressionCorpus {
            dataset_root,
            output_dir,
            router_bundle,
            router_evidence,
            method,
            expected_dimension,
            shard_index,
            shard_count,
            jobs,
            determinism_checks,
            manifest_out,
        } => {
            let manifest = embed_regression_corpus(&RegressionEmbeddingBatchOptions {
                dataset_root: &dataset_root,
                output_dir: &output_dir,
                router_bundle: &router_bundle,
                router_evidence: &router_evidence,
                method: &method,
                expected_dimension,
                shard_index,
                shard_count,
                jobs,
                determinism_checks,
            })?;
            dope_kernel::production::write_canonical(&manifest_out, &manifest)?;
            println!(
                "{}",
                serde_json::json!({
                    "discovered": manifest.discovered,
                    "assigned": manifest.assigned,
                    "success": manifest.success,
                    "schema_mismatch": manifest.schema_mismatch,
                    "failed": manifest.failed,
                    "determinism_checked": manifest.determinism_checked,
                    "dimension": manifest.router.dimension,
                    "manifest_out": manifest_out,
                })
            );
            if manifest.schema_mismatch != 0 || manifest.failed != 0 {
                return Err(DopeError::Data(format!(
                    "regression embedding shard has {} schema mismatches and {} failures",
                    manifest.schema_mismatch, manifest.failed
                )));
            }
        }
        Command::VerifyRegressionEmbeddings {
            output_dir,
            router_bundle,
            router_evidence,
            method,
            expected_dimension,
            expected_count,
        } => {
            let qualification = qualify_embedding_router(&router_bundle, &router_evidence)?;
            if qualification.dimension != expected_dimension {
                return Err(DopeError::Data(format!(
                    "qualified router dimension is {}, expected {expected_dimension}",
                    qualification.dimension
                )));
            }
            let verification =
                verify_regression_embeddings(&output_dir, &method, &qualification, expected_count)?;
            println!("{}", serde_json::to_string(&verification)?);
        }
        Command::Convert {
            dataset_dir,
            real_holdout_dir,
            task,
            out,
            seed,
            runtime_dictionary_bytes,
            supported_datasets,
            synthetic_seed,
        } => {
            let result = convert_release(
                &dataset_dir,
                &real_holdout_dir,
                parse_task(&task)?,
                &out,
                &ConversionOptions {
                    seed,
                    runtime_dictionary_bytes,
                    supported_datasets,
                    synthetic_csv_seeds: synthetic_seed,
                    ..Default::default()
                },
            )?;
            println!(
                "{{\"certified\":{},\"artifact_bytes\":{},\"out\":{}}}",
                result.report.certified,
                result.report.byte_accounting.artifact_bytes,
                serde_json::to_string(&out)?
            );
        }
        Command::Compile {
            dataset_dir,
            task,
            out,
            report,
            pareto_frontier,
            seed,
            deadline_seconds,
            language,
            candidate,
            neural_target_weight,
            neural_structural_penalty,
        } => {
            let options = CompileOptions {
                seed,
                deadline: deadline_seconds.map(Duration::from_secs_f64),
                language: language.as_deref().map(load_language).transpose()?,
                backend_id: candidate,
                neural_target_weight,
                neural_structural_penalty,
                ..Default::default()
            };
            let result = compile_kernel_from_dir(&dataset_dir, parse_task(&task)?, &options)?;
            fs::write(&out, &result.artifact)
                .map_err(|error| dope_kernel::error::io_error(&out, error))?;
            let report =
                report.unwrap_or_else(|| PathBuf::from(format!("{}.report.json", out.display())));
            let frontier = pareto_frontier
                .unwrap_or_else(|| PathBuf::from(format!("{}.pareto.json", out.display())));
            write_json(&report, &result.report)?;
            write_json(&frontier, &result.pareto_frontier)?;
            println!(
                "{{\"artifact_bytes\":{},\"compliant\":{},\"out\":{}}}",
                result.artifact.len(),
                result.report.compliant,
                serde_json::to_string(&out)?
            );
        }
        Command::Inspect { kernel, out } => {
            let inspection = inspect_kernel(&kernel)?;
            if let Some(path) = out {
                write_json(&path, &inspection)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&inspection)?);
            }
        }
        Command::Sample {
            kernel,
            rows,
            out,
            seed,
        } => {
            let loaded = load_kernel(&kernel)?;
            sample_kernel_to_csv(&loaded, SampleOptions { rows, seed }, &out)?;
            println!(
                "{{\"out\":{},\"rows\":{rows}}}",
                serde_json::to_string(&out)?
            );
        }
        Command::PackCorpus {
            corpus,
            out,
            seed,
            shard_mib,
            archive_zstd,
        } => {
            fs::create_dir_all(&out).map_err(|error| dope_kernel::error::io_error(&out, error))?;
            let (inventory, manifest) = build_split_manifest_multi(&corpus, seed)?;
            write_inventory(&inventory, &out.join("inventory.json"))?;
            write_exclusions(&inventory, &out.join("exclusions.jsonl"))?;
            write_split_manifest(&manifest, &out.join("split-manifest.json"))?;
            let packed = dope_kernel::packed::pack_corpus(
                &manifest,
                &out,
                (shard_mib * 1024.0 * 1024.0) as usize,
                archive_zstd,
            )?;
            println!("{}", serde_json::to_string(&packed)?);
        }
        Command::InventoryCorpus { corpus, out, seed } => {
            fs::create_dir_all(&out).map_err(|error| dope_kernel::error::io_error(&out, error))?;
            let (inventory, manifest) = build_split_manifest_multi(&corpus, seed)?;
            write_inventory(&inventory, &out.join("inventory.json"))?;
            write_exclusions(&inventory, &out.join("exclusions.jsonl"))?;
            write_split_manifest(&manifest, &out.join("split-manifest.json"))?;
            println!(
                "{{\"discovered_paths\":{},\"eligible_paths\":{},\"excluded_paths\":{},\"manifest_checksum\":{}}}",
                inventory.discovered_paths,
                inventory.eligible_paths,
                inventory.exclusions.len(),
                serde_json::to_string(&manifest.checksum)?,
            );
        }
        Command::FinalizeInventory {
            inventory,
            out,
            seed,
        } => {
            fs::create_dir_all(&out).map_err(|error| dope_kernel::error::io_error(&out, error))?;
            let inventory = refresh_inventory_lineages(&load_inventory(&inventory)?)?;
            let manifest = split_inventory(&inventory, seed)?;
            write_inventory(&inventory, &out.join("inventory.json"))?;
            write_exclusions(&inventory, &out.join("exclusions.jsonl"))?;
            write_split_manifest(&manifest, &out.join("split-manifest.json"))?;
            println!(
                "{{\"discovered_paths\":{},\"eligible_paths\":{},\"excluded_paths\":{},\"inventory_checksum\":{},\"manifest_checksum\":{}}}",
                inventory.discovered_paths,
                inventory.eligible_paths,
                inventory.exclusions.len(),
                serde_json::to_string(&inventory.checksum)?,
                serde_json::to_string(&manifest.checksum)?,
            );
        }
        Command::SplitValidation {
            manifest,
            out,
            seed,
        } => {
            let bytes = fs::read(&manifest)
                .map_err(|error| dope_kernel::error::io_error(&manifest, error))?;
            let source = serde_json::from_slice(&bytes)?;
            let submanifest = split_validation_manifest(&source, seed)?;
            write_json(&out, &submanifest)?;
            println!(
                "{{\"assignments\":{},\"validation_cert_sealed\":true,\"out\":{}}}",
                submanifest.assignments.len(),
                serde_json::to_string(&out)?
            );
        }
        Command::CalibrateLanguage { corpus, out, seed } => {
            if seed != 1729 {
                return Err(DopeError::Data(
                    "the release run predeclares seed 1729; stability seeds are fixed internally"
                        .into(),
                ));
            }
            let calibration = train_language(&corpus, &out, &TrainingOptions::default())?;
            println!(
                "{{\"dictionary_bytes\":{},\"release_eligible\":{},\"out\":{}}}",
                calibration.language_artifact.dictionary_bytes,
                calibration.language_artifact.release_eligible,
                serde_json::to_string(&out)?
            );
        }
        Command::Certify {
            real_dir,
            kernel,
            out,
            seed,
            runtime_dictionary_bytes,
            supported_datasets,
        } => {
            let report = certify_kernel(
                &real_dir,
                &kernel,
                &out,
                seed,
                runtime_dictionary_bytes,
                supported_datasets,
            )?;
            println!(
                "{{\"certified\":{},\"out\":{}}}",
                report.certified,
                serde_json::to_string(&out)?
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse().command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

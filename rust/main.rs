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
use dope_kernel::certification::certify_kernel_with_policy;
use dope_kernel::certification::{GoldCellOptions, evaluate_gold_cell};
use dope_kernel::codec::load_kernel;
use dope_kernel::compiler::{CompileOptions, compile_kernel_from_dir};
use dope_kernel::contract::{AnonymizationTier, ReleasePolicy};
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
    MetadataMode, TargetSelector, embed_dataset_with_metadata, qualify_embedding_router,
    write_embedding_csv,
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
use sha2::Digest;

include!("main/cli.rs");

fn parse_task(value: &str) -> Result<Task> {
    Task::parse(value).ok_or_else(|| DopeError::Data("task must be regression or binary".into()))
}

fn parse_policy(
    tier: &str,
    max_artifact_bytes: Option<usize>,
    require_formal_dp: bool,
) -> Result<ReleasePolicy> {
    ReleasePolicy::new(
        tier.parse::<AnonymizationTier>().map_err(DopeError::Data)?,
        max_artifact_bytes,
        require_formal_dp,
    )
}

fn write_json(path: &PathBuf, value: &impl serde::Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(path, bytes).map_err(|error| dope_kernel::error::io_error(path, error))
}

fn main() -> ExitCode {
    match run(Cli::parse().command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            let hint = error.repair_hint();
            eprintln!(
                "repair_receipt: {}",
                serde_json::to_string(&hint).expect("static repair hint")
            );
            ExitCode::FAILURE
        }
    }
}

include!("main/campaign_commands.rs");
include!("main/mixed_commands.rs");

include!("main/dispatch.rs");
include!("main/metadata_commands.rs");
include!("main/release_commands.rs");

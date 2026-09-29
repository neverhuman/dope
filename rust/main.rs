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
        Command::Campaign { command } => run_campaign(command)?,
        Command::EmbedDataset {
            csv,
            target_column,
            target_index,
            task,
            router_bundle,
            router_evidence,
            out,
            vector_out,
            restricted_metadata,
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
            let embedding = embed_dataset_with_metadata(
                &csv,
                selector,
                parse_task(&task)?,
                &router_bundle,
                &router_evidence,
                if restricted_metadata {
                    MetadataMode::RestrictedResearch
                } else {
                    MetadataMode::Public
                },
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
            tier,
            max_artifact_bytes,
            require_formal_dp,
        } => {
            let release_policy = parse_policy(&tier, max_artifact_bytes, require_formal_dp)?;
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
                    compile_options: CompileOptions {
                        beam_width: match release_policy.tier {
                            AnonymizationTier::L3 => Some(4),
                            AnonymizationTier::L2 => Some(8),
                            AnonymizationTier::L1 | AnonymizationTier::L0 => Some(16),
                        },
                        release_policy,
                        ..Default::default()
                    },
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
            tier,
            max_artifact_bytes,
            require_formal_dp,
        } => {
            let options = CompileOptions {
                release_policy: parse_policy(&tier, max_artifact_bytes, require_formal_dp)?,
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
            let report = match report {
                Some(path) => path,
                None => PathBuf::from(format!("{}.report.json", out.display())),
            };
            let frontier = match pareto_frontier {
                Some(path) => path,
                None => PathBuf::from(format!("{}.pareto.json", out.display())),
            };
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
            tier,
            max_artifact_bytes,
            require_formal_dp,
        } => {
            let policy = parse_policy(&tier, max_artifact_bytes, require_formal_dp)?;
            let report = certify_kernel_with_policy(
                &real_dir,
                &kernel,
                &out,
                seed,
                runtime_dictionary_bytes,
                supported_datasets,
                &policy,
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

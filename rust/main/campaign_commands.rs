fn run_campaign(command: CampaignCommand) -> Result<()> {
    match command {
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
    };
    Ok(())
}

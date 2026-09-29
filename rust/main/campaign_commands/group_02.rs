fn run_campaign_group_02(command: CampaignCommand) -> Result<()> {
    match command {
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
            },
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
            },
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
            },
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
            },
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
            },
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
            },
        CampaignCommand::ExportMetrics {
                state_dir,
                out,
                receipt_key,
            } => {
                let metrics = export_metrics(&state_dir, &out, &receipt_key)?;
                println!("{}", serde_json::to_string(&metrics)?);
            },
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
            },
        CampaignCommand::FreezeRouter {
                state_dir,
                bundle,
                evidence,
                allow_failed_frontier,
            } => {
                let state = freeze_router(&state_dir, &bundle, &evidence, allow_failed_frontier)?;
                println!("{}", serde_json::to_string(&state)?);
            },
        CampaignCommand::OpenValidationCert {
                state_dir,
                authorization,
            } => {
                let state = open_validation_cert(&state_dir, &authorization)?;
                println!("{}", serde_json::to_string(&state)?);
            },
        _ => return Err(DopeError::Unsupported("campaign command routing mismatch".into())),
    };
    Ok(())
}

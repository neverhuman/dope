|| -> Result<MetricsBundle> {


    let count_extra_json = |directory: &Path| -> Result<usize> {
        if !directory.is_dir() {
            return Ok(0);
        }
        let mut extra = 0usize;
        for entry in fs::read_dir(directory).map_err(|error| io_error(directory, error))? {
            let path = entry.map_err(|error| io_error(directory, error))?.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "json")
                && path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .is_none_or(|stem| !expected_identities.contains(stem))
            {
                extra += 1;
            }
        }
        Ok(extra)
    };
    let extra_blocks = count_extra_json(&cert_out.join("blocks"))?;
    let extra_receipts = count_extra_json(&cert_out.join("receipts"))?;
    let reconciliation = CertReceiptReconciliation {
        format: "dope-cert-receipt-reconciliation".into(),
        version: 1,
        expected_blocks: cohort.records.len(),
        reconciled_blocks: receipts_reconciled,
        extra_blocks,
        extra_receipts,
        issues,
    };
    write_canonical(&out.join("receipt-reconciliation.json"), &reconciliation)?;

    let missing_lineages = cohort
        .records
        .len()
        .saturating_sub(evaluated_lineages.len());
    let kpi_report = aggregate_kpis(
        &kpi_cells,
        cohort.records.len(),
        missing_lineages,
        "validation-cert",
    )?;
    write_canonical(&out.join("validation-kpis.json"), &kpi_report)?;
    let coverage_entries = coverage_groups
        .into_iter()
        .map(|((task, profile, auditor, size_multiplier), aggregate)| {
            let missing = aggregate
                .eligible
                .saturating_sub(aggregate.evaluated + aggregate.timed_out);
            CoverageEntry {
                task,
                structural_profile: profile,
                auditor,
                size_multiplier,
                eligible: aggregate.eligible,
                evaluated: aggregate.evaluated,
                missing,
                timed_out: aggregate.timed_out,
                required: true,
            }
        })
        .collect::<Vec<_>>();
    let required_complete = !coverage_entries.is_empty()
        && coverage_entries
            .iter()
            .all(|entry| entry.missing == 0 && entry.timed_out == 0);
    let coverage = CoverageReport {
        format: "dope-kpi-coverage".into(),
        version: 1,
        tier: "validation-cert".into(),
        entries: coverage_entries,
        required_complete,
    };
    coverage.validate()?;
    write_canonical(&out.join("kpi-coverage.json"), &coverage)?;

    let mut frontier = Vec::new();
    for candidate in &candidate_ids {
        let aggregate = candidate_aggregates
            .get(candidate)
            .expect("initialized candidate aggregate");
        let attempted = cohort.records.len() * cells_per_candidate;
        frontier.push(CandidateFrontierEntry {
            candidate_id: candidate.clone(),
            implementation_hash: empirical_backends()
                .into_iter()
                .find(|backend| backend.id == *candidate)
                .map(|backend| backend.implementation_hash)
                .expect("frozen candidate implementation"),
            attempted,
            succeeded: aggregate.succeeded,
            failed: attempted.saturating_sub(aggregate.succeeded + aggregate.timed_out),
            timed_out: aggregate.timed_out,
            mean_retention: (aggregate.retention_count > 0)
                .then_some(aggregate.retention_sum / aggregate.retention_count.max(1) as f64),
            runtime_p95_ms: histogram_percentile(&aggregate.runtimes, aggregate.succeeded, 0.95),
            peak_memory_p95_bytes: histogram_percentile(
                &aggregate.memories,
                aggregate.succeeded,
                0.95,
            ),
            artifact_bytes_p95: histogram_percentile(
                &aggregate.artifact_bytes,
                aggregate.succeeded,
                0.95,
            ),
        });
    }
    write_canonical(&out.join("candidate-frontier.json"), &frontier)?;

    let regret = mean_and_upper_95(&regrets).map(|(mean, _)| mean);
    let random_regret = mean_and_upper_95(&random_regrets).map(|(mean, _)| mean);
    let fixed_regret = mean_and_upper_95(&fixed_regrets).map(|(mean, _)| mean);
    let regret_by_profile = profile_regrets
        .iter()
        .filter_map(|(profile, values)| {
            mean_and_upper_95(values).map(|(mean_regret, upper_95)| {
                (
                    profile.clone(),
                    ProfileRegretDetail {
                        outcomes: values.len(),
                        mean_regret,
                        upper_95,
                    },
                )
            })
        })
        .collect::<BTreeMap<_, _>>();
    let maximum_profile_regret_upper = regret_by_profile
        .values()
        .map(|detail| detail.upper_95)
        .reduce(f64::max);
    let top_four_oracle_recall = (complete_oracle_outcomes > 0)
        .then_some(top_four_hits as f64 / complete_oracle_outcomes as f64);
    let reference_hypervolume = if hypervolume_pairs.is_empty() {
        None
    } else {
        Some(
            hypervolume_pairs
                .iter()
                .map(|(_, fixed)| fixed.abs())
                .sum::<f64>()
                / hypervolume_pairs.len() as f64,
        )
    };
    let hypervolume_improvements = reference_hypervolume
        .filter(|reference| *reference > 0.0)
        .map(|reference| {
            hypervolume_pairs
                .iter()
                .map(|(routed, fixed)| (routed - fixed) / reference.max(1e-9))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let (paired_hypervolume_improvement, paired_hypervolume_ci_lower) =
        mean_and_lower_95(&hypervolume_improvements)
            .map_or((None, None), |(mean, lower)| (Some(mean), Some(lower)));
    let router_inference_p95_ms = float_percentile(&mut router_inference_times, 0.95);
    let sketch_p95_ms = float_percentile(&mut router_sketch_times, 0.95);
    let work_avoided_fraction = (valid_expected_cells > 0)
        .then_some(1.0 - selected_expected_cells as f64 / valid_expected_cells as f64);
    let latency_reduction_fraction =
        (exhaustive_runtime_ms > 0.0).then_some(1.0 - selected_runtime_ms / exhaustive_runtime_ms);
    let actual_beats_random = regret
        .zip(random_regret)
        .is_some_and(|(routed, random)| routed < random);
    let actual_beats_fixed = regret
        .zip(fixed_regret)
        .is_some_and(|(routed, fixed)| routed < fixed);
    let all_auditors_exercised = auditor_availability.values().all(|count| *count > 0);
    let certification = CertificationDetail {
        eligible_lineages: cohort.records.len(),
        evaluated_lineages: evaluated_lineages.len(),
        missing_lineages,
        timeout_lineages: timeout_lineages.len(),
        exhaustive_cells: expected_cells,
        routed_cells,
        shadow_cells,
        failed_cells: expected_cells.saturating_sub(succeeded_cells),
        top_four_oracle_recall,
        mean_candidate_regret: regret,
        random_candidate_regret: random_regret,
        best_fixed_candidate_id: router_evidence.best_fixed_candidate_id.clone(),
        best_fixed_candidate_regret: fixed_regret,
        beats_random_candidate: actual_beats_random,
        beats_best_fixed_candidate: actual_beats_fixed,
        maximum_profile_regret_upper,
        regret_by_profile,
        work_avoided_fraction,
        latency_reduction_fraction,
        router_bundle_bytes: fs::metadata(&frozen_router_path)
            .map_err(|error| io_error(&frozen_router_path, error))?
            .len(),
        router_inference_p95_ms,
        sketch_p95_ms,
        paired_hypervolume_improvement,
        paired_hypervolume_ci_lower,
        candidate_availability,
        auditor_availability,
        candidate_implementation_hashes: empirical_backends()
            .into_iter()
            .map(|candidate| (candidate.id, candidate.implementation_hash))
            .collect(),
        auditor_implementation_hashes: auditor_specs()
            .into_iter()
            .map(|auditor| (auditor.id, auditor.frozen_hash))
            .collect(),
        receipts_expected: cohort.records.len(),
        receipts_reconciled,
    };
    write_canonical(&out.join("certification-detail.json"), &certification)?;

    let signatures_valid = receipts_reconciled == cohort.records.len()
        && reconciliation.issues.is_empty()
        && extra_blocks == 0
        && extra_receipts == 0;
    let all_candidates_exercised_both_tasks = candidate_aggregates.values().all(|aggregate| {
        aggregate.tasks.contains("binary") && aggregate.tasks.contains("regression")
    });
    let gate_evidence = GateEvidence {
        ptf_v1: kpi_report.ptf_v1,
        calibration_degradation_max,
        rare_class_tail_subgroup_retention_min: rare_tail_subgroup_retention_min,
        driver_agreement_min,
        joint_fidelity_min,
        query_p95_normalized_error_max: query_error_max,
        nominal_95_coverage_min: nominal_coverage_min,
        nominal_95_coverage_max: nominal_coverage_max,
        type_i_error_max,
        membership_auc_max,
        feature_importance_complete: feature_importance_complete && valid_expected_cells > 0,
        feature_importance_applicable,
        feature_importance_spearman_min: if feature_importance_missing_rank {
            None
        } else {
            feature_importance_spearman_min
        },
        feature_importance_top_k_jaccard_min,
        attribute_inference_advantage_max: attribute_advantage_max,
        exact_copies,
        near_copies,
        canary_extractions,
        lineage_leaks,
        validation_training_regret_upper: Some(router_evidence.maximum_profile_regret_upper),
        across_seed_validation_loss_stddev,
        uncertainty_coverage_min: nominal_coverage_min,
        uncertainty_coverage_max: nominal_coverage_max,
        beats_random_router: router_evidence.beats_random,
        beats_best_fixed_candidate: router_evidence.beats_best_fixed,
        learned_representation_used: true,
        pareto_hypervolume_improvement: paired_hypervolume_improvement,
        pareto_hypervolume_ci_lower: paired_hypervolume_ci_lower,
        all_candidates_exercised_both_tasks,
        all_auditors_pinned_and_exercised: all_auditors_exercised,
        byte_reconciliation_exact: signatures_valid && all_artifact_bytes_nonzero,
        signatures_and_receipts_valid: signatures_valid,
        controller_free_gib: controller.available_bytes / GIB,
        evaluator_free_gib: evaluator.available_bytes / GIB,
    };
    write_canonical(&out.join("gate-evidence.json"), &gate_evidence)?;
    let mut failed_gates = gate_evidence
        .failed_gates(&crate::production::KpiContract::embedded()?, &coverage)
        .into_iter()
        .collect::<BTreeSet<_>>();
    failed_gates.extend(
        router_evidence
            .failed_gates()
            .into_iter()
            .map(|gate| format!("router_{gate}")),
    );
    if !top_four_oracle_recall.is_some_and(|value| value >= 0.99) {
        failed_gates.insert("validation_cert_top_four_oracle_recall".into());
    }
    if !maximum_profile_regret_upper.is_some_and(|value| value <= 0.02) {
        failed_gates.insert("validation_cert_profile_regret".into());
    }
    if !actual_beats_random {
        failed_gates.insert("validation_cert_beats_random".into());
    }
    if !actual_beats_fixed {
        failed_gates.insert("validation_cert_beats_best_fixed".into());
    }
    if !router_inference_p95_ms.is_some_and(|value| value <= 5.0) {
        failed_gates.insert("validation_cert_router_inference_p95".into());
    }
    if !sketch_p95_ms.is_some_and(|value| value <= 1_000.0) {
        failed_gates.insert("validation_cert_sketch_p95".into());
    }
    let failed_gates = failed_gates.into_iter().collect::<Vec<_>>();
    let kpi = KpiSummary::from_evidence(
        Some(&kpi_report),
        Some(&coverage),
        maximum_profile_regret_upper,
    );
    let mut counts = BTreeMap::new();
    let non_timeout_failures = explicit_failed_cells.saturating_sub(timed_out_cells);
    let pending = expected_cells.saturating_sub(succeeded_cells + explicit_failed_cells);
    counts.insert(JobState::Pending, pending);
    counts.insert(JobState::Leased, 0);
    counts.insert(JobState::Running, 0);
    counts.insert(JobState::Succeeded, succeeded_cells);
    counts.insert(JobState::Failed, non_timeout_failures);
    counts.insert(JobState::TimedOut, timed_out_cells);
    let terminal = succeeded_cells + explicit_failed_cells;
    let status = crate::ledger::LedgerStatus {
        format: "dope-campaign-ledger-status".into(),
        version: 1,
        counts,
        total: expected_cells,
        training_progress: (expected_cells > 0).then_some(terminal as f64 / expected_cells as f64),
        completed_gold_labels: succeeded_cells,
        failed_jobs: non_timeout_failures,
        timed_out_jobs: timed_out_cells,
        throughput_jobs_per_hour: None,
        coverage: kpi.coverage,
        current_router_validation_regret: maximum_profile_regret_upper,
        ptf_v1: kpi_report.ptf_v1,
        production_score_available: kpi.production_score_available && failed_gates.is_empty(),
        updated_unix_seconds: unix_now(),
    };
    let bundle = MetricsBundle {
        format: "dope-campaign-metrics".into(),
        version: 1,
        status,
        kpi,
        candidate_frontier: frontier,
        router: Some(router_evidence),
        failed_gates,
        validation_cert_open_count: state.validation_cert_open_count,
        sealed_test_open_count: state.sealed_test_open_count,
        receipts_reconciled,
        certification: Some(certification),
    };
    write_canonical(&out.join("metrics-summary.json"), &bundle)?;
    Ok(bundle)

}

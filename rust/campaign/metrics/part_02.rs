

pub fn export_metrics(state_dir: &Path, out: &Path, receipt_key: &Path) -> Result<MetricsBundle> {
    let state = load_state(state_dir)?;
    fs::create_dir_all(out).map_err(|error| io_error(out, error))?;
    let key = load_receipt_key(receipt_key)?;
    if hashes(&key).sha256 != state.receipt_key_sha256 {
        return Err(DopeError::Data(
            "metrics receipt key differs from freeze".into(),
        ));
    }
    let ledger = CampaignLedger::open(&ledger_path(state_dir))?;
    let specs = ledger.all_specs()?;
    let receipts = ledger.all_receipts()?;
    let specs_by_id = specs
        .iter()
        .map(|(id, state, spec)| (id.as_str(), (*state, spec)))
        .collect::<BTreeMap<_, _>>();
    let signatures_valid = receipts.iter().all(|receipt| {
        receipt.verify_signature(&key).is_ok()
            && specs_by_id
                .get(receipt.job_id.as_str())
                .is_some_and(|(_, spec)| receipt.validate_against(spec).is_ok())
    });
    let mut last_receipts = BTreeMap::<&str, &JobReceipt>::new();
    for receipt in &receipts {
        let replace = last_receipts
            .get(receipt.job_id.as_str())
            .is_none_or(|existing| receipt.attempt > existing.attempt);
        if replace {
            last_receipts.insert(receipt.job_id.as_str(), receipt);
        }
    }
    let receipts_reconciled = specs
        .iter()
        .filter(|(id, state, _)| {
            matches!(
                state,
                JobState::Succeeded | JobState::Failed | JobState::TimedOut
            ) && last_receipts
                .get(id.as_str())
                .is_some_and(|receipt| receipt.state == *state)
        })
        .count();
    let mut metrics_file = File::create(out.join("metrics.jsonl"))
        .map_err(|error| io_error(out.join("metrics.jsonl"), error))?;
    for receipt in &receipts {
        metrics_file
            .write_all(&canonical_json(receipt)?)
            .map_err(|error| io_error(out, error))?;
        metrics_file
            .write_all(b"\n")
            .map_err(|error| io_error(out, error))?;
    }
    ledger.export_receipts(&out.join("receipts.json"))?;

    let successful = receipts
        .iter()
        .filter(|receipt| receipt.state == JobState::Succeeded)
        .filter_map(|receipt| receipt.evidence.as_ref())
        .collect::<Vec<_>>();
    let cert_successful = successful
        .iter()
        .filter(|evidence| evidence.phase == "validation_cert" && evidence.routed)
        .copied()
        .collect::<Vec<_>>();
    let cert_cells = cert_successful
        .iter()
        .map(|evidence| KpiCell {
            task: if evidence.task == "binary" {
                Task::Binary
            } else {
                Task::Regression
            },
            train_rows: evidence.train_rows,
            features: evidence.features,
            auditor: evidence.auditor_id.clone(),
            size_multiplier: evidence.size_multiplier,
            lineage_group_id: evidence.lineage_group_id.clone(),
            generation_seed: evidence.generation_seed,
            auditor_seed: evidence.auditor_seed,
            null_loss: evidence.null_loss,
            trtr_loss: evidence.trtr_loss,
            tstr_loss: evidence.tstr_loss,
            calibration_degradation: evidence.calibration_degradation,
            rare_class_or_tail_retention: evidence.rare_class_or_tail_retention,
            supported_subgroup_retention: evidence.supported_subgroup_retention,
            nominal_95_coverage: evidence.nominal_95_coverage,
        })
        .collect::<Vec<_>>();
    let eligible_lineages = specs
        .iter()
        .filter(|(_, _, spec)| spec.phase == "validation_cert" && spec.routed)
        .map(|(_, _, spec)| &spec.dataset_lineage)
        .collect::<BTreeSet<_>>()
        .len();
    let evaluated_lineages = cert_cells
        .iter()
        .map(|cell| &cell.lineage_group_id)
        .collect::<BTreeSet<_>>()
        .len();
    let kpi_report = aggregate_kpis(
        &cert_cells,
        eligible_lineages,
        eligible_lineages.saturating_sub(evaluated_lineages),
        "validation-cert",
    )?;
    write_canonical(&out.join("validation-kpis.json"), &kpi_report)?;

    let mut coverage_groups =
        BTreeMap::<(String, String, String, usize), (usize, usize, usize, usize)>::new();
    for (_, job_state, spec) in specs
        .iter()
        .filter(|(_, _, spec)| spec.phase == "validation_cert" && spec.routed)
    {
        let task = spec
            .structural_profile
            .split('/')
            .next()
            .unwrap_or("unknown")
            .to_owned();
        let entry = coverage_groups
            .entry((
                task,
                spec.structural_profile.clone(),
                spec.auditor_spec.clone(),
                spec.size_multiplier,
            ))
            .or_default();
        entry.0 += 1;
        match job_state {
            JobState::Succeeded => entry.1 += 1,
            JobState::TimedOut => entry.3 += 1,
            _ => entry.2 += 1,
        }
    }
    let entries = coverage_groups
        .into_iter()
        .map(
            |((task, profile, auditor, size), (eligible, evaluated, missing, timed_out))| {
                CoverageEntry {
                    task,
                    structural_profile: profile,
                    auditor,
                    size_multiplier: size,
                    eligible,
                    evaluated,
                    missing,
                    timed_out,
                    required: matches!(size, 1 | 4),
                }
            },
        )
        .collect::<Vec<_>>();
    let required_complete = !entries.is_empty()
        && entries
            .iter()
            .all(|entry| !entry.required || (entry.missing == 0 && entry.timed_out == 0));
    let coverage = CoverageReport {
        format: "dope-kpi-coverage".into(),
        version: 1,
        tier: "validation-cert".into(),
        entries,
        required_complete,
    };
    coverage.validate()?;
    write_canonical(&out.join("kpi-coverage.json"), &coverage)?;

    let router: Option<RouterEvidence> = ledger.get_metadata("router_evidence")?;
    let candidate_regret = router
        .as_ref()
        .map(|value| value.maximum_profile_regret_upper);
    let kpi = KpiSummary::from_evidence(Some(&kpi_report), Some(&coverage), candidate_regret);
    let mut frontier = Vec::new();
    for backend in empirical_backends() {
        let matching_specs = specs
            .iter()
            .filter(|(_, _, spec)| spec.candidate_spec == backend.id)
            .collect::<Vec<_>>();
        let matching = successful
            .iter()
            .filter(|evidence| evidence.candidate_id == backend.id)
            .copied()
            .collect::<Vec<_>>();
        let mut runtimes = matching
            .iter()
            .map(|evidence| evidence.runtime_ms)
            .collect::<Vec<_>>();
        let mut memories = matching
            .iter()
            .map(|evidence| evidence.peak_memory_bytes)
            .collect::<Vec<_>>();
        let mut bytes = matching
            .iter()
            .map(|evidence| evidence.artifact_bytes)
            .collect::<Vec<_>>();
        let retentions = matching
            .iter()
            .filter_map(|evidence| {
                let denominator = evidence.null_loss - evidence.trtr_loss;
                (denominator.abs() > f64::EPSILON)
                    .then_some((evidence.null_loss - evidence.tstr_loss) / denominator)
            })
            .collect::<Vec<_>>();
        frontier.push(CandidateFrontierEntry {
            candidate_id: backend.id,
            implementation_hash: backend.implementation_hash,
            attempted: matching_specs.len(),
            succeeded: matching.len(),
            failed: matching_specs
                .iter()
                .filter(|(_, state, _)| *state == JobState::Failed)
                .count(),
            timed_out: matching_specs
                .iter()
                .filter(|(_, state, _)| *state == JobState::TimedOut)
                .count(),
            mean_retention: (!retentions.is_empty())
                .then(|| retentions.iter().sum::<f64>() / retentions.len() as f64),
            runtime_p95_ms: percentile(&mut runtimes, 0.95),
            peak_memory_p95_bytes: percentile(&mut memories, 0.95),
            artifact_bytes_p95: percentile(&mut bytes, 0.95),
        });
    }
    write_canonical(&out.join("candidate-frontier.json"), &frontier)?;

    let candidate_tasks = successful.iter().fold(
        BTreeMap::<String, BTreeSet<String>>::new(),
        |mut map, evidence| {
            map.entry(evidence.candidate_id.clone())
                .or_default()
                .insert(evidence.task.clone());
            map
        },
    );
    let exercised_auditors = successful
        .iter()
        .map(|evidence| &evidence.auditor_id)
        .collect::<BTreeSet<_>>();
    let mut seed_groups = BTreeMap::<(String, String, String, usize), Vec<f64>>::new();
    for evidence in &cert_successful {
        seed_groups
            .entry((
                evidence.lineage_group_id.clone(),
                evidence.candidate_id.clone(),
                evidence.auditor_id.clone(),
                evidence.size_multiplier,
            ))
            .or_default()
            .push(evidence.tstr_loss);
    }
    let standard_deviation = |values: &[f64]| {
        if values.len() < 2 {
            return 0.0;
        }
        let average = values.iter().sum::<f64>() / values.len() as f64;
        (values
            .iter()
            .map(|value| (value - average).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64)
            .sqrt()
    };
    let across_seed_validation_loss_stddev = seed_groups
        .values()
        .map(|values| standard_deviation(values))
        .reduce(f64::max);
    let byte_reconciliation_exact = !specs.is_empty()
        && receipts_reconciled == specs.len()
        && signatures_valid
        && successful.iter().all(|evidence| {
            evidence.artifact_bytes > 0
                && is_lower_hex(&evidence.artifact_sha256, &[64])
                && is_lower_hex(&evidence.artifact_blake3, &[64])
        });
    let (
        feature_importance_complete,
        feature_importance_applicable,
        feature_importance_spearman_min,
        feature_importance_top_k_jaccard_min,
    ) = aggregate_importance(cert_successful.iter().copied());
    let evidence = GateEvidence {
        ptf_v1: kpi_report.ptf_v1,
        calibration_degradation_max: option_max(
            cert_successful.iter().map(|e| e.calibration_degradation),
        ),
        rare_class_tail_subgroup_retention_min: option_min(cert_successful.iter().flat_map(|e| {
            [
                e.rare_class_or_tail_retention,
                e.supported_subgroup_retention,
            ]
        })),
        driver_agreement_min: option_min(cert_successful.iter().map(|e| e.driver_agreement)),
        joint_fidelity_min: option_min(cert_successful.iter().map(|e| e.joint_fidelity)),
        query_p95_normalized_error_max: option_max(
            cert_successful.iter().map(|e| e.query_p95_normalized_error),
        ),
        nominal_95_coverage_min: option_min(cert_successful.iter().map(|e| e.nominal_95_coverage)),
        nominal_95_coverage_max: option_max(cert_successful.iter().map(|e| e.nominal_95_coverage)),
        type_i_error_max: option_max(cert_successful.iter().map(|e| e.type_i_error)),
        membership_auc_max: option_max(cert_successful.iter().map(|e| e.membership_auc)),
        feature_importance_complete,
        feature_importance_applicable,
        feature_importance_spearman_min,
        feature_importance_top_k_jaccard_min,
        attribute_inference_advantage_max: option_max(
            cert_successful
                .iter()
                .map(|e| e.attribute_inference_advantage),
        ),
        exact_copies: cert_successful.iter().map(|e| e.exact_copies).sum(),
        near_copies: cert_successful.iter().map(|e| e.near_copies).sum(),
        canary_extractions: cert_successful.iter().map(|e| e.canary_extractions).sum(),
        lineage_leaks: cert_successful.iter().map(|e| e.lineage_leaks).sum(),
        validation_training_regret_upper: candidate_regret,
        across_seed_validation_loss_stddev,
        uncertainty_coverage_min: option_min(cert_successful.iter().map(|e| e.nominal_95_coverage)),
        uncertainty_coverage_max: option_max(cert_successful.iter().map(|e| e.nominal_95_coverage)),
        beats_random_router: router.as_ref().is_some_and(|value| value.beats_random),
        beats_best_fixed_candidate: router.as_ref().is_some_and(|value| value.beats_best_fixed),
        learned_representation_used: router.is_some(),
        pareto_hypervolume_improvement: router
            .as_ref()
            .map(|value| value.paired_hypervolume_improvement),
        pareto_hypervolume_ci_lower: router
            .as_ref()
            .map(|value| value.paired_hypervolume_ci_lower),
        all_candidates_exercised_both_tasks: empirical_backends().iter().all(|backend| {
            candidate_tasks
                .get(&backend.id)
                .is_some_and(|tasks| tasks.len() == 2)
        }),
        all_auditors_pinned_and_exercised: auditor_specs()
            .iter()
            .all(|auditor| exercised_auditors.contains(&auditor.id)),
        byte_reconciliation_exact,
        signatures_and_receipts_valid: signatures_valid,
        controller_free_gib: available_bytes(state_dir)? / GIB,
        evaluator_free_gib: available_bytes(state_dir)? / GIB,
    };
    write_canonical(&out.join("gate-evidence.json"), &evidence)?;
    let failed_gates =
        evidence.failed_gates(&crate::production::KpiContract::embedded()?, &coverage);
    let status = ledger.status()?;
    ledger.put_metadata(
        "status_evidence",
        &serde_json::json!({
            "router_regret": candidate_regret,
            "ptf_v1": kpi_report.ptf_v1,
            "coverage": kpi.coverage,
            "production_score_available": kpi.production_score_available
        }),
    )?;
    let bundle = MetricsBundle {
        format: "dope-campaign-metrics".into(),
        version: 1,
        status,
        kpi,
        candidate_frontier: frontier,
        router,
        failed_gates,
        validation_cert_open_count: state.validation_cert_open_count,
        sealed_test_open_count: state.sealed_test_open_count,
        receipts_reconciled,
        certification: None,
    };
    write_canonical(&out.join("metrics-summary.json"), &bundle)?;
    Ok(bundle)
}

#[derive(Default)]
struct CertCandidateAggregate {
    succeeded: usize,
    timed_out: usize,
    retention_sum: f64,
    retention_count: usize,
    runtimes: BTreeMap<u64, usize>,
    memories: BTreeMap<u64, usize>,
    artifact_bytes: BTreeMap<u64, usize>,
    tasks: BTreeSet<String>,
}

#[derive(Default)]
struct CertCoverageAggregate {
    eligible: usize,
    evaluated: usize,
    timed_out: usize,
}

#[derive(Clone, Copy)]
struct CertCandidateOutcome {
    utility: f64,
    runtime_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct CertReceiptReconciliation {
    format: String,
    version: u8,
    expected_blocks: usize,
    reconciled_blocks: usize,
    extra_blocks: usize,
    extra_receipts: usize,
    issues: Vec<String>,
}

fn histogram_percentile(
    histogram: &BTreeMap<u64, usize>,
    cells: usize,
    quantile: f64,
) -> Option<u64> {
    if cells == 0 {
        return None;
    }
    let index = ((cells - 1) as f64 * quantile).ceil() as usize;
    let mut cumulative = 0usize;
    for (&value, &count) in histogram {
        cumulative += count;
        if cumulative > index {
            return Some(value);
        }
    }
    None
}

/// Reconstructs the final validation-cert metrics directly from content-addressed
/// record blocks. Only blocks with a valid xbabe3 keyed receipt and exact frozen
/// source, binary, router, cohort, and block hashes are admitted as evidence.
pub fn export_validation_cert_metrics(
    state_dir: &Path,
    cert_out: &Path,
    cohort_path: &Path,
    out: &Path,
    receipt_key: &Path,
) -> Result<MetricsBundle> {
    let state = load_state(state_dir)?;
    if state.phase != CampaignPhase::ValidationCertOpen
        || state.validation_cert_open_count != 1
        || state.sealed_test_open_count != 0
    {
        return Err(DopeError::Data(
            "validation-cert metrics require exactly one authorization event".into(),
        ));
    }
    require_durable_space(state_dir)?;
    fs::create_dir_all(out).map_err(|error| io_error(out, error))?;
    let key = load_receipt_key(receipt_key)?;
    if hashes(&key).sha256 != state.receipt_key_sha256 {
        return Err(DopeError::Data(
            "validation-cert receipt key differs from freeze".into(),
        ));
    }
    if file_hashes(&state_dir.join("source.tar"))? != state.source_tree {
        return Err(DopeError::Data(
            "frozen source archive no longer matches campaign state".into(),
        ));
    }
    let frozen_router_path = state_dir.join("router.bundle");
    let frozen_router_evidence_path = state_dir.join("router-evidence.json");
    let router_hashes = file_hashes(&frozen_router_path)?;
    let router_evidence_hashes = file_hashes(&frozen_router_evidence_path)?;
    if state.router_bundle.as_ref() != Some(&router_hashes)
        || state.router_evidence.as_ref() != Some(&router_evidence_hashes)
    {
        return Err(DopeError::Data(
            "router artifacts no longer match the frozen campaign".into(),
        ));
    }
    let router_bundle: RouterBundle = read_json(&frozen_router_path)?;
    router_bundle.validate()?;
    let router_evidence: RouterEvidence = read_json(&frozen_router_evidence_path)?;
    if router_evidence.format != "dope-router-evidence" || router_evidence.version != 1 {
        return Err(DopeError::Data("invalid frozen router evidence".into()));
    }
    let host_inventory_path = state_dir.join("host-inventory.json");
    if file_hashes(&host_inventory_path)? != state.host_inventory {
        return Err(DopeError::Data(
            "host inventory no longer matches campaign state".into(),
        ));
    }
    let host_inventory: HostInventory = read_json(&host_inventory_path)?;
    host_inventory.validate(&state.source_tree.sha256, &state.corpus_manifest.sha256)?;
    let evaluator = host_inventory
        .workers
        .iter()
        .find(|worker| worker.host == "xbabe3")
        .ok_or_else(|| DopeError::Data("xbabe3 is absent from host inventory".into()))?;
    let controller = host_inventory
        .workers
        .iter()
        .find(|worker| worker.host == "xbabe1")
        .ok_or_else(|| DopeError::Data("xbabe1 is absent from host inventory".into()))?;

    let cohort: CohortPlan = read_json(cohort_path)?;
    let cohort_hashes = file_hashes(cohort_path)?;
    if cohort.format != "dope-campaign-cohort"
        || cohort.version != 1
        || cohort.kind != "validation-cert"
        || cohort.records.len() != state.validation_cert_lineage_groups
        || cohort
            .records
            .iter()
            .map(|record| record.lineage_group_id.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != state.validation_cert_lineage_groups
        || cohort
            .records
            .iter()
            .any(|record| record.partition != "validation_cert")
    {
        return Err(DopeError::Data(
            "validation-cert cohort does not contain the exact guarded lineage population".into(),
        ));
    }
    let validation_path = state_dir.join("validation-manifest.json");
    if file_hashes(&validation_path)? != state.validation_manifest {
        return Err(DopeError::Data(
            "validation manifest no longer matches campaign state".into(),
        ));
    }
    let validation: ValidationSubmanifest = read_json(&validation_path)?;
    validate_validation_submanifest(&validation)?;
    let cert_assignments = validation
        .assignments
        .iter()
        .filter(|assignment| assignment.partition == "validation-cert")
        .map(|assignment| {
            (
                assignment.dataset_id.as_str(),
                assignment.lineage_group_id.as_str(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if cohort.records.iter().any(|record| {
        cert_assignments.get(record.dataset_id.as_str()).copied()
            != Some(record.lineage_group_id.as_str())
    }) {
        return Err(DopeError::Data(
            "validation-cert cohort differs from the frozen assignment".into(),
        ));
    }

    let candidate_ids = empirical_backends()
        .into_iter()
        .map(|candidate| candidate.id)
        .collect::<Vec<_>>();
    let auditor_ids = auditor_specs()
        .into_iter()
        .map(|auditor| auditor.id)
        .collect::<Vec<_>>();
    let size_multipliers = [1usize, 4];
    let generation_seeds = [1829u64, 99238, 196647];
    let auditor_seeds = [57721u64, 161803, 271828];
    let cells_per_candidate =
        auditor_ids.len() * size_multipliers.len() * generation_seeds.len() * auditor_seeds.len();
    let cells_per_record = candidate_ids.len() * cells_per_candidate;
    let expected_cells = cohort.records.len() * cells_per_record;
    let mut candidate_aggregates = candidate_ids
        .iter()
        .map(|candidate| (candidate.clone(), CertCandidateAggregate::default()))
        .collect::<BTreeMap<_, _>>();
    let mut candidate_availability = candidate_ids
        .iter()
        .map(|candidate| (candidate.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut auditor_availability = auditor_ids
        .iter()
        .map(|auditor| (auditor.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut coverage_groups =
        BTreeMap::<(String, String, String, usize), CertCoverageAggregate>::new();
    for record in &cohort.records {
        for auditor in &auditor_ids {
            for size in size_multipliers {
                coverage_groups
                    .entry((
                        record.task.clone(),
                        record.structural_profile.clone(),
                        auditor.clone(),
                        size,
                    ))
                    .or_default()
                    .eligible += generation_seeds.len() * auditor_seeds.len();
            }
        }
    }

    let metrics_partial = out.join("metrics.jsonl.partial");
    let receipts_partial = out.join("receipts.jsonl.partial");
    let mut metrics_writer = BufWriter::new(
        File::create(&metrics_partial).map_err(|error| io_error(&metrics_partial, error))?,
    );
    let mut receipts_writer = BufWriter::new(
        File::create(&receipts_partial).map_err(|error| io_error(&receipts_partial, error))?,
    );
    let mut issues = Vec::new();
    let mut expected_identities = BTreeSet::new();
    let mut receipts_reconciled = 0usize;
    let mut succeeded_cells = 0usize;
    let mut explicit_failed_cells = 0usize;
    let mut timed_out_cells = 0usize;
    let mut routed_cells = 0usize;
    let mut shadow_cells = 0usize;
    let mut selected_expected_cells = 0usize;
    let mut valid_expected_cells = 0usize;
    let mut selected_runtime_ms = 0f64;
    let mut exhaustive_runtime_ms = 0f64;
    let mut all_artifact_bytes_nonzero = true;
    let mut evaluated_lineages = BTreeSet::new();
    let mut timeout_lineages = BTreeSet::new();
    let mut kpi_cells = Vec::with_capacity(cohort.records.len() * cells_per_candidate);
    let mut router_sketch_times = Vec::with_capacity(cohort.records.len());
    let mut router_inference_times = Vec::with_capacity(cohort.records.len());
    let mut regrets = Vec::new();
    let mut random_regrets = Vec::new();
    let mut fixed_regrets = Vec::new();
    let mut profile_regrets = BTreeMap::<String, Vec<f64>>::new();
    let mut hypervolume_pairs = Vec::new();
    let mut top_four_hits = 0usize;
    let mut complete_oracle_outcomes = 0usize;
    let mut calibration_degradation_max = None;
    let mut rare_tail_subgroup_retention_min = None;
    let mut driver_agreement_min = None;
    let mut joint_fidelity_min = None;
    let mut query_error_max = None;
    let mut nominal_coverage_min = None;
    let mut nominal_coverage_max = None;
    let mut type_i_error_max = None;
    let mut membership_auc_max = None;
    let mut attribute_advantage_max = None;
    let mut across_seed_validation_loss_stddev = None;
    let mut feature_importance_complete = true;
    let mut feature_importance_applicable = false;
    let mut feature_importance_spearman_min = None::<f64>;
    let mut feature_importance_top_k_jaccard_min = None::<f64>;
    let mut feature_importance_missing_rank = false;
    let mut exact_copies = 0usize;
    let mut near_copies = 0usize;
    let mut canary_extractions = 0usize;
    let mut lineage_leaks = 0usize;

    for record in &cohort.records {
        let matrix_identity = gold_record_matrix_identity(&GoldMatrixIdentity {
            phase: "validation_cert",
            dataset_id: &record.dataset_id,
            candidate_ids: &candidate_ids,
            auditor_ids: &auditor_ids,
            size_multipliers: &size_multipliers,
            generation_seeds: &generation_seeds,
            auditor_seeds: &auditor_seeds,
            primary_only: false,
            neural_target_weight: 2.0,
            neural_structural_penalty: 0.0,
            routing_sha256: Some(&router_hashes.sha256),
        })?;
        expected_identities.insert(matrix_identity.clone());
        let block_path = cert_out
            .join("blocks")
            .join(format!("{matrix_identity}.json"));
        let receipt_path = cert_out
            .join("receipts")
            .join(format!("{matrix_identity}.json"));
        let admitted = (|| -> Result<(GoldRecordBlock, GoldBlockReceipt)> {
            let block: GoldRecordBlock = read_json(&block_path)?;
            validate_gold_record_block(
                &block,
                record,
                "validation_cert",
                &matrix_identity,
                &candidate_ids,
                &auditor_ids,
                Some(&router_hashes.sha256),
            )?;
            let receipt: GoldBlockReceipt = read_json(&receipt_path)?;
            receipt.verify(&key)?;
            let block_hashes = file_hashes(&block_path)?;
            if receipt.dataset_id != record.dataset_id
                || receipt.matrix_identity != matrix_identity
                || receipt.phase != "validation_cert"
                || receipt.worker != "xbabe3"
                || receipt.source_sha256 != state.source_tree.sha256
                || receipt.source_sha256 != evaluator.source_sha256
                || receipt.evaluator_binary_sha256 != evaluator.binary_sha256
                || receipt.router_bundle_sha256.as_deref() != Some(&router_hashes.sha256)
                || receipt.cohort != cohort_hashes
                || receipt.block != block_hashes
                || receipt.evidence_cells != block.evidence.len()
                || receipt.failed_cells != block.failures.len()
            {
                return Err(DopeError::Data(
                    "gold block receipt does not reconcile with frozen xbabe3 inputs".into(),
                ));
            }
            Ok((block, receipt))
        })();
        let (block, receipt) = match admitted {
            Ok(value) => value,
            Err(error) => {
                issues.push(format!("{}: {error}", record.dataset_id));
                continue;
            }
        };
        receipts_reconciled += 1;
        receipts_writer
            .write_all(&canonical_json(&receipt)?)
            .and_then(|_| receipts_writer.write_all(b"\n"))
            .map_err(|error| io_error(&receipts_partial, error))?;
        valid_expected_cells += cells_per_record;
        let routing = block
            .routing
            .as_ref()
            .expect("validated validation-cert block has routing");
        router_sketch_times.push(routing.sketch_ms);
        router_inference_times.push(routing.inference_ms);
        selected_expected_cells += routing.selected_candidates.len() * cells_per_candidate;
        let top_candidate = &routing.ranked_candidates[0];
        let mut top_cells = Vec::new();
        let mut block_coverage = BTreeMap::<(String, usize), usize>::new();
        for cell in &block.evidence {
            succeeded_cells += 1;
            let aggregate = candidate_aggregates
                .get_mut(&cell.candidate_id)
                .expect("validated candidate");
            aggregate.succeeded += 1;
            aggregate.tasks.insert(cell.task.clone());
            *aggregate.runtimes.entry(cell.runtime_ms).or_default() += 1;
            *aggregate
                .memories
                .entry(cell.peak_memory_bytes)
                .or_default() += 1;
            *aggregate
                .artifact_bytes
                .entry(cell.artifact_bytes)
                .or_default() += 1;
            let denominator = cell.null_loss - cell.trtr_loss;
            if denominator.abs() > f64::EPSILON {
                aggregate.retention_sum += (cell.null_loss - cell.tstr_loss) / denominator;
                aggregate.retention_count += 1;
            }
            *candidate_availability
                .get_mut(&cell.candidate_id)
                .expect("validated candidate") += 1;
            *auditor_availability
                .get_mut(&cell.auditor_id)
                .expect("validated auditor") += 1;
            all_artifact_bytes_nonzero &= cell.artifact_bytes > 0;
            exhaustive_runtime_ms += cell.runtime_ms as f64;
            if cell.routed {
                routed_cells += 1;
                selected_runtime_ms += cell.runtime_ms as f64;
            } else {
                shadow_cells += 1;
            }
            if &cell.candidate_id == top_candidate {
                top_cells.push(cell);
                *block_coverage
                    .entry((cell.auditor_id.clone(), cell.size_multiplier))
                    .or_default() += 1;
                coverage_groups
                    .get_mut(&(
                        record.task.clone(),
                        record.structural_profile.clone(),
                        cell.auditor_id.clone(),
                        cell.size_multiplier,
                    ))
                    .expect("initialized coverage group")
                    .evaluated += 1;
                metrics_writer
                    .write_all(&canonical_json(cell)?)
                    .and_then(|_| metrics_writer.write_all(b"\n"))
                    .map_err(|error| io_error(&metrics_partial, error))?;
            }
        }
        explicit_failed_cells += block.failures.len();
        let mut top_timeout_by_auditor = BTreeMap::<String, usize>::new();
        let mut block_has_top_timeout = false;
        for failure in &block.failures {
            if is_timeout_failure(&failure.error) {
                timed_out_cells += 1;
                candidate_aggregates
                    .get_mut(&failure.candidate_id)
                    .expect("validated failure candidate")
                    .timed_out += 1;
                if &failure.candidate_id == top_candidate {
                    *top_timeout_by_auditor
                        .entry(failure.auditor_id.clone())
                        .or_default() += 1;
                    block_has_top_timeout = true;
                }
            }
        }
        if block_has_top_timeout {
            timeout_lineages.insert(record.lineage_group_id.clone());
        }
        for auditor in &auditor_ids {
            let mut remaining_timeouts = top_timeout_by_auditor
                .get(auditor)
                .copied()
                .unwrap_or_default();
            for size in size_multipliers {
                let evaluated = block_coverage
                    .get(&(auditor.clone(), size))
                    .copied()
                    .unwrap_or_default();
                let timed_out = remaining_timeouts.min(
                    generation_seeds
                        .len()
                        .saturating_mul(auditor_seeds.len())
                        .saturating_sub(evaluated),
                );
                remaining_timeouts -= timed_out;
                coverage_groups
                    .get_mut(&(
                        record.task.clone(),
                        record.structural_profile.clone(),
                        auditor.clone(),
                        size,
                    ))
                    .expect("initialized coverage group")
                    .timed_out += timed_out;
            }
        }

        if top_cells.len() == cells_per_candidate {
            evaluated_lineages.insert(record.lineage_group_id.clone());
            let mut seed_groups = BTreeMap::<(String, usize), Vec<f64>>::new();
            for cell in &top_cells {
                kpi_cells.push(KpiCell {
                    task: table_task(&cell.task)?,
                    train_rows: cell.train_rows,
                    features: cell.features,
                    auditor: cell.auditor_id.clone(),
                    size_multiplier: cell.size_multiplier,
                    lineage_group_id: cell.lineage_group_id.clone(),
                    generation_seed: cell.generation_seed,
                    auditor_seed: cell.auditor_seed,
                    null_loss: cell.null_loss,
                    trtr_loss: cell.trtr_loss,
                    tstr_loss: cell.tstr_loss,
                    calibration_degradation: cell.calibration_degradation,
                    rare_class_or_tail_retention: cell.rare_class_or_tail_retention,
                    supported_subgroup_retention: cell.supported_subgroup_retention,
                    nominal_95_coverage: cell.nominal_95_coverage,
                });
                update_optional_max(
                    &mut calibration_degradation_max,
                    cell.calibration_degradation,
                );
                update_optional_min(
                    &mut rare_tail_subgroup_retention_min,
                    cell.rare_class_or_tail_retention,
                );
                update_optional_min(
                    &mut rare_tail_subgroup_retention_min,
                    cell.supported_subgroup_retention,
                );
                update_optional_min(&mut driver_agreement_min, cell.driver_agreement);
                update_optional_min(&mut joint_fidelity_min, cell.joint_fidelity);
                update_optional_max(&mut query_error_max, cell.query_p95_normalized_error);
                update_optional_min(&mut nominal_coverage_min, cell.nominal_95_coverage);
                update_optional_max(&mut nominal_coverage_max, cell.nominal_95_coverage);
                update_optional_max(&mut type_i_error_max, cell.type_i_error);
                update_optional_max(&mut membership_auc_max, cell.membership_auc);
                feature_importance_complete &= cell.version == JOB_EVIDENCE_VERSION
                    && cell.feature_importance_feature_count == cell.features
                    && cell.feature_importance_real_shares.len() == cell.features
                    && cell.feature_importance_synthetic_shares.len() == cell.features;
                if cell.feature_importance_informative_count >= 3 {
                    feature_importance_applicable = true;
                    if let Some(value) = cell.feature_importance_spearman {
                        update_optional_min(&mut feature_importance_spearman_min, Some(value));
                    } else {
                        feature_importance_missing_rank = true;
                    }
                    update_optional_min(
                        &mut feature_importance_top_k_jaccard_min,
                        Some(cell.feature_importance_top_k_agreement),
                    );
                }
                update_optional_max(
                    &mut attribute_advantage_max,
                    cell.attribute_inference_advantage,
                );
                exact_copies += cell.exact_copies;
                near_copies += cell.near_copies;
                canary_extractions += cell.canary_extractions;
                lineage_leaks += cell.lineage_leaks;
                seed_groups
                    .entry((cell.auditor_id.clone(), cell.size_multiplier))
                    .or_default()
                    .push(cell.tstr_loss);
            }
            for values in seed_groups.values() {
                let deviation = if values.len() > 1 {
                    let average = values.iter().sum::<f64>() / values.len() as f64;
                    (values
                        .iter()
                        .map(|value| (value - average).powi(2))
                        .sum::<f64>()
                        / (values.len() - 1) as f64)
                        .sqrt()
                } else {
                    0.0
                };
                update_optional_max(&mut across_seed_validation_loss_stddev, Some(deviation));
            }
        }

        if block.failures.is_empty() && block.evidence.len() == cells_per_record {
            let mut outcomes = BTreeMap::<String, CertCandidateOutcome>::new();
            for candidate in &candidate_ids {
                let matching = block
                    .evidence
                    .iter()
                    .filter(|cell| &cell.candidate_id == candidate)
                    .collect::<Vec<_>>();
                if matching.len() != cells_per_candidate {
                    continue;
                }
                outcomes.insert(
                    candidate.clone(),
                    CertCandidateOutcome {
                        utility: matching
                            .iter()
                            .map(|cell| bounded_router_utility(cell))
                            .sum::<f64>()
                            / matching.len() as f64,
                        runtime_ms: matching
                            .iter()
                            .map(|cell| cell.runtime_ms as f64)
                            .sum::<f64>()
                            / matching.len() as f64,
                    },
                );
            }
            if outcomes.len() == candidate_ids.len() {
                complete_oracle_outcomes += 1;
                let (_oracle_id, oracle) = outcomes
                    .iter()
                    .max_by(|left, right| {
                        left.1
                            .utility
                            .total_cmp(&right.1.utility)
                            .then_with(|| right.0.cmp(left.0))
                    })
                    .expect("complete candidate outcomes");
                let selected = outcomes
                    .get(top_candidate)
                    .expect("validated top candidate outcome");
                let selected_regret = (oracle.utility - selected.utility).max(0.0);
                regrets.push(selected_regret);
                profile_regrets
                    .entry(record.structural_profile.clone())
                    .or_default()
                    .push(selected_regret);
                let top_four_best = routing
                    .top_four_candidates
                    .iter()
                    .filter_map(|candidate| outcomes.get(candidate))
                    .map(|outcome| outcome.utility)
                    .reduce(f64::max)
                    .unwrap_or(f64::NEG_INFINITY);
                top_four_hits += usize::from(oracle.utility - top_four_best <= 1e-9);
                let mut random_hasher = blake3::Hasher::new();
                random_hasher.update(record.lineage_group_id.as_bytes());
                random_hasher.update(record.structural_profile.as_bytes());
                let random_index =
                    usize::from(random_hasher.finalize().as_bytes()[1]) % candidate_ids.len();
                let random = outcomes
                    .get(&candidate_ids[random_index])
                    .expect("frozen random candidate");
                random_regrets.push((oracle.utility - random.utility).max(0.0));
                if let Some(fixed) = outcomes.get(&router_evidence.best_fixed_candidate_id) {
                    fixed_regrets.push((oracle.utility - fixed.utility).max(0.0));
                    let max_runtime = outcomes
                        .values()
                        .map(|outcome| outcome.runtime_ms)
                        .reduce(f64::max)
                        .unwrap_or(0.0)
                        + 1.0;
                    let hypervolume = |outcome: &CertCandidateOutcome| {
                        outcome.utility.max(0.0) * (max_runtime - outcome.runtime_ms).max(0.0)
                    };
                    hypervolume_pairs.push((hypervolume(selected), hypervolume(fixed)));
                }
            }
        }
    }
    metrics_writer
        .flush()
        .map_err(|error| io_error(&metrics_partial, error))?;
    metrics_writer
        .get_ref()
        .sync_all()
        .map_err(|error| io_error(&metrics_partial, error))?;
    receipts_writer
        .flush()
        .map_err(|error| io_error(&receipts_partial, error))?;
    receipts_writer
        .get_ref()
        .sync_all()
        .map_err(|error| io_error(&receipts_partial, error))?;
    drop(metrics_writer);
    drop(receipts_writer);
    fs::rename(&metrics_partial, out.join("metrics.jsonl"))
        .map_err(|error| io_error(out.join("metrics.jsonl"), error))?;
    fs::rename(&receipts_partial, out.join("receipts.jsonl"))
        .map_err(|error| io_error(out.join("receipts.jsonl"), error))?;

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

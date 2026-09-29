|| -> Result<MetricsBundle> {


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
    include!("finalize_metrics.rs")()
}

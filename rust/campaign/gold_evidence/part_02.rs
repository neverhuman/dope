

pub fn summarize_evidence(metric_paths: &[PathBuf], out: &Path) -> Result<MeasuredEvidenceSummary> {
    if metric_paths.is_empty() {
        return Err(DopeError::Data("evidence metrics inputs are empty".into()));
    }
    #[derive(Default)]
    struct CandidateAggregate {
        cells: usize,
        bounded_utility_sum: f64,
        runtimes: BTreeMap<u64, usize>,
        memories: BTreeMap<u64, usize>,
        artifact_bytes: BTreeMap<u64, usize>,
    }
    let histogram_percentile = |histogram: &BTreeMap<u64, usize>, cells: usize| {
        if cells == 0 {
            return 0;
        }
        let index = ((cells - 1) as f64 * 0.95).ceil() as usize;
        let mut cumulative = 0usize;
        for (&value, &count) in histogram {
            cumulative += count;
            if cumulative > index {
                return value;
            }
        }
        0
    };
    let auditor_ids = auditor_specs()
        .into_iter()
        .map(|auditor| auditor.id)
        .collect::<Vec<_>>();
    let mut full_replicates = BTreeMap::<(String, String, String, String, bool), u128>::new();
    let mut other_identities = BTreeSet::<String>::new();
    let mut candidates = BTreeMap::<String, CandidateAggregate>::new();
    let mut lineage_groups = BTreeSet::<String>::new();
    let mut lineage_profile_outcomes = BTreeSet::<(String, String)>::new();
    let mut auditors = BTreeSet::<String>::new();
    let mut tasks = BTreeSet::<String>::new();
    let mut cells = 0usize;
    let mut calibration_degradation_max = None;
    let mut rare_tail_retention_min = None;
    let mut subgroup_retention_min = None;
    let mut nominal_coverage_min = None;
    let mut nominal_coverage_max = None;
    let mut driver_agreement_min = None;
    let mut joint_fidelity_min = None;
    let mut query_error_max = None;
    let mut type_i_error_max = None;
    let mut membership_auc_max = None;
    let mut attribute_advantage_max = None;
    let mut importance_complete = true;
    let mut importance_applicable = false;
    let mut importance_spearman_min = None::<f64>;
    let mut importance_jaccard_min = None::<f64>;
    let mut importance_missing_rank = false;
    let mut exact_copies = 0usize;
    let mut near_copies = 0usize;
    let update_min = |slot: &mut Option<f64>, value: Option<f64>| {
        if let Some(value) = value {
            *slot = Some(slot.map_or(value, |existing| existing.min(value)));
        }
    };
    let update_max = |slot: &mut Option<f64>, value: Option<f64>| {
        if let Some(value) = value {
            *slot = Some(slot.map_or(value, |existing| existing.max(value)));
        }
    };
    for_each_evidence(metric_paths, &mut |cell| {
        if let Ok(slot) = router_replicate_slot(&cell, &auditor_ids) {
            let seen = full_replicates
                .entry((
                    cell.phase.clone(),
                    cell.lineage_group_id.clone(),
                    cell.structural_profile.clone(),
                    cell.candidate_id.clone(),
                    cell.routed,
                ))
                .or_default();
            let bit = 1u128 << slot;
            if *seen & bit != 0 {
                return Err(DopeError::Data(
                    "evidence metrics contain duplicate cell identities".into(),
                ));
            }
            *seen |= bit;
        } else {
            let identity = hashes(&canonical_json(&serde_json::json!({
                "phase": cell.phase,
                "lineage": cell.lineage_group_id,
                "profile": cell.structural_profile,
                "candidate": cell.candidate_id,
                "auditor": cell.auditor_id,
                "size_multiplier": cell.size_multiplier,
                "generation_seed": cell.generation_seed,
                "auditor_seed": cell.auditor_seed,
                "routed": cell.routed,
            }))?)
            .blake3;
            if !other_identities.insert(identity) {
                return Err(DopeError::Data(
                    "evidence metrics contain duplicate cell identities".into(),
                ));
            }
        }
        let aggregate = candidates.entry(cell.candidate_id.clone()).or_default();
        aggregate.cells += 1;
        aggregate.bounded_utility_sum += bounded_router_utility(&cell);
        *aggregate.runtimes.entry(cell.runtime_ms).or_default() += 1;
        *aggregate
            .memories
            .entry(cell.peak_memory_bytes)
            .or_default() += 1;
        *aggregate
            .artifact_bytes
            .entry(cell.artifact_bytes)
            .or_default() += 1;
        lineage_groups.insert(cell.lineage_group_id.clone());
        lineage_profile_outcomes.insert((
            cell.lineage_group_id.clone(),
            cell.structural_profile.clone(),
        ));
        auditors.insert(cell.auditor_id.clone());
        tasks.insert(cell.task.clone());
        update_max(
            &mut calibration_degradation_max,
            cell.calibration_degradation,
        );
        update_min(
            &mut rare_tail_retention_min,
            cell.rare_class_or_tail_retention,
        );
        update_min(
            &mut subgroup_retention_min,
            cell.supported_subgroup_retention,
        );
        update_min(&mut nominal_coverage_min, cell.nominal_95_coverage);
        update_max(&mut nominal_coverage_max, cell.nominal_95_coverage);
        update_min(&mut driver_agreement_min, cell.driver_agreement);
        update_min(&mut joint_fidelity_min, cell.joint_fidelity);
        update_max(&mut query_error_max, cell.query_p95_normalized_error);
        update_max(&mut type_i_error_max, cell.type_i_error);
        update_max(&mut membership_auc_max, cell.membership_auc);
        importance_complete &= cell.version == JOB_EVIDENCE_VERSION
            && cell.feature_importance_feature_count == cell.features
            && cell.feature_importance_real_shares.len() == cell.features
            && cell.feature_importance_synthetic_shares.len() == cell.features;
        if cell.feature_importance_informative_count >= 2 {
            importance_applicable = true;
            if let Some(value) = cell.feature_importance_spearman {
                update_min(&mut importance_spearman_min, Some(value));
            } else {
                importance_missing_rank = true;
            }
            update_min(
                &mut importance_jaccard_min,
                Some(cell.feature_importance_top_k_agreement),
            );
        }
        update_max(
            &mut attribute_advantage_max,
            cell.attribute_inference_advantage,
        );
        exact_copies += cell.exact_copies;
        near_copies += cell.near_copies;
        cells += 1;
        Ok(())
    })?;
    let mut candidate_frontier = candidates
        .into_iter()
        .map(|(candidate_id, aggregate)| MeasuredCandidateSummary {
            implementation_hash: empirical_backends()
                .into_iter()
                .find(|backend| backend.id == candidate_id)
                .map(|backend| backend.implementation_hash)
                .unwrap_or_default(),
            candidate_id,
            cells: aggregate.cells,
            mean_bounded_utility: aggregate.bounded_utility_sum / aggregate.cells.max(1) as f64,
            runtime_p95_ms: histogram_percentile(&aggregate.runtimes, aggregate.cells),
            peak_memory_p95_bytes: histogram_percentile(&aggregate.memories, aggregate.cells),
            artifact_p95_bytes: histogram_percentile(&aggregate.artifact_bytes, aggregate.cells),
        })
        .collect::<Vec<_>>();
    candidate_frontier.sort_by(|left, right| {
        right
            .mean_bounded_utility
            .total_cmp(&left.mean_bounded_utility)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    let summary = MeasuredEvidenceSummary {
        format: "dope-measured-evidence-summary".into(),
        version: 2,
        cells,
        lineage_groups: lineage_groups.len(),
        lineage_profile_outcomes: lineage_profile_outcomes.len(),
        candidates: candidate_frontier.len(),
        auditors: auditors.len(),
        tasks: tasks.into_iter().collect(),
        candidate_frontier,
        calibration_degradation_max,
        rare_tail_retention_min,
        subgroup_retention_min,
        nominal_coverage_min,
        nominal_coverage_max,
        driver_agreement_min,
        joint_fidelity_min,
        query_error_max,
        type_i_error_max,
        membership_auc_max,
        feature_importance_complete: cells > 0 && importance_complete,
        feature_importance_applicable: importance_applicable,
        feature_importance_spearman_min: if importance_missing_rank {
            None
        } else {
            importance_spearman_min
        },
        feature_importance_top_k_jaccard_min: importance_jaccard_min,
        attribute_advantage_max,
        exact_copies,
        near_copies,
    };
    write_canonical(out, &summary)?;
    Ok(summary)
}

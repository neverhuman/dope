
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
    include!("validation_metrics/aggregate_records.rs")()
}

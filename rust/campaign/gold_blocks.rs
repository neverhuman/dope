pub struct RoutedCertOptions<'a> {
    pub state_dir: &'a Path,
    pub cohort_path: &'a Path,
    pub router_bundle: &'a Path,
    pub out: &'a Path,
    pub shard_index: usize,
    pub shard_count: usize,
}

pub fn run_routed_cert_matrix(options: &RoutedCertOptions<'_>) -> Result<BaselineSummary> {
    require_host("xbabe3", "routed validation-cert evaluation")?;
    let state = load_state(options.state_dir)?;
    if state.phase != CampaignPhase::ValidationCertOpen
        || state.validation_cert_open_count != 1
        || state.sealed_test_open_count != 0
    {
        return Err(DopeError::Data(
            "routed validation-cert requires the one authorization event".into(),
        ));
    }
    let bundle_hashes = file_hashes(options.router_bundle)?;
    if state.router_bundle.as_ref() != Some(&bundle_hashes) {
        return Err(DopeError::Data(
            "validation-cert router differs from the frozen bundle".into(),
        ));
    }
    let cohort: CohortPlan = read_json(options.cohort_path)?;
    if cohort.kind != "validation-cert"
        || cohort.records.len() != state.validation_cert_lineage_groups
        || cohort
            .records
            .iter()
            .map(|record| &record.lineage_group_id)
            .collect::<BTreeSet<_>>()
            .len()
            != state.validation_cert_lineage_groups
    {
        return Err(DopeError::Data(
            "validation-cert cohort is incomplete or not guarded".into(),
        ));
    }
    let bundle: RouterBundle = read_json(options.router_bundle)?;
    let router = QuantizedRouter::new(bundle)?;
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
    run_gold_matrix_internal(
        &GoldMatrixOptions {
            cohort_path: options.cohort_path,
            out: options.out,
            candidate_ids: &candidate_ids,
            auditor_ids: &auditor_ids,
            size_multipliers: &size_multipliers,
            generation_seeds: &generation_seeds,
            auditor_seeds: &auditor_seeds,
            shard_index: options.shard_index,
            shard_count: options.shard_count,
            primary_only: false,
            blocks_only: true,
            neural_target_weight: 2.0,
            neural_structural_penalty: 0.0,
        },
        Some(&RoutedCertContext {
            router: &router,
            bundle_sha256: &bundle_hashes.sha256,
        }),
    )
}

fn validate_gold_record_block(
    block: &GoldRecordBlock,
    record: &CohortRecord,
    phase: &str,
    expected_identity: &str,
    candidate_ids: &[String],
    auditor_ids: &[String],
    routing_sha256: Option<&str>,
) -> Result<()> {
    let candidates = candidate_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let auditors = auditor_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let routed_candidates = match (routing_sha256, &block.routing) {
        (None, None) => BTreeSet::new(),
        (Some(expected), Some(routing))
            if routing.format == "dope-gold-routing-decision"
                && routing.version == 1
                && routing.router_bundle_sha256 == expected
                && is_lower_hex(&routing.prediction_sha256, &[64])
                && routing.sketch_ms.is_finite()
                && routing.sketch_ms >= 0.0
                && routing.inference_ms.is_finite()
                && routing.inference_ms >= 0.0
                && routing.ranked_candidates.len() == candidate_ids.len()
                && routing
                    .ranked_candidates
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    == candidate_ids.len()
                && routing
                    .ranked_candidates
                    .iter()
                    .all(|candidate| candidates.contains(candidate.as_str()))
                && routing.top_four_candidates
                    == routing
                        .ranked_candidates
                        .iter()
                        .take(4)
                        .cloned()
                        .collect::<Vec<_>>()
                && !routing.selected_candidates.is_empty()
                && routing.selected_candidates.len() <= candidate_ids.len()
                && routing
                    .selected_candidates
                    .iter()
                    .all(|candidate| candidates.contains(candidate.as_str()))
                && routing
                    .selected_candidates
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    == routing.selected_candidates.len() =>
        {
            routing
                .selected_candidates
                .iter()
                .map(String::as_str)
                .collect()
        }
        _ => {
            return Err(DopeError::Data(format!(
                "invalid routing decision in gold block {}",
                record.dataset_id
            )));
        }
    };
    let optional_finite = |value: Option<f64>| value.is_none_or(f64::is_finite);
    let mut identities = BTreeSet::new();
    for cell in &block.evidence {
        let identity = (
            cell.candidate_id.as_str(),
            cell.auditor_id.as_str(),
            cell.size_multiplier,
            cell.generation_seed,
            cell.auditor_seed,
        );
        if cell.format != "dope-job-evidence"
            || !(2..=JOB_EVIDENCE_VERSION).contains(&cell.version)
            || cell.phase != phase
            || cell.task != record.task
            || cell.lineage_group_id != record.lineage_group_id
            || cell.structural_profile != record.structural_profile
            || cell.train_rows == 0
            || cell.features == 0
            || !candidates.contains(cell.candidate_id.as_str())
            || !auditors.contains(cell.auditor_id.as_str())
            || ![1, 4].contains(&cell.size_multiplier)
            || ![1829, 99238, 196647].contains(&cell.generation_seed)
            || ![57721, 161803, 271828].contains(&cell.auditor_seed)
            || cell.routed != routed_candidates.contains(cell.candidate_id.as_str())
            || ![cell.null_loss, cell.trtr_loss, cell.tstr_loss]
                .into_iter()
                .all(|value| value.is_finite() && value >= 0.0)
            || !is_lower_hex(&cell.artifact_sha256, &[64])
            || !is_lower_hex(&cell.artifact_blake3, &[64])
            || ![
                cell.calibration_degradation,
                cell.rare_class_or_tail_retention,
                cell.supported_subgroup_retention,
                cell.nominal_95_coverage,
                cell.driver_agreement,
                cell.joint_fidelity,
                cell.query_p95_normalized_error,
                cell.type_i_error,
                cell.membership_auc,
                cell.attribute_inference_advantage,
                cell.feature_importance_spearman,
            ]
            .into_iter()
            .all(optional_finite)
            || !cell.feature_importance_top_k_agreement.is_finite()
            || !(0.0..=1.0).contains(&cell.feature_importance_top_k_agreement)
            || cell.feature_importance_feature_count == 0
            || cell.feature_importance_feature_count > cell.features
            || cell.feature_importance_informative_count > cell.feature_importance_feature_count
            || cell.peak_cpu_memory_bytes != cell.peak_memory_bytes
            || cell
                .fitting_time_ms
                .saturating_add(cell.sampling_time_ms)
                .saturating_add(cell.auditor_time_ms)
                > cell.runtime_ms
            || !identities.insert(identity)
        {
            return Err(DopeError::Data(format!(
                "invalid gold evidence in record block {}",
                record.dataset_id
            )));
        }
    }
    let expected_cells = candidate_ids.len() * auditor_ids.len() * 2 * 3 * 3;
    if block.format != "dope-gold-record-block"
        || block.version != GOLD_RECORD_BLOCK_VERSION
        || block.dataset_id != record.dataset_id
        || block.matrix_identity != expected_identity
        || block.evidence.len() + block.failures.len() != expected_cells
        || block.failures.iter().any(|failure| {
            failure.dataset_id != record.dataset_id
                || !candidates.contains(failure.candidate_id.as_str())
                || !auditors.contains(failure.auditor_id.as_str())
                || failure.error.is_empty()
        })
    {
        return Err(DopeError::Data(format!(
            "invalid or incomplete gold record block {}",
            record.dataset_id
        )));
    }
    Ok(())
}

pub fn sign_gold_blocks(
    out: &Path,
    cohort_path: &Path,
    worker: &str,
    source_sha256: &str,
    evaluator_binary_sha256: &str,
    router_bundle: Option<&Path>,
    receipt_key: &Path,
) -> Result<GoldReceiptSummary> {
    if !matches!(worker, "xbabe1" | "xbabe2" | "xbabe3")
        || !is_lower_hex(source_sha256, &[64])
        || !is_lower_hex(evaluator_binary_sha256, &[64])
    {
        return Err(DopeError::Data(
            "invalid gold receipt signer identity".into(),
        ));
    }
    let key = load_receipt_key(receipt_key)?;
    let cohort: CohortPlan = read_json(cohort_path)?;
    let phase = match cohort.kind.as_str() {
        "training-gold" => "training_gold",
        "validation-select" => "validation_select",
        "validation-cert" => "validation_cert",
        _ => return Err(DopeError::Data("unsupported gold receipt cohort".into())),
    };
    let router_bundle_sha256 = router_bundle
        .map(file_hashes)
        .transpose()?
        .map(|hashes| hashes.sha256);
    if (phase == "validation_cert") != router_bundle_sha256.is_some() {
        return Err(DopeError::Data(
            "validation-cert receipts require exactly one frozen router bundle".into(),
        ));
    }
    let cohort_hashes = file_hashes(cohort_path)?;
    let records = cohort
        .records
        .iter()
        .map(|record| (record.dataset_id.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    if records.len() != cohort.records.len() {
        return Err(DopeError::Data(
            "gold receipt cohort contains duplicate dataset IDs".into(),
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
    let block_dir = out.join("blocks");
    let receipt_dir = out.join("receipts");
    fs::create_dir_all(&receipt_dir).map_err(|error| io_error(&receipt_dir, error))?;
    let mut block_paths = fs::read_dir(&block_dir)
        .map_err(|error| io_error(&block_dir, error))?
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .map_err(|error| io_error(&block_dir, error))
        })
        .collect::<Result<Vec<_>>>()?;
    block_paths.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "json")
    });
    block_paths.sort();
    let mut evidence_cells = 0usize;
    let mut failed_cells = 0usize;
    for block_path in &block_paths {
        let block: GoldRecordBlock = read_json(block_path)?;
        let record = records.get(block.dataset_id.as_str()).ok_or_else(|| {
            DopeError::Data(format!(
                "gold block dataset {} is absent from its cohort",
                block.dataset_id
            ))
        })?;
        let matrix_identity = gold_record_matrix_identity(&GoldMatrixIdentity {
            phase,
            dataset_id: &record.dataset_id,
            candidate_ids: &candidate_ids,
            auditor_ids: &auditor_ids,
            size_multipliers: &size_multipliers,
            generation_seeds: &generation_seeds,
            auditor_seeds: &auditor_seeds,
            primary_only: false,
            neural_target_weight: block.neural_target_weight,
            neural_structural_penalty: block.neural_structural_penalty,
            routing_sha256: router_bundle_sha256.as_deref(),
        })?;
        if block_path.file_stem().and_then(|value| value.to_str()) != Some(&matrix_identity) {
            return Err(DopeError::Data(format!(
                "gold block filename does not match its identity at {}",
                block_path.display()
            )));
        }
        validate_gold_record_block(
            &block,
            record,
            phase,
            &matrix_identity,
            &candidate_ids,
            &auditor_ids,
            router_bundle_sha256.as_deref(),
        )?;
        let block_hashes = file_hashes(block_path)?;
        let receipt_path = receipt_dir.join(format!("{matrix_identity}.json"));
        if receipt_path.is_file() {
            let receipt: GoldBlockReceipt = read_json(&receipt_path)?;
            receipt.verify(&key)?;
            if receipt.dataset_id != block.dataset_id
                || receipt.matrix_identity != matrix_identity
                || receipt.phase != phase
                || receipt.worker != worker
                || receipt.source_sha256 != source_sha256
                || receipt.evaluator_binary_sha256 != evaluator_binary_sha256
                || receipt.router_bundle_sha256 != router_bundle_sha256
                || receipt.cohort != cohort_hashes
                || receipt.block != block_hashes
                || receipt.evidence_cells != block.evidence.len()
                || receipt.failed_cells != block.failures.len()
            {
                return Err(DopeError::Data(format!(
                    "gold receipt does not reconcile at {}",
                    receipt_path.display()
                )));
            }
        } else {
            let mut receipt = GoldBlockReceipt {
                format: "dope-gold-block-receipt".into(),
                version: GOLD_BLOCK_RECEIPT_VERSION,
                dataset_id: block.dataset_id.clone(),
                matrix_identity: matrix_identity.clone(),
                phase: phase.into(),
                worker: worker.into(),
                source_sha256: source_sha256.into(),
                evaluator_binary_sha256: evaluator_binary_sha256.into(),
                router_bundle_sha256: router_bundle_sha256.clone(),
                cohort: cohort_hashes.clone(),
                block: block_hashes,
                evidence_cells: block.evidence.len(),
                failed_cells: block.failures.len(),
                signed_unix_seconds: unix_now(),
                signature: None,
            };
            receipt.sign(&key)?;
            write_canonical(&receipt_path, &receipt)?;
        }
        evidence_cells += block.evidence.len();
        failed_cells += block.failures.len();
    }
    let summary = GoldReceiptSummary {
        format: "dope-gold-receipt-summary".into(),
        version: 1,
        worker: worker.into(),
        blocks: block_paths.len(),
        evidence_cells,
        failed_cells,
        cohort: cohort_hashes,
    };
    write_canonical(
        &out.join(format!("receipt-summary-{worker}.json")),
        &summary,
    )?;
    Ok(summary)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GoldBlockStatus {
    pub format: String,
    pub version: u8,
    pub phase: String,
    pub eligible_lineages: usize,
    pub completed_lineages: usize,
    pub missing_lineages: usize,
    pub attempted_cells: usize,
    pub completed_cells: usize,
    pub failed_cells: usize,
    pub throughput_lineages_per_hour: Option<f64>,
    pub coverage: f64,
    pub candidate_successes: BTreeMap<String, usize>,
    pub auditor_successes: BTreeMap<String, usize>,
    pub invalid_blocks: Vec<String>,
    pub router_regret: Option<f64>,
    pub ptf_v1: Option<f64>,
    pub production_score_available: bool,
    pub updated_unix_seconds: u64,
}

pub fn gold_block_status(
    cohort_path: &Path,
    gold_out: &Path,
    router_bundle: Option<&Path>,
    metrics_path: Option<&Path>,
) -> Result<GoldBlockStatus> {
    let cohort: CohortPlan = read_json(cohort_path)?;
    let phase = match cohort.kind.as_str() {
        "training-gold" => "training_gold",
        "validation-select" => "validation_select",
        "validation-cert" => "validation_cert",
        _ => return Err(DopeError::Data("unsupported gold status cohort".into())),
    };
    let routing_sha256 = router_bundle
        .map(file_hashes)
        .transpose()?
        .map(|hash| hash.sha256);
    if (phase == "validation_cert") != routing_sha256.is_some() {
        return Err(DopeError::Data(
            "validation-cert block status requires its frozen router bundle".into(),
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
    let cells_per_record = candidate_ids.len()
        * auditor_ids.len()
        * size_multipliers.len()
        * generation_seeds.len()
        * auditor_seeds.len();
    let mut completed_lineages = 0usize;
    let mut completed_cells = 0usize;
    let mut failed_cells = 0usize;
    let mut candidate_successes = candidate_ids
        .iter()
        .map(|candidate| (candidate.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut auditor_successes = auditor_ids
        .iter()
        .map(|auditor| (auditor.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut invalid_blocks = Vec::new();
    let mut first_modified = None::<SystemTime>;
    let mut last_modified = None::<SystemTime>;
    for record in &cohort.records {
        let matrix_identity = gold_record_matrix_identity(&GoldMatrixIdentity {
            phase,
            dataset_id: &record.dataset_id,
            candidate_ids: &candidate_ids,
            auditor_ids: &auditor_ids,
            size_multipliers: &size_multipliers,
            generation_seeds: &generation_seeds,
            auditor_seeds: &auditor_seeds,
            primary_only: false,
            neural_target_weight: 2.0,
            neural_structural_penalty: 0.0,
            routing_sha256: routing_sha256.as_deref(),
        })?;
        let path = gold_out
            .join("blocks")
            .join(format!("{matrix_identity}.json"));
        if !path.is_file() {
            continue;
        }
        let block = match read_json::<GoldRecordBlock>(&path).and_then(|block| {
            validate_gold_record_block(
                &block,
                record,
                phase,
                &matrix_identity,
                &candidate_ids,
                &auditor_ids,
                routing_sha256.as_deref(),
            )?;
            Ok(block)
        }) {
            Ok(block) => block,
            Err(error) => {
                invalid_blocks.push(format!("{}: {error}", record.dataset_id));
                continue;
            }
        };
        completed_lineages += 1;
        completed_cells += block.evidence.len();
        failed_cells += block.failures.len();
        for cell in block.evidence {
            *candidate_successes
                .get_mut(&cell.candidate_id)
                .expect("validated candidate") += 1;
            *auditor_successes
                .get_mut(&cell.auditor_id)
                .expect("validated auditor") += 1;
        }
        let modified = fs::metadata(&path)
            .map_err(|error| io_error(&path, error))?
            .modified()
            .map_err(|error| io_error(&path, error))?;
        first_modified = Some(first_modified.map_or(modified, |current| current.min(modified)));
        last_modified = Some(last_modified.map_or(modified, |current| current.max(modified)));
    }
    let throughput_lineages_per_hour = first_modified
        .zip(last_modified)
        .and_then(|(first, last)| last.duration_since(first).ok())
        .filter(|elapsed| !elapsed.is_zero())
        .map(|elapsed| completed_lineages as f64 / elapsed.as_secs_f64() * 3_600.0);
    let metrics = metrics_path.map(read_json::<MetricsBundle>).transpose()?;
    let router_regret = metrics
        .as_ref()
        .and_then(|metrics| metrics.kpi.candidate_regret);
    let ptf_v1 = metrics.as_ref().and_then(|metrics| metrics.kpi.ptf_v1);
    let production_score_available = metrics
        .as_ref()
        .is_some_and(|metrics| metrics.kpi.production_score_available);
    Ok(GoldBlockStatus {
        format: "dope-gold-block-status".into(),
        version: 1,
        phase: phase.into(),
        eligible_lineages: cohort.records.len(),
        completed_lineages,
        missing_lineages: cohort.records.len().saturating_sub(completed_lineages),
        attempted_cells: cohort.records.len() * cells_per_record,
        completed_cells,
        failed_cells,
        throughput_lineages_per_hour,
        coverage: completed_lineages as f64 / cohort.records.len().max(1) as f64,
        candidate_successes,
        auditor_successes,
        invalid_blocks,
        router_regret,
        ptf_v1,
        production_score_available,
        updated_unix_seconds: unix_now(),
    })
}

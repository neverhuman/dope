

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

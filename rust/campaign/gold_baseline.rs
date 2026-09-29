
/// Runs the first honest matrix with one fixed generation and auditor seed.
/// Each cell is persisted independently so interruption and restart do not
/// discard completed evidence.
pub fn run_baseline(
    cohort_path: &Path,
    out: &Path,
    candidate_ids: &[String],
    auditor_ids: &[String],
    shard_index: usize,
    shard_count: usize,
    primary_only: bool,
) -> Result<BaselineSummary> {
    if shard_count == 0 || shard_index >= shard_count {
        return Err(DopeError::Data("invalid baseline shard".into()));
    }
    let cohort: CohortPlan = read_json(cohort_path)?;
    if cohort.format != "dope-campaign-cohort" || cohort.kind != "baseline" {
        return Err(DopeError::Data(
            "baseline requires a frozen baseline cohort".into(),
        ));
    }
    let frozen_candidates = empirical_backends()
        .into_iter()
        .map(|value| value.id)
        .collect::<BTreeSet<_>>();
    let frozen_auditors = auditor_specs()
        .into_iter()
        .map(|value| value.id)
        .collect::<BTreeSet<_>>();
    if candidate_ids.is_empty()
        || auditor_ids.is_empty()
        || candidate_ids
            .iter()
            .any(|value| !frozen_candidates.contains(value))
        || auditor_ids
            .iter()
            .any(|value| !frozen_auditors.contains(value))
    {
        return Err(DopeError::Data(
            "baseline candidates and auditors must be nonempty frozen IDs".into(),
        ));
    }
    let cell_dir = out.join("cells");
    let cache_dir = out.join("cache");
    fs::create_dir_all(&cell_dir).map_err(|error| io_error(&cell_dir, error))?;
    let selected = cohort
        .records
        .iter()
        .enumerate()
        .filter(|(index, _)| index % shard_count == shard_index)
        .map(|(_, record)| record)
        .collect::<Vec<_>>();
    let mut evidence = Vec::new();
    let mut failures = Vec::new();
    for record in &selected {
        for candidate_id in candidate_ids {
            for auditor_id in auditor_ids {
                let identity = hashes(&canonical_json(&serde_json::json!({
                    "dataset": record.dataset_id,
                    "candidate": candidate_id,
                    "auditor": auditor_id,
                    "generation_seed": 1829,
                    "auditor_seed": 57721,
                    "primary_only": primary_only
                }))?)
                .blake3;
                let path = cell_dir.join(format!("{identity}.json"));
                if path.is_file() {
                    evidence.push(read_json::<JobEvidence>(&path)?);
                    continue;
                }
                match evaluate_gold_cell(&GoldCellOptions {
                    dataset_dir: &record.dataset_path,
                    task: if record.task == "binary" {
                        Task::Binary
                    } else {
                        Task::Regression
                    },
                    candidate_id,
                    auditor_id,
                    lineage_group_id: &record.lineage_group_id,
                    structural_profile: &record.structural_profile,
                    phase: "validation_select",
                    size_multiplier: 1,
                    generation_seed: 1829,
                    auditor_seed: 57721,
                    routed: false,
                    ancillary: !primary_only,
                    cache_dir: Some(&cache_dir),
                    neural_target_weight: 2.0,
                    neural_structural_penalty: 0.0,
                }) {
                    Ok(cell) => {
                        write_canonical(&path, &cell)?;
                        evidence.push(cell);
                    }
                    Err(error) => failures.push(BaselineFailure {
                        dataset_id: record.dataset_id.clone(),
                        candidate_id: candidate_id.clone(),
                        auditor_id: auditor_id.clone(),
                        size_multiplier: None,
                        generation_seed: None,
                        auditor_seed: None,
                        error: error.to_string(),
                    }),
                }
            }
        }
    }
    evidence.sort_by(|left, right| {
        left.structural_profile
            .cmp(&right.structural_profile)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
            .then_with(|| left.auditor_id.cmp(&right.auditor_id))
    });
    let metrics_path = out.join(format!("metrics-shard-{shard_index:02}.jsonl"));
    let mut metrics =
        File::create(&metrics_path).map_err(|error| io_error(&metrics_path, error))?;
    for cell in &evidence {
        metrics
            .write_all(&canonical_json(cell)?)
            .and_then(|_| metrics.write_all(b"\n"))
            .map_err(|error| io_error(&metrics_path, error))?;
    }
    let summary = BaselineSummary {
        format: "dope-baseline-summary".into(),
        version: 1,
        cohort_records: selected.len(),
        attempted: selected.len() * candidate_ids.len() * auditor_ids.len(),
        succeeded: evidence.len(),
        failed: failures.len(),
        failures,
    };
    write_canonical(
        &out.join(format!("summary-shard-{shard_index:02}.json")),
        &summary,
    )?;
    Ok(summary)
}

pub struct GoldMatrixOptions<'a> {
    pub cohort_path: &'a Path,
    pub out: &'a Path,
    pub candidate_ids: &'a [String],
    pub auditor_ids: &'a [String],
    pub size_multipliers: &'a [usize],
    pub generation_seeds: &'a [u64],
    pub auditor_seeds: &'a [u64],
    pub shard_index: usize,
    pub shard_count: usize,
    pub primary_only: bool,
    pub blocks_only: bool,
    pub neural_target_weight: f64,
    pub neural_structural_penalty: f64,
}

/// Executes the exhaustive frozen matrix. It is deliberately restart-safe at
/// cell granularity; terminal failures are retained in the shard summary and
/// are never replaced with another candidate's evidence.
pub fn run_gold_matrix(options: &GoldMatrixOptions<'_>) -> Result<BaselineSummary> {
    run_gold_matrix_internal(options, None)
}

struct RoutedCertContext<'a> {
    router: &'a QuantizedRouter,
    bundle_sha256: &'a str,
}

fn run_gold_matrix_internal(
    options: &GoldMatrixOptions<'_>,
    routed_cert: Option<&RoutedCertContext<'_>>,
) -> Result<BaselineSummary> {
    if options.shard_count == 0
        || options.shard_index >= options.shard_count
        || options.candidate_ids.is_empty()
        || options.auditor_ids.is_empty()
        || options.size_multipliers.is_empty()
        || options.generation_seeds.is_empty()
        || options.auditor_seeds.is_empty()
        || options
            .size_multipliers
            .iter()
            .any(|value| !matches!(value, 1 | 4))
        || options
            .auditor_seeds
            .iter()
            .any(|value| ![57721, 161803, 271828].contains(value))
        || !matches!(options.neural_target_weight, 2.0 | 4.0)
        || !matches!(options.neural_structural_penalty, 0.0 | 0.1)
    {
        return Err(DopeError::Data(
            "invalid exhaustive gold matrix options".into(),
        ));
    }
    let cohort: CohortPlan = read_json(options.cohort_path)?;
    let phase = match cohort.kind.as_str() {
        "training-gold" => "training_gold",
        "validation-select" => "validation_select",
        "validation-cert" if routed_cert.is_some() => "validation_cert",
        _ => {
            return Err(DopeError::Data(
                "gold matrix cohort is not authorized for this command".into(),
            ));
        }
    };
    let frozen_candidates = empirical_backends()
        .into_iter()
        .map(|value| value.id)
        .collect::<BTreeSet<_>>();
    let frozen_auditors = auditor_specs()
        .into_iter()
        .map(|value| value.id)
        .collect::<BTreeSet<_>>();
    if options
        .candidate_ids
        .iter()
        .any(|value| !frozen_candidates.contains(value))
        || options
            .auditor_ids
            .iter()
            .any(|value| !frozen_auditors.contains(value))
    {
        return Err(DopeError::Data(
            "gold matrix contains an unfrozen candidate or auditor".into(),
        ));
    }
    let cell_dir = options.out.join("cells");
    let cache_dir = options.out.join("cache");
    let block_dir = options.out.join("blocks");
    fs::create_dir_all(&cell_dir).map_err(|error| io_error(&cell_dir, error))?;
    fs::create_dir_all(&block_dir).map_err(|error| io_error(&block_dir, error))?;
    let selected = cohort
        .records
        .iter()
        .enumerate()
        .filter(|(index, _)| index % options.shard_count == options.shard_index)
        .map(|(_, record)| record)
        .collect::<Vec<_>>();
    let mut evidence = Vec::new();
    let mut failures = Vec::new();
    let cells_per_record = options.candidate_ids.len()
        * options.auditor_ids.len()
        * options.size_multipliers.len()
        * options.generation_seeds.len()
        * options.auditor_seeds.len();
    // Every matrix, including the 78-cell discovery matrix, is represented by
    // one durable record block. This gives failures and small campaigns the
    // same restart and provenance guarantees as exhaustive confirmation.
    let use_record_blocks = true;
    let mut succeeded_total = 0usize;
    for record in &selected {
        let matrix_identity = gold_record_matrix_identity(&GoldMatrixIdentity {
            phase,
            dataset_id: &record.dataset_id,
            candidate_ids: options.candidate_ids,
            auditor_ids: options.auditor_ids,
            size_multipliers: options.size_multipliers,
            generation_seeds: options.generation_seeds,
            auditor_seeds: options.auditor_seeds,
            primary_only: options.primary_only,
            neural_target_weight: options.neural_target_weight,
            neural_structural_penalty: options.neural_structural_penalty,
            routing_sha256: routed_cert.map(|context| context.bundle_sha256),
        })?;
        let block_path = block_dir.join(format!("{matrix_identity}.json"));
        if use_record_blocks && block_path.is_file() {
            let block: Option<GoldRecordBlock> = match read_json(&block_path) {
                Ok(block) => Some(block),
                Err(error) => {
                    let quarantine = options.out.join("operational-failures/corrupt-blocks");
                    fs::create_dir_all(&quarantine)
                        .map_err(|create_error| io_error(&quarantine, create_error))?;
                    let quarantined = quarantine.join(format!(
                        "{matrix_identity}-{}-{}.json",
                        unix_now(),
                        std::process::id()
                    ));
                    fs::rename(&block_path, &quarantined)
                        .map_err(|rename_error| io_error(&quarantined, rename_error))?;
                    write_canonical(
                        &quarantined.with_extension("error.json"),
                        &serde_json::json!({
                            "format": "dope-corrupt-gold-block-incident",
                            "version": 1,
                            "matrix_identity": matrix_identity,
                            "quarantined_path": quarantined,
                            "error": error.to_string()
                        }),
                    )?;
                    None
                }
            };
            if let Some(block) = block {
                if block.format != "dope-gold-record-block"
                    || block.version != GOLD_RECORD_BLOCK_VERSION
                    || block.dataset_id != record.dataset_id
                    || block.matrix_identity != matrix_identity
                    || block.evidence.len() + block.failures.len() != cells_per_record
                    || match (routed_cert, &block.routing) {
                        (None, None) => block.evidence.iter().any(|cell| cell.routed),
                        (Some(context), Some(routing)) => {
                            routing.router_bundle_sha256 != context.bundle_sha256
                                || routing.selected_candidates.is_empty()
                                || block.evidence.iter().any(|cell| {
                                    cell.routed
                                        != routing.selected_candidates.contains(&cell.candidate_id)
                                })
                        }
                        _ => true,
                    }
                {
                    return Err(DopeError::Data(format!(
                        "gold record block identity mismatch at {}",
                        block_path.display()
                    )));
                }
                succeeded_total += block.evidence.len();
                if !options.blocks_only {
                    evidence.extend(block.evidence);
                }
                failures.extend(block.failures);
                continue;
            }
        }
        let evidence_start = evidence.len();
        let failures_start = failures.len();
        let routing = if let Some(context) = routed_cert {
            let table = Table::read_dataset_dir(&record.dataset_path, table_task(&record.task)?)?;
            let sketch_started = Instant::now();
            let sketch = DatasetSketch::from_train(&table, table_task(&record.task)?);
            let sketch_ms = sketch_started.elapsed().as_secs_f64() * 1_000.0;
            let inference_started = Instant::now();
            let predictions = context.router.predict(&sketch)?;
            let inference_ms = inference_started.elapsed().as_secs_f64() * 1_000.0;
            let mut ranked = predictions.clone();
            ranked.sort_by(|left, right| {
                right
                    .retention_lower
                    .total_cmp(&left.retention_lower)
                    .then_with(|| left.expected_regret.total_cmp(&right.expected_regret))
                    .then_with(|| left.candidate_id.cmp(&right.candidate_id))
            });
            let ranked_candidates = ranked
                .iter()
                .map(|prediction| prediction.candidate_id.clone())
                .collect::<Vec<_>>();
            let top_four_candidates = ranked_candidates.iter().take(4).cloned().collect();
            let selected_candidates =
                SelectionPolicy::default().select(&predictions, "independent_quantile");
            Some(GoldRoutingDecision {
                format: "dope-gold-routing-decision".into(),
                version: 1,
                router_bundle_sha256: context.bundle_sha256.into(),
                ranked_candidates,
                top_four_candidates,
                selected_candidates,
                prediction_sha256: hashes(&canonical_json(&predictions)?).sha256,
                sketch_ms,
                inference_ms,
            })
        } else {
            None
        };
        for candidate_id in options.candidate_ids {
            for &size_multiplier in options.size_multipliers {
                for &generation_seed in options.generation_seeds {
                    let mut pending = Vec::new();
                    for auditor_id in options.auditor_ids {
                        for &auditor_seed in options.auditor_seeds {
                            let identity = hashes(&canonical_json(&serde_json::json!({
                                "phase": phase,
                                "dataset": record.dataset_id,
                                "candidate": candidate_id,
                                "auditor": auditor_id,
                                "size_multiplier": size_multiplier,
                                "generation_seed": generation_seed,
                                "auditor_seed": auditor_seed,
                                "primary_only": options.primary_only,
                                "neural_target_weight": options.neural_target_weight,
                                "neural_structural_penalty": options.neural_structural_penalty
                            }))?)
                            .blake3;
                            let path = cell_dir.join(format!("{identity}.json"));
                            if !use_record_blocks && path.is_file() {
                                evidence.push(read_json::<JobEvidence>(&path)?);
                                continue;
                            }
                            pending.push((auditor_id, auditor_seed, path));
                        }
                    }
                    if pending.is_empty() {
                        continue;
                    }
                    let preparation_options = GoldCellOptions {
                        dataset_dir: &record.dataset_path,
                        task: table_task(&record.task)?,
                        candidate_id,
                        auditor_id: pending[0].0,
                        lineage_group_id: &record.lineage_group_id,
                        structural_profile: &record.structural_profile,
                        phase,
                        size_multiplier,
                        generation_seed,
                        auditor_seed: pending[0].1,
                        routed: routing.as_ref().is_some_and(|routing| {
                            routing.selected_candidates.contains(candidate_id)
                        }),
                        ancillary: !options.primary_only,
                        cache_dir: Some(&cache_dir),
                        neural_target_weight: options.neural_target_weight,
                        neural_structural_penalty: options.neural_structural_penalty,
                    };
                    match prepare_gold_cell(&preparation_options) {
                        Ok(prepared) => {
                            let (native_pending, gpu_pending): (Vec<_>, Vec<_>) = pending
                                .into_iter()
                                .partition(|(auditor_id, _, _)| !auditor_id.starts_with("gpu_"));
                            let native_results = native_pending
                                .par_iter()
                                .map(|(auditor_id, auditor_seed, path)| {
                                    let cell_options = GoldCellOptions {
                                        auditor_id,
                                        auditor_seed: *auditor_seed,
                                        ..preparation_options
                                    };
                                    (
                                        *auditor_id,
                                        *auditor_seed,
                                        path,
                                        evaluate_prepared_gold_cell(&cell_options, &prepared),
                                    )
                                })
                                .collect::<Vec<_>>();
                            for (auditor_id, auditor_seed, path, result) in native_results {
                                match result {
                                    Ok(cell) => {
                                        if !use_record_blocks {
                                            write_canonical(path, &cell)?;
                                        }
                                        evidence.push(cell);
                                    }
                                    Err(error) => failures.push(BaselineFailure {
                                        dataset_id: record.dataset_id.clone(),
                                        candidate_id: candidate_id.clone(),
                                        auditor_id: auditor_id.clone(),
                                        size_multiplier: Some(size_multiplier),
                                        generation_seed: Some(generation_seed),
                                        auditor_seed: Some(auditor_seed),
                                        error: error.to_string(),
                                    }),
                                }
                            }
                            // tch's process-global seed must not race inside a worker.
                            for (auditor_id, auditor_seed, path) in gpu_pending {
                                let cell_options = GoldCellOptions {
                                    auditor_id,
                                    auditor_seed,
                                    ..preparation_options
                                };
                                match evaluate_prepared_gold_cell(&cell_options, &prepared) {
                                    Ok(cell) => {
                                        if !use_record_blocks {
                                            write_canonical(&path, &cell)?;
                                        }
                                        evidence.push(cell);
                                    }
                                    Err(error) => failures.push(BaselineFailure {
                                        dataset_id: record.dataset_id.clone(),
                                        candidate_id: candidate_id.clone(),
                                        auditor_id: auditor_id.clone(),
                                        size_multiplier: Some(size_multiplier),
                                        generation_seed: Some(generation_seed),
                                        auditor_seed: Some(auditor_seed),
                                        error: error.to_string(),
                                    }),
                                }
                            }
                        }
                        Err(error) => {
                            for (auditor_id, auditor_seed, _) in pending {
                                failures.push(BaselineFailure {
                                    dataset_id: record.dataset_id.clone(),
                                    candidate_id: candidate_id.clone(),
                                    auditor_id: auditor_id.clone(),
                                    size_multiplier: Some(size_multiplier),
                                    generation_seed: Some(generation_seed),
                                    auditor_seed: Some(auditor_seed),
                                    error: error.to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }
        if use_record_blocks {
            write_canonical(
                &block_path,
                &GoldRecordBlock {
                    format: "dope-gold-record-block".into(),
                    version: GOLD_RECORD_BLOCK_VERSION,
                    dataset_id: record.dataset_id.clone(),
                    matrix_identity,
                    routing,
                    evidence: evidence[evidence_start..].to_vec(),
                    failures: failures[failures_start..].to_vec(),
                    neural_target_weight: options.neural_target_weight,
                    neural_structural_penalty: options.neural_structural_penalty,
                },
            )?;
            succeeded_total += evidence.len() - evidence_start;
            if options.blocks_only {
                evidence.truncate(evidence_start);
            }
        }
    }
    evidence.sort_by(|left, right| {
        left.lineage_group_id
            .cmp(&right.lineage_group_id)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
            .then_with(|| left.auditor_id.cmp(&right.auditor_id))
            .then_with(|| left.size_multiplier.cmp(&right.size_multiplier))
            .then_with(|| left.generation_seed.cmp(&right.generation_seed))
            .then_with(|| left.auditor_seed.cmp(&right.auditor_seed))
    });
    if !options.blocks_only {
        let metrics_path = options
            .out
            .join(format!("metrics-shard-{:04}.jsonl", options.shard_index));
        let mut metrics =
            File::create(&metrics_path).map_err(|error| io_error(&metrics_path, error))?;
        for cell in &evidence {
            metrics
                .write_all(&canonical_json(cell)?)
                .and_then(|_| metrics.write_all(b"\n"))
                .map_err(|error| io_error(&metrics_path, error))?;
        }
    } else {
        write_canonical(
            &options.out.join(format!(
                "block-manifest-shard-{:04}.json",
                options.shard_index
            )),
            &serde_json::json!({
                "format": "dope-gold-block-shard-manifest",
                "version": GOLD_RECORD_BLOCK_VERSION,
                "shard_index": options.shard_index,
                "shard_count": options.shard_count,
                "cohort_records": selected.len(),
                "succeeded": succeeded_total,
                "failed": failures.len()
            }),
        )?;
    }
    let attempted = selected.len()
        * options.candidate_ids.len()
        * options.auditor_ids.len()
        * options.size_multipliers.len()
        * options.generation_seeds.len()
        * options.auditor_seeds.len();
    let summary = BaselineSummary {
        format: "dope-gold-matrix-summary".into(),
        version: 1,
        cohort_records: selected.len(),
        attempted,
        succeeded: if use_record_blocks {
            succeeded_total
        } else {
            evidence.len()
        },
        failed: failures.len(),
        failures,
    };
    write_canonical(
        &options
            .out
            .join(format!("summary-shard-{:04}.json", options.shard_index)),
        &summary,
    )?;
    Ok(summary)
}

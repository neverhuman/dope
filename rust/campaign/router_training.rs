
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RouterBenchmarkReport {
    pub format: String,
    pub version: u8,
    pub rows: usize,
    pub features: usize,
    pub repeats: usize,
    pub sketch_p95_ms: f64,
    pub router_inference_p95_ms: Option<f64>,
    pub router_bundle_bytes: Option<u64>,
    pub deterministic_inference: Option<bool>,
}

pub fn benchmark_router(
    router_bundle: Option<&Path>,
    rows: usize,
    features: usize,
    repeats: usize,
) -> Result<RouterBenchmarkReport> {
    if rows == 0 || !(1..=2_000).contains(&features) || repeats < 2 {
        return Err(DopeError::Data(
            "router benchmark requires rows > 0, 1..=2000 features, and at least two repeats"
                .into(),
        ));
    }
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        (value >> 40) as f32 / (1u32 << 24) as f32
    };
    let values = (0..rows * features).map(|_| next()).collect::<Vec<_>>();
    let target = (0..rows)
        .map(|row| {
            (0..features.min(8))
                .map(|feature| values[row * features + feature])
                .sum::<f32>()
                / features.min(8) as f32
        })
        .collect::<Vec<_>>();
    let table = Table::from_arrays(&values, &target, rows, features, Task::Regression)?;
    let mut sketches = Vec::with_capacity(repeats);
    let mut sketch_times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        sketches.push(DatasetSketch::from_train(&table, Task::Regression));
        sketch_times.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    let sketch_p95_ms = float_percentile(&mut sketch_times, 0.95).unwrap_or(f64::INFINITY);
    let (router_inference_p95_ms, router_bundle_bytes, deterministic_inference) =
        if let Some(path) = router_bundle {
            let bundle: RouterBundle = read_json(path)?;
            let router = QuantizedRouter::new(bundle)?;
            let mut prediction_hashes = BTreeSet::new();
            let mut inference_times = Vec::with_capacity(repeats);
            for sketch in &sketches {
                let started = Instant::now();
                let prediction = router.predict(sketch)?;
                inference_times.push(started.elapsed().as_secs_f64() * 1_000.0);
                prediction_hashes.insert(hashes(&canonical_json(&prediction)?).sha256);
            }
            (
                float_percentile(&mut inference_times, 0.95),
                Some(
                    fs::metadata(path)
                        .map_err(|error| io_error(path, error))?
                        .len(),
                ),
                Some(prediction_hashes.len() == 1),
            )
        } else {
            (None, None, None)
        };
    Ok(RouterBenchmarkReport {
        format: "dope-router-benchmark".into(),
        version: 1,
        rows,
        features,
        repeats,
        sketch_p95_ms,
        router_inference_p95_ms,
        router_bundle_bytes,
        deterministic_inference,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct CachedSketch {
    format: String,
    version: u8,
    lineage_group_id: String,
    structural_profile: String,
    sketch: DatasetSketch,
    compute_ms: f64,
}

#[derive(Clone, Debug, Default)]
struct RouterCellAggregate {
    seen_replicates: u128,
    count: usize,
    retention_sum: f64,
    runtime_ms_sum: f64,
    peak_memory_bytes_sum: f64,
    artifact_bytes_sum: f64,
}

fn router_bounded_retention(cell: &JobEvidence) -> f64 {
    let improvement = cell.null_loss - cell.trtr_loss;
    let informative = improvement >= 0.01 * cell.null_loss.abs();
    if informative && improvement.abs() > f64::EPSILON {
        // Router utility is bounded: exceeding TRTR is a full success, while
        // negative retention has no extra routing value. KPI aggregation
        // retains the required unclamped statistic.
        ((cell.null_loss - cell.tstr_loss) / improvement).clamp(0.0, 1.0)
    } else {
        f64::from(cell.tstr_loss <= cell.trtr_loss + 0.01 * cell.null_loss)
    }
}

fn router_replicate_slot(cell: &JobEvidence, auditor_ids: &[String]) -> Result<usize> {
    let auditor = auditor_ids
        .iter()
        .position(|id| id == &cell.auditor_id)
        .ok_or_else(|| DopeError::Data(format!("unknown router auditor {}", cell.auditor_id)))?;
    let size = [1, 4]
        .iter()
        .position(|&value| value == cell.size_multiplier)
        .ok_or_else(|| DopeError::Data("router metric has a non-gold size multiplier".into()))?;
    let generation = [1829, 99238, 196647]
        .iter()
        .position(|&value| value == cell.generation_seed)
        .ok_or_else(|| DopeError::Data("router metric has an unfrozen generation seed".into()))?;
    let auditor_seed = [57721, 161803, 271828]
        .iter()
        .position(|&value| value == cell.auditor_seed)
        .ok_or_else(|| DopeError::Data("router metric has an unfrozen auditor seed".into()))?;
    Ok((((auditor * 2) + size) * 3 + generation) * 3 + auditor_seed)
}

pub fn train_router_from_metrics(
    cohort_paths: &[PathBuf],
    metric_paths: &[PathBuf],
    sketch_cache: Option<&Path>,
    bundle_out: &Path,
    report_out: &Path,
    evidence_out: &Path,
) -> Result<(RouterTrainingReport, RouterEvidence)> {
    if cohort_paths.is_empty() || metric_paths.is_empty() {
        return Err(DopeError::Data(
            "router cohort and metrics inputs must be nonempty".into(),
        ));
    }
    let cohorts = cohort_paths
        .iter()
        .map(|path| read_json::<CohortPlan>(path))
        .collect::<Result<Vec<_>>>()?;
    if cohorts
        .iter()
        .any(|cohort| !matches!(cohort.kind.as_str(), "training-gold" | "validation-select"))
    {
        return Err(DopeError::Data(
            "router training accepts only training-gold and validation-select cohorts".into(),
        ));
    }
    let training_lineage_ids = cohorts
        .iter()
        .filter(|cohort| cohort.kind == "training-gold")
        .flat_map(|cohort| &cohort.records)
        .map(|record| record.lineage_group_id.clone())
        .collect::<BTreeSet<_>>();
    let validation_lineage_ids = cohorts
        .iter()
        .filter(|cohort| cohort.kind == "validation-select")
        .flat_map(|cohort| &cohort.records)
        .map(|record| record.lineage_group_id.clone())
        .collect::<BTreeSet<_>>();
    if !training_lineage_ids.is_disjoint(&validation_lineage_ids) {
        return Err(DopeError::Data(
            "router training and validation-select lineages overlap".into(),
        ));
    }
    let records = cohorts
        .iter()
        .flat_map(|cohort| &cohort.records)
        .map(|record| {
            (
                (
                    record.lineage_group_id.as_str(),
                    record.structural_profile.as_str(),
                ),
                record,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let records_by_dataset = cohorts
        .iter()
        .flat_map(|cohort| &cohort.records)
        .map(|record| (record.dataset_id.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    let candidate_ids = empirical_backends()
        .into_iter()
        .map(|backend| backend.id)
        .collect::<Vec<_>>();
    let candidate_set = candidate_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let auditor_ids = auditor_specs()
        .into_iter()
        .map(|auditor| auditor.id)
        .collect::<Vec<_>>();
    let mut grouped = BTreeMap::<(String, String, String), RouterCellAggregate>::new();
    let mut sketch_jobs = BTreeMap::new();
    let mut input_cells = 0usize;
    let mut expanded_metric_paths = Vec::new();
    for path in metric_paths {
        if path.is_dir() {
            let mut entries = fs::read_dir(path)
                .map_err(|error| io_error(path, error))?
                .map(|entry| {
                    entry
                        .map(|entry| entry.path())
                        .map_err(|error| io_error(path, error))
                })
                .collect::<Result<Vec<_>>>()?;
            entries.retain(|entry| {
                entry
                    .extension()
                    .is_some_and(|extension| extension == "json")
            });
            entries.sort();
            expanded_metric_paths.extend(entries);
        } else {
            expanded_metric_paths.push(path.clone());
        }
    }
    if expanded_metric_paths.is_empty() {
        return Err(DopeError::Data(
            "router metrics inputs contain no evidence files".into(),
        ));
    }
    let mut consume_cell = |cell: JobEvidence| -> Result<()> {
        let record = records
            .get(&(
                cell.lineage_group_id.as_str(),
                cell.structural_profile.as_str(),
            ))
            .ok_or_else(|| {
                DopeError::Data(format!(
                    "router metric outcome {}/{} is absent from its cohort",
                    cell.lineage_group_id, cell.structural_profile
                ))
            })?;
        if !candidate_set.contains(cell.candidate_id.as_str())
            || cell.task != record.task
            || cell.lineage_group_id != record.lineage_group_id
            || cell.structural_profile != record.structural_profile
        {
            return Err(DopeError::Data(
                "router metric does not match its frozen cohort record".into(),
            ));
        }
        let slot = router_replicate_slot(&cell, &auditor_ids)?;
        let bit = 1u128 << slot;
        let aggregate = grouped
            .entry((
                cell.lineage_group_id.clone(),
                cell.structural_profile.clone(),
                cell.candidate_id.clone(),
            ))
            .or_default();
        if aggregate.seen_replicates & bit != 0 {
            return Err(DopeError::Data(format!(
                "duplicate router metric replicate for {}/{}/{}",
                cell.lineage_group_id, cell.structural_profile, cell.candidate_id
            )));
        }
        aggregate.seen_replicates |= bit;
        aggregate.count += 1;
        aggregate.retention_sum += router_bounded_retention(&cell);
        aggregate.runtime_ms_sum += cell.runtime_ms as f64;
        aggregate.peak_memory_bytes_sum += cell.peak_memory_bytes as f64;
        aggregate.artifact_bytes_sum += cell.artifact_bytes as f64;
        sketch_jobs.insert(
            (
                cell.lineage_group_id.clone(),
                cell.structural_profile.clone(),
            ),
            (*record).clone(),
        );
        input_cells += 1;
        Ok(())
    };
    let gold_sizes = [1usize, 4];
    let gold_generation_seeds = [1829u64, 99238, 196647];
    let gold_auditor_seeds = [57721u64, 161803, 271828];
    for path in &expanded_metric_paths {
        let file = File::open(path).map_err(|error| io_error(path, error))?;
        let mut lines = std::io::BufReader::new(file).lines();
        let Some(first_line) = lines.next() else {
            continue;
        };
        let first_line = first_line.map_err(|error| io_error(path, error))?;
        if first_line.trim().is_empty() {
            continue;
        }
        let first_value: Value = serde_json::from_str(&first_line)?;
        if first_value.get("format").and_then(Value::as_str) == Some("dope-gold-record-block") {
            let block: GoldRecordBlock = serde_json::from_value(first_value)?;
            if lines.next().is_some() {
                return Err(DopeError::Data(format!(
                    "gold record block contains trailing data at {}",
                    path.display()
                )));
            }
            let record = records_by_dataset
                .get(block.dataset_id.as_str())
                .ok_or_else(|| {
                    DopeError::Data(format!(
                        "gold record block {} is absent from router cohorts",
                        block.dataset_id
                    ))
                })?;
            let phase = match record.partition.as_str() {
                "training_gold" => "training_gold",
                "validation_select" => "validation_select",
                _ => {
                    return Err(DopeError::Data(
                        "router training cannot consume certification evidence".into(),
                    ));
                }
            };
            let matrix_identity = gold_record_matrix_identity(&GoldMatrixIdentity {
                phase,
                dataset_id: &record.dataset_id,
                candidate_ids: &candidate_ids,
                auditor_ids: &auditor_ids,
                size_multipliers: &gold_sizes,
                generation_seeds: &gold_generation_seeds,
                auditor_seeds: &gold_auditor_seeds,
                primary_only: false,
                neural_target_weight: 2.0,
                neural_structural_penalty: 0.0,
                routing_sha256: None,
            })?;
            if path.file_stem().and_then(|stem| stem.to_str()) != Some(&matrix_identity) {
                return Err(DopeError::Data(format!(
                    "router gold block filename differs from its matrix identity at {}",
                    path.display()
                )));
            }
            validate_gold_record_block(
                &block,
                record,
                phase,
                &matrix_identity,
                &candidate_ids,
                &auditor_ids,
                None,
            )?;
            for cell in block.evidence {
                consume_cell(cell)?;
            }
            continue;
        }
        consume_cell(serde_json::from_value(first_value)?)?;
        for line in lines {
            let line = line.map_err(|error| io_error(path, error))?;
            if !line.trim().is_empty() {
                consume_cell(serde_json::from_str(&line)?)?;
            }
        }
    }
    if let Some(cache) = sketch_cache {
        fs::create_dir_all(cache).map_err(|error| io_error(cache, error))?;
    }
    let sketch_results = sketch_jobs
        .into_par_iter()
        .map(|((lineage, profile), record)| {
            let cache_path = sketch_cache.map(|cache| {
                cache.join(format!(
                    "{}.json",
                    hashes(format!("{}\0{}\0{}", record.dataset_id, lineage, profile).as_bytes())
                        .blake3
                ))
            });
            if let Some(path) = cache_path.as_ref().filter(|path| path.is_file()) {
                let cached: CachedSketch = read_json(path)?;
                if cached.format != "dope-router-sketch-cache"
                    || cached.version != 1
                    || cached.lineage_group_id != lineage
                    || cached.structural_profile != profile
                    || cached.sketch.version != DATASET_SKETCH_VERSION
                    || !cached.compute_ms.is_finite()
                    || cached.compute_ms < 0.0
                {
                    return Err(DopeError::Data(format!(
                        "router sketch cache identity mismatch at {}",
                        path.display()
                    )));
                }
                return Ok(((lineage, profile), cached.sketch, cached.compute_ms));
            }
            let table = Table::read_dataset_dir(&record.dataset_path, table_task(&record.task)?)?;
            let started = Instant::now();
            let sketch = DatasetSketch::from_train(&table, table_task(&record.task)?);
            let compute_ms = started.elapsed().as_secs_f64() * 1_000.0;
            if let Some(path) = cache_path {
                write_canonical(
                    &path,
                    &CachedSketch {
                        format: "dope-router-sketch-cache".into(),
                        version: 1,
                        lineage_group_id: lineage.clone(),
                        structural_profile: profile.clone(),
                        sketch: sketch.clone(),
                        compute_ms,
                    },
                )?;
            }
            Ok(((lineage, profile), sketch, compute_ms))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut sketches = BTreeMap::<(String, String), DatasetSketch>::new();
    let mut sketch_times = Vec::<f64>::new();
    for (key, sketch, compute_ms) in sketch_results {
        sketches.insert(key, sketch);
        sketch_times.push(compute_ms);
    }
    let replicates_per_candidate = grouped.values().map(|value| value.count).max().unwrap_or(0);
    let mut outcome_counts = BTreeMap::<(String, String), BTreeMap<String, usize>>::new();
    for ((lineage, profile, candidate), values) in &grouped {
        outcome_counts
            .entry((lineage.clone(), profile.clone()))
            .or_default()
            .insert(candidate.clone(), values.count);
    }
    let complete_outcomes = outcome_counts
        .iter()
        .filter(|(_, counts)| {
            counts.len() == candidate_ids.len()
                && counts.keys().map(String::as_str).collect::<BTreeSet<_>>() == candidate_set
                && counts
                    .values()
                    .all(|&count| count == replicates_per_candidate)
        })
        .map(|(outcome, _)| outcome.clone())
        .collect::<BTreeSet<_>>();
    let excluded_incomplete_outcomes = outcome_counts.len() - complete_outcomes.len();
    if complete_outcomes.is_empty() {
        return Err(DopeError::Data(
            "router metrics contain no complete lineage/profile outcomes".into(),
        ));
    }
    let mut labels = grouped
        .into_iter()
        .filter(|((lineage, profile, _), _)| {
            complete_outcomes.contains(&(lineage.clone(), profile.clone()))
        })
        .map(|((lineage, profile, candidate), values)| {
            let count = values.count as f64;
            let sketch_key = (lineage.clone(), profile.clone());
            RouterLabel {
                lineage_group_id: lineage,
                structural_profile: profile,
                candidate_id: candidate,
                sketch: sketches[&sketch_key].clone(),
                retention: (values.retention_sum / count) as f32,
                runtime_ms: (values.runtime_ms_sum / count) as f32,
                peak_memory_bytes: (values.peak_memory_bytes_sum / count) as f32,
                artifact_bytes: (values.artifact_bytes_sum / count) as f32,
                failed: false,
            }
        })
        .collect::<Vec<_>>();
    labels.sort_by(|left, right| {
        left.lineage_group_id
            .cmp(&right.lineage_group_id)
            .then_with(|| left.structural_profile.cmp(&right.structural_profile))
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    let training_evidence_sha256 = hashes(&canonical_json(&serde_json::json!({
        "labels": labels,
        "input_cells": input_cells,
        "replicates_per_candidate": replicates_per_candidate,
        "excluded_incomplete_outcomes": excluded_incomplete_outcomes,
        "validation_lineages": validation_lineage_ids
    }))?)
    .sha256;
    let label_lineages = labels
        .iter()
        .map(|label| label.lineage_group_id.clone())
        .collect::<BTreeSet<_>>();
    let effective_validation_lineages = validation_lineage_ids
        .intersection(&label_lineages)
        .cloned()
        .collect::<BTreeSet<_>>();
    let explicit_validation = (!training_lineage_ids.is_empty()
        && !validation_lineage_ids.is_empty())
    .then_some(&effective_validation_lineages);
    let (bundle, mut report) = train_distilled_router(
        &labels,
        &candidate_ids,
        &training_evidence_sha256,
        explicit_validation,
    )?;
    report.input_cells = input_cells;
    report.replicates_per_candidate = replicates_per_candidate;
    report.complete_lineage_profile_outcomes = complete_outcomes.len();
    report.excluded_incomplete_outcomes = excluded_incomplete_outcomes;
    write_canonical(bundle_out, &bundle)?;
    write_canonical(report_out, &report)?;
    let router = QuantizedRouter::new(bundle.clone())?;
    let mut inference_times = Vec::new();
    for sketch in sketches.values() {
        for _ in 0..5 {
            let started = Instant::now();
            let _ = router.predict(sketch)?;
            inference_times.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
    }
    let p95 = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values
            .get(((values.len().saturating_sub(1)) as f64 * 0.95).ceil() as usize)
            .copied()
            .unwrap_or(f64::INFINITY)
    };
    let evidence = RouterEvidence {
        format: "dope-router-evidence".into(),
        version: 1,
        top_four_oracle_recall: report.student_top_four_oracle_recall,
        maximum_profile_regret_upper: report.student_maximum_profile_regret_upper,
        beats_random: report.student_mean_regret < report.random_mean_regret,
        beats_best_fixed: report.student_mean_regret < report.best_fixed_mean_regret,
        beats_ga2m: report.student_mean_regret < report.ga2m_mean_regret,
        paired_hypervolume_improvement: report.paired_hypervolume_improvement,
        paired_hypervolume_ci_lower: report.paired_hypervolume_ci_lower,
        student_max_abs_difference: report.student_max_abs_difference,
        top_choice_agreement: report.top_choice_agreement,
        bundle_bytes: report.bundle_bytes,
        inference_p95_ms: p95(&mut inference_times),
        sketch_p95_ms: p95(&mut sketch_times),
        training_evidence_sha256,
        best_fixed_candidate_id: report.best_fixed_candidate_id.clone(),
        student_mean_regret: report.student_mean_regret,
        random_mean_regret: report.random_mean_regret,
        best_fixed_mean_regret: report.best_fixed_mean_regret,
        ga2m_mean_regret: report.ga2m_mean_regret,
    };
    write_canonical(evidence_out, &evidence)?;
    Ok((report, evidence))
}

fn table_task(task: &str) -> Result<Task> {
    match task {
        "binary" => Ok(Task::Binary),
        "regression" => Ok(Task::Regression),
        _ => Err(DopeError::Data(format!("unknown cohort task {task}"))),
    }
}

pub fn freeze_router(
    state_dir: &Path,
    bundle_path: &Path,
    evidence_path: &Path,
    allow_failed_frontier: bool,
) -> Result<CampaignState> {
    let mut state = load_state(state_dir)?;
    if state.phase != CampaignPhase::Frozen {
        return Err(DopeError::Data("router may be frozen exactly once".into()));
    }
    let bundle: RouterBundle = read_json(bundle_path)?;
    bundle.validate()?;
    if !bundle.trained {
        return Err(DopeError::Data("untrained router cannot be frozen".into()));
    }
    let evidence: RouterEvidence = read_json(evidence_path)?;
    if evidence.format != "dope-router-evidence" || evidence.version != 1 {
        return Err(DopeError::Data("invalid router evidence format".into()));
    }
    let failed = evidence.failed_gates();
    if !failed.is_empty() && !allow_failed_frontier {
        return Err(DopeError::Data(format!(
            "router freeze gates failed (use the explicit measured failed-frontier path to continue certification without promotion): {}",
            failed.join(",")
        )));
    }
    let bundle_hash = copy_file(bundle_path, &state_dir.join("router.bundle"))?;
    let evidence_hash = copy_file(evidence_path, &state_dir.join("router-evidence.json"))?;
    if bundle.training_evidence_sha256.as_deref() != Some(&evidence.training_evidence_sha256) {
        return Err(DopeError::Data(
            "router bundle and training evidence hashes differ".into(),
        ));
    }
    state.phase = CampaignPhase::RouterFrozen;
    state.router_bundle = Some(bundle_hash);
    state.router_evidence = Some(evidence_hash);
    state.validate()?;
    write_canonical(&state_path(state_dir), &state)?;
    CampaignLedger::open(&ledger_path(state_dir))?.put_metadata("router_evidence", &evidence)?;
    Ok(state)
}

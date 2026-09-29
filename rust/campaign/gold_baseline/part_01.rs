
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
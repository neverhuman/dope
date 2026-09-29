

fn cached_synthetic_ancillary(
    options: &GoldCellOptions<'_>,
    artifact_sha256: &str,
    train: &Table,
    test: &Table,
    synthetic: &Table,
) -> Result<(CachedSyntheticAncillary, CacheStatus)> {
    let identity = canonical_json(&serde_json::json!({
        "format": "dope-gold-synthetic-ancillary-cache-key",
        "version": 1,
        "lineage_group_id": options.lineage_group_id,
        "candidate_id": options.candidate_id,
        "size_multiplier": options.size_multiplier,
        "generation_seed": options.generation_seed,
        "artifact_sha256": artifact_sha256
    }))?;
    let path = options
        .cache_dir
        .map(|cache| cache_path(cache, "synthetic-ancillary", &identity, "json"));
    if let Some(path) = path.as_ref().filter(|path| path.is_file()) {
        let cached: CachedSyntheticAncillary =
            serde_json::from_slice(&fs::read(path).map_err(|error| io_error(path, error))?)?;
        if cached.format != "dope-gold-synthetic-ancillary-cache"
            || cached.version != 1
            || cached.lineage_group_id != options.lineage_group_id
            || cached.candidate_id != options.candidate_id
            || cached.size_multiplier != options.size_multiplier
            || cached.generation_seed != options.generation_seed
            || cached.artifact_sha256 != artifact_sha256
            || [
                cached.driver_agreement,
                cached.joint_fidelity,
                cached.query_p95_normalized_error,
                cached.type_i_error,
                cached.membership_auc,
                cached.attribute_inference_advantage,
            ]
            .into_iter()
            .any(|value| !value.is_finite())
        {
            return Err(DopeError::Data(format!(
                "synthetic ancillary cache identity mismatch at {}",
                path.display()
            )));
        }
        return Ok((cached, CacheStatus::Hit));
    }
    let cached = CachedSyntheticAncillary {
        format: "dope-gold-synthetic-ancillary-cache".into(),
        version: 1,
        lineage_group_id: options.lineage_group_id.into(),
        candidate_id: options.candidate_id.into(),
        size_multiplier: options.size_multiplier,
        generation_seed: options.generation_seed,
        artifact_sha256: artifact_sha256.into(),
        driver_agreement: driver_agreement(train, synthetic),
        joint_fidelity: joint_fidelity(train, synthetic),
        query_p95_normalized_error: query_p95_normalized_error(train, synthetic),
        type_i_error: type_i_error(train, synthetic),
        membership_auc: membership_auc(synthetic, train, test),
        attribute_inference_advantage: attribute_inference_advantage(synthetic, train, test),
        exact_copies: exact_copy_count(train, synthetic),
        near_copies: near_copy_count(train, synthetic),
    };
    let status = if path.is_some() {
        CacheStatus::Miss
    } else {
        CacheStatus::Disabled
    };
    if let Some(path) = path {
        atomic_cache_write(&path, &canonical_json(&cached)?)?;
    }
    Ok((cached, status))
}

#[cfg(target_os = "linux")]
fn peak_memory_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("VmHWM:")
                    .and_then(|value| value.split_whitespace().next())
                    .and_then(|value| value.parse::<u64>().ok())
            })
        })
        .and_then(|kib| kib.checked_mul(1024))
        .unwrap_or(0)
}

#[cfg(not(target_os = "linux"))]
fn peak_memory_bytes() -> u64 {
    0
}

fn clear_gpu_peak_memory() {
    #[cfg(feature = "gpu-training")]
    crate::libtorch::clear_last_gpu_peak_memory_bytes();
}

fn take_gpu_peak_memory() -> Option<u64> {
    #[cfg(feature = "gpu-training")]
    {
        crate::libtorch::take_last_gpu_peak_memory_bytes()
    }
    #[cfg(not(feature = "gpu-training"))]
    {
        None
    }
}

fn maximum_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

pub struct PreparedGoldCell {
    lineage_group_id: String,
    candidate_id: String,
    task: Task,
    size_multiplier: usize,
    generation_seed: u64,
    train: Table,
    test: Table,
    synthetic: Table,
    artifact_bytes: u64,
    artifact_hashes: ContentHashes,
    lineage_leaks: usize,
    invalid_rows: usize,
    schema_violations: usize,
    nondeterministic_output: bool,
    fitting_time_ms: u64,
    sampling_time_ms: u64,
    artifact_cache_status: CacheStatus,
    peak_cpu_memory_bytes: u64,
    peak_gpu_memory_bytes: Option<u64>,
    preparation_runtime_ms: u64,
}

fn artifact_lineage_leaks(artifact: &[u8], options: &GoldCellOptions<'_>) -> usize {
    let mut probes = vec![
        options.lineage_group_id.as_bytes().to_vec(),
        options.dataset_dir.to_string_lossy().as_bytes().to_vec(),
    ];
    if let Some(name) = options.dataset_dir.file_name() {
        probes.push(name.to_string_lossy().as_bytes().to_vec());
    }
    probes
        .into_iter()
        .filter(|probe| probe.len() >= 4)
        .map(|probe| {
            artifact
                .windows(probe.len())
                .filter(|window| *window == probe)
                .count()
        })
        .sum()
}

fn marginal_contains(marginal: &Marginal, value: f32) -> bool {
    match marginal {
        Marginal::Constant { value: expected } => (value - expected).abs() <= 1e-5,
        Marginal::Bernoulli { .. } => (value - value.round()).abs() <= 1e-6,
        Marginal::Grid { values, .. } => values
            .iter()
            .any(|expected| (value - expected).abs() <= 1e-5),
        Marginal::QuantileSpline { values } => values
            .first()
            .zip(values.last())
            .is_some_and(|(minimum, maximum)| value >= *minimum - 1e-5 && value <= *maximum + 1e-5),
        Marginal::Beta { .. } => (0.0..=1.0).contains(&value),
        Marginal::Gaussian { .. } => value.is_finite(),
        Marginal::Histogram { edges, .. } => edges
            .first()
            .zip(edges.last())
            .is_some_and(|(minimum, maximum)| value >= *minimum - 1e-5 && value <= *maximum + 1e-5),
        Marginal::ZeroInflated { point, base, .. } => {
            (value - point).abs() <= 1e-5 || marginal_contains(base, value)
        }
    }
}

fn sample_diagnostics(sample: &[f32], kernel: &Kernel, rows: usize) -> (usize, usize) {
    let stride = kernel.features as usize + 1;
    let mut invalid_rows = 0usize;
    let mut schema_violations = 0usize;
    for row in sample.chunks_exact(stride).take(rows) {
        let mut invalid = false;
        for (feature, &value) in row.iter().take(kernel.features as usize).enumerate() {
            if value.is_infinite() {
                invalid = true;
                schema_violations += 1;
            } else if value.is_nan() {
                schema_violations += usize::from(kernel.schema[feature].missing_probability == 0.0);
            } else if !marginal_contains(&kernel.marginals[feature], value)
                || (kernel.schema[feature].kind == SchemaKind::Binary
                    && (value - value.round()).abs() > 1e-6)
            {
                schema_violations += 1;
            }
        }
        let target = row[stride - 1];
        if !target.is_finite()
            || !(0.0..=1.0).contains(&target)
            || (kernel.task == Task::Binary && (target - target.round()).abs() > 1e-6)
        {
            invalid = true;
            schema_violations += 1;
        }
        invalid_rows += usize::from(invalid);
    }
    (invalid_rows, schema_violations)
}

/// Fits and generates one immutable candidate × size × generation-seed table.
/// The guarded holdout is opened only after the bounded artifact and synthetic
/// table exist. Auditors can then score this preparation without regenerating it.
pub fn prepare_gold_cell(options: &GoldCellOptions<'_>) -> Result<PreparedGoldCell> {
    if !matches!(
        options.phase,
        "training_gold" | "validation_select" | "validation_cert"
    ) || !matches!(options.size_multiplier, 1 | 4)
    {
        return Err(DopeError::Data("invalid gold cell phase or size".into()));
    }
    let started = Instant::now();
    let starting_cpu_memory = peak_memory_bytes();
    let train = Table::read_dataset_dir(options.dataset_dir, options.task)?;
    clear_gpu_peak_memory();
    let (artifact, artifact_cache_status, fitting_time_ms) = cached_artifact(options, &train)?;
    let peak_gpu_memory_bytes = take_gpu_peak_memory();
    let artifact_bytes = artifact.len() as u64;
    let artifact_hashes = hashes(&artifact);
    let lineage_leaks = artifact_lineage_leaks(&artifact, options);
    let kernel = decode_kernel(&artifact)?;
    let synthetic_rows = train.rows.saturating_mul(options.size_multiplier);
    let generation_started = Instant::now();
    let sample = sample_kernel(
        &LoadedKernel::V3(Box::new(kernel.clone())),
        SampleOptions {
            rows: synthetic_rows,
            seed: Some(options.generation_seed),
        },
    )?;
    if generation_started.elapsed() > Duration::from_secs(60) {
        return Err(DopeError::Data(
            "generation timeout exceeded the frozen 60 second limit".into(),
        ));
    }
    let first_sampling_time = generation_started.elapsed();
    let repeat_started = Instant::now();
    let repeated = sample_kernel(
        &LoadedKernel::V3(Box::new(kernel.clone())),
        SampleOptions {
            rows: synthetic_rows,
            seed: Some(options.generation_seed),
        },
    )?;
    if repeat_started.elapsed() > Duration::from_secs(60) {
        return Err(DopeError::Data(
            "repeated generation timeout exceeded the frozen 60 second limit".into(),
        ));
    }
    let sampling_time_ms = first_sampling_time
        .saturating_add(repeat_started.elapsed())
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;
    let nondeterministic_output = sample
        .iter()
        .map(|value| value.to_bits())
        .ne(repeated.iter().map(|value| value.to_bits()));
    let (invalid_rows, schema_violations) = sample_diagnostics(&sample, &kernel, synthetic_rows);
    let synthetic = table_from_sample(&sample, synthetic_rows, train.features);

    // The guarded evaluator-owned holdout is opened only after fitting and
    // generation, preserving the no-test-data generator boundary.
    let test_path = options.dataset_dir.join("test.csv");
    if !test_path.is_file() {
        return Err(DopeError::Data(
            "gold evaluation requires evaluator-owned test.csv".into(),
        ));
    }
    let test = Table::read_csv(&test_path, options.task)?;
    if test.features != train.features {
        return Err(DopeError::Data(
            "gold train and test feature widths differ".into(),
        ));
    }
    Ok(PreparedGoldCell {
        lineage_group_id: options.lineage_group_id.into(),
        candidate_id: options.candidate_id.into(),
        task: options.task,
        size_multiplier: options.size_multiplier,
        generation_seed: options.generation_seed,
        train,
        test,
        synthetic,
        artifact_bytes,
        artifact_hashes,
        lineage_leaks,
        invalid_rows,
        schema_violations,
        nondeterministic_output,
        fitting_time_ms,
        sampling_time_ms,
        artifact_cache_status,
        peak_cpu_memory_bytes: starting_cpu_memory.max(peak_memory_bytes()),
        peak_gpu_memory_bytes,
        preparation_runtime_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
    })
}

include!("gold_cells.rs");
include!("privacy_attacks.rs");

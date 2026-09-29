
/// Scores one auditor × auditor-seed cell from a frozen generated table.
pub fn evaluate_prepared_gold_cell(
    options: &GoldCellOptions<'_>,
    prepared: &PreparedGoldCell,
) -> Result<JobEvidence> {
    if prepared.lineage_group_id != options.lineage_group_id
        || prepared.candidate_id != options.candidate_id
        || prepared.task != options.task
        || prepared.size_multiplier != options.size_multiplier
        || prepared.generation_seed != options.generation_seed
    {
        return Err(DopeError::Data(
            "prepared gold table does not match requested cell".into(),
        ));
    }
    let started = Instant::now();
    let train = &prepared.train;
    let test = &prepared.test;
    let synthetic = &prepared.synthetic;
    let artifact_hashes = &prepared.artifact_hashes;
    let auditor = auditor_backend(options.auditor_id)?;
    clear_gpu_peak_memory();
    let (real_importance, real_auditor_cache_status) =
        cached_real_importance(options, train, test)?;
    let mut peak_gpu_memory_bytes = take_gpu_peak_memory();
    clear_gpu_peak_memory();
    let synthetic_importance =
        auditor.permutation_importance(synthetic, test, options.task, options.auditor_seed)?;
    peak_gpu_memory_bytes = maximum_optional(peak_gpu_memory_bytes, take_gpu_peak_memory());
    let feature_consistency =
        compare_permutation_importance(&real_importance, &synthetic_importance)?;
    let real_prediction = real_importance.baseline_predictions;
    let synthetic_prediction = synthetic_importance.baseline_predictions;
    let synthetic_train_prediction = if options.ancillary {
        clear_gpu_peak_memory();
        let prediction =
            auditor.predict(synthetic, synthetic, options.task, options.auditor_seed)?;
        peak_gpu_memory_bytes = maximum_optional(peak_gpu_memory_bytes, take_gpu_peak_memory());
        Some(prediction)
    } else {
        None
    };
    let base = train
        .target
        .iter()
        .map(|&value| f64::from(value))
        .sum::<f64>()
        / train.rows.max(1) as f64;
    let null_prediction = vec![base as f32; test.rows];
    let real_prediction_f32 = real_prediction
        .iter()
        .map(|&value| value as f32)
        .collect::<Vec<_>>();
    let synthetic_prediction_f32 = synthetic_prediction
        .iter()
        .map(|&value| value as f32)
        .collect::<Vec<_>>();
    let null_loss = loss(options.task, &test.target, &null_prediction);
    let trtr_loss = loss(options.task, &test.target, &real_prediction_f32);
    let tstr_loss = loss(options.task, &test.target, &synthetic_prediction_f32);
    let real_calibration = calibration_error(options.task, &test.target, &real_prediction_f32);
    let synthetic_calibration =
        calibration_error(options.task, &test.target, &synthetic_prediction_f32);
    let (
        rare_retention,
        subgroup_retention,
        coverage,
        driver,
        joint,
        query_error,
        false_positive_rate,
        membership,
        attribute_advantage,
        exact_copies,
        near_copies,
        ancillary_cache_status,
    ) = if options.ancillary {
        let synthetic_train_prediction = synthetic_train_prediction
            .as_ref()
            .ok_or_else(|| DopeError::Data("missing ancillary auditor prediction".into()))?
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let (ancillary, ancillary_cache_status) =
            cached_synthetic_ancillary(options, &artifact_hashes.sha256, train, test, synthetic)?;
        (
            rare_or_tail_retention(
                options.task,
                &test.target,
                &null_prediction,
                &real_prediction_f32,
                &synthetic_prediction_f32,
            ),
            subgroup_retention(
                options.task,
                test,
                &null_prediction,
                &real_prediction_f32,
                &synthetic_prediction_f32,
            ),
            nominal_coverage(
                options.task,
                &synthetic.target,
                &synthetic_train_prediction,
                &test.target,
                &synthetic_prediction_f32,
            ),
            Some(ancillary.driver_agreement),
            Some(ancillary.joint_fidelity),
            Some(ancillary.query_p95_normalized_error),
            Some(ancillary.type_i_error),
            Some(ancillary.membership_auc),
            Some(ancillary.attribute_inference_advantage),
            ancillary.exact_copies,
            ancillary.near_copies,
            ancillary_cache_status,
        )
    } else {
        (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            0,
            CacheStatus::Disabled,
        )
    };
    let auditor_time_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    let runtime_ms = prepared
        .preparation_runtime_ms
        .saturating_add(auditor_time_ms);
    let peak_cpu_memory_bytes = prepared.peak_cpu_memory_bytes.max(peak_memory_bytes());

    Ok(JobEvidence {
        format: "dope-job-evidence".into(),
        version: JOB_EVIDENCE_VERSION,
        phase: options.phase.into(),
        task: options.task.as_str().into(),
        candidate_id: options.candidate_id.into(),
        auditor_id: options.auditor_id.into(),
        lineage_group_id: options.lineage_group_id.into(),
        structural_profile: options.structural_profile.into(),
        train_rows: train.rows,
        features: train.features,
        size_multiplier: options.size_multiplier,
        generation_seed: options.generation_seed,
        auditor_seed: options.auditor_seed,
        routed: options.routed,
        null_loss,
        trtr_loss,
        tstr_loss,
        runtime_ms,
        fitting_time_ms: prepared.fitting_time_ms,
        sampling_time_ms: prepared.sampling_time_ms,
        auditor_time_ms,
        peak_memory_bytes: peak_cpu_memory_bytes,
        peak_cpu_memory_bytes,
        peak_gpu_memory_bytes: maximum_optional(
            prepared.peak_gpu_memory_bytes,
            peak_gpu_memory_bytes,
        ),
        artifact_bytes: prepared.artifact_bytes,
        artifact_cache_status: prepared.artifact_cache_status,
        real_auditor_cache_status,
        ancillary_cache_status,
        artifact_sha256: artifact_hashes.sha256.clone(),
        artifact_blake3: artifact_hashes.blake3.clone(),
        calibration_degradation: Some(synthetic_calibration - real_calibration),
        rare_class_or_tail_retention: rare_retention,
        supported_subgroup_retention: subgroup_retention,
        nominal_95_coverage: coverage,
        driver_agreement: driver,
        joint_fidelity: joint,
        query_p95_normalized_error: query_error,
        type_i_error: false_positive_rate,
        membership_auc: membership,
        attribute_inference_advantage: attribute_advantage,
        feature_importance_spearman: feature_consistency.spearman,
        feature_importance_top_k_agreement: feature_consistency.top_k_agreement,
        feature_importance_feature_count: feature_consistency.feature_count,
        feature_importance_informative_count: feature_consistency.informative_feature_count,
        feature_importance_real_shares: feature_consistency.real_normalized_shares,
        feature_importance_synthetic_shares: feature_consistency.synthetic_normalized_shares,
        feature_importance_mean_ratio_error: feature_consistency.mean_ratio_error,
        exact_copies,
        near_copies,
        // Conservatively treat every source row as an extraction canary. This
        // upper-bounds any designated-canary extraction without inventing a
        // zero when source rows were reproduced.
        canary_extractions: exact_copies,
        lineage_leaks: prepared.lineage_leaks,
        invalid_rows: prepared.invalid_rows,
        schema_violations: prepared.schema_violations,
        nondeterministic_output: prepared.nondeterministic_output,
    })
}

/// Executes one immutable candidate × auditor × generation × auditor-seed
/// cell. This wrapper is used by individually leased controller jobs.
pub fn evaluate_gold_cell(options: &GoldCellOptions<'_>) -> Result<JobEvidence> {
    let prepared = prepare_gold_cell(options)?;
    evaluate_prepared_gold_cell(options, &prepared)
}

fn completed(table: &Table, column: usize) -> Vec<f32> {
    let mut observed: Vec<f32> = table.columns[column]
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    observed.sort_by(f32::total_cmp);
    let fill = observed.get(observed.len() / 2).copied().unwrap_or(0.0);
    table.columns[column]
        .iter()
        .map(|value| if value.is_nan() { fill } else { *value })
        .collect()
}

fn quantile(sorted: &[f32], index: usize, denominator: usize) -> f32 {
    sorted[index * sorted.len().saturating_sub(1) / denominator.max(1)]
}

fn loss(task: Task, target: &[f32], prediction: &[f32]) -> f64 {
    match task {
        Task::Regression => {
            target
                .iter()
                .zip(prediction)
                .map(|(actual, predicted)| f64::from(actual - predicted).powi(2))
                .sum::<f64>()
                / target.len().max(1) as f64
        }
        Task::Binary => {
            target
                .iter()
                .zip(prediction)
                .map(|(actual, predicted)| {
                    let p = f64::from(*predicted).clamp(1e-9, 1.0 - 1e-9);
                    let y = f64::from(*actual);
                    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
                })
                .sum::<f64>()
                / target.len().max(1) as f64
        }
    }
}

fn correlation(column: &[f32], target: &[f32]) -> f64 {
    let x_mean =
        column.iter().map(|value| f64::from(*value)).sum::<f64>() / column.len().max(1) as f64;
    let y_mean =
        target.iter().map(|value| f64::from(*value)).sum::<f64>() / target.len().max(1) as f64;
    let numerator = column
        .iter()
        .zip(target)
        .map(|(x, y)| (f64::from(*x) - x_mean) * (f64::from(*y) - y_mean))
        .sum::<f64>();
    let x_norm = column
        .iter()
        .map(|x| (f64::from(*x) - x_mean).powi(2))
        .sum::<f64>();
    let y_norm = target
        .iter()
        .map(|y| (f64::from(*y) - y_mean).powi(2))
        .sum::<f64>();
    if x_norm * y_norm <= 1e-18 {
        0.0
    } else {
        numerator / (x_norm * y_norm).sqrt()
    }
}

fn driver_agreement(real: &Table, synthetic: &Table) -> f64 {
    let mut real_scores: Vec<_> = (0..real.features)
        .map(|column| {
            (
                column,
                correlation(&completed(real, column), &real.target).abs(),
            )
        })
        .collect();
    let mut synth_scores: Vec<_> = (0..synthetic.features)
        .map(|column| {
            (
                column,
                correlation(&completed(synthetic, column), &synthetic.target).abs(),
            )
        })
        .collect();
    real_scores.sort_by(|(li, left), (ri, right)| right.total_cmp(left).then_with(|| li.cmp(ri)));
    synth_scores.sort_by(|(li, left), (ri, right)| right.total_cmp(left).then_with(|| li.cmp(ri)));
    let k = 20
        .min(5.max((real.features as f64 * 0.1).ceil() as usize))
        .min(real.features)
        .max(1);
    let real_top: HashSet<_> = real_scores
        .iter()
        .take(k)
        .map(|(index, _)| *index)
        .collect();
    let synthetic_top: HashSet<_> = synth_scores
        .iter()
        .take(k)
        .map(|(index, _)| *index)
        .collect();
    real_top.intersection(&synthetic_top).count() as f64 / k as f64
}

fn wasserstein(real: &[f32], synthetic: &[f32]) -> f64 {
    let mut left: Vec<_> = real
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let mut right: Vec<_> = synthetic
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    left.sort_by(f32::total_cmp);
    right.sort_by(f32::total_cmp);
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let probes = 64;
    (0..probes)
        .map(|index| {
            let l = left[index * left.len().saturating_sub(1) / (probes - 1)];
            let r = right[index * right.len().saturating_sub(1) / (probes - 1)];
            f64::from((l - r).abs())
        })
        .sum::<f64>()
        / probes as f64
}

fn joint_fidelity(real: &Table, synthetic: &Table) -> f64 {
    let (marginal, dependence) = fidelity_components(real, synthetic);
    0.55 * marginal + 0.45 * dependence
}

fn fidelity_components(real: &Table, synthetic: &Table) -> (f64, f64) {
    let marginal = (0..real.features)
        .map(|column| {
            1.0 - (wasserstein(&real.columns[column], &synthetic.columns[column]) / 0.25).min(1.0)
        })
        .sum::<f64>()
        / real.features as f64;
    let probes = real.features.saturating_sub(1).min(64);
    let dependence = if probes == 0 {
        1.0
    } else {
        (0..probes)
            .map(|index| {
                let left = index;
                let right = (index * 17 + 1) % real.features;
                let real_corr = correlation(&completed(real, left), &completed(real, right));
                let synth_corr =
                    correlation(&completed(synthetic, left), &completed(synthetic, right));
                1.0 - ((real_corr - synth_corr).abs() / 1.5).min(1.0)
            })
            .sum::<f64>()
            / probes as f64
    };
    (marginal, dependence)
}

fn nearest_distances(reference: &Table, query: &Table) -> Vec<f64> {
    let reference_columns: Vec<_> = (0..reference.features)
        .map(|column| completed(reference, column))
        .collect();
    let query_columns: Vec<_> = (0..query.features)
        .map(|column| completed(query, column))
        .collect();
    let query_rows = query.rows.min(256);
    (0..query_rows)
        .into_par_iter()
        .map(|query_row| {
            (0..reference.rows)
                .map(|reference_row| {
                    (0..reference.features)
                        .map(|column| {
                            let delta = f64::from(
                                query_columns[column][query_row]
                                    - reference_columns[column][reference_row],
                            );
                            delta * delta
                        })
                        .sum::<f64>()
                })
                .fold(f64::INFINITY, f64::min)
                .sqrt()
        })
        .collect()
}

fn membership_auc(synthetic: &Table, train: &Table, test: &Table) -> f64 {
    let train_distances = nearest_distances(synthetic, train);
    let test_distances = nearest_distances(synthetic, test);
    let mut wins = 0.0;
    for train in &train_distances {
        for test in &test_distances {
            if train < test {
                wins += 1.0
            } else if train == test {
                wins += 0.5
            }
        }
    }
    wins / (train_distances.len() * test_distances.len()).max(1) as f64
}

fn joint_table(table: &Table) -> Table {
    let mut joint = table.clone();
    joint.columns.push(table.target.clone());
    joint.features += 1;
    joint
}
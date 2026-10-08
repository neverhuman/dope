fn near_copy_count(reference: &Table, synthetic: &Table) -> usize {
    const THRESHOLD: f64 = 1e-3;
    let mut groups = BTreeMap::<Vec<u64>, Vec<usize>>::new();
    for row in 0..reference.rows {
        groups
            .entry(missingness_key(reference, row))
            .or_default()
            .push(row);
    }
    (0..synthetic.rows)
        .filter(|query| {
            let key = missingness_key(synthetic, *query);
            let Some(rows) = groups.get(&key) else {
                return false;
            };
            let projection = (0..=reference.features)
                .find(|column| row_value(synthetic, *query, *column).is_finite())
                .unwrap_or(reference.features);
            let mut ordered: Vec<_> = rows
                .iter()
                .map(|row| (row_value(reference, *row, projection), *row))
                .collect();
            ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
            let query_value = row_value(synthetic, *query, projection);
            let radius = THRESHOLD * ((reference.features + 1) as f64).sqrt();
            let start = ordered
                .partition_point(|(value, _)| f64::from(*value) < f64::from(query_value) - radius);
            ordered[start..]
                .iter()
                .take_while(|(value, _)| f64::from(*value) <= f64::from(query_value) + radius)
                .any(|(_, row)| {
                    let squared = (0..=reference.features)
                        .filter_map(|column| {
                            let left = row_value(reference, *row, column);
                            let right = row_value(synthetic, *query, column);
                            left.is_finite().then(|| f64::from(left - right).powi(2))
                        })
                        .sum::<f64>();
                    (squared / (reference.features + 1) as f64).sqrt() <= THRESHOLD
                })
        })
        .count()
}

fn attribute_inference_advantage(synthetic: &Table, train: &Table, test: &Table) -> f64 {
    let query_rows = test.rows.min(128);
    let reference_step = synthetic.rows.div_ceil(4096).max(1);
    let test_columns: Vec<_> = (0..test.features)
        .map(|column| completed(test, column))
        .collect();
    let synthetic_columns: Vec<_> = (0..synthetic.features)
        .map(|column| completed(synthetic, column))
        .collect();
    let hidden_features = train.features.min(16);
    let references = (0..synthetic.rows)
        .step_by(reference_step)
        .collect::<Vec<_>>();
    let baselines = (0..hidden_features)
        .map(|hidden| {
            let values = completed(train, hidden);
            values.iter().sum::<f32>() / values.len().max(1) as f32
        })
        .collect::<Vec<_>>();
    let mut baseline_squared = vec![0.0f64; hidden_features];
    let mut attack_squared = vec![0.0f64; hidden_features];
    for row in 0..query_rows {
        let total_distances = references
            .iter()
            .map(|&candidate| {
                (0..train.features)
                    .map(|column| {
                        let delta =
                            test_columns[column][row] - synthetic_columns[column][candidate];
                        f64::from(delta).powi(2)
                    })
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        for hidden in 0..hidden_features {
            baseline_squared[hidden] +=
                f64::from(test_columns[hidden][row] - baselines[hidden]).powi(2);
            let nearest = references
                .iter()
                .zip(&total_distances)
                .min_by(|(left, left_total), (right, right_total)| {
                    let excluding_hidden = |candidate: usize, total: f64| {
                        let delta =
                            test_columns[hidden][row] - synthetic_columns[hidden][candidate];
                        total - f64::from(delta).powi(2)
                    };
                    excluding_hidden(**left, **left_total)
                        .total_cmp(&excluding_hidden(**right, **right_total))
                })
                .map(|(&candidate, _)| candidate)
                .unwrap_or(0);
            attack_squared[hidden] +=
                f64::from(test_columns[hidden][row] - synthetic_columns[hidden][nearest]).powi(2);
        }
    }
    (0..hidden_features)
        .map(|hidden| {
            ((baseline_squared[hidden] - attack_squared[hidden])
                / baseline_squared[hidden].max(1e-12))
            .max(0.0)
        })
        .reduce(f64::max)
        .unwrap_or(0.0)
}

fn one_sided_clustered_lower(metrics: &[MetricVector], rows: usize) -> Option<f64> {
    let mut clusters = BTreeMap::<u64, Vec<f64>>::new();
    for metric in metrics
        .iter()
        .filter(|metric| metric.synthetic_rows == rows)
    {
        if let Some(retention) = metric.retention {
            clusters
                .entry(metric.generation_seed)
                .or_default()
                .push(retention);
        }
    }
    let means: Vec<_> = clusters
        .values()
        .map(|values| values.iter().sum::<f64>() / values.len() as f64)
        .collect();
    if means.is_empty() {
        return None;
    }
    let mut bootstrap = Vec::new();
    for left in &means {
        for middle in &means {
            for right in &means {
                bootstrap.push((left + middle + right) / 3.0);
            }
        }
    }
    bootstrap.sort_by(f64::total_cmp);
    bootstrap.get(bootstrap.len() / 20).copied()
}

pub fn certify_kernel(
    real_dir: &Path,
    kernel_path: &Path,
    out: &Path,
    seed: u64,
    runtime_dictionary_bytes: usize,
    supported_datasets: usize,
) -> Result<CertificationReport> {
    certify_kernel_with_policy(
        real_dir,
        kernel_path,
        out,
        seed,
        runtime_dictionary_bytes,
        supported_datasets,
        &ReleasePolicy::new(AnonymizationTier::L0, None, false)?,
    )
}

pub fn certify_kernel_with_policy(
    real_dir: &Path,
    kernel_path: &Path,
    out: &Path,
    seed: u64,
    runtime_dictionary_bytes: usize,
    supported_datasets: usize,
    policy: &ReleasePolicy,
) -> Result<CertificationReport> {
    policy.validate()?;
    let artifact_bytes = fs::metadata(kernel_path)
        .map_err(|error| io_error(kernel_path, error))?
        .len() as usize;
    if !policy.accepts_artifact_bytes(artifact_bytes) {
        return Err(DopeError::Data(format!(
            "artifact exceeds release-policy byte limit: {artifact_bytes} bytes"
        )));
    }
    let loaded = load_kernel(kernel_path)?;
    let source = match &loaded {
        LoadedKernel::V3(kernel) | LoadedKernel::V2(kernel) => kernel,
        LoadedKernel::V1(_) => {
            return Err(DopeError::Unsupported(
                "certification requires a canonical V2 or V3 artifact".into(),
            ));
        }
    };
    let train = Table::read_dataset_dir_with_policy(real_dir, source.task, policy)?;
    let test_path = real_dir.join("test.csv");
    if !test_path.is_file() {
        return Err(DopeError::Data(
            "guarded certification requires an evaluator-owned test.csv".into(),
        ));
    }
    let test = Table::read_csv_with_policy(&test_path, source.task, policy)?;
    if train.features != test.features {
        return Err(DopeError::Data(
            "real train.csv and evaluator test.csv have different widths".into(),
        ));
    }
    let base = train.target.iter().sum::<f32>() / train.rows.max(1) as f32;
    let null_prediction = vec![base; test.rows];
    let null_loss = loss(source.task, &test.target, &null_prediction);
    let minority = if train.target.iter().filter(|&&value| value == 1.0).count() * 2 <= train.rows {
        1.0
    } else {
        0.0
    };
    let rare_train_indices = rare_slice_indices(&train, source.task, minority);
    let rare_test_indices = rare_slice_indices(&test, source.task, minority);
    let rare_slice_required = rare_train_indices.len() >= 10;
    let rare_slice_covered = !rare_slice_required || rare_test_indices.len() >= 10;
    let rare_tables = (rare_slice_required && rare_slice_covered).then(|| {
        (
            selected_rows(&train, &rare_train_indices),
            selected_rows(&test, &rare_test_indices),
        )
    });

    let specs = auditor_specs();
    let mut real_importances = BTreeMap::new();
    for spec in &specs {
        if !spec.available {
            continue;
        }
        let auditor = auditor_backend(&spec.id)?;
        for auditor_seed in AUDITOR_SEEDS {
            real_importances.insert(
                (spec.id.clone(), auditor_seed),
                auditor.permutation_importance(&train, &test, source.task, auditor_seed)?,
            );
        }
    }

    let sizes: Vec<_> = [1usize, 2, 4, 8]
        .into_iter()
        .map(|multiplier| train.rows.saturating_mul(multiplier))
        .collect();
    let gold_rows = [train.rows, train.rows.saturating_mul(4)];
    let contract = KpiContract::embedded()?;
    contract.validate()?;
    let minimum_informative_features = contract
        .release_gates
        .feature_importance_min_informative_features
        .unwrap_or(3);
    let mut metrics_by_auditor = BTreeMap::<String, Vec<MetricVector>>::new();
    let mut importance_spearman = Vec::new();
    let mut importance_jaccard = Vec::new();
    let mut importance_applicable = false;
    let mut importance_complete = true;
    let mut importance_evidence = Vec::new();
    let mut drivers = Vec::new();
    let mut joints = Vec::new();
    let mut marginal_fidelities = Vec::new();
    let mut dependence_fidelities = Vec::new();
    let mut fitness_diagnostics = Vec::new();
    let mut queries = Vec::new();
    let mut type_i_errors = Vec::new();
    let mut memberships = Vec::new();
    let mut feature_memberships = Vec::new();
    let mut joint_memberships = Vec::new();
    let mut rare_memberships = Vec::new();
    let mut rare_attributes = Vec::new();
    let mut attribute_advantages = Vec::new();
    let mut exact_copies = 0usize;
    let mut near_copies = 0usize;

    for (size_index, synthetic_rows) in sizes.iter().copied().enumerate() {
        for generation in 0..GENERATION_REPEATS {
            let generation_seed = seed
                .wrapping_add((size_index as u64 + 1).wrapping_mul(1_000_003))
                .wrapping_add((generation as u64 + 1).wrapping_mul(97_409));
            let sampled = sample_kernel(
                &loaded,
                SampleOptions {
                    rows: synthetic_rows,
                    seed: Some(generation_seed),
                },
            )?;
            let synthetic = table_from_sample(&sampled, synthetic_rows, train.features);
            let gold = gold_rows.contains(&synthetic_rows);
            if gold {
                drivers.push(driver_agreement(&train, &synthetic));
                let (marginal, dependence) = fidelity_components(&train, &synthetic);
                marginal_fidelities.push(marginal);
                dependence_fidelities.push(dependence);
                joints.push(0.55 * marginal + 0.45 * dependence);
                if let Some(diagnostics) =
                    evaluate_fitness_diagnostics(&test, &synthetic, generation_seed)
                {
                    fitness_diagnostics.push(diagnostics);
                }
                queries.push(query_p95_normalized_error(&train, &synthetic));
                type_i_errors.push(type_i_error(&train, &synthetic));
                let feature_attack = membership_auc(&synthetic, &train, &test);
                let joint_attack = membership_auc(
                    &joint_table(&synthetic),
                    &joint_table(&train),
                    &joint_table(&test),
                );
                feature_memberships.push(feature_attack);
                joint_memberships.push(joint_attack);
                memberships.push(feature_attack.max(joint_attack));
                if let Some((rare_train, rare_test)) = &rare_tables {
                    rare_memberships.push(membership_auc(&synthetic, rare_train, rare_test));
                    rare_attributes.push(attribute_inference_advantage(
                        &synthetic, rare_train, rare_test,
                    ));
                }
                attribute_advantages.push(attribute_inference_advantage(&synthetic, &train, &test));
                exact_copies += exact_copy_count(&train, &synthetic);
                near_copies += near_copy_count(&train, &synthetic);
            }
            for spec in &specs {
                if !spec.available {
                    continue;
                }
                let auditor = auditor_backend(&spec.id)?;
                for auditor_seed in AUDITOR_SEEDS {
                    let real = &real_importances[&(spec.id.clone(), auditor_seed)];
                    let real_test = real
                        .baseline_predictions
                        .iter()
                        .map(|&value| value as f32)
                        .collect::<Vec<_>>();
                    let synthetic_importance = if gold {
                        Some(auditor.permutation_importance(
                            &synthetic,
                            &test,
                            source.task,
                            auditor_seed,
                        )?)
                    } else {
                        None
                    };
                    let synthetic_test = if let Some(importance) = &synthetic_importance {
                        importance
                            .baseline_predictions
                            .iter()
                            .map(|&value| value as f32)
                            .collect::<Vec<_>>()
                    } else {
                        auditor
                            .predict(&synthetic, &test, source.task, auditor_seed)?
                            .into_iter()
                            .map(|value| value as f32)
                            .collect::<Vec<_>>()
                    };
                    if let Some(synthetic_importance) = &synthetic_importance {
                        let comparison =
                            compare_permutation_importance(real, synthetic_importance)?;
                        importance_complete &= comparison.feature_count == train.features;
                        if comparison.informative_feature_count >= minimum_informative_features {
                            importance_applicable = true;
                            importance_spearman.push(comparison.spearman);
                            importance_jaccard.push(comparison.top_k_agreement);
                        }
                        importance_evidence.push(ImportanceEvidence {
                            auditor_id: spec.id.clone(),
                            auditor_seed,
                            synthetic_rows,
                            generation_seed,
                            consistency: comparison,
                        });
                    }
                    let synthetic_train = auditor
                        .predict(&synthetic, &synthetic, source.task, auditor_seed)?
                        .into_iter()
                        .map(|value| value as f32)
                        .collect::<Vec<_>>();
                    let real_loss = loss(source.task, &test.target, &real_test);
                    let synthetic_loss = loss(source.task, &test.target, &synthetic_test);
                    let decision = retention_decision(null_loss, real_loss, synthetic_loss);
                    metrics_by_auditor
                        .entry(spec.id.clone())
                        .or_default()
                        .push(MetricVector {
                            synthetic_rows,
                            generation_seed,
                            auditor_seed,
                            null_loss,
                            trtr_loss: real_loss,
                            tstr_loss: synthetic_loss,
                            informative: decision.informative,
                            retention: decision.retention,
                            absolute_noninferiority_passed: decision.absolute_noninferiority_passed,
                            calibration_degradation: calibration_error(
                                source.task,
                                &test.target,
                                &synthetic_test,
                            ) - calibration_error(
                                source.task,
                                &test.target,
                                &real_test,
                            ),
                            rare_class_or_tail_retention: rare_or_tail_retention(
                                source.task,
                                &test.target,
                                &null_prediction,
                                &real_test,
                                &synthetic_test,
                            ),
                            supported_subgroup_retention: subgroup_retention(
                                source.task,
                                &test,
                                &null_prediction,
                                &real_test,
                                &synthetic_test,
                            ),
                            nominal_95_coverage: nominal_coverage(
                                source.task,
                                &synthetic.target,
                                &synthetic_train,
                                &test.target,
                                &synthetic_test,
                            ),
                        });
                }
            }
        }
    }

    let report_for = |spec: &crate::contract::AuditorSpec, metrics: Vec<MetricVector>| {
        let mut bounds = BTreeMap::new();
        for (multiplier, rows) in GOLD_SIZE_MULTIPLIERS.into_iter().zip(gold_rows) {
            bounds.insert(
                format!("{multiplier}n"),
                one_sided_clustered_lower(&metrics, rows),
            );
        }
        let gold: Vec<_> = metrics
            .iter()
            .filter(|metric| gold_rows.contains(&metric.synthetic_rows))
            .collect();
        let utility = gold.iter().all(|metric| {
            if metric.informative {
                bounds
                    .get(if metric.synthetic_rows == train.rows {
                        "1n"
                    } else {
                        "4n"
                    })
                    .and_then(|value| *value)
                    .is_some_and(|lower| lower >= 0.99)
            } else {
                metric.absolute_noninferiority_passed
            }
        });
        let ancillary = gold.iter().all(|metric| {
            metric.calibration_degradation <= 0.02
                && metric
                    .rare_class_or_tail_retention
                    .is_none_or(|value| value >= 0.95)
                && metric
                    .supported_subgroup_retention
                    .is_none_or(|value| value >= 0.95)
        });
        AuditorReport {
            available: true,
            backend: spec.backend.clone(),
            version: spec.frozen_version.clone(),
            hash: spec.frozen_hash.clone(),
            telemetry_disabled: spec.telemetry_disabled,
            null_loss: Some(null_loss),
            real_trained_loss: metrics.first().map(|metric| metric.trtr_loss),
            synthetic_trained_loss: metrics.first().map(|metric| metric.tstr_loss),
            retention: metrics
                .iter()
                .filter_map(|metric| metric.retention)
                .reduce(f64::min),
            metrics,
            one_sided_95_lower_by_multiplier: bounds,
            gold_gate_passed: utility && ancillary,
            reason: None,
        }
    };
    include!("privacy_attacks/finalize_certification.rs")()
}

/// Scores one mapped triple without reading a path. `certify_kernel` is not used:
/// that entry requires an evaluator-owned `test.csv`.
pub fn score_holdout_tables(
    fit: &crate::data::Table,
    synthetic: &crate::data::Table,
    holdout: &crate::data::Table,
    generation_seed: u64,
) -> HoldoutTableScores {
    let feature_attack = membership_auc(synthetic, fit, holdout);
    let joint_attack = membership_auc(
        &joint_table(synthetic),
        &joint_table(fit),
        &joint_table(holdout),
    );
    let (marginal, dependence) = fidelity_components(fit, synthetic);
    let diagnostics = evaluate_fitness_diagnostics(holdout, synthetic, generation_seed);
    HoldoutTableScores {
        membership_auc: feature_attack.max(joint_attack),
        attribute_inference_advantage: attribute_inference_advantage(synthetic, fit, holdout),
        marginal_fidelity: marginal,
        dependence_fidelity: dependence,
        sliced_wasserstein_fidelity: diagnostics
            .as_ref()
            .map(|item| item.sliced_wasserstein_fidelity),
        mmd_fidelity: diagnostics.as_ref().map(|item| item.mmd_fidelity),
        coverage_realism: diagnostics.as_ref().map(|item| item.coverage_realism),
        driver_fidelity: driver_fidelity(fit, synthetic),
        driver_importance: "absolute_target_correlation",
    }
}

/// One-sided 95% lower bound used as the utility component before clamping.
pub fn retention_lower_bound(values: &[f64]) -> Option<f64> {
    crate::production::one_sided_lower(values, 0.95)
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct HoldoutTableScores {
    pub membership_auc: f64,
    pub attribute_inference_advantage: f64,
    pub marginal_fidelity: f64,
    pub dependence_fidelity: f64,
    pub sliced_wasserstein_fidelity: Option<f64>,
    pub mmd_fidelity: Option<f64>,
    pub coverage_realism: Option<f64>,
    pub driver_fidelity: Option<f64>,
    pub driver_importance: &'static str,
}

fn driver_fidelity(fit: &crate::data::Table, synthetic: &crate::data::Table) -> Option<f64> {
    if fit.features == 0 || fit.features != synthetic.features {
        return None;
    }
    let real: Vec<(usize, f64)> = (0..fit.features)
        .map(|column| {
            (
                column,
                correlation(&completed(fit, column), &fit.target).abs(),
            )
        })
        .filter(|(_, value)| value.is_finite() && *value > 1e-12)
        .collect();
    if real.len() < 2 {
        return None;
    }
    let mut paired = Vec::new();
    for (column, real_score) in real {
        let synthetic_score = correlation(&completed(synthetic, column), &synthetic.target).abs();
        if synthetic_score.is_finite() {
            paired.push((column, real_score, synthetic_score));
        }
    }
    if paired.len() < 2 {
        return None;
    }
    let real_ranks = average_rank_values(
        &paired
            .iter()
            .map(|(_, value, _)| *value)
            .collect::<Vec<_>>(),
    );
    let synthetic_ranks = average_rank_values(
        &paired
            .iter()
            .map(|(_, _, value)| *value)
            .collect::<Vec<_>>(),
    );
    let spearman = rank_correlation(&real_ranks, &synthetic_ranks)?;
    let top_k = paired.len().div_ceil(5).min(10);
    let top = |position: usize| {
        let mut values = paired.clone();
        values.sort_by(|left, right| {
            let score = |row: &(usize, f64, f64)| if position == 1 { row.1 } else { row.2 };
            score(right)
                .total_cmp(&score(left))
                .then_with(|| left.0.cmp(&right.0))
        });
        values
            .into_iter()
            .take(top_k)
            .map(|row| row.0)
            .collect::<std::collections::BTreeSet<_>>()
    };
    let real_top = top(1);
    let synthetic_top = top(2);
    let union = real_top.union(&synthetic_top).count();
    let overlap = if union == 0 {
        1.0
    } else {
        real_top.intersection(&synthetic_top).count() as f64 / union as f64
    };
    let rank = ((spearman + 1.0) / 2.0).clamp(0.0, 1.0);
    Some(0.5 * rank + 0.5 * overlap.clamp(0.0, 1.0))
}

fn average_rank_values(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| {
        values[left]
            .total_cmp(&values[right])
            .then_with(|| left.cmp(&right))
    });
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let rank = (start + end - 1) as f64 / 2.0 + 1.0;
        for index in &order[start..end] {
            ranks[*index] = rank;
        }
        start = end;
    }
    ranks
}

fn rank_correlation(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut numerator = 0.0;
    let mut left_scale = 0.0;
    let mut right_scale = 0.0;
    for (left, right) in left.iter().zip(right) {
        let left_delta = left - left_mean;
        let right_delta = right - right_mean;
        numerator += left_delta * right_delta;
        left_scale += left_delta * left_delta;
        right_scale += right_delta * right_delta;
    }
    let denominator = (left_scale * right_scale).sqrt();
    (denominator > f64::EPSILON).then_some(numerator / denominator)
}

#[cfg(test)]
include!("privacy_attacks/tests.rs");

#[cfg(test)]
mod holdout_table_score {
    use super::*;
    use crate::data::Table;
    use crate::model::Task;

    fn regression(columns: &[f32], target: &[f32], rows: usize) -> Table {
        Table::from_arrays(columns, target, rows, 1, Task::Regression).unwrap()
    }

    #[test]
    fn identical_fit_and_synthetic_keep_full_marginal_fidelity() {
        let column = [0.05, 0.15, 0.25, 0.35, 0.55, 0.65, 0.75, 0.85];
        let target = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80];
        let fit = regression(&column, &target, 8);
        let holdout = regression(
            &[0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82],
            &[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
            8,
        );
        let scores = score_holdout_tables(&fit, &fit, &holdout, 11);
        assert!((scores.marginal_fidelity - 1.0).abs() < 1e-9);
        assert!((scores.dependence_fidelity - 1.0).abs() < 1e-9);
        assert!(scores.membership_auc.is_finite());
        assert!(scores.attribute_inference_advantage.is_finite());
        assert_eq!(scores.driver_importance, "absolute_target_correlation");
        assert!(scores.sliced_wasserstein_fidelity.is_some());
        let lower = retention_lower_bound(&[0.2, 0.5, 0.8]).unwrap();
        assert!((lower + 0.0057563382544248975).abs() < 1e-9);
    }
}

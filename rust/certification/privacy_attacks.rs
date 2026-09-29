
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

    let mut auditors = BTreeMap::new();
    for spec in specs {
        let report = if spec.available {
            report_for(
                &spec,
                metrics_by_auditor.remove(&spec.id).unwrap_or_default(),
            )
        } else {
            AuditorReport {
                available: false,
                backend: spec.backend.clone(),
                version: spec.frozen_version.clone(),
                hash: spec.frozen_hash.clone(),
                telemetry_disabled: spec.telemetry_disabled,
                null_loss: None,
                real_trained_loss: None,
                synthetic_trained_loss: None,
                retention: None,
                metrics: Vec::new(),
                one_sided_95_lower_by_multiplier: BTreeMap::new(),
                gold_gate_passed: false,
                reason: spec.reason.clone(),
            }
        };
        auditors.insert(spec.id, report);
    }

    let minimum_gold_one_sided_95_retention = auditors
        .values()
        .filter(|auditor| auditor.available)
        .flat_map(|auditor| auditor.one_sided_95_lower_by_multiplier.values())
        .filter_map(|value| *value)
        .reduce(f64::min);
    let driver = drivers.into_iter().reduce(f64::min).unwrap_or(0.0);
    let joint = joints.into_iter().reduce(f64::min).unwrap_or(0.0);
    let query = queries
        .into_iter()
        .reduce(f64::max)
        .unwrap_or(f64::INFINITY);
    let type_i = type_i_errors.into_iter().reduce(f64::max).unwrap_or(1.0);
    let membership = memberships.into_iter().reduce(f64::max).unwrap_or(1.0);
    let feature_attack_max = feature_memberships
        .into_iter()
        .reduce(f64::max)
        .unwrap_or(1.0);
    let joint_attack_max = joint_memberships
        .into_iter()
        .reduce(f64::max)
        .unwrap_or(1.0);
    let rare_slice_auc = rare_memberships.into_iter().reduce(f64::max);
    let rare_slice_attribute = rare_attributes.into_iter().reduce(f64::max);
    let attribute = attribute_advantages
        .into_iter()
        .reduce(f64::max)
        .unwrap_or(1.0);
    let native_gold: Vec<_> = auditors
        .values()
        .filter(|auditor| auditor.available)
        .flat_map(|auditor| &auditor.metrics)
        .filter(|metric| gold_rows.contains(&metric.synthetic_rows))
        .collect();
    let structure_spearman = importance_spearman
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>()
        .and_then(|values| values.into_iter().reduce(f64::min));
    let structure_jaccard = importance_jaccard.into_iter().reduce(f64::min);
    let mut gates: BTreeMap<String, bool> = BTreeMap::new();
    gates.insert(
        "auditor_family_coverage".into(),
        auditors.values().all(|auditor| auditor.available),
    );
    gates.insert(
        "tstr_retention_or_low_signal_noninferiority".into(),
        auditors.values().all(|auditor| auditor.gold_gate_passed),
    );
    gates.insert(
        "calibration".into(),
        native_gold
            .iter()
            .all(|metric| metric.calibration_degradation <= 0.02),
    );
    gates.insert(
        "rare_class_and_tail".into(),
        native_gold.iter().all(|metric| {
            metric
                .rare_class_or_tail_retention
                .is_none_or(|value| value >= 0.95)
        }),
    );
    gates.insert(
        "supported_subgroups".into(),
        native_gold.iter().all(|metric| {
            metric
                .supported_subgroup_retention
                .is_none_or(|value| value >= 0.95)
        }),
    );
    gates.insert("driver".into(), driver >= 0.95);
    gates.insert(
        "feature_view_coverage".into(),
        importance_complete && !importance_evidence.is_empty(),
    );
    gates.insert(
        "feature_importance_spearman".into(),
        !importance_applicable
            || structure_spearman.is_some_and(|value| {
                value
                    >= contract
                        .release_gates
                        .feature_importance_spearman_min
                        .unwrap_or(0.70)
            }),
    );
    gates.insert(
        "feature_importance_top_k_jaccard".into(),
        !importance_applicable
            || structure_jaccard.is_some_and(|value| {
                value
                    >= contract
                        .release_gates
                        .feature_importance_top_k_jaccard_min
                        .unwrap_or(0.50)
            }),
    );
    gates.insert("joint_fidelity".into(), joint >= 0.90);
    gates.insert("query_fidelity".into(), query <= 0.05);
    gates.insert(
        "nominal_95_coverage".into(),
        native_gold.iter().all(|metric| {
            metric
                .nominal_95_coverage
                .is_some_and(|coverage| (0.90..=0.98).contains(&coverage))
        }),
    );
    gates.insert("type_i_error".into(), type_i <= 0.06);
    gates.insert(
        "membership_inference".into(),
        membership <= contract.release_gates.membership_auc_max,
    );
    gates.insert("rare_slice_privacy_coverage".into(), rare_slice_covered);
    gates.insert(
        "rare_slice_membership".into(),
        !rare_slice_required
            || rare_slice_auc
                .is_some_and(|value| value <= contract.release_gates.membership_auc_max),
    );
    gates.insert(
        "rare_slice_attribute_inference".into(),
        !rare_slice_required
            || rare_slice_attribute.is_some_and(|value| {
                value <= contract.release_gates.attribute_inference_advantage_max
            }),
    );
    gates.insert("attribute_inference".into(), attribute <= 0.05);
    gates.insert("exact_copies".into(), exact_copies == 0);
    gates.insert("near_copies".into(), near_copies == 0);
    gates.insert(
        "runtime_size".into(),
        runtime_dictionary_bytes <= 64 * 1024 * 1024,
    );
    gates.insert(
        "artifact_size".into(),
        policy.accepts_artifact_bytes(artifact_bytes),
    );
    let failed_gates: Vec<_> = gates
        .iter()
        .filter(|(_, passed)| !**passed)
        .map(|(name, _)| name.clone())
        .collect();
    let artifact_bytes = fs::metadata(kernel_path)
        .map_err(|error| io_error(kernel_path, error))?
        .len() as usize;
    let effective_bytes =
        artifact_bytes + runtime_dictionary_bytes.div_ceil(supported_datasets.max(1));
    let marginal_fidelity_min = marginal_fidelities.into_iter().reduce(f64::min);
    let dependence_fidelity_min = dependence_fidelities.into_iter().reduce(f64::min);
    let diagnostics_complete = fitness_diagnostics.len() == 2 * GENERATION_REPEATS;
    let worst_diagnostics = |metric: fn(&FitnessDiagnostics) -> f64| {
        diagnostics_complete
            .then(|| fitness_diagnostics.iter().map(metric).reduce(f64::min))
            .flatten()
    };
    let sliced_fidelity_min = worst_diagnostics(|item| item.sliced_wasserstein_fidelity);
    let mmd_fidelity_min = worst_diagnostics(|item| item.mmd_fidelity);
    let coverage_realism_min = worst_diagnostics(|item| item.coverage_realism);
    let pareto_vector = ParetoVector {
        utility_transfer: minimum_gold_one_sided_95_retention.map(|value| value.clamp(0.0, 1.0)),
        driver_fidelity: structure_spearman
            .zip(structure_jaccard)
            .map(|(rank, top_k)| {
                0.5 * ((rank + 1.0) / 2.0).clamp(0.0, 1.0) + 0.5 * top_k.clamp(0.0, 1.0)
            }),
        distribution_fidelity: marginal_fidelity_min
            .zip(sliced_fidelity_min)
            .zip(mmd_fidelity_min)
            .map(|((marginal, sliced), mmd)| 0.50 * marginal + 0.25 * sliced + 0.25 * mmd),
        structure_fidelity: dependence_fidelity_min,
        coverage_realism: coverage_realism_min,
        compactness: policy
            .maximum_artifact_bytes
            .map(|limit| (1.0 - artifact_bytes as f64 / limit as f64).clamp(0.0, 1.0)),
    };
    let raw_metric_vector = BTreeMap::from([
        (
            "ptf_v1_gold_lower_bound".into(),
            minimum_gold_one_sided_95_retention,
        ),
        ("feature_importance_spearman_min".into(), structure_spearman),
        (
            "feature_importance_top_k_jaccard_min".into(),
            structure_jaccard,
        ),
        ("driver_agreement_min".into(), Some(driver)),
        ("joint_fidelity_min".into(), Some(joint)),
        ("marginal_w1_fidelity_min".into(), marginal_fidelity_min),
        (
            "sliced_wasserstein_fidelity_min".into(),
            sliced_fidelity_min,
        ),
        ("mmd_fidelity_min".into(), mmd_fidelity_min),
        (
            "prdc_precision_min".into(),
            worst_diagnostics(|item| item.prdc_precision),
        ),
        (
            "prdc_recall_min".into(),
            worst_diagnostics(|item| item.prdc_recall),
        ),
        (
            "prdc_density_min".into(),
            worst_diagnostics(|item| item.prdc_density),
        ),
        (
            "prdc_coverage_min".into(),
            worst_diagnostics(|item| item.prdc_coverage),
        ),
        (
            "pairwise_dependence_fidelity_min".into(),
            dependence_fidelity_min,
        ),
        ("query_p95_normalized_error_max".into(), Some(query)),
        ("type_i_error_max".into(), Some(type_i)),
        ("membership_auc_max".into(), Some(membership)),
        ("rare_slice_membership_auc_max".into(), rare_slice_auc),
        (
            "rare_slice_attribute_advantage_max".into(),
            rare_slice_attribute,
        ),
        ("attribute_inference_advantage_max".into(), Some(attribute)),
        ("exact_copy_count".into(), Some(exact_copies as f64)),
        ("near_copy_count".into(), Some(near_copies as f64)),
        ("artifact_bytes".into(), Some(artifact_bytes as f64)),
    ]);
    let master_fitness = MasterFitnessReport::evaluate(
        contract
            .master_fitness
            .as_ref()
            .expect("validated v2 fitness contract"),
        gates.clone(),
        pareto_vector,
        raw_metric_vector,
    );
    let report = CertificationReport {
        format: "dope-kernel-certification".into(),
        version: 5,
        release_policy: policy.clone(),
        release_policy_hash: policy.hash(),
        certified: failed_gates.is_empty(),
        gates,
        failed_gates,
        auditors,
        feature_importance: importance_evidence,
        minimum_gold_one_sided_95_retention,
        fidelity: FidelityReport {
            worst_driver_agreement: driver,
            worst_joint_fidelity: joint,
            worst_query_p95_normalized_error: query,
            maximum_type_i_error: type_i,
        },
        privacy: PrivacyReport {
            attack_suite_version: 1,
            membership_attacks: BTreeMap::from([
                ("nearest_features".into(), feature_attack_max),
                ("nearest_features_and_target".into(), joint_attack_max),
            ]),
            maximum_membership_auc: membership,
            rare_slice_supported: rare_slice_required,
            rare_slice_membership_auc: rare_slice_auc,
            rare_slice_attribute_advantage: rare_slice_attribute,
            maximum_attribute_inference_advantage: attribute,
            exact_copy_count: exact_copies,
            near_copy_count: near_copies,
            near_copy_definition: "identical missingness and normalized RMS distance <= 1e-3"
                .into(),
        },
        fitness_diagnostics,
        master_fitness: Some(master_fitness),
        isolation: IsolationEvidence {
            generator_opened: vec!["train.csv".into()],
            evaluator_opened_real_test: true,
            downstream_tuning:
                "frozen hyperparameters; TRTR fits real train only; TSTR fits synthetic only".into(),
            synthetic_sizes: sizes,
            generation_repeats: GENERATION_REPEATS,
            auditor_seeds: AUDITOR_SEEDS.to_vec(),
        },
        geometric_mean_retention: minimum_gold_one_sided_95_retention.unwrap_or(0.0),
        driver_agreement: driver,
        joint_fidelity: joint,
        membership_inference_auc: membership,
        synthetic_duplicate_rate: exact_copies as f64
            / (GENERATION_REPEATS * (train.rows + train.rows.saturating_mul(4))).max(1) as f64,
        primary_synthetic_rows: train.rows,
        secondary_synthetic_rows: train.rows.saturating_mul(4),
        artifact_bytes,
        effective_bytes,
        bits_per_original_cell: 8.0 * artifact_bytes as f64
            / (train.rows * (train.features + 1)).max(1) as f64,
        compression_vs_packed_f32: (train.rows * (train.features + 1) * 4) as f64
            / effective_bytes.max(1) as f64,
    };
    fs::write(out, serde_json::to_vec_pretty(&report)?).map_err(|error| io_error(out, error))?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_cache_is_atomic_and_immutable() {
        let root = std::env::temp_dir().join(format!(
            "dope-content-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let path = root.join("aa").join("cell.bin");
        atomic_cache_write(&path, b"first").unwrap();
        atomic_cache_write(&path, b"second").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"first");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn shared_retention_scale() {
        assert_eq!(null_normalized_excess_loss_retention(1.0, 0.5, 0.75), 0.5);
        assert_eq!(null_normalized_excess_loss_retention(1.0, 0.5, 0.4), 1.2);
        assert!((null_normalized_excess_loss_retention(1.0, 0.5, 1.1) + 0.2).abs() < 1e-12);
    }

    #[test]
    fn low_signal_uses_absolute_noninferiority() {
        let passed = retention_decision(1.0, 0.995, 1.004);
        assert!(!passed.informative);
        assert_eq!(passed.retention, None);
        assert!(passed.absolute_noninferiority_passed);
        let failed = retention_decision(1.0, 0.995, 1.006);
        assert!(!failed.absolute_noninferiority_passed);
    }

    #[test]
    fn exact_and_near_copy_checks_include_missingness() {
        let real = Table {
            rows: 2,
            features: 2,
            columns: vec![vec![0.1, f32::NAN], vec![0.2, 0.8]],
            target: vec![0.3, 0.7],
        };
        let exact = real.clone();
        assert_eq!(exact_copy_count(&real, &exact), 2);
        assert_eq!(near_copy_count(&real, &exact), 2);
        let changed_mask = Table {
            rows: 1,
            features: 2,
            columns: vec![vec![0.1], vec![f32::NAN]],
            target: vec![0.3],
        };
        assert_eq!(near_copy_count(&real, &changed_mask), 0);
    }

    #[test]
    fn certification_rejects_10241_bytes_before_decoding() {
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("certification-byte-boundary-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let artifact = root.join("candidate.dpk");
        let output = root.join("certification.json");
        let policy = ReleasePolicy::new(AnonymizationTier::L3, None, false).unwrap();

        fs::write(&artifact, vec![b'X'; 10_240]).unwrap();
        let at_limit = certify_kernel_with_policy(&root, &artifact, &output, 1, 0, 1, &policy)
            .unwrap_err()
            .to_string();
        assert!(!at_limit.contains("byte limit"));

        fs::write(&artifact, vec![b'X'; 10_241]).unwrap();
        let over_limit = certify_kernel_with_policy(&root, &artifact, &output, 1, 0, 1, &policy)
            .unwrap_err()
            .to_string();
        assert!(over_limit.contains("byte limit"));
        assert!(!output.exists());
        fs::remove_dir_all(root).unwrap();
    }
}

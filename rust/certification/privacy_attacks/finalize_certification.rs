|| -> Result<CertificationReport> {


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



/// Selects only among validation-select candidates whose complete gate
/// decision is positive. Replications are averaged within the
/// `(structural_profile, lineage)` cohort unit before candidates are compared.
pub fn select_final_family(
    outcomes: &[DeepOutcome],
    qualifications: &[Qualification],
) -> Result<FinalFamilySelection> {
    let mut decisions = BTreeMap::new();
    for qualification in qualifications {
        if !NEW_CANDIDATES.contains(&qualification.candidate_id.as_str())
            || qualification.implementation_hash
                != candidate_implementation_hash(&qualification.candidate_id)
            || qualification.qualifies != qualification.failed_gates.is_empty()
            || decisions
                .insert(qualification.candidate_id.clone(), qualification)
                .is_some()
        {
            return Err(DopeError::Data(
                "invalid or duplicate final-family qualification".into(),
            ));
        }
    }
    let averaged = averaged_outcomes(outcomes)?;
    let mut eligible = Vec::new();
    for (candidate, qualification) in decisions {
        if !qualification.qualifies {
            continue;
        }
        let candidate_outcomes = outcomes
            .iter()
            .filter(|outcome| outcome.candidate_id == candidate)
            .collect::<Vec<_>>();
        let configurations = candidate_outcomes
            .iter()
            .filter_map(|outcome| outcome.configuration)
            .collect::<Vec<_>>();
        let unique_configurations = configurations
            .iter()
            .copied()
            .map(LossConfiguration::id)
            .collect::<BTreeSet<_>>();
        if configurations.len() != candidate_outcomes.len() || unique_configurations.len() != 1 {
            return Err(DopeError::Data(format!(
                "validation-select configuration is not frozen for {candidate}"
            )));
        }
        let configuration = configurations[0];
        let cohort_values = averaged
            .iter()
            .filter(|((_, _, id), _)| id == &candidate)
            .filter_map(|(_, outcome)| {
                outcome
                    .metrics
                    .bounded_retention
                    .zip(outcome.metrics.regret)
                    .map(|(retention, regret)| (outcome.profile.clone(), retention, regret))
            })
            .collect::<Vec<_>>();
        if cohort_values.is_empty() {
            return Err(DopeError::Data(format!(
                "qualified candidate {candidate} has no complete validation-select cohort units"
            )));
        }
        let mean_bounded_retention = cohort_values
            .iter()
            .map(|(_, retention, _)| retention)
            .sum::<f64>()
            / cohort_values.len() as f64;
        let mut profile_regret = BTreeMap::<&str, (f64, usize)>::new();
        for (profile, _, regret) in &cohort_values {
            let entry = profile_regret.entry(profile).or_default();
            entry.0 += regret;
            entry.1 += 1;
        }
        let worst_profile_regret = profile_regret
            .into_values()
            .map(|(sum, count)| sum / count as f64)
            .reduce(f64::max)
            .ok_or_else(|| DopeError::Data("missing final profile regret".into()))?;
        let maximum_artifact_bytes = candidate_outcomes
            .iter()
            .filter_map(|outcome| outcome.artifact_bytes)
            .max()
            .ok_or_else(|| DopeError::Data("missing final artifact measurement".into()))?;
        let mut runtime_groups = BTreeMap::<(&str, &str), (u128, usize)>::new();
        for outcome in &candidate_outcomes {
            let runtime = outcome
                .runtime_ms
                .ok_or_else(|| DopeError::Data("missing final runtime measurement".into()))?;
            let entry = runtime_groups
                .entry((&outcome.structural_profile, &outcome.lineage_group_id))
                .or_default();
            entry.0 += u128::from(runtime);
            entry.1 += 1;
        }
        let mean_runtime_ms = runtime_groups
            .values()
            .map(|(sum, count)| *sum as f64 / *count as f64)
            .sum::<f64>()
            / runtime_groups.len() as f64;
        eligible.push(FinalCandidateScore {
            candidate_id: candidate.clone(),
            implementation_hash: qualification.implementation_hash.clone(),
            configuration,
            cohort_units: cohort_values.len(),
            mean_bounded_retention,
            worst_profile_regret,
            maximum_artifact_bytes,
            mean_runtime_ms,
        });
    }
    eligible.sort_by(|left, right| {
        right
            .mean_bounded_retention
            .total_cmp(&left.mean_bounded_retention)
            .then_with(|| {
                left.worst_profile_regret
                    .total_cmp(&right.worst_profile_regret)
            })
            .then_with(|| {
                left.maximum_artifact_bytes
                    .cmp(&right.maximum_artifact_bytes)
            })
            .then_with(|| left.mean_runtime_ms.total_cmp(&right.mean_runtime_ms))
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    let winner = eligible.first().cloned();
    Ok(FinalFamilySelection {
        format: "dope-deep-final-family-selection".into(),
        version: 1,
        recommendation: winner.as_ref().map_or_else(
            || "retain_current_frontier".into(),
            |winner| format!("promote_dataset_specific_{}", winner.candidate_id),
        ),
        winner,
        eligible_candidates: eligible,
    })
}

fn one_sided_regression_pvalue(values: &[f64], margin: f64) -> f64 {
    if values.len() < 2 {
        return 1.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let error = (variance / values.len() as f64).sqrt();
    if error <= f64::EPSILON {
        return f64::from(mean <= margin);
    }
    let distribution =
        StudentsT::new(0.0, 1.0, (values.len() - 1) as f64).expect("positive degrees of freedom");
    1.0 - distribution.cdf((mean - margin) / error)
}

fn holm_rejects(mut pvalues: Vec<f64>, alpha: f64) -> bool {
    pvalues.sort_by(f64::total_cmp);
    pvalues
        .first()
        .is_some_and(|pvalue| *pvalue <= alpha / pvalues.len() as f64)
}

fn profile_regret_groups(
    outcomes: &[DeepOutcome],
    candidate: &str,
) -> BTreeMap<String, Vec<(String, f64)>> {
    let mut sums = BTreeMap::<(String, String, String, usize, String), (f64, usize)>::new();
    for outcome in outcomes {
        if !matches!(outcome.candidate_id.as_str(), id if id == candidate || id == LEGACY_DEEP_CHAMPION)
        {
            continue;
        }
        let Some(regret) = outcome.metrics.as_ref().and_then(|metrics| metrics.regret) else {
            continue;
        };
        let entry = sums
            .entry((
                outcome.structural_profile.clone(),
                outcome.lineage_group_id.clone(),
                outcome.auditor_id.clone(),
                outcome.size_multiplier,
                outcome.candidate_id.clone(),
            ))
            .or_default();
        entry.0 += regret;
        entry.1 += 1;
    }
    let means = sums
        .into_iter()
        .map(|(key, (sum, count))| (key, sum / count as f64))
        .collect::<BTreeMap<_, _>>();
    let mut groups = BTreeMap::<String, Vec<(String, f64)>>::new();
    for ((profile, lineage, auditor, size, id), value) in &means {
        if id != candidate {
            continue;
        }
        if let Some(champion) = means.get(&(
            profile.clone(),
            lineage.clone(),
            auditor.clone(),
            *size,
            LEGACY_DEEP_CHAMPION.into(),
        )) {
            groups
                .entry(profile.clone())
                .or_default()
                .push((lineage.clone(), value - champion));
        }
    }
    groups
}

pub fn qualify_confirmation_candidate(
    candidate: &str,
    outcomes: &[DeepOutcome],
    scheduled_cells: usize,
) -> Result<Qualification> {
    if !NEW_CANDIDATES.contains(&candidate) || scheduled_cells == 0 {
        return Err(DopeError::Data(
            "confirmation qualification requires a scheduled new candidate".into(),
        ));
    }
    for outcome in outcomes {
        outcome.validate()?;
    }
    let averaged = averaged_outcomes(outcomes)?;
    let lift = paired_interval(&metric_differences(&averaged, candidate, |metric| {
        metric.bounded_retention
    }));
    let mut cell_sums = BTreeMap::<(String, String, String, usize, String), (f64, usize)>::new();
    for outcome in outcomes {
        let Some(retention) = outcome
            .metrics
            .as_ref()
            .and_then(|metrics| metrics.bounded_retention)
        else {
            continue;
        };
        let entry = cell_sums
            .entry((
                outcome.lineage_group_id.clone(),
                outcome.structural_profile.clone(),
                outcome.auditor_id.clone(),
                outcome.size_multiplier,
                outcome.candidate_id.clone(),
            ))
            .or_default();
        entry.0 += retention;
        entry.1 += 1;
    }
    let cell_means = cell_sums
        .into_iter()
        .map(|(key, (sum, count))| (key, sum / count as f64))
        .collect::<BTreeMap<_, _>>();
    let mut oracle_by_lineage = BTreeMap::<(String, String), (f64, usize)>::new();
    for ((lineage, profile, auditor, size, id), retention) in &cell_means {
        if id != candidate {
            continue;
        }
        let oracle_values = LEGACY_DEEP_CANDIDATES
            .iter()
            .filter_map(|legacy| {
                cell_means.get(&(
                    lineage.clone(),
                    profile.clone(),
                    auditor.clone(),
                    *size,
                    (*legacy).into(),
                ))
            })
            .copied()
            .collect::<Vec<_>>();
        if oracle_values.len() == LEGACY_DEEP_CANDIDATES.len() {
            let oracle = oracle_values.into_iter().reduce(f64::max).unwrap();
            let entry = oracle_by_lineage
                .entry((profile.clone(), lineage.clone()))
                .or_default();
            entry.0 += retention - oracle;
            entry.1 += 1;
        }
    }
    let oracle_differences = oracle_by_lineage
        .into_values()
        .map(|(sum, count)| sum / count as f64)
        .collect::<Vec<_>>();
    let oracle_lift = (!oracle_differences.is_empty())
        .then(|| oracle_differences.iter().sum::<f64>() / oracle_differences.len() as f64);
    let candidate_outcomes = outcomes
        .iter()
        .filter(|outcome| outcome.candidate_id == candidate)
        .collect::<Vec<_>>();
    let frozen_configurations = candidate_outcomes
        .iter()
        .filter_map(|outcome| outcome.configuration.map(LossConfiguration::id))
        .collect::<BTreeSet<_>>();
    let successes = candidate_outcomes
        .iter()
        .filter(|outcome| outcome.state == EvidenceState::Succeeded)
        .count();
    let success_coverage = successes as f64 / scheduled_cells as f64;
    let mut profile_attempts = BTreeMap::<&str, (usize, usize)>::new();
    for outcome in &candidate_outcomes {
        let entry = profile_attempts
            .entry(&outcome.structural_profile)
            .or_default();
        entry.0 += 1;
        entry.1 += usize::from(outcome.state == EvidenceState::Succeeded);
    }
    let minimum_profile_coverage = profile_attempts
        .values()
        .map(|(attempted, succeeded)| *succeeded as f64 / *attempted as f64)
        .reduce(f64::min)
        .unwrap_or(0.0);
    let maximum_artifact_bytes = candidate_outcomes
        .iter()
        .filter_map(|outcome| outcome.artifact_bytes)
        .max();
    let feature_importance_groups = averaged
        .iter()
        .filter(|((_, _, id), _)| id == candidate)
        .count();
    let feature_importance_informative_groups = averaged
        .iter()
        .filter(|((_, _, id), value)| {
            id == candidate && value.metrics.feature_importance_spearman.is_some()
        })
        .count();
    let feature_importance_coverage =
        feature_importance_informative_groups as f64 / feature_importance_groups.max(1) as f64;
    let feature_importance_spearman_delta =
        paired_interval(&metric_differences(&averaged, candidate, |metric| {
            metric.feature_importance_spearman
        }));
    let feature_importance_top_k_delta =
        paired_interval(&metric_differences(&averaged, candidate, |metric| {
            metric.feature_importance_top_k_agreement
        }));
    let mut failed_gates = Vec::new();
    if frozen_configurations.len() != 1
        || candidate_outcomes
            .iter()
            .any(|outcome| outcome.configuration.is_none())
    {
        failed_gates.push("configuration_not_frozen".into());
    }
    if lift.is_none_or(|interval| interval.mean < 0.10 || interval.lower_one_sided_95 <= 0.0) {
        failed_gates.push("overall_retention_lift".into());
    }
    if oracle_lift.is_none_or(|value| value < 0.0) {
        failed_gates.push("legacy_deep_oracle_lift".into());
    }
    let mut regression_pvalues = Vec::new();
    for (profile, regret_groups) in profile_regret_groups(outcomes, candidate) {
        let regret_differences = regret_groups
            .iter()
            .map(|(_, difference)| *difference)
            .collect::<Vec<_>>();
        let informative = regret_differences
            .iter()
            .filter(|value| value.abs() > 1e-12)
            .count();
        if regret_differences.len() >= 100
            && informative >= 30
            && regret_differences.iter().sum::<f64>() / regret_differences.len() as f64 >= 0.0
        {
            failed_gates.push(format!("profile_regret_not_lower:{profile}"));
        }
        let mut clustered = BTreeMap::<String, (f64, usize)>::new();
        for (lineage, difference) in regret_groups {
            let entry = clustered.entry(lineage).or_default();
            entry.0 += difference;
            entry.1 += 1;
        }
        let clustered = clustered
            .into_values()
            .map(|(sum, count)| sum / count as f64)
            .collect::<Vec<_>>();
        if !clustered.is_empty() {
            regression_pvalues.push(one_sided_regression_pvalue(&clustered, 0.01));
        }
    }
    if holm_rejects(regression_pvalues, 0.05) {
        failed_gates.push("holm_supported_profile_regret_regression".into());
    }
    if candidate_outcomes.len() != scheduled_cells
        || success_coverage < 0.99
        || minimum_profile_coverage < 0.95
    {
        failed_gates.push("cell_coverage".into());
    }
    if maximum_artifact_bytes.is_none_or(|bytes| bytes > MAX_NEURAL_ARTIFACT_BYTES as u64) {
        failed_gates.push("pilot_artifact_size".into());
    }
    if candidate_outcomes.iter().any(|outcome| {
        outcome.invalid_rows > 0 || outcome.schema_violations > 0 || outcome.nondeterministic_output
    }) {
        failed_gates.push("validity_or_determinism".into());
    }
    if !safeguards_noninferior(&averaged, candidate) {
        failed_gates.push("privacy_or_fidelity_noninferiority".into());
    }
    if feature_importance_coverage < 0.80
        || feature_importance_spearman_delta
            .is_none_or(|interval| interval.lower_one_sided_95 < -0.01)
        || feature_importance_top_k_delta.is_none_or(|interval| interval.lower_one_sided_95 < -0.01)
    {
        failed_gates.push("feature_importance_consistency".into());
    }
    Ok(Qualification {
        candidate_id: candidate.into(),
        implementation_hash: candidate_implementation_hash(candidate),
        qualifies: failed_gates.is_empty(),
        overall_lift: lift,
        oracle_lift,
        success_coverage,
        minimum_profile_coverage,
        feature_importance_coverage,
        feature_importance_spearman_delta,
        feature_importance_top_k_delta,
        maximum_artifact_bytes,
        failed_gates,
    })
}

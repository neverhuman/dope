
fn add_cell_comparisons(outcomes: &mut [DeepOutcome]) {
    let measured = outcomes
        .iter()
        .filter_map(|outcome| {
            outcome
                .metrics
                .as_ref()
                .and_then(|metrics| metrics.bounded_retention)
                .map(|retention| {
                    (
                        (
                            outcome.lineage_group_id.clone(),
                            outcome.structural_profile.clone(),
                            outcome.auditor_id.clone(),
                            outcome.size_multiplier,
                            outcome.generation_seed,
                            outcome.auditor_seed,
                            outcome.candidate_id.clone(),
                        ),
                        retention,
                    )
                })
        })
        .collect::<BTreeMap<_, _>>();
    for outcome in outcomes {
        let Some(metrics) = outcome.metrics.as_mut() else {
            continue;
        };
        let cell = |candidate: &str| {
            measured.get(&(
                outcome.lineage_group_id.clone(),
                outcome.structural_profile.clone(),
                outcome.auditor_id.clone(),
                outcome.size_multiplier,
                outcome.generation_seed,
                outcome.auditor_seed,
                candidate.to_string(),
            ))
        };
        let Some(retention) = metrics.bounded_retention else {
            continue;
        };
        metrics.absolute_lift = cell(LEGACY_DEEP_CHAMPION).map(|champion| retention - champion);
        let oracle = NEW_CANDIDATES
            .into_iter()
            .chain(LEGACY_DEEP_CANDIDATES)
            .chain(REFERENCE_FRONTIER)
            .filter_map(cell)
            .copied()
            .reduce(f64::max);
        metrics.regret = oracle.map(|oracle| oracle - retention);
        metrics.oracle_recall = oracle.map(|oracle| f64::from(retention >= oracle - 1e-12));
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct PairedInterval {
    pub groups: usize,
    pub mean: f64,
    pub lower_one_sided_95: f64,
    pub upper_one_sided_95: f64,
}

fn paired_interval(values: &[f64]) -> Option<PairedInterval> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() == 1 {
        return Some(PairedInterval {
            groups: 1,
            mean,
            lower_one_sided_95: f64::NEG_INFINITY,
            upper_one_sided_95: f64::INFINITY,
        });
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let critical = StudentsT::new(0.0, 1.0, (values.len() - 1) as f64)
        .ok()?
        .inverse_cdf(0.95);
    let margin = critical * (variance / values.len() as f64).sqrt();
    Some(PairedInterval {
        groups: values.len(),
        mean,
        lower_one_sided_95: mean - margin,
        upper_one_sided_95: mean + margin,
    })
}

#[derive(Clone, Debug)]
struct AveragedOutcome {
    profile: String,
    candidate: String,
    metrics: DeepMetrics,
}

fn averaged_outcomes(
    outcomes: &[DeepOutcome],
) -> Result<BTreeMap<(String, String, String), AveragedOutcome>> {
    let mut sums =
        BTreeMap::<(String, String, String), (DeepMetrics, usize, Option<LossConfiguration>)>::new(
        );
    for outcome in outcomes {
        outcome.validate()?;
        if let Some(metrics) = &outcome.metrics {
            let key = (
                outcome.lineage_group_id.clone(),
                outcome.structural_profile.clone(),
                outcome.candidate_id.clone(),
            );
            match sums.entry(key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((metrics.clone(), 1, outcome.configuration));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if entry.get().2 != outcome.configuration {
                        return Err(DopeError::Data(
                            "candidate evidence mixes discovery configurations".into(),
                        ));
                    }
                    entry.get_mut().0.add_assign(metrics);
                    entry.get_mut().1 += 1;
                }
            }
        }
    }
    Ok(sums
        .into_iter()
        .map(|(key, (mut metrics, count, _configuration))| {
            metrics.divide(count as f64);
            (
                key,
                AveragedOutcome {
                    profile: String::new(),
                    candidate: String::new(),
                    metrics,
                },
            )
        })
        .map(|((lineage, profile, candidate), mut value)| {
            value.profile = profile.clone();
            value.candidate = candidate.clone();
            ((lineage, profile, candidate), value)
        })
        .collect())
}

fn metric_differences(
    averaged: &BTreeMap<(String, String, String), AveragedOutcome>,
    candidate: &str,
    metric: impl Fn(&DeepMetrics) -> Option<f64>,
) -> Vec<f64> {
    averaged
        .iter()
        .filter(|((_, _, id), _)| id == candidate)
        .filter_map(|((lineage, profile, _), value)| {
            averaged
                .get(&(
                    lineage.clone(),
                    profile.clone(),
                    LEGACY_DEEP_CHAMPION.into(),
                ))
                .and_then(|champion| {
                    metric(&value.metrics)
                        .zip(metric(&champion.metrics))
                        .map(|(candidate, champion)| candidate - champion)
                })
        })
        .collect()
}

fn safeguards_noninferior(
    averaged: &BTreeMap<(String, String, String), AveragedOutcome>,
    candidate: &str,
) -> bool {
    let upper_bounded = [
        |m: &DeepMetrics| m.calibration_degradation,
        |m: &DeepMetrics| m.query_p95_error,
        |m: &DeepMetrics| m.type_i_error,
        |m: &DeepMetrics| m.membership_auc,
        |m: &DeepMetrics| m.attribute_inference_advantage,
    ]
    .iter()
    .all(|metric| {
        paired_interval(&metric_differences(averaged, candidate, metric))
            .is_some_and(|interval| interval.upper_one_sided_95 <= 0.01)
    });
    let lower_bounded = [
        |m: &DeepMetrics| m.driver_agreement,
        |m: &DeepMetrics| m.joint_fidelity,
        |m: &DeepMetrics| m.rare_tail_retention,
        |m: &DeepMetrics| m.supported_subgroup_retention,
        |m: &DeepMetrics| m.feature_importance_spearman,
        |m: &DeepMetrics| m.feature_importance_top_k_agreement,
    ]
    .iter()
    .all(|metric| {
        paired_interval(&metric_differences(averaged, candidate, metric))
            .is_some_and(|interval| interval.lower_one_sided_95 >= -0.01)
    });
    let counts_safe = averaged
        .iter()
        .filter(|((_, _, id), _)| id == candidate)
        .all(|((lineage, profile, _), value)| {
            averaged
                .get(&(
                    lineage.clone(),
                    profile.clone(),
                    LEGACY_DEEP_CHAMPION.into(),
                ))
                .is_some_and(|champion| {
                    value
                        .metrics
                        .exact_copies
                        .zip(champion.metrics.exact_copies)
                        .is_some_and(|(candidate, champion)| candidate <= champion)
                        && value
                            .metrics
                            .near_copies
                            .zip(champion.metrics.near_copies)
                            .is_some_and(|(candidate, champion)| candidate <= champion)
                        && value
                            .metrics
                            .canary_extractions
                            .zip(champion.metrics.canary_extractions)
                            .is_some_and(|(candidate, champion)| candidate <= champion)
                        && value
                            .metrics
                            .lineage_leaks
                            .zip(champion.metrics.lineage_leaks)
                            .is_some_and(|(candidate, champion)| candidate <= champion)
                        && value.metrics.canary_extractions == Some(0)
                        && value.metrics.lineage_leaks == Some(0)
                })
        });
    upper_bounded && lower_bounded && counts_safe
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DiscoverySelection {
    pub candidate_id: String,
    pub implementation_hash: String,
    pub configuration: Option<LossConfiguration>,
    pub macro_retention: Option<f64>,
    pub safe: bool,
    pub failure: Option<String>,
}

pub fn select_discovery_configurations(
    outcomes: &[DeepOutcome],
) -> Result<Vec<DiscoverySelection>> {
    let mut selections = Vec::new();
    for architecture in NEW_CANDIDATES {
        let mut safe = Vec::new();
        for configuration in LossConfiguration::GRID {
            let candidate_cells = outcomes
                .iter()
                .filter(|outcome| {
                    outcome.candidate_id == architecture
                        && outcome.configuration == Some(configuration)
                })
                .collect::<Vec<_>>();
            let champion_cells = outcomes
                .iter()
                .filter(|outcome| outcome.candidate_id == LEGACY_DEEP_CHAMPION)
                .collect::<Vec<_>>();
            if candidate_cells.is_empty()
                || candidate_cells.len() != champion_cells.len()
                || candidate_cells
                    .iter()
                    .any(|outcome| outcome.state != EvidenceState::Succeeded)
                || champion_cells
                    .iter()
                    .any(|outcome| outcome.state != EvidenceState::Succeeded)
            {
                continue;
            }
            let filtered = outcomes
                .iter()
                .filter(|outcome| {
                    outcome.candidate_id == architecture
                        && outcome.configuration == Some(configuration)
                        || outcome.candidate_id == LEGACY_DEEP_CHAMPION
                })
                .cloned()
                .collect::<Vec<_>>();
            let averaged = averaged_outcomes(&filtered)?;
            if safeguards_noninferior(&averaged, architecture) {
                let profile_means = averaged
                    .iter()
                    .filter(|((_, _, candidate), _)| candidate == architecture)
                    .fold(BTreeMap::<&str, Vec<f64>>::new(), |mut map, (_, value)| {
                        if let Some(retention) = value.metrics.bounded_retention {
                            map.entry(&value.profile).or_default().push(retention);
                        }
                        map
                    });
                if !profile_means.is_empty() {
                    let macro_retention = profile_means
                        .values()
                        .map(|values| values.iter().sum::<f64>() / values.len() as f64)
                        .sum::<f64>()
                        / profile_means.len() as f64;
                    let artifact = filtered
                        .iter()
                        .filter(|outcome| outcome.candidate_id == architecture)
                        .filter_map(|outcome| outcome.artifact_bytes)
                        .max()
                        .unwrap_or(u64::MAX);
                    let runtime = filtered
                        .iter()
                        .filter(|outcome| outcome.candidate_id == architecture)
                        .filter_map(|outcome| outcome.runtime_ms)
                        .max()
                        .unwrap_or(u64::MAX);
                    safe.push((configuration, macro_retention, artifact, runtime));
                }
            }
        }
        safe.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.2.cmp(&right.2))
                .then_with(|| left.3.cmp(&right.3))
                .then_with(|| left.0.id().cmp(&right.0.id()))
        });
        selections.push(
            if let Some((configuration, retention, _, _)) = safe.first() {
                DiscoverySelection {
                    candidate_id: architecture.into(),
                    implementation_hash: candidate_implementation_hash(architecture),
                    configuration: Some(*configuration),
                    macro_retention: Some(*retention),
                    safe: true,
                    failure: None,
                }
            } else {
                DiscoverySelection {
                    candidate_id: architecture.into(),
                    implementation_hash: candidate_implementation_hash(architecture),
                    configuration: None,
                    macro_retention: None,
                    safe: false,
                    failure: Some("no loss configuration passed safeguard noninferiority".into()),
                }
            },
        );
    }
    Ok(selections)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Qualification {
    pub candidate_id: String,
    pub implementation_hash: String,
    pub qualifies: bool,
    pub overall_lift: Option<PairedInterval>,
    pub oracle_lift: Option<f64>,
    pub success_coverage: f64,
    pub minimum_profile_coverage: f64,
    pub feature_importance_coverage: f64,
    pub feature_importance_spearman_delta: Option<PairedInterval>,
    pub feature_importance_top_k_delta: Option<PairedInterval>,
    pub maximum_artifact_bytes: Option<u64>,
    pub failed_gates: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FinalCandidateScore {
    pub candidate_id: String,
    pub implementation_hash: String,
    pub configuration: LossConfiguration,
    pub cohort_units: usize,
    pub mean_bounded_retention: f64,
    pub worst_profile_regret: f64,
    pub maximum_artifact_bytes: u64,
    pub mean_runtime_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FinalFamilySelection {
    pub format: String,
    pub version: u8,
    pub winner: Option<FinalCandidateScore>,
    pub eligible_candidates: Vec<FinalCandidateScore>,
    pub recommendation: String,
}
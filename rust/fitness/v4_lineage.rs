pub const V4_SAMPLE_SEEDS: [u64; 3] = [101, 211, 307];

pub const V4_SEED_VARIANCE_METRICS: [&str; 13] = [
    "retention_mean",
    "retention_sample_sd",
    "retention_min",
    "retention_max",
    "retention_lcb",
    "driver_fidelity_mean",
    "driver_fidelity_sample_sd",
    "driver_fidelity_min",
    "driver_fidelity_max",
    "seed_score_mean",
    "seed_score_sample_sd",
    "seed_score_min",
    "seed_score_max",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct V4LineageAggregationContract {
    pub sample_seeds: Vec<u64>,
    pub all_seeds_must_pass_gates: bool,
    pub utility_bound_rule: String,
    pub control_baseline_aggregation: String,
    pub driver_attribution_aggregation: String,
    pub representation_aggregation: String,
    pub compression_aggregation: String,
    pub fidelity_aggregation: String,
    pub privacy_aggregation: String,
    pub seed_variance_reported: bool,
    pub seed_variance_metrics: Vec<String>,
}

impl V4LineageAggregationContract {
    pub fn is_frozen(&self) -> bool {
        self.sample_seeds == V4_SAMPLE_SEEDS
            && self.all_seeds_must_pass_gates
            && self.utility_bound_rule == "lcb_le_mean_plus_1e_12"
            && self.control_baseline_aggregation == "max_across_seeds"
            && self.driver_attribution_aggregation == "min_composite_driver_fidelity_seed"
            && self.representation_aggregation == "min_representation_closeness_seed"
            && self.compression_aggregation == "max_artifact_bytes_and_encoded_numbers_across_seeds"
            && self.fidelity_aggregation == "min_across_seeds"
            && self.privacy_aggregation == "max_risk_across_seeds"
            && self.seed_variance_reported
            && self
                .seed_variance_metrics
                .iter()
                .map(String::as_str)
                .eq(V4_SEED_VARIANCE_METRICS)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct V4SeedObservation {
    pub sample_seed: u64,
    pub components: ScoredComponentsV4,
    pub observation: MfsV4Observation,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct V4SeedVarianceSummary {
    pub sample_seeds: Vec<u64>,
    pub retention_mean: f64,
    pub retention_sample_sd: f64,
    pub retention_min: f64,
    pub retention_max: f64,
    pub retention_lcb: f64,
    pub driver_fidelity_mean: f64,
    pub driver_fidelity_sample_sd: f64,
    pub driver_fidelity_min: f64,
    pub driver_fidelity_max: f64,
    pub seed_score_mean: f64,
    pub seed_score_sample_sd: f64,
    pub seed_score_min: f64,
    pub seed_score_max: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MasterFitnessV4LineageReport {
    pub report: MasterFitnessV4Report,
    pub seed_variance: Option<V4SeedVarianceSummary>,
    pub seed_reports: Vec<MasterFitnessV4Report>,
}

fn sample_mean_and_sd(values: &[f64]) -> Option<(f64, f64)> {
    if values.len() < 2 || values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mean = values.iter().sum::<f64>() / (values.len() as f64);
    if !mean.is_finite() {
        return None;
    }
    let var = values
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / ((values.len() - 1) as f64);
    if !var.is_finite() {
        return None;
    }
    Some((mean, var.max(0.0).sqrt()))
}

/// Evaluates an MFS-v4 lineage across the declared sample seeds (`[101, 211, 307]`).
/// Every seed must individually pass `evaluate_v4`, share identical table/protocol
/// invariants, and the one-sided lower confidence bound (`lower_bound`, bounded by
/// `<= mean(retention) + 1e-12`) must beat the worst-seed (`max`) control baselines,
/// while driver attribution, representation closeness, knowledge compression,
/// fidelity, and privacy use worst-case aggregation across seeds.
pub fn evaluate_v4_lineage<F>(
    contract: &MasterFitnessV4Contract,
    seeds: &[V4SeedObservation],
    lower_bound: F,
) -> MasterFitnessV4LineageReport
where
    F: FnOnce(&[f64]) -> Option<f64>,
{
    let mut ordered: Vec<usize> = (0..seeds.len()).collect();
    ordered.sort_by_key(|&i| seeds[i].sample_seed);
    let complete_seed_set = ordered.len() == V4_SAMPLE_SEEDS.len()
        && ordered
            .iter()
            .map(|&i| seeds[i].sample_seed)
            .eq(V4_SAMPLE_SEEDS);
    let idx: Vec<usize> = if complete_seed_set {
        ordered
    } else {
        (0..seeds.len()).collect()
    };
    let seed_reports: Vec<MasterFitnessV4Report> = idx
        .iter()
        .map(|&i| evaluate_v4(contract, &seeds[i].components, &seeds[i].observation))
        .collect();
    if seeds.is_empty() {
        let hard_gates = BTreeMap::from([
            ("complete_seed_set".into(), false),
            ("lineage_schema_consistent".into(), false),
            ("all_seeds_eligible".into(), false),
            ("utility_lcb_valid".into(), false),
        ]);
        return MasterFitnessV4LineageReport {
            report: MasterFitnessV4Report {
                version: contract.version,
                eligible: false,
                hard_gates,
                score: None,
                utility_transfer: None,
                driver_fidelity: None,
                representation_closeness: None,
                knowledge_compression: None,
                control_contrastive_lift: None,
                driver_attribution: None,
                compression_metrics: None,
                unavailable_terms: contract
                    .weights
                    .entries()
                    .into_iter()
                    .map(|(n, _)| n.into())
                    .collect(),
            },
            seed_variance: None,
            seed_reports,
        };
    }

    let first = &seeds[idx[0]].observation;
    let schema_consistent = complete_seed_set
        && idx.iter().all(|&i| {
            let obs = &seeds[i].observation;
            (
                obs.real_rows,
                obs.real_columns,
                obs.real_importances.len(),
                obs.synthetic_importances.len(),
            ) == (
                first.real_rows,
                first.real_columns,
                first.real_importances.len(),
                first.synthetic_importances.len(),
            ) && obs.normalizer == first.normalizer
                && obs.utility_protocol == first.utility_protocol
                && obs.utility_auditors == first.utility_auditors
                && obs.encoder == first.encoder
        });
    let all_seeds_eligible = complete_seed_set
        && schema_consistent
        && seed_reports.iter().all(|r| r.eligible && r.score.is_some());

    let retentions: Vec<f64> = idx.iter().map(|&i| seeds[i].observation.retention).collect();
    let ret_stats = complete_seed_set
        .then(|| sample_mean_and_sd(&retentions))
        .flatten();
    let valid_lcb = ret_stats
        .and_then(|(mean, _)| lower_bound(&retentions).map(|val| (val, mean)))
        .filter(|&(val, mean)| val.is_finite() && val <= mean + 1e-12)
        .map(|(val, _)| val.clamp(0.0, 1.0));

    let worst_drv = (complete_seed_set
        && schema_consistent
        && seed_reports
            .iter()
            .all(|r| r.driver_fidelity.is_some() && r.driver_attribution.is_some()))
    .then(|| {
        seed_reports
            .iter()
            .enumerate()
            .min_by(|&(ia, ra), &(ib, rb)| {
                ra.driver_fidelity
                    .unwrap_or(-1.0)
                    .total_cmp(&rb.driver_fidelity.unwrap_or(-1.0))
                    .then_with(|| {
                        ra.driver_attribution
                            .map(|d| d.top_k_overlap)
                            .unwrap_or(-1.0)
                            .total_cmp(
                                &rb.driver_attribution
                                    .map(|d| d.top_k_overlap)
                                    .unwrap_or(-1.0),
                            )
                    })
                    .then_with(|| seeds[idx[ia]].sample_seed.cmp(&seeds[idx[ib]].sample_seed))
            })
            .map(|(pos, _)| &seeds[idx[pos]].observation)
    })
    .flatten();
    let worst_repr = (complete_seed_set
        && seed_reports
            .iter()
            .all(|r| r.representation_closeness.is_some()))
    .then(|| {
        seed_reports
            .iter()
            .enumerate()
            .min_by(|&(ia, ra), &(ib, rb)| {
                ra.representation_closeness
                    .unwrap_or(-1.0)
                    .total_cmp(&rb.representation_closeness.unwrap_or(-1.0))
                    .then_with(|| seeds[idx[ia]].sample_seed.cmp(&seeds[idx[ib]].sample_seed))
            })
            .map(|(pos, _)| &seeds[idx[pos]].observation)
    })
    .flatten();

    let finite_max = |f: fn(&MfsV4Observation) -> f64| {
        if complete_seed_set && idx.iter().all(|&i| f(&seeds[i].observation).is_finite()) {
            idx.iter()
                .map(|&i| f(&seeds[i].observation))
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            f64::NAN
        }
    };
    let unit_max = |f: fn(&MfsV4Observation) -> f64| {
        if complete_seed_set
            && idx.iter().all(|&i| {
                let v = f(&seeds[i].observation);
                v.is_finite() && (0.0..=1.0).contains(&v)
            })
        {
            idx.iter()
                .map(|&i| f(&seeds[i].observation))
                .fold(0.0, f64::max)
        } else {
            f64::NAN
        }
    };
    let unit_min = |f: fn(&ScoredComponentsV4) -> Option<f64>| {
        complete_seed_set
            .then(|| {
                idx.iter()
                    .map(|&i| component_in_unit(f(&seeds[i].components)))
                    .collect::<Option<Vec<_>>>()
                    .and_then(|vals| vals.into_iter().reduce(f64::min))
            })
            .flatten()
    };
    let bootstrap_4n = if complete_seed_set
        && idx
            .iter()
            .all(|&i| seeds[i].observation.real_bootstrap_4n_retention.is_some())
    {
        Some(
            if idx.iter().all(|&i| {
                seeds[i]
                    .observation
                    .real_bootstrap_4n_retention
                    .is_some_and(f64::is_finite)
            }) {
                idx.iter()
                    .filter_map(|&i| seeds[i].observation.real_bootstrap_4n_retention)
                    .fold(f64::NEG_INFINITY, f64::max)
            } else {
                f64::NAN
            },
        )
    } else if complete_seed_set
        && idx.iter().any(|&i| {
            seeds[i]
                .observation
                .real_bootstrap_4n_retention
                .is_some_and(|v| !v.is_finite())
        })
    {
        Some(f64::NAN)
    } else {
        None
    };

    let agg_components = ScoredComponentsV4 {
        distribution_fidelity: unit_min(|c| c.distribution_fidelity),
        structure_fidelity: unit_min(|c| c.structure_fidelity),
        coverage_realism: unit_min(|c| c.coverage_realism),
    };
    let agg_observation = MfsV4Observation {
        encoder: if complete_seed_set && idx.iter().all(|&i| seeds[i].observation.encoder == first.encoder) {
            first.encoder.clone()
        } else {
            String::new()
        },
        distance: worst_repr.map_or(f64::NAN, |o| o.distance),
        d_null: worst_repr.map_or(f64::NAN, |o| o.d_null),
        d_match: worst_repr.map_or(f64::NAN, |o| o.d_match),
        gap_mitra: worst_repr.map_or(f64::NAN, |o| o.gap_mitra),
        gap_tabicl: worst_repr.map_or(f64::NAN, |o| o.gap_tabicl),
        exact_row_matches: if complete_seed_set {
            idx.iter()
                .map(|&i| seeds[i].observation.exact_row_matches)
                .max()
                .unwrap_or(1)
        } else {
            1
        },
        near_copy_ok: complete_seed_set && idx.iter().all(|&i| seeds[i].observation.near_copy_ok),
        cleartext_absent: complete_seed_set
            && idx.iter().all(|&i| seeds[i].observation.cleartext_absent),
        membership_auc: unit_max(|o| o.membership_auc),
        attribute_inference_advantage: unit_max(|o| o.attribute_inference_advantage),
        artifact_bytes: if complete_seed_set
            && idx
                .iter()
                .all(|&i| (1..=10_240).contains(&seeds[i].observation.artifact_bytes))
        {
            idx.iter()
                .map(|&i| seeds[i].observation.artifact_bytes)
                .max()
                .unwrap_or(0)
        } else {
            0
        },
        encoded_numbers: if complete_seed_set
            && idx.iter().all(|&i| seeds[i].observation.encoded_numbers >= 1)
        {
            idx.iter()
                .map(|&i| seeds[i].observation.encoded_numbers)
                .max()
                .unwrap_or(0)
        } else {
            0
        },
        real_rows: if schema_consistent { first.real_rows } else { 0 },
        real_columns: if schema_consistent { first.real_columns } else { 0 },
        normalizer: if complete_seed_set && idx.iter().all(|&i| seeds[i].observation.normalizer == first.normalizer) {
            first.normalizer.clone()
        } else {
            String::new()
        },
        utility_protocol: if complete_seed_set
            && idx.iter().all(|&i| seeds[i].observation.utility_protocol == first.utility_protocol)
        {
            first.utility_protocol.clone()
        } else {
            String::new()
        },
        utility_auditors: if complete_seed_set
            && idx.iter().all(|&i| seeds[i].observation.utility_auditors == first.utility_auditors)
        {
            first.utility_auditors.clone()
        } else {
            Vec::new()
        },
        retention: valid_lcb.unwrap_or(f64::NAN),
        predictor_only_retention: finite_max(|o| o.predictor_only_retention),
        independent_marginals_retention: finite_max(|o| o.independent_marginals_retention),
        real_bootstrap_4n_retention: bootstrap_4n,
        real_importances: worst_drv.map_or_else(Vec::new, |o| o.real_importances.clone()),
        synthetic_importances: worst_drv.map_or_else(Vec::new, |o| o.synthetic_importances.clone()),
        real_signs: worst_drv.and_then(|o| o.real_signs.clone()),
        synthetic_signs: worst_drv.and_then(|o| o.synthetic_signs.clone()),
        real_vs_real_control_complete: complete_seed_set
            && idx
                .iter()
                .all(|&i| seeds[i].observation.real_vs_real_control_complete),
        required_size_cells_complete: complete_seed_set
            && idx
                .iter()
                .all(|&i| seeds[i].observation.required_size_cells_complete),
        artifact_sampling_verified: complete_seed_set
            && idx
                .iter()
                .all(|&i| seeds[i].observation.artifact_sampling_verified),
        metric_implementations_locked: complete_seed_set
            && idx
                .iter()
                .all(|&i| seeds[i].observation.metric_implementations_locked),
        control_contrastive_baselines_complete: complete_seed_set
            && idx
                .iter()
                .all(|&i| seeds[i].observation.control_contrastive_baselines_complete),
        model_driver_attribution_measured: complete_seed_set
            && idx
                .iter()
                .all(|&i| seeds[i].observation.model_driver_attribution_measured),
    };

    let mut report = evaluate_v4(contract, &agg_components, &agg_observation);
    for seed_report in &seed_reports {
        for (gate, passed) in &seed_report.hard_gates {
            if !passed {
                report.hard_gates.insert(gate.clone(), false);
            }
        }
    }
    if !report.hard_gates.get("positive_predictor_lift").copied().unwrap_or(false)
        || !report.hard_gates.get("positive_dependence_lift").copied().unwrap_or(false)
    {
        report.utility_transfer = None;
        report.knowledge_compression = None;
        for term in ["utility_transfer", "knowledge_compression"] {
            if !report.unavailable_terms.iter().any(|t| t == term) {
                report.unavailable_terms.push(term.into());
            }
        }
    }
    for (k, v) in [
        ("complete_seed_set", complete_seed_set),
        ("lineage_schema_consistent", schema_consistent),
        ("all_seeds_eligible", all_seeds_eligible),
        ("utility_lcb_valid", valid_lcb.is_some()),
    ] {
        report.hard_gates.insert(k.into(), v);
    }
    if !complete_seed_set || !schema_consistent || !all_seeds_eligible || valid_lcb.is_none() {
        report.eligible = false;
        report.score = None;
    }

    let seed_variance = all_seeds_eligible.then_some(()).and_then(|()| {
        let lcb_val = valid_lcb?;
        let (ret_mean, ret_sd) = ret_stats?;
        let drv_vals: Vec<f64> = seed_reports
            .iter()
            .filter_map(|r| r.driver_fidelity)
            .collect();
        let score_vals: Vec<f64> = seed_reports.iter().filter_map(|r| r.score).collect();
        let (drv_mean, drv_sd) = sample_mean_and_sd(&drv_vals)?;
        let (score_mean, score_sd) = sample_mean_and_sd(&score_vals)?;
        Some(V4SeedVarianceSummary {
            sample_seeds: V4_SAMPLE_SEEDS.to_vec(),
            retention_mean: ret_mean,
            retention_sample_sd: ret_sd,
            retention_min: retentions.iter().copied().fold(f64::INFINITY, f64::min),
            retention_max: retentions.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            retention_lcb: lcb_val,
            driver_fidelity_mean: drv_mean,
            driver_fidelity_sample_sd: drv_sd,
            driver_fidelity_min: drv_vals.iter().copied().fold(f64::INFINITY, f64::min),
            driver_fidelity_max: drv_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            seed_score_mean: score_mean,
            seed_score_sample_sd: score_sd,
            seed_score_min: score_vals.iter().copied().fold(f64::INFINITY, f64::min),
            seed_score_max: score_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        })
    });

    MasterFitnessV4LineageReport {
        report,
        seed_variance,
        seed_reports,
    }
}

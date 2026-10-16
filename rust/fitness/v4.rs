/// MFS-v4 upgrades MFS-v3 with control-contrastive utility lift (over
/// `predictor_only`, `independent_marginals`, and `real_bootstrap_4n`),
/// holdout model-driver attribution agreement (NDCG@k, Top-k Jaccard, L1
/// attribution share, and directional/interaction sign agreement), and
/// MDL knowledge compression (rewarding fewer encoded numbers and smaller
/// byte payloads alongside the hard 10,240-byte L3 gate).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FitnessWeightsV4 {
    pub utility_transfer: f64,
    pub driver_fidelity: f64,
    pub representation_closeness: f64,
    pub knowledge_compression: f64,
    pub distribution_fidelity: f64,
    pub structure_fidelity: f64,
    pub coverage_realism: f64,
}

impl FitnessWeightsV4 {
    fn entries(&self) -> [(&'static str, f64); 7] {
        [
            ("utility_transfer", self.utility_transfer),
            ("driver_fidelity", self.driver_fidelity),
            ("representation_closeness", self.representation_closeness),
            ("knowledge_compression", self.knowledge_compression),
            ("distribution_fidelity", self.distribution_fidelity),
            ("structure_fidelity", self.structure_fidelity),
            ("coverage_realism", self.coverage_realism),
        ]
    }

    pub fn is_valid(&self) -> bool {
        self.entries()
            .iter()
            .all(|(_, weight)| weight.is_finite() && *weight > 0.0)
            && (self.entries().iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() <= 1e-12
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MasterFitnessV4Contract {
    pub version: u8,
    pub scalarization: String,
    pub privacy_soft_weight: f64,
    pub epsilon: f64,
    pub weights: FitnessWeightsV4,
    #[serde(default)]
    pub epsilon_rule: String,
    #[serde(default)]
    pub prerequisites: Vec<String>,
    #[serde(default)]
    pub score_range: [u8; 2],
}

pub const V4_PREREQUISITES: [&str; 6] = [
    "real_vs_real_control_complete",
    "required_size_cells_complete",
    "artifact_sampling_verified",
    "metric_implementations_locked",
    "control_contrastive_baselines_complete",
    "model_driver_attribution_measured",
];

pub const V4_TABULAR_AUDITORS: &[&str] = &["kumo_tabular_l", "catboost", "mitra_v2", "tabicl2"];

impl MasterFitnessV4Contract {
    pub fn is_frozen(&self) -> bool {
        self.version == 4
            && self.scalarization == "geometric_mean"
            && self.privacy_soft_weight == 0.0
            && self.epsilon == 1e-6
            && self.epsilon_rule == "max_component_epsilon"
            && self.score_range == [0, 100]
            && self
                .prerequisites
                .iter()
                .map(String::as_str)
                .eq(V4_PREREQUISITES)
            && self.weights.is_valid()
            && self.weights
                == FitnessWeightsV4 {
                    utility_transfer: 0.30,
                    driver_fidelity: 0.20,
                    representation_closeness: 0.15,
                    knowledge_compression: 0.15,
                    distribution_fidelity: 0.10,
                    structure_fidelity: 0.05,
                    coverage_realism: 0.05,
                }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoredComponentsV4 {
    pub distribution_fidelity: Option<f64>,
    pub structure_fidelity: Option<f64>,
    pub coverage_realism: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MfsV4Observation {
    pub encoder: String,
    pub distance: f64,
    pub d_null: f64,
    pub d_match: f64,
    pub gap_mitra: f64,
    pub gap_tabicl: f64,
    pub exact_row_matches: u64,
    pub near_copy_ok: bool,
    pub cleartext_absent: bool,
    pub membership_auc: f64,
    pub attribute_inference_advantage: f64,
    pub artifact_bytes: u64,
    pub encoded_numbers: u64,
    pub real_rows: u64,
    pub real_columns: u64,
    pub normalizer: String,
    pub utility_protocol: String,
    pub utility_auditors: Vec<String>,
    pub retention: f64,
    pub predictor_only_retention: f64,
    pub independent_marginals_retention: f64,
    pub real_bootstrap_4n_retention: Option<f64>,
    pub real_importances: Vec<f64>,
    pub synthetic_importances: Vec<f64>,
    pub real_signs: Option<Vec<i8>>,
    pub synthetic_signs: Option<Vec<i8>>,
    pub real_vs_real_control_complete: bool,
    pub required_size_cells_complete: bool,
    pub artifact_sampling_verified: bool,
    pub metric_implementations_locked: bool,
    pub control_contrastive_baselines_complete: bool,
    pub model_driver_attribution_measured: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MasterFitnessV4Report {
    pub version: u8,
    pub eligible: bool,
    pub hard_gates: BTreeMap<String, bool>,
    pub score: Option<f64>,
    pub utility_transfer: Option<f64>,
    pub driver_fidelity: Option<f64>,
    pub representation_closeness: Option<f64>,
    pub knowledge_compression: Option<f64>,
    pub control_contrastive_lift: Option<crate::transfer_metrics::ControlContrastiveLift>,
    pub driver_attribution: Option<crate::transfer_metrics::ModelDriverAttribution>,
    pub compression_metrics: Option<crate::transfer_metrics::KnowledgeCompressionMetrics>,
    pub unavailable_terms: Vec<String>,
}

fn v4_tabular_transfer_ok(normalizer: &str, protocol: &str, auditors: &[String]) -> bool {
    normalizer == crate::representation::NORMALIZER
        && protocol == crate::representation::TABULAR_PROTOCOL
        && auditors.len() == V4_TABULAR_AUDITORS.len()
        && V4_TABULAR_AUDITORS
            .iter()
            .all(|name| auditors.iter().any(|got| got == name))
}

/// Evaluates MFS-v4 from raw verified observations. Caller-supplied values for
/// `utility_transfer`, `driver_fidelity`, `representation_closeness`, and
/// `knowledge_compression` are not accepted; all four are derived from the
/// observation under the frozen v4 contract and hard release gates.
pub fn evaluate_v4(
    contract: &MasterFitnessV4Contract,
    components: &ScoredComponentsV4,
    observation: &MfsV4Observation,
) -> MasterFitnessV4Report {
    let shared_map = observation.normalizer == crate::representation::NORMALIZER;
    let closeness = if shared_map {
        crate::representation::representation_closeness(
            observation.distance,
            observation.d_null,
            observation.d_match,
            observation.gap_mitra,
            observation.gap_tabicl,
        )
    } else {
        None
    };
    let tabular_ok = v4_tabular_transfer_ok(
        &observation.normalizer,
        &observation.utility_protocol,
        &observation.utility_auditors,
    );
    let baseline_lift = if observation.control_contrastive_baselines_complete {
        crate::transfer_metrics::control_contrastive_lift(
            observation.retention,
            observation.predictor_only_retention,
            observation.independent_marginals_retention,
            observation.real_bootstrap_4n_retention,
        )
        .ok()
    } else {
        None
    };
    let lift = baseline_lift.filter(|_| tabular_ok);
    let utility_transfer = lift
        .filter(|item| item.positive_predictor_lift && item.positive_dependence_lift)
        .and_then(|item| component_in_unit(Some(item.contrastive_utility_score)));
    let driver_attr = if observation.model_driver_attribution_measured
        && (observation.real_columns as usize) > observation.real_importances.len()
    {
        crate::transfer_metrics::driver_attribution_agreement(
            &observation.real_importances,
            &observation.synthetic_importances,
            observation.real_signs.as_deref(),
            observation.synthetic_signs.as_deref(),
        )
        .ok()
    } else {
        None
    };
    let driver_fidelity = driver_attr
        .filter(|item| item.top_k_overlap > 0.0 && item.composite_driver_fidelity > 0.0)
        .and_then(|item| component_in_unit(Some(item.composite_driver_fidelity)));
    let compression = crate::transfer_metrics::knowledge_compression_metrics(
        observation.artifact_bytes,
        observation.encoded_numbers,
        observation.real_rows,
        observation.real_columns,
        10_240,
        observation.retention,
    )
    .ok();
    let knowledge_compression = compression
        .filter(|item| utility_transfer.is_some() && item.knowledge_compression_score > 0.0)
        .and_then(|item| component_in_unit(Some(item.knowledge_compression_score)));
    let distribution_fidelity = component_in_unit(components.distribution_fidelity);
    let structure_fidelity = component_in_unit(components.structure_fidelity);
    let coverage_realism = component_in_unit(components.coverage_realism);

    let mut hard_gates = BTreeMap::new();
    hard_gates.insert("exact_row_match".into(), observation.exact_row_matches == 0);
    hard_gates.insert("near_copy".into(), observation.near_copy_ok);
    hard_gates.insert("cleartext_absent".into(), observation.cleartext_absent);
    hard_gates.insert(
        "clean_encoder".into(),
        crate::representation::headline_encoder(&observation.encoder),
    );
    hard_gates.insert(
        "membership_auc".into(),
        observation.membership_auc.is_finite()
            && (0.0..=0.55).contains(&observation.membership_auc),
    );
    hard_gates.insert(
        "attribute_inference".into(),
        observation.attribute_inference_advantage.is_finite()
            && (0.0..=0.05).contains(&observation.attribute_inference_advantage),
    );
    hard_gates.insert(
        "tier_bytes".into(),
        (1..=10_240).contains(&observation.artifact_bytes),
    );
    hard_gates.insert(
        "encoded_numbers".into(),
        observation.encoded_numbers >= 1 && compression.is_some(),
    );
    hard_gates.insert(
        "positive_predictor_lift".into(),
        lift.is_some_and(|item| item.positive_predictor_lift),
    );
    hard_gates.insert(
        "positive_dependence_lift".into(),
        lift.is_some_and(|item| item.positive_dependence_lift),
    );
    for (key, passed) in V4_PREREQUISITES.into_iter().zip([
        observation.real_vs_real_control_complete,
        observation.required_size_cells_complete,
        observation.artifact_sampling_verified,
        observation.metric_implementations_locked,
        observation.control_contrastive_baselines_complete && baseline_lift.is_some(),
        observation.model_driver_attribution_measured && driver_attr.is_some(),
    ]) {
        hard_gates.insert(key.into(), passed);
    }
    hard_gates.insert("frozen_contract".into(), contract.is_frozen());
    hard_gates.insert("shared_normalizer".into(), shared_map);
    hard_gates.insert("tabular_transfer".into(), tabular_ok);
    hard_gates.insert("representation_closeness".into(), closeness.is_some());

    let named_terms = [
        ("utility_transfer", utility_transfer),
        ("driver_fidelity", driver_fidelity),
        ("representation_closeness", closeness),
        ("knowledge_compression", knowledge_compression),
        ("distribution_fidelity", distribution_fidelity),
        ("structure_fidelity", structure_fidelity),
        ("coverage_realism", coverage_realism),
    ];
    let unavailable_terms = named_terms
        .into_iter()
        .filter(|(_, value)| value.is_none())
        .map(|(name, _)| name.to_string())
        .collect::<Vec<_>>();
    let gates_pass =
        contract.is_frozen() && !hard_gates.is_empty() && hard_gates.values().all(|passed| *passed);
    let score = if gates_pass && unavailable_terms.is_empty() {
        let mut log_score = 0.0;
        for (name, value) in named_terms {
            let value = value.unwrap_or(0.0);
            let weight = contract
                .weights
                .entries()
                .into_iter()
                .find(|(key, _)| *key == name)
                .map(|(_, weight)| weight)
                .unwrap_or(0.0);
            log_score += weight * value.max(contract.epsilon).ln();
        }
        let total: f64 = contract.weights.entries().iter().map(|(_, w)| w).sum();
        (total > 0.0).then_some((100.0 * (log_score / total).exp()).clamp(0.0, 100.0))
    } else {
        None
    };

    MasterFitnessV4Report {
        version: contract.version,
        eligible: score.is_some(),
        hard_gates,
        score,
        utility_transfer,
        driver_fidelity,
        representation_closeness: closeness,
        knowledge_compression,
        control_contrastive_lift: lift,
        driver_attribution: driver_attr,
        compression_metrics: compression,
        unavailable_terms,
    }
}

pub const V4_SAMPLE_SEEDS: [u64; 3] = [101, 211, 307];

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

    let worst_drv = seed_reports
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
        .unwrap_or(first);
    let worst_repr = seed_reports
        .iter()
        .enumerate()
        .min_by(|&(ia, ra), &(ib, rb)| {
            ra.representation_closeness
                .unwrap_or(-1.0)
                .total_cmp(&rb.representation_closeness.unwrap_or(-1.0))
                .then_with(|| seeds[idx[ia]].sample_seed.cmp(&seeds[idx[ib]].sample_seed))
        })
        .map(|(pos, _)| &seeds[idx[pos]].observation)
        .unwrap_or(first);

    let finite_max = |f: fn(&MfsV4Observation) -> f64| {
        if idx.iter().all(|&i| f(&seeds[i].observation).is_finite()) {
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
        idx.iter()
            .map(|&i| component_in_unit(f(&seeds[i].components)))
            .collect::<Option<Vec<_>>>()
            .and_then(|vals| vals.into_iter().reduce(f64::min))
    };
    let bootstrap_4n = if idx
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
                    .fold(f64::INFINITY, f64::min)
            } else {
                f64::NAN
            },
        )
    } else if idx.iter().any(|&i| {
        seeds[i]
            .observation
            .real_bootstrap_4n_retention
            .is_some_and(|v| !v.is_finite())
    }) {
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
        distance: worst_repr.distance,
        d_null: worst_repr.d_null,
        d_match: worst_repr.d_match,
        gap_mitra: worst_repr.gap_mitra,
        gap_tabicl: worst_repr.gap_tabicl,
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
        real_importances: worst_drv.real_importances.clone(),
        synthetic_importances: worst_drv.synthetic_importances.clone(),
        real_signs: worst_drv.real_signs.clone(),
        synthetic_signs: worst_drv.synthetic_signs.clone(),
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


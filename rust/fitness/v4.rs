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
    let lift = if tabular_ok && observation.control_contrastive_baselines_complete {
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
    let utility_transfer = lift.and_then(|item| component_in_unit(Some(item.contrastive_utility_score)));
    let driver_attr = if observation.model_driver_attribution_measured {
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
    let driver_fidelity =
        driver_attr.and_then(|item| component_in_unit(Some(item.composite_driver_fidelity)));
    let compression = crate::transfer_metrics::knowledge_compression_metrics(
        observation.artifact_bytes,
        observation.encoded_numbers,
        observation.real_rows,
        observation.real_columns,
        10_240,
        observation.retention,
    )
    .ok();
    let knowledge_compression =
        compression.and_then(|item| component_in_unit(Some(item.knowledge_compression_score)));
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
        observation.control_contrastive_baselines_complete,
        observation.model_driver_attribution_measured,
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

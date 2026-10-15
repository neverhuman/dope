use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// MFS-v2 ranks candidates only after every release gate passes. Terms without
/// measured evidence stay absent; an absent term makes the scalar unavailable.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FitnessWeights {
    pub utility_transfer: f64,
    pub driver_fidelity: f64,
    pub distribution_fidelity: f64,
    pub structure_fidelity: f64,
    pub coverage_realism: f64,
    pub compactness: f64,
}

impl FitnessWeights {
    fn entries(&self) -> [(&'static str, f64); 6] {
        [
            ("utility_transfer", self.utility_transfer),
            ("driver_fidelity", self.driver_fidelity),
            ("distribution_fidelity", self.distribution_fidelity),
            ("structure_fidelity", self.structure_fidelity),
            ("coverage_realism", self.coverage_realism),
            ("compactness", self.compactness),
        ]
    }

    pub fn is_valid(&self) -> bool {
        self.entries()
            .iter()
            .all(|(_, weight)| weight.is_finite() && *weight >= 0.0)
            && (self.entries().iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() <= 1e-12
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MasterFitnessContract {
    pub version: u8,
    pub scalarization: String,
    pub privacy_soft_weight: f64,
    pub epsilon: f64,
    pub weights: FitnessWeights,
}

impl MasterFitnessContract {
    pub fn is_frozen_v2(&self) -> bool {
        self.version == 2
            && self.scalarization == "geometric_mean"
            && self.privacy_soft_weight == 0.0
            && self.epsilon == 1e-6
            && self.weights
                == FitnessWeights {
                    utility_transfer: 0.30,
                    driver_fidelity: 0.20,
                    distribution_fidelity: 0.20,
                    structure_fidelity: 0.15,
                    coverage_realism: 0.10,
                    compactness: 0.05,
                }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ParetoVector {
    pub utility_transfer: Option<f64>,
    pub driver_fidelity: Option<f64>,
    pub distribution_fidelity: Option<f64>,
    pub structure_fidelity: Option<f64>,
    pub coverage_realism: Option<f64>,
    pub compactness: Option<f64>,
}

impl ParetoVector {
    fn entries(&self) -> [(&'static str, Option<f64>); 6] {
        [
            ("utility_transfer", self.utility_transfer),
            ("driver_fidelity", self.driver_fidelity),
            ("distribution_fidelity", self.distribution_fidelity),
            ("structure_fidelity", self.structure_fidelity),
            ("coverage_realism", self.coverage_realism),
            ("compactness", self.compactness),
        ]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MasterFitnessReport {
    pub version: u8,
    pub scalarization: String,
    pub privacy_soft_weight: f64,
    pub eligible: bool,
    pub hard_gates: BTreeMap<String, bool>,
    pub pareto_vector: ParetoVector,
    pub raw_metric_vector: BTreeMap<String, Option<f64>>,
    pub weights: FitnessWeights,
    pub score: Option<f64>,
    pub weight_sensitivity: BTreeMap<String, Option<f64>>,
    pub unavailable_terms: Vec<String>,
}

fn score(vector: &ParetoVector, weights: &FitnessWeights, epsilon: f64) -> Option<f64> {
    let terms = vector.entries();
    let weight_entries = weights.entries();
    let total_weight: f64 = weight_entries.iter().map(|(_, value)| value).sum();
    if total_weight <= 0.0 {
        return None;
    }
    let mut log_score = 0.0;
    for ((_, value), (_, weight)) in terms.iter().zip(weight_entries) {
        if weight == 0.0 {
            continue;
        }
        let value = value.filter(|value| value.is_finite() && (0.0..=1.0).contains(value))?;
        log_score += weight * (epsilon + value).ln();
    }
    Some(100.0 * (log_score / total_weight).exp())
}

impl MasterFitnessReport {
    pub fn evaluate(
        contract: &MasterFitnessContract,
        hard_gates: BTreeMap<String, bool>,
        pareto_vector: ParetoVector,
        raw_metric_vector: BTreeMap<String, Option<f64>>,
    ) -> Self {
        let eligible = !hard_gates.is_empty() && hard_gates.values().all(|passed| *passed);
        let unavailable_terms = pareto_vector
            .entries()
            .into_iter()
            .filter(|(_, value)| !value.is_some_and(|v| v.is_finite() && (0.0..=1.0).contains(&v)))
            .map(|(name, _)| name.to_string())
            .collect();
        let mut weight_sensitivity = BTreeMap::new();
        let peers = [
            ("default", contract.weights.clone()),
            (
                "utility_heavy",
                FitnessWeights {
                    utility_transfer: 0.40,
                    driver_fidelity: 0.20,
                    distribution_fidelity: 0.15,
                    structure_fidelity: 0.10,
                    coverage_realism: 0.10,
                    compactness: 0.05,
                },
            ),
            (
                "driver_heavy",
                FitnessWeights {
                    utility_transfer: 0.25,
                    driver_fidelity: 0.30,
                    distribution_fidelity: 0.15,
                    structure_fidelity: 0.15,
                    coverage_realism: 0.10,
                    compactness: 0.05,
                },
            ),
        ];
        for (name, weights) in peers {
            weight_sensitivity.insert(
                name.to_string(),
                eligible
                    .then(|| score(&pareto_vector, &weights, contract.epsilon))
                    .flatten(),
            );
        }
        for (name, _) in contract.weights.entries() {
            let mut weights = contract.weights.clone();
            match name {
                "utility_transfer" => weights.utility_transfer = 0.0,
                "driver_fidelity" => weights.driver_fidelity = 0.0,
                "distribution_fidelity" => weights.distribution_fidelity = 0.0,
                "structure_fidelity" => weights.structure_fidelity = 0.0,
                "coverage_realism" => weights.coverage_realism = 0.0,
                "compactness" => weights.compactness = 0.0,
                _ => unreachable!(),
            }
            weight_sensitivity.insert(
                format!("without_{name}"),
                eligible
                    .then(|| score(&pareto_vector, &weights, contract.epsilon))
                    .flatten(),
            );
        }
        let score = weight_sensitivity["default"];
        Self {
            version: contract.version,
            scalarization: contract.scalarization.clone(),
            privacy_soft_weight: contract.privacy_soft_weight,
            eligible,
            hard_gates,
            pareto_vector,
            raw_metric_vector,
            weights: contract.weights.clone(),
            score,
            weight_sensitivity,
            unavailable_terms,
        }
    }
}

/// MFS-v3 adds representation closeness and keeps privacy as a hard gate.
/// The v2 contract and `MasterFitnessReport::evaluate` are unchanged.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FitnessWeightsV3 {
    pub utility_transfer: f64,
    pub representation_closeness: f64,
    pub driver_fidelity: f64,
    pub distribution_fidelity: f64,
    pub structure_fidelity: f64,
    pub coverage_realism: f64,
    pub compactness: f64,
}

impl FitnessWeightsV3 {
    fn entries(&self) -> [(&'static str, f64); 7] {
        [
            ("utility_transfer", self.utility_transfer),
            ("representation_closeness", self.representation_closeness),
            ("driver_fidelity", self.driver_fidelity),
            ("distribution_fidelity", self.distribution_fidelity),
            ("structure_fidelity", self.structure_fidelity),
            ("coverage_realism", self.coverage_realism),
            ("compactness", self.compactness),
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
pub struct MasterFitnessV3Contract {
    pub version: u8,
    pub scalarization: String,
    pub privacy_soft_weight: f64,
    pub epsilon: f64,
    pub weights: FitnessWeightsV3,
    #[serde(default)]
    pub epsilon_rule: String,
    #[serde(default)]
    pub prerequisites: Vec<String>,
}

const V3_PREREQUISITES: [&str; 4] = [
    "real_vs_real_control_complete",
    "required_size_cells_complete",
    "artifact_sampling_verified",
    "metric_implementations_locked",
];

impl MasterFitnessV3Contract {
    pub fn is_frozen(&self) -> bool {
        self.version == 3
            && self.scalarization == "geometric_mean"
            && self.privacy_soft_weight == 0.0
            && self.epsilon == 1e-6
            && self.epsilon_rule == "max_component_epsilon"
            && self
                .prerequisites
                .iter()
                .map(String::as_str)
                .eq(V3_PREREQUISITES)
            && self.weights.is_valid()
            && self.weights
                == FitnessWeightsV3 {
                    utility_transfer: 0.35,
                    representation_closeness: 0.15,
                    driver_fidelity: 0.15,
                    distribution_fidelity: 0.15,
                    structure_fidelity: 0.10,
                    coverage_realism: 0.05,
                    compactness: 0.05,
                }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoredComponentsV3 {
    pub utility_transfer: Option<f64>,
    pub driver_fidelity: Option<f64>,
    pub distribution_fidelity: Option<f64>,
    pub structure_fidelity: Option<f64>,
    pub coverage_realism: Option<f64>,
    pub compactness: Option<f64>,
}

impl ScoredComponentsV3 {
    fn entries(&self) -> [(&'static str, Option<f64>); 6] {
        [
            ("utility_transfer", self.utility_transfer),
            ("driver_fidelity", self.driver_fidelity),
            ("distribution_fidelity", self.distribution_fidelity),
            ("structure_fidelity", self.structure_fidelity),
            ("coverage_realism", self.coverage_realism),
            ("compactness", self.compactness),
        ]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepresentationObservation {
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
    pub normalizer: String,
    pub utility_protocol: String,
    pub utility_auditors: Vec<String>,
    pub real_vs_real_control_complete: bool,
    pub required_size_cells_complete: bool,
    pub artifact_sampling_verified: bool,
    pub metric_implementations_locked: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MasterFitnessV3Report {
    pub version: u8,
    pub eligible: bool,
    pub hard_gates: BTreeMap<String, bool>,
    pub score: Option<f64>,
    pub representation_closeness: Option<f64>,
    pub unavailable_terms: Vec<String>,
}

fn component_in_unit(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
}

/// Returns null unless the frozen v3 weights, the security gates, and the
/// computed representation component are all present. A caller-supplied
/// closeness number is not accepted.
pub fn evaluate_v3(
    contract: &MasterFitnessV3Contract,
    components: &ScoredComponentsV3,
    observation: &RepresentationObservation,
) -> MasterFitnessV3Report {
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
    let tabular_ok = crate::representation::tabular_transfer_ok(
        &observation.normalizer,
        &observation.utility_protocol,
        &observation.utility_auditors,
    );
    let mut scored = components.clone();
    if !tabular_ok {
        scored.utility_transfer = None;
    }
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
    for (key, passed) in V3_PREREQUISITES.into_iter().zip([
        observation.real_vs_real_control_complete,
        observation.required_size_cells_complete,
        observation.artifact_sampling_verified,
        observation.metric_implementations_locked,
    ]) {
        hard_gates.insert(key.into(), passed);
    }
    hard_gates.insert("frozen_contract".into(), contract.is_frozen());
    hard_gates.insert("shared_normalizer".into(), shared_map);
    hard_gates.insert("tabular_transfer".into(), tabular_ok);
    hard_gates.insert("representation_closeness".into(), closeness.is_some());
    let mut unavailable_terms = scored
        .entries()
        .into_iter()
        .filter(|(_, value)| component_in_unit(*value).is_none())
        .map(|(name, _)| name.to_string())
        .collect::<Vec<_>>();
    if closeness.is_none() {
        unavailable_terms.push("representation_closeness".into());
    }
    let gates_pass =
        contract.is_frozen() && !hard_gates.is_empty() && hard_gates.values().all(|passed| *passed);
    let measured = scored
        .entries()
        .into_iter()
        .map(|(_, value)| component_in_unit(value))
        .collect::<Option<Vec<_>>>();
    let score = if gates_pass {
        measured.and_then(|values| {
            let closeness = closeness?;
            let mut log_score =
                contract.weights.representation_closeness * closeness.max(contract.epsilon).ln();
            let named = [
                ("utility_transfer", values[0]),
                ("driver_fidelity", values[1]),
                ("distribution_fidelity", values[2]),
                ("structure_fidelity", values[3]),
                ("coverage_realism", values[4]),
                ("compactness", values[5]),
            ];
            for (name, value) in named {
                let weight = contract
                    .weights
                    .entries()
                    .into_iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, weight)| weight)
                    .unwrap_or(0.0);
                log_score += weight * value.max(contract.epsilon).ln();
            }
            let total = contract
                .weights
                .entries()
                .iter()
                .map(|(_, weight)| weight)
                .sum::<f64>();
            (total > 0.0).then_some((100.0 * (log_score / total).exp()).clamp(0.0, 100.0))
        })
    } else {
        None
    };
    MasterFitnessV3Report {
        version: contract.version,
        eligible: score.is_some(),
        hard_gates,
        score,
        representation_closeness: closeness,
        unavailable_terms,
    }
}

#[cfg(test)]
include!("fitness/tests.rs");

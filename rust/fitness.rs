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
mod tests {
    use super::*;

    fn contract() -> MasterFitnessContract {
        MasterFitnessContract {
            version: 2,
            scalarization: "geometric_mean".into(),
            privacy_soft_weight: 0.0,
            epsilon: 1e-6,
            weights: FitnessWeights {
                utility_transfer: 0.30,
                driver_fidelity: 0.20,
                distribution_fidelity: 0.20,
                structure_fidelity: 0.15,
                coverage_realism: 0.10,
                compactness: 0.05,
            },
        }
    }

    #[test]
    fn gate_failure_cannot_be_bought_with_soft_scores() {
        let vector = ParetoVector {
            utility_transfer: Some(1.0),
            driver_fidelity: Some(1.0),
            distribution_fidelity: Some(1.0),
            structure_fidelity: Some(1.0),
            coverage_realism: Some(1.0),
            compactness: Some(1.0),
        };
        let report = MasterFitnessReport::evaluate(
            &contract(),
            BTreeMap::from([("privacy".into(), false)]),
            vector,
            BTreeMap::new(),
        );
        assert!(!report.eligible);
        assert_eq!(report.score, None);
        assert!(report.weight_sensitivity.values().all(Option::is_none));
    }

    #[test]
    fn missing_coverage_does_not_become_a_perfect_score() {
        let vector = ParetoVector {
            utility_transfer: Some(1.0),
            driver_fidelity: Some(1.0),
            distribution_fidelity: Some(1.0),
            structure_fidelity: Some(1.0),
            compactness: Some(1.0),
            ..ParetoVector::default()
        };
        let report = MasterFitnessReport::evaluate(
            &contract(),
            BTreeMap::from([("privacy".into(), true)]),
            vector,
            BTreeMap::new(),
        );
        assert_eq!(report.score, None);
        assert_eq!(report.unavailable_terms, vec!["coverage_realism"]);
        assert!(report.weight_sensitivity["without_coverage_realism"].is_some());
    }

    #[test]
    fn complete_gate_passer_uses_geometric_mean_and_archives_sensitivity() {
        let vector = ParetoVector {
            utility_transfer: Some(0.8),
            driver_fidelity: Some(0.6),
            distribution_fidelity: Some(0.9),
            structure_fidelity: Some(0.7),
            coverage_realism: Some(0.5),
            compactness: Some(0.4),
        };
        let report = MasterFitnessReport::evaluate(
            &contract(),
            BTreeMap::from([("privacy".into(), true), ("utility".into(), true)]),
            vector,
            BTreeMap::new(),
        );
        let expected = 100.0
            * ((0.8_f64 + 1e-6).ln() * 0.30
                + (0.6_f64 + 1e-6).ln() * 0.20
                + (0.9_f64 + 1e-6).ln() * 0.20
                + (0.7_f64 + 1e-6).ln() * 0.15
                + (0.5_f64 + 1e-6).ln() * 0.10
                + (0.4_f64 + 1e-6).ln() * 0.05)
                .exp();
        assert!(report.eligible);
        assert!((report.score.unwrap() - expected).abs() < 1e-10);
        assert_eq!(report.weight_sensitivity.len(), 9);
        assert!(report.weight_sensitivity.values().all(Option::is_some));
    }

    fn v3_contract() -> MasterFitnessV3Contract {
        MasterFitnessV3Contract {
            version: 3,
            scalarization: "geometric_mean".into(),
            privacy_soft_weight: 0.0,
            epsilon: 1e-6,
            epsilon_rule: "max_component_epsilon".into(),
            prerequisites: V3_PREREQUISITES
                .iter()
                .map(|key| (*key).to_string())
                .collect(),
            weights: FitnessWeightsV3 {
                utility_transfer: 0.35,
                representation_closeness: 0.15,
                driver_fidelity: 0.15,
                distribution_fidelity: 0.15,
                structure_fidelity: 0.10,
                coverage_realism: 0.05,
                compactness: 0.05,
            },
        }
    }

    fn passing_components() -> ScoredComponentsV3 {
        ScoredComponentsV3 {
            utility_transfer: Some(0.5),
            driver_fidelity: Some(0.5),
            distribution_fidelity: Some(0.5),
            structure_fidelity: Some(0.5),
            coverage_realism: Some(0.5),
            compactness: Some(0.5),
        }
    }

    fn passing_observation() -> RepresentationObservation {
        RepresentationObservation {
            real_vs_real_control_complete: true,
            required_size_cells_complete: true,
            artifact_sampling_verified: true,
            metric_implementations_locked: true,
            encoder: "kumo_tabular_l".into(),
            distance: 0.2,
            d_null: 0.5,
            d_match: 0.4,
            gap_mitra: -0.1,
            gap_tabicl: -0.2,
            exact_row_matches: 0,
            near_copy_ok: true,
            cleartext_absent: true,
            membership_auc: 0.5,
            attribute_inference_advantage: 0.0,
            artifact_bytes: 1_000,
            normalizer: crate::representation::NORMALIZER.into(),
            utility_protocol: crate::representation::TABULAR_PROTOCOL.into(),
            utility_auditors: crate::representation::TABULAR_AUDITORS
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
        }
    }

    #[test]
    fn v3_geometric_mean_uses_the_frozen_weights() {
        let report = evaluate_v3(
            &v3_contract(),
            &passing_components(),
            &passing_observation(),
        );
        assert!(report.eligible);
        assert!((report.representation_closeness.unwrap() - 0.5).abs() < 1e-12);
        assert!((report.score.unwrap() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn v3_perfect_profile_is_exactly_100() {
        let mut components = passing_components();
        components.utility_transfer = Some(1.0);
        components.driver_fidelity = Some(1.0);
        components.distribution_fidelity = Some(1.0);
        components.structure_fidelity = Some(1.0);
        components.coverage_realism = Some(1.0);
        components.compactness = Some(1.0);
        let mut observation = passing_observation();
        observation.distance = 0.0;
        assert_eq!(
            evaluate_v3(&v3_contract(), &components, &observation).score,
            Some(100.0)
        );
    }

    #[test]
    fn v3_positive_byte_boundaries_are_enforced() {
        for bytes in [0, 10_241] {
            let mut observation = passing_observation();
            observation.artifact_bytes = bytes;
            let report = evaluate_v3(&v3_contract(), &passing_components(), &observation);
            assert_eq!(report.score, None);
            assert!(!report.hard_gates["tier_bytes"]);
        }
        for bytes in [1, 10_240] {
            let mut observation = passing_observation();
            observation.artifact_bytes = bytes;
            assert!(evaluate_v3(&v3_contract(), &passing_components(), &observation).eligible);
        }
    }

    #[test]
    fn v3_each_missing_prerequisite_is_null() {
        for key in V3_PREREQUISITES {
            let mut observation = passing_observation();
            match key {
                "real_vs_real_control_complete" => {
                    observation.real_vs_real_control_complete = false
                }
                "required_size_cells_complete" => observation.required_size_cells_complete = false,
                "artifact_sampling_verified" => observation.artifact_sampling_verified = false,
                "metric_implementations_locked" => {
                    observation.metric_implementations_locked = false
                }
                _ => unreachable!(),
            }
            let report = evaluate_v3(&v3_contract(), &passing_components(), &observation);
            assert_eq!(report.score, None);
            assert!(!report.hard_gates[key]);
        }
    }

    #[test]
    fn v3_membership_auc_has_a_probability_domain() {
        for auc in [-1.0, 0.551, f64::NAN, f64::INFINITY] {
            let mut observation = passing_observation();
            observation.membership_auc = auc;
            let report = evaluate_v3(&v3_contract(), &passing_components(), &observation);
            assert_eq!(report.score, None);
            assert!(!report.hard_gates["membership_auc"]);
        }
        for auc in [0.0, 0.55] {
            let mut observation = passing_observation();
            observation.membership_auc = auc;
            assert!(evaluate_v3(&v3_contract(), &passing_components(), &observation).eligible);
        }
    }

    #[test]
    fn v3_attribute_advantage_has_a_probability_domain() {
        for advantage in [-1.0, 0.051, f64::NAN, f64::INFINITY] {
            let mut observation = passing_observation();
            observation.attribute_inference_advantage = advantage;
            let report = evaluate_v3(&v3_contract(), &passing_components(), &observation);
            assert_eq!(report.score, None);
            assert!(!report.hard_gates["attribute_inference"]);
        }
        for advantage in [0.0, 0.05] {
            let mut observation = passing_observation();
            observation.attribute_inference_advantage = advantage;
            assert!(evaluate_v3(&v3_contract(), &passing_components(), &observation).eligible);
        }
    }

    #[test]
    fn v3_exact_row_match_is_null() {
        let mut observation = passing_observation();
        observation.exact_row_matches = 1;
        let report = evaluate_v3(&v3_contract(), &passing_components(), &observation);
        assert!(!report.hard_gates["exact_row_match"]);
        assert_eq!(report.score, None);
    }

    #[test]
    fn v3_cleartext_byte_is_null() {
        let mut observation = passing_observation();
        observation.cleartext_absent =
            !crate::representation::artifact_has_cleartext(b"header,age\n", &[b"age"]);
        let report = evaluate_v3(&v3_contract(), &passing_components(), &observation);
        assert_eq!(report.score, None);
        assert!(!report.hard_gates["cleartext_absent"]);
    }

    #[test]
    fn v3_refused_encoder_is_null() {
        let mut observation = passing_observation();
        observation.encoder = "foundation".into();
        let report = evaluate_v3(&v3_contract(), &passing_components(), &observation);
        assert!(!report.hard_gates["clean_encoder"]);
        assert_eq!(report.score, None);
    }

    #[test]
    fn v3_zero_match_distance_and_sign_flip_are_null() {
        let mut observation = passing_observation();
        observation.d_match = 0.0;
        assert_eq!(
            evaluate_v3(&v3_contract(), &passing_components(), &observation).score,
            None
        );
        observation.d_match = 0.4;
        observation.gap_mitra = 0.2;
        let report = evaluate_v3(&v3_contract(), &passing_components(), &observation);
        assert!(report.representation_closeness.is_none());
        assert_eq!(report.score, None);
    }

    #[test]
    fn v3_auditor_retention_cannot_fill_tabular_transfer() {
        let mut observation = passing_observation();
        observation.utility_protocol = "auditor_retention".into();
        observation.utility_auditors = vec!["catboost".into()];
        let report = evaluate_v3(&v3_contract(), &passing_components(), &observation);
        assert!(!report.hard_gates["tabular_transfer"]);
        assert!(
            report
                .unavailable_terms
                .iter()
                .any(|term| term == "utility_transfer")
        );
        assert_eq!(report.score, None);
    }

    #[test]
    fn embedded_v3_contract_matches_the_build_digest() {
        use sha2::{Digest, Sha256};
        let bytes = include_bytes!("../production/kpi-contract-v3.json");
        let normalized = bytes.strip_suffix(b"\n").unwrap_or(bytes);
        let digest = format!("{:x}", Sha256::digest(normalized));
        assert_eq!(digest, env!("DOPE_KPI_CONTRACT_V3_SHA256"));
        let parsed: serde_json::Value = serde_json::from_slice(normalized).unwrap();
        let contract: MasterFitnessV3Contract =
            serde_json::from_value(parsed["master_fitness"].clone()).unwrap();
        assert!(contract.is_frozen());
    }
}

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
}

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
        assert!(report
            .unavailable_terms
            .iter()
            .any(|term| term == "utility_transfer"));
        assert_eq!(report.score, None);
    }

    #[test]
    fn embedded_v3_contract_matches_the_build_digest() {
        use sha2::{Digest, Sha256};
        let bytes = include_bytes!("../../production/kpi-contract-v3.json");
        let normalized = bytes.strip_suffix(b"\n").unwrap_or(bytes);
        let digest = format!("{:x}", Sha256::digest(normalized));
        assert_eq!(digest, env!("DOPE_KPI_CONTRACT_V3_SHA256"));
        let parsed: serde_json::Value = serde_json::from_slice(normalized).unwrap();
        let contract: MasterFitnessV3Contract =
            serde_json::from_value(parsed["master_fitness"].clone()).unwrap();
        assert!(contract.is_frozen());
    }

    fn v4_contract() -> MasterFitnessV4Contract {
        MasterFitnessV4Contract {
            version: 4,
            scalarization: "geometric_mean".into(),
            privacy_soft_weight: 0.0,
            epsilon: 1e-6,
            epsilon_rule: "max_component_epsilon".into(),
            prerequisites: V4_PREREQUISITES.iter().map(|k| (*k).to_string()).collect(),
            score_range: [0, 100],
            weights: FitnessWeightsV4 {
                utility_transfer: 0.30,
                driver_fidelity: 0.20,
                representation_closeness: 0.15,
                knowledge_compression: 0.15,
                distribution_fidelity: 0.10,
                structure_fidelity: 0.05,
                coverage_realism: 0.05,
            },
        }
    }

    fn v4_components() -> ScoredComponentsV4 {
        ScoredComponentsV4 {
            distribution_fidelity: Some(0.85),
            structure_fidelity: Some(0.80),
            coverage_realism: Some(0.90),
        }
    }

    fn v4_observation() -> MfsV4Observation {
        MfsV4Observation {
            encoder: "kumo_tabular_l".into(),
            distance: 0.2,
            d_null: 0.5,
            d_match: 0.4,
            gap_mitra: -0.1,
            gap_tabicl: -0.2,
            exact_row_matches: 0,
            near_copy_ok: true,
            cleartext_absent: true,
            membership_auc: 0.50,
            attribute_inference_advantage: 0.01,
            artifact_bytes: 1_024,
            encoded_numbers: 64,
            real_rows: 1_024,
            real_columns: 8,
            normalizer: crate::representation::NORMALIZER.into(),
            utility_protocol: crate::representation::TABULAR_PROTOCOL.into(),
            utility_auditors: V4_TABULAR_AUDITORS
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            retention: 0.90,
            predictor_only_retention: 0.70,
            independent_marginals_retention: 0.65,
            real_bootstrap_4n_retention: Some(0.85),
            real_importances: vec![0.40, 0.30, 0.20, 0.10],
            synthetic_importances: vec![0.45, 0.25, 0.20, 0.10],
            real_signs: Some(vec![1, -1, 1, -1]),
            synthetic_signs: Some(vec![1, -1, 1, 1]),
            real_vs_real_control_complete: true,
            required_size_cells_complete: true,
            artifact_sampling_verified: true,
            metric_implementations_locked: true,
            control_contrastive_baselines_complete: true,
            model_driver_attribution_measured: true,
        }
    }

    #[test]
    fn embedded_v4_contract_matches_the_build_digest() {
        use sha2::{Digest, Sha256};
        let bytes = include_bytes!("../../production/kpi-contract-v4.json");
        let normalized = bytes.strip_suffix(b"\n").unwrap_or(bytes);
        let digest = format!("{:x}", Sha256::digest(normalized));
        assert_eq!(digest, env!("DOPE_KPI_CONTRACT_V4_SHA256"));
        let parsed: serde_json::Value = serde_json::from_slice(normalized).unwrap();
        let contract: MasterFitnessV4Contract =
            serde_json::from_value(parsed["master_fitness"].clone()).unwrap();
        assert!(contract.is_frozen());
    }

    #[test]
    fn v4_refuses_unmeasured_directional_attribution() {
        let mut observation = v4_observation();
        observation.real_signs = None;
        observation.synthetic_signs = None;
        let report = evaluate_v4(&v4_contract(), &v4_components(), &observation);
        assert_eq!(report.driver_fidelity, None);
        assert_eq!(report.driver_attribution, None);
        assert_eq!(report.score, None);
        assert!(!report.eligible);
        assert!(!report.hard_gates["model_driver_attribution_measured"]);
        assert!(report.unavailable_terms.iter().any(|term| term == "driver_fidelity"));

        let mut truncated = v4_observation();
        truncated.real_signs = Some(vec![1, -1]);
        truncated.synthetic_signs = Some(vec![1, -1]);
        let truncated_report = evaluate_v4(&v4_contract(), &v4_components(), &truncated);
        assert_eq!(truncated_report.driver_fidelity, None);
        assert_eq!(truncated_report.driver_attribution, None);
        assert_eq!(truncated_report.score, None);
        assert!(!truncated_report.eligible);
        assert!(!truncated_report.hard_gates["model_driver_attribution_measured"]);
    }

    #[test]
    fn v4_refuses_overflowed_attribution_normalizers() {
        let mut observation = v4_observation();
        observation.real_importances = vec![f64::MAX / 32.0; 500];
        observation.synthetic_importances = observation.real_importances.clone();
        observation.synthetic_importances[100..].fill(0.0);
        observation.real_signs = Some(vec![1; 500]);
        observation.synthetic_signs = Some(vec![1; 500]);
        let report = evaluate_v4(&v4_contract(), &v4_components(), &observation);
        assert_eq!(report.driver_fidelity, None);
        assert_eq!(report.driver_attribution, None);
        assert_eq!(report.score, None);
        assert!(!report.eligible);
        assert!(!report.hard_gates["model_driver_attribution_measured"]);
    }

    #[test]
    fn v4_enforces_control_lift_driver_attribution_and_rewards_knowledge_compression() {
        let contract = v4_contract();
        let components = v4_components();
        let compact_obs = v4_observation();
        let compact_report = evaluate_v4(&contract, &components, &compact_obs);
        assert!(compact_report.eligible);
        assert!(compact_report.score.unwrap() > 0.0);
        assert!((compact_report.utility_transfer.unwrap() - 0.935).abs() < 1e-12);
        assert!((compact_report.driver_fidelity.unwrap() - 0.95).abs() < 1e-12);

        let mut bloated_obs = v4_observation();
        bloated_obs.artifact_bytes = 8_192;
        bloated_obs.encoded_numbers = 512;
        let bloated_report = evaluate_v4(&contract, &components, &bloated_obs);
        assert!(bloated_report.eligible);
        assert!(
            compact_report.knowledge_compression.unwrap()
                > bloated_report.knowledge_compression.unwrap()
        );
        assert!(compact_report.score.unwrap() > bloated_report.score.unwrap());

        let mut no_predictor_lift = v4_observation();
        no_predictor_lift.predictor_only_retention = 0.90;
        let failed_lift = evaluate_v4(&contract, &components, &no_predictor_lift);
        assert!(!failed_lift.eligible);
        assert!(!failed_lift.hard_gates["positive_predictor_lift"]);
        assert_eq!(failed_lift.score, None);
    }

    #[test]
    fn v4_refuses_nonfinite_or_overflowed_control_baselines() {
        for bad_predictor in [f64::NAN, f64::INFINITY, -f64::MAX] {
            let mut observation = v4_observation();
            if bad_predictor == -f64::MAX {
                observation.retention = f64::MAX;
            }
            observation.predictor_only_retention = bad_predictor;
            let report = evaluate_v4(&v4_contract(), &v4_components(), &observation);
            assert!(!report.eligible);
            assert_eq!(report.score, None);
            assert_eq!(report.utility_transfer, None);
            assert_eq!(report.control_contrastive_lift, None);
            assert!(!report.hard_gates["control_contrastive_baselines_complete"]);
            assert!(!report.hard_gates["positive_predictor_lift"]);
            assert!(report
                .unavailable_terms
                .iter()
                .any(|term| term == "utility_transfer"));
        }
    }
}

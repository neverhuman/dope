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
        observation.real_columns = 501;
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
        assert_eq!(failed_lift.utility_transfer, None);
        assert_eq!(failed_lift.knowledge_compression, None);
        assert!(failed_lift
            .unavailable_terms
            .iter()
            .any(|term| term == "utility_transfer"));
        assert!(failed_lift
            .unavailable_terms
            .iter()
            .any(|term| term == "knowledge_compression"));
        assert_eq!(failed_lift.score, None);

        let mut over_budget = v4_observation();
        over_budget.artifact_bytes = 12_288;
        let over_budget_report = evaluate_v4(&contract, &components, &over_budget);
        assert!(!over_budget_report.eligible);
        assert!(!over_budget_report.hard_gates["tier_bytes"]);
        assert_eq!(over_budget_report.knowledge_compression, None);
        assert_eq!(over_budget_report.score, None);

        let mut eq_cols = v4_observation();
        eq_cols.real_columns = eq_cols.real_importances.len() as u64;
        let eq_col_report = evaluate_v4(&contract, &components, &eq_cols);
        assert!(!eq_col_report.eligible);
        assert_eq!(eq_col_report.driver_fidelity, None);
        assert!(!eq_col_report.hard_gates["model_driver_attribution_measured"]);

        let mut plus_one_cols = v4_observation();
        plus_one_cols.real_columns = (plus_one_cols.real_importances.len() + 1) as u64;
        let plus_one_report = evaluate_v4(&contract, &components, &plus_one_cols);
        assert!(plus_one_report.eligible);
        assert!(plus_one_report.score.is_some());

        let mut disjoint_obs = v4_observation();
        disjoint_obs.real_importances = vec![0.60, 0.40, 0.0, 0.0];
        disjoint_obs.synthetic_importances = vec![0.0, 0.0, 0.50, 0.50];
        disjoint_obs.real_signs = Some(vec![1, -1, 0, 0]);
        disjoint_obs.synthetic_signs = Some(vec![1, -1, 1, -1]);
        let disjoint_report = evaluate_v4(&contract, &components, &disjoint_obs);
        assert!(!disjoint_report.eligible);
        assert_eq!(disjoint_report.driver_fidelity, None);
        assert!(disjoint_report
            .unavailable_terms
            .iter()
            .any(|term| term == "driver_fidelity"));
        assert_eq!(disjoint_report.score, None);
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

    fn v4_lineage_seeds() -> Vec<V4SeedObservation> {
        let mut s101 = v4_observation();
        s101.retention = 0.88;
        s101.artifact_bytes = 1_024;
        s101.encoded_numbers = 64;

        let mut s211 = v4_observation();
        s211.retention = 0.90;
        s211.artifact_bytes = 1_536;
        s211.encoded_numbers = 80;
        s211.synthetic_importances = vec![0.35, 0.35, 0.20, 0.10];

        let mut s307 = v4_observation();
        s307.retention = 0.92;
        s307.artifact_bytes = 2_048;
        s307.encoded_numbers = 96;

        vec![
            V4SeedObservation {
                sample_seed: 307,
                components: v4_components(),
                observation: s307,
            },
            V4SeedObservation {
                sample_seed: 101,
                components: v4_components(),
                observation: s101,
            },
            V4SeedObservation {
                sample_seed: 211,
                components: v4_components(),
                observation: s211,
            },
        ]
    }

    #[test]
    fn v4_lineage_evaluates_three_seeds_and_reports_seed_variance() {
        let contract = v4_contract();
        let seeds = v4_lineage_seeds();
        let lineage = evaluate_v4_lineage(&contract, &seeds, |_rets| Some(0.86));
        assert!(lineage.report.eligible);
        assert!(lineage.report.score.is_some());
        assert!(lineage.report.hard_gates["complete_seed_set"]);
        assert!(lineage.report.hard_gates["lineage_schema_consistent"]);
        assert!(lineage.report.hard_gates["all_seeds_eligible"]);
        assert!(lineage.report.hard_gates["utility_lcb_valid"]);
        let comp = lineage.report.compression_metrics.unwrap();
        assert_eq!(comp.encoded_numbers, 96);
        let var = lineage.seed_variance.expect("seed variance must be present");
        assert_eq!(var.sample_seeds, V4_SAMPLE_SEEDS);
        assert!((var.retention_mean - 0.90).abs() < 1e-12);
        assert!((var.retention_sample_sd - 0.02).abs() < 1e-12);
        assert!((var.retention_min - 0.88).abs() < 1e-12);
        assert!((var.retention_max - 0.92).abs() < 1e-12);
        assert!((var.retention_lcb - 0.86).abs() < 1e-12);
        assert!(var.driver_fidelity_sample_sd > 0.0);
        assert!(
            (lineage.report.driver_fidelity.unwrap() - var.driver_fidelity_min).abs() < 1e-12
        );
        assert!(var.seed_score_sample_sd > 0.0);
    }

    #[test]
    fn v4_lineage_enforces_exact_lcb_boundary_and_worst_seed_controls() {
        let contract = v4_contract();
        let seeds = v4_lineage_seeds();

        // Exact boundary: lcb == mean (0.90) passes; lcb > mean + 1e-12 fails closed.
        let at_mean = evaluate_v4_lineage(&contract, &seeds, |rets| {
            Some(rets.iter().sum::<f64>() / rets.len() as f64)
        });
        assert!(at_mean.report.eligible);
        assert!(at_mean.report.hard_gates["utility_lcb_valid"]);

        let above_mean = evaluate_v4_lineage(&contract, &seeds, |rets| {
            Some(rets.iter().sum::<f64>() / rets.len() as f64 + 2e-12)
        });
        assert!(!above_mean.report.eligible);
        assert_eq!(above_mean.report.score, None);
        assert!(!above_mean.report.hard_gates["utility_lcb_valid"]);
        assert_eq!(above_mean.seed_variance, None);

        // Worst-seed control baseline: seed 211 has predictor_only_retention = 0.87 (still < 0.90 on that seed),
        // but lineage LCB is 0.86 <= 0.87, so lineage fails positive_predictor_lift.
        let mut tough_control = v4_lineage_seeds();
        tough_control[2].observation.predictor_only_retention = 0.87;
        let failed_control = evaluate_v4_lineage(&contract, &tough_control, |_rets| Some(0.86));
        assert!(!failed_control.report.eligible);
        assert_eq!(failed_control.report.score, None);
        assert!(!failed_control.report.hard_gates["positive_predictor_lift"]);
        assert_eq!(failed_control.report.utility_transfer, None);

        // Single-seed failure cannot be hidden even if LCB > worst control baseline.
        let mut one_bad_seed = v4_lineage_seeds();
        one_bad_seed[1].observation.retention = 0.68; // <= 0.70 on seed 101
        let bad_seed_lineage = evaluate_v4_lineage(&contract, &one_bad_seed, |_rets| Some(0.80));
        assert!(!bad_seed_lineage.report.eligible);
        assert_eq!(bad_seed_lineage.report.score, None);
        assert!(!bad_seed_lineage.report.hard_gates["all_seeds_eligible"]);
        assert!(!bad_seed_lineage.report.hard_gates["positive_predictor_lift"]);
        assert_eq!(bad_seed_lineage.report.utility_transfer, None);
        assert_eq!(bad_seed_lineage.seed_variance, None);

        // Schema inconsistency across seeds fails closed.
        let mut mismatch = v4_lineage_seeds();
        mismatch[0].observation.real_rows = 2_048;
        let mismatch_lineage = evaluate_v4_lineage(&contract, &mismatch, |_rets| Some(0.86));
        assert!(!mismatch_lineage.report.eligible);
        assert!(!mismatch_lineage.report.hard_gates["lineage_schema_consistent"]);
        assert_eq!(mismatch_lineage.report.score, None);

        // Incomplete seed set fails closed.
        let incomplete = evaluate_v4_lineage(&contract, &seeds[..2], |_rets| Some(0.86));
        assert!(!incomplete.report.eligible);
        assert!(!incomplete.report.hard_gates["complete_seed_set"]);
        assert_eq!(incomplete.report.score, None);
    }
}

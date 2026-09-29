

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn record(profile: &str, lineage: usize) -> CohortRecord {
        CohortRecord {
            dataset_id: format!("d{lineage}"),
            dataset_path: PathBuf::from(format!("/{lineage}")),
            task: "regression".into(),
            rows: 128,
            features: 8,
            lineage_group_id: format!("{profile}-l{lineage}"),
            structural_profile: profile.into(),
            partition: "training_gold".into(),
        }
    }

    fn final_outcome(
        candidate: &str,
        profile: &str,
        retention: f64,
        regret: f64,
        artifact_bytes: u64,
        runtime_ms: u64,
    ) -> DeepOutcome {
        DeepOutcome {
            task: "regression".into(),
            lineage_group_id: "shared-lineage".into(),
            structural_profile: profile.into(),
            candidate_id: candidate.into(),
            configuration: Some(LossConfiguration::GRID[0]),
            generation_seed: 1_829,
            auditor_seed: 57_721,
            auditor_id: "elastic_net_glm".into(),
            size_multiplier: 1,
            state: EvidenceState::Succeeded,
            metrics: Some(DeepMetrics {
                bounded_retention: Some(retention),
                regret: Some(regret),
                ..Default::default()
            }),
            failure_reason: None,
            artifact_bytes: Some(artifact_bytes),
            runtime_ms: Some(runtime_ms),
            fitting_time_ms: Some(10),
            sampling_time_ms: Some(5),
            auditor_time_ms: Some(runtime_ms.saturating_sub(15)),
            peak_cpu_memory_bytes: Some(1_024),
            peak_gpu_memory_bytes: None,
            artifact_cache_status: Some(CacheStatus::Miss),
            real_auditor_cache_status: Some(CacheStatus::Miss),
            ancillary_cache_status: Some(CacheStatus::Disabled),
            invalid_rows: 0,
            schema_violations: 0,
            nondeterministic_output: false,
        }
    }

    fn passing(candidate: &str) -> Qualification {
        Qualification {
            candidate_id: candidate.into(),
            implementation_hash: candidate_implementation_hash(candidate),
            qualifies: true,
            overall_lift: None,
            oracle_lift: Some(0.0),
            success_coverage: 1.0,
            minimum_profile_coverage: 1.0,
            feature_importance_coverage: 1.0,
            feature_importance_spearman_delta: None,
            feature_importance_top_k_delta: None,
            maximum_artifact_bytes: Some(1_024),
            failed_gates: Vec::new(),
        }
    }

    #[test]
    fn cohorts_take_first_eight_then_next_thirty_two_without_overlap() {
        let plan = CohortPlan {
            format: "dope-campaign-cohort".into(),
            version: 1,
            kind: "training-gold".into(),
            records: (0..100)
                .map(|lineage| record(if lineage < 50 { "a" } else { "b" }, lineage))
                .collect(),
            exclusions: Vec::new(),
        };
        let deep = plan_deep_cohorts(&plan).unwrap();
        assert_eq!(deep.discovery.len(), 16);
        assert_eq!(deep.confirmation.len(), 64);
        let discovery = deep
            .discovery
            .iter()
            .map(|record| &record.lineage_group_id)
            .collect::<BTreeSet<_>>();
        assert!(
            deep.confirmation
                .iter()
                .all(|record| !discovery.contains(&record.lineage_group_id))
        );
        assert_eq!(deep.confirmation_cells(14), 64 * 14 * 108);
        assert_eq!(
            EXPECTED_CONFIRMATION_LINEAGES * 14 * 108,
            MAX_CONFIRMATION_CELLS
        );
        assert_eq!(
            validation_select_cells(3).unwrap(),
            MAX_VALIDATION_SELECT_CELLS
        );
    }

    #[test]
    fn bounded_expansion_follows_the_frozen_deficit_rule() {
        assert_eq!(
            bounded_expansion(0.049, "query_error"),
            ExpansionDecision::None
        );
        assert_eq!(
            bounded_expansion(0.05, "query_error"),
            ExpansionDecision::Ctgan
        );
        assert_eq!(
            bounded_expansion(0.08, "rare_tail_retention"),
            ExpansionDecision::Tabddpm
        );
        assert_eq!(
            bounded_expansion(0.08, "privacy"),
            ExpansionDecision::PublishFailedFrontier
        );
    }

    #[test]
    fn final_selector_uses_profile_lineage_units_and_frozen_tiebreaks() {
        let outcomes = vec![
            final_outcome("tvae", "a", 0.8, 0.2, 1_000, 30),
            final_outcome("tvae", "b", 0.8, 0.2, 1_000, 30),
            final_outcome("tabsyn", "a", 0.8, 0.1, 2_000, 20),
            final_outcome("tabsyn", "b", 0.8, 0.1, 2_000, 20),
        ];
        let selection =
            select_final_family(&outcomes, &[passing("tvae"), passing("tabsyn")]).unwrap();
        let winner = selection.winner.unwrap();
        assert_eq!(winner.candidate_id, "tabsyn");
        assert_eq!(winner.cohort_units, 2);
    }

    #[test]
    fn final_selector_recommends_frontier_when_no_candidate_passes() {
        let selection = select_final_family(&[], &[]).unwrap();
        assert_eq!(selection.winner, None);
        assert_eq!(selection.recommendation, "retain_current_frontier");
        let card = build_deep_model_card(
            &selection,
            &DeepCampaignReport {
                format: "dope-deep-joint-campaign-report".into(),
                version: 1,
                slices: Vec::new(),
                failures: Vec::new(),
                paired_pareto_hypervolume: BTreeMap::new(),
            },
        )
        .unwrap();
        assert!(card.contains("retain the current frontier"));
        assert!(card.contains("never pooled"));
    }
}

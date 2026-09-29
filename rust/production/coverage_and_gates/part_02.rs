

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_contract_is_canonical_and_bound() {
        let contract = KpiContract::embedded().unwrap();
        contract.validate().unwrap();
        let digest = hashes(&canonical_json(&contract).unwrap());
        assert_eq!(digest.sha256, KPI_CONTRACT_SHA256);
        assert_eq!(digest.blake3, KPI_CONTRACT_BLAKE3);
        let historical = KpiContract::embedded_v1().unwrap();
        historical.validate().unwrap();
        assert_eq!(historical.version, 1);
    }

    #[test]
    fn shape_profiles_do_not_depend_on_names_or_paths() {
        assert_eq!(
            StructuralProfile::from_shape(Task::Binary, 31, 2_000)
                .unwrap()
                .id(),
            "binary/<32/257-2000"
        );
        assert_eq!(
            StructuralProfile::from_shape(Task::Regression, 1_024, 17)
                .unwrap()
                .id(),
            "regression/>=1024/17-64"
        );
    }

    #[test]
    fn kpi_retention_is_unclamped_and_lineage_weighted() {
        let mut cells = Vec::new();
        for lineage in 0..100 {
            for generation_seed in 0..3 {
                for auditor_seed in AUDITOR_SEEDS {
                    cells.push(KpiCell {
                        task: Task::Regression,
                        train_rows: 100,
                        features: 4,
                        auditor: "elastic_net_glm".into(),
                        size_multiplier: 1,
                        lineage_group_id: format!("lineage-{lineage}"),
                        generation_seed,
                        auditor_seed,
                        null_loss: 1.0,
                        trtr_loss: 0.5,
                        tstr_loss: if lineage == 0 { 0.4 } else { 0.505 },
                        calibration_degradation: None,
                        rare_class_or_tail_retention: None,
                        supported_subgroup_retention: None,
                        nominal_95_coverage: None,
                    });
                }
            }
        }
        let report = aggregate_kpis(&cells, 100, 0, "unit").unwrap();
        assert_eq!(report.independently_gated_profiles.len(), 1);
        assert!(report.cells[0].mean_unclamped_retention.unwrap() > 0.99);
        assert!(report.ptf_v1.unwrap() < report.cells[0].mean_unclamped_retention.unwrap());
    }

    #[test]
    fn low_signal_uses_absolute_noninferiority() {
        let cells = vec![KpiCell {
            task: Task::Binary,
            train_rows: 20,
            features: 2,
            auditor: "ga2m".into(),
            size_multiplier: 4,
            lineage_group_id: "g".into(),
            generation_seed: 7,
            auditor_seed: AUDITOR_SEEDS[0],
            null_loss: 1.0,
            trtr_loss: 0.995,
            tstr_loss: 1.004,
            calibration_degradation: None,
            rare_class_or_tail_retention: None,
            supported_subgroup_retention: None,
            nominal_95_coverage: None,
        }];
        let report = aggregate_kpis(&cells, 1, 0, "unit").unwrap();
        assert_eq!(report.ptf_v1, None);
        assert_eq!(report.low_signal_compliance_overall, Some(1.0));
    }

    #[test]
    fn empty_kpi_summary_is_explicitly_unmeasured() {
        let summary = KpiSummary::from_evidence(None, None, None);
        let value = serde_json::to_value(&summary).unwrap();
        assert!(value["ptf_v1"].is_null());
        assert!(value["low_signal_compliance"].is_null());
        assert!(value["coverage"].is_null());
        assert!(value["candidate_regret"].is_null());
        assert_eq!(summary.auditor_availability.required, 6);
        assert_eq!(
            summary.auditor_availability.available,
            crate::contract::auditor_specs()
                .iter()
                .filter(|auditor| auditor.available)
                .count()
        );
        assert!(!summary.production_score_available);
    }

    #[test]
    fn canonical_writes_replace_atomically_without_temporary_debris() {
        let directory =
            std::env::temp_dir().join(format!("dope-canonical-write-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("value.json");
        write_canonical(&path, &serde_json::json!({"value": 1})).unwrap();
        write_canonical(&path, &serde_json::json!({"value": 2})).unwrap();
        assert_eq!(read_json::<Value>(&path).unwrap()["value"], 2);
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        let _ = fs::remove_dir_all(directory);
    }
}

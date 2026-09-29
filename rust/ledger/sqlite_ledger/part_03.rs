

fn throughput(receipts: &[JobReceipt]) -> Option<f64> {
    let first = receipts
        .iter()
        .map(|receipt| receipt.started_unix_seconds)
        .min()?;
    let last = receipts
        .iter()
        .map(|receipt| receipt.finished_unix_seconds)
        .max()?;
    let hours = (last.saturating_sub(first).max(1)) as f64 / 3_600.0;
    Some(receipts.len() as f64 / hours)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> JobSpec {
        JobSpec {
            format: "dope-job-spec".into(),
            version: 1,
            source_commit: "a".repeat(64),
            environment_lock_sha256: "b".repeat(64),
            corpus_manifest_sha256: "c".repeat(64),
            kpi_contract_sha256: "d".repeat(64),
            phase: "training_gold".into(),
            routed: false,
            candidate_spec: "independent_quantile".into(),
            auditor_spec: "elastic_net_glm".into(),
            dataset_lineage: "lineage-1".into(),
            structural_profile: "regression/32-255/1-16".into(),
            size_multiplier: 1,
            generation_seed: 42,
            auditor_seed: 57721,
            tuning_split: "synthetic_only_cv".into(),
            command: vec!["true".into()],
            environment: BTreeMap::new(),
            expected_outputs: BTreeMap::from([("metrics".into(), PathBuf::from("metrics.json"))]),
            timeout_seconds: 60,
        }
    }

    fn ledger() -> (CampaignLedger, PathBuf) {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "dope-ledger-{}-{}.sqlite",
            std::process::id(),
            unique
        ));
        let _ = fs::remove_file(&path);
        (CampaignLedger::initialize(&path).unwrap(), path)
    }

    fn evidence(spec: &JobSpec) -> JobEvidence {
        JobEvidence {
            format: "dope-job-evidence".into(),
            version: 2,
            phase: spec.phase.clone(),
            task: "regression".into(),
            candidate_id: spec.candidate_spec.clone(),
            auditor_id: spec.auditor_spec.clone(),
            lineage_group_id: spec.dataset_lineage.clone(),
            structural_profile: spec.structural_profile.clone(),
            train_rows: 100,
            features: 4,
            size_multiplier: spec.size_multiplier,
            generation_seed: spec.generation_seed,
            auditor_seed: spec.auditor_seed,
            routed: spec.routed,
            null_loss: 1.0,
            trtr_loss: 0.5,
            tstr_loss: 0.6,
            runtime_ms: 10,
            fitting_time_ms: 3,
            sampling_time_ms: 2,
            auditor_time_ms: 5,
            peak_memory_bytes: 1_024,
            peak_cpu_memory_bytes: 1_024,
            peak_gpu_memory_bytes: None,
            artifact_bytes: 512,
            artifact_cache_status: CacheStatus::Miss,
            real_auditor_cache_status: CacheStatus::Miss,
            ancillary_cache_status: CacheStatus::Disabled,
            artifact_sha256: "a".repeat(64),
            artifact_blake3: "b".repeat(64),
            calibration_degradation: None,
            rare_class_or_tail_retention: None,
            supported_subgroup_retention: None,
            nominal_95_coverage: None,
            driver_agreement: None,
            joint_fidelity: None,
            query_p95_normalized_error: None,
            type_i_error: None,
            membership_auc: None,
            attribute_inference_advantage: None,
            feature_importance_spearman: Some(0.5),
            feature_importance_top_k_agreement: 0.5,
            feature_importance_feature_count: 4,
            feature_importance_informative_count: 3,
            feature_importance_real_shares: Vec::new(),
            feature_importance_synthetic_shares: Vec::new(),
            feature_importance_mean_ratio_error: None,
            exact_copies: 0,
            near_copies: 0,
            canary_extractions: 0,
            lineage_leaks: 0,
            invalid_rows: 0,
            schema_violations: 0,
            nondeterministic_output: false,
        }
    }

    #[test]
    fn ledger_is_idempotent_and_receipts_are_immutable() {
        let (ledger, path) = ledger();
        assert_eq!(ledger.insert_specs(&[spec()]).unwrap(), 1);
        assert_eq!(ledger.insert_specs(&[spec()]).unwrap(), 0);
        let leased = ledger.lease_next("xbabe1", 600).unwrap().unwrap();
        ledger.mark_running(&leased.job_id, "xbabe1").unwrap();
        let mut receipt = JobReceipt {
            format: "dope-job-receipt".into(),
            version: 1,
            job_id: leased.job_id.clone(),
            attempt: leased.attempt,
            worker: "xbabe1".into(),
            state: JobState::Succeeded,
            started_unix_seconds: now(),
            finished_unix_seconds: now(),
            source_commit: leased.spec.source_commit.clone(),
            environment_lock_sha256: leased.spec.environment_lock_sha256.clone(),
            corpus_manifest_sha256: leased.spec.corpus_manifest_sha256.clone(),
            kpi_contract_sha256: leased.spec.kpi_contract_sha256.clone(),
            output_hashes: BTreeMap::from([(
                "metrics".into(),
                ContentHashes {
                    sha256: "e".repeat(64),
                    blake3: "f".repeat(64),
                },
            )]),
            failure_class: None,
            failure: None,
            command_exit_code: Some(0),
            evidence: Some(evidence(&leased.spec)),
            signature: None,
        };
        let key = [7; 32];
        receipt.sign(&key).unwrap();
        ledger.finish_signed(&receipt, &key).unwrap();
        let status = ledger.status().unwrap();
        assert_eq!(status.counts[&JobState::Succeeded], 1);
        assert_eq!(status.completed_gold_labels, 1);
        assert_eq!(status.training_progress, Some(1.0));
        assert_eq!(status.current_router_validation_regret, None);
        assert_eq!(status.ptf_v1, None);
        assert!(!status.production_score_available);
        assert!(ledger.finish_signed(&receipt, &key).is_err());
        drop(ledger);
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(path.with_extension("sqlite-wal"));
        let _ = fs::remove_file(path.with_extension("sqlite-shm"));
    }

    #[test]
    fn expired_infrastructure_leases_retry_only_twice() {
        let (ledger, path) = ledger();
        ledger.insert_specs(&[spec()]).unwrap();
        for expected_attempt in 1..=3 {
            let leased = ledger.lease_next("worker", 1).unwrap().unwrap();
            assert_eq!(leased.attempt, expected_attempt);
            ledger.reclaim_expired(u64::MAX).unwrap();
        }
        assert_eq!(ledger.status().unwrap().counts[&JobState::TimedOut], 1);
        drop(ledger);
        let _ = fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_and_gpu_worker_cannot_open_validation_cert() {
        for host in ["xbabe1", "xbabe2"] {
            assert!(require_named_host(host, "xbabe3", "validation-cert authorization").is_err());
        }
        assert!(require_named_host("xbabe3", "xbabe3", "validation-cert authorization").is_ok());
    }

    #[test]
    fn router_promotion_requires_every_frozen_gate() {
        let evidence = RouterEvidence {
            format: "dope-router-evidence".into(),
            version: 1,
            top_four_oracle_recall: 0.99,
            maximum_profile_regret_upper: 0.02,
            beats_random: true,
            beats_best_fixed: true,
            beats_ga2m: true,
            paired_hypervolume_improvement: 0.05,
            paired_hypervolume_ci_lower: 0.001,
            student_max_abs_difference: 0.001,
            top_choice_agreement: 0.999,
            bundle_bytes: 1024,
            inference_p95_ms: 5.0,
            sketch_p95_ms: 1_000.0,
            training_evidence_sha256: "a".repeat(64),
            best_fixed_candidate_id: empirical_backends()[0].id.clone(),
            student_mean_regret: 0.001,
            random_mean_regret: 0.01,
            best_fixed_mean_regret: 0.005,
            ga2m_mean_regret: 0.004,
        };
        assert!(evidence.failed_gates().is_empty());
        let mut failed = evidence;
        failed.beats_ga2m = false;
        assert_eq!(failed.failed_gates(), vec!["beats_ga2m"]);
    }

    #[test]
    fn receipt_key_parser_rejects_non_hex() {
        let path = std::env::temp_dir().join(format!("dope-key-{}", std::process::id()));
        fs::write(&path, "not-a-key").unwrap();
        assert!(load_receipt_key(&path).is_err());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn receipt_key_creation_is_exclusive_and_private() {
        use std::os::unix::fs::PermissionsExt;

        let path = std::env::temp_dir().join(format!("dope-new-key-{}", std::process::id()));
        let _ = fs::remove_file(&path);
        let hashes = create_receipt_key(&path).unwrap();
        assert!(is_lower_hex(&hashes.sha256, &[64]));
        assert!(load_receipt_key(&path).is_ok());
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(create_receipt_key(&path).is_err());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn certification_statistics_do_not_fill_missing_evidence() {
        assert_eq!(mean_and_upper_95(&[]), None);
        assert_eq!(mean_and_lower_95(&[]), None);
        assert_eq!(float_percentile(&mut [], 0.95), None);
        let mut histogram = BTreeMap::from([(1, 1), (5, 2), (9, 1)]);
        assert_eq!(histogram_percentile(&histogram, 4, 0.95), Some(9));
        histogram.clear();
        assert_eq!(histogram_percentile(&histogram, 0, 0.95), None);
        assert!(is_timeout_failure("CUDA deadline exceeded"));
        assert!(!is_timeout_failure("deterministic model failure"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_cache_is_atomic_and_immutable() {
        let root = std::env::temp_dir().join(format!(
            "dope-content-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let path = root.join("aa").join("cell.bin");
        atomic_cache_write(&path, b"first").unwrap();
        atomic_cache_write(&path, b"second").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"first");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn shared_retention_scale() {
        assert_eq!(null_normalized_excess_loss_retention(1.0, 0.5, 0.75), 0.5);
        assert_eq!(null_normalized_excess_loss_retention(1.0, 0.5, 0.4), 1.2);
        assert!((null_normalized_excess_loss_retention(1.0, 0.5, 1.1) + 0.2).abs() < 1e-12);
    }

    #[test]
    fn low_signal_uses_absolute_noninferiority() {
        let passed = retention_decision(1.0, 0.995, 1.004);
        assert!(!passed.informative);
        assert_eq!(passed.retention, None);
        assert!(passed.absolute_noninferiority_passed);
        let failed = retention_decision(1.0, 0.995, 1.006);
        assert!(!failed.absolute_noninferiority_passed);
    }

    #[test]
    fn exact_and_near_copy_checks_include_missingness() {
        let real = Table {
            rows: 2,
            features: 2,
            columns: vec![vec![0.1, f32::NAN], vec![0.2, 0.8]],
            target: vec![0.3, 0.7],
        };
        let exact = real.clone();
        assert_eq!(exact_copy_count(&real, &exact), 2);
        assert_eq!(near_copy_count(&real, &exact), 2);
        let changed_mask = Table {
            rows: 1,
            features: 2,
            columns: vec![vec![0.1], vec![f32::NAN]],
            target: vec![0.3],
        };
        assert_eq!(near_copy_count(&real, &changed_mask), 0);
    }

    #[test]
    fn certification_rejects_10241_bytes_before_decoding() {
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("certification-byte-boundary-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let artifact = root.join("candidate.dpk");
        let output = root.join("certification.json");
        let policy = ReleasePolicy::new(AnonymizationTier::L3, None, false).unwrap();

        fs::write(&artifact, vec![b'X'; 10_240]).unwrap();
        let at_limit = certify_kernel_with_policy(&root, &artifact, &output, 1, 0, 1, &policy)
            .unwrap_err()
            .to_string();
        assert!(!at_limit.contains("byte limit"));

        fs::write(&artifact, vec![b'X'; 10_241]).unwrap();
        let over_limit = certify_kernel_with_policy(&root, &artifact, &output, 1, 0, 1, &policy)
            .unwrap_err()
            .to_string();
        assert!(over_limit.contains("byte limit"));
        assert!(!output.exists());
        fs::remove_dir_all(root).unwrap();
    }
}

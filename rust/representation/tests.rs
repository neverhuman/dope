#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_matches_rfc_4231_case_1() {
        let key = [0x0bu8; 20];
        let digest = hmac_sha256(&key, b"Hi There");
        assert_eq!(
            hex16(&digest),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn midrank_clamps_inside_the_open_unit_interval() {
        let fit = [1.0, 2.0, 4.0];
        let got = midrank_quantile(&fit, &[0.0, 1.0, 3.0, 4.0, 9.0]).unwrap();
        let expect = [0.5 / 3.0, 0.5 / 3.0, 2.0 / 3.0, 2.5 / 3.0, 2.5 / 3.0];
        for (actual, expected) in got.iter().zip(expect) {
            assert!((actual - expected).abs() < 1e-12);
            assert!(*actual > 0.0 && *actual < 1.0);
        }
    }

    #[test]
    fn synthetic_column_uses_the_real_fit() {
        let real = [0.0, 1.0, 2.0];
        let synthetic = [10.0, 11.0, 12.0];
        let frozen = midrank_quantile(&real, &synthetic).unwrap();
        let refit = midrank_quantile(&synthetic, &synthetic).unwrap();
        let upper = 2.5 / 3.0;
        assert_ne!(frozen, refit);
        assert!(frozen.iter().all(|value| (*value - upper).abs() < 1e-12));
        assert!(frozen.iter().all(|value| *value < 1.0));
    }

    #[test]
    fn manifest_keeps_hashes_and_drops_source_text() {
        let manifest = hash_manifest(
            b"salt",
            &[("age", 1, 1), ("red", 2, 1)],
            HEADLINE_ENCODER,
            100,
        );
        let encoded = format!("{manifest:?}");
        assert!(!encoded.contains("age"));
        assert!(!encoded.contains("red"));
        assert_eq!(manifest.columns[0].hash.len(), 16);
        assert_eq!(manifest.salt_id.len(), 16);
        assert!(!artifact_has_cleartext(
            encoded.as_bytes(),
            &[b"age", b"red"]
        ));
    }

    #[test]
    fn cleartext_scan_catches_a_header_byte_string() {
        assert!(artifact_has_cleartext(b"col,age\n1", &[b"age"]));
        assert!(!artifact_has_cleartext(b"col,a1e\n1", &[b"age"]));
    }

    #[test]
    fn closeness_rejects_a_zero_match_and_a_sign_flip() {
        assert!(representation_closeness(0.2, 0.5, 0.0, -0.1, -0.1).is_none());
        assert!(representation_closeness(0.2, 0.5, 0.4, 0.1, -0.1).is_none());
        let value = representation_closeness(0.2, 0.5, 0.4, -0.1, -0.2).unwrap();
        assert!((value - 0.5).abs() < 1e-12);
    }

    #[test]
    fn lookup_stays_outside_the_repo_and_holds_the_label() {
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let refused = write_local_lookup(
            &repo.join("production"),
            &repo,
            b"salt",
            &[("age", 1, 1)],
            &["red"],
        );
        assert!(refused.is_err());
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        // Self-hosted runners set the process temp directory inside the checkout.
        // The lookup has to be created somewhere the guard accepts.
        let tmp = ["/dev/shm", "/var/tmp", "/tmp"]
            .into_iter()
            .map(|base| PathBuf::from(base).join(format!("dope-mfs-v3-lookup-{stamp}")))
            .chain(std::iter::once(
                std::env::temp_dir().join(format!("dope-mfs-v3-lookup-{stamp}")),
            ))
            .find(|dir| lookup_inside_repo(dir, &repo).ok() == Some(false))
            .expect("a directory outside the repository");
        fs::create_dir_all(&tmp).unwrap();
        let path = write_local_lookup(&tmp, &repo, b"salt", &[("age", 1, 1)], &["red"]).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        let lookup = fs::read(&path).unwrap();
        assert!(artifact_has_cleartext(&lookup, &[b"age", b"red"]));
        let fixture = tmp.join("artifact.csv");
        fs::write(&fixture, b"header,age\n1").unwrap();
        assert!(artifact_has_cleartext(
            &fs::read(&fixture).unwrap(),
            &[b"age"]
        ));
        let manifest = hash_manifest(b"salt", &[("age", 1, 1)], HEADLINE_ENCODER, 100);
        let encoded = format!("{manifest:?}");
        assert!(!artifact_has_cleartext(
            encoded.as_bytes(),
            &[b"age", b"red"]
        ));
        assert!(!encoded.contains(&path.display().to_string()));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn tabular_transfer_requires_the_clean_models_and_the_shared_map() {
        let auditors: Vec<String> = TABULAR_AUDITORS
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        assert!(tabular_transfer_ok(NORMALIZER, TABULAR_PROTOCOL, &auditors));
        assert!(!tabular_transfer_ok(
            NORMALIZER,
            "auditor_retention",
            &auditors
        ));
        assert!(!tabular_transfer_ok(
            NORMALIZER,
            TABULAR_PROTOCOL,
            &["catboost".into()]
        ));
    }

    #[test]
    fn refused_encoder_is_not_the_headline() {
        assert!(headline_encoder(HEADLINE_ENCODER));
        assert!(!headline_encoder("foundation"));
        assert!(!headline_encoder("tabpfn35"));
        assert!(REFUSED_ENCODERS.contains(&"foundation"));
        assert!(SENSITIVITY_ENCODERS.contains(&"mitra_v2"));
    }

    #[test]
    fn midrank_and_unit_half_distance_refuse_overflow() {
        assert!(midrank_quantile(&[-1e308, 1e308], &[0.0]).is_none());
        assert!(midrank_quantile(&[-1e308, 1e308], &[0.5e308]).is_none());
        assert!(unit_half_distance(&[1e200, 0.0], &[0.0, 1e200]).is_none());
    }
}

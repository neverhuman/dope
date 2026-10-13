#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_digests() {
        assert_eq!(
            content_fingerprint(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            content_fingerprint(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn canonical_subset_rejects_duplicate_keys_and_floats() {
        assert_eq!(
            parse_json_subset(br#"{"a":1,"a":2}"#).unwrap_err().code(),
            "duplicate_json_key"
        );
        assert_eq!(
            parse_json_subset(b"1.5").unwrap_err().code(),
            "invalid_manifest"
        );
        assert_eq!(
            parse_json_subset(b"NaN").unwrap_err().code(),
            "nonfinite_json_number"
        );
    }

    #[test]
    fn expansion_cap_rejects_a_tiny_declared_bomb() {
        let mut artifact = vec![0u8; HEADER_BYTES];
        artifact[..4].copy_from_slice(b"DPK3");
        artifact[4] = 3;
        artifact[5] = 2;
        artifact[6] = 1;
        artifact[8] = 8;
        artifact[12] = 1;
        let payload = [0x81, 0x01, 0x80, 0x80, 0x80, 0x80, 0x08, 0x00];
        artifact[32..40].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        artifact.extend_from_slice(&payload);
        let error = preflight_dpk(&artifact, &BundleLimits::default()).unwrap_err();
        assert_eq!(error.code(), "expansion_cap_exceeded");
        assert!(artifact.len() < 128);
    }

    #[test]
    fn byte_sums_reject_u64_overflow() {
        assert_eq!(
            checked_add(u64::MAX, 1).unwrap_err().code(),
            "integer_overflow"
        );
    }

    fn hex_bytes(value: &str) -> Vec<u8> {
        (0..value.len())
            .step_by(2)
            .map(|index| u8::from_str_radix(&value[index..index + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn golden_accounting_matches_python_inventory() {
        let file = include_bytes!("../../schemas/bundle-golden-v1.json");
        let parsed = parse_json_subset(file).unwrap();
        assert_eq!(canonical_subset(&parsed).unwrap(), file);
        let cases = object(&parsed).unwrap().get("cases").unwrap();
        let Json::Array(cases) = cases else {
            panic!("cases")
        };
        for case in cases {
            let case = object(case).unwrap();
            let name = json_str(case.get("name").unwrap()).unwrap();
            let root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("target/vnext/pr07-golden")
                .join(name);
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).unwrap();
            let manifest = canonical_subset(case.get("manifest").unwrap()).unwrap();
            fs::write(root.join(MANIFEST_NAME), &manifest).unwrap();
            let Json::Object(files) = case.get("files_hex").unwrap() else {
                panic!("files")
            };
            for (path, encoded) in files {
                fs::write(root.join(path), hex_bytes(json_str(encoded).unwrap())).unwrap();
            }
            let expected = json_str(case.get("manifest_sha256").unwrap()).unwrap();
            assert_eq!(content_fingerprint(&manifest), expected);
            let verified = verify_bundle(&root, expected, &BundleLimits::default()).unwrap();
            let got = verified.accounting();
            let accounting = object(case.get("accounting").unwrap()).unwrap();
            let number =
                |key: &str| json_u64(accounting.get(key).unwrap(), "schema_mismatch").unwrap();
            assert_eq!(got.sampling_payload_bytes, number("sampling_payload_bytes"));
            assert_eq!(
                got.shared_learned_state_bytes,
                number("shared_learned_state_bytes")
            );
            assert_eq!(
                got.standalone_learned_state_bytes,
                number("standalone_learned_state_bytes")
            );
            assert_eq!(
                got.amortized_learned_state_bytes,
                number("amortized_learned_state_bytes")
            );
            assert_eq!(got.runtime_package_bytes, number("runtime_package_bytes"));
            assert_eq!(
                got.evidence_authorization_bytes,
                number("evidence_authorization_bytes")
            );
            assert_eq!(
                got.materialized_rows_bytes,
                number("materialized_rows_bytes")
            );
            assert_eq!(got.transform_bytes, number("transform_bytes"));
            assert_eq!(got.dictionary_bytes, number("dictionary_bytes"));
            assert_eq!(got.manifest_bytes, number("manifest_bytes"));
            assert_eq!(got.charged_sampling_bytes, number("charged_sampling_bytes"));
            assert_eq!(got.amortization_count, number("amortization_count"));
            let charged = got.charged_sampling_bytes;
            assert_eq!(charged_sampling_bytes(&verified), charged);
            assert!(within_budget(charged, charged));
            assert!(!within_budget(charged, charged - 1));
            assert!(within_budget(charged, charged + 1));
            assert!(got.standalone_learned_state_bytes >= got.amortized_learned_state_bytes);
            match json_str(accounting.get("transport_framing").unwrap()).unwrap() {
                "unmeasured" => {
                    assert_eq!(got.transport_framing, TransportFraming::Unmeasured);
                    assert!(got.complete_delivery_bytes.is_none());
                    assert!(accounting.get("complete_delivery_bytes").is_none());
                }
                "included" => {
                    assert_eq!(got.transport_framing, TransportFraming::Included);
                    assert_eq!(
                        got.complete_delivery_bytes,
                        Some(number("complete_delivery_bytes"))
                    );
                }
                _ => panic!("transport"),
            }
        }
    }

    fn open_directory(path: &Path) -> OwnedFd {
        let name = c_string(path.as_os_str().as_bytes()).unwrap();
        owned_fd({
            // SAFETY: `name` is a NUL-terminated path this test created. `open`
            // uses O_DIRECTORY, O_NOFOLLOW, and O_CLOEXEC.
            unsafe { open(name.as_ptr(), O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC, 0) }
        })
        .unwrap()
    }

    #[test]
    fn replaced_directory_path_does_not_change_the_fd_listing() {
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/vnext/pr07-fd-walk");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("real")).unwrap();
        fs::create_dir_all(base.join("decoy")).unwrap();
        fs::write(base.join("real/manifest.json"), b"{}").unwrap();
        fs::write(base.join("real/keep.bin"), b"ab").unwrap();
        fs::write(base.join("decoy/manifest.json"), b"{}").unwrap();
        fs::write(base.join("decoy/keep.bin"), b"ab").unwrap();
        fs::write(base.join("decoy/secret.bin"), b"hidden").unwrap();
        let real = base.join("real");
        let directory = open_directory(&real);
        fs::rename(&real, base.join("real.moved")).unwrap();
        std::os::unix::fs::symlink(base.join("decoy"), &real).unwrap();
        assert!(real.symlink_metadata().unwrap().file_type().is_symlink());
        let declared = BTreeSet::from(["manifest.json".to_string(), "keep.bin".to_string()]);
        let prefixes = BTreeSet::new();
        let mut found = BTreeSet::new();
        walk(
            directory,
            "",
            &mut WalkLists {
                declared: &declared,
                prefixes: &prefixes,
                found: &mut found,
            },
            WalkBudget {
                max_entries: 33,
                max_name_bytes: 33 * 64,
            },
        )
        .unwrap();
        assert_eq!(found, declared);
        assert!(!found.contains("secret.bin"));
    }

    #[test]
    fn directory_stream_stops_on_the_first_undeclared_or_cap() {
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/vnext/pr07-fd-cap");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        fs::write(base.join("extra.bin"), b"x").unwrap();
        let declared = BTreeSet::new();
        let prefixes = BTreeSet::new();
        let mut found = BTreeSet::new();
        let error = walk(
            open_directory(&base),
            "",
            &mut WalkLists {
                declared: &declared,
                prefixes: &prefixes,
                found: &mut found,
            },
            WalkBudget {
                max_entries: 8,
                max_name_bytes: 8 * 64,
            },
        )
        .unwrap_err();
        assert_eq!(error.code(), "undeclared_component");
        assert!(found.is_empty());
        assert!(!error.to_string().contains("extra.bin"));

        let error = walk(
            open_directory(&base),
            "",
            &mut WalkLists {
                declared: &declared,
                prefixes: &prefixes,
                found: &mut found,
            },
            WalkBudget {
                max_entries: 0,
                max_name_bytes: 64,
            },
        )
        .unwrap_err();
        assert_eq!(error.code(), "too_many_components");

        let error = walk(
            open_directory(&base),
            "",
            &mut WalkLists {
                declared: &declared,
                prefixes: &prefixes,
                found: &mut found,
            },
            WalkBudget {
                max_entries: 8,
                max_name_bytes: 1,
            },
        )
        .unwrap_err();
        assert_eq!(error.code(), "too_many_components");
    }

    fn dpk_header(magic: &[u8], major: u8, minor: u8) -> Vec<u8> {
        let mut artifact = vec![0u8; HEADER_BYTES];
        artifact[..4].copy_from_slice(magic);
        artifact[4] = major;
        artifact[5] = minor;
        artifact[6] = 1;
        artifact[8] = 8;
        artifact[12] = 1;
        artifact
    }

    fn with_payload(mut artifact: Vec<u8>, payload: &[u8]) -> Vec<u8> {
        artifact[32..40].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        artifact.extend_from_slice(payload);
        artifact
    }

    #[test]
    fn preflight_accepts_historical_dpk3_minors_through_three() {
        for minor in [0u8, 1, 2, 3] {
            let summary =
                preflight_dpk(&dpk_header(b"DPK3", 3, minor), &BundleLimits::default()).unwrap();
            assert_eq!(summary.minor, minor);
            assert!(summary.v3);
        }
        assert_eq!(
            preflight_dpk(&dpk_header(b"DPK3", 3, 4), &BundleLimits::default())
                .unwrap_err()
                .code(),
            "unsupported_dpk_version"
        );
        assert!(
            !preflight_dpk(&dpk_header(b"DPK2", 2, 0), &BundleLimits::default())
                .unwrap()
                .v3
        );
        assert!(preflight_dpk(&dpk_header(b"DPK2", 2, 1), &BundleLimits::default()).is_ok());
        assert_eq!(
            preflight_dpk(&dpk_header(b"DPK2", 2, 2), &BundleLimits::default())
                .unwrap_err()
                .code(),
            "unsupported_dpk_version"
        );
        let early = with_payload(dpk_header(b"DPK3", 3, 1), &[0x06, 0x00]);
        assert_eq!(
            preflight_dpk(&early, &BundleLimits::default())
                .unwrap_err()
                .code(),
            "unknown_section"
        );
        let current = with_payload(dpk_header(b"DPK3", 3, 2), &[0x06, 0x00]);
        assert!(preflight_dpk(&current, &BundleLimits::default()).is_ok());
    }

    fn measured(
        id: &str,
        role: Role,
        bytes: u64,
        required: bool,
        depends_on: &[&str],
    ) -> Component {
        Component {
            id: id.to_string(),
            role,
            path: format!("{id}.bin"),
            sha256: "ab".repeat(32),
            bytes,
            kind: Kind::Opaque,
            required,
            consumed_by_sampler: role == Role::Payload,
            depends_on: depends_on.iter().map(|item| (*item).to_string()).collect(),
            measured_bytes: bytes,
        }
    }

    #[test]
    fn optional_payload_dependency_stays_in_the_sampling_budget() {
        let linked = vec![
            measured("payload", Role::Payload, 4, true, &["evidence"]),
            measured("evidence", Role::Evidence, 7, false, &[]),
        ];
        let accounting = account_inventory(&linked, 10, 1, TransportFraming::Unmeasured).unwrap();
        assert_eq!(accounting.charged_sampling_bytes, 21);

        let detached = vec![
            measured("payload", Role::Payload, 4, true, &[]),
            measured("evidence", Role::Evidence, 7, false, &[]),
        ];
        let accounting = account_inventory(&detached, 10, 1, TransportFraming::Unmeasured).unwrap();
        assert_eq!(accounting.charged_sampling_bytes, 14);

        let rows = vec![
            measured("payload", Role::Payload, 4, true, &["rows"]),
            measured("rows", Role::Rows, 9, false, &[]),
        ];
        let accounting = account_inventory(&rows, 10, 1, TransportFraming::Unmeasured).unwrap();
        assert_eq!(accounting.charged_sampling_bytes, 23);

        let dictionary = vec![
            measured("payload", Role::Payload, 4, true, &["dictionary"]),
            measured("dictionary", Role::Dictionary, 5, false, &[]),
        ];
        let accounting =
            account_inventory(&dictionary, 10, 1, TransportFraming::Unmeasured).unwrap();
        assert_eq!(accounting.charged_sampling_bytes, 19);
        assert_eq!(accounting.dictionary_bytes, 5);
    }

    #[test]
    fn open_flags_come_from_libc_not_x86_64_literals() {
        let source = include_str!("../bundle.rs");
        assert!(source.contains("libc::O_NOFOLLOW"));
        assert!(source.contains("libc::O_DIRECTORY"));
        assert!(source.contains("libc::O_CLOEXEC"));
        assert!(source.contains("libc::O_NONBLOCK"));
        assert!(source.contains("libc::AT_SYMLINK_NOFOLLOW"));
        assert!(!source.contains("const O_NOFOLLOW: i32 = 0x20000"));
        assert!(!source.contains("const O_DIRECTORY: i32 = 0x10000"));
    }
}

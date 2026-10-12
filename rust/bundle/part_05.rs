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
}

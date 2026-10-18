#[cfg(test)]
pub mod adversarial {
    use super::*;

    #[test]
    fn canonical_container_order_is_deterministic_across_permutations() {
        let a = canonical_container(&json!([
            {"name": "weights.bin", "hex": "01020304"},
            {"name": "header.json", "hex": "7b7d"}
        ]))
        .unwrap();
        let b = canonical_container(&json!([
            {"name": "header.json", "hex": "7b7d"},
            {"name": "weights.bin", "hex": "01020304"}
        ]))
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn packing_and_denominator_attacks_never_shrink_charged_bytes() {
        let honest = evaluate(&json!({
            "sections": [
                {"name": "quantized_packed", "hex": "deadbeefcafebabe"}
            ],
            "closure_files": [
                {"name": "decoder_lut", "hex": "00112233445566778899aabbccddeeff", "scope": "runtime", "dataset_specific": false}
            ]
        }));
        let attacked = evaluate(&json!({
            "sections": [
                {"name": "quantized_packed", "hex": "deadbeefcafebabe"}
            ],
            "closure_files": [
                {"name": "decoder_lut", "hex": "00112233445566778899aabbccddeeff", "scope": "runtime", "dataset_specific": false}
            ],
            "encoded_numbers": 1,
            "encoded_scalars": 1,
            "row_count": 1_000_000,
            "denominator": 1000
        }));
        assert_eq!(honest["status"], "complete");
        assert_eq!(honest["payload_bytes"], attacked["payload_bytes"]);
        assert_eq!(
            honest["sampling_closure_bytes"],
            attacked["sampling_closure_bytes"]
        );
    }

    #[test]
    fn hidden_sidecar_checkpoint_and_path_traversal_attacks_fail_closed() {
        // Dataset-specific sidecar disguised as closure file is rejected.
        let sidecar = evaluate(&json!({
            "sections": [{"name": "payload", "hex": "0102"}],
            "closure_files": [{"name": "sidecar", "hex": "0102", "scope": "runtime", "dataset_specific": true}]
        }));
        assert_eq!(sidecar["status"], "failed");
        assert_eq!(sidecar["reason"], "undeclared_or_dataset_specific_closure");

        // Overlapping section and closure entry is rejected.
        let dup = evaluate(&json!({
            "sections": [{"name": "prior.bin", "hex": "0102"}],
            "closure_files": [{"name": "prior.bin", "hex": "0102", "scope": "shared_prior", "dataset_specific": false}]
        }));
        assert_eq!(dup["status"], "failed");
        assert_eq!(dup["reason"], "duplicate_charge_entry");

        // Path traversal or uppercase section names are rejected.
        for bad_name in ["../source.csv", "/etc/passwd", "UpperCase", ""] {
            let res = evaluate(&json!({
                "sections": [{"name": bad_name, "hex": "0102"}],
                "closure_files": []
            }));
            assert_eq!(res["status"], "failed");
            assert_eq!(res["reason"], "invalid_section_name");
        }

        // Undeclared sidecar/checkpoint/source read in sampler trace fails closed.
        assert_eq!(
            verify_sampler_access_trace(
                &["payload.dpk"],
                &["runtime.so"],
                &["payload.dpk", "hidden_checkpoint.pt"]
            ),
            Err(MetricError::new("undeclared_sampler_file_access"))
        );
    }

    #[test]
    fn copied_rows_and_gzip_controls_are_charged_full_container_and_closure_bytes() {
        let copied_rows = evaluate(&json!({
            "sections": [
                {"name": "copied_rows.bin", "hex": "ff".repeat(256)}
            ],
            "closure_files": [
                {"name": "row_sampler.py", "hex": "aa".repeat(64), "scope": "runtime", "dataset_specific": false}
            ]
        }));
        assert_eq!(copied_rows["status"], "complete");
        assert!(copied_rows["payload_bytes"].as_u64().unwrap() > 256);
        assert_eq!(
            copied_rows["sampling_closure_bytes"].as_u64().unwrap(),
            copied_rows["payload_bytes"].as_u64().unwrap() + 64
        );

        let gzip_control = evaluate(&json!({
            "sections": [
                {"name": "train.csv.gz", "hex": "1f8b0800".repeat(32)}
            ],
            "closure_files": [
                {"name": "gunzip_replay.py", "hex": "bb".repeat(48), "scope": "runtime", "dataset_specific": false}
            ]
        }));
        assert_eq!(gzip_control["status"], "complete");
        assert!(gzip_control["payload_bytes"].as_u64().unwrap() > 128);
        assert_eq!(
            gzip_control["sampling_closure_bytes"].as_u64().unwrap(),
            gzip_control["payload_bytes"].as_u64().unwrap() + 48
        );
    }
}

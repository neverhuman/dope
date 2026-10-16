//! Deterministic reference container and exact byte scopes (T-034, T-035).
//!
//! This fixture container is a conformance format, not a replacement DPK
//! decoder. No row count, scalar count, or denominator can reduce charged bytes.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::{MetricError, MetricResult, count, failure_with_status, require};

pub const MAGIC: &[u8; 4] = b"TDL1";

fn valid_section_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > 64 {
        return false;
    }
    if !(bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit()) {
        return false;
    }
    bytes[1..]
        .iter()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(*b, b'_' | b'.' | b'-'))
}

fn decode_hex_lowercase(encoded: &str) -> MetricResult<Vec<u8>> {
    require(encoded.len() % 2 == 0, "invalid_section_hex")?;
    let bytes = encoded.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let nibble = |b: u8| -> MetricResult<u8> {
            match b {
                b'0'..=b'9' => Ok(b - b'0'),
                b'a'..=b'f' => Ok(b - b'a' + 10),
                _ => Err(MetricError::new("invalid_section_hex")),
            }
        };
        let hi = nibble(chunk[0])?;
        let lo = nibble(chunk[1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn parse_sections(value: &Value) -> MetricResult<BTreeMap<String, Vec<u8>>> {
    let arr = value
        .as_array()
        .ok_or(MetricError::new("invalid_sections"))?;
    let mut sections = BTreeMap::new();
    for item in arr {
        let obj = item
            .as_object()
            .ok_or(MetricError::new("invalid_section"))?;
        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .ok_or(MetricError::new("invalid_section_name"))?;
        require(valid_section_name(name), "invalid_section_name")?;
        require(!sections.contains_key(name), "duplicate_section")?;
        let hex_str = obj
            .get("hex")
            .and_then(Value::as_str)
            .ok_or(MetricError::new("invalid_section_hex"))?;
        let decoded = decode_hex_lowercase(hex_str)?;
        sections.insert(name.to_owned(), decoded);
    }
    Ok(sections)
}

pub fn canonical_container(sections_value: &Value) -> MetricResult<Vec<u8>> {
    let sections = parse_sections(sections_value)?;
    Ok(encode_container(&sections))
}

fn encode_container(sections: &BTreeMap<String, Vec<u8>>) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(sections.len() as u32).to_le_bytes());
    for (name, content) in sections {
        let token = name.as_bytes();
        out.extend_from_slice(&(token.len() as u32).to_le_bytes());
        out.extend_from_slice(token);
        out.extend_from_slice(&(content.len() as u64).to_le_bytes());
        out.extend_from_slice(content);
    }
    out
}

pub fn evaluate(data: &Value) -> Value {
    match evaluate_inner(data) {
        Ok(val) => val,
        Err(err) => failure_with_status(err.reason, err.status),
    }
}

fn evaluate_inner(data: &Value) -> MetricResult<Value> {
    let obj = data.as_object().ok_or(MetricError::new("invalid_input"))?;
    if obj.get("sections").is_none_or(Value::is_null)
        || obj.get("closure_files").is_none_or(Value::is_null)
    {
        return Err(MetricError::with_status(
            "missing_charge_manifest",
            "incomplete",
        ));
    }
    let sections = parse_sections(&obj["sections"])?;
    require(!sections.is_empty(), "empty_payload")?;
    let closure = parse_sections(&obj["closure_files"])?;
    let closure_arr = obj["closure_files"]
        .as_array()
        .ok_or(MetricError::new("invalid_sections"))?;
    let valid_closure_meta = closure_arr.iter().all(|file| {
        let scope_ok = matches!(
            file.get("scope").and_then(Value::as_str),
            Some("runtime" | "shared_prior")
        );
        let ds_ok = file.get("dataset_specific").and_then(Value::as_bool) == Some(false);
        scope_ok && ds_ok
    });
    require(valid_closure_meta, "undeclared_or_dataset_specific_closure")?;

    let section_names: BTreeSet<&String> = sections.keys().collect();
    let closure_names: BTreeSet<&String> = closure.keys().collect();
    require(
        section_names.is_disjoint(&closure_names),
        "duplicate_charge_entry",
    )?;

    let artifact = encode_container(&sections);
    let payload = artifact.len() as u64;
    let content_sum: u64 = sections.values().map(|c| c.len() as u64).sum();
    let framing_bytes = payload - content_sum;
    let runtime: u64 = closure.values().map(|c| c.len() as u64).sum();
    let total = count(&json!(payload.saturating_add(runtime)), 0)?;
    let users = match obj.get("amortization_count") {
        Some(v) if !v.is_null() => count(v, 1)?,
        _ => 1,
    };
    let amortized = (payload as f64) + (runtime as f64) / (users as f64);
    let sha = format!("{:x}", Sha256::digest(&artifact));

    let mut section_bytes = Map::new();
    for (name, content) in &sections {
        section_bytes.insert(name.clone(), json!(content.len() as u64));
    }

    Ok(json!({
        "status": "complete",
        "reason": Value::Null,
        "payload_bytes": payload,
        "section_bytes": Value::Object(section_bytes),
        "framing_bytes": framing_bytes,
        "shared_bytes": runtime,
        "sampling_closure_bytes": total,
        "cold_bundle_bytes": total,
        "amortized_bytes": amortized,
        "artifact_sha256": sha,
    }))
}

/// Verify that a runtime file-access trace opens only declared payload sections
/// or closure files and never reads undeclared sidecars, checkpoints, or source data.
pub fn verify_sampler_access_trace(
    declared_sections: &[&str],
    declared_closure: &[&str],
    observed_reads: &[&str],
) -> MetricResult<()> {
    let allowed: BTreeSet<&str> = declared_sections
        .iter()
        .chain(declared_closure.iter())
        .copied()
        .collect();
    require(!observed_reads.is_empty(), "empty_sampler_trace")?;
    for &path in observed_reads {
        require(
            valid_section_name(path) && allowed.contains(path),
            "undeclared_sampler_file_access",
        )?;
    }
    Ok(())
}

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

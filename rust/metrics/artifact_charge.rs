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

fn encode_hex_lowercase(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

pub fn dpk_section_name(tag: u8) -> MetricResult<&'static str> {
    match tag {
        1 => Ok("01_schema"),
        2 => Ok("02_marginals"),
        3 => Ok("03_dependence"),
        4 => Ok("04_target"),
        5 => Ok("05_noise"),
        6 => Ok("06_neural"),
        _ => Err(MetricError::new("invalid_section_name")),
    }
}

/// Extract the exact encoded section slices of a native DPK3 artifact into a
/// canonical `TDL1` `"sections"` JSON array (`[{"name": ..., "hex": ...}, ...]`).
pub fn dpk_to_tdl1_sections(dpk_bytes: &[u8]) -> MetricResult<Value> {
    let accounting = crate::codec::account_artifact(dpk_bytes)
        .map_err(|_| MetricError::new("invalid_dpk_artifact"))?;
    require(!accounting.sections.is_empty(), "empty_payload")?;
    let mut cursor = accounting.header_bytes;
    let mut items = Vec::with_capacity(accounting.sections.len());
    for section in &accounting.sections {
        let name = dpk_section_name(section.tag)?;
        let content_start = cursor
            .checked_add(section.framing_bytes)
            .ok_or(MetricError::new("invalid_dpk_artifact"))?;
        let content_end = content_start
            .checked_add(section.encoded_bytes)
            .ok_or(MetricError::new("invalid_dpk_artifact"))?;
        let slice = dpk_bytes
            .get(content_start..content_end)
            .ok_or(MetricError::new("invalid_dpk_artifact"))?;
        cursor = content_end;
        items.push(json!({
            "name": name,
            "hex": encode_hex_lowercase(slice),
        }));
    }
    require(cursor == dpk_bytes.len(), "invalid_dpk_artifact")?;
    Ok(Value::Array(items))
}

/// Decode and validate a canonical binary `TDL1` container.
pub fn decode_container(bytes: &[u8]) -> MetricResult<BTreeMap<String, Vec<u8>>> {
    require(
        bytes.len() >= 8 && bytes.starts_with(MAGIC),
        "invalid_container",
    )?;
    let count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    require(count > 0, "empty_payload")?;
    let mut offset = 8usize;
    let mut sections = BTreeMap::new();
    let mut prev_name: Option<String> = None;
    for _ in 0..count {
        let name_len_end = offset
            .checked_add(4)
            .ok_or(MetricError::new("invalid_container"))?;
        let name_len_bytes = bytes
            .get(offset..name_len_end)
            .ok_or(MetricError::new("invalid_container"))?;
        let name_len = u32::from_le_bytes(name_len_bytes.try_into().unwrap()) as usize;
        offset = name_len_end;

        let name_end = offset
            .checked_add(name_len)
            .ok_or(MetricError::new("invalid_container"))?;
        let name_bytes = bytes
            .get(offset..name_end)
            .ok_or(MetricError::new("invalid_container"))?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| MetricError::new("invalid_section_name"))?;
        require(valid_section_name(name), "invalid_section_name")?;
        if let Some(prev) = &prev_name {
            if name == prev {
                return Err(MetricError::new("duplicate_section"));
            }
            require(name > prev.as_str(), "noncanonical_section_order")?;
        }
        prev_name = Some(name.to_owned());
        offset = name_end;

        let content_len_end = offset
            .checked_add(8)
            .ok_or(MetricError::new("invalid_container"))?;
        let content_len_bytes = bytes
            .get(offset..content_len_end)
            .ok_or(MetricError::new("invalid_container"))?;
        let content_len =
            usize::try_from(u64::from_le_bytes(content_len_bytes.try_into().unwrap()))
                .map_err(|_| MetricError::new("invalid_container"))?;
        offset = content_len_end;

        let content_end = offset
            .checked_add(content_len)
            .ok_or(MetricError::new("invalid_container"))?;
        let content = bytes
            .get(offset..content_end)
            .ok_or(MetricError::new("invalid_container"))?;
        offset = content_end;

        sections.insert(name.to_owned(), content.to_vec());
    }
    require(offset == bytes.len(), "invalid_container")?;
    Ok(sections)
}

/// Produce `ArtifactAccounting` for a canonical `TDL1` container so `dope inspect`
/// and `artifact_charge` share identical header, section framing, and payload totals.
pub fn account_tdl1_container(bytes: &[u8]) -> MetricResult<crate::codec::ArtifactAccounting> {
    let sections = decode_container(bytes)?;
    let mut accounted = Vec::with_capacity(sections.len());
    for (idx, (name, content)) in sections.iter().enumerate() {
        let tag = u8::try_from(idx + 1).map_err(|_| MetricError::new("invalid_container"))?;
        accounted.push(crate::codec::SectionAccounting {
            tag,
            framing_bytes: 12 + name.len(),
            raw_bytes: content.len(),
            encoded_bytes: content.len(),
            compressed: false,
        });
    }
    Ok(crate::codec::ArtifactAccounting {
        header_bytes: 8,
        sections: accounted,
        total_bytes: bytes.len(),
    })
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

fn parse_closure_files(closure_value: &Value) -> MetricResult<BTreeMap<String, Vec<u8>>> {
    let closure = parse_sections(closure_value)?;
    let closure_arr = closure_value
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
    Ok(closure)
}

fn finalize_charge(
    artifact: &[u8],
    section_lengths: &BTreeMap<String, u64>,
    closure: &BTreeMap<String, Vec<u8>>,
    users: u64,
) -> MetricResult<Value> {
    let section_names: BTreeSet<&String> = section_lengths.keys().collect();
    let closure_names: BTreeSet<&String> = closure.keys().collect();
    require(
        section_names.is_disjoint(&closure_names),
        "duplicate_charge_entry",
    )?;
    let payload = artifact.len() as u64;
    let content_sum: u64 = section_lengths.values().copied().sum();
    require(payload >= content_sum, "invalid_container")?;
    let framing_bytes = payload - content_sum;
    let runtime: u64 = closure.values().map(|c| c.len() as u64).sum();
    let total = count(&json!(payload.saturating_add(runtime)), 0)?;
    let amortized = (payload as f64) + (runtime as f64) / (users as f64);
    let sha = format!("{:x}", Sha256::digest(artifact));

    let mut section_bytes = Map::new();
    for (name, &len) in section_lengths {
        section_bytes.insert(name.clone(), json!(len));
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

/// Evaluate exact byte charge directly on a binary `TDL1` container or native `DPK3` artifact.
/// A null closure declaration is unmeasured; an explicit empty array declares no extra files.
pub fn evaluate_artifact_bytes(
    artifact: &[u8],
    closure_files: &Value,
    amortization_count: Option<u64>,
) -> Value {
    let inner = (|| -> MetricResult<Value> {
        if closure_files.is_null() {
            return Err(MetricError::with_status(
                "missing_charge_manifest",
                "incomplete",
            ));
        }
        let closure = parse_closure_files(closure_files)?;
        let users = match amortization_count {
            Some(u) => count(&json!(u), 1)?,
            None => 1,
        };
        let mut section_lengths = BTreeMap::new();
        if artifact.starts_with(MAGIC) {
            let sections = decode_container(artifact)?;
            for (name, content) in sections {
                section_lengths.insert(name, content.len() as u64);
            }
        } else if artifact.starts_with(crate::codec::MAGIC) {
            let accounting = crate::codec::account_artifact(artifact)
                .map_err(|_| MetricError::new("invalid_dpk_artifact"))?;
            require(!accounting.sections.is_empty(), "empty_payload")?;
            for section in &accounting.sections {
                let name = dpk_section_name(section.tag)?;
                section_lengths.insert(name.to_owned(), section.encoded_bytes as u64);
            }
        } else {
            return Err(MetricError::new("invalid_container"));
        }
        finalize_charge(artifact, &section_lengths, &closure, users)
    })();
    match inner {
        Ok(val) => val,
        Err(err) => failure_with_status(err.reason, err.status),
    }
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
    let closure = parse_closure_files(&obj["closure_files"])?;
    let users = match obj.get("amortization_count") {
        Some(v) if !v.is_null() => count(v, 1)?,
        _ => 1,
    };
    let artifact = encode_container(&sections);
    let section_lengths: BTreeMap<String, u64> = sections
        .iter()
        .map(|(name, content)| (name.clone(), content.len() as u64))
        .collect();
    finalize_charge(&artifact, &section_lengths, &closure, users)
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

include!("artifact_charge/adversarial.rs");

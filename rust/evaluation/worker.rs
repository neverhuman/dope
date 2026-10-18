//! Bounded, fixture-only native model worker ABI. No query-label field, paths,
//! credentials, pickle, vendor import or process launch exists in this contract.
//! Runtime isolation and actual checkpoint/GPU qualification remain executor work.
use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use super::{encoded, sha};
use crate::error::{DopeError, Result};
use crate::evidence::{canonical_json, parse_json};

mod schema;
#[cfg(test)]
mod tests;

pub const MAX_ENVELOPE: usize = 4 * 1024 * 1024;
pub const MAX_BUFFER: usize = 64 * 1024 * 1024;
pub const MAX_TOTAL: usize = 128 * 1024 * 1024;
pub type Buffers = BTreeMap<String, Vec<u8>>;

fn invalid() -> DopeError {
    DopeError::Data("worker_abi_invalid".into())
}
fn require(ok: bool) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid()) }
}
fn n(value: &Value) -> Option<u64> {
    value.as_u64()
}
fn number(value: &Value) -> u64 {
    n(value).unwrap_or(u64::MAX)
}
fn array(value: &Value) -> &[Value] {
    value.as_array().map(Vec::as_slice).unwrap_or_default()
}
fn hash(value: &Value) -> Result<String> {
    Ok(sha(&canonical_json(value).map_err(|_| invalid())?))
}
fn decode(bytes: &[u8], index: usize) -> Result<Value> {
    require(bytes.len() <= MAX_ENVELOPE)?;
    let value = parse_json(bytes).map_err(|_| invalid())?;
    schema::shape(&value, index)?;
    Ok(value)
}
fn predict(value: &Value) -> bool {
    matches!(
        value["method"].as_str(),
        Some("predict" | "predict_perturbations")
    )
}
fn active(value: &Value) -> bool {
    predict(value) || value["method"] == "fit_or_prepare_context"
}
fn unique(ids: &[Value]) -> bool {
    ids.iter()
        .filter_map(Value::as_str)
        .collect::<BTreeSet<_>>()
        .len()
        == ids.len()
}
fn buffer(descriptor: &Value, bytes: &[u8]) -> Result<()> {
    require(
        bytes.len() <= MAX_BUFFER
            && number(&descriptor["byte_length"]) == bytes.len() as u64
            && number(&descriptor["rows"])
                .checked_mul(number(&descriptor["columns"]))
                .and_then(|n| n.checked_mul(8))
                == Some(bytes.len() as u64)
            && array(&descriptor["row_ids"]).len() as u64 == number(&descriptor["rows"])
            && unique(array(&descriptor["row_ids"]))
            && descriptor["sha256"] == sha(bytes),
    )?;
    require(
        bytes
            .chunks_exact(8)
            .all(|b| f64::from_le_bytes(b.try_into().expect("eight-byte chunk")).is_finite()),
    )
}
fn output(value: &Value) -> Result<()> {
    let classes = array(&value["class_order"]);
    require(unique(classes) && classes.len() <= 4096)?;
    require(match value["kind"].as_str() {
        Some("probability") => !classes.is_empty() && value["units"] == "probability",
        Some("point") => classes.is_empty() && value["units"] == "target_original",
        Some("distribution") => {
            classes.is_empty() && value["units"] == "distribution_target_original"
        }
        _ => false,
    })
}
fn runtime_version(value: &Value) -> bool {
    let Some(text) = value.as_str() else {
        return false;
    };
    if text.len() > 128 {
        return false;
    }
    let (version, suffix) = text.split_once(['-', '+']).unwrap_or((text, ""));
    let parts: Vec<_> = version.split('.').collect();
    parts.len() >= 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
        && (version.len() == text.len() || !suffix.is_empty())
        && suffix
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}

/// Constructible only after schema, byte and custodian-pin verification. The
/// request SHA is the cache identity and covers every policy/count/resource pin.
pub struct ValidatedRequest(Value);
impl ValidatedRequest {
    pub fn identity(&self) -> &str {
        self.0["request_sha256"].as_str().expect("validated digest")
    }
}

/// Trusted pins are supplied independently by the owner, not read from the worker.
/// A matching digest still does not prove a vendor checkpoint was executed.
pub fn validate_request(
    bytes: &[u8],
    buffers: &Buffers,
    expected_pins: &Value,
) -> Result<ValidatedRequest> {
    let request = decode(bytes, 0)?;
    require(
        request["pins"] == *expected_pins && runtime_version(&request["pins"]["runtime_version"]),
    )?;
    output(&request["output"])?;
    let inputs = request["inputs"].as_object().ok_or_else(invalid)?;
    let names: BTreeSet<_> = inputs
        .iter()
        .filter(|(_, v)| !v.is_null())
        .map(|(k, _)| k)
        .collect();
    require(
        names == buffers.keys().collect()
            && buffers
                .values()
                .try_fold(0usize, |n, b| n.checked_add(b.len()))
                .is_some_and(|n| n <= MAX_TOTAL),
    )?;
    for (name, bytes) in buffers {
        buffer(&inputs[name], bytes)?;
    }
    let counts = &request["counts"];
    let policy = &request["policy"];
    require(
        (policy["subsampling"] == "none" || policy["truncation"] == "none")
            && (policy["query_policy"] != "rowwise" || policy["query_batch_size"] == 1),
    )?;
    let ids = array(&request["context_row_ids"]);
    require(
        ["train_features", "train_labels"]
            .iter()
            .all(|k| inputs[*k].is_null() != active(&request))
            && inputs["query_features"].is_null() != predict(&request),
    )?;
    if active(&request) {
        require(!request["feature_schema_sha256"].is_null())?;
        let train = &inputs["train_features"];
        let labels = &inputs["train_labels"];
        let provided = number(&policy["primary_n"]) * number(&policy["sample_multiplier"]);
        require(
            !request["training_role"].is_null()
                && ["requested_rows", "produced_rows", "provided_rows"]
                    .iter()
                    .all(|k| number(&counts[*k]) == provided)
                && number(&train["rows"]) == provided
                && number(&labels["rows"]) == provided
                && labels["columns"] == 1
                && labels["row_ids"] == train["row_ids"],
        )?;
        let train_ids = array(&train["row_ids"]);
        if request["output"]["kind"] == "probability" {
            let classes = array(&request["output"]["class_order"]).len() as f64;
            require(buffers["train_labels"].chunks_exact(8).all(|b| {
                let v = f64::from_le_bytes(b.try_into().expect("eight-byte chunk"));
                v.fract() == 0.0 && v >= 0.0 && v < classes
            }))?;
        }
        let train_set: BTreeSet<_> = train_ids.iter().filter_map(Value::as_str).collect();
        require(
            !ids.is_empty()
                && ids.len() as u64 == number(&counts["consumed_rows"])
                && ids.len() as u64 <= provided.min(number(&policy["context_cap"]))
                && unique(ids)
                && ids
                    .iter()
                    .all(|id| id.as_str().is_some_and(|id| train_set.contains(id))),
        )?;
        if policy["subsampling"] == "none" {
            let expected = if policy["truncation"] == "head" {
                provided.min(number(&policy["context_cap"]))
            } else {
                provided
            };
            require(ids.len() as u64 == expected && train_ids.get(..ids.len()) == Some(ids))?;
        }
        if predict(&request) {
            let query = &inputs["query_features"];
            require(
                query["columns"] == train["columns"]
                    && number(&query["rows"]) <= number(&policy["query_batch_size"])
                    && array(&query["row_ids"])
                        .iter()
                        .all(|id| id.as_str().is_some_and(|id| !train_set.contains(id)))
                    && (policy["query_policy"] != "rowwise" || query["rows"] == 1),
            )?;
        }
    } else {
        require(
            ids.is_empty()
                && request["feature_schema_sha256"].is_null()
                && request["training_role"].is_null()
                && counts
                    .as_object()
                    .ok_or_else(invalid)?
                    .values()
                    .all(|v| v == 0),
        )?;
    }
    let context = serde_json::json!({"train_features": inputs["train_features"],
        "train_labels": inputs["train_labels"], "context_row_ids": ids});
    require(
        request["context_sha256"] == hash(&context)?
            && request["query_sha256"] == hash(&inputs["query_features"])?,
    )?;
    let mut unsigned = request.clone();
    unsigned
        .as_object_mut()
        .ok_or_else(invalid)?
        .remove("request_sha256");
    require(request["request_sha256"] == hash(&unsigned)?)?;
    Ok(ValidatedRequest(request))
}

/// Rust verifies/reduces the response. It deliberately cannot certify external
/// runtime, checkpoint or GPU execution based on worker self-reports.
pub fn validate_response(
    bytes: &[u8],
    buffers: &Buffers,
    request: &ValidatedRequest,
) -> Result<Value> {
    let response = decode(bytes, 1)?;
    let request = &request.0;
    for key in [
        "request_sha256",
        "pins",
        "context_sha256",
        "query_sha256",
        "output",
    ] {
        require(response[key] == request[key])?;
    }
    for key in ["requested_rows", "produced_rows", "provided_rows"] {
        require(response["counts"][key] == request["counts"][key])?;
    }
    for key in ["wall_ms", "memory_bytes", "vram_bytes"] {
        require(number(&response["usage"][key]) <= number(&request["resources"][key]))?;
    }
    let descriptor = &response["prediction"];
    if response["status"] != "ok" {
        require(
            !response["error"].is_null()
                && descriptor.is_null()
                && buffers.is_empty()
                && number(&response["counts"]["consumed_rows"])
                    <= number(&request["counts"]["consumed_rows"]),
        )?;
    } else {
        require(response["error"].is_null() && response["counts"] == request["counts"])?;
        let expected: BTreeSet<_> = if predict(request) {
            ["prediction"].into_iter().collect()
        } else {
            BTreeSet::new()
        };
        require(
            descriptor.is_null() != predict(request)
                && buffers.keys().map(String::as_str).collect::<BTreeSet<_>>() == expected,
        )?;
        if predict(request) {
            let bytes = &buffers["prediction"];
            buffer(descriptor, bytes)?;
            require(descriptor["row_ids"] == request["inputs"]["query_features"]["row_ids"])?;
            let kind = &request["output"]["kind"];
            let columns = number(&descriptor["columns"]);
            require(
                (kind != "point" || columns == 1)
                    && (kind != "probability"
                        || columns == array(&request["output"]["class_order"]).len() as u64),
            )?;
            if kind == "probability" {
                for row in bytes.chunks_exact(columns as usize * 8) {
                    let values = row
                        .chunks_exact(8)
                        .map(|b| f64::from_le_bytes(b.try_into().expect("eight-byte chunk")));
                    let mut sum = 0.0;
                    for value in values {
                        require((0.0..=1.0).contains(&value))?;
                        sum += value;
                    }
                    require((sum - 1.0).abs() <= 1e-7)?;
                }
            }
        }
    }
    // The closed schema requires false/fixture_only even for the qualify method.
    Ok(
        serde_json::json!({"request_sha256": request["request_sha256"],
        "qualified": false, "qualification": "fixture_only", "response_sha256": hash(&response)?}),
    )
}

/// Serialize only the validated contract; errors never include user input.
pub fn request_bytes(request: &ValidatedRequest) -> Result<Vec<u8>> {
    encoded(&request.0).map_err(|_| invalid())
}

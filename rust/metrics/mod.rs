//! TDL (Transfer Description Length) metric core: task-conditioned transfer,
//! worst-family cluster bootstrap aggregation, grouped permutation driver
//! fidelity, byte-budget Pareto frontier, and canonical byte accounting.
//!
//! All public metric surfaces share [`METRIC_NAME`] (`"TDL"`) so the display
//! identifier can be renamed in one place once decision D-01 is finalized.

use std::collections::{BTreeMap, BTreeSet};

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value, json};

pub mod aggregate;
pub mod artifact_charge;
pub mod drivers;
pub mod frontier;
pub mod golden;
pub mod transfer;

/// Working public metric name (decision D-01).
pub const METRIC_NAME: &str = "TDL";

/// Maximum exact IEEE-754 integer accepted by conformance admission.
pub const MAX_SAFE_INTEGER: i64 = (1i64 << 53) - 1;

/// Default float parity tolerance between Rust and Python golden outputs.
pub const GOLDEN_TOLERANCE: f64 = 1e-12;

/// Source-free metric evaluation error carrying a stable reason code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricError {
    pub reason: &'static str,
    pub status: &'static str,
}

impl MetricError {
    pub const fn new(reason: &'static str) -> Self {
        Self {
            reason,
            status: "failed",
        }
    }

    pub const fn with_status(reason: &'static str, status: &'static str) -> Self {
        Self { reason, status }
    }
}

pub type MetricResult<T> = std::result::Result<T, MetricError>;

pub fn require(condition: bool, reason: &'static str) -> MetricResult<()> {
    if condition {
        Ok(())
    } else {
        Err(MetricError::new(reason))
    }
}

pub fn failure(reason: &str) -> Value {
    failure_with_status(reason, "failed")
}

pub fn failure_with_status(reason: &str, status: &str) -> Value {
    json!({
        "status": status,
        "reason": reason,
    })
}

pub fn number(value: &Value) -> MetricResult<f64> {
    match value {
        Value::Number(n) => {
            if !n.is_f64() {
                let i = n
                    .as_i64()
                    .ok_or(MetricError::new("nonfinite_or_invalid_number"))?;
                require(
                    (-MAX_SAFE_INTEGER..=MAX_SAFE_INTEGER).contains(&i),
                    "nonfinite_or_invalid_number",
                )?;
            }
            let v = n
                .as_f64()
                .ok_or(MetricError::new("nonfinite_or_invalid_number"))?;
            require(v.is_finite(), "nonfinite_or_invalid_number")?;
            Ok(v)
        }
        _ => Err(MetricError::new("nonfinite_or_invalid_number")),
    }
}

pub fn count(value: &Value, minimum: u64) -> MetricResult<u64> {
    match value {
        Value::Number(n) => {
            require(!n.is_f64(), "invalid_count")?;
            let i = n.as_i64().ok_or(MetricError::new("invalid_count"))?;
            require(
                i >= 0 && (i as u64) >= minimum && i <= MAX_SAFE_INTEGER,
                "invalid_count",
            )?;
            Ok(i as u64)
        }
        _ => Err(MetricError::new("invalid_count")),
    }
}

pub fn unit(value: &Value) -> MetricResult<f64> {
    let v = number(value)?;
    require((0.0..=1.0).contains(&v), "invalid_probability")?;
    Ok(v)
}

pub fn vector(value: &Value, nonnegative: bool) -> MetricResult<Vec<f64>> {
    let arr = value.as_array().ok_or(MetricError::new("invalid_vector"))?;
    require(!arr.is_empty(), "invalid_vector")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let v = number(item)?;
        out.push(v);
    }
    require(
        !nonnegative || out.iter().all(|v| *v >= 0.0),
        "negative_value",
    )?;
    Ok(out)
}

pub fn mean(values: &[f64]) -> MetricResult<f64> {
    require(!values.is_empty(), "empty_mean")?;
    let len = values.len() as f64;
    let mut total = 0.0_f64;
    for &v in values {
        require(v.is_finite(), "nonfinite_or_invalid_number")?;
        let step = v / len;
        require(step.is_finite(), "nonfinite_or_invalid_number")?;
        total += step;
        require(total.is_finite(), "nonfinite_or_invalid_number")?;
    }
    Ok(total)
}

pub fn sum_checked(values: &[f64]) -> MetricResult<f64> {
    let mut total = 0.0_f64;
    for &v in values {
        require(v.is_finite(), "nonfinite_or_invalid_number")?;
        total += v;
        require(total.is_finite(), "nonfinite_or_invalid_number")?;
    }
    Ok(total)
}

/// Strictly parse JSON rejecting duplicate keys and non-finite numbers.
pub fn strict_parse_json(text: &str) -> MetricResult<Value> {
    struct StrictValue;

    impl<'de> DeserializeSeed<'de> for StrictValue {
        type Value = Value;

        fn deserialize<D>(self, deserializer: D) -> std::result::Result<Value, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = Value;

                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("strict JSON value")
                }

                fn visit_bool<E>(self, v: bool) -> std::result::Result<Value, E> {
                    Ok(Value::Bool(v))
                }

                fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Value, E> {
                    Ok(Value::Number(v.into()))
                }

                fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Value, E> {
                    Ok(Value::Number(v.into()))
                }

                fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Value, E> {
                    Number::from_f64(v)
                        .map(Value::Number)
                        .ok_or_else(|| E::custom("nonstandard_json_number"))
                }

                fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Value, E> {
                    Ok(Value::String(v.to_owned()))
                }

                fn visit_string<E>(self, v: String) -> std::result::Result<Value, E> {
                    Ok(Value::String(v))
                }

                fn visit_unit<E>(self) -> std::result::Result<Value, E> {
                    Ok(Value::Null)
                }

                fn visit_seq<A: SeqAccess<'de>>(
                    self,
                    mut seq: A,
                ) -> std::result::Result<Value, A::Error> {
                    let mut out = Vec::new();
                    while let Some(v) = seq.next_element_seed(StrictValue)? {
                        out.push(v);
                    }
                    Ok(Value::Array(out))
                }

                fn visit_map<A: MapAccess<'de>>(
                    self,
                    mut map: A,
                ) -> std::result::Result<Value, A::Error> {
                    let mut out = Map::new();
                    while let Some(k) = map.next_key::<String>()? {
                        let v = map.next_value_seed(StrictValue)?;
                        if out.insert(k, v).is_some() {
                            return Err(de::Error::custom("duplicate_json_key"));
                        }
                    }
                    Ok(Value::Object(out))
                }
            }
            deserializer.deserialize_any(V)
        }
    }

    let bytes = text.as_bytes();
    let mut i = 0;
    let mut in_string = false;
    let mut escaped = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_string = true;
            i += 1;
            continue;
        }
        if b == b'-' || b.is_ascii_digit() {
            let start = i;
            i += 1;
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || matches!(bytes[i], b'.' | b'e' | b'E' | b'+' | b'-'))
            {
                i += 1;
            }
            let tok = &text[start..i];
            let has_digit = tok.bytes().any(|c| c.is_ascii_digit());
            let is_int = has_digit && !tok.bytes().any(|c| matches!(c, b'.' | b'e' | b'E' | b'+'));
            if is_int {
                let ok = if tok.starts_with('-') {
                    tok.parse::<i64>().is_ok()
                } else {
                    tok.parse::<u64>().is_ok()
                };
                if !ok {
                    return Err(MetricError::new("invalid_json"));
                }
            }
            continue;
        }
        i += 1;
    }

    let mut de = serde_json::Deserializer::from_str(text);
    let value = StrictValue.deserialize(&mut de).map_err(|err| {
        let msg = err.to_string();
        if msg.contains("duplicate_json_key") {
            MetricError::new("duplicate_json_key")
        } else {
            MetricError::new("nonstandard_json_number")
        }
    })?;
    de.end()
        .map_err(|_| MetricError::new("nonstandard_json_number"))?;
    Ok(value)
}

/// Compare two JSON values with exact shape/type/integer/bool/null equality and
/// finite float tolerance (`<= tolerance`).
pub fn values_close(actual: &Value, expected: &Value, tolerance: f64) -> bool {
    match (actual, expected) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(x, y)| values_close(x, y, tolerance))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.keys().all(|k| {
                    b.get(k)
                        .is_some_and(|bv| values_close(&a[k], bv, tolerance))
                })
        }
        (Value::Number(a), Value::Number(b)) => {
            if !b.is_f64() {
                !a.is_f64() && a.as_i64() == b.as_i64() && a.as_u64() == b.as_u64()
            } else {
                match (a.as_f64(), b.as_f64()) {
                    (Some(af), Some(bf)) => {
                        af.is_finite() && bf.is_finite() && (af - bf).abs() <= tolerance
                    }
                    _ => false,
                }
            }
        }
        _ => false,
    }
}

/// Recursively collect all string values starting with `"TBD"` inside a JSON value.
pub fn collect_tbd_fields(value: &Value, prefix: &str, out: &mut BTreeMap<String, String>) {
    match value {
        Value::String(s) if s.starts_with("TBD") => {
            out.insert(prefix.to_owned(), s.clone());
        }
        Value::Array(items) => {
            for (idx, item) in items.iter().enumerate() {
                let path = format!("{prefix}[{idx}]");
                collect_tbd_fields(item, &path, out);
            }
        }
        Value::Object(map) => {
            for (k, v) in map {
                let path = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                collect_tbd_fields(v, &path, out);
            }
        }
        _ => {}
    }
}

/// Validate a `tdl-contract-v1.json` document, refusing `protocol_frozen == true`
/// whenever any field is `null` or `"TBD*"` (T-020).
pub fn validate_tdl_contract(contract: &Value) -> MetricResult<()> {
    let obj = contract
        .as_object()
        .ok_or(MetricError::new("invalid_tdl_contract"))?;
    let required = [
        "format",
        "contract_id",
        "version",
        "metric_name",
        "status",
        "protocol_frozen",
        "headline_operating_point",
        "operating_points",
        "secondary_fixed_budgets_kib",
        "byte_grid_kib",
        "byte_grid_bytes",
        "byte_scopes",
        "transfer_protocol",
        "aggregation_policy",
        "driver_protocol",
        "learner_families",
        "sample_size_policy",
        "seed_policy",
        "multiplicity_policy",
        "evaluator_isolation",
        "result_states",
        "pending_decisions",
        "historical_contracts",
    ];
    for key in required {
        require(
            obj.get(key).is_some_and(|v| !v.is_null()),
            "missing_required_contract_field",
        )?;
    }
    require(
        obj["format"] == "dope-tdl-contract"
            && obj["contract_id"] == "tdl-contract-v1"
            && obj["version"] == 1,
        "invalid_tdl_contract_header",
    )?;
    let frozen = obj["protocol_frozen"]
        .as_bool()
        .ok_or(MetricError::new("invalid_protocol_frozen"))?;
    let mut tbds = BTreeMap::new();
    collect_tbd_fields(contract, "", &mut tbds);
    if frozen {
        require(obj["status"] == "frozen", "frozen_contract_status_mismatch")?;
        require(tbds.is_empty(), "cannot_freeze_contract_with_tbd")?;
    } else {
        require(obj["status"] == "draft", "draft_contract_status_mismatch")?;
    }
    let grid_kib = vector(&obj["byte_grid_kib"], true)?;
    let grid_bytes = obj["byte_grid_bytes"]
        .as_array()
        .ok_or(MetricError::new("invalid_byte_grid"))?;
    require(
        grid_kib.len() == 10 && grid_bytes.len() == 10,
        "invalid_byte_grid",
    )?;
    for (kib, bytes_val) in grid_kib.iter().zip(grid_bytes.iter()) {
        let b = count(bytes_val, 512)?;
        require(
            ((*kib * 1024.0) - (b as f64)).abs() <= 1e-9,
            "byte_grid_kib_bytes_mismatch",
        )?;
    }
    let required_families: BTreeSet<&str> = obj["learner_families"]["required_families"]
        .as_array()
        .ok_or(MetricError::new("invalid_learner_families"))?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let held_out_families: BTreeSet<&str> = obj["learner_families"]["held_out_families"]
        .as_array()
        .ok_or(MetricError::new("invalid_learner_families"))?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    require(
        !required_families.is_empty()
            && !held_out_families.is_empty()
            && required_families.is_disjoint(&held_out_families),
        "learner_family_overlap",
    )?;
    Ok(())
}

/// C00 admission only; candidate scoring/calibration is implemented in later packets.
pub mod candidate {
    use super::*;
    use sha2::{Digest, Sha256};

    const PROTOCOL_SCHEMA: &str = include_str!("../../schemas/tdl-protocol-v1.schema.json");
    const RESULT_SCHEMA: &str = include_str!("../../schemas/tdl-result-v1.schema.json");

    /// Domain-separated identities use sorted UTF-8 JSON and safe integers only.
    /// Rational values are explicit numerator/denominator objects, never floats.
    pub fn identity(kind: &str, payload: &Value) -> MetricResult<String> {
        require(
            [
                "protocol",
                "task",
                "artifact",
                "method",
                "panel",
                "campaign",
                "release_profile",
            ]
            .contains(&kind),
            "candidate_identity_kind",
        )?;
        fn canonical(value: &Value) -> MetricResult<Value> {
            Ok(match value {
                Value::Number(n) => {
                    require(!n.is_f64(), "candidate_noncanonical_number")?;
                    number(value)?;
                    value.clone()
                }
                Value::Array(items) => {
                    Value::Array(items.iter().map(canonical).collect::<MetricResult<_>>()?)
                }
                Value::Object(items) => {
                    let mut out = Map::new();
                    for key in items.keys().collect::<BTreeSet<_>>() {
                        out.insert(key.clone(), canonical(&items[key])?);
                    }
                    Value::Object(out)
                }
                _ => value.clone(),
            })
        }
        let bytes = serde_json::to_vec(&canonical(payload)?)
            .map_err(|_| MetricError::new("candidate_canonical_encoding"))?;
        let mut digest = Sha256::new();
        digest.update(b"dope.identity.v1\0");
        digest.update(kind.as_bytes());
        digest.update([0]);
        digest.update(bytes);
        Ok(format!("{kind}:{:x}", digest.finalize()))
    }

    pub fn protocol_id(protocol: &Value) -> MetricResult<String> {
        let mut body = protocol
            .as_object()
            .ok_or(MetricError::new("candidate_schema_mismatch"))?
            .clone();
        body.remove("protocol_id");
        identity("protocol", &Value::Object(body))
    }

    fn hexadecimal(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    }

    // Closed, local schema vocabulary. Unknown keywords/patterns fail closed;
    // this is not a general-purpose or externally extensible JSON Schema engine.
    fn schema_matches(value: &Value, schema: &Value, root: &Value) -> bool {
        let Some(keywords) = schema.as_object() else {
            return false;
        };
        const SUPPORTED: &[&str] = &[
            "$schema",
            "$id",
            "title",
            "$defs",
            "$ref",
            "type",
            "const",
            "enum",
            "anyOf",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "minItems",
            "uniqueItems",
            "minimum",
            "maximum",
            "pattern",
        ];
        if keywords
            .keys()
            .any(|key| !SUPPORTED.contains(&key.as_str()))
        {
            return false;
        }
        if let Some(reference) = schema["$ref"].as_str() {
            return reference
                .strip_prefix('#')
                .and_then(|p| root.pointer(p))
                .is_some_and(|target| schema_matches(value, target, root));
        }
        if let Some(options) = schema["anyOf"].as_array()
            && !options
                .iter()
                .any(|option| schema_matches(value, option, root))
        {
            return false;
        }
        if let Some(expected) = keywords.get("const")
            && value != expected
        {
            return false;
        }
        if let Some(options) = schema["enum"].as_array()
            && !options.contains(value)
        {
            return false;
        }
        if let Some(kind) = schema["type"].as_str() {
            let valid = match kind {
                "object" => value.is_object(),
                "array" => value.is_array(),
                "string" => value.is_string(),
                "null" => value.is_null(),
                "boolean" => value.is_boolean(),
                "integer" => value.as_number().is_some_and(|n| !n.is_f64()),
                _ => false,
            };
            if !valid {
                return false;
            }
        }
        if let Some(properties) = schema["properties"].as_object() {
            let Some(object) = value.as_object() else {
                return false;
            };
            if schema["required"].as_array().is_none_or(|keys| {
                keys.iter()
                    .any(|key| key.as_str().is_none_or(|k| !object.contains_key(k)))
            }) {
                return false;
            }
            if schema["additionalProperties"] != false
                || object.keys().any(|key| !properties.contains_key(key))
            {
                return false;
            }
            if object
                .iter()
                .any(|(key, item)| !schema_matches(item, &properties[key], root))
            {
                return false;
            }
        }
        if let Some(items) = value.as_array() {
            if let Some(minimum) = schema["minItems"].as_u64()
                && items.len() < minimum as usize
            {
                return false;
            }
            if schema["uniqueItems"] == true
                && items
                    .iter()
                    .enumerate()
                    .any(|(i, v)| items[..i].contains(v))
            {
                return false;
            }
            if let Some(item_schema) = keywords.get("items")
                && items
                    .iter()
                    .any(|item| !schema_matches(item, item_schema, root))
            {
                return false;
            }
        }
        if let Some(n) = value.as_f64() {
            if schema["minimum"]
                .as_f64()
                .is_some_and(|minimum| n < minimum)
                || schema["maximum"]
                    .as_f64()
                    .is_some_and(|maximum| n > maximum)
            {
                return false;
            }
        }
        if let Some(pattern) = schema["pattern"].as_str() {
            let Some(text) = value.as_str() else {
                return false;
            };
            if pattern == "^[0-9a-f]{64}$" {
                return hexadecimal(text);
            }
            return pattern
                .strip_prefix('^')
                .and_then(|p| p.strip_suffix("[0-9a-f]{64}$"))
                .and_then(|prefix| text.strip_prefix(prefix))
                .is_some_and(hexadecimal);
        }
        true
    }

    pub fn validate_protocol(protocol: &Value) -> MetricResult<()> {
        let expected = protocol_id(protocol)?;
        let schema = strict_parse_json(PROTOCOL_SCHEMA)?;
        require(
            schema_matches(protocol, &schema, &schema),
            "candidate_schema_mismatch",
        )?;
        let parent = format!(
            "{:x}",
            Sha256::digest(include_bytes!("../../contracts/tdl-contract-v1.json"))
        );
        require(
            protocol["parent_contract_sha256"] == parent,
            "candidate_parent_digest_mismatch",
        )?;
        require(
            protocol["protocol_id"] == expected,
            "candidate_protocol_digest_mismatch",
        )?;
        if let Some(panel) = protocol["panel"].as_object() {
            let families = panel["required_families"]
                .as_array()
                .ok_or(MetricError::new("candidate_inconsistent_panel"))?;
            let models = panel["eligible_models"]
                .as_array()
                .ok_or(MetricError::new("candidate_inconsistent_panel"))?;
            let mut ids = BTreeSet::new();
            for model in models {
                require(
                    families.contains(&model["family"])
                        && ids.insert(model["id"].as_str().unwrap_or_default()),
                    "candidate_inconsistent_panel",
                )?;
                if let Some(roster) =
                    protocol["roles"][model["role"].as_str().unwrap_or_default()].as_array()
                {
                    require(
                        roster.contains(&model["id"]),
                        "candidate_inconsistent_panel",
                    )?;
                }
            }
            require(
                families
                    .iter()
                    .all(|family| models.iter().any(|model| model["family"] == *family)),
                "candidate_inconsistent_panel",
            )?;
            for role in ["development", "confirmation"] {
                if let Some(roster) = protocol["roles"][role].as_array() {
                    require(
                        roster.iter().all(|id| {
                            models
                                .iter()
                                .any(|model| model["id"] == *id && model["role"] == role)
                        }),
                        "candidate_inconsistent_panel",
                    )?;
                }
            }
        }
        if let (Some(development), Some(confirmation)) = (
            protocol["roles"]["development"].as_array(),
            protocol["roles"]["confirmation"].as_array(),
        ) {
            require(
                !development.iter().any(|id| confirmation.contains(id)),
                "candidate_inconsistent_panel",
            )?;
        }
        Ok(())
    }

    pub fn parse_protocol(text: &str) -> MetricResult<Value> {
        let protocol = strict_parse_json(text)?;
        validate_protocol(&protocol)?;
        Ok(protocol)
    }

    /// A mode fence, not a scorer or qualification result. Legacy replay uses
    /// the existing v1 component-case reader and cannot enter candidate mode.
    #[derive(Debug, PartialEq, Eq)]
    pub enum ReceiptMode {
        CandidateAdmission,
        LegacyReplay,
    }

    pub fn parse_receipt(
        protocol: &Value,
        text: &str,
        legacy_replay: bool,
    ) -> MetricResult<ReceiptMode> {
        let receipt = strict_parse_json(text)?;
        admit_receipt(protocol, &receipt, legacy_replay)
    }

    pub fn admit_receipt(
        protocol: &Value,
        receipt: &Value,
        legacy_replay: bool,
    ) -> MetricResult<ReceiptMode> {
        if receipt["contract_id"] == "tdl-contract-v1" {
            require(legacy_replay, "legacy_receipt_requires_replay")?;
            require(
                receipt
                    .as_object()
                    .is_some_and(|o| o.len() == 2 && o.contains_key("case")),
                "invalid_legacy_replay",
            )?;
            golden::run_case(&receipt["case"])?;
            return Ok(ReceiptMode::LegacyReplay);
        }
        require(!legacy_replay, "candidate_not_legacy_receipt")?;
        validate_protocol(protocol)?;
        require(
            receipt["protocol_id"] == protocol["protocol_id"],
            "candidate_receipt_protocol_mismatch",
        )?;
        identity("artifact", receipt)?; // refuse lossy/noncanonical numbers before schema admission
        let schema = strict_parse_json(RESULT_SCHEMA)?;
        require(
            schema_matches(receipt, &schema, &schema),
            "candidate_schema_mismatch",
        )?;
        let identities = receipt["identities"]
            .as_object()
            .ok_or(MetricError::new("candidate_schema_mismatch"))?;
        for (kind, entity) in identities {
            let mut body = entity
                .as_object()
                .ok_or(MetricError::new("candidate_identity_mismatch"))?
                .clone();
            body.remove("id");
            require(
                entity["id"] == identity(kind, &Value::Object(body))?,
                "candidate_identity_mismatch",
            )?;
        }
        let artifact = &identities["artifact"]["bindings"];
        let campaign = &identities["campaign"]["bindings"];
        require(
            artifact["protocol_id"] == protocol["protocol_id"]
                && campaign["protocol_id"] == protocol["protocol_id"]
                && artifact["task_id"] == identities["task"]["id"]
                && artifact["method_id"] == identities["method"]["id"]
                && campaign["panel_id"] == identities["panel"]["id"],
            "candidate_binding_mismatch",
        )?;
        Ok(ReceiptMode::CandidateAdmission)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate_fixture(name: &str) -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tdl-v2")
            .join(name);
        strict_parse_json(&std::fs::read_to_string(path).expect("candidate fixture"))
            .expect("strict candidate fixture")
    }

    #[test]
    fn candidate_hash_goldens_and_parent_are_bound() {
        let protocol = candidate::parse_protocol(include_str!(
            "../../contracts/tdl-contract-v2.candidate.json"
        ))
        .unwrap();
        for golden in candidate_fixture("canonical-hashes.json")
            .as_array()
            .unwrap()
        {
            assert_eq!(
                candidate::identity(golden["kind"].as_str().unwrap(), &golden["payload"]).unwrap(),
                golden["identity"]
            );
            assert_eq!(
                strict_parse_json(golden["canonical"].as_str().unwrap()).unwrap(),
                golden["payload"]
            );
        }
        assert_eq!(
            candidate::protocol_id(&protocol).unwrap(),
            protocol["protocol_id"]
        );
        let receipt = candidate_fixture("candidate-result.json");
        assert_eq!(
            candidate::admit_receipt(&protocol, &receipt, false).unwrap(),
            candidate::ReceiptMode::CandidateAdmission
        );
    }

    #[test]
    fn candidate_admission_counterexamples_fail_without_echoing_values() {
        let protocol = candidate::parse_protocol(include_str!(
            "../../contracts/tdl-contract-v2.candidate.json"
        ))
        .unwrap();
        let receipt = candidate_fixture("candidate-result.json");
        for case in candidate_fixture("admission-cases.json")
            .as_array()
            .unwrap()
        {
            let is_protocol = case["target"] == "protocol";
            let mut value = if is_protocol {
                protocol.clone()
            } else {
                receipt.clone()
            };
            if let Some(panel) = case.get("setup_panel") {
                value["panel"] = panel.clone();
            }
            let path = case["path"].as_str().unwrap();
            if case["remove"] == true {
                let (parent, key) = path.rsplit_once('/').unwrap();
                value
                    .pointer_mut(parent)
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .remove(key);
            } else {
                *value.pointer_mut(path).unwrap() = case["value"].clone();
            }
            if case["resign_identity"] == true {
                let entity = &mut value["identities"]["artifact"];
                let mut body = entity.as_object().unwrap().clone();
                body.remove("id");
                entity["id"] =
                    json!(candidate::identity("artifact", &Value::Object(body)).unwrap());
            }
            let outcome = if is_protocol {
                if case["resign"] == true {
                    match candidate::protocol_id(&value) {
                        Ok(digest) => value["protocol_id"] = json!(digest),
                        Err(error) => {
                            assert_eq!(error.reason, case["reason"].as_str().unwrap());
                            continue;
                        }
                    }
                }
                candidate::validate_protocol(&value)
            } else {
                candidate::admit_receipt(&protocol, &value, false).map(|_| ())
            };
            let error = outcome.expect_err("candidate counterexample must fail");
            assert_eq!(error.reason, case["reason"].as_str().unwrap());
            assert!(!format!("{error:?}").contains("CANARY_PRIVATE"));
        }
    }

    #[test]
    fn candidate_strict_ingress_and_explicit_legacy_replay_fence() {
        let protocol = candidate::parse_protocol(include_str!(
            "../../contracts/tdl-contract-v2.candidate.json"
        ))
        .unwrap();
        let legacy = candidate_fixture("legacy-replay.json");
        for case in candidate_fixture("invalid-ingress.json")
            .as_array()
            .unwrap()
        {
            let text = case["text"].as_str().unwrap();
            for error in [
                candidate::parse_protocol(text).unwrap_err(),
                candidate::parse_receipt(&protocol, text, false).unwrap_err(),
            ] {
                assert_eq!(error.reason, case["reason"].as_str().unwrap());
                assert!(!format!("{error:?}").contains("CANARY_PRIVATE"));
            }
        }
        assert_eq!(
            candidate::admit_receipt(&protocol, &legacy, false),
            Err(MetricError::new("legacy_receipt_requires_replay"))
        );
        assert_eq!(
            candidate::admit_receipt(&protocol, &legacy, true).unwrap(),
            candidate::ReceiptMode::LegacyReplay
        );
        assert_eq!(
            golden::run_case(&legacy["case"]).unwrap(),
            legacy["case"]["expected"]
        );
        let receipt = candidate_fixture("candidate-result.json");
        assert_eq!(
            candidate::admit_receipt(&protocol, &receipt, true),
            Err(MetricError::new("candidate_not_legacy_receipt"))
        );
    }

    #[test]
    fn tdl_contract_v1_validates_and_rejects_freeze_while_tbd_remains() {
        let raw = include_str!("../../contracts/tdl-contract-v1.json");
        let contract = strict_parse_json(raw).expect("tdl-contract-v1.json must be strict JSON");
        validate_tdl_contract(&contract).expect("draft tdl-contract-v1.json must validate");
        assert_eq!(contract["metric_name"], METRIC_NAME);
        assert_eq!(contract["protocol_frozen"], false);
        assert_eq!(contract["headline_operating_point"], "T90/D80");

        let mut tbds = BTreeMap::new();
        collect_tbd_fields(&contract, "", &mut tbds);
        assert!(
            !tbds.is_empty(),
            "draft contract must record pending TBD decisions"
        );

        let mut prematurely_frozen = contract.clone();
        prematurely_frozen["protocol_frozen"] = json!(true);
        prematurely_frozen["status"] = json!("frozen");
        assert_eq!(
            validate_tdl_contract(&prematurely_frozen),
            Err(MetricError::new("cannot_freeze_contract_with_tbd"))
        );
    }

    #[test]
    fn tdl_report_aggregate_is_recomputable_from_auditor_reports() {
        let rows = json!([
            {"source_family": "s1", "dataset": "d1", "seed": 101, "family": "tree", "auditor": "catboost", "transfer": 0.94},
            {"source_family": "s1", "dataset": "d1", "seed": 101, "family": "linear", "auditor": "elastic_net", "transfer": 0.92},
            {"source_family": "s1", "dataset": "d1", "seed": 211, "family": "tree", "auditor": "catboost", "transfer": 0.93},
            {"source_family": "s1", "dataset": "d1", "seed": 211, "family": "linear", "auditor": "elastic_net", "transfer": 0.91}
        ]);
        let agg = aggregate::evaluate(&json!({
            "required_families": ["tree", "linear"],
            "required_auditors": {"tree": ["catboost"], "linear": ["elastic_net"]},
            "required_seeds": [101, 211],
            "rows": rows,
            "bootstrap_replicates": 200,
            "bootstrap_seed": 1729,
            "alpha": 0.05
        }));
        assert_eq!(agg["status"], "complete");
        assert!((agg["robust_transfer"].as_f64().unwrap() - 0.915).abs() <= 1e-12);
    }
}

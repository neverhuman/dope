//! C00 admission only; candidate scoring/calibration is implemented in later packets.
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

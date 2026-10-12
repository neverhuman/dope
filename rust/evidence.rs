//! New vNext evidence boundary. Historical production/ledger receipts are unchanged.
//! A verified receipt binds bytes to an independently trusted executor policy;
//! it does not grant scientific truth, isolation, privacy or release authority.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use rustix::fs::{Mode, OFlags, open, openat};
use serde::Deserializer;
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value, json};
use sha2::{Digest, Sha256};

use crate::error::{DopeError, Result};

const MAX_INTEGER: u64 = (1 << 53) - 1;
const MAX_MANIFEST: usize = 16 * 1024 * 1024;
const RUN_SCHEMA: &str = include_str!("../schemas/transfer-run-v1.schema.json");
const BEHAVIOR_SCHEMA: &str = include_str!("../schemas/behavior-contract-v1.schema.json");
const FREEZE_SCHEMA: &str = include_str!("../schemas/freeze-v1.schema.json");
const FIT_FIELDS: [&str; 8] = [
    "data_hash",
    "split_hash",
    "generator_hash",
    "configuration_hash",
    "feature_transform_hash",
    "target_transform_hash",
    "generator_fit_seed",
    "global_seed",
];

fn error(code: &'static str) -> DopeError {
    DopeError::Data(code.into())
}
fn require(condition: bool, code: &'static str) -> Result<()> {
    if condition { Ok(()) } else { Err(error(code)) }
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn is_digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

// serde_json::Value alone silently accepts duplicate object keys.
struct StrictValue(Value);
struct StrictSeed<'a>(&'a mut std::vec::IntoIter<Number>);
impl<'de> DeserializeSeed<'de> for StrictSeed<'_> {
    type Value = StrictValue;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Self::Value, D::Error> {
        struct StrictVisitor<'a>(&'a mut std::vec::IntoIter<Number>);
        impl<'de> Visitor<'de> for StrictVisitor<'_> {
            type Value = StrictValue;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("finite JSON")
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Self::Value, E> {
                Ok(StrictValue(v.into()))
            }
            fn visit_u64<E: de::Error>(
                self,
                _placeholder: u64,
            ) -> std::result::Result<Self::Value, E> {
                self.0
                    .next()
                    .map(|n| StrictValue(Value::Number(n)))
                    .ok_or_else(|| E::custom("invalid_number_stream"))
            }
            fn visit_i64<E: de::Error>(
                self,
                placeholder: i64,
            ) -> std::result::Result<Self::Value, E> {
                self.visit_u64(placeholder as u64)
            }
            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Self::Value, E> {
                Ok(StrictValue(v.into()))
            }
            fn visit_none<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(StrictValue(Value::Null))
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(StrictValue(Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(v) = a.next_element_seed(StrictSeed(&mut *self.0))? {
                    values.push(v.0);
                }
                Ok(StrictValue(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = Map::new();
                while let Some(k) = a.next_key::<String>()? {
                    if values.contains_key(&k) {
                        return Err(de::Error::custom("duplicate_json_key"));
                    }
                    values.insert(k, a.next_value_seed(StrictSeed(&mut *self.0))?.0);
                }
                Ok(StrictValue(Value::Object(values)))
            }
        }
        d.deserialize_any(StrictVisitor(self.0))
    }
}

fn number_token(token: &str) -> Result<Number> {
    // JSON number grammar; replacing tokens below must not admit e.g. 01 or 1.
    let bytes = token.as_bytes();
    let mut i = usize::from(bytes.first() == Some(&b'-'));
    if bytes.get(i) == Some(&b'0') {
        i += 1;
    } else {
        require(
            bytes.get(i).is_some_and(|b| (b'1'..=b'9').contains(b)),
            "invalid_finite_json",
        )?;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
    }
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        let start = i;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        require(i > start, "invalid_finite_json")?;
    }
    if bytes.get(i).is_some_and(|b| matches!(b, b'e' | b'E')) {
        i += 1;
        if bytes.get(i).is_some_and(|b| matches!(b, b'+' | b'-')) {
            i += 1;
        }
        let start = i;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        require(i > start, "invalid_finite_json")?;
    }
    require(i == bytes.len(), "invalid_finite_json")?;
    if token.contains(['.', 'e', 'E']) {
        Number::from_f64(
            token
                .parse::<f64>()
                .map_err(|_| error("invalid_finite_json"))?,
        )
        .ok_or_else(|| error("nonfinite_json_number"))
    } else {
        let n = token
            .parse::<i64>()
            .map_err(|_| error("integer_out_of_range"))?;
        require(n.unsigned_abs() <= MAX_INTEGER, "integer_out_of_range")?;
        Ok(n.into())
    }
}

pub fn parse_json(data: &[u8]) -> Result<Value> {
    // Parse numeric tokens with std's correctly rounded f64 parser, and let
    // serde handle JSON structure, escaping and duplicate-key detection. The
    // number stream restores original scalar types in encounter order. This
    // confines numeric behavior to vNext without changing historical parsers.
    let mut offset = 0;
    let mut quoted = false;
    let mut copied = 0;
    let mut structure = Vec::with_capacity(data.len());
    let mut numbers = Vec::new();
    while offset < data.len() {
        match data[offset] {
            b'\\' if quoted => offset += 2,
            b'"' => {
                quoted = !quoted;
                offset += 1;
            }
            b'-' | b'0'..=b'9' if !quoted => {
                let start = offset;
                offset += 1;
                while offset < data.len()
                    && matches!(data[offset], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
                {
                    offset += 1;
                }
                let token = std::str::from_utf8(&data[start..offset])
                    .map_err(|_| error("invalid_finite_json"))?;
                numbers.push(number_token(token)?);
                structure.extend_from_slice(&data[copied..start]);
                structure.push(b'0');
                copied = offset;
            }
            _ => offset += 1,
        }
    }
    structure.extend_from_slice(&data[copied..]);
    let mut numbers = numbers.into_iter();
    let mut deserializer = serde_json::Deserializer::from_slice(&structure);
    let value = StrictSeed(&mut numbers)
        .deserialize(&mut deserializer)
        .map_err(|_| error("invalid_finite_json"))?
        .0;
    deserializer
        .end()
        .map_err(|_| error("invalid_finite_json"))?;
    require(numbers.next().is_none(), "invalid_number_stream")?;
    Ok(value)
}

/// dope-finite-json-v1, not RFC 8785. Float/int representations remain distinct.
pub fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    fn encode(value: &Value, out: &mut String) -> Result<()> {
        match value {
            Value::Object(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().collect::<BTreeMap<_, _>>().iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(
                        &serde_json::to_string(k).map_err(|_| error("invalid_json_encoding"))?,
                    );
                    out.push(':');
                    encode(v, out)?;
                }
                out.push('}');
            }
            Value::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    encode(v, out)?;
                }
                out.push(']');
            }
            Value::Number(n) if n.is_f64() => {
                let v = n.as_f64().ok_or_else(|| error("nonfinite_json_number"))?;
                require(v.is_finite(), "nonfinite_json_number")?;
                out.push_str(&format!("{v:.17e}"));
            }
            Value::Number(n) => {
                let magnitude = n
                    .as_i64()
                    .map(i64::unsigned_abs)
                    .or_else(|| n.as_u64())
                    .ok_or_else(|| error("integer_out_of_range"))?;
                require(magnitude <= MAX_INTEGER, "integer_out_of_range")?;
                out.push_str(&n.to_string());
            }
            _ => out.push_str(
                &serde_json::to_string(value).map_err(|_| error("invalid_json_encoding"))?,
            ),
        }
        Ok(())
    }
    let mut out = String::new();
    encode(value, &mut out)?;
    Ok(out.into_bytes())
}
fn digest(value: &Value) -> Result<String> {
    Ok(hash(&canonical_json(value)?))
}

// Closed Draft 2020-12 subset used by the three tracked schemas. Both runtimes
// validate against these sources. Unsupported shape changes need verifier changes.
fn validate(value: &Value, schema: &Value) -> Result<()> {
    const KNOWN: &[&str] = &[
        "$schema",
        "$id",
        "$defs",
        "title",
        "type",
        "anyOf",
        "allOf",
        "if",
        "then",
        "else",
        "const",
        "enum",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "minItems",
        "uniqueItems",
        "minLength",
        "pattern",
        "minimum",
        "maximum",
    ];
    if let Some(m) = schema.as_object() {
        require(
            m.keys().all(|k| KNOWN.contains(&k.as_str())),
            "unsupported_schema_keyword",
        )?;
    }
    if let Some(alternatives) = schema["anyOf"].as_array() {
        require(
            alternatives.iter().any(|s| validate(value, s).is_ok()),
            "schema_mismatch",
        )?;
    }
    if let Some(t) = schema["type"].as_str() {
        let valid = match t {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "integer" => value.as_number().is_some_and(|n| !n.is_f64()),
            "number" => value.is_number(),
            "boolean" => value.is_boolean(),
            "null" => value.is_null(),
            _ => false,
        };
        require(valid, "schema_mismatch")?;
    }
    if let Some(c) = schema.get("const") {
        require(
            canonical_json(value)? == canonical_json(c)?,
            "schema_mismatch",
        )?;
    }
    if let Some(a) = schema["enum"].as_array() {
        require(a.iter().any(|v| v == value), "schema_mismatch")?;
    }
    if let Some(m) = value.as_object() {
        if let Some(required) = schema["required"].as_array() {
            require(
                required
                    .iter()
                    .all(|k| k.as_str().is_some_and(|k| m.contains_key(k))),
                "schema_mismatch",
            )?;
        }
        let properties = schema["properties"].as_object();
        if schema["additionalProperties"] == false {
            require(
                m.keys()
                    .all(|k| properties.is_some_and(|p| p.contains_key(k))),
                "schema_mismatch",
            )?;
        }
        if let Some(properties) = properties {
            for (k, s) in properties {
                if let Some(v) = m.get(k) {
                    validate(v, s)?;
                }
            }
        }
    }
    if let Some(a) = value.as_array() {
        require(
            a.len() as u64 >= schema["minItems"].as_u64().unwrap_or(0),
            "schema_mismatch",
        )?;
        if schema["uniqueItems"] == true {
            let mut unique = BTreeSet::new();
            for v in a {
                require(unique.insert(canonical_json(v)?), "schema_mismatch")?;
            }
        }
        if let Some(s) = schema.get("items") {
            for v in a {
                validate(v, s)?;
            }
        }
    }
    if let Some(s) = value.as_str() {
        require(
            s.chars().count() as u64 >= schema["minLength"].as_u64().unwrap_or(0),
            "schema_mismatch",
        )?;
        if schema.get("pattern").is_some() {
            require(
                schema["pattern"] == "^[0-9a-f]{64}$" && is_digest(s),
                "schema_mismatch",
            )?;
        }
    }
    if let Some(n) = value.as_f64() {
        require(
            n >= schema["minimum"].as_f64().unwrap_or(f64::NEG_INFINITY)
                && n <= schema["maximum"].as_f64().unwrap_or(f64::INFINITY),
            "schema_mismatch",
        )?;
    }
    if let Some(a) = schema["allOf"].as_array() {
        for s in a {
            validate(value, s)?;
        }
    }
    if let Some(condition) = schema.get("if") {
        let branch = if validate(value, condition).is_ok() {
            "then"
        } else {
            "else"
        };
        if let Some(s) = schema.get(branch) {
            validate(value, s)?;
        }
    }
    Ok(())
}
fn schema(text: &str) -> Result<Value> {
    parse_json(text.as_bytes())
}
pub fn validate_behavior_contract(value: &Value) -> Result<()> {
    canonical_json(value)?;
    validate(value, &schema(BEHAVIOR_SCHEMA)?)?;
    let required = value["learners"]["required_families"]
        .as_array()
        .ok_or_else(|| error("schema_mismatch"))?;
    let held_out = value["learners"]["held_out_families"]
        .as_array()
        .ok_or_else(|| error("schema_mismatch"))?;
    require(
        !required.iter().any(|v| held_out.contains(v)),
        "learner_holdout_overlap",
    )
}

#[derive(Clone, Debug)]
pub struct EvaluationIdentity(Value);
impl EvaluationIdentity {
    pub fn new(value: Value) -> Result<Self> {
        canonical_json(&value)?;
        validate(&value, &schema(RUN_SCHEMA)?["$defs"]["EvaluationIdentity"])?;
        Ok(Self(value))
    }
    pub fn value(&self) -> &Value {
        &self.0
    }
}
pub fn scientific_cache_key(identity: &EvaluationIdentity) -> Result<String> {
    digest(&json!({"format":"dope-scientific-cache-key", "version":1, "identity":identity.0}))
}
pub fn fit_replicate_key(identity: &EvaluationIdentity) -> Result<String> {
    let fields: Map<String, Value> = FIT_FIELDS
        .iter()
        .map(|k| (k.to_string(), identity.0[k].clone()))
        .collect();
    digest(&json!({"format":"dope-fit-replicate", "version":1, "identity":fields}))
}

/// Must be provided independently by a trusted executor/custodian. Copying these
/// pins from an untrusted receipt establishes no executor authority.
pub struct VerificationPolicy {
    pub identity: EvaluationIdentity,
    pub executor_hash: String,
    pub receipt_hash: String,
    pub freeze_hash: Option<String>,
}
#[derive(Debug)]
pub struct UnverifiedReceipt(Value);
impl UnverifiedReceipt {
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Ok(Self(parse_json(data)?))
    }
}
#[derive(Debug)]
pub struct VerifiedEvaluationReceipt {
    receipt: Value,
    identity: EvaluationIdentity,
    receipt_hash: String,
}
impl VerifiedEvaluationReceipt {
    pub fn value(&self) -> &Value {
        &self.receipt
    }
    pub fn identity(&self) -> &EvaluationIdentity {
        &self.identity
    }
    pub fn receipt_hash(&self) -> &str {
        &self.receipt_hash
    }
}

fn text(v: &Value) -> Result<&str> {
    v.as_str().ok_or_else(|| error("schema_mismatch"))
}
fn uint(v: &Value) -> Result<u64> {
    v.as_u64().ok_or_else(|| error("schema_mismatch"))
}
fn closed(v: &Value, fields: &[&str]) -> Result<()> {
    require(
        v.as_object()
            .is_some_and(|m| m.len() == fields.len() && fields.iter().all(|k| m.contains_key(*k))),
        "invalid_dependency_manifest",
    )
}
fn version(v: &Value, format: &str) -> bool {
    v["format"] == format && v["version"].as_u64() == Some(1)
}

// Open relative to directory descriptors with NOFOLLOW for every component.
// The root directory is supplied by the trusted verifier, not by the receipt.
fn read_file(root: &Path, path: &str, manifest: bool) -> Result<(String, u64, Vec<u8>)> {
    require(
        !path.contains(['\\', '\0']) && path.split('/').all(|p| !matches!(p, "" | "." | "..")),
        "unsafe_dependency_path",
    )?;
    let directory_flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let mut directory =
        open(root, directory_flags, Mode::empty()).map_err(|_| error("dependency_io_failed"))?;
    let mut pieces = path.split('/').peekable();
    let mut file = loop {
        let part = pieces
            .next()
            .ok_or_else(|| error("unsafe_dependency_path"))?;
        if pieces.peek().is_none() {
            let fd = openat(
                &directory,
                part,
                OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
                Mode::empty(),
            )
            .map_err(|_| error("dependency_io_failed"))?;
            break File::from(fd);
        }
        directory = openat(&directory, part, directory_flags, Mode::empty())
            .map_err(|_| error("dependency_io_failed"))?;
    };
    require(
        file.metadata()
            .map_err(|_| error("dependency_io_failed"))?
            .is_file(),
        "dependency_not_regular_file",
    )?;
    let mut hasher = Sha256::new();
    let mut size = 0_u64;
    let mut data = Vec::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|_| error("dependency_io_failed"))?;
        if n == 0 {
            break;
        }
        size = size
            .checked_add(n as u64)
            .ok_or_else(|| error("dependency_too_large"))?;
        hasher.update(&buffer[..n]);
        if manifest {
            require(size <= MAX_MANIFEST as u64, "manifest_too_large")?;
            data.extend_from_slice(&buffer[..n]);
        }
    }
    Ok((format!("{:x}", hasher.finalize()), size, data))
}
fn visit<'a>(
    key: &'a str,
    nodes: &BTreeMap<&'a str, &'a Value>,
    visited: &mut BTreeSet<&'a str>,
    active: &mut BTreeSet<&'a str>,
) -> Result<()> {
    let node = nodes.get(key).ok_or_else(|| error("dependency_missing"))?;
    require(!active.contains(key), "dependency_cycle")?;
    if visited.contains(key) {
        return Ok(());
    }
    active.insert(key);
    for child in node["requires"]
        .as_array()
        .ok_or_else(|| error("schema_mismatch"))?
    {
        visit(text(child)?, nodes, visited, active)?;
    }
    active.remove(key);
    visited.insert(key);
    Ok(())
}
fn children(node: &Value) -> Result<BTreeSet<&str>> {
    node["requires"]
        .as_array()
        .ok_or_else(|| error("schema_mismatch"))?
        .iter()
        .map(text)
        .collect()
}

pub fn verify_receipt(
    unverified: UnverifiedReceipt,
    root: &Path,
    policy: &VerificationPolicy,
) -> Result<VerifiedEvaluationReceipt> {
    let receipt = unverified.0;
    validate(&receipt, &schema(RUN_SCHEMA)?)?;
    let receipt_hash = digest(&receipt)?;
    require(
        is_digest(&policy.executor_hash)
            && is_digest(&policy.receipt_hash)
            && policy.freeze_hash.as_deref().is_none_or(is_digest),
        "invalid_trust_pin",
    )?;
    require(
        receipt_hash == policy.receipt_hash && receipt["identity"] == policy.identity.0,
        "receipt_identity_mismatch",
    )?;
    require(
        receipt["executor_hash"] == policy.executor_hash
            && receipt["freeze_hash"].as_str() == policy.freeze_hash.as_deref(),
        "executor_or_freeze_mismatch",
    )?;
    require(receipt["execution"] == "complete", "evaluation_incomplete")?;
    require(
        receipt["projection"] == "private",
        "private_evidence_required",
    )?;
    let identity = &receipt["identity"];
    let produced = uint(&identity["produced_rows"])?;
    let consumed = uint(&identity["consumed_rows"])?;
    require(
        produced == uint(&identity["requested_rows"])?
            && consumed > 0
            && consumed <= produced
            && uint(&identity["context_rows"])? <= consumed,
        "row_count_mismatch",
    )?;
    let mut nodes = BTreeMap::new();
    let mut paths = BTreeSet::new();
    for node in receipt["dependencies"]
        .as_array()
        .ok_or_else(|| error("schema_mismatch"))?
    {
        require(
            nodes.insert(text(&node["sha256"])?, node).is_none()
                && paths.insert(text(&node["path"])?),
            "dependency_alias",
        )?;
    }
    let mut roots = BTreeSet::new();
    for (field, value) in identity
        .as_object()
        .ok_or_else(|| error("schema_mismatch"))?
    {
        if field.ends_with("_hash") {
            roots.insert(text(value)?);
        }
    }
    roots.insert(text(&receipt["executor_hash"])?);
    roots.insert(text(&receipt["paired_evidence_hash"])?);
    if let Some(f) = receipt["freeze_hash"].as_str() {
        roots.insert(f);
    }
    let mut visited = BTreeSet::new();
    let mut active = BTreeSet::new();
    for key in roots {
        visit(key, &nodes, &mut visited, &mut active)?;
    }
    require(visited.len() == nodes.len(), "unbound_dependency")?;
    let mut manifests = BTreeMap::new();
    for (key, node) in &nodes {
        let manifest = matches!(
            text(&node["kind"])?,
            "protocol" | "split" | "fit" | "artifact_inventory" | "paired_evidence" | "freeze"
        );
        let (actual, size, data) = read_file(root, text(&node["path"])?, manifest)?;
        require(
            actual == *key && size == uint(&node["bytes"])?,
            "dependency_bytes_mismatch",
        )?;
        if manifest {
            manifests.insert(*key, parse_json(&data)?);
        }
    }
    for (field, key) in identity
        .as_object()
        .ok_or_else(|| error("schema_mismatch"))?
    {
        if let Some(kind) = field.strip_suffix("_hash") {
            require(
                nodes[text(key)?]["kind"] == kind,
                "dependency_kind_mismatch",
            )?;
        }
    }
    let executor = text(&receipt["executor_hash"])?;
    let paired_key = text(&receipt["paired_evidence_hash"])?;
    require(
        nodes[executor]["kind"] == "executor" && nodes[paired_key]["kind"] == "paired_evidence",
        "dependency_kind_mismatch",
    )?;
    let protocol_key = text(&identity["protocol_hash"])?;
    let split_key = text(&identity["split_hash"])?;
    let split = &manifests[split_key];
    closed(
        split,
        &[
            "format",
            "version",
            "data_hash",
            "query_hash",
            "partition_role",
            "fit_rows",
            "query_rows",
        ],
    )?;
    require(
        version(split, "dope-split-manifest")
            && [
                "data_hash",
                "query_hash",
                "partition_role",
                "fit_rows",
                "query_rows",
            ]
            .iter()
            .all(|k| split[k] == identity[k]),
        "split_identity_mismatch",
    )?;
    require(
        children(nodes[split_key])?
            == BTreeSet::from([
                text(&identity["data_hash"])?,
                text(&identity["query_hash"])?,
            ]),
        "split_dependency_mismatch",
    )?;
    validate_behavior_contract(&manifests[protocol_key])?;
    let protocol = &manifests[protocol_key];
    for (declared, actual) in [
        ("table_schema_hash", "schema_hash"),
        ("task_manifest_hash", "task_hash"),
        ("population_history_hash", "source_family_hash"),
    ] {
        require(
            protocol["workload"][declared].is_null()
                || protocol["workload"][declared] == identity[actual],
            "protocol_workload_mismatch",
        )?;
    }
    let mut commitments = BTreeSet::new();
    for section in ["freeze_requirements", "workload", "disclosure"] {
        for (field, value) in protocol[section]
            .as_object()
            .ok_or_else(|| error("schema_mismatch"))?
        {
            if (section == "freeze_requirements" || field.ends_with("_hash")) && !value.is_null() {
                commitments.insert(text(value)?);
            }
        }
    }
    require(
        commitments.is_subset(&children(nodes[protocol_key])?),
        "protocol_commitment_unbound",
    )?;
    let inventory_key = text(&identity["artifact_inventory_hash"])?;
    let inventory = &manifests[inventory_key];
    closed(inventory, &["format", "version", "files"])?;
    require(
        version(inventory, "dope-artifact-inventory"),
        "invalid_artifact_inventory",
    )?;
    let files: Vec<&str> = inventory["files"]
        .as_array()
        .ok_or_else(|| error("invalid_artifact_inventory"))?
        .iter()
        .map(text)
        .collect::<Result<_>>()?;
    let artifacts: BTreeSet<&str> = nodes
        .iter()
        .filter(|(_, n)| n["kind"] == "artifact")
        .map(|(k, _)| *k)
        .collect();
    require(
        !artifacts.is_empty()
            && files.len() == artifacts.len()
            && files.iter().copied().collect::<BTreeSet<_>>() == artifacts
            && children(nodes[inventory_key])? == artifacts,
        "artifact_inventory_mismatch",
    )?;
    let bytes = artifacts.iter().try_fold(0_u64, |s, k| {
        s.checked_add(uint(&nodes[k]["bytes"])?)
            .ok_or_else(|| error("sampling_byte_sum_mismatch"))
    })?;
    require(
        bytes == uint(&receipt["sampling_payload_bytes"])?,
        "sampling_byte_sum_mismatch",
    )?;
    let fit_key = text(&identity["fit_hash"])?;
    let fit = &manifests[fit_key];
    let mut fit_keys = vec!["format", "version", "state_hash"];
    fit_keys.extend(FIT_FIELDS);
    closed(fit, &fit_keys)?;
    require(
        version(fit, "dope-generator-fit") && FIT_FIELDS.iter().all(|k| fit[k] == identity[k]),
        "fit_identity_mismatch",
    )?;
    let state_key = text(&fit["state_hash"])?;
    let mut fit_requires = BTreeSet::from([state_key]);
    for field in FIT_FIELDS {
        if field.ends_with("_hash") {
            fit_requires.insert(text(&fit[field])?);
        }
    }
    require(
        nodes
            .get(state_key)
            .is_some_and(|n| n["kind"] == "fit_state")
            && children(nodes[fit_key])? == fit_requires,
        "fit_state_unbound",
    )?;
    let paired = &manifests[paired_key];
    closed(
        paired,
        &[
            "format",
            "version",
            "query_hash",
            "metric_hash",
            "loss",
            "loss_space",
            "loss_unit",
            "real",
            "synthetic",
            "null",
        ],
    )?;
    require(
        version(paired, "dope-paired-losses")
            && ["loss", "loss_space", "loss_unit"]
                .iter()
                .all(|k| paired[k] == receipt[k])
            && ["query_hash", "metric_hash"]
                .iter()
                .all(|k| paired[k] == identity[k]),
        "paired_identity_mismatch",
    )?;
    require(
        children(nodes[paired_key])?
            == BTreeSet::from([
                text(&identity["query_hash"])?,
                text(&identity["metric_hash"])?,
            ]),
        "paired_dependency_mismatch",
    )?;
    for label in ["real", "synthetic", "null"] {
        let losses = paired[label]
            .as_array()
            .ok_or_else(|| error("invalid_paired_losses"))?;
        require(
            losses.len() as u64 == uint(&identity["query_rows"])?,
            "invalid_paired_losses",
        )?;
        let mut mean = 0.0;
        let mut compensation = 0.0;
        for loss in losses {
            let x = loss
                .as_f64()
                .ok_or_else(|| error("invalid_paired_losses"))?;
            require(x.is_finite() && x >= 0.0, "invalid_paired_losses")?;
            let term = x / losses.len() as f64 - compensation;
            let sum = mean + term;
            compensation = (sum - mean) - term;
            mean = sum;
        }
        let supplied = receipt["losses"][label]
            .as_f64()
            .ok_or_else(|| error("invalid_paired_losses"))?;
        require(
            mean.is_finite()
                && (mean - supplied).abs() <= 1e-12_f64.max(1e-12 * mean.abs().max(supplied.abs())),
            "paired_loss_sum_mismatch",
        )?;
    }
    if let Some(key) = receipt["freeze_hash"].as_str() {
        require(nodes[key]["kind"] == "freeze", "dependency_kind_mismatch")?;
        let freeze = &manifests[key];
        validate(freeze, &schema(FREEZE_SCHEMA)?)?;
        require(
            [
                "protocol_hash",
                "artifact_inventory_hash",
                "configuration_hash",
                "split_hash",
                "evaluation_code_hash",
            ]
            .iter()
            .all(|k| freeze[k] == identity[k])
                && freeze["executor_hash"] == policy.executor_hash
                && freeze["cell_hash"] == scientific_cache_key(&policy.identity)?,
            "frozen_selection_mismatch",
        )?;
        for (field, kind) in [
            ("selection_receipt_hash", "selection_receipt"),
            ("access_ledger_hash", "access_ledger"),
        ] {
            let dependency = text(&freeze[field])?;
            require(
                nodes.get(dependency).is_some_and(|n| n["kind"] == kind)
                    && children(nodes[key])?.contains(dependency),
                "freeze_dependency_missing",
            )?;
        }
        if let Some(parent) = freeze["parent_freeze_hash"].as_str() {
            require(
                nodes.get(parent).is_some_and(|n| n["kind"] == "freeze")
                    && children(nodes[key])?.contains(parent),
                "freeze_parent_missing",
            )?;
        }
        require(
            manifests[protocol_key]["protocol_frozen"] == true,
            "protocol_unfrozen",
        )?;
    }
    require(
        identity["partition_role"] != "confirmation" || receipt["freeze_hash"].is_string(),
        "confirmation_freeze_required",
    )?;
    let identity = EvaluationIdentity::new(identity.clone())?;
    Ok(VerifiedEvaluationReceipt {
        receipt,
        identity,
        receipt_hash,
    })
}

pub fn commit_completed_receipt(cache: &Path, receipt: &VerifiedEvaluationReceipt) -> Result<()> {
    let key = scientific_cache_key(receipt.identity())?;
    let envelope = json!({"format":"dope-scientific-cache", "version":1, "state":"complete", "cache_key":key, "receipt_hash":receipt.receipt_hash, "receipt":receipt.receipt});
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(cache)
        .map_err(|_| error("cache_commit_failed"))?;
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let stage = cache.join(format!(
        ".staging-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&stage)
        .map_err(|_| error("cache_commit_failed"))?;
    file.write_all(&canonical_json(&envelope)?)
        .and_then(|_| file.sync_all())
        .map_err(|_| error("cache_commit_failed"))?;
    fs::rename(stage, cache.join(format!("{key}.json")))
        .map_err(|_| error("cache_commit_failed"))?;
    File::open(cache)
        .and_then(|f| f.sync_all())
        .map_err(|_| error("cache_commit_failed"))?;
    Ok(())
}
pub fn load_completed_receipt(
    cache: &Path,
    root: &Path,
    policy: &VerificationPolicy,
) -> Result<VerifiedEvaluationReceipt> {
    let key = scientific_cache_key(&policy.identity)?;
    let (_, _, data) = read_file(cache, &format!("{key}.json"), true)?;
    let envelope = parse_json(&data)?;
    closed(
        &envelope,
        &[
            "format",
            "version",
            "state",
            "cache_key",
            "receipt_hash",
            "receipt",
        ],
    )?;
    require(
        version(&envelope, "dope-scientific-cache")
            && envelope["state"] == "complete"
            && envelope["cache_key"] == key
            && envelope["receipt_hash"] == policy.receipt_hash,
        "cache_incomplete_or_mismatched",
    )?;
    verify_receipt(UnverifiedReceipt(envelope["receipt"].clone()), root, policy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    struct Fixture {
        root: PathBuf,
        receipt: Value,
    }
    fn node(
        root: &Path,
        nodes: &mut Vec<Value>,
        kind: &str,
        data: &[u8],
        requires: Vec<Value>,
    ) -> Value {
        let path = format!("{}.blob", nodes.len());
        fs::write(root.join(&path), data).unwrap();
        let key = hash(data);
        nodes.push(json!({"sha256":key, "path":path, "bytes":data.len(), "kind":kind, "requires":requires}));
        key.into()
    }
    impl Fixture {
        fn new() -> Self {
            Self::with_role(false)
        }
        fn with_role(confirmation: bool) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
                "target/vnext/rust-evidence-fixtures/{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&root).unwrap();
            let mut nodes = vec![];
            let mut identity = Map::new();
            let s = schema(RUN_SCHEMA).unwrap();
            for field in s["$defs"]["EvaluationIdentity"]["properties"]
                .as_object()
                .unwrap()
                .keys()
            {
                if let Some(kind) = field.strip_suffix("_hash") {
                    if !matches!(kind, "protocol" | "fit" | "artifact_inventory" | "split") {
                        identity.insert(
                            field.clone(),
                            node(&root, &mut nodes, kind, field.as_bytes(), vec![]),
                        );
                    }
                }
            }
            for (field, value) in json!({"partition_role":"selection", "global_seed":7, "generator_fit_seed":11, "sample_seed":13, "learner_seed":17, "fit_rows":3, "requested_rows":4, "produced_rows":4, "consumed_rows":4, "query_rows":3, "context_rows":4, "ensemble_count":1, "precision":"float64", "determinism":"deterministic"}).as_object().unwrap() {
                identity.insert(field.clone(), value.clone());
            }
            if confirmation {
                identity.insert("partition_role".into(), "confirmation".into());
            }
            let split = json!({"format":"dope-split-manifest", "version":1, "data_hash":identity["data_hash"], "query_hash":identity["query_hash"], "partition_role":identity["partition_role"], "fit_rows":identity["fit_rows"], "query_rows":identity["query_rows"]});
            let split_key = node(
                &root,
                &mut nodes,
                "split",
                &canonical_json(&split).unwrap(),
                vec![
                    identity["data_hash"].clone(),
                    identity["query_hash"].clone(),
                ],
            );
            identity.insert("split_hash".into(), split_key);
            let mut protocol =
                parse_json(include_bytes!("../production/behavior-contract-v1.json")).unwrap();
            let mut commitments = vec![];
            if confirmation {
                let policy = node(
                    &root,
                    &mut nodes,
                    "policy",
                    b"fixture-policy-without-real-authorization",
                    vec![],
                );
                protocol["status"] = "frozen".into();
                protocol["protocol_frozen"] = true.into();
                protocol["confirmation_allowed"] = true.into();
                for v in protocol["freeze_requirements"]
                    .as_object_mut()
                    .unwrap()
                    .values_mut()
                {
                    *v = policy.clone();
                }
                for v in protocol["margins"].as_object_mut().unwrap().values_mut() {
                    *v = json!(0.1);
                }
                for v in protocol["budgets"].as_object_mut().unwrap().values_mut() {
                    *v = json!(1);
                }
                for (declared, actual) in [
                    ("table_schema_hash", "schema_hash"),
                    ("task_manifest_hash", "task_hash"),
                    ("population_history_hash", "source_family_hash"),
                ] {
                    protocol["workload"][declared] = identity[actual].clone();
                    commitments.push(identity[actual].clone());
                }
                protocol["learners"]["held_out_families"] = json!(["held_out"]);
                protocol["disclosure"]["protected_unit"] = "row".into();
                protocol["disclosure"]["auxiliary_information_hash"] = policy.clone();
                protocol["disclosure"]["repeated_release_policy_hash"] = policy.clone();
                protocol["disclosure"]["privacy_route"] = "scoped".into();
                protocol["disclosure"]["legal_route"] = "scoped".into();
                protocol["disclosure"]["purpose_grants"] = json!(["customer_modeling"]);
                commitments.push(policy);
            }
            let protocol_bytes = if confirmation {
                canonical_json(&protocol).unwrap()
            } else {
                include_bytes!("../production/behavior-contract-v1.json").to_vec()
            };
            identity.insert(
                "protocol_hash".into(),
                node(&root, &mut nodes, "protocol", &protocol_bytes, commitments),
            );
            let state = node(&root, &mut nodes, "fit_state", b"fixture-fit-state", vec![]);
            let mut fit = json!({"format":"dope-generator-fit", "version":1, "state_hash":state});
            let mut fit_requires = vec![state];
            for field in FIT_FIELDS {
                fit[field] = identity[field].clone();
                if field.ends_with("_hash") {
                    fit_requires.push(identity[field].clone());
                }
            }
            identity.insert(
                "fit_hash".into(),
                node(
                    &root,
                    &mut nodes,
                    "fit",
                    &canonical_json(&fit).unwrap(),
                    fit_requires,
                ),
            );
            let artifact = node(
                &root,
                &mut nodes,
                "artifact",
                b"fixture-sampling-artifact",
                vec![],
            );
            let inventory =
                json!({"format":"dope-artifact-inventory", "version":1, "files":[artifact]});
            identity.insert(
                "artifact_inventory_hash".into(),
                node(
                    &root,
                    &mut nodes,
                    "artifact_inventory",
                    &canonical_json(&inventory).unwrap(),
                    vec![artifact],
                ),
            );
            let paired = json!({"format":"dope-paired-losses", "version":1, "query_hash":identity["query_hash"], "metric_hash":identity["metric_hash"], "loss":"mse", "loss_space":"original_target_units", "loss_unit":"target_units_squared", "real":[0,1,2], "synthetic":[1,2,3], "null":[2,3,4]});
            let paired_key = node(
                &root,
                &mut nodes,
                "paired_evidence",
                &canonical_json(&paired).unwrap(),
                vec![
                    identity["query_hash"].clone(),
                    identity["metric_hash"].clone(),
                ],
            );
            let executor = node(&root, &mut nodes, "executor", b"fixture-executor", vec![]);
            let freeze_hash = if confirmation {
                let selection = node(
                    &root,
                    &mut nodes,
                    "selection_receipt",
                    b"fixture-selection-receipt",
                    vec![],
                );
                let access = node(
                    &root,
                    &mut nodes,
                    "access_ledger",
                    b"fixture-access-ledger",
                    vec![],
                );
                let mut freeze = json!({"format":"dope-freeze", "version":1, "selection_receipt_hash":selection, "executor_hash":executor, "access_ledger_hash":access, "cell_hash":scientific_cache_key(&EvaluationIdentity::new(identity.clone().into()).unwrap()).unwrap(), "parent_freeze_hash":null, "protocol_frozen":true, "confirmation_allowed":true});
                for field in [
                    "protocol_hash",
                    "artifact_inventory_hash",
                    "configuration_hash",
                    "split_hash",
                    "evaluation_code_hash",
                ] {
                    freeze[field] = identity[field].clone();
                }
                node(
                    &root,
                    &mut nodes,
                    "freeze",
                    &canonical_json(&freeze).unwrap(),
                    vec![selection, access],
                )
            } else {
                Value::Null
            };
            let receipt = json!({"format":"dope-transfer-run", "version":1, "identity":identity, "executor_hash":executor, "freeze_hash":freeze_hash, "dependencies":nodes, "sampling_payload_bytes":b"fixture-sampling-artifact".len(), "paired_evidence_hash":paired_key, "loss":"mse", "loss_space":"original_target_units", "loss_unit":"target_units_squared", "losses":{"real":1.0,"synthetic":2.0,"null":3.0}, "execution":"complete", "attempt":{"attempt_id":"a".repeat(64),"parent_attempt_id":null,"exit_code":0,"wall_time_ms":1,"cpu_time_ms":0,"peak_memory_bytes":1}, "projection":"private", "release":"restricted_local", "transfer":"unmeasured", "empirical_privacy":"unmeasured", "formal_mechanism":"none", "legal_route":"unreviewed"});
            Self { root, receipt }
        }
        fn policy(&self) -> VerificationPolicy {
            VerificationPolicy {
                identity: EvaluationIdentity::new(self.receipt["identity"].clone()).unwrap(),
                executor_hash: text(&self.receipt["executor_hash"]).unwrap().into(),
                receipt_hash: digest(&self.receipt).unwrap(),
                freeze_hash: self.receipt["freeze_hash"].as_str().map(String::from),
            }
        }
        fn verify(&self) -> Result<VerifiedEvaluationReceipt> {
            verify_receipt(
                UnverifiedReceipt::from_bytes(&canonical_json(&self.receipt)?)?,
                &self.root,
                &self.policy(),
            )
        }
    }

    #[test]
    fn canonical_json_matches_python_golden_and_rejects_duplicates() {
        let value = json!({"z":-0.0,"b":0.3,"a":[true,null,"é"]});
        assert_eq!(
            String::from_utf8(canonical_json(&value).unwrap()).unwrap(),
            "{\"a\":[true,null,\"é\"],\"b\":2.99999999999999989e-1,\"z\":-0.00000000000000000e0}"
        );
        assert_eq!(
            digest(&json!({"a":1,"b":2})).unwrap(),
            "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
        );
        for data in [
            br#"{"a":1,"a":2}"#.as_slice(),
            br#"{"x":{"a":1,"a":2}}"#,
            b"NaN",
            b"Infinity",
            b"1e999",
            b"9007199254740992",
            b"184467440737095516160",
            br#""\ud800""#,
        ] {
            assert!(parse_json(data).is_err());
        }
    }

    #[test]
    fn numeric_parser_roundtrips_extremes_and_preserves_json_grammar() {
        let original = json!([
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            -0.0,
            0.3,
            1e-100,
            1e100
        ]);
        let encoded = canonical_json(&original).unwrap();
        assert_eq!(
            canonical_json(&parse_json(&encoded).unwrap()).unwrap(),
            encoded
        );
        assert_eq!(
            canonical_json(&parse_json(br#"{"z":1.0,"a":2,"digit\"1":3.5}"#).unwrap()).unwrap(),
            br#"{"a":2,"digit\"1":3.50000000000000000e0,"z":1.00000000000000000e0}"#
        );
        for input in [
            b"01".as_slice(),
            b"1.",
            b"-.1",
            b"1e",
            b"--1",
            b"[1 2]",
            b"[1,]",
            b"1 true",
        ] {
            assert!(parse_json(input).is_err());
        }
    }

    #[test]
    fn completed_receipt_reverifies_atomic_cache_and_corruption() {
        let f = Fixture::new();
        let v = f.verify().unwrap();
        let cache = f.root.join("cache");
        commit_completed_receipt(&cache, &v).unwrap();
        assert_eq!(
            fs::metadata(&cache).unwrap().permissions().mode() & 0o077,
            0
        );
        assert_eq!(
            fs::metadata(cache.join(format!(
                "{}.json",
                scientific_cache_key(v.identity()).unwrap()
            )))
            .unwrap()
            .permissions()
            .mode()
                & 0o077,
            0
        );
        assert_eq!(
            load_completed_receipt(&cache, &f.root, &f.policy())
                .unwrap()
                .receipt_hash(),
            v.receipt_hash()
        );
        fs::write(
            f.root
                .join(text(&f.receipt["dependencies"][0]["path"]).unwrap()),
            b"corrupt",
        )
        .unwrap();
        assert!(load_completed_receipt(&cache, &f.root, &f.policy()).is_err());
    }

    #[test]
    fn full_identity_changes_cache_attempts_and_aliases_do_not_change_fit() {
        let mut f = Fixture::new();
        let first = f.verify().unwrap();
        let key = scientific_cache_key(first.identity()).unwrap();
        assert_eq!(
            key,
            "43b5caa1ed64df2fc60133d256ffc7d5e5afdd0866c1abec8c4f82b19176994f"
        );
        assert_eq!(
            fit_replicate_key(first.identity()).unwrap(),
            "6774caa4afc0fcad64da7139d288b94a849ecc786c7a0cfa45112e36175da5a1"
        );
        for (field, value) in first.identity().value().as_object().unwrap() {
            let mut changed = first.identity().value().clone();
            changed[field] = if field.ends_with("_hash") {
                "f".repeat(64).into()
            } else if let Some(n) = value.as_u64() {
                (n + 1).into()
            } else {
                match field.as_str() {
                    "partition_role" => "exploratory",
                    "precision" => "float32",
                    _ => "seeded_best_effort",
                }
                .into()
            };
            assert_ne!(
                key,
                scientific_cache_key(&EvaluationIdentity::new(changed).unwrap()).unwrap(),
                "{field}"
            );
        }
        f.receipt["attempt"]["attempt_id"] = "b".repeat(64).into();
        for n in f.receipt["dependencies"].as_array_mut().unwrap() {
            let old = text(&n["path"]).unwrap();
            let alias = format!("alias-{old}");
            fs::hard_link(f.root.join(old), f.root.join(&alias)).unwrap();
            n["path"] = alias.into();
        }
        let after = f.verify().unwrap();
        assert_eq!(key, scientific_cache_key(after.identity()).unwrap());
        assert_eq!(
            fit_replicate_key(first.identity()).unwrap(),
            fit_replicate_key(after.identity()).unwrap()
        );
        assert_ne!(first.receipt_hash(), after.receipt_hash());
    }

    #[test]
    fn executor_receipt_pins_and_confirmation_fail_closed() {
        let f = Fixture::new();
        let mut policy = f.policy();
        policy.receipt_hash = "a".repeat(64);
        assert!(verify_receipt(UnverifiedReceipt(f.receipt.clone()), &f.root, &policy).is_err());
        policy = f.policy();
        policy.executor_hash = "a".repeat(64);
        assert!(verify_receipt(UnverifiedReceipt(f.receipt.clone()), &f.root, &policy).is_err());
        let mut f = Fixture::new();
        f.receipt["identity"]["partition_role"] = "confirmation".into();
        assert!(f.verify().is_err());
    }

    #[test]
    fn fabricated_booleans_versions_byte_sums_losses_and_rows_fail() {
        for (field, value) in [
            ("version", json!(true)),
            ("version", json!(2)),
            ("version", json!(1.0)),
            ("sampling_payload_bytes", json!(true)),
            ("sampling_payload_bytes", json!(999)),
            ("loss_unit", json!("target_units")),
            ("projection", json!("sanitized_public")),
            ("release", json!("eligible")),
            ("artifact_sampling_verified", json!(true)),
        ] {
            let mut f = Fixture::new();
            f.receipt[field] = value;
            assert!(f.verify().is_err(), "{field}");
        }
        let mut f = Fixture::new();
        f.receipt["losses"]["synthetic"] = json!(0);
        assert!(f.verify().is_err());
        f = Fixture::new();
        f.receipt["losses"]["real"] = json!(true);
        assert!(f.verify().is_err());
        for field in [
            "produced_rows",
            "consumed_rows",
            "context_rows",
            "query_rows",
        ] {
            let mut f = Fixture::new();
            f.receipt["identity"][field] = json!(999);
            assert!(f.verify().is_err());
        }
    }

    #[test]
    fn dag_cycles_missing_aliases_and_inconsistent_metadata_fail() {
        for mutation in ["cycle", "missing", "alias", "bytes", "kind"] {
            let mut f = Fixture::new();
            let node = f.receipt["dependencies"][0].clone();
            match mutation {
                "cycle" => f.receipt["dependencies"][0]["requires"] = json!([node["sha256"]]),
                "missing" => {
                    f.receipt["dependencies"].as_array_mut().unwrap().remove(0);
                }
                "alias" => f.receipt["dependencies"].as_array_mut().unwrap().push(node),
                "bytes" => f.receipt["dependencies"][0]["bytes"] = json!(999),
                _ => f.receipt["dependencies"][0]["kind"] = json!("policy"),
            }
            assert!(f.verify().is_err(), "{mutation}");
        }
    }

    #[test]
    fn unsafe_paths_symlinks_and_diagnostics_do_not_expose_values() {
        for path in [
            "../private-source-value",
            "/private-source-value",
            "./private-source-value",
            "private-source-value\\secret",
        ] {
            let mut f = Fixture::new();
            f.receipt["dependencies"][0]["path"] = path.into();
            let message = f.verify().unwrap_err().to_string();
            assert!(!message.contains("private-source-value"));
        }
        let mut f = Fixture::new();
        let target = f
            .root
            .join(text(&f.receipt["dependencies"][0]["path"]).unwrap());
        std::os::unix::fs::symlink(target, f.root.join("link")).unwrap();
        f.receipt["dependencies"][0]["path"] = "link".into();
        assert!(f.verify().is_err());
    }

    #[test]
    fn partial_cache_and_failure_dispositions_cannot_be_successes() {
        let f = Fixture::new();
        let cache = f.root.join("cache");
        let v = f.verify().unwrap();
        commit_completed_receipt(&cache, &v).unwrap();
        let key = scientific_cache_key(v.identity()).unwrap();
        fs::rename(
            cache.join(format!("{key}.json")),
            cache.join(".staging-interrupted"),
        )
        .unwrap();
        assert!(load_completed_receipt(&cache, &f.root, &f.policy()).is_err());
        for state in [
            "unsupported",
            "timeout",
            "oom",
            "invalid_output",
            "byte_infeasible",
            "protocol_violation",
            "unresolved",
        ] {
            let mut f = Fixture::new();
            f.receipt["execution"] = state.into();
            f.receipt["losses"] = Value::Null;
            f.receipt["paired_evidence_hash"] = Value::Null;
            assert!(f.verify().is_err());
        }
    }

    #[test]
    fn confirmation_verifies_frozen_commitments_and_rejects_substitution() {
        let mut f = Fixture::with_role(true);
        let v = f.verify().unwrap();
        let cache = f.root.join("cache");
        commit_completed_receipt(&cache, &v).unwrap();
        load_completed_receipt(&cache, &f.root, &f.policy()).unwrap();
        let mut policy = f.policy();
        let mut identity = policy.identity.value().clone();
        identity["partition_role"] = "selection".into();
        policy.identity = EvaluationIdentity::new(identity).unwrap();
        assert!(load_completed_receipt(&cache, &f.root, &policy).is_err());
        f.receipt["freeze_hash"] = Value::Null;
        assert!(f.verify().is_err());
    }

    #[test]
    fn draft_contract_preserves_historical_bytes_and_rejects_false_freeze() {
        let c = parse_json(include_bytes!("../production/behavior-contract-v1.json")).unwrap();
        validate_behavior_contract(&c).unwrap();
        for (path, expected) in c["historical_contract_sha256"].as_object().unwrap() {
            assert_eq!(
                hash(&fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join(path)).unwrap()),
                expected.as_str().unwrap()
            );
        }
        let mut invalid = c;
        invalid["status"] = "frozen".into();
        invalid["protocol_frozen"] = true.into();
        invalid["confirmation_allowed"] = true.into();
        assert!(validate_behavior_contract(&invalid).is_err());
    }
}

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

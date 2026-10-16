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

#[cfg(test)]
mod tests {
    use super::*;

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
            "rows": rows,
            "bootstrap_replicates": 200,
            "bootstrap_seed": 1729,
            "alpha": 0.05
        }));
        assert_eq!(agg["status"], "complete");
        assert!((agg["robust_transfer"].as_f64().unwrap() - 0.915).abs() <= 1e-12);
    }
}

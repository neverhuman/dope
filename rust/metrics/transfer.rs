//! Paired original-unit losses and informative-task admission (T-030).

use serde_json::{Value, json};

use super::{
    MetricError, MetricResult, failure, failure_with_status, mean, number, require, vector,
};

pub fn evaluate(data: &Value) -> Value {
    match evaluate_inner(data) {
        Ok(val) => val,
        Err(err) => failure_with_status(err.reason, err.status),
    }
}

fn evaluate_inner(data: &Value) -> MetricResult<Value> {
    let obj = data.as_object().ok_or(MetricError::new("invalid_input"))?;
    let required = [
        "null_losses",
        "real_losses",
        "synthetic_losses",
        "real_gain_lcb",
        "constant_target",
    ];
    if required
        .iter()
        .any(|k| obj.get(*k).is_none_or(Value::is_null))
    {
        return Err(MetricError::with_status(
            "missing_transfer_cell",
            "incomplete",
        ));
    }
    let null = vector(&obj["null_losses"], true)?;
    let real = vector(&obj["real_losses"], true)?;
    let syn = vector(&obj["synthetic_losses"], true)?;
    require(
        null.len() == real.len() && real.len() == syn.len(),
        "unpaired_losses",
    )?;
    let constant_target = obj["constant_target"]
        .as_bool()
        .ok_or(MetricError::new("invalid_constant_target"))?;
    let lcb = number(&obj["real_gain_lcb"])?;
    let absolute = match obj.get("absolute_gain_floor") {
        Some(v) if !v.is_null() => number(v)?,
        _ => 1e-12,
    };
    let relative = match obj.get("relative_gain_floor") {
        Some(v) if !v.is_null() => number(v)?,
        _ => 0.01,
    };
    require(absolute >= 0.0 && relative >= 0.0, "invalid_gain_floor")?;
    let ln = mean(&null)?;
    let lr = mean(&real)?;
    let ls = mean(&syn)?;
    let gain = ln - lr;
    require(gain.is_finite(), "nonfinite_or_invalid_number")?;
    if constant_target {
        return Err(MetricError::with_status("constant_target", "uninformative"));
    }
    let rel_floor = relative * ln.abs();
    require(rel_floor.is_finite(), "nonfinite_or_invalid_number")?;
    if gain <= 0.0 || gain < absolute.max(rel_floor) || lcb <= 0.0 {
        return Err(MetricError::with_status(
            "real_gain_not_informative",
            "uninformative",
        ));
    }
    let syn_gain = ln - ls;
    require(syn_gain.is_finite(), "nonfinite_or_invalid_number")?;
    let retention = syn_gain / gain;
    require(retention.is_finite(), "nonfinite_or_invalid_number")?;
    let clipped = retention.clamp(0.0, 1.0);
    Ok(json!({
        "status": "informative",
        "reason": Value::Null,
        "null_loss": ln,
        "real_loss": lr,
        "synthetic_loss": ls,
        "real_gain": gain,
        "raw_transfer": retention,
        "clipped_transfer": clipped,
    }))
}

#[allow(dead_code)]
pub fn is_informative(report: &Value) -> bool {
    report.get("status").and_then(Value::as_str) == Some("informative")
}

#[allow(dead_code)]
pub fn invalid_transfer_failure(reason: &str) -> Value {
    failure(reason)
}

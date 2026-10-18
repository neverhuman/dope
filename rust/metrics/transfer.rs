//! Paired original-unit losses and informative-task admission (T-030).

use crate::transfer_metrics::{LossSpace, PairedLosses, TdlMargin};
use serde_json::{Value, json};

use super::{
    MetricError, MetricResult, count, failure, failure_with_status, mean, number, require, vector,
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

/// Candidate paired-loss diagnostics. Statistical qualification belongs to C02/C04;
/// an informative report or a nonpositive point margin grants no qualification.
pub fn evaluate_tdl(data: &Value) -> Value {
    match evaluate_tdl_inner(data) {
        Ok(report) => report,
        Err(err) => failure_with_status(err.reason, err.status),
    }
}

fn primitive<T>(value: Result<T, crate::transfer_metrics::MetricError>) -> MetricResult<T> {
    value.map_err(|err| {
        MetricError::new(match err {
            crate::transfer_metrics::MetricError::Nonfinite => "nonfinite_or_invalid_number",
            _ => "invalid_paired_loss_evidence",
        })
    })
}

fn tdl_pairs(data: &Value) -> MetricResult<PairedLosses> {
    let task = data["task"]
        .as_str()
        .ok_or(MetricError::new("invalid_task"))?;
    if task == "paired_losses" {
        let mut pairs = primitive(PairedLosses::new(
            vector(&data["real_losses"], true)?,
            vector(&data["synthetic_losses"], true)?,
            vector(&data["null_losses"], true)?,
        ))?;
        pairs.loss_space = match data["loss_space"].as_str() {
            Some("original_mse") => LossSpace::OriginalMse,
            Some("discrete_log_loss") => LossSpace::DiscreteLogLoss,
            _ => return Err(MetricError::new("invalid_loss_space")),
        };
        return Ok(pairs);
    }
    if task == "regression" {
        return primitive(PairedLosses::regression_transformed(
            &vector(&data["targets"], false)?,
            &vector(&data["real_predictions"], false)?,
            &vector(&data["synthetic_predictions"], false)?,
            &vector(&data["null_predictions"], false)?,
            number(&data["prediction_offset"])?,
            number(&data["prediction_scale"])?,
        ));
    }
    require(task == "binary" || task == "multiclass", "invalid_task")?;
    let order = data["class_order"]
        .as_array()
        .ok_or(MetricError::new("invalid_class_order"))?;
    let mut classes = std::collections::BTreeSet::new();
    for label in order {
        let label = label
            .as_str()
            .ok_or(MetricError::new("invalid_class_order"))?;
        require(
            !label.is_empty() && classes.insert(label),
            "invalid_class_order",
        )?;
    }
    require(
        order.len() >= 2 && (task != "binary" || order.len() == 2),
        "invalid_class_order",
    )?;
    let targets = data["targets"]
        .as_array()
        .ok_or(MetricError::new("invalid_targets"))?
        .iter()
        .map(|y| usize::try_from(count(y, 0)?).map_err(|_| MetricError::new("invalid_targets")))
        .collect::<MetricResult<Vec<_>>>()?;
    let matrix = |key: &str| -> MetricResult<Vec<Vec<f64>>> {
        data[key]
            .as_array()
            .ok_or(MetricError::new("invalid_probabilities"))?
            .iter()
            .map(|row| vector(row, true))
            .collect()
    };
    primitive(PairedLosses::classification(
        &targets,
        &matrix("real_predictions")?,
        &matrix("synthetic_predictions")?,
        &matrix("null_predictions")?,
        order.len(),
    ))
}

fn evaluate_tdl_inner(data: &Value) -> MetricResult<Value> {
    require(data.is_object(), "invalid_input")?;
    match data.get("required_model_present") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => {
            return Err(MetricError::with_status(
                "missing_required_model",
                "incomplete",
            ));
        }
        Some(Value::Bool(true)) => (),
        _ => return Err(MetricError::new("invalid_required_model")),
    }
    let margin = TdlMargin {
        tau: match data.get("tau") {
            None => TdlMargin::default().tau,
            Some(value) => number(value)?,
        },
    };
    // Validate tau even if the historical gate will refuse the cell.
    require((0.0..=1.0).contains(&margin.tau), "invalid_tau")?;
    let pairs = tdl_pairs(data)?;
    let (real, synthetic, null_losses) = pairs.loss_vectors();
    let mut gate = data.clone();
    gate["real_losses"] = json!(real);
    gate["synthetic_losses"] = json!(synthetic);
    gate["null_losses"] = json!(null_losses);
    // Reuse historical informative-reference defaults; candidate calibration remains null.
    let mut report = evaluate(&gate);
    if !is_informative(&report) {
        return Ok(report);
    }
    let harm = primitive(pairs.query_harm_margin(margin))?;
    let mean_harm = mean(&harm)?;
    report["tau"] = json!(margin.tau);
    report["loss_space"] = json!(match pairs.loss_space {
        LossSpace::OriginalMse => "original_mse",
        LossSpace::DiscreteLogLoss => "discrete_log_loss",
    });
    report["query_harm_margin"] = json!(harm);
    report["harm_margin"] = json!(mean_harm);
    report["benefit_margin"] = json!(-mean_harm);
    Ok(report)
}

//! Smallest tested passing artifact, with a cumulative budget frontier (T-033).

use std::collections::BTreeSet;

use serde_json::{Value, json};

use super::{MetricError, MetricResult, count, failure_with_status, number, require, unit};

fn parse_unique_strings(
    value: Option<&Value>,
    invalid_err: &'static str,
    dup_err: &'static str,
) -> MetricResult<BTreeSet<String>> {
    let arr = value
        .and_then(Value::as_array)
        .ok_or(MetricError::new(invalid_err))?;
    require(!arr.is_empty(), invalid_err)?;
    let mut set = BTreeSet::new();
    for item in arr {
        let s = item.as_str().ok_or(MetricError::new(invalid_err))?;
        require(!s.is_empty(), invalid_err)?;
        require(set.insert(s.to_owned()), dup_err)?;
    }
    Ok(set)
}

pub fn evaluate(data: &Value) -> Value {
    match evaluate_inner(data) {
        Ok(val) => val,
        Err(err) => failure_with_status(err.reason, err.status),
    }
}

fn evaluate_inner(data: &Value) -> MetricResult<Value> {
    let obj = data.as_object().ok_or(MetricError::new("invalid_input"))?;
    let cells_val = match obj.get("cells") {
        None | Some(Value::Null) => {
            return Err(MetricError::with_status(
                "missing_budget_cell",
                "incomplete",
            ));
        }
        Some(v) => v,
    };
    let cells = cells_val
        .as_array()
        .ok_or(MetricError::new("invalid_cells"))?;
    if cells.is_empty() {
        return Err(MetricError::with_status(
            "missing_budget_cell",
            "incomplete",
        ));
    }

    let required_families = parse_unique_strings(
        obj.get("required_families"),
        "invalid_families",
        "duplicate_family",
    )?;
    let required_gates =
        parse_unique_strings(obj.get("required_gates"), "invalid_gates", "duplicate_gate")?;
    let tt = match obj.get("transfer_threshold") {
        Some(v) if !v.is_null() => unit(v)?,
        _ => 0.90,
    };
    let dt = match obj.get("driver_threshold") {
        Some(v) if !v.is_null() => unit(v)?,
        _ => 0.80,
    };

    let mut budgets = BTreeSet::new();
    let mut admitted: Vec<(u64, f64, f64, bool)> = Vec::new();
    let mut states: Vec<&'static str> = Vec::new();

    for cell_val in cells {
        let cell = cell_val
            .as_object()
            .ok_or(MetricError::new("invalid_cells"))?;
        let size = count(
            cell.get("artifact_bytes")
                .ok_or(MetricError::new("invalid_count"))?,
            1,
        )?;
        require(budgets.insert(size), "duplicate_budget")?;
        let state = match cell.get("status").and_then(Value::as_str) {
            Some("complete") => "complete",
            Some("unqualified") => "unqualified",
            Some("incomplete") => "incomplete",
            Some("failed") => "failed",
            _ => return Err(MetricError::new("invalid_cell_status")),
        };
        states.push(state);
        if state != "complete" {
            continue;
        }
        let observed = match cell.get("families").and_then(Value::as_array) {
            Some(arr) => {
                let obs_set: BTreeSet<&str> = arr.iter().filter_map(Value::as_str).collect();
                if !required_families
                    .iter()
                    .all(|req| obs_set.contains(req.as_str()))
                {
                    states.push("incomplete");
                    continue;
                }
                obs_set
            }
            None => {
                states.push("incomplete");
                continue;
            }
        };
        let _ = observed;
        let measured = match cell.get("gates").and_then(Value::as_object) {
            Some(m) => {
                if required_gates
                    .iter()
                    .any(|g| m.get(g).is_none_or(Value::is_null))
                {
                    states.push("incomplete");
                    continue;
                }
                m
            }
            None => {
                states.push("incomplete");
                continue;
            }
        };
        let mut all_gates = true;
        for g in &required_gates {
            let b = measured[g]
                .as_bool()
                .ok_or(MetricError::new("invalid_gate"))?;
            all_gates &= b;
        }
        if cell.get("lcb_transfer").is_none_or(Value::is_null)
            || cell.get("driver_fidelity").is_none_or(Value::is_null)
        {
            states.push("incomplete");
            continue;
        }
        let transfer = number(&cell["lcb_transfer"])?;
        let driver = unit(&cell["driver_fidelity"])?;
        admitted.push((size, transfer, driver, all_gates));
    }

    for &(bad_state, bad_reason) in &[
        ("failed", "budget_cell_failed"),
        ("unqualified", "budget_cell_unqualified"),
        ("incomplete", "budget_cell_incomplete"),
    ] {
        if states.contains(&bad_state) {
            return Err(MetricError::with_status(bad_reason, bad_state));
        }
    }

    admitted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut rows = Vec::with_capacity(admitted.len());
    let mut passing = Vec::new();
    let mut best: Option<f64> = None;

    for (size, transfer, driver, passed_gates) in admitted {
        let passed = passed_gates && transfer >= tt && driver >= dt;
        if passed {
            passing.push(size);
        }
        if passed_gates && driver >= dt {
            best = Some(match best {
                Some(prev) => prev.max(transfer),
                None => transfer,
            });
        }
        rows.push(json!({
            "artifact_bytes": size,
            "lcb_transfer": transfer,
            "driver_fidelity": driver,
            "tdl_pass": passed,
            "frontier_transfer": best,
        }));
    }

    let largest_tested = budgets.iter().next_back().copied().unwrap_or(0);
    let tdl_bytes = passing.iter().min().copied();
    let status = if tdl_bytes.is_some() {
        "qualified"
    } else {
        "not_reached_in_tested_range"
    };

    Ok(json!({
        "status": status,
        "reason": Value::Null,
        "tdl_bytes": tdl_bytes,
        "largest_tested_bytes": largest_tested,
        "frontier": rows,
    }))
}

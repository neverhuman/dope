//! Smallest tested passing artifact, with a cumulative budget frontier (T-033).

use std::collections::{BTreeMap, BTreeSet};

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

fn parse_grid(value: Option<&Value>) -> MetricResult<Option<BTreeSet<u64>>> {
    let val = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(v) => v,
    };
    let arr = val
        .as_array()
        .ok_or(MetricError::new("invalid_byte_grid"))?;
    require(!arr.is_empty(), "invalid_byte_grid")?;
    let mut out = BTreeSet::new();
    for item in arr {
        let b = count(item, 1)?;
        require(out.insert(b), "duplicate_budget")?;
    }
    Ok(Some(out))
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
    let raw_grid = match obj.get("byte_grid") {
        Some(v) if !v.is_null() => Some(v),
        _ => obj.get("declared_grid_bytes"),
    };
    let declared_grid = parse_grid(raw_grid)?;
    // T-122 B5: byte_grid is mandatory; do not invent it from cell budgets or artifact sizes.
    let Some(declared_grid) = declared_grid else {
        return Err(MetricError::with_status(
            "missing_budget_cell",
            "incomplete",
        ));
    };
    let tt = match obj.get("transfer_threshold") {
        Some(v) if !v.is_null() => unit(v)?,
        _ => 0.90,
    };
    let dt = match obj.get("driver_threshold") {
        Some(v) if !v.is_null() => unit(v)?,
        _ => 0.80,
    };
    // Budget-grid claim mode vs artifact-only; byte_grid is mandatory in both (above).
    let has_grid = cells.iter().any(|c| {
        c.as_object()
            .and_then(|m| m.get("budget_bytes"))
            .is_some_and(|v| !v.is_null())
    });

    let mut budgets = BTreeSet::new();
    type AdmittedCell = (u64, u64, f64, f64, bool, BTreeMap<String, bool>);
    let mut admitted: Vec<AdmittedCell> = Vec::new();
    let mut states: Vec<&'static str> = Vec::new();

    for cell_val in cells {
        let cell = cell_val
            .as_object()
            .ok_or(MetricError::new("invalid_cells"))?;
        let (budget, size_opt) = match cell.get("budget_bytes") {
            Some(bv) if !bv.is_null() => {
                let b = count(bv, 1)?;
                let s_opt = match cell.get("artifact_bytes") {
                    Some(sv) if !sv.is_null() => {
                        let s = count(sv, 1)?;
                        require(s <= b, "artifact_exceeds_budget")?;
                        Some(s)
                    }
                    _ => None,
                };
                (b, s_opt)
            }
            _ => {
                let s = count(
                    cell.get("artifact_bytes")
                        .ok_or(MetricError::new("invalid_count"))?,
                    1,
                )?;
                (s, Some(s))
            }
        };
        require(declared_grid.contains(&budget), "undeclared_budget")?;
        require(budgets.insert(budget), "duplicate_budget")?;
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
        let Some(size) = size_opt else {
            states.push("incomplete");
            continue;
        };
        match cell.get("families") {
            None | Some(Value::Null) => {
                states.push("incomplete");
                continue;
            }
            Some(Value::Array(arr)) => {
                let mut obs_set = BTreeSet::new();
                for item in arr {
                    let s = item.as_str().ok_or(MetricError::new("invalid_families"))?;
                    require(!s.is_empty(), "invalid_families")?;
                    obs_set.insert(s);
                }
                if !required_families
                    .iter()
                    .all(|req| obs_set.contains(req.as_str()))
                {
                    states.push("incomplete");
                    continue;
                }
            }
            Some(_) => return Err(MetricError::new("invalid_families")),
        }
        let measured = match cell.get("gates") {
            None | Some(Value::Null) => {
                states.push("incomplete");
                continue;
            }
            Some(Value::Object(m)) => {
                if required_gates
                    .iter()
                    .any(|g| m.get(g).is_none_or(Value::is_null))
                {
                    states.push("incomplete");
                    continue;
                }
                m
            }
            Some(_) => return Err(MetricError::new("invalid_gate")),
        };
        let mut all_gates = true;
        let mut gate_map = BTreeMap::new();
        for g in &required_gates {
            let b = measured[g]
                .as_bool()
                .ok_or(MetricError::new("invalid_gate"))?;
            all_gates &= b;
            gate_map.insert(g.clone(), b);
        }
        if cell.get("lcb_transfer").is_none_or(Value::is_null)
            || cell.get("driver_fidelity").is_none_or(Value::is_null)
        {
            states.push("incomplete");
            continue;
        }
        let transfer = number(&cell["lcb_transfer"])?;
        let driver = unit(&cell["driver_fidelity"])?;
        admitted.push((budget, size, transfer, driver, all_gates, gate_map));
    }

    for &(bad_state, bad_reason) in &[
        ("failed", "budget_cell_failed"),
        ("unqualified", "budget_cell_unqualified"),
    ] {
        if states.contains(&bad_state) {
            return Err(MetricError::with_status(bad_reason, bad_state));
        }
    }
    if budgets != declared_grid {
        return Err(MetricError::with_status(
            "missing_budget_cell",
            "incomplete",
        ));
    }
    if states.contains(&"incomplete") {
        return Err(MetricError::with_status(
            "budget_cell_incomplete",
            "incomplete",
        ));
    }

    admitted.sort_by(|a, b| a.0.cmp(&b.0));
    if admitted.iter().all(|item| !item.4) {
        let always_failing = required_gates
            .iter()
            .find(|g| admitted.iter().all(|item| !item.5[*g]));
        let failed_gate = match always_failing {
            Some(g) => g.as_str(),
            None => required_gates
                .iter()
                .find(|g| admitted.iter().any(|item| !item.5[*g]))
                .map(String::as_str)
                .unwrap_or("unknown"),
        };
        return Ok(failure_with_status(
            &format!("gate_failed_{failed_gate}"),
            "unqualified",
        ));
    }

    let mut rows = Vec::with_capacity(admitted.len());
    let mut qualifying_idx: Option<usize> = None;
    let mut best: Option<f64> = None;

    for (idx, &(budget, size, transfer, driver, passed_gates, _)) in admitted.iter().enumerate() {
        let passed = passed_gates && transfer >= tt && driver >= dt;
        if passed && qualifying_idx.is_none() {
            qualifying_idx = Some(idx);
        }
        if passed_gates && driver >= dt {
            best = Some(match best {
                Some(prev) => prev.max(transfer),
                None => transfer,
            });
        }
        if has_grid {
            rows.push(json!({
                "budget_bytes": budget,
                "artifact_bytes": size,
                "lcb_transfer": transfer,
                "driver_fidelity": driver,
                "tdl_pass": passed,
                "frontier_transfer": best,
            }));
        } else {
            rows.push(json!({
                "artifact_bytes": size,
                "lcb_transfer": transfer,
                "driver_fidelity": driver,
                "tdl_pass": passed,
                "frontier_transfer": best,
            }));
        }
    }

    let largest_tested = budgets.iter().next_back().copied().unwrap_or(0);
    let tdl_bytes = qualifying_idx.map(|i| admitted[i].1);
    let status = if qualifying_idx.is_some() {
        "qualified"
    } else {
        "not_reached_in_tested_range"
    };

    if has_grid {
        let tdl_budget = qualifying_idx.map(|i| admitted[i].0);
        let interval = qualifying_idx.map(|i| {
            let prev = if i == 0 { 0 } else { admitted[i - 1].0 };
            json!([prev, admitted[i].0])
        });
        Ok(json!({
            "status": status,
            "reason": Value::Null,
            "tdl_bytes": tdl_bytes,
            "tdl_budget_bytes": tdl_budget,
            "tdl_interval_bytes": interval,
            "largest_tested_bytes": largest_tested,
            "frontier": rows,
        }))
    } else {
        Ok(json!({
            "status": status,
            "reason": Value::Null,
            "tdl_bytes": tdl_bytes,
            "largest_tested_bytes": largest_tested,
            "frontier": rows,
        }))
    }
}

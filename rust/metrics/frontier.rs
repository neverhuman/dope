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
    let result = match data.get("protocol_version") {
        None => evaluate_inner(data),
        Some(v) if v.as_u64() == Some(1) && !v.is_f64() => evaluate_inner(data),
        Some(v) if v.as_u64() == Some(2) && !v.is_f64() => evaluate_v2(data),
        _ => Err(MetricError::new("invalid_frontier_version")),
    };
    match result {
        Ok(val) => val,
        Err(err) => failure_with_status(err.reason, err.status),
    }
}

/// V2 consumes evaluator-owned joint qualification decisions for a frozen
/// method/seed and roster. The golden envelope remains version 1 so the shared
/// historical runner can dispatch this component's explicit protocol version.
fn evaluate_v2(data: &Value) -> MetricResult<Value> {
    use crate::transfer_metrics::frontier::{ArtifactTrial, non_monotone, roster_coverage};
    let obj = data.as_object().ok_or(MetricError::new("invalid_input"))?;
    let roster = parse_unique_strings(
        obj.get("admitted_tasks"),
        "invalid_roster",
        "duplicate_task",
    )?;
    let task_indices: BTreeMap<_, _> = roster
        .iter()
        .enumerate()
        .map(|(i, task)| (task.as_str(), i))
        .collect();
    for key in ["method_id", "seed_id"] {
        require(
            obj.get(key)
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty()),
            "invalid_frontier_identity",
        )?;
    }
    let grid = parse_grid(obj.get("byte_grid"))?.ok_or(MetricError::new("invalid_byte_grid"))?;
    let monotone = match obj.get("require_monotone") {
        None => false,
        Some(v) => v
            .as_bool()
            .ok_or(MetricError::new("invalid_monotone_policy"))?,
    };
    let cells = obj
        .get("cells")
        .and_then(Value::as_array)
        .ok_or(MetricError::new("invalid_cells"))?;
    let mut trials = Vec::new();
    let mut identities = BTreeSet::new();
    for cell in cells {
        let cell = cell.as_object().ok_or(MetricError::new("invalid_cells"))?;
        for key in ["method_id", "seed_id"] {
            require(cell.get(key) == obj.get(key), "frontier_identity_mismatch")?;
        }
        let task = cell
            .get("task_id")
            .and_then(Value::as_str)
            .ok_or(MetricError::new("invalid_roster_task"))?;
        let task_index = *task_indices
            .get(task)
            .ok_or(MetricError::new("unscheduled_task"))?;
        let candidate = cell
            .get("candidate_id")
            .and_then(Value::as_str)
            .ok_or(MetricError::new("invalid_candidate_identity"))?;
        require(!candidate.is_empty(), "invalid_candidate_identity")?;
        require(
            identities.insert((task_index, candidate)),
            "duplicate_candidate",
        )?;
        let status = cell
            .get("status")
            .and_then(Value::as_str)
            .ok_or(MetricError::new("invalid_cell_status"))?;
        require(
            matches!(
                status,
                "qualified"
                    | "unqualified"
                    | "failed"
                    | "crashed"
                    | "technical_failure"
                    | "incomplete"
                    | "timeout"
                    | "timed_out"
                    | "oom"
                    | "unsupported"
                    | "unevaluated"
                    | "unresolved"
            ),
            "invalid_cell_status",
        )?;
        let cap = count(cell.get("search_cap_bytes").unwrap_or(&Value::Null), 1)?;
        let bytes = match cell.get("artifact_bytes") {
            None | Some(Value::Null) => None,
            Some(v) => Some(count(v, 1)?),
        };
        require(bytes.is_none_or(|b| b <= cap), "artifact_exceeds_budget")?;
        require(
            status != "qualified" || bytes.is_some(),
            "missing_qualified_artifact",
        )?;
        trials.push(ArtifactTrial {
            task_index,
            search_cap_bytes: cap,
            artifact_bytes: bytes,
            qualified: status == "qualified",
        });
    }
    let mut rows = Vec::new();
    for cap in grid {
        let coverage = roster_coverage(roster.len(), &trials, cap, monotone)
            .map_err(|_| MetricError::new("invalid_frontier_evidence"))?;
        rows.push(json!({"budget_bytes": cap, "qualified_tasks": coverage.qualified_tasks, "admitted_tasks": coverage.admitted_tasks, "coverage": coverage.coverage}));
    }
    let minimum = roster_coverage(roster.len(), &trials, u64::MAX, monotone)
        .map_err(|_| MetricError::new("invalid_frontier_evidence"))?;
    Ok(json!({
        "protocol_version": 2, "status": "complete", "reason": Value::Null,
        "admitted_tasks": roster.len(), "minimum_qualified_bytes": minimum.minimum_bytes,
        "frontier": rows, "require_monotone": monotone,
        "non_monotone_tasks": (0..roster.len()).filter(|task| non_monotone(&trials, *task)).count(),
        "global_optimum_claimed": false,
    }))
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

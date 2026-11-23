//! Side inputs of `dope_fleet_macro`: scored checkpoints, the ASR guard state,
//! the corpus index and optional per-run exposure logs.

use super::scan::{num, Run};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Guard events kept for the chart annotations.
const MAX_GUARD_EVENTS: usize = 200;

fn jsonl(path: &Path) -> Vec<Value> {
    let Ok(file) = fs::File::open(path) else {
        return Vec::new();
    };
    BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| serde_json::from_str::<Value>(&l).ok())
        .filter(Value::is_object)
        .collect()
}

fn short(v: &Value, key: &str, max: usize) -> Value {
    match v.get(key) {
        Some(Value::String(s)) => Value::String(s.chars().take(max).collect()),
        Some(Value::Number(n)) => Value::Number(n.clone()),
        Some(Value::Array(a)) => Value::Array(
            a.iter()
                .filter_map(Value::as_str)
                .take(16)
                .map(|s| Value::String(s.chars().take(40).collect()))
                .collect(),
        ),
        _ => Value::Null,
    }
}

/// Time of the latest logged row at or before `step` (rows are step-ordered in time).
fn time_at_step(run: &Run, step: u64) -> Option<f64> {
    let idx = run.rows.partition_point(|r| r.step <= step);
    idx.checked_sub(1).map(|i| run.rows[i].t)
}

/// Scored checkpoints placed on the campaign clock, with a summary per variant.
pub fn checkpoints(path: Option<&Path>, runs: &[Run], noise: f64) -> Value {
    let Some(path) = path else {
        return Value::Null;
    };
    let by_name: BTreeMap<&str, &Run> = runs.iter().map(|r| (r.name.as_str(), r)).collect();
    let mut points = Vec::new();
    let mut by_variant: BTreeMap<String, Vec<(f64, String, u64)>> = BTreeMap::new();
    let (mut metric, mut panel) = (Value::Null, Value::Null);
    for row in jsonl(path) {
        let (Some(run), Some(step), Some(value)) = (
            row.get("run").and_then(Value::as_str),
            num(&row, "step"),
            num(&row, "value"),
        ) else {
            continue;
        };
        let step = step as u64;
        let variant = row
            .get("variant")
            .and_then(Value::as_str)
            .unwrap_or("full")
            .to_string();
        if metric.is_null() {
            metric = short(&row, "metric", 64);
            panel = short(&row, "panel", 160);
        }
        let found = by_name.get(run);
        let at = found.and_then(|r| time_at_step(r, step));
        points.push(json!({
            "run": run, "family": super::scan::family_of(run), "step": step,
            "variant": variant, "value": value, "at": at,
            "host": found.map(|r| r.host.clone()),
        }));
        by_variant
            .entry(variant)
            .or_default()
            .push((value, run.to_string(), step));
    }
    points.sort_by(|a, b| {
        let key = |v: &Value| v["at"].as_f64().unwrap_or(f64::MAX);
        key(a).total_cmp(&key(b))
    });
    let mut summary = Map::new();
    for (variant, vals) in by_variant {
        let values: Vec<f64> = vals.iter().map(|v| v.0).collect();
        let best = vals.iter().max_by(|a, b| a.0.total_cmp(&b.0));
        summary.insert(
            variant,
            json!({
                "n": vals.len(),
                "above_noise": values.iter().filter(|v| **v > noise).count(),
                "below_zero": values.iter().filter(|v| **v < 0.0).count(),
                "median": super::scan::median(&values),
                "best": best.map(|b| json!({"value": b.0, "run": b.1, "step": b.2})),
            }),
        );
    }
    json!({
        "metric": metric, "panel": panel, "noise": noise,
        "points": points, "summary": summary,
    })
}

/// ASR guard levels and its event log (read only).
pub fn guard(dir: Option<&Path>) -> Value {
    let Some(dir) = dir else {
        return Value::Null;
    };
    let state: Value = fs::read_to_string(dir.join("guard_state.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(Value::Null);
    let all = jsonl(&dir.join("guard_events.jsonl"));
    let keep = all.len().saturating_sub(MAX_GUARD_EVENTS);
    let events: Vec<Value> = all[keep..]
        .iter()
        .map(|e| {
            json!({
                "at": num(e, "t"), "utc": short(e, "utc", 32), "event": short(e, "event", 32),
                "job": short(e, "job", 16), "hosts": short(e, "hosts", 40),
                "level": short(e, "level", 8), "level_before": short(e, "level_before", 8),
                "reason": if e.get("reason").is_some() { short(e, "reason", 200) } else { short(e, "why", 200) },
            })
        })
        .collect();
    json!({
        "levels": state.get("levels").cloned().unwrap_or(Value::Null),
        "paused": state.get("paused").cloned().unwrap_or(Value::Null),
        "events_total": all.len(),
        "last_event": events.last().cloned(),
        "events": events,
    })
}

/// Training tables in the corpus index (`split == "train"`), and all tables.
pub fn corpus(path: Option<&Path>) -> (Option<u64>, Option<u64>) {
    let Some(path) = path else {
        return (None, None);
    };
    let rows = jsonl(path);
    if rows.is_empty() {
        return (None, None);
    }
    let train = rows
        .iter()
        .filter(|r| r.get("split").and_then(Value::as_str) == Some("train"))
        .count();
    (Some(train as u64), Some(rows.len() as u64))
}

#[derive(Default)]
struct Exposure {
    steps: u64,
    dataset_entries: u64,
    split_entries: u64,
    within_step_dataset_dups: u64,
    datasets: BTreeSet<String>,
    splits: BTreeSet<String>,
}

fn strings(v: &Value, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn read_exposure(path: &Path) -> Option<Exposure> {
    let rows = jsonl(path);
    if rows.is_empty() {
        return None;
    }
    let mut e = Exposure::default();
    for row in rows {
        let datasets = strings(&row, "datasets");
        let splits = strings(&row, "splits");
        e.steps += 1;
        e.dataset_entries += datasets.len() as u64;
        e.split_entries += splits.len() as u64;
        let distinct: BTreeSet<&String> = datasets.iter().collect();
        e.within_step_dataset_dups += (datasets.len() - distinct.len()) as u64;
        e.datasets.extend(datasets);
        e.splits.extend(splits);
    }
    Some(e)
}

/// Exact uniqueness from optional `exposure.jsonl` files: per run and fleet-wide.
pub fn exposure(runs: &[Run]) -> Value {
    let mut per_run = Vec::new();
    let (mut all_d, mut all_s) = (BTreeSet::new(), BTreeSet::new());
    let (mut d_entries, mut s_entries) = (0u64, 0u64);
    for run in runs {
        let Some(e) = read_exposure(&run.dir.join("exposure.jsonl")) else {
            continue;
        };
        per_run.push(json!({
            "run": run.name, "host": run.host, "steps": e.steps,
            "dataset_entries": e.dataset_entries, "distinct_datasets": e.datasets.len(),
            "split_entries": e.split_entries, "distinct_splits": e.splits.len(),
            "split_repeats": e.split_entries - e.splits.len() as u64,
            "dataset_repeats": e.dataset_entries - e.datasets.len() as u64,
            "within_step_dataset_dups": e.within_step_dataset_dups,
        }));
        d_entries += e.dataset_entries;
        s_entries += e.split_entries;
        all_d.extend(e.datasets);
        all_s.extend(e.splits);
    }
    if per_run.is_empty() {
        return Value::Null;
    }
    json!({
        "runs": per_run,
        "fleet": {
            "runs": per_run.len(),
            "dataset_entries": d_entries, "distinct_datasets": all_d.len(),
            "split_entries": s_entries, "distinct_splits": all_s.len(),
            "split_repeats": s_entries - all_s.len() as u64,
            "dataset_repeats": d_entries - all_d.len() as u64,
        },
    })
}

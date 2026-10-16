//! Family means, worst-family gate, and paired cluster bootstrap (T-031).
//!
//! Resamples `source_family -> dataset -> seed`, preserving every auditor
//! family in one paired draw. Input rows are seed-level paired retention
//! observations.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

use super::{MetricError, MetricResult, count, failure_with_status, mean, number, require, unit};

/// Explicit portable 64-bit XorShift RNG (`13, 7, 17`); zero seed maps to `1`.
#[derive(Clone, Debug)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    pub fn index(&mut self, length: usize) -> usize {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (self.state % (length as u64)) as usize
    }
}

fn parse_names(
    value: Option<&Value>,
    invalid_err: &'static str,
    dup_err: &'static str,
) -> MetricResult<Vec<String>> {
    let arr = value
        .and_then(Value::as_array)
        .ok_or(MetricError::new(invalid_err))?;
    require(!arr.is_empty(), invalid_err)?;
    let mut names = Vec::with_capacity(arr.len());
    let mut seen = BTreeSet::new();
    for item in arr {
        let s = item.as_str().ok_or(MetricError::new(invalid_err))?;
        require(!s.is_empty(), invalid_err)?;
        require(seen.insert(s.to_owned()), dup_err)?;
        names.push(s.to_owned());
    }
    names.sort();
    Ok(names)
}

fn parse_required_auditors(
    value: Option<&Value>,
    families: &[String],
) -> MetricResult<Option<BTreeMap<String, BTreeSet<String>>>> {
    let val = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(v) => v,
    };
    let obj = val
        .as_object()
        .ok_or(MetricError::new("invalid_auditors"))?;
    require(
        obj.len() == families.len() && families.iter().all(|f| obj.contains_key(f)),
        "invalid_auditors",
    )?;
    let mut out = BTreeMap::new();
    for family in families {
        let names = parse_names(
            obj.get(family),
            "invalid_auditors",
            "duplicate_auditor",
        )?;
        out.insert(family.clone(), names.into_iter().collect());
    }
    Ok(Some(out))
}

fn parse_required_seeds(value: Option<&Value>) -> MetricResult<Option<BTreeSet<u64>>> {
    let val = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(v) => v,
    };
    let arr = val.as_array().ok_or(MetricError::new("invalid_seeds"))?;
    require(!arr.is_empty(), "invalid_seeds")?;
    let mut out = BTreeSet::new();
    for item in arr {
        let s = count(item, 0)?;
        require(out.insert(s), "duplicate_seed")?;
    }
    Ok(Some(out))
}

type CellMap = BTreeMap<String, f64>;
type SeedMap = BTreeMap<u64, CellMap>;
type DatasetMap = BTreeMap<String, SeedMap>;
type SourceTree = BTreeMap<String, DatasetMap>;

pub fn evaluate(data: &Value) -> Value {
    match evaluate_inner(data) {
        Ok(val) => val,
        Err(err) => failure_with_status(err.reason, err.status),
    }
}

fn evaluate_inner(data: &Value) -> MetricResult<Value> {
    let obj = data.as_object().ok_or(MetricError::new("invalid_input"))?;
    let families = parse_names(
        obj.get("required_families"),
        "invalid_families",
        "duplicate_family",
    )?;
    let family_set: BTreeSet<&str> = families.iter().map(String::as_str).collect();
    let declared_auditors = parse_required_auditors(obj.get("required_auditors"), &families)?;
    let declared_seeds = parse_required_seeds(obj.get("required_seeds"))?;

    let rows_val = match obj.get("rows") {
        None | Some(Value::Null) => {
            return Err(MetricError::with_status(
                "missing_family_or_cell",
                "incomplete",
            ));
        }
        Some(v) => v,
    };
    let rows = rows_val
        .as_array()
        .ok_or(MetricError::new("invalid_rows"))?;
    if rows.is_empty() {
        return Err(MetricError::with_status(
            "missing_family_or_cell",
            "incomplete",
        ));
    }

    type FamilyAuditorCells = BTreeMap<String, BTreeMap<String, f64>>;
    type RawClusterTree = BTreeMap<String, BTreeMap<String, BTreeMap<u64, FamilyAuditorCells>>>;
    let mut raw_tree: RawClusterTree = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut observed_auditors: BTreeMap<String, BTreeSet<String>> = families
        .iter()
        .map(|f| (f.clone(), BTreeSet::new()))
        .collect();

    for row_val in rows {
        let row = row_val
            .as_object()
            .ok_or(MetricError::new("invalid_rows"))?;
        let get_str = |key: &str| -> MetricResult<String> {
            let s = row
                .get(key)
                .and_then(Value::as_str)
                .ok_or(MetricError::new("invalid_cluster"))?;
            require(!s.is_empty(), "invalid_cluster")?;
            Ok(s.to_owned())
        };
        let source_family = get_str("source_family")?;
        let dataset = get_str("dataset")?;
        let family = get_str("family")?;
        let auditor = get_str("auditor")?;
        require(family_set.contains(family.as_str()), "unexpected_family")?;
        if let Some(ref decl_aud) = declared_auditors {
            require(
                decl_aud[&family].contains(&auditor),
                "unexpected_auditor",
            )?;
        }
        let seed = count(row.get("seed").ok_or(MetricError::new("invalid_count"))?, 0)?;
        if let Some(ref decl_seeds) = declared_seeds {
            require(decl_seeds.contains(&seed), "unexpected_seed")?;
        }
        let obs_key = (
            source_family.clone(),
            dataset.clone(),
            seed,
            family.clone(),
            auditor.clone(),
        );
        require(seen.insert(obs_key), "duplicate_observation")?;
        if let Some(st_val) = row.get("status")
            && !st_val.is_null()
        {
            let st = st_val
                .as_str()
                .ok_or(MetricError::new("invalid_row_status"))?;
            match st {
                "failed" => return Err(MetricError::new("row_failed")),
                "incomplete" | "uninformative" => {
                    return Err(MetricError::with_status(
                        "missing_family_or_cell",
                        "incomplete",
                    ));
                }
                "complete" | "informative" => {}
                _ => return Err(MetricError::new("invalid_row_status")),
            }
        }
        if row.get("informative").and_then(Value::as_bool) == Some(false) {
            return Err(MetricError::with_status(
                "missing_family_or_cell",
                "incomplete",
            ));
        }
        let transfer_val = match row.get("transfer") {
            None | Some(Value::Null) => {
                return Err(MetricError::with_status(
                    "missing_family_or_cell",
                    "incomplete",
                ));
            }
            Some(v) => number(v)?.min(1.0),
        };
        observed_auditors
            .get_mut(&family)
            .unwrap()
            .insert(auditor.clone());
        raw_tree
            .entry(source_family)
            .or_default()
            .entry(dataset)
            .or_default()
            .entry(seed)
            .or_default()
            .entry(family)
            .or_default()
            .insert(auditor, transfer_val);
    }

    let required_auditors = declared_auditors.unwrap_or(observed_auditors);
    if families.iter().any(|f| required_auditors[f].is_empty()) {
        return Err(MetricError::with_status(
            "missing_family_or_cell",
            "incomplete",
        ));
    }

    let mut tree: SourceTree = BTreeMap::new();
    for (source, datasets) in raw_tree {
        let mut ds_map = BTreeMap::new();
        for (dataset, seeds) in datasets {
            let observed_seed_set: BTreeSet<u64> = seeds.keys().copied().collect();
            let expected_seeds = declared_seeds.as_ref().unwrap_or(&observed_seed_set);
            if &observed_seed_set != expected_seeds {
                return Err(MetricError::with_status(
                    "missing_family_or_cell",
                    "incomplete",
                ));
            }
            let mut seed_map = BTreeMap::new();
            for (seed, fam_auditors) in seeds {
                if fam_auditors.len() != families.len()
                    || !families.iter().all(|f| fam_auditors.contains_key(f))
                {
                    return Err(MetricError::with_status(
                        "missing_family_or_cell",
                        "incomplete",
                    ));
                }
                let mut cell_means = BTreeMap::new();
                for family in &families {
                    let auditors = &fam_auditors[family];
                    let aud_keys: BTreeSet<String> = auditors.keys().cloned().collect();
                    if aud_keys != required_auditors[family] {
                        return Err(MetricError::with_status(
                            "missing_family_or_cell",
                            "incomplete",
                        ));
                    }
                    let vals: Vec<f64> = auditors.values().copied().collect();
                    cell_means.insert(family.clone(), mean(&vals)?);
                }
                seed_map.insert(seed, cell_means);
            }
            ds_map.insert(dataset, seed_map);
        }
        tree.insert(source, ds_map);
    }

    let sources: Vec<&String> = tree.keys().collect();
    let estimate = |mut rng: Option<&mut XorShift64>| -> MetricResult<BTreeMap<String, f64>> {
        let mut source_estimates: Vec<BTreeMap<String, f64>> = Vec::with_capacity(sources.len());
        for i in 0..sources.len() {
            let s_idx = match rng.as_deref_mut() {
                Some(r) => r.index(sources.len()),
                None => i,
            };
            let datasets = &tree[sources[s_idx]];
            let ds_names: Vec<&String> = datasets.keys().collect();
            let mut ds_estimates: Vec<BTreeMap<String, f64>> = Vec::with_capacity(ds_names.len());
            for j in 0..ds_names.len() {
                let d_idx = match rng.as_deref_mut() {
                    Some(r) => r.index(ds_names.len()),
                    None => j,
                };
                let seeds = &datasets[ds_names[d_idx]];
                let seed_ids: Vec<&u64> = seeds.keys().collect();
                let mut picked_cells: Vec<&CellMap> = Vec::with_capacity(seed_ids.len());
                for k in 0..seed_ids.len() {
                    let k_idx = match rng.as_deref_mut() {
                        Some(r) => r.index(seed_ids.len()),
                        None => k,
                    };
                    picked_cells.push(&seeds[seed_ids[k_idx]]);
                }
                let mut ds_fam = BTreeMap::new();
                for f in &families {
                    let vals: Vec<f64> = picked_cells.iter().map(|c| c[f]).collect();
                    ds_fam.insert(f.clone(), mean(&vals)?);
                }
                ds_estimates.push(ds_fam);
            }
            let mut src_fam = BTreeMap::new();
            for f in &families {
                let vals: Vec<f64> = ds_estimates.iter().map(|d| d[f]).collect();
                src_fam.insert(f.clone(), mean(&vals)?);
            }
            source_estimates.push(src_fam);
        }
        let mut out = BTreeMap::new();
        for f in &families {
            let vals: Vec<f64> = source_estimates.iter().map(|s| s[f]).collect();
            out.insert(f.clone(), mean(&vals)?);
        }
        Ok(out)
    };

    let points = estimate(None)?;
    let replicates = match obj.get("bootstrap_replicates") {
        Some(v) if !v.is_null() => count(v, 1)?,
        _ => 1000,
    };
    require(replicates <= 100_000, "bootstrap_budget_exceeded")?;
    let alpha = match obj.get("alpha") {
        Some(v) if !v.is_null() => unit(v)?,
        _ => 0.05,
    };
    require(alpha > 0.0 && alpha < 0.5, "invalid_alpha")?;
    let seed = match obj.get("bootstrap_seed") {
        Some(v) if !v.is_null() => count(v, 0)?,
        _ => 1729,
    };
    let mut rng = XorShift64::new(seed);
    let mut samples = Vec::with_capacity(replicates as usize);
    for _ in 0..replicates {
        let draw = estimate(Some(&mut rng))?;
        let worst = draw.values().copied().fold(f64::INFINITY, f64::min);
        samples.push(worst);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let q_idx = (alpha * ((replicates - 1) as f64)).floor() as usize;
    let lcb = samples[q_idx];

    let point_vals: Vec<f64> = families.iter().map(|f| points[f]).collect();
    let robust_transfer = point_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let mean_transfer = mean(&point_vals)?;
    let mut family_means_map = Map::new();
    for (k, v) in points {
        family_means_map.insert(k, json!(v));
    }

    Ok(json!({
        "status": "complete",
        "reason": Value::Null,
        "family_means": Value::Object(family_means_map),
        "robust_transfer": robust_transfer,
        "mean_transfer": mean_transfer,
        "lcb": lcb,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn adding_worse_family_never_increases_robust_transfer(
            tree_a in 0.1f64..1.0f64,
            linear_a in 0.1f64..1.0f64,
            worse_delta in 0.0f64..0.5f64,
        ) {
            let worst_base = tree_a.min(linear_a);
            let neural_a = (worst_base - worse_delta).max(0.0);
            let base = evaluate(&json!({
                "required_families": ["tree", "linear"],
                "rows": [
                    {"source_family": "s1", "dataset": "d1", "seed": 101, "family": "tree", "auditor": "cb", "transfer": tree_a},
                    {"source_family": "s1", "dataset": "d1", "seed": 101, "family": "linear", "auditor": "ridge", "transfer": linear_a}
                ],
                "bootstrap_replicates": 50,
                "bootstrap_seed": 1729,
                "alpha": 0.05
            }));
            let extended = evaluate(&json!({
                "required_families": ["tree", "linear", "neural"],
                "rows": [
                    {"source_family": "s1", "dataset": "d1", "seed": 101, "family": "tree", "auditor": "cb", "transfer": tree_a},
                    {"source_family": "s1", "dataset": "d1", "seed": 101, "family": "linear", "auditor": "ridge", "transfer": linear_a},
                    {"source_family": "s1", "dataset": "d1", "seed": 101, "family": "neural", "auditor": "mlp", "transfer": neural_a}
                ],
                "bootstrap_replicates": 50,
                "bootstrap_seed": 1729,
                "alpha": 0.05
            }));
            prop_assert!(extended["robust_transfer"].as_f64().unwrap() <= base["robust_transfer"].as_f64().unwrap() + 1e-12);
        }
    }
}

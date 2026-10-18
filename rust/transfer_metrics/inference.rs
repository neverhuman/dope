//! Development crossed source/task inference; never scientific qualification.
//! Product source/task draws retain source:task shocks; seed namespaces cross
//! cells, while fit/sample/learner and paired entity draws stay inside cells.
use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::metrics::aggregate::XorShift64;
use crate::metrics::{MetricError, MetricResult, mean, require};

mod design;
pub use design::{CrossedDesign, EvaluationRun, InferenceMargin, LossEvidence, task_substream};

struct Cell<'a> {
    fits: Vec<&'a str>,
    samples: Vec<&'a str>,
    learners: Vec<&'a str>,
    namespace: Option<&'a str>,
    seeds: Vec<&'a str>,
    clusters: Vec<&'a str>,
    pairs: &'a LossEvidence,
    values: BTreeMap<(&'a str, &'a str, &'a str), Vec<f64>>,
}

pub struct BootstrapEffects {
    pub status: &'static str,
    pub estimates: BTreeMap<String, f64>,
    pub samples: BTreeMap<String, Vec<f64>>,
    pub draws: usize,
    pub family_effects: BTreeMap<String, BTreeMap<String, f64>>,
}

fn refused(status: &'static str, draws: usize) -> BootstrapEffects {
    BootstrapEffects {
        status,
        estimates: BTreeMap::new(),
        samples: BTreeMap::new(),
        draws,
        family_effects: BTreeMap::new(),
    }
}

fn picks<'a>(ids: &[&'a str], rng: &mut XorShift64) -> Vec<&'a str> {
    ids.iter().map(|_| ids[rng.index(ids.len())]).collect()
}

/// Default production-design draw count; fixtures may request a smaller budget.
pub const DEFAULT_DRAWS: usize = 10_000;

pub fn bootstrap_effects(
    design: &CrossedDesign,
    endpoint: &str,
    estimand: &str,
    draws: usize,
    seed: u64,
) -> MetricResult<BootstrapEffects> {
    design.validate()?;
    require((99..=100_000).contains(&draws), "invalid_bootstrap_budget")?;
    require(
        matches!(endpoint, "z" | "gap" | "normalized_gap")
            && matches!(estimand, "method_new_source" | "fixed_artifact"),
        "invalid_estimand",
    )?;
    if design.runs.iter().any(|r| r.pairs.is_none()) {
        return Ok(refused("incomplete", draws));
    }
    let sources: Vec<_> = design
        .runs
        .iter()
        .map(|r| r.family_id.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let tasks: Vec<_> = design
        .runs
        .iter()
        .map(|r| r.task_axis_id.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let methods: Vec<_> = design
        .required_methods
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut cells = BTreeMap::new();
    let mut namespaces = BTreeMap::new();
    let mut query_sets = BTreeMap::new();
    let mut paired_axes = BTreeMap::new();
    for &source in &sources {
        for &task in &tasks {
            for &method in &methods {
                let runs: Vec<_> = design
                    .runs
                    .iter()
                    .filter(|r| {
                        r.family_id == source && r.task_axis_id == task && r.method_id == method
                    })
                    .collect();
                if runs.is_empty() {
                    return Ok(refused("incomplete", draws));
                }
                let first = runs[0];
                let pairs = first.pairs.as_ref().unwrap();
                let fits: Vec<_> = runs
                    .iter()
                    .map(|r| r.fit_id.as_str())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let samples: Vec<_> = runs
                    .iter()
                    .map(|r| r.sample_id.as_str())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let learners: Vec<_> = runs
                    .iter()
                    .map(|r| r.learner_seed_id.as_str())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let axes = (samples.clone(), learners.clone());
                if paired_axes
                    .insert((source, task), axes.clone())
                    .is_some_and(|v| v != axes)
                {
                    return Ok(refused("incomplete", draws));
                }
                if runs.len() != fits.len() * samples.len() * learners.len() {
                    return Ok(refused("incomplete", draws));
                }
                let signature = (&pairs.query_ids, &pairs.cluster_ids);
                require(
                    query_sets
                        .insert((source, task), signature)
                        .is_none_or(|v| v == signature),
                    "unpaired_queries",
                )?;
                let mut values = BTreeMap::new();
                for r in &runs {
                    let p = r.pairs.as_ref().unwrap();
                    require(
                        (&p.query_ids, &p.cluster_ids) == signature,
                        "unpaired_queries",
                    )?;
                    values.insert(
                        (
                            r.fit_id.as_str(),
                            r.sample_id.as_str(),
                            r.learner_seed_id.as_str(),
                        ),
                        r.values(endpoint)?,
                    );
                }
                let namespace = first.global_seed_namespace.as_deref();
                let seeds: Vec<_> = fits
                    .iter()
                    .map(|f| {
                        runs.iter()
                            .find(|r| &r.fit_id == f)
                            .unwrap()
                            .global_seed_id
                            .as_deref()
                            .unwrap_or("")
                    })
                    .collect();
                if let Some(ns) = namespace {
                    let ordered: Vec<_> = seeds
                        .iter()
                        .copied()
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect();
                    require(ordered.len() == seeds.len(), "duplicate_fit_seed")?;
                    if namespaces
                        .insert(ns, ordered.clone())
                        .is_some_and(|v| v != ordered)
                    {
                        return Ok(refused("incomplete", draws));
                    }
                }
                require(
                    estimand != "fixed_artifact"
                        || (fits.len(), samples.len(), learners.len()) == (1, 1, 1),
                    "invalid_fixed_artifact_design",
                )?;
                if estimand == "method_new_source" && fits.len() < 2 {
                    return Ok(refused("insufficient_fits", draws));
                }
                let clusters: Vec<_> = pairs
                    .cluster_ids
                    .iter()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                if clusters.len() < 2 {
                    return Ok(refused("insufficient_query_clusters", draws));
                }
                cells.insert(
                    (source, task, method),
                    Cell {
                        fits,
                        samples,
                        learners,
                        namespace,
                        seeds,
                        clusters,
                        pairs,
                        values,
                    },
                );
            }
        }
    }
    if estimand == "method_new_source" && sources.len() < 2 {
        return Ok(refused("insufficient_families", draws));
    }
    if estimand == "method_new_source" && tasks.len() < 2 {
        return Ok(refused("insufficient_tasks", draws));
    }
    let mut output = BootstrapEffects {
        status: "complete",
        draws,
        estimates: BTreeMap::new(),
        family_effects: BTreeMap::new(),
        samples: methods
            .iter()
            .map(|m| (m.to_string(), Vec::with_capacity(draws)))
            .collect(),
    };
    for &method in &methods {
        let source_values: MetricResult<Vec<_>> = sources
            .iter()
            .map(|s| {
                let values: MetricResult<Vec<_>> = tasks
                    .iter()
                    .map(|t| {
                        let cell = &cells[&(*s, *t, method)];
                        let values: MetricResult<Vec<_>> =
                            cell.values.values().map(|v| mean(v)).collect();
                        mean(&values?)
                    })
                    .collect();
                mean(&values?)
            })
            .collect();
        let values = source_values?;
        output.estimates.insert(method.to_string(), mean(&values)?);
        output.family_effects.insert(
            method.to_string(),
            sources.iter().map(|s| s.to_string()).zip(values).collect(),
        );
    }
    let mut rng = XorShift64::new(seed);
    for _ in 0..draws {
        let global: BTreeMap<_, _> = if estimand == "method_new_source" {
            namespaces
                .iter()
                .map(|(ns, ids)| (*ns, picks(ids, &mut rng)))
                .collect()
        } else {
            BTreeMap::new()
        };
        let mut task_values = BTreeMap::new();
        for &source in &sources {
            for &task in &tasks {
                let first = &cells[&(source, task, methods[0])];
                let selected = picks(&first.clusters, &mut rng);
                let weights: Vec<_> = first
                    .pairs
                    .cluster_ids
                    .iter()
                    .map(|c| selected.iter().filter(|s| **s == c).count() as f64)
                    .collect();
                let total: f64 = weights.iter().sum();
                let mut sample_draws = BTreeMap::new();
                let mut learner_draws = BTreeMap::new();
                for &method in &methods {
                    let cell = &cells[&(source, task, method)];
                    let fits = if estimand == "fixed_artifact" {
                        cell.fits.clone()
                    } else if let Some(ns) = cell.namespace {
                        global[ns]
                            .iter()
                            .map(|s| cell.fits[cell.seeds.iter().position(|v| v == s).unwrap()])
                            .collect()
                    } else {
                        picks(&cell.fits, &mut rng)
                    };
                    let mut fit_means = Vec::new();
                    for (fi, fit) in fits.into_iter().enumerate() {
                        let samples = sample_draws.entry(fi).or_insert_with(|| {
                            if estimand == "fixed_artifact" {
                                cell.samples.clone()
                            } else {
                                picks(&cell.samples, &mut rng)
                            }
                        });
                        let mut sample_means = Vec::new();
                        for (si, &sample) in samples.iter().enumerate() {
                            let learners = learner_draws.entry((fi, si)).or_insert_with(|| {
                                if estimand == "fixed_artifact" {
                                    cell.learners.clone()
                                } else {
                                    picks(&cell.learners, &mut rng)
                                }
                            });
                            let values: Vec<_> = learners
                                .iter()
                                .map(|l| {
                                    cell.values[&(fit, sample, *l)]
                                        .iter()
                                        .zip(&weights)
                                        .map(|(v, w)| v * w)
                                        .sum::<f64>()
                                        / total
                                })
                                .collect();
                            sample_means.push(mean(&values)?);
                        }
                        fit_means.push(mean(&sample_means)?);
                    }
                    task_values.insert((source, task, method), mean(&fit_means)?);
                }
            }
        }
        let selected_sources = if estimand == "fixed_artifact" {
            sources.clone()
        } else {
            picks(&sources, &mut rng)
        };
        let selected_tasks = if estimand == "fixed_artifact" {
            tasks.clone()
        } else {
            picks(&tasks, &mut rng)
        };
        for &method in &methods {
            let values: MetricResult<Vec<_>> = selected_sources
                .iter()
                .map(|s| {
                    mean(
                        &selected_tasks
                            .iter()
                            .map(|t| task_values[&(*s, *t, method)])
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            output
                .samples
                .get_mut(method)
                .unwrap()
                .push(mean(&values?)?);
        }
    }
    Ok(output)
}

mod simultaneous;
pub use simultaneous::simultaneous_ucb;

/// Robust sensitivity has a distinct estimand and supplies no replacement bound.
pub fn heavy_tail_sensitivity(boot: &BootstrapEffects, blocks: usize) -> MetricResult<Value> {
    require(blocks >= 3 && blocks % 2 == 1, "invalid_robust_blocks")?;
    let mut estimates = BTreeMap::new();
    for (method, effects) in &boot.family_effects {
        if boot.status != "complete" || effects.len() < blocks {
            return Ok(
                json!({"status": "insufficient_families", "qualification": false, "bounds": null}),
            );
        }
        let values: Vec<_> = effects.values().copied().collect();
        let mut means = Vec::new();
        for i in 0..blocks {
            means.push(mean(
                &values
                    .iter()
                    .skip(i)
                    .step_by(blocks)
                    .copied()
                    .collect::<Vec<_>>(),
            )?);
        }
        means.sort_by(f64::total_cmp);
        estimates.insert(method, means[blocks / 2]);
    }
    require(!estimates.is_empty(), "missing_robust_evidence")?;
    Ok(
        json!({"status": "sensitivity_only", "estimand": "median_of_source_block_means", "estimates": estimates, "blocks": blocks, "qualification": false, "bounds": null, "primary_losses_clipped": false}),
    )
}

pub fn evaluate(data: &Value) -> Value {
    let inner = || -> MetricResult<Value> {
        require(
            data["inference_version"].as_u64() == Some(2),
            "invalid_inference_version",
        )?;
        let design: CrossedDesign = serde_json::from_value(data.clone())
            .map_err(|_| MetricError::new("invalid_crossed_design"))?;
        let draws = data
            .get("bootstrap_draws")
            .map_or(Ok(DEFAULT_DRAWS as u64), |v| {
                v.as_u64()
                    .ok_or(MetricError::new("invalid_bootstrap_budget"))
            })?;
        let seed = data.get("bootstrap_seed").map_or(Ok(0), |v| {
            v.as_u64().ok_or(MetricError::new("invalid_bootstrap_seed"))
        })?;
        let boot = bootstrap_effects(
            &design,
            data.get("endpoint")
                .map_or(Some("z"), Value::as_str)
                .ok_or(MetricError::new("invalid_endpoint"))?,
            data.get("estimand")
                .map_or(Some("method_new_source"), Value::as_str)
                .ok_or(MetricError::new("invalid_estimand"))?,
            usize::try_from(draws).map_err(|_| MetricError::new("invalid_bootstrap_budget"))?,
            seed,
        )?;
        let alpha = data
            .get("alpha")
            .map_or(Some(0.05), Value::as_f64)
            .ok_or(MetricError::new("invalid_inference_alpha"))?;
        let mut result = simultaneous_ucb(&boot, alpha)?;
        result["qualification"] = json!(false);
        result["calibration"] = json!("unmeasured");
        result["campaign"] = json!("unmeasured");
        Ok(result)
    };
    match inner() {
        Ok(v) => v,
        Err(_) => json!({"status": "failed", "reason": "invalid_crossed_design"}),
    }
}

#[cfg(test)]
#[path = "inference_tests.rs"]
mod tests;

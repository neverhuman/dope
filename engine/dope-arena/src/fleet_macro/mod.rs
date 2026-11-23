//! `dope.fleet-macro/v1`: one campaign-wide view over every training run of
//! the fleet: throughput, tables trained, learning, checkpoints, topology and
//! data uniqueness, bucketed on the wall clock. Read only; never writes into
//! a run directory.

pub mod extras;
pub mod scan;

use scan::{increments, median, quantile, Run};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub const SCHEMA: &str = "dope.fleet-macro/v1";
/// Rows used for a run's start and end loss.
const LOSS_EDGE_ROWS: usize = 20;
/// Idle buckets after the last row are drawn up to `now` only within this span.
const MAX_IDLE_TAIL_S: f64 = 7.0 * 86_400.0;
/// Synthetic-prior share of the V7 sampler (`s3_v7_batch_ops.py`, 20%), used
/// only when a V7 run does not log `n_synth_prior`.
const V7_RECIPE_SYNTH_SHARE: f64 = 0.20;
/// Window of the sustained "now" rate: tables completed in the last 5 minutes.
const RECENT_S: f64 = 300.0;
/// Idle spans are filled with explicit zero buckets only below this count.
const MAX_BUCKETS: i64 = 100_000;

#[derive(Clone, Debug)]
pub struct Config {
    pub roots: Vec<PathBuf>,
    pub checkpoint_kpi: Option<PathBuf>,
    pub corpus_index: Option<PathBuf>,
    pub guard_dir: Option<PathBuf>,
    pub bucket_s: u64,
    pub active_window_s: f64,
    pub kpi_noise: f64,
    pub now: f64,
}

/// Per-run roll-up kept beside its JSON form for the fleet totals.
struct RunStats {
    tables: f64,
    synth_est: Option<f64>,
    synth_basis: &'static str,
    active: bool,
}

#[derive(Default)]
struct Bucket {
    tables: f64,
    by_host: BTreeMap<String, f64>,
    runs: BTreeSet<usize>,
    hosts: BTreeSet<String>,
    rel_loss: Vec<f64>,
    ckpts: u64,
}

/// `x` rounded to `places` decimals, never a negative zero.
fn round(x: f64, places: i32) -> f64 {
    let m = 10f64.powi(places);
    (x * m).round() / m + 0.0
}

fn edge_loss(run: &Run, tail: bool) -> Option<f64> {
    let losses: Vec<f64> = run.rows.iter().filter_map(|r| r.loss).collect();
    // Start and end windows never overlap: at most half the rows each.
    let n = losses
        .len()
        .min(LOSS_EDGE_ROWS)
        .min((losses.len() / 2).max(1));
    if n == 0 {
        return None;
    }
    let slice = if tail {
        &losses[losses.len() - n..]
    } else {
        &losses[..n]
    };
    median(slice)
}

/// Synthetic-prior episodes of a run: logged per step, else the V7 recipe share.
fn synth_estimate(run: &Run, tables: f64) -> (Option<f64>, &'static str) {
    let (mut synth, mut eps) = (0.0, 0.0);
    for r in &run.rows {
        if let (Some(s), Some(e)) = (r.synth, r.eps) {
            synth += s;
            eps += e;
        }
    }
    if eps > 0.0 {
        return (Some(tables * synth / eps), "logged");
    }
    if run.arch.contains("v7") {
        return (Some(tables * V7_RECIPE_SYNTH_SHARE), "recipe_estimate");
    }
    (None, "not_logged")
}

fn run_json(run: &Run, cfg: &Config, stats: &RunStats) -> Value {
    let first = run.rows.first().map(|r| r.t);
    let last = run.rows.last().map(|r| r.t);
    let span = match (first, last) {
        (Some(a), Some(b)) if b > a => Some(b - a),
        _ => None,
    };
    let (ls, le) = (edge_loss(run, false), edge_loss(run, true));
    let change = match (ls, le) {
        (Some(a), Some(b)) if a.abs() > 1e-12 => Some(round((b / a - 1.0) * 100.0, 2)),
        _ => None,
    };
    let real = stats.tables - stats.synth_est.unwrap_or(0.0);
    json!({
        "run": run.name, "host": run.host, "family": run.family, "arch": run.arch,
        "recipe": run.recipe, "world_size": run.world_size, "warm_start": run.warm_start,
        "sync_mode": run.sync_mode, "sync_rounds": run.sync_rounds,
        "first_at": first, "last_at": last, "rows": run.rows.len(), "bad_rows": run.bad_rows,
        "last_step": run.rows.last().map(|r| r.step),
        "tables": stats.tables.round(),
        "synth_est": stats.synth_est.map(f64::round), "synth_basis": stats.synth_basis,
        "real_est": real.round(),
        "tables_per_s_last": run.rows.last().and_then(|r| r.tps),
        "tables_per_s_mean": span.map(|s| round(stats.tables / s, 2)),
        "loss_start": ls.map(|x| round(x, 5)), "loss_end": le.map(|x| round(x, 5)),
        "loss_change_pct": change,
        "checkpoints_saved": run.ckpt_times.len(),
        "active": stats.active,
        "active_window_s": cfg.active_window_s,
    })
}

/// Checkpoint save times within an hour of the run's logged span (file times
/// outside it, e.g. from a copy that did not keep them, are not placed).
fn placed_checkpoints(run: &Run) -> Vec<f64> {
    let (Some(first), Some(last)) = (run.rows.first(), run.rows.last()) else {
        return Vec::new();
    };
    run.ckpt_times
        .iter()
        .copied()
        .filter(|t| *t >= first.t - 3600.0 && *t <= last.t + 3600.0)
        .collect()
}

fn bucket_of(t: f64, size: u64) -> i64 {
    (t / size as f64).floor() as i64 * size as i64
}

/// Fill buckets with throughput, activity, relative loss and checkpoint saves.
fn fill_buckets(runs: &[Run], cfg: &Config) -> BTreeMap<i64, Bucket> {
    let size = cfg.bucket_s.max(1);
    let mut buckets: BTreeMap<i64, Bucket> = BTreeMap::new();
    for (i, run) in runs.iter().enumerate() {
        for (t, inc) in increments(&run.rows) {
            let b = buckets.entry(bucket_of(t, size)).or_default();
            b.tables += inc;
            *b.by_host.entry(run.host.clone()).or_default() += inc;
            b.runs.insert(i);
            b.hosts.insert(run.host.clone());
        }
        if let Some(base) = edge_loss(run, false).filter(|b| *b > 1e-12) {
            let mut per: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
            for r in &run.rows {
                if let Some(l) = r.loss {
                    let e = per.entry(bucket_of(r.t, size)).or_default();
                    e.0 += l / base;
                    e.1 += 1.0;
                }
            }
            for (k, (sum, n)) in per {
                buckets.entry(k).or_default().rel_loss.push(sum / n);
            }
        }
        for t in placed_checkpoints(run) {
            buckets.entry(bucket_of(t, size)).or_default().ckpts += 1;
        }
    }
    // Explicit zero buckets for idle spans, up to `now` when it is near the data.
    if let (Some(&lo), Some(&hi)) = (buckets.keys().next(), buckets.keys().next_back()) {
        let end = if cfg.now - hi as f64 <= MAX_IDLE_TAIL_S {
            bucket_of(cfg.now, size).max(hi)
        } else {
            hi
        };
        let mut k = if (end - lo) / size as i64 > MAX_BUCKETS {
            end
        } else {
            lo
        };
        while k <= end {
            buckets.entry(k).or_default();
            k += size as i64;
        }
    }
    buckets
}

fn series(buckets: &BTreeMap<i64, Bucket>, size: u64) -> (Vec<Value>, Option<(f64, i64)>) {
    let (mut cum_tables, mut cum_ckpts) = (0.0, 0u64);
    let mut peak: Option<(f64, i64)> = None;
    let mut out = Vec::with_capacity(buckets.len());
    for (&t, b) in buckets {
        cum_tables += b.tables;
        cum_ckpts += b.ckpts;
        let rate = b.tables / size as f64;
        if rate > peak.map_or(0.0, |p| p.0) {
            peak = Some((rate, t));
        }
        let by_host: Map<String, Value> = b
            .by_host
            .iter()
            .map(|(h, v)| (h.clone(), json!(round(v / size as f64, 3))))
            .collect();
        out.push(json!({
            "t": t, "tables_per_s": round(rate, 3), "by_host": by_host,
            "active_runs": b.runs.len(), "active_hosts": b.hosts.len(),
            "cum_tables": cum_tables.round(), "checkpoints_saved": b.ckpts,
            "cum_checkpoints": cum_ckpts,
            "loss_rel_median": median(&b.rel_loss).map(|x| round(x, 4)),
            "loss_rel_p25": quantile(&b.rel_loss, 0.25).map(|x| round(x, 4)),
            "loss_rel_p75": quantile(&b.rel_loss, 0.75).map(|x| round(x, 4)),
            "loss_runs": b.rel_loss.len(),
        }));
    }
    (out, peak)
}

/// A run that exchanges weights with other runs through a sync directory
/// (DiLoCo / local SGD), as opposed to an independent or data-parallel run.
pub fn is_synced(run: &Run) -> bool {
    !matches!(
        run.sync_mode.as_str(),
        "independent" | "independent_ws1" | "ddp" | "ddp_gloo"
    )
}

fn sync_groups(runs: &[Run]) -> Vec<Value> {
    type Group<'a> = (String, BTreeSet<&'a str>, BTreeSet<&'a str>, u64);
    let mut groups: BTreeMap<String, Group> = BTreeMap::new();
    for r in runs.iter().filter(|r| is_synced(r)) {
        let key = r
            .sync_group
            .clone()
            .unwrap_or_else(|| format!("{} (sync dir not recorded)", r.sync_mode));
        let g = groups.entry(key).or_default();
        g.0 = r.sync_mode.clone();
        g.1.insert(r.name.as_str());
        g.2.insert(r.host.as_str());
        g.3 = g.3.max(r.sync_rounds);
    }
    groups
        .into_iter()
        .map(|(k, (mode, names, hosts, rounds))| {
            json!({"group": k, "mode": mode, "runs": names.len(), "hosts": hosts.len(),
                   "max_round": rounds, "members": names})
        })
        .collect()
}

fn topology(runs: &[Run]) -> Value {
    let multi: Vec<Value> = runs
        .iter()
        .filter(|r| r.world_size > 1)
        .map(|r| json!({"run": r.name, "host": r.host, "world_size": r.world_size}))
        .collect();
    let warm: Vec<Value> = runs
        .iter()
        .filter_map(|r| {
            r.warm_start
                .as_ref()
                .map(|w| json!({"run": r.name, "from": w}))
        })
        .collect();
    let mut hosts: BTreeMap<&str, (u64, u64, u64, BTreeSet<&str>)> = BTreeMap::new();
    for r in runs {
        let h = hosts.entry(r.host.as_str()).or_default();
        h.0 += 1;
        h.1 += u64::from(r.world_size > 1);
        h.2 += u64::from(is_synced(r));
        h.3.insert(r.family.as_str());
    }
    let independent = runs
        .iter()
        .filter(|r| r.world_size <= 1 && r.warm_start.is_none() && !is_synced(r))
        .count();
    let groups = sync_groups(runs);
    let shared_groups = groups
        .iter()
        .filter(|g| g["runs"].as_u64() > Some(1))
        .count();
    let hosts: Vec<Value> = hosts
        .into_iter()
        .map(|(h, (n, m, s, f))| {
            json!({"host": h, "runs": n, "multi_rank_runs": m, "synced_runs": s, "families": f.len()})
        })
        .collect();
    json!({
        "runs": runs.len(),
        "independent_runs": independent,
        "data_parallel_runs": multi,
        "warm_started_runs": warm,
        "sync_groups": groups,
        "cross_run_weight_sharing": !warm.is_empty() || shared_groups > 0,
        "max_world_size": runs.iter().map(|r| r.world_size).max().unwrap_or(0),
        "hosts": hosts,
        "note": "Data-parallel runs average gradients across their own ranks only and log rank 0; the logs do not record which hosts those ranks ran on. Runs in a sync group exchange weights through a shared sync directory (sync_mode, sync_round). Every other run is its own model and starts from no other run's weights unless listed under warm_started_runs.",
    })
}

fn uniqueness(cfg: &Config, runs: &[Run], stats: &[RunStats]) -> Value {
    let (corpus_train, corpus_all) = extras::corpus(cfg.corpus_index.as_deref());
    let episodes: f64 = stats.iter().map(|s| s.tables).sum();
    let synth: f64 = stats.iter().filter_map(|s| s.synth_est).sum();
    let real = episodes - synth;
    let per_run_reuse = |s: &RunStats| {
        corpus_train
            .filter(|c| *c > 0)
            .map(|c| (s.tables - s.synth_est.unwrap_or(0.0)) / c as f64)
    };
    let max_run = runs
        .iter()
        .zip(stats)
        .filter_map(|(r, s)| per_run_reuse(s).map(|x| (x, r.name.clone())))
        .max_by(|a, b| a.0.total_cmp(&b.0));
    let mut basis: BTreeMap<&str, u64> = BTreeMap::new();
    for s in stats {
        *basis.entry(s.synth_basis).or_default() += 1;
    }
    json!({
        "episode_counter": "unique_splits_seen (trainer episode counter; not deduplicated)",
        "episodes_total": episodes.round(),
        "synthetic_est": synth.round(),
        "real_episodes_est": real.round(),
        "synth_basis_runs": basis,
        "corpus_train_tables": corpus_train,
        "corpus_tables": corpus_all,
        "fleet_avg_uses_per_train_table_est": corpus_train.filter(|c| *c > 0).map(|c| round(real / c as f64, 2)),
        "max_run_avg_uses_per_train_table_est": max_run.as_ref().map(|m| round(m.0, 2)),
        "max_run": max_run.map(|m| m.1),
        "distinct_datasets": Value::Null,
        "distinct_splits": Value::Null,
        "exposure": extras::exposure(runs),
    })
}

/// Build the whole `dope.fleet-macro/v1` document.
pub fn build(cfg: &Config) -> Value {
    let files = scan::find_train_files(&cfg.roots);
    let runs: Vec<Run> = files
        .iter()
        .filter_map(|(train, root)| scan::load_run(train, root))
        .filter(|r| !r.rows.is_empty())
        .collect();
    let window_start = cfg.now - cfg.active_window_s;
    let stats: Vec<RunStats> = runs
        .iter()
        .map(|r| {
            let tables: f64 = increments(&r.rows).iter().map(|x| x.1).sum();
            let (synth_est, synth_basis) = synth_estimate(r, tables);
            let active = r
                .rows
                .last()
                .is_some_and(|x| x.t >= window_start && x.t <= cfg.now + 60.0);
            RunStats {
                tables,
                synth_est,
                synth_basis,
                active,
            }
        })
        .collect();
    let buckets = fill_buckets(&runs, cfg);
    let (series, peak) = series(&buckets, cfg.bucket_s.max(1));
    let done_since = |span: f64| -> f64 {
        runs.iter()
            .flat_map(|r| increments(&r.rows))
            .filter(|(t, _)| *t > cfg.now - span && *t <= cfg.now)
            .map(|x| x.1)
            .sum()
    };
    let active_runs: Vec<usize> = (0..runs.len()).filter(|i| stats[*i].active).collect();
    let reported: f64 = active_runs
        .iter()
        .filter_map(|i| runs[*i].rows.last()?.tps)
        .sum();
    let active_hosts: BTreeSet<&str> = active_runs.iter().map(|i| runs[*i].host.as_str()).collect();
    let hosts: BTreeSet<&str> = runs.iter().map(|r| r.host.as_str()).collect();
    let first = runs
        .iter()
        .filter_map(|r| r.rows.first())
        .map(|r| r.t)
        .reduce(f64::min);
    let last = runs
        .iter()
        .filter_map(|r| r.rows.last())
        .map(|r| r.t)
        .reduce(f64::max);
    let run_hours: f64 = runs
        .iter()
        .map(|r| r.rows.last().map_or(0.0, |l| l.t) - r.rows.first().map_or(0.0, |f| f.t))
        .sum::<f64>()
        / 3600.0;
    json!({
        "schema": SCHEMA,
        "generated_at": round(cfg.now, 1),
        "bucket_s": cfg.bucket_s,
        "active_window_s": cfg.active_window_s,
        "inputs": {
            "roots": cfg.roots, "train_files": files.len(), "runs": runs.len(),
            "rows": runs.iter().map(|r| r.rows.len()).sum::<usize>(),
            "bad_rows": runs.iter().map(|r| r.bad_rows).sum::<usize>(),
            "checkpoint_kpi": cfg.checkpoint_kpi, "corpus_index": cfg.corpus_index,
            "guard_dir": cfg.guard_dir,
        },
        "headline": {
            "tables_per_s_now": round(done_since(RECENT_S) / RECENT_S, 2),
            "tables_per_s_now_window_s": RECENT_S,
            "tables_per_s_reported": round(reported, 2),
            "active_runs": active_runs.len(), "active_hosts": active_hosts.len(),
            "tables_per_s_1h": round(done_since(3600.0) / 3600.0, 2),
            "peak": peak.map(|(v, t)| json!({"tables_per_s": round(v, 2), "t": t})),
            "tables_total": stats.iter().map(|s| s.tables).sum::<f64>().round(),
            "runs_total": runs.len(), "hosts_total": hosts.len(),
            "first_row_at": first, "last_row_at": last,
            "run_hours": round(run_hours, 1),
            "checkpoints_saved": runs.iter().map(|r| r.ckpt_times.len()).sum::<usize>(),
            "checkpoints_unplaced": runs
                .iter()
                .map(|r| r.ckpt_times.len() - placed_checkpoints(r).len())
                .sum::<usize>(),
        },
        "hosts": hosts,
        "guard": extras::guard(cfg.guard_dir.as_deref()),
        "series": series,
        "runs": runs.iter().zip(&stats).map(|(r, s)| run_json(r, cfg, s)).collect::<Vec<_>>(),
        "checkpoint_kpi": extras::checkpoints(cfg.checkpoint_kpi.as_deref(), &runs, cfg.kpi_noise),
        "topology": topology(&runs),
        "uniqueness": uniqueness(cfg, &runs, &stats),
    })
}

//! Accepted resident/session bounds, with the original six-run reduction.

use serde::Serialize;

pub const RUNS: usize = 6;
pub const THREADS: usize = 8;
pub const MAX_ROWS: usize = 1_024;
pub const OVERHEAD_LIMIT_MS: f64 = 5.0;
pub const LATENCY_P50_LIMIT_MS: f64 = 30.0;

/// This release check has no automatic device selection or backend fallback.
pub fn canonical_device(input: &str) -> Result<String, &'static str> {
    if input == "cpu" {
        return Ok(input.into());
    }
    let ordinal = input
        .strip_prefix("cuda:")
        .filter(|x| !x.is_empty() && x.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|x| x.parse::<usize>().ok())
        .ok_or("resident release device must be cpu or cuda:N")?;
    Ok(format!("cuda:{ordinal}"))
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct StageTimes {
    pub prepare_ms: f64,
    pub conditioning_ms: f64,
    pub forward_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct Summary {
    pub warmup_runs_discarded: usize,
    pub measured_runs: usize,
    pub overhead_max_ms: f64,
    pub none_p50_ms: f64,
    pub none_p95_ms: f64,
    pub overhead_limit_ms: f64,
    pub none_p50_limit_ms: f64,
    pub passed: bool,
}

/// Read cost is measured once and included in every reconstructed latency.
/// All samples must be valid, including the discarded warm-up. Bounds are closed.
pub fn summarize(
    read_ms: f64,
    runs: &[StageTimes],
    rows: usize,
    threads: usize,
) -> Result<Summary, &'static str> {
    if !(1..=MAX_ROWS).contains(&rows) {
        return Err("resident release bounds require 1..=1024 fit rows");
    }
    if threads != THREADS {
        return Err("resident release bounds require RAYON_NUM_THREADS=8");
    }
    if runs.len() != RUNS {
        return Err("resident release bounds require six runs, including one warm-up");
    }
    let valid = |x: f64| x.is_finite() && x >= 0.0;
    if !valid(read_ms) {
        return Err("invalid CSV read timing");
    }
    let mut latencies = Vec::with_capacity(RUNS - 1);
    let mut overhead_max = 0.0_f64;
    for (index, run) in runs.iter().enumerate() {
        if ![run.prepare_ms, run.conditioning_ms, run.forward_ms]
            .into_iter()
            .all(valid)
        {
            return Err("invalid resident stage timing");
        }
        let overhead = run.conditioning_ms + run.forward_ms;
        let latency = read_ms + run.prepare_ms + overhead;
        if !valid(overhead) || !valid(latency) {
            return Err("resident timing sum overflow");
        }
        if latency == 0.0 {
            return Err("resident trials must have positive total latency");
        }
        if index != 0 {
            overhead_max = overhead_max.max(overhead);
            latencies.push(latency);
        }
    }
    latencies.sort_by(f64::total_cmp);
    let p50 = latencies[2];
    // Five measured runs: NumPy's linear 95th percentile is 80% into [3, 4].
    let p95 = latencies[3] + (latencies[4] - latencies[3]) * 0.8;
    Ok(Summary {
        warmup_runs_discarded: 1,
        measured_runs: RUNS - 1,
        overhead_max_ms: overhead_max,
        none_p50_ms: p50,
        none_p95_ms: p95,
        overhead_limit_ms: OVERHEAD_LIMIT_MS,
        none_p50_limit_ms: LATENCY_P50_LIMIT_MS,
        passed: overhead_max <= OVERHEAD_LIMIT_MS && p50 <= LATENCY_P50_LIMIT_MS,
    })
}

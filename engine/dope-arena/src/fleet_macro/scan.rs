//! Discovery and parsing of training runs for `dope_fleet_macro`: every
//! `train.jsonl` under the given roots, read once into compact rows.

use serde_json::Value;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Directory names never descended into while looking for runs.
const SKIP_DIRS: [&str; 6] = ["checkpoints", "code", "_work", "_logs", "packets", "target"];
/// Run directories sit at most this many levels below a root.
const MAX_DEPTH: usize = 4;
/// Config keys that would mean a run started from another run's weights.
const WARM_START_KEYS: [&str; 6] = [
    "init_from",
    "init_checkpoint",
    "resume_from",
    "warm_start",
    "load_checkpoint",
    "parent_run",
];

/// One logged optimizer step (fields the macro view needs, nothing else).
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub t: f64,
    pub step: u64,
    /// Trainer's cumulative episode counter (`episodes_seen`, formerly `unique_splits_seen`).
    pub cum: Option<f64>,
    pub tps: Option<f64>,
    pub loss: Option<f64>,
    /// Synthetic-prior episodes in this step (`n_synth_prior`).
    pub synth: Option<f64>,
    /// Episodes in this step (`n_reg + n_bin + n_mc`), when all three are logged.
    pub eps: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Run {
    pub name: String,
    pub host: String,
    pub family: String,
    pub dir: PathBuf,
    pub arch: String,
    pub recipe: String,
    pub world_size: u32,
    pub warm_start: Option<String>,
    /// Weight-sync mode (`independent_ws1`, `ddp_gloo`, `diloco_outer_momentum`, ...).
    pub sync_mode: String,
    /// Sync directory shared by a weight-sync group, when the config records it.
    pub sync_group: Option<String>,
    /// Highest logged `sync_round`.
    pub sync_rounds: u64,
    /// Rows in time order.
    pub rows: Vec<Row>,
    pub bad_rows: usize,
    /// Modification times of saved `checkpoints/*.safetensors`.
    pub ckpt_times: Vec<f64>,
}

/// A finite JSON number.
pub fn num(v: &Value, key: &str) -> Option<f64> {
    v.get(key)
        .and_then(Value::as_f64)
        .filter(|x| x.is_finite() && x.abs() < 1e15)
}

fn text(v: &Value, key: &str) -> Option<String> {
    v.get(key)
        .and_then(Value::as_str)
        .map(|s| s.chars().take(80).collect())
}

/// Every `train.jsonl` below `roots` (sorted, depth-limited, `SKIP_DIRS` excluded).
pub fn find_train_files(roots: &[PathBuf]) -> Vec<(PathBuf, PathBuf)> {
    let mut out = Vec::new();
    for root in roots {
        walk(root, 0, root, &mut out);
    }
    out.sort();
    out.dedup();
    out
}

fn walk(dir: &Path, depth: usize, root: &Path, out: &mut Vec<(PathBuf, PathBuf)>) {
    if depth > MAX_DEPTH {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        if kind.is_file() && name == "train.jsonl" {
            out.push((path, root.to_path_buf()));
        } else if kind.is_dir() && !SKIP_DIRS.contains(&name.as_str()) && !name.starts_with('.') {
            walk(&path, depth + 1, root, out);
        }
    }
}

/// Host of a run directory: `<host>/runs/<run>` names the host, else the root's name.
pub fn host_of(run_dir: &Path, root: &Path) -> String {
    let parent = run_dir.parent();
    if let Some(p) = parent.filter(|p| p.file_name().is_some_and(|n| n == "runs")) {
        if let Some(h) = p.parent().and_then(Path::file_name) {
            return h.to_string_lossy().into_owned();
        }
    }
    root.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "local".to_string())
}

/// Run family: the name without a trailing seed suffix `-s<digits>`.
pub fn family_of(name: &str) -> String {
    if let Some(pos) = name.rfind("-s") {
        let tail = &name[pos + 2..];
        if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) {
            return name[..pos].to_string();
        }
    }
    name.to_string()
}

fn parse_row(v: &Value) -> Option<Row> {
    let t = num(v, "time")?;
    let step = num(v, "step").filter(|s| *s >= 0.0)? as u64;
    let eps = match (num(v, "n_reg"), num(v, "n_bin"), num(v, "n_mc")) {
        (Some(a), Some(b), Some(c)) => Some(a + b + c),
        _ => None,
    };
    Some(Row {
        t,
        step,
        cum: num(v, "episodes_seen").or_else(|| num(v, "unique_splits_seen")),
        tps: num(v, "tables_per_s"),
        loss: num(v, "loss"),
        synth: num(v, "n_synth_prior"),
        eps,
    })
}

fn read_config(dir: &Path) -> Value {
    fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(Value::Null)
}

fn checkpoint_times(dir: &Path) -> Vec<f64> {
    let Ok(entries) = fs::read_dir(dir.join("checkpoints")) else {
        return Vec::new();
    };
    let mut times: Vec<f64> = entries
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
        .filter_map(|e| e.metadata().ok()?.modified().ok())
        .filter_map(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
        .collect();
    times.sort_by(f64::total_cmp);
    times
}

/// Read one run; `None` when the file cannot be opened.
pub fn load_run(train: &Path, root: &Path) -> Option<Run> {
    let dir = train.parent()?.to_path_buf();
    let file = fs::File::open(train).ok()?;
    let config = read_config(&dir);
    let mut rows = Vec::new();
    let mut bad_rows = 0;
    let mut ws_rows: Option<u32> = None;
    let (mut arch, mut recipe) = (text(&config, "arch"), text(&config, "recipe"));
    let mut sync_mode = text(&config, "sync_mode");
    let mut sync_rounds = 0u64;
    for line in BufReader::new(file).lines() {
        let Ok(line) = line else {
            bad_rows += 1;
            continue;
        };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(&line) else {
            bad_rows += 1;
            continue;
        };
        match parse_row(&v) {
            Some(r) => rows.push(r),
            None => bad_rows += 1,
        }
        if let Some(w) = num(&v, "world_size") {
            ws_rows = Some(ws_rows.unwrap_or(1).max(w as u32));
        }
        if arch.is_none() {
            arch = text(&v, "arch");
        }
        if let Some(m) = text(&v, "sync_mode") {
            sync_mode = Some(m);
        }
        if let Some(r) = num(&v, "sync_round").filter(|r| *r >= 0.0) {
            sync_rounds = sync_rounds.max(r as u64);
        }
        if recipe.is_none() {
            recipe = text(&v, "recipe");
        }
    }
    rows.sort_by(|a, b| a.t.total_cmp(&b.t).then(a.step.cmp(&b.step)));
    let world_size = ws_rows
        .or_else(|| num(&config, "world_size").map(|w| w as u32))
        .unwrap_or(1)
        .max(1);
    let warm_start = WARM_START_KEYS
        .iter()
        .find_map(|k| text(&config, k).filter(|s| !s.is_empty()));
    let sync_mode = sync_mode
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| if world_size > 1 { "ddp" } else { "independent" }.to_string());
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    Some(Run {
        host: host_of(&dir, root),
        family: family_of(&name),
        name,
        arch: arch.unwrap_or_default(),
        recipe: recipe.unwrap_or_default(),
        world_size,
        warm_start,
        sync_group: text(&config, "sync_group")
            .or_else(|| text(&config, "sync_dir"))
            .filter(|s| !s.is_empty()),
        sync_mode,
        sync_rounds,
        rows,
        bad_rows,
        ckpt_times: checkpoint_times(&dir),
        dir,
    })
}

/// Train-step increments `(time, tables)`: differences of the cumulative counter,
/// else logged rate times elapsed time (each gap capped at 600 s).
pub fn increments(rows: &[Row]) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(rows.len());
    let mut prev_cum = 0.0;
    let mut prev_t: Option<f64> = None;
    for r in rows {
        let inc = match (r.cum, r.tps, prev_t) {
            (Some(c), _, _) => {
                let d = (c - prev_cum).max(0.0);
                prev_cum = prev_cum.max(c);
                d
            }
            (None, Some(rate), Some(p)) => rate * (r.t - p).clamp(0.0, 600.0),
            _ => 0.0,
        };
        prev_t = Some(r.t);
        out.push((r.t, inc));
    }
    out
}

/// Median of a slice (sorted copy); `None` when empty.
pub fn median(values: &[f64]) -> Option<f64> {
    quantile(values, 0.5)
}

/// Linear-interpolated quantile of a slice; `None` when empty.
pub fn quantile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let pos = q.clamp(0.0, 1.0) * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    Some(v[lo] + (v[hi] - v[lo]) * (pos - lo as f64))
}

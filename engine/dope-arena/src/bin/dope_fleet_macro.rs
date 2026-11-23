//! `dope_fleet_macro`: writes the campaign-wide `dope.fleet-macro/v1` JSON the
//! DOPE dashboard serves at `/api/macro`.
//!
//! ```text
//! dope_fleet_macro --root ~/dope-data/fleet-mirror \
//!     --checkpoint-kpi ~/dope-data/dashboard/checkpoint_kpi.jsonl \
//!     --corpus-index ~/dope-data/flat/s3-v250/index.jsonl \
//!     --guard-dir ~/dope/.agent/tools/cpu_fleet/state \
//!     --out ~/dope-data/dashboard/fleet_macro.json [--watch 60]
//! ```
//!
//! Read only: run directories, the guard directory and the index are never written.

#[path = "../fleet_macro/mod.rs"]
mod fleet_macro;

use clap::Parser;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Parser)]
#[command(
    name = "dope_fleet_macro",
    about = "Campaign-wide DOPE fleet telemetry (dope.fleet-macro/v1)"
)]
struct Cli {
    /// Directory holding runs (`<host>/runs/<run>/train.jsonl` or `<run>/train.jsonl`); repeatable.
    #[arg(long = "root", required = true)]
    roots: Vec<PathBuf>,
    /// Scored checkpoints (`{"run","step","variant","metric","value","panel"}` per line).
    #[arg(long)]
    checkpoint_kpi: Option<PathBuf>,
    /// Corpus index (`index.jsonl`, one table per line with `split`).
    #[arg(long)]
    corpus_index: Option<PathBuf>,
    /// ASR guard state directory (`guard_state.json`, `guard_events.jsonl`).
    #[arg(long)]
    guard_dir: Option<PathBuf>,
    /// Bucket width of the time series, in seconds.
    #[arg(long, default_value_t = 300)]
    bucket_s: u64,
    /// A run counts as active when its last row is at most this old.
    #[arg(long, default_value_t = 120.0)]
    active_window_s: f64,
    /// Checkpoint KPI change treated as noise (the contract's noise floor).
    #[arg(long, default_value_t = 0.01)]
    kpi_noise: f64,
    /// Clock override (Unix seconds), for reproducible output.
    #[arg(long)]
    now: Option<f64>,
    /// Output file, replaced atomically.
    #[arg(long)]
    out: PathBuf,
    /// Rebuild every N seconds until killed.
    #[arg(long)]
    watch: Option<u64>,
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)
}

fn once(cli: &Cli) -> Result<String, String> {
    let cfg = fleet_macro::Config {
        roots: cli.roots.clone(),
        checkpoint_kpi: cli.checkpoint_kpi.clone(),
        corpus_index: cli.corpus_index.clone(),
        guard_dir: cli.guard_dir.clone(),
        bucket_s: cli.bucket_s.max(1),
        active_window_s: cli.active_window_s.max(0.0),
        kpi_noise: cli.kpi_noise,
        now: cli.now.unwrap_or_else(unix_now),
    };
    let doc = fleet_macro::build(&cfg);
    let bytes = serde_json::to_vec(&doc).map_err(|e| format!("serialize: {e}"))?;
    write_atomic(&cli.out, &bytes).map_err(|e| format!("write {}: {e}", cli.out.display()))?;
    let h = &doc["headline"];
    Ok(format!(
        "{{\"schema\":\"{}\",\"runs\":{},\"rows\":{},\"tables_per_s_now\":{},\"tables_total\":{},\"bytes\":{}}}",
        fleet_macro::SCHEMA,
        doc["inputs"]["runs"],
        doc["inputs"]["rows"],
        h["tables_per_s_now"],
        h["tables_total"],
        bytes.len()
    ))
}

fn main() {
    let cli = Cli::parse();
    loop {
        match once(&cli) {
            Ok(line) => println!("{line}"),
            Err(e) => {
                eprintln!("dope_fleet_macro: {e}");
                if cli.watch.is_none() {
                    std::process::exit(1);
                }
            }
        }
        match cli.watch {
            Some(secs) => std::thread::sleep(Duration::from_secs(secs.max(5))),
            None => break,
        }
    }
}

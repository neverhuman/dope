//! Measure the accepted resident/session limits on one explicitly supplied fit CSV.
//!
//! RAYON_NUM_THREADS=8 cargo run --release --manifest-path engine/Cargo.toml \
//!   -p dope-core --example resident_release_check [--features cuda] -- \
//!   CHECKPOINT CSV TARGET regression|classification|multiclass cpu|cuda:N
//! Exit 0: measured bounds pass; 1: measured bounds fail; 2: refused/unavailable.

#[path = "resident_release_check/limits.rs"]
mod limits;

use candle_core::{Device, Tensor};
use dope_core::landmark::{LandmarkBundle, LandmarkMode};
use dope_core::mlc_v3::MlcV3Model;
use dope_core::prepare::{build_conditioning, prepare_table_cells, DopeConditioning, PrepTask};
use dope_core::RawTable;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "CHECKPOINT CSV TARGET regression|classification|multiclass cpu|cuda:N";
const CHECKPOINT_SHA256: &str = "d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c";
const CHECKPOINT_BYTES: u64 = 1_781_800;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn forward(model: &MlcV3Model, cond: &DopeConditioning, dev: &Device) -> Res<f64> {
    let start = Instant::now();
    let columns: Vec<f32> = cond.col_landmarks.iter().flatten().copied().collect();
    let ds = Tensor::from_slice(&cond.vec256, (1, 256), dev)?;
    let cs = Tensor::from_slice(&columns, (1, 64, 17), dev)?;
    let values = model
        .encode_features_629_tensor(&ds, &cs, "high")?
        .to_vec2::<f32>()?;
    // Retain the original benchmark span through upload and completed host readback.
    let elapsed = ms(start);
    if values.len() != 1 || values[0].len() != 629 || !values[0].iter().all(|x| x.is_finite()) {
        return Err("resident forward did not produce 629 finite features".into());
    }
    Ok(elapsed)
}

fn check() -> Res<bool> {
    let args: Vec<String> = std::env::args_os()
        .map(|arg| {
            arg.into_string()
                .map_err(|_| "arguments must be valid UTF-8")
        })
        .collect::<Result<_, _>>()?;
    let [_, checkpoint, csv, target, task_name, device_name] = args.as_slice() else {
        return Err(USAGE.into());
    };
    let task = match task_name.as_str() {
        "regression" => PrepTask::Regression,
        "classification" => PrepTask::Classification,
        "multiclass" => PrepTask::Multiclass,
        _ => return Err("unsupported task; use regression, classification or multiclass".into()),
    };
    if std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|x| x.parse::<usize>().ok())
        != Some(limits::THREADS)
        || rayon::current_num_threads() != limits::THREADS
    {
        return Err("resident release bounds require RAYON_NUM_THREADS=8".into());
    }
    let device_name = limits::canonical_device(device_name)?;
    let device = dope_core::device::parse(&device_name)
        .map_err(|_| "requested inference device is unavailable")?;
    let load_start = Instant::now();
    let metadata = std::fs::symlink_metadata(checkpoint)
        .map_err(|_| "cannot read released checkpoint metadata")?;
    if !metadata.is_file() || metadata.len() != CHECKPOINT_BYTES {
        return Err("released checkpoint must be a regular file with the expected size".into());
    }
    let weights = std::fs::read(checkpoint).map_err(|_| "cannot read released checkpoint")?;
    if format!("{:x}", Sha256::digest(&weights)) != CHECKPOINT_SHA256 {
        return Err("released checkpoint SHA-256 mismatch".into());
    }
    let model = MlcV3Model::from_safetensors_bytes_on_device(&weights, &device)
        .map_err(|_| "cannot load released checkpoint on the requested device")?;
    let model_load_ms = ms(load_start);

    let read_start = Instant::now();
    let raw = RawTable::read(Path::new(csv)).map_err(|_| "cannot read fit CSV")?;
    let target_index = raw
        .column_index(target)
        .map_err(|_| "target column not found")?;
    let mut x = vec![Vec::new(); raw.rows()];
    for column in raw
        .columns
        .iter()
        .enumerate()
        .filter(|(c, _)| *c != target_index)
    {
        for (r, value) in column.1.iter().enumerate() {
            x[r].push(value.clone());
        }
    }
    let y = raw.columns[target_index].clone();
    let read_ms = ms(read_start);
    if !(1..=limits::MAX_ROWS).contains(&raw.rows()) {
        return Err("resident release bounds require 1..=1024 fit rows".into());
    }

    let mut runs = Vec::with_capacity(limits::RUNS);
    let mut source_digest = None;
    let mut conditioning_digest = None;
    for _ in 0..limits::RUNS {
        let start = Instant::now();
        let table =
            prepare_table_cells(&x, &y, task).map_err(|_| "fit table preparation failed")?;
        let prepare_ms = ms(start);
        let start = Instant::now();
        let cond = build_conditioning(&table, &LandmarkBundle::zero(), None, LandmarkMode::None)
            .map_err(|_| "zero-landmark conditioning failed")?;
        let conditioning_ms = ms(start);
        let forward_ms = forward(&model, &cond, &device)
            .map_err(|_| "resident forward failed or produced nonfinite features")?;
        // Digest accounting is outside the measured spans, after the warm-up forward.
        if source_digest.is_none() {
            source_digest = Some(table.source_rows_digest());
            conditioning_digest = Some(cond.digest());
        }
        runs.push(limits::StageTimes {
            prepare_ms,
            conditioning_ms,
            forward_ms,
        });
    }
    let summary = limits::summarize(read_ms, &runs, raw.rows(), rayon::current_num_threads())?;
    let binary = std::fs::read(std::env::current_exe()?)?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "schema": "dope.resident-release-check/v1",
            "status": if summary.passed { "passed" } else { "failed" },
            "regime": "resident_session",
            "checkpoint_sha256": model.inventory.sha256,
            "executable_sha256": format!("{:x}", Sha256::digest(&binary)),
            "source_rows_digest": source_digest,
            "conditioning_digest": conditioning_digest,
            "task": task_name,
            "device": device_name,
            "actual_device": format!("{:?}", device.location()),
            "threads": rayon::current_num_threads(),
            "rows": raw.rows(),
            "features": raw.width() - 1,
            "model_load_ms": model_load_ms,
            "read_ms": read_ms,
            "runs": runs,
            "summary": summary,
            "scope": "One caller-supplied fit fixture; real-cohort provenance supplied separately. Resident/session bounds only; model load excluded. Does not requalify the none/full speedup ratio or cold CLI latency."
        }))?
    );
    Ok(summary.passed)
}

fn main() -> ExitCode {
    match check() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("resident release check refused: {error}");
            ExitCode::from(2)
        }
    }
}

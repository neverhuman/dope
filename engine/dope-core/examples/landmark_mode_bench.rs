//! Stage-timed latency of `mlc-v3 encode` landmark modes, in one warm process.
//!
//! `cargo run --release --example landmark_mode_bench [--features cuda] -- \
//!   CHECKPOINT CSV TARGET TASK DEVICE REPS [WORK_BUDGET] [MODES]`
//! TASK is regression|classification|multiclass, DEVICE cpu|cuda:N. Prints one JSON
//! object: cold model load, then per repetition the stage timings (ms) of none,
//! progressive (zero, macro, larger, fine; N=300) and full (unsampled and N=300),
//! each split into landmarks, conditioning and forward. MODES (comma list of
//! none,progressive,full_sub300,full; default all) selects what is timed.
//! Measurements only.
use candle_core::{Device, Tensor};
use dope_core::landmark::{LandmarkBundle, LandmarkMode, LandmarkOptions, SubsampleSpec};
use dope_core::landmark::{LandmarkGroup, PROGRESSIVE_ORDER};
use dope_core::mlc_v3::MlcV3Model;
use dope_core::prepare::{
    build_conditioning, build_conditioning_sampled, prepare_table_cells,
    progressive_landmarks_for_table, DopeConditioning, NativeReportView, PrepTask, PreparedTable,
};
use dope_core::RawTable;
use serde_json::{json, Value};
use std::path::Path;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn forward(model: &MlcV3Model, cond: &DopeConditioning, dev: &Device) -> Res<f64> {
    let t = Instant::now();
    let cols: Vec<f32> = cond.col_landmarks.iter().flatten().copied().collect();
    let ds = Tensor::from_slice(&cond.vec256, (1, 256), dev)?;
    let cs = Tensor::from_slice(&cols, (1, 64, 17), dev)?;
    model
        .encode_features_629_tensor(&ds, &cs, "high")?
        .to_vec2::<f32>()?;
    Ok(ms(t))
}

/// `sub = None` runs the unsampled panel; Fine (or Full) is conditioned as Full.
fn landmark_run(
    model: &MlcV3Model,
    table: &PreparedTable,
    opts: &LandmarkOptions,
    sub: Option<SubsampleSpec>,
    dev: &Device,
    groups: &[LandmarkGroup],
) -> Res<Value> {
    let t = Instant::now();
    let mut driver = progressive_landmarks_for_table(table, opts, sub)?;
    let positions = driver.selection().map(|s| s.positions.clone());
    let mut steps = vec![json!({"stage": "driver_new", "ms": ms(t)})];
    for (k, &group) in groups.iter().enumerate() {
        let t = Instant::now();
        let (bundle, sig) = driver.next_group(group)?;
        let landmarks = ms(t);
        let t = Instant::now();
        let view = NativeReportView::from_signature(sig);
        let mode = if k + 1 == groups.len() {
            LandmarkMode::Full
        } else {
            LandmarkMode::Progressive
        };
        let cond =
            build_conditioning_sampled(table, &bundle, Some(&view), mode, positions.as_deref())?;
        let conditioning = ms(t);
        steps.push(json!({"stage": group.as_str(), "landmarks_ms": landmarks,
            "conditioning_ms": conditioning, "forward_ms": forward(model, &cond, dev)?,
            "status": cond.bundle.status, "digest": cond.digest()}));
    }
    Ok(json!(steps))
}

fn main() -> Res<()> {
    let a: Vec<String> = std::env::args().collect();
    let [_, ckpt, csv, target, task, device, reps, ..] = a.as_slice() else {
        return Err("usage: CHECKPOINT CSV TARGET TASK DEVICE REPS [WORK_BUDGET] [MODES]".into());
    };
    let reps: usize = reps.parse()?;
    let dev = dope_core::device::parse(device)?;
    let mut opts = LandmarkOptions::default();
    if let Some(budget) = a.get(7) {
        opts.work_budget = budget.parse()?;
    }
    let modes = a
        .get(8)
        .map_or("none,progressive,full_sub300,full", String::as_str);
    let want = |m: &str| modes.split(',').any(|x| x == m);
    let task = match task.as_str() {
        "regression" => PrepTask::Regression,
        "classification" => PrepTask::Classification,
        _ => PrepTask::Multiclass,
    };
    let t = Instant::now();
    let model = MlcV3Model::load_safetensors_on_device(Path::new(ckpt), &dev)?;
    let load_ms = ms(t);
    let t = Instant::now();
    let raw = RawTable::read(Path::new(csv))?;
    let ti = raw.column_index(target)?;
    let mut x = vec![Vec::new(); raw.rows()];
    for (c, col) in raw.columns.iter().enumerate().filter(|(c, _)| *c != ti) {
        let _ = c;
        for (r, v) in col.iter().enumerate() {
            x[r].push(v.clone());
        }
    }
    let y = raw.columns[ti].clone();
    let read_ms = ms(t);
    let full_groups = [
        LandmarkGroup::Macro,
        LandmarkGroup::Larger,
        LandmarkGroup::Fine,
    ];
    let mut runs = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        let table = prepare_table_cells(&x, &y, task)?;
        let prepare_ms = ms(t);
        let mut run = json!({"prepare_ms": prepare_ms});
        if want("none") {
            let t = Instant::now();
            let cond =
                build_conditioning(&table, &LandmarkBundle::zero(), None, LandmarkMode::None)?;
            run["none"] =
                json!({"conditioning_ms": ms(t), "forward_ms": forward(&model, &cond, &dev)?});
        }
        if want("progressive") {
            let sub = Some(SubsampleSpec::PROGRESSIVE_DEFAULT);
            run["progressive"] =
                landmark_run(&model, &table, &opts, sub, &dev, &PROGRESSIVE_ORDER)?;
        }
        if want("full_sub300") {
            let sub = Some(SubsampleSpec::new(300, 0));
            run["full_sub300"] = landmark_run(&model, &table, &opts, sub, &dev, &full_groups)?;
        }
        if want("full") {
            run["full"] = landmark_run(&model, &table, &opts, None, &dev, &full_groups)?;
        }
        runs.push(run);
    }
    println!(
        "{}",
        serde_json::to_string(&json!({"csv": csv, "rows": raw.rows(), "device": device,
        "work_budget": opts.work_budget, "load_ms": load_ms, "read_ms": read_ms, "runs": runs}))?
    );
    Ok(())
}

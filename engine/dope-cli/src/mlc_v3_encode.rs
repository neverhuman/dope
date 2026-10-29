//! `dope mlc-v3 encode`: optional/progressive landmark conditioning and 629-D encoding.

use super::resolve_weights_path;
use clap::Args;
use dope_core::landmark::{
    LandmarkBundle, LandmarkMode, LandmarkOptions, SubsampleSpec, PROGRESSIVE_ORDER,
};
use dope_core::mlc_v3::MlcV3Model;
use dope_core::prepare::{
    build_conditioning, build_conditioning_sampled, prepare_table_cells,
    progressive_landmarks_for_table, DopeConditioning, NativeReportView, PrepTask,
};
use dope_core::schema::{infer, IntakeOptions, Task};
use dope_core::RawTable;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::PathBuf;

#[path = "mlc_v3_encode/receipt.rs"]
mod receipt;

#[derive(Args)]
pub(crate) struct EncodeV3Args {
    #[arg(long)]
    pub(crate) weights: Option<PathBuf>,
    #[arg(long, default_value = "multitask", value_parser = ["multitask", "maxtrain97"])]
    pub(crate) preset: String,
    #[arg(long)]
    pub(crate) csv: PathBuf,
    #[arg(short, long)]
    pub(crate) target: String,
    #[arg(long, value_parser = ["regression", "classification", "multiclass"])]
    pub(crate) task: Option<String>,
    #[arg(long, default_value = "full", value_parser = ["none", "fast", "progressive", "full"])]
    pub(crate) landmark_mode: String,
    #[arg(long)]
    pub(crate) landmark_subsample: Option<usize>,
    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,
    #[arg(long)]
    pub(crate) bundle_json: Option<PathBuf>,
    #[arg(long, default_value = "cpu")]
    pub(crate) device: String,
    #[arg(long)]
    pub(crate) json: bool,
    #[arg(short, long)]
    pub(crate) out: Option<PathBuf>,
    /// Write named snapshot receipts to a new JSONL file, separate from encode output.
    #[arg(long)]
    pub(crate) encoding_receipt: Option<PathBuf>,
}

fn encode_cond_on_model(
    model: &MlcV3Model,
    cond: &DopeConditioning,
    cpu_pool: Option<&rayon::ThreadPool>,
) -> Result<(Vec<f32>, String), String> {
    let feat = if model.device().is_cpu() {
        let col_vecs: Vec<Vec<f32>> = cond.col_landmarks.iter().map(|r| r.to_vec()).collect();
        let run = || model.encode_features_629(&cond.vec256, &col_vecs, "high");
        match cpu_pool {
            Some(pool) => pool.install(run).map_err(|e| e.to_string())?,
            None => run().map_err(|e| e.to_string())?,
        }
    } else {
        model
            .encode_conditioning_629(&cond.vec256, &cond.col_landmarks, "high")
            .map_err(|e| e.to_string())?
    };
    let mut h = Sha256::new();
    for v in &feat {
        h.update(v.to_le_bytes());
    }
    let sha = format!("{:x}", h.finalize());
    Ok((feat, sha))
}

pub(super) fn run_encode(a: EncodeV3Args) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let map_err = |e: dope_core::DopeError| e.to_string();
    let mode: LandmarkMode = a.landmark_mode.parse().map_err(map_err)?;
    let dev = dope_core::device::parse(&a.device).map_err(map_err)?;
    let cpu_pool = if dev.is_cpu() {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .map_err(|_| "could not configure 1-thread CPU reference pool".to_string())?,
        )
    } else {
        None
    };
    let dev_desc = match &cpu_pool {
        Some(pool) => pool.install(|| dope_core::device::describe(&dev)),
        None => dope_core::device::describe(&dev),
    };
    let weights_path = resolve_weights_path(&a.weights, &a.preset)?;
    let model = MlcV3Model::load_safetensors_on_device(&weights_path, &dev).map_err(map_err)?;

    let raw = RawTable::read(&a.csv).map_err(map_err)?;
    let target_idx = raw.column_index(&a.target).map_err(map_err)?;
    let nrows = raw.rows();
    let ncols = raw.columns.len();
    let prep_task = match a.task.as_deref() {
        Some("regression") => PrepTask::Regression,
        Some("multiclass") => PrepTask::Multiclass,
        Some("classification") => {
            let uniq: std::collections::BTreeSet<&str> = raw.columns[target_idx]
                .iter()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();
            if uniq.len() > 2 {
                PrepTask::Multiclass
            } else {
                PrepTask::Classification
            }
        }
        None => {
            let schema = infer(&raw, target_idx, &IntakeOptions::default()).map_err(map_err)?;
            match schema.task {
                Task::Regression => PrepTask::Regression,
                Task::Classification => {
                    let uniq: std::collections::BTreeSet<&str> = raw.columns[target_idx]
                        .iter()
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect();
                    if uniq.len() > 2 {
                        PrepTask::Multiclass
                    } else {
                        PrepTask::Classification
                    }
                }
            }
        }
        Some(other) => return Err(format!("unsupported task `{other}`")),
    };
    // Move the cells into row-major order rather than cloning every one.
    let mut columns = raw.columns;
    let y_cells = std::mem::take(&mut columns[target_idx]);
    let mut x_cells = vec![Vec::with_capacity(ncols.saturating_sub(1)); nrows];
    for (col_idx, col) in columns.into_iter().enumerate() {
        if col_idx == target_idx {
            continue;
        }
        for (r, val) in col.into_iter().enumerate() {
            x_cells[r].push(val);
        }
    }

    let prepared = prepare_table_cells(&x_cells, &y_cells, prep_task).map_err(map_err)?;
    let supplied_bundle: Option<LandmarkBundle> = match &a.bundle_json {
        Some(p) => {
            let s = std::fs::read_to_string(p).map_err(|e| e.to_string())?;
            Some(serde_json::from_str(&s).map_err(|e| e.to_string())?)
        }
        None => None,
    };

    let mut receipts = a
        .encoding_receipt
        .as_ref()
        .map(|path| {
            receipt::ReceiptWriter::create(
                path,
                a.out.as_deref(),
                prepared.computed_source_rows_digest(),
                model.inventory.sha256.clone(),
                dev_desc.clone(),
            )
        })
        .transpose()?;
    let mut out_w = crate::writer(&a.out)?;
    let emit_line =
        |w: &mut Box<dyn Write>, val: &serde_json::Value, pretty: bool| -> Result<(), String> {
            let s = if pretty {
                serde_json::to_string_pretty(val).map_err(|e| e.to_string())?
            } else {
                serde_json::to_string(val).map_err(|e| e.to_string())?
            };
            writeln!(w, "{s}").map_err(|e| e.to_string())?;
            w.flush().map_err(|e| e.to_string())
        };

    match mode {
        LandmarkMode::None | LandmarkMode::Fast => {
            let b = supplied_bundle.unwrap_or_else(LandmarkBundle::zero);
            let cond = build_conditioning(&prepared, &b, None, mode).map_err(map_err)?;
            let (feat, feat_sha) = encode_cond_on_model(&model, &cond, cpu_pool.as_ref())?;
            let doc = serde_json::json!({
                "seq": 0,
                "mode": mode.as_str(),
                "device": dev_desc,
                "recipe": cond.recipe,
                "status": cond.bundle.status,
                "subsample_n": cond.bundle.subsample_n,
                "source_digest": cond.bundle.source_digest,
                "bundle_digest": cond.bundle.conditioning_digest(),
                "conditioning_digest": cond.digest(),
                "features629_sha256": feat_sha,
                "features629": feat,
                "latency_ms": t0.elapsed().as_secs_f64() * 1000.0,
            });
            emit_line(&mut out_w, &doc, !a.json)?;
            if let Some(receipts) = &mut receipts {
                receipts.emit(0, &cond, &feat_sha, None)?;
            }
        }
        LandmarkMode::Progressive => {
            let opts = LandmarkOptions {
                seed: a.seed,
                ..LandmarkOptions::mlc_v3_training()
            };
            let opts_digest = dope_core::landmark::bundle::options_digest(&opts);
            let zero = LandmarkBundle::zero();
            let cond0 =
                build_conditioning(&prepared, &zero, None, LandmarkMode::None).map_err(map_err)?;
            let (feat0, sha0) = encode_cond_on_model(&model, &cond0, cpu_pool.as_ref())?;
            let doc0 = serde_json::json!({
                "seq": 0,
                "mode": "progressive",
                "group": "none",
                "device": dev_desc,
                "recipe": cond0.recipe,
                "status": cond0.bundle.status,
                "subsample_n": cond0.bundle.subsample_n,
                "source_digest": cond0.bundle.source_digest,
                "options_digest": opts_digest,
                "work_budget": opts.work_budget,
                "bundle_digest": cond0.bundle.conditioning_digest(),
                "conditioning_digest": cond0.digest(),
                "features629_sha256": sha0,
                "features629": feat0,
                "latency_ms": t0.elapsed().as_secs_f64() * 1000.0,
            });
            emit_line(&mut out_w, &doc0, false)?;
            if let Some(receipts) = &mut receipts {
                receipts.emit(0, &cond0, &sha0, Some(&opts))?;
            }

            let sub_n = a
                .landmark_subsample
                .unwrap_or(SubsampleSpec::PROGRESSIVE_DEFAULT.n);
            let sub = if sub_n == 0 {
                None
            } else {
                Some(SubsampleSpec::new(sub_n, a.seed))
            };
            let mut driver =
                progressive_landmarks_for_table(&prepared, &opts, sub).map_err(map_err)?;
            let sample_pos = driver.selection().map(|s| s.positions.clone());
            for (idx, &group) in PROGRESSIVE_ORDER.iter().enumerate() {
                let (bundle, sig) = driver.next_group(group).map_err(map_err)?;
                let rep_view = NativeReportView::from_signature(sig);
                let step_mode = if group == dope_core::landmark::LandmarkGroup::Fine {
                    LandmarkMode::Full
                } else {
                    LandmarkMode::Progressive
                };
                let cond = build_conditioning_sampled(
                    &prepared,
                    &bundle,
                    Some(&rep_view),
                    step_mode,
                    sample_pos.as_deref(),
                )
                .map_err(map_err)?;
                let (feat, sha) = encode_cond_on_model(&model, &cond, cpu_pool.as_ref())?;
                let doc = serde_json::json!({
                    "seq": idx + 1,
                    "mode": "progressive",
                    "group": group.as_str(),
                    "device": dev_desc,
                    "recipe": cond.recipe,
                    "status": cond.bundle.status,
                    "subsample_n": cond.bundle.subsample_n,
                    "source_digest": cond.bundle.source_digest,
                    "options_digest": opts_digest,
                    "work_budget": opts.work_budget,
                    "bundle_digest": cond.bundle.conditioning_digest(),
                    "conditioning_digest": cond.digest(),
                    "features629_sha256": sha,
                    "features629": feat,
                    "latency_ms": t0.elapsed().as_secs_f64() * 1000.0,
                });
                emit_line(&mut out_w, &doc, false)?;
                if let Some(receipts) = &mut receipts {
                    receipts.emit(idx + 1, &cond, &sha, Some(&opts))?;
                }
            }
        }
        LandmarkMode::Full => {
            let opts = LandmarkOptions {
                seed: a.seed,
                ..LandmarkOptions::mlc_v3_training()
            };
            let opts_digest = dope_core::landmark::bundle::options_digest(&opts);
            let sub = a
                .landmark_subsample
                .filter(|&n| n > 0)
                .map(|n| SubsampleSpec::new(n, a.seed));
            let mut driver =
                progressive_landmarks_for_table(&prepared, &opts, sub).map_err(map_err)?;
            let sample_pos = driver.selection().map(|s| s.positions.clone());
            let all_ids: Vec<u8> = (1..=23).collect();
            let (computed_bundle, sig) = driver.run_ids(&all_ids).map_err(map_err)?;
            let rep_view = NativeReportView::from_signature(sig);
            let b = supplied_bundle.unwrap_or(computed_bundle);
            let cond = build_conditioning_sampled(
                &prepared,
                &b,
                Some(&rep_view),
                LandmarkMode::Full,
                sample_pos.as_deref(),
            )
            .map_err(map_err)?;
            let (feat, sha) = encode_cond_on_model(&model, &cond, cpu_pool.as_ref())?;
            let doc = serde_json::json!({
                "seq": 0,
                "mode": "full",
                "device": dev_desc,
                "recipe": cond.recipe,
                "status": cond.bundle.status,
                "subsample_n": cond.bundle.subsample_n,
                "source_digest": cond.bundle.source_digest,
                "options_digest": opts_digest,
                "work_budget": opts.work_budget,
                "bundle_digest": cond.bundle.conditioning_digest(),
                "conditioning_digest": cond.digest(),
                "features629_sha256": sha,
                "features629": feat,
                "latency_ms": t0.elapsed().as_secs_f64() * 1000.0,
            });
            emit_line(&mut out_w, &doc, !a.json)?;
            if let Some(receipts) = &mut receipts {
                receipts.emit(0, &cond, &sha, Some(&opts))?;
            }
        }
    }
    Ok(())
}

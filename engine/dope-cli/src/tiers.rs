//! Reproducible per-dataset pretrained artifact packaging; no network or publication.
use crate::args::TierArgs;
use dope_core::{fit, schema::Task, FitOptions, ModelSize, RawTable};
use serde_json::json;
use sha2::{Digest, Sha256};

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn fit_tiers(args: TierArgs) -> Result<(), String> {
    if args.out.exists() {
        return Err("tier output directory already exists; choose a new directory".into());
    }
    if [&args.dataset_id, &args.source, &args.data_license]
        .iter()
        .any(|s| s.trim().is_empty())
    {
        return Err("dataset-id, source and data-license must be nonempty".into());
    }
    let task = match args.task.as_deref() {
        None => None,
        Some("regression") => Some(Task::Regression),
        Some("classification") => Some(Task::Classification),
        _ => return Err("--task must be regression or classification".into()),
    };
    let raw = std::fs::read(&args.csv).map_err(|e| e.to_string())?;
    let table = RawTable::from_reader(raw.as_slice()).map_err(|e| e.to_string())?;
    let target = table
        .column_index(&args.target)
        .map_err(|e| e.to_string())?;
    let dev = dope_core::device::parse(&args.device).map_err(|e| e.to_string())?;
    let mut artifacts = Vec::new();
    let mut inventory = Vec::new();
    for (name, size) in [
        ("small", ModelSize::Small),
        ("medium", ModelSize::Medium),
        ("large", ModelSize::Large),
    ] {
        let fitted = fit(
            &table,
            target,
            &FitOptions {
                size: Some(size),
                seed: args.seed,
                task,
                device: dev.clone(),
                ..Default::default()
            },
        )
        .map_err(|e| format!("{name}: {e}; hint: {}", e.repair_hint()))?;
        let file = format!("{name}.dope");
        inventory.push(json!({
            "size": name, "file": file, "budget_bytes": size.budget(),
            "bytes": fitted.bytes.len(), "sha256": sha256(&fitted.bytes),
            "parameters": fitted.report.parameters, "self_audit": fitted.report.self_audit,
            "task": fitted.report.task, "training_rows": fitted.report.rows,
        }));
        artifacts.push((file, fitted.bytes));
    }
    // All fits must succeed before creating output. Catalog is the completion marker.
    let executable = std::fs::read(std::env::current_exe().map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let catalog = json!({
        "schema_version": 1, "kind": "per-dataset-pretrained-generators",
        "dataset_id": args.dataset_id, "source": args.source, "data_license": args.data_license,
        "training_csv_sha256": sha256(&raw), "seed": args.seed,
        "target_index": target, "requested_task": args.task,
        "engine_version": env!("CARGO_PKG_VERSION"), "engine_binary_sha256": sha256(&executable),
        "device": dope_core::device::describe(&dev),
        "qualification": "training-self-audit-only", "release_authorized": false,
        "artifacts": inventory,
    });
    std::fs::create_dir(&args.out).map_err(|e| e.to_string())?;
    for (file, bytes) in artifacts {
        std::fs::write(args.out.join(file), bytes).map_err(|e| e.to_string())?;
    }
    std::fs::write(args.out.join("MODEL_CARD.md"),
        "# DOPE pretrained dataset generators\n\nEach artifact is fitted to this catalog's dataset and target. It can be sampled immediately, without training or access to the source CSV. It is not a general model for unseen tables.\n\nSmall: 2 KiB; medium: 4 KiB; large: 16 KiB maximum whole-file size. Actual size and full-generator scalar counts are in catalog.json. Larger budgets offer capacity, not guaranteed lift.\n\nTraining self-audits are development estimates, not external test results, privacy certification or release approval. Review data rights, leakage, category labels, quantiles and external transfer results before public release. Data license is recorded separately; it does not establish weight licensing or permission to redistribute.\n\nVerify the artifact SHA-256 against catalog.json, then run:\n\n```bash\ndope inspect small.dope --json\ndope sample small.dope -n 10000 -o synthetic.csv\n```\n")
        .map_err(|e| e.to_string())?;
    std::fs::write(
        args.out.join("catalog.json"),
        serde_json::to_vec_pretty(&catalog).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "compiled small/medium/large artifacts and catalog into {}",
        args.out.display()
    );
    Ok(())
}

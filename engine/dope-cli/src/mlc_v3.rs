//! `dope mlc-v3` CLI commands: 100% pure-Rust SafeTensors inference, pinned SHA-256 weight
//! resolution, and `DML3`/`CSC1` synthetic sampling.

use crate::style::{
    print_mlc_v3_sample_report, print_mlc_v3_weights_report, print_mlc_v3_wire_report, Style,
};
use clap::{Args, Subcommand};
use dope_core::mlc_v3::{
    sample_synthetic_table_with_draws, Dml3Envelope, FixedSampleDraws, MlcV3ForwardCheckInput,
    MlcV3ForwardCheckOutput, MlcV3Model,
};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) const DEFAULT_RELEASE_TAG: &str = "v1.0.0";
pub(crate) const REPO_SLUG: &str = "neverhuman/dope";

#[derive(Clone, Copy, Debug)]
pub(crate) struct PinnedAsset {
    pub(crate) preset: &'static str,
    pub(crate) filename: &'static str,
    pub(crate) bytes: usize,
    pub(crate) sha256: &'static str,
    pub(crate) kind: &'static str,
}

pub(crate) const PINNED_WEIGHTS: [PinnedAsset; 2] = [
    PinnedAsset {
        preset: "multitask",
        filename: "dope_mlc_v3_multitask_best.safetensors",
        bytes: 1_781_800,
        sha256: "d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c",
        kind: "safetensors-weights",
    },
    PinnedAsset {
        preset: "maxtrain97",
        filename: "dope_mlc_v3_maxtrain97_best.safetensors",
        bytes: 1_781_800,
        sha256: "9f7b5560e930e929391b30791fec5d0ed6441974e7f7ac44456167f0e9b334ef",
        kind: "safetensors-weights",
    },
];

pub(crate) const PINNED_SAMPLE_WIRE: PinnedAsset = PinnedAsset {
    preset: "sample-wire",
    filename: "packet_high_regression.dml3",
    bytes: 2_327,
    sha256: "975d3504e01f526c5aaee0b2094ba89ac981bf8333589cc0c777350b0acc756b",
    kind: "dml3-wire-fixture",
};

#[derive(Subcommand)]
pub(crate) enum MlcV3Command {
    /// Fetch and verify pinned DOPE-MLC-v3 .safetensors weights into local cache.
    FetchWeights(FetchWeightsArgs),
    /// Verify and inspect a DOPE-MLC-v3 .safetensors weight file.
    InspectWeights(InspectWeightsArgs),
    /// Verify and inspect a DML3/CSC1 or DML2 wire packet.
    InspectWire(InspectWireArgs),
    /// Run LandmarkHyperRouter (and optional TargetNet/CovNet forward check).
    Route(RouteArgs),
    /// Sample a synthetic CSV table from a DML3/CSC1 wire packet and .safetensors weights.
    Sample(SampleV3Args),
    /// Encode a CSV table into 629-D DOPE-MLC-v3 features with optional/progressive landmarks.
    Encode(EncodeV3Args),
}

#[derive(Args)]
pub(crate) struct FetchWeightsArgs {
    #[arg(long, default_value = "multitask", value_parser = ["multitask", "maxtrain97", "all"])]
    pub(crate) preset: String,
    #[arg(long, default_value = DEFAULT_RELEASE_TAG)]
    pub(crate) tag: String,
    #[arg(long)]
    pub(crate) from_dir: Option<PathBuf>,
    #[arg(long)]
    pub(crate) out_dir: Option<PathBuf>,
    #[arg(long)]
    pub(crate) no_sample_wire: bool,
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Args)]
pub(crate) struct InspectWeightsArgs {
    #[arg(long)]
    pub(crate) weights: Option<PathBuf>,
    #[arg(long, default_value = "multitask", value_parser = ["multitask", "maxtrain97"])]
    pub(crate) preset: String,
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Args)]
pub(crate) struct InspectWireArgs {
    #[arg(long)]
    pub(crate) wire: PathBuf,
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Args)]
pub(crate) struct RouteArgs {
    #[arg(long)]
    pub(crate) weights: Option<PathBuf>,
    #[arg(long, default_value = "multitask", value_parser = ["multitask", "maxtrain97"])]
    pub(crate) preset: String,
    #[arg(long, value_parser = ["low", "small", "medium", "high", "ultra"])]
    pub(crate) level: String,
    #[arg(long)]
    pub(crate) conditioning_json: PathBuf,
    #[arg(long)]
    pub(crate) forward_input_json: Option<PathBuf>,
}

#[derive(Args)]
pub(crate) struct SampleV3Args {
    #[arg(long)]
    pub(crate) weights: Option<PathBuf>,
    #[arg(long, default_value = "multitask", value_parser = ["multitask", "maxtrain97"])]
    pub(crate) preset: String,
    #[arg(long)]
    pub(crate) wire: PathBuf,
    #[arg(short = 'n', long, default_value_t = 0)]
    pub(crate) rows: usize,
    #[arg(long, default_value_t = 20261023)]
    pub(crate) seed: u64,
    #[arg(long, default_value = "raw", value_parser = ["raw", "scaled"])]
    pub(crate) space: String,
    #[arg(long)]
    pub(crate) draws_json: Option<PathBuf>,
    #[arg(long)]
    pub(crate) json_out: Option<PathBuf>,
    #[arg(short, long)]
    pub(crate) out: Option<PathBuf>,
}

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
}

fn non_empty_env(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn default_weights_dir() -> Result<PathBuf, String> {
    if let Some(dir) = non_empty_env("DOPE_WEIGHTS_DIR") {
        return Ok(dir);
    }
    if let Some(xdg) = non_empty_env("XDG_CACHE_HOME") {
        return Ok(xdg.join("dope/weights"));
    }
    if let Some(home) = non_empty_env("HOME") {
        return Ok(home.join(".cache/dope/weights"));
    }
    Err("could not determine weights cache directory; set DOPE_WEIGHTS_DIR".into())
}

fn require_regular_file(path: &Path, label: &str) -> Result<(), String> {
    let d = path.display();
    let meta = std::fs::symlink_metadata(path).map_err(|e| format!("cannot stat {d}: {e}"))?;
    if meta.file_type().is_symlink() || !meta.file_type().is_file() {
        return Err(format!("refusing non-regular or symlinked {label}: {d}"));
    }
    Ok(())
}

fn verify_regular_file_digest(
    path: &Path,
    exp_len: usize,
    exp_sha: &str,
) -> Result<String, String> {
    require_regular_file(path, "asset path")?;
    let d = path.display();
    let data = std::fs::read(path).map_err(|e| format!("cannot read {d}: {e}"))?;
    if data.len() != exp_len {
        let n = data.len();
        return Err(format!(
            "asset byte length mismatch for {d}: expected {exp_len} B, got {n} B"
        ));
    }
    let sha = format!("{:x}", Sha256::digest(&data));
    if sha != exp_sha {
        return Err(format!(
            "asset SHA-256 mismatch for {d}: expected {exp_sha}, got {sha}"
        ));
    }
    Ok(sha)
}

fn resolve_weights_path(explicit: &Option<PathBuf>, preset: &str) -> Result<PathBuf, String> {
    if let Some(p) = explicit {
        return Ok(p.clone());
    }
    let spec = PINNED_WEIGHTS.iter().find(|w| w.preset == preset);
    let spec = spec.ok_or_else(|| format!("unknown preset `{preset}`"))?;
    let mut cands = Vec::new();
    if let Some(dir) = non_empty_env("DOPE_WEIGHTS_DIR") {
        cands.push(dir.join(spec.filename));
    }
    if let Ok(cache_dir) = default_weights_dir() {
        cands.push(cache_dir.join(spec.filename));
    }
    if let Some(parent) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        cands.push(parent.join(spec.filename));
        if let Some(prefix) = parent.parent() {
            cands.push(prefix.join("share/dope/weights").join(spec.filename));
        }
    }
    cands.push(PathBuf::from(spec.filename));
    cands.push(PathBuf::from("target/model-release-v1").join(spec.filename));
    for cand in cands {
        if cand.exists() {
            verify_regular_file_digest(&cand, spec.bytes, spec.sha256)?;
            return Ok(cand);
        }
    }
    let (f, b, h) = (spec.filename, spec.bytes, &spec.sha256[..16]);
    Err(format!("pinned DOPE-MLC-v3 weights `{f}` ({b} B, sha256: {h}…) not found in local cache.\n  hint: run `dope mlc-v3 fetch-weights --preset {preset}` or pass `--weights <PATH>`"))
}

fn validate_release_tag(tag: &str) -> Result<(), String> {
    let t = tag.trim();
    let ok_chars = t
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'));
    if t.is_empty() || t.eq_ignore_ascii_case("latest") || t.contains("..") || !ok_chars {
        return Err(
            "release tag must be a concrete version tag (e.g. v1.0.0), not 'latest'".into(),
        );
    }
    Ok(())
}

fn fetch_one_asset(
    asset: &PinnedAsset,
    tag: &str,
    from_dir: Option<&Path>,
    out_dir: &Path,
) -> Result<serde_json::Value, String> {
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;
    let dest = out_dir.join(asset.filename);
    if std::fs::symlink_metadata(&dest).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(format!(
            "refusing to overwrite symlink at destination: {}",
            dest.display()
        ));
    }
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let tmp = out_dir.join(format!(
        ".{}.tmp.{}.{}",
        asset.filename,
        std::process::id(),
        ns.as_nanos()
    ));
    let source_desc = if let Some(src_dir) = from_dir {
        let src = src_dir.join(asset.filename);
        require_regular_file(&src, "source asset")?;
        std::fs::copy(&src, &tmp).map_err(|e| format!("failed copying {}: {e}", src.display()))?;
        format!("local-dir:{}", src.display())
    } else {
        let url = format!(
            "https://github.com/{REPO_SLUG}/releases/download/{tag}/{}",
            asset.filename
        );
        let flags = [
            "-fsSL",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--retry",
            "2",
            "-o",
        ];
        let st = Command::new("curl")
            .args(flags)
            .arg(&tmp)
            .arg(&url)
            .status();
        let st = st.map_err(|e| format!("failed to launch curl for {url}: {e}"))?;
        if !st.success() {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!("download failed for {url} (exit {st})"));
        }
        url
    };
    if let Err(err) = verify_regular_file_digest(&tmp, asset.bytes, asset.sha256) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    if asset.kind == "safetensors-weights" {
        if let Err(e) = MlcV3Model::load_safetensors(&tmp) {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!(
                "downloaded weights failed SafeTensors structural verification: {e}"
            ));
        }
    } else if asset.kind == "dml3-wire-fixture" {
        let raw = std::fs::read(&tmp).map_err(|e| e.to_string())?;
        if let Err(e) = Dml3Envelope::from_bytes(&raw) {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!(
                "downloaded wire packet failed DML3 verification: {e}"
            ));
        }
    }
    std::fs::rename(&tmp, &dest).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("atomic rename to {} failed: {e}", dest.display())
    })?;
    Ok(serde_json::json!({
        "preset": asset.preset, "filename": asset.filename, "kind": asset.kind,
        "path": dest.display().to_string(), "bytes": asset.bytes, "sha256": asset.sha256,
        "release_tag": tag, "source": source_desc, "verification_level": "sha256-content-integrity",
    }))
}

pub(crate) fn run(cmd: MlcV3Command, style: Style) -> Result<(), String> {
    let map_err = |e: dope_core::DopeError| e.to_string();
    let print_json = |v: &serde_json::Value| -> Result<(), String> {
        println!(
            "{}",
            serde_json::to_string_pretty(v).map_err(|e| e.to_string())?
        );
        Ok(())
    };
    match cmd {
        MlcV3Command::FetchWeights(a) => {
            validate_release_tag(&a.tag)?;
            let out_dir = match &a.out_dir {
                Some(d) => d.clone(),
                None => default_weights_dir()?,
            };
            let mut targets: Vec<&PinnedAsset> = match a.preset.as_str() {
                "multitask" => vec![&PINNED_WEIGHTS[0]],
                "maxtrain97" => vec![&PINNED_WEIGHTS[1]],
                "all" => vec![&PINNED_WEIGHTS[0], &PINNED_WEIGHTS[1]],
                _ => return Err("--preset must be multitask, maxtrain97, or all".into()),
            };
            if !a.no_sample_wire {
                targets.push(&PINNED_SAMPLE_WIRE);
            }
            let results: Vec<serde_json::Value> = targets
                .into_iter()
                .map(|asset| fetch_one_asset(asset, &a.tag, a.from_dir.as_deref(), &out_dir))
                .collect::<Result<_, _>>()?;
            if a.json {
                let note = "SHA-256 checksums and SafeTensors/DML3 structural checks verify content integrity against the pinned release manifest; cryptographic publisher provenance requires Sigstore OIDC attestation verification.";
                print_json(&serde_json::json!({
                    "schema": "dope-mlc-v3-weights-fetch-v1", "release_tag": a.tag,
                    "out_dir": out_dir.display().to_string(),
                    "verification_level": "sha256-content-integrity",
                    "provenance_note": note, "assets": results,
                }))?;
            } else {
                let ok = style.green_bold("✓");
                let tg = style.cyan_bold(&a.tag);
                let dir = style.bold(out_dir.display().to_string());
                println!(
                    "{ok} verified {} asset(s) for release tag {tg} -> {dir}",
                    results.len()
                );
                for item in &results {
                    let f = item["filename"].as_str().unwrap_or("");
                    let b = item["bytes"].as_u64().unwrap_or(0);
                    let h = &item["sha256"].as_str().unwrap_or("")[..16];
                    println!("  {} {f:<42} {b:>9} B  sha256:{h}…", style.green("•"));
                }
                let vnote = "verification: sha256-content-integrity + SafeTensors/DML3 structural check (not a cryptographic signature/attestation)";
                println!("  {}", style.dim(vnote));
            }
        }
        MlcV3Command::InspectWeights(a) => {
            let weights_path = resolve_weights_path(&a.weights, &a.preset)?;
            let m = MlcV3Model::load_safetensors(&weights_path)
                .map_err(map_err)?
                .inventory;
            if !style.enabled || a.json {
                print_json(&serde_json::json!({
                    "format": "dope-mlc-v3-safetensors", "tensor_count": m.tensor_count,
                    "parameter_count": m.parameter_count, "byte_len": m.byte_len,
                    "sha256": m.sha256, "runtime": "rust-candle-cpu",
                }))?;
            } else {
                print_mlc_v3_weights_report(&weights_path, &m, style);
            }
        }
        MlcV3Command::InspectWire(a) => {
            let raw = std::fs::read(&a.wire).map_err(|e| e.to_string())?;
            let env = Dml3Envelope::from_bytes(&raw).map_err(map_err)?;
            let enc = &env.encoding;
            if !style.enabled || a.json {
                print_json(&serde_json::json!({
                    "wire_format": env.wire_format, "wire_bytes": env.wire_bytes,
                    "tier_cap_bytes": env.tier_cap_bytes, "within_tier_cap": env.within_tier_cap,
                    "crc32": env.crc32, "compressed_flags": env.compressed_flags,
                    "level": enc.level, "task": enc.task, "code_width": enc.code_width,
                    "n_train": enc.n_train, "p_use": enc.p_use,
                    "include_v3_primitives": enc.target_bundle.include_v3_primitives,
                    "has_csc1_scaler": env.scaler.is_some(), "k_classes": enc.target_bundle.k_classes,
                }))?;
            } else {
                print_mlc_v3_wire_report(&a.wire, &env, style);
            }
        }
        MlcV3Command::Route(a) => {
            let weights_path = resolve_weights_path(&a.weights, &a.preset)?;
            let model = MlcV3Model::load_safetensors(&weights_path).map_err(map_err)?;
            let raw_json =
                std::fs::read_to_string(&a.conditioning_json).map_err(|e| e.to_string())?;
            let vec256: Vec<f32> = serde_json::from_str(&raw_json).map_err(|e| e.to_string())?;
            let routed = model.router.forward(&vec256, &a.level).map_err(map_err)?;
            if let Some(fwd_path) = &a.forward_input_json {
                let fwd_raw = std::fs::read_to_string(fwd_path).map_err(|e| e.to_string())?;
                let fwd_in: MlcV3ForwardCheckInput =
                    serde_json::from_str(&fwd_raw).map_err(|e| e.to_string())?;
                let fwd_out = MlcV3ForwardCheckOutput::evaluate(
                    routed,
                    &model.target_net,
                    &model.cov_net,
                    &fwd_in,
                )
                .map_err(map_err)?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&fwd_out).map_err(|e| e.to_string())?
                );
            } else {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&routed).map_err(|e| e.to_string())?
                );
            }
        }
        MlcV3Command::Sample(a) => {
            let weights_path = resolve_weights_path(&a.weights, &a.preset)?;
            let model = MlcV3Model::load_safetensors(&weights_path).map_err(map_err)?;
            let wire_bytes = std::fs::read(&a.wire).map_err(|e| e.to_string())?;
            let env = Dml3Envelope::from_bytes(&wire_bytes).map_err(map_err)?;
            let draws: Option<FixedSampleDraws> = match &a.draws_json {
                Some(p) => {
                    let s = std::fs::read_to_string(p).map_err(|e| e.to_string())?;
                    Some(serde_json::from_str(&s).map_err(|e| e.to_string())?)
                }
                None => None,
            };
            let sampled =
                sample_synthetic_table_with_draws(&model, &env, a.rows, a.seed, draws.as_ref())
                    .map_err(map_err)?;
            if let Some(jp) = &a.json_out {
                let bytes = serde_json::to_vec_pretty(&sampled).map_err(|e| e.to_string())?;
                std::fs::write(jp, bytes).map_err(|e| e.to_string())?;
            }
            let (xs, ys) = if a.space == "scaled" {
                (&sampled.x_scaled, &sampled.y_scaled)
            } else {
                (&sampled.x_raw, &sampled.y_raw)
            };
            let mut csv_bytes = Vec::new();
            dope_core::table::write_csv(&mut csv_bytes, &sampled.headers, ys.len(), |r, c| {
                dope_core::schema::format_number(if c < xs[r].len() { xs[r][c] } else { ys[r] })
            })
            .map_err(map_err)?;
            match &a.out {
                Some(p) => {
                    std::fs::write(p, &csv_bytes).map_err(|e| e.to_string())?;
                    let nf = sampled.headers.len().saturating_sub(1);
                    print_mlc_v3_sample_report(&a.wire, p, ys.len(), nf, &a.space, &env, style);
                }
                None => std::io::stdout()
                    .write_all(&csv_bytes)
                    .map_err(|e| e.to_string())?,
            }
        }
        MlcV3Command::Encode(a) => run_encode(a)?,
    }
    Ok(())
}

fn encode_cond_on_model(
    model: &MlcV3Model,
    cond: &dope_core::prepare::DopeConditioning,
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

fn run_encode(a: EncodeV3Args) -> Result<(), String> {
    use dope_core::landmark::{
        LandmarkBundle, LandmarkMode, LandmarkOptions, SubsampleSpec, PROGRESSIVE_ORDER,
    };
    use dope_core::prepare::{
        build_conditioning, build_conditioning_sampled, prepare_table_cells,
        progressive_landmarks_for_table, NativeReportView, PrepTask,
    };
    use dope_core::schema::{infer, IntakeOptions, Task};
    use dope_core::RawTable;

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
    let mut x_cells = vec![Vec::with_capacity(ncols.saturating_sub(1)); nrows];
    for (col_idx, col) in raw.columns.iter().enumerate() {
        if col_idx == target_idx {
            continue;
        }
        for (r, val) in col.iter().enumerate() {
            x_cells[r].push(val.clone());
        }
    }
    let y_cells = raw.columns[target_idx].clone();
    let prep_task = match a.task.as_deref() {
        Some("regression") => PrepTask::Regression,
        Some("multiclass") => PrepTask::Multiclass,
        Some("classification") => {
            let uniq: std::collections::BTreeSet<&str> = y_cells
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
                    let uniq: std::collections::BTreeSet<&str> = y_cells
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

    let prepared = prepare_table_cells(&x_cells, &y_cells, prep_task).map_err(map_err)?;
    let supplied_bundle: Option<LandmarkBundle> = match &a.bundle_json {
        Some(p) => {
            let s = std::fs::read_to_string(p).map_err(|e| e.to_string())?;
            Some(serde_json::from_str(&s).map_err(|e| e.to_string())?)
        }
        None => None,
    };

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
        }
    }
    Ok(())
}

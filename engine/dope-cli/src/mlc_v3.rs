//! `dope mlc-v3` CLI commands: 100% pure-Rust SafeTensors inference, pinned SHA-256 weight
//! resolution, and `DML3`/`CSC1` synthetic sampling.

use crate::style::Style;
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
    /// Fetch and verify pinned DOPE-MLC-v3 .safetensors weights (and optional sample .dml3 packet) into local cache.
    FetchWeights(FetchWeightsArgs),
    /// Verify and inspect a DOPE-MLC-v3 .safetensors weight file (56 F32 tensors, 444,168 parameters).
    InspectWeights(InspectWeightsArgs),
    /// Verify and inspect a DML3/CSC1 or DML2 wire packet.
    InspectWire(InspectWireArgs),
    /// Run LandmarkHyperRouter (and optional TargetNet/CovNet forward check) using .safetensors weights.
    Route(RouteArgs),
    /// Sample a synthetic CSV table from a DML3/CSC1 wire packet and .safetensors weights in 100% Rust.
    Sample(SampleV3Args),
}

#[derive(Args)]
pub(crate) struct FetchWeightsArgs {
    /// Weight preset to fetch: multitask, maxtrain97, or all.
    #[arg(long, default_value = "multitask", value_parser = ["multitask", "maxtrain97", "all"])]
    pub(crate) preset: String,
    /// Concrete release tag resolved once for all assets (never floating 'latest').
    #[arg(long, default_value = DEFAULT_RELEASE_TAG)]
    pub(crate) tag: String,
    /// Copy and verify from a local staged release directory instead of downloading over HTTPS.
    #[arg(long)]
    pub(crate) from_dir: Option<PathBuf>,
    /// Destination directory for verified weights (defaults to $DOPE_WEIGHTS_DIR or ~/.cache/dope/weights).
    #[arg(long)]
    pub(crate) out_dir: Option<PathBuf>,
    /// Skip fetching the companion 2,327-byte packet_high_regression.dml3 example wire packet.
    #[arg(long)]
    pub(crate) no_sample_wire: bool,
    /// Emit structured JSON summary on stdout.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Args)]
pub(crate) struct InspectWeightsArgs {
    /// Path to .safetensors weights; if omitted, resolves pinned preset from local cache/install.
    #[arg(long)]
    pub(crate) weights: Option<PathBuf>,
    /// Preset when --weights is omitted: multitask or maxtrain97.
    #[arg(long, default_value = "multitask", value_parser = ["multitask", "maxtrain97"])]
    pub(crate) preset: String,
    /// Emit structured JSON summary on stdout even in an interactive terminal.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Args)]
pub(crate) struct InspectWireArgs {
    #[arg(long)]
    pub(crate) wire: PathBuf,
    /// Emit structured JSON summary on stdout even in an interactive terminal.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Args)]
pub(crate) struct RouteArgs {
    /// Path to .safetensors weights; if omitted, resolves pinned preset from local cache/install.
    #[arg(long)]
    pub(crate) weights: Option<PathBuf>,
    /// Preset when --weights is omitted: multitask or maxtrain97.
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
    /// Path to .safetensors weights; if omitted, resolves pinned preset from local cache/install.
    #[arg(long)]
    pub(crate) weights: Option<PathBuf>,
    /// Preset when --weights is omitted: multitask or maxtrain97.
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

fn default_weights_dir() -> Result<PathBuf, String> {
    if let Some(dir) = std::env::var_os("DOPE_WEIGHTS_DIR").filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(xdg).join("dope").join("weights"));
    }
    if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(home).join(".cache/dope/weights"));
    }
    Err(
        "could not determine weights cache directory; set DOPE_WEIGHTS_DIR or pass --weights"
            .into(),
    )
}

fn verify_regular_file_digest(
    path: &Path,
    expected_bytes: usize,
    expected_sha256: &str,
) -> Result<String, String> {
    let meta = std::fs::symlink_metadata(path)
        .map_err(|e| format!("cannot stat {}: {e}", path.display()))?;
    if meta.file_type().is_symlink() || !meta.file_type().is_file() {
        return Err(format!(
            "refusing non-regular or symlinked asset path: {}",
            path.display()
        ));
    }
    let data = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if data.len() != expected_bytes {
        return Err(format!(
            "asset byte length mismatch for {}: expected {} B, got {} B",
            path.display(),
            expected_bytes,
            data.len()
        ));
    }
    let actual_sha = format!("{:x}", Sha256::digest(&data));
    if actual_sha != expected_sha256 {
        return Err(format!(
            "asset SHA-256 mismatch for {}: expected {}, got {}",
            path.display(),
            expected_sha256,
            actual_sha
        ));
    }
    Ok(actual_sha)
}

fn resolve_weights_path(explicit: &Option<PathBuf>, preset: &str) -> Result<PathBuf, String> {
    if let Some(p) = explicit {
        return Ok(p.clone());
    }
    let spec = PINNED_WEIGHTS
        .iter()
        .find(|w| w.preset == preset)
        .ok_or_else(|| format!("unknown preset `{preset}`"))?;
    let mut candidates = Vec::new();
    if let Some(dir) = std::env::var_os("DOPE_WEIGHTS_DIR").filter(|v| !v.is_empty()) {
        candidates.push(PathBuf::from(dir).join(spec.filename));
    }
    if let Ok(cache_dir) = default_weights_dir() {
        candidates.push(cache_dir.join(spec.filename));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join(spec.filename));
            if let Some(prefix) = parent.parent() {
                candidates.push(prefix.join("share/dope/weights").join(spec.filename));
            }
        }
    }
    candidates.push(PathBuf::from(spec.filename));
    candidates.push(PathBuf::from("target/model-release-v1").join(spec.filename));
    for cand in candidates {
        if cand.exists() {
            verify_regular_file_digest(&cand, spec.bytes, spec.sha256)?;
            return Ok(cand);
        }
    }
    Err(format!(
        "pinned DOPE-MLC-v3 weights `{}` ({} B, sha256: {}…) not found in local cache.\n  hint: run `dope mlc-v3 fetch-weights --preset {}` or pass `--weights <PATH>`",
        spec.filename, spec.bytes, &spec.sha256[..16], preset
    ))
}

fn validate_release_tag(tag: &str) -> Result<(), String> {
    let t = tag.trim();
    if t.is_empty()
        || t.eq_ignore_ascii_case("latest")
        || t.contains('/')
        || t.contains("..")
        || !t
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
    {
        return Err("release tag must be a concrete version tag (e.g. v1.0.0 or dope-engine-v1.0.0), not 'latest'".into());
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
    if std::fs::symlink_metadata(&dest)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
    {
        return Err(format!(
            "refusing to overwrite symlink at destination: {}",
            dest.display()
        ));
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = out_dir.join(format!(
        ".{}.tmp.{}.{nanos}",
        asset.filename,
        std::process::id()
    ));
    let source_desc = if let Some(src_dir) = from_dir {
        let src = src_dir.join(asset.filename);
        let meta = std::fs::symlink_metadata(&src)
            .map_err(|e| format!("missing source asset {}: {e}", src.display()))?;
        if meta.file_type().is_symlink() || !meta.file_type().is_file() {
            return Err(format!(
                "refusing non-regular or symlinked source asset: {}",
                src.display()
            ));
        }
        std::fs::copy(&src, &tmp)
            .map_err(|e| format!("failed copying {} to {}: {e}", src.display(), tmp.display()))?;
        format!("local-dir:{}", src.display())
    } else {
        let url = format!(
            "https://github.com/{REPO_SLUG}/releases/download/{tag}/{}",
            asset.filename
        );
        let status = Command::new("curl")
            .args([
                "-fsSL",
                "--proto",
                "=https",
                "--tlsv1.2",
                "--retry",
                "2",
                "-o",
            ])
            .arg(&tmp)
            .arg(&url)
            .status()
            .map_err(|e| format!("failed to launch curl for {url}: {e}"))?;
        if !status.success() {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!("download failed for {url} (exit {status})"));
        }
        url
    };
    if let Err(err) = verify_regular_file_digest(&tmp, asset.bytes, asset.sha256) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    if asset.kind == "safetensors-weights" {
        if let Err(err) = MlcV3Model::load_safetensors(&tmp) {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!(
                "downloaded weights failed SafeTensors structural verification: {err}"
            ));
        }
    } else if asset.kind == "dml3-wire-fixture" {
        let raw = std::fs::read(&tmp).map_err(|e| e.to_string())?;
        if let Err(err) = Dml3Envelope::from_bytes(&raw) {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!(
                "downloaded wire packet failed DML3 verification: {err}"
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
                let report = serde_json::json!({
                    "schema": "dope-mlc-v3-weights-fetch-v1", "release_tag": a.tag,
                    "out_dir": out_dir.display().to_string(), "verification_level": "sha256-content-integrity",
                    "provenance_note": "SHA-256 checksums and SafeTensors/DML3 structural checks verify content integrity against the pinned release manifest; cryptographic publisher provenance requires Sigstore OIDC attestation verification.",
                    "assets": results,
                });
                print_json(&report)?;
            } else {
                println!(
                    "{} verified {} asset(s) for release tag {} -> {}",
                    style.green_bold("✓"),
                    results.len(),
                    style.cyan_bold(&a.tag),
                    style.bold(out_dir.display().to_string())
                );
                for item in &results {
                    println!(
                        "  {} {:<42} {:>9} B  sha256:{}…",
                        style.green("•"),
                        item["filename"].as_str().unwrap_or(""),
                        item["bytes"].as_u64().unwrap_or(0),
                        &item["sha256"].as_str().unwrap_or("")[..16]
                    );
                }
                let note = "verification: sha256-content-integrity + SafeTensors/DML3 structural check (not a cryptographic signature/attestation)";
                println!("  {}", style.dim(note));
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
                println!(
                    "{} {}  {} bytes ({} tensors · {} parameters)",
                    style.cyan_bold("●"),
                    style.bold(weights_path.display().to_string()),
                    style.green_bold(crate::style::group(m.byte_len)),
                    style.bold(crate::style::group(m.tensor_count)),
                    style.bold(crate::style::group(m.parameter_count)),
                );
                println!("  {}       dope-mlc-v3-safetensors", style.dim("format"));
                println!(
                    "  {}      {}",
                    style.dim("runtime"),
                    style.green_bold("rust-candle-cpu · 0 Python/LibTorch deps")
                );
                println!("  {}       {}", style.dim("sha256"), m.sha256);
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
                println!(
                    "{} {}  {} / {} bytes ({} · tier `{}`) {}",
                    style.cyan_bold("●"),
                    style.bold(a.wire.display().to_string()),
                    style.green_bold(crate::style::group(env.wire_bytes)),
                    crate::style::group(env.tier_cap_bytes),
                    style.bold(&env.wire_format),
                    style.cyan_bold(&enc.level),
                    style.budget_bar(env.wire_bytes, env.tier_cap_bytes, 12),
                );
                println!(
                    "  {}         {} (k_classes = {}) · {} train rows × {} features · {}",
                    style.dim("task"),
                    style.bold(&enc.task),
                    enc.target_bundle.k_classes,
                    crate::style::group(enc.n_train),
                    enc.p_use,
                    style.green("0 source rows stored"),
                );
                println!(
                    "  {}     code_width = {} · v3_primitives = {} · csc1_scaler = {}",
                    style.dim("encoding"),
                    enc.code_width,
                    enc.target_bundle.include_v3_primitives,
                    env.scaler.is_some(),
                );
                let crc_str = env
                    .crc32
                    .map_or_else(|| "none".to_string(), |c| format!("0x{c:08x}"));
                println!(
                    "  {}    within_tier_cap = {} · crc32 = {} · flags = 0x{:02x}",
                    style.dim("integrity"),
                    if env.within_tier_cap {
                        style.green_bold("true")
                    } else {
                        style.red_bold("false")
                    },
                    crc_str,
                    env.compressed_flags,
                );
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
                Some(p) => Some(
                    serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?,
                ),
                None => None,
            };
            let sampled =
                sample_synthetic_table_with_draws(&model, &env, a.rows, a.seed, draws.as_ref())
                    .map_err(map_err)?;
            if let Some(jp) = &a.json_out {
                std::fs::write(
                    jp,
                    serde_json::to_vec_pretty(&sampled).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
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
                    if style.enabled {
                        println!(
                            "{} sampled {} synthetic rows ({} features + target, {} space) from {} ({} · {} · {} B) -> {}",
                            style.green_bold("✓"),
                            style.bold(crate::style::group(ys.len())),
                            sampled.headers.len().saturating_sub(1),
                            style.bold(&a.space),
                            style.bold(a.wire.display().to_string()),
                            &env.wire_format,
                            style.cyan_bold(&env.encoding.level),
                            crate::style::group(env.wire_bytes),
                            style.cyan_bold(p.display().to_string()),
                        );
                    }
                }
                None => std::io::stdout()
                    .write_all(&csv_bytes)
                    .map_err(|e| e.to_string())?,
            }
        }
    }
    Ok(())
}

//! Pure-Rust `DOPEMAT1` CLI for DOPE v3 (`DSE8` v1 + FIT-only Evidence Bank + `SingleEncV8`).
//!
//! Drop-in `dope_kpi emit-cli` external arm and parity/inspection CLI:
//! - `dope_v8_emit fit --in A.mat --task regression|binary|multiclass --budget 2048 [--normalized-only] [--checkpoint w.safetensors] [--code-tier 1] [--skew-t] --out m.bin`
//! - `dope_v8_emit sample --model m.bin [--checkpoint w.safetensors] [--n-heads 4] --rows N --seed S --out S.mat`
//! - `dope_v8_emit evidence --in A.mat --task regression|binary|multiclass [--tier t0|t1|t2|t3] [--seed S] --out ev.json`
//! - `dope_v8_emit embed --model m.bin --out emb.json`
//! - `dope_v8_emit truncate --model m.bin --budget 1024 --out m_trunc.bin`
//! - `dope_v8_emit inspect --model m.bin`

use clap::{Parser, Subcommand};
use dope_core::evidence::{extract as extract_evidence, EvidenceInput, EvidenceTier};
use dope_core::single_encoding::v8::{
    decode_dse8_rows, extract_jopedime_embedding, fit_dse8_packet, fit_dse8_packet_with_encoder,
    fit_unit_meta, truncate_dse8_packet, unpack_dse8_packet, Dse8ColumnMeta, Dse8FitOptions,
    SingleEncV8Weights,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "dope_v8_emit",
    about = "Pure-Rust DSE8 v1 + FIT-only Evidence Bank + SingleEncV8 CLI"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Fit a strict `<= budget` CRC32-verified DSE8 v1 wire packet from A.mat.
    Fit {
        #[arg(long = "in")]
        input: PathBuf,
        #[arg(long)]
        task: String,
        #[arg(long, default_value_t = 0)]
        classes: usize,
        #[arg(long, default_value_t = 2048)]
        budget: usize,
        #[arg(long, default_value_t = false)]
        normalized_only: bool,
        #[arg(long, default_value_t = false)]
        no_trees: bool,
        #[arg(long, default_value_t = false)]
        no_regimes: bool,
        #[arg(long, default_value_t = false)]
        no_effects: bool,
        #[arg(long, default_value_t = false)]
        no_resid: bool,
        #[arg(long, default_value_t = false)]
        skew_t: bool,
        #[arg(long, default_value_t = 0)]
        code_tier: u8,
        #[arg(long = "checkpoint", visible_alias = "weights")]
        checkpoint: Option<PathBuf>,
        #[arg(long, default_value_t = 4)]
        n_heads: usize,
        #[arg(long = "col-meta")]
        col_meta: Option<PathBuf>,
        /// Without `--col-meta`: compute FIT-only column meta in the packet's unit space
        /// (`fit_unit_meta`, the same meta the V8 training episodes use).
        #[arg(long, default_value_t = false, conflicts_with = "col_meta")]
        fit_meta: bool,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Decode synthetic rows (`DOPEMAT1`, target last, labels sampled) from a DSE8 packet.
    Sample {
        #[arg(long = "model", visible_alias = "packet")]
        model: PathBuf,
        #[arg(long = "checkpoint", visible_alias = "weights")]
        checkpoint: Option<PathBuf>,
        #[arg(long, default_value_t = 4)]
        n_heads: usize,
        #[arg(long)]
        rows: usize,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Extract FIT-only Evidence tensors (JSON) for parity checks & verification.
    Evidence {
        #[arg(long = "in")]
        input: PathBuf,
        #[arg(long)]
        task: String,
        #[arg(long, default_value_t = 0)]
        classes: usize,
        #[arg(long, default_value = "t2")]
        tier: String,
        #[arg(long, default_value_t = false)]
        target_free: bool,
        #[arg(long = "col-meta")]
        col_meta: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Export deterministic 64-D JopeDime embedding from a DSE8 packet.
    Embed {
        #[arg(long = "model", visible_alias = "packet")]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Nested-prefix truncate a DSE8 packet to `<= budget` without refitting.
    Truncate {
        #[arg(long = "model", visible_alias = "packet")]
        model: PathBuf,
        #[arg(long)]
        budget: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Inspect value-free DSE8 packet header & section layout as JSON.
    Inspect {
        #[arg(long = "model", visible_alias = "packet")]
        model: PathBuf,
    },
}

use dope_arena::dopemat::{read_dopemat, write_atomic, write_dopemat};

fn parse_task(task: &str) -> Result<u8, String> {
    match task {
        "regression" => Ok(0),
        "binary" => Ok(1),
        "multiclass" => Ok(2),
        _ => Err("unsupported_task".into()),
    }
}

fn load_col_meta(path: Option<&Path>) -> Result<Option<Vec<Dse8ColumnMeta>>, String> {
    let Some(p) = path else {
        return Ok(None);
    };
    let s = fs::read_to_string(p).map_err(|e| format!("read_meta_failed: {e}"))?;
    let meta: Vec<Dse8ColumnMeta> =
        serde_json::from_str(&s).map_err(|e| format!("parse_meta_failed: {e}"))?;
    Ok(Some(meta))
}

fn main() {
    let cli = Cli::parse();
    let res = match cli.cmd {
        Cmd::Fit {
            input,
            task,
            classes: _,
            budget,
            normalized_only,
            no_trees,
            no_regimes,
            no_effects,
            no_resid,
            skew_t,
            code_tier,
            checkpoint,
            n_heads,
            col_meta,
            fit_meta,
            seed,
            out,
        } => (|| -> Result<(), String> {
            let task_id = parse_task(&task)?;
            let (x, y) = read_dopemat(&input)?;
            let meta = if fit_meta {
                Some(
                    fit_unit_meta(&x, &y, task_id, normalized_only)
                        .map_err(|e| format!("fit_meta_failed: {e}"))?,
                )
            } else {
                load_col_meta(col_meta.as_deref())?
            };
            let opts = Dse8FitOptions {
                task_id,
                budget_bytes: budget,
                normalized_only,
                enable_trees: !no_trees,
                enable_regimes: !no_regimes,
                enable_effects: !no_effects,
                enable_residual_model: !no_resid,
                enable_skew_t: skew_t,
                code_tier,
                col_meta: meta.as_deref(),
                seed,
            };
            let fit_res = if let Some(wp) = checkpoint {
                let wb = fs::read(&wp).map_err(|e| format!("read_ckpt_failed: {e}"))?;
                let w = SingleEncV8Weights::from_safetensors_bytes(&wb, n_heads)
                    .map_err(|e| format!("load_ckpt_failed: {e}"))?;
                fit_dse8_packet_with_encoder(&x, &y, &opts, &w)
            } else {
                fit_dse8_packet(&x, &y, &opts)
            };
            match fit_res {
                Ok(pkt_bytes) => {
                    write_atomic(&out, &pkt_bytes)?;
                    let sha = format!("{:x}", Sha256::digest(&pkt_bytes));
                    println!(
                        "{}",
                        serde_json::json!({
                            "status": "ok",
                            "format": "DSE8v1",
                            "bytes": pkt_bytes.len(),
                            "packet_bytes": pkt_bytes.len(),
                            "packet_sha256": sha,
                        })
                    );
                    Ok(())
                }
                Err(dope_core::DopeError::Budget(_)) => {
                    println!("{}", serde_json::json!({"status": "over_budget"}));
                    Ok(())
                }
                Err(e) => Err(format!("fit_failed: {e}")),
            }
        })(),
        Cmd::Sample {
            model,
            checkpoint,
            n_heads,
            rows,
            seed,
            out,
        } => (|| -> Result<(), String> {
            let pkt_bytes = fs::read(&model).map_err(|e| format!("read_model_failed: {e}"))?;
            let pkt = unpack_dse8_packet(&pkt_bytes).map_err(|e| format!("unpack_failed: {e}"))?;
            let weights = match checkpoint {
                Some(wp) => {
                    let wb = fs::read(&wp).map_err(|e| format!("read_ckpt_failed: {e}"))?;
                    Some(
                        SingleEncV8Weights::from_safetensors_bytes(&wb, n_heads)
                            .map_err(|e| format!("load_ckpt_failed: {e}"))?,
                    )
                }
                None => None,
            };
            let tbl = decode_dse8_rows(&pkt, weights.as_ref(), rows, seed)
                .map_err(|e| format!("decode_failed: {e}"))?;
            let x_cols = tbl.x.len();
            let mut body = Vec::with_capacity(rows * (x_cols + 1));
            for r in 0..rows {
                for c in 0..x_cols {
                    body.push(tbl.x[c][r] as f32);
                }
                body.push(tbl.y[r] as f32);
            }
            write_dopemat(&out, rows, x_cols, &body)
        })(),
        Cmd::Evidence {
            input,
            task,
            classes,
            tier,
            target_free,
            col_meta,
            seed,
            out,
        } => (|| -> Result<(), String> {
            let task_id = parse_task(&task)?;
            let (x_raw, y) = read_dopemat(&input)?;
            let x_unit: Vec<Vec<f32>> = x_raw
                .iter()
                .map(|c| c.iter().map(|&v| v as f32).collect())
                .collect();
            let meta = load_col_meta(col_meta.as_deref())?;
            let ev_meta: Option<Vec<dope_core::evidence::ColumnMeta>> = meta.as_ref().map(|ms| {
                ms.iter()
                    .map(dope_core::evidence::ColumnMeta::from)
                    .collect()
            });
            let tier_enum = match tier.to_ascii_lowercase().as_str() {
                "t0" => EvidenceTier::T0,
                "t1" => EvidenceTier::T1,
                "t2" => EvidenceTier::T2,
                "t3" => EvidenceTier::T3,
                _ => return Err("invalid_tier".into()),
            };
            let k_eff = if classes > 0 {
                classes
            } else if task_id == 0 {
                1
            } else if task_id == 1 {
                2
            } else {
                let mut uniq: Vec<i32> = y
                    .iter()
                    .filter(|v| v.is_finite())
                    .map(|v| v.round() as i32)
                    .collect();
                uniq.sort_unstable();
                uniq.dedup();
                uniq.len().clamp(2, 8)
            };
            let ev = extract_evidence(&EvidenceInput {
                x_unit: &x_unit,
                x_raw: Some(&x_raw),
                y: &y,
                task_id,
                n_classes: k_eff,
                col_meta: ev_meta.as_deref(),
                packet_probe: None,
                tier: tier_enum,
                target_free,
                seed,
            })
            .map_err(|e| format!("evidence_failed: {e}"))?;
            let sha_hex: String = ev
                .layout_sha256
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            let json = serde_json::json!({
                "p": ev.p,
                "k_eff": ev.k_eff,
                "layout_sha256": sha_hex,
                "col": ev.col.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                "col_mask": ev.col_mask.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                "edge": ev.edge.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                "edge_mask": ev.edge_mask,
                "class_tok": ev.class_tok.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                "class_col": ev.class_col.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                "group_tok": ev.group_tok.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                "glob": ev.glob.to_vec(),
                "glob_mask": ev.glob_mask.to_vec(),
                "col_composite_signal": ev.col_composite_signal,
            });
            let bytes = serde_json::to_vec(&json).map_err(|e| format!("json_failed: {e}"))?;
            write_atomic(&out, &bytes)
        })(),
        Cmd::Embed { model, out } => (|| -> Result<(), String> {
            let pkt_bytes = fs::read(&model).map_err(|e| format!("read_model_failed: {e}"))?;
            let pkt = unpack_dse8_packet(&pkt_bytes).map_err(|e| format!("unpack_failed: {e}"))?;
            let emb = extract_jopedime_embedding(&pkt);
            let json = serde_json::json!({
                "dim": emb.len(),
                "code_tier": pkt.code.tier,
                "embedding": emb,
            });
            write_atomic(
                &out,
                &serde_json::to_vec(&json).map_err(|e| format!("json_failed: {e}"))?,
            )
        })(),
        Cmd::Truncate { model, budget, out } => (|| -> Result<(), String> {
            let pkt_bytes = fs::read(&model).map_err(|e| format!("read_model_failed: {e}"))?;
            let trunc = truncate_dse8_packet(&pkt_bytes, budget)
                .map_err(|e| format!("truncate_failed: {e}"))?;
            write_atomic(&out, &trunc)?;
            println!(
                "{}",
                serde_json::json!({
                    "status": "ok",
                    "orig_bytes": pkt_bytes.len(),
                    "bytes": trunc.len(),
                })
            );
            Ok(())
        })(),
        Cmd::Inspect { model } => (|| -> Result<(), String> {
            let pkt_bytes = fs::read(&model).map_err(|e| format!("read_model_failed: {e}"))?;
            let pkt = unpack_dse8_packet(&pkt_bytes).map_err(|e| format!("unpack_failed: {e}"))?;
            let sha = format!("{:x}", Sha256::digest(&pkt_bytes));
            println!(
                "{}",
                serde_json::json!({
                    "format": "DSE8v1",
                    "bytes": pkt_bytes.len(),
                    "sha256": sha,
                    "task_id": pkt.header.task_id,
                    "k_eff": pkt.header.k_eff,
                    "p_eff": pkt.header.p_eff,
                    "x_cols_raw": pkt.header.x_cols_raw,
                    "normalized_only": pkt.card.normalized_only,
                    "rank_use": pkt.header.rank_use,
                    "knot_level": pkt.header.knot_level,
                    "nominal_groups": pkt.nominal_groups.len(),
                    "regimes": pkt.regimes.regimes.len(),
                    "effects": pkt.effects.len(),
                    "pairs": pkt.pairs.len(),
                    "trees": pkt.trees.trees.len(),
                    "resid_cols": pkt.resid_model.cols.len(),
                    "code_tier": pkt.code.tier,
                })
            );
            Ok(())
        })(),
    };
    if let Err(err) = res {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

//! Pure-Rust `DOPEMAT1` `fit` / `sample` CLI for `SingleEncV7` (`<= 2 KiB`, `source_free`).
//!
//! Drop-in `dope_kpi emit-cli` external arm:
//! - `dope_v7_emit fit --in A.mat --task regression|binary|multiclass [--classes K] --budget 2048 --out m.bin`
//! - `dope_v7_emit sample --model m.bin [--checkpoint w.safetensors] [--n-heads 4] --rows N --seed S --out S.mat`

use clap::{Parser, Subcommand};
use dope_core::rng::Rng;
use dope_core::single_encoding::{
    fit_dse7_packet, sample_zero_init_rows, unpack_dse7_packet, SingleEncV7Weights,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const DOPEMAT_MAGIC: &[u8; 8] = b"DOPEMAT1";

#[derive(Parser)]
#[command(
    name = "dope_v7_emit",
    about = "Pure-Rust self-contained SingleEncV7 DSE7 packet fit & sample CLI"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Fit a strict `<= budget` self-contained CRC32-verified DSE7 v2 wire packet from A.mat.
    Fit {
        #[arg(long = "in")]
        input: PathBuf,
        #[arg(long)]
        task: String,
        #[arg(long, default_value_t = 0)]
        classes: usize,
        #[arg(long, default_value_t = 2048)]
        budget: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Decode synthetic raw-unit rows (`DOPEMAT1`, target last, labels sampled) from packet + optional checkpoint.
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
}

use dope_arena::dopemat::{read_dopemat, write_atomic};

fn write_dopemat(path: &Path, rows: usize, x_cols: usize, body: &[f32]) -> Result<(), String> {
    let mut buf = Vec::with_capacity(24 + body.len() * 4);
    buf.extend_from_slice(DOPEMAT_MAGIC);
    buf.extend_from_slice(&(rows as u32).to_le_bytes());
    buf.extend_from_slice(&(x_cols as u32).to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    for &v in body {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    write_atomic(path, &buf)
}

fn main() {
    let cli = Cli::parse();
    let code = match cli.cmd {
        Cmd::Fit {
            input,
            task,
            classes: _,
            budget,
            out,
        } => run_fit(&input, &task, budget, &out),
        Cmd::Sample {
            model,
            checkpoint,
            n_heads,
            rows,
            seed,
            out,
        } => run_sample(&model, checkpoint.as_deref(), n_heads, rows, seed, &out),
    };
    if let Err(err) = code {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run_fit(input: &Path, task: &str, budget: usize, out: &Path) -> Result<(), String> {
    let task_id: u8 = match task {
        "regression" => 0,
        "binary" => 1,
        "multiclass" => 2,
        _ => return Err("unsupported_task".into()),
    };
    let (x, y) = read_dopemat(input)?;
    match fit_dse7_packet(&x, &y, task_id, budget) {
        Ok(pkt_bytes) => {
            write_atomic(out, &pkt_bytes)?;
            let sha = format!("{:x}", Sha256::digest(&pkt_bytes));
            println!(
                "{}",
                serde_json::json!({
                    "status": "ok",
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
}

fn run_sample(
    model_path: &Path,
    checkpoint_path: Option<&Path>,
    n_heads: usize,
    rows: usize,
    seed: u64,
    out: &Path,
) -> Result<(), String> {
    let pkt_bytes = fs::read(model_path).map_err(|e| format!("read_model_failed: {e}"))?;
    let (pkt, hdr, card) =
        unpack_dse7_packet(&pkt_bytes).map_err(|e| format!("unpack_failed: {e}"))?;

    let decoded = if let Some(wp) = checkpoint_path {
        let w_bytes = fs::read(wp).map_err(|e| format!("read_checkpoint_failed: {e}"))?;
        let model = SingleEncV7Weights::from_safetensors_bytes(&w_bytes, n_heads)
            .map_err(|e| format!("load_checkpoint_failed: {e}"))?;
        model.sample_counter_rows(&pkt, rows, seed)
    } else {
        sample_zero_init_rows(&pkt, rows, seed)
    };

    let mut lbl_rng = Rng::new(seed ^ 0xD5E7_CAFE_BABE_1234);
    let mut body = Vec::with_capacity(rows * (card.x_cols_raw + 1));
    let k_eff = hdr.k_eff as usize;
    for row in &decoded {
        let x_raw = card.inverse_row_x(&row.x);
        body.extend_from_slice(&x_raw);
        let y_raw: f32 = match hdr.task_id {
            0 => (card.y_mean as f64 + (card.y_sd as f64) * row.y_reg as f64) as f32,
            1 => {
                let p1 = 1.0 / (1.0 + (-row.logits[0].clamp(-10.0, 10.0)).exp());
                let idx = usize::from(lbl_rng.uniform() < p1 as f64);
                card.classes.get(idx).copied().unwrap_or(idx as i32) as f32
            }
            _ => {
                let max_l = row.logits[..k_eff]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut probs = [0.0f64; 8];
                let mut sum = 0.0f64;
                for (c, slot) in probs.iter_mut().enumerate().take(k_eff) {
                    let e = ((row.logits[c] - max_l).clamp(-20.0, 0.0) as f64).exp();
                    *slot = e;
                    sum += e;
                }
                let u = lbl_rng.uniform() * sum.max(1e-12);
                let mut acc = 0.0;
                let mut pick = k_eff.saturating_sub(1);
                for (c, &p) in probs.iter().enumerate().take(k_eff) {
                    acc += p;
                    if u <= acc {
                        pick = c;
                        break;
                    }
                }
                card.classes.get(pick).copied().unwrap_or(pick as i32) as f32
            }
        };
        body.push(y_raw);
    }
    write_dopemat(out, rows, card.x_cols_raw, &body)?;
    Ok(())
}

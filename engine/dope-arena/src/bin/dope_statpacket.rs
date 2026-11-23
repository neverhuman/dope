//! `dope_statpacket`: fit, sample, truncate, inspect and embed
//! `dope.single-encoding/v1` statistics packets. Tables travel as `DOPEMAT1`
//! files in raw units (features first, target last); every command prints one
//! JSON object on stdout. Errors go to stderr and never contain data values,
//! file contents or file names.

use clap::{Parser, Subcommand};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

// The module is compiled into this binary until the library crate lists it;
// library items the command line does not reach are still exercised by tests.
#[allow(dead_code)]
#[path = "../statpacket/mod.rs"]
mod statpacket;

use statpacket::codec::{decoder_sha8, ledger, truncate, unpack};
use statpacket::decode::{Decoder, LabelRule};
use statpacket::embed::{embed, EMBED_PREFIX};
use statpacket::fit::{fit_packet, FitTarget, Size};
use statpacket::matio::Mat;
use statpacket::packet::{PacketLevel, PacketTask};
use statpacket::scale::ScaleCard;
use statpacket::StatError;

#[derive(Parser)]
#[command(
    name = "dope_statpacket",
    about = "Stat packet v0 (dope.single-encoding/v1)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fit a packet and its owner-side scale card from FIT rows.
    Fit {
        /// DOPEMAT1 file with the FIT rows in raw units, target last.
        #[arg(long = "in", alias = "input")]
        input: PathBuf,
        /// regression, binary, multiclass or multilabel.
        #[arg(long)]
        task: String,
        /// Class count (multiclass) or label count (multilabel).
        #[arg(long, default_value_t = 0)]
        classes: u8,
        /// Packet file to write.
        #[arg(long = "out", alias = "packet")]
        out: PathBuf,
        #[arg(long)]
        card: PathBuf,
        /// Byte budget of the packet.
        #[arg(long, alias = "max-bytes", default_value_t = 2048)]
        budget: usize,
        /// Highest level to keep (A..E) before the budget is applied.
        #[arg(long)]
        level: Option<String>,
    },
    /// Sample rows `row0 .. row0 + rows` (features and target) from a packet.
    #[command(alias = "decode")]
    Sample {
        #[arg(long)]
        packet: PathBuf,
        /// Scale card; without it the rows stay in unit space.
        #[arg(long)]
        card: Option<PathBuf>,
        #[arg(long)]
        rows: usize,
        #[arg(long, default_value_t = 0)]
        row0: u64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// `sampled` (default) or `regime`: how a class label is drawn.
        #[arg(long, default_value = "sampled")]
        labels: String,
        #[arg(long = "out", alias = "output")]
        out: PathBuf,
    },
    /// Cut a packet down to a lower level.
    Truncate {
        #[arg(long)]
        packet: PathBuf,
        #[arg(long)]
        level: String,
        #[arg(long = "out", alias = "output")]
        out: PathBuf,
    },
    /// Print the header and the byte ledger of a packet (no data values).
    #[command(alias = "info")]
    Inspect {
        #[arg(long)]
        packet: PathBuf,
    },
    /// Print the 64-D and 128-D int8 embedding of a packet.
    Embed {
        #[arg(long)]
        packet: PathBuf,
    },
}

fn read(path: &Path) -> Result<Vec<u8>, StatError> {
    std::fs::read(path).map_err(|e| StatError::Io(e.kind()))
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), StatError> {
    std::fs::write(path, bytes).map_err(|e| StatError::Io(e.kind()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn level_arg(name: &str) -> Result<PacketLevel, StatError> {
    PacketLevel::from_name(name).ok_or(StatError::Invalid("level must be one of A, B, C, D, E"))
}

fn task_arg(name: &str, classes: u8) -> Result<(PacketTask, u8), StatError> {
    match name {
        "regression" => Ok((PacketTask::Regression, 0)),
        "binary" => Ok((PacketTask::Binary, 2)),
        "multiclass" => Ok((PacketTask::Multiclass, classes)),
        "multilabel" => Ok((PacketTask::Multilabel, classes)),
        _ => Err(StatError::Invalid(
            "task must be regression, binary, multiclass or multilabel",
        )),
    }
}

/// Target cells as small non-negative integers below `limit`.
fn integer_labels(values: &[f64], limit: usize) -> Result<Vec<u16>, StatError> {
    values
        .iter()
        .map(|&v| {
            let ok = v.is_finite() && v >= 0.0 && v.fract() == 0.0 && (v as usize) < limit;
            ok.then_some(v as u16).ok_or(StatError::Invalid(
                "a label is outside the declared classes",
            ))
        })
        .collect()
}

/// Header fields and byte ledger of a packed file: integers and names only.
fn describe(bytes: &[u8]) -> Result<serde_json::Value, StatError> {
    let packet = unpack(bytes)?;
    Ok(json!({
        "schema": "dope.single-encoding/v1",
        "header": {
            "magic": String::from_utf8_lossy(&bytes[..4]),
            "version": bytes[4],
            "kind": bytes[5],
            "task": packet.task,
            "classes": packet.k,
            "cols": packet.p,
            "level": packet.level.name(),
            "bit_profile": bytes[11],
            "sections": bytes[12],
            "log2_rows": packet.log2_n,
            "sparse_target": packet.target_sparse,
            "decoder_sha8": hex(&decoder_sha8()),
        },
        "packet_bytes": bytes.len(),
        "packet_sha256": sha256_hex(bytes),
        "ledger": ledger(bytes)?,
    }))
}

fn fit(
    input: &Path,
    task: &str,
    classes: u8,
    out: &Path,
    card_path: &Path,
    budget: usize,
    level: Option<&str>,
) -> Result<serde_json::Value, StatError> {
    let mat = Mat::read(input)?;
    let (task, k) = task_arg(task, classes)?;
    let want_y = if task == PacketTask::Multilabel {
        k as usize
    } else {
        1
    };
    if mat.y_cols != want_y || mat.x_cols == 0 {
        return Err(StatError::Invalid(
            "input does not carry the declared target columns",
        ));
    }
    let raw = mat.features();
    let (values, ids, mut cells): (Vec<f64>, Vec<u16>, Vec<u8>);
    let target = match task {
        PacketTask::Regression => {
            values = mat.target(0);
            FitTarget::Regression(&values)
        }
        PacketTask::Binary | PacketTask::Multiclass => {
            ids = integer_labels(&mat.target(0), k as usize)?;
            FitTarget::Classes(&ids)
        }
        PacketTask::Multilabel => {
            cells = Vec::with_capacity(mat.rows * k as usize);
            for row in 0..mat.rows {
                for label in 0..k as usize {
                    let cell = mat.data[row * mat.width() + mat.x_cols + label] as f64;
                    cells.push(integer_labels(&[cell], 2)?[0] as u8);
                }
            }
            FitTarget::Multilabel(&cells)
        }
    };
    let size = Size {
        level: match level {
            Some(name) => level_arg(name)?,
            None => PacketLevel::E,
        },
        max_bytes: Some(budget),
    };
    let fitted = fit_packet(&raw, target, task, k, size, mat.rows)?;
    let card_text = fitted.card.to_json()?;
    write(out, &fitted.bytes)?;
    write(card_path, card_text.as_bytes())?;
    let mut report = describe(&fitted.bytes)?;
    report["status"] = json!(if fitted.budget.fits {
        "ok"
    } else {
        "over_budget"
    });
    report["rows"] = json!(mat.rows);
    report["cols"] = json!(mat.x_cols);
    report["level"] = json!(fitted.budget.level.name());
    report["budget"] = json!(budget);
    report["max_bytes"] = json!(budget);
    report["budget_step"] = json!(fitted.budget.step);
    report["target_cols"] = json!(fitted.budget.target_cols);
    report["main_effect_cols"] = json!(fitted.budget.main_effect_cols);
    report["card_bytes"] = json!(card_text.len());
    Ok(report)
}

fn sample(
    packet: &Path,
    card: Option<&Path>,
    rows: usize,
    row0: u64,
    seed: u64,
    labels: &str,
    out: &Path,
) -> Result<serde_json::Value, StatError> {
    let rule = match labels {
        "sampled" => LabelRule::Sampled,
        "regime" => LabelRule::Regime,
        _ => return Err(StatError::Invalid("labels must be sampled or regime")),
    };
    let bytes = read(packet)?;
    let decoder = Decoder::new(&unpack(&bytes)?)?.with_label_rule(rule);
    let card = match card {
        Some(path) => {
            let text = String::from_utf8(read(path)?)
                .map_err(|_| StatError::Invalid("scale card is not UTF-8"))?;
            Some(ScaleCard::from_json(&text)?)
        }
        None => None,
    };
    let mat = Mat {
        rows,
        x_cols: decoder.cols(),
        y_cols: decoder.targets(),
        data: decoder.table(card.as_ref(), seed, row0, rows)?,
    };
    mat.write(out)?;
    Ok(json!({
        "rows": rows,
        "row0": row0,
        "seed": seed,
        "cols": decoder.cols(),
        "target_cols": decoder.targets(),
        "units": if card.is_some() { "raw" } else { "unit" },
        "labels": labels,
        "packet_bytes": bytes.len(),
        "packet_sha256": sha256_hex(&bytes),
    }))
}

fn run(cli: Cli) -> Result<serde_json::Value, StatError> {
    match cli.command {
        Command::Fit {
            input,
            task,
            classes,
            out,
            card,
            budget,
            level,
        } => fit(
            &input,
            &task,
            classes,
            &out,
            &card,
            budget,
            level.as_deref(),
        ),
        Command::Sample {
            packet,
            card,
            rows,
            row0,
            seed,
            labels,
            out,
        } => sample(&packet, card.as_deref(), rows, row0, seed, &labels, &out),
        Command::Truncate { packet, level, out } => {
            let cut = truncate(&read(&packet)?, level_arg(&level)?)?;
            write(&out, &cut)?;
            describe(&cut)
        }
        Command::Inspect { packet } => describe(&read(&packet)?),
        Command::Embed { packet } => {
            let bytes = read(&packet)?;
            let full = embed(&unpack(&bytes)?);
            Ok(json!({
                "packet_sha256": sha256_hex(&bytes),
                "embedding_64": full[..EMBED_PREFIX].to_vec(),
                "embedding_128": full.to_vec(),
            }))
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("dope_statpacket: {err}");
            ExitCode::from(2)
        }
    }
}

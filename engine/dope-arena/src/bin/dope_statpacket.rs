//! `dope_statpacket`: fit, inspect, truncate and decode `dope.single-encoding/v1`
//! statistics packets. Tables travel as `DOPEMAT1` files in raw units (features
//! first, target last); reports are one JSON object on stdout. Errors go to
//! stderr and never contain data values or file contents.

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

use statpacket::codec::{fit_to_budget, ledger, pack, truncate, unpack};
use statpacket::decode::Decoder;
use statpacket::fit::{fit_features_for_budget, FitInput, Labels};
use statpacket::matio::Mat;
use statpacket::packet::{Level, Task};
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
        /// DOPEMAT1 file with the FIT rows in raw units.
        #[arg(long)]
        input: PathBuf,
        /// regression, binary, multiclass or multilabel.
        #[arg(long)]
        task: String,
        /// Class count (multiclass) or label count (multilabel).
        #[arg(long, default_value_t = 0)]
        classes: u8,
        #[arg(long)]
        packet: PathBuf,
        #[arg(long)]
        card: PathBuf,
        #[arg(long, default_value_t = 2048)]
        max_bytes: usize,
        /// Highest level to keep (A..E) before the budget is applied.
        #[arg(long)]
        level: Option<String>,
    },
    /// Print the byte ledger of a packet.
    Info {
        #[arg(long)]
        packet: PathBuf,
    },
    /// Cut a packet down to a lower level.
    Truncate {
        #[arg(long)]
        packet: PathBuf,
        #[arg(long)]
        level: String,
        #[arg(long)]
        output: PathBuf,
    },
    /// Sample feature rows `row0 .. row0 + rows` from a packet.
    Decode {
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
        #[arg(long)]
        output: PathBuf,
    },
}

fn read(path: &Path) -> Result<Vec<u8>, StatError> {
    std::fs::read(path).map_err(|e| StatError::Io(e.kind()))
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), StatError> {
    std::fs::write(path, bytes).map_err(|e| StatError::Io(e.kind()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn level_arg(name: &str) -> Result<Level, StatError> {
    Level::from_name(name).ok_or(StatError::Invalid("level must be one of A, B, C, D, E"))
}

fn task_arg(name: &str, classes: u8) -> Result<(Task, u8), StatError> {
    match name {
        "regression" => Ok((Task::Regression, 0)),
        "binary" => Ok((Task::Binary, 2)),
        "multiclass" => Ok((Task::Multiclass, classes)),
        "multilabel" => Ok((Task::Multilabel, classes)),
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

fn fit(
    input: &Path,
    task: &str,
    classes: u8,
    packet_path: &Path,
    card_path: &Path,
    max_bytes: usize,
    level: Option<&str>,
) -> Result<serde_json::Value, StatError> {
    let mat = Mat::read(input)?;
    let (task, k) = task_arg(task, classes)?;
    let want_y = if task == Task::Multilabel {
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
    let mut card = ScaleCard::fit(&raw, mat.rows, mat.x_cols)?;
    let unit = card.to_unit_rows(&raw)?;
    let (ids, mut cells): (Vec<u16>, Vec<u8>);
    let labels = match task {
        Task::Regression => {
            card = card.with_regression_target(&mat.target(0))?;
            Labels::None
        }
        Task::Binary | Task::Multiclass => {
            ids = integer_labels(&mat.target(0), k as usize)?;
            card = card.with_classes((0..k).map(f64::from).collect());
            Labels::Classes(&ids)
        }
        Task::Multilabel => {
            cells = Vec::with_capacity(mat.rows * k as usize);
            for row in 0..mat.rows {
                for label in 0..k as usize {
                    let cell = mat.data[row * mat.width() + mat.x_cols + label] as f64;
                    cells.push(integer_labels(&[cell], 2)?[0] as u8);
                }
            }
            Labels::Multilabel(&cells)
        }
    };
    let fit_input = FitInput {
        x: &unit,
        rows: mat.rows,
        cols: mat.x_cols,
        task,
        k,
        labels,
    };
    let mut packet = fit_features_for_budget(&fit_input, max_bytes)?;
    if let Some(name) = level {
        packet = packet.at_level(level_arg(name)?)?;
    }
    let fitted = fit_to_budget(&packet, max_bytes);
    let bytes = pack(&fitted.packet, fitted.level)?;
    write(packet_path, &bytes)?;
    write(card_path, card.to_json()?.as_bytes())?;
    Ok(json!({
        "schema": "dope.single-encoding/v1",
        "status": if fitted.fits { "ok" } else { "over_budget" },
        "rows": mat.rows,
        "cols": mat.x_cols,
        "level": fitted.level.name(),
        "packet_bytes": bytes.len(),
        "max_bytes": max_bytes,
        "budget_step": fitted.step,
        "target_cols": fitted.target_cols,
        "main_effect_cols": fitted.main_effect_cols,
        "packet_sha256": sha256_hex(&bytes),
        "ledger": ledger(&bytes)?,
    }))
}

fn run(cli: Cli) -> Result<serde_json::Value, StatError> {
    match cli.command {
        Command::Fit {
            input,
            task,
            classes,
            packet,
            card,
            max_bytes,
            level,
        } => fit(
            &input,
            &task,
            classes,
            &packet,
            &card,
            max_bytes,
            level.as_deref(),
        ),
        Command::Info { packet } => {
            let bytes = read(&packet)?;
            Ok(json!({ "packet_sha256": sha256_hex(&bytes), "ledger": ledger(&bytes)? }))
        }
        Command::Truncate {
            packet,
            level,
            output,
        } => {
            let cut = truncate(&read(&packet)?, level_arg(&level)?)?;
            write(&output, &cut)?;
            Ok(json!({ "packet_sha256": sha256_hex(&cut), "ledger": ledger(&cut)? }))
        }
        Command::Decode {
            packet,
            card,
            rows,
            row0,
            seed,
            output,
        } => {
            let bytes = read(&packet)?;
            let decoder = Decoder::new(&unpack(&bytes)?)?;
            let (data, units) = match card {
                Some(path) => {
                    let text = String::from_utf8(read(&path)?)
                        .map_err(|_| StatError::Invalid("scale card is not UTF-8"))?;
                    let card = ScaleCard::from_json(&text)?;
                    (decoder.rows_raw(&card, seed, row0, rows)?, "raw")
                }
                None => (decoder.rows_unit(seed, row0, rows), "unit"),
            };
            let mat = Mat {
                rows,
                x_cols: decoder.cols(),
                y_cols: 0,
                data,
            };
            mat.write(&output)?;
            Ok(json!({
                "rows": rows,
                "row0": row0,
                "seed": seed,
                "cols": decoder.cols(),
                "units": units,
                "packet_bytes": bytes.len(),
                "packet_sha256": sha256_hex(&bytes),
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

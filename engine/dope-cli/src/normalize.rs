//! Explicit owner-private NC1 workflow. No CSV target/kind inference.
use crate::safe_io;
use clap::Subcommand;
use dope_core::nc1::{DeclaredSchema, InputTable, NormOptions, ScaleCard, MAX_JSON_BYTES};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Subcommand)]
pub(crate) enum NormalizeCommand {
    /// Fit a card on declared FIT row IDs; export private card/FIT matrix and public schema.
    Fit {
        #[arg(long)]
        request: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Apply a frozen card. Request may explicitly select feature-only prediction input.
    Apply {
        #[arg(long)]
        card: PathBuf,
        #[arg(long)]
        request: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Owner-private regression target inverse; predictions never clipped.
    InverseTarget {
        #[arg(long)]
        card: PathBuf,
        #[arg(long)]
        request: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
    },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FitRequest {
    table: InputTable,
    declared: DeclaredSchema,
    fit_rows: Vec<u64>,
    #[serde(default)]
    options: NormOptions,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ApplyRequest {
    table: InputTable,
    rows: Vec<u64>,
    #[serde(default)]
    features_only: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InverseRequest {
    values: Vec<f64>,
}
fn card(path: &std::path::Path) -> Result<ScaleCard, String> {
    ScaleCard::from_private_json(&safe_io::read_bytes(path, MAX_JSON_BYTES)?)
        .map_err(|e| e.to_string())
}
pub(crate) fn run(command: NormalizeCommand) -> Result<(), String> {
    match command {
        NormalizeCommand::Fit { request, out } => {
            let input: FitRequest = safe_io::read_json(&request, MAX_JSON_BYTES)
                .map_err(|_| "invalid_normalization_request")?;
            let card = ScaleCard::fit(
                &input.table,
                &input.declared,
                &input.fit_rows,
                input.options,
            )
            .map_err(|e| e.to_string())?;
            let (matrix, receipt) = card
                .apply(&input.table, &input.fit_rows)
                .map_err(|e| e.to_string())?;
            safe_io::write_directory(
                &out,
                &[
                    (
                        "ScaleCard.private.json",
                        card.to_private_json().map_err(|e| e.to_string())?,
                    ),
                    ("schema.public.json", safe_io::json(card.public_schema())?),
                    (
                        "fit.unit.private.json",
                        matrix.to_private_json().map_err(|e| e.to_string())?,
                    ),
                    ("apply.private.json", safe_io::json(&receipt)?),
                ],
            )?;
            println!(
                "{}",
                serde_json::json!({"recipe":"nc1-minmax-onehot-v1","fit_rows":input.fit_rows.len(),"columns":card.public_schema().columns,
                "fit_scope_sha256":card.fit_scope_sha256(),"transform_sha256":card.transform_sha256(),"private_export":true})
            );
        }
        NormalizeCommand::Apply {
            card: path,
            request,
            out,
        } => {
            let card = card(&path)?;
            let input: ApplyRequest = safe_io::read_json(&request, MAX_JSON_BYTES)
                .map_err(|_| "invalid_normalization_request")?;
            let (matrix, receipt) = if input.features_only {
                card.apply_features(&input.table, &input.rows)
            } else {
                card.apply(&input.table, &input.rows)
            }
            .map_err(|e| e.to_string())?;
            safe_io::write_directory(
                &out,
                &[
                    (
                        "matrix.unit.private.json",
                        matrix.to_private_json().map_err(|e| e.to_string())?,
                    ),
                    ("apply.private.json", safe_io::json(&receipt)?),
                ],
            )?;
            println!(
                "{}",
                serde_json::json!({"rows":matrix.x().len(),"columns":matrix.schema().columns,"private_export":true})
            );
        }
        NormalizeCommand::InverseTarget {
            card: path,
            request,
            out,
        } => {
            let card = card(&path)?;
            let input: InverseRequest = safe_io::read_json(&request, MAX_JSON_BYTES)
                .map_err(|_| "invalid_normalization_request")?;
            if input.values.is_empty() {
                return Err("empty_predictions".into());
            }
            let values: Vec<f64> = input
                .values
                .iter()
                .map(|&v| card.inverse_target(v).map_err(|e| e.to_string()))
                .collect::<Result<_, _>>()?;
            safe_io::write_new(
                &out,
                &safe_io::json(&serde_json::json!({"values":values,"private_export":true}))?,
            )?;
            println!(
                "{}",
                serde_json::json!({"rows":values.len(),"private_export":true})
            );
        }
    }
    Ok(())
}

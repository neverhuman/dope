//! Explicit experimental learned commands. Frozen local weights; no downloads or backend fallback.
use clap::{Args, Subcommand};
use dope_core::{learned, levels::Level, Result};
use std::{
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Subcommand)]
pub(crate) enum LearnedCommand {
    /// Encode a declared fit/context CSV using frozen shared weights.
    Encode(Encode),
    /// Sample using only decoder weights, code and charged positional schema.
    Sample(Sample),
    /// Inspect code identity and complete dependency byte costs.
    Inspect(Inspect),
}
#[derive(Args)]
pub(crate) struct Encode {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    csv: PathBuf,
    #[arg(long)]
    target: String,
    /// Explicit source-order numeric/categorical kinds, including the declared target.
    #[arg(long, value_delimiter = ',')]
    kinds: Vec<String>,
    #[arg(long,value_parser=["regression","classification"])]
    task: String,
    #[arg(long, default_value = "low")]
    levels: String,
    #[arg(long, default_value = "64KiB")]
    max_bytes: String,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    target_free: bool,
    #[arg(long)]
    out: PathBuf,
}
#[derive(Args)]
pub(crate) struct Sample {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    code: PathBuf,
    #[arg(long)]
    level: String,
    #[arg(long)]
    rows: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 32)]
    steps: usize,
    /// Gradual categorical reveal, or the initial experimental redraw recipe.
    #[arg(long, default_value = "reveal-v1", value_parser = ["reveal-v1", "redraw-v1"])]
    categorical_schedule: String,
    #[arg(long)]
    out: PathBuf,
}
#[derive(Args)]
pub(crate) struct Inspect {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    code: PathBuf,
}
fn bad() -> dope_core::DopeError {
    dope_core::DopeError::Artifact("learned command input/output contract failed".into())
}
fn code(path: &Path) -> Result<Vec<u8>> {
    let meta = std::fs::symlink_metadata(path)?;
    if !meta.is_file() || meta.len() > 1048576 {
        return Err(bad());
    }
    Ok(std::fs::read(path)?)
}
fn publish(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = format!(
        ".dope-learned-{}-{}.partial",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| bad())?
            .as_nanos()
    );
    let staging = parent.join(name);
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&staging)?;
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::hard_link(&staging, path)?;
        Ok(())
    })();
    std::fs::remove_file(staging)?;
    result
}
fn execute(command: LearnedCommand) -> Result<()> {
    match command {
        LearnedCommand::Encode(a) => {
            let selected = Level::parse_list(&a.levels)?;
            let through = *selected.last().ok_or_else(bad)?;
            let max_bytes = super::parse_bytes(&a.max_bytes).map_err(|_| bad())?;
            let (encoder, manifest) = learned::load_encoder(&a.model)?;
            let context = learned::intake::read_context(
                &a.csv,
                &a.target,
                &a.kinds,
                &a.task,
                &manifest.graph,
                a.seed,
            )?;
            let local = learned::LocalEncoding::encode(
                &encoder,
                manifest,
                &context.values,
                &context.observed,
                context.schema,
                !a.target_free,
                through,
            )?;
            let exported = local.export(&selected, max_bytes)?;
            publish(&a.out, exported.bytes())?;
            println!(
                "{}",
                serde_json::json!({"format":"dope-table-code-v1","bytes":exported.bytes().len(),
                "levels":exported.manifest.exported_levels,"certified":false})
            );
        }
        LearnedCommand::Sample(a) => {
            let (decoder, model) = learned::load_decoder(&a.model)?;
            let (code, manifest) = learned::read_code(&code(&a.code)?, &model)?;
            let levels = Level::parse_list(&a.level)?;
            if levels.len() != 1
                || !manifest
                    .exported_levels
                    .iter()
                    .any(|l| l == levels[0].name())
            {
                return Err(bad());
            }
            let width = [256, 1024, 4096, 5120][levels[0] as usize];
            let schedule = if a.categorical_schedule == "reveal-v1" {
                learned::CategoricalSchedule::RevealV1
            } else {
                learned::CategoricalSchedule::RedrawV1
            };
            let samples = learned::sample_with_schedule(
                &decoder,
                &code.narrow(0, 0, width)?,
                &manifest.schema,
                a.rows,
                a.seed,
                a.steps,
                schedule,
            )?;
            let headers: Vec<_> = (0..samples.schema.columns.len())
                .map(|i| format!("column_{i}"))
                .collect();
            let mut bytes = Vec::new();
            dope_core::table::write_csv(&mut bytes, &headers, samples.rows.len(), |r, c| {
                samples.rows[r][c]
                    .map(|v| v.to_string())
                    .unwrap_or_default()
            })?;
            publish(&a.out, &bytes)?;
        }
        LearnedCommand::Inspect(a) => {
            let model = learned::read_manifest(&a.model)?;
            let bytes = code(&a.code)?;
            let (_, manifest) = learned::read_code(&bytes, &model)?;
            println!(
                "{}",
                serde_json::json!({"format":manifest.format,"code_bytes":bytes.len(),
                "coordinate_space_id":manifest.coordinate_space_id,"levels":manifest.exported_levels,
                "target_included":manifest.target_included,"model_manifest_bytes":std::fs::metadata(a.model.join("model.json"))?.len(),
                "encoder_weight_bytes":model.weights["encoder"].bytes,"decoder_weight_bytes":model.weights["decoder"].bytes,
                "evidence":manifest.evidence,"certified":false})
            );
        }
    }
    Ok(())
}
pub(crate) fn run(command: LearnedCommand) -> std::result::Result<(), String> {
    execute(command).map_err(|_|format!("{}; reason: model, typed context, level or output is invalid; common fixes: check hashes, explicit kinds/target, bounds and a fresh output path; hint: preserve the failed receipt and repair inputs; docs: research/levels/README.md",bad()))
}

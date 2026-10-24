//! `dope mlc-v3` CLI commands: 100% pure-Rust SafeTensors inference and `DML3`/`CSC1` synthetic sampling.

use clap::{Args, Subcommand};
use dope_core::mlc_v3::{
    sample_synthetic_table_with_draws, Dml3Envelope, FixedSampleDraws, MlcV3ForwardCheckInput,
    MlcV3ForwardCheckOutput, MlcV3Model,
};
use std::io::Write;
use std::path::PathBuf;

#[derive(Subcommand)]
pub(crate) enum MlcV3Command {
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
pub(crate) struct InspectWeightsArgs {
    #[arg(long)]
    pub(crate) weights: PathBuf,
}

#[derive(Args)]
pub(crate) struct InspectWireArgs {
    #[arg(long)]
    pub(crate) wire: PathBuf,
}

#[derive(Args)]
pub(crate) struct RouteArgs {
    #[arg(long)]
    pub(crate) weights: PathBuf,
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
    pub(crate) weights: PathBuf,
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

pub(crate) fn run(cmd: MlcV3Command) -> Result<(), String> {
    let map_err = |e: dope_core::DopeError| e.to_string();
    match cmd {
        MlcV3Command::InspectWeights(a) => {
            let model = MlcV3Model::load_safetensors(&a.weights).map_err(map_err)?;
            let out = serde_json::json!({
                "format": "dope-mlc-v3-safetensors",
                "tensor_count": model.inventory.tensor_count,
                "parameter_count": model.inventory.parameter_count,
                "byte_len": model.inventory.byte_len,
                "sha256": model.inventory.sha256,
                "runtime": "rust-candle-cpu",
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
            );
        }
        MlcV3Command::InspectWire(a) => {
            let raw = std::fs::read(&a.wire).map_err(|e| e.to_string())?;
            let env = Dml3Envelope::from_bytes(&raw).map_err(map_err)?;
            let out = serde_json::json!({
                "wire_format": env.wire_format,
                "wire_bytes": env.wire_bytes,
                "tier_cap_bytes": env.tier_cap_bytes,
                "within_tier_cap": env.within_tier_cap,
                "crc32": env.crc32,
                "compressed_flags": env.compressed_flags,
                "level": env.encoding.level,
                "task": env.encoding.task,
                "code_width": env.encoding.code_width,
                "n_train": env.encoding.n_train,
                "p_use": env.encoding.p_use,
                "include_v3_primitives": env.encoding.target_bundle.include_v3_primitives,
                "has_csc1_scaler": env.scaler.is_some(),
                "k_classes": env.encoding.target_bundle.k_classes,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
            );
        }
        MlcV3Command::Route(a) => {
            let model = MlcV3Model::load_safetensors(&a.weights).map_err(map_err)?;
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
            let model = MlcV3Model::load_safetensors(&a.weights).map_err(map_err)?;
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
                let jbytes = serde_json::to_vec_pretty(&sampled).map_err(|e| e.to_string())?;
                std::fs::write(jp, jbytes).map_err(|e| e.to_string())?;
            }
            let (xs, ys) = if a.space == "scaled" {
                (&sampled.x_scaled, &sampled.y_scaled)
            } else {
                (&sampled.x_raw, &sampled.y_raw)
            };
            let mut csv_bytes = Vec::new();
            dope_core::table::write_csv(&mut csv_bytes, &sampled.headers, ys.len(), |r, c| {
                if c < xs[r].len() {
                    dope_core::schema::format_number(xs[r][c])
                } else {
                    dope_core::schema::format_number(ys[r])
                }
            })
            .map_err(map_err)?;
            match &a.out {
                Some(p) => std::fs::write(p, &csv_bytes).map_err(|e| e.to_string())?,
                None => std::io::stdout()
                    .write_all(&csv_bytes)
                    .map_err(|e| e.to_string())?,
            }
        }
    }
    Ok(())
}

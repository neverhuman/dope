//! Fully local DML4 validation, packing and source-free unit-space sampling.
use crate::safe_io;
use clap::Subcommand;
use dope_core::mlc_v4::{self, Packet};
use std::path::PathBuf;
#[derive(Subcommand)]
pub(crate) enum MlcV4Command {
    /// Validate a typed packet JSON and enforce its complete serialized cap.
    Pack {
        #[arg(long)]
        input: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Inspect only public structural identities of a validated packet.
    Inspect {
        #[arg(long)]
        packet: PathBuf,
    },
    /// Sample the packet's closed unit-space graph, without source/card/encoder.
    Sample {
        #[arg(long)]
        packet: PathBuf,
        #[arg(short = 'n', long, default_value_t = 1000)]
        rows: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(short, long)]
        out: PathBuf,
    },
}
pub(crate) fn run(command: MlcV4Command) -> Result<(), String> {
    match command {
        MlcV4Command::Pack { input, out } => {
            let packet: Packet =
                safe_io::read_json(&input, 65536).map_err(|_| "invalid_dml4_request")?;
            let bytes = mlc_v4::encode(&packet).map_err(|e| e.to_string())?;
            safe_io::write_new(&out, &bytes)?;
            println!(
                "{}",
                serde_json::json!({"bytes":bytes.len(),"columns":packet.schema.columns,"contract":packet.contract})
            );
        }
        MlcV4Command::Inspect { packet } => {
            let bytes = safe_io::read_bytes(&packet, 65536)?;
            let packet = mlc_v4::decode(&bytes).map_err(|e| e.to_string())?;
            println!(
                "{}",
                serde_json::json!({"contract":packet.contract,"level":packet.level,"bytes":bytes.len(),"schema":packet.schema,"schema_sha256":packet.schema_sha256,"identity":packet.identity})
            );
        }
        MlcV4Command::Sample {
            packet,
            rows,
            seed,
            out,
        } => {
            let bytes = safe_io::read_bytes(&packet, 65536)?;
            let packet = mlc_v4::decode(&bytes).map_err(|e| e.to_string())?;
            let table = mlc_v4::sample(&packet, rows, seed).map_err(|e| e.to_string())?;
            safe_io::write_new(&out, &safe_io::json(&table)?)?;
            println!(
                "{}",
                serde_json::json!({"rows":table.y.len(),"columns":packet.schema.columns,"space":"unit","qualification_state":"not_run"})
            );
        }
    }
    Ok(())
}

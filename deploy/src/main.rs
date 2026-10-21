use clap::{Parser, Subcommand};
use dope_kernel::sample::SampleOptions;
use dope_runtime::{Activation, Carrier, DeviceRequest, pack, read_bounded};
use sha2::{Digest, Sha256};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    version,
    about = "Offline Rust sampler; development SafeTensors transport"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Export native DPK and typed TVAE weights into one charged carrier.
    Pack {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value = "legacy-v1")]
        activation: Activation,
    },
    /// Sample normalized rows. The explicitly declared target remains last.
    Sample {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        rows: usize,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
        #[arg(long, value_enum, default_value = "auto")]
        device: DeviceRequest,
    },
    /// Inspect bytes and numerical profile without exposing fitted parameters.
    Inspect {
        #[arg(long)]
        artifact: PathBuf,
    },
}

fn create(path: &std::path::Path) -> Result<std::fs::File, &'static str> {
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path).map_err(|_| "output_unavailable_or_exists")
}

fn run(args: Args) -> Result<serde_json::Value, &'static str> {
    match args.command {
        Command::Pack {
            artifact,
            out,
            activation,
        } => {
            let bytes = pack(&read_bounded(&artifact)?, activation)?;
            create(&out)?
                .write_all(&bytes)
                .map_err(|_| "output_write_failed")?;
            Ok(
                serde_json::json!({"state":"packed", "charged_bytes":bytes.len(),
                "sha256":format!("{:x}", Sha256::digest(&bytes)), "qualified":false}),
            )
        }
        Command::Inspect { artifact } => {
            let bytes = read_bounded(&artifact)?;
            let carrier = Carrier::load(&bytes)?;
            Ok(
                serde_json::json!({"state":"inspected", "charged_bytes":carrier.charged_bytes,
                "features":carrier.kernel.features, "activation":format!("{:?}",carrier.activation),
                "sha256":format!("{:x}", Sha256::digest(&bytes)),
                "cold_closure_bytes":null, "qualified":false, "certified":false}),
            )
        }
        Command::Sample {
            artifact,
            out,
            rows,
            seed,
            device,
        } => {
            let carrier = Carrier::load(&read_bounded(&artifact)?)?;
            let (values, used) = carrier.sample(
                SampleOptions {
                    rows,
                    seed: Some(seed),
                },
                device,
            )?;
            let width = carrier.kernel.features as usize + 1;
            let mut file = BufWriter::new(create(&out)?);
            for row in values.chunks_exact(width) {
                for (index, value) in row.iter().enumerate() {
                    if index != 0 {
                        file.write_all(b",").map_err(|_| "output_write_failed")?;
                    }
                    write!(file, "{value:.8}").map_err(|_| "output_write_failed")?;
                }
                file.write_all(b"\n").map_err(|_| "output_write_failed")?;
            }
            file.flush().map_err(|_| "output_write_failed")?;
            Ok(
                serde_json::json!({"state":"sampled", "rows":rows, "device_used":used,
                "charged_bytes":carrier.charged_bytes,"qualified":false}),
            )
        }
    }
}

fn main() -> std::process::ExitCode {
    // Clap's own argument errors can echo private paths. Emit fixed diagnostics.
    let args = match Args::try_parse() {
        Ok(args) => args,
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            print!("{error}");
            return std::process::ExitCode::SUCCESS;
        }
        Err(_) => {
            eprintln!("{{\"error\":\"invalid_arguments\"}}");
            return std::process::ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(receipt) => {
            println!("{receipt}");
            std::process::ExitCode::SUCCESS
        }
        Err(code) => {
            eprintln!("{{\"error\":\"{code}\"}}");
            std::process::ExitCode::from(2)
        }
    }
}

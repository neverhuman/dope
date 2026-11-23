//! Source-free replay and packing diagnostics for archived V8 DPA1 artifacts.

#[path = "dope_v8_replay/trace.rs"]
mod trace;

use clap::{Parser, Subcommand, ValueEnum};
use dope_arena::dopemat::write_dopemat;
use dope_core::auto::v8::AutoV8Options;
use dope_core::auto::v8_dpa::fit_dpa1_v8_with_code_reservation;
use dope_core::auto::{parse_dpa1, V8Decoder, V8Model};
use dope_core::schema::Task;
use dope_core::single_encoding::v8::{
    unpack_dse8_packet, DecodedV8Row, Dse8Packet, ReplayArmV8, ReplayPlanV8, DSE8_MAGIC,
    V8_MAX_RANK,
};
use dope_core::table::RawTable;
use serde::Serialize;
use std::error::Error;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "dope_v8_replay")]
struct Cli {
    #[arg(long, global = true)]
    weights: Option<PathBuf>,
    #[arg(long, global = true, default_value_t = 4)]
    heads: usize,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Materialize {
        #[arg(long)]
        packets: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Sample {
        packet: PathBuf,
        #[arg(long)]
        arm: Arm,
        #[arg(long)]
        rows: usize,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Export native random primitives and served packet-unit rows for Torch parity.
    Trace {
        packet: PathBuf,
        #[arg(long)]
        arm: Arm,
        #[arg(long)]
        rows: usize,
        #[arg(long)]
        seed: u64,
    },
    ReserveCode {
        fit: PathBuf,
        #[arg(long)]
        target: String,
        #[arg(long, value_enum)]
        task: ReplayTask,
        #[arg(long)]
        seed: u64,
        #[arg(long, default_value_t = 2048)]
        budget: usize,
        #[arg(long, default_value_t = 1)]
        code_tier: u8,
        #[arg(long)]
        out: PathBuf,
    },
    Inspect {
        packet: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum ReplayTask {
    Regression,
    Classification,
}

#[derive(Clone, Copy, ValueEnum)]
enum Arm {
    Analytic,
    Learned,
    LearnedX,
    LearnedY,
    MaskInactiveRanks,
    NoKnots,
    NoFlow,
    NoLinearCorrection,
    NoGatedHead,
    NoResidualScale,
}

impl From<Arm> for ReplayArmV8 {
    fn from(arm: Arm) -> Self {
        match arm {
            Arm::Analytic => Self::Analytic,
            Arm::Learned => Self::Learned,
            Arm::LearnedX => Self::LearnedX,
            Arm::LearnedY => Self::LearnedY,
            Arm::MaskInactiveRanks => Self::MaskInactiveRanks,
            Arm::NoKnots => Self::NoKnots,
            Arm::NoFlow => Self::NoFlow,
            Arm::NoLinearCorrection => Self::NoLinearCorrection,
            Arm::NoGatedHead => Self::NoGatedHead,
            Arm::NoResidualScale => Self::NoResidualScale,
        }
    }
}

#[derive(Clone, Default, Serialize)]
struct Moments {
    rows: usize,
    mean: f64,
    #[serde(skip)]
    m2: f64,
    variance: f64,
}

impl Moments {
    fn push(&mut self, value: f32) {
        self.rows += 1;
        let delta = f64::from(value) - self.mean;
        self.mean += delta / self.rows as f64;
        self.m2 += delta * (f64::from(value) - self.mean);
        self.variance = self.m2 / self.rows as f64;
    }
}

#[derive(Clone, Default, Serialize)]
struct Column {
    both_observed: usize,
    feature_changes: usize,
    mask_changes: usize,
    #[serde(skip)]
    abs_change_sum: f64,
    mean_abs_change: f64,
    latent: Moments,
    analytic_latent: Moments,
}

#[derive(Serialize)]
struct PairedDiagnostics {
    active_columns: Vec<Column>,
    factors: [Moments; V8_MAX_RANK],
    conditional: Moments,
    analytic_conditional: Moments,
    target: Moments,
    analytic_target: Moments,
    target_changes: usize,
}

impl PairedDiagnostics {
    fn new(width: usize) -> Self {
        Self {
            active_columns: vec![Column::default(); width],
            factors: Default::default(),
            conditional: Moments::default(),
            analytic_conditional: Moments::default(),
            target: Moments::default(),
            analytic_target: Moments::default(),
            target_changes: 0,
        }
    }

    fn observe(
        &mut self,
        row: &DecodedV8Row,
        base: &DecodedV8Row,
        factors: [f32; V8_MAX_RANK],
        latent: &[f32],
        analytic_latent: &[f32],
    ) {
        for (moment, value) in self.factors.iter_mut().zip(factors) {
            moment.push(value);
        }
        for (j, column) in self.active_columns.iter_mut().enumerate() {
            let missing = row.missing_active[j];
            let base_missing = base.missing_active[j];
            column.mask_changes += usize::from(missing != base_missing);
            if !missing && !base_missing {
                column.both_observed += 1;
                column.feature_changes += usize::from(row.x_unit[j] != base.x_unit[j]);
                column.abs_change_sum += f64::from(row.x_unit[j] - base.x_unit[j]).abs();
                column.mean_abs_change = column.abs_change_sum / column.both_observed as f64;
            }
            column.latent.push(latent[j]);
            column.analytic_latent.push(analytic_latent[j]);
        }
        self.conditional.push(row.logits[0]);
        self.analytic_conditional.push(base.logits[0]);
        self.target.push(row.y_unit);
        self.analytic_target.push(base.y_unit);
        self.target_changes += usize::from(row.y_unit != base.y_unit);
    }
}

#[derive(Serialize)]
struct Layout {
    active_width: usize,
    rank: u8,
    code_tier: u8,
    code_bytes: usize,
    regimes: usize,
    effects: usize,
    pairs: usize,
    trees: usize,
    residual: bool,
}

impl From<&Dse8Packet> for Layout {
    fn from(pkt: &Dse8Packet) -> Self {
        Self {
            active_width: pkt.card.col_indices.len(),
            rank: pkt.header.rank_use,
            code_tier: pkt.code.tier,
            code_bytes: pkt.code.packed.len(),
            regimes: pkt.regimes.regimes.len(),
            effects: pkt.effects.len(),
            pairs: pkt.pairs.len(),
            trees: pkt.trees.trees.len(),
            residual: !pkt.resid_model.cols.is_empty()
                || pkt.resid_model.q5_resid.iter().any(|&value| value != 0.0),
        }
    }
}

fn sample(
    decoder: Option<&V8Decoder>,
    packet: &Dse8Packet,
    model: &V8Model,
    arm: Arm,
    rows: usize,
    seed: u64,
    out: &Path,
) -> Result<(), Box<dyn Error>> {
    let analytic = ReplayPlanV8::new(packet, model.weights(), ReplayArmV8::Analytic)?;
    let plan = ReplayPlanV8::new(packet, model.weights(), arm.into())?;
    let mut diagnostics = PairedDiagnostics::new(packet.card.col_indices.len());
    let samples = (0..rows as u64).map(|row| {
        let mut analytic_latent = Vec::new();
        let base = analytic.row_at(seed, row, |_, values| analytic_latent = values);
        let mut factors = [0.0; V8_MAX_RANK];
        let mut latent = Vec::new();
        let sample = plan.row_at(seed, row, |factor_draw, values| {
            factors = factor_draw;
            latent = values;
        });
        diagnostics.observe(&sample, &base, factors, &latent, &analytic_latent);
        sample
    });
    if let Some(decoder) = decoder {
        decoder.write_decoded_csv(samples, File::create(out)?)?;
    } else {
        let body: Vec<_> = samples
            .flat_map(|row| {
                row.x_raw
                    .into_iter()
                    .chain([row.y_raw])
                    .map(|value| value as f32)
            })
            .collect();
        write_dopemat(out, rows, packet.card.x_cols_raw, &body)?;
    }
    println!("{}", serde_json::to_string(&diagnostics)?);
    Ok(())
}

enum Artifact {
    Wrapped(V8Decoder),
    Bare(Dse8Packet),
}

impl Artifact {
    fn open(bytes: &[u8], model: &V8Model) -> Result<Self, Box<dyn Error>> {
        let artifact = if bytes.starts_with(DSE8_MAGIC) {
            Self::Bare(unpack_dse8_packet(bytes)?)
        } else {
            let container = parse_dpa1(bytes)?;
            let decoder = V8Decoder::open(&container, Some(model), false)?;
            if !decoder.is_neural() {
                return Err("replay requires a fingerprint-bound packet".into());
            }
            Self::Wrapped(decoder)
        };
        if artifact.packet().weights_fp != Some(model.fingerprint()) {
            return Err("replay requires the packet's fingerprint-bound checkpoint".into());
        }
        Ok(artifact)
    }

    fn packet(&self) -> &Dse8Packet {
        match self {
            Self::Wrapped(decoder) => decoder.packet(),
            Self::Bare(packet) => packet,
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    if let Command::Materialize { packets, out } = &cli.command {
        let name = out.file_name().ok_or("missing packet name")?;
        fs::copy(packets.join(name), out)?;
        return Ok(());
    }
    let model = V8Model::from_safetensors(
        &fs::read(cli.weights.ok_or("replay requires --weights")?)?,
        cli.heads,
    )?;
    match cli.command {
        Command::Materialize { .. } => unreachable!(),
        Command::Sample {
            packet,
            arm,
            rows,
            seed,
            out,
        } => {
            let bytes = fs::read(packet)?;
            let artifact = Artifact::open(&bytes, &model)?;
            let decoder = match &artifact {
                Artifact::Wrapped(decoder) => Some(decoder),
                Artifact::Bare(_) => None,
            };
            sample(decoder, artifact.packet(), &model, arm, rows, seed, &out)?;
        }
        Command::Trace {
            packet,
            arm,
            rows,
            seed,
        } => {
            let artifact = Artifact::open(&fs::read(packet)?, &model)?;
            let plan = ReplayPlanV8::new(artifact.packet(), model.weights(), arm.into())?;
            trace::write(artifact.packet(), &plan, seed, rows)?;
        }
        Command::ReserveCode {
            fit,
            target,
            task,
            seed,
            budget,
            code_tier,
            out,
        } => {
            let raw = RawTable::read(&fit)?;
            let target = raw.column_index(&target)?;
            let opts = AutoV8Options {
                budget,
                code_tier,
                format1: dope_core::compile::FitOptions {
                    task: Some(match task {
                        ReplayTask::Regression => Task::Regression,
                        ReplayTask::Classification => Task::Classification,
                    }),
                    seed,
                    ..Default::default()
                },
                ..Default::default()
            };
            let fitted = fit_dpa1_v8_with_code_reservation(&raw, target, &opts, &model)?;
            fs::write(out, fitted.bytes)?;
        }
        Command::Inspect { packet } => {
            let bytes = fs::read(packet)?;
            let artifact = Artifact::open(&bytes, &model)?;
            println!(
                "{}",
                serde_json::to_string(&Layout::from(artifact.packet()))?
            );
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("V8 replay: {error}");
        std::process::exit(1);
    }
}

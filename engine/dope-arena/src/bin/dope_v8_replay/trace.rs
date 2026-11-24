use dope_core::single_encoding::v8::rng_v8::{
    counter_normal, counter_student_t_scale, counter_uniform, STREAM_ATOM, STREAM_FACTOR,
    STREAM_MISSING, STREAM_NOISE, STREAM_REGIME, STREAM_TARGET,
};
use dope_core::single_encoding::v8::{Dse8Packet, ReplayPlanV8, V8_MAX_RANK};
use serde::Serialize;
use std::array;
use std::io::{self, Write};

#[derive(Serialize)]
struct Draws {
    factor_normal: [f32; V8_MAX_RANK],
    noise_normal: Vec<f32>,
    student_t_scale: f32,
    regime_uniform: f64,
    missing_uniform: Vec<f32>,
    atom_uniform: Vec<f32>,
    nominal_uniform: Vec<f64>,
    target_normal: f32,
    target_uniform: f64,
}

fn draws(packet: &Dse8Packet, seed: u64, index: u64) -> Draws {
    let columns = 0..u64::from(packet.header.p_eff);
    Draws {
        factor_normal: array::from_fn(|lane| {
            counter_normal(seed, index, STREAM_FACTOR, lane as u64) as f32
        }),
        noise_normal: columns
            .clone()
            .map(|column| counter_normal(seed, index, STREAM_NOISE, column) as f32)
            .collect(),
        student_t_scale: counter_student_t_scale(seed, index, packet.header.inv_nu),
        regime_uniform: counter_uniform(seed, index, STREAM_REGIME, 0),
        missing_uniform: columns
            .clone()
            .map(|column| counter_uniform(seed, index, STREAM_MISSING, column) as f32)
            .collect(),
        atom_uniform: columns
            .map(|column| counter_uniform(seed, index, STREAM_ATOM, column) as f32)
            .collect(),
        nominal_uniform: (0..packet.nominal_groups.len())
            .map(|group| counter_uniform(seed, index, STREAM_ATOM, 1024 + group as u64))
            .collect(),
        target_normal: counter_normal(seed, index, STREAM_TARGET, 0) as f32,
        target_uniform: counter_uniform(seed, index, STREAM_TARGET, 0),
    }
}

#[derive(Serialize)]
struct Trace<'a> {
    schema: &'static str,
    row: u64,
    draws: Draws,
    factors: [f32; V8_MAX_RANK],
    latents: &'a [f32],
    x_unit: &'a [f32],
    missing: &'a [bool],
    conditional: &'a [f32],
    y_unit: f32,
}

pub(super) fn write(
    packet: &Dse8Packet,
    plan: &ReplayPlanV8,
    seed: u64,
    rows: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for index in 0..rows as u64 {
        let (mut factors, mut latents) = ([0.0; V8_MAX_RANK], Vec::new());
        let row = plan.row_at(seed, index, |factor, latent| {
            factors = factor;
            latents = latent;
        });
        serde_json::to_writer(
            &mut stdout,
            &Trace {
                schema: "dope.v8-training-parity-trace/v1",
                row: index,
                draws: draws(packet, seed, index),
                factors,
                latents: &latents,
                x_unit: &row.x_unit,
                missing: &row.missing_active,
                conditional: &row.logits[..packet.linear.o_dim],
                y_unit: row.y_unit,
            },
        )?;
        stdout.write_all(b"\n")?;
    }
    stdout.flush()?;
    Ok(())
}

//! `DPA1` tag `8` (`Encoder::V8`) container integration for `dope fit --format v8|v8-analytic|auto-v8`
//! and `dope sample`.
//!
//! Layout inside `DPA1(tag=8, fp)`:
//! - `fp == NO_FINGERPRINT` (`[0; 8]`): weightless analytic Step-0 `DSE8` v1 packet.
//! - `fp != NO_FINGERPRINT`: neural `SingleEncV8` packet; `fp` MUST match both the loaded
//!   `SingleEncV8Weights::fingerprint()` and the inner `TAG_WEIGHTS_FP` TLV section.
//! - `payload = v7_schema::encode_section(schema, anonymize) || dse8_packet_bytes`.
//!
//! The entire file (13-byte `DPA1` header + compressed schema section + `DSE8` packet) strictly
//! respects the requested byte budget (default `2048 B`).

use super::container::{
    fingerprint_hex, wrap_dpa1, Dpa1, Encoder, DPA1_HEADER_BYTES, NO_FINGERPRINT,
};
mod sampling;

use super::v7_schema::{decode_section, V7Schema};
use super::v8::{fit_auto_v8_with_payload_room, AutoV8Options, V8Choice};
use super::v8_budget::prepare_dpa_v8_budget;
pub use super::v8_budget::{
    compare_reserved_and_serving_dpa1, dpa_v8_wrapper_bytes_from_json,
    estimate_dpa_v8_wrapper_bytes, DpaV8KindSpec, DpaV8SchemaSpec, Dse8PackComparison,
    Dse8SectionDiff,
};
use crate::error::{DopeError, Result};
use crate::schema::{infer, IntakeOptions, Kind, Schema};
use crate::single_encoding::v8::{
    fit_dse8_packet, fit_dse8_packet_with_code_reservation, fit_dse8_packet_with_encoder,
    fit_unit_meta, unpack_dse8_packet, Dse8FitOptions, Dse8Packet, SingleEncV8Weights,
};
use crate::table::{is_missing, parse_number, RawTable};
use serde::Serialize;
use std::time::Instant;

/// SingleEncV8 weights and SHA-256 identity used by `DPA1` tag `8` headers and `TAG_WEIGHTS_FP`.
#[derive(Clone, Debug)]
pub struct V8Model {
    weights: SingleEncV8Weights,
}

impl V8Model {
    /// Fail-closed parser for `SingleEncV8` SafeTensors (`F32` or `BF16`, `rank <= 2`, `<= 1 GiB`)
    /// with automatic attention-head inference.
    pub fn from_safetensors_auto(bytes: &[u8]) -> Result<Self> {
        let weights = SingleEncV8Weights::from_safetensors_auto(bytes)?;
        Ok(Self { weights })
    }

    /// Fail-closed parser for `SingleEncV8` SafeTensors (`F32` or `BF16`, `rank <= 2`, `<= 1 GiB`).
    pub fn from_safetensors(bytes: &[u8], n_heads: usize) -> Result<Self> {
        let weights = SingleEncV8Weights::from_safetensors_bytes(bytes, n_heads)?;
        Ok(Self { weights })
    }

    pub fn weights(&self) -> &SingleEncV8Weights {
        &self.weights
    }

    pub fn sha256(&self) -> [u8; 32] {
        self.weights.sha256()
    }

    pub fn fingerprint(&self) -> [u8; 8] {
        self.weights.fingerprint()
    }
}

/// Requested V8 fitting mode from `dope fit --format ...`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum V8DpaMode {
    /// 3-way grouped acceptance guard: `Format1` vs `DSE8 Analytic` vs `DSE8 Neural`.
    AutoV8,
    /// Explicit `DSE8` (neural if `model.is_some()`, else analytic Step-0).
    V8,
    /// Explicit weightless `DSE8` analytic Step-0 (`NO_FINGERPRINT`).
    V8Analytic,
}

impl V8DpaMode {
    pub fn name(self) -> &'static str {
        match self {
            V8DpaMode::AutoV8 => "auto-v8",
            V8DpaMode::V8 => "v8",
            V8DpaMode::V8Analytic => "v8-analytic",
        }
    }
}

/// Value-free summary of a V8 `DPA1` fit (never echoes raw cells, headers, or extrema).
#[derive(Clone, Debug, Serialize)]
pub struct V8DpaReport {
    pub mode: String,
    pub encoder: String,
    pub choice: V8Choice,
    pub bytes: usize,
    pub budget: usize,
    pub task: String,
    pub rows: usize,
    pub features: usize,
    pub score_format1: Option<f64>,
    pub score_dse8_analytic: Option<f64>,
    pub score_dse8_neural: Option<f64>,
    pub v8_weights: String,
    pub note: String,
    pub seconds: f64,
}

pub struct V8DpaFitted {
    pub bytes: Vec<u8>,
    pub encoder: Encoder,
    pub choice: V8Choice,
    pub report: V8DpaReport,
}

fn task_id_for(classes: usize) -> u8 {
    match classes {
        0 => 0,
        2 => 1,
        _ => 2,
    }
}

fn target_value(schema: &Schema, cell: &str) -> Option<f64> {
    if is_missing(cell) {
        return None;
    }
    match &schema.target.kind {
        Kind::Numeric(_) => parse_number(cell),
        Kind::Categorical(spec) => spec.levels.iter().position(|l| l == cell).map(|k| k as f64),
    }
}

type PacketInputs = (Vec<Vec<f64>>, Vec<f64>, Vec<u64>);

fn packet_inputs(raw: &RawTable, target: usize, schema: &Schema) -> Result<PacketInputs> {
    let cells = &raw.columns[target];
    let rows: Vec<usize> = (0..raw.rows())
        .filter(|&r| target_value(schema, &cells[r]).is_some())
        .collect();
    let y: Vec<f64> = rows
        .iter()
        .filter_map(|&r| target_value(schema, &cells[r]))
        .collect();
    let row_ids: Vec<u64> = rows.iter().map(|&r| r as u64).collect();
    let mut x = Vec::with_capacity(schema.features.len());
    for f in &schema.features {
        let c = raw
            .headers
            .iter()
            .position(|h| *h == f.name)
            .ok_or_else(|| DopeError::Input("a feature column is missing from the input".into()))?;
        let col = &raw.columns[c];
        x.push(
            rows.iter()
                .map(|&r| match &f.kind {
                    Kind::Numeric(_) => parse_number(&col[r]).unwrap_or(f64::NAN),
                    Kind::Categorical(spec) if !is_missing(&col[r]) => {
                        spec.level_of(&col[r]).map_or(f64::NAN, |k| k as f64)
                    }
                    Kind::Categorical(_) => f64::NAN,
                })
                .collect(),
        );
    }
    Ok((x, y, row_ids))
}

/// Fit a V8 or 3-way Auto-V8 `DPA1` artifact strictly within `opts.budget` bytes.
pub fn fit_dpa1_v8(
    raw: &RawTable,
    target: usize,
    opts: &AutoV8Options,
    model: Option<&V8Model>,
    mode: V8DpaMode,
) -> Result<V8DpaFitted> {
    fit_dpa1_v8_inner(raw, target, opts, model, mode, false)
}

/// Fit the experimental direct neural arm with code charged before optional analytic sections.
pub fn fit_dpa1_v8_with_code_reservation(
    raw: &RawTable,
    target: usize,
    opts: &AutoV8Options,
    model: &V8Model,
) -> Result<V8DpaFitted> {
    fit_dpa1_v8_inner(raw, target, opts, Some(model), V8DpaMode::V8, true)
}

fn fit_dpa1_v8_inner(
    raw: &RawTable,
    target: usize,
    opts: &AutoV8Options,
    model: Option<&V8Model>,
    mode: V8DpaMode,
    reserve_code: bool,
) -> Result<V8DpaFitted> {
    let start = Instant::now();
    let budget = opts.budget;
    let room = budget
        .checked_sub(DPA1_HEADER_BYTES)
        .filter(|&r| r > 0)
        .ok_or_else(|| {
            DopeError::Budget(format!(
                "{budget} bytes cannot hold the {DPA1_HEADER_BYTES}-byte DPA1 header and a payload"
            ))
        })?;

    let intake = IntakeOptions {
        task: opts.format1.task,
        ..Default::default()
    };
    let schema = infer(raw, target, &intake)?;
    let v7s = V7Schema::from_schema(&schema);
    let prep = prepare_dpa_v8_budget(&v7s, budget, opts.format1.anonymize)?;
    let section = prep.schema_section;
    let (x_raw, y_raw, row_ids) = packet_inputs(raw, target, &schema)?;
    let tid = task_id_for(v7s.classes());
    let active_model = match mode {
        V8DpaMode::V8Analytic => None,
        V8DpaMode::V8 | V8DpaMode::AutoV8 => model,
    };

    let (encoder, choice, fp, payload, sf1, sp0, snn, note) = match mode {
        V8DpaMode::V8Analytic | V8DpaMode::V8 => {
            if prep.wrapper_bytes > budget {
                return Err(DopeError::Budget(
                    "the V8 column schema alone exceeds the budget (try --anonymize)".into(),
                ));
            }
            let packet_room = prep.packet_budget;
            let unit_meta = fit_unit_meta(&x_raw, &y_raw, tid, opts.normalized_only).ok();
            let dse8_opts = Dse8FitOptions {
                task_id: tid,
                budget_bytes: packet_room,
                normalized_only: opts.normalized_only,
                enable_trees: opts.enable_trees,
                enable_regimes: opts.enable_regimes,
                enable_effects: opts.enable_effects,
                enable_residual_model: opts.enable_residual_model,
                enable_skew_t: opts.enable_skew_t,
                code_tier: if active_model.is_some() {
                    opts.code_tier.max(1)
                } else {
                    0
                },
                reserve_neural_code: false,
                col_meta: unit_meta.as_deref(),
                fit_row_ids: active_model
                    .filter(|m| m.weights().active_rank_v1)
                    .map(|_| row_ids.as_slice()),
                seed: opts.format1.seed,
            };
            let (pkt, ch, fp) = match active_model {
                Some(m) => (
                    if reserve_code {
                        fit_dse8_packet_with_code_reservation(
                            &x_raw,
                            &y_raw,
                            &dse8_opts,
                            m.weights(),
                        )?
                    } else {
                        fit_dse8_packet_with_encoder(&x_raw, &y_raw, &dse8_opts, m.weights())?
                    },
                    V8Choice::Dse8Neural,
                    m.fingerprint(),
                ),
                None => (
                    fit_dse8_packet(&x_raw, &y_raw, &dse8_opts)?,
                    V8Choice::Dse8Analytic,
                    NO_FINGERPRINT,
                ),
            };
            let pl = [&section[..], &pkt[..]].concat();
            (
                Encoder::V8,
                ch,
                fp,
                pl,
                None,
                None,
                None,
                format!("direct {}", mode.name()),
            )
        }
        V8DpaMode::AutoV8 => {
            // If the V8 schema section leaves too little room for a DSE8 packet, fall back
            // directly to Format 1 inside `room` instead of failing over_budget!
            let packet_room = prep.packet_budget;
            if packet_room < 96 {
                let mut f1_opts = opts.format1.clone();
                f1_opts.budget = Some(room);
                f1_opts.task = Some(schema.task);
                let f1 = crate::compile::fit(raw, target, &f1_opts)?;
                (
                    Encoder::Format1,
                    V8Choice::Format1,
                    NO_FINGERPRINT,
                    f1.bytes,
                    None,
                    None,
                    None,
                    "format1_fallback: schema_exceeds_v8_room".into(),
                )
            } else {
                // Fit Format 1 inside `room`, DSE8 inside `packet_room`, using the 3-way guard!
                let mut inner_opts = opts.clone();
                inner_opts.budget = packet_room;
                inner_opts.format1.budget = Some(room);
                let fitted = fit_auto_v8_with_payload_room(
                    raw,
                    target,
                    &inner_opts,
                    active_model.map(|m| m.weights()),
                    Some(room),
                )?;
                let r = fitted.report;
                match fitted.choice {
                    V8Choice::Format1 => (
                        Encoder::Format1,
                        V8Choice::Format1,
                        NO_FINGERPRINT,
                        fitted.bytes,
                        r.score_format1,
                        r.score_dse8_analytic,
                        r.score_dse8_neural,
                        r.note,
                    ),
                    V8Choice::Dse8Analytic => {
                        let pl = [&section[..], &fitted.bytes[..]].concat();
                        (
                            Encoder::V8,
                            V8Choice::Dse8Analytic,
                            NO_FINGERPRINT,
                            pl,
                            r.score_format1,
                            r.score_dse8_analytic,
                            r.score_dse8_neural,
                            r.note,
                        )
                    }
                    V8Choice::Dse8Neural => {
                        let fp = active_model.map_or(NO_FINGERPRINT, V8Model::fingerprint);
                        let pl = [&section[..], &fitted.bytes[..]].concat();
                        (
                            Encoder::V8,
                            V8Choice::Dse8Neural,
                            fp,
                            pl,
                            r.score_format1,
                            r.score_dse8_analytic,
                            r.score_dse8_neural,
                            r.note,
                        )
                    }
                }
            }
        }
    };

    let bytes = wrap_dpa1(encoder, fp, &payload);
    if bytes.len() > budget {
        return Err(DopeError::Budget(format!(
            "the {} artifact ({} B) exceeds {budget} bytes",
            encoder.name(),
            bytes.len()
        )));
    }
    let task_s = format!("{:?}", schema.task).to_lowercase();
    let report = V8DpaReport {
        mode: mode.name().into(),
        encoder: encoder.name().into(),
        choice,
        bytes: bytes.len(),
        budget,
        task: task_s,
        rows: y_raw.len(),
        features: schema.features.len(),
        score_format1: sf1,
        score_dse8_analytic: sp0,
        score_dse8_neural: snn,
        v8_weights: fingerprint_hex(&fp),
        note,
        seconds: start.elapsed().as_secs_f64(),
    };
    Ok(V8DpaFitted {
        bytes,
        encoder,
        choice,
        report,
    })
}

/// Source-free decoder for `DPA1` tag `8` (`Encoder::V8`) artifacts.
pub struct V8Decoder {
    schema: V7Schema,
    packet: Dse8Packet,
    header_fp: [u8; 8],
}

impl V8Decoder {
    /// Open and validate a `DPA1` tag `8` container. Fails closed on fingerprint mismatch,
    /// missing required neural weights (unless `zero_init`), schema mismatch, or packet corruption.
    pub fn open(dpa: &Dpa1<'_>, model: Option<&V8Model>, zero_init: bool) -> Result<Self> {
        if dpa.encoder != Encoder::V8 {
            return Err(DopeError::Artifact(
                "DPA1 payload is not a V8 packet".into(),
            ));
        }
        if dpa.fingerprint != NO_FINGERPRINT && !zero_init {
            let Some(m) = model else {
                return Err(DopeError::Artifact(format!(
                    "V8 artifact requires SingleEncV8 weights {} (pass --v8-weights or --v8-zero-init)",
                    fingerprint_hex(&dpa.fingerprint)
                )));
            };
            if m.fingerprint() != dpa.fingerprint {
                return Err(DopeError::Artifact(format!(
                    "V8 weights fingerprint mismatch: artifact needs {} but loaded weights are {}",
                    fingerprint_hex(&dpa.fingerprint),
                    fingerprint_hex(&m.fingerprint())
                )));
            }
        }
        let (schema, used) = decode_section(dpa.payload)?;
        let packet = unpack_dse8_packet(&dpa.payload[used..])?;
        if packet.card.x_cols_raw != schema.features.len() {
            return Err(DopeError::Artifact(
                "V8 payload is inconsistent (feature count)".into(),
            ));
        }
        let k = schema.classes();
        if packet.header.task_id != task_id_for(k) {
            return Err(DopeError::Artifact(
                "V8 payload is inconsistent (task)".into(),
            ));
        }
        Ok(Self {
            schema,
            packet,
            header_fp: dpa.fingerprint,
        })
    }

    pub fn headers(&self) -> Vec<String> {
        self.schema.headers()
    }

    pub fn features(&self) -> usize {
        self.schema.features.len()
    }

    pub fn is_neural(&self) -> bool {
        self.header_fp != NO_FINGERPRINT
    }
}

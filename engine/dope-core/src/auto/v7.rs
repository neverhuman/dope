//! The V7 candidate of `--format auto`: the SingleEncV7 DSE7 v2 packet of the table's
//! numbers, behind the column schema that turns decoded numbers back into cells.
//!
//! Features follow the Format 1 intake schema (same columns, order and labels): numeric
//! cells enter the packet in raw units, categorical cells as their level index. The target
//! is the raw value (regression) or the class index `0..k` (classification), so the packet's
//! class table holds indices into the stored labels. Rows without a target are left out.

use super::container::{Dpa1, Encoder};
use super::v7_schema::{decode_section, encode_section, V7Kind, V7Schema};
use crate::error::{DopeError, Result};
use crate::rng::Rng;
use crate::schema::{infer, IntakeOptions, Kind, Schema, Task};
use crate::single_encoding::{
    fit_dse7_packet, sample_zero_init_rows, unpack_dse7_packet, DecodedV7Row, Dse7Header,
    RefinedV7Packet, ScaleCardV7, SingleEncV7Packet, SingleEncV7Weights, MAX_COLS, MAX_RANK,
};
use crate::table::{is_missing, parse_number, RawTable};
use sha2::{Digest, Sha256};
use std::io::Write;

/// Label stream of `dope_v7_emit sample`, kept so both decoders draw the same classes.
const LABEL_SALT: u64 = 0xD5E7_CAFE_BABE_1234;

/// SingleEncV7 weights and the SHA-256 that identifies them in `DPA1` headers.
#[derive(Clone, Debug)]
pub struct V7Model {
    weights: SingleEncV7Weights,
    sha256: [u8; 32],
}

impl V7Model {
    /// Load canonical F32 SingleEncV7 SafeTensors (fail closed, as the runtime loader).
    pub fn from_safetensors(bytes: &[u8], n_heads: usize) -> Result<Self> {
        let weights = SingleEncV7Weights::from_safetensors_bytes(bytes, n_heads)?;
        Ok(Self {
            weights,
            sha256: Sha256::digest(bytes).into(),
        })
    }

    pub fn weights(&self) -> &SingleEncV7Weights {
        &self.weights
    }

    /// SHA-256 of the SafeTensors file the weights were loaded from.
    pub fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    /// First 8 bytes of [`V7Model::sha256`], stored in every V7 `DPA1` header.
    pub fn fingerprint(&self) -> [u8; 8] {
        let mut f = [0u8; 8];
        f.copy_from_slice(&self.sha256[..8]);
        f
    }
}

/// A fitted V7 payload: the stored schema section, the packet, and the full (never
/// anonymized) in-memory schema used by the internal check.
pub(crate) struct V7Fit {
    pub(crate) schema: V7Schema,
    section: Vec<u8>,
    packet: Vec<u8>,
}

impl V7Fit {
    pub(crate) fn payload(&self) -> Vec<u8> {
        [&self.section[..], &self.packet[..]].concat()
    }

    pub(crate) fn decoder(&self) -> Result<V7Decoder> {
        V7Decoder::new(self.schema.clone(), &self.packet)
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

/// Column-major packet inputs (`x[feature][row]`, NaN = missing) and targets.
fn packet_inputs(
    raw: &RawTable,
    target: usize,
    schema: &Schema,
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    let cells = &raw.columns[target];
    let rows: Vec<usize> = (0..raw.rows())
        .filter(|&r| target_value(schema, &cells[r]).is_some())
        .collect();
    let y = rows
        .iter()
        .filter_map(|&r| target_value(schema, &cells[r]))
        .collect();
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
    Ok((x, y))
}

/// DSE7 task id: 0 regression, 1 binary, 2 multiclass.
fn task_id(classes: usize) -> u8 {
    match classes {
        0 => 0,
        2 => 1,
        _ => 2,
    }
}

/// Fit the V7 payload of `raw` within `room` bytes (schema section plus packet).
pub(crate) fn fit(
    raw: &RawTable,
    target: usize,
    task: Option<Task>,
    anonymize: bool,
    room: usize,
) -> Result<V7Fit> {
    let intake = IntakeOptions {
        task,
        ..Default::default()
    };
    let schema = infer(raw, target, &intake)?;
    let v7 = V7Schema::from_schema(&schema);
    let section = encode_section(&v7, anonymize)?;
    let packet_room = room.checked_sub(section.len()).ok_or_else(|| {
        DopeError::Budget("the V7 column schema alone exceeds the budget (try --anonymize)".into())
    })?;
    let (x, y) = packet_inputs(raw, target, &schema)?;
    let packet =
        fit_dse7_packet(&x, &y, task_id(v7.classes()), packet_room).map_err(|e| match e {
            DopeError::Budget(_) => DopeError::Budget(format!(
                "the V7 packet does not fit the {packet_room} bytes left after its column schema"
            )),
            other => other,
        })?;
    Ok(V7Fit {
        schema: v7,
        section,
        packet,
    })
}

/// How V7 rows are decoded.
#[derive(Clone, Copy, Debug)]
pub enum V7Decode<'a> {
    /// The SingleEncV7 network: the refined packet, then counter rows.
    Trained(&'a V7Model),
    /// The analytic zero-init decoder (no neural weights; the step-0 baseline that
    /// `dope_v7_emit sample` uses without `--checkpoint`). Rows are drawn in one batch.
    ZeroInit,
}

/// A validated V7 payload, ready to sample.
pub struct V7Decoder {
    schema: V7Schema,
    pkt: SingleEncV7Packet,
    hdr: Dse7Header,
    card: ScaleCardV7,
}

impl V7Decoder {
    fn new(schema: V7Schema, packet: &[u8]) -> Result<Self> {
        let (pkt, hdr, card) = unpack_dse7_packet(packet)?;
        let bad = |what: &str| DopeError::Artifact(format!("V7 payload is inconsistent ({what})"));
        if card.x_cols_raw != schema.features.len() {
            return Err(bad("feature count"));
        }
        let k = schema.classes();
        if hdr.task_id != task_id(k) {
            return Err(bad("task"));
        }
        if k > 0 && card.classes.iter().any(|&c| c < 0 || c as usize >= k) {
            return Err(bad("class table"));
        }
        Ok(Self {
            schema,
            pkt,
            hdr,
            card,
        })
    }

    fn require_v7(dpa: &Dpa1<'_>) -> Result<()> {
        match dpa.encoder {
            Encoder::V7 => Ok(()),
            Encoder::Format1 | Encoder::V8 => Err(DopeError::Artifact(
                "DPA1 payload is not a V7 packet".into(),
            )),
        }
    }

    /// Open a V7 payload for [`V7Decode::ZeroInit`], which uses no weights, so the
    /// weights fingerprint is not checked.
    pub fn open_weightless(dpa: &Dpa1<'_>) -> Result<Self> {
        Self::require_v7(dpa)?;
        let (schema, used) = decode_section(dpa.payload)?;
        Self::new(schema, &dpa.payload[used..])
    }

    /// Open the V7 payload of a parsed container with the weights it was fitted with.
    pub fn open(dpa: &Dpa1<'_>, model: &V7Model) -> Result<Self> {
        Self::require_v7(dpa)?;
        if dpa.fingerprint != model.fingerprint() {
            return Err(DopeError::Artifact(format!(
                "V7 weights fingerprint mismatch: the artifact needs weights {} but the loaded weights are {}; sample it with the dope binary or --v7-weights file that fitted it, or refit",
                super::fingerprint_hex(&dpa.fingerprint),
                super::fingerprint_hex(&model.fingerprint())
            )));
        }
        Self::open_weightless(dpa)
    }

    /// Output header: features, then the target last.
    pub fn headers(&self) -> Vec<String> {
        self.schema.headers()
    }

    pub fn features(&self) -> usize {
        self.schema.features.len()
    }

    /// Decode `rows` rows; `emit(features, target)` receives each row as cells.
    fn each_row(
        &self,
        how: V7Decode<'_>,
        rows: usize,
        seed: u64,
        mut emit: impl FnMut(Vec<String>, String) -> Result<()>,
    ) -> Result<()> {
        let mut labels = Rng::new(seed ^ LABEL_SALT);
        let mut put = |row: &DecodedV7Row| {
            let x = self.card.inverse_row_x(&row.x);
            let cells = self
                .schema
                .features
                .iter()
                .zip(&x)
                .map(|(c, &v)| cell(&c.kind, v))
                .collect();
            let y = self.target_cell(row, &mut labels);
            emit(cells, y)
        };
        match how {
            V7Decode::Trained(model) => {
                let refined = model.weights.refine_packet(&self.pkt);
                (0..rows).try_for_each(|r| put(&counter_row(&refined, seed, r)))
            }
            V7Decode::ZeroInit => sample_zero_init_rows(&self.pkt, rows, seed)
                .iter()
                .try_for_each(put),
        }
    }

    fn target_cell(&self, row: &DecodedV7Row, labels: &mut Rng) -> String {
        let class_label = |pick: usize| {
            let id = self.card.classes.get(pick).copied().unwrap_or(pick as i32);
            match &self.schema.target.kind {
                V7Kind::Categorical(l) => l[(id.max(0) as usize).min(l.len() - 1)].clone(),
                _ => String::new(),
            }
        };
        match self.hdr.task_id {
            0 => {
                let y = (self.card.y_mean as f64 + self.card.y_sd as f64 * row.y_reg as f64) as f32;
                cell(&self.schema.target.kind, y)
            }
            1 => {
                let p1 = 1.0 / (1.0 + (-row.logits[0].clamp(-10.0, 10.0)).exp());
                class_label(usize::from(labels.uniform() < p1 as f64))
            }
            _ => {
                let k = self.hdr.k_eff as usize;
                let max_l = row.logits[..k]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let probs: Vec<f64> = row.logits[..k]
                    .iter()
                    .map(|&l| ((l - max_l).clamp(-20.0, 0.0) as f64).exp())
                    .collect();
                let u = labels.uniform() * probs.iter().sum::<f64>().max(1e-12);
                let (mut acc, mut pick) = (0.0, k.saturating_sub(1));
                for (c, &p) in probs.iter().enumerate() {
                    acc += p;
                    if u <= acc {
                        pick = c;
                        break;
                    }
                }
                class_label(pick)
            }
        }
    }

    /// Stream `rows` synthetic rows as CSV (header first, target last).
    pub fn write_csv<W: Write>(
        &self,
        how: V7Decode<'_>,
        rows: usize,
        seed: u64,
        w: W,
    ) -> Result<()> {
        let mut csv = csv::Writer::from_writer(w);
        csv.write_record(self.headers())?;
        self.each_row(how, rows, seed, |mut cells, y| {
            cells.push(y);
            csv.write_record(&cells)?;
            Ok(())
        })?;
        csv.flush()?;
        Ok(())
    }

    /// The same rows as a table (internal check).
    pub(crate) fn table(&self, how: V7Decode<'_>, rows: usize, seed: u64) -> Result<RawTable> {
        let mut columns = vec![Vec::with_capacity(rows); self.features() + 1];
        self.each_row(how, rows, seed, |cells, y| {
            for (col, v) in columns.iter_mut().zip(cells.into_iter().chain([y])) {
                col.push(v);
            }
            Ok(())
        })?;
        Ok(RawTable {
            headers: self.headers(),
            columns,
        })
    }
}

/// Row `index` of `SingleEncV7Weights::sample_counter_rows(pkt, _, seed)`, decoded on its own
/// so rows can be streamed (same counter seed and noise order).
pub(crate) fn counter_row(refined: &RefinedV7Packet, seed: u64, index: usize) -> DecodedV7Row {
    let mut rng = Rng::new(seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut zf = [0.0f32; MAX_RANK];
    for v in &mut zf {
        *v = rng.normal() as f32;
    }
    let mut zn = [0.0f32; MAX_COLS];
    for v in &mut zn {
        *v = rng.normal() as f32;
    }
    let yn = rng.normal() as f32;
    SingleEncV7Weights::decode_row(refined, &zf, &zn, yn)
}

/// One decoded number as a cell; non-finite numbers become missing cells.
fn cell(kind: &V7Kind, v: f32) -> String {
    match kind {
        V7Kind::Categorical(labels) if v.is_finite() => {
            labels[(v.round().max(0.0) as usize).min(labels.len() - 1)].clone()
        }
        _ if !v.is_finite() => String::new(),
        V7Kind::Integer => format!("{}", v.round() + 0.0),
        _ => format!("{}", v + 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_round_integers_snap_levels_and_drop_non_finite() {
        let levels = V7Kind::Categorical(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(cell(&levels, 1.4), "b");
        assert_eq!(cell(&levels, -3.0), "a");
        assert_eq!(cell(&levels, 9.0), "c");
        assert_eq!(cell(&levels, f32::NAN), "");
        assert_eq!(cell(&V7Kind::Integer, 2.6), "3");
        assert_eq!(cell(&V7Kind::Integer, -0.2), "0");
        assert_eq!(cell(&V7Kind::Real, 0.1), "0.1");
        assert_eq!(cell(&V7Kind::Real, -0.0), "0");
        assert_eq!(cell(&V7Kind::Real, f32::INFINITY), "");
        assert_eq!((task_id(0), task_id(2), task_id(5)), (0, 1, 2));
    }

    #[test]
    fn streamed_rows_equal_sample_counter_rows() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/single-enc-v7/model.safetensors"
        );
        let model = V7Model::from_safetensors(&std::fs::read(path).unwrap(), 4).unwrap();
        let mut rng = Rng::new(1);
        let n = 200;
        let x: Vec<Vec<f64>> = (0..5)
            .map(|_| (0..n).map(|_| rng.normal()).collect())
            .collect();
        let y: Vec<f64> = (0..n).map(|r| x[0][r] + 0.5 * x[1][r]).collect();
        let (pkt, _, _) = unpack_dse7_packet(&fit_dse7_packet(&x, &y, 0, 2048).unwrap()).unwrap();
        let all = model.weights().sample_counter_rows(&pkt, 37, 99);
        let refined = model.weights().refine_packet(&pkt);
        for (i, row) in all.iter().enumerate() {
            let one = counter_row(&refined, 99, i);
            assert_eq!((one.x, one.logits), (row.x, row.logits), "row {i}");
            assert_eq!(one.y_reg.to_bits(), row.y_reg.to_bits(), "row {i}");
        }
    }
}

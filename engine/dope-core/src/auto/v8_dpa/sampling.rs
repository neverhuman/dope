use super::NO_FINGERPRINT;
use super::{V8Decoder, V8Model};
use crate::auto::v7_schema::V7Kind;
use crate::error::{DopeError, Result};
use crate::single_encoding::v8::{
    build_decode_plan_v8, decode_dse8_row_at, DecodedV8Row, Dse8Packet,
};
use std::io::Write;

impl V8Decoder {
    /// Decode `rows` synthetic rows strictly from the artifact and write target-last CSV.
    pub fn write_csv<W: Write>(
        &self,
        model: Option<&V8Model>,
        zero_init: bool,
        rows: usize,
        seed: u64,
        w: W,
    ) -> Result<()> {
        let use_weights = if zero_init || self.header_fp == NO_FINGERPRINT {
            None
        } else {
            model.map(|m| m.weights())
        };
        let plan = build_decode_plan_v8(&self.packet, use_weights)?;
        self.write_decoded_csv(
            (0..rows).map(|r| decode_dse8_row_at(&plan, seed, r as u64)),
            w,
        )
    }

    pub fn packet(&self) -> &Dse8Packet {
        &self.packet
    }

    /// Write decoded V8 rows in the same target-last format as the served decoder.
    pub fn write_decoded_csv<W: Write>(
        &self,
        rows: impl IntoIterator<Item = DecodedV8Row>,
        w: W,
    ) -> Result<()> {
        let mut csv = csv::Writer::from_writer(w);
        csv.write_record(self.headers())?;
        let p = self.schema.features.len();
        let mut row_cells = Vec::with_capacity(p + 1);
        for row in rows {
            if row.x_raw.len() != p {
                return Err(DopeError::Artifact("dse8_decoded_width_mismatch".into()));
            }
            row_cells.clear();
            for (feature, value) in self.schema.features.iter().zip(row.x_raw) {
                row_cells.push(format_cell(&feature.kind, value));
            }
            row_cells.push(format_cell(&self.schema.target.kind, row.y_raw));
            csv.write_record(&row_cells)?;
        }
        csv.flush()?;
        Ok(())
    }
}

fn format_cell(kind: &V7Kind, value: f64) -> String {
    if !value.is_finite() {
        return String::new();
    }
    match kind {
        V7Kind::Real => {
            let v = if value == 0.0 { 0.0 } else { value };
            format!("{v}")
        }
        V7Kind::Integer => {
            let r = value.round();
            format!("{}", if r == 0.0 { 0 } else { r as i64 })
        }
        V7Kind::Categorical(labels) => {
            if labels.is_empty() {
                return String::new();
            }
            let idx = (value.round() as isize).clamp(0, labels.len() as isize - 1) as usize;
            labels[idx].clone()
        }
    }
}

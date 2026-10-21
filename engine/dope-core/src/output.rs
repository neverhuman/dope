//! Turning model-space samples into tables.

use crate::encode::MlEncoder;
use crate::error::Result;
use crate::program::Program;
use crate::sample::Samples;
use crate::schema::format_number;
use crate::table::write_csv;
use std::io::Write;

/// Write synthetic rows in original units and labels (target last).
pub fn write_raw<W: Write>(program: &Program, s: &Samples, writer: W) -> Result<()> {
    let mut headers: Vec<String> = program
        .schema
        .features
        .iter()
        .map(|f| f.name.clone())
        .collect();
    headers.push(program.schema.target.name.clone());
    let d = program.features();
    write_csv(writer, &headers, s.y.len(), |r, c| {
        if c < d {
            program.schema.features[c].to_raw(s.x[c][r])
        } else {
            program.schema.target_to_raw(s.y[r])
        }
    })
}

/// Write synthetic rows ML-ready (one-hot, [0,1] numerics, target last).
pub fn write_ml<W: Write>(enc: &MlEncoder<'_>, s: &Samples, writer: W) -> Result<()> {
    let headers = enc.headers();
    let rows = enc.rows(&s.x, &s.y);
    write_csv(writer, &headers, rows.len(), |r, c| {
        format_number(rows[r][c])
    })
}

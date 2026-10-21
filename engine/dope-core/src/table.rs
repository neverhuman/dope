//! Raw CSV tables: headers plus string cells, before any typing.

use crate::error::{DopeError, Result};
use std::io::{Read, Write};
use std::path::Path;

/// Cells recognised as missing (case-insensitive, after trimming).
const MISSING: [&str; 9] = ["", "na", "n/a", "nan", "null", "none", "?", "-", "#n/a"];

pub fn is_missing(cell: &str) -> bool {
    let t = cell.trim();
    MISSING.iter().any(|m| t.eq_ignore_ascii_case(m))
}

/// Parse a numeric cell (allows surrounding whitespace and thousands commas).
pub fn parse_number(cell: &str) -> Option<f64> {
    let t = cell.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<f64>()
        .ok()
        .or_else(|| t.replace(',', "").parse::<f64>().ok())
        .filter(|v| v.is_finite())
}

#[derive(Clone, Debug)]
pub struct RawTable {
    pub headers: Vec<String>,
    /// Column-major string cells, trimmed.
    pub columns: Vec<Vec<String>>,
}

impl RawTable {
    pub fn rows(&self) -> usize {
        self.columns.first().map_or(0, Vec::len)
    }

    pub fn width(&self) -> usize {
        self.headers.len()
    }

    pub fn column_index(&self, name: &str) -> Result<usize> {
        if let Some(i) = self.headers.iter().position(|h| h == name) {
            return Ok(i);
        }
        if let Ok(i) = name.parse::<usize>() {
            if i < self.headers.len() {
                return Ok(i);
            }
        }
        Err(DopeError::Input(
            "target column not found (give its header name or zero-based index)".into(),
        ))
    }

    pub fn from_reader<R: Read>(reader: R) -> Result<Self> {
        let mut csv = csv::ReaderBuilder::new()
            .flexible(false)
            .trim(csv::Trim::All)
            .from_reader(reader);
        let headers: Vec<String> = csv.headers()?.iter().map(str::to_string).collect();
        if headers.is_empty() {
            return Err(DopeError::Input("CSV has no header row".into()));
        }
        let mut seen = std::collections::BTreeSet::new();
        for h in &headers {
            if !seen.insert(h.as_str()) {
                return Err(DopeError::Input("duplicate column header".into()));
            }
        }
        let mut columns = vec![Vec::new(); headers.len()];
        for record in csv.records() {
            let record = record?;
            for (c, cell) in record.iter().enumerate() {
                columns[c].push(cell.to_string());
            }
        }
        if columns[0].is_empty() {
            return Err(DopeError::Input("CSV has no data rows".into()));
        }
        Ok(Self { headers, columns })
    }

    pub fn read(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        Self::from_reader(std::io::BufReader::new(file))
    }

    /// Keep only the listed rows (in order).
    pub fn select_rows(&self, rows: &[usize]) -> Self {
        Self {
            headers: self.headers.clone(),
            columns: self
                .columns
                .iter()
                .map(|c| rows.iter().map(|&r| c[r].clone()).collect())
                .collect(),
        }
    }

    pub fn write<W: Write>(&self, writer: W) -> Result<()> {
        write_csv(writer, &self.headers, self.rows(), |r, c| {
            self.columns[c][r].clone()
        })
    }
}

/// Write a CSV from a cell callback `(row, column) -> String`.
pub fn write_csv<W: Write, F: Fn(usize, usize) -> String>(
    writer: W,
    headers: &[String],
    rows: usize,
    cell: F,
) -> Result<()> {
    let mut csv = csv::Writer::from_writer(writer);
    csv.write_record(headers)?;
    let mut record = Vec::with_capacity(headers.len());
    for r in 0..rows {
        record.clear();
        for c in 0..headers.len() {
            record.push(cell(r, c));
        }
        csv.write_record(&record)?;
    }
    csv.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_and_flags_missing() {
        let t = RawTable::from_reader("a,b\n1, x\nNA,y\n".as_bytes()).unwrap();
        assert_eq!(t.rows(), 2);
        assert_eq!(t.columns[1][0], "x");
        assert!(is_missing(&t.columns[0][1]));
        assert_eq!(parse_number("1,234.5"), Some(1234.5));
    }
}

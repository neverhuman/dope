//! Append-only JSONL Arena ledger with SHA-256 verification and resumable cell tracking.

use crate::{ArenaError, ArenaResult};
use dope_core::dvs::v3::FidelityCellRequest;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

pub fn append_cell(path: &Path, cell: &FidelityCellRequest) -> ArenaResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut f, cell)?;
    f.write_all(b"\n")?;
    f.flush()?;
    Ok(())
}

pub fn read_ledger(path: &Path) -> ArenaResult<(Vec<FidelityCellRequest>, String)> {
    let raw = std::fs::read(path)?;
    let sha = format!("{:x}", Sha256::digest(&raw));
    let reader = BufReader::new(raw.as_slice());
    let mut cells = Vec::new();
    let mut seen = BTreeSet::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let cell: FidelityCellRequest = serde_json::from_str(trimmed)?;
        if !seen.insert(cell.cell_id.clone()) {
            return Err(ArenaError::Contract(format!(
                "duplicate cell_id in Arena ledger: {}",
                cell.cell_id
            )));
        }
        cells.push(cell);
    }
    Ok((cells, sha))
}

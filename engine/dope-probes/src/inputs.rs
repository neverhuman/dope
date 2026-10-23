//! Pinned probe inputs (`bench/real-world/probes/inputs.json`) and the deterministic split.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::nprng::permutation;
use crate::{Error, Result};

#[derive(Debug, Deserialize)]
pub struct SplitParams {
    pub seed: u32,
    pub test_share: f64,
    pub max_test_rows: usize,
    pub max_train_rows: usize,
    pub target: String,
}

#[derive(Debug, Deserialize)]
pub struct TableDigests {
    /// SHA-256 of the gzip file as downloaded from the pinned PMLB commit.
    pub tsv_gz: String,
    /// SHA-256 of the decompressed TSV the split reads.
    pub tsv: String,
    /// SHA-256 of the split files this tool writes.
    pub train_csv: String,
    pub test_csv: String,
}

#[derive(Debug, Deserialize)]
pub struct Inputs {
    pub pmlb_base: String,
    pub split: SplitParams,
    pub classification: Vec<String>,
    pub tables: BTreeMap<String, TableDigests>,
}

impl Inputs {
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        serde_json::from_str(&text).map_err(|e| Error::json(path, e))
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Row indices of the test and training files: shuffle with the pinned seed, the test file
/// takes the first `min(share * n, max_test)` rows, training the next `max_train` rows.
pub fn split_indices(n: usize, p: &SplitParams) -> (Vec<usize>, Vec<usize>) {
    let idx = permutation(p.seed, n);
    let n_test = ((p.test_share * n as f64) as usize).min(p.max_test_rows);
    let test = idx[..n_test].to_vec();
    let train = idx[n_test..]
        .iter()
        .copied()
        .take(p.max_train_rows)
        .collect();
    (test, train)
}

/// Split one tab-separated table into `train.csv` and `test.csv` under `out`. Fields are
/// carried through verbatim; the returned pair holds the two files' SHA-256 digests.
pub fn split_table(tsv: &[u8], p: &SplitParams, out: &Path) -> Result<(String, String)> {
    let text =
        std::str::from_utf8(tsv).map_err(|e| Error::msg(format!("TSV is not UTF-8: {e}")))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| Error::msg("empty TSV"))?
        .split('\t')
        .collect();
    if !header.contains(&p.target.as_str()) {
        return Err(Error::msg("declared target column is missing"));
    }
    let rows: Vec<Vec<&str>> = lines.map(|l| l.split('\t').collect()).collect();
    if let Some((i, r)) = rows
        .iter()
        .enumerate()
        .find(|(_, r)| r.len() != header.len())
    {
        return Err(Error::msg(format!(
            "row {} has {} fields, header has {}",
            i + 1,
            r.len(),
            header.len()
        )));
    }
    let (test, train) = split_indices(rows.len(), p);
    fs::create_dir_all(out).map_err(|e| Error::io(out, e))?;
    let write = |name: &str, idx: &[usize]| -> Result<String> {
        let mut w = csv::WriterBuilder::new().from_writer(Vec::new());
        w.write_record(&header).map_err(Error::csv)?;
        for &i in idx {
            w.write_record(&rows[i]).map_err(Error::csv)?;
        }
        let bytes = w.into_inner().map_err(|e| Error::msg(e.to_string()))?;
        let path = out.join(name);
        let mut f = fs::File::create(&path).map_err(|e| Error::io(&path, e))?;
        f.write_all(&bytes).map_err(|e| Error::io(&path, e))?;
        Ok(sha256_hex(&bytes))
    };
    Ok((write("train.csv", &train)?, write("test.csv", &test)?))
}

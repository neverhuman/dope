//! The levels-probe driver: fits, evaluations and pooled samples through a `dope` binary, and
//! the byte columns recomputed from saved artifacts.

use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::Command;

use super::{Level, LevelsRow, PerLevel, Pooled, SingleAtSum, LEVELS};
use crate::{Error, Result};

const BUDGETS: [&str; 4] = ["2KiB", "4KiB", "16KiB", "64KiB"];

/// Byte columns recomputed from saved artifacts `<dir>/<table>/<level>.dope`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteColumns {
    pub bytes: [u64; 4],
    pub inflated: [u64; 4],
    pub metadata: [u64; 4],
    pub brotli_alone: [u64; 4],
    pub conditional: [i64; 4],
}

impl ByteColumns {
    /// `1 − (marginal low+med+high) / (standalone low+med+high)`.
    pub fn chain_saving(&self) -> f64 {
        let chain: i64 = self.conditional[..3].iter().sum();
        let alone: u64 = self.brotli_alone[..3].iter().sum();
        1.0 - chain as f64 / alone as f64
    }

    pub fn from_row(r: &LevelsRow) -> Self {
        ByteColumns {
            bytes: LEVELS.map(|l| r.levels.get(l).bytes),
            inflated: LEVELS.map(|l| r.levels.get(l).inflated),
            metadata: LEVELS.map(|l| r.levels.get(l).metadata),
            brotli_alone: LEVELS.map(|l| r.levels.get(l).brotli_alone),
            conditional: LEVELS.map(|l| *r.conditional_bytes.get(l)),
        }
    }
}

/// Program bytes, inflated size, metadata length, standalone brotli size and the marginal
/// chain sizes of one table's four level artifacts.
pub fn byte_columns(dir: &Path) -> Result<ByteColumns> {
    let mut c = ByteColumns {
        bytes: [0; 4],
        inflated: [0; 4],
        metadata: [0; 4],
        brotli_alone: [0; 4],
        conditional: [0; 4],
    };
    let mut chain: Vec<u8> = Vec::new();
    for (k, l) in LEVELS.iter().enumerate() {
        let m = dir.join(format!("{l}.dope"));
        let p = inflate(&m)?;
        c.bytes[k] = fs::metadata(&m).map_err(|e| Error::io(&m, e))?.len();
        c.inflated[k] = p.len() as u64;
        c.metadata[k] = u64::from(program_metadata_len(&p)?);
        c.brotli_alone[k] = brotli_len(&p);
        let before = if chain.is_empty() {
            0
        } else {
            brotli_len(&chain) as i64
        };
        chain.extend_from_slice(&p);
        c.conditional[k] = brotli_len(&chain) as i64 - before;
    }
    Ok(c)
}

fn program_metadata_len(p: &[u8]) -> Result<u32> {
    let head: [u8; 4] = p
        .get(..4)
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| Error::msg("program shorter than its length prefix"))?;
    Ok(u32::from_le_bytes(head))
}

fn run(dope: &Path, args: &[&dyn AsRef<std::ffi::OsStr>]) -> Result<String> {
    let out = Command::new(dope)
        .args(args.iter().map(|a| a.as_ref()))
        .output()
        .map_err(|e| Error::io(dope, e))?;
    if !out.status.success() {
        return Err(Error::engine(format!(
            "{} failed with status {} (private stderr omitted)",
            dope.display(),
            out.status
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let text = fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
    serde_json::from_str(&text).map_err(|e| Error::json(path, e))
}

/// The artifact's inflated program: the `dope` tensor of the safetensors file, brotli-decoded.
pub fn inflate(path: &Path) -> Result<Vec<u8>> {
    let bytes = fs::read(path).map_err(|e| Error::io(path, e))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|_| Error::msg(format!("{}: invalid safetensors artifact", path.display())))?;
    let t = st
        .tensor("dope")
        .map_err(|_| Error::msg(format!("{}: required tensor is missing", path.display())))?;
    let mut out = Vec::new();
    brotli::Decompressor::new(t.data(), 4096)
        .read_to_end(&mut out)
        .map_err(|e| Error::io(path, e))?;
    Ok(out)
}

/// Brotli at quality 11 and a 22-bit window (the reference encoder's defaults).
pub fn brotli_len(bytes: &[u8]) -> u64 {
    let mut out = Vec::new();
    let params = brotli::enc::BrotliEncoderParams {
        quality: 11,
        lgwin: 22,
        ..Default::default()
    };
    let mut input = bytes;
    // Writing into a Vec cannot fail.
    let _ = brotli::BrotliCompress(&mut input, &mut out, &params);
    out.len() as u64
}

fn evaluate(
    dope: &Path,
    model: &Path,
    train: &Path,
    test: &Path,
    synth: Option<&Path>,
) -> Result<(f64, serde_json::Map<String, serde_json::Value>)> {
    let json = match synth {
        None => model.with_extension("eval.json"),
        Some(s) => s.with_extension("csv.eval.json"),
    };
    let mut args: Vec<&dyn AsRef<std::ffi::OsStr>> = vec![
        &"eval",
        &"-m",
        &model,
        &"--train",
        &train,
        &"--test",
        &test,
        &"--factor",
        &"1",
        &"--json",
        &json,
    ];
    if let Some(s) = synth.as_ref() {
        args.push(&"--synth");
        args.push(s);
    }
    run(dope, &args)?;
    let r = read_json(&json)?;
    let robust = r["robust_retention"]
        .as_f64()
        .ok_or_else(|| Error::engine("eval report lacks robust_retention"))?;
    let mut per = serde_json::Map::new();
    for a in r["auditors"].as_array().into_iter().flatten() {
        if a["informative"].as_bool() == Some(true) {
            if let Some(name) = a["auditor"].as_str() {
                per.insert(name.to_string(), a["retention"].clone());
            }
        }
    }
    Ok((robust, per))
}

/// Run the levels protocol on one table directory (`train.csv`, `test.csv`, target `target`).
pub fn probe_table(dope: &Path, tdir: &Path, out: &Path) -> Result<LevelsRow> {
    let table = tdir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let (train, test) = (tdir.join("train.csv"), tdir.join("test.csv"));
    let w = out.join(&table);
    fs::create_dir_all(&w).map_err(|e| Error::io(&w, e))?;
    let mut levels = Vec::new();
    for (lvl, budget) in LEVELS.iter().zip(BUDGETS) {
        let m = w.join(format!("{lvl}.dope"));
        let report = w.join(format!("{lvl}.fit.json"));
        run(
            dope,
            &[
                &"fit",
                &train,
                &"-t",
                &"target",
                &"--budget",
                &budget,
                &"-o",
                &m,
                &"--report",
                &report,
                &"-q",
            ],
        )?;
        let rep = read_json(&report)?;
        let (robust, per) = evaluate(dope, &m, &train, &test, None)?;
        levels.push((rep, robust, per));
    }
    let bc = byte_columns(&w)?;
    let mut built = levels
        .into_iter()
        .enumerate()
        .map(|(k, (rep, robust, per))| Level {
            bytes: bc.bytes[k],
            inflated: bc.inflated[k],
            metadata: bc.metadata[k],
            brotli_alone: bc.brotli_alone[k],
            trees: rep["trees"].clone(),
            dependence: rep["dependence"].clone(),
            regimes: rep["regimes"].clone(),
            plan: serde_json::json!([rep["knots"], rep["inputs"], rep["hidden"]]),
            self_audit: rep["self_audit"].as_f64(),
            robust,
            per_auditor: per,
        });
    let (low, med, high, ultra) = match (built.next(), built.next(), built.next(), built.next()) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => return Err(Error::msg("missing level")),
    };
    let cond = bc.conditional;
    let independent_sum_lmh = (low.brotli_alone + med.brotli_alone + high.brotli_alone) as i64;
    let total = low.bytes + med.bytes + high.bytes;
    let m = w.join("sum.dope");
    let report = w.join("sum.fit.json");
    let total_s = total.to_string();
    run(
        dope,
        &[
            &"fit",
            &train,
            &"-t",
            &"target",
            &"--budget",
            &total_s,
            &"-o",
            &m,
            &"--report",
            &report,
            &"-q",
        ],
    )?;
    let single = SingleAtSum {
        budget: total,
        bytes: fs::metadata(&m).map_err(|e| Error::io(&m, e))?.len(),
        robust: evaluate(dope, &m, &train, &test, None)?.0,
    };
    // Pool samples from the three independent programs: equal thirds, as many rows as train.
    let n = fs::read_to_string(&train)
        .map_err(|e| Error::io(&train, e))?
        .lines()
        .count()
        .saturating_sub(1);
    let mut pooled = String::new();
    for (i, lvl) in ["low", "med", "high"].iter().enumerate() {
        let f = w.join(format!("{lvl}.syn.csv"));
        let rows = (n / 3 + usize::from(i < n % 3)).to_string();
        let seed = (7 + i).to_string();
        let model = w.join(format!("{lvl}.dope"));
        run(
            dope,
            &[&"sample", &model, &"-n", &rows, &"--seed", &seed, &"-o", &f],
        )?;
        let text = fs::read_to_string(&f).map_err(|e| Error::io(&f, e))?;
        let mut lines = text.lines();
        let header = lines.next().unwrap_or_default();
        if i == 0 {
            pooled += header;
            pooled.push('\n');
        }
        for l in lines.filter(|l| !l.is_empty()) {
            pooled += l;
            pooled.push('\n');
        }
    }
    let pool = w.join("pool.csv");
    fs::write(&pool, pooled).map_err(|e| Error::io(&pool, e))?;
    let pooled_robust = evaluate(dope, &w.join("high.dope"), &train, &test, Some(&pool))?.0;
    Ok(LevelsRow {
        table,
        levels: PerLevel {
            low,
            med,
            high,
            ultra,
        },
        conditional_bytes: PerLevel {
            low: cond[0],
            med: cond[1],
            high: cond[2],
            ultra: cond[3],
        },
        independent_sum_lmh,
        conditional_sum_lmh: cond[0] + cond[1] + cond[2],
        single_at_sum: single,
        pooled_lmh: Pooled {
            robust: pooled_robust,
        },
    })
}

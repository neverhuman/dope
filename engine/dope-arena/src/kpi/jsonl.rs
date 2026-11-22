//! Canonical JSON (sorted keys, no whitespace), JSONL files and staged writes.
//!
//! Records are rendered through `serde_json::Value` and re-emitted with keys
//! in byte order, so the bytes do not depend on struct field order or on any
//! `serde_json` feature flag. Parse errors name the file and line only; they
//! never repeat record content.

use dope_arena::{ArenaError, ArenaResult};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// A failed file operation, reported with the path and the error kind.
pub fn file_error(path: &Path, err: &std::io::Error) -> ArenaError {
    ArenaError::Contract(format!("{}: {}", path.display(), err.kind()))
}

pub fn read_file(path: &Path) -> ArenaResult<Vec<u8>> {
    std::fs::read(path).map_err(|e| file_error(path, &e))
}

/// Streaming SHA-256 of a file (constant memory).
pub fn file_sha256(path: &Path) -> ArenaResult<String> {
    let mut file = File::open(path).map_err(|e| file_error(path, &e))?;
    let mut hasher = Sha256::new();
    let mut block = vec![0u8; 1 << 20];
    loop {
        let got = file.read(&mut block).map_err(|e| file_error(path, &e))?;
        if got == 0 {
            break;
        }
        hasher.update(&block[..got]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn write_value(out: &mut String, value: &Value) -> ArenaResult<()> {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            out.push('{');
            for (position, (key, item)) in entries.into_iter().enumerate() {
                if position > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key)?);
                out.push(':');
                write_value(out, item)?;
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (position, item) in items.iter().enumerate() {
                if position > 0 {
                    out.push(',');
                }
                write_value(out, item)?;
            }
            out.push(']');
        }
        scalar => out.push_str(&serde_json::to_string(scalar)?),
    }
    Ok(())
}

/// One-line JSON with object keys sorted at every depth.
pub fn canonical_json<T: Serialize>(record: &T) -> ArenaResult<String> {
    let value = serde_json::to_value(record)?;
    let mut out = String::new();
    write_value(&mut out, &value)?;
    Ok(out)
}

/// Indented JSON with sorted keys and a trailing newline, for single-document artifacts.
pub fn canonical_pretty<T: Serialize>(record: &T) -> ArenaResult<String> {
    // Re-parsing the canonical text keeps the sorted order whichever map type
    // `serde_json::Value` is built with (ordered map or insertion order).
    let sorted: Value = serde_json::from_str(&canonical_json(record)?)?;
    let mut text = serde_json::to_string_pretty(&sorted)?;
    text.push('\n');
    Ok(text)
}

/// Write through a sibling staging file and rename, so readers never see a partial file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> ArenaResult<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| file_error(parent, &e))?;
        }
    }
    let mut staging_name = path.file_name().unwrap_or_default().to_os_string();
    staging_name.push(".staging");
    let staging: PathBuf = path.with_file_name(staging_name);
    let mut file = File::create(&staging).map_err(|e| file_error(&staging, &e))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| file_error(&staging, &e))?;
    drop(file);
    std::fs::rename(&staging, path).map_err(|e| file_error(path, &e))
}

/// Canonical JSONL bytes: one sorted-key object per line, `\n` terminated.
pub fn jsonl_bytes<T: Serialize>(records: &[T]) -> ArenaResult<Vec<u8>> {
    let mut out = String::new();
    for record in records {
        out.push_str(&canonical_json(record)?);
        out.push('\n');
    }
    Ok(out.into_bytes())
}

/// Write canonical JSONL and return the SHA-256 of the bytes written.
pub fn write_jsonl<T: Serialize>(path: &Path, records: &[T]) -> ArenaResult<String> {
    let bytes = jsonl_bytes(records)?;
    write_atomic(path, &bytes)?;
    Ok(sha256_hex(&bytes))
}

/// Parse JSONL bytes; blank lines are skipped and errors cite `label` and the line number.
pub fn parse_jsonl<T: DeserializeOwned>(label: &str, bytes: &[u8]) -> ArenaResult<Vec<T>> {
    let mut records = Vec::new();
    for (number, line) in bytes.split(|&b| b == b'\n').enumerate() {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let record = serde_json::from_slice(line).map_err(|e| {
            ArenaError::Contract(format!(
                "{label}: line {}: malformed record ({:?})",
                number + 1,
                e.classify()
            ))
        })?;
        records.push(record);
    }
    Ok(records)
}

pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> ArenaResult<Vec<T>> {
    parse_jsonl(&path.display().to_string(), &read_file(path)?)
}

/// Parse a single JSON document without echoing its content on failure.
pub fn parse_json<T: DeserializeOwned>(label: &str, bytes: &[u8]) -> ArenaResult<T> {
    serde_json::from_slice(bytes)
        .map_err(|e| ArenaError::Contract(format!("{label}: malformed JSON ({:?})", e.classify())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::collections::BTreeMap;

    #[derive(Debug, Deserialize, PartialEq, Serialize)]
    struct Record {
        zeta: u64,
        alpha: String,
        mid: Vec<i32>,
        nested: BTreeMap<String, u8>,
    }

    fn sample(zeta: u64) -> Record {
        Record {
            zeta,
            alpha: "a\"b".into(),
            mid: vec![3, -1],
            nested: BTreeMap::from([("y".to_string(), 2), ("x".to_string(), 1)]),
        }
    }

    #[test]
    fn canonical_json_sorts_keys_at_every_depth() {
        assert_eq!(
            canonical_json(&sample(7)).unwrap(),
            r#"{"alpha":"a\"b","mid":[3,-1],"nested":{"x":1,"y":2},"zeta":7}"#
        );
    }

    #[test]
    fn pretty_output_keeps_sorted_keys_and_ends_with_a_newline() {
        let text = canonical_pretty(&sample(7)).unwrap();
        assert!(text.ends_with("}\n"));
        let at = |key: &str| text.find(key).unwrap();
        assert!(at("\"alpha\"") < at("\"mid\"") && at("\"mid\"") < at("\"nested\""));
        assert!(at("\"x\"") < at("\"y\"") && at("\"nested\"") < at("\"zeta\""));
        let back: Record = parse_json("pretty", text.as_bytes()).unwrap();
        assert_eq!(back, sample(7));
    }

    #[test]
    fn jsonl_round_trips_and_reports_its_digest() {
        let dir = crate::kpi::unit_test_dir("jsonl-round-trip");
        let path = dir.join("nested").join("records.jsonl");
        let records = vec![sample(1), sample(u64::MAX)];
        let digest = write_jsonl(&path, &records).unwrap();
        assert_eq!(digest, file_sha256(&path).unwrap());
        assert_eq!(digest, sha256_hex(&std::fs::read(&path).unwrap()));
        assert_eq!(read_jsonl::<Record>(&path).unwrap(), records);
        assert!(!path.with_file_name("records.jsonl.staging").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn parse_errors_cite_the_line_and_never_the_content() {
        let bytes = b"{\"zeta\":1,\"alpha\":\"x\",\"mid\":[],\"nested\":{}}\n\n{\"zeta\":\"SECRETVALUE\"}\n";
        let err = parse_jsonl::<Record>("records.jsonl", bytes)
            .unwrap_err()
            .to_string();
        assert!(err.contains("records.jsonl: line 3"), "{err}");
        assert!(!err.contains("SECRETVALUE"), "{err}");
        let err = parse_json::<Record>("META.json", b"{\"zeta\": SECRETVALUE")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("META.json") && !err.contains("SECRETVALUE"),
            "{err}"
        );
    }

    #[test]
    fn missing_file_reports_path_and_kind() {
        let err = read_jsonl::<Record>(Path::new("/nonexistent/kpi/records.jsonl"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("records.jsonl") && err.contains("not found"),
            "{err}"
        );
    }
}

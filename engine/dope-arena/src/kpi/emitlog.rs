//! Emission log `emit.jsonl` (formats.md section 6): one row per attempted file.
//!
//! An emitter appends its rows to `<emit-dir>/emit.jsonl`; a native control
//! arm writes `<emit-dir>/<arm_id>/emit.jsonl` instead, so it never rewrites
//! a log that another writer may be appending to. The scorer reads both
//! places through this one loader and keeps the rows of a single arm.

use super::jsonl::{read_jsonl, write_jsonl};
use dope_arena::{ArenaError, ArenaResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const EMIT_LOG: &str = "emit.jsonl";
/// `artifact_class` values written by this binary (an emitter may also log `rows_at_sampler`).
pub const CLASS_CONTROL: &str = "control";
pub const CLASS_SOURCE_FREE: &str = "source_free";
pub const CLASS_ROWS_AT_SAMPLER: &str = "rows_at_sampler";
/// Log rows a worker process hands to its parent for a native arm.
pub const EMIT_PART: &str = "emit.part.jsonl";
pub const EMIT_OK: &str = "ok";
pub const EMIT_FAILED: &str = "failed";
pub const EMIT_OVER_BUDGET: &str = "over_budget";
pub const EMIT_UNSUPPORTED: &str = "unsupported";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EmitRow {
    pub cell_id: String,
    pub sample_seed: u64,
    pub multiplier: u32,
    pub arm_id: String,
    #[serde(default)]
    pub arch_id: String,
    #[serde(default)]
    pub weights_sha256: String,
    pub artifact_class: String,
    #[serde(default)]
    pub packet_bytes: u64,
    #[serde(default)]
    pub packet_sha256: String,
    #[serde(default)]
    pub cold_bytes: u64,
    #[serde(default)]
    pub rows: u64,
    #[serde(default)]
    pub file: String,
    pub status: String,
    #[serde(default)]
    pub reason: String,
    /// Wall-clock measurements of the emitter; never copied into a scored output.
    #[serde(default)]
    pub encode_ms: f64,
    #[serde(default)]
    pub decode_ms: f64,
    /// Smallest fit-row support behind any discrete level the generator can
    /// emit, when the emitter declares it (not part of formats.md section 6).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_emitted_support: Option<u64>,
}

/// Whether `class` is one of the three artifact classes of formats.md section 6.
pub fn known_class(class: &str) -> bool {
    [CLASS_SOURCE_FREE, CLASS_ROWS_AT_SAMPLER, CLASS_CONTROL].contains(&class)
}

/// Key of one attempted file within an arm.
pub type EmitKey = (String, u64, u32);

/// An arm id is one path component of plain characters.
pub fn check_arm_id(arm_id: &str) -> ArenaResult<()> {
    let plain = |b: u8| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.');
    let ok = !arm_id.is_empty()
        && arm_id.len() <= 96
        && arm_id.bytes().all(plain)
        && !arm_id.starts_with('.');
    if ok {
        Ok(())
    } else {
        Err(ArenaError::Contract(
            "--arm must be 1-96 characters of [A-Za-z0-9_.-] and not start with a dot".into(),
        ))
    }
}

/// A reason code safe to copy into a ledger. Emitters log fixed codes; any
/// text that is not a plain code (it could quote data) is replaced as a whole.
pub fn sanitize_reason(reason: &str) -> String {
    let plain = |c: char| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | ':' | '.');
    let is_code = !reason.is_empty()
        && reason.len() <= 80
        && reason.starts_with(|c: char| c.is_ascii_alphabetic())
        && reason.chars().all(plain);
    if is_code {
        reason.to_string()
    } else {
        "nonconforming_reason".to_string()
    }
}

pub fn arm_log_path(emit_dir: &Path, arm_id: &str) -> PathBuf {
    emit_dir.join(arm_id).join(EMIT_LOG)
}

/// Write the rows of a native arm to its own log, ordered by (cell, sample
/// seed, multiplier) whatever order they were produced in.
pub fn write_arm_log(emit_dir: &Path, arm_id: &str, rows: &mut [EmitRow]) -> ArenaResult<String> {
    rows.sort_by(|a, b| {
        (&a.cell_id, a.sample_seed, a.multiplier).cmp(&(&b.cell_id, b.sample_seed, b.multiplier))
    });
    write_jsonl(&arm_log_path(emit_dir, arm_id), rows)
}

/// Rows keyed for the scorer; the same file twice is a contract error.
pub fn index(rows: Vec<EmitRow>) -> ArenaResult<BTreeMap<EmitKey, EmitRow>> {
    let mut keyed = BTreeMap::new();
    for row in rows {
        let key = (row.cell_id.clone(), row.sample_seed, row.multiplier);
        if keyed.insert(key, row).is_some() {
            return Err(ArenaError::Contract(
                "an attempted file is logged twice for this arm".into(),
            ));
        }
    }
    Ok(keyed)
}

/// Rows of `arm_id` from the shared log and from the arm's own log.
/// The same (cell, sample seed, multiplier) twice is a contract error.
pub fn load_arm(emit_dir: &Path, arm_id: &str) -> ArenaResult<BTreeMap<EmitKey, EmitRow>> {
    let mut rows = Vec::new();
    for path in [emit_dir.join(EMIT_LOG), arm_log_path(emit_dir, arm_id)] {
        if path.is_file() {
            rows.extend(read_jsonl::<EmitRow>(&path)?);
        }
    }
    rows.retain(|row| row.arm_id == arm_id);
    index(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(arm: &str, cell: &str, seed: u64, multiplier: u32) -> EmitRow {
        EmitRow {
            cell_id: cell.into(),
            sample_seed: seed,
            multiplier,
            arm_id: arm.into(),
            arch_id: String::new(),
            weights_sha256: String::new(),
            artifact_class: CLASS_CONTROL.into(),
            packet_bytes: 0,
            packet_sha256: String::new(),
            cold_bytes: 0,
            rows: 12,
            file: format!("{arm}/x.mat"),
            status: EMIT_OK.into(),
            reason: String::new(),
            encode_ms: 0.0,
            decode_ms: 0.0,
            min_emitted_support: None,
        }
    }

    #[test]
    fn loader_merges_both_logs_and_keeps_one_arm() {
        let dir = crate::kpi::unit_test_dir("emitlog-merge");
        // The shared log as the Python emitter writes it: extra precision, two arms.
        let shared = concat!(
            "{\"arch_id\":\"single_enc_v6\",\"arm_id\":\"model\",\"artifact_class\":\"source_free\",",
            "\"cell_id\":\"p:h:full:d0\",\"cold_bytes\":0,\"decode_ms\":31.662,\"encode_ms\":208.4,",
            "\"file\":\"model/p__h__full__d0.s0.m4.mat\",\"multiplier\":4,\"packet_bytes\":1624,",
            "\"packet_sha256\":\"ab\",\"reason\":\"\",\"rows\":4096,\"sample_seed\":0,\"status\":\"ok\",",
            "\"weights_sha256\":\"cd\"}\n",
            "{\"arm_id\":\"other\",\"artifact_class\":\"control\",\"cell_id\":\"p:h:full:d0\",",
            "\"multiplier\":4,\"sample_seed\":0,\"status\":\"failed\",\"reason\":\"x\"}\n",
        );
        std::fs::write(dir.join(EMIT_LOG), shared).unwrap();
        let mut own = vec![
            row("model", "p:h:full:d0", 1, 4),
            row("model", "p:h:full:d0", 0, 1),
        ];
        write_arm_log(&dir, "model", &mut own).unwrap();
        assert_eq!(own[0].sample_seed, 0, "the log is written in key order");

        let loaded = load_arm(&dir, "model").unwrap();
        assert_eq!(loaded.len(), 3);
        let first = &loaded[&("p:h:full:d0".to_string(), 0, 4)];
        assert_eq!((first.packet_bytes, first.rows), (1624, 4096));
        assert_eq!(first.artifact_class, "source_free");
        assert_eq!(first.min_emitted_support, None);
        assert_eq!(loaded[&("p:h:full:d0".to_string(), 1, 4)], own[1]);
        assert_eq!(load_arm(&dir, "other").unwrap().len(), 1);
        assert!(load_arm(&dir, "absent").unwrap().is_empty());

        // The same file logged in both places is refused.
        write_arm_log(&dir, "model", &mut [row("model", "p:h:full:d0", 0, 4)]).unwrap();
        let err = load_arm(&dir, "model").unwrap_err().to_string();
        assert!(err.contains("logged twice"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn control_rows_serialise_without_optional_fields() {
        let line = crate::kpi::jsonl::canonical_json(&row("copy_a", "c", 0, 4)).unwrap();
        assert!(!line.contains("min_emitted_support"));
        assert!(line.starts_with("{\"arch_id\":\"\",\"arm_id\":\"copy_a\""));
        let mut declared = row("copy_a", "c", 0, 4);
        declared.min_emitted_support = Some(12);
        let line = crate::kpi::jsonl::canonical_json(&declared).unwrap();
        assert_eq!(serde_json::from_str::<EmitRow>(&line).unwrap(), declared);
    }

    #[test]
    fn arm_ids_and_reasons_are_restricted_to_plain_characters() {
        for good in ["twin_c", "v6-step.5000", "A"] {
            check_arm_id(good).unwrap();
        }
        for bad in ["", "a/b", "..", ".hidden", "a b", "a:b", &"x".repeat(97)] {
            assert!(check_arm_id(bad).is_err(), "{bad:?}");
        }
        assert!(known_class("source_free") && known_class("rows_at_sampler"));
        assert!(known_class("control") && !known_class("Control") && !known_class(""));
        assert_eq!(
            sanitize_reason("encode_exception:ValueError"),
            "encode_exception:ValueError"
        );
        for text in [
            "bad value 3.75 in 'age' (row 9)",
            "",
            "3.75",
            &"z".repeat(81),
        ] {
            assert_eq!(sanitize_reason(text), "nonconforming_reason", "{text:?}");
        }
    }
}

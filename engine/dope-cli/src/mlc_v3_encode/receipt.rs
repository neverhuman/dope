//! Opt-in snapshot companion; the primary encode event contract is independent.

use dope_core::landmark::{LandmarkOptions, LandmarkStatus};
use dope_core::prepare::DopeConditioning;
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

#[derive(Serialize)]
struct EmbeddingIdentity {
    model: &'static str,
    layout: &'static str,
    encoder_level: &'static str,
    dimensions: usize,
    dtype: &'static str,
}

/// A successful encoding snapshot, not a whole-run completion certificate.
#[derive(Serialize)]
struct LandmarkEncodingReceipt<'a> {
    schema: &'static str,
    seq: usize,
    version: u64,
    status: LandmarkStatus,
    mask: [bool; 23],
    subsample_n: Option<usize>,
    source_digest: &'a str,
    landmark_source_digest: &'a str,
    conditioning_digest: &'a str,
    conditioning_recipe: &'a str,
    conditioning_level: &'a str,
    options_digest: Option<String>,
    work_budget: Option<u64>,
    weights_sha256: &'a str,
    embedding_identity: EmbeddingIdentity,
    features629_sha256: &'a str,
    device: &'a str,
}

#[derive(Default)]
struct ConditioningVersion {
    last_digest: Option<String>,
    version: u64,
}

impl ConditioningVersion {
    fn observe(&mut self, digest: &str) -> u64 {
        if self
            .last_digest
            .as_deref()
            .is_some_and(|last| last != digest)
        {
            self.version += 1;
        }
        self.last_digest = Some(digest.to_owned());
        self.version
    }
}

pub(super) struct ReceiptWriter {
    file: File,
    source_digest: String,
    weights_sha256: String,
    device: String,
    version: ConditioningVersion,
}

impl ReceiptWriter {
    pub(super) fn create(
        path: &Path,
        primary_output: Option<&Path>,
        source_digest: String,
        weights_sha256: String,
        device: String,
    ) -> Result<Self, String> {
        // Never truncate existing evidence or an input, even through a symlink.
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|_| "could not create new encoding receipt".to_string())?;
        let receipt_path = path
            .canonicalize()
            .map_err(|_| "could not resolve encoding receipt".to_string())?;
        if primary_output.and_then(|p| p.canonicalize().ok()).as_ref() == Some(&receipt_path) {
            return Err("encoding receipt must differ from primary output".into());
        }
        Ok(Self {
            file,
            source_digest,
            weights_sha256,
            device,
            version: ConditioningVersion::default(),
        })
    }

    pub(super) fn emit(
        &mut self,
        seq: usize,
        cond: &DopeConditioning,
        features_sha256: &str,
        options: Option<&LandmarkOptions>,
    ) -> Result<(), String> {
        let bundle = cond.bundle.sanitized();
        let digest = cond.digest();
        let receipt = LandmarkEncodingReceipt {
            schema: "dope.landmark-encoding/v1",
            seq,
            version: self.version.observe(&digest),
            status: bundle.status,
            mask: bundle.mask,
            subsample_n: bundle.subsample_n,
            source_digest: &self.source_digest,
            landmark_source_digest: &bundle.source_digest,
            conditioning_digest: &digest,
            conditioning_recipe: &cond.recipe,
            conditioning_level: &cond.level,
            options_digest: options.map(dope_core::landmark::bundle::options_digest),
            work_budget: options.map(|opts| opts.work_budget),
            weights_sha256: &self.weights_sha256,
            embedding_identity: EmbeddingIdentity {
                model: "mlc-v3",
                layout: "High629",
                encoder_level: "high",
                dimensions: 629,
                dtype: "f32",
            },
            features629_sha256: features_sha256,
            device: &self.device,
        };
        serde_json::to_writer(&mut self.file, &receipt)
            .and_then(|()| {
                self.file.write_all(b"\n").map_err(serde_json::Error::io)?;
                self.file.flush().map_err(serde_json::Error::io)
            })
            .map_err(|_| "could not write encoding receipt".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dope_core::landmark::{LandmarkBundle, LandmarkMode};
    use dope_core::prepare::{build_conditioning, prepare_table_cells, PrepTask};

    fn zero_conditioning() -> DopeConditioning {
        let x = vec![
            vec!["1".into()],
            vec!["2".into()],
            vec!["3".into()],
            vec!["4".into()],
        ];
        let y = vec!["0".into(), "1".into(), "2".into(), "3".into()];
        let table = prepare_table_cells(&x, &y, PrepTask::Regression).unwrap();
        build_conditioning(&table, &LandmarkBundle::zero(), None, LandmarkMode::None).unwrap()
    }

    #[test]
    fn versions_follow_effective_conditioning_including_reversions() {
        let mut version = ConditioningVersion::default();
        assert_eq!(version.observe("a"), 0);
        assert_eq!(version.observe("a"), 0);
        assert_eq!(version.observe("b"), 1);
        assert_eq!(version.observe("b"), 1);
        assert_eq!(version.observe("a"), 2);
    }

    #[test]
    fn coverage_is_sanitized_without_advancing_unchanged_conditioning() {
        let path =
            std::env::temp_dir().join(format!("dope-receipt-unit-{}.jsonl", std::process::id()));
        let mut writer =
            ReceiptWriter::create(&path, None, "a".repeat(64), "b".repeat(64), "cpu".into())
                .unwrap();
        let mut cond = zero_conditioning();
        writer.emit(0, &cond, &"c".repeat(64), None).unwrap();
        // A finite measured zero is present; a nonfinite slot must be absent.
        cond.bundle.mask[0] = true;
        cond.bundle.mask[1] = true;
        cond.bundle.values[1] = f32::NAN;
        cond.bundle.status = LandmarkStatus::Full;
        writer.emit(1, &cond, &"c".repeat(64), None).unwrap();
        drop(writer);
        let records: Vec<serde_json::Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records[0]["status"], "none");
        assert_eq!(records[1]["status"], serde_json::json!({"partial": 1}));
        assert_eq!(records[1]["mask"][0], true);
        assert_eq!(records[1]["mask"][1], false);
        assert_eq!(records[1]["version"], 0);
        assert_eq!(
            records[0]["conditioning_digest"],
            records[1]["conditioning_digest"]
        );
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn receipt_write_failure_is_not_success() {
        let mut writer = ReceiptWriter {
            file: OpenOptions::new().write(true).open("/dev/full").unwrap(),
            source_digest: "a".repeat(64),
            weights_sha256: "b".repeat(64),
            device: "cpu".into(),
            version: ConditioningVersion::default(),
        };
        assert_eq!(
            writer
                .emit(0, &zero_conditioning(), &"c".repeat(64), None)
                .unwrap_err(),
            "could not write encoding receipt"
        );
    }
}

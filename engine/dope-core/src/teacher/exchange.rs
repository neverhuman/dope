//! External teacher exchange seam (`dope teacher export` / `--teacher exchange:DIR`).
//!
//! Enables plugging unmodified external teachers (CatBoost, LightGBM, XGBoost, TabPFN-2.5,
//! TabICLv2) via a frozen file seam while recording `seen_teacher_families` so no model used
//! as a teacher can ever serve as a held-out auditor.

use crate::error::{DopeError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

pub const TEACHER_EXCHANGE_PROTOCOL: &str = "dope-teacher-exchange-v1";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherExchangeBundle {
    pub protocol: String,
    /// Dataset/split commitment matching the exact `BUILD` rows used to produce OOF predictions.
    pub build_split_sha256: String,
    pub task: String,
    pub num_outputs: usize,
    /// Model families used inside this external teacher — strictly barred from held-out auditing!
    pub seen_teacher_families: Vec<String>,
    /// Out-of-fold teacher predictions on `BUILD` rows `[n_build * num_outputs]`.
    pub oof_predictions: Vec<f64>,
    /// Optional teacher predictions on exported `Q_X` probe rows `[n_probes * num_outputs]`.
    pub probe_predictions: Option<Vec<f64>>,
    /// Optional per-row member agreement scores in `[0, 1]` for transfer weighting.
    pub probe_agreement: Option<Vec<f64>>,
}

impl TeacherExchangeBundle {
    pub fn validate(&self, expected_build_rows: usize, expected_outputs: usize) -> Result<()> {
        if self.protocol != TEACHER_EXCHANGE_PROTOCOL {
            return Err(DopeError::Input(
                "unsupported teacher exchange protocol".into(),
            ));
        }
        if self.seen_teacher_families.is_empty() {
            return Err(DopeError::Input(
                "external teacher exchange must declare non-empty seen_teacher_families".into(),
            ));
        }
        if self.num_outputs != expected_outputs {
            return Err(DopeError::Input(
                "teacher exchange num_outputs does not match task".into(),
            ));
        }
        if self.oof_predictions.len() != expected_build_rows * expected_outputs {
            return Err(DopeError::Input(
                "teacher exchange oof_predictions length mismatch".into(),
            ));
        }
        if !self.oof_predictions.iter().all(|v| v.is_finite()) {
            return Err(DopeError::Input(
                "teacher exchange contains non-finite OOF predictions".into(),
            ));
        }
        if let Some(ref probes) = self.probe_predictions {
            if probes.len() % expected_outputs != 0 || !probes.iter().all(|v| v.is_finite()) {
                return Err(DopeError::Input(
                    "teacher exchange probe_predictions invalid".into(),
                ));
            }
            if let Some(ref agr) = self.probe_agreement {
                let np = probes.len() / expected_outputs;
                if agr.len() != np || !agr.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                {
                    return Err(DopeError::Input(
                        "teacher exchange probe_agreement invalid".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Strict independence check: refuse any auditor whose family was seen as a teacher!
    pub fn assert_auditor_unseen(&self, auditor_family: &str) -> Result<()> {
        let needle = auditor_family.trim().to_ascii_lowercase();
        if self
            .seen_teacher_families
            .iter()
            .any(|s| s.trim().to_ascii_lowercase() == needle)
        {
            return Err(DopeError::Input(
                "auditor family was seen inside the teacher and cannot serve as held-out auditor"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn save_dir(&self, dir: &Path) -> Result<String> {
        std::fs::create_dir_all(dir)?;
        let raw = serde_json::to_vec_pretty(self)
            .map_err(|_| DopeError::Input("failed to serialize teacher exchange".into()))?;
        let sha = format!("{:x}", Sha256::digest(&raw));
        std::fs::write(dir.join("teacher_exchange.json"), &raw)?;
        Ok(sha)
    }

    pub fn load_dir(dir: &Path) -> Result<Self> {
        let raw = std::fs::read(dir.join("teacher_exchange.json"))?;
        serde_json::from_slice(&raw)
            .map_err(|_| DopeError::Input("invalid teacher_exchange.json".into()))
    }
}

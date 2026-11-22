//! Panel manifest and cryptographic commitment verification for dev and sealed panels.

use crate::{ArenaError, ArenaResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PanelRole {
    Development,
    Quick24,
    Pinned16,
    SealedConfirmation,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PanelDatasetEntry {
    pub dataset_id: String,
    pub family_id: String,
    pub target_column: String,
    pub task: String,
    pub group_column: Option<String>,
    pub time_column: Option<String>,
    pub source_sha256: String,
    pub rows: usize,
    pub features: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PanelManifest {
    pub protocol: String,
    pub panel_id: String,
    pub role: PanelRole,
    pub datasets: Vec<PanelDatasetEntry>,
    pub manifest_sha256: Option<String>,
}

impl PanelManifest {
    pub fn load(path: &Path) -> ArenaResult<Self> {
        let bytes = std::fs::read(path)?;
        let manifest: Self = serde_json::from_slice(&bytes)?;
        manifest.verify()?;
        Ok(manifest)
    }

    /// Canonical SHA-256 over the sorted dataset inventory (independent of `manifest_sha256`).
    pub fn canonical_sha256(&self) -> ArenaResult<String> {
        let mut copy = self.clone();
        copy.manifest_sha256 = None;
        copy.datasets.sort_by(|a, b| {
            a.family_id
                .cmp(&b.family_id)
                .then(a.dataset_id.cmp(&b.dataset_id))
        });
        let raw = serde_json::to_vec(&copy)?;
        Ok(format!("{:x}", Sha256::digest(&raw)))
    }

    pub fn unique_families(&self) -> usize {
        self.datasets
            .iter()
            .map(|d| d.family_id.as_str())
            .collect::<BTreeSet<_>>()
            .len()
    }

    pub fn verify(&self) -> ArenaResult<()> {
        if self.protocol != "dope-fidelity/v3" {
            return Err(ArenaError::Contract("unsupported panel protocol".into()));
        }
        if self.datasets.is_empty() {
            return Err(ArenaError::Contract("empty panel dataset list".into()));
        }
        let mut seen_ids = BTreeSet::new();
        for ds in &self.datasets {
            if ds.dataset_id.is_empty()
                || ds.family_id.is_empty()
                || ds.target_column.is_empty()
                || !matches!(ds.task.as_str(), "regression" | "binary" | "multiclass")
            {
                return Err(ArenaError::Contract("invalid panel dataset entry".into()));
            }
            if !seen_ids.insert(ds.dataset_id.as_str()) {
                return Err(ArenaError::Contract("duplicate dataset_id in panel".into()));
            }
            if ds.source_sha256.len() != 64
                || !ds
                    .source_sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err(ArenaError::Contract("invalid source_sha256 hex pin".into()));
            }
        }
        if self.role == PanelRole::SealedConfirmation && self.unique_families() < 90 {
            return Err(ArenaError::Contract(
                "sealed confirmation panel requires at least 90 unique physical source families"
                    .into(),
            ));
        }
        if let Some(ref declared) = self.manifest_sha256 {
            let actual = self.canonical_sha256()?;
            if declared != &actual {
                return Err(ArenaError::Contract(
                    "panel manifest_sha256 commitment mismatch".into(),
                ));
            }
        }
        Ok(())
    }
}

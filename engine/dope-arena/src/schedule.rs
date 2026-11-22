//! Complete scheduled benchmark matrix & isolated third-party worker authenticity seam.

use crate::panel::PanelManifest;
use crate::{ArenaError, ArenaResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MethodRegistryEntry {
    pub authenticity: String,
    pub allowed_in_headline_claims: bool,
    pub implementation: String,
    pub legacy_alias: Option<String>,
    pub notes: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MethodsRegistry {
    pub schema: String,
    pub purpose: Option<String>,
    pub methods: BTreeMap<String, MethodRegistryEntry>,
}

impl MethodsRegistry {
    pub fn load(path: &Path) -> ArenaResult<Self> {
        let bytes = std::fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub fn verify_headline_eligible(&self, method_id: &str) -> ArenaResult<()> {
        let entry = self.methods.get(method_id).ok_or_else(|| {
            ArenaError::Contract(format!(
                "unregistered method in methods-registry: {method_id}"
            ))
        })?;
        if !entry.allowed_in_headline_claims || entry.authenticity == "internal_standin" {
            return Err(ArenaError::Contract(format!(
                "method {method_id} is an internal stand-in and barred from headline claims"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScheduledCellSpec {
    pub cell_id: String,
    pub dataset_id: String,
    pub family_id: String,
    pub method_id: String,
    pub tier: String,
    pub mode: String,
    pub auditor_id: String,
    pub data_seed: u64,
    pub sample_seed: u64,
}

/// Expand the full Cartesian benchmark schedule so missing/failed cells can never vanish.
#[allow(clippy::too_many_arguments)]
pub fn build_schedule(
    panel: &PanelManifest,
    methods: &[String],
    tiers: &[String],
    modes: &[String],
    auditors: &[String],
    data_seeds: &[u64],
    sample_seeds: &[u64],
) -> Vec<ScheduledCellSpec> {
    let mut out = Vec::new();
    for ds in &panel.datasets {
        for m in methods {
            for t in tiers {
                for mode in modes {
                    for aud in auditors {
                        for &dseed in data_seeds {
                            for &sseed in sample_seeds {
                                let raw = format!(
                                    "{}|{}|{}|{}|{}|{}|{}",
                                    ds.dataset_id, m, t, mode, aud, dseed, sseed
                                );
                                let digest = format!("{:x}", Sha256::digest(raw.as_bytes()));
                                out.push(ScheduledCellSpec {
                                    cell_id: digest[..16].to_string(),
                                    dataset_id: ds.dataset_id.clone(),
                                    family_id: ds.family_id.clone(),
                                    method_id: m.clone(),
                                    tier: t.clone(),
                                    mode: mode.clone(),
                                    auditor_id: aud.clone(),
                                    data_seed: dseed,
                                    sample_seed: sseed,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

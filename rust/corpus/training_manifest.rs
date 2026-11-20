//! Role-aware training manifest.
//!
//! Historical inventory readers stay in place. A directory that carries this
//! schema with a protected role is not opened as `train.csv` / `test.csv`.
//! A marker that is not a regular file is refused. Accepted TRAIN bases are
//! distinct realized tables. DEV content and any non-TRAIN ancestor stay out
//! of that count. Row order and a new generator seed do not create another base.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::{DopeError, Result, io_error};

pub const SCHEMA: &str = "dope-training-base/v1";
pub const MARKER_FILE: &str = "dope-training-base.json";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceRole {
    Train,
    Dev,
    Confirm,
    Recall,
    Test,
    Query,
    Holdout,
}

impl SourceRole {
    fn parse(raw: &str) -> Result<Self> {
        match raw {
            "TRAIN" => Ok(Self::Train),
            "DEV" => Ok(Self::Dev),
            "CONFIRM" => Ok(Self::Confirm),
            "RECALL" => Ok(Self::Recall),
            "TEST" => Ok(Self::Test),
            "QUERY" => Ok(Self::Query),
            "HOLDOUT" => Ok(Self::Holdout),
            _ => Err(DopeError::Data("training_manifest_role".into())),
        }
    }

    fn protected(self) -> bool {
        matches!(
            self,
            Self::Confirm | Self::Test | Self::Query | Self::Holdout
        )
    }

    fn seals_train_content(self) -> bool {
        self.protected() || matches!(self, Self::Dev)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Origin {
    Real,
    StructuralSynthetic,
    RealConditioned,
    Counterfactual,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskKind {
    Regression,
    Binary,
    Multiclass,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Generator {
    pub mechanism_id: String,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct TrainingBase {
    pub base_dataset_id: String,
    pub physical_rows_sha256: Vec<String>,
    pub source_family_ids: Vec<String>,
    pub ancestor_base_ids: Vec<String>,
    pub origin: Origin,
    pub generator: Option<Generator>,
    pub role: SourceRole,
    pub task: TaskKind,
    pub target_declaration_sha256: String,
    /// Parent base id. Views keep the parent's physical identity and do not count.
    pub view_of: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rejection {
    pub base_dataset_id: String,
    pub code: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Admission {
    pub accepted_base_count: usize,
    pub accepted_base_ids: Vec<String>,
    pub rejected: Vec<Rejection>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Marker {
    schema: String,
    role: String,
}

struct Prepared {
    item: TrainingBase,
    identity: String,
}

/// True when the legacy inventory must not open table bytes in `dir`.
pub fn legacy_blocks_read(dir: &Path) -> Result<bool> {
    let path = dir.join(MARKER_FILE);
    match fs::symlink_metadata(&path) {
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(false),
        Ok(meta) if meta.file_type().is_file() => {}
        Ok(_) => return Err(DopeError::Data("training_manifest_schema".into())),
        Err(error) => return Err(io_error(&path, error)),
    }
    let bytes = fs::read(&path).map_err(|error| io_error(&path, error))?;
    let marker: Marker = serde_json::from_slice(&bytes)
        .map_err(|_| DopeError::Data("training_manifest_schema".into()))?;
    if marker.schema != SCHEMA {
        return Err(DopeError::Data("training_manifest_schema".into()));
    }
    Ok(SourceRole::parse(&marker.role)?.seals_train_content())
}

pub fn physical_base_identity(rows: &[String]) -> Result<String> {
    if rows.is_empty() {
        return Err(DopeError::Data("training_manifest_shape".into()));
    }
    let mut ordered = Vec::with_capacity(rows.len());
    for row in rows {
        digest_text(row)?;
        ordered.push(row.as_str());
    }
    ordered.sort_unstable();
    let mut payload = Vec::new();
    for row in ordered {
        payload.extend(row.as_bytes());
        payload.push(b'\n');
    }
    Ok(sha256_hex(&payload))
}

pub fn admit(items: &[TrainingBase]) -> Result<Admission> {
    let mut prepared = Vec::with_capacity(items.len());
    let mut ids = BTreeSet::new();
    for item in items {
        if item.base_dataset_id.is_empty() || !ids.insert(item.base_dataset_id.clone()) {
            return Err(DopeError::Data("training_manifest_shape".into()));
        }
        if item.source_family_ids.is_empty()
            || item
                .source_family_ids
                .iter()
                .any(|family| family.is_empty())
        {
            return Err(DopeError::Data("training_manifest_shape".into()));
        }
        digest_text(&item.target_declaration_sha256)?;
        if matches!(
            item.origin,
            Origin::RealConditioned | Origin::Counterfactual
        ) && item.ancestor_base_ids.is_empty()
        {
            return Err(DopeError::Data("training_manifest_ancestry".into()));
        }
        let identity = physical_base_identity(&item.physical_rows_sha256)?;
        prepared.push(Prepared {
            item: item.clone(),
            identity,
        });
    }
    let by_id: BTreeMap<&str, &Prepared> = prepared
        .iter()
        .map(|item| (item.item.base_dataset_id.as_str(), item))
        .collect();
    let protected_families: BTreeSet<&str> = prepared
        .iter()
        .filter(|item| item.item.role.seals_train_content())
        .flat_map(|item| item.item.source_family_ids.iter().map(String::as_str))
        .collect();
    let protected_identities: BTreeSet<&str> = prepared
        .iter()
        .filter(|item| item.item.role.seals_train_content())
        .map(|item| item.identity.as_str())
        .collect();

    let mut accepted_ids = Vec::new();
    let mut accepted_identity: BTreeMap<&str, Option<u64>> = BTreeMap::new();
    let mut rejected = Vec::new();
    for item in prepared.iter().filter(|item| item.item.view_of.is_none()) {
        if let Some(code) = base_rejection(item, &by_id, &protected_families, &protected_identities)
        {
            rejected.push(Rejection {
                base_dataset_id: item.item.base_dataset_id.clone(),
                code,
            });
            continue;
        }
        if !item.item.role.protected() && item.item.role != SourceRole::Train {
            continue;
        }
        if item.item.role.protected() {
            continue;
        }
        if let Some(prior_seed) = accepted_identity.get(item.identity.as_str()) {
            let code = if *prior_seed != seed_of(item) {
                "training_manifest_seed"
            } else {
                "training_manifest_duplicate"
            };
            rejected.push(Rejection {
                base_dataset_id: item.item.base_dataset_id.clone(),
                code,
            });
            continue;
        }
        accepted_identity.insert(item.identity.as_str(), seed_of(item));
        accepted_ids.push(item.item.base_dataset_id.clone());
    }
    let accepted_set: BTreeSet<&str> = accepted_ids.iter().map(String::as_str).collect();
    for item in prepared.iter().filter(|item| item.item.view_of.is_some()) {
        let parent_id = item.item.view_of.as_deref().unwrap_or("");
        let Some(parent) = by_id.get(parent_id) else {
            rejected.push(Rejection {
                base_dataset_id: item.item.base_dataset_id.clone(),
                code: "training_manifest_ancestry",
            });
            continue;
        };
        if parent.identity != item.identity || parent.item.view_of.is_some() {
            rejected.push(Rejection {
                base_dataset_id: item.item.base_dataset_id.clone(),
                code: "training_manifest_view",
            });
            continue;
        }
        if item.item.role == SourceRole::Train && parent.item.role != SourceRole::Train {
            rejected.push(Rejection {
                base_dataset_id: item.item.base_dataset_id.clone(),
                code: "training_manifest_ancestry",
            });
            continue;
        }
        if item.item.role == SourceRole::Train && !accepted_set.contains(parent_id) {
            rejected.push(Rejection {
                base_dataset_id: item.item.base_dataset_id.clone(),
                code: "parent_rejected",
            });
            continue;
        }
        if let Some(code) = base_rejection(item, &by_id, &protected_families, &protected_identities)
        {
            rejected.push(Rejection {
                base_dataset_id: item.item.base_dataset_id.clone(),
                code,
            });
        }
    }
    Ok(Admission {
        accepted_base_count: accepted_ids.len(),
        accepted_base_ids: accepted_ids,
        rejected,
    })
}

fn base_rejection<'a>(
    item: &'a Prepared,
    by_id: &BTreeMap<&str, &'a Prepared>,
    protected_families: &BTreeSet<&str>,
    protected_identities: &BTreeSet<&str>,
) -> Option<&'static str> {
    if item.item.role != SourceRole::Train {
        return None;
    }
    if protected_identities.contains(item.identity.as_str())
        || item
            .item
            .source_family_ids
            .iter()
            .any(|family| protected_families.contains(family.as_str()))
    {
        return Some("training_manifest_ancestry");
    }
    let mut stack = BTreeSet::new();
    if ancestry_blocked(item.item.base_dataset_id.as_str(), by_id, &mut stack) {
        return Some("training_manifest_ancestry");
    }
    None
}

fn ancestry_blocked<'a>(
    id: &'a str,
    by_id: &BTreeMap<&str, &'a Prepared>,
    stack: &mut BTreeSet<&'a str>,
) -> bool {
    let Some(item) = by_id.get(id) else {
        return true;
    };
    if item.item.role != SourceRole::Train {
        return true;
    }
    if !stack.insert(id) {
        return true;
    }
    let blocked = item.item.ancestor_base_ids.iter().any(|ancestor| {
        !by_id.contains_key(ancestor.as_str()) || ancestry_blocked(ancestor.as_str(), by_id, stack)
    });
    stack.remove(id);
    blocked
}

fn seed_of(item: &Prepared) -> Option<u64> {
    item.item.generator.as_ref().map(|generator| generator.seed)
}

fn digest_text(value: &str) -> Result<()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err(DopeError::Data("training_manifest_digest".into()))
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
#[path = "training_manifest_tests.rs"]
mod tests;

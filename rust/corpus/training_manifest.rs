//! Role-aware training manifest.
//!
//! Historical inventory readers stay in place. A directory that carries this
//! schema with a protected role is not opened as `train.csv` / `test.csv`.
//! Accepted TRAIN bases are distinct realized tables: row order, a new target,
//! and a new generator seed do not create another base.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
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
    if !path.is_file() {
        return Ok(false);
    }
    let bytes = fs::read(&path).map_err(|error| io_error(&path, error))?;
    let marker: Marker = serde_json::from_slice(&bytes)
        .map_err(|_| DopeError::Data("training_manifest_schema".into()))?;
    if marker.schema != SCHEMA {
        return Err(DopeError::Data("training_manifest_schema".into()));
    }
    Ok(SourceRole::parse(&marker.role)?.protected())
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
    let protected_ids: BTreeSet<&str> = prepared
        .iter()
        .filter(|item| item.item.role.protected())
        .map(|item| item.item.base_dataset_id.as_str())
        .collect();
    let protected_families: BTreeSet<&str> = prepared
        .iter()
        .filter(|item| item.item.role.protected())
        .flat_map(|item| item.item.source_family_ids.iter().map(String::as_str))
        .collect();
    let protected_identities: BTreeSet<&str> = prepared
        .iter()
        .filter(|item| item.item.role.protected())
        .map(|item| item.identity.as_str())
        .collect();

    let mut accepted_ids = Vec::new();
    let mut accepted_identity: BTreeMap<&str, Option<u64>> = BTreeMap::new();
    let mut rejected = Vec::new();
    for item in prepared.iter().filter(|item| item.item.view_of.is_none()) {
        if let Some(code) = base_rejection(
            item,
            &by_id,
            &protected_ids,
            &protected_families,
            &protected_identities,
        ) {
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
    protected_ids: &BTreeSet<&str>,
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
    if ancestry_blocked(
        item.item.base_dataset_id.as_str(),
        by_id,
        protected_ids,
        &mut stack,
    ) {
        return Some("training_manifest_ancestry");
    }
    None
}

fn ancestry_blocked<'a>(
    id: &'a str,
    by_id: &BTreeMap<&str, &'a Prepared>,
    protected_ids: &BTreeSet<&str>,
    stack: &mut BTreeSet<&'a str>,
) -> bool {
    let Some(item) = by_id.get(id) else {
        return true;
    };
    if item.item.role.protected() || protected_ids.contains(id) {
        return true;
    }
    if !stack.insert(id) {
        return true;
    }
    let blocked = item.item.ancestor_base_ids.iter().any(|ancestor| {
        !by_id.contains_key(ancestor.as_str())
            || ancestry_blocked(ancestor.as_str(), by_id, protected_ids, stack)
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
mod tests {
    use super::*;
    use std::io::Write;

    fn hex(fill: u8) -> String {
        format!("{fill:02x}").repeat(32)
    }

    fn train(id: &str, rows: Vec<String>, families: &[&str]) -> TrainingBase {
        TrainingBase {
            base_dataset_id: id.into(),
            physical_rows_sha256: rows,
            source_family_ids: families
                .iter()
                .map(|family| (*family).to_string())
                .collect(),
            ancestor_base_ids: Vec::new(),
            origin: Origin::Real,
            generator: None,
            role: SourceRole::Train,
            task: TaskKind::Regression,
            target_declaration_sha256: hex(0x11),
            view_of: None,
        }
    }

    #[test]
    fn legacy_inventory_does_not_read_protected_bytes() {
        let root = Path::new("target/training-manifest").join(format!("{}", std::process::id()));
        let public = root.join("public");
        let sealed = root.join("sealed");
        fs::create_dir_all(&public).unwrap();
        fs::create_dir_all(&sealed).unwrap();
        for (path, body) in [
            (public.join("train.csv"), "0,1,0\n"),
            (public.join("test.csv"), "1,0,1\n"),
            (sealed.join("train.csv"), "SECRET_LABEL,1\n"),
            (sealed.join("test.csv"), "SECRET_LABEL,0\n"),
        ] {
            let mut file = fs::File::create(path).unwrap();
            write!(file, "{body}").unwrap();
        }
        fs::write(
            sealed.join(MARKER_FILE),
            r#"{"schema":"dope-training-base/v1","role":"CONFIRM"}"#,
        )
        .unwrap();
        let inventory = super::super::inventory_corpus(&[root.clone()]).unwrap();
        assert_eq!(inventory.datasets.len(), 1);
        assert_eq!(inventory.exclusions.len(), 1);
        assert_eq!(inventory.exclusions[0].reason, "protected");
        assert_eq!(
            inventory.exclusions[0].detail,
            "training_manifest_protected"
        );
        let rendered = format!("{inventory:?}");
        assert!(!rendered.contains("SECRET_LABEL"));
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn malformed_marker_refuses_before_table_bytes() {
        let root =
            Path::new("target/training-manifest-bad").join(format!("{}", std::process::id()));
        let sealed = root.join("sealed");
        fs::create_dir_all(&sealed).unwrap();
        fs::write(sealed.join("train.csv"), "SECRET_LABEL,1\n").unwrap();
        fs::write(sealed.join("test.csv"), "SECRET_LABEL,0\n").unwrap();
        fs::write(sealed.join(MARKER_FILE), "{").unwrap();
        let error = super::super::inventory_corpus(&[root.clone()]).unwrap_err();
        let rendered = error.to_string();
        assert!(rendered.contains("training_manifest_schema"));
        assert!(!rendered.contains("SECRET_LABEL"));
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn derived_family_alias_is_refused() {
        let mut protected = train("secret-source", vec![hex(1)], &["family-secret"]);
        protected.role = SourceRole::Confirm;
        let mut renamed = train("public-name", vec![hex(2)], &["family-secret"]);
        renamed.ancestor_base_ids = vec!["secret-source".into()];
        let mut derived = train("derived", vec![hex(3)], &["other-family"]);
        derived.ancestor_base_ids = vec!["secret-source".into()];
        let copied = train("copied-bytes", vec![hex(1)], &["unrelated"]);
        let admission = admit(&[protected, renamed, derived, copied]).unwrap();
        assert_eq!(admission.accepted_base_count, 0);
        assert_eq!(admission.rejected.len(), 3);
        assert!(
            admission
                .rejected
                .iter()
                .all(|rejection| rejection.code == "training_manifest_ancestry")
        );
    }

    #[test]
    fn reordering_does_not_increment_base_count() {
        let first = train("base", vec![hex(1), hex(2)], &["family-a"]);
        let mut reordered = train("reordered", vec![hex(2), hex(1)], &["family-a"]);
        reordered.target_declaration_sha256 = hex(0x22);
        let mut view = train("view", vec![hex(2), hex(1)], &["family-a"]);
        view.view_of = Some("base".into());
        view.target_declaration_sha256 = hex(0x33);
        let admission = admit(&[first, reordered, view]).unwrap();
        assert_eq!(admission.accepted_base_count, 1);
        assert_eq!(admission.accepted_base_ids, vec!["base".to_string()]);
        assert_eq!(admission.rejected.len(), 1);
        assert_eq!(admission.rejected[0].code, "training_manifest_duplicate");
        assert_eq!(
            physical_base_identity(&[hex(1), hex(2)]).unwrap(),
            physical_base_identity(&[hex(2), hex(1)]).unwrap()
        );
    }

    #[test]
    fn seed_collision_is_rejected_by_realized_content() {
        let mut first = train("draw-a", vec![hex(4), hex(5)], &["mech"]);
        first.origin = Origin::StructuralSynthetic;
        first.generator = Some(Generator {
            mechanism_id: "generator-a".into(),
            seed: 1,
        });
        let mut second = train("draw-b", vec![hex(5), hex(4)], &["mech"]);
        second.origin = Origin::StructuralSynthetic;
        second.generator = Some(Generator {
            mechanism_id: "generator-a".into(),
            seed: 2,
        });
        let mut distinct = train("draw-c", vec![hex(6)], &["mech"]);
        distinct.origin = Origin::StructuralSynthetic;
        distinct.generator = Some(Generator {
            mechanism_id: "generator-a".into(),
            seed: 2,
        });
        let admission = admit(&[first, second, distinct]).unwrap();
        assert_eq!(admission.accepted_base_count, 2);
        assert_eq!(
            admission.rejected,
            vec![Rejection {
                base_dataset_id: "draw-b".into(),
                code: "training_manifest_seed",
            }]
        );
    }
}

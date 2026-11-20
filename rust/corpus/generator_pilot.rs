//! Deterministic generator pilot.
//!
//! Admitted TRAIN bases are counted from realized rows. Views and a
//! conditioning ancestor stay outside that count. An unsupported task is
//! written to the reject ledger instead of being dropped.

use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use super::training_manifest::{
    Generator, Origin, Rejection, SourceRole, TaskKind, TrainingBase, admit,
};
use crate::error::{DopeError, Result};

pub const SCHEMA: &str = "dope-generator-pilot/v1";
pub const PILOT_BASES: usize = 100_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Arm {
    pub mechanism_id: String,
    pub weight: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Plan {
    pub seed: u64,
    pub shard_id: u32,
    pub bases: usize,
    pub arms: Vec<Arm>,
    pub task: String,
    pub views_per_base: usize,
    pub ancestor_role: SourceRole,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PilotReport {
    pub schema: &'static str,
    pub requested_bases: usize,
    pub accepted_base_count: usize,
    pub conditioning_base_count: usize,
    pub requested_views: usize,
    pub view_count: usize,
    pub rejected: Vec<Rejection>,
    pub mechanism_counts: BTreeMap<String, usize>,
    pub content_sha256: Option<String>,
    pub sample_row_sha256: Option<String>,
}

struct BaseDraft {
    id: String,
    rows: Vec<String>,
    family: String,
    origin: Origin,
    generator: Option<Generator>,
    role: SourceRole,
    task: TaskKind,
    target: String,
    ancestors: Vec<String>,
    view_of: Option<String>,
}

struct Mechanism {
    origin: Origin,
    conditioned: bool,
}

pub fn run(plan: &Plan) -> Result<PilotReport> {
    validate(plan)?;
    let requested_views = plan
        .bases
        .checked_mul(plan.views_per_base)
        .ok_or_else(|| DopeError::Data("generator_pilot_limit".into()))?;
    if !supported_task(&plan.task) {
        return Ok(PilotReport {
            schema: SCHEMA,
            requested_bases: plan.bases,
            accepted_base_count: 0,
            conditioning_base_count: 0,
            requested_views,
            view_count: 0,
            rejected: (0..plan.bases)
                .map(|index| Rejection {
                    base_dataset_id: base_id(plan.shard_id, index),
                    code: "generator_pilot_task",
                })
                .collect(),
            mechanism_counts: BTreeMap::new(),
            content_sha256: None,
            sample_row_sha256: None,
        });
    }
    let task = task_kind(&plan.task);
    let target = sha256_hex(format!("{SCHEMA}\n{task_label}\n", task_label = plan.task).as_bytes());
    let mut items = Vec::new();
    let mut requested = BTreeSet::new();
    let mut view_ids = BTreeSet::new();
    let mut rows = Vec::with_capacity(plan.bases);
    let mut mechanisms = Vec::with_capacity(plan.bases);
    let needs_ancestor = plan
        .arms
        .iter()
        .any(|arm| registry(&arm.mechanism_id).is_some_and(|mechanism| mechanism.conditioned));
    let ancestor = ancestor_id(plan.shard_id);
    if needs_ancestor {
        let digest = row_digest("ancestor", plan, 0);
        items.push(record(BaseDraft {
            id: ancestor.clone(),
            rows: vec![digest],
            family: "pilot-ancestor".to_string(),
            origin: Origin::Real,
            generator: None,
            role: plan.ancestor_role,
            task,
            target: target.clone(),
            ancestors: Vec::new(),
            view_of: None,
        }));
    }
    for index in 0..plan.bases {
        let arm = pick(plan, index)?;
        let mechanism = registry(&arm.mechanism_id).expect("validated arm");
        let digest = row_digest(&arm.mechanism_id, plan, index);
        let id = base_id(plan.shard_id, index);
        let ancestors = if mechanism.conditioned {
            vec![ancestor.clone()]
        } else {
            Vec::new()
        };
        rows.push(digest.clone());
        mechanisms.push(arm.mechanism_id.clone());
        requested.insert(id.clone());
        let family = format!("pilot-{}", arm.mechanism_id);
        let generator = Some(Generator {
            mechanism_id: arm.mechanism_id.clone(),
            seed: plan.seed,
        });
        items.push(record(BaseDraft {
            id: id.clone(),
            rows: vec![digest.clone()],
            family: family.clone(),
            origin: mechanism.origin,
            generator: generator.clone(),
            role: SourceRole::Train,
            task,
            target: target.clone(),
            ancestors: ancestors.clone(),
            view_of: None,
        }));
        for view in 0..plan.views_per_base {
            let view_name = view_id(plan.shard_id, index, view);
            view_ids.insert(view_name.clone());
            items.push(record(BaseDraft {
                id: view_name,
                rows: vec![digest.clone()],
                family: family.clone(),
                origin: mechanism.origin,
                generator: generator.clone(),
                role: SourceRole::Train,
                task,
                target: target.clone(),
                ancestors: ancestors.clone(),
                view_of: Some(id.clone()),
            }));
        }
    }
    let admission = admit(&items)?;
    let mut rejected = Vec::new();
    for rejection in &admission.rejected {
        if view_ids.contains(&rejection.base_dataset_id) {
            continue;
        }
        if requested.contains(&rejection.base_dataset_id) {
            rejected.push(rejection.clone());
        }
    }
    let admitted_views = items
        .iter()
        .filter(|item| item.view_of.is_some())
        .filter(|item| {
            !admission
                .rejected
                .iter()
                .any(|rejection| rejection.base_dataset_id == item.base_dataset_id)
        })
        .count();
    let accepted_ids: BTreeSet<&str> = admission
        .accepted_base_ids
        .iter()
        .map(String::as_str)
        .collect();
    let mut mechanism_counts = BTreeMap::new();
    let mut accepted_rows = Vec::new();
    for (index, mechanism_id) in mechanisms.iter().enumerate() {
        if accepted_ids.contains(base_id(plan.shard_id, index).as_str()) {
            *mechanism_counts.entry(mechanism_id.clone()).or_insert(0) += 1;
            accepted_rows.push(rows[index].as_str());
        }
    }
    let accepted_base_count = mechanism_counts.values().sum();
    Ok(PilotReport {
        schema: SCHEMA,
        requested_bases: plan.bases,
        accepted_base_count,
        conditioning_base_count: usize::from(accepted_ids.contains(ancestor.as_str())),
        requested_views,
        view_count: admitted_views,
        rejected,
        mechanism_counts,
        content_sha256: (!accepted_rows.is_empty()).then(|| content_digest(&accepted_rows)),
        sample_row_sha256: accepted_rows.first().map(|row| (*row).to_string()),
    })
}

fn validate(plan: &Plan) -> Result<()> {
    if plan.bases == 0 || plan.bases > PILOT_BASES {
        return Err(DopeError::Data("generator_pilot_limit".into()));
    }
    if plan.views_per_base > 2 || (plan.views_per_base > 0 && plan.bases > 64) {
        return Err(DopeError::Data("generator_pilot_limit".into()));
    }
    if plan.arms.is_empty() {
        return Err(DopeError::Data("generator_pilot_mechanism".into()));
    }
    let mut seen = BTreeSet::new();
    for arm in &plan.arms {
        if arm.weight == 0
            || !seen.insert(arm.mechanism_id.as_str())
            || registry(&arm.mechanism_id).is_none()
        {
            return Err(DopeError::Data("generator_pilot_mechanism".into()));
        }
    }
    Ok(())
}

fn registry(mechanism_id: &str) -> Option<Mechanism> {
    match mechanism_id {
        "independent" => Some(Mechanism {
            origin: Origin::StructuralSynthetic,
            conditioned: false,
        }),
        "real-conditioned" => Some(Mechanism {
            origin: Origin::RealConditioned,
            conditioned: true,
        }),
        "counterfactual" => Some(Mechanism {
            origin: Origin::Counterfactual,
            conditioned: true,
        }),
        _ => None,
    }
}

fn supported_task(task: &str) -> bool {
    matches!(task, "regression" | "binary" | "multiclass")
}

fn task_kind(task: &str) -> TaskKind {
    match task {
        "binary" => TaskKind::Binary,
        "multiclass" => TaskKind::Multiclass,
        _ => TaskKind::Regression,
    }
}

fn pick(plan: &Plan, index: usize) -> Result<&Arm> {
    let total: u64 = plan.arms.iter().map(|arm| u64::from(arm.weight)).sum();
    let mut draw = mix(plan.seed, plan.shard_id, index as u64) % total;
    for arm in &plan.arms {
        let weight = u64::from(arm.weight);
        if draw < weight {
            return Ok(arm);
        }
        draw -= weight;
    }
    Err(DopeError::Data("generator_pilot_mechanism".into()))
}

fn record(draft: BaseDraft) -> TrainingBase {
    TrainingBase {
        base_dataset_id: draft.id,
        physical_rows_sha256: draft.rows,
        source_family_ids: vec![draft.family],
        ancestor_base_ids: draft.ancestors,
        origin: draft.origin,
        generator: draft.generator,
        role: draft.role,
        task: draft.task,
        target_declaration_sha256: draft.target,
        view_of: draft.view_of,
    }
}

fn ancestor_id(shard: u32) -> String {
    format!("a{shard:010}")
}

fn base_id(shard: u32, index: usize) -> String {
    format!("b{shard:010}-{index:06}")
}

fn view_id(shard: u32, index: usize, view: usize) -> String {
    format!("v{shard:010}-{index:06}-{view}")
}

fn row_digest(mechanism: &str, plan: &Plan, index: usize) -> String {
    sha256_hex(
        format!(
            "{SCHEMA}\n{mechanism}\n{}\n{}\n{index}\n{}",
            plan.seed, plan.shard_id, plan.task
        )
        .as_bytes(),
    )
}

fn content_digest(rows: &[&str]) -> String {
    let mut payload = Vec::new();
    for row in rows {
        payload.extend(row.as_bytes());
        payload.push(b'\n');
    }
    sha256_hex(&payload)
}

fn mix(seed: u64, shard: u32, index: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(SCHEMA.as_bytes());
    hasher.update(seed.to_le_bytes());
    hasher.update(shard.to_le_bytes());
    hasher.update(index.to_le_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
#[path = "generator_pilot_tests.rs"]
mod tests;

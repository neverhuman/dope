use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::campaign::{CohortPlan, CohortRecord, GoldRecordBlock};
use crate::codec::{MAX_NEURAL_ARTIFACT_BYTES, decode_kernel, encode_kernel};
use crate::contract::candidate_implementation_hash;
use crate::error::{DopeError, Result};
use crate::ledger::{CacheStatus, JOB_EVIDENCE_VERSION, JobEvidence};
use crate::model::{JointNetwork, Kernel, KernelProgram, Marginal, QuantizedLinear};
use crate::production::{canonical_json, hashes, read_json};

pub const DISCOVERY_GENERATION_SEED: u64 = 1_829;
pub const DISCOVERY_AUDITOR_SEED: u64 = 57_721;
pub const CONFIRMATION_GENERATION_SEEDS: [u64; 3] = [1_829, 99_238, 196_647];
pub const CONFIRMATION_AUDITOR_SEEDS: [u64; 3] = [57_721, 161_803, 271_828];
pub const CONFIRMATION_SIZE_MULTIPLIERS: [usize; 2] = [1, 4];
pub const AUDITORS_PER_CELL: usize = 6;
pub const CONFIRMATION_CELLS_PER_CANDIDATE_LINEAGE: usize = 108;
pub const EXPECTED_DISCOVERY_LINEAGES: usize = 206;
pub const EXPECTED_CONFIRMATION_LINEAGES: usize = 601;
pub const VALIDATION_SELECT_LINEAGES: usize = 11_892;
pub const MAX_CONFIRMATION_CELLS: usize = 908_712;
pub const MAX_VALIDATION_SELECT_CELLS: usize = 17_980_704;
pub const DURABLE_FREE_BYTES: u64 = 150 * 1024 * 1024 * 1024;
pub const FIT_TIMEOUT_SECONDS: u64 = 600;
pub const GENERATION_TIMEOUT_SECONDS: u64 = 60;
pub const GPU_MEMORY_LIMIT_BYTES: u64 = 16 * 1024 * 1024 * 1024;
pub const ROUTER_ARTIFACT_LIMIT_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DeepSourceEntry {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub blake3: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DeepFreezeManifest {
    pub format: String,
    pub version: u8,
    pub evidence_version: u8,
    pub artifact_format: String,
    pub source_tree_sha256: String,
    pub source_tree_blake3: String,
    pub source_files: Vec<DeepSourceEntry>,
    pub binary_bytes: u64,
    pub binary_sha256: String,
    pub binary_blake3: String,
    pub candidate_implementation_hashes: BTreeMap<String, String>,
}

fn source_files(root: &Path, directory: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries = fs::read_dir(directory)
        .map_err(|error| crate::error::io_error(directory, error))?
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .map_err(|error| crate::error::io_error(directory, error))
        })
        .collect::<Result<Vec<_>>>()?;
    entries.sort();
    for path in entries {
        if path.is_dir() {
            let relative = path.strip_prefix(root).unwrap_or(&path);
            if relative.components().any(|component| {
                matches!(
                    component.as_os_str().to_str(),
                    Some(".git" | ".pytest_cache" | "__pycache__" | "runs" | "target")
                )
            }) {
                continue;
            }
            source_files(root, &path, files)?;
        } else if path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

pub fn freeze_deep_implementation(source_root: &Path, binary: &Path) -> Result<DeepFreezeManifest> {
    if !source_root.is_dir() || !binary.is_file() {
        return Err(DopeError::Data(
            "deep freeze requires a source directory and built binary".into(),
        ));
    }
    let mut paths = Vec::new();
    source_files(source_root, source_root, &mut paths)?;
    let mut source_entries = Vec::new();
    for path in paths {
        let bytes = fs::read(&path).map_err(|error| crate::error::io_error(&path, error))?;
        let content = hashes(&bytes);
        source_entries.push(DeepSourceEntry {
            path: path
                .strip_prefix(source_root)
                .map_err(|_| DopeError::Data("source path escaped its root".into()))?
                .to_string_lossy()
                .replace('\\', "/"),
            bytes: bytes.len() as u64,
            sha256: content.sha256,
            blake3: content.blake3,
        });
    }
    source_entries.sort_by(|left, right| left.path.cmp(&right.path));
    let source_hashes = hashes(&canonical_json(&source_entries)?);
    let binary_bytes = fs::read(binary).map_err(|error| crate::error::io_error(binary, error))?;
    let binary_hashes = hashes(&binary_bytes);
    let candidate_implementation_hashes = NEW_CANDIDATES
        .into_iter()
        .chain(std::iter::once(LEGACY_DEEP_CHAMPION))
        .map(|candidate| {
            (
                candidate.to_string(),
                candidate_implementation_hash(candidate),
            )
        })
        .collect();
    Ok(DeepFreezeManifest {
        format: "dope-deep-source-binary-implementation-freeze".into(),
        version: 1,
        evidence_version: JOB_EVIDENCE_VERSION,
        artifact_format: "DPK3.2".into(),
        source_tree_sha256: source_hashes.sha256,
        source_tree_blake3: source_hashes.blake3,
        source_files: source_entries,
        binary_bytes: binary_bytes.len() as u64,
        binary_sha256: binary_hashes.sha256,
        binary_blake3: binary_hashes.blake3,
        candidate_implementation_hashes,
    })
}

pub const NEW_CANDIDATES: [&str; 3] = ["tvae", "single_table_autoregressive_transformer", "tabsyn"];
pub const LEGACY_DEEP_CANDIDATES: [&str; 7] = [
    "legacy_tvae",
    "legacy_ctgan",
    "legacy_taegan",
    "legacy_tabddpm",
    "legacy_tabsyn",
    "legacy_single_table_autoregressive_transformer",
    "legacy_masked_diffusion_transformer",
];
pub const LEGACY_DEEP_CHAMPION: &str = "legacy_tvae";
pub const REFERENCE_FRONTIER: [&str; 4] = [
    "tabsds_rank",
    "sparse_gaussian_copula",
    "forest_diffusion_vp",
    "synthetic_pretrained_cross_table",
];

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct LossConfiguration {
    pub target_weight: f64,
    pub structural_penalty: f64,
}

impl LossConfiguration {
    pub const GRID: [Self; 4] = [
        Self {
            target_weight: 2.0,
            structural_penalty: 0.0,
        },
        Self {
            target_weight: 2.0,
            structural_penalty: 0.1,
        },
        Self {
            target_weight: 4.0,
            structural_penalty: 0.0,
        },
        Self {
            target_weight: 4.0,
            structural_penalty: 0.1,
        },
    ];

    pub fn id(self) -> String {
        format!(
            "target-{}-structural-{}",
            self.target_weight as usize,
            if self.structural_penalty == 0.0 {
                "0"
            } else {
                "0.1"
            }
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct DeepCohortPlan {
    pub format: String,
    pub version: u8,
    pub source_kind: String,
    pub discovery: Vec<CohortRecord>,
    pub confirmation: Vec<CohortRecord>,
}

fn lineage_order(record: &CohortRecord) -> ([u8; 32], String) {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"deep-joint-cohort-v1");
    hasher.update(record.structural_profile.as_bytes());
    hasher.update(record.lineage_group_id.as_bytes());
    (*hasher.finalize().as_bytes(), record.dataset_id.clone())
}

pub fn plan_deep_cohorts(training_gold: &CohortPlan) -> Result<DeepCohortPlan> {
    if training_gold.kind != "training-gold" {
        return Err(DopeError::Data(
            "deep discovery and confirmation require the training-gold cohort".into(),
        ));
    }
    let mut canonical = BTreeMap::<(String, String), CohortRecord>::new();
    for record in &training_gold.records {
        let key = (
            record.structural_profile.clone(),
            record.lineage_group_id.clone(),
        );
        let entry = canonical.entry(key).or_insert_with(|| record.clone());
        if lineage_order(record) < lineage_order(entry) {
            *entry = record.clone();
        }
    }
    let mut profiles = BTreeMap::<String, Vec<CohortRecord>>::new();
    for ((profile, _), record) in canonical {
        profiles.entry(profile).or_default().push(record);
    }
    for records in profiles.values_mut() {
        records.sort_by_key(lineage_order);
    }
    let mut discovery = Vec::new();
    let mut confirmation = Vec::new();
    for records in profiles.values() {
        let discovery_count = records.len().min(8);
        let confirmation_count = records.len().saturating_sub(discovery_count).min(32);
        discovery.extend(records[..discovery_count].iter().cloned());
        confirmation.extend(
            records[discovery_count..discovery_count + confirmation_count]
                .iter()
                .cloned(),
        );
    }
    discovery.sort_by(|left, right| {
        left.structural_profile
            .cmp(&right.structural_profile)
            .then_with(|| lineage_order(left).cmp(&lineage_order(right)))
    });
    confirmation.sort_by(|left, right| {
        left.structural_profile
            .cmp(&right.structural_profile)
            .then_with(|| lineage_order(left).cmp(&lineage_order(right)))
    });
    let discovery_cells = discovery
        .iter()
        .map(|record| {
            (
                record.structural_profile.as_str(),
                record.lineage_group_id.as_str(),
            )
        })
        .collect::<BTreeSet<_>>();
    if confirmation.iter().any(|record| {
        discovery_cells.contains(&(
            record.structural_profile.as_str(),
            record.lineage_group_id.as_str(),
        ))
    }) {
        return Err(DopeError::Data(
            "deep discovery and confirmation cohorts overlap".into(),
        ));
    }
    Ok(DeepCohortPlan {
        format: "dope-deep-joint-cohorts".into(),
        version: 1,
        source_kind: training_gold.kind.clone(),
        discovery,
        confirmation,
    })
}

pub fn validation_select_cells(qualifying_candidates: usize) -> Result<usize> {
    if qualifying_candidates > NEW_CANDIDATES.len() {
        return Err(DopeError::Data(
            "validation-select has more candidates than the frozen deep campaign".into(),
        ));
    }
    Ok(VALIDATION_SELECT_LINEAGES
        * (qualifying_candidates + LEGACY_DEEP_CANDIDATES.len() + REFERENCE_FRONTIER.len())
        * CONFIRMATION_CELLS_PER_CANDIDATE_LINEAGE)
}

impl DeepCohortPlan {
    pub fn stage_cohort(&self, stage: &str) -> Result<CohortPlan> {
        let records = match stage {
            "discovery" => self.discovery.clone(),
            "confirmation" => self.confirmation.clone(),
            _ => {
                return Err(DopeError::Data(
                    "deep cohort stage must be discovery or confirmation".into(),
                ));
            }
        };
        Ok(CohortPlan {
            format: "dope-campaign-cohort".into(),
            version: 1,
            kind: "training-gold".into(),
            records,
            exclusions: Vec::new(),
        })
    }

    pub fn discovery_cells(&self) -> usize {
        // 12 neural configurations plus the fixed legacy champion, one size,
        // one generation seed, one auditor seed, and all six auditors.
        self.discovery.len() * (NEW_CANDIDATES.len() * LossConfiguration::GRID.len() + 1) * 6
    }

    pub fn confirmation_cells(&self, candidates: usize) -> usize {
        self.confirmation.len() * candidates * CONFIRMATION_CELLS_PER_CANDIDATE_LINEAGE
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceState {
    Succeeded,
    Failed,
    TimedOut,
    Oom,
    Malformed,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct DeepMetrics {
    pub null_loss: Option<f64>,
    pub trtr_loss: Option<f64>,
    pub tstr_loss: Option<f64>,
    pub bounded_retention: Option<f64>,
    pub absolute_lift: Option<f64>,
    pub regret: Option<f64>,
    pub oracle_recall: Option<f64>,
    pub calibration_degradation: Option<f64>,
    pub query_p95_error: Option<f64>,
    pub type_i_error: Option<f64>,
    pub membership_auc: Option<f64>,
    pub attribute_inference_advantage: Option<f64>,
    pub driver_agreement: Option<f64>,
    pub joint_fidelity: Option<f64>,
    pub rare_tail_retention: Option<f64>,
    pub supported_subgroup_retention: Option<f64>,
    pub nominal_coverage: Option<f64>,
    pub feature_importance_spearman: Option<f64>,
    pub feature_importance_top_k_agreement: Option<f64>,
    pub feature_importance_informative_groups: Option<f64>,
    pub exact_copies: Option<u64>,
    pub near_copies: Option<u64>,
    pub canary_extractions: Option<u64>,
    pub lineage_leaks: Option<u64>,
}
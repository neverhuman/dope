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

impl DeepMetrics {
    fn finite(&self) -> bool {
        [
            self.null_loss,
            self.trtr_loss,
            self.tstr_loss,
            self.bounded_retention,
            self.absolute_lift,
            self.regret,
            self.oracle_recall,
            self.calibration_degradation,
            self.query_p95_error,
            self.type_i_error,
            self.membership_auc,
            self.attribute_inference_advantage,
            self.driver_agreement,
            self.joint_fidelity,
            self.rare_tail_retention,
            self.supported_subgroup_retention,
            self.nominal_coverage,
            self.feature_importance_spearman,
            self.feature_importance_top_k_agreement,
            self.feature_importance_informative_groups,
        ]
        .into_iter()
        .flatten()
        .all(f64::is_finite)
    }

    fn add_assign(&mut self, other: &Self) {
        let add = |left: &mut Option<f64>, right: Option<f64>| {
            *left = left.zip(right).map(|(left, right)| left + right);
        };
        add(&mut self.null_loss, other.null_loss);
        add(&mut self.trtr_loss, other.trtr_loss);
        add(&mut self.tstr_loss, other.tstr_loss);
        add(&mut self.bounded_retention, other.bounded_retention);
        add(&mut self.absolute_lift, other.absolute_lift);
        add(&mut self.regret, other.regret);
        add(&mut self.oracle_recall, other.oracle_recall);
        add(
            &mut self.calibration_degradation,
            other.calibration_degradation,
        );
        add(&mut self.query_p95_error, other.query_p95_error);
        add(&mut self.type_i_error, other.type_i_error);
        add(&mut self.membership_auc, other.membership_auc);
        add(
            &mut self.attribute_inference_advantage,
            other.attribute_inference_advantage,
        );
        add(&mut self.driver_agreement, other.driver_agreement);
        add(&mut self.joint_fidelity, other.joint_fidelity);
        add(&mut self.rare_tail_retention, other.rare_tail_retention);
        add(
            &mut self.supported_subgroup_retention,
            other.supported_subgroup_retention,
        );
        add(&mut self.nominal_coverage, other.nominal_coverage);
        add(
            &mut self.feature_importance_spearman,
            other.feature_importance_spearman,
        );
        add(
            &mut self.feature_importance_top_k_agreement,
            other.feature_importance_top_k_agreement,
        );
        add(
            &mut self.feature_importance_informative_groups,
            other.feature_importance_informative_groups,
        );
        let add_count = |left: &mut Option<u64>, right: Option<u64>| {
            *left = left.zip(right).map(|(left, right)| left + right);
        };
        add_count(&mut self.exact_copies, other.exact_copies);
        add_count(&mut self.near_copies, other.near_copies);
        add_count(&mut self.canary_extractions, other.canary_extractions);
        add_count(&mut self.lineage_leaks, other.lineage_leaks);
    }

    fn divide(&mut self, denominator: f64) {
        for value in [
            &mut self.null_loss,
            &mut self.trtr_loss,
            &mut self.tstr_loss,
            &mut self.bounded_retention,
            &mut self.absolute_lift,
            &mut self.regret,
            &mut self.oracle_recall,
            &mut self.calibration_degradation,
            &mut self.query_p95_error,
            &mut self.type_i_error,
            &mut self.membership_auc,
            &mut self.attribute_inference_advantage,
            &mut self.driver_agreement,
            &mut self.joint_fidelity,
            &mut self.rare_tail_retention,
            &mut self.supported_subgroup_retention,
            &mut self.nominal_coverage,
            &mut self.feature_importance_spearman,
            &mut self.feature_importance_top_k_agreement,
            &mut self.feature_importance_informative_groups,
        ]
        .into_iter()
        .flatten()
        {
            *value /= denominator;
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DeepOutcome {
    pub task: String,
    pub lineage_group_id: String,
    pub structural_profile: String,
    pub candidate_id: String,
    pub configuration: Option<LossConfiguration>,
    pub generation_seed: u64,
    pub auditor_seed: u64,
    pub auditor_id: String,
    pub size_multiplier: usize,
    pub state: EvidenceState,
    /// Must be absent unless state is `succeeded`.
    pub metrics: Option<DeepMetrics>,
    #[serde(default)]
    pub failure_reason: Option<String>,
    pub artifact_bytes: Option<u64>,
    pub runtime_ms: Option<u64>,
    pub fitting_time_ms: Option<u64>,
    pub sampling_time_ms: Option<u64>,
    pub auditor_time_ms: Option<u64>,
    pub peak_cpu_memory_bytes: Option<u64>,
    pub peak_gpu_memory_bytes: Option<u64>,
    pub artifact_cache_status: Option<CacheStatus>,
    pub real_auditor_cache_status: Option<CacheStatus>,
    pub ancillary_cache_status: Option<CacheStatus>,
    pub invalid_rows: usize,
    pub schema_violations: usize,
    pub nondeterministic_output: bool,
}

impl DeepOutcome {
    pub fn validate(&self) -> Result<()> {
        if (self.state == EvidenceState::Succeeded)
            != (self.metrics.is_some()
                && self.artifact_bytes.is_some()
                && self.runtime_ms.is_some()
                && self.fitting_time_ms.is_some()
                && self.sampling_time_ms.is_some()
                && self.auditor_time_ms.is_some()
                && self.peak_cpu_memory_bytes.is_some()
                && self.artifact_cache_status.is_some()
                && self.real_auditor_cache_status.is_some()
                && self.ancillary_cache_status.is_some()
                && self.failure_reason.is_none())
            || (self.state != EvidenceState::Succeeded
                && self.failure_reason.as_deref().is_none_or(str::is_empty))
            || self
                .metrics
                .as_ref()
                .is_some_and(|metrics| !metrics.finite())
            || self.artifact_bytes.is_some_and(|bytes| bytes == 0)
            || !matches!(self.task.as_str(), "binary" | "regression")
        {
            return Err(DopeError::Data(
                "deep outcome has an inconsistent measured/failed state".into(),
            ));
        }
        Ok(())
    }
}

fn bounded_retention(evidence: &JobEvidence) -> f64 {
    let improvement = evidence.null_loss - evidence.trtr_loss;
    if improvement >= 0.01 * evidence.null_loss.abs() && improvement.abs() > f64::EPSILON {
        ((evidence.null_loss - evidence.tstr_loss) / improvement).clamp(0.0, 1.0)
    } else {
        f64::from(evidence.tstr_loss <= evidence.trtr_loss + 0.01 * evidence.null_loss.abs())
    }
}

fn failure_state(error: &str) -> EvidenceState {
    let error = error.to_ascii_lowercase();
    if error.contains("timeout") || error.contains("deadline") {
        EvidenceState::TimedOut
    } else if error.contains("out of memory") || error.contains("oom") {
        EvidenceState::Oom
    } else if error.contains("malformed") || error.contains("schema") || error.contains("corrupt") {
        EvidenceState::Malformed
    } else {
        EvidenceState::Failed
    }
}

fn block_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for path in paths {
        if path.is_dir() {
            for entry in fs::read_dir(path).map_err(|error| crate::error::io_error(path, error))? {
                let entry = entry.map_err(|error| crate::error::io_error(path, error))?;
                let candidate = entry.path();
                if candidate.extension().and_then(|value| value.to_str()) == Some("json") {
                    files.push(candidate);
                }
            }
        } else {
            files.push(path.clone());
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

type DeepOutcomeKey = (String, String, String, String, String, usize, u64, u64);
type DeepOutcomeMap = BTreeMap<DeepOutcomeKey, DeepOutcome>;

/// Reconstructs seed-addressed deep evidence from durable gold blocks. Failed
/// cells retain their exact matrix coordinates and every derived comparison is
/// computed from measured peers in the same cell.
pub fn rebuild_deep_outcomes(
    cohort: &CohortPlan,
    block_paths: &[PathBuf],
) -> Result<Vec<DeepOutcome>> {
    let records = cohort
        .records
        .iter()
        .map(|record| (record.dataset_id.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    let mut outcomes = DeepOutcomeMap::new();
    for path in block_files(block_paths)? {
        let block: GoldRecordBlock = read_json(&path)?;
        if block.format != "dope-gold-record-block" || block.version != 2 {
            return Err(DopeError::Data(format!(
                "invalid gold block for deep evidence: {}",
                path.display()
            )));
        }
        let record = records
            .get(block.dataset_id.as_str())
            .ok_or_else(|| DopeError::Data("gold block dataset is absent from cohort".into()))?;
        let configuration = LossConfiguration {
            target_weight: block.neural_target_weight,
            structural_penalty: block.neural_structural_penalty,
        };
        if !LossConfiguration::GRID.contains(&configuration) {
            return Err(DopeError::Data(
                "gold block contains an unfrozen neural loss configuration".into(),
            ));
        }
        for evidence in block.evidence {
            if !(2..=JOB_EVIDENCE_VERSION).contains(&evidence.version) {
                return Err(DopeError::Data(
                    "gold block mixes superseded job evidence with the neural campaign".into(),
                ));
            }
            let is_new = NEW_CANDIDATES.contains(&evidence.candidate_id.as_str());
            let metrics = DeepMetrics {
                null_loss: Some(evidence.null_loss),
                trtr_loss: Some(evidence.trtr_loss),
                tstr_loss: Some(evidence.tstr_loss),
                bounded_retention: Some(bounded_retention(&evidence)),
                absolute_lift: None,
                regret: None,
                oracle_recall: None,
                calibration_degradation: evidence.calibration_degradation,
                query_p95_error: evidence.query_p95_normalized_error,
                type_i_error: evidence.type_i_error,
                membership_auc: evidence.membership_auc,
                attribute_inference_advantage: evidence.attribute_inference_advantage,
                driver_agreement: evidence.driver_agreement,
                joint_fidelity: evidence.joint_fidelity,
                rare_tail_retention: evidence.rare_class_or_tail_retention,
                supported_subgroup_retention: evidence.supported_subgroup_retention,
                nominal_coverage: evidence.nominal_95_coverage,
                feature_importance_spearman: evidence.feature_importance_spearman,
                feature_importance_top_k_agreement: Some(
                    evidence.feature_importance_top_k_agreement,
                ),
                feature_importance_informative_groups: Some(
                    evidence.feature_importance_informative_count as f64,
                ),
                exact_copies: Some(evidence.exact_copies as u64),
                near_copies: Some(evidence.near_copies as u64),
                canary_extractions: Some(evidence.canary_extractions as u64),
                lineage_leaks: Some(evidence.lineage_leaks as u64),
            };
            let outcome = DeepOutcome {
                task: evidence.task,
                lineage_group_id: evidence.lineage_group_id,
                structural_profile: evidence.structural_profile,
                candidate_id: evidence.candidate_id,
                configuration: is_new.then_some(configuration),
                generation_seed: evidence.generation_seed,
                auditor_seed: evidence.auditor_seed,
                auditor_id: evidence.auditor_id,
                size_multiplier: evidence.size_multiplier,
                state: EvidenceState::Succeeded,
                metrics: Some(metrics),
                failure_reason: None,
                artifact_bytes: Some(evidence.artifact_bytes),
                runtime_ms: Some(evidence.runtime_ms),
                fitting_time_ms: Some(evidence.fitting_time_ms),
                sampling_time_ms: Some(evidence.sampling_time_ms),
                auditor_time_ms: Some(evidence.auditor_time_ms),
                peak_cpu_memory_bytes: Some(evidence.peak_cpu_memory_bytes),
                peak_gpu_memory_bytes: evidence.peak_gpu_memory_bytes,
                artifact_cache_status: Some(evidence.artifact_cache_status),
                real_auditor_cache_status: Some(evidence.real_auditor_cache_status),
                ancillary_cache_status: Some(evidence.ancillary_cache_status),
                invalid_rows: evidence.invalid_rows,
                schema_violations: evidence.schema_violations,
                nondeterministic_output: evidence.nondeterministic_output,
            };
            insert_deep_outcome(&mut outcomes, outcome)?;
        }
        for failure in block.failures {
            let size_multiplier = failure.size_multiplier.ok_or_else(|| {
                DopeError::Data("gold failure is missing its size multiplier".into())
            })?;
            let generation_seed = failure.generation_seed.ok_or_else(|| {
                DopeError::Data("gold failure is missing its generation seed".into())
            })?;
            let auditor_seed = failure.auditor_seed.ok_or_else(|| {
                DopeError::Data("gold failure is missing its auditor seed".into())
            })?;
            let is_new = NEW_CANDIDATES.contains(&failure.candidate_id.as_str());
            let outcome = DeepOutcome {
                task: record.task.clone(),
                lineage_group_id: record.lineage_group_id.clone(),
                structural_profile: record.structural_profile.clone(),
                candidate_id: failure.candidate_id,
                configuration: is_new.then_some(configuration),
                generation_seed,
                auditor_seed,
                auditor_id: failure.auditor_id,
                size_multiplier,
                state: failure_state(&failure.error),
                metrics: None,
                failure_reason: Some(failure.error),
                artifact_bytes: None,
                runtime_ms: None,
                fitting_time_ms: None,
                sampling_time_ms: None,
                auditor_time_ms: None,
                peak_cpu_memory_bytes: None,
                peak_gpu_memory_bytes: None,
                artifact_cache_status: None,
                real_auditor_cache_status: None,
                ancillary_cache_status: None,
                invalid_rows: 0,
                schema_violations: 0,
                nondeterministic_output: false,
            };
            insert_deep_outcome(&mut outcomes, outcome)?;
        }
    }
    let mut outcomes = outcomes.into_values().collect::<Vec<_>>();
    add_cell_comparisons(&mut outcomes);
    outcomes.sort_by(|left, right| {
        left.lineage_group_id
            .cmp(&right.lineage_group_id)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
            .then_with(|| left.auditor_id.cmp(&right.auditor_id))
            .then_with(|| left.size_multiplier.cmp(&right.size_multiplier))
            .then_with(|| left.generation_seed.cmp(&right.generation_seed))
            .then_with(|| left.auditor_seed.cmp(&right.auditor_seed))
    });
    Ok(outcomes)
}

fn insert_deep_outcome(outcomes: &mut DeepOutcomeMap, outcome: DeepOutcome) -> Result<()> {
    let configuration = outcome
        .configuration
        .map(LossConfiguration::id)
        .unwrap_or_default();
    let key = (
        outcome.lineage_group_id.clone(),
        outcome.structural_profile.clone(),
        outcome.candidate_id.clone(),
        configuration,
        outcome.auditor_id.clone(),
        outcome.size_multiplier,
        outcome.generation_seed,
        outcome.auditor_seed,
    );
    if let Some(existing) = outcomes.insert(key, outcome.clone())
        && existing != outcome
    {
        return Err(DopeError::Data(
            "duplicate gold cells contain inconsistent deep evidence".into(),
        ));
    }
    Ok(())
}

fn add_cell_comparisons(outcomes: &mut [DeepOutcome]) {
    let measured = outcomes
        .iter()
        .filter_map(|outcome| {
            outcome
                .metrics
                .as_ref()
                .and_then(|metrics| metrics.bounded_retention)
                .map(|retention| {
                    (
                        (
                            outcome.lineage_group_id.clone(),
                            outcome.structural_profile.clone(),
                            outcome.auditor_id.clone(),
                            outcome.size_multiplier,
                            outcome.generation_seed,
                            outcome.auditor_seed,
                            outcome.candidate_id.clone(),
                        ),
                        retention,
                    )
                })
        })
        .collect::<BTreeMap<_, _>>();
    for outcome in outcomes {
        let Some(metrics) = outcome.metrics.as_mut() else {
            continue;
        };
        let cell = |candidate: &str| {
            measured.get(&(
                outcome.lineage_group_id.clone(),
                outcome.structural_profile.clone(),
                outcome.auditor_id.clone(),
                outcome.size_multiplier,
                outcome.generation_seed,
                outcome.auditor_seed,
                candidate.to_string(),
            ))
        };
        let Some(retention) = metrics.bounded_retention else {
            continue;
        };
        metrics.absolute_lift = cell(LEGACY_DEEP_CHAMPION).map(|champion| retention - champion);
        let oracle = NEW_CANDIDATES
            .into_iter()
            .chain(LEGACY_DEEP_CANDIDATES)
            .chain(REFERENCE_FRONTIER)
            .filter_map(cell)
            .copied()
            .reduce(f64::max);
        metrics.regret = oracle.map(|oracle| oracle - retention);
        metrics.oracle_recall = oracle.map(|oracle| f64::from(retention >= oracle - 1e-12));
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct PairedInterval {
    pub groups: usize,
    pub mean: f64,
    pub lower_one_sided_95: f64,
    pub upper_one_sided_95: f64,
}

fn paired_interval(values: &[f64]) -> Option<PairedInterval> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() == 1 {
        return Some(PairedInterval {
            groups: 1,
            mean,
            lower_one_sided_95: f64::NEG_INFINITY,
            upper_one_sided_95: f64::INFINITY,
        });
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let critical = StudentsT::new(0.0, 1.0, (values.len() - 1) as f64)
        .ok()?
        .inverse_cdf(0.95);
    let margin = critical * (variance / values.len() as f64).sqrt();
    Some(PairedInterval {
        groups: values.len(),
        mean,
        lower_one_sided_95: mean - margin,
        upper_one_sided_95: mean + margin,
    })
}

#[derive(Clone, Debug)]
struct AveragedOutcome {
    profile: String,
    candidate: String,
    metrics: DeepMetrics,
}

fn averaged_outcomes(
    outcomes: &[DeepOutcome],
) -> Result<BTreeMap<(String, String, String), AveragedOutcome>> {
    let mut sums =
        BTreeMap::<(String, String, String), (DeepMetrics, usize, Option<LossConfiguration>)>::new(
        );
    for outcome in outcomes {
        outcome.validate()?;
        if let Some(metrics) = &outcome.metrics {
            let key = (
                outcome.lineage_group_id.clone(),
                outcome.structural_profile.clone(),
                outcome.candidate_id.clone(),
            );
            match sums.entry(key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((metrics.clone(), 1, outcome.configuration));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if entry.get().2 != outcome.configuration {
                        return Err(DopeError::Data(
                            "candidate evidence mixes discovery configurations".into(),
                        ));
                    }
                    entry.get_mut().0.add_assign(metrics);
                    entry.get_mut().1 += 1;
                }
            }
        }
    }
    Ok(sums
        .into_iter()
        .map(|(key, (mut metrics, count, _configuration))| {
            metrics.divide(count as f64);
            (
                key,
                AveragedOutcome {
                    profile: String::new(),
                    candidate: String::new(),
                    metrics,
                },
            )
        })
        .map(|((lineage, profile, candidate), mut value)| {
            value.profile = profile.clone();
            value.candidate = candidate.clone();
            ((lineage, profile, candidate), value)
        })
        .collect())
}

fn metric_differences(
    averaged: &BTreeMap<(String, String, String), AveragedOutcome>,
    candidate: &str,
    metric: impl Fn(&DeepMetrics) -> Option<f64>,
) -> Vec<f64> {
    averaged
        .iter()
        .filter(|((_, _, id), _)| id == candidate)
        .filter_map(|((lineage, profile, _), value)| {
            averaged
                .get(&(
                    lineage.clone(),
                    profile.clone(),
                    LEGACY_DEEP_CHAMPION.into(),
                ))
                .and_then(|champion| {
                    metric(&value.metrics)
                        .zip(metric(&champion.metrics))
                        .map(|(candidate, champion)| candidate - champion)
                })
        })
        .collect()
}

fn safeguards_noninferior(
    averaged: &BTreeMap<(String, String, String), AveragedOutcome>,
    candidate: &str,
) -> bool {
    let upper_bounded = [
        |m: &DeepMetrics| m.calibration_degradation,
        |m: &DeepMetrics| m.query_p95_error,
        |m: &DeepMetrics| m.type_i_error,
        |m: &DeepMetrics| m.membership_auc,
        |m: &DeepMetrics| m.attribute_inference_advantage,
    ]
    .iter()
    .all(|metric| {
        paired_interval(&metric_differences(averaged, candidate, metric))
            .is_some_and(|interval| interval.upper_one_sided_95 <= 0.01)
    });
    let lower_bounded = [
        |m: &DeepMetrics| m.driver_agreement,
        |m: &DeepMetrics| m.joint_fidelity,
        |m: &DeepMetrics| m.rare_tail_retention,
        |m: &DeepMetrics| m.supported_subgroup_retention,
        |m: &DeepMetrics| m.feature_importance_spearman,
        |m: &DeepMetrics| m.feature_importance_top_k_agreement,
    ]
    .iter()
    .all(|metric| {
        paired_interval(&metric_differences(averaged, candidate, metric))
            .is_some_and(|interval| interval.lower_one_sided_95 >= -0.01)
    });
    let counts_safe = averaged
        .iter()
        .filter(|((_, _, id), _)| id == candidate)
        .all(|((lineage, profile, _), value)| {
            averaged
                .get(&(
                    lineage.clone(),
                    profile.clone(),
                    LEGACY_DEEP_CHAMPION.into(),
                ))
                .is_some_and(|champion| {
                    value
                        .metrics
                        .exact_copies
                        .zip(champion.metrics.exact_copies)
                        .is_some_and(|(candidate, champion)| candidate <= champion)
                        && value
                            .metrics
                            .near_copies
                            .zip(champion.metrics.near_copies)
                            .is_some_and(|(candidate, champion)| candidate <= champion)
                        && value
                            .metrics
                            .canary_extractions
                            .zip(champion.metrics.canary_extractions)
                            .is_some_and(|(candidate, champion)| candidate <= champion)
                        && value
                            .metrics
                            .lineage_leaks
                            .zip(champion.metrics.lineage_leaks)
                            .is_some_and(|(candidate, champion)| candidate <= champion)
                        && value.metrics.canary_extractions == Some(0)
                        && value.metrics.lineage_leaks == Some(0)
                })
        });
    upper_bounded && lower_bounded && counts_safe
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DiscoverySelection {
    pub candidate_id: String,
    pub implementation_hash: String,
    pub configuration: Option<LossConfiguration>,
    pub macro_retention: Option<f64>,
    pub safe: bool,
    pub failure: Option<String>,
}

pub fn select_discovery_configurations(
    outcomes: &[DeepOutcome],
) -> Result<Vec<DiscoverySelection>> {
    let mut selections = Vec::new();
    for architecture in NEW_CANDIDATES {
        let mut safe = Vec::new();
        for configuration in LossConfiguration::GRID {
            let candidate_cells = outcomes
                .iter()
                .filter(|outcome| {
                    outcome.candidate_id == architecture
                        && outcome.configuration == Some(configuration)
                })
                .collect::<Vec<_>>();
            let champion_cells = outcomes
                .iter()
                .filter(|outcome| outcome.candidate_id == LEGACY_DEEP_CHAMPION)
                .collect::<Vec<_>>();
            if candidate_cells.is_empty()
                || candidate_cells.len() != champion_cells.len()
                || candidate_cells
                    .iter()
                    .any(|outcome| outcome.state != EvidenceState::Succeeded)
                || champion_cells
                    .iter()
                    .any(|outcome| outcome.state != EvidenceState::Succeeded)
            {
                continue;
            }
            let filtered = outcomes
                .iter()
                .filter(|outcome| {
                    outcome.candidate_id == architecture
                        && outcome.configuration == Some(configuration)
                        || outcome.candidate_id == LEGACY_DEEP_CHAMPION
                })
                .cloned()
                .collect::<Vec<_>>();
            let averaged = averaged_outcomes(&filtered)?;
            if safeguards_noninferior(&averaged, architecture) {
                let profile_means = averaged
                    .iter()
                    .filter(|((_, _, candidate), _)| candidate == architecture)
                    .fold(BTreeMap::<&str, Vec<f64>>::new(), |mut map, (_, value)| {
                        if let Some(retention) = value.metrics.bounded_retention {
                            map.entry(&value.profile).or_default().push(retention);
                        }
                        map
                    });
                if !profile_means.is_empty() {
                    let macro_retention = profile_means
                        .values()
                        .map(|values| values.iter().sum::<f64>() / values.len() as f64)
                        .sum::<f64>()
                        / profile_means.len() as f64;
                    let artifact = filtered
                        .iter()
                        .filter(|outcome| outcome.candidate_id == architecture)
                        .filter_map(|outcome| outcome.artifact_bytes)
                        .max()
                        .unwrap_or(u64::MAX);
                    let runtime = filtered
                        .iter()
                        .filter(|outcome| outcome.candidate_id == architecture)
                        .filter_map(|outcome| outcome.runtime_ms)
                        .max()
                        .unwrap_or(u64::MAX);
                    safe.push((configuration, macro_retention, artifact, runtime));
                }
            }
        }
        safe.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.2.cmp(&right.2))
                .then_with(|| left.3.cmp(&right.3))
                .then_with(|| left.0.id().cmp(&right.0.id()))
        });
        selections.push(
            if let Some((configuration, retention, _, _)) = safe.first() {
                DiscoverySelection {
                    candidate_id: architecture.into(),
                    implementation_hash: candidate_implementation_hash(architecture),
                    configuration: Some(*configuration),
                    macro_retention: Some(*retention),
                    safe: true,
                    failure: None,
                }
            } else {
                DiscoverySelection {
                    candidate_id: architecture.into(),
                    implementation_hash: candidate_implementation_hash(architecture),
                    configuration: None,
                    macro_retention: None,
                    safe: false,
                    failure: Some("no loss configuration passed safeguard noninferiority".into()),
                }
            },
        );
    }
    Ok(selections)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Qualification {
    pub candidate_id: String,
    pub implementation_hash: String,
    pub qualifies: bool,
    pub overall_lift: Option<PairedInterval>,
    pub oracle_lift: Option<f64>,
    pub success_coverage: f64,
    pub minimum_profile_coverage: f64,
    pub feature_importance_coverage: f64,
    pub feature_importance_spearman_delta: Option<PairedInterval>,
    pub feature_importance_top_k_delta: Option<PairedInterval>,
    pub maximum_artifact_bytes: Option<u64>,
    pub failed_gates: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FinalCandidateScore {
    pub candidate_id: String,
    pub implementation_hash: String,
    pub configuration: LossConfiguration,
    pub cohort_units: usize,
    pub mean_bounded_retention: f64,
    pub worst_profile_regret: f64,
    pub maximum_artifact_bytes: u64,
    pub mean_runtime_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FinalFamilySelection {
    pub format: String,
    pub version: u8,
    pub winner: Option<FinalCandidateScore>,
    pub eligible_candidates: Vec<FinalCandidateScore>,
    pub recommendation: String,
}

/// Selects only among validation-select candidates whose complete gate
/// decision is positive. Replications are averaged within the
/// `(structural_profile, lineage)` cohort unit before candidates are compared.
pub fn select_final_family(
    outcomes: &[DeepOutcome],
    qualifications: &[Qualification],
) -> Result<FinalFamilySelection> {
    let mut decisions = BTreeMap::new();
    for qualification in qualifications {
        if !NEW_CANDIDATES.contains(&qualification.candidate_id.as_str())
            || qualification.implementation_hash
                != candidate_implementation_hash(&qualification.candidate_id)
            || qualification.qualifies != qualification.failed_gates.is_empty()
            || decisions
                .insert(qualification.candidate_id.clone(), qualification)
                .is_some()
        {
            return Err(DopeError::Data(
                "invalid or duplicate final-family qualification".into(),
            ));
        }
    }
    let averaged = averaged_outcomes(outcomes)?;
    let mut eligible = Vec::new();
    for (candidate, qualification) in decisions {
        if !qualification.qualifies {
            continue;
        }
        let candidate_outcomes = outcomes
            .iter()
            .filter(|outcome| outcome.candidate_id == candidate)
            .collect::<Vec<_>>();
        let configurations = candidate_outcomes
            .iter()
            .filter_map(|outcome| outcome.configuration)
            .collect::<Vec<_>>();
        let unique_configurations = configurations
            .iter()
            .copied()
            .map(LossConfiguration::id)
            .collect::<BTreeSet<_>>();
        if configurations.len() != candidate_outcomes.len() || unique_configurations.len() != 1 {
            return Err(DopeError::Data(format!(
                "validation-select configuration is not frozen for {candidate}"
            )));
        }
        let configuration = configurations[0];
        let cohort_values = averaged
            .iter()
            .filter(|((_, _, id), _)| id == &candidate)
            .filter_map(|(_, outcome)| {
                outcome
                    .metrics
                    .bounded_retention
                    .zip(outcome.metrics.regret)
                    .map(|(retention, regret)| (outcome.profile.clone(), retention, regret))
            })
            .collect::<Vec<_>>();
        if cohort_values.is_empty() {
            return Err(DopeError::Data(format!(
                "qualified candidate {candidate} has no complete validation-select cohort units"
            )));
        }
        let mean_bounded_retention = cohort_values
            .iter()
            .map(|(_, retention, _)| retention)
            .sum::<f64>()
            / cohort_values.len() as f64;
        let mut profile_regret = BTreeMap::<&str, (f64, usize)>::new();
        for (profile, _, regret) in &cohort_values {
            let entry = profile_regret.entry(profile).or_default();
            entry.0 += regret;
            entry.1 += 1;
        }
        let worst_profile_regret = profile_regret
            .into_values()
            .map(|(sum, count)| sum / count as f64)
            .reduce(f64::max)
            .ok_or_else(|| DopeError::Data("missing final profile regret".into()))?;
        let maximum_artifact_bytes = candidate_outcomes
            .iter()
            .filter_map(|outcome| outcome.artifact_bytes)
            .max()
            .ok_or_else(|| DopeError::Data("missing final artifact measurement".into()))?;
        let mut runtime_groups = BTreeMap::<(&str, &str), (u128, usize)>::new();
        for outcome in &candidate_outcomes {
            let runtime = outcome
                .runtime_ms
                .ok_or_else(|| DopeError::Data("missing final runtime measurement".into()))?;
            let entry = runtime_groups
                .entry((&outcome.structural_profile, &outcome.lineage_group_id))
                .or_default();
            entry.0 += u128::from(runtime);
            entry.1 += 1;
        }
        let mean_runtime_ms = runtime_groups
            .values()
            .map(|(sum, count)| *sum as f64 / *count as f64)
            .sum::<f64>()
            / runtime_groups.len() as f64;
        eligible.push(FinalCandidateScore {
            candidate_id: candidate.clone(),
            implementation_hash: qualification.implementation_hash.clone(),
            configuration,
            cohort_units: cohort_values.len(),
            mean_bounded_retention,
            worst_profile_regret,
            maximum_artifact_bytes,
            mean_runtime_ms,
        });
    }
    eligible.sort_by(|left, right| {
        right
            .mean_bounded_retention
            .total_cmp(&left.mean_bounded_retention)
            .then_with(|| {
                left.worst_profile_regret
                    .total_cmp(&right.worst_profile_regret)
            })
            .then_with(|| {
                left.maximum_artifact_bytes
                    .cmp(&right.maximum_artifact_bytes)
            })
            .then_with(|| left.mean_runtime_ms.total_cmp(&right.mean_runtime_ms))
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    let winner = eligible.first().cloned();
    Ok(FinalFamilySelection {
        format: "dope-deep-final-family-selection".into(),
        version: 1,
        recommendation: winner.as_ref().map_or_else(
            || "retain_current_frontier".into(),
            |winner| format!("promote_dataset_specific_{}", winner.candidate_id),
        ),
        winner,
        eligible_candidates: eligible,
    })
}

fn one_sided_regression_pvalue(values: &[f64], margin: f64) -> f64 {
    if values.len() < 2 {
        return 1.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let error = (variance / values.len() as f64).sqrt();
    if error <= f64::EPSILON {
        return f64::from(mean <= margin);
    }
    let distribution =
        StudentsT::new(0.0, 1.0, (values.len() - 1) as f64).expect("positive degrees of freedom");
    1.0 - distribution.cdf((mean - margin) / error)
}

fn holm_rejects(mut pvalues: Vec<f64>, alpha: f64) -> bool {
    pvalues.sort_by(f64::total_cmp);
    pvalues
        .first()
        .is_some_and(|pvalue| *pvalue <= alpha / pvalues.len() as f64)
}

fn profile_regret_groups(
    outcomes: &[DeepOutcome],
    candidate: &str,
) -> BTreeMap<String, Vec<(String, f64)>> {
    let mut sums = BTreeMap::<(String, String, String, usize, String), (f64, usize)>::new();
    for outcome in outcomes {
        if !matches!(outcome.candidate_id.as_str(), id if id == candidate || id == LEGACY_DEEP_CHAMPION)
        {
            continue;
        }
        let Some(regret) = outcome.metrics.as_ref().and_then(|metrics| metrics.regret) else {
            continue;
        };
        let entry = sums
            .entry((
                outcome.structural_profile.clone(),
                outcome.lineage_group_id.clone(),
                outcome.auditor_id.clone(),
                outcome.size_multiplier,
                outcome.candidate_id.clone(),
            ))
            .or_default();
        entry.0 += regret;
        entry.1 += 1;
    }
    let means = sums
        .into_iter()
        .map(|(key, (sum, count))| (key, sum / count as f64))
        .collect::<BTreeMap<_, _>>();
    let mut groups = BTreeMap::<String, Vec<(String, f64)>>::new();
    for ((profile, lineage, auditor, size, id), value) in &means {
        if id != candidate {
            continue;
        }
        if let Some(champion) = means.get(&(
            profile.clone(),
            lineage.clone(),
            auditor.clone(),
            *size,
            LEGACY_DEEP_CHAMPION.into(),
        )) {
            groups
                .entry(profile.clone())
                .or_default()
                .push((lineage.clone(), value - champion));
        }
    }
    groups
}

pub fn qualify_confirmation_candidate(
    candidate: &str,
    outcomes: &[DeepOutcome],
    scheduled_cells: usize,
) -> Result<Qualification> {
    if !NEW_CANDIDATES.contains(&candidate) || scheduled_cells == 0 {
        return Err(DopeError::Data(
            "confirmation qualification requires a scheduled new candidate".into(),
        ));
    }
    for outcome in outcomes {
        outcome.validate()?;
    }
    let averaged = averaged_outcomes(outcomes)?;
    let lift = paired_interval(&metric_differences(&averaged, candidate, |metric| {
        metric.bounded_retention
    }));
    let mut cell_sums = BTreeMap::<(String, String, String, usize, String), (f64, usize)>::new();
    for outcome in outcomes {
        let Some(retention) = outcome
            .metrics
            .as_ref()
            .and_then(|metrics| metrics.bounded_retention)
        else {
            continue;
        };
        let entry = cell_sums
            .entry((
                outcome.lineage_group_id.clone(),
                outcome.structural_profile.clone(),
                outcome.auditor_id.clone(),
                outcome.size_multiplier,
                outcome.candidate_id.clone(),
            ))
            .or_default();
        entry.0 += retention;
        entry.1 += 1;
    }
    let cell_means = cell_sums
        .into_iter()
        .map(|(key, (sum, count))| (key, sum / count as f64))
        .collect::<BTreeMap<_, _>>();
    let mut oracle_by_lineage = BTreeMap::<(String, String), (f64, usize)>::new();
    for ((lineage, profile, auditor, size, id), retention) in &cell_means {
        if id != candidate {
            continue;
        }
        let oracle_values = LEGACY_DEEP_CANDIDATES
            .iter()
            .filter_map(|legacy| {
                cell_means.get(&(
                    lineage.clone(),
                    profile.clone(),
                    auditor.clone(),
                    *size,
                    (*legacy).into(),
                ))
            })
            .copied()
            .collect::<Vec<_>>();
        if oracle_values.len() == LEGACY_DEEP_CANDIDATES.len() {
            let oracle = oracle_values.into_iter().reduce(f64::max).unwrap();
            let entry = oracle_by_lineage
                .entry((profile.clone(), lineage.clone()))
                .or_default();
            entry.0 += retention - oracle;
            entry.1 += 1;
        }
    }
    let oracle_differences = oracle_by_lineage
        .into_values()
        .map(|(sum, count)| sum / count as f64)
        .collect::<Vec<_>>();
    let oracle_lift = (!oracle_differences.is_empty())
        .then(|| oracle_differences.iter().sum::<f64>() / oracle_differences.len() as f64);
    let candidate_outcomes = outcomes
        .iter()
        .filter(|outcome| outcome.candidate_id == candidate)
        .collect::<Vec<_>>();
    let frozen_configurations = candidate_outcomes
        .iter()
        .filter_map(|outcome| outcome.configuration.map(LossConfiguration::id))
        .collect::<BTreeSet<_>>();
    let successes = candidate_outcomes
        .iter()
        .filter(|outcome| outcome.state == EvidenceState::Succeeded)
        .count();
    let success_coverage = successes as f64 / scheduled_cells as f64;
    let mut profile_attempts = BTreeMap::<&str, (usize, usize)>::new();
    for outcome in &candidate_outcomes {
        let entry = profile_attempts
            .entry(&outcome.structural_profile)
            .or_default();
        entry.0 += 1;
        entry.1 += usize::from(outcome.state == EvidenceState::Succeeded);
    }
    let minimum_profile_coverage = profile_attempts
        .values()
        .map(|(attempted, succeeded)| *succeeded as f64 / *attempted as f64)
        .reduce(f64::min)
        .unwrap_or(0.0);
    let maximum_artifact_bytes = candidate_outcomes
        .iter()
        .filter_map(|outcome| outcome.artifact_bytes)
        .max();
    let feature_importance_groups = averaged
        .iter()
        .filter(|((_, _, id), _)| id == candidate)
        .count();
    let feature_importance_informative_groups = averaged
        .iter()
        .filter(|((_, _, id), value)| {
            id == candidate && value.metrics.feature_importance_spearman.is_some()
        })
        .count();
    let feature_importance_coverage =
        feature_importance_informative_groups as f64 / feature_importance_groups.max(1) as f64;
    let feature_importance_spearman_delta =
        paired_interval(&metric_differences(&averaged, candidate, |metric| {
            metric.feature_importance_spearman
        }));
    let feature_importance_top_k_delta =
        paired_interval(&metric_differences(&averaged, candidate, |metric| {
            metric.feature_importance_top_k_agreement
        }));
    let mut failed_gates = Vec::new();
    if frozen_configurations.len() != 1
        || candidate_outcomes
            .iter()
            .any(|outcome| outcome.configuration.is_none())
    {
        failed_gates.push("configuration_not_frozen".into());
    }
    if lift.is_none_or(|interval| interval.mean < 0.10 || interval.lower_one_sided_95 <= 0.0) {
        failed_gates.push("overall_retention_lift".into());
    }
    if oracle_lift.is_none_or(|value| value < 0.0) {
        failed_gates.push("legacy_deep_oracle_lift".into());
    }
    let mut regression_pvalues = Vec::new();
    for (profile, regret_groups) in profile_regret_groups(outcomes, candidate) {
        let regret_differences = regret_groups
            .iter()
            .map(|(_, difference)| *difference)
            .collect::<Vec<_>>();
        let informative = regret_differences
            .iter()
            .filter(|value| value.abs() > 1e-12)
            .count();
        if regret_differences.len() >= 100
            && informative >= 30
            && regret_differences.iter().sum::<f64>() / regret_differences.len() as f64 >= 0.0
        {
            failed_gates.push(format!("profile_regret_not_lower:{profile}"));
        }
        let mut clustered = BTreeMap::<String, (f64, usize)>::new();
        for (lineage, difference) in regret_groups {
            let entry = clustered.entry(lineage).or_default();
            entry.0 += difference;
            entry.1 += 1;
        }
        let clustered = clustered
            .into_values()
            .map(|(sum, count)| sum / count as f64)
            .collect::<Vec<_>>();
        if !clustered.is_empty() {
            regression_pvalues.push(one_sided_regression_pvalue(&clustered, 0.01));
        }
    }
    if holm_rejects(regression_pvalues, 0.05) {
        failed_gates.push("holm_supported_profile_regret_regression".into());
    }
    if candidate_outcomes.len() != scheduled_cells
        || success_coverage < 0.99
        || minimum_profile_coverage < 0.95
    {
        failed_gates.push("cell_coverage".into());
    }
    if maximum_artifact_bytes.is_none_or(|bytes| bytes > MAX_NEURAL_ARTIFACT_BYTES as u64) {
        failed_gates.push("pilot_artifact_size".into());
    }
    if candidate_outcomes.iter().any(|outcome| {
        outcome.invalid_rows > 0 || outcome.schema_violations > 0 || outcome.nondeterministic_output
    }) {
        failed_gates.push("validity_or_determinism".into());
    }
    if !safeguards_noninferior(&averaged, candidate) {
        failed_gates.push("privacy_or_fidelity_noninferiority".into());
    }
    if feature_importance_coverage < 0.80
        || feature_importance_spearman_delta
            .is_none_or(|interval| interval.lower_one_sided_95 < -0.01)
        || feature_importance_top_k_delta.is_none_or(|interval| interval.lower_one_sided_95 < -0.01)
    {
        failed_gates.push("feature_importance_consistency".into());
    }
    Ok(Qualification {
        candidate_id: candidate.into(),
        implementation_hash: candidate_implementation_hash(candidate),
        qualifies: failed_gates.is_empty(),
        overall_lift: lift,
        oracle_lift,
        success_coverage,
        minimum_profile_coverage,
        feature_importance_coverage,
        feature_importance_spearman_delta,
        feature_importance_top_k_delta,
        maximum_artifact_bytes,
        failed_gates,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DeepSliceReport {
    pub dimension: String,
    pub key: String,
    pub candidate_id: String,
    pub attempted_cells: usize,
    pub succeeded_cells: usize,
    pub failed_cells: usize,
    pub timed_out_cells: usize,
    pub lineage_groups: usize,
    pub metrics: DeepMetrics,
    pub bounded_retention_interval: Option<PairedInterval>,
    pub absolute_lift_interval: Option<PairedInterval>,
    pub mean_runtime_ms: Option<f64>,
    pub mean_fitting_time_ms: Option<f64>,
    pub mean_sampling_time_ms: Option<f64>,
    pub mean_auditor_time_ms: Option<f64>,
    pub peak_cpu_memory_bytes: Option<u64>,
    pub peak_gpu_memory_bytes: Option<u64>,
    pub maximum_artifact_bytes: Option<u64>,
    pub artifact_cache_statuses: BTreeMap<String, usize>,
    pub real_auditor_cache_statuses: BTreeMap<String, usize>,
    pub ancillary_cache_statuses: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DeepCampaignReport {
    pub format: String,
    pub version: u8,
    pub slices: Vec<DeepSliceReport>,
    pub failures: Vec<DeepOutcome>,
    pub paired_pareto_hypervolume: BTreeMap<String, Option<PairedInterval>>,
}

#[derive(Default)]
struct SliceAccumulator {
    attempted: usize,
    succeeded: usize,
    failed: usize,
    timed_out: usize,
    runtimes: Vec<u64>,
    fitting_times: Vec<u64>,
    sampling_times: Vec<u64>,
    auditor_times: Vec<u64>,
    cpu_memory: Vec<u64>,
    gpu_memory: Vec<u64>,
    artifact_bytes: Vec<u64>,
    artifact_cache_statuses: BTreeMap<String, usize>,
    real_auditor_cache_statuses: BTreeMap<String, usize>,
    ancillary_cache_statuses: BTreeMap<String, usize>,
}

/// Builds the publication slices without substituting zero for a missing
/// metric. Seed/auditor replications are averaged inside each lineage before
/// lineages receive equal weight.
pub fn build_deep_campaign_report(outcomes: &[DeepOutcome]) -> Result<DeepCampaignReport> {
    for candidate in NEW_CANDIDATES {
        let configurations = outcomes
            .iter()
            .filter(|outcome| outcome.candidate_id == candidate)
            .filter_map(|outcome| outcome.configuration.map(LossConfiguration::id))
            .collect::<BTreeSet<_>>();
        if configurations.len() > 1 {
            return Err(DopeError::Data(format!(
                "campaign report mixes frozen configurations for {candidate}"
            )));
        }
    }
    let overall_averaged = averaged_outcomes(outcomes)?;
    let mut cells = BTreeMap::<(String, String, String), SliceAccumulator>::new();
    let mut lineage_metrics =
        BTreeMap::<(String, String, String, String, String), (DeepMetrics, usize)>::new();
    let mut failures = Vec::new();
    for outcome in outcomes {
        outcome.validate()?;
        if outcome.state != EvidenceState::Succeeded {
            failures.push(outcome.clone());
        }
        let dimensions = [
            ("overall", "all".to_string()),
            ("task", outcome.task.clone()),
            ("profile", outcome.structural_profile.clone()),
            ("auditor", outcome.auditor_id.clone()),
            ("size", outcome.size_multiplier.to_string()),
        ];
        for (dimension, key) in dimensions {
            let cell_key = (
                dimension.to_string(),
                key.clone(),
                outcome.candidate_id.clone(),
            );
            let cell = cells.entry(cell_key.clone()).or_default();
            cell.attempted += 1;
            match outcome.state {
                EvidenceState::Succeeded => cell.succeeded += 1,
                EvidenceState::TimedOut => cell.timed_out += 1,
                _ => cell.failed += 1,
            }
            if let Some(value) = outcome.runtime_ms {
                cell.runtimes.push(value);
            }
            if let Some(value) = outcome.fitting_time_ms {
                cell.fitting_times.push(value);
            }
            if let Some(value) = outcome.sampling_time_ms {
                cell.sampling_times.push(value);
            }
            if let Some(value) = outcome.auditor_time_ms {
                cell.auditor_times.push(value);
            }
            if let Some(value) = outcome.peak_cpu_memory_bytes {
                cell.cpu_memory.push(value);
            }
            if let Some(value) = outcome.peak_gpu_memory_bytes {
                cell.gpu_memory.push(value);
            }
            if let Some(value) = outcome.artifact_bytes {
                cell.artifact_bytes.push(value);
            }
            let record_cache = |counts: &mut BTreeMap<String, usize>,
                                status: Option<CacheStatus>| {
                if let Some(status) = status {
                    *counts
                        .entry(format!("{status:?}").to_ascii_lowercase())
                        .or_default() += 1;
                }
            };
            record_cache(
                &mut cell.artifact_cache_statuses,
                outcome.artifact_cache_status,
            );
            record_cache(
                &mut cell.real_auditor_cache_statuses,
                outcome.real_auditor_cache_status,
            );
            record_cache(
                &mut cell.ancillary_cache_statuses,
                outcome.ancillary_cache_status,
            );
            if let Some(metrics) = &outcome.metrics {
                let lineage_key = (
                    cell_key.0,
                    cell_key.1,
                    cell_key.2,
                    outcome.structural_profile.clone(),
                    outcome.lineage_group_id.clone(),
                );
                match lineage_metrics.entry(lineage_key) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert((metrics.clone(), 1));
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        entry.get_mut().0.add_assign(metrics);
                        entry.get_mut().1 += 1;
                    }
                }
            }
        }
    }
    let mut by_slice = BTreeMap::<(String, String, String), Vec<DeepMetrics>>::new();
    for ((dimension, key, candidate, _, _), (mut metrics, count)) in lineage_metrics {
        metrics.divide(count as f64);
        by_slice
            .entry((dimension, key, candidate))
            .or_default()
            .push(metrics);
    }
    let mean = |values: &[u64]| {
        (!values.is_empty()).then(|| values.iter().sum::<u64>() as f64 / values.len() as f64)
    };
    let mut slices = Vec::new();
    for (key, cell) in cells {
        let lineage_values = by_slice.remove(&key).unwrap_or_default();
        let mut metrics = lineage_values.first().cloned().unwrap_or_default();
        for value in lineage_values.iter().skip(1) {
            metrics.add_assign(value);
        }
        if !lineage_values.is_empty() {
            metrics.divide(lineage_values.len() as f64);
        }
        let bounded_retention = lineage_values
            .iter()
            .filter_map(|metrics| metrics.bounded_retention)
            .collect::<Vec<_>>();
        let absolute_lift = lineage_values
            .iter()
            .filter_map(|metrics| metrics.absolute_lift)
            .collect::<Vec<_>>();
        slices.push(DeepSliceReport {
            dimension: key.0,
            key: key.1,
            candidate_id: key.2,
            attempted_cells: cell.attempted,
            succeeded_cells: cell.succeeded,
            failed_cells: cell.failed,
            timed_out_cells: cell.timed_out,
            lineage_groups: lineage_values.len(),
            metrics,
            bounded_retention_interval: paired_interval(&bounded_retention),
            absolute_lift_interval: paired_interval(&absolute_lift),
            mean_runtime_ms: mean(&cell.runtimes),
            mean_fitting_time_ms: mean(&cell.fitting_times),
            mean_sampling_time_ms: mean(&cell.sampling_times),
            mean_auditor_time_ms: mean(&cell.auditor_times),
            peak_cpu_memory_bytes: cell.cpu_memory.into_iter().max(),
            peak_gpu_memory_bytes: cell.gpu_memory.into_iter().max(),
            maximum_artifact_bytes: cell.artifact_bytes.into_iter().max(),
            artifact_cache_statuses: cell.artifact_cache_statuses,
            real_auditor_cache_statuses: cell.real_auditor_cache_statuses,
            ancillary_cache_statuses: cell.ancillary_cache_statuses,
        });
    }
    slices.sort_by(|left, right| {
        left.dimension
            .cmp(&right.dimension)
            .then_with(|| left.key.cmp(&right.key))
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    let hypervolume = |metrics: &DeepMetrics| {
        metrics
            .bounded_retention
            .zip(metrics.joint_fidelity)
            .zip(metrics.membership_auc)
            .map(|((retention, fidelity), membership)| {
                retention.clamp(0.0, 1.0)
                    * fidelity.clamp(0.0, 1.0)
                    * (1.0 - membership).clamp(0.0, 1.0)
            })
    };
    let paired_pareto_hypervolume = NEW_CANDIDATES
        .into_iter()
        .map(|candidate| {
            let differences = metric_differences(&overall_averaged, candidate, hypervolume);
            (candidate.to_string(), paired_interval(&differences))
        })
        .collect();
    Ok(DeepCampaignReport {
        format: "dope-deep-joint-campaign-report".into(),
        version: 1,
        slices,
        failures,
        paired_pareto_hypervolume,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DatasetArtifactEntry {
    pub dataset_id: String,
    pub file_name: String,
    pub bytes: u64,
    pub sha256: String,
    pub blake3: String,
    pub rows_fitted: u64,
    pub features: u32,
    pub training_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DeepArtifactManifest {
    pub format: String,
    pub version: u8,
    pub family: String,
    pub configuration: LossConfiguration,
    pub implementation_hash: String,
    pub compilation_scope: String,
    pub representative_dataset_id: String,
    pub artifacts: Vec<DatasetArtifactEntry>,
}

pub fn build_deep_artifact_manifest(
    selection: &FinalFamilySelection,
    artifacts: &[(String, PathBuf)],
    representative_dataset_id: &str,
) -> Result<DeepArtifactManifest> {
    let winner = selection.winner.as_ref().ok_or_else(|| {
        DopeError::Data("cannot publish deep artifacts without a passing final family".into())
    })?;
    if artifacts.is_empty() {
        return Err(DopeError::Data("deep artifact manifest is empty".into()));
    }
    let mut seen = BTreeSet::new();
    let mut entries = Vec::new();
    for (dataset_id, path) in artifacts {
        if dataset_id.is_empty() || !seen.insert(dataset_id.clone()) {
            return Err(DopeError::Data(
                "deep artifact manifest contains an invalid dataset ID".into(),
            ));
        }
        if path.extension().and_then(|value| value.to_str()) != Some("dpk") {
            return Err(DopeError::Data(
                "deep artifact manifest requires .dpk filenames".into(),
            ));
        }
        let bytes = fs::read(path).map_err(|error| crate::error::io_error(path, error))?;
        if &bytes[..bytes.len().min(6)] != b"DPK3\x03\x02" {
            return Err(DopeError::Data(format!(
                "deep artifact is not DPK3.2: {}",
                path.display()
            )));
        }
        let kernel = decode_kernel(&bytes)?;
        let KernelProgram::NeuralJoint(generator) = &kernel.program else {
            return Err(DopeError::Data(
                "deep artifact manifest contains a non-neural kernel".into(),
            ));
        };
        if generator.implementation_hash != winner.implementation_hash
            || generator.architecture
                != match winner.candidate_id.as_str() {
                    "tvae" => crate::model::NeuralArchitecture::Tvae,
                    "single_table_autoregressive_transformer" => {
                        crate::model::NeuralArchitecture::MaskedAutoregressiveTransformer
                    }
                    "tabsyn" => crate::model::NeuralArchitecture::TabSyn,
                    _ => {
                        return Err(DopeError::Data(
                            "final family is not a frozen neural candidate".into(),
                        ));
                    }
                }
        {
            return Err(DopeError::Data(
                "artifact implementation does not match the frozen winner".into(),
            ));
        }
        let content = hashes(&bytes);
        entries.push(DatasetArtifactEntry {
            dataset_id: dataset_id.clone(),
            file_name: path
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| DopeError::Data("artifact filename is not UTF-8".into()))?
                .into(),
            bytes: bytes.len() as u64,
            sha256: content.sha256,
            blake3: content.blake3,
            rows_fitted: kernel.rows_fitted,
            features: kernel.features,
            training_hash: generator.training_hash.clone(),
        });
    }
    entries.sort_by(|left, right| left.dataset_id.cmp(&right.dataset_id));
    if !seen.contains(representative_dataset_id) {
        return Err(DopeError::Data(
            "representative artifact is absent from the manifest".into(),
        ));
    }
    Ok(DeepArtifactManifest {
        format: "dope-deep-dpk-artifact-manifest".into(),
        version: 1,
        family: winner.candidate_id.clone(),
        configuration: winner.configuration,
        implementation_hash: winner.implementation_hash.clone(),
        compilation_scope: "one independent fit and DPK3.2 artifact per dataset".into(),
        representative_dataset_id: representative_dataset_id.into(),
        artifacts: entries,
    })
}

fn display_metric(value: Option<f64>) -> String {
    value.map_or_else(|| "missing".into(), |value| format!("{value:.6}"))
}

pub fn build_deep_model_card(
    selection: &FinalFamilySelection,
    report: &DeepCampaignReport,
) -> Result<String> {
    if report.format != "dope-deep-joint-campaign-report" {
        return Err(DopeError::Data("invalid deep campaign report".into()));
    }
    let (family, configuration, implementation, decision) = selection.winner.as_ref().map_or_else(
        || {
            (
                "none".to_string(),
                "none".to_string(),
                "none".to_string(),
                "No deep family passed every frozen gate; retain the current frontier.".to_string(),
            )
        },
        |winner| {
            (
                winner.candidate_id.clone(),
                winner.configuration.id(),
                winner.implementation_hash.clone(),
                format!(
                    "Promote {} only as a compiler for dataset-specific DPK3.2 artifacts.",
                    winner.candidate_id
                ),
            )
        },
    );
    let overall = selection.winner.as_ref().and_then(|winner| {
        report.slices.iter().find(|slice| {
            slice.dimension == "overall"
                && slice.key == "all"
                && slice.candidate_id == winner.candidate_id
        })
    });
    let metrics = overall.map(|slice| &slice.metrics);
    let metric = |field: fn(&DeepMetrics) -> Option<f64>| display_metric(metrics.and_then(field));
    let runtime = overall
        .and_then(|slice| slice.mean_runtime_ms)
        .map_or_else(|| "missing".into(), |value| format!("{value:.3} ms"));
    let artifact = overall
        .and_then(|slice| slice.maximum_artifact_bytes)
        .map_or_else(|| "missing".into(), |value| format!("{value} bytes"));
    Ok(format!(
        "# Deep joint generator model card\n\n\
         ## Decision\n\n{decision}\n\n\
         Family: `{family}`  \nConfiguration: `{configuration}`  \nImplementation hash: `{implementation}`\n\n\
         ## Architecture and data accounting\n\n\
         The frozen candidates are TVAE, a masked autoregressive transformer, and TabSyn. \
         Discovery contains 206 datasets and 108,523 rows; confirmation contains 601 disjoint \
         datasets and 409,686 rows; validation-select contains 11,892 lineages and 11,620,492 rows. \
         These rows are distributed across independent per-dataset fits. They are never pooled \
         into one cross-dataset network. The selected family, if any, compiles one DPK3.2 artifact \
         for each dataset and sampling remains pure Rust.\n\n\
         ## Synthetic-to-real transfer\n\n\
         Mean bounded retention: {}  \nAbsolute lift over legacy TVAE: {}  \nRegret: {}  \nOracle recall: {}\n\n\
         ## Feature-importance consistency\n\n\
         Permutation-importance Spearman: {}  \nTop-k agreement: {}  \nInformative feature groups: {}  \nUndefined Spearman values remain missing and are not replaced by zero.\n\n\
         ## Fidelity, privacy, and safeguards\n\n\
         Driver agreement: {}  \nJoint fidelity: {}  \nMembership AUC: {}  \nAttribute-inference advantage: {}  \nExact copies: {}  \nCanary extractions: {}\n\n\
         ## Operations\n\n\
         Mean per-cell runtime: {runtime}  \nMaximum artifact: {artifact}\n\n\
         ## Limitations\n\n\
         Promotion is valid only for the frozen source, binary, configuration, cohort split, \
         auditors, seeds, coverage rules, and artifact/parity gates. Validation-cert and sealed-test \
         are outside family selection. Missing, failed, timed-out, OOM, or inadequately informative \
         cells cannot be imputed as successful evidence.\n",
        metric(|m| m.bounded_retention),
        metric(|m| m.absolute_lift),
        metric(|m| m.regret),
        metric(|m| m.oracle_recall),
        metric(|m| m.feature_importance_spearman),
        metric(|m| m.feature_importance_top_k_agreement),
        metric(|m| m.feature_importance_informative_groups),
        metric(|m| m.driver_agreement),
        metric(|m| m.joint_fidelity),
        metric(|m| m.membership_auc),
        metric(|m| m.attribute_inference_advantage),
        metrics
            .and_then(|m| m.exact_copies)
            .map_or_else(|| "missing".into(), |v| v.to_string()),
        metrics
            .and_then(|m| m.canary_extractions)
            .map_or_else(|| "missing".into(), |v| v.to_string()),
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpansionDecision {
    None,
    Ctgan,
    Tabddpm,
    PublishFailedFrontier,
}

pub fn bounded_expansion(
    maximum_safe_discovery_lift: f64,
    largest_deficit: &str,
) -> ExpansionDecision {
    if maximum_safe_discovery_lift < 0.05 {
        return ExpansionDecision::None;
    }
    match largest_deficit {
        "joint_fidelity" | "query_error" => ExpansionDecision::Ctgan,
        "rare_tail_retention" | "multimodal_coverage" => ExpansionDecision::Tabddpm,
        "privacy" | "copying" | "calibration" => ExpansionDecision::PublishFailedFrontier,
        _ => ExpansionDecision::None,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CompressionEvidence {
    pub artifact_bytes: u64,
    pub retention_before: f64,
    pub retention_after: f64,
    pub profile_conclusions_unchanged: bool,
    pub safeguard_gates_pass: bool,
    pub deterministic_draws: usize,
    pub maximum_libtorch_rust_difference: f64,
    pub discrete_decision_agreement: f64,
}

impl CompressionEvidence {
    pub fn validate(&self) -> Result<()> {
        if self.artifact_bytes > ROUTER_ARTIFACT_LIMIT_BYTES
            || !self.retention_before.is_finite()
            || !self.retention_after.is_finite()
            || self.retention_before - self.retention_after > 0.01
            || !self.profile_conclusions_unchanged
            || !self.safeguard_gates_pass
            || self.deterministic_draws != 100_000
            || !self.maximum_libtorch_rust_difference.is_finite()
            || self.maximum_libtorch_rust_difference > 1e-3
            || !self.discrete_decision_agreement.is_finite()
            || self.discrete_decision_agreement < 0.999
        {
            return Err(DopeError::Data(
                "compressed joint generator does not satisfy router handoff gates".into(),
            ));
        }
        Ok(())
    }
}

/// Produces the exact router-handoff bytes once the already-int8 joint model
/// fits the compressed ceiling. Models over the ceiling remain failed evidence
/// until an offline structured-pruning pass emits a new frozen kernel.
pub fn prepare_router_artifact(kernel: &Kernel) -> Result<Vec<u8>> {
    if !matches!(kernel.program, KernelProgram::NeuralJoint(_)) {
        return Err(DopeError::Data(
            "router deep handoff requires a neural joint kernel".into(),
        ));
    }
    let artifact = encode_kernel(kernel)?;
    if artifact.len() as u64 > ROUTER_ARTIFACT_LIMIT_BYTES {
        return Err(DopeError::Data(
            "joint artifact still exceeds 4 MiB after int8 export; structured pruning is required"
                .into(),
        ));
    }
    Ok(artifact)
}

const COMPRESSION_SCHEDULE: [(f64, usize); 4] = [(0.10, 129), (0.20, 65), (0.35, 33), (0.50, 17)];

fn sparsify_output_channels(layer: &mut QuantizedLinear, fraction: f64) {
    let input = layer.input_dim as usize;
    let output = layer.output_dim as usize;
    let prune = ((output as f64 * fraction).floor() as usize).min(output.saturating_sub(1));
    let mut channels = (0..output)
        .map(|channel| {
            let salience = layer.weights[channel * input..(channel + 1) * input]
                .iter()
                .map(|weight| f64::from(weight.unsigned_abs()))
                .sum::<f64>()
                * f64::from(layer.scales[channel])
                + f64::from(layer.biases[channel].abs());
            (salience, channel)
        })
        .collect::<Vec<_>>();
    channels.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    for (_, channel) in channels.into_iter().take(prune) {
        layer.weights[channel * input..(channel + 1) * input].fill(0);
        layer.biases[channel] = 0.0;
    }
}

fn reduce_marginal_knots(marginal: &mut Marginal, maximum: usize) {
    match marginal {
        Marginal::QuantileSpline { values } if values.len() > maximum => {
            let last = values.len() - 1;
            *values = (0..maximum)
                .map(|index| values[index * last / (maximum - 1)])
                .collect();
        }
        Marginal::ZeroInflated { base, .. } => reduce_marginal_knots(base, maximum),
        _ => {}
    }
}

fn compressed_kernel(kernel: &Kernel, fraction: f64, maximum_knots: usize) -> Kernel {
    let mut compressed = kernel.clone();
    for marginal in &mut compressed.marginals {
        reduce_marginal_knots(marginal, maximum_knots);
    }
    let KernelProgram::NeuralJoint(generator) = &mut compressed.program else {
        return compressed;
    };
    reduce_marginal_knots(&mut generator.target_marginal, maximum_knots);
    match &mut generator.network {
        JointNetwork::Tvae { decoder } => {
            sparsify_output_channels(&mut decoder.hidden_1, fraction);
            sparsify_output_channels(&mut decoder.hidden_2, fraction);
        }
        JointNetwork::MaskedAutoregressiveTransformer { transformer } => {
            for block in &mut transformer.blocks {
                sparsify_output_channels(&mut block.feed_forward_1, fraction);
            }
        }
        JointNetwork::TabSyn {
            decoder, denoiser, ..
        } => {
            sparsify_output_channels(&mut decoder.hidden_1, fraction);
            sparsify_output_channels(&mut decoder.hidden_2, fraction);
            sparsify_output_channels(&mut denoiser.hidden_1, fraction);
            sparsify_output_channels(&mut denoiser.hidden_2, fraction);
            sparsify_output_channels(&mut denoiser.hidden_3, fraction);
        }
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"joint-router-compression-v1");
    hasher.update(generator.training_hash.as_bytes());
    hasher.update(&fraction.to_bits().to_le_bytes());
    hasher.update(&(maximum_knots as u64).to_le_bytes());
    generator.training_hash = hasher.finalize().to_hex().to_string();
    compressed
}

/// Applies the frozen train-only compression schedule. The first artifact that
/// reaches 4 MiB is returned only when independent compression/parity evidence
/// satisfies every frozen gate and reconciles to the exact byte count.
pub fn prepare_router_artifact_with_compression(
    kernel: &Kernel,
    evidence: &CompressionEvidence,
) -> Result<Vec<u8>> {
    if !matches!(kernel.program, KernelProgram::NeuralJoint(_)) {
        return Err(DopeError::Data(
            "router deep compression requires a neural joint kernel".into(),
        ));
    }
    let artifact = prepare_router_artifact(kernel);
    if let Ok(artifact) = artifact {
        return Ok(artifact);
    }
    evidence.validate()?;
    for (fraction, maximum_knots) in COMPRESSION_SCHEDULE {
        let compressed = compressed_kernel(kernel, fraction, maximum_knots);
        compressed.validate().map_err(DopeError::Data)?;
        let artifact = encode_kernel(&compressed)?;
        if artifact.len() as u64 <= ROUTER_ARTIFACT_LIMIT_BYTES {
            if artifact.len() as u64 != evidence.artifact_bytes {
                return Err(DopeError::Data(
                    "compression evidence does not reconcile to the emitted artifact".into(),
                ));
            }
            return Ok(artifact);
        }
    }
    Err(DopeError::Data(
        "joint artifact exceeds 4 MiB after the complete frozen compression schedule".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn record(profile: &str, lineage: usize) -> CohortRecord {
        CohortRecord {
            dataset_id: format!("d{lineage}"),
            dataset_path: PathBuf::from(format!("/{lineage}")),
            task: "regression".into(),
            rows: 128,
            features: 8,
            lineage_group_id: format!("{profile}-l{lineage}"),
            structural_profile: profile.into(),
            partition: "training_gold".into(),
        }
    }

    fn final_outcome(
        candidate: &str,
        profile: &str,
        retention: f64,
        regret: f64,
        artifact_bytes: u64,
        runtime_ms: u64,
    ) -> DeepOutcome {
        DeepOutcome {
            task: "regression".into(),
            lineage_group_id: "shared-lineage".into(),
            structural_profile: profile.into(),
            candidate_id: candidate.into(),
            configuration: Some(LossConfiguration::GRID[0]),
            generation_seed: 1_829,
            auditor_seed: 57_721,
            auditor_id: "elastic_net_glm".into(),
            size_multiplier: 1,
            state: EvidenceState::Succeeded,
            metrics: Some(DeepMetrics {
                bounded_retention: Some(retention),
                regret: Some(regret),
                ..Default::default()
            }),
            failure_reason: None,
            artifact_bytes: Some(artifact_bytes),
            runtime_ms: Some(runtime_ms),
            fitting_time_ms: Some(10),
            sampling_time_ms: Some(5),
            auditor_time_ms: Some(runtime_ms.saturating_sub(15)),
            peak_cpu_memory_bytes: Some(1_024),
            peak_gpu_memory_bytes: None,
            artifact_cache_status: Some(CacheStatus::Miss),
            real_auditor_cache_status: Some(CacheStatus::Miss),
            ancillary_cache_status: Some(CacheStatus::Disabled),
            invalid_rows: 0,
            schema_violations: 0,
            nondeterministic_output: false,
        }
    }

    fn passing(candidate: &str) -> Qualification {
        Qualification {
            candidate_id: candidate.into(),
            implementation_hash: candidate_implementation_hash(candidate),
            qualifies: true,
            overall_lift: None,
            oracle_lift: Some(0.0),
            success_coverage: 1.0,
            minimum_profile_coverage: 1.0,
            feature_importance_coverage: 1.0,
            feature_importance_spearman_delta: None,
            feature_importance_top_k_delta: None,
            maximum_artifact_bytes: Some(1_024),
            failed_gates: Vec::new(),
        }
    }

    #[test]
    fn cohorts_take_first_eight_then_next_thirty_two_without_overlap() {
        let plan = CohortPlan {
            format: "dope-campaign-cohort".into(),
            version: 1,
            kind: "training-gold".into(),
            records: (0..100)
                .map(|lineage| record(if lineage < 50 { "a" } else { "b" }, lineage))
                .collect(),
            exclusions: Vec::new(),
        };
        let deep = plan_deep_cohorts(&plan).unwrap();
        assert_eq!(deep.discovery.len(), 16);
        assert_eq!(deep.confirmation.len(), 64);
        let discovery = deep
            .discovery
            .iter()
            .map(|record| &record.lineage_group_id)
            .collect::<BTreeSet<_>>();
        assert!(
            deep.confirmation
                .iter()
                .all(|record| !discovery.contains(&record.lineage_group_id))
        );
        assert_eq!(deep.confirmation_cells(14), 64 * 14 * 108);
        assert_eq!(
            EXPECTED_CONFIRMATION_LINEAGES * 14 * 108,
            MAX_CONFIRMATION_CELLS
        );
        assert_eq!(
            validation_select_cells(3).unwrap(),
            MAX_VALIDATION_SELECT_CELLS
        );
    }

    #[test]
    fn bounded_expansion_follows_the_frozen_deficit_rule() {
        assert_eq!(
            bounded_expansion(0.049, "query_error"),
            ExpansionDecision::None
        );
        assert_eq!(
            bounded_expansion(0.05, "query_error"),
            ExpansionDecision::Ctgan
        );
        assert_eq!(
            bounded_expansion(0.08, "rare_tail_retention"),
            ExpansionDecision::Tabddpm
        );
        assert_eq!(
            bounded_expansion(0.08, "privacy"),
            ExpansionDecision::PublishFailedFrontier
        );
    }

    #[test]
    fn final_selector_uses_profile_lineage_units_and_frozen_tiebreaks() {
        let outcomes = vec![
            final_outcome("tvae", "a", 0.8, 0.2, 1_000, 30),
            final_outcome("tvae", "b", 0.8, 0.2, 1_000, 30),
            final_outcome("tabsyn", "a", 0.8, 0.1, 2_000, 20),
            final_outcome("tabsyn", "b", 0.8, 0.1, 2_000, 20),
        ];
        let selection =
            select_final_family(&outcomes, &[passing("tvae"), passing("tabsyn")]).unwrap();
        let winner = selection.winner.unwrap();
        assert_eq!(winner.candidate_id, "tabsyn");
        assert_eq!(winner.cohort_units, 2);
    }

    #[test]
    fn final_selector_recommends_frontier_when_no_candidate_passes() {
        let selection = select_final_family(&[], &[]).unwrap();
        assert_eq!(selection.winner, None);
        assert_eq!(selection.recommendation, "retain_current_frontier");
        let card = build_deep_model_card(
            &selection,
            &DeepCampaignReport {
                format: "dope-deep-joint-campaign-report".into(),
                version: 1,
                slices: Vec::new(),
                failures: Vec::new(),
                paired_pareto_hypervolume: BTreeMap::new(),
            },
        )
        .unwrap();
        assert!(card.contains("retain the current frontier"));
        assert!(card.contains("never pooled"));
    }
}

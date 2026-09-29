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

include!("deep_campaign/cell_comparison.rs");
include!("deep_campaign/slice_reports.rs");

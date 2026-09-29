use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};

use crate::error::{DopeError, Result, io_error};
use crate::production::{ContentHashes, canonical_json, hashes, read_json};

pub const HEARTBEAT_SECONDS: u64 = 30;
pub const LEASE_SECONDS: u64 = 600;
pub const MAX_INFRASTRUCTURE_RETRIES: u8 = 2;
pub const JOB_EVIDENCE_VERSION: u8 = 3;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CacheStatus {
    Disabled,
    Hit,
    Miss,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq, Ord, PartialOrd)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Pending,
    Leased,
    Running,
    Succeeded,
    Failed,
    TimedOut,
}

impl JobState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Leased => "leased",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::TimedOut => "timed_out",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "pending" => Ok(Self::Pending),
            "leased" => Ok(Self::Leased),
            "running" => Ok(Self::Running),
            "succeeded" => Ok(Self::Succeeded),
            "failed" => Ok(Self::Failed),
            "timed_out" => Ok(Self::TimedOut),
            _ => Err(DopeError::Data(format!("unknown job state {value}"))),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    Infrastructure,
    DeterministicModel,
    Timeout,
    ContractMismatch,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct JobEvidence {
    pub format: String,
    pub version: u8,
    pub phase: String,
    pub task: String,
    pub candidate_id: String,
    pub auditor_id: String,
    pub lineage_group_id: String,
    pub structural_profile: String,
    pub train_rows: usize,
    pub features: usize,
    pub size_multiplier: usize,
    pub generation_seed: u64,
    pub auditor_seed: u64,
    pub routed: bool,
    pub null_loss: f64,
    pub trtr_loss: f64,
    pub tstr_loss: f64,
    pub runtime_ms: u64,
    pub fitting_time_ms: u64,
    pub sampling_time_ms: u64,
    pub auditor_time_ms: u64,
    pub peak_memory_bytes: u64,
    pub peak_cpu_memory_bytes: u64,
    pub peak_gpu_memory_bytes: Option<u64>,
    pub artifact_bytes: u64,
    pub artifact_cache_status: CacheStatus,
    pub real_auditor_cache_status: CacheStatus,
    pub ancillary_cache_status: CacheStatus,
    #[serde(default)]
    pub artifact_sha256: String,
    #[serde(default)]
    pub artifact_blake3: String,
    pub calibration_degradation: Option<f64>,
    pub rare_class_or_tail_retention: Option<f64>,
    pub supported_subgroup_retention: Option<f64>,
    pub nominal_95_coverage: Option<f64>,
    pub driver_agreement: Option<f64>,
    pub joint_fidelity: Option<f64>,
    pub query_p95_normalized_error: Option<f64>,
    pub type_i_error: Option<f64>,
    pub membership_auc: Option<f64>,
    pub attribute_inference_advantage: Option<f64>,
    pub feature_importance_spearman: Option<f64>,
    pub feature_importance_top_k_agreement: f64,
    pub feature_importance_feature_count: usize,
    pub feature_importance_informative_count: usize,
    #[serde(default)]
    pub feature_importance_real_shares: Vec<(usize, f64)>,
    #[serde(default)]
    pub feature_importance_synthetic_shares: Vec<(usize, f64)>,
    #[serde(default)]
    pub feature_importance_mean_ratio_error: Option<f64>,
    pub exact_copies: usize,
    pub near_copies: usize,
    pub canary_extractions: usize,
    pub lineage_leaks: usize,
    #[serde(default)]
    pub invalid_rows: usize,
    #[serde(default)]
    pub schema_violations: usize,
    #[serde(default)]
    pub nondeterministic_output: bool,
}

impl JobEvidence {
    pub fn validate_against(&self, spec: &JobSpec) -> Result<()> {
        let optional_finite = |value: Option<f64>| value.is_none_or(f64::is_finite);
        if self.format != "dope-job-evidence"
            || !(self.version == 2 || self.version == JOB_EVIDENCE_VERSION)
            || !matches!(
                self.phase.as_str(),
                "training_gold" | "validation_select" | "validation_cert"
            )
            || !matches!(self.task.as_str(), "binary" | "regression")
            || self.candidate_id != spec.candidate_spec
            || self.auditor_id != spec.auditor_spec
            || self.lineage_group_id != spec.dataset_lineage
            || self.structural_profile != spec.structural_profile
            || self.train_rows == 0
            || self.features == 0
            || self.size_multiplier != spec.size_multiplier
            || self.generation_seed != spec.generation_seed
            || self.auditor_seed != spec.auditor_seed
            || self.phase != spec.phase
            || self.routed != spec.routed
            || ![self.null_loss, self.trtr_loss, self.tstr_loss]
                .into_iter()
                .all(|value| value.is_finite() && value >= 0.0)
            || !is_lower_hex(&self.artifact_sha256, 64)
            || !is_lower_hex(&self.artifact_blake3, 64)
            || ![
                self.calibration_degradation,
                self.rare_class_or_tail_retention,
                self.supported_subgroup_retention,
                self.nominal_95_coverage,
                self.driver_agreement,
                self.joint_fidelity,
                self.query_p95_normalized_error,
                self.type_i_error,
                self.membership_auc,
                self.attribute_inference_advantage,
                self.feature_importance_spearman,
            ]
            .into_iter()
            .all(optional_finite)
            || !self.feature_importance_top_k_agreement.is_finite()
            || !(0.0..=1.0).contains(&self.feature_importance_top_k_agreement)
            || self.feature_importance_feature_count == 0
            || self.feature_importance_feature_count > self.features
            || self.feature_importance_informative_count > self.feature_importance_feature_count
            || (self.version == JOB_EVIDENCE_VERSION
                && (self.feature_importance_feature_count != self.features
                    || self.feature_importance_real_shares.len() != self.features
                    || self.feature_importance_synthetic_shares.len() != self.features
                    || self
                        .feature_importance_real_shares
                        .iter()
                        .zip(&self.feature_importance_synthetic_shares)
                        .enumerate()
                        .any(|(index, (real, synthetic))| {
                            real.0 != index
                                || synthetic.0 != index
                                || !real.1.is_finite()
                                || !synthetic.1.is_finite()
                                || real.1 < 0.0
                                || synthetic.1 < 0.0
                        })
                    || !optional_finite(self.feature_importance_mean_ratio_error)))
            || self.peak_cpu_memory_bytes != self.peak_memory_bytes
            || self
                .fitting_time_ms
                .saturating_add(self.sampling_time_ms)
                .saturating_add(self.auditor_time_ms)
                > self.runtime_ms
        {
            return Err(DopeError::Data(
                "job evidence does not match its immutable specification".into(),
            ));
        }
        Ok(())
    }
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct JobSpec {
    pub format: String,
    pub version: u8,
    pub source_commit: String,
    pub environment_lock_sha256: String,
    pub corpus_manifest_sha256: String,
    pub kpi_contract_sha256: String,
    pub phase: String,
    pub routed: bool,
    pub candidate_spec: String,
    pub auditor_spec: String,
    pub dataset_lineage: String,
    pub structural_profile: String,
    pub size_multiplier: usize,
    pub generation_seed: u64,
    pub auditor_seed: u64,
    pub tuning_split: String,
    pub command: Vec<String>,
    pub environment: BTreeMap<String, String>,
    pub expected_outputs: BTreeMap<String, PathBuf>,
    pub timeout_seconds: u64,
}

impl JobSpec {
    pub fn validate(&self) -> Result<()> {
        let digest = |value: &str| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        };
        if self.format != "dope-job-spec"
            || self.version != 1
            || !(digest(&self.source_commit)
                || (self.source_commit.len() == 40
                    && self
                        .source_commit
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))))
            || !digest(&self.environment_lock_sha256)
            || !digest(&self.corpus_manifest_sha256)
            || !digest(&self.kpi_contract_sha256)
            || !matches!(
                self.phase.as_str(),
                "training_gold" | "validation_select" | "validation_cert"
            )
            || self.candidate_spec.is_empty()
            || self.auditor_spec.is_empty()
            || self.dataset_lineage.is_empty()
            || self.structural_profile.is_empty()
            || ![1, 2, 4, 8].contains(&self.size_multiplier)
            || ![57721, 161803, 271828].contains(&self.auditor_seed)
            || !["real_inner_cv", "synthetic_only_cv", "fixed_no_tuning"]
                .contains(&self.tuning_split.as_str())
            || self.command.is_empty()
            || !(self.expected_outputs.contains_key("metrics")
                || self.expected_outputs.contains_key("evidence"))
            || self.timeout_seconds == 0
        {
            return Err(DopeError::Data("invalid frozen job specification".into()));
        }
        Ok(())
    }

    pub fn id(&self) -> Result<String> {
        self.validate()?;
        Ok(hashes(&canonical_json(self)?).blake3)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct JobReceipt {
    pub format: String,
    pub version: u8,
    pub job_id: String,
    pub attempt: u8,
    pub worker: String,
    pub state: JobState,
    pub started_unix_seconds: u64,
    pub finished_unix_seconds: u64,
    pub source_commit: String,
    pub environment_lock_sha256: String,
    pub corpus_manifest_sha256: String,
    pub kpi_contract_sha256: String,
    pub output_hashes: BTreeMap<String, ContentHashes>,
    pub failure_class: Option<FailureClass>,
    pub failure: Option<String>,
    pub command_exit_code: Option<i32>,
    pub evidence: Option<JobEvidence>,
    pub signature: Option<String>,
}

impl JobReceipt {
    pub fn id(&self) -> Result<String> {
        Ok(hashes(&canonical_json(self)?).blake3)
    }

    pub fn validate_against(&self, spec: &JobSpec) -> Result<()> {
        if self.format != "dope-job-receipt"
            || self.version != 1
            || self.job_id != spec.id()?
            || self.source_commit != spec.source_commit
            || self.environment_lock_sha256 != spec.environment_lock_sha256
            || self.corpus_manifest_sha256 != spec.corpus_manifest_sha256
            || self.kpi_contract_sha256 != spec.kpi_contract_sha256
            || self.finished_unix_seconds < self.started_unix_seconds
            || (self.state == JobState::Succeeded
                && (self.failure.is_some()
                    || self.failure_class.is_some()
                    || self.output_hashes.keys().collect::<Vec<_>>()
                        != spec.expected_outputs.keys().collect::<Vec<_>>()
                    || self.evidence.is_none()))
            || (self.state != JobState::Succeeded && self.evidence.is_some())
        {
            return Err(DopeError::Data(
                "job receipt does not match its immutable specification".into(),
            ));
        }
        if let Some(evidence) = &self.evidence {
            evidence.validate_against(spec)?;
        }
        Ok(())
    }

    fn signing_bytes(&self) -> Result<Vec<u8>> {
        let mut unsigned = self.clone();
        unsigned.signature = None;
        canonical_json(&unsigned)
    }

    pub fn sign(&mut self, key: &[u8; 32]) -> Result<()> {
        self.signature = Some(format!(
            "blake3-keyed-v1:{}",
            blake3::keyed_hash(key, &self.signing_bytes()?).to_hex()
        ));
        Ok(())
    }

    pub fn verify_signature(&self, key: &[u8; 32]) -> Result<()> {
        let expected = format!(
            "blake3-keyed-v1:{}",
            blake3::keyed_hash(key, &self.signing_bytes()?).to_hex()
        );
        if self.signature.as_deref() != Some(&expected) {
            return Err(DopeError::Data("invalid worker receipt signature".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct LeasedJob {
    pub job_id: String,
    pub attempt: u8,
    pub lease_owner: String,
    pub lease_until_unix_seconds: u64,
    pub spec: JobSpec,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LedgerStatus {
    pub format: String,
    pub version: u8,
    pub counts: BTreeMap<JobState, usize>,
    pub total: usize,
    pub training_progress: Option<f64>,
    pub completed_gold_labels: usize,
    pub failed_jobs: usize,
    pub timed_out_jobs: usize,
    pub throughput_jobs_per_hour: Option<f64>,
    pub coverage: Option<f64>,
    pub current_router_validation_regret: Option<f64>,
    pub ptf_v1: Option<f64>,
    pub production_score_available: bool,
    pub updated_unix_seconds: u64,
}

include!("ledger/sqlite_ledger.rs");

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
pub const JOB_EVIDENCE_VERSION: u8 = 2;

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
            || self.version != JOB_EVIDENCE_VERSION
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
            || self.feature_importance_feature_count > 64
            || self.feature_importance_informative_count > self.feature_importance_feature_count
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

#[derive(Clone, Debug)]
pub struct CampaignLedger {
    path: PathBuf,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl CampaignLedger {
    pub fn initialize(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
        }
        let ledger = Self {
            path: path.to_path_buf(),
        };
        let connection = ledger.connection()?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;
             PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS jobs (
                 job_id TEXT PRIMARY KEY,
                 spec_json BLOB NOT NULL,
                 state TEXT NOT NULL CHECK(state IN ('pending','leased','running','succeeded','failed','timed_out')),
                 attempts INTEGER NOT NULL DEFAULT 0,
                 lease_owner TEXT,
                 lease_until INTEGER,
                 heartbeat_at INTEGER,
                 last_error TEXT,
                 created_at INTEGER NOT NULL,
                 updated_at INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS receipts (
                 receipt_id TEXT PRIMARY KEY,
                 job_id TEXT NOT NULL REFERENCES jobs(job_id),
                 attempt INTEGER NOT NULL,
                 receipt_json BLOB NOT NULL,
                 created_at INTEGER NOT NULL,
                 UNIQUE(job_id, attempt)
             );
             CREATE TABLE IF NOT EXISTS metadata (
                 key TEXT PRIMARY KEY,
                 value_json BLOB NOT NULL,
                 updated_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS jobs_state_idx ON jobs(state, job_id);",
        )?;
        Ok(ledger)
    }

    pub fn open(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Err(DopeError::Data(format!(
                "campaign ledger does not exist: {}",
                path.display()
            )));
        }
        let ledger = Self {
            path: path.to_path_buf(),
        };
        let connection = ledger.connection()?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value_json BLOB NOT NULL,
                updated_at INTEGER NOT NULL
            );",
        )?;
        Ok(ledger)
    }

    fn connection(&self) -> Result<Connection> {
        let connection = Connection::open(&self.path)?;
        // Multiple short-lived handles are intentionally used by the scheduler
        // and by recovery tooling.  WAL removes reader/writer contention, but
        // writers can still briefly overlap; wait deterministically instead of
        // surfacing a transient SQLITE_BUSY failure.
        connection.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(connection)
    }

    pub fn insert_specs(&self, specs: &[JobSpec]) -> Result<usize> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let timestamp = now();
        let mut inserted = 0;
        for spec in specs {
            let job_id = spec.id()?;
            let bytes = canonical_json(spec)?;
            inserted += transaction.execute(
                "INSERT OR IGNORE INTO jobs(job_id,spec_json,state,created_at,updated_at)
                 VALUES (?1,?2,'pending',?3,?3)",
                params![job_id, bytes, timestamp],
            )?;
            let existing: Vec<u8> = transaction.query_row(
                "SELECT spec_json FROM jobs WHERE job_id=?1",
                params![job_id],
                |row| row.get(0),
            )?;
            if existing != bytes {
                return Err(DopeError::Data(
                    "content-addressed job collision or mutated specification".into(),
                ));
            }
        }
        transaction.commit()?;
        Ok(inserted)
    }

    pub fn lease_next(&self, worker: &str, lease_seconds: u64) -> Result<Option<LeasedJob>> {
        self.lease_next_for(
            worker,
            lease_seconds,
            &["training_gold", "validation_select", "validation_cert"],
        )
    }

    pub fn lease_next_for(
        &self,
        worker: &str,
        lease_seconds: u64,
        allowed_phases: &[&str],
    ) -> Result<Option<LeasedJob>> {
        if worker.is_empty() || lease_seconds == 0 {
            return Err(DopeError::Data("invalid lease request".into()));
        }
        if allowed_phases.is_empty() {
            return Ok(None);
        }
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let selected = {
            let mut statement = transaction.prepare(
                "SELECT job_id,spec_json,attempts FROM jobs
                 WHERE state='pending' ORDER BY job_id",
            )?;
            let rows = statement.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, u8>(2)?,
                ))
            })?;
            let mut selected = None;
            for row in rows {
                let row = row?;
                let spec: JobSpec = serde_json::from_slice(&row.1)?;
                if allowed_phases.contains(&spec.phase.as_str()) {
                    selected = Some(row);
                    break;
                }
            }
            selected
        };
        let Some((job_id, bytes, attempts)) = selected else {
            transaction.commit()?;
            return Ok(None);
        };
        let timestamp = now();
        let lease_until = timestamp.saturating_add(lease_seconds);
        let attempt = attempts.saturating_add(1);
        let changed = transaction.execute(
            "UPDATE jobs SET state='leased',attempts=?2,lease_owner=?3,lease_until=?4,
             heartbeat_at=?5,updated_at=?5 WHERE job_id=?1 AND state='pending'",
            params![job_id, attempt, worker, lease_until, timestamp],
        )?;
        if changed != 1 {
            return Err(DopeError::Data("job lease race was not serialized".into()));
        }
        transaction.commit()?;
        let spec: JobSpec = serde_json::from_slice(&bytes)?;
        spec.validate()?;
        Ok(Some(LeasedJob {
            job_id,
            attempt,
            lease_owner: worker.into(),
            lease_until_unix_seconds: lease_until,
            spec,
        }))
    }

    pub fn get_spec(&self, job_id: &str) -> Result<JobSpec> {
        let connection = self.connection()?;
        let bytes: Vec<u8> = connection.query_row(
            "SELECT spec_json FROM jobs WHERE job_id=?1",
            params![job_id],
            |row| row.get(0),
        )?;
        let spec: JobSpec = serde_json::from_slice(&bytes)?;
        if spec.id()? != job_id {
            return Err(DopeError::Data(
                "ledger job ID no longer matches stored bytes".into(),
            ));
        }
        Ok(spec)
    }

    pub fn mark_running(&self, job_id: &str, worker: &str) -> Result<()> {
        let connection = self.connection()?;
        let timestamp = now();
        let changed = connection.execute(
            "UPDATE jobs SET state='running',heartbeat_at=?3,updated_at=?3
             WHERE job_id=?1 AND state='leased' AND lease_owner=?2 AND lease_until>=?3",
            params![job_id, worker, timestamp],
        )?;
        if changed != 1 {
            return Err(DopeError::Data(
                "job cannot enter running state without a live matching lease".into(),
            ));
        }
        Ok(())
    }

    pub fn heartbeat(&self, job_id: &str, worker: &str, lease_seconds: u64) -> Result<()> {
        let connection = self.connection()?;
        let timestamp = now();
        let changed = connection.execute(
            "UPDATE jobs SET heartbeat_at=?3,lease_until=?4,updated_at=?3
             WHERE job_id=?1 AND lease_owner=?2 AND state IN ('leased','running')",
            params![
                job_id,
                worker,
                timestamp,
                timestamp.saturating_add(lease_seconds)
            ],
        )?;
        if changed != 1 {
            return Err(DopeError::Data("heartbeat does not own a live job".into()));
        }
        Ok(())
    }

    pub fn finish(&self, receipt: &JobReceipt) -> Result<()> {
        let spec = self.get_spec(&receipt.job_id)?;
        receipt.validate_against(&spec)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (state, attempts, lease_owner): (String, u8, Option<String>) = transaction.query_row(
            "SELECT state,attempts,lease_owner FROM jobs WHERE job_id=?1",
            params![receipt.job_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        if !matches!(
            JobState::parse(&state)?,
            JobState::Leased | JobState::Running
        ) || attempts != receipt.attempt
            || lease_owner.as_deref() != Some(&receipt.worker)
        {
            return Err(DopeError::Data(
                "receipt is stale or job is already terminal".into(),
            ));
        }
        let receipt_id = receipt.id()?;
        let bytes = canonical_json(receipt)?;
        transaction.execute(
            "INSERT INTO receipts(receipt_id,job_id,attempt,receipt_json,created_at)
             VALUES (?1,?2,?3,?4,?5)",
            params![receipt_id, receipt.job_id, receipt.attempt, bytes, now()],
        )?;
        let next_state = match receipt.state {
            JobState::Succeeded => JobState::Succeeded,
            JobState::TimedOut => JobState::TimedOut,
            JobState::Failed => {
                if receipt.failure_class == Some(FailureClass::Infrastructure)
                    && receipt.attempt <= MAX_INFRASTRUCTURE_RETRIES
                {
                    JobState::Pending
                } else {
                    JobState::Failed
                }
            }
            _ => {
                return Err(DopeError::Data(
                    "worker receipt must be succeeded, failed, or timed_out".into(),
                ));
            }
        };
        transaction.execute(
            "UPDATE jobs SET state=?2,lease_owner=NULL,lease_until=NULL,heartbeat_at=NULL,
             last_error=?3,updated_at=?4 WHERE job_id=?1",
            params![receipt.job_id, next_state.as_str(), receipt.failure, now()],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn finish_signed(&self, receipt: &JobReceipt, key: &[u8; 32]) -> Result<()> {
        receipt.verify_signature(key)?;
        self.finish(receipt)
    }

    pub fn reclaim_expired(&self, timestamp: u64) -> Result<usize> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        // SQLite INTEGER is signed; callers may use u64::MAX as a deterministic
        // "reclaim everything" sentinel during recovery/tests.
        let sqlite_timestamp = timestamp.min(i64::MAX as u64);
        let mut statement = transaction.prepare(
            "SELECT job_id,attempts FROM jobs
             WHERE state IN ('leased','running') AND lease_until<?1 ORDER BY job_id",
        )?;
        let expired: Vec<(String, u8)> = statement
            .query_map(params![sqlite_timestamp], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?
            .collect::<std::result::Result<_, _>>()?;
        drop(statement);
        for (job_id, attempts) in &expired {
            let state = if *attempts <= MAX_INFRASTRUCTURE_RETRIES {
                JobState::Pending
            } else {
                JobState::TimedOut
            };
            transaction.execute(
                "UPDATE jobs SET state=?2,lease_owner=NULL,lease_until=NULL,heartbeat_at=NULL,
                 last_error='lease expired',updated_at=?3 WHERE job_id=?1",
                params![job_id, state.as_str(), sqlite_timestamp],
            )?;
        }
        transaction.commit()?;
        Ok(expired.len())
    }

    pub fn status(&self) -> Result<LedgerStatus> {
        let connection = self.connection()?;
        let mut counts = BTreeMap::new();
        for state in [
            JobState::Pending,
            JobState::Leased,
            JobState::Running,
            JobState::Succeeded,
            JobState::Failed,
            JobState::TimedOut,
        ] {
            let count: usize = connection.query_row(
                "SELECT COUNT(*) FROM jobs WHERE state=?1",
                params![state.as_str()],
                |row| row.get(0),
            )?;
            counts.insert(state, count);
        }
        let total: usize = counts.values().sum();
        let receipts = self.all_receipts()?;
        let completed_gold_labels = receipts
            .iter()
            .filter(|receipt| receipt.state == JobState::Succeeded && receipt.evidence.is_some())
            .count();
        let terminal =
            completed_gold_labels + counts[&JobState::Failed] + counts[&JobState::TimedOut];
        let measured: Option<serde_json::Value> = self.get_metadata("status_evidence")?;
        let router_regret = measured
            .as_ref()
            .and_then(|value| value.get("router_regret"))
            .and_then(serde_json::Value::as_f64);
        let ptf_v1 = measured
            .as_ref()
            .and_then(|value| value.get("ptf_v1"))
            .and_then(serde_json::Value::as_f64);
        let measured_coverage = measured
            .as_ref()
            .and_then(|value| value.get("coverage"))
            .and_then(serde_json::Value::as_f64);
        let production_score_available = measured
            .as_ref()
            .and_then(|value| value.get("production_score_available"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        Ok(LedgerStatus {
            format: "dope-campaign-ledger-status".into(),
            version: 1,
            total,
            training_progress: (total > 0).then_some(terminal as f64 / total as f64),
            completed_gold_labels,
            failed_jobs: counts[&JobState::Failed],
            timed_out_jobs: counts[&JobState::TimedOut],
            throughput_jobs_per_hour: throughput(&receipts),
            coverage: measured_coverage
                .or_else(|| (total > 0).then_some(terminal as f64 / total as f64)),
            // Router validation and PTF are imported only by the frozen evidence pipeline.
            // Job completion alone is not a quality measurement.
            current_router_validation_regret: router_regret,
            ptf_v1,
            production_score_available,
            counts,
            updated_unix_seconds: now(),
        })
    }

    pub fn successful_receipt(&self, job_id: &str) -> Result<Option<JobReceipt>> {
        let connection = self.connection()?;
        let bytes: Option<Vec<u8>> = connection
            .query_row(
                "SELECT r.receipt_json FROM receipts r JOIN jobs j ON j.job_id=r.job_id
                 WHERE r.job_id=?1 AND j.state='succeeded' ORDER BY r.attempt DESC LIMIT 1",
                params![job_id],
                |row| row.get(0),
            )
            .optional()?;
        bytes
            .map(|value| serde_json::from_slice(&value).map_err(Into::into))
            .transpose()
    }

    pub fn export_receipts(&self, path: &Path) -> Result<ContentHashes> {
        let receipts = self.all_receipts()?;
        let bytes = canonical_json(&receipts)?;
        fs::write(path, &bytes).map_err(|error| io_error(path, error))?;
        Ok(hashes(&bytes))
    }

    pub fn load_specs(path: &Path) -> Result<Vec<JobSpec>> {
        read_json(path)
    }

    pub fn all_specs(&self) -> Result<Vec<(String, JobState, JobSpec)>> {
        let connection = self.connection()?;
        let mut statement =
            connection.prepare("SELECT job_id,state,spec_json FROM jobs ORDER BY job_id")?;
        statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            })?
            .map(|row| {
                let (id, state, bytes) = row?;
                Ok((
                    id,
                    JobState::parse(&state)?,
                    serde_json::from_slice(&bytes)?,
                ))
            })
            .collect()
    }

    pub fn all_receipts(&self) -> Result<Vec<JobReceipt>> {
        let connection = self.connection()?;
        let mut statement =
            connection.prepare("SELECT receipt_json FROM receipts ORDER BY job_id,attempt")?;
        statement
            .query_map([], |row| row.get::<_, Vec<u8>>(0))?
            .map(|row| Ok(serde_json::from_slice(&row?)?))
            .collect()
    }

    pub fn put_metadata<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        if key.is_empty() {
            return Err(DopeError::Data("metadata key cannot be empty".into()));
        }
        self.connection()?.execute(
            "INSERT INTO metadata(key,value_json,updated_at) VALUES (?1,?2,?3)
             ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,
             updated_at=excluded.updated_at",
            params![key, canonical_json(value)?, now()],
        )?;
        Ok(())
    }

    pub fn get_metadata<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        let bytes: Option<Vec<u8>> = self
            .connection()?
            .query_row(
                "SELECT value_json FROM metadata WHERE key=?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?;
        bytes
            .map(|bytes| serde_json::from_slice(&bytes).map_err(Into::into))
            .transpose()
    }
}

fn throughput(receipts: &[JobReceipt]) -> Option<f64> {
    let first = receipts
        .iter()
        .map(|receipt| receipt.started_unix_seconds)
        .min()?;
    let last = receipts
        .iter()
        .map(|receipt| receipt.finished_unix_seconds)
        .max()?;
    let hours = (last.saturating_sub(first).max(1)) as f64 / 3_600.0;
    Some(receipts.len() as f64 / hours)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> JobSpec {
        JobSpec {
            format: "dope-job-spec".into(),
            version: 1,
            source_commit: "a".repeat(64),
            environment_lock_sha256: "b".repeat(64),
            corpus_manifest_sha256: "c".repeat(64),
            kpi_contract_sha256: "d".repeat(64),
            phase: "training_gold".into(),
            routed: false,
            candidate_spec: "independent_quantile".into(),
            auditor_spec: "elastic_net_glm".into(),
            dataset_lineage: "lineage-1".into(),
            structural_profile: "regression/32-255/1-16".into(),
            size_multiplier: 1,
            generation_seed: 42,
            auditor_seed: 57721,
            tuning_split: "synthetic_only_cv".into(),
            command: vec!["true".into()],
            environment: BTreeMap::new(),
            expected_outputs: BTreeMap::from([("metrics".into(), PathBuf::from("metrics.json"))]),
            timeout_seconds: 60,
        }
    }

    fn ledger() -> (CampaignLedger, PathBuf) {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "dope-ledger-{}-{}.sqlite",
            std::process::id(),
            unique
        ));
        let _ = fs::remove_file(&path);
        (CampaignLedger::initialize(&path).unwrap(), path)
    }

    fn evidence(spec: &JobSpec) -> JobEvidence {
        JobEvidence {
            format: "dope-job-evidence".into(),
            version: JOB_EVIDENCE_VERSION,
            phase: spec.phase.clone(),
            task: "regression".into(),
            candidate_id: spec.candidate_spec.clone(),
            auditor_id: spec.auditor_spec.clone(),
            lineage_group_id: spec.dataset_lineage.clone(),
            structural_profile: spec.structural_profile.clone(),
            train_rows: 100,
            features: 4,
            size_multiplier: spec.size_multiplier,
            generation_seed: spec.generation_seed,
            auditor_seed: spec.auditor_seed,
            routed: spec.routed,
            null_loss: 1.0,
            trtr_loss: 0.5,
            tstr_loss: 0.6,
            runtime_ms: 10,
            fitting_time_ms: 3,
            sampling_time_ms: 2,
            auditor_time_ms: 5,
            peak_memory_bytes: 1_024,
            peak_cpu_memory_bytes: 1_024,
            peak_gpu_memory_bytes: None,
            artifact_bytes: 512,
            artifact_cache_status: CacheStatus::Miss,
            real_auditor_cache_status: CacheStatus::Miss,
            ancillary_cache_status: CacheStatus::Disabled,
            artifact_sha256: "a".repeat(64),
            artifact_blake3: "b".repeat(64),
            calibration_degradation: None,
            rare_class_or_tail_retention: None,
            supported_subgroup_retention: None,
            nominal_95_coverage: None,
            driver_agreement: None,
            joint_fidelity: None,
            query_p95_normalized_error: None,
            type_i_error: None,
            membership_auc: None,
            attribute_inference_advantage: None,
            feature_importance_spearman: Some(0.5),
            feature_importance_top_k_agreement: 0.5,
            feature_importance_feature_count: 4,
            feature_importance_informative_count: 3,
            exact_copies: 0,
            near_copies: 0,
            canary_extractions: 0,
            lineage_leaks: 0,
            invalid_rows: 0,
            schema_violations: 0,
            nondeterministic_output: false,
        }
    }

    #[test]
    fn ledger_is_idempotent_and_receipts_are_immutable() {
        let (ledger, path) = ledger();
        assert_eq!(ledger.insert_specs(&[spec()]).unwrap(), 1);
        assert_eq!(ledger.insert_specs(&[spec()]).unwrap(), 0);
        let leased = ledger.lease_next("xbabe1", 600).unwrap().unwrap();
        ledger.mark_running(&leased.job_id, "xbabe1").unwrap();
        let mut receipt = JobReceipt {
            format: "dope-job-receipt".into(),
            version: 1,
            job_id: leased.job_id.clone(),
            attempt: leased.attempt,
            worker: "xbabe1".into(),
            state: JobState::Succeeded,
            started_unix_seconds: now(),
            finished_unix_seconds: now(),
            source_commit: leased.spec.source_commit.clone(),
            environment_lock_sha256: leased.spec.environment_lock_sha256.clone(),
            corpus_manifest_sha256: leased.spec.corpus_manifest_sha256.clone(),
            kpi_contract_sha256: leased.spec.kpi_contract_sha256.clone(),
            output_hashes: BTreeMap::from([(
                "metrics".into(),
                ContentHashes {
                    sha256: "e".repeat(64),
                    blake3: "f".repeat(64),
                },
            )]),
            failure_class: None,
            failure: None,
            command_exit_code: Some(0),
            evidence: Some(evidence(&leased.spec)),
            signature: None,
        };
        let key = [7; 32];
        receipt.sign(&key).unwrap();
        ledger.finish_signed(&receipt, &key).unwrap();
        let status = ledger.status().unwrap();
        assert_eq!(status.counts[&JobState::Succeeded], 1);
        assert_eq!(status.completed_gold_labels, 1);
        assert_eq!(status.training_progress, Some(1.0));
        assert_eq!(status.current_router_validation_regret, None);
        assert_eq!(status.ptf_v1, None);
        assert!(!status.production_score_available);
        assert!(ledger.finish_signed(&receipt, &key).is_err());
        drop(ledger);
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(path.with_extension("sqlite-wal"));
        let _ = fs::remove_file(path.with_extension("sqlite-shm"));
    }

    #[test]
    fn expired_infrastructure_leases_retry_only_twice() {
        let (ledger, path) = ledger();
        ledger.insert_specs(&[spec()]).unwrap();
        for expected_attempt in 1..=3 {
            let leased = ledger.lease_next("worker", 1).unwrap().unwrap();
            assert_eq!(leased.attempt, expected_attempt);
            ledger.reclaim_expired(u64::MAX).unwrap();
        }
        assert_eq!(ledger.status().unwrap().counts[&JobState::TimedOut], 1);
        drop(ledger);
        let _ = fs::remove_file(path);
    }
}

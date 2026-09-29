use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufWriter, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::access::{AccessRole, ValidationAccessBroker};
use crate::certification::{
    GoldCellOptions, evaluate_gold_cell, evaluate_prepared_gold_cell, prepare_gold_cell,
};
use crate::contract::{auditor_specs, empirical_backends};
use crate::corpus::{
    SplitManifest, ValidationSubmanifest, validate_split_manifest, validate_validation_submanifest,
};
use crate::data::Table;
use crate::error::{DopeError, Result, io_error};
use crate::ledger::{
    CampaignLedger, FailureClass, HEARTBEAT_SECONDS, JOB_EVIDENCE_VERSION, JobEvidence, JobReceipt,
    JobState, LEASE_SECONDS, LeasedJob,
};
use crate::model::Task;
use crate::production::{
    ContentHashes, CoverageEntry, CoverageReport, GateEvidence, KPI_CONTRACT_SHA256, KpiCell,
    KpiSummary, StructuralProfile, aggregate_kpis, canonical_json, file_hashes, hashes, read_json,
    write_canonical,
};
use crate::router::{
    DATASET_SKETCH_VERSION, DatasetSketch, MAX_ROUTER_BUNDLE_BYTES, QuantizedRouter, RouterBundle,
    RouterLabel, RouterTrainingReport, SelectionPolicy, train_distilled_router,
};

pub const MIN_DURABLE_FREE_GIB: u64 = 150;
const GIB: u64 = 1024 * 1024 * 1024;

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn current_hostname() -> Result<String> {
    let path = Path::new("/etc/hostname");
    let hostname = fs::read_to_string(path).map_err(|error| io_error(path, error))?;
    hostname
        .trim()
        .split('.')
        .next()
        .filter(|hostname| !hostname.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| DopeError::Data("host identity is empty".into()))
}

fn require_named_host(actual: &str, expected: &str, operation: &str) -> Result<()> {
    if actual != expected {
        return Err(DopeError::Data(format!(
            "{operation} is restricted to {expected}; current host is {actual}"
        )));
    }
    Ok(())
}

fn require_host(expected: &str, operation: &str) -> Result<()> {
    require_named_host(&current_hostname()?, expected, operation)
}

pub fn available_bytes(path: &Path) -> Result<u64> {
    let original_path = path.to_path_buf();
    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| DopeError::Data("storage path contains a NUL byte".into()))?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `path` is a live NUL-terminated string and `stats` points to
    // writable storage for the duration of the libc call.
    if unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return Err(io_error(original_path, std::io::Error::last_os_error()));
    }
    // SAFETY: statvfs returned success and initialized the structure.
    let stats = unsafe { stats.assume_init() };
    Ok(stats.f_bavail.saturating_mul(stats.f_frsize))
}

fn require_durable_space(path: &Path) -> Result<u64> {
    let bytes = available_bytes(path)?;
    if bytes < MIN_DURABLE_FREE_GIB * GIB {
        return Err(DopeError::Data(format!(
            "campaign admission stopped: durable free space is {:.2} GiB, below {MIN_DURABLE_FREE_GIB} GiB",
            bytes as f64 / GIB as f64
        )));
    }
    Ok(bytes)
}

fn is_lower_hex(value: &str, lengths: &[usize]) -> bool {
    lengths.contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn load_receipt_key(path: &Path) -> Result<[u8; 32]> {
    let text = fs::read_to_string(path).map_err(|error| io_error(path, error))?;
    let text = text.trim();
    if !is_lower_hex(text, &[64]) {
        return Err(DopeError::Data(
            "receipt key file must contain exactly 32 lowercase hexadecimal bytes".into(),
        ));
    }
    let mut key = [0u8; 32];
    for (index, slot) in key.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16)
            .map_err(|_| DopeError::Data("invalid receipt key".into()))?;
    }
    Ok(key)
}

pub fn create_receipt_key(path: &Path) -> Result<ContentHashes> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
    }
    let mut random = File::open("/dev/urandom")
        .map_err(|error| DopeError::Data(format!("cannot open OS randomness: {error}")))?;
    let mut key = [0u8; 32];
    random
        .read_exact(&mut key)
        .map_err(|error| DopeError::Data(format!("cannot read OS randomness: {error}")))?;
    let encoded = key
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .map_err(|error| io_error(path, error))?;
    output
        .write_all(encoded.as_bytes())
        .and_then(|_| output.write_all(b"\n"))
        .and_then(|_| output.sync_all())
        .map_err(|error| io_error(path, error))?;
    file_hashes(path)
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CorpusSampleVerification {
    pub format: String,
    pub version: u8,
    pub manifest: ContentHashes,
    pub selection_numerator: usize,
    pub selection_denominator: usize,
    pub sampled_datasets: usize,
    pub sample_sha256: String,
    pub sample_blake3: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CorpusSamplePlan {
    pub format: String,
    pub version: u8,
    pub paths: Vec<PathBuf>,
}

pub fn corpus_sample_plan(manifest_path: &Path) -> Result<CorpusSamplePlan> {
    let manifest: SplitManifest = read_json(manifest_path)?;
    validate_split_manifest(&manifest)?;
    let mut paths = manifest
        .datasets
        .iter()
        .filter(|record| {
            let digest = blake3::hash(record.dataset_id.as_bytes());
            let mut prefix = [0u8; 8];
            prefix.copy_from_slice(&digest.as_bytes()[..8]);
            u64::from_le_bytes(prefix) % 100 == 0
        })
        .map(|record| record.path.join("train.csv"))
        .collect::<Vec<_>>();
    paths.sort();
    Ok(CorpusSamplePlan {
        format: "dope-corpus-sample-plan".into(),
        version: 1,
        paths,
    })
}

pub fn verify_corpus_sample(manifest_path: &Path) -> Result<CorpusSampleVerification> {
    let manifest: SplitManifest = read_json(manifest_path)?;
    validate_split_manifest(&manifest)?;
    let mut selected = manifest
        .datasets
        .iter()
        .filter(|record| {
            let digest = blake3::hash(record.dataset_id.as_bytes());
            let mut prefix = [0u8; 8];
            prefix.copy_from_slice(&digest.as_bytes()[..8]);
            u64::from_le_bytes(prefix) % 100 == 0
        })
        .collect::<Vec<_>>();
    selected.sort_by(|left, right| left.dataset_id.cmp(&right.dataset_id));
    if selected.is_empty() {
        return Err(DopeError::Data(
            "deterministic corpus sample unexpectedly selected no datasets".into(),
        ));
    }
    let mut aggregate_sha = Sha256::new();
    let mut aggregate_blake = blake3::Hasher::new();
    for record in &selected {
        let path = record.path.join("train.csv");
        let mut file = File::open(&path).map_err(|error| io_error(&path, error))?;
        let mut file_sha = Sha256::new();
        let mut file_blake = blake3::Hasher::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|error| io_error(&path, error))?;
            if read == 0 {
                break;
            }
            file_sha.update(&buffer[..read]);
            file_blake.update(&buffer[..read]);
        }
        let file_sha = file_sha.finalize();
        let file_blake = file_blake.finalize();
        let id = record.dataset_id.as_bytes();
        let id_len = (id.len() as u64).to_le_bytes();
        aggregate_sha.update(id_len);
        aggregate_sha.update(id);
        aggregate_sha.update(file_sha);
        aggregate_blake.update(&id_len);
        aggregate_blake.update(id);
        aggregate_blake.update(file_blake.as_bytes());
    }
    Ok(CorpusSampleVerification {
        format: "dope-corpus-sample-verification".into(),
        version: 1,
        manifest: file_hashes(manifest_path)?,
        selection_numerator: 1,
        selection_denominator: 100,
        sampled_datasets: selected.len(),
        sample_sha256: format!("{:x}", aggregate_sha.finalize()),
        sample_blake3: aggregate_blake.finalize().to_hex().to_string(),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CohortRecord {
    pub dataset_id: String,
    pub dataset_path: PathBuf,
    pub task: String,
    pub rows: usize,
    pub features: usize,
    pub lineage_group_id: String,
    pub structural_profile: String,
    pub partition: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CohortPlan {
    pub format: String,
    pub version: u8,
    pub kind: String,
    pub records: Vec<CohortRecord>,
    #[serde(default)]
    pub exclusions: Vec<CohortExclusion>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CohortExclusion {
    pub dataset_id: String,
    pub reason: String,
}

fn cohort_order(seed: u64, profile: &str, lineage: &str, dataset: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    for value in [profile, lineage, dataset] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// Selects campaign cohorts without opening validation-cert. Candidates are
/// admitted only after their train table agrees with the frozen inventory;
/// malformed or stale inventory entries are retained as explicit exclusions.
pub fn plan_cohort(manifest_path: &Path, validation_path: &Path, kind: &str) -> Result<CohortPlan> {
    if !matches!(kind, "baseline" | "training-gold" | "validation-select") {
        return Err(DopeError::Data(
            "cohort kind must be baseline, training-gold, or validation-select".into(),
        ));
    }
    let manifest: SplitManifest = read_json(manifest_path)?;
    validate_split_manifest(&manifest)?;
    let validation: ValidationSubmanifest = read_json(validation_path)?;
    validate_validation_submanifest(&validation)?;
    if validation.source_manifest_checksum != manifest.checksum {
        return Err(DopeError::Data(
            "cohort manifests do not share a source checksum".into(),
        ));
    }
    let select = validation
        .assignments
        .iter()
        .filter(|assignment| assignment.partition == "validation-select")
        .map(|assignment| {
            (
                assignment.dataset_id.as_str(),
                assignment.lineage_group_id.as_str(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut canonical = BTreeMap::<(String, String), ([u8; 32], CohortRecord)>::new();
    for record in &manifest.datasets {
        if kind == "training-gold"
            && (record.origin == "mirror"
                || record.duplicate_of.is_some()
                || record.near_duplicate_of.is_some())
        {
            continue;
        }
        let (eligible, lineage, partition) = if kind == "training-gold" {
            (
                record.split == "train",
                record.group_id.as_str(),
                "training_gold",
            )
        } else {
            (
                record.split == "validation" && select.contains_key(record.dataset_id.as_str()),
                select
                    .get(record.dataset_id.as_str())
                    .copied()
                    .unwrap_or(record.group_id.as_str()),
                "validation_select",
            )
        };
        if !eligible {
            continue;
        }
        let features = record.cols.saturating_sub(1);
        let profile = StructuralProfile::from_shape(record.task, record.rows, features)?.id();
        let order = cohort_order(manifest.seed, &profile, lineage, &record.dataset_id);
        let value = CohortRecord {
            dataset_id: record.dataset_id.clone(),
            dataset_path: record.path.clone(),
            task: record.task.as_str().into(),
            rows: record.rows,
            features,
            lineage_group_id: lineage.into(),
            structural_profile: profile.clone(),
            partition: partition.into(),
        };
        let cohort_profile = if kind == "validation-select" {
            String::new()
        } else {
            profile
        };
        let entry = canonical
            .entry((cohort_profile, lineage.into()))
            .or_insert_with(|| (order, value.clone()));
        if order < entry.0 {
            *entry = (order, value);
        }
    }
    let mut by_profile = BTreeMap::<String, Vec<([u8; 32], CohortRecord)>>::new();
    for ((profile, _), value) in canonical {
        by_profile.entry(profile).or_default().push(value);
    }
    let mut records = Vec::new();
    let mut exclusions = Vec::new();
    let mut profile_batches = by_profile.into_iter().collect::<Vec<_>>();
    profile_batches.sort_by(|left, right| {
        left.1
            .len()
            .cmp(&right.1.len())
            .then_with(|| left.0.cmp(&right.0))
    });
    for (_, mut profile_records) in profile_batches {
        profile_records.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.dataset_id.cmp(&right.1.dataset_id))
        });
        let take = match kind {
            "baseline" => 1,
            "training-gold" => 120,
            _ => usize::MAX,
        };
        let mut admitted = 0usize;
        for (_, record) in profile_records {
            if admitted == take {
                break;
            }
            if kind == "validation-select" {
                records.push(record);
                admitted += 1;
                continue;
            }
            match Table::read_dataset_dir(&record.dataset_path, table_task(&record.task)?) {
                Ok(table) if table.rows == record.rows && table.features == record.features => {
                    records.push(record);
                    admitted += 1;
                }
                Ok(table) => exclusions.push(CohortExclusion {
                    dataset_id: record.dataset_id,
                    reason: format!(
                        "inventory shape {}/{} differs from train shape {}/{}",
                        record.rows, record.features, table.rows, table.features
                    ),
                }),
                Err(error) => exclusions.push(CohortExclusion {
                    dataset_id: record.dataset_id,
                    reason: error.to_string(),
                }),
            }
        }
    }
    records.sort_by(|left, right| {
        left.structural_profile
            .cmp(&right.structural_profile)
            .then_with(|| left.lineage_group_id.cmp(&right.lineage_group_id))
    });
    if kind == "validation-select"
        && records
            .iter()
            .map(|record| &record.lineage_group_id)
            .collect::<BTreeSet<_>>()
            .len()
            != validation.validation_select_lineage_groups
    {
        return Err(DopeError::Data(
            "canonical validation-select cohort lost required lineages".into(),
        ));
    }
    Ok(CohortPlan {
        format: "dope-campaign-cohort".into(),
        version: 1,
        kind: kind.into(),
        records,
        exclusions,
    })
}

/// Materializes validation-cert paths only after the single durable
/// authorization event. This command is intended to execute on xbabe3; the
/// resulting cohort never participates in router training.
pub fn plan_validation_cert_cohort(
    state_dir: &Path,
    manifest_path: &Path,
    validation_path: &Path,
) -> Result<CohortPlan> {
    require_host("xbabe3", "validation-cert cohort planning")?;
    let state = load_state(state_dir)?;
    if state.phase != CampaignPhase::ValidationCertOpen
        || state.validation_cert_open_count != 1
        || state.sealed_test_open_count != 0
    {
        return Err(DopeError::Data(
            "validation-cert cohort requires the one durable authorization event".into(),
        ));
    }
    let manifest: SplitManifest = read_json(manifest_path)?;
    validate_split_manifest(&manifest)?;
    let validation: ValidationSubmanifest = read_json(validation_path)?;
    validate_validation_submanifest(&validation)?;
    if validation.source_manifest_checksum != manifest.checksum
        || file_hashes(manifest_path)? != state.corpus_manifest
        || file_hashes(validation_path)? != state.validation_manifest
    {
        return Err(DopeError::Data(
            "validation-cert manifests differ from the frozen campaign".into(),
        ));
    }
    let broker = ValidationAccessBroker::new(&validation, AccessRole::FinalReleaseEvaluator, true)?;
    let assignments = validation
        .assignments
        .iter()
        .map(|assignment| (assignment.dataset_id.as_str(), assignment))
        .collect::<BTreeMap<_, _>>();
    let mut canonical = BTreeMap::<String, ([u8; 32], CohortRecord)>::new();
    for record in &manifest.datasets {
        let Some(assignment) = assignments.get(record.dataset_id.as_str()) else {
            continue;
        };
        if record.split != "validation" || assignment.partition != "validation-cert" {
            continue;
        }
        broker.authorize_lineage(&validation, &assignment.lineage_group_id)?;
        let features = record.cols.saturating_sub(1);
        let profile = StructuralProfile::from_shape(record.task, record.rows, features)?.id();
        let order = cohort_order(
            manifest.seed,
            &profile,
            &assignment.lineage_group_id,
            &record.dataset_id,
        );
        let value = CohortRecord {
            dataset_id: record.dataset_id.clone(),
            dataset_path: record.path.clone(),
            task: record.task.as_str().into(),
            rows: record.rows,
            features,
            lineage_group_id: assignment.lineage_group_id.clone(),
            structural_profile: profile,
            partition: "validation_cert".into(),
        };
        let entry = canonical
            .entry(assignment.lineage_group_id.clone())
            .or_insert_with(|| (order, value.clone()));
        if order < entry.0 {
            *entry = (order, value);
        }
    }
    let mut records = canonical
        .into_values()
        .map(|(_, record)| record)
        .collect::<Vec<_>>();
    records.sort_by(|left, right| {
        left.structural_profile
            .cmp(&right.structural_profile)
            .then_with(|| left.lineage_group_id.cmp(&right.lineage_group_id))
    });
    if records.len() != state.validation_cert_lineage_groups
        || records
            .iter()
            .map(|record| &record.lineage_group_id)
            .collect::<BTreeSet<_>>()
            .len()
            != state.validation_cert_lineage_groups
    {
        return Err(DopeError::Data(
            "canonical validation-cert cohort lost required lineages".into(),
        ));
    }
    Ok(CohortPlan {
        format: "dope-campaign-cohort".into(),
        version: 1,
        kind: "validation-cert".into(),
        records,
        exclusions: Vec::new(),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BaselineFailure {
    pub dataset_id: String,
    pub candidate_id: String,
    pub auditor_id: String,
    #[serde(default)]
    pub size_multiplier: Option<usize>,
    #[serde(default)]
    pub generation_seed: Option<u64>,
    #[serde(default)]
    pub auditor_seed: Option<u64>,
    pub error: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BaselineSummary {
    pub format: String,
    pub version: u8,
    pub cohort_records: usize,
    pub attempted: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub failures: Vec<BaselineFailure>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct GoldRecordBlock {
    pub(crate) format: String,
    pub(crate) version: u8,
    pub(crate) dataset_id: String,
    matrix_identity: String,
    #[serde(default)]
    routing: Option<GoldRoutingDecision>,
    pub(crate) evidence: Vec<JobEvidence>,
    pub(crate) failures: Vec<BaselineFailure>,
    #[serde(default = "default_neural_target_weight")]
    pub(crate) neural_target_weight: f64,
    #[serde(default)]
    pub(crate) neural_structural_penalty: f64,
}

const GOLD_RECORD_BLOCK_VERSION: u8 = 2;
const GOLD_BLOCK_RECEIPT_VERSION: u8 = 2;

fn default_neural_target_weight() -> f64 {
    2.0
}

struct GoldMatrixIdentity<'a> {
    phase: &'a str,
    dataset_id: &'a str,
    candidate_ids: &'a [String],
    auditor_ids: &'a [String],
    size_multipliers: &'a [usize],
    generation_seeds: &'a [u64],
    auditor_seeds: &'a [u64],
    primary_only: bool,
    neural_target_weight: f64,
    neural_structural_penalty: f64,
    routing_sha256: Option<&'a str>,
}

fn gold_record_matrix_identity(identity: &GoldMatrixIdentity<'_>) -> Result<String> {
    let mut value = serde_json::json!({
        "format": "dope-gold-record-block-key",
        "version": GOLD_RECORD_BLOCK_VERSION,
        "phase": identity.phase,
        "dataset": identity.dataset_id,
        "candidates": identity.candidate_ids,
        "auditors": identity.auditor_ids,
        "size_multipliers": identity.size_multipliers,
        "generation_seeds": identity.generation_seeds,
        "auditor_seeds": identity.auditor_seeds,
        "primary_only": identity.primary_only,
        "neural_target_weight": identity.neural_target_weight,
        "neural_structural_penalty": identity.neural_structural_penalty
    });
    if let Some(routing_sha256) = identity.routing_sha256 {
        value
            .as_object_mut()
            .expect("gold matrix identity is an object")
            .insert(
                "routing_sha256".into(),
                Value::String(routing_sha256.into()),
            );
    }
    Ok(hashes(&canonical_json(&value)?).blake3)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct GoldRoutingDecision {
    format: String,
    version: u8,
    router_bundle_sha256: String,
    ranked_candidates: Vec<String>,
    top_four_candidates: Vec<String>,
    selected_candidates: Vec<String>,
    prediction_sha256: String,
    sketch_ms: f64,
    inference_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GoldBlockReceipt {
    pub format: String,
    pub version: u8,
    pub dataset_id: String,
    pub matrix_identity: String,
    pub phase: String,
    pub worker: String,
    pub source_sha256: String,
    pub evaluator_binary_sha256: String,
    #[serde(default)]
    pub router_bundle_sha256: Option<String>,
    pub cohort: ContentHashes,
    pub block: ContentHashes,
    pub evidence_cells: usize,
    pub failed_cells: usize,
    pub signed_unix_seconds: u64,
    pub signature: Option<String>,
}

impl GoldBlockReceipt {
    fn signing_bytes(&self) -> Result<Vec<u8>> {
        let mut unsigned = self.clone();
        unsigned.signature = None;
        canonical_json(&unsigned)
    }

    fn sign(&mut self, key: &[u8; 32]) -> Result<()> {
        self.signature = Some(format!(
            "blake3-keyed-v1:{}",
            blake3::keyed_hash(key, &self.signing_bytes()?).to_hex()
        ));
        Ok(())
    }

    pub fn verify(&self, key: &[u8; 32]) -> Result<()> {
        let expected = format!(
            "blake3-keyed-v1:{}",
            blake3::keyed_hash(key, &self.signing_bytes()?).to_hex()
        );
        if self.format != "dope-gold-block-receipt"
            || self.version != GOLD_BLOCK_RECEIPT_VERSION
            || !matches!(
                self.phase.as_str(),
                "training_gold" | "validation_select" | "validation_cert"
            )
            || !matches!(self.worker.as_str(), "xbabe1" | "xbabe2" | "xbabe3")
            || !is_lower_hex(&self.source_sha256, &[64])
            || !is_lower_hex(&self.evaluator_binary_sha256, &[64])
            || self
                .router_bundle_sha256
                .as_ref()
                .is_some_and(|value| !is_lower_hex(value, &[64]))
            || !is_lower_hex(&self.matrix_identity, &[64])
            || self.signature.as_deref() != Some(&expected)
        {
            return Err(DopeError::Data("invalid gold block receipt".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GoldReceiptSummary {
    pub format: String,
    pub version: u8,
    pub worker: String,
    pub blocks: usize,
    pub evidence_cells: usize,
    pub failed_cells: usize,
    pub cohort: ContentHashes,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MeasuredCandidateSummary {
    pub candidate_id: String,
    pub implementation_hash: String,
    pub cells: usize,
    pub mean_bounded_utility: f64,
    pub runtime_p95_ms: u64,
    pub peak_memory_p95_bytes: u64,
    pub artifact_p95_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MeasuredEvidenceSummary {
    pub format: String,
    pub version: u8,
    pub cells: usize,
    pub lineage_groups: usize,
    pub lineage_profile_outcomes: usize,
    pub candidates: usize,
    pub auditors: usize,
    pub tasks: Vec<String>,
    pub candidate_frontier: Vec<MeasuredCandidateSummary>,
    pub calibration_degradation_max: Option<f64>,
    pub rare_tail_retention_min: Option<f64>,
    pub subgroup_retention_min: Option<f64>,
    pub nominal_coverage_min: Option<f64>,
    pub nominal_coverage_max: Option<f64>,
    pub driver_agreement_min: Option<f64>,
    pub joint_fidelity_min: Option<f64>,
    pub query_error_max: Option<f64>,
    pub type_i_error_max: Option<f64>,
    pub membership_auc_max: Option<f64>,
    pub feature_importance_complete: bool,
    pub feature_importance_applicable: bool,
    pub feature_importance_spearman_min: Option<f64>,
    pub feature_importance_top_k_jaccard_min: Option<f64>,
    pub attribute_advantage_max: Option<f64>,
    pub exact_copies: usize,
    pub near_copies: usize,
}

fn aggregate_importance<'a>(
    cells: impl IntoIterator<Item = &'a JobEvidence>,
) -> (bool, bool, Option<f64>, Option<f64>) {
    let mut count = 0usize;
    let mut complete = true;
    let mut applicable = false;
    let mut spearman = None::<f64>;
    let mut jaccard = None::<f64>;
    let mut missing_rank = false;
    for cell in cells {
        count += 1;
        complete &= cell.version == JOB_EVIDENCE_VERSION
            && cell.feature_importance_feature_count == cell.features
            && cell.feature_importance_real_shares.len() == cell.features
            && cell.feature_importance_synthetic_shares.len() == cell.features;
        if cell.feature_importance_informative_count >= 2 {
            applicable = true;
            if let Some(value) = cell.feature_importance_spearman {
                spearman = Some(spearman.map_or(value, |minimum| minimum.min(value)));
            } else {
                missing_rank = true;
            }
            let value = cell.feature_importance_top_k_agreement;
            jaccard = Some(jaccard.map_or(value, |minimum| minimum.min(value)));
        }
    }
    (
        count > 0 && complete,
        applicable,
        if missing_rank { None } else { spearman },
        jaccard,
    )
}

fn bounded_router_utility(evidence: &JobEvidence) -> f64 {
    let improvement = evidence.null_loss - evidence.trtr_loss;
    if improvement >= 0.01 * evidence.null_loss.abs() && improvement.abs() > f64::EPSILON {
        ((evidence.null_loss - evidence.tstr_loss) / improvement).clamp(0.0, 1.0)
    } else {
        f64::from(evidence.tstr_loss <= evidence.trtr_loss + 0.01 * evidence.null_loss)
    }
}

fn for_each_evidence(
    paths: &[PathBuf],
    callback: &mut impl FnMut(JobEvidence) -> Result<()>,
) -> Result<()> {
    let mut expanded = Vec::new();
    for path in paths {
        if path.is_dir() {
            let mut entries = fs::read_dir(path)
                .map_err(|error| io_error(path, error))?
                .map(|entry| {
                    entry
                        .map(|entry| entry.path())
                        .map_err(|error| io_error(path, error))
                })
                .collect::<Result<Vec<_>>>()?;
            entries.retain(|entry| {
                entry
                    .extension()
                    .is_some_and(|extension| extension == "json")
            });
            entries.sort();
            expanded.extend(entries);
        } else {
            expanded.push(path.clone());
        }
    }
    for path in expanded {
        let file = File::open(&path).map_err(|error| io_error(&path, error))?;
        let mut lines = std::io::BufReader::new(file).lines();
        let Some(first_line) = lines.next() else {
            continue;
        };
        let first_line = first_line.map_err(|error| io_error(&path, error))?;
        if first_line.trim().is_empty() {
            continue;
        }
        let first_value: Value = serde_json::from_str(&first_line)?;
        if first_value.get("format").and_then(Value::as_str) == Some("dope-gold-record-block") {
            let block: GoldRecordBlock = serde_json::from_value(first_value)?;
            if block.version != GOLD_RECORD_BLOCK_VERSION || lines.next().is_some() {
                return Err(DopeError::Data(format!(
                    "invalid evidence record block at {}",
                    path.display()
                )));
            }
            for cell in block.evidence {
                callback(cell)?;
            }
        } else {
            callback(serde_json::from_value(first_value)?)?;
            for line in lines {
                let line = line.map_err(|error| io_error(&path, error))?;
                if !line.trim().is_empty() {
                    callback(serde_json::from_str(&line)?)?;
                }
            }
        }
    }
    Ok(())
}

pub fn summarize_evidence(metric_paths: &[PathBuf], out: &Path) -> Result<MeasuredEvidenceSummary> {
    if metric_paths.is_empty() {
        return Err(DopeError::Data("evidence metrics inputs are empty".into()));
    }
    #[derive(Default)]
    struct CandidateAggregate {
        cells: usize,
        bounded_utility_sum: f64,
        runtimes: BTreeMap<u64, usize>,
        memories: BTreeMap<u64, usize>,
        artifact_bytes: BTreeMap<u64, usize>,
    }
    let histogram_percentile = |histogram: &BTreeMap<u64, usize>, cells: usize| {
        if cells == 0 {
            return 0;
        }
        let index = ((cells - 1) as f64 * 0.95).ceil() as usize;
        let mut cumulative = 0usize;
        for (&value, &count) in histogram {
            cumulative += count;
            if cumulative > index {
                return value;
            }
        }
        0
    };
    let auditor_ids = auditor_specs()
        .into_iter()
        .map(|auditor| auditor.id)
        .collect::<Vec<_>>();
    let mut full_replicates = BTreeMap::<(String, String, String, String, bool), u128>::new();
    let mut other_identities = BTreeSet::<String>::new();
    let mut candidates = BTreeMap::<String, CandidateAggregate>::new();
    let mut lineage_groups = BTreeSet::<String>::new();
    let mut lineage_profile_outcomes = BTreeSet::<(String, String)>::new();
    let mut auditors = BTreeSet::<String>::new();
    let mut tasks = BTreeSet::<String>::new();
    let mut cells = 0usize;
    let mut calibration_degradation_max = None;
    let mut rare_tail_retention_min = None;
    let mut subgroup_retention_min = None;
    let mut nominal_coverage_min = None;
    let mut nominal_coverage_max = None;
    let mut driver_agreement_min = None;
    let mut joint_fidelity_min = None;
    let mut query_error_max = None;
    let mut type_i_error_max = None;
    let mut membership_auc_max = None;
    let mut attribute_advantage_max = None;
    let mut importance_complete = true;
    let mut importance_applicable = false;
    let mut importance_spearman_min = None::<f64>;
    let mut importance_jaccard_min = None::<f64>;
    let mut importance_missing_rank = false;
    let mut exact_copies = 0usize;
    let mut near_copies = 0usize;
    let update_min = |slot: &mut Option<f64>, value: Option<f64>| {
        if let Some(value) = value {
            *slot = Some(slot.map_or(value, |existing| existing.min(value)));
        }
    };
    let update_max = |slot: &mut Option<f64>, value: Option<f64>| {
        if let Some(value) = value {
            *slot = Some(slot.map_or(value, |existing| existing.max(value)));
        }
    };
    for_each_evidence(metric_paths, &mut |cell| {
        if let Ok(slot) = router_replicate_slot(&cell, &auditor_ids) {
            let seen = full_replicates
                .entry((
                    cell.phase.clone(),
                    cell.lineage_group_id.clone(),
                    cell.structural_profile.clone(),
                    cell.candidate_id.clone(),
                    cell.routed,
                ))
                .or_default();
            let bit = 1u128 << slot;
            if *seen & bit != 0 {
                return Err(DopeError::Data(
                    "evidence metrics contain duplicate cell identities".into(),
                ));
            }
            *seen |= bit;
        } else {
            let identity = hashes(&canonical_json(&serde_json::json!({
                "phase": cell.phase,
                "lineage": cell.lineage_group_id,
                "profile": cell.structural_profile,
                "candidate": cell.candidate_id,
                "auditor": cell.auditor_id,
                "size_multiplier": cell.size_multiplier,
                "generation_seed": cell.generation_seed,
                "auditor_seed": cell.auditor_seed,
                "routed": cell.routed,
            }))?)
            .blake3;
            if !other_identities.insert(identity) {
                return Err(DopeError::Data(
                    "evidence metrics contain duplicate cell identities".into(),
                ));
            }
        }
        let aggregate = candidates.entry(cell.candidate_id.clone()).or_default();
        aggregate.cells += 1;
        aggregate.bounded_utility_sum += bounded_router_utility(&cell);
        *aggregate.runtimes.entry(cell.runtime_ms).or_default() += 1;
        *aggregate
            .memories
            .entry(cell.peak_memory_bytes)
            .or_default() += 1;
        *aggregate
            .artifact_bytes
            .entry(cell.artifact_bytes)
            .or_default() += 1;
        lineage_groups.insert(cell.lineage_group_id.clone());
        lineage_profile_outcomes.insert((
            cell.lineage_group_id.clone(),
            cell.structural_profile.clone(),
        ));
        auditors.insert(cell.auditor_id.clone());
        tasks.insert(cell.task.clone());
        update_max(
            &mut calibration_degradation_max,
            cell.calibration_degradation,
        );
        update_min(
            &mut rare_tail_retention_min,
            cell.rare_class_or_tail_retention,
        );
        update_min(
            &mut subgroup_retention_min,
            cell.supported_subgroup_retention,
        );
        update_min(&mut nominal_coverage_min, cell.nominal_95_coverage);
        update_max(&mut nominal_coverage_max, cell.nominal_95_coverage);
        update_min(&mut driver_agreement_min, cell.driver_agreement);
        update_min(&mut joint_fidelity_min, cell.joint_fidelity);
        update_max(&mut query_error_max, cell.query_p95_normalized_error);
        update_max(&mut type_i_error_max, cell.type_i_error);
        update_max(&mut membership_auc_max, cell.membership_auc);
        importance_complete &= cell.version == JOB_EVIDENCE_VERSION
            && cell.feature_importance_feature_count == cell.features
            && cell.feature_importance_real_shares.len() == cell.features
            && cell.feature_importance_synthetic_shares.len() == cell.features;
        if cell.feature_importance_informative_count >= 2 {
            importance_applicable = true;
            if let Some(value) = cell.feature_importance_spearman {
                update_min(&mut importance_spearman_min, Some(value));
            } else {
                importance_missing_rank = true;
            }
            update_min(
                &mut importance_jaccard_min,
                Some(cell.feature_importance_top_k_agreement),
            );
        }
        update_max(
            &mut attribute_advantage_max,
            cell.attribute_inference_advantage,
        );
        exact_copies += cell.exact_copies;
        near_copies += cell.near_copies;
        cells += 1;
        Ok(())
    })?;
    let mut candidate_frontier = candidates
        .into_iter()
        .map(|(candidate_id, aggregate)| MeasuredCandidateSummary {
            implementation_hash: empirical_backends()
                .into_iter()
                .find(|backend| backend.id == candidate_id)
                .map(|backend| backend.implementation_hash)
                .unwrap_or_default(),
            candidate_id,
            cells: aggregate.cells,
            mean_bounded_utility: aggregate.bounded_utility_sum / aggregate.cells.max(1) as f64,
            runtime_p95_ms: histogram_percentile(&aggregate.runtimes, aggregate.cells),
            peak_memory_p95_bytes: histogram_percentile(&aggregate.memories, aggregate.cells),
            artifact_p95_bytes: histogram_percentile(&aggregate.artifact_bytes, aggregate.cells),
        })
        .collect::<Vec<_>>();
    candidate_frontier.sort_by(|left, right| {
        right
            .mean_bounded_utility
            .total_cmp(&left.mean_bounded_utility)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    let summary = MeasuredEvidenceSummary {
        format: "dope-measured-evidence-summary".into(),
        version: 2,
        cells,
        lineage_groups: lineage_groups.len(),
        lineage_profile_outcomes: lineage_profile_outcomes.len(),
        candidates: candidate_frontier.len(),
        auditors: auditors.len(),
        tasks: tasks.into_iter().collect(),
        candidate_frontier,
        calibration_degradation_max,
        rare_tail_retention_min,
        subgroup_retention_min,
        nominal_coverage_min,
        nominal_coverage_max,
        driver_agreement_min,
        joint_fidelity_min,
        query_error_max,
        type_i_error_max,
        membership_auc_max,
        feature_importance_complete: cells > 0 && importance_complete,
        feature_importance_applicable: importance_applicable,
        feature_importance_spearman_min: if importance_missing_rank {
            None
        } else {
            importance_spearman_min
        },
        feature_importance_top_k_jaccard_min: importance_jaccard_min,
        attribute_advantage_max,
        exact_copies,
        near_copies,
    };
    write_canonical(out, &summary)?;
    Ok(summary)
}

/// Runs the first honest matrix with one fixed generation and auditor seed.
/// Each cell is persisted independently so interruption and restart do not
/// discard completed evidence.
pub fn run_baseline(
    cohort_path: &Path,
    out: &Path,
    candidate_ids: &[String],
    auditor_ids: &[String],
    shard_index: usize,
    shard_count: usize,
    primary_only: bool,
) -> Result<BaselineSummary> {
    if shard_count == 0 || shard_index >= shard_count {
        return Err(DopeError::Data("invalid baseline shard".into()));
    }
    let cohort: CohortPlan = read_json(cohort_path)?;
    if cohort.format != "dope-campaign-cohort" || cohort.kind != "baseline" {
        return Err(DopeError::Data(
            "baseline requires a frozen baseline cohort".into(),
        ));
    }
    let frozen_candidates = empirical_backends()
        .into_iter()
        .map(|value| value.id)
        .collect::<BTreeSet<_>>();
    let frozen_auditors = auditor_specs()
        .into_iter()
        .map(|value| value.id)
        .collect::<BTreeSet<_>>();
    if candidate_ids.is_empty()
        || auditor_ids.is_empty()
        || candidate_ids
            .iter()
            .any(|value| !frozen_candidates.contains(value))
        || auditor_ids
            .iter()
            .any(|value| !frozen_auditors.contains(value))
    {
        return Err(DopeError::Data(
            "baseline candidates and auditors must be nonempty frozen IDs".into(),
        ));
    }
    let cell_dir = out.join("cells");
    let cache_dir = out.join("cache");
    fs::create_dir_all(&cell_dir).map_err(|error| io_error(&cell_dir, error))?;
    let selected = cohort
        .records
        .iter()
        .enumerate()
        .filter(|(index, _)| index % shard_count == shard_index)
        .map(|(_, record)| record)
        .collect::<Vec<_>>();
    let mut evidence = Vec::new();
    let mut failures = Vec::new();
    for record in &selected {
        for candidate_id in candidate_ids {
            for auditor_id in auditor_ids {
                let identity = hashes(&canonical_json(&serde_json::json!({
                    "dataset": record.dataset_id,
                    "candidate": candidate_id,
                    "auditor": auditor_id,
                    "generation_seed": 1829,
                    "auditor_seed": 57721,
                    "primary_only": primary_only
                }))?)
                .blake3;
                let path = cell_dir.join(format!("{identity}.json"));
                if path.is_file() {
                    evidence.push(read_json::<JobEvidence>(&path)?);
                    continue;
                }
                match evaluate_gold_cell(&GoldCellOptions {
                    dataset_dir: &record.dataset_path,
                    task: if record.task == "binary" {
                        Task::Binary
                    } else {
                        Task::Regression
                    },
                    candidate_id,
                    auditor_id,
                    lineage_group_id: &record.lineage_group_id,
                    structural_profile: &record.structural_profile,
                    phase: "validation_select",
                    size_multiplier: 1,
                    generation_seed: 1829,
                    auditor_seed: 57721,
                    routed: false,
                    ancillary: !primary_only,
                    cache_dir: Some(&cache_dir),
                    neural_target_weight: 2.0,
                    neural_structural_penalty: 0.0,
                }) {
                    Ok(cell) => {
                        write_canonical(&path, &cell)?;
                        evidence.push(cell);
                    }
                    Err(error) => failures.push(BaselineFailure {
                        dataset_id: record.dataset_id.clone(),
                        candidate_id: candidate_id.clone(),
                        auditor_id: auditor_id.clone(),
                        size_multiplier: None,
                        generation_seed: None,
                        auditor_seed: None,
                        error: error.to_string(),
                    }),
                }
            }
        }
    }
    evidence.sort_by(|left, right| {
        left.structural_profile
            .cmp(&right.structural_profile)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
            .then_with(|| left.auditor_id.cmp(&right.auditor_id))
    });
    let metrics_path = out.join(format!("metrics-shard-{shard_index:02}.jsonl"));
    let mut metrics =
        File::create(&metrics_path).map_err(|error| io_error(&metrics_path, error))?;
    for cell in &evidence {
        metrics
            .write_all(&canonical_json(cell)?)
            .and_then(|_| metrics.write_all(b"\n"))
            .map_err(|error| io_error(&metrics_path, error))?;
    }
    let summary = BaselineSummary {
        format: "dope-baseline-summary".into(),
        version: 1,
        cohort_records: selected.len(),
        attempted: selected.len() * candidate_ids.len() * auditor_ids.len(),
        succeeded: evidence.len(),
        failed: failures.len(),
        failures,
    };
    write_canonical(
        &out.join(format!("summary-shard-{shard_index:02}.json")),
        &summary,
    )?;
    Ok(summary)
}

pub struct GoldMatrixOptions<'a> {
    pub cohort_path: &'a Path,
    pub out: &'a Path,
    pub candidate_ids: &'a [String],
    pub auditor_ids: &'a [String],
    pub size_multipliers: &'a [usize],
    pub generation_seeds: &'a [u64],
    pub auditor_seeds: &'a [u64],
    pub shard_index: usize,
    pub shard_count: usize,
    pub primary_only: bool,
    pub blocks_only: bool,
    pub neural_target_weight: f64,
    pub neural_structural_penalty: f64,
}

/// Executes the exhaustive frozen matrix. It is deliberately restart-safe at
/// cell granularity; terminal failures are retained in the shard summary and
/// are never replaced with another candidate's evidence.
pub fn run_gold_matrix(options: &GoldMatrixOptions<'_>) -> Result<BaselineSummary> {
    run_gold_matrix_internal(options, None)
}

struct RoutedCertContext<'a> {
    router: &'a QuantizedRouter,
    bundle_sha256: &'a str,
}

fn run_gold_matrix_internal(
    options: &GoldMatrixOptions<'_>,
    routed_cert: Option<&RoutedCertContext<'_>>,
) -> Result<BaselineSummary> {
    if options.shard_count == 0
        || options.shard_index >= options.shard_count
        || options.candidate_ids.is_empty()
        || options.auditor_ids.is_empty()
        || options.size_multipliers.is_empty()
        || options.generation_seeds.is_empty()
        || options.auditor_seeds.is_empty()
        || options
            .size_multipliers
            .iter()
            .any(|value| !matches!(value, 1 | 4))
        || options
            .auditor_seeds
            .iter()
            .any(|value| ![57721, 161803, 271828].contains(value))
        || !matches!(options.neural_target_weight, 2.0 | 4.0)
        || !matches!(options.neural_structural_penalty, 0.0 | 0.1)
    {
        return Err(DopeError::Data(
            "invalid exhaustive gold matrix options".into(),
        ));
    }
    let cohort: CohortPlan = read_json(options.cohort_path)?;
    let phase = match cohort.kind.as_str() {
        "training-gold" => "training_gold",
        "validation-select" => "validation_select",
        "validation-cert" if routed_cert.is_some() => "validation_cert",
        _ => {
            return Err(DopeError::Data(
                "gold matrix cohort is not authorized for this command".into(),
            ));
        }
    };
    let frozen_candidates = empirical_backends()
        .into_iter()
        .map(|value| value.id)
        .collect::<BTreeSet<_>>();
    let frozen_auditors = auditor_specs()
        .into_iter()
        .map(|value| value.id)
        .collect::<BTreeSet<_>>();
    if options
        .candidate_ids
        .iter()
        .any(|value| !frozen_candidates.contains(value))
        || options
            .auditor_ids
            .iter()
            .any(|value| !frozen_auditors.contains(value))
    {
        return Err(DopeError::Data(
            "gold matrix contains an unfrozen candidate or auditor".into(),
        ));
    }
    let cell_dir = options.out.join("cells");
    let cache_dir = options.out.join("cache");
    let block_dir = options.out.join("blocks");
    fs::create_dir_all(&cell_dir).map_err(|error| io_error(&cell_dir, error))?;
    fs::create_dir_all(&block_dir).map_err(|error| io_error(&block_dir, error))?;
    let selected = cohort
        .records
        .iter()
        .enumerate()
        .filter(|(index, _)| index % options.shard_count == options.shard_index)
        .map(|(_, record)| record)
        .collect::<Vec<_>>();
    let mut evidence = Vec::new();
    let mut failures = Vec::new();
    let cells_per_record = options.candidate_ids.len()
        * options.auditor_ids.len()
        * options.size_multipliers.len()
        * options.generation_seeds.len()
        * options.auditor_seeds.len();
    // Every matrix, including the 78-cell discovery matrix, is represented by
    // one durable record block. This gives failures and small campaigns the
    // same restart and provenance guarantees as exhaustive confirmation.
    let use_record_blocks = true;
    let mut succeeded_total = 0usize;
    for record in &selected {
        let matrix_identity = gold_record_matrix_identity(&GoldMatrixIdentity {
            phase,
            dataset_id: &record.dataset_id,
            candidate_ids: options.candidate_ids,
            auditor_ids: options.auditor_ids,
            size_multipliers: options.size_multipliers,
            generation_seeds: options.generation_seeds,
            auditor_seeds: options.auditor_seeds,
            primary_only: options.primary_only,
            neural_target_weight: options.neural_target_weight,
            neural_structural_penalty: options.neural_structural_penalty,
            routing_sha256: routed_cert.map(|context| context.bundle_sha256),
        })?;
        let block_path = block_dir.join(format!("{matrix_identity}.json"));
        if use_record_blocks && block_path.is_file() {
            let block: Option<GoldRecordBlock> = match read_json(&block_path) {
                Ok(block) => Some(block),
                Err(error) => {
                    let quarantine = options.out.join("operational-failures/corrupt-blocks");
                    fs::create_dir_all(&quarantine)
                        .map_err(|create_error| io_error(&quarantine, create_error))?;
                    let quarantined = quarantine.join(format!(
                        "{matrix_identity}-{}-{}.json",
                        unix_now(),
                        std::process::id()
                    ));
                    fs::rename(&block_path, &quarantined)
                        .map_err(|rename_error| io_error(&quarantined, rename_error))?;
                    write_canonical(
                        &quarantined.with_extension("error.json"),
                        &serde_json::json!({
                            "format": "dope-corrupt-gold-block-incident",
                            "version": 1,
                            "matrix_identity": matrix_identity,
                            "quarantined_path": quarantined,
                            "error": error.to_string()
                        }),
                    )?;
                    None
                }
            };
            if let Some(block) = block {
                if block.format != "dope-gold-record-block"
                    || block.version != GOLD_RECORD_BLOCK_VERSION
                    || block.dataset_id != record.dataset_id
                    || block.matrix_identity != matrix_identity
                    || block.evidence.len() + block.failures.len() != cells_per_record
                    || match (routed_cert, &block.routing) {
                        (None, None) => block.evidence.iter().any(|cell| cell.routed),
                        (Some(context), Some(routing)) => {
                            routing.router_bundle_sha256 != context.bundle_sha256
                                || routing.selected_candidates.is_empty()
                                || block.evidence.iter().any(|cell| {
                                    cell.routed
                                        != routing.selected_candidates.contains(&cell.candidate_id)
                                })
                        }
                        _ => true,
                    }
                {
                    return Err(DopeError::Data(format!(
                        "gold record block identity mismatch at {}",
                        block_path.display()
                    )));
                }
                succeeded_total += block.evidence.len();
                if !options.blocks_only {
                    evidence.extend(block.evidence);
                }
                failures.extend(block.failures);
                continue;
            }
        }
        let evidence_start = evidence.len();
        let failures_start = failures.len();
        let routing = if let Some(context) = routed_cert {
            let table = Table::read_dataset_dir(&record.dataset_path, table_task(&record.task)?)?;
            let sketch_started = Instant::now();
            let sketch = DatasetSketch::from_train(&table, table_task(&record.task)?);
            let sketch_ms = sketch_started.elapsed().as_secs_f64() * 1_000.0;
            let inference_started = Instant::now();
            let predictions = context.router.predict(&sketch)?;
            let inference_ms = inference_started.elapsed().as_secs_f64() * 1_000.0;
            let mut ranked = predictions.clone();
            ranked.sort_by(|left, right| {
                right
                    .retention_lower
                    .total_cmp(&left.retention_lower)
                    .then_with(|| left.expected_regret.total_cmp(&right.expected_regret))
                    .then_with(|| left.candidate_id.cmp(&right.candidate_id))
            });
            let ranked_candidates = ranked
                .iter()
                .map(|prediction| prediction.candidate_id.clone())
                .collect::<Vec<_>>();
            let top_four_candidates = ranked_candidates.iter().take(4).cloned().collect();
            let selected_candidates =
                SelectionPolicy::default().select(&predictions, "independent_quantile");
            Some(GoldRoutingDecision {
                format: "dope-gold-routing-decision".into(),
                version: 1,
                router_bundle_sha256: context.bundle_sha256.into(),
                ranked_candidates,
                top_four_candidates,
                selected_candidates,
                prediction_sha256: hashes(&canonical_json(&predictions)?).sha256,
                sketch_ms,
                inference_ms,
            })
        } else {
            None
        };
        for candidate_id in options.candidate_ids {
            for &size_multiplier in options.size_multipliers {
                for &generation_seed in options.generation_seeds {
                    let mut pending = Vec::new();
                    for auditor_id in options.auditor_ids {
                        for &auditor_seed in options.auditor_seeds {
                            let identity = hashes(&canonical_json(&serde_json::json!({
                                "phase": phase,
                                "dataset": record.dataset_id,
                                "candidate": candidate_id,
                                "auditor": auditor_id,
                                "size_multiplier": size_multiplier,
                                "generation_seed": generation_seed,
                                "auditor_seed": auditor_seed,
                                "primary_only": options.primary_only,
                                "neural_target_weight": options.neural_target_weight,
                                "neural_structural_penalty": options.neural_structural_penalty
                            }))?)
                            .blake3;
                            let path = cell_dir.join(format!("{identity}.json"));
                            if !use_record_blocks && path.is_file() {
                                evidence.push(read_json::<JobEvidence>(&path)?);
                                continue;
                            }
                            pending.push((auditor_id, auditor_seed, path));
                        }
                    }
                    if pending.is_empty() {
                        continue;
                    }
                    let preparation_options = GoldCellOptions {
                        dataset_dir: &record.dataset_path,
                        task: table_task(&record.task)?,
                        candidate_id,
                        auditor_id: pending[0].0,
                        lineage_group_id: &record.lineage_group_id,
                        structural_profile: &record.structural_profile,
                        phase,
                        size_multiplier,
                        generation_seed,
                        auditor_seed: pending[0].1,
                        routed: routing.as_ref().is_some_and(|routing| {
                            routing.selected_candidates.contains(candidate_id)
                        }),
                        ancillary: !options.primary_only,
                        cache_dir: Some(&cache_dir),
                        neural_target_weight: options.neural_target_weight,
                        neural_structural_penalty: options.neural_structural_penalty,
                    };
                    match prepare_gold_cell(&preparation_options) {
                        Ok(prepared) => {
                            let (native_pending, gpu_pending): (Vec<_>, Vec<_>) = pending
                                .into_iter()
                                .partition(|(auditor_id, _, _)| !auditor_id.starts_with("gpu_"));
                            let native_results = native_pending
                                .par_iter()
                                .map(|(auditor_id, auditor_seed, path)| {
                                    let cell_options = GoldCellOptions {
                                        auditor_id,
                                        auditor_seed: *auditor_seed,
                                        ..preparation_options
                                    };
                                    (
                                        *auditor_id,
                                        *auditor_seed,
                                        path,
                                        evaluate_prepared_gold_cell(&cell_options, &prepared),
                                    )
                                })
                                .collect::<Vec<_>>();
                            for (auditor_id, auditor_seed, path, result) in native_results {
                                match result {
                                    Ok(cell) => {
                                        if !use_record_blocks {
                                            write_canonical(path, &cell)?;
                                        }
                                        evidence.push(cell);
                                    }
                                    Err(error) => failures.push(BaselineFailure {
                                        dataset_id: record.dataset_id.clone(),
                                        candidate_id: candidate_id.clone(),
                                        auditor_id: auditor_id.clone(),
                                        size_multiplier: Some(size_multiplier),
                                        generation_seed: Some(generation_seed),
                                        auditor_seed: Some(auditor_seed),
                                        error: error.to_string(),
                                    }),
                                }
                            }
                            // tch's process-global seed must not race inside a worker.
                            for (auditor_id, auditor_seed, path) in gpu_pending {
                                let cell_options = GoldCellOptions {
                                    auditor_id,
                                    auditor_seed,
                                    ..preparation_options
                                };
                                match evaluate_prepared_gold_cell(&cell_options, &prepared) {
                                    Ok(cell) => {
                                        if !use_record_blocks {
                                            write_canonical(&path, &cell)?;
                                        }
                                        evidence.push(cell);
                                    }
                                    Err(error) => failures.push(BaselineFailure {
                                        dataset_id: record.dataset_id.clone(),
                                        candidate_id: candidate_id.clone(),
                                        auditor_id: auditor_id.clone(),
                                        size_multiplier: Some(size_multiplier),
                                        generation_seed: Some(generation_seed),
                                        auditor_seed: Some(auditor_seed),
                                        error: error.to_string(),
                                    }),
                                }
                            }
                        }
                        Err(error) => {
                            for (auditor_id, auditor_seed, _) in pending {
                                failures.push(BaselineFailure {
                                    dataset_id: record.dataset_id.clone(),
                                    candidate_id: candidate_id.clone(),
                                    auditor_id: auditor_id.clone(),
                                    size_multiplier: Some(size_multiplier),
                                    generation_seed: Some(generation_seed),
                                    auditor_seed: Some(auditor_seed),
                                    error: error.to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }
        if use_record_blocks {
            write_canonical(
                &block_path,
                &GoldRecordBlock {
                    format: "dope-gold-record-block".into(),
                    version: GOLD_RECORD_BLOCK_VERSION,
                    dataset_id: record.dataset_id.clone(),
                    matrix_identity,
                    routing,
                    evidence: evidence[evidence_start..].to_vec(),
                    failures: failures[failures_start..].to_vec(),
                    neural_target_weight: options.neural_target_weight,
                    neural_structural_penalty: options.neural_structural_penalty,
                },
            )?;
            succeeded_total += evidence.len() - evidence_start;
            if options.blocks_only {
                evidence.truncate(evidence_start);
            }
        }
    }
    evidence.sort_by(|left, right| {
        left.lineage_group_id
            .cmp(&right.lineage_group_id)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
            .then_with(|| left.auditor_id.cmp(&right.auditor_id))
            .then_with(|| left.size_multiplier.cmp(&right.size_multiplier))
            .then_with(|| left.generation_seed.cmp(&right.generation_seed))
            .then_with(|| left.auditor_seed.cmp(&right.auditor_seed))
    });
    if !options.blocks_only {
        let metrics_path = options
            .out
            .join(format!("metrics-shard-{:04}.jsonl", options.shard_index));
        let mut metrics =
            File::create(&metrics_path).map_err(|error| io_error(&metrics_path, error))?;
        for cell in &evidence {
            metrics
                .write_all(&canonical_json(cell)?)
                .and_then(|_| metrics.write_all(b"\n"))
                .map_err(|error| io_error(&metrics_path, error))?;
        }
    } else {
        write_canonical(
            &options.out.join(format!(
                "block-manifest-shard-{:04}.json",
                options.shard_index
            )),
            &serde_json::json!({
                "format": "dope-gold-block-shard-manifest",
                "version": GOLD_RECORD_BLOCK_VERSION,
                "shard_index": options.shard_index,
                "shard_count": options.shard_count,
                "cohort_records": selected.len(),
                "succeeded": succeeded_total,
                "failed": failures.len()
            }),
        )?;
    }
    let attempted = selected.len()
        * options.candidate_ids.len()
        * options.auditor_ids.len()
        * options.size_multipliers.len()
        * options.generation_seeds.len()
        * options.auditor_seeds.len();
    let summary = BaselineSummary {
        format: "dope-gold-matrix-summary".into(),
        version: 1,
        cohort_records: selected.len(),
        attempted,
        succeeded: if use_record_blocks {
            succeeded_total
        } else {
            evidence.len()
        },
        failed: failures.len(),
        failures,
    };
    write_canonical(
        &options
            .out
            .join(format!("summary-shard-{:04}.json", options.shard_index)),
        &summary,
    )?;
    Ok(summary)
}

pub struct RoutedCertOptions<'a> {
    pub state_dir: &'a Path,
    pub cohort_path: &'a Path,
    pub router_bundle: &'a Path,
    pub out: &'a Path,
    pub shard_index: usize,
    pub shard_count: usize,
}

pub fn run_routed_cert_matrix(options: &RoutedCertOptions<'_>) -> Result<BaselineSummary> {
    require_host("xbabe3", "routed validation-cert evaluation")?;
    let state = load_state(options.state_dir)?;
    if state.phase != CampaignPhase::ValidationCertOpen
        || state.validation_cert_open_count != 1
        || state.sealed_test_open_count != 0
    {
        return Err(DopeError::Data(
            "routed validation-cert requires the one authorization event".into(),
        ));
    }
    let bundle_hashes = file_hashes(options.router_bundle)?;
    if state.router_bundle.as_ref() != Some(&bundle_hashes) {
        return Err(DopeError::Data(
            "validation-cert router differs from the frozen bundle".into(),
        ));
    }
    let cohort: CohortPlan = read_json(options.cohort_path)?;
    if cohort.kind != "validation-cert"
        || cohort.records.len() != state.validation_cert_lineage_groups
        || cohort
            .records
            .iter()
            .map(|record| &record.lineage_group_id)
            .collect::<BTreeSet<_>>()
            .len()
            != state.validation_cert_lineage_groups
    {
        return Err(DopeError::Data(
            "validation-cert cohort is incomplete or not guarded".into(),
        ));
    }
    let bundle: RouterBundle = read_json(options.router_bundle)?;
    let router = QuantizedRouter::new(bundle)?;
    let candidate_ids = empirical_backends()
        .into_iter()
        .map(|candidate| candidate.id)
        .collect::<Vec<_>>();
    let auditor_ids = auditor_specs()
        .into_iter()
        .map(|auditor| auditor.id)
        .collect::<Vec<_>>();
    let size_multipliers = [1usize, 4];
    let generation_seeds = [1829u64, 99238, 196647];
    let auditor_seeds = [57721u64, 161803, 271828];
    run_gold_matrix_internal(
        &GoldMatrixOptions {
            cohort_path: options.cohort_path,
            out: options.out,
            candidate_ids: &candidate_ids,
            auditor_ids: &auditor_ids,
            size_multipliers: &size_multipliers,
            generation_seeds: &generation_seeds,
            auditor_seeds: &auditor_seeds,
            shard_index: options.shard_index,
            shard_count: options.shard_count,
            primary_only: false,
            blocks_only: true,
            neural_target_weight: 2.0,
            neural_structural_penalty: 0.0,
        },
        Some(&RoutedCertContext {
            router: &router,
            bundle_sha256: &bundle_hashes.sha256,
        }),
    )
}

fn validate_gold_record_block(
    block: &GoldRecordBlock,
    record: &CohortRecord,
    phase: &str,
    expected_identity: &str,
    candidate_ids: &[String],
    auditor_ids: &[String],
    routing_sha256: Option<&str>,
) -> Result<()> {
    let candidates = candidate_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let auditors = auditor_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let routed_candidates = match (routing_sha256, &block.routing) {
        (None, None) => BTreeSet::new(),
        (Some(expected), Some(routing))
            if routing.format == "dope-gold-routing-decision"
                && routing.version == 1
                && routing.router_bundle_sha256 == expected
                && is_lower_hex(&routing.prediction_sha256, &[64])
                && routing.sketch_ms.is_finite()
                && routing.sketch_ms >= 0.0
                && routing.inference_ms.is_finite()
                && routing.inference_ms >= 0.0
                && routing.ranked_candidates.len() == candidate_ids.len()
                && routing
                    .ranked_candidates
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    == candidate_ids.len()
                && routing
                    .ranked_candidates
                    .iter()
                    .all(|candidate| candidates.contains(candidate.as_str()))
                && routing.top_four_candidates
                    == routing
                        .ranked_candidates
                        .iter()
                        .take(4)
                        .cloned()
                        .collect::<Vec<_>>()
                && !routing.selected_candidates.is_empty()
                && routing.selected_candidates.len() <= candidate_ids.len()
                && routing
                    .selected_candidates
                    .iter()
                    .all(|candidate| candidates.contains(candidate.as_str()))
                && routing
                    .selected_candidates
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    == routing.selected_candidates.len() =>
        {
            routing
                .selected_candidates
                .iter()
                .map(String::as_str)
                .collect()
        }
        _ => {
            return Err(DopeError::Data(format!(
                "invalid routing decision in gold block {}",
                record.dataset_id
            )));
        }
    };
    let optional_finite = |value: Option<f64>| value.is_none_or(f64::is_finite);
    let mut identities = BTreeSet::new();
    for cell in &block.evidence {
        let identity = (
            cell.candidate_id.as_str(),
            cell.auditor_id.as_str(),
            cell.size_multiplier,
            cell.generation_seed,
            cell.auditor_seed,
        );
        if cell.format != "dope-job-evidence"
            || !(2..=JOB_EVIDENCE_VERSION).contains(&cell.version)
            || cell.phase != phase
            || cell.task != record.task
            || cell.lineage_group_id != record.lineage_group_id
            || cell.structural_profile != record.structural_profile
            || cell.train_rows == 0
            || cell.features == 0
            || !candidates.contains(cell.candidate_id.as_str())
            || !auditors.contains(cell.auditor_id.as_str())
            || ![1, 4].contains(&cell.size_multiplier)
            || ![1829, 99238, 196647].contains(&cell.generation_seed)
            || ![57721, 161803, 271828].contains(&cell.auditor_seed)
            || cell.routed != routed_candidates.contains(cell.candidate_id.as_str())
            || ![cell.null_loss, cell.trtr_loss, cell.tstr_loss]
                .into_iter()
                .all(|value| value.is_finite() && value >= 0.0)
            || !is_lower_hex(&cell.artifact_sha256, &[64])
            || !is_lower_hex(&cell.artifact_blake3, &[64])
            || ![
                cell.calibration_degradation,
                cell.rare_class_or_tail_retention,
                cell.supported_subgroup_retention,
                cell.nominal_95_coverage,
                cell.driver_agreement,
                cell.joint_fidelity,
                cell.query_p95_normalized_error,
                cell.type_i_error,
                cell.membership_auc,
                cell.attribute_inference_advantage,
                cell.feature_importance_spearman,
            ]
            .into_iter()
            .all(optional_finite)
            || !cell.feature_importance_top_k_agreement.is_finite()
            || !(0.0..=1.0).contains(&cell.feature_importance_top_k_agreement)
            || cell.feature_importance_feature_count == 0
            || cell.feature_importance_feature_count > cell.features
            || cell.feature_importance_informative_count > cell.feature_importance_feature_count
            || cell.peak_cpu_memory_bytes != cell.peak_memory_bytes
            || cell
                .fitting_time_ms
                .saturating_add(cell.sampling_time_ms)
                .saturating_add(cell.auditor_time_ms)
                > cell.runtime_ms
            || !identities.insert(identity)
        {
            return Err(DopeError::Data(format!(
                "invalid gold evidence in record block {}",
                record.dataset_id
            )));
        }
    }
    let expected_cells = candidate_ids.len() * auditor_ids.len() * 2 * 3 * 3;
    if block.format != "dope-gold-record-block"
        || block.version != GOLD_RECORD_BLOCK_VERSION
        || block.dataset_id != record.dataset_id
        || block.matrix_identity != expected_identity
        || block.evidence.len() + block.failures.len() != expected_cells
        || block.failures.iter().any(|failure| {
            failure.dataset_id != record.dataset_id
                || !candidates.contains(failure.candidate_id.as_str())
                || !auditors.contains(failure.auditor_id.as_str())
                || failure.error.is_empty()
        })
    {
        return Err(DopeError::Data(format!(
            "invalid or incomplete gold record block {}",
            record.dataset_id
        )));
    }
    Ok(())
}

pub fn sign_gold_blocks(
    out: &Path,
    cohort_path: &Path,
    worker: &str,
    source_sha256: &str,
    evaluator_binary_sha256: &str,
    router_bundle: Option<&Path>,
    receipt_key: &Path,
) -> Result<GoldReceiptSummary> {
    if !matches!(worker, "xbabe1" | "xbabe2" | "xbabe3")
        || !is_lower_hex(source_sha256, &[64])
        || !is_lower_hex(evaluator_binary_sha256, &[64])
    {
        return Err(DopeError::Data(
            "invalid gold receipt signer identity".into(),
        ));
    }
    let key = load_receipt_key(receipt_key)?;
    let cohort: CohortPlan = read_json(cohort_path)?;
    let phase = match cohort.kind.as_str() {
        "training-gold" => "training_gold",
        "validation-select" => "validation_select",
        "validation-cert" => "validation_cert",
        _ => return Err(DopeError::Data("unsupported gold receipt cohort".into())),
    };
    let router_bundle_sha256 = router_bundle
        .map(file_hashes)
        .transpose()?
        .map(|hashes| hashes.sha256);
    if (phase == "validation_cert") != router_bundle_sha256.is_some() {
        return Err(DopeError::Data(
            "validation-cert receipts require exactly one frozen router bundle".into(),
        ));
    }
    let cohort_hashes = file_hashes(cohort_path)?;
    let records = cohort
        .records
        .iter()
        .map(|record| (record.dataset_id.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    if records.len() != cohort.records.len() {
        return Err(DopeError::Data(
            "gold receipt cohort contains duplicate dataset IDs".into(),
        ));
    }
    let candidate_ids = empirical_backends()
        .into_iter()
        .map(|candidate| candidate.id)
        .collect::<Vec<_>>();
    let auditor_ids = auditor_specs()
        .into_iter()
        .map(|auditor| auditor.id)
        .collect::<Vec<_>>();
    let size_multipliers = [1usize, 4];
    let generation_seeds = [1829u64, 99238, 196647];
    let auditor_seeds = [57721u64, 161803, 271828];
    let block_dir = out.join("blocks");
    let receipt_dir = out.join("receipts");
    fs::create_dir_all(&receipt_dir).map_err(|error| io_error(&receipt_dir, error))?;
    let mut block_paths = fs::read_dir(&block_dir)
        .map_err(|error| io_error(&block_dir, error))?
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .map_err(|error| io_error(&block_dir, error))
        })
        .collect::<Result<Vec<_>>>()?;
    block_paths.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "json")
    });
    block_paths.sort();
    let mut evidence_cells = 0usize;
    let mut failed_cells = 0usize;
    for block_path in &block_paths {
        let block: GoldRecordBlock = read_json(block_path)?;
        let record = records.get(block.dataset_id.as_str()).ok_or_else(|| {
            DopeError::Data(format!(
                "gold block dataset {} is absent from its cohort",
                block.dataset_id
            ))
        })?;
        let matrix_identity = gold_record_matrix_identity(&GoldMatrixIdentity {
            phase,
            dataset_id: &record.dataset_id,
            candidate_ids: &candidate_ids,
            auditor_ids: &auditor_ids,
            size_multipliers: &size_multipliers,
            generation_seeds: &generation_seeds,
            auditor_seeds: &auditor_seeds,
            primary_only: false,
            neural_target_weight: block.neural_target_weight,
            neural_structural_penalty: block.neural_structural_penalty,
            routing_sha256: router_bundle_sha256.as_deref(),
        })?;
        if block_path.file_stem().and_then(|value| value.to_str()) != Some(&matrix_identity) {
            return Err(DopeError::Data(format!(
                "gold block filename does not match its identity at {}",
                block_path.display()
            )));
        }
        validate_gold_record_block(
            &block,
            record,
            phase,
            &matrix_identity,
            &candidate_ids,
            &auditor_ids,
            router_bundle_sha256.as_deref(),
        )?;
        let block_hashes = file_hashes(block_path)?;
        let receipt_path = receipt_dir.join(format!("{matrix_identity}.json"));
        if receipt_path.is_file() {
            let receipt: GoldBlockReceipt = read_json(&receipt_path)?;
            receipt.verify(&key)?;
            if receipt.dataset_id != block.dataset_id
                || receipt.matrix_identity != matrix_identity
                || receipt.phase != phase
                || receipt.worker != worker
                || receipt.source_sha256 != source_sha256
                || receipt.evaluator_binary_sha256 != evaluator_binary_sha256
                || receipt.router_bundle_sha256 != router_bundle_sha256
                || receipt.cohort != cohort_hashes
                || receipt.block != block_hashes
                || receipt.evidence_cells != block.evidence.len()
                || receipt.failed_cells != block.failures.len()
            {
                return Err(DopeError::Data(format!(
                    "gold receipt does not reconcile at {}",
                    receipt_path.display()
                )));
            }
        } else {
            let mut receipt = GoldBlockReceipt {
                format: "dope-gold-block-receipt".into(),
                version: GOLD_BLOCK_RECEIPT_VERSION,
                dataset_id: block.dataset_id.clone(),
                matrix_identity: matrix_identity.clone(),
                phase: phase.into(),
                worker: worker.into(),
                source_sha256: source_sha256.into(),
                evaluator_binary_sha256: evaluator_binary_sha256.into(),
                router_bundle_sha256: router_bundle_sha256.clone(),
                cohort: cohort_hashes.clone(),
                block: block_hashes,
                evidence_cells: block.evidence.len(),
                failed_cells: block.failures.len(),
                signed_unix_seconds: unix_now(),
                signature: None,
            };
            receipt.sign(&key)?;
            write_canonical(&receipt_path, &receipt)?;
        }
        evidence_cells += block.evidence.len();
        failed_cells += block.failures.len();
    }
    let summary = GoldReceiptSummary {
        format: "dope-gold-receipt-summary".into(),
        version: 1,
        worker: worker.into(),
        blocks: block_paths.len(),
        evidence_cells,
        failed_cells,
        cohort: cohort_hashes,
    };
    write_canonical(
        &out.join(format!("receipt-summary-{worker}.json")),
        &summary,
    )?;
    Ok(summary)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GoldBlockStatus {
    pub format: String,
    pub version: u8,
    pub phase: String,
    pub eligible_lineages: usize,
    pub completed_lineages: usize,
    pub missing_lineages: usize,
    pub attempted_cells: usize,
    pub completed_cells: usize,
    pub failed_cells: usize,
    pub throughput_lineages_per_hour: Option<f64>,
    pub coverage: f64,
    pub candidate_successes: BTreeMap<String, usize>,
    pub auditor_successes: BTreeMap<String, usize>,
    pub invalid_blocks: Vec<String>,
    pub router_regret: Option<f64>,
    pub ptf_v1: Option<f64>,
    pub production_score_available: bool,
    pub updated_unix_seconds: u64,
}

pub fn gold_block_status(
    cohort_path: &Path,
    gold_out: &Path,
    router_bundle: Option<&Path>,
    metrics_path: Option<&Path>,
) -> Result<GoldBlockStatus> {
    let cohort: CohortPlan = read_json(cohort_path)?;
    let phase = match cohort.kind.as_str() {
        "training-gold" => "training_gold",
        "validation-select" => "validation_select",
        "validation-cert" => "validation_cert",
        _ => return Err(DopeError::Data("unsupported gold status cohort".into())),
    };
    let routing_sha256 = router_bundle
        .map(file_hashes)
        .transpose()?
        .map(|hash| hash.sha256);
    if (phase == "validation_cert") != routing_sha256.is_some() {
        return Err(DopeError::Data(
            "validation-cert block status requires its frozen router bundle".into(),
        ));
    }
    let candidate_ids = empirical_backends()
        .into_iter()
        .map(|candidate| candidate.id)
        .collect::<Vec<_>>();
    let auditor_ids = auditor_specs()
        .into_iter()
        .map(|auditor| auditor.id)
        .collect::<Vec<_>>();
    let size_multipliers = [1usize, 4];
    let generation_seeds = [1829u64, 99238, 196647];
    let auditor_seeds = [57721u64, 161803, 271828];
    let cells_per_record = candidate_ids.len()
        * auditor_ids.len()
        * size_multipliers.len()
        * generation_seeds.len()
        * auditor_seeds.len();
    let mut completed_lineages = 0usize;
    let mut completed_cells = 0usize;
    let mut failed_cells = 0usize;
    let mut candidate_successes = candidate_ids
        .iter()
        .map(|candidate| (candidate.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut auditor_successes = auditor_ids
        .iter()
        .map(|auditor| (auditor.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut invalid_blocks = Vec::new();
    let mut first_modified = None::<SystemTime>;
    let mut last_modified = None::<SystemTime>;
    for record in &cohort.records {
        let matrix_identity = gold_record_matrix_identity(&GoldMatrixIdentity {
            phase,
            dataset_id: &record.dataset_id,
            candidate_ids: &candidate_ids,
            auditor_ids: &auditor_ids,
            size_multipliers: &size_multipliers,
            generation_seeds: &generation_seeds,
            auditor_seeds: &auditor_seeds,
            primary_only: false,
            neural_target_weight: 2.0,
            neural_structural_penalty: 0.0,
            routing_sha256: routing_sha256.as_deref(),
        })?;
        let path = gold_out
            .join("blocks")
            .join(format!("{matrix_identity}.json"));
        if !path.is_file() {
            continue;
        }
        let block = match read_json::<GoldRecordBlock>(&path).and_then(|block| {
            validate_gold_record_block(
                &block,
                record,
                phase,
                &matrix_identity,
                &candidate_ids,
                &auditor_ids,
                routing_sha256.as_deref(),
            )?;
            Ok(block)
        }) {
            Ok(block) => block,
            Err(error) => {
                invalid_blocks.push(format!("{}: {error}", record.dataset_id));
                continue;
            }
        };
        completed_lineages += 1;
        completed_cells += block.evidence.len();
        failed_cells += block.failures.len();
        for cell in block.evidence {
            *candidate_successes
                .get_mut(&cell.candidate_id)
                .expect("validated candidate") += 1;
            *auditor_successes
                .get_mut(&cell.auditor_id)
                .expect("validated auditor") += 1;
        }
        let modified = fs::metadata(&path)
            .map_err(|error| io_error(&path, error))?
            .modified()
            .map_err(|error| io_error(&path, error))?;
        first_modified = Some(first_modified.map_or(modified, |current| current.min(modified)));
        last_modified = Some(last_modified.map_or(modified, |current| current.max(modified)));
    }
    let throughput_lineages_per_hour = first_modified
        .zip(last_modified)
        .and_then(|(first, last)| last.duration_since(first).ok())
        .filter(|elapsed| !elapsed.is_zero())
        .map(|elapsed| completed_lineages as f64 / elapsed.as_secs_f64() * 3_600.0);
    let metrics = metrics_path.map(read_json::<MetricsBundle>).transpose()?;
    let router_regret = metrics
        .as_ref()
        .and_then(|metrics| metrics.kpi.candidate_regret);
    let ptf_v1 = metrics.as_ref().and_then(|metrics| metrics.kpi.ptf_v1);
    let production_score_available = metrics
        .as_ref()
        .is_some_and(|metrics| metrics.kpi.production_score_available);
    Ok(GoldBlockStatus {
        format: "dope-gold-block-status".into(),
        version: 1,
        phase: phase.into(),
        eligible_lineages: cohort.records.len(),
        completed_lineages,
        missing_lineages: cohort.records.len().saturating_sub(completed_lineages),
        attempted_cells: cohort.records.len() * cells_per_record,
        completed_cells,
        failed_cells,
        throughput_lineages_per_hour,
        coverage: completed_lineages as f64 / cohort.records.len().max(1) as f64,
        candidate_successes,
        auditor_successes,
        invalid_blocks,
        router_regret,
        ptf_v1,
        production_score_available,
        updated_unix_seconds: unix_now(),
    })
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CampaignPhase {
    Frozen,
    RouterFrozen,
    ValidationCertOpen,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CampaignState {
    pub format: String,
    pub version: u8,
    pub phase: CampaignPhase,
    pub source_commit: String,
    pub source_tree: ContentHashes,
    pub environment_lock: ContentHashes,
    pub host_inventory: ContentHashes,
    pub source_hashes: ContentHashes,
    pub sbom: ContentHashes,
    pub notices: ContentHashes,
    pub checkpoint_manifest: ContentHashes,
    pub corpus_manifest: ContentHashes,
    pub corpus_manifest_checksum: String,
    pub validation_manifest: ContentHashes,
    pub validation_select_lineage_groups: usize,
    pub validation_cert_lineage_groups: usize,
    pub kpi_contract_sha256: String,
    pub receipt_key_sha256: String,
    pub router_bundle: Option<ContentHashes>,
    pub router_evidence: Option<ContentHashes>,
    pub validation_cert_open_count: usize,
    pub sealed_test_open_count: usize,
    pub frozen_unix_seconds: u64,
}

impl CampaignState {
    pub fn validate(&self) -> Result<()> {
        if self.format != "dope-campaign-state"
            || self.version != 1
            || !is_lower_hex(&self.source_commit, &[40, 64])
            || self.kpi_contract_sha256 != KPI_CONTRACT_SHA256
            || self.validation_select_lineage_groups != 11_892
            || self.validation_cert_lineage_groups != 7_820
            || self.validation_cert_open_count > 1
            || self.sealed_test_open_count != 0
            || (self.phase == CampaignPhase::Frozen
                && (self.router_bundle.is_some() || self.router_evidence.is_some()))
            || (self.phase != CampaignPhase::Frozen
                && (self.router_bundle.is_none() || self.router_evidence.is_none()))
            || (self.phase == CampaignPhase::ValidationCertOpen
                && self.validation_cert_open_count != 1)
        {
            return Err(DopeError::Data("invalid durable campaign state".into()));
        }
        Ok(())
    }
}

pub struct FreezeOptions<'a> {
    pub state_dir: &'a Path,
    pub specs: &'a Path,
    pub source_commit: &'a str,
    pub source_tree: &'a Path,
    pub environment_lock: &'a Path,
    pub host_inventory: &'a Path,
    pub source_hashes: &'a Path,
    pub sbom: &'a Path,
    pub notices: &'a Path,
    pub checkpoint_manifest: &'a Path,
    pub corpus_manifest: &'a Path,
    pub validation_manifest: &'a Path,
    pub receipt_key: &'a Path,
}

fn state_path(state_dir: &Path) -> PathBuf {
    state_dir.join("campaign-state.json")
}

pub fn ledger_path(state_dir: &Path) -> PathBuf {
    state_dir.join("campaign.sqlite")
}

pub fn load_state(state_dir: &Path) -> Result<CampaignState> {
    let state: CampaignState = read_json(&state_path(state_dir))?;
    state.validate()?;
    Ok(state)
}

fn validate_environment(path: &Path) -> Result<()> {
    let value: Value = read_json(path)?;
    let object = value
        .as_object()
        .ok_or_else(|| DopeError::Data("environment lock must be an object".into()))?;
    let forbidden = [
        "python",
        "python_packages",
        "sklearn",
        "xgboost",
        "catboost",
        "tabpfn",
    ];
    if object.get("production_language").and_then(Value::as_str) != Some("rust-only")
        || object.get("rust").and_then(Value::as_str) != Some("1.88.0")
        || object.get("cuda").and_then(Value::as_str) != Some("12.8")
        || object
            .get("gpu_training")
            .and_then(Value::as_object)
            .and_then(|gpu| gpu.get("libtorch"))
            .and_then(Value::as_str)
            != Some("2.7.0")
        || object
            .get("gpu_training")
            .and_then(Value::as_object)
            .and_then(|gpu| gpu.get("tch_rs"))
            .and_then(Value::as_str)
            != Some("0.20.0")
        || object
            .get("gpu_training")
            .and_then(Value::as_object)
            .and_then(|gpu| gpu.get("cublas_workspace_config"))
            .and_then(Value::as_str)
            != Some(":4096:8")
        || object
            .get("gpu_training")
            .and_then(Value::as_object)
            .and_then(|gpu| gpu.get("cudnn_benchmark"))
            .and_then(Value::as_bool)
            != Some(false)
        || object
            .get("gpu_training")
            .and_then(Value::as_object)
            .and_then(|gpu| gpu.get("deterministic_seeds_required"))
            .and_then(Value::as_bool)
            != Some(true)
        || forbidden.iter().any(|key| object.contains_key(*key))
    {
        return Err(DopeError::Data(
            "environment lock is not the frozen Rust 1.88/tch-rs/libtorch 2.7/CUDA 12.8 contract"
                .into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct WorkerInventory {
    pub host: String,
    pub rust: String,
    pub cuda: String,
    pub driver: String,
    pub gpu: String,
    pub source_sha256: String,
    pub binary_sha256: String,
    pub corpus_manifest_sha256: String,
    pub corpus_sample_sha256: String,
    #[serde(default)]
    pub available_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct HostInventory {
    pub format: String,
    pub version: u8,
    pub tch_rs: String,
    pub libtorch: String,
    pub workers: Vec<WorkerInventory>,
}

impl HostInventory {
    pub fn validate(&self, source_sha256: &str, corpus_sha256: &str) -> Result<()> {
        let hosts = self
            .workers
            .iter()
            .map(|worker| worker.host.as_str())
            .collect::<BTreeSet<_>>();
        let samples = self
            .workers
            .iter()
            .map(|worker| worker.corpus_sample_sha256.as_str())
            .collect::<BTreeSet<_>>();
        if self.format != "dope-host-inventory"
            || self.version != 1
            || self.tch_rs != "0.20.0"
            || self.libtorch != "2.7.0"
            || hosts != BTreeSet::from(["xbabe1", "xbabe2", "xbabe3"])
            || samples.len() != 1
            || self.workers.iter().any(|worker| {
                worker.rust != "1.88.0"
                    || worker.cuda != "12.8"
                    || worker.driver.is_empty()
                    || worker.gpu.is_empty()
                    || worker.source_sha256 != source_sha256
                    || worker.corpus_manifest_sha256 != corpus_sha256
                    || !is_lower_hex(&worker.binary_sha256, &[64])
                    || !is_lower_hex(&worker.corpus_sample_sha256, &[64])
                    || (matches!(worker.host.as_str(), "xbabe1" | "xbabe3")
                        && worker.available_bytes < MIN_DURABLE_FREE_GIB * GIB)
            })
        {
            return Err(DopeError::Data(
                "host inventory or deterministic corpus sample differs across xbabe workers".into(),
            ));
        }
        Ok(())
    }
}

fn copy_file(source: &Path, destination: &Path) -> Result<ContentHashes> {
    fs::copy(source, destination).map_err(|error| io_error(destination, error))?;
    file_hashes(destination)
}

pub fn freeze(options: &FreezeOptions<'_>) -> Result<CampaignState> {
    if !is_lower_hex(options.source_commit, &[40, 64]) {
        return Err(DopeError::Data(
            "source commit must be lowercase hexadecimal".into(),
        ));
    }
    fs::create_dir_all(options.state_dir).map_err(|error| io_error(options.state_dir, error))?;
    require_durable_space(options.state_dir)?;
    if state_path(options.state_dir).exists() {
        return Err(DopeError::Data("campaign is already frozen".into()));
    }
    validate_environment(options.environment_lock)?;
    let corpus: SplitManifest = read_json(options.corpus_manifest)?;
    validate_split_manifest(&corpus)?;
    let validation: ValidationSubmanifest = read_json(options.validation_manifest)?;
    validate_validation_submanifest(&validation)?;
    if validation.source_manifest_checksum != corpus.checksum
        || validation.validation_select_lineage_groups != 11_892
        || validation.validation_cert_lineage_groups != 7_820
    {
        return Err(DopeError::Data(
            "validation manifest does not match the frozen corpus lineage allocation".into(),
        ));
    }
    let source_tree = file_hashes(options.source_tree)?;
    let environment_lock = file_hashes(options.environment_lock)?;
    let corpus_manifest = file_hashes(options.corpus_manifest)?;
    let validation_manifest = file_hashes(options.validation_manifest)?;
    let host_inventory: HostInventory = read_json(options.host_inventory)?;
    host_inventory.validate(&source_tree.sha256, &corpus_manifest.sha256)?;
    let host_inventory_hash = file_hashes(options.host_inventory)?;
    let source_hashes = file_hashes(options.source_hashes)?;
    let sbom = file_hashes(options.sbom)?;
    let notices = file_hashes(options.notices)?;
    let checkpoint_manifest = file_hashes(options.checkpoint_manifest)?;
    let receipt_key = load_receipt_key(options.receipt_key)?;
    let receipt_key_sha256 = hashes(&receipt_key).sha256;
    let specs = CampaignLedger::load_specs(options.specs)?;
    let candidates = empirical_backends()
        .into_iter()
        .map(|backend| backend.id)
        .collect::<BTreeSet<_>>();
    let auditors = auditor_specs()
        .into_iter()
        .map(|auditor| auditor.id)
        .collect::<BTreeSet<_>>();
    for spec in &specs {
        spec.validate()?;
        if spec.source_commit != options.source_commit
            || spec.environment_lock_sha256 != environment_lock.sha256
            || spec.corpus_manifest_sha256 != corpus_manifest.sha256
            || spec.kpi_contract_sha256 != KPI_CONTRACT_SHA256
            || !candidates.contains(&spec.candidate_spec)
            || !auditors.contains(&spec.auditor_spec)
        {
            return Err(DopeError::Data(
                "job specification differs from frozen campaign inputs".into(),
            ));
        }
    }
    copy_file(options.source_tree, &options.state_dir.join("source.tar"))?;
    copy_file(
        options.environment_lock,
        &options.state_dir.join("environment-lock.json"),
    )?;
    copy_file(
        options.host_inventory,
        &options.state_dir.join("host-inventory.json"),
    )?;
    copy_file(
        options.source_hashes,
        &options.state_dir.join("source-hashes.json"),
    )?;
    copy_file(options.sbom, &options.state_dir.join("sbom.json"))?;
    copy_file(options.notices, &options.state_dir.join("notices.txt"))?;
    copy_file(
        options.checkpoint_manifest,
        &options.state_dir.join("checkpoint-manifest.json"),
    )?;
    copy_file(
        options.corpus_manifest,
        &options.state_dir.join("corpus-manifest.json"),
    )?;
    copy_file(
        options.validation_manifest,
        &options.state_dir.join("validation-manifest.json"),
    )?;
    copy_file(options.specs, &options.state_dir.join("job-specs.json"))?;
    let ledger = CampaignLedger::initialize(&ledger_path(options.state_dir))?;
    ledger.insert_specs(&specs)?;
    let state = CampaignState {
        format: "dope-campaign-state".into(),
        version: 1,
        phase: CampaignPhase::Frozen,
        source_commit: options.source_commit.into(),
        source_tree,
        environment_lock,
        host_inventory: host_inventory_hash,
        source_hashes,
        sbom,
        notices,
        checkpoint_manifest,
        corpus_manifest,
        corpus_manifest_checksum: corpus.checksum,
        validation_manifest,
        validation_select_lineage_groups: validation.validation_select_lineage_groups,
        validation_cert_lineage_groups: validation.validation_cert_lineage_groups,
        kpi_contract_sha256: KPI_CONTRACT_SHA256.into(),
        receipt_key_sha256,
        router_bundle: None,
        router_evidence: None,
        validation_cert_open_count: 0,
        sealed_test_open_count: 0,
        frozen_unix_seconds: unix_now(),
    };
    state.validate()?;
    write_canonical(&state_path(options.state_dir), &state)?;
    Ok(state)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RouterEvidence {
    pub format: String,
    pub version: u8,
    pub top_four_oracle_recall: f64,
    pub maximum_profile_regret_upper: f64,
    pub beats_random: bool,
    pub beats_best_fixed: bool,
    pub beats_ga2m: bool,
    pub paired_hypervolume_improvement: f64,
    pub paired_hypervolume_ci_lower: f64,
    pub student_max_abs_difference: f64,
    pub top_choice_agreement: f64,
    pub bundle_bytes: u64,
    pub inference_p95_ms: f64,
    pub sketch_p95_ms: f64,
    pub training_evidence_sha256: String,
    #[serde(default, deserialize_with = "deserialize_nullable_string")]
    pub best_fixed_candidate_id: String,
    #[serde(default)]
    pub student_mean_regret: f64,
    #[serde(default)]
    pub random_mean_regret: f64,
    #[serde(default)]
    pub best_fixed_mean_regret: f64,
    #[serde(default)]
    pub ga2m_mean_regret: f64,
}

fn deserialize_nullable_string<'de, D>(deserializer: D) -> std::result::Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
}

impl RouterEvidence {
    pub fn failed_gates(&self) -> Vec<String> {
        let mut failed = Vec::new();
        let mut gate = |name: &str, passed: bool| {
            if !passed {
                failed.push(name.into());
            }
        };
        gate(
            "top_four_oracle_recall",
            self.top_four_oracle_recall >= 0.99,
        );
        gate("profile_regret", self.maximum_profile_regret_upper <= 0.02);
        gate("beats_random", self.beats_random);
        gate("beats_best_fixed", self.beats_best_fixed);
        gate("beats_ga2m", self.beats_ga2m);
        gate(
            "paired_hypervolume",
            self.paired_hypervolume_improvement >= 0.05 && self.paired_hypervolume_ci_lower > 0.0,
        );
        gate(
            "student_parity",
            self.student_max_abs_difference <= 1e-3 && self.top_choice_agreement >= 0.999,
        );
        gate(
            "router_bundle_bytes",
            self.bundle_bytes <= MAX_ROUTER_BUNDLE_BYTES as u64,
        );
        gate("router_inference_p95", self.inference_p95_ms <= 5.0);
        gate("sketch_p95", self.sketch_p95_ms <= 1_000.0);
        gate(
            "router_training_evidence",
            is_lower_hex(&self.training_evidence_sha256, &[64]),
        );
        gate(
            "best_fixed_candidate",
            empirical_backends()
                .iter()
                .any(|candidate| candidate.id == self.best_fixed_candidate_id),
        );
        failed
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RouterBenchmarkReport {
    pub format: String,
    pub version: u8,
    pub rows: usize,
    pub features: usize,
    pub repeats: usize,
    pub sketch_p95_ms: f64,
    pub router_inference_p95_ms: Option<f64>,
    pub router_bundle_bytes: Option<u64>,
    pub deterministic_inference: Option<bool>,
}

pub fn benchmark_router(
    router_bundle: Option<&Path>,
    rows: usize,
    features: usize,
    repeats: usize,
) -> Result<RouterBenchmarkReport> {
    if rows == 0 || !(1..=2_000).contains(&features) || repeats < 2 {
        return Err(DopeError::Data(
            "router benchmark requires rows > 0, 1..=2000 features, and at least two repeats"
                .into(),
        ));
    }
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        (value >> 40) as f32 / (1u32 << 24) as f32
    };
    let values = (0..rows * features).map(|_| next()).collect::<Vec<_>>();
    let target = (0..rows)
        .map(|row| {
            (0..features.min(8))
                .map(|feature| values[row * features + feature])
                .sum::<f32>()
                / features.min(8) as f32
        })
        .collect::<Vec<_>>();
    let table = Table::from_arrays(&values, &target, rows, features, Task::Regression)?;
    let mut sketches = Vec::with_capacity(repeats);
    let mut sketch_times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        sketches.push(DatasetSketch::from_train(&table, Task::Regression));
        sketch_times.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    let sketch_p95_ms = float_percentile(&mut sketch_times, 0.95).unwrap_or(f64::INFINITY);
    let (router_inference_p95_ms, router_bundle_bytes, deterministic_inference) =
        if let Some(path) = router_bundle {
            let bundle: RouterBundle = read_json(path)?;
            let router = QuantizedRouter::new(bundle)?;
            let mut prediction_hashes = BTreeSet::new();
            let mut inference_times = Vec::with_capacity(repeats);
            for sketch in &sketches {
                let started = Instant::now();
                let prediction = router.predict(sketch)?;
                inference_times.push(started.elapsed().as_secs_f64() * 1_000.0);
                prediction_hashes.insert(hashes(&canonical_json(&prediction)?).sha256);
            }
            (
                float_percentile(&mut inference_times, 0.95),
                Some(
                    fs::metadata(path)
                        .map_err(|error| io_error(path, error))?
                        .len(),
                ),
                Some(prediction_hashes.len() == 1),
            )
        } else {
            (None, None, None)
        };
    Ok(RouterBenchmarkReport {
        format: "dope-router-benchmark".into(),
        version: 1,
        rows,
        features,
        repeats,
        sketch_p95_ms,
        router_inference_p95_ms,
        router_bundle_bytes,
        deterministic_inference,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct CachedSketch {
    format: String,
    version: u8,
    lineage_group_id: String,
    structural_profile: String,
    sketch: DatasetSketch,
    compute_ms: f64,
}

#[derive(Clone, Debug, Default)]
struct RouterCellAggregate {
    seen_replicates: u128,
    count: usize,
    retention_sum: f64,
    runtime_ms_sum: f64,
    peak_memory_bytes_sum: f64,
    artifact_bytes_sum: f64,
}

fn router_bounded_retention(cell: &JobEvidence) -> f64 {
    let improvement = cell.null_loss - cell.trtr_loss;
    let informative = improvement >= 0.01 * cell.null_loss.abs();
    if informative && improvement.abs() > f64::EPSILON {
        // Router utility is bounded: exceeding TRTR is a full success, while
        // negative retention has no extra routing value. KPI aggregation
        // retains the required unclamped statistic.
        ((cell.null_loss - cell.tstr_loss) / improvement).clamp(0.0, 1.0)
    } else {
        f64::from(cell.tstr_loss <= cell.trtr_loss + 0.01 * cell.null_loss)
    }
}

fn router_replicate_slot(cell: &JobEvidence, auditor_ids: &[String]) -> Result<usize> {
    let auditor = auditor_ids
        .iter()
        .position(|id| id == &cell.auditor_id)
        .ok_or_else(|| DopeError::Data(format!("unknown router auditor {}", cell.auditor_id)))?;
    let size = [1, 4]
        .iter()
        .position(|&value| value == cell.size_multiplier)
        .ok_or_else(|| DopeError::Data("router metric has a non-gold size multiplier".into()))?;
    let generation = [1829, 99238, 196647]
        .iter()
        .position(|&value| value == cell.generation_seed)
        .ok_or_else(|| DopeError::Data("router metric has an unfrozen generation seed".into()))?;
    let auditor_seed = [57721, 161803, 271828]
        .iter()
        .position(|&value| value == cell.auditor_seed)
        .ok_or_else(|| DopeError::Data("router metric has an unfrozen auditor seed".into()))?;
    Ok((((auditor * 2) + size) * 3 + generation) * 3 + auditor_seed)
}

pub fn train_router_from_metrics(
    cohort_paths: &[PathBuf],
    metric_paths: &[PathBuf],
    sketch_cache: Option<&Path>,
    bundle_out: &Path,
    report_out: &Path,
    evidence_out: &Path,
) -> Result<(RouterTrainingReport, RouterEvidence)> {
    if cohort_paths.is_empty() || metric_paths.is_empty() {
        return Err(DopeError::Data(
            "router cohort and metrics inputs must be nonempty".into(),
        ));
    }
    let cohorts = cohort_paths
        .iter()
        .map(|path| read_json::<CohortPlan>(path))
        .collect::<Result<Vec<_>>>()?;
    if cohorts
        .iter()
        .any(|cohort| !matches!(cohort.kind.as_str(), "training-gold" | "validation-select"))
    {
        return Err(DopeError::Data(
            "router training accepts only training-gold and validation-select cohorts".into(),
        ));
    }
    let training_lineage_ids = cohorts
        .iter()
        .filter(|cohort| cohort.kind == "training-gold")
        .flat_map(|cohort| &cohort.records)
        .map(|record| record.lineage_group_id.clone())
        .collect::<BTreeSet<_>>();
    let validation_lineage_ids = cohorts
        .iter()
        .filter(|cohort| cohort.kind == "validation-select")
        .flat_map(|cohort| &cohort.records)
        .map(|record| record.lineage_group_id.clone())
        .collect::<BTreeSet<_>>();
    if !training_lineage_ids.is_disjoint(&validation_lineage_ids) {
        return Err(DopeError::Data(
            "router training and validation-select lineages overlap".into(),
        ));
    }
    let records = cohorts
        .iter()
        .flat_map(|cohort| &cohort.records)
        .map(|record| {
            (
                (
                    record.lineage_group_id.as_str(),
                    record.structural_profile.as_str(),
                ),
                record,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let records_by_dataset = cohorts
        .iter()
        .flat_map(|cohort| &cohort.records)
        .map(|record| (record.dataset_id.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    let candidate_ids = empirical_backends()
        .into_iter()
        .map(|backend| backend.id)
        .collect::<Vec<_>>();
    let candidate_set = candidate_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let auditor_ids = auditor_specs()
        .into_iter()
        .map(|auditor| auditor.id)
        .collect::<Vec<_>>();
    let mut grouped = BTreeMap::<(String, String, String), RouterCellAggregate>::new();
    let mut sketch_jobs = BTreeMap::new();
    let mut input_cells = 0usize;
    let mut expanded_metric_paths = Vec::new();
    for path in metric_paths {
        if path.is_dir() {
            let mut entries = fs::read_dir(path)
                .map_err(|error| io_error(path, error))?
                .map(|entry| {
                    entry
                        .map(|entry| entry.path())
                        .map_err(|error| io_error(path, error))
                })
                .collect::<Result<Vec<_>>>()?;
            entries.retain(|entry| {
                entry
                    .extension()
                    .is_some_and(|extension| extension == "json")
            });
            entries.sort();
            expanded_metric_paths.extend(entries);
        } else {
            expanded_metric_paths.push(path.clone());
        }
    }
    if expanded_metric_paths.is_empty() {
        return Err(DopeError::Data(
            "router metrics inputs contain no evidence files".into(),
        ));
    }
    let mut consume_cell = |cell: JobEvidence| -> Result<()> {
        let record = records
            .get(&(
                cell.lineage_group_id.as_str(),
                cell.structural_profile.as_str(),
            ))
            .ok_or_else(|| {
                DopeError::Data(format!(
                    "router metric outcome {}/{} is absent from its cohort",
                    cell.lineage_group_id, cell.structural_profile
                ))
            })?;
        if !candidate_set.contains(cell.candidate_id.as_str())
            || cell.task != record.task
            || cell.lineage_group_id != record.lineage_group_id
            || cell.structural_profile != record.structural_profile
        {
            return Err(DopeError::Data(
                "router metric does not match its frozen cohort record".into(),
            ));
        }
        let slot = router_replicate_slot(&cell, &auditor_ids)?;
        let bit = 1u128 << slot;
        let aggregate = grouped
            .entry((
                cell.lineage_group_id.clone(),
                cell.structural_profile.clone(),
                cell.candidate_id.clone(),
            ))
            .or_default();
        if aggregate.seen_replicates & bit != 0 {
            return Err(DopeError::Data(format!(
                "duplicate router metric replicate for {}/{}/{}",
                cell.lineage_group_id, cell.structural_profile, cell.candidate_id
            )));
        }
        aggregate.seen_replicates |= bit;
        aggregate.count += 1;
        aggregate.retention_sum += router_bounded_retention(&cell);
        aggregate.runtime_ms_sum += cell.runtime_ms as f64;
        aggregate.peak_memory_bytes_sum += cell.peak_memory_bytes as f64;
        aggregate.artifact_bytes_sum += cell.artifact_bytes as f64;
        sketch_jobs.insert(
            (
                cell.lineage_group_id.clone(),
                cell.structural_profile.clone(),
            ),
            (*record).clone(),
        );
        input_cells += 1;
        Ok(())
    };
    let gold_sizes = [1usize, 4];
    let gold_generation_seeds = [1829u64, 99238, 196647];
    let gold_auditor_seeds = [57721u64, 161803, 271828];
    for path in &expanded_metric_paths {
        let file = File::open(path).map_err(|error| io_error(path, error))?;
        let mut lines = std::io::BufReader::new(file).lines();
        let Some(first_line) = lines.next() else {
            continue;
        };
        let first_line = first_line.map_err(|error| io_error(path, error))?;
        if first_line.trim().is_empty() {
            continue;
        }
        let first_value: Value = serde_json::from_str(&first_line)?;
        if first_value.get("format").and_then(Value::as_str) == Some("dope-gold-record-block") {
            let block: GoldRecordBlock = serde_json::from_value(first_value)?;
            if lines.next().is_some() {
                return Err(DopeError::Data(format!(
                    "gold record block contains trailing data at {}",
                    path.display()
                )));
            }
            let record = records_by_dataset
                .get(block.dataset_id.as_str())
                .ok_or_else(|| {
                    DopeError::Data(format!(
                        "gold record block {} is absent from router cohorts",
                        block.dataset_id
                    ))
                })?;
            let phase = match record.partition.as_str() {
                "training_gold" => "training_gold",
                "validation_select" => "validation_select",
                _ => {
                    return Err(DopeError::Data(
                        "router training cannot consume certification evidence".into(),
                    ));
                }
            };
            let matrix_identity = gold_record_matrix_identity(&GoldMatrixIdentity {
                phase,
                dataset_id: &record.dataset_id,
                candidate_ids: &candidate_ids,
                auditor_ids: &auditor_ids,
                size_multipliers: &gold_sizes,
                generation_seeds: &gold_generation_seeds,
                auditor_seeds: &gold_auditor_seeds,
                primary_only: false,
                neural_target_weight: 2.0,
                neural_structural_penalty: 0.0,
                routing_sha256: None,
            })?;
            if path.file_stem().and_then(|stem| stem.to_str()) != Some(&matrix_identity) {
                return Err(DopeError::Data(format!(
                    "router gold block filename differs from its matrix identity at {}",
                    path.display()
                )));
            }
            validate_gold_record_block(
                &block,
                record,
                phase,
                &matrix_identity,
                &candidate_ids,
                &auditor_ids,
                None,
            )?;
            for cell in block.evidence {
                consume_cell(cell)?;
            }
            continue;
        }
        consume_cell(serde_json::from_value(first_value)?)?;
        for line in lines {
            let line = line.map_err(|error| io_error(path, error))?;
            if !line.trim().is_empty() {
                consume_cell(serde_json::from_str(&line)?)?;
            }
        }
    }
    if let Some(cache) = sketch_cache {
        fs::create_dir_all(cache).map_err(|error| io_error(cache, error))?;
    }
    let sketch_results = sketch_jobs
        .into_par_iter()
        .map(|((lineage, profile), record)| {
            let cache_path = sketch_cache.map(|cache| {
                cache.join(format!(
                    "{}.json",
                    hashes(format!("{}\0{}\0{}", record.dataset_id, lineage, profile).as_bytes())
                        .blake3
                ))
            });
            if let Some(path) = cache_path.as_ref().filter(|path| path.is_file()) {
                let cached: CachedSketch = read_json(path)?;
                if cached.format != "dope-router-sketch-cache"
                    || cached.version != 1
                    || cached.lineage_group_id != lineage
                    || cached.structural_profile != profile
                    || cached.sketch.version != DATASET_SKETCH_VERSION
                    || !cached.compute_ms.is_finite()
                    || cached.compute_ms < 0.0
                {
                    return Err(DopeError::Data(format!(
                        "router sketch cache identity mismatch at {}",
                        path.display()
                    )));
                }
                return Ok(((lineage, profile), cached.sketch, cached.compute_ms));
            }
            let table = Table::read_dataset_dir(&record.dataset_path, table_task(&record.task)?)?;
            let started = Instant::now();
            let sketch = DatasetSketch::from_train(&table, table_task(&record.task)?);
            let compute_ms = started.elapsed().as_secs_f64() * 1_000.0;
            if let Some(path) = cache_path {
                write_canonical(
                    &path,
                    &CachedSketch {
                        format: "dope-router-sketch-cache".into(),
                        version: 1,
                        lineage_group_id: lineage.clone(),
                        structural_profile: profile.clone(),
                        sketch: sketch.clone(),
                        compute_ms,
                    },
                )?;
            }
            Ok(((lineage, profile), sketch, compute_ms))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut sketches = BTreeMap::<(String, String), DatasetSketch>::new();
    let mut sketch_times = Vec::<f64>::new();
    for (key, sketch, compute_ms) in sketch_results {
        sketches.insert(key, sketch);
        sketch_times.push(compute_ms);
    }
    let replicates_per_candidate = grouped.values().map(|value| value.count).max().unwrap_or(0);
    let mut outcome_counts = BTreeMap::<(String, String), BTreeMap<String, usize>>::new();
    for ((lineage, profile, candidate), values) in &grouped {
        outcome_counts
            .entry((lineage.clone(), profile.clone()))
            .or_default()
            .insert(candidate.clone(), values.count);
    }
    let complete_outcomes = outcome_counts
        .iter()
        .filter(|(_, counts)| {
            counts.len() == candidate_ids.len()
                && counts.keys().map(String::as_str).collect::<BTreeSet<_>>() == candidate_set
                && counts
                    .values()
                    .all(|&count| count == replicates_per_candidate)
        })
        .map(|(outcome, _)| outcome.clone())
        .collect::<BTreeSet<_>>();
    let excluded_incomplete_outcomes = outcome_counts.len() - complete_outcomes.len();
    if complete_outcomes.is_empty() {
        return Err(DopeError::Data(
            "router metrics contain no complete lineage/profile outcomes".into(),
        ));
    }
    let mut labels = grouped
        .into_iter()
        .filter(|((lineage, profile, _), _)| {
            complete_outcomes.contains(&(lineage.clone(), profile.clone()))
        })
        .map(|((lineage, profile, candidate), values)| {
            let count = values.count as f64;
            let sketch_key = (lineage.clone(), profile.clone());
            RouterLabel {
                lineage_group_id: lineage,
                structural_profile: profile,
                candidate_id: candidate,
                sketch: sketches[&sketch_key].clone(),
                retention: (values.retention_sum / count) as f32,
                runtime_ms: (values.runtime_ms_sum / count) as f32,
                peak_memory_bytes: (values.peak_memory_bytes_sum / count) as f32,
                artifact_bytes: (values.artifact_bytes_sum / count) as f32,
                failed: false,
            }
        })
        .collect::<Vec<_>>();
    labels.sort_by(|left, right| {
        left.lineage_group_id
            .cmp(&right.lineage_group_id)
            .then_with(|| left.structural_profile.cmp(&right.structural_profile))
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    let training_evidence_sha256 = hashes(&canonical_json(&serde_json::json!({
        "labels": labels,
        "input_cells": input_cells,
        "replicates_per_candidate": replicates_per_candidate,
        "excluded_incomplete_outcomes": excluded_incomplete_outcomes,
        "validation_lineages": validation_lineage_ids
    }))?)
    .sha256;
    let label_lineages = labels
        .iter()
        .map(|label| label.lineage_group_id.clone())
        .collect::<BTreeSet<_>>();
    let effective_validation_lineages = validation_lineage_ids
        .intersection(&label_lineages)
        .cloned()
        .collect::<BTreeSet<_>>();
    let explicit_validation = (!training_lineage_ids.is_empty()
        && !validation_lineage_ids.is_empty())
    .then_some(&effective_validation_lineages);
    let (bundle, mut report) = train_distilled_router(
        &labels,
        &candidate_ids,
        &training_evidence_sha256,
        explicit_validation,
    )?;
    report.input_cells = input_cells;
    report.replicates_per_candidate = replicates_per_candidate;
    report.complete_lineage_profile_outcomes = complete_outcomes.len();
    report.excluded_incomplete_outcomes = excluded_incomplete_outcomes;
    write_canonical(bundle_out, &bundle)?;
    write_canonical(report_out, &report)?;
    let router = QuantizedRouter::new(bundle.clone())?;
    let mut inference_times = Vec::new();
    for sketch in sketches.values() {
        for _ in 0..5 {
            let started = Instant::now();
            let _ = router.predict(sketch)?;
            inference_times.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
    }
    let p95 = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values
            .get(((values.len().saturating_sub(1)) as f64 * 0.95).ceil() as usize)
            .copied()
            .unwrap_or(f64::INFINITY)
    };
    let evidence = RouterEvidence {
        format: "dope-router-evidence".into(),
        version: 1,
        top_four_oracle_recall: report.student_top_four_oracle_recall,
        maximum_profile_regret_upper: report.student_maximum_profile_regret_upper,
        beats_random: report.student_mean_regret < report.random_mean_regret,
        beats_best_fixed: report.student_mean_regret < report.best_fixed_mean_regret,
        beats_ga2m: report.student_mean_regret < report.ga2m_mean_regret,
        paired_hypervolume_improvement: report.paired_hypervolume_improvement,
        paired_hypervolume_ci_lower: report.paired_hypervolume_ci_lower,
        student_max_abs_difference: report.student_max_abs_difference,
        top_choice_agreement: report.top_choice_agreement,
        bundle_bytes: report.bundle_bytes,
        inference_p95_ms: p95(&mut inference_times),
        sketch_p95_ms: p95(&mut sketch_times),
        training_evidence_sha256,
        best_fixed_candidate_id: report.best_fixed_candidate_id.clone(),
        student_mean_regret: report.student_mean_regret,
        random_mean_regret: report.random_mean_regret,
        best_fixed_mean_regret: report.best_fixed_mean_regret,
        ga2m_mean_regret: report.ga2m_mean_regret,
    };
    write_canonical(evidence_out, &evidence)?;
    Ok((report, evidence))
}

fn table_task(task: &str) -> Result<Task> {
    match task {
        "binary" => Ok(Task::Binary),
        "regression" => Ok(Task::Regression),
        _ => Err(DopeError::Data(format!("unknown cohort task {task}"))),
    }
}

pub fn freeze_router(
    state_dir: &Path,
    bundle_path: &Path,
    evidence_path: &Path,
    allow_failed_frontier: bool,
) -> Result<CampaignState> {
    let mut state = load_state(state_dir)?;
    if state.phase != CampaignPhase::Frozen {
        return Err(DopeError::Data("router may be frozen exactly once".into()));
    }
    let bundle: RouterBundle = read_json(bundle_path)?;
    bundle.validate()?;
    if !bundle.trained {
        return Err(DopeError::Data("untrained router cannot be frozen".into()));
    }
    let evidence: RouterEvidence = read_json(evidence_path)?;
    if evidence.format != "dope-router-evidence" || evidence.version != 1 {
        return Err(DopeError::Data("invalid router evidence format".into()));
    }
    let failed = evidence.failed_gates();
    if !failed.is_empty() && !allow_failed_frontier {
        return Err(DopeError::Data(format!(
            "router freeze gates failed (use the explicit measured failed-frontier path to continue certification without promotion): {}",
            failed.join(",")
        )));
    }
    let bundle_hash = copy_file(bundle_path, &state_dir.join("router.bundle"))?;
    let evidence_hash = copy_file(evidence_path, &state_dir.join("router-evidence.json"))?;
    if bundle.training_evidence_sha256.as_deref() != Some(&evidence.training_evidence_sha256) {
        return Err(DopeError::Data(
            "router bundle and training evidence hashes differ".into(),
        ));
    }
    state.phase = CampaignPhase::RouterFrozen;
    state.router_bundle = Some(bundle_hash);
    state.router_evidence = Some(evidence_hash);
    state.validate()?;
    write_canonical(&state_path(state_dir), &state)?;
    CampaignLedger::open(&ledger_path(state_dir))?.put_metadata("router_evidence", &evidence)?;
    Ok(state)
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ValidationAuthorizationEvent {
    pub format: String,
    pub version: u8,
    pub source_commit: String,
    pub router_bundle_sha256: String,
    pub authorization_token_sha256: String,
    pub opened_unix_seconds: u64,
    pub validation_cert_open_count: usize,
    pub sealed_test_open_count: usize,
}

pub fn open_validation_cert(state_dir: &Path, authorization: &Path) -> Result<CampaignState> {
    require_host("xbabe3", "validation-cert authorization")?;
    let mut state = load_state(state_dir)?;
    if state.phase != CampaignPhase::RouterFrozen || state.validation_cert_open_count != 0 {
        return Err(DopeError::Data(
            "validation-cert requires a frozen router and a single unopened authorization".into(),
        ));
    }
    let token = fs::read(authorization).map_err(|error| io_error(authorization, error))?;
    if token.is_empty() {
        return Err(DopeError::Data("validation authorization is empty".into()));
    }
    let event = ValidationAuthorizationEvent {
        format: "dope-validation-cert-authorization".into(),
        version: 1,
        source_commit: state.source_commit.clone(),
        router_bundle_sha256: state
            .router_bundle
            .as_ref()
            .expect("validated router state")
            .sha256
            .clone(),
        authorization_token_sha256: hashes(&token).sha256,
        opened_unix_seconds: unix_now(),
        validation_cert_open_count: 1,
        sealed_test_open_count: 0,
    };
    write_canonical(
        &state_dir.join("validation-cert-authorization.json"),
        &event,
    )?;
    state.phase = CampaignPhase::ValidationCertOpen;
    state.validation_cert_open_count = 1;
    state.validate()?;
    write_canonical(&state_path(state_dir), &state)?;
    Ok(state)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ControllerRequest {
    Lease { worker: String },
    MarkRunning { job_id: String, worker: String },
    Heartbeat { job_id: String, worker: String },
    Finish { receipt: Box<JobReceipt> },
    Status,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ControllerResponse {
    pub ok: bool,
    pub error: Option<String>,
    pub leased: Option<LeasedJob>,
    pub status: Option<crate::ledger::LedgerStatus>,
}

impl ControllerResponse {
    fn ok() -> Self {
        Self {
            ok: true,
            error: None,
            leased: None,
            status: None,
        }
    }

    fn error(error: impl ToString) -> Self {
        Self {
            ok: false,
            error: Some(error.to_string()),
            leased: None,
            status: None,
        }
    }
}

fn handle_request(
    state_dir: &Path,
    ledger: &CampaignLedger,
    key: &[u8; 32],
    request: ControllerRequest,
) -> Result<ControllerResponse> {
    match request {
        ControllerRequest::Lease { worker } => {
            require_durable_space(state_dir)?;
            ledger.reclaim_expired(unix_now())?;
            let state = load_state(state_dir)?;
            let phases: &[&str] = if worker == "xbabe3" {
                if state.phase != CampaignPhase::ValidationCertOpen {
                    &[]
                } else {
                    &["validation_cert"]
                }
            } else {
                &["training_gold", "validation_select"]
            };
            let mut response = ControllerResponse::ok();
            response.leased = ledger.lease_next_for(&worker, LEASE_SECONDS, phases)?;
            Ok(response)
        }
        ControllerRequest::MarkRunning { job_id, worker } => {
            ledger.mark_running(&job_id, &worker)?;
            Ok(ControllerResponse::ok())
        }
        ControllerRequest::Heartbeat { job_id, worker } => {
            ledger.heartbeat(&job_id, &worker, LEASE_SECONDS)?;
            Ok(ControllerResponse::ok())
        }
        ControllerRequest::Finish { receipt } => {
            ledger.finish_signed(&receipt, key)?;
            Ok(ControllerResponse::ok())
        }
        ControllerRequest::Status => {
            let mut response = ControllerResponse::ok();
            response.status = Some(ledger.status()?);
            Ok(response)
        }
    }
}

pub fn serve(
    state_dir: &Path,
    bind: &str,
    receipt_key: &Path,
    max_requests: Option<usize>,
) -> Result<()> {
    let state = load_state(state_dir)?;
    let key = load_receipt_key(receipt_key)?;
    if hashes(&key).sha256 != state.receipt_key_sha256 {
        return Err(DopeError::Data(
            "controller receipt key differs from freeze".into(),
        ));
    }
    let ledger = CampaignLedger::open(&ledger_path(state_dir))?;
    let listener = TcpListener::bind(bind)
        .map_err(|error| DopeError::Data(format!("cannot bind controller {bind}: {error}")))?;
    for (index, stream) in listener.incoming().enumerate() {
        let mut stream = stream.map_err(|error| DopeError::Data(error.to_string()))?;
        let mut bytes = Vec::new();
        stream
            .read_to_end(&mut bytes)
            .map_err(|error| DopeError::Data(error.to_string()))?;
        let response = match serde_json::from_slice::<ControllerRequest>(&bytes) {
            Ok(request) => handle_request(state_dir, &ledger, &key, request)
                .unwrap_or_else(ControllerResponse::error),
            Err(error) => ControllerResponse::error(error),
        };
        stream
            .write_all(&canonical_json(&response)?)
            .map_err(|error| DopeError::Data(error.to_string()))?;
        if max_requests.is_some_and(|limit| index + 1 >= limit) {
            break;
        }
    }
    Ok(())
}

fn controller_call(address: &str, request: &ControllerRequest) -> Result<ControllerResponse> {
    let mut stream = TcpStream::connect(address).map_err(|error| {
        DopeError::Data(format!("controller {address} is unreachable: {error}"))
    })?;
    stream
        .write_all(&canonical_json(request)?)
        .map_err(|error| DopeError::Data(error.to_string()))?;
    stream
        .shutdown(Shutdown::Write)
        .map_err(|error| DopeError::Data(error.to_string()))?;
    let mut bytes = Vec::new();
    stream
        .read_to_end(&mut bytes)
        .map_err(|error| DopeError::Data(error.to_string()))?;
    let response: ControllerResponse = serde_json::from_slice(&bytes)?;
    if !response.ok {
        return Err(DopeError::Data(
            response
                .error
                .unwrap_or_else(|| "controller rejected request".into()),
        ));
    }
    Ok(response)
}

fn execute_job(
    controller: &str,
    worker: &str,
    work_dir: &Path,
    leased: &LeasedJob,
    key: &[u8; 32],
) -> Result<JobReceipt> {
    let started = unix_now();
    controller_call(
        controller,
        &ControllerRequest::MarkRunning {
            job_id: leased.job_id.clone(),
            worker: worker.into(),
        },
    )?;
    let mut command = Command::new(&leased.spec.command[0]);
    command
        .args(&leased.spec.command[1..])
        .current_dir(work_dir)
        .envs(&leased.spec.environment)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    let mut failure_class = None;
    let mut failure = None;
    let mut exit_code = None;
    let mut state = JobState::Failed;
    let mut output_hashes = BTreeMap::new();
    let mut evidence = None;
    match command.spawn() {
        Err(error) => {
            failure_class = Some(FailureClass::Infrastructure);
            failure = Some(format!("failed to spawn frozen command: {error}"));
        }
        Ok(mut child) => {
            let deadline = Instant::now() + Duration::from_secs(leased.spec.timeout_seconds);
            let mut next_heartbeat = Instant::now() + Duration::from_secs(HEARTBEAT_SECONDS);
            loop {
                if let Some(status) = child
                    .try_wait()
                    .map_err(|error| DopeError::Data(error.to_string()))?
                {
                    exit_code = status.code();
                    if status.success() {
                        let outputs = leased
                            .spec
                            .expected_outputs
                            .iter()
                            .map(|(name, path)| {
                                let path = if path.is_absolute() {
                                    path.clone()
                                } else {
                                    work_dir.join(path)
                                };
                                Ok((name.clone(), path.clone(), file_hashes(&path)?))
                            })
                            .collect::<Result<Vec<_>>>();
                        match outputs {
                            Ok(outputs) => {
                                for (name, _, digest) in &outputs {
                                    output_hashes.insert(name.clone(), digest.clone());
                                }
                                let metric_path = outputs
                                    .iter()
                                    .find(|(name, _, _)| name == "metrics" || name == "evidence")
                                    .map(|(_, path, _)| path);
                                match metric_path
                                    .map(|path| read_json::<JobEvidence>(path))
                                    .transpose()
                                {
                                    Ok(Some(value))
                                        if value.validate_against(&leased.spec).is_ok() =>
                                    {
                                        evidence = Some(value);
                                        state = JobState::Succeeded;
                                    }
                                    Ok(_) => {
                                        failure_class = Some(FailureClass::ContractMismatch);
                                        failure = Some("successful command did not emit typed metrics evidence".into());
                                    }
                                    Err(error) => {
                                        failure_class = Some(FailureClass::ContractMismatch);
                                        failure = Some(error.to_string());
                                    }
                                }
                            }
                            Err(error) => {
                                failure_class = Some(FailureClass::ContractMismatch);
                                failure = Some(error.to_string());
                            }
                        }
                    } else {
                        failure_class = Some(FailureClass::DeterministicModel);
                        failure = Some(format!("frozen command exited with status {status}"));
                    }
                    break;
                }
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    state = JobState::TimedOut;
                    failure_class = Some(FailureClass::Timeout);
                    failure = Some("frozen command exceeded its timeout".into());
                    break;
                }
                if Instant::now() >= next_heartbeat {
                    controller_call(
                        controller,
                        &ControllerRequest::Heartbeat {
                            job_id: leased.job_id.clone(),
                            worker: worker.into(),
                        },
                    )?;
                    next_heartbeat = Instant::now() + Duration::from_secs(HEARTBEAT_SECONDS);
                }
                thread::sleep(Duration::from_millis(100));
            }
        }
    }
    let mut receipt = JobReceipt {
        format: "dope-job-receipt".into(),
        version: 1,
        job_id: leased.job_id.clone(),
        attempt: leased.attempt,
        worker: worker.into(),
        state,
        started_unix_seconds: started,
        finished_unix_seconds: unix_now(),
        source_commit: leased.spec.source_commit.clone(),
        environment_lock_sha256: leased.spec.environment_lock_sha256.clone(),
        corpus_manifest_sha256: leased.spec.corpus_manifest_sha256.clone(),
        kpi_contract_sha256: leased.spec.kpi_contract_sha256.clone(),
        output_hashes,
        failure_class,
        failure,
        command_exit_code: exit_code,
        evidence,
        signature: None,
    };
    receipt.sign(key)?;
    Ok(receipt)
}

pub fn worker(
    controller: &str,
    worker: &str,
    work_dir: &Path,
    receipt_key: &Path,
    once: bool,
) -> Result<usize> {
    let key = load_receipt_key(receipt_key)?;
    let mut completed = 0;
    loop {
        let response = controller_call(
            controller,
            &ControllerRequest::Lease {
                worker: worker.into(),
            },
        )?;
        let Some(leased) = response.leased else { break };
        let receipt = execute_job(controller, worker, work_dir, &leased, &key)?;
        controller_call(
            controller,
            &ControllerRequest::Finish {
                receipt: Box::new(receipt),
            },
        )?;
        completed += 1;
        if once {
            break;
        }
    }
    Ok(completed)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CandidateFrontierEntry {
    pub candidate_id: String,
    #[serde(default)]
    pub implementation_hash: String,
    pub attempted: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub timed_out: usize,
    pub mean_retention: Option<f64>,
    pub runtime_p95_ms: Option<u64>,
    pub peak_memory_p95_bytes: Option<u64>,
    pub artifact_bytes_p95: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MetricsBundle {
    pub format: String,
    pub version: u8,
    pub status: crate::ledger::LedgerStatus,
    pub kpi: KpiSummary,
    pub candidate_frontier: Vec<CandidateFrontierEntry>,
    pub router: Option<RouterEvidence>,
    pub failed_gates: Vec<String>,
    pub validation_cert_open_count: usize,
    pub sealed_test_open_count: usize,
    pub receipts_reconciled: usize,
    #[serde(default)]
    pub certification: Option<CertificationDetail>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProfileRegretDetail {
    pub outcomes: usize,
    pub mean_regret: f64,
    pub upper_95: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CertificationDetail {
    pub eligible_lineages: usize,
    pub evaluated_lineages: usize,
    pub missing_lineages: usize,
    pub timeout_lineages: usize,
    pub exhaustive_cells: usize,
    pub routed_cells: usize,
    pub shadow_cells: usize,
    pub failed_cells: usize,
    pub top_four_oracle_recall: Option<f64>,
    pub mean_candidate_regret: Option<f64>,
    pub random_candidate_regret: Option<f64>,
    pub best_fixed_candidate_id: String,
    pub best_fixed_candidate_regret: Option<f64>,
    pub beats_random_candidate: bool,
    pub beats_best_fixed_candidate: bool,
    pub maximum_profile_regret_upper: Option<f64>,
    pub regret_by_profile: BTreeMap<String, ProfileRegretDetail>,
    pub work_avoided_fraction: Option<f64>,
    pub latency_reduction_fraction: Option<f64>,
    pub router_bundle_bytes: u64,
    pub router_inference_p95_ms: Option<f64>,
    pub sketch_p95_ms: Option<f64>,
    pub paired_hypervolume_improvement: Option<f64>,
    pub paired_hypervolume_ci_lower: Option<f64>,
    pub candidate_availability: BTreeMap<String, usize>,
    pub auditor_availability: BTreeMap<String, usize>,
    pub candidate_implementation_hashes: BTreeMap<String, String>,
    pub auditor_implementation_hashes: BTreeMap<String, String>,
    pub receipts_expected: usize,
    pub receipts_reconciled: usize,
}

fn percentile(values: &mut [u64], quantile: f64) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some(values[((values.len() - 1) as f64 * quantile).ceil() as usize])
}

fn option_max(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    values.flatten().reduce(f64::max)
}

fn option_min(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    values.flatten().reduce(f64::min)
}

pub fn export_metrics(state_dir: &Path, out: &Path, receipt_key: &Path) -> Result<MetricsBundle> {
    let state = load_state(state_dir)?;
    fs::create_dir_all(out).map_err(|error| io_error(out, error))?;
    let key = load_receipt_key(receipt_key)?;
    if hashes(&key).sha256 != state.receipt_key_sha256 {
        return Err(DopeError::Data(
            "metrics receipt key differs from freeze".into(),
        ));
    }
    let ledger = CampaignLedger::open(&ledger_path(state_dir))?;
    let specs = ledger.all_specs()?;
    let receipts = ledger.all_receipts()?;
    let specs_by_id = specs
        .iter()
        .map(|(id, state, spec)| (id.as_str(), (*state, spec)))
        .collect::<BTreeMap<_, _>>();
    let signatures_valid = receipts.iter().all(|receipt| {
        receipt.verify_signature(&key).is_ok()
            && specs_by_id
                .get(receipt.job_id.as_str())
                .is_some_and(|(_, spec)| receipt.validate_against(spec).is_ok())
    });
    let mut last_receipts = BTreeMap::<&str, &JobReceipt>::new();
    for receipt in &receipts {
        let replace = last_receipts
            .get(receipt.job_id.as_str())
            .is_none_or(|existing| receipt.attempt > existing.attempt);
        if replace {
            last_receipts.insert(receipt.job_id.as_str(), receipt);
        }
    }
    let receipts_reconciled = specs
        .iter()
        .filter(|(id, state, _)| {
            matches!(
                state,
                JobState::Succeeded | JobState::Failed | JobState::TimedOut
            ) && last_receipts
                .get(id.as_str())
                .is_some_and(|receipt| receipt.state == *state)
        })
        .count();
    let mut metrics_file = File::create(out.join("metrics.jsonl"))
        .map_err(|error| io_error(out.join("metrics.jsonl"), error))?;
    for receipt in &receipts {
        metrics_file
            .write_all(&canonical_json(receipt)?)
            .map_err(|error| io_error(out, error))?;
        metrics_file
            .write_all(b"\n")
            .map_err(|error| io_error(out, error))?;
    }
    ledger.export_receipts(&out.join("receipts.json"))?;

    let successful = receipts
        .iter()
        .filter(|receipt| receipt.state == JobState::Succeeded)
        .filter_map(|receipt| receipt.evidence.as_ref())
        .collect::<Vec<_>>();
    let cert_successful = successful
        .iter()
        .filter(|evidence| evidence.phase == "validation_cert" && evidence.routed)
        .copied()
        .collect::<Vec<_>>();
    let cert_cells = cert_successful
        .iter()
        .map(|evidence| KpiCell {
            task: if evidence.task == "binary" {
                Task::Binary
            } else {
                Task::Regression
            },
            train_rows: evidence.train_rows,
            features: evidence.features,
            auditor: evidence.auditor_id.clone(),
            size_multiplier: evidence.size_multiplier,
            lineage_group_id: evidence.lineage_group_id.clone(),
            generation_seed: evidence.generation_seed,
            auditor_seed: evidence.auditor_seed,
            null_loss: evidence.null_loss,
            trtr_loss: evidence.trtr_loss,
            tstr_loss: evidence.tstr_loss,
            calibration_degradation: evidence.calibration_degradation,
            rare_class_or_tail_retention: evidence.rare_class_or_tail_retention,
            supported_subgroup_retention: evidence.supported_subgroup_retention,
            nominal_95_coverage: evidence.nominal_95_coverage,
        })
        .collect::<Vec<_>>();
    let eligible_lineages = specs
        .iter()
        .filter(|(_, _, spec)| spec.phase == "validation_cert" && spec.routed)
        .map(|(_, _, spec)| &spec.dataset_lineage)
        .collect::<BTreeSet<_>>()
        .len();
    let evaluated_lineages = cert_cells
        .iter()
        .map(|cell| &cell.lineage_group_id)
        .collect::<BTreeSet<_>>()
        .len();
    let kpi_report = aggregate_kpis(
        &cert_cells,
        eligible_lineages,
        eligible_lineages.saturating_sub(evaluated_lineages),
        "validation-cert",
    )?;
    write_canonical(&out.join("validation-kpis.json"), &kpi_report)?;

    let mut coverage_groups =
        BTreeMap::<(String, String, String, usize), (usize, usize, usize, usize)>::new();
    for (_, job_state, spec) in specs
        .iter()
        .filter(|(_, _, spec)| spec.phase == "validation_cert" && spec.routed)
    {
        let task = spec
            .structural_profile
            .split('/')
            .next()
            .unwrap_or("unknown")
            .to_owned();
        let entry = coverage_groups
            .entry((
                task,
                spec.structural_profile.clone(),
                spec.auditor_spec.clone(),
                spec.size_multiplier,
            ))
            .or_default();
        entry.0 += 1;
        match job_state {
            JobState::Succeeded => entry.1 += 1,
            JobState::TimedOut => entry.3 += 1,
            _ => entry.2 += 1,
        }
    }
    let entries = coverage_groups
        .into_iter()
        .map(
            |((task, profile, auditor, size), (eligible, evaluated, missing, timed_out))| {
                CoverageEntry {
                    task,
                    structural_profile: profile,
                    auditor,
                    size_multiplier: size,
                    eligible,
                    evaluated,
                    missing,
                    timed_out,
                    required: matches!(size, 1 | 4),
                }
            },
        )
        .collect::<Vec<_>>();
    let required_complete = !entries.is_empty()
        && entries
            .iter()
            .all(|entry| !entry.required || (entry.missing == 0 && entry.timed_out == 0));
    let coverage = CoverageReport {
        format: "dope-kpi-coverage".into(),
        version: 1,
        tier: "validation-cert".into(),
        entries,
        required_complete,
    };
    coverage.validate()?;
    write_canonical(&out.join("kpi-coverage.json"), &coverage)?;

    let router: Option<RouterEvidence> = ledger.get_metadata("router_evidence")?;
    let candidate_regret = router
        .as_ref()
        .map(|value| value.maximum_profile_regret_upper);
    let kpi = KpiSummary::from_evidence(Some(&kpi_report), Some(&coverage), candidate_regret);
    let mut frontier = Vec::new();
    for backend in empirical_backends() {
        let matching_specs = specs
            .iter()
            .filter(|(_, _, spec)| spec.candidate_spec == backend.id)
            .collect::<Vec<_>>();
        let matching = successful
            .iter()
            .filter(|evidence| evidence.candidate_id == backend.id)
            .copied()
            .collect::<Vec<_>>();
        let mut runtimes = matching
            .iter()
            .map(|evidence| evidence.runtime_ms)
            .collect::<Vec<_>>();
        let mut memories = matching
            .iter()
            .map(|evidence| evidence.peak_memory_bytes)
            .collect::<Vec<_>>();
        let mut bytes = matching
            .iter()
            .map(|evidence| evidence.artifact_bytes)
            .collect::<Vec<_>>();
        let retentions = matching
            .iter()
            .filter_map(|evidence| {
                let denominator = evidence.null_loss - evidence.trtr_loss;
                (denominator.abs() > f64::EPSILON)
                    .then_some((evidence.null_loss - evidence.tstr_loss) / denominator)
            })
            .collect::<Vec<_>>();
        frontier.push(CandidateFrontierEntry {
            candidate_id: backend.id,
            implementation_hash: backend.implementation_hash,
            attempted: matching_specs.len(),
            succeeded: matching.len(),
            failed: matching_specs
                .iter()
                .filter(|(_, state, _)| *state == JobState::Failed)
                .count(),
            timed_out: matching_specs
                .iter()
                .filter(|(_, state, _)| *state == JobState::TimedOut)
                .count(),
            mean_retention: (!retentions.is_empty())
                .then(|| retentions.iter().sum::<f64>() / retentions.len() as f64),
            runtime_p95_ms: percentile(&mut runtimes, 0.95),
            peak_memory_p95_bytes: percentile(&mut memories, 0.95),
            artifact_bytes_p95: percentile(&mut bytes, 0.95),
        });
    }
    write_canonical(&out.join("candidate-frontier.json"), &frontier)?;

    let candidate_tasks = successful.iter().fold(
        BTreeMap::<String, BTreeSet<String>>::new(),
        |mut map, evidence| {
            map.entry(evidence.candidate_id.clone())
                .or_default()
                .insert(evidence.task.clone());
            map
        },
    );
    let exercised_auditors = successful
        .iter()
        .map(|evidence| &evidence.auditor_id)
        .collect::<BTreeSet<_>>();
    let mut seed_groups = BTreeMap::<(String, String, String, usize), Vec<f64>>::new();
    for evidence in &cert_successful {
        seed_groups
            .entry((
                evidence.lineage_group_id.clone(),
                evidence.candidate_id.clone(),
                evidence.auditor_id.clone(),
                evidence.size_multiplier,
            ))
            .or_default()
            .push(evidence.tstr_loss);
    }
    let standard_deviation = |values: &[f64]| {
        if values.len() < 2 {
            return 0.0;
        }
        let average = values.iter().sum::<f64>() / values.len() as f64;
        (values
            .iter()
            .map(|value| (value - average).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64)
            .sqrt()
    };
    let across_seed_validation_loss_stddev = seed_groups
        .values()
        .map(|values| standard_deviation(values))
        .reduce(f64::max);
    let byte_reconciliation_exact = !specs.is_empty()
        && receipts_reconciled == specs.len()
        && signatures_valid
        && successful.iter().all(|evidence| {
            evidence.artifact_bytes > 0
                && is_lower_hex(&evidence.artifact_sha256, &[64])
                && is_lower_hex(&evidence.artifact_blake3, &[64])
        });
    let (
        feature_importance_complete,
        feature_importance_applicable,
        feature_importance_spearman_min,
        feature_importance_top_k_jaccard_min,
    ) = aggregate_importance(cert_successful.iter().copied());
    let evidence = GateEvidence {
        ptf_v1: kpi_report.ptf_v1,
        calibration_degradation_max: option_max(
            cert_successful.iter().map(|e| e.calibration_degradation),
        ),
        rare_class_tail_subgroup_retention_min: option_min(cert_successful.iter().flat_map(|e| {
            [
                e.rare_class_or_tail_retention,
                e.supported_subgroup_retention,
            ]
        })),
        driver_agreement_min: option_min(cert_successful.iter().map(|e| e.driver_agreement)),
        joint_fidelity_min: option_min(cert_successful.iter().map(|e| e.joint_fidelity)),
        query_p95_normalized_error_max: option_max(
            cert_successful.iter().map(|e| e.query_p95_normalized_error),
        ),
        nominal_95_coverage_min: option_min(cert_successful.iter().map(|e| e.nominal_95_coverage)),
        nominal_95_coverage_max: option_max(cert_successful.iter().map(|e| e.nominal_95_coverage)),
        type_i_error_max: option_max(cert_successful.iter().map(|e| e.type_i_error)),
        membership_auc_max: option_max(cert_successful.iter().map(|e| e.membership_auc)),
        feature_importance_complete,
        feature_importance_applicable,
        feature_importance_spearman_min,
        feature_importance_top_k_jaccard_min,
        attribute_inference_advantage_max: option_max(
            cert_successful
                .iter()
                .map(|e| e.attribute_inference_advantage),
        ),
        exact_copies: cert_successful.iter().map(|e| e.exact_copies).sum(),
        near_copies: cert_successful.iter().map(|e| e.near_copies).sum(),
        canary_extractions: cert_successful.iter().map(|e| e.canary_extractions).sum(),
        lineage_leaks: cert_successful.iter().map(|e| e.lineage_leaks).sum(),
        validation_training_regret_upper: candidate_regret,
        across_seed_validation_loss_stddev,
        uncertainty_coverage_min: option_min(cert_successful.iter().map(|e| e.nominal_95_coverage)),
        uncertainty_coverage_max: option_max(cert_successful.iter().map(|e| e.nominal_95_coverage)),
        beats_random_router: router.as_ref().is_some_and(|value| value.beats_random),
        beats_best_fixed_candidate: router.as_ref().is_some_and(|value| value.beats_best_fixed),
        learned_representation_used: router.is_some(),
        pareto_hypervolume_improvement: router
            .as_ref()
            .map(|value| value.paired_hypervolume_improvement),
        pareto_hypervolume_ci_lower: router
            .as_ref()
            .map(|value| value.paired_hypervolume_ci_lower),
        all_candidates_exercised_both_tasks: empirical_backends().iter().all(|backend| {
            candidate_tasks
                .get(&backend.id)
                .is_some_and(|tasks| tasks.len() == 2)
        }),
        all_auditors_pinned_and_exercised: auditor_specs()
            .iter()
            .all(|auditor| exercised_auditors.contains(&auditor.id)),
        byte_reconciliation_exact,
        signatures_and_receipts_valid: signatures_valid,
        controller_free_gib: available_bytes(state_dir)? / GIB,
        evaluator_free_gib: available_bytes(state_dir)? / GIB,
    };
    write_canonical(&out.join("gate-evidence.json"), &evidence)?;
    let failed_gates =
        evidence.failed_gates(&crate::production::KpiContract::embedded()?, &coverage);
    let status = ledger.status()?;
    ledger.put_metadata(
        "status_evidence",
        &serde_json::json!({
            "router_regret": candidate_regret,
            "ptf_v1": kpi_report.ptf_v1,
            "coverage": kpi.coverage,
            "production_score_available": kpi.production_score_available
        }),
    )?;
    let bundle = MetricsBundle {
        format: "dope-campaign-metrics".into(),
        version: 1,
        status,
        kpi,
        candidate_frontier: frontier,
        router,
        failed_gates,
        validation_cert_open_count: state.validation_cert_open_count,
        sealed_test_open_count: state.sealed_test_open_count,
        receipts_reconciled,
        certification: None,
    };
    write_canonical(&out.join("metrics-summary.json"), &bundle)?;
    Ok(bundle)
}

#[derive(Default)]
struct CertCandidateAggregate {
    succeeded: usize,
    timed_out: usize,
    retention_sum: f64,
    retention_count: usize,
    runtimes: BTreeMap<u64, usize>,
    memories: BTreeMap<u64, usize>,
    artifact_bytes: BTreeMap<u64, usize>,
    tasks: BTreeSet<String>,
}

#[derive(Default)]
struct CertCoverageAggregate {
    eligible: usize,
    evaluated: usize,
    timed_out: usize,
}

#[derive(Clone, Copy)]
struct CertCandidateOutcome {
    utility: f64,
    runtime_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct CertReceiptReconciliation {
    format: String,
    version: u8,
    expected_blocks: usize,
    reconciled_blocks: usize,
    extra_blocks: usize,
    extra_receipts: usize,
    issues: Vec<String>,
}

fn histogram_percentile(
    histogram: &BTreeMap<u64, usize>,
    cells: usize,
    quantile: f64,
) -> Option<u64> {
    if cells == 0 {
        return None;
    }
    let index = ((cells - 1) as f64 * quantile).ceil() as usize;
    let mut cumulative = 0usize;
    for (&value, &count) in histogram {
        cumulative += count;
        if cumulative > index {
            return Some(value);
        }
    }
    None
}

fn mean_and_upper_95(values: &[f64]) -> Option<(f64, f64)> {
    if values.is_empty() {
        return None;
    }
    let average = values.iter().sum::<f64>() / values.len() as f64;
    let deviation = if values.len() > 1 {
        (values
            .iter()
            .map(|value| (value - average).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    Some((
        average,
        average + 1.645 * deviation / (values.len() as f64).sqrt(),
    ))
}

fn mean_and_lower_95(values: &[f64]) -> Option<(f64, f64)> {
    if values.is_empty() {
        return None;
    }
    let average = values.iter().sum::<f64>() / values.len() as f64;
    let deviation = if values.len() > 1 {
        (values
            .iter()
            .map(|value| (value - average).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    Some((
        average,
        average - 1.96 * deviation / (values.len() as f64).sqrt(),
    ))
}

fn float_percentile(values: &mut [f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[((values.len() - 1) as f64 * quantile).ceil() as usize])
}

fn update_optional_max(target: &mut Option<f64>, value: Option<f64>) {
    if let Some(value) = value {
        *target = Some(target.map_or(value, |current| current.max(value)));
    }
}

fn update_optional_min(target: &mut Option<f64>, value: Option<f64>) {
    if let Some(value) = value {
        *target = Some(target.map_or(value, |current| current.min(value)));
    }
}

fn is_timeout_failure(error: &str) -> bool {
    let error = error.to_ascii_lowercase();
    error.contains("timeout") || error.contains("timed out") || error.contains("deadline")
}

/// Reconstructs the final validation-cert metrics directly from content-addressed
/// record blocks. Only blocks with a valid xbabe3 keyed receipt and exact frozen
/// source, binary, router, cohort, and block hashes are admitted as evidence.
pub fn export_validation_cert_metrics(
    state_dir: &Path,
    cert_out: &Path,
    cohort_path: &Path,
    out: &Path,
    receipt_key: &Path,
) -> Result<MetricsBundle> {
    let state = load_state(state_dir)?;
    if state.phase != CampaignPhase::ValidationCertOpen
        || state.validation_cert_open_count != 1
        || state.sealed_test_open_count != 0
    {
        return Err(DopeError::Data(
            "validation-cert metrics require exactly one authorization event".into(),
        ));
    }
    require_durable_space(state_dir)?;
    fs::create_dir_all(out).map_err(|error| io_error(out, error))?;
    let key = load_receipt_key(receipt_key)?;
    if hashes(&key).sha256 != state.receipt_key_sha256 {
        return Err(DopeError::Data(
            "validation-cert receipt key differs from freeze".into(),
        ));
    }
    if file_hashes(&state_dir.join("source.tar"))? != state.source_tree {
        return Err(DopeError::Data(
            "frozen source archive no longer matches campaign state".into(),
        ));
    }
    let frozen_router_path = state_dir.join("router.bundle");
    let frozen_router_evidence_path = state_dir.join("router-evidence.json");
    let router_hashes = file_hashes(&frozen_router_path)?;
    let router_evidence_hashes = file_hashes(&frozen_router_evidence_path)?;
    if state.router_bundle.as_ref() != Some(&router_hashes)
        || state.router_evidence.as_ref() != Some(&router_evidence_hashes)
    {
        return Err(DopeError::Data(
            "router artifacts no longer match the frozen campaign".into(),
        ));
    }
    let router_bundle: RouterBundle = read_json(&frozen_router_path)?;
    router_bundle.validate()?;
    let router_evidence: RouterEvidence = read_json(&frozen_router_evidence_path)?;
    if router_evidence.format != "dope-router-evidence" || router_evidence.version != 1 {
        return Err(DopeError::Data("invalid frozen router evidence".into()));
    }
    let host_inventory_path = state_dir.join("host-inventory.json");
    if file_hashes(&host_inventory_path)? != state.host_inventory {
        return Err(DopeError::Data(
            "host inventory no longer matches campaign state".into(),
        ));
    }
    let host_inventory: HostInventory = read_json(&host_inventory_path)?;
    host_inventory.validate(&state.source_tree.sha256, &state.corpus_manifest.sha256)?;
    let evaluator = host_inventory
        .workers
        .iter()
        .find(|worker| worker.host == "xbabe3")
        .ok_or_else(|| DopeError::Data("xbabe3 is absent from host inventory".into()))?;
    let controller = host_inventory
        .workers
        .iter()
        .find(|worker| worker.host == "xbabe1")
        .ok_or_else(|| DopeError::Data("xbabe1 is absent from host inventory".into()))?;

    let cohort: CohortPlan = read_json(cohort_path)?;
    let cohort_hashes = file_hashes(cohort_path)?;
    if cohort.format != "dope-campaign-cohort"
        || cohort.version != 1
        || cohort.kind != "validation-cert"
        || cohort.records.len() != state.validation_cert_lineage_groups
        || cohort
            .records
            .iter()
            .map(|record| record.lineage_group_id.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != state.validation_cert_lineage_groups
        || cohort
            .records
            .iter()
            .any(|record| record.partition != "validation_cert")
    {
        return Err(DopeError::Data(
            "validation-cert cohort does not contain the exact guarded lineage population".into(),
        ));
    }
    let validation_path = state_dir.join("validation-manifest.json");
    if file_hashes(&validation_path)? != state.validation_manifest {
        return Err(DopeError::Data(
            "validation manifest no longer matches campaign state".into(),
        ));
    }
    let validation: ValidationSubmanifest = read_json(&validation_path)?;
    validate_validation_submanifest(&validation)?;
    let cert_assignments = validation
        .assignments
        .iter()
        .filter(|assignment| assignment.partition == "validation-cert")
        .map(|assignment| {
            (
                assignment.dataset_id.as_str(),
                assignment.lineage_group_id.as_str(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if cohort.records.iter().any(|record| {
        cert_assignments.get(record.dataset_id.as_str()).copied()
            != Some(record.lineage_group_id.as_str())
    }) {
        return Err(DopeError::Data(
            "validation-cert cohort differs from the frozen assignment".into(),
        ));
    }

    let candidate_ids = empirical_backends()
        .into_iter()
        .map(|candidate| candidate.id)
        .collect::<Vec<_>>();
    let auditor_ids = auditor_specs()
        .into_iter()
        .map(|auditor| auditor.id)
        .collect::<Vec<_>>();
    let size_multipliers = [1usize, 4];
    let generation_seeds = [1829u64, 99238, 196647];
    let auditor_seeds = [57721u64, 161803, 271828];
    let cells_per_candidate =
        auditor_ids.len() * size_multipliers.len() * generation_seeds.len() * auditor_seeds.len();
    let cells_per_record = candidate_ids.len() * cells_per_candidate;
    let expected_cells = cohort.records.len() * cells_per_record;
    let mut candidate_aggregates = candidate_ids
        .iter()
        .map(|candidate| (candidate.clone(), CertCandidateAggregate::default()))
        .collect::<BTreeMap<_, _>>();
    let mut candidate_availability = candidate_ids
        .iter()
        .map(|candidate| (candidate.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut auditor_availability = auditor_ids
        .iter()
        .map(|auditor| (auditor.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut coverage_groups =
        BTreeMap::<(String, String, String, usize), CertCoverageAggregate>::new();
    for record in &cohort.records {
        for auditor in &auditor_ids {
            for size in size_multipliers {
                coverage_groups
                    .entry((
                        record.task.clone(),
                        record.structural_profile.clone(),
                        auditor.clone(),
                        size,
                    ))
                    .or_default()
                    .eligible += generation_seeds.len() * auditor_seeds.len();
            }
        }
    }

    let metrics_partial = out.join("metrics.jsonl.partial");
    let receipts_partial = out.join("receipts.jsonl.partial");
    let mut metrics_writer = BufWriter::new(
        File::create(&metrics_partial).map_err(|error| io_error(&metrics_partial, error))?,
    );
    let mut receipts_writer = BufWriter::new(
        File::create(&receipts_partial).map_err(|error| io_error(&receipts_partial, error))?,
    );
    let mut issues = Vec::new();
    let mut expected_identities = BTreeSet::new();
    let mut receipts_reconciled = 0usize;
    let mut succeeded_cells = 0usize;
    let mut explicit_failed_cells = 0usize;
    let mut timed_out_cells = 0usize;
    let mut routed_cells = 0usize;
    let mut shadow_cells = 0usize;
    let mut selected_expected_cells = 0usize;
    let mut valid_expected_cells = 0usize;
    let mut selected_runtime_ms = 0f64;
    let mut exhaustive_runtime_ms = 0f64;
    let mut all_artifact_bytes_nonzero = true;
    let mut evaluated_lineages = BTreeSet::new();
    let mut timeout_lineages = BTreeSet::new();
    let mut kpi_cells = Vec::with_capacity(cohort.records.len() * cells_per_candidate);
    let mut router_sketch_times = Vec::with_capacity(cohort.records.len());
    let mut router_inference_times = Vec::with_capacity(cohort.records.len());
    let mut regrets = Vec::new();
    let mut random_regrets = Vec::new();
    let mut fixed_regrets = Vec::new();
    let mut profile_regrets = BTreeMap::<String, Vec<f64>>::new();
    let mut hypervolume_pairs = Vec::new();
    let mut top_four_hits = 0usize;
    let mut complete_oracle_outcomes = 0usize;
    let mut calibration_degradation_max = None;
    let mut rare_tail_subgroup_retention_min = None;
    let mut driver_agreement_min = None;
    let mut joint_fidelity_min = None;
    let mut query_error_max = None;
    let mut nominal_coverage_min = None;
    let mut nominal_coverage_max = None;
    let mut type_i_error_max = None;
    let mut membership_auc_max = None;
    let mut attribute_advantage_max = None;
    let mut across_seed_validation_loss_stddev = None;
    let mut feature_importance_complete = true;
    let mut feature_importance_applicable = false;
    let mut feature_importance_spearman_min = None::<f64>;
    let mut feature_importance_top_k_jaccard_min = None::<f64>;
    let mut feature_importance_missing_rank = false;
    let mut exact_copies = 0usize;
    let mut near_copies = 0usize;
    let mut canary_extractions = 0usize;
    let mut lineage_leaks = 0usize;

    for record in &cohort.records {
        let matrix_identity = gold_record_matrix_identity(&GoldMatrixIdentity {
            phase: "validation_cert",
            dataset_id: &record.dataset_id,
            candidate_ids: &candidate_ids,
            auditor_ids: &auditor_ids,
            size_multipliers: &size_multipliers,
            generation_seeds: &generation_seeds,
            auditor_seeds: &auditor_seeds,
            primary_only: false,
            neural_target_weight: 2.0,
            neural_structural_penalty: 0.0,
            routing_sha256: Some(&router_hashes.sha256),
        })?;
        expected_identities.insert(matrix_identity.clone());
        let block_path = cert_out
            .join("blocks")
            .join(format!("{matrix_identity}.json"));
        let receipt_path = cert_out
            .join("receipts")
            .join(format!("{matrix_identity}.json"));
        let admitted = (|| -> Result<(GoldRecordBlock, GoldBlockReceipt)> {
            let block: GoldRecordBlock = read_json(&block_path)?;
            validate_gold_record_block(
                &block,
                record,
                "validation_cert",
                &matrix_identity,
                &candidate_ids,
                &auditor_ids,
                Some(&router_hashes.sha256),
            )?;
            let receipt: GoldBlockReceipt = read_json(&receipt_path)?;
            receipt.verify(&key)?;
            let block_hashes = file_hashes(&block_path)?;
            if receipt.dataset_id != record.dataset_id
                || receipt.matrix_identity != matrix_identity
                || receipt.phase != "validation_cert"
                || receipt.worker != "xbabe3"
                || receipt.source_sha256 != state.source_tree.sha256
                || receipt.source_sha256 != evaluator.source_sha256
                || receipt.evaluator_binary_sha256 != evaluator.binary_sha256
                || receipt.router_bundle_sha256.as_deref() != Some(&router_hashes.sha256)
                || receipt.cohort != cohort_hashes
                || receipt.block != block_hashes
                || receipt.evidence_cells != block.evidence.len()
                || receipt.failed_cells != block.failures.len()
            {
                return Err(DopeError::Data(
                    "gold block receipt does not reconcile with frozen xbabe3 inputs".into(),
                ));
            }
            Ok((block, receipt))
        })();
        let (block, receipt) = match admitted {
            Ok(value) => value,
            Err(error) => {
                issues.push(format!("{}: {error}", record.dataset_id));
                continue;
            }
        };
        receipts_reconciled += 1;
        receipts_writer
            .write_all(&canonical_json(&receipt)?)
            .and_then(|_| receipts_writer.write_all(b"\n"))
            .map_err(|error| io_error(&receipts_partial, error))?;
        valid_expected_cells += cells_per_record;
        let routing = block
            .routing
            .as_ref()
            .expect("validated validation-cert block has routing");
        router_sketch_times.push(routing.sketch_ms);
        router_inference_times.push(routing.inference_ms);
        selected_expected_cells += routing.selected_candidates.len() * cells_per_candidate;
        let top_candidate = &routing.ranked_candidates[0];
        let mut top_cells = Vec::new();
        let mut block_coverage = BTreeMap::<(String, usize), usize>::new();
        for cell in &block.evidence {
            succeeded_cells += 1;
            let aggregate = candidate_aggregates
                .get_mut(&cell.candidate_id)
                .expect("validated candidate");
            aggregate.succeeded += 1;
            aggregate.tasks.insert(cell.task.clone());
            *aggregate.runtimes.entry(cell.runtime_ms).or_default() += 1;
            *aggregate
                .memories
                .entry(cell.peak_memory_bytes)
                .or_default() += 1;
            *aggregate
                .artifact_bytes
                .entry(cell.artifact_bytes)
                .or_default() += 1;
            let denominator = cell.null_loss - cell.trtr_loss;
            if denominator.abs() > f64::EPSILON {
                aggregate.retention_sum += (cell.null_loss - cell.tstr_loss) / denominator;
                aggregate.retention_count += 1;
            }
            *candidate_availability
                .get_mut(&cell.candidate_id)
                .expect("validated candidate") += 1;
            *auditor_availability
                .get_mut(&cell.auditor_id)
                .expect("validated auditor") += 1;
            all_artifact_bytes_nonzero &= cell.artifact_bytes > 0;
            exhaustive_runtime_ms += cell.runtime_ms as f64;
            if cell.routed {
                routed_cells += 1;
                selected_runtime_ms += cell.runtime_ms as f64;
            } else {
                shadow_cells += 1;
            }
            if &cell.candidate_id == top_candidate {
                top_cells.push(cell);
                *block_coverage
                    .entry((cell.auditor_id.clone(), cell.size_multiplier))
                    .or_default() += 1;
                coverage_groups
                    .get_mut(&(
                        record.task.clone(),
                        record.structural_profile.clone(),
                        cell.auditor_id.clone(),
                        cell.size_multiplier,
                    ))
                    .expect("initialized coverage group")
                    .evaluated += 1;
                metrics_writer
                    .write_all(&canonical_json(cell)?)
                    .and_then(|_| metrics_writer.write_all(b"\n"))
                    .map_err(|error| io_error(&metrics_partial, error))?;
            }
        }
        explicit_failed_cells += block.failures.len();
        let mut top_timeout_by_auditor = BTreeMap::<String, usize>::new();
        let mut block_has_top_timeout = false;
        for failure in &block.failures {
            if is_timeout_failure(&failure.error) {
                timed_out_cells += 1;
                candidate_aggregates
                    .get_mut(&failure.candidate_id)
                    .expect("validated failure candidate")
                    .timed_out += 1;
                if &failure.candidate_id == top_candidate {
                    *top_timeout_by_auditor
                        .entry(failure.auditor_id.clone())
                        .or_default() += 1;
                    block_has_top_timeout = true;
                }
            }
        }
        if block_has_top_timeout {
            timeout_lineages.insert(record.lineage_group_id.clone());
        }
        for auditor in &auditor_ids {
            let mut remaining_timeouts = top_timeout_by_auditor
                .get(auditor)
                .copied()
                .unwrap_or_default();
            for size in size_multipliers {
                let evaluated = block_coverage
                    .get(&(auditor.clone(), size))
                    .copied()
                    .unwrap_or_default();
                let timed_out = remaining_timeouts.min(
                    generation_seeds
                        .len()
                        .saturating_mul(auditor_seeds.len())
                        .saturating_sub(evaluated),
                );
                remaining_timeouts -= timed_out;
                coverage_groups
                    .get_mut(&(
                        record.task.clone(),
                        record.structural_profile.clone(),
                        auditor.clone(),
                        size,
                    ))
                    .expect("initialized coverage group")
                    .timed_out += timed_out;
            }
        }

        if top_cells.len() == cells_per_candidate {
            evaluated_lineages.insert(record.lineage_group_id.clone());
            let mut seed_groups = BTreeMap::<(String, usize), Vec<f64>>::new();
            for cell in &top_cells {
                kpi_cells.push(KpiCell {
                    task: table_task(&cell.task)?,
                    train_rows: cell.train_rows,
                    features: cell.features,
                    auditor: cell.auditor_id.clone(),
                    size_multiplier: cell.size_multiplier,
                    lineage_group_id: cell.lineage_group_id.clone(),
                    generation_seed: cell.generation_seed,
                    auditor_seed: cell.auditor_seed,
                    null_loss: cell.null_loss,
                    trtr_loss: cell.trtr_loss,
                    tstr_loss: cell.tstr_loss,
                    calibration_degradation: cell.calibration_degradation,
                    rare_class_or_tail_retention: cell.rare_class_or_tail_retention,
                    supported_subgroup_retention: cell.supported_subgroup_retention,
                    nominal_95_coverage: cell.nominal_95_coverage,
                });
                update_optional_max(
                    &mut calibration_degradation_max,
                    cell.calibration_degradation,
                );
                update_optional_min(
                    &mut rare_tail_subgroup_retention_min,
                    cell.rare_class_or_tail_retention,
                );
                update_optional_min(
                    &mut rare_tail_subgroup_retention_min,
                    cell.supported_subgroup_retention,
                );
                update_optional_min(&mut driver_agreement_min, cell.driver_agreement);
                update_optional_min(&mut joint_fidelity_min, cell.joint_fidelity);
                update_optional_max(&mut query_error_max, cell.query_p95_normalized_error);
                update_optional_min(&mut nominal_coverage_min, cell.nominal_95_coverage);
                update_optional_max(&mut nominal_coverage_max, cell.nominal_95_coverage);
                update_optional_max(&mut type_i_error_max, cell.type_i_error);
                update_optional_max(&mut membership_auc_max, cell.membership_auc);
                feature_importance_complete &= cell.version == JOB_EVIDENCE_VERSION
                    && cell.feature_importance_feature_count == cell.features
                    && cell.feature_importance_real_shares.len() == cell.features
                    && cell.feature_importance_synthetic_shares.len() == cell.features;
                if cell.feature_importance_informative_count >= 2 {
                    feature_importance_applicable = true;
                    if let Some(value) = cell.feature_importance_spearman {
                        update_optional_min(&mut feature_importance_spearman_min, Some(value));
                    } else {
                        feature_importance_missing_rank = true;
                    }
                    update_optional_min(
                        &mut feature_importance_top_k_jaccard_min,
                        Some(cell.feature_importance_top_k_agreement),
                    );
                }
                update_optional_max(
                    &mut attribute_advantage_max,
                    cell.attribute_inference_advantage,
                );
                exact_copies += cell.exact_copies;
                near_copies += cell.near_copies;
                canary_extractions += cell.canary_extractions;
                lineage_leaks += cell.lineage_leaks;
                seed_groups
                    .entry((cell.auditor_id.clone(), cell.size_multiplier))
                    .or_default()
                    .push(cell.tstr_loss);
            }
            for values in seed_groups.values() {
                let deviation = if values.len() > 1 {
                    let average = values.iter().sum::<f64>() / values.len() as f64;
                    (values
                        .iter()
                        .map(|value| (value - average).powi(2))
                        .sum::<f64>()
                        / (values.len() - 1) as f64)
                        .sqrt()
                } else {
                    0.0
                };
                update_optional_max(&mut across_seed_validation_loss_stddev, Some(deviation));
            }
        }

        if block.failures.is_empty() && block.evidence.len() == cells_per_record {
            let mut outcomes = BTreeMap::<String, CertCandidateOutcome>::new();
            for candidate in &candidate_ids {
                let matching = block
                    .evidence
                    .iter()
                    .filter(|cell| &cell.candidate_id == candidate)
                    .collect::<Vec<_>>();
                if matching.len() != cells_per_candidate {
                    continue;
                }
                outcomes.insert(
                    candidate.clone(),
                    CertCandidateOutcome {
                        utility: matching
                            .iter()
                            .map(|cell| bounded_router_utility(cell))
                            .sum::<f64>()
                            / matching.len() as f64,
                        runtime_ms: matching
                            .iter()
                            .map(|cell| cell.runtime_ms as f64)
                            .sum::<f64>()
                            / matching.len() as f64,
                    },
                );
            }
            if outcomes.len() == candidate_ids.len() {
                complete_oracle_outcomes += 1;
                let (_oracle_id, oracle) = outcomes
                    .iter()
                    .max_by(|left, right| {
                        left.1
                            .utility
                            .total_cmp(&right.1.utility)
                            .then_with(|| right.0.cmp(left.0))
                    })
                    .expect("complete candidate outcomes");
                let selected = outcomes
                    .get(top_candidate)
                    .expect("validated top candidate outcome");
                let selected_regret = (oracle.utility - selected.utility).max(0.0);
                regrets.push(selected_regret);
                profile_regrets
                    .entry(record.structural_profile.clone())
                    .or_default()
                    .push(selected_regret);
                let top_four_best = routing
                    .top_four_candidates
                    .iter()
                    .filter_map(|candidate| outcomes.get(candidate))
                    .map(|outcome| outcome.utility)
                    .reduce(f64::max)
                    .unwrap_or(f64::NEG_INFINITY);
                top_four_hits += usize::from(oracle.utility - top_four_best <= 1e-9);
                let mut random_hasher = blake3::Hasher::new();
                random_hasher.update(record.lineage_group_id.as_bytes());
                random_hasher.update(record.structural_profile.as_bytes());
                let random_index =
                    usize::from(random_hasher.finalize().as_bytes()[1]) % candidate_ids.len();
                let random = outcomes
                    .get(&candidate_ids[random_index])
                    .expect("frozen random candidate");
                random_regrets.push((oracle.utility - random.utility).max(0.0));
                if let Some(fixed) = outcomes.get(&router_evidence.best_fixed_candidate_id) {
                    fixed_regrets.push((oracle.utility - fixed.utility).max(0.0));
                    let max_runtime = outcomes
                        .values()
                        .map(|outcome| outcome.runtime_ms)
                        .reduce(f64::max)
                        .unwrap_or(0.0)
                        + 1.0;
                    let hypervolume = |outcome: &CertCandidateOutcome| {
                        outcome.utility.max(0.0) * (max_runtime - outcome.runtime_ms).max(0.0)
                    };
                    hypervolume_pairs.push((hypervolume(selected), hypervolume(fixed)));
                }
            }
        }
    }
    metrics_writer
        .flush()
        .map_err(|error| io_error(&metrics_partial, error))?;
    metrics_writer
        .get_ref()
        .sync_all()
        .map_err(|error| io_error(&metrics_partial, error))?;
    receipts_writer
        .flush()
        .map_err(|error| io_error(&receipts_partial, error))?;
    receipts_writer
        .get_ref()
        .sync_all()
        .map_err(|error| io_error(&receipts_partial, error))?;
    drop(metrics_writer);
    drop(receipts_writer);
    fs::rename(&metrics_partial, out.join("metrics.jsonl"))
        .map_err(|error| io_error(out.join("metrics.jsonl"), error))?;
    fs::rename(&receipts_partial, out.join("receipts.jsonl"))
        .map_err(|error| io_error(out.join("receipts.jsonl"), error))?;

    let count_extra_json = |directory: &Path| -> Result<usize> {
        if !directory.is_dir() {
            return Ok(0);
        }
        let mut extra = 0usize;
        for entry in fs::read_dir(directory).map_err(|error| io_error(directory, error))? {
            let path = entry.map_err(|error| io_error(directory, error))?.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "json")
                && path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .is_none_or(|stem| !expected_identities.contains(stem))
            {
                extra += 1;
            }
        }
        Ok(extra)
    };
    let extra_blocks = count_extra_json(&cert_out.join("blocks"))?;
    let extra_receipts = count_extra_json(&cert_out.join("receipts"))?;
    let reconciliation = CertReceiptReconciliation {
        format: "dope-cert-receipt-reconciliation".into(),
        version: 1,
        expected_blocks: cohort.records.len(),
        reconciled_blocks: receipts_reconciled,
        extra_blocks,
        extra_receipts,
        issues,
    };
    write_canonical(&out.join("receipt-reconciliation.json"), &reconciliation)?;

    let missing_lineages = cohort
        .records
        .len()
        .saturating_sub(evaluated_lineages.len());
    let kpi_report = aggregate_kpis(
        &kpi_cells,
        cohort.records.len(),
        missing_lineages,
        "validation-cert",
    )?;
    write_canonical(&out.join("validation-kpis.json"), &kpi_report)?;
    let coverage_entries = coverage_groups
        .into_iter()
        .map(|((task, profile, auditor, size_multiplier), aggregate)| {
            let missing = aggregate
                .eligible
                .saturating_sub(aggregate.evaluated + aggregate.timed_out);
            CoverageEntry {
                task,
                structural_profile: profile,
                auditor,
                size_multiplier,
                eligible: aggregate.eligible,
                evaluated: aggregate.evaluated,
                missing,
                timed_out: aggregate.timed_out,
                required: true,
            }
        })
        .collect::<Vec<_>>();
    let required_complete = !coverage_entries.is_empty()
        && coverage_entries
            .iter()
            .all(|entry| entry.missing == 0 && entry.timed_out == 0);
    let coverage = CoverageReport {
        format: "dope-kpi-coverage".into(),
        version: 1,
        tier: "validation-cert".into(),
        entries: coverage_entries,
        required_complete,
    };
    coverage.validate()?;
    write_canonical(&out.join("kpi-coverage.json"), &coverage)?;

    let mut frontier = Vec::new();
    for candidate in &candidate_ids {
        let aggregate = candidate_aggregates
            .get(candidate)
            .expect("initialized candidate aggregate");
        let attempted = cohort.records.len() * cells_per_candidate;
        frontier.push(CandidateFrontierEntry {
            candidate_id: candidate.clone(),
            implementation_hash: empirical_backends()
                .into_iter()
                .find(|backend| backend.id == *candidate)
                .map(|backend| backend.implementation_hash)
                .expect("frozen candidate implementation"),
            attempted,
            succeeded: aggregate.succeeded,
            failed: attempted.saturating_sub(aggregate.succeeded + aggregate.timed_out),
            timed_out: aggregate.timed_out,
            mean_retention: (aggregate.retention_count > 0)
                .then_some(aggregate.retention_sum / aggregate.retention_count.max(1) as f64),
            runtime_p95_ms: histogram_percentile(&aggregate.runtimes, aggregate.succeeded, 0.95),
            peak_memory_p95_bytes: histogram_percentile(
                &aggregate.memories,
                aggregate.succeeded,
                0.95,
            ),
            artifact_bytes_p95: histogram_percentile(
                &aggregate.artifact_bytes,
                aggregate.succeeded,
                0.95,
            ),
        });
    }
    write_canonical(&out.join("candidate-frontier.json"), &frontier)?;

    let regret = mean_and_upper_95(&regrets).map(|(mean, _)| mean);
    let random_regret = mean_and_upper_95(&random_regrets).map(|(mean, _)| mean);
    let fixed_regret = mean_and_upper_95(&fixed_regrets).map(|(mean, _)| mean);
    let regret_by_profile = profile_regrets
        .iter()
        .filter_map(|(profile, values)| {
            mean_and_upper_95(values).map(|(mean_regret, upper_95)| {
                (
                    profile.clone(),
                    ProfileRegretDetail {
                        outcomes: values.len(),
                        mean_regret,
                        upper_95,
                    },
                )
            })
        })
        .collect::<BTreeMap<_, _>>();
    let maximum_profile_regret_upper = regret_by_profile
        .values()
        .map(|detail| detail.upper_95)
        .reduce(f64::max);
    let top_four_oracle_recall = (complete_oracle_outcomes > 0)
        .then_some(top_four_hits as f64 / complete_oracle_outcomes as f64);
    let reference_hypervolume = if hypervolume_pairs.is_empty() {
        None
    } else {
        Some(
            hypervolume_pairs
                .iter()
                .map(|(_, fixed)| fixed.abs())
                .sum::<f64>()
                / hypervolume_pairs.len() as f64,
        )
    };
    let hypervolume_improvements = reference_hypervolume
        .filter(|reference| *reference > 0.0)
        .map(|reference| {
            hypervolume_pairs
                .iter()
                .map(|(routed, fixed)| (routed - fixed) / reference.max(1e-9))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let (paired_hypervolume_improvement, paired_hypervolume_ci_lower) =
        mean_and_lower_95(&hypervolume_improvements)
            .map_or((None, None), |(mean, lower)| (Some(mean), Some(lower)));
    let router_inference_p95_ms = float_percentile(&mut router_inference_times, 0.95);
    let sketch_p95_ms = float_percentile(&mut router_sketch_times, 0.95);
    let work_avoided_fraction = (valid_expected_cells > 0)
        .then_some(1.0 - selected_expected_cells as f64 / valid_expected_cells as f64);
    let latency_reduction_fraction =
        (exhaustive_runtime_ms > 0.0).then_some(1.0 - selected_runtime_ms / exhaustive_runtime_ms);
    let actual_beats_random = regret
        .zip(random_regret)
        .is_some_and(|(routed, random)| routed < random);
    let actual_beats_fixed = regret
        .zip(fixed_regret)
        .is_some_and(|(routed, fixed)| routed < fixed);
    let all_auditors_exercised = auditor_availability.values().all(|count| *count > 0);
    let certification = CertificationDetail {
        eligible_lineages: cohort.records.len(),
        evaluated_lineages: evaluated_lineages.len(),
        missing_lineages,
        timeout_lineages: timeout_lineages.len(),
        exhaustive_cells: expected_cells,
        routed_cells,
        shadow_cells,
        failed_cells: expected_cells.saturating_sub(succeeded_cells),
        top_four_oracle_recall,
        mean_candidate_regret: regret,
        random_candidate_regret: random_regret,
        best_fixed_candidate_id: router_evidence.best_fixed_candidate_id.clone(),
        best_fixed_candidate_regret: fixed_regret,
        beats_random_candidate: actual_beats_random,
        beats_best_fixed_candidate: actual_beats_fixed,
        maximum_profile_regret_upper,
        regret_by_profile,
        work_avoided_fraction,
        latency_reduction_fraction,
        router_bundle_bytes: fs::metadata(&frozen_router_path)
            .map_err(|error| io_error(&frozen_router_path, error))?
            .len(),
        router_inference_p95_ms,
        sketch_p95_ms,
        paired_hypervolume_improvement,
        paired_hypervolume_ci_lower,
        candidate_availability,
        auditor_availability,
        candidate_implementation_hashes: empirical_backends()
            .into_iter()
            .map(|candidate| (candidate.id, candidate.implementation_hash))
            .collect(),
        auditor_implementation_hashes: auditor_specs()
            .into_iter()
            .map(|auditor| (auditor.id, auditor.frozen_hash))
            .collect(),
        receipts_expected: cohort.records.len(),
        receipts_reconciled,
    };
    write_canonical(&out.join("certification-detail.json"), &certification)?;

    let signatures_valid = receipts_reconciled == cohort.records.len()
        && reconciliation.issues.is_empty()
        && extra_blocks == 0
        && extra_receipts == 0;
    let all_candidates_exercised_both_tasks = candidate_aggregates.values().all(|aggregate| {
        aggregate.tasks.contains("binary") && aggregate.tasks.contains("regression")
    });
    let gate_evidence = GateEvidence {
        ptf_v1: kpi_report.ptf_v1,
        calibration_degradation_max,
        rare_class_tail_subgroup_retention_min: rare_tail_subgroup_retention_min,
        driver_agreement_min,
        joint_fidelity_min,
        query_p95_normalized_error_max: query_error_max,
        nominal_95_coverage_min: nominal_coverage_min,
        nominal_95_coverage_max: nominal_coverage_max,
        type_i_error_max,
        membership_auc_max,
        feature_importance_complete: feature_importance_complete && valid_expected_cells > 0,
        feature_importance_applicable,
        feature_importance_spearman_min: if feature_importance_missing_rank {
            None
        } else {
            feature_importance_spearman_min
        },
        feature_importance_top_k_jaccard_min,
        attribute_inference_advantage_max: attribute_advantage_max,
        exact_copies,
        near_copies,
        canary_extractions,
        lineage_leaks,
        validation_training_regret_upper: Some(router_evidence.maximum_profile_regret_upper),
        across_seed_validation_loss_stddev,
        uncertainty_coverage_min: nominal_coverage_min,
        uncertainty_coverage_max: nominal_coverage_max,
        beats_random_router: router_evidence.beats_random,
        beats_best_fixed_candidate: router_evidence.beats_best_fixed,
        learned_representation_used: true,
        pareto_hypervolume_improvement: paired_hypervolume_improvement,
        pareto_hypervolume_ci_lower: paired_hypervolume_ci_lower,
        all_candidates_exercised_both_tasks,
        all_auditors_pinned_and_exercised: all_auditors_exercised,
        byte_reconciliation_exact: signatures_valid && all_artifact_bytes_nonzero,
        signatures_and_receipts_valid: signatures_valid,
        controller_free_gib: controller.available_bytes / GIB,
        evaluator_free_gib: evaluator.available_bytes / GIB,
    };
    write_canonical(&out.join("gate-evidence.json"), &gate_evidence)?;
    let mut failed_gates = gate_evidence
        .failed_gates(&crate::production::KpiContract::embedded()?, &coverage)
        .into_iter()
        .collect::<BTreeSet<_>>();
    failed_gates.extend(
        router_evidence
            .failed_gates()
            .into_iter()
            .map(|gate| format!("router_{gate}")),
    );
    if !top_four_oracle_recall.is_some_and(|value| value >= 0.99) {
        failed_gates.insert("validation_cert_top_four_oracle_recall".into());
    }
    if !maximum_profile_regret_upper.is_some_and(|value| value <= 0.02) {
        failed_gates.insert("validation_cert_profile_regret".into());
    }
    if !actual_beats_random {
        failed_gates.insert("validation_cert_beats_random".into());
    }
    if !actual_beats_fixed {
        failed_gates.insert("validation_cert_beats_best_fixed".into());
    }
    if !router_inference_p95_ms.is_some_and(|value| value <= 5.0) {
        failed_gates.insert("validation_cert_router_inference_p95".into());
    }
    if !sketch_p95_ms.is_some_and(|value| value <= 1_000.0) {
        failed_gates.insert("validation_cert_sketch_p95".into());
    }
    let failed_gates = failed_gates.into_iter().collect::<Vec<_>>();
    let kpi = KpiSummary::from_evidence(
        Some(&kpi_report),
        Some(&coverage),
        maximum_profile_regret_upper,
    );
    let mut counts = BTreeMap::new();
    let non_timeout_failures = explicit_failed_cells.saturating_sub(timed_out_cells);
    let pending = expected_cells.saturating_sub(succeeded_cells + explicit_failed_cells);
    counts.insert(JobState::Pending, pending);
    counts.insert(JobState::Leased, 0);
    counts.insert(JobState::Running, 0);
    counts.insert(JobState::Succeeded, succeeded_cells);
    counts.insert(JobState::Failed, non_timeout_failures);
    counts.insert(JobState::TimedOut, timed_out_cells);
    let terminal = succeeded_cells + explicit_failed_cells;
    let status = crate::ledger::LedgerStatus {
        format: "dope-campaign-ledger-status".into(),
        version: 1,
        counts,
        total: expected_cells,
        training_progress: (expected_cells > 0).then_some(terminal as f64 / expected_cells as f64),
        completed_gold_labels: succeeded_cells,
        failed_jobs: non_timeout_failures,
        timed_out_jobs: timed_out_cells,
        throughput_jobs_per_hour: None,
        coverage: kpi.coverage,
        current_router_validation_regret: maximum_profile_regret_upper,
        ptf_v1: kpi_report.ptf_v1,
        production_score_available: kpi.production_score_available && failed_gates.is_empty(),
        updated_unix_seconds: unix_now(),
    };
    let bundle = MetricsBundle {
        format: "dope-campaign-metrics".into(),
        version: 1,
        status,
        kpi,
        candidate_frontier: frontier,
        router: Some(router_evidence),
        failed_gates,
        validation_cert_open_count: state.validation_cert_open_count,
        sealed_test_open_count: state.sealed_test_open_count,
        receipts_reconciled,
        certification: Some(certification),
    };
    write_canonical(&out.join("metrics-summary.json"), &bundle)?;
    Ok(bundle)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReleaseDecision {
    pub format: String,
    pub version: u8,
    pub release_identity: Option<String>,
    pub measured_ptf_v1: Option<f64>,
    pub failed_gates: Vec<String>,
    pub sealed_test_open_count: usize,
}

pub fn failed_release_decision(
    state_dir: &Path,
    metrics: &MetricsBundle,
) -> Result<ReleaseDecision> {
    let state = load_state(state_dir)?;
    let decision = ReleaseDecision {
        format: "dope-release-decision".into(),
        version: 1,
        release_identity: None,
        measured_ptf_v1: metrics.kpi.ptf_v1,
        failed_gates: metrics.failed_gates.clone(),
        sealed_test_open_count: state.sealed_test_open_count,
    };
    Ok(decision)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_and_gpu_worker_cannot_open_validation_cert() {
        for host in ["xbabe1", "xbabe2"] {
            assert!(require_named_host(host, "xbabe3", "validation-cert authorization").is_err());
        }
        assert!(require_named_host("xbabe3", "xbabe3", "validation-cert authorization").is_ok());
    }

    #[test]
    fn router_promotion_requires_every_frozen_gate() {
        let evidence = RouterEvidence {
            format: "dope-router-evidence".into(),
            version: 1,
            top_four_oracle_recall: 0.99,
            maximum_profile_regret_upper: 0.02,
            beats_random: true,
            beats_best_fixed: true,
            beats_ga2m: true,
            paired_hypervolume_improvement: 0.05,
            paired_hypervolume_ci_lower: 0.001,
            student_max_abs_difference: 0.001,
            top_choice_agreement: 0.999,
            bundle_bytes: 1024,
            inference_p95_ms: 5.0,
            sketch_p95_ms: 1_000.0,
            training_evidence_sha256: "a".repeat(64),
            best_fixed_candidate_id: empirical_backends()[0].id.clone(),
            student_mean_regret: 0.001,
            random_mean_regret: 0.01,
            best_fixed_mean_regret: 0.005,
            ga2m_mean_regret: 0.004,
        };
        assert!(evidence.failed_gates().is_empty());
        let mut failed = evidence;
        failed.beats_ga2m = false;
        assert_eq!(failed.failed_gates(), vec!["beats_ga2m"]);
    }

    #[test]
    fn receipt_key_parser_rejects_non_hex() {
        let path = std::env::temp_dir().join(format!("dope-key-{}", std::process::id()));
        fs::write(&path, "not-a-key").unwrap();
        assert!(load_receipt_key(&path).is_err());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn receipt_key_creation_is_exclusive_and_private() {
        use std::os::unix::fs::PermissionsExt;

        let path = std::env::temp_dir().join(format!("dope-new-key-{}", std::process::id()));
        let _ = fs::remove_file(&path);
        let hashes = create_receipt_key(&path).unwrap();
        assert!(is_lower_hex(&hashes.sha256, &[64]));
        assert!(load_receipt_key(&path).is_ok());
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(create_receipt_key(&path).is_err());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn certification_statistics_do_not_fill_missing_evidence() {
        assert_eq!(mean_and_upper_95(&[]), None);
        assert_eq!(mean_and_lower_95(&[]), None);
        assert_eq!(float_percentile(&mut [], 0.95), None);
        let mut histogram = BTreeMap::from([(1, 1), (5, 2), (9, 1)]);
        assert_eq!(histogram_percentile(&histogram, 4, 0.95), Some(9));
        histogram.clear();
        assert_eq!(histogram_percentile(&histogram, 0, 0.95), None);
        assert!(is_timeout_failure("CUDA deadline exceeded"));
        assert!(!is_timeout_failure("deterministic model failure"));
    }
}

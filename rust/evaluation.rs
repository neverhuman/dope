//! Nonadaptive protected evaluation. Deploy the owner endpoint under a separate UID.
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::access::{deny_evaluator_network, effective_uid, peer_uid, read_regular};
use crate::certification::{CertificationReport, certify_frozen_tables};
use crate::codec::decode_kernel;
use crate::compiler::{CompileOptions, CompileResult, compile_kernel_from_arrays};
use crate::contract::ReleasePolicy;
use crate::data::Table;
use crate::error::{DopeError, Result};
use crate::evidence::{canonical_json, parse_json, validate_behavior_contract};
use crate::model::Task;

pub mod worker;

include!("evaluation/owner.rs");

fn denied() -> DopeError {
    DopeError::Data("protected_evaluation_denied".into())
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn encoded(value: &impl Serialize) -> Result<Vec<u8>> {
    canonical_json(&serde_json::to_value(value)?)
}
fn digest_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Only development bytes enter this type; no path, array or callback for final labels.
pub struct SelectionView {
    table: Table,
    task: Task,
    fit_hash: String,
    policy: ReleasePolicy,
}
impl SelectionView {
    pub fn open(train: &Path, task: Task, policy: &ReleasePolicy) -> Result<Self> {
        if effective_uid() == 0 {
            return Err(denied());
        }
        let bytes = read_regular(train, None, false)?;
        Ok(Self {
            table: table(&bytes, task, policy)?,
            task,
            fit_hash: sha(&bytes),
            policy: policy.clone(),
        })
    }
    pub fn select(&self, options: &CompileOptions) -> Result<CompileResult> {
        if !options.neural_target_weight.is_finite()
            || !options.neural_structural_penalty.is_finite()
            || serde_json::to_value(&options.release_policy)? != serde_json::to_value(&self.policy)?
            || std::env::vars_os()
                .any(|(key, _)| key.to_string_lossy().starts_with("DOPE_RESEARCH_"))
        {
            return Err(denied());
        }
        let mut flat = Vec::with_capacity(self.table.rows * self.table.features);
        for row in 0..self.table.rows {
            for column in &self.table.columns {
                flat.push(column[row]);
            }
        }
        compile_kernel_from_arrays(
            &flat,
            &self.table.target,
            self.table.rows,
            self.table.features,
            self.task,
            options,
        )
    }
    pub fn freeze(
        &self,
        compiled: &CompileResult,
        options: &CompileOptions,
        seed: u64,
        runtime_dictionary_bytes: usize,
        supported_datasets: usize,
    ) -> Result<FrozenSelection> {
        let selected = &compiled.report.selected_candidate;
        if compiled.candidate_artifacts.get(selected) != Some(&compiled.artifact) {
            return Err(denied());
        }
        let configuration = json!({"policy":options.release_policy,"seed":options.seed,"deadline":options.deadline.map(|d|d.as_nanos().to_string()),"beam_width":options.beam_width,"quantization_profiles":options.quantization_profiles,"language":options.language,"backend_id":options.backend_id,"neural_target_weight":options.neural_target_weight,"neural_structural_penalty":options.neural_structural_penalty,"evaluation_seed":seed,"runtime_dictionary_bytes":runtime_dictionary_bytes,"supported_datasets":supported_datasets});
        let frozen = FrozenSelection {
            candidate_id: selected.clone(),
            artifact: compiled.artifact.clone(),
            fit_hash: self.fit_hash.clone(),
            configuration_hash: sha(&canonical_json(&configuration)?),
            configuration,
            selection_receipt_hash: sha(&encoded(&compiled.report)?),
            evaluation_code_hash: evaluation_code_hash(),
            seed,
            runtime_dictionary_bytes,
            supported_datasets,
            policy: self.policy.clone(),
        };
        frozen.validate()?;
        Ok(frozen)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenSelection {
    candidate_id: String,
    artifact: Vec<u8>,
    fit_hash: String,
    configuration_hash: String,
    configuration: serde_json::Value,
    selection_receipt_hash: String,
    evaluation_code_hash: String,
    seed: u64,
    runtime_dictionary_bytes: usize,
    supported_datasets: usize,
    policy: ReleasePolicy,
}
impl FrozenSelection {
    pub fn identity(&self) -> Result<String> {
        Ok(sha(&encoded(self)?))
    }
    pub fn artifact(&self) -> &[u8] {
        &self.artifact
    }
    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }
    pub(crate) fn parameters(&self) -> (u64, usize, usize, &ReleasePolicy) {
        (
            self.seed,
            self.runtime_dictionary_bytes,
            self.supported_datasets,
            &self.policy,
        )
    }
    fn validate(&self) -> Result<()> {
        self.policy.validate()?;
        if self.configuration_hash != sha(&canonical_json(&self.configuration)?)
            || self.configuration["evaluation_seed"] != self.seed
            || self.configuration["runtime_dictionary_bytes"] != self.runtime_dictionary_bytes
            || self.configuration["supported_datasets"] != self.supported_datasets
            || self.configuration["policy"] != serde_json::to_value(&self.policy)?
            || self.evaluation_code_hash != evaluation_code_hash()
            || self.supported_datasets == 0
            || !self.policy.accepts_artifact_bytes(self.artifact.len())
            || [
                &self.fit_hash,
                &self.configuration_hash,
                &self.selection_receipt_hash,
            ]
            .iter()
            .any(|v| !digest_valid(v))
            || crate::codec::reencode_canonical(&self.artifact)? != self.artifact
        {
            return Err(denied());
        }
        Ok(())
    }
}

pub fn evaluation_code_hash() -> String {
    let mut h = Sha256::new();
    h.update(include_bytes!("evaluation.rs"));
    h.update(include_bytes!("evaluation/owner.rs"));
    h.update(include_bytes!("evaluation/worker.rs"));
    h.update(include_bytes!("evaluation/worker/schema.rs"));
    h.update(include_bytes!("../schemas/model-worker-v1.schema.json"));
    h.update(include_bytes!("access.rs"));
    h.update(include_bytes!("certification.rs"));
    h.update(include_bytes!("certification/privacy_attacks.rs"));
    h.update(include_bytes!("certification/privacy_attacks/certify.rs"));
    h.update(include_bytes!("certification/part_01.rs"));
    h.update(include_bytes!("certification/part_02.rs"));
    h.update(include_bytes!(
        "certification/privacy_attacks/finalize_certification.rs"
    ));
    h.update(include_bytes!("../Cargo.lock"));
    h.update(json!(crate::contract::auditor_specs()).to_string());
    h.update(crate::contract::candidate_implementation_hash(
        "frozen_evaluation",
    ));
    format!("{:x}", h.finalize())
}
pub fn frozen_batch_hash(batch: &[FrozenSelection]) -> Result<String> {
    Ok(sha(&encoded(&batch)?))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationDeclaration {
    pub confirmation_id: String,
    pub batch_hash: String,
    pub protocol_hash: String,
    pub fit_hash: String,
    pub query_hash: String,
    pub producer_uid: u32,
    pub max_attempts: u32,
    pub query_open_budget: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationEvent {
    pub confirmation_id: String,
    pub batch_hash: String,
    pub declaration_hash: String,
    pub attempt: u32,
    pub kind: String,
    pub query_opens_charged: u32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationOutcome {
    pub batch_hash: String,
    pub reports: Vec<CertificationReport>,
    pub events: Vec<EvaluationEvent>,
    pub query_opens_charged: u32,
    pub sealed_test_open_count: usize,
}

/// A live connection authenticated by kernel credentials; never serializable.
pub struct EvaluationCapability {
    connection: UnixStream,
}
impl EvaluationCapability {
    pub fn connect(socket: &Path, evaluator_uid: u32) -> Result<Self> {
        if effective_uid() == 0 || evaluator_uid == effective_uid() {
            return Err(denied());
        }
        let metadata = fs::symlink_metadata(socket).map_err(|_| denied())?;
        use std::os::unix::fs::{FileTypeExt, MetadataExt};
        if !metadata.file_type().is_socket() || metadata.uid() != evaluator_uid {
            return Err(denied());
        }
        let connection = UnixStream::connect(socket).map_err(|_| denied())?;
        if peer_uid(&connection)? != evaluator_uid {
            return Err(denied());
        }
        connection
            .set_read_timeout(Some(Duration::from_secs(600)))
            .map_err(|_| denied())?;
        connection
            .set_write_timeout(Some(Duration::from_secs(60)))
            .map_err(|_| denied())?;
        Ok(Self { connection })
    }
}

/// Exactly the supplied frozen batch is evaluated. This interface has no ranking or callbacks.
pub fn evaluate_frozen_batch(
    mut capability: EvaluationCapability,
    batch: &[FrozenSelection],
) -> Result<EvaluationOutcome> {
    if batch.is_empty() || batch.len() > 64 {
        return Err(denied());
    }
    for frozen in batch {
        frozen.validate()?;
    }
    let identity = frozen_batch_hash(batch)?;
    let request = encoded(&batch)?;
    if request.len() > 16 * 1024 * 1024 {
        return Err(denied());
    }
    capability
        .connection
        .write_all(&request)
        .map_err(|_| denied())?;
    capability
        .connection
        .shutdown(std::net::Shutdown::Write)
        .map_err(|_| denied())?;
    let mut bytes = Vec::new();
    capability
        .connection
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| denied())?;
    let outcome: EvaluationOutcome =
        serde_json::from_value(parse_json(&bytes)?).map_err(|_| denied())?;
    let last = outcome.events.last().ok_or_else(denied)?;
    let charged = outcome.events.iter().try_fold(0u32, |total, event| {
        total
            .checked_add(event.query_opens_charged)
            .ok_or_else(denied)
    })?;
    if outcome.batch_hash != identity
        || outcome.reports.len() != batch.len()
        || last.kind != "completed"
        || outcome.events.iter().any(|event| {
            event.batch_hash != identity
                || event.confirmation_id != last.confirmation_id
                || event.declaration_hash != last.declaration_hash
        })
        || charged != outcome.query_opens_charged
        || outcome.sealed_test_open_count
            != outcome
                .events
                .iter()
                .filter(|event| event.kind == "query_opened")
                .count()
        || outcome.reports.iter().zip(batch).any(|(report, frozen)| {
            report.artifact_bytes != frozen.artifact.len()
                || serde_json::to_value(&report.release_policy).ok()
                    != serde_json::to_value(&frozen.policy).ok()
        })
    {
        return Err(denied());
    }
    Ok(outcome)
}

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
use crate::codec::{decode_kernel, encode_kernel};
use crate::compiler::{CompileOptions, CompileResult, compile_kernel_from_arrays};
use crate::contract::ReleasePolicy;
use crate::data::Table;
use crate::error::{DopeError, Result};
use crate::evidence::{canonical_json, parse_json, validate_behavior_contract};
use crate::model::Task;

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
            || encode_kernel(&decode_kernel(&self.artifact)?)? != self.artifact
        {
            return Err(denied());
        }
        Ok(())
    }
}

pub fn evaluation_code_hash() -> String {
    let mut h = Sha256::new();
    h.update(include_bytes!("evaluation.rs"));
    h.update(include_bytes!("access.rs"));
    h.update(include_bytes!("certification.rs"));
    h.update(include_bytes!("certification/privacy_attacks.rs"));
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

fn table(bytes: &[u8], task: Task, policy: &ReleasePolicy) -> Result<Table> {
    let mut flat = Vec::new();
    let mut target = Vec::new();
    let mut features = None;
    for record in csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_reader(bytes)
        .records()
    {
        let record = record.map_err(|_| denied())?;
        if record.len() < 2 || features.is_some_and(|n| n != record.len() - 1) {
            return Err(denied());
        }
        features = Some(record.len() - 1);
        for field in record.iter().take(record.len() - 1) {
            flat.push(
                if field.trim().is_empty() || field.eq_ignore_ascii_case("nan") {
                    f32::NAN
                } else {
                    field.parse().map_err(|_| denied())?
                },
            );
        }
        target.push(record[record.len() - 1].parse().map_err(|_| denied())?);
    }
    Table::from_arrays_with_policy(
        &flat,
        &target,
        target.len(),
        features.ok_or_else(denied)?,
        task,
        policy,
    )
    .map_err(|_| denied())
}

fn event(
    ledger: &mut File,
    declaration: &EvaluationDeclaration,
    attempt: u32,
    kind: &str,
    charge: u32,
) -> Result<EvaluationEvent> {
    let value = EvaluationEvent {
        confirmation_id: declaration.confirmation_id.clone(),
        batch_hash: declaration.batch_hash.clone(),
        declaration_hash: sha(&encoded(declaration)?),
        attempt,
        kind: kind.into(),
        query_opens_charged: charge,
    };
    ledger
        .write_all(&encoded(&value)?)
        .and_then(|_| ledger.write_all(b"\n"))
        .and_then(|_| ledger.sync_all())
        .map_err(|_| denied())?;
    Ok(value)
}

/// Owner process only. Keep the root private (0700), files owner-only, and the
/// declaration immutable before starting the endpoint. The socket may be shared.
/// This installs a process-wide network/exec deny filter; run in a dedicated process.
pub fn serve_frozen_batch(listener: &UnixListener, root: &Path) -> Result<()> {
    let uid = effective_uid();
    use std::os::unix::fs::MetadataExt;
    let metadata = fs::symlink_metadata(root).map_err(|_| denied())?;
    if !metadata.is_dir() || metadata.uid() != uid || metadata.mode() & 0o777 != 0o700 {
        return Err(denied());
    }
    let declaration: EvaluationDeclaration = serde_json::from_value(parse_json(&read_regular(
        &root.join("declaration.json"),
        Some(uid),
        true,
    )?)?)
    .map_err(|_| denied())?;
    if declaration.producer_uid == uid
        || declaration.producer_uid == 0
        || declaration.fit_hash == declaration.query_hash
        || declaration.max_attempts == 0
        || declaration.query_open_budget == 0
        || [
            &declaration.confirmation_id,
            &declaration.batch_hash,
            &declaration.protocol_hash,
            &declaration.fit_hash,
            &declaration.query_hash,
        ]
        .iter()
        .any(|v| !digest_valid(v))
    {
        return Err(denied());
    }
    let protocol_bytes = read_regular(&root.join("protocol.json"), Some(uid), true)?;
    let protocol = parse_json(&protocol_bytes)?;
    validate_behavior_contract(&protocol)?;
    if sha(&protocol_bytes) != declaration.protocol_hash
        || protocol["protocol_frozen"] != true
        || protocol["confirmation_allowed"] != true
    {
        return Err(denied());
    }
    let (mut connection, _) = listener.accept().map_err(|_| denied())?;
    if peer_uid(&connection)? != declaration.producer_uid {
        return Err(denied());
    }
    connection
        .set_read_timeout(Some(Duration::from_secs(60)))
        .map_err(|_| denied())?;
    let mut request = Vec::new();
    (&mut connection)
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut request)
        .map_err(|_| denied())?;
    let batch: Vec<FrozenSelection> =
        serde_json::from_value(parse_json(&request)?).map_err(|_| denied())?;
    if batch.is_empty() || batch.len() > 64 || frozen_batch_hash(&batch)? != declaration.batch_hash
    {
        return Err(denied());
    }
    for frozen in &batch {
        frozen.validate()?;
        if frozen.fit_hash != declaration.fit_hash {
            return Err(denied());
        }
    }
    deny_evaluator_network()?;
    // The owner controls the ledger directory, independently of producer paths.
    let path = root.join(format!("{}.jsonl", declaration.confirmation_id));
    let mut ledger = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
        .map_err(|_| denied())?;
    let metadata = ledger.metadata().map_err(|_| denied())?;
    if !metadata.is_file()
        || metadata.uid() != uid
        || metadata.mode() & 0o077 != 0
        || metadata.nlink() != 1
        || metadata.len() > 16 * 1024 * 1024
    {
        return Err(denied());
    }
    // SAFETY: flock operates on this live file descriptor; the lock is held until drop.
    if unsafe { libc::flock(ledger.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(denied());
    }
    File::open(root)
        .and_then(|f| f.sync_all())
        .map_err(|_| denied())?;
    let mut history = String::new();
    ledger.read_to_string(&mut history).map_err(|_| denied())?;
    let previous = history
        .lines()
        .map(|line| serde_json::from_str::<EvaluationEvent>(line).map_err(|_| denied()))
        .collect::<Result<Vec<_>>>()?;
    let declaration_hash = sha(&encoded(&declaration)?);
    if previous.iter().any(|e| {
        e.batch_hash != declaration.batch_hash
            || e.declaration_hash != declaration_hash
            || e.confirmation_id != declaration.confirmation_id
            || e.attempt == 0
            || !matches!(
                e.kind.as_str(),
                "started" | "query_access" | "query_opened" | "interrupted"
            )
            || e.query_opens_charged != u32::from(e.kind == "query_access")
    }) {
        return Err(denied());
    }
    let attempt = previous
        .iter()
        .map(|e| e.attempt)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(denied)?;
    let charged = previous.iter().try_fold(0u32, |total, e| {
        total.checked_add(e.query_opens_charged).ok_or_else(denied)
    })?;
    let cost = u32::try_from(batch.len()).map_err(|_| denied())?;
    if attempt > declaration.max_attempts
        || charged
            .checked_add(cost)
            .is_none_or(|n| n > declaration.query_open_budget)
    {
        return Err(denied());
    }
    let mut events = previous;
    events.push(event(&mut ledger, &declaration, attempt, "started", 0)?);
    let mut reports = Vec::new();
    for (index, frozen) in batch.iter().enumerate() {
        // Charge and fsync BEFORE opening labels, including failed/interrupted attempts.
        events.push(event(
            &mut ledger,
            &declaration,
            attempt,
            "query_access",
            1,
        )?);
        let result = (|| {
            let fit_bytes = read_regular(&root.join("train.csv"), Some(uid), true)?;
            let query_bytes = read_regular(&root.join("test.csv"), Some(uid), true)?;
            if sha(&fit_bytes) != declaration.fit_hash
                || sha(&query_bytes) != declaration.query_hash
            {
                return Err(denied());
            }
            events.push(event(
                &mut ledger,
                &declaration,
                attempt,
                "query_opened",
                0,
            )?);
            let task = decode_kernel(frozen.artifact())?.task;
            let fit = table(&fit_bytes, task, &frozen.policy)?;
            let query = table(&query_bytes, task, &frozen.policy)?;
            let artifact_path = root.join(format!(
                "{}-{attempt}-{index}.dpk",
                declaration.confirmation_id
            ));
            let mut artifact = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&artifact_path)
                .map_err(|_| denied())?;
            artifact
                .write_all(frozen.artifact())
                .and_then(|_| artifact.sync_all())
                .map_err(|_| denied())?;
            let out = root.join(format!(
                "{}-{attempt}-{index}.certification.json",
                declaration.confirmation_id
            ));
            certify_frozen_tables(frozen, &artifact_path, &out, fit, query)
        })();
        match result {
            Ok(report) => reports.push(report),
            Err(error) => {
                let kind = match &error {
                    DopeError::Io { source, .. }
                        if matches!(
                            source.kind(),
                            std::io::ErrorKind::Interrupted
                                | std::io::ErrorKind::TimedOut
                                | std::io::ErrorKind::WouldBlock
                        ) =>
                    {
                        "interrupted"
                    }
                    _ => "rejected",
                };
                event(&mut ledger, &declaration, attempt, kind, 0)?;
                return Err(denied());
            }
        }
    }
    events.push(event(&mut ledger, &declaration, attempt, "completed", 0)?);
    let opens = events
        .iter()
        .filter(|event| event.kind == "query_opened")
        .count();
    let outcome = EvaluationOutcome {
        batch_hash: declaration.batch_hash,
        reports,
        events,
        query_opens_charged: charged + cost,
        sealed_test_open_count: opens,
    };
    connection
        .write_all(&encoded(&outcome)?)
        .map_err(|_| denied())?;
    Ok(())
}

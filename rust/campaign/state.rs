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
    // Historical router receipts encoded a missing fixed candidate as null.
    // Retain that one wire representation for read-only receipt verification.
    match Option::<String>::deserialize(deserializer)? {
        Some(candidate_id) => Ok(candidate_id),
        None => Ok(String::new()),
    }
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

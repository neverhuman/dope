
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CoverageEntry {
    pub task: String,
    pub structural_profile: String,
    pub auditor: String,
    pub size_multiplier: usize,
    pub eligible: usize,
    pub evaluated: usize,
    pub missing: usize,
    pub timed_out: usize,
    pub required: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CoverageReport {
    pub format: String,
    pub version: u8,
    pub tier: String,
    pub entries: Vec<CoverageEntry>,
    pub required_complete: bool,
}

impl CoverageReport {
    pub fn validate(&self) -> Result<()> {
        let complete = !self.entries.is_empty()
            && self.entries.iter().all(|entry| {
                entry.eligible == entry.evaluated + entry.missing + entry.timed_out
                    && (!entry.required || (entry.missing == 0 && entry.timed_out == 0))
            });
        if self.format != "dope-kpi-coverage"
            || self.version != 1
            || self.required_complete != complete
        {
            return Err(DopeError::Data("invalid KPI coverage report".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GateEvidence {
    pub ptf_v1: Option<f64>,
    pub calibration_degradation_max: Option<f64>,
    pub rare_class_tail_subgroup_retention_min: Option<f64>,
    pub driver_agreement_min: Option<f64>,
    pub joint_fidelity_min: Option<f64>,
    pub query_p95_normalized_error_max: Option<f64>,
    pub nominal_95_coverage_min: Option<f64>,
    pub nominal_95_coverage_max: Option<f64>,
    pub type_i_error_max: Option<f64>,
    pub membership_auc_max: Option<f64>,
    #[serde(default)]
    pub feature_importance_complete: bool,
    #[serde(default)]
    pub feature_importance_applicable: bool,
    #[serde(default)]
    pub feature_importance_spearman_min: Option<f64>,
    #[serde(default)]
    pub feature_importance_top_k_jaccard_min: Option<f64>,
    pub attribute_inference_advantage_max: Option<f64>,
    pub exact_copies: usize,
    pub near_copies: usize,
    pub canary_extractions: usize,
    pub lineage_leaks: usize,
    pub validation_training_regret_upper: Option<f64>,
    pub across_seed_validation_loss_stddev: Option<f64>,
    pub uncertainty_coverage_min: Option<f64>,
    pub uncertainty_coverage_max: Option<f64>,
    pub beats_random_router: bool,
    pub beats_best_fixed_candidate: bool,
    pub learned_representation_used: bool,
    pub pareto_hypervolume_improvement: Option<f64>,
    pub pareto_hypervolume_ci_lower: Option<f64>,
    pub all_candidates_exercised_both_tasks: bool,
    pub all_auditors_pinned_and_exercised: bool,
    pub byte_reconciliation_exact: bool,
    pub signatures_and_receipts_valid: bool,
    pub controller_free_gib: u64,
    pub evaluator_free_gib: u64,
}

impl GateEvidence {
    pub fn failed_gates(&self, contract: &KpiContract, coverage: &CoverageReport) -> Vec<String> {
        let gates = &contract.release_gates;
        let mut failed = Vec::new();
        let mut require = |name: &str, passed: bool| {
            if !passed {
                failed.push(name.into());
            }
        };
        require("ptf_v1", self.ptf_v1.is_some_and(|v| v >= gates.ptf_v1_min));
        require(
            "calibration",
            self.calibration_degradation_max
                .is_some_and(|v| v <= gates.calibration_degradation_max),
        );
        require(
            "rare_tail_subgroup",
            self.rare_class_tail_subgroup_retention_min
                .is_some_and(|v| v >= gates.rare_tail_subgroup_retention_min),
        );
        require(
            "driver",
            self.driver_agreement_min
                .is_some_and(|v| v >= gates.driver_agreement_min),
        );
        require(
            "joint_fidelity",
            self.joint_fidelity_min
                .is_some_and(|v| v >= gates.joint_fidelity_min),
        );
        require(
            "query_fidelity",
            self.query_p95_normalized_error_max
                .is_some_and(|v| v <= gates.query_p95_normalized_error_max),
        );
        require(
            "nominal_95_coverage",
            self.nominal_95_coverage_min
                .is_some_and(|v| v >= gates.nominal_95_coverage_min)
                && self
                    .nominal_95_coverage_max
                    .is_some_and(|v| v <= gates.nominal_95_coverage_max),
        );
        require(
            "type_i_error",
            self.type_i_error_max
                .is_some_and(|v| v <= gates.type_i_error_max),
        );
        require(
            "membership",
            self.membership_auc_max
                .is_some_and(|v| v <= gates.membership_auc_max),
        );
        if contract.version >= 2 {
            require("feature_view_coverage", self.feature_importance_complete);
            if self.feature_importance_applicable {
                require(
                    "feature_importance_spearman",
                    self.feature_importance_spearman_min.is_some_and(|value| {
                        value >= gates.feature_importance_spearman_min.unwrap_or(0.70)
                    }),
                );
                require(
                    "feature_importance_top_k_jaccard",
                    self.feature_importance_top_k_jaccard_min
                        .is_some_and(|value| {
                            value >= gates.feature_importance_top_k_jaccard_min.unwrap_or(0.50)
                        }),
                );
            }
        }
        require(
            "attribute_inference",
            self.attribute_inference_advantage_max
                .is_some_and(|v| v <= gates.attribute_inference_advantage_max),
        );
        require(
            "copy_and_leakage",
            self.exact_copies == 0
                && self.near_copies == 0
                && self.canary_extractions == 0
                && self.lineage_leaks == 0,
        );
        require(
            "generalization_regret",
            self.validation_training_regret_upper
                .is_some_and(|v| v <= gates.validation_training_regret_upper_max),
        );
        require(
            "seed_stability",
            self.across_seed_validation_loss_stddev
                .is_some_and(|v| v <= gates.across_seed_validation_loss_stddev_max),
        );
        require(
            "uncertainty_coverage",
            self.uncertainty_coverage_min.is_some_and(|v| v >= 0.90)
                && self.uncertainty_coverage_max.is_some_and(|v| v <= 0.98),
        );
        require(
            "router",
            self.beats_random_router && self.beats_best_fixed_candidate,
        );
        require(
            "learned_representation",
            !self.learned_representation_used
                || (self
                    .pareto_hypervolume_improvement
                    .is_some_and(|v| v >= 0.05)
                    && self.pareto_hypervolume_ci_lower.is_some_and(|v| v > 0.0)),
        );
        require(
            "candidate_coverage",
            self.all_candidates_exercised_both_tasks,
        );
        require("auditor_coverage", self.all_auditors_pinned_and_exercised);
        require("byte_reconciliation", self.byte_reconciliation_exact);
        require(
            "signatures_and_receipts",
            self.signatures_and_receipts_valid,
        );
        require(
            "coverage",
            coverage.required_complete && coverage.validate().is_ok(),
        );
        require("controller_disk", self.controller_free_gib >= 150);
        require("evaluator_disk", self.evaluator_free_gib >= 150);
        failed
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ReleaseFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub blake3: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ReleaseManifest {
    pub format: String,
    pub version: u8,
    pub release_identity: String,
    pub source_commit: String,
    pub source_tag_object_sha256: String,
    pub environment_lock: ContentHashes,
    pub kpi_contract: ContentHashes,
    pub run_contract: ContentHashes,
    pub files: Vec<ReleaseFile>,
    pub sealed_test_open_count: usize,
    pub signature_kind: String,
    pub public_key_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct PromotionReceipt {
    pub format: String,
    pub version: u8,
    pub rc_manifest_sha256: String,
    pub authorization_token_sha256: String,
    pub sealed_opened_unix_seconds: u64,
    pub sealed_test_open_count: usize,
    pub outcome: String,
    pub final_release_identity: Option<String>,
    pub evidence_sha256: String,
}

pub const REQUIRED_RC_FILES: [&str; 15] = [
    "kernel.dpk",
    "conversion-report.json",
    "synthetic-model.bundle",
    "tstr-certification.json",
    "kpi-contract.json",
    "training-kpis.json",
    "validation-kpis.json",
    "kpi-coverage.json",
    "candidate-frontier.json",
    "formal-dp-frontier.json",
    "run-contract.json",
    "environment-lock.json",
    "sbom.json",
    "notices.txt",
    "receipts.json",
];

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct BuildRcOptions {
    pub source_commit: String,
    pub source_tag_object_sha256: String,
    pub signature_kind: String,
    pub public_key_sha256: String,
}

/// Builds an RC identity only after all frozen evidence gates have passed.
pub fn build_rc_manifest(
    root: &Path,
    evidence: &GateEvidence,
    coverage: &CoverageReport,
    options: &BuildRcOptions,
) -> Result<ReleaseManifest> {
    let contract = KpiContract::embedded()?;
    contract.validate()?;
    coverage.validate()?;
    let failed = evidence.failed_gates(&contract, coverage);
    if !failed.is_empty() {
        return Err(DopeError::Data(format!(
            "release candidate gates failed: {}",
            failed.join(",")
        )));
    }
    if !(is_lower_hex(&options.source_commit, 40) || is_lower_hex(&options.source_commit, 64))
        || !is_lower_hex(&options.source_tag_object_sha256, 64)
        || options.signature_kind.is_empty()
        || !is_lower_hex(&options.public_key_sha256, 64)
    {
        return Err(DopeError::Data(
            "invalid RC source or signature identity".into(),
        ));
    }
    let run_path = root.join("run-contract.json");
    let run_contract: RunContract = read_json(&run_path)?;
    run_contract.validate()?;
    if run_contract.source_commit != options.source_commit {
        return Err(DopeError::Data(
            "RC source commit differs from frozen run contract".into(),
        ));
    }
    let run_hashes = file_hashes(&run_path)?;
    if run_hashes != run_contract.id()? {
        return Err(DopeError::Data("run contract is not canonical".into()));
    }
    let environment_lock = file_hashes(&root.join("environment-lock.json"))?;
    if environment_lock != run_contract.environment_lock {
        return Err(DopeError::Data(
            "environment lock differs from frozen run contract".into(),
        ));
    }
    let kpi_contract = file_hashes(&root.join("kpi-contract.json"))?;
    if kpi_contract.sha256 != KPI_CONTRACT_SHA256 || kpi_contract.blake3 != KPI_CONTRACT_BLAKE3 {
        return Err(DopeError::Data(
            "release KPI contract differs from binary identity".into(),
        ));
    }
    let files = inventory_release_files(root, &REQUIRED_RC_FILES)?;
    Ok(ReleaseManifest {
        format: "dope-release-manifest".into(),
        version: 1,
        release_identity: RELEASE_CANDIDATE_VERSION.into(),
        source_commit: options.source_commit.clone(),
        source_tag_object_sha256: options.source_tag_object_sha256.clone(),
        environment_lock,
        kpi_contract,
        run_contract: run_hashes,
        files,
        sealed_test_open_count: 0,
        signature_kind: options.signature_kind.clone(),
        public_key_sha256: options.public_key_sha256.clone(),
    })
}

pub fn inventory_release_files(root: &Path, names: &[&str]) -> Result<Vec<ReleaseFile>> {
    let expected = names.iter().copied().collect::<BTreeSet<_>>();
    let mut observed = BTreeSet::new();
    for entry in fs::read_dir(root).map_err(|error| io_error(root, error))? {
        let entry = entry.map_err(|error| io_error(root, error))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !expected.contains(name.as_str()) {
            return Err(DopeError::Data(format!("unexpected release file: {name}")));
        }
        let metadata =
            fs::symlink_metadata(entry.path()).map_err(|error| io_error(entry.path(), error))?;
        if !metadata.file_type().is_file() {
            return Err(DopeError::Data(format!(
                "release file is not a regular file: {name}"
            )));
        }
        observed.insert(name);
    }
    if observed.len() != expected.len() {
        return Err(DopeError::Data(
            "release file inventory is incomplete".into(),
        ));
    }
    let mut files = Vec::with_capacity(names.len());
    for name in names {
        let path = root.join(name);
        let metadata = fs::symlink_metadata(&path).map_err(|error| io_error(&path, error))?;
        if !metadata.is_file() {
            return Err(DopeError::Data(format!(
                "release input is not a file: {name}"
            )));
        }
        let digest = file_hashes(&path)?;
        files.push(ReleaseFile {
            path: (*name).into(),
            bytes: metadata.len(),
            sha256: digest.sha256,
            blake3: digest.blake3,
        });
    }
    files.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(files)
}

pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn copy_canonical_json<T: for<'de> Deserialize<'de> + Serialize>(
    source: &Path,
    destination: &Path,
) -> Result<ContentHashes> {
    let value: T = read_json(source)?;
    write_canonical(destination, &value)
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct FrozenContractReceipt {
    pub format: String,
    pub version: u8,
    pub run_contract: ContentHashes,
    pub kpi_contract: ContentHashes,
    pub environment_lock: ContentHashes,
    pub source_commit: String,
    pub output: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_contract_is_canonical_and_bound() {
        let contract = KpiContract::embedded().unwrap();
        contract.validate().unwrap();
        let digest = hashes(&canonical_json(&contract).unwrap());
        assert_eq!(digest.sha256, KPI_CONTRACT_SHA256);
        assert_eq!(digest.blake3, KPI_CONTRACT_BLAKE3);
        let historical = KpiContract::embedded_v1().unwrap();
        historical.validate().unwrap();
        assert_eq!(historical.version, 1);
    }

    #[test]
    fn shape_profiles_do_not_depend_on_names_or_paths() {
        assert_eq!(
            StructuralProfile::from_shape(Task::Binary, 31, 2_000)
                .unwrap()
                .id(),
            "binary/<32/257-2000"
        );
        assert_eq!(
            StructuralProfile::from_shape(Task::Regression, 1_024, 17)
                .unwrap()
                .id(),
            "regression/>=1024/17-64"
        );
    }

    #[test]
    fn kpi_retention_is_unclamped_and_lineage_weighted() {
        let mut cells = Vec::new();
        for lineage in 0..100 {
            for generation_seed in 0..3 {
                for auditor_seed in AUDITOR_SEEDS {
                    cells.push(KpiCell {
                        task: Task::Regression,
                        train_rows: 100,
                        features: 4,
                        auditor: "elastic_net_glm".into(),
                        size_multiplier: 1,
                        lineage_group_id: format!("lineage-{lineage}"),
                        generation_seed,
                        auditor_seed,
                        null_loss: 1.0,
                        trtr_loss: 0.5,
                        tstr_loss: if lineage == 0 { 0.4 } else { 0.505 },
                        calibration_degradation: None,
                        rare_class_or_tail_retention: None,
                        supported_subgroup_retention: None,
                        nominal_95_coverage: None,
                    });
                }
            }
        }
        let report = aggregate_kpis(&cells, 100, 0, "unit").unwrap();
        assert_eq!(report.independently_gated_profiles.len(), 1);
        assert!(report.cells[0].mean_unclamped_retention.unwrap() > 0.99);
        assert!(report.ptf_v1.unwrap() < report.cells[0].mean_unclamped_retention.unwrap());
    }

    #[test]
    fn low_signal_uses_absolute_noninferiority() {
        let cells = vec![KpiCell {
            task: Task::Binary,
            train_rows: 20,
            features: 2,
            auditor: "ga2m".into(),
            size_multiplier: 4,
            lineage_group_id: "g".into(),
            generation_seed: 7,
            auditor_seed: AUDITOR_SEEDS[0],
            null_loss: 1.0,
            trtr_loss: 0.995,
            tstr_loss: 1.004,
            calibration_degradation: None,
            rare_class_or_tail_retention: None,
            supported_subgroup_retention: None,
            nominal_95_coverage: None,
        }];
        let report = aggregate_kpis(&cells, 1, 0, "unit").unwrap();
        assert_eq!(report.ptf_v1, None);
        assert_eq!(report.low_signal_compliance_overall, Some(1.0));
    }

    #[test]
    fn empty_kpi_summary_is_explicitly_unmeasured() {
        let summary = KpiSummary::from_evidence(None, None, None);
        let value = serde_json::to_value(&summary).unwrap();
        assert!(value["ptf_v1"].is_null());
        assert!(value["low_signal_compliance"].is_null());
        assert!(value["coverage"].is_null());
        assert!(value["candidate_regret"].is_null());
        assert_eq!(summary.auditor_availability.required, 6);
        assert_eq!(
            summary.auditor_availability.available,
            crate::contract::auditor_specs()
                .iter()
                .filter(|auditor| auditor.available)
                .count()
        );
        assert!(!summary.production_score_available);
    }

    #[test]
    fn canonical_writes_replace_atomically_without_temporary_debris() {
        let directory =
            std::env::temp_dir().join(format!("dope-canonical-write-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("value.json");
        write_canonical(&path, &serde_json::json!({"value": 1})).unwrap();
        write_canonical(&path, &serde_json::json!({"value": 2})).unwrap();
        assert_eq!(read_json::<Value>(&path).unwrap()["value"], 2);
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        let _ = fs::remove_dir_all(directory);
    }
}

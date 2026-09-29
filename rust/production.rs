use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::contract::{
    AUDITOR_SEEDS, GENERATION_REPEATS, GOLD_SIZE_MULTIPLIERS, RELEASE_SEED,
    SYNTHETIC_SIZE_MULTIPLIERS, auditor_specs, empirical_backends,
};
use crate::error::{DopeError, Result, io_error};
use crate::fitness::MasterFitnessContract;
use crate::model::Task;

/// Public identity of ordinary development builds. An RC identity is emitted
/// only in a successfully validated release-candidate manifest.
pub const DEVELOPMENT_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const RELEASE_CANDIDATE_VERSION: &str = "1.0.0-rc.1";
pub const KPI_CONTRACT_SHA256: &str = env!("DOPE_KPI_CONTRACT_SHA256");
pub const KPI_CONTRACT_BLAKE3: &str = env!("DOPE_KPI_CONTRACT_BLAKE3");
pub const KPI_CONTRACT_V1_SHA256: &str = env!("DOPE_KPI_CONTRACT_V1_SHA256");
pub const KPI_CONTRACT_V1_BLAKE3: &str = env!("DOPE_KPI_CONTRACT_V1_BLAKE3");
pub const CORPUS_MANIFEST_SHA256: &str =
    "045a90e9c0448a547368ec8cac2ca0f8fe1b110b04a0442a33ef407496c75973";
pub const CORPUS_INVENTORY_SHA256: &str =
    "7b905c576a42e9c0b51f4a743d172fd21aa320ba37690bee216c2ccf09a8702e";
pub const EMPIRICAL_ARTIFACT_CEILING_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ContentHashes {
    pub sha256: String,
    pub blake3: String,
}

pub fn canonical_value(value: &Value) -> Value {
    match value {
        Value::Object(entries) => Value::Object(
            entries
                .iter()
                .map(|(key, value)| (key.clone(), canonical_value(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(canonical_value).collect()),
        _ => value.clone(),
    }
}

pub fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&canonical_value(
        &serde_json::to_value(value)?,
    ))?)
}

pub fn hashes(bytes: &[u8]) -> ContentHashes {
    ContentHashes {
        sha256: format!("{:x}", Sha256::digest(bytes)),
        blake3: blake3::hash(bytes).to_hex().to_string(),
    }
}

pub fn file_hashes(path: &Path) -> Result<ContentHashes> {
    let bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    Ok(hashes(&bytes))
}

pub fn write_canonical<T: Serialize>(path: &Path, value: &T) -> Result<ContentHashes> {
    let bytes = canonical_json(value)?;
    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| DopeError::Data("canonical output path has no UTF-8 filename".into()))?;
    let staged_path = parent.join(format!(
        ".{filename}.tmp-{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let write_result = (|| -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staged_path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&staged_path, path)?;
        File::open(parent)?.sync_all()
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&staged_path);
        return Err(io_error(path, error));
    }
    Ok(hashes(&bytes))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConfidenceContract {
    pub method: String,
    pub cluster: String,
    pub equal_weight: String,
    pub one_sided_level: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LowSignalContract {
    pub improvement_threshold_fraction: f64,
    pub noninferiority_fraction_of_null: f64,
    pub overall_compliance: f64,
    pub profile_compliance: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProfileGatingContract {
    pub minimum_groups: usize,
    pub minimum_informative_groups: usize,
    pub sparse_profile: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProfileBands {
    pub tasks: Vec<String>,
    pub row_bands: Vec<String>,
    pub feature_bands: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReleaseGates {
    pub ptf_v1_min: f64,
    pub calibration_degradation_max: f64,
    pub rare_tail_subgroup_retention_min: f64,
    pub driver_agreement_min: f64,
    pub joint_fidelity_min: f64,
    pub query_p95_normalized_error_max: f64,
    pub nominal_95_coverage_min: f64,
    pub nominal_95_coverage_max: f64,
    pub type_i_error_max: f64,
    pub membership_auc_max: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_importance_spearman_min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_importance_top_k_jaccard_min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_importance_min_informative_features: Option<usize>,
    pub attribute_inference_advantage_max: f64,
    pub validation_training_regret_upper_max: f64,
    pub across_seed_validation_loss_stddev_max: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KpiContract {
    pub format: String,
    pub version: u8,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tier_byte_limits: BTreeMap<String, usize>,
    pub primary_kpi: String,
    pub release_seed: u64,
    pub generation_repeats: usize,
    pub auditor_seeds: Vec<u64>,
    pub required_size_multipliers: Vec<usize>,
    pub scaling_size_multipliers: Vec<usize>,
    pub confidence: ConfidenceContract,
    pub low_signal: LowSignalContract,
    pub profile_gating: ProfileGatingContract,
    pub profiles: ProfileBands,
    pub release_gates: ReleaseGates,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub master_fitness: Option<MasterFitnessContract>,
}

impl KpiContract {
    pub fn embedded() -> Result<Self> {
        Ok(serde_json::from_slice(include_bytes!(
            "../production/kpi-contract.json"
        ))?)
    }

    pub fn embedded_v1() -> Result<Self> {
        Ok(serde_json::from_slice(include_bytes!(
            "../production/kpi-contract-v1.json"
        ))?)
    }

    pub fn validate(&self) -> Result<()> {
        let valid = self.format == "dope-kpi-contract"
            && (self.version == 1 || self.version == 2)
            && self.primary_kpi == "PTF-v1"
            && self.release_seed == RELEASE_SEED
            && self.generation_repeats == GENERATION_REPEATS
            && self.auditor_seeds == AUDITOR_SEEDS
            && self.required_size_multipliers == GOLD_SIZE_MULTIPLIERS
            && self.scaling_size_multipliers == SYNTHETIC_SIZE_MULTIPLIERS
            && self.confidence.method == "paired_student_t"
            && self.confidence.cluster == "lineage_group"
            && self.confidence.equal_weight == "lineage_group"
            && (self.confidence.one_sided_level - 0.95).abs() <= f64::EPSILON
            && self.profile_gating.minimum_groups == 100
            && self.profile_gating.minimum_informative_groups == 30
            && self.release_gates.ptf_v1_min == 0.99
            && match self.version {
                1 => {
                    self.tier_byte_limits.is_empty()
                        && self.release_gates.feature_importance_spearman_min.is_none()
                        && self
                            .release_gates
                            .feature_importance_top_k_jaccard_min
                            .is_none()
                        && self
                            .release_gates
                            .feature_importance_min_informative_features
                            .is_none()
                        && self.master_fitness.is_none()
                }
                2 => {
                    self.tier_byte_limits.get("l3") == Some(&10_240)
                        && self.tier_byte_limits.get("l2") == Some(&32_768)
                        && self.tier_byte_limits.len() == 2
                        && self.release_gates.membership_auc_max == 0.55
                        && self.release_gates.feature_importance_spearman_min == Some(0.70)
                        && self.release_gates.feature_importance_top_k_jaccard_min == Some(0.50)
                        && self
                            .release_gates
                            .feature_importance_min_informative_features
                            == Some(3)
                        && self
                            .master_fitness
                            .as_ref()
                            .is_some_and(MasterFitnessContract::is_frozen_v2)
                }
                _ => false,
            };
        if !valid {
            return Err(DopeError::Data(
                "KPI contract differs from the frozen PTF-v1 definition".into(),
            ));
        }
        let digest = hashes(&canonical_json(self)?);
        let (sha256, blake3) = if self.version == 1 {
            (KPI_CONTRACT_V1_SHA256, KPI_CONTRACT_V1_BLAKE3)
        } else {
            (KPI_CONTRACT_SHA256, KPI_CONTRACT_BLAKE3)
        };
        if digest.sha256 != sha256 || digest.blake3 != blake3 {
            return Err(DopeError::Data(
                "KPI contract does not match the binary-embedded identity".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CorpusContract {
    pub manifest_sha256: String,
    pub inventory_sha256: String,
    pub train_lineage_groups: usize,
    pub validation_lineage_groups: usize,
    pub sealed_test_lineage_groups: usize,
    pub mirrors_weight: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ValidationSplitContract {
    pub hash: String,
    pub seed: u64,
    pub selection_numerator: usize,
    pub selection_denominator: usize,
    pub stratified_by: Vec<String>,
    pub sealed_rows_opened: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RunContract {
    pub format: String,
    pub version: u8,
    pub release_identity: String,
    pub source_commit: String,
    pub environment_lock: ContentHashes,
    pub sbom_sha256: String,
    pub package_hashes_sha256: String,
    pub cuda_inventory_sha256: String,
    pub checkpoint_manifest_sha256: String,
    pub notices_sha256: String,
    pub corpus: CorpusContract,
    pub validation_split: ValidationSplitContract,
    pub kpi_contract: ContentHashes,
    pub empirical_candidates: Vec<String>,
    pub formal_dp_backends: Vec<String>,
    pub auditors: Vec<String>,
    pub synthetic_sizes: Vec<usize>,
    pub generation_seeds_per_lineage: usize,
    pub auditor_seeds: Vec<u64>,
    pub artifact_ceiling_bytes: u64,
    pub controller: String,
    pub primary_worker: String,
    pub guarded_evaluator: String,
    pub lease_seconds: u64,
    pub heartbeat_seconds: u64,
    pub infrastructure_retries: u8,
    pub created_unix_seconds: u64,
}

impl RunContract {
    pub fn validate(&self) -> Result<()> {
        let required_candidates: BTreeSet<_> = empirical_backends()
            .into_iter()
            .map(|backend| backend.id)
            .collect();
        let required_auditors: BTreeSet<_> = auditor_specs()
            .into_iter()
            .map(|auditor| auditor.id)
            .collect();
        let candidates: BTreeSet<_> = self.empirical_candidates.iter().cloned().collect();
        let auditors: BTreeSet<_> = self.auditors.iter().cloned().collect();
        let required_dp: BTreeSet<_> = [
            "dp_graphical_mst",
            "dp_kd_tree",
            "dp_tbart",
            "dp_tvae",
            "dp_tabsyn",
            "dp_conditional_transformer",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        let dp: BTreeSet<_> = self.formal_dp_backends.iter().cloned().collect();
        let digest_fields_valid = [
            &self.sbom_sha256,
            &self.package_hashes_sha256,
            &self.cuda_inventory_sha256,
            &self.checkpoint_manifest_sha256,
            &self.notices_sha256,
            &self.validation_split.hash,
        ]
        .into_iter()
        .all(|value| is_lower_hex(value, 64));
        if self.format != "dope-run-contract"
            || self.version != 1
            || self.release_identity != RELEASE_CANDIDATE_VERSION
            || !digest_fields_valid
            || !(is_lower_hex(&self.source_commit, 40) || is_lower_hex(&self.source_commit, 64))
            || self.environment_lock.sha256.len() != 64
            || self.environment_lock.blake3.len() != 64
            || self.corpus.manifest_sha256 != CORPUS_MANIFEST_SHA256
            || self.corpus.inventory_sha256 != CORPUS_INVENTORY_SHA256
            || self.corpus.train_lineage_groups != 90_729
            || self.corpus.validation_lineage_groups != 19_712
            || self.corpus.sealed_test_lineage_groups != 19_684
            || self.corpus.mirrors_weight != 0
            || self.validation_split.seed != RELEASE_SEED
            || self.validation_split.selection_numerator != 3
            || self.validation_split.selection_denominator != 5
            || self.validation_split.sealed_rows_opened
            || self.kpi_contract.sha256 != KPI_CONTRACT_SHA256
            || self.kpi_contract.blake3 != KPI_CONTRACT_BLAKE3
            || candidates != required_candidates
            || auditors != required_auditors
            || dp != required_dp
            || self.synthetic_sizes != SYNTHETIC_SIZE_MULTIPLIERS
            || self.generation_seeds_per_lineage != GENERATION_REPEATS
            || self.auditor_seeds != AUDITOR_SEEDS
            || self.artifact_ceiling_bytes != EMPIRICAL_ARTIFACT_CEILING_BYTES
            || self.controller != "xbabe1"
            || self.primary_worker != "xbabe1"
            || self.guarded_evaluator != "xbabe3"
            || self.lease_seconds != 600
            || self.heartbeat_seconds != 30
            || self.infrastructure_retries != 2
        {
            return Err(DopeError::Data(
                "run contract differs from the frozen production campaign".into(),
            ));
        }
        Ok(())
    }

    pub fn id(&self) -> Result<ContentHashes> {
        self.validate()?;
        Ok(hashes(&canonical_json(self)?))
    }
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Ord, PartialOrd)]
pub struct StructuralProfile {
    pub task: String,
    pub row_band: String,
    pub feature_band: String,
}

impl StructuralProfile {
    pub fn from_shape(task: Task, rows: usize, features: usize) -> Result<Self> {
        if !(1..=2_000).contains(&features) {
            return Err(DopeError::Data(
                "production profiles support 1 to 2,000 features".into(),
            ));
        }
        let row_band = match rows {
            0..=31 => "<32",
            32..=255 => "32-255",
            256..=1023 => "256-1023",
            _ => ">=1024",
        };
        let feature_band = match features {
            1..=16 => "1-16",
            17..=64 => "17-64",
            65..=256 => "65-256",
            _ => "257-2000",
        };
        Ok(Self {
            task: task.as_str().into(),
            row_band: row_band.into(),
            feature_band: feature_band.into(),
        })
    }

    pub fn id(&self) -> String {
        format!("{}/{}/{}", self.task, self.row_band, self.feature_band)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KpiCell {
    pub task: Task,
    pub train_rows: usize,
    pub features: usize,
    pub auditor: String,
    pub size_multiplier: usize,
    pub lineage_group_id: String,
    pub generation_seed: u64,
    pub auditor_seed: u64,
    pub null_loss: f64,
    pub trtr_loss: f64,
    pub tstr_loss: f64,
    pub calibration_degradation: Option<f64>,
    pub rare_class_or_tail_retention: Option<f64>,
    pub supported_subgroup_retention: Option<f64>,
    pub nominal_95_coverage: Option<f64>,
}

impl KpiCell {
    pub fn validate(&self, contract: &KpiContract) -> Result<()> {
        if self.train_rows == 0
            || !(1..=2_000).contains(&self.features)
            || !contract
                .scaling_size_multipliers
                .contains(&self.size_multiplier)
            || !contract.auditor_seeds.contains(&self.auditor_seed)
            || ![self.null_loss, self.trtr_loss, self.tstr_loss]
                .into_iter()
                .all(|value| value.is_finite() && value >= 0.0)
            || self.lineage_group_id.is_empty()
            || !contract.auditor_seeds.contains(&self.auditor_seed)
        {
            return Err(DopeError::Data("invalid KPI cell".into()));
        }
        if !auditor_specs()
            .iter()
            .any(|auditor| auditor.id == self.auditor)
        {
            return Err(DopeError::Data("KPI cell names an unfrozen auditor".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KpiAggregate {
    pub task: String,
    pub structural_profile: String,
    pub auditor: String,
    pub size_multiplier: usize,
    pub lineage_groups: usize,
    pub informative_groups: usize,
    pub low_signal_groups: usize,
    pub mean_unclamped_retention: Option<f64>,
    pub one_sided_95_lower_bound: Option<f64>,
    pub low_signal_compliance: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KpiReport {
    pub format: String,
    pub version: u8,
    pub primary_kpi: String,
    pub ptf_v1: Option<f64>,
    pub cells: Vec<KpiAggregate>,
    pub independently_gated_profiles: Vec<String>,
    pub folded_profiles: BTreeMap<String, String>,
    pub low_signal_compliance_overall: Option<f64>,
    pub low_signal_compliance_by_profile: BTreeMap<String, f64>,
    pub eligible_population: usize,
    pub evaluated_numerator: usize,
    pub missing_or_timeout_denominator: usize,
    pub lineage_weighting: String,
    pub coverage_tier: String,
    pub confidence_method: String,
    pub kpi_contract: ContentHashes,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct AuditorAvailabilitySummary {
    pub required: usize,
    pub available: usize,
    pub unavailable: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KpiSummary {
    pub format: String,
    pub version: u8,
    pub ptf_v1: Option<f64>,
    pub low_signal_compliance: Option<f64>,
    pub coverage: Option<f64>,
    pub missing_cells: usize,
    pub candidate_regret: Option<f64>,
    pub auditor_availability: AuditorAvailabilitySummary,
    pub production_score_available: bool,
}

impl KpiSummary {
    pub fn from_evidence(
        report: Option<&KpiReport>,
        coverage_report: Option<&CoverageReport>,
        candidate_regret: Option<f64>,
    ) -> Self {
        let auditors = auditor_specs();
        let unavailable = auditors
            .iter()
            .filter(|auditor| !auditor.available)
            .map(|auditor| auditor.id.clone())
            .collect::<Vec<_>>();
        let (coverage, missing_cells) = if let Some(coverage_report) = coverage_report {
            let eligible = coverage_report
                .entries
                .iter()
                .map(|entry| entry.eligible)
                .sum::<usize>();
            let evaluated = coverage_report
                .entries
                .iter()
                .map(|entry| entry.evaluated)
                .sum::<usize>();
            let missing = coverage_report
                .entries
                .iter()
                .map(|entry| entry.missing + entry.timed_out)
                .sum();
            (
                (eligible > 0).then_some(evaluated as f64 / eligible as f64),
                missing,
            )
        } else {
            (
                report.and_then(|report| {
                    (report.eligible_population > 0).then_some(
                        report.evaluated_numerator as f64 / report.eligible_population as f64,
                    )
                }),
                report.map_or(0, |report| report.missing_or_timeout_denominator),
            )
        };
        let ptf_v1 = report.and_then(|report| report.ptf_v1);
        let required_complete = coverage_report
            .is_some_and(|coverage| coverage.validate().is_ok() && coverage.required_complete);
        let production_score_available = ptf_v1.is_some()
            && required_complete
            && unavailable.is_empty()
            && empirical_backends().iter().all(|backend| backend.available);
        Self {
            format: "dope-kpi-summary".into(),
            version: 1,
            ptf_v1,
            low_signal_compliance: report.and_then(|report| report.low_signal_compliance_overall),
            coverage,
            missing_cells,
            candidate_regret,
            auditor_availability: AuditorAvailabilitySummary {
                required: auditors.len(),
                available: auditors.len() - unavailable.len(),
                unavailable,
            },
            production_score_available,
        }
    }
}

type AggregateKey = (String, String, String, usize);

#[derive(Clone, Copy)]
struct LossTriple {
    null: f64,
    trtr: f64,
    tstr: f64,
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<_> = values.collect();
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn one_sided_lower(values: &[f64], level: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let average = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() == 1 {
        return Some(average);
    }
    let variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let distribution = StudentsT::new(0.0, 1.0, (values.len() - 1) as f64).ok()?;
    Some(average - distribution.inverse_cdf(level) * (variance / values.len() as f64).sqrt())
}

pub fn aggregate_kpis(
    cells: &[KpiCell],
    eligible_population: usize,
    missing_or_timeout_denominator: usize,
    coverage_tier: &str,
) -> Result<KpiReport> {
    let contract = KpiContract::embedded()?;
    contract.validate()?;
    for cell in cells {
        cell.validate(&contract)?;
    }

    let mut profile_lineages = BTreeMap::<String, BTreeMap<String, Vec<LossTriple>>>::new();
    for cell in cells {
        let profile =
            StructuralProfile::from_shape(cell.task, cell.train_rows, cell.features)?.id();
        profile_lineages
            .entry(profile)
            .or_default()
            .entry(cell.lineage_group_id.clone())
            .or_default()
            .push(LossTriple {
                null: cell.null_loss,
                trtr: cell.trtr_loss,
                tstr: cell.tstr_loss,
            });
    }
    let mut independently_gated = Vec::new();
    let mut folded_profiles = BTreeMap::new();
    for (profile, lineages) in &profile_lineages {
        let informative = lineages
            .values()
            .filter(|values| {
                let null = mean(values.iter().map(|value| value.null));
                let trtr = mean(values.iter().map(|value| value.trtr));
                null - trtr >= contract.low_signal.improvement_threshold_fraction * null.abs()
            })
            .count();
        if lineages.len() >= contract.profile_gating.minimum_groups
            && informative >= contract.profile_gating.minimum_informative_groups
        {
            independently_gated.push(profile.clone());
            folded_profiles.insert(profile.clone(), profile.clone());
        } else {
            let task = profile.split('/').next().unwrap_or("unknown");
            folded_profiles.insert(profile.clone(), format!("{task}/other"));
        }
    }
    independently_gated.sort();

    let mut grouped = BTreeMap::<AggregateKey, BTreeMap<String, Vec<LossTriple>>>::new();
    for cell in cells {
        let raw_profile =
            StructuralProfile::from_shape(cell.task, cell.train_rows, cell.features)?.id();
        let profile = folded_profiles[&raw_profile].clone();
        grouped
            .entry((
                cell.task.as_str().into(),
                profile,
                cell.auditor.clone(),
                cell.size_multiplier,
            ))
            .or_default()
            .entry(cell.lineage_group_id.clone())
            .or_default()
            .push(LossTriple {
                null: cell.null_loss,
                trtr: cell.trtr_loss,
                tstr: cell.tstr_loss,
            });
    }

    let mut aggregates = Vec::new();
    let mut all_low_signal = Vec::<(String, bool)>::new();
    for ((task, profile, auditor, size_multiplier), lineages) in grouped {
        let mut retentions = Vec::new();
        let mut low_signal = Vec::new();
        for values in lineages.values() {
            let null = mean(values.iter().map(|value| value.null));
            let trtr = mean(values.iter().map(|value| value.trtr));
            let tstr = mean(values.iter().map(|value| value.tstr));
            if null - trtr >= contract.low_signal.improvement_threshold_fraction * null.abs() {
                retentions.push((null - tstr) / (null - trtr));
            } else {
                let passed =
                    tstr <= trtr + contract.low_signal.noninferiority_fraction_of_null * null;
                low_signal.push(passed);
                all_low_signal.push((profile.clone(), passed));
            }
        }
        aggregates.push(KpiAggregate {
            task,
            structural_profile: profile,
            auditor,
            size_multiplier,
            lineage_groups: lineages.len(),
            informative_groups: retentions.len(),
            low_signal_groups: low_signal.len(),
            mean_unclamped_retention: (!retentions.is_empty())
                .then(|| retentions.iter().sum::<f64>() / retentions.len() as f64),
            one_sided_95_lower_bound: one_sided_lower(
                &retentions,
                contract.confidence.one_sided_level,
            ),
            low_signal_compliance: (!low_signal.is_empty()).then(|| {
                low_signal.iter().filter(|passed| **passed).count() as f64 / low_signal.len() as f64
            }),
        });
    }
    let required: BTreeSet<_> = contract.required_size_multipliers.iter().copied().collect();
    let ptf_v1 = aggregates
        .iter()
        .filter(|cell| required.contains(&cell.size_multiplier))
        .filter_map(|cell| cell.one_sided_95_lower_bound)
        .reduce(f64::min);
    let low_signal_compliance_overall = (!all_low_signal.is_empty()).then(|| {
        all_low_signal.iter().filter(|(_, passed)| *passed).count() as f64
            / all_low_signal.len() as f64
    });
    let mut by_profile_raw = BTreeMap::<String, (usize, usize)>::new();
    for (profile, passed) in all_low_signal {
        let entry = by_profile_raw.entry(profile).or_default();
        entry.0 += passed as usize;
        entry.1 += 1;
    }
    let low_signal_compliance_by_profile = by_profile_raw
        .into_iter()
        .map(|(profile, (passed, total))| (profile, passed as f64 / total as f64))
        .collect();
    let evaluated_numerator = cells
        .iter()
        .map(|cell| &cell.lineage_group_id)
        .collect::<BTreeSet<_>>()
        .len();
    Ok(KpiReport {
        format: "dope-kpi-aggregate".into(),
        version: 1,
        primary_kpi: "PTF-v1".into(),
        ptf_v1,
        cells: aggregates,
        independently_gated_profiles: independently_gated,
        folded_profiles,
        low_signal_compliance_overall,
        low_signal_compliance_by_profile,
        eligible_population,
        evaluated_numerator,
        missing_or_timeout_denominator,
        lineage_weighting: "equal_lineage_group".into(),
        coverage_tier: coverage_tier.into(),
        confidence_method: "paired_lineage_clustered_one_sided_student_t_95".into(),
        kpi_contract: ContentHashes {
            sha256: KPI_CONTRACT_SHA256.into(),
            blake3: KPI_CONTRACT_BLAKE3.into(),
        },
    })
}

include!("production/coverage_and_gates.rs");

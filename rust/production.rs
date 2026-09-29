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
use crate::model::Task;

/// Public identity of ordinary development builds. An RC identity is emitted
/// only in a successfully validated release-candidate manifest.
pub const DEVELOPMENT_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const RELEASE_CANDIDATE_VERSION: &str = "1.0.0-rc.1";
pub const KPI_CONTRACT_SHA256: &str = env!("DOPE_KPI_CONTRACT_SHA256");
pub const KPI_CONTRACT_BLAKE3: &str = env!("DOPE_KPI_CONTRACT_BLAKE3");
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
    let temporary = parent.join(format!(
        ".{filename}.tmp-{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let write_result = (|| -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        File::open(parent)?.sync_all()
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
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
    pub attribute_inference_advantage_max: f64,
    pub validation_training_regret_upper_max: f64,
    pub across_seed_validation_loss_stddev_max: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KpiContract {
    pub format: String,
    pub version: u8,
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
}

impl KpiContract {
    pub fn embedded() -> Result<Self> {
        Ok(serde_json::from_slice(include_bytes!(
            "../production/kpi-contract.json"
        ))?)
    }

    pub fn validate(&self) -> Result<()> {
        let valid = self.format == "dope-kpi-contract"
            && self.version == 1
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
            && self.release_gates.ptf_v1_min == 0.99;
        if !valid {
            return Err(DopeError::Data(
                "KPI contract differs from the frozen PTF-v1 definition".into(),
            ));
        }
        let digest = hashes(&canonical_json(self)?);
        if digest.sha256 != KPI_CONTRACT_SHA256 || digest.blake3 != KPI_CONTRACT_BLAKE3 {
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
    let mut files = Vec::with_capacity(names.len());
    for name in names {
        let path = root.join(name);
        let metadata = fs::metadata(&path).map_err(|error| io_error(&path, error))?;
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

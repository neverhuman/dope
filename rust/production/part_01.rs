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
        Ok(serde_json::from_slice(include_bytes!("../../production/kpi-contract.json"))?)
    }

    pub fn embedded_v1() -> Result<Self> {
        Ok(serde_json::from_slice(include_bytes!("../../production/kpi-contract-v1.json"))?)
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
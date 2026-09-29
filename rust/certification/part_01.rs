use std::collections::{BTreeMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::auditor::{
    FeatureImportanceConsistency, PermutationImportance, auditor_backend,
    compare_permutation_importance, implementation_hash,
};
use crate::codec::{LoadedKernel, decode_kernel, load_kernel};
use crate::compiler::CompileOptions;
use crate::contract::{
    AUDITOR_SEEDS, AnonymizationTier, GENERATION_REPEATS, GOLD_SIZE_MULTIPLIERS, RELEASE_SEED,
    ReleasePolicy, auditor_specs, candidate_implementation_hash, empirical_backend,
};
use crate::data::Table;
use crate::error::{DopeError, Result, io_error};
use crate::fitness::{MasterFitnessReport, ParetoVector};
use crate::fitness_metrics::{FitnessDiagnostics, evaluate as evaluate_fitness_diagnostics};
use crate::ledger::{CacheStatus, JOB_EVIDENCE_VERSION, JobEvidence};
use crate::model::{Kernel, Marginal, SchemaKind, Task};
use crate::production::{ContentHashes, KpiContract, canonical_json, hashes};
use crate::sample::{SampleOptions, sample_kernel};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricVector {
    pub synthetic_rows: usize,
    pub generation_seed: u64,
    pub auditor_seed: u64,
    pub null_loss: f64,
    pub trtr_loss: f64,
    pub tstr_loss: f64,
    pub informative: bool,
    pub retention: Option<f64>,
    pub absolute_noninferiority_passed: bool,
    pub calibration_degradation: f64,
    pub rare_class_or_tail_retention: Option<f64>,
    pub supported_subgroup_retention: Option<f64>,
    pub nominal_95_coverage: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RetentionDecision {
    pub informative: bool,
    pub retention: Option<f64>,
    pub absolute_noninferiority_passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditorReport {
    pub available: bool,
    pub backend: String,
    pub version: String,
    pub hash: String,
    pub telemetry_disabled: bool,
    pub null_loss: Option<f64>,
    pub real_trained_loss: Option<f64>,
    pub synthetic_trained_loss: Option<f64>,
    pub retention: Option<f64>,
    pub metrics: Vec<MetricVector>,
    pub one_sided_95_lower_by_multiplier: BTreeMap<String, Option<f64>>,
    pub gold_gate_passed: bool,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FidelityReport {
    pub worst_driver_agreement: f64,
    pub worst_joint_fidelity: f64,
    pub worst_query_p95_normalized_error: f64,
    pub maximum_type_i_error: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrivacyReport {
    pub attack_suite_version: u8,
    pub membership_attacks: BTreeMap<String, f64>,
    pub maximum_membership_auc: f64,
    pub rare_slice_supported: bool,
    pub rare_slice_membership_auc: Option<f64>,
    pub rare_slice_attribute_advantage: Option<f64>,
    pub maximum_attribute_inference_advantage: f64,
    pub exact_copy_count: usize,
    pub near_copy_count: usize,
    pub near_copy_definition: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IsolationEvidence {
    pub generator_opened: Vec<String>,
    pub evaluator_opened_real_test: bool,
    pub downstream_tuning: String,
    pub synthetic_sizes: Vec<usize>,
    pub generation_repeats: usize,
    pub auditor_seeds: Vec<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CertificationReport {
    pub format: String,
    pub version: u8,
    pub release_policy: ReleasePolicy,
    pub release_policy_hash: String,
    pub certified: bool,
    pub gates: BTreeMap<String, bool>,
    pub failed_gates: Vec<String>,
    pub auditors: BTreeMap<String, AuditorReport>,
    pub feature_importance: Vec<ImportanceEvidence>,
    pub minimum_gold_one_sided_95_retention: Option<f64>,
    pub fidelity: FidelityReport,
    pub privacy: PrivacyReport,
    #[serde(default)]
    pub fitness_diagnostics: Vec<FitnessDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub master_fitness: Option<MasterFitnessReport>,
    pub isolation: IsolationEvidence,
    pub geometric_mean_retention: f64,
    pub driver_agreement: f64,
    pub joint_fidelity: f64,
    pub membership_inference_auc: f64,
    pub synthetic_duplicate_rate: f64,
    pub primary_synthetic_rows: usize,
    pub secondary_synthetic_rows: usize,
    pub artifact_bytes: usize,
    pub effective_bytes: usize,
    pub bits_per_original_cell: f64,
    pub compression_vs_packed_f32: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportanceEvidence {
    pub auditor_id: String,
    pub auditor_seed: u64,
    pub synthetic_rows: usize,
    pub generation_seed: u64,
    pub consistency: FeatureImportanceConsistency,
}

pub fn null_normalized_excess_loss_retention(
    null_loss: f64,
    real_loss: f64,
    synthetic_loss: f64,
) -> f64 {
    let real_excess = null_loss - real_loss;
    if real_excess.abs() <= 1e-12 {
        return f64::NAN;
    }
    (null_loss - synthetic_loss) / real_excess
}

pub fn retention_decision(
    null_loss: f64,
    real_loss: f64,
    synthetic_loss: f64,
) -> RetentionDecision {
    let informative = null_loss - real_loss >= 0.01 * null_loss.abs();
    RetentionDecision {
        informative,
        retention: informative
            .then(|| null_normalized_excess_loss_retention(null_loss, real_loss, synthetic_loss)),
        absolute_noninferiority_passed: synthetic_loss <= real_loss + 0.01 * null_loss,
    }
}

fn row_major(table: &Table) -> Vec<f32> {
    let mut output = vec![0.0; table.rows * table.features];
    for row in 0..table.rows {
        for column in 0..table.features {
            output[row * table.features + column] = table.columns[column][row];
        }
    }
    output
}

fn table_from_sample(sample: &[f32], rows: usize, features: usize) -> Table {
    let stride = features + 1;
    let columns = (0..features)
        .map(|column| (0..rows).map(|row| sample[row * stride + column]).collect())
        .collect();
    let target = (0..rows)
        .map(|row| sample[row * stride + features])
        .collect();
    Table {
        rows,
        features,
        columns,
        target,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GoldCellOptions<'a> {
    pub dataset_dir: &'a Path,
    pub task: Task,
    pub candidate_id: &'a str,
    pub auditor_id: &'a str,
    pub lineage_group_id: &'a str,
    pub structural_profile: &'a str,
    pub phase: &'a str,
    pub size_multiplier: usize,
    pub generation_seed: u64,
    pub auditor_seed: u64,
    pub routed: bool,
    pub ancillary: bool,
    pub cache_dir: Option<&'a Path>,
    pub neural_target_weight: f64,
    pub neural_structural_penalty: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct CachedAuditorPrediction {
    format: String,
    version: u8,
    lineage_group_id: String,
    auditor_id: String,
    auditor_seed: u64,
    task: String,
    rows: usize,
    features: usize,
    importance: PermutationImportance,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct CachedSyntheticAncillary {
    format: String,
    version: u8,
    lineage_group_id: String,
    candidate_id: String,
    size_multiplier: usize,
    generation_seed: u64,
    artifact_sha256: String,
    driver_agreement: f64,
    joint_fidelity: f64,
    query_p95_normalized_error: f64,
    type_i_error: f64,
    membership_auc: f64,
    attribute_inference_advantage: f64,
    exact_copies: usize,
    near_copies: usize,
}

fn cache_path(cache: &Path, kind: &str, identity: &[u8], extension: &str) -> PathBuf {
    let digest = hashes(identity).blake3;
    cache
        .join(kind)
        .join(&digest[..2])
        .join(format!("{digest}.{extension}"))
}

fn atomic_cache_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.is_file() {
        return Ok(());
    }
    let parent = path
        .parent()
        .ok_or_else(|| DopeError::Data("cache path has no parent".into()))?;
    fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
    let staged_path = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staged_path)
    {
        Ok(file) => file,
        Err(_error) if staged_path.is_file() || path.is_file() => return Ok(()),
        Err(error) => return Err(io_error(&staged_path, error)),
    };
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| io_error(&staged_path, error))?;
    match fs::rename(&staged_path, path) {
        Ok(()) => Ok(()),
        Err(_) if path.is_file() => {
            fs::remove_file(&staged_path).map_err(|error| io_error(&staged_path, error))?;
            Ok(())
        }
        Err(error) => Err(io_error(path, error)),
    }
}

fn cached_artifact(
    options: &GoldCellOptions<'_>,
    train: &Table,
) -> Result<(Vec<u8>, CacheStatus, u64)> {
    let identity = canonical_json(&serde_json::json!({
        "format": "dope-gold-artifact-cache-key",
        "version": 2,
        "lineage_group_id": options.lineage_group_id,
        "candidate_id": options.candidate_id,
        "candidate_implementation": candidate_implementation_hash(options.candidate_id),
        "task": options.task.as_str(),
        "rows": train.rows,
        "features": train.features,
        "release_seed": RELEASE_SEED,
        "neural_target_weight": options.neural_target_weight,
        "neural_structural_penalty": options.neural_structural_penalty
    }))?;
    let path = options
        .cache_dir
        .map(|cache| cache_path(cache, "artifacts", &identity, "dope"));
    if let Some(path) = path.as_ref().filter(|path| path.is_file()) {
        let artifact = fs::read(path).map_err(|error| io_error(path, error))?;
        decode_kernel(&artifact)?;
        return Ok((artifact, CacheStatus::Hit, 0));
    }
    let fitting_started = Instant::now();
    let artifact = empirical_backend(options.candidate_id)?
        .fit(
            &row_major(train),
            &train.target,
            train.rows,
            train.features,
            options.task,
            &CompileOptions {
                seed: Some(RELEASE_SEED),
                neural_target_weight: options.neural_target_weight,
                neural_structural_penalty: options.neural_structural_penalty,
                ..Default::default()
            },
        )?
        .artifact;
    let fitting_time_ms = fitting_started
        .elapsed()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;
    let status = if path.is_some() {
        CacheStatus::Miss
    } else {
        CacheStatus::Disabled
    };
    if let Some(path) = path {
        atomic_cache_write(&path, &artifact)?;
    }
    Ok((artifact, status, fitting_time_ms))
}

fn cached_real_importance(
    options: &GoldCellOptions<'_>,
    train: &Table,
    test: &Table,
) -> Result<(PermutationImportance, CacheStatus)> {
    let identity = canonical_json(&serde_json::json!({
        "format": "dope-gold-real-auditor-cache-key",
        "version": 2,
        "lineage_group_id": options.lineage_group_id,
        "auditor_id": options.auditor_id,
        "auditor_implementation": implementation_hash(options.auditor_id),
        "auditor_seed": options.auditor_seed,
        "task": options.task.as_str(),
        "train_rows": train.rows,
        "test_rows": test.rows,
        "features": train.features
    }))?;
    let path = options
        .cache_dir
        .map(|cache| cache_path(cache, "real-auditor-predictions", &identity, "json"));
    if let Some(path) = path.as_ref().filter(|path| path.is_file()) {
        let cached: CachedAuditorPrediction =
            serde_json::from_slice(&fs::read(path).map_err(|error| io_error(path, error))?)?;
        if cached.format != "dope-gold-real-auditor-cache"
            || cached.version != 2
            || cached.lineage_group_id != options.lineage_group_id
            || cached.auditor_id != options.auditor_id
            || cached.auditor_seed != options.auditor_seed
            || cached.task != options.task.as_str()
            || cached.rows != test.rows
            || cached.features != test.features
            || cached.importance.baseline_predictions.len() != test.rows
            || cached
                .importance
                .baseline_predictions
                .iter()
                .any(|value| !value.is_finite())
            || cached.importance.feature_indices.len() != cached.importance.importances.len()
            || cached.importance.feature_indices.len() != train.features
        {
            return Err(DopeError::Data(format!(
                "real auditor cache identity mismatch at {}",
                path.display()
            )));
        }
        return Ok((cached.importance, CacheStatus::Hit));
    }
    let importance = auditor_backend(options.auditor_id)?.permutation_importance(
        train,
        test,
        options.task,
        options.auditor_seed,
    )?;
    let status = if path.is_some() {
        CacheStatus::Miss
    } else {
        CacheStatus::Disabled
    };
    if let Some(path) = path {
        atomic_cache_write(
            &path,
            &canonical_json(&CachedAuditorPrediction {
                format: "dope-gold-real-auditor-cache".into(),
                version: 2,
                lineage_group_id: options.lineage_group_id.into(),
                auditor_id: options.auditor_id.into(),
                auditor_seed: options.auditor_seed,
                task: options.task.as_str().into(),
                rows: test.rows,
                features: test.features,
                importance: importance.clone(),
            })?,
        )?;
    }
    Ok((importance, status))
}
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

fn cached_synthetic_ancillary(
    options: &GoldCellOptions<'_>,
    artifact_sha256: &str,
    train: &Table,
    test: &Table,
    synthetic: &Table,
) -> Result<(CachedSyntheticAncillary, CacheStatus)> {
    let identity = canonical_json(&serde_json::json!({
        "format": "dope-gold-synthetic-ancillary-cache-key",
        "version": 1,
        "lineage_group_id": options.lineage_group_id,
        "candidate_id": options.candidate_id,
        "size_multiplier": options.size_multiplier,
        "generation_seed": options.generation_seed,
        "artifact_sha256": artifact_sha256
    }))?;
    let path = options
        .cache_dir
        .map(|cache| cache_path(cache, "synthetic-ancillary", &identity, "json"));
    if let Some(path) = path.as_ref().filter(|path| path.is_file()) {
        let cached: CachedSyntheticAncillary =
            serde_json::from_slice(&fs::read(path).map_err(|error| io_error(path, error))?)?;
        if cached.format != "dope-gold-synthetic-ancillary-cache"
            || cached.version != 1
            || cached.lineage_group_id != options.lineage_group_id
            || cached.candidate_id != options.candidate_id
            || cached.size_multiplier != options.size_multiplier
            || cached.generation_seed != options.generation_seed
            || cached.artifact_sha256 != artifact_sha256
            || [
                cached.driver_agreement,
                cached.joint_fidelity,
                cached.query_p95_normalized_error,
                cached.type_i_error,
                cached.membership_auc,
                cached.attribute_inference_advantage,
            ]
            .into_iter()
            .any(|value| !value.is_finite())
        {
            return Err(DopeError::Data(format!(
                "synthetic ancillary cache identity mismatch at {}",
                path.display()
            )));
        }
        return Ok((cached, CacheStatus::Hit));
    }
    let cached = CachedSyntheticAncillary {
        format: "dope-gold-synthetic-ancillary-cache".into(),
        version: 1,
        lineage_group_id: options.lineage_group_id.into(),
        candidate_id: options.candidate_id.into(),
        size_multiplier: options.size_multiplier,
        generation_seed: options.generation_seed,
        artifact_sha256: artifact_sha256.into(),
        driver_agreement: driver_agreement(train, synthetic),
        joint_fidelity: joint_fidelity(train, synthetic),
        query_p95_normalized_error: query_p95_normalized_error(train, synthetic),
        type_i_error: type_i_error(train, synthetic),
        membership_auc: membership_auc(synthetic, train, test),
        attribute_inference_advantage: attribute_inference_advantage(synthetic, train, test),
        exact_copies: exact_copy_count(train, synthetic),
        near_copies: near_copy_count(train, synthetic),
    };
    let status = if path.is_some() {
        CacheStatus::Miss
    } else {
        CacheStatus::Disabled
    };
    if let Some(path) = path {
        atomic_cache_write(&path, &canonical_json(&cached)?)?;
    }
    Ok((cached, status))
}

#[cfg(target_os = "linux")]
fn peak_memory_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("VmHWM:")
                    .and_then(|value| value.split_whitespace().next())
                    .and_then(|value| value.parse::<u64>().ok())
            })
        })
        .and_then(|kib| kib.checked_mul(1024))
        .unwrap_or(0)
}

#[cfg(not(target_os = "linux"))]
fn peak_memory_bytes() -> u64 {
    0
}

fn clear_gpu_peak_memory() {
    #[cfg(feature = "gpu-training")]
    crate::libtorch::clear_last_gpu_peak_memory_bytes();
}

fn take_gpu_peak_memory() -> Option<u64> {
    #[cfg(feature = "gpu-training")]
    {
        crate::libtorch::take_last_gpu_peak_memory_bytes()
    }
    #[cfg(not(feature = "gpu-training"))]
    {
        None
    }
}

fn maximum_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

pub struct PreparedGoldCell {
    lineage_group_id: String,
    candidate_id: String,
    task: Task,
    size_multiplier: usize,
    generation_seed: u64,
    train: Table,
    test: Table,
    synthetic: Table,
    artifact_bytes: u64,
    artifact_hashes: ContentHashes,
    lineage_leaks: usize,
    invalid_rows: usize,
    schema_violations: usize,
    nondeterministic_output: bool,
    fitting_time_ms: u64,
    sampling_time_ms: u64,
    artifact_cache_status: CacheStatus,
    peak_cpu_memory_bytes: u64,
    peak_gpu_memory_bytes: Option<u64>,
    preparation_runtime_ms: u64,
}

fn artifact_lineage_leaks(artifact: &[u8], options: &GoldCellOptions<'_>) -> usize {
    let mut probes = vec![
        options.lineage_group_id.as_bytes().to_vec(),
        options.dataset_dir.to_string_lossy().as_bytes().to_vec(),
    ];
    if let Some(name) = options.dataset_dir.file_name() {
        probes.push(name.to_string_lossy().as_bytes().to_vec());
    }
    probes
        .into_iter()
        .filter(|probe| probe.len() >= 4)
        .map(|probe| {
            artifact
                .windows(probe.len())
                .filter(|window| *window == probe)
                .count()
        })
        .sum()
}

fn marginal_contains(marginal: &Marginal, value: f32) -> bool {
    match marginal {
        Marginal::Constant { value: expected } => (value - expected).abs() <= 1e-5,
        Marginal::Bernoulli { .. } => (value - value.round()).abs() <= 1e-6,
        Marginal::Grid { values, .. } => values
            .iter()
            .any(|expected| (value - expected).abs() <= 1e-5),
        Marginal::QuantileSpline { values } => values
            .first()
            .zip(values.last())
            .is_some_and(|(minimum, maximum)| value >= *minimum - 1e-5 && value <= *maximum + 1e-5),
        Marginal::Beta { .. } => (0.0..=1.0).contains(&value),
        Marginal::Gaussian { .. } => value.is_finite(),
        Marginal::Histogram { edges, .. } => edges
            .first()
            .zip(edges.last())
            .is_some_and(|(minimum, maximum)| value >= *minimum - 1e-5 && value <= *maximum + 1e-5),
        Marginal::ZeroInflated { point, base, .. } => {
            (value - point).abs() <= 1e-5 || marginal_contains(base, value)
        }
    }
}

fn sample_diagnostics(sample: &[f32], kernel: &Kernel, rows: usize) -> (usize, usize) {
    let stride = kernel.features as usize + 1;
    let mut invalid_rows = 0usize;
    let mut schema_violations = 0usize;
    for row in sample.chunks_exact(stride).take(rows) {
        let mut invalid = false;
        for (feature, &value) in row.iter().take(kernel.features as usize).enumerate() {
            if value.is_infinite() {
                invalid = true;
                schema_violations += 1;
            } else if value.is_nan() {
                schema_violations += usize::from(kernel.schema[feature].missing_probability == 0.0);
            } else if !marginal_contains(&kernel.marginals[feature], value)
                || (kernel.schema[feature].kind == SchemaKind::Binary
                    && (value - value.round()).abs() > 1e-6)
            {
                schema_violations += 1;
            }
        }
        let target = row[stride - 1];
        if !target.is_finite()
            || !(0.0..=1.0).contains(&target)
            || (kernel.task == Task::Binary && (target - target.round()).abs() > 1e-6)
        {
            invalid = true;
            schema_violations += 1;
        }
        invalid_rows += usize::from(invalid);
    }
    (invalid_rows, schema_violations)
}

/// Fits and generates one immutable candidate × size × generation-seed table.
/// The guarded holdout is opened only after the bounded artifact and synthetic
/// table exist. Auditors can then score this preparation without regenerating it.
pub fn prepare_gold_cell(options: &GoldCellOptions<'_>) -> Result<PreparedGoldCell> {
    if !matches!(
        options.phase,
        "training_gold" | "validation_select" | "validation_cert"
    ) || !matches!(options.size_multiplier, 1 | 4)
    {
        return Err(DopeError::Data("invalid gold cell phase or size".into()));
    }
    let started = Instant::now();
    let starting_cpu_memory = peak_memory_bytes();
    let train = Table::read_dataset_dir(options.dataset_dir, options.task)?;
    clear_gpu_peak_memory();
    let (artifact, artifact_cache_status, fitting_time_ms) = cached_artifact(options, &train)?;
    let peak_gpu_memory_bytes = take_gpu_peak_memory();
    let artifact_bytes = artifact.len() as u64;
    let artifact_hashes = hashes(&artifact);
    let lineage_leaks = artifact_lineage_leaks(&artifact, options);
    let kernel = decode_kernel(&artifact)?;
    let synthetic_rows = train.rows.saturating_mul(options.size_multiplier);
    let generation_started = Instant::now();
    let sample = sample_kernel(
        &LoadedKernel::V3(Box::new(kernel.clone())),
        SampleOptions {
            rows: synthetic_rows,
            seed: Some(options.generation_seed),
        },
    )?;
    if generation_started.elapsed() > Duration::from_secs(60) {
        return Err(DopeError::Data(
            "generation timeout exceeded the frozen 60 second limit".into(),
        ));
    }
    let first_sampling_time = generation_started.elapsed();
    let repeat_started = Instant::now();
    let repeated = sample_kernel(
        &LoadedKernel::V3(Box::new(kernel.clone())),
        SampleOptions {
            rows: synthetic_rows,
            seed: Some(options.generation_seed),
        },
    )?;
    if repeat_started.elapsed() > Duration::from_secs(60) {
        return Err(DopeError::Data(
            "repeated generation timeout exceeded the frozen 60 second limit".into(),
        ));
    }
    let sampling_time_ms = first_sampling_time
        .saturating_add(repeat_started.elapsed())
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;
    let nondeterministic_output = sample
        .iter()
        .map(|value| value.to_bits())
        .ne(repeated.iter().map(|value| value.to_bits()));
    let (invalid_rows, schema_violations) = sample_diagnostics(&sample, &kernel, synthetic_rows);
    let synthetic = table_from_sample(&sample, synthetic_rows, train.features);

    // The guarded evaluator-owned holdout is opened only after fitting and
    // generation, preserving the no-test-data generator boundary.
    let test_path = options.dataset_dir.join("test.csv");
    if !test_path.is_file() {
        return Err(DopeError::Data(
            "gold evaluation requires evaluator-owned test.csv".into(),
        ));
    }
    let test = Table::read_csv(&test_path, options.task)?;
    if test.features != train.features {
        return Err(DopeError::Data(
            "gold train and test feature widths differ".into(),
        ));
    }
    Ok(PreparedGoldCell {
        lineage_group_id: options.lineage_group_id.into(),
        candidate_id: options.candidate_id.into(),
        task: options.task,
        size_multiplier: options.size_multiplier,
        generation_seed: options.generation_seed,
        train,
        test,
        synthetic,
        artifact_bytes,
        artifact_hashes,
        lineage_leaks,
        invalid_rows,
        schema_violations,
        nondeterministic_output,
        fitting_time_ms,
        sampling_time_ms,
        artifact_cache_status,
        peak_cpu_memory_bytes: starting_cpu_memory.max(peak_memory_bytes()),
        peak_gpu_memory_bytes,
        preparation_runtime_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
    })
}

include!("certification/gold_cells.rs");
include!("certification/privacy_attacks.rs");

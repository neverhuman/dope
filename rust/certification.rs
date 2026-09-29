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
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
    {
        Ok(file) => file,
        Err(_error) if temporary.is_file() || path.is_file() => return Ok(()),
        Err(error) => return Err(io_error(&temporary, error)),
    };
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| io_error(&temporary, error))?;
    match fs::rename(&temporary, path) {
        Ok(()) => Ok(()),
        Err(_) if path.is_file() => {
            fs::remove_file(&temporary).map_err(|error| io_error(&temporary, error))?;
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
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage initializes the supplied rusage object on success.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status == 0 {
        // Linux reports ru_maxrss in KiB.
        unsafe { usage.assume_init().ru_maxrss.max(0) as u64 * 1024 }
    } else {
        0
    }
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

/// Scores one auditor × auditor-seed cell from a frozen generated table.
pub fn evaluate_prepared_gold_cell(
    options: &GoldCellOptions<'_>,
    prepared: &PreparedGoldCell,
) -> Result<JobEvidence> {
    if prepared.lineage_group_id != options.lineage_group_id
        || prepared.candidate_id != options.candidate_id
        || prepared.task != options.task
        || prepared.size_multiplier != options.size_multiplier
        || prepared.generation_seed != options.generation_seed
    {
        return Err(DopeError::Data(
            "prepared gold table does not match requested cell".into(),
        ));
    }
    let started = Instant::now();
    let train = &prepared.train;
    let test = &prepared.test;
    let synthetic = &prepared.synthetic;
    let artifact_hashes = &prepared.artifact_hashes;
    let auditor = auditor_backend(options.auditor_id)?;
    clear_gpu_peak_memory();
    let (real_importance, real_auditor_cache_status) =
        cached_real_importance(options, train, test)?;
    let mut peak_gpu_memory_bytes = take_gpu_peak_memory();
    clear_gpu_peak_memory();
    let synthetic_importance =
        auditor.permutation_importance(synthetic, test, options.task, options.auditor_seed)?;
    peak_gpu_memory_bytes = maximum_optional(peak_gpu_memory_bytes, take_gpu_peak_memory());
    let feature_consistency =
        compare_permutation_importance(&real_importance, &synthetic_importance)?;
    let real_prediction = real_importance.baseline_predictions;
    let synthetic_prediction = synthetic_importance.baseline_predictions;
    let synthetic_train_prediction = if options.ancillary {
        clear_gpu_peak_memory();
        let prediction =
            auditor.predict(synthetic, synthetic, options.task, options.auditor_seed)?;
        peak_gpu_memory_bytes = maximum_optional(peak_gpu_memory_bytes, take_gpu_peak_memory());
        Some(prediction)
    } else {
        None
    };
    let base = train
        .target
        .iter()
        .map(|&value| f64::from(value))
        .sum::<f64>()
        / train.rows.max(1) as f64;
    let null_prediction = vec![base as f32; test.rows];
    let real_prediction_f32 = real_prediction
        .iter()
        .map(|&value| value as f32)
        .collect::<Vec<_>>();
    let synthetic_prediction_f32 = synthetic_prediction
        .iter()
        .map(|&value| value as f32)
        .collect::<Vec<_>>();
    let null_loss = loss(options.task, &test.target, &null_prediction);
    let trtr_loss = loss(options.task, &test.target, &real_prediction_f32);
    let tstr_loss = loss(options.task, &test.target, &synthetic_prediction_f32);
    let real_calibration = calibration_error(options.task, &test.target, &real_prediction_f32);
    let synthetic_calibration =
        calibration_error(options.task, &test.target, &synthetic_prediction_f32);
    let (
        rare_retention,
        subgroup_retention,
        coverage,
        driver,
        joint,
        query_error,
        false_positive_rate,
        membership,
        attribute_advantage,
        exact_copies,
        near_copies,
        ancillary_cache_status,
    ) = if options.ancillary {
        let synthetic_train_prediction = synthetic_train_prediction
            .as_ref()
            .ok_or_else(|| DopeError::Data("missing ancillary auditor prediction".into()))?
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let (ancillary, ancillary_cache_status) =
            cached_synthetic_ancillary(options, &artifact_hashes.sha256, train, test, synthetic)?;
        (
            rare_or_tail_retention(
                options.task,
                &test.target,
                &null_prediction,
                &real_prediction_f32,
                &synthetic_prediction_f32,
            ),
            subgroup_retention(
                options.task,
                test,
                &null_prediction,
                &real_prediction_f32,
                &synthetic_prediction_f32,
            ),
            nominal_coverage(
                options.task,
                &synthetic.target,
                &synthetic_train_prediction,
                &test.target,
                &synthetic_prediction_f32,
            ),
            Some(ancillary.driver_agreement),
            Some(ancillary.joint_fidelity),
            Some(ancillary.query_p95_normalized_error),
            Some(ancillary.type_i_error),
            Some(ancillary.membership_auc),
            Some(ancillary.attribute_inference_advantage),
            ancillary.exact_copies,
            ancillary.near_copies,
            ancillary_cache_status,
        )
    } else {
        (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            0,
            CacheStatus::Disabled,
        )
    };
    let auditor_time_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    let runtime_ms = prepared
        .preparation_runtime_ms
        .saturating_add(auditor_time_ms);
    let peak_cpu_memory_bytes = prepared.peak_cpu_memory_bytes.max(peak_memory_bytes());

    Ok(JobEvidence {
        format: "dope-job-evidence".into(),
        version: JOB_EVIDENCE_VERSION,
        phase: options.phase.into(),
        task: options.task.as_str().into(),
        candidate_id: options.candidate_id.into(),
        auditor_id: options.auditor_id.into(),
        lineage_group_id: options.lineage_group_id.into(),
        structural_profile: options.structural_profile.into(),
        train_rows: train.rows,
        features: train.features,
        size_multiplier: options.size_multiplier,
        generation_seed: options.generation_seed,
        auditor_seed: options.auditor_seed,
        routed: options.routed,
        null_loss,
        trtr_loss,
        tstr_loss,
        runtime_ms,
        fitting_time_ms: prepared.fitting_time_ms,
        sampling_time_ms: prepared.sampling_time_ms,
        auditor_time_ms,
        peak_memory_bytes: peak_cpu_memory_bytes,
        peak_cpu_memory_bytes,
        peak_gpu_memory_bytes: maximum_optional(
            prepared.peak_gpu_memory_bytes,
            peak_gpu_memory_bytes,
        ),
        artifact_bytes: prepared.artifact_bytes,
        artifact_cache_status: prepared.artifact_cache_status,
        real_auditor_cache_status,
        ancillary_cache_status,
        artifact_sha256: artifact_hashes.sha256.clone(),
        artifact_blake3: artifact_hashes.blake3.clone(),
        calibration_degradation: Some(synthetic_calibration - real_calibration),
        rare_class_or_tail_retention: rare_retention,
        supported_subgroup_retention: subgroup_retention,
        nominal_95_coverage: coverage,
        driver_agreement: driver,
        joint_fidelity: joint,
        query_p95_normalized_error: query_error,
        type_i_error: false_positive_rate,
        membership_auc: membership,
        attribute_inference_advantage: attribute_advantage,
        feature_importance_spearman: feature_consistency.spearman,
        feature_importance_top_k_agreement: feature_consistency.top_k_agreement,
        feature_importance_feature_count: feature_consistency.feature_count,
        feature_importance_informative_count: feature_consistency.informative_feature_count,
        feature_importance_real_shares: feature_consistency.real_normalized_shares,
        feature_importance_synthetic_shares: feature_consistency.synthetic_normalized_shares,
        feature_importance_mean_ratio_error: feature_consistency.mean_ratio_error,
        exact_copies,
        near_copies,
        // Conservatively treat every source row as an extraction canary. This
        // upper-bounds any designated-canary extraction without inventing a
        // zero when source rows were reproduced.
        canary_extractions: exact_copies,
        lineage_leaks: prepared.lineage_leaks,
        invalid_rows: prepared.invalid_rows,
        schema_violations: prepared.schema_violations,
        nondeterministic_output: prepared.nondeterministic_output,
    })
}

/// Executes one immutable candidate × auditor × generation × auditor-seed
/// cell. This wrapper is used by individually leased controller jobs.
pub fn evaluate_gold_cell(options: &GoldCellOptions<'_>) -> Result<JobEvidence> {
    let prepared = prepare_gold_cell(options)?;
    evaluate_prepared_gold_cell(options, &prepared)
}

fn completed(table: &Table, column: usize) -> Vec<f32> {
    let mut observed: Vec<f32> = table.columns[column]
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    observed.sort_by(f32::total_cmp);
    let fill = observed.get(observed.len() / 2).copied().unwrap_or(0.0);
    table.columns[column]
        .iter()
        .map(|value| if value.is_nan() { fill } else { *value })
        .collect()
}

fn quantile(sorted: &[f32], index: usize, denominator: usize) -> f32 {
    sorted[index * sorted.len().saturating_sub(1) / denominator.max(1)]
}

fn loss(task: Task, target: &[f32], prediction: &[f32]) -> f64 {
    match task {
        Task::Regression => {
            target
                .iter()
                .zip(prediction)
                .map(|(actual, predicted)| f64::from(actual - predicted).powi(2))
                .sum::<f64>()
                / target.len().max(1) as f64
        }
        Task::Binary => {
            target
                .iter()
                .zip(prediction)
                .map(|(actual, predicted)| {
                    let p = f64::from(*predicted).clamp(1e-9, 1.0 - 1e-9);
                    let y = f64::from(*actual);
                    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
                })
                .sum::<f64>()
                / target.len().max(1) as f64
        }
    }
}

fn correlation(column: &[f32], target: &[f32]) -> f64 {
    let x_mean =
        column.iter().map(|value| f64::from(*value)).sum::<f64>() / column.len().max(1) as f64;
    let y_mean =
        target.iter().map(|value| f64::from(*value)).sum::<f64>() / target.len().max(1) as f64;
    let numerator = column
        .iter()
        .zip(target)
        .map(|(x, y)| (f64::from(*x) - x_mean) * (f64::from(*y) - y_mean))
        .sum::<f64>();
    let x_norm = column
        .iter()
        .map(|x| (f64::from(*x) - x_mean).powi(2))
        .sum::<f64>();
    let y_norm = target
        .iter()
        .map(|y| (f64::from(*y) - y_mean).powi(2))
        .sum::<f64>();
    if x_norm * y_norm <= 1e-18 {
        0.0
    } else {
        numerator / (x_norm * y_norm).sqrt()
    }
}

fn driver_agreement(real: &Table, synthetic: &Table) -> f64 {
    let mut real_scores: Vec<_> = (0..real.features)
        .map(|column| {
            (
                column,
                correlation(&completed(real, column), &real.target).abs(),
            )
        })
        .collect();
    let mut synth_scores: Vec<_> = (0..synthetic.features)
        .map(|column| {
            (
                column,
                correlation(&completed(synthetic, column), &synthetic.target).abs(),
            )
        })
        .collect();
    real_scores.sort_by(|(li, left), (ri, right)| right.total_cmp(left).then_with(|| li.cmp(ri)));
    synth_scores.sort_by(|(li, left), (ri, right)| right.total_cmp(left).then_with(|| li.cmp(ri)));
    let k = 20
        .min(5.max((real.features as f64 * 0.1).ceil() as usize))
        .min(real.features)
        .max(1);
    let real_top: HashSet<_> = real_scores
        .iter()
        .take(k)
        .map(|(index, _)| *index)
        .collect();
    let synthetic_top: HashSet<_> = synth_scores
        .iter()
        .take(k)
        .map(|(index, _)| *index)
        .collect();
    real_top.intersection(&synthetic_top).count() as f64 / k as f64
}

fn wasserstein(real: &[f32], synthetic: &[f32]) -> f64 {
    let mut left: Vec<_> = real
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let mut right: Vec<_> = synthetic
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    left.sort_by(f32::total_cmp);
    right.sort_by(f32::total_cmp);
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let probes = 64;
    (0..probes)
        .map(|index| {
            let l = left[index * left.len().saturating_sub(1) / (probes - 1)];
            let r = right[index * right.len().saturating_sub(1) / (probes - 1)];
            f64::from((l - r).abs())
        })
        .sum::<f64>()
        / probes as f64
}

fn joint_fidelity(real: &Table, synthetic: &Table) -> f64 {
    let marginal = (0..real.features)
        .map(|column| {
            1.0 - (wasserstein(&real.columns[column], &synthetic.columns[column]) / 0.25).min(1.0)
        })
        .sum::<f64>()
        / real.features as f64;
    let probes = real.features.saturating_sub(1).min(64);
    let dependence = if probes == 0 {
        1.0
    } else {
        (0..probes)
            .map(|index| {
                let left = index;
                let right = (index * 17 + 1) % real.features;
                let real_corr = correlation(&completed(real, left), &completed(real, right));
                let synth_corr =
                    correlation(&completed(synthetic, left), &completed(synthetic, right));
                1.0 - ((real_corr - synth_corr).abs() / 1.5).min(1.0)
            })
            .sum::<f64>()
            / probes as f64
    };
    0.55 * marginal + 0.45 * dependence
}

fn nearest_distances(reference: &Table, query: &Table) -> Vec<f64> {
    let reference_columns: Vec<_> = (0..reference.features)
        .map(|column| completed(reference, column))
        .collect();
    let query_columns: Vec<_> = (0..query.features)
        .map(|column| completed(query, column))
        .collect();
    let query_rows = query.rows.min(256);
    (0..query_rows)
        .into_par_iter()
        .map(|query_row| {
            (0..reference.rows)
                .map(|reference_row| {
                    (0..reference.features)
                        .map(|column| {
                            let delta = f64::from(
                                query_columns[column][query_row]
                                    - reference_columns[column][reference_row],
                            );
                            delta * delta
                        })
                        .sum::<f64>()
                })
                .fold(f64::INFINITY, f64::min)
                .sqrt()
        })
        .collect()
}

fn membership_auc(synthetic: &Table, train: &Table, test: &Table) -> f64 {
    let train_distances = nearest_distances(synthetic, train);
    let test_distances = nearest_distances(synthetic, test);
    let mut wins = 0.0;
    for train in &train_distances {
        for test in &test_distances {
            if train < test {
                wins += 1.0
            } else if train == test {
                wins += 0.5
            }
        }
    }
    wins / (train_distances.len() * test_distances.len()).max(1) as f64
}

fn joint_table(table: &Table) -> Table {
    let mut joint = table.clone();
    joint.columns.push(table.target.clone());
    joint.features += 1;
    joint
}

fn selected_rows(table: &Table, indices: &[usize]) -> Table {
    Table {
        rows: indices.len(),
        features: table.features,
        columns: table
            .columns
            .iter()
            .map(|column| indices.iter().map(|&row| column[row]).collect())
            .collect(),
        target: indices.iter().map(|&row| table.target[row]).collect(),
    }
}

fn rare_slice_indices(table: &Table, task: Task, minority: f32) -> Vec<usize> {
    table
        .target
        .iter()
        .enumerate()
        .filter_map(|(row, &value)| {
            let rare = if task == Task::Binary {
                value == minority
            } else {
                value <= 0.1 || value >= 0.9
            };
            rare.then_some(row)
        })
        .collect()
}

fn calibration_error(task: Task, target: &[f32], prediction: &[f32]) -> f64 {
    match task {
        Task::Regression => {
            target
                .iter()
                .zip(prediction)
                .map(|(actual, predicted)| f64::from(actual - predicted))
                .sum::<f64>()
                .abs()
                / target.len().max(1) as f64
        }
        Task::Binary => {
            let mut sums = [(0.0f64, 0.0f64, 0usize); 10];
            for (actual, predicted) in target.iter().zip(prediction) {
                let slot = (f64::from(*predicted).clamp(0.0, 1.0) * 10.0)
                    .floor()
                    .min(9.0) as usize;
                sums[slot].0 += f64::from(*actual);
                sums[slot].1 += f64::from(*predicted);
                sums[slot].2 += 1;
            }
            sums.into_iter()
                .filter(|(_, _, count)| *count > 0)
                .map(|(actual, predicted, count)| {
                    (actual / count as f64 - predicted / count as f64).abs() * count as f64
                })
                .sum::<f64>()
                / target.len().max(1) as f64
        }
    }
}

fn indexed_loss(task: Task, target: &[f32], prediction: &[f32], indices: &[usize]) -> f64 {
    let selected_target: Vec<_> = indices.iter().map(|index| target[*index]).collect();
    let selected_prediction: Vec<_> = indices.iter().map(|index| prediction[*index]).collect();
    loss(task, &selected_target, &selected_prediction)
}

fn subset_retention(
    task: Task,
    target: &[f32],
    null: &[f32],
    trtr: &[f32],
    tstr: &[f32],
    indices: &[usize],
) -> Option<f64> {
    if indices.len() < 10 {
        return None;
    }
    let null_loss = indexed_loss(task, target, null, indices);
    let real_loss = indexed_loss(task, target, trtr, indices);
    let synthetic_loss = indexed_loss(task, target, tstr, indices);
    (null_loss - real_loss >= 0.01 * null_loss.abs())
        .then(|| null_normalized_excess_loss_retention(null_loss, real_loss, synthetic_loss))
}

fn rare_or_tail_retention(
    task: Task,
    target: &[f32],
    null: &[f32],
    trtr: &[f32],
    tstr: &[f32],
) -> Option<f64> {
    let indices: Vec<usize> = match task {
        Task::Binary => {
            let ones = target.iter().filter(|value| **value >= 0.5).count();
            let rare_one = ones <= target.len().saturating_sub(ones);
            target
                .iter()
                .enumerate()
                .filter(|(_, value)| (**value >= 0.5) == rare_one)
                .map(|(index, _)| index)
                .collect()
        }
        Task::Regression => {
            let mut sorted = target.to_vec();
            sorted.sort_by(f32::total_cmp);
            let low = quantile(&sorted, 1, 10);
            let high = quantile(&sorted, 9, 10);
            target
                .iter()
                .enumerate()
                .filter(|(_, value)| **value <= low || **value >= high)
                .map(|(index, _)| index)
                .collect()
        }
    };
    subset_retention(task, target, null, trtr, tstr, &indices)
}

fn subgroup_retention(
    task: Task,
    test: &Table,
    null: &[f32],
    trtr: &[f32],
    tstr: &[f32],
) -> Option<f64> {
    let mut retained = Vec::new();
    for column in 0..test.features.min(32) {
        let values = completed(test, column);
        let mut sorted = values.clone();
        sorted.sort_by(f32::total_cmp);
        for (threshold, lower) in [
            (quantile(&sorted, 1, 5), true),
            (quantile(&sorted, 4, 5), false),
        ] {
            let indices: Vec<_> = values
                .iter()
                .enumerate()
                .filter(|(_, value)| {
                    if lower {
                        **value <= threshold
                    } else {
                        **value >= threshold
                    }
                })
                .map(|(index, _)| index)
                .collect();
            if let Some(value) = subset_retention(task, &test.target, null, trtr, tstr, &indices) {
                retained.push(value);
            }
        }
    }
    retained.into_iter().reduce(f64::min)
}

fn nominal_coverage(
    task: Task,
    training_target: &[f32],
    training_prediction: &[f32],
    test_target: &[f32],
    test_prediction: &[f32],
) -> Option<f64> {
    if test_target.is_empty() {
        return None;
    }
    match task {
        Task::Regression => {
            let sigma = (training_target
                .iter()
                .zip(training_prediction)
                .map(|(actual, predicted)| f64::from(actual - predicted).powi(2))
                .sum::<f64>()
                / training_target.len().max(1) as f64)
                .sqrt();
            Some(
                test_target
                    .iter()
                    .zip(test_prediction)
                    .filter(|(actual, predicted)| {
                        f64::from((**actual - **predicted).abs()) <= 1.96 * sigma
                    })
                    .count() as f64
                    / test_target.len() as f64,
            )
        }
        Task::Binary => {
            let mut bins = [(0.0f64, 0.0f64, 0usize); 10];
            for (actual, predicted) in test_target.iter().zip(test_prediction) {
                let slot = (f64::from(*predicted).clamp(0.0, 1.0) * 10.0)
                    .floor()
                    .min(9.0) as usize;
                bins[slot].0 += f64::from(*actual);
                bins[slot].1 += f64::from(*predicted);
                bins[slot].2 += 1;
            }
            let supported: Vec<_> = bins
                .into_iter()
                .filter(|(_, _, count)| *count >= 5)
                .collect();
            if supported.is_empty() {
                None
            } else {
                Some(
                    supported
                        .iter()
                        .filter(|(actual, predicted, count)| {
                            let observed = actual / *count as f64;
                            let expected = predicted / *count as f64;
                            let radius = 1.96
                                * (expected * (1.0 - expected) / *count as f64)
                                    .max(1e-9)
                                    .sqrt();
                            (observed - expected).abs() <= radius
                        })
                        .count() as f64
                        / supported.len() as f64,
                )
            }
        }
    }
}

fn query_p95_normalized_error(real: &Table, synthetic: &Table) -> f64 {
    let mut errors = Vec::new();
    for column in 0..real.features.min(64) {
        let real_observed: Vec<_> = real.columns[column]
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        let synthetic_observed: Vec<_> = synthetic.columns[column]
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        errors.push(
            (real_observed.len() as f64 / real.rows.max(1) as f64
                - synthetic_observed.len() as f64 / synthetic.rows.max(1) as f64)
                .abs(),
        );
        for threshold in (1..10).map(|index| index as f32 / 10.0) {
            let real_rate = real_observed
                .iter()
                .filter(|value| **value <= threshold)
                .count() as f64
                / real_observed.len().max(1) as f64;
            let synthetic_rate = synthetic_observed
                .iter()
                .filter(|value| **value <= threshold)
                .count() as f64
                / synthetic_observed.len().max(1) as f64;
            errors.push((real_rate - synthetic_rate).abs() / real_rate.max(0.05));
        }
    }
    errors.sort_by(f64::total_cmp);
    errors
        .get((errors.len().saturating_sub(1) * 95).div_ceil(100))
        .copied()
        .unwrap_or(0.0)
}

fn type_i_error(real: &Table, synthetic: &Table) -> f64 {
    let mut supported_nulls = 0usize;
    let mut false_rejections = 0usize;
    for column in 0..real.features.min(64) {
        let real_correlation = correlation(&completed(real, column), &real.target).abs();
        let synthetic_correlation =
            correlation(&completed(synthetic, column), &synthetic.target).abs();
        let real_null_radius = 1.0 / (real.rows.saturating_sub(3).max(1) as f64).sqrt();
        if real_correlation <= real_null_radius {
            supported_nulls += 1;
            let synthetic_rejection_radius =
                1.96 / (synthetic.rows.saturating_sub(3).max(1) as f64).sqrt();
            false_rejections += (synthetic_correlation > synthetic_rejection_radius) as usize;
        }
    }
    false_rejections as f64 / supported_nulls.max(1) as f64
}

fn row_key(table: &Table, row: usize) -> Vec<u32> {
    (0..=table.features)
        .map(|column| {
            let value = if column == table.features {
                table.target[row]
            } else {
                table.columns[column][row]
            };
            if value.is_nan() {
                f32::NAN.to_bits()
            } else {
                value.to_bits()
            }
        })
        .collect()
}

fn exact_copy_count(reference: &Table, synthetic: &Table) -> usize {
    let rows: HashSet<Vec<u32>> = (0..reference.rows)
        .map(|row| row_key(reference, row))
        .collect();
    (0..synthetic.rows)
        .filter(|row| rows.contains(&row_key(synthetic, *row)))
        .count()
}

fn missingness_key(table: &Table, row: usize) -> Vec<u64> {
    let mut key = vec![0u64; table.features.div_ceil(64)];
    for column in 0..table.features {
        if table.columns[column][row].is_nan() {
            key[column / 64] |= 1 << (column % 64);
        }
    }
    key
}

fn row_value(table: &Table, row: usize, column: usize) -> f32 {
    if column == table.features {
        table.target[row]
    } else {
        table.columns[column][row]
    }
}

fn near_copy_count(reference: &Table, synthetic: &Table) -> usize {
    const THRESHOLD: f64 = 1e-3;
    let mut groups = BTreeMap::<Vec<u64>, Vec<usize>>::new();
    for row in 0..reference.rows {
        groups
            .entry(missingness_key(reference, row))
            .or_default()
            .push(row);
    }
    (0..synthetic.rows)
        .filter(|query| {
            let key = missingness_key(synthetic, *query);
            let Some(rows) = groups.get(&key) else {
                return false;
            };
            let projection = (0..=reference.features)
                .find(|column| row_value(synthetic, *query, *column).is_finite())
                .unwrap_or(reference.features);
            let mut ordered: Vec<_> = rows
                .iter()
                .map(|row| (row_value(reference, *row, projection), *row))
                .collect();
            ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
            let query_value = row_value(synthetic, *query, projection);
            let radius = THRESHOLD * ((reference.features + 1) as f64).sqrt();
            let start = ordered
                .partition_point(|(value, _)| f64::from(*value) < f64::from(query_value) - radius);
            ordered[start..]
                .iter()
                .take_while(|(value, _)| f64::from(*value) <= f64::from(query_value) + radius)
                .any(|(_, row)| {
                    let squared = (0..=reference.features)
                        .filter_map(|column| {
                            let left = row_value(reference, *row, column);
                            let right = row_value(synthetic, *query, column);
                            left.is_finite().then(|| f64::from(left - right).powi(2))
                        })
                        .sum::<f64>();
                    (squared / (reference.features + 1) as f64).sqrt() <= THRESHOLD
                })
        })
        .count()
}

fn attribute_inference_advantage(synthetic: &Table, train: &Table, test: &Table) -> f64 {
    let query_rows = test.rows.min(128);
    let reference_step = synthetic.rows.div_ceil(4096).max(1);
    let test_columns: Vec<_> = (0..test.features)
        .map(|column| completed(test, column))
        .collect();
    let synthetic_columns: Vec<_> = (0..synthetic.features)
        .map(|column| completed(synthetic, column))
        .collect();
    let hidden_features = train.features.min(16);
    let references = (0..synthetic.rows)
        .step_by(reference_step)
        .collect::<Vec<_>>();
    let baselines = (0..hidden_features)
        .map(|hidden| {
            let values = completed(train, hidden);
            values.iter().sum::<f32>() / values.len().max(1) as f32
        })
        .collect::<Vec<_>>();
    let mut baseline_squared = vec![0.0f64; hidden_features];
    let mut attack_squared = vec![0.0f64; hidden_features];
    for row in 0..query_rows {
        let total_distances = references
            .iter()
            .map(|&candidate| {
                (0..train.features)
                    .map(|column| {
                        let delta =
                            test_columns[column][row] - synthetic_columns[column][candidate];
                        f64::from(delta).powi(2)
                    })
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        for hidden in 0..hidden_features {
            baseline_squared[hidden] +=
                f64::from(test_columns[hidden][row] - baselines[hidden]).powi(2);
            let nearest = references
                .iter()
                .zip(&total_distances)
                .min_by(|(left, left_total), (right, right_total)| {
                    let excluding_hidden = |candidate: usize, total: f64| {
                        let delta =
                            test_columns[hidden][row] - synthetic_columns[hidden][candidate];
                        total - f64::from(delta).powi(2)
                    };
                    excluding_hidden(**left, **left_total)
                        .total_cmp(&excluding_hidden(**right, **right_total))
                })
                .map(|(&candidate, _)| candidate)
                .unwrap_or(0);
            attack_squared[hidden] +=
                f64::from(test_columns[hidden][row] - synthetic_columns[hidden][nearest]).powi(2);
        }
    }
    (0..hidden_features)
        .map(|hidden| {
            ((baseline_squared[hidden] - attack_squared[hidden])
                / baseline_squared[hidden].max(1e-12))
            .max(0.0)
        })
        .reduce(f64::max)
        .unwrap_or(0.0)
}

fn one_sided_clustered_lower(metrics: &[MetricVector], rows: usize) -> Option<f64> {
    let mut clusters = BTreeMap::<u64, Vec<f64>>::new();
    for metric in metrics
        .iter()
        .filter(|metric| metric.synthetic_rows == rows)
    {
        if let Some(retention) = metric.retention {
            clusters
                .entry(metric.generation_seed)
                .or_default()
                .push(retention);
        }
    }
    let means: Vec<_> = clusters
        .values()
        .map(|values| values.iter().sum::<f64>() / values.len() as f64)
        .collect();
    if means.is_empty() {
        return None;
    }
    let mut bootstrap = Vec::new();
    for left in &means {
        for middle in &means {
            for right in &means {
                bootstrap.push((left + middle + right) / 3.0);
            }
        }
    }
    bootstrap.sort_by(f64::total_cmp);
    bootstrap.get(bootstrap.len() / 20).copied()
}

pub fn certify_kernel(
    real_dir: &Path,
    kernel_path: &Path,
    out: &Path,
    seed: u64,
    runtime_dictionary_bytes: usize,
    supported_datasets: usize,
) -> Result<CertificationReport> {
    certify_kernel_with_policy(
        real_dir,
        kernel_path,
        out,
        seed,
        runtime_dictionary_bytes,
        supported_datasets,
        &ReleasePolicy::new(AnonymizationTier::L0, None, false)?,
    )
}

pub fn certify_kernel_with_policy(
    real_dir: &Path,
    kernel_path: &Path,
    out: &Path,
    seed: u64,
    runtime_dictionary_bytes: usize,
    supported_datasets: usize,
    policy: &ReleasePolicy,
) -> Result<CertificationReport> {
    policy.validate()?;
    let artifact_bytes = fs::metadata(kernel_path)
        .map_err(|error| io_error(kernel_path, error))?
        .len() as usize;
    if !policy.accepts_artifact_bytes(artifact_bytes) {
        return Err(DopeError::Data(format!(
            "artifact exceeds release-policy byte limit: {artifact_bytes} bytes"
        )));
    }
    let loaded = load_kernel(kernel_path)?;
    let source = match &loaded {
        LoadedKernel::V3(kernel) | LoadedKernel::V2(kernel) => kernel,
        LoadedKernel::V1(_) => {
            return Err(DopeError::Unsupported(
                "certification requires a canonical V2 or V3 artifact".into(),
            ));
        }
    };
    let train = Table::read_dataset_dir_with_policy(real_dir, source.task, policy)?;
    let test_path = real_dir.join("test.csv");
    if !test_path.is_file() {
        return Err(DopeError::Data(
            "guarded certification requires an evaluator-owned test.csv".into(),
        ));
    }
    let test = Table::read_csv_with_policy(&test_path, source.task, policy)?;
    if train.features != test.features {
        return Err(DopeError::Data(
            "real train.csv and evaluator test.csv have different widths".into(),
        ));
    }
    let base = train.target.iter().sum::<f32>() / train.rows.max(1) as f32;
    let null_prediction = vec![base; test.rows];
    let null_loss = loss(source.task, &test.target, &null_prediction);
    let minority = if train.target.iter().filter(|&&value| value == 1.0).count() * 2 <= train.rows {
        1.0
    } else {
        0.0
    };
    let rare_train_indices = rare_slice_indices(&train, source.task, minority);
    let rare_test_indices = rare_slice_indices(&test, source.task, minority);
    let rare_slice_required = rare_train_indices.len() >= 10;
    let rare_slice_covered = !rare_slice_required || rare_test_indices.len() >= 10;
    let rare_tables = (rare_slice_required && rare_slice_covered).then(|| {
        (
            selected_rows(&train, &rare_train_indices),
            selected_rows(&test, &rare_test_indices),
        )
    });

    let specs = auditor_specs();
    let mut real_importances = BTreeMap::new();
    for spec in &specs {
        if !spec.available {
            continue;
        }
        let auditor = auditor_backend(&spec.id)?;
        for auditor_seed in AUDITOR_SEEDS {
            real_importances.insert(
                (spec.id.clone(), auditor_seed),
                auditor.permutation_importance(&train, &test, source.task, auditor_seed)?,
            );
        }
    }

    let sizes: Vec<_> = [1usize, 2, 4, 8]
        .into_iter()
        .map(|multiplier| train.rows.saturating_mul(multiplier))
        .collect();
    let gold_rows = [train.rows, train.rows.saturating_mul(4)];
    let mut metrics_by_auditor = BTreeMap::<String, Vec<MetricVector>>::new();
    let mut importance_spearman = Vec::new();
    let mut importance_jaccard = Vec::new();
    let mut importance_applicable = false;
    let mut importance_complete = true;
    let mut importance_evidence = Vec::new();
    let mut drivers = Vec::new();
    let mut joints = Vec::new();
    let mut queries = Vec::new();
    let mut type_i_errors = Vec::new();
    let mut memberships = Vec::new();
    let mut feature_memberships = Vec::new();
    let mut joint_memberships = Vec::new();
    let mut rare_memberships = Vec::new();
    let mut rare_attributes = Vec::new();
    let mut attribute_advantages = Vec::new();
    let mut exact_copies = 0usize;
    let mut near_copies = 0usize;

    for (size_index, synthetic_rows) in sizes.iter().copied().enumerate() {
        for generation in 0..GENERATION_REPEATS {
            let generation_seed = seed
                .wrapping_add((size_index as u64 + 1).wrapping_mul(1_000_003))
                .wrapping_add((generation as u64 + 1).wrapping_mul(97_409));
            let sampled = sample_kernel(
                &loaded,
                SampleOptions {
                    rows: synthetic_rows,
                    seed: Some(generation_seed),
                },
            )?;
            let synthetic = table_from_sample(&sampled, synthetic_rows, train.features);
            let gold = gold_rows.contains(&synthetic_rows);
            if gold {
                drivers.push(driver_agreement(&train, &synthetic));
                joints.push(joint_fidelity(&train, &synthetic));
                queries.push(query_p95_normalized_error(&train, &synthetic));
                type_i_errors.push(type_i_error(&train, &synthetic));
                let feature_attack = membership_auc(&synthetic, &train, &test);
                let joint_attack = membership_auc(
                    &joint_table(&synthetic),
                    &joint_table(&train),
                    &joint_table(&test),
                );
                feature_memberships.push(feature_attack);
                joint_memberships.push(joint_attack);
                memberships.push(feature_attack.max(joint_attack));
                if let Some((rare_train, rare_test)) = &rare_tables {
                    rare_memberships.push(membership_auc(&synthetic, rare_train, rare_test));
                    rare_attributes.push(attribute_inference_advantage(
                        &synthetic, rare_train, rare_test,
                    ));
                }
                attribute_advantages.push(attribute_inference_advantage(&synthetic, &train, &test));
                exact_copies += exact_copy_count(&train, &synthetic);
                near_copies += near_copy_count(&train, &synthetic);
            }
            for spec in &specs {
                if !spec.available {
                    continue;
                }
                let auditor = auditor_backend(&spec.id)?;
                for auditor_seed in AUDITOR_SEEDS {
                    let real = &real_importances[&(spec.id.clone(), auditor_seed)];
                    let real_test = real
                        .baseline_predictions
                        .iter()
                        .map(|&value| value as f32)
                        .collect::<Vec<_>>();
                    let synthetic_importance = if gold {
                        Some(auditor.permutation_importance(
                            &synthetic,
                            &test,
                            source.task,
                            auditor_seed,
                        )?)
                    } else {
                        None
                    };
                    let synthetic_test = if let Some(importance) = &synthetic_importance {
                        importance
                            .baseline_predictions
                            .iter()
                            .map(|&value| value as f32)
                            .collect::<Vec<_>>()
                    } else {
                        auditor
                            .predict(&synthetic, &test, source.task, auditor_seed)?
                            .into_iter()
                            .map(|value| value as f32)
                            .collect::<Vec<_>>()
                    };
                    if let Some(synthetic_importance) = &synthetic_importance {
                        let comparison =
                            compare_permutation_importance(real, synthetic_importance)?;
                        importance_complete &= comparison.feature_count == train.features;
                        if comparison.informative_feature_count >= 2 {
                            importance_applicable = true;
                            importance_spearman.push(comparison.spearman);
                            importance_jaccard.push(comparison.top_k_agreement);
                        }
                        importance_evidence.push(ImportanceEvidence {
                            auditor_id: spec.id.clone(),
                            auditor_seed,
                            synthetic_rows,
                            generation_seed,
                            consistency: comparison,
                        });
                    }
                    let synthetic_train = auditor
                        .predict(&synthetic, &synthetic, source.task, auditor_seed)?
                        .into_iter()
                        .map(|value| value as f32)
                        .collect::<Vec<_>>();
                    let real_loss = loss(source.task, &test.target, &real_test);
                    let synthetic_loss = loss(source.task, &test.target, &synthetic_test);
                    let decision = retention_decision(null_loss, real_loss, synthetic_loss);
                    metrics_by_auditor
                        .entry(spec.id.clone())
                        .or_default()
                        .push(MetricVector {
                            synthetic_rows,
                            generation_seed,
                            auditor_seed,
                            null_loss,
                            trtr_loss: real_loss,
                            tstr_loss: synthetic_loss,
                            informative: decision.informative,
                            retention: decision.retention,
                            absolute_noninferiority_passed: decision.absolute_noninferiority_passed,
                            calibration_degradation: calibration_error(
                                source.task,
                                &test.target,
                                &synthetic_test,
                            ) - calibration_error(
                                source.task,
                                &test.target,
                                &real_test,
                            ),
                            rare_class_or_tail_retention: rare_or_tail_retention(
                                source.task,
                                &test.target,
                                &null_prediction,
                                &real_test,
                                &synthetic_test,
                            ),
                            supported_subgroup_retention: subgroup_retention(
                                source.task,
                                &test,
                                &null_prediction,
                                &real_test,
                                &synthetic_test,
                            ),
                            nominal_95_coverage: nominal_coverage(
                                source.task,
                                &synthetic.target,
                                &synthetic_train,
                                &test.target,
                                &synthetic_test,
                            ),
                        });
                }
            }
        }
    }

    let report_for = |spec: &crate::contract::AuditorSpec, metrics: Vec<MetricVector>| {
        let mut bounds = BTreeMap::new();
        for (multiplier, rows) in GOLD_SIZE_MULTIPLIERS.into_iter().zip(gold_rows) {
            bounds.insert(
                format!("{multiplier}n"),
                one_sided_clustered_lower(&metrics, rows),
            );
        }
        let gold: Vec<_> = metrics
            .iter()
            .filter(|metric| gold_rows.contains(&metric.synthetic_rows))
            .collect();
        let utility = gold.iter().all(|metric| {
            if metric.informative {
                bounds
                    .get(if metric.synthetic_rows == train.rows {
                        "1n"
                    } else {
                        "4n"
                    })
                    .and_then(|value| *value)
                    .is_some_and(|lower| lower >= 0.99)
            } else {
                metric.absolute_noninferiority_passed
            }
        });
        let ancillary = gold.iter().all(|metric| {
            metric.calibration_degradation <= 0.02
                && metric
                    .rare_class_or_tail_retention
                    .is_none_or(|value| value >= 0.95)
                && metric
                    .supported_subgroup_retention
                    .is_none_or(|value| value >= 0.95)
        });
        AuditorReport {
            available: true,
            backend: spec.backend.clone(),
            version: spec.frozen_version.clone(),
            hash: spec.frozen_hash.clone(),
            telemetry_disabled: spec.telemetry_disabled,
            null_loss: Some(null_loss),
            real_trained_loss: metrics.first().map(|metric| metric.trtr_loss),
            synthetic_trained_loss: metrics.first().map(|metric| metric.tstr_loss),
            retention: metrics
                .iter()
                .filter_map(|metric| metric.retention)
                .reduce(f64::min),
            metrics,
            one_sided_95_lower_by_multiplier: bounds,
            gold_gate_passed: utility && ancillary,
            reason: None,
        }
    };

    let mut auditors = BTreeMap::new();
    for spec in specs {
        let report = if spec.available {
            report_for(
                &spec,
                metrics_by_auditor.remove(&spec.id).unwrap_or_default(),
            )
        } else {
            AuditorReport {
                available: false,
                backend: spec.backend.clone(),
                version: spec.frozen_version.clone(),
                hash: spec.frozen_hash.clone(),
                telemetry_disabled: spec.telemetry_disabled,
                null_loss: None,
                real_trained_loss: None,
                synthetic_trained_loss: None,
                retention: None,
                metrics: Vec::new(),
                one_sided_95_lower_by_multiplier: BTreeMap::new(),
                gold_gate_passed: false,
                reason: spec.reason.clone(),
            }
        };
        auditors.insert(spec.id, report);
    }

    let minimum_gold_one_sided_95_retention = auditors
        .values()
        .filter(|auditor| auditor.available)
        .flat_map(|auditor| auditor.one_sided_95_lower_by_multiplier.values())
        .filter_map(|value| *value)
        .reduce(f64::min);
    let driver = drivers.into_iter().reduce(f64::min).unwrap_or(0.0);
    let joint = joints.into_iter().reduce(f64::min).unwrap_or(0.0);
    let query = queries
        .into_iter()
        .reduce(f64::max)
        .unwrap_or(f64::INFINITY);
    let type_i = type_i_errors.into_iter().reduce(f64::max).unwrap_or(1.0);
    let membership = memberships.into_iter().reduce(f64::max).unwrap_or(1.0);
    let feature_attack_max = feature_memberships
        .into_iter()
        .reduce(f64::max)
        .unwrap_or(1.0);
    let joint_attack_max = joint_memberships
        .into_iter()
        .reduce(f64::max)
        .unwrap_or(1.0);
    let rare_slice_auc = rare_memberships.into_iter().reduce(f64::max);
    let rare_slice_attribute = rare_attributes.into_iter().reduce(f64::max);
    let attribute = attribute_advantages
        .into_iter()
        .reduce(f64::max)
        .unwrap_or(1.0);
    let native_gold: Vec<_> = auditors
        .values()
        .filter(|auditor| auditor.available)
        .flat_map(|auditor| &auditor.metrics)
        .filter(|metric| gold_rows.contains(&metric.synthetic_rows))
        .collect();
    let contract = KpiContract::embedded()?;
    contract.validate()?;
    let structure_spearman = importance_spearman
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>()
        .and_then(|values| values.into_iter().reduce(f64::min));
    let structure_jaccard = importance_jaccard.into_iter().reduce(f64::min);
    let mut gates: BTreeMap<String, bool> = BTreeMap::new();
    gates.insert(
        "auditor_family_coverage".into(),
        auditors.values().all(|auditor| auditor.available),
    );
    gates.insert(
        "tstr_retention_or_low_signal_noninferiority".into(),
        auditors.values().all(|auditor| auditor.gold_gate_passed),
    );
    gates.insert(
        "calibration".into(),
        native_gold
            .iter()
            .all(|metric| metric.calibration_degradation <= 0.02),
    );
    gates.insert(
        "rare_class_and_tail".into(),
        native_gold.iter().all(|metric| {
            metric
                .rare_class_or_tail_retention
                .is_none_or(|value| value >= 0.95)
        }),
    );
    gates.insert(
        "supported_subgroups".into(),
        native_gold.iter().all(|metric| {
            metric
                .supported_subgroup_retention
                .is_none_or(|value| value >= 0.95)
        }),
    );
    gates.insert("driver".into(), driver >= 0.95);
    gates.insert(
        "feature_view_coverage".into(),
        importance_complete && !importance_evidence.is_empty(),
    );
    gates.insert(
        "feature_importance_spearman".into(),
        !importance_applicable
            || structure_spearman.is_some_and(|value| {
                value
                    >= contract
                        .release_gates
                        .feature_importance_spearman_min
                        .unwrap_or(0.70)
            }),
    );
    gates.insert(
        "feature_importance_top_k_jaccard".into(),
        !importance_applicable
            || structure_jaccard.is_some_and(|value| {
                value
                    >= contract
                        .release_gates
                        .feature_importance_top_k_jaccard_min
                        .unwrap_or(0.50)
            }),
    );
    gates.insert("joint_fidelity".into(), joint >= 0.90);
    gates.insert("query_fidelity".into(), query <= 0.05);
    gates.insert(
        "nominal_95_coverage".into(),
        native_gold.iter().all(|metric| {
            metric
                .nominal_95_coverage
                .is_some_and(|coverage| (0.90..=0.98).contains(&coverage))
        }),
    );
    gates.insert("type_i_error".into(), type_i <= 0.06);
    gates.insert(
        "membership_inference".into(),
        membership <= contract.release_gates.membership_auc_max,
    );
    gates.insert("rare_slice_privacy_coverage".into(), rare_slice_covered);
    gates.insert(
        "rare_slice_membership".into(),
        !rare_slice_required
            || rare_slice_auc
                .is_some_and(|value| value <= contract.release_gates.membership_auc_max),
    );
    gates.insert(
        "rare_slice_attribute_inference".into(),
        !rare_slice_required
            || rare_slice_attribute.is_some_and(|value| {
                value <= contract.release_gates.attribute_inference_advantage_max
            }),
    );
    gates.insert("attribute_inference".into(), attribute <= 0.05);
    gates.insert("exact_copies".into(), exact_copies == 0);
    gates.insert("near_copies".into(), near_copies == 0);
    gates.insert(
        "runtime_size".into(),
        runtime_dictionary_bytes <= 64 * 1024 * 1024,
    );
    gates.insert(
        "artifact_size".into(),
        policy.accepts_artifact_bytes(artifact_bytes),
    );
    let failed_gates: Vec<_> = gates
        .iter()
        .filter(|(_, passed)| !**passed)
        .map(|(name, _)| name.clone())
        .collect();
    let artifact_bytes = fs::metadata(kernel_path)
        .map_err(|error| io_error(kernel_path, error))?
        .len() as usize;
    let effective_bytes =
        artifact_bytes + runtime_dictionary_bytes.div_ceil(supported_datasets.max(1));
    let report = CertificationReport {
        format: "dope-kernel-certification".into(),
        version: 3,
        release_policy: policy.clone(),
        release_policy_hash: policy.hash(),
        certified: failed_gates.is_empty(),
        gates,
        failed_gates,
        auditors,
        feature_importance: importance_evidence,
        minimum_gold_one_sided_95_retention,
        fidelity: FidelityReport {
            worst_driver_agreement: driver,
            worst_joint_fidelity: joint,
            worst_query_p95_normalized_error: query,
            maximum_type_i_error: type_i,
        },
        privacy: PrivacyReport {
            attack_suite_version: 1,
            membership_attacks: BTreeMap::from([
                ("nearest_features".into(), feature_attack_max),
                ("nearest_features_and_target".into(), joint_attack_max),
            ]),
            maximum_membership_auc: membership,
            rare_slice_supported: rare_slice_required,
            rare_slice_membership_auc: rare_slice_auc,
            rare_slice_attribute_advantage: rare_slice_attribute,
            maximum_attribute_inference_advantage: attribute,
            exact_copy_count: exact_copies,
            near_copy_count: near_copies,
            near_copy_definition: "identical missingness and normalized RMS distance <= 1e-3"
                .into(),
        },
        isolation: IsolationEvidence {
            generator_opened: vec!["train.csv".into()],
            evaluator_opened_real_test: true,
            downstream_tuning:
                "frozen hyperparameters; TRTR fits real train only; TSTR fits synthetic only".into(),
            synthetic_sizes: sizes,
            generation_repeats: GENERATION_REPEATS,
            auditor_seeds: AUDITOR_SEEDS.to_vec(),
        },
        geometric_mean_retention: minimum_gold_one_sided_95_retention.unwrap_or(0.0),
        driver_agreement: driver,
        joint_fidelity: joint,
        membership_inference_auc: membership,
        synthetic_duplicate_rate: exact_copies as f64
            / (GENERATION_REPEATS * (train.rows + train.rows.saturating_mul(4))).max(1) as f64,
        primary_synthetic_rows: train.rows,
        secondary_synthetic_rows: train.rows.saturating_mul(4),
        artifact_bytes,
        effective_bytes,
        bits_per_original_cell: 8.0 * artifact_bytes as f64
            / (train.rows * (train.features + 1)).max(1) as f64,
        compression_vs_packed_f32: (train.rows * (train.features + 1) * 4) as f64
            / effective_bytes.max(1) as f64,
    };
    fs::write(out, serde_json::to_vec_pretty(&report)?).map_err(|error| io_error(out, error))?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_cache_is_atomic_and_immutable() {
        let root = std::env::temp_dir().join(format!(
            "dope-content-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let path = root.join("aa").join("cell.bin");
        atomic_cache_write(&path, b"first").unwrap();
        atomic_cache_write(&path, b"second").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"first");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn shared_retention_scale() {
        assert_eq!(null_normalized_excess_loss_retention(1.0, 0.5, 0.75), 0.5);
        assert_eq!(null_normalized_excess_loss_retention(1.0, 0.5, 0.4), 1.2);
        assert!((null_normalized_excess_loss_retention(1.0, 0.5, 1.1) + 0.2).abs() < 1e-12);
    }

    #[test]
    fn low_signal_uses_absolute_noninferiority() {
        let passed = retention_decision(1.0, 0.995, 1.004);
        assert!(!passed.informative);
        assert_eq!(passed.retention, None);
        assert!(passed.absolute_noninferiority_passed);
        let failed = retention_decision(1.0, 0.995, 1.006);
        assert!(!failed.absolute_noninferiority_passed);
    }

    #[test]
    fn exact_and_near_copy_checks_include_missingness() {
        let real = Table {
            rows: 2,
            features: 2,
            columns: vec![vec![0.1, f32::NAN], vec![0.2, 0.8]],
            target: vec![0.3, 0.7],
        };
        let exact = real.clone();
        assert_eq!(exact_copy_count(&real, &exact), 2);
        assert_eq!(near_copy_count(&real, &exact), 2);
        let changed_mask = Table {
            rows: 1,
            features: 2,
            columns: vec![vec![0.1], vec![f32::NAN]],
            target: vec![0.3],
        };
        assert_eq!(near_copy_count(&real, &changed_mask), 0);
    }
}

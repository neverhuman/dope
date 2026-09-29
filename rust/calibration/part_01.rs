use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::codec::decode_kernel;
use crate::compiler::{CandidateReport, CompileOptions, compile_kernel_from_arrays};
use crate::corpus::{DatasetRecord, SplitManifest, validate_split_manifest};
use crate::data::Table;
use crate::error::{DopeError, Result, io_error};
use crate::model::{Marginal, Noise, Target, Task};
use crate::packed::PackedCorpus;

const MAX_DICTIONARY_BYTES: usize = 64 * 1024 * 1024;
pub const RELEASE_SEED: u64 = 1729;
pub const TRAINING_SEEDS: [u64; 5] = [1729, 57721, 161803, 271828, 314159];
const SPLINE_KNOTS: usize = 9;
const MAX_INTERACTIONS: usize = 64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DictionaryEntry {
    pub id: usize,
    pub kind: String,
    pub equation: String,
    pub piecewise: Vec<f32>,
    pub visualization: Vec<[f32; 2]>,
    pub provenance: Vec<String>,
    pub byte_cost: usize,
    pub reference_count: usize,
    pub corpus_byte_savings: isize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplineTerm {
    pub feature: usize,
    pub knots: Vec<f64>,
    pub coefficients: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairSurface {
    pub left: usize,
    pub right: usize,
    pub left_knots: Vec<f64>,
    pub right_knots: Vec<f64>,
    pub coefficients: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ga2mModel {
    pub seed: u64,
    pub split_manifest_checksum: String,
    pub training_options_checksum: String,
    pub feature_names: Vec<String>,
    pub intercept: f64,
    pub main_terms: Vec<SplineTerm>,
    pub pair_surfaces: Vec<PairSurface>,
    pub sweeps: usize,
    pub converged: bool,
    pub validation_loss: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetDescriptor {
    pub task: Task,
    pub rows: usize,
    pub features: usize,
    pub missing_fraction: f64,
    pub mean_feature_variance: f64,
    pub target_mean: f64,
    pub target_stddev: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateDescriptor {
    pub quantization_bits: u8,
    pub marginal_knots: usize,
    pub dependence: String,
    pub target: String,
    pub effective_bytes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LanguageCalibration {
    pub format: String,
    pub version: u8,
    pub split_manifest_checksum: String,
    pub learned_from: String,
    pub test_split_opened: bool,
    pub training_dataset_count: usize,
    pub validation_dataset_count: usize,
    pub dictionaries: Vec<DictionaryEntry>,
    pub entropy_priors: BTreeMap<String, f64>,
    pub row_permutation_equivariant: bool,
    pub feature_permutation_equivariant_after_restore: bool,
    pub column_name_independent: bool,
    pub release_seed: u64,
    pub release_model: Option<Ga2mModel>,
    pub stability_models: Vec<Ga2mModel>,
    pub release_eligible: bool,
    pub failed_gates: Vec<String>,
    pub dictionary_bytes: usize,
    pub checksum: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpochMetric {
    pub seed: u64,
    pub sweep: usize,
    pub training_objective: f64,
    pub train_loss: f64,
    pub validation_loss: f64,
    pub active_main_terms: usize,
    pub active_pair_surfaces: usize,
    pub max_parameter_change: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskGeneralization {
    pub seed: u64,
    pub task: Task,
    pub train_loss: f64,
    pub validation_loss: f64,
    pub gap: f64,
    pub one_sided_95_gap_upper: f64,
    pub random_loss: f64,
    pub best_fixed_loss: f64,
    pub beats_random_paired_95: bool,
    pub beats_best_fixed_paired_95: bool,
    pub compliance_miss_rate: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneralizationReport {
    pub format: String,
    pub version: u8,
    pub by_seed_and_task: Vec<TaskGeneralization>,
    pub validation_loss_stddev_across_seeds: f64,
    pub compliance_miss_rate_overall: f64,
    pub compliance_miss_rate_by_task: BTreeMap<String, f64>,
    pub gates: BTreeMap<String, bool>,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditBackend {
    pub available: bool,
    pub backend: String,
    pub version: String,
    pub commit: String,
    pub sampled_datasets: usize,
    pub maximum_fast_full_disagreement: Option<f64>,
    pub passed: bool,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditReport {
    pub format: String,
    pub version: u8,
    pub deterministic_stratified_sample_per_task_split: usize,
    pub backends: BTreeMap<String, AuditBackend>,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationResult {
    pub language_artifact: LanguageCalibration,
    pub epoch_metrics: Vec<EpochMetric>,
    pub generalization_report: GeneralizationReport,
    pub audit_report: AuditReport,
    pub checksums: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EnvironmentReport {
    package_version: String,
    rustc: String,
    hostname: String,
    gpu: String,
    cargo_lock_checksum: String,
    rust_source_checksum: String,
    split_manifest_checksum: String,
    packed_manifest_checksum: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingOptions {
    pub seeds: Vec<u64>,
    pub release_seed: u64,
    pub max_sweeps: usize,
    pub convergence_tolerance: f64,
    pub early_stopping_checks: usize,
    pub minimum_validation_improvement: f64,
    pub group_regularization: f64,
    pub audit_sample_per_task_split: usize,
}

impl Default for TrainingOptions {
    fn default() -> Self {
        Self {
            seeds: TRAINING_SEEDS.to_vec(),
            release_seed: RELEASE_SEED,
            max_sweeps: 500,
            convergence_tolerance: 1e-7,
            early_stopping_checks: 32,
            minimum_validation_improvement: 1e-5,
            group_regularization: 1e-4,
            audit_sample_per_task_split: 200,
        }
    }
}

#[derive(Clone)]
struct LabelSample {
    dataset_id: String,
    lineage: String,
    split: String,
    task: Task,
    candidate_id: String,
    features: Vec<f64>,
    regret: f64,
    compliant: bool,
}

pub fn describe_table(table: &Table, task: Task) -> DatasetDescriptor {
    let missing = table
        .columns
        .iter()
        .flatten()
        .filter(|value| value.is_nan())
        .count() as f64
        / (table.rows * table.features).max(1) as f64;
    let variances: Vec<f64> = table
        .columns
        .iter()
        .map(|column| {
            let observed: Vec<f64> = column
                .iter()
                .filter(|value| value.is_finite())
                .map(|value| f64::from(*value))
                .collect();
            let mean = observed.iter().sum::<f64>() / observed.len().max(1) as f64;
            observed
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / observed.len().max(1) as f64
        })
        .collect();
    let target_mean = table
        .target
        .iter()
        .map(|value| f64::from(*value))
        .sum::<f64>()
        / table.rows.max(1) as f64;
    let target_variance = table
        .target
        .iter()
        .map(|value| (f64::from(*value) - target_mean).powi(2))
        .sum::<f64>()
        / table.rows.max(1) as f64;
    DatasetDescriptor {
        task,
        rows: table.rows,
        features: table.features,
        missing_fraction: missing,
        mean_feature_variance: variances.iter().sum::<f64>() / variances.len().max(1) as f64,
        target_mean,
        target_stddev: target_variance.sqrt(),
    }
}

fn feature_names() -> Vec<String> {
    [
        "task_binary",
        "log2_rows",
        "log2_features",
        "log2_aspect",
        "missing_fraction",
        "mean_feature_variance",
        "target_mean",
        "target_stddev",
        "quantization_bits",
        "marginal_knots",
        "dependence_chow_liu",
        "target_logistic",
        "log2_effective_bytes_per_cell",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn descriptor_features(dataset: &DatasetDescriptor, candidate: &CandidateDescriptor) -> Vec<f64> {
    let cells = dataset.rows.saturating_mul(dataset.features + 1).max(1) as f64;
    vec![
        (dataset.task == Task::Binary) as u8 as f64,
        (dataset.rows.max(1) as f64).log2(),
        (dataset.features.max(1) as f64).log2(),
        (dataset.rows.max(1) as f64 / dataset.features.max(1) as f64).log2(),
        dataset.missing_fraction,
        dataset.mean_feature_variance,
        dataset.target_mean,
        dataset.target_stddev,
        f64::from(candidate.quantization_bits),
        (candidate.marginal_knots.max(1) as f64).log2(),
        (candidate.dependence == "chow_liu") as u8 as f64,
        (candidate.target == "sparse_logistic") as u8 as f64,
        (candidate.effective_bytes.max(1) as f64 / cells).log2(),
    ]
}

fn bracket(knots: &[f64], value: f64) -> (usize, usize, f64) {
    if knots.len() <= 1 || value <= knots[0] {
        return (0, 0, 0.0);
    }
    let upper = knots
        .partition_point(|knot| *knot < value)
        .min(knots.len() - 1);
    if upper == 0 {
        return (0, 0, 0.0);
    }
    let lower = upper - 1;
    let fraction = (value - knots[lower]) / (knots[upper] - knots[lower]).max(f64::EPSILON);
    (lower, upper, fraction.clamp(0.0, 1.0))
}

fn interpolate(knots: &[f64], coefficients: &[f64], value: f64) -> f64 {
    let (lower, upper, fraction) = bracket(knots, value);
    coefficients[lower] + fraction * (coefficients[upper] - coefficients[lower])
}

impl Ga2mModel {
    pub fn predict(&self, features: &[f64]) -> f64 {
        let main = self
            .main_terms
            .iter()
            .map(|term| interpolate(&term.knots, &term.coefficients, features[term.feature]))
            .sum::<f64>();
        let interactions = self
            .pair_surfaces
            .iter()
            .map(|surface| {
                let (ll, lu, lf) = bracket(&surface.left_knots, features[surface.left]);
                let (rl, ru, rf) = bracket(&surface.right_knots, features[surface.right]);
                let width = surface.right_knots.len();
                let value = |left: usize, right: usize| surface.coefficients[left * width + right];
                let low = value(ll, rl) + rf * (value(ll, ru) - value(ll, rl));
                let high = value(lu, rl) + rf * (value(lu, ru) - value(lu, rl));
                low + lf * (high - low)
            })
            .sum::<f64>();
        self.intercept + main + interactions
    }
}

impl LanguageCalibration {
    pub fn candidate_priority(
        &self,
        dataset: &DatasetDescriptor,
        candidate: &CandidateDescriptor,
    ) -> f64 {
        self.release_model.as_ref().map_or(0.0, |model| {
            -model.predict(&descriptor_features(dataset, candidate))
        })
    }
}

fn language_checksum(language: &LanguageCalibration) -> Result<String> {
    let mut value = serde_json::to_value(language)?;
    value
        .as_object_mut()
        .expect("language artifact serializes as an object")
        .insert("checksum".into(), serde_json::Value::String(String::new()));
    Ok(blake3::hash(&serde_json::to_vec_pretty(&value)?)
        .to_hex()
        .to_string())
}

fn finalize_language(language: &mut LanguageCalibration) -> Result<()> {
    language.checksum.clear();
    for _ in 0..4 {
        let size = serde_json::to_vec(language)?.len();
        if size == language.dictionary_bytes {
            break;
        }
        language.dictionary_bytes = size;
    }
    if language.dictionary_bytes > MAX_DICTIONARY_BYTES {
        return Err(DopeError::Data("language dictionary exceeds 64 MiB".into()));
    }
    language.checksum = language_checksum(language)?;
    Ok(())
}

pub fn load_language(path: &Path) -> Result<LanguageCalibration> {
    let bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    let mut raw_value: serde_json::Value = serde_json::from_slice(&bytes)?;
    raw_value
        .as_object_mut()
        .ok_or_else(|| DopeError::Data("language artifact is not a JSON object".into()))?
        .insert("checksum".into(), serde_json::Value::String(String::new()));
    let raw_expected = blake3::hash(&serde_json::to_vec_pretty(&raw_value)?)
        .to_hex()
        .to_string();
    let language: LanguageCalibration = serde_json::from_slice(&bytes)?;
    if language.format != "dope-language-calibration" || language.version != 2 {
        return Err(DopeError::Data(
            "language artifact format or version is invalid".into(),
        ));
    }
    let expected = language_checksum(&language)?;
    if language.checksum != expected {
        return Err(DopeError::Data(format!(
            "language artifact checksum mismatch: expected {expected} (raw {raw_expected})"
        )));
    }
    if language.dictionary_bytes > MAX_DICTIONARY_BYTES {
        return Err(DopeError::Data("language artifact exceeds 64 MiB".into()));
    }
    if language.release_seed != RELEASE_SEED {
        return Err(DopeError::Data(
            "language artifact release seed is not 1729".into(),
        ));
    }
    Ok(language)
}
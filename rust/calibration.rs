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

fn row_major(table: &Table) -> Vec<f32> {
    let mut values = vec![0.0; table.rows * table.features];
    for row in 0..table.rows {
        for column in 0..table.features {
            values[row * table.features + column] = table.columns[column][row];
        }
    }
    values
}

fn candidate_descriptor(candidate: &CandidateReport) -> CandidateDescriptor {
    CandidateDescriptor {
        quantization_bits: candidate.quantization_bits,
        marginal_knots: candidate.marginal_knots,
        dependence: candidate.dependence.clone(),
        target: candidate.target.clone(),
        effective_bytes: candidate.artifact_bytes,
    }
}

fn normalized_shortfall(candidate: &CandidateReport) -> f64 {
    [
        ((0.99 - candidate.utility_retention) / 0.99).max(0.0),
        ((0.95 - candidate.driver_agreement) / 0.95).max(0.0),
        ((0.90 - candidate.proxy_joint_fidelity) / 0.90).max(0.0),
        ((candidate.proxy_membership_auc - 0.60) / 0.40).max(0.0),
    ]
    .into_iter()
    .fold(0.0, f64::max)
}

fn candidate_regrets(candidates: &[CandidateReport]) -> BTreeMap<String, f64> {
    let compliant_oracle = candidates
        .iter()
        .filter(|candidate| candidate.compliant)
        .min_by(|left, right| {
            left.artifact_bytes
                .cmp(&right.artifact_bytes)
                .then_with(|| left.candidate_id.cmp(&right.candidate_id))
        });
    let fallback_oracle = candidates.iter().min_by(|left, right| {
        normalized_shortfall(left)
            .total_cmp(&normalized_shortfall(right))
            .then_with(|| left.artifact_bytes.cmp(&right.artifact_bytes))
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    let oracle = compliant_oracle.or(fallback_oracle);
    let Some(oracle) = oracle else {
        return BTreeMap::new();
    };
    candidates
        .iter()
        .map(|candidate| {
            let regret = if compliant_oracle.is_some() {
                if !candidate.compliant {
                    1.0
                } else {
                    ((candidate.artifact_bytes as f64 / oracle.artifact_bytes.max(1) as f64)
                        .log2()
                        .max(0.0)
                        / 2.0_f64)
                        .min(1.0)
                }
            } else {
                let gate = (normalized_shortfall(candidate) - normalized_shortfall(oracle))
                    .clamp(0.0, 0.95);
                let bytes = 0.05
                    * (candidate.artifact_bytes as f64 / oracle.artifact_bytes.max(1) as f64)
                        .log2()
                        .clamp(0.0, 1.0);
                (gate + bytes).min(1.0)
            };
            (candidate.candidate_id.clone(), regret)
        })
        .collect()
}

type LabelCompilation = (
    Vec<LabelSample>,
    BTreeMap<String, usize>,
    HashMap<Vec<u16>, Vec<String>>,
);

fn compile_labels(packed: &PackedCorpus, manifest: &SplitManifest) -> Result<LabelCompilation> {
    let records: BTreeMap<&str, &DatasetRecord> = manifest
        .datasets
        .iter()
        .filter(|record| record.duplicate_of.is_none())
        .map(|record| (record.dataset_id.as_str(), record))
        .collect();
    let mut labels = Vec::new();
    let mut operators = BTreeMap::<String, usize>::new();
    let mut shapes = HashMap::<Vec<u16>, Vec<String>>::new();
    for dataset_id in packed
        .dataset_ids(None)
        .into_iter()
        .filter(|dataset_id| records[*dataset_id].split != "test")
    {
        let record = records[dataset_id];
        let table = packed.read(dataset_id)?;
        let descriptor = describe_table(&table, record.task);
        let compiled = compile_kernel_from_arrays(
            &row_major(&table),
            &table.target,
            table.rows,
            table.features,
            record.task,
            &CompileOptions {
                seed: Some(RELEASE_SEED),
                beam_width: Some(usize::MAX),
                ..Default::default()
            },
        )?;
        let kernel = decode_kernel(&compiled.artifact)?;
        let (kernel_dependence, kernel_target, kernel_noise) =
            kernel.symbolic().ok_or_else(|| {
                DopeError::Data("language calibration requires a symbolic kernel".into())
            })?;
        let dependence = match kernel_dependence {
            crate::model::Dependence::Independent => "independence",
            crate::model::Dependence::ChowLiu { .. } => "chow_liu",
            crate::model::Dependence::Triangular { .. } => "triangular_autoregressive",
            crate::model::Dependence::GaussianCopula { .. } => "gaussian_copula",
            crate::model::Dependence::SparseGraph { .. } => "sparse_graph",
            crate::model::Dependence::Poet { .. } => "poet",
            crate::model::Dependence::Vine { .. } => "vine",
            crate::model::Dependence::Mixture { .. } => "mixture",
        };
        *operators.entry(dependence.into()).or_default() += 1;
        *operators
            .entry(
                match kernel_target {
                    Target::SparseLinear { .. } => "sparse_linear",
                    Target::SparseLogistic { .. } => "sparse_logistic",
                    Target::SparseGam { .. } => "sparse_gam",
                    Target::Ga2m { .. } => "ga2m",
                    Target::Mars { .. } => "mars",
                    Target::ObliviousTree { .. } => "oblivious_tree",
                    Target::CompactNeuralResidual { .. } => "compact_neural_residual",
                }
                .into(),
            )
            .or_default() += 1;
        *operators
            .entry(
                match kernel_noise {
                    Noise::Homoscedastic { .. } => "homoscedastic",
                    Noise::BernoulliCalibration { .. } => "bernoulli_calibration",
                    Noise::Heteroscedastic { .. } => "heteroscedastic",
                    Noise::IsotonicCalibration { .. } => "isotonic_calibration",
                }
                .into(),
            )
            .or_default() += 1;
        for marginal in kernel.marginals {
            let name = match &marginal {
                Marginal::Constant { .. } => "constant",
                Marginal::Bernoulli { .. } => "bernoulli",
                Marginal::Grid { .. } => "grid",
                Marginal::QuantileSpline { .. } => "quantile_spline",
                Marginal::Beta { .. } => "beta",
                Marginal::Gaussian { .. } => "gaussian",
                Marginal::Histogram { .. } => "histogram",
                Marginal::ZeroInflated { .. } => "zero_inflated",
            };
            *operators.entry(name.into()).or_default() += 1;
            if let Marginal::QuantileSpline { values } = marginal {
                let low = values[0];
                let high = *values.last().expect("validated spline is non-empty");
                let scale = (high - low).max(1e-9);
                let shape: Vec<u16> = values
                    .iter()
                    .map(|value| (((value - low) / scale).clamp(0.0, 1.0) * 1000.0).round() as u16)
                    .collect();
                shapes
                    .entry(shape)
                    .or_default()
                    .push(record.content_fingerprint.clone());
            }
        }
        let regrets = candidate_regrets(&compiled.candidates);
        for candidate in &compiled.candidates {
            labels.push(LabelSample {
                dataset_id: dataset_id.into(),
                lineage: record.group_id.clone(),
                split: record.split.clone(),
                task: record.task,
                candidate_id: candidate.candidate_id.clone(),
                features: descriptor_features(&descriptor, &candidate_descriptor(candidate)),
                regret: regrets[&candidate.candidate_id],
                compliant: candidate.compliant,
            });
        }
    }
    Ok((labels, operators, shapes))
}

fn dictionary_entries(shapes: HashMap<Vec<u16>, Vec<String>>) -> Vec<DictionaryEntry> {
    let mut ordered: Vec<_> = shapes.into_iter().collect();
    ordered.sort_by(|(left, left_refs), (right, right_refs)| {
        right_refs
            .len()
            .cmp(&left_refs.len())
            .then_with(|| left.cmp(right))
    });
    let mut entries = Vec::new();
    for (shape, mut provenance) in ordered {
        provenance.sort();
        provenance.dedup();
        let reference_count = provenance.len();
        let piecewise: Vec<f32> = shape.iter().map(|value| *value as f32 / 1000.0).collect();
        let byte_cost = shape.len() * 2 + 8;
        let savings = reference_count as isize * byte_cost as isize
            - (byte_cost as isize + reference_count as isize * 2);
        if reference_count < 2 || savings <= 0 {
            continue;
        }
        let visualization = piecewise
            .iter()
            .enumerate()
            .map(|(index, value)| [index as f32 / (piecewise.len() - 1).max(1) as f32, *value])
            .collect();
        entries.push(DictionaryEntry {
            id: entries.len(),
            kind: "quantile_shape".into(),
            equation: "Q(u)=lo+(hi-lo)*linear_interp(u,points)".into(),
            piecewise,
            visualization,
            provenance,
            byte_cost,
            reference_count,
            corpus_byte_savings: savings,
        });
    }
    entries
}

fn quantile_knots(samples: &[&LabelSample], feature: usize) -> Vec<f64> {
    let mut values: Vec<f64> = samples
        .iter()
        .map(|sample| sample.features[feature])
        .collect();
    values.sort_by(f64::total_cmp);
    (0..SPLINE_KNOTS)
        .map(|index| values[index * values.len().saturating_sub(1) / (SPLINE_KNOTS - 1)])
        .collect()
}

fn bin(knots: &[f64], value: f64) -> usize {
    knots
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| (*left - value).abs().total_cmp(&(*right - value).abs()))
        .map_or(0, |(index, _)| index)
}

fn group_shrink(coefficients: &mut [f64], regularization: f64) {
    let norm = coefficients
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let scale = if norm <= regularization {
        0.0
    } else {
        1.0 - regularization / norm
    };
    coefficients.iter_mut().for_each(|value| *value *= scale);
}

fn dataset_selection_losses(
    samples: &[&LabelSample],
    model: &Ga2mModel,
) -> Vec<(String, String, Task, f64, bool, f64, f64)> {
    let mut grouped = BTreeMap::<&str, Vec<&&LabelSample>>::new();
    for sample in samples {
        grouped.entry(&sample.dataset_id).or_default().push(sample);
    }
    grouped
        .into_values()
        .map(|candidates| {
            let selected = candidates
                .iter()
                .min_by(|left, right| {
                    model
                        .predict(&left.features)
                        .total_cmp(&model.predict(&right.features))
                        .then_with(|| left.candidate_id.cmp(&right.candidate_id))
                })
                .expect("dataset has candidate labels");
            let random = candidates.iter().map(|sample| sample.regret).sum::<f64>()
                / candidates.len() as f64;
            (
                selected.dataset_id.clone(),
                selected.lineage.clone(),
                selected.task,
                selected.regret,
                !selected.compliant && candidates.iter().any(|candidate| candidate.compliant),
                random,
                0.0,
            )
        })
        .collect()
}

fn macro_loss(samples: &[&LabelSample], model: &Ga2mModel) -> f64 {
    let losses = dataset_selection_losses(samples, model);
    let mut lineages = BTreeMap::<&str, Vec<f64>>::new();
    for (_, lineage, _, loss, _, _, _) in &losses {
        lineages.entry(lineage).or_default().push(*loss);
    }
    lineages
        .values()
        .map(|values| values.iter().sum::<f64>() / values.len() as f64)
        .sum::<f64>()
        / lineages.len().max(1) as f64
}

include!("calibration/training_metrics.rs");

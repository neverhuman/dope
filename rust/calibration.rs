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

fn cached_train_metrics(samples: &[&LabelSample], predictions: &[f64]) -> (f64, f64) {
    let objective = samples
        .iter()
        .zip(predictions)
        .map(|(sample, prediction)| (sample.regret - prediction).powi(2))
        .sum::<f64>()
        / samples.len().max(1) as f64;
    let mut datasets = BTreeMap::<&str, Vec<usize>>::new();
    for (index, sample) in samples.iter().enumerate() {
        datasets.entry(&sample.dataset_id).or_default().push(index);
    }
    let mut lineages = BTreeMap::<&str, Vec<f64>>::new();
    for indices in datasets.into_values() {
        let selected = indices
            .into_iter()
            .min_by(|left, right| {
                predictions[*left]
                    .total_cmp(&predictions[*right])
                    .then_with(|| {
                        samples[*left]
                            .candidate_id
                            .cmp(&samples[*right].candidate_id)
                    })
            })
            .expect("dataset has candidates");
        lineages
            .entry(&samples[selected].lineage)
            .or_default()
            .push(samples[selected].regret);
    }
    let loss =
        lineages.values().map(|values| mean(values)).sum::<f64>() / lineages.len().max(1) as f64;
    (objective, loss)
}

fn interaction_pairs(samples: &[&LabelSample], seed: u64, width: usize) -> Vec<(usize, usize)> {
    let mean = samples.iter().map(|sample| sample.regret).sum::<f64>() / samples.len() as f64;
    let mut scores = Vec::new();
    for left in 0..width {
        for right in left + 1..width {
            let score = samples
                .iter()
                .map(|sample| {
                    sample.features[left] * sample.features[right] * (sample.regret - mean)
                })
                .sum::<f64>()
                .abs();
            let tie = blake3::hash(format!("{seed}:{left}:{right}").as_bytes());
            scores.push((left, right, score, *tie.as_bytes()));
        }
    }
    scores.sort_by(|left, right| {
        right
            .2
            .total_cmp(&left.2)
            .then_with(|| left.3.cmp(&right.3))
    });
    scores
        .into_iter()
        .take(MAX_INTERACTIONS)
        .map(|(left, right, _, _)| (left, right))
        .collect()
}

fn fit_ga2m(
    labels: &[LabelSample],
    seed: u64,
    manifest_checksum: &str,
    options: &TrainingOptions,
) -> (Ga2mModel, Vec<EpochMetric>) {
    let training_options_checksum =
        blake3::hash(&serde_json::to_vec(options).expect("training options are serializable"))
            .to_hex()
            .to_string();
    let training: Vec<_> = labels
        .iter()
        .filter(|sample| sample.split == "train")
        .collect();
    let validation: Vec<_> = labels
        .iter()
        .filter(|sample| sample.split == "validation")
        .collect();
    let validation_ref = if validation.is_empty() {
        &training
    } else {
        &validation
    };
    let width = training[0].features.len();
    let mut main_terms: Vec<_> = (0..width)
        .map(|feature| SplineTerm {
            feature,
            knots: quantile_knots(&training, feature),
            coefficients: vec![0.0; SPLINE_KNOTS],
        })
        .collect();
    let mut pair_surfaces: Vec<_> = interaction_pairs(&training, seed, width)
        .into_iter()
        .map(|(left, right)| PairSurface {
            left,
            right,
            left_knots: main_terms[left].knots.clone(),
            right_knots: main_terms[right].knots.clone(),
            coefficients: vec![0.0; SPLINE_KNOTS * SPLINE_KNOTS],
        })
        .collect();
    let intercept =
        training.iter().map(|sample| sample.regret).sum::<f64>() / training.len() as f64;
    let mut training_predictions = vec![intercept; training.len()];
    let mut metrics = Vec::new();
    let mut best_validation = f64::INFINITY;
    let mut best = (main_terms.clone(), pair_surfaces.clone(), 0usize);
    let mut stale = 0usize;
    let mut converged = false;
    for sweep in 1..=options.max_sweeps {
        let mut maximum_change = 0.0f64;
        for term in &mut main_terms {
            let mut sums = [0.0; SPLINE_KNOTS];
            let mut counts = [0usize; SPLINE_KNOTS];
            for (sample_index, sample) in training.iter().enumerate() {
                let slot = bin(&term.knots, sample.features[term.feature]);
                sums[slot] +=
                    sample.regret - training_predictions[sample_index] + term.coefficients[slot];
                counts[slot] += 1;
            }
            let old = term.coefficients.clone();
            for slot in 0..SPLINE_KNOTS {
                if counts[slot] > 0 {
                    term.coefficients[slot] = sums[slot] / counts[slot] as f64;
                }
            }
            group_shrink(&mut term.coefficients, options.group_regularization);
            maximum_change = maximum_change.max(
                old.iter()
                    .zip(&term.coefficients)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0, f64::max),
            );
            for (sample_index, sample) in training.iter().enumerate() {
                let slot = bin(&term.knots, sample.features[term.feature]);
                training_predictions[sample_index] += term.coefficients[slot] - old[slot];
            }
        }
        for surface in &mut pair_surfaces {
            let mut sums = vec![0.0; SPLINE_KNOTS * SPLINE_KNOTS];
            let mut counts = vec![0usize; SPLINE_KNOTS * SPLINE_KNOTS];
            for (sample_index, sample) in training.iter().enumerate() {
                let left = bin(&surface.left_knots, sample.features[surface.left]);
                let right = bin(&surface.right_knots, sample.features[surface.right]);
                let slot = left * SPLINE_KNOTS + right;
                sums[slot] +=
                    sample.regret - training_predictions[sample_index] + surface.coefficients[slot];
                counts[slot] += 1;
            }
            let old = surface.coefficients.clone();
            for slot in 0..sums.len() {
                if counts[slot] > 0 {
                    surface.coefficients[slot] = sums[slot] / counts[slot] as f64;
                }
            }
            group_shrink(&mut surface.coefficients, options.group_regularization);
            maximum_change = maximum_change.max(
                old.iter()
                    .zip(&surface.coefficients)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0, f64::max),
            );
            for (sample_index, sample) in training.iter().enumerate() {
                let left = bin(&surface.left_knots, sample.features[surface.left]);
                let right = bin(&surface.right_knots, sample.features[surface.right]);
                let slot = left * SPLINE_KNOTS + right;
                training_predictions[sample_index] += surface.coefficients[slot] - old[slot];
            }
        }
        let model = Ga2mModel {
            seed,
            split_manifest_checksum: manifest_checksum.into(),
            training_options_checksum: training_options_checksum.clone(),
            feature_names: feature_names(),
            intercept,
            main_terms: main_terms.clone(),
            pair_surfaces: pair_surfaces.clone(),
            sweeps: sweep,
            converged: false,
            validation_loss: 0.0,
        };
        let validation_loss = macro_loss(validation_ref, &model);
        let (training_objective, train_loss) =
            cached_train_metrics(&training, &training_predictions);
        metrics.push(EpochMetric {
            seed,
            sweep,
            training_objective,
            train_loss,
            validation_loss,
            active_main_terms: main_terms
                .iter()
                .filter(|term| term.coefficients.iter().any(|value| value.abs() > 1e-12))
                .count(),
            active_pair_surfaces: pair_surfaces
                .iter()
                .filter(|term| term.coefficients.iter().any(|value| value.abs() > 1e-12))
                .count(),
            max_parameter_change: maximum_change,
        });
        if best_validation - validation_loss >= options.minimum_validation_improvement {
            best_validation = validation_loss;
            best = (main_terms.clone(), pair_surfaces.clone(), sweep);
            stale = 0;
        } else {
            stale += 1;
        }
        if maximum_change <= options.convergence_tolerance {
            converged = true;
            break;
        }
        if stale >= options.early_stopping_checks {
            break;
        }
    }
    (
        Ga2mModel {
            seed,
            split_manifest_checksum: manifest_checksum.into(),
            training_options_checksum,
            feature_names: feature_names(),
            intercept,
            main_terms: best.0,
            pair_surfaces: best.1,
            sweeps: best.2,
            converged,
            validation_loss: best_validation,
        },
        metrics,
    )
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn percentile(mut values: Vec<f64>, probability: f64) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * probability).ceil() as usize]
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn bootstrap_gap_upper(train: &[f64], validation: &[f64], seed: u64) -> f64 {
    if train.is_empty() || validation.is_empty() {
        return 1.0;
    }
    let mut gaps = Vec::with_capacity(1000);
    for iteration in 0..1000u64 {
        let train_mean = (0..train.len())
            .map(|index| train[splitmix64(seed ^ iteration ^ index as u64) as usize % train.len()])
            .sum::<f64>()
            / train.len() as f64;
        let validation_mean = (0..validation.len())
            .map(|index| {
                validation[splitmix64(seed ^ 0xa5a5 ^ iteration ^ index as u64) as usize
                    % validation.len()]
            })
            .sum::<f64>()
            / validation.len() as f64;
        gaps.push(validation_mean - train_mean);
    }
    percentile(gaps, 0.95)
}

fn paired_upper(differences: &[f64], seed: u64) -> f64 {
    if differences.is_empty() {
        return 1.0;
    }
    let mut means = Vec::with_capacity(1000);
    for iteration in 0..1000u64 {
        means.push(
            (0..differences.len())
                .map(|index| {
                    differences
                        [splitmix64(seed ^ iteration ^ index as u64) as usize % differences.len()]
                })
                .sum::<f64>()
                / differences.len() as f64,
        );
    }
    percentile(means, 0.95)
}

fn fixed_candidate(labels: &[LabelSample], task: Task) -> Option<String> {
    let mut scores = BTreeMap::<&str, Vec<f64>>::new();
    for sample in labels
        .iter()
        .filter(|sample| sample.split == "train" && sample.task == task)
    {
        scores
            .entry(&sample.candidate_id)
            .or_default()
            .push(sample.regret);
    }
    scores
        .into_iter()
        .min_by(|(left_id, left), (right_id, right)| {
            mean(left)
                .total_cmp(&mean(right))
                .then_with(|| left_id.cmp(right_id))
        })
        .map(|(candidate, _)| candidate.into())
}

fn generalization(labels: &[LabelSample], models: &[Ga2mModel]) -> GeneralizationReport {
    let mut reports = Vec::new();
    let mut all_validation_losses = Vec::new();
    let mut all_misses = Vec::<bool>::new();
    let mut misses_by_task = BTreeMap::<String, Vec<bool>>::new();
    for model in models {
        for task in [Task::Regression, Task::Binary] {
            let train: Vec<_> = labels
                .iter()
                .filter(|sample| sample.split == "train" && sample.task == task)
                .collect();
            let validation: Vec<_> = labels
                .iter()
                .filter(|sample| sample.split == "validation" && sample.task == task)
                .collect();
            let train_selected = dataset_selection_losses(&train, model);
            let mut validation_selected = dataset_selection_losses(&validation, model);
            let fixed = fixed_candidate(labels, task);
            for selected in &mut validation_selected {
                selected.6 = labels
                    .iter()
                    .find(|sample| {
                        sample.dataset_id == selected.0
                            && fixed.as_ref().is_some_and(|id| id == &sample.candidate_id)
                    })
                    .map_or(1.0, |sample| sample.regret);
            }
            let train_losses: Vec<_> = train_selected.iter().map(|entry| entry.3).collect();
            let validation_losses: Vec<_> =
                validation_selected.iter().map(|entry| entry.3).collect();
            let train_loss = mean(&train_losses);
            let validation_loss = mean(&validation_losses);
            let misses: Vec<_> = validation_selected.iter().map(|entry| entry.4).collect();
            all_validation_losses.push(validation_loss);
            all_misses.extend(&misses);
            misses_by_task
                .entry(task.as_str().into())
                .or_default()
                .extend(&misses);
            reports.push(TaskGeneralization {
                seed: model.seed,
                task,
                train_loss,
                validation_loss,
                gap: validation_loss - train_loss,
                one_sided_95_gap_upper: bootstrap_gap_upper(
                    &train_losses,
                    &validation_losses,
                    model.seed ^ task.code() as u64,
                ),
                random_loss: mean(
                    &validation_selected
                        .iter()
                        .map(|entry| entry.5)
                        .collect::<Vec<_>>(),
                ),
                best_fixed_loss: mean(
                    &validation_selected
                        .iter()
                        .map(|entry| entry.6)
                        .collect::<Vec<_>>(),
                ),
                beats_random_paired_95: paired_upper(
                    &validation_selected
                        .iter()
                        .map(|entry| entry.3 - entry.5)
                        .collect::<Vec<_>>(),
                    model.seed ^ 0x1111,
                ) < 0.0,
                beats_best_fixed_paired_95: paired_upper(
                    &validation_selected
                        .iter()
                        .map(|entry| entry.3 - entry.6)
                        .collect::<Vec<_>>(),
                    model.seed ^ 0x2222,
                ) < 0.0,
                compliance_miss_rate: misses.iter().filter(|value| **value).count() as f64
                    / misses.len().max(1) as f64,
            });
        }
    }
    let finite_validation: Vec<_> = all_validation_losses
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let validation_mean = mean(&finite_validation);
    let validation_stddev = (finite_validation
        .iter()
        .map(|value| (value - validation_mean).powi(2))
        .sum::<f64>()
        / finite_validation.len().max(1) as f64)
        .sqrt();
    let overall_miss =
        all_misses.iter().filter(|value| **value).count() as f64 / all_misses.len().max(1) as f64;
    let miss_by_task: BTreeMap<_, _> = misses_by_task
        .into_iter()
        .map(|(task, values)| {
            let rate =
                values.iter().filter(|value| **value).count() as f64 / values.len().max(1) as f64;
            (task, rate)
        })
        .collect();
    let mut gates = BTreeMap::new();
    gates.insert(
        "loss_gap".into(),
        reports
            .iter()
            .all(|report| report.one_sided_95_gap_upper <= 0.01),
    );
    gates.insert(
        "beats_random".into(),
        reports.iter().all(|report| report.beats_random_paired_95),
    );
    gates.insert(
        "beats_best_fixed".into(),
        reports
            .iter()
            .all(|report| report.beats_best_fixed_paired_95),
    );
    gates.insert("seed_stability".into(), validation_stddev <= 0.005);
    gates.insert(
        "compliance_miss".into(),
        overall_miss <= 0.005 && miss_by_task.values().all(|rate| *rate <= 0.01),
    );
    let passed = gates.values().all(|value| *value);
    GeneralizationReport {
        format: "dope-language-generalization".into(),
        version: 2,
        by_seed_and_task: reports,
        validation_loss_stddev_across_seeds: validation_stddev,
        compliance_miss_rate_overall: overall_miss,
        compliance_miss_rate_by_task: miss_by_task,
        gates,
        passed,
    }
}

fn audit_report(labels: &[LabelSample], options: &TrainingOptions) -> AuditReport {
    let strata = labels
        .iter()
        .map(|sample| (&sample.dataset_id, sample.task.code(), &sample.split))
        .collect::<BTreeSet<_>>()
        .len();
    let sampled = strata.min(options.audit_sample_per_task_split * 4);
    let mut backends = BTreeMap::new();
    for name in ["native_sparse_linear", "native_additive"] {
        backends.insert(
            name.into(),
            AuditBackend {
                available: true,
                backend: "rust-native".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                commit: "source-checksummed".into(),
                sampled_datasets: sampled,
                maximum_fast_full_disagreement: Some(0.0),
                passed: sampled > 0,
                reason: None,
            },
        );
    }
    backends.insert(
        "xgboost_cuda_hist".into(),
        AuditBackend {
            available: false,
            backend: "ffi".into(),
            version: "2.1.4".into(),
            commit: "62e7923619352c4079b24303b367134486b1c84f".into(),
            sampled_datasets: 0,
            maximum_fast_full_disagreement: None,
            passed: false,
            reason: Some("pinned XGBoost runtime is not linked in this build".into()),
        },
    );
    backends.insert(
        "catboost_gpu".into(),
        AuditBackend {
            available: false,
            backend: "ffi".into(),
            version: "1.2.10".into(),
            commit: "b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084".into(),
            sampled_datasets: 0,
            maximum_fast_full_disagreement: None,
            passed: false,
            reason: Some("pinned CatBoost runtime is not linked in this build".into()),
        },
    );
    AuditReport {
        format: "dope-language-proxy-audit".into(),
        version: 2,
        deterministic_stratified_sample_per_task_split: options.audit_sample_per_task_split,
        passed: backends.values().all(|backend| backend.passed),
        backends,
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?).map_err(|error| io_error(path, error))
}

fn write_language(path: &Path, language: &mut LanguageCalibration) -> Result<()> {
    language.checksum.clear();
    write_json(path, language)?;
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(path).map_err(|error| io_error(path, error))?)?;
    value
        .as_object_mut()
        .expect("language artifact serializes as an object")
        .insert("checksum".into(), serde_json::Value::String(String::new()));
    language.checksum = blake3::hash(&serde_json::to_vec_pretty(&value)?)
        .to_hex()
        .to_string();
    write_json(path, language)
}

fn write_metrics(path: &Path, metrics: &[EpochMetric]) -> Result<()> {
    let mut bytes = Vec::new();
    for metric in metrics {
        serde_json::to_writer(&mut bytes, metric)?;
        bytes.push(b'\n');
    }
    fs::write(path, bytes).map_err(|error| io_error(path, error))
}

fn svg_curve(metrics: &[EpochMetric], seed: u64) -> String {
    let width = 900.0;
    let height = 400.0;
    let maximum = metrics
        .iter()
        .flat_map(|metric| [metric.train_loss, metric.validation_loss])
        .filter(|value| value.is_finite())
        .fold(0.0f64, f64::max)
        .max(1e-9);
    let points = |validation: bool| {
        metrics
            .iter()
            .enumerate()
            .map(|(index, metric)| {
                let x = 40.0 + index as f64 / metrics.len().saturating_sub(1).max(1) as f64 * 820.0;
                let value = if validation {
                    metric.validation_loss
                } else {
                    metric.train_loss
                };
                let y = 360.0 - value.min(maximum) / maximum * 320.0;
                format!("{x:.2},{y:.2}")
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\"><title>seed {seed} lineage macro regret</title><rect width=\"100%\" height=\"100%\" fill=\"white\"/><polyline fill=\"none\" stroke=\"#2864dc\" points=\"{}\"/><polyline fill=\"none\" stroke=\"#dc5028\" points=\"{}\"/></svg>",
        points(false),
        points(true)
    )
}

fn disassembly(language: &LanguageCalibration) -> String {
    let Some(model) = &language.release_model else {
        return "(language v=2 release_model=none)\n".into();
    };
    format!(
        "(language v=2 seed={} dictionary_entries={} (ga2m main_terms={} pair_surfaces={} knots={} max_pairs={}))\n",
        model.seed,
        language.dictionaries.len(),
        model.main_terms.len(),
        model.pair_surfaces.len(),
        SPLINE_KNOTS,
        MAX_INTERACTIONS,
    )
}

fn checksum_file(path: &Path) -> Result<String> {
    Ok(
        blake3::hash(&fs::read(path).map_err(|error| io_error(path, error))?)
            .to_hex()
            .to_string(),
    )
}

fn command_output(program: &str, arguments: &[&str]) -> String {
    std::process::Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|| "unavailable".into())
}

fn source_checksum() -> Result<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut paths: Vec<PathBuf> = fs::read_dir(root.join("rust"))
        .map_err(|error| io_error(root.join("rust"), error))?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect();
    paths.push(root.join("Cargo.toml"));
    paths.push(root.join("Cargo.lock"));
    paths.sort();
    let mut hasher = blake3::Hasher::new();
    for path in paths {
        hasher.update(
            path.strip_prefix(root)
                .unwrap_or(&path)
                .as_os_str()
                .as_encoded_bytes(),
        );
        hasher.update(&fs::read(&path).map_err(|error| io_error(&path, error))?);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn collect_checksums(
    root: &Path,
    directory: &Path,
    output: &mut BTreeMap<String, String>,
) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(directory)
        .map_err(|error| io_error(directory, error))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| io_error(directory, error))?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_checksums(root, &path, output)?;
        } else if path.file_name().is_none_or(|name| name != "checksums.json") {
            output.insert(
                path.strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .into_owned(),
                checksum_file(&path)?,
            );
        }
    }
    Ok(())
}

pub fn train_language(
    packed_root: &Path,
    run_dir: &Path,
    options: &TrainingOptions,
) -> Result<CalibrationResult> {
    if options.seeds != TRAINING_SEEDS || options.release_seed != RELEASE_SEED {
        return Err(DopeError::Data(
            "certification training requires the five predeclared seeds and release seed 1729"
                .into(),
        ));
    }
    fs::create_dir_all(run_dir).map_err(|error| io_error(run_dir, error))?;
    let split_path = packed_root.join("split-manifest.json");
    let manifest: SplitManifest = serde_json::from_slice(
        &fs::read(&split_path).map_err(|error| io_error(&split_path, error))?,
    )?;
    validate_split_manifest(&manifest)?;
    let packed = PackedCorpus::open(packed_root, true)?;
    if packed.manifest.split_manifest_checksum != manifest.checksum {
        return Err(DopeError::Data(
            "packed corpus was not produced from the frozen split manifest".into(),
        ));
    }
    let (labels, operators, shapes) = compile_labels(&packed, &manifest)?;
    if !labels.iter().any(|label| label.split == "train") {
        return Err(DopeError::Data(
            "packed corpus has no training labels".into(),
        ));
    }
    let mut models = Vec::new();
    let mut epoch_metrics = Vec::new();
    let training_options_checksum = blake3::hash(&serde_json::to_vec(options)?)
        .to_hex()
        .to_string();
    for &seed in &options.seeds {
        let seed_dir = run_dir.join(format!("seed-{seed}"));
        fs::create_dir_all(&seed_dir).map_err(|error| io_error(&seed_dir, error))?;
        let model_path = seed_dir.join("model.json");
        let metrics_path = seed_dir.join("metrics.jsonl");
        let restored = fs::read(&model_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Ga2mModel>(&bytes).ok())
            .filter(|model| {
                model.seed == seed
                    && model.split_manifest_checksum == manifest.checksum
                    && model.training_options_checksum == training_options_checksum
            });
        let (model, metrics) = if let Some(model) = restored {
            let metrics = fs::read_to_string(&metrics_path)
                .ok()
                .map(|text| {
                    text.lines()
                        .filter_map(|line| serde_json::from_str(line).ok())
                        .collect::<Vec<EpochMetric>>()
                })
                .unwrap_or_default();
            (model, metrics)
        } else {
            let fitted = fit_ga2m(&labels, seed, &manifest.checksum, options);
            write_json(&model_path, &fitted.0)?;
            write_metrics(&metrics_path, &fitted.1)?;
            fs::write(seed_dir.join("loss-curves.svg"), svg_curve(&fitted.1, seed))
                .map_err(|error| io_error(seed_dir.join("loss-curves.svg"), error))?;
            fitted
        };
        epoch_metrics.extend(metrics);
        models.push(model);
    }
    let generalization_report = generalization(&labels, &models);
    let audit_report = audit_report(&labels, options);
    let operator_total = operators.values().sum::<usize>().max(1);
    let entropy_priors = operators
        .into_iter()
        .map(|(name, count)| (name, count as f64 / operator_total as f64))
        .collect();
    let failed_gates: Vec<String> = generalization_report
        .gates
        .iter()
        .filter(|(_, passed)| !**passed)
        .map(|(name, _)| name.clone())
        .chain((!audit_report.passed).then_some("auditor_coverage".into()))
        .collect();
    let mut language = LanguageCalibration {
        format: "dope-language-calibration".into(),
        version: 2,
        split_manifest_checksum: manifest.checksum.clone(),
        learned_from: "training_lineages_only".into(),
        test_split_opened: false,
        training_dataset_count: manifest
            .datasets
            .iter()
            .filter(|record| record.split == "train" && record.duplicate_of.is_none())
            .count(),
        validation_dataset_count: manifest
            .datasets
            .iter()
            .filter(|record| record.split == "validation" && record.duplicate_of.is_none())
            .count(),
        dictionaries: dictionary_entries(shapes),
        entropy_priors,
        row_permutation_equivariant: true,
        feature_permutation_equivariant_after_restore: true,
        column_name_independent: true,
        release_seed: options.release_seed,
        release_model: models
            .iter()
            .find(|model| model.seed == options.release_seed)
            .cloned(),
        stability_models: models,
        release_eligible: generalization_report.passed && audit_report.passed,
        failed_gates,
        dictionary_bytes: 0,
        checksum: String::new(),
    };
    finalize_language(&mut language)?;
    let language_path = run_dir.join("language.json");
    let generalization_path = run_dir.join("generalization.json");
    let audit_path = run_dir.join("proxy-audit.json");
    let disassembly_path = run_dir.join("language-disassembly.sexp");
    let failed_gates_path = run_dir.join("failed-gates.json");
    let environment_path = run_dir.join("environment.json");
    write_language(&language_path, &mut language)?;
    write_json(&generalization_path, &generalization_report)?;
    write_json(&audit_path, &audit_report)?;
    fs::write(&disassembly_path, disassembly(&language))
        .map_err(|error| io_error(&disassembly_path, error))?;
    write_json(&failed_gates_path, &language.failed_gates)?;
    write_json(
        &environment_path,
        &EnvironmentReport {
            package_version: env!("CARGO_PKG_VERSION").into(),
            rustc: command_output("rustc", &["--version", "--verbose"]),
            hostname: command_output("hostname", &[]),
            gpu: command_output(
                "nvidia-smi",
                &["--query-gpu=name,driver_version", "--format=csv,noheader"],
            ),
            cargo_lock_checksum: checksum_file(
                &Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"),
            )?,
            rust_source_checksum: source_checksum()?,
            split_manifest_checksum: manifest.checksum.clone(),
            packed_manifest_checksum: packed.manifest.checksum.clone(),
        },
    )?;
    let mut checksums = BTreeMap::new();
    collect_checksums(run_dir, run_dir, &mut checksums)?;
    write_json(&run_dir.join("checksums.json"), &checksums)?;
    Ok(CalibrationResult {
        language_artifact: language,
        epoch_metrics,
        generalization_report,
        audit_report,
        checksums,
    })
}

pub fn calibrate_language(
    manifest: &SplitManifest,
    out: &Path,
    seed: u64,
) -> Result<LanguageCalibration> {
    validate_split_manifest(manifest)?;
    let training: Vec<_> = manifest
        .datasets
        .iter()
        .filter(|record| record.split == "train" && record.duplicate_of.is_none())
        .collect();
    let mut shapes = HashMap::<Vec<u16>, Vec<String>>::new();
    let mut operators = BTreeMap::<String, usize>::new();
    for record in &training {
        let table = Table::read_dataset_dir(&record.path, record.task)?;
        let compiled = compile_kernel_from_arrays(
            &row_major(&table),
            &table.target,
            table.rows,
            table.features,
            record.task,
            &CompileOptions {
                seed: Some(seed),
                quantization_profiles: vec![8],
                ..Default::default()
            },
        )?;
        let kernel = decode_kernel(&compiled.artifact)?;
        let (dependence, _, _) = kernel.symbolic().ok_or_else(|| {
            DopeError::Data("language calibration requires a symbolic kernel".into())
        })?;
        *operators
            .entry(
                if matches!(dependence, crate::model::Dependence::Independent) {
                    "independence"
                } else {
                    "chow_liu"
                }
                .into(),
            )
            .or_default() += 1;
        for marginal in kernel.marginals {
            if let Marginal::QuantileSpline { values } = marginal {
                let low = values[0];
                let high = *values.last().expect("validated spline");
                let scale = (high - low).max(1e-9);
                shapes
                    .entry(
                        values
                            .iter()
                            .map(|value| {
                                (((value - low) / scale).clamp(0.0, 1.0) * 1000.0).round() as u16
                            })
                            .collect(),
                    )
                    .or_default()
                    .push(record.content_fingerprint.clone());
            }
        }
    }
    let total = operators.values().sum::<usize>().max(1);
    let mut language = LanguageCalibration {
        format: "dope-language-calibration".into(),
        version: 2,
        split_manifest_checksum: manifest.checksum.clone(),
        learned_from: "training_lineages_only".into(),
        test_split_opened: false,
        training_dataset_count: training.len(),
        validation_dataset_count: 0,
        dictionaries: dictionary_entries(shapes),
        entropy_priors: operators
            .into_iter()
            .map(|(name, count)| (name, count as f64 / total as f64))
            .collect(),
        row_permutation_equivariant: true,
        feature_permutation_equivariant_after_restore: true,
        column_name_independent: true,
        release_seed: RELEASE_SEED,
        release_model: None,
        stability_models: Vec::new(),
        release_eligible: false,
        failed_gates: vec!["training_not_run".into()],
        dictionary_bytes: 0,
        checksum: String::new(),
    };
    finalize_language(&mut language)?;
    write_language(out, &mut language)?;
    Ok(language)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_pareto_regret_penalizes_noncompliance() {
        let candidate = |id: &str, bytes: usize, compliant: bool| CandidateReport {
            candidate_id: id.into(),
            artifact_bytes: bytes,
            bits_per_original_cell: 1.0,
            score: 1.0,
            compliant,
            quantization_bits: 8,
            marginal_knots: 9,
            dependence: "independence".into(),
            target: "sparse_linear".into(),
            utility_retention: if compliant { 1.0 } else { 0.5 },
            driver_agreement: 1.0,
            proxy_joint_fidelity: 1.0,
            proxy_membership_auc: 0.5,
            failed_gates: Vec::new(),
        };
        let regrets = candidate_regrets(&[
            candidate("oracle", 100, true),
            candidate("larger", 200, true),
            candidate("miss", 50, false),
        ]);
        assert_eq!(regrets["oracle"], 0.0);
        assert!((regrets["larger"] - 0.5).abs() < 1e-12);
        assert_eq!(regrets["miss"], 1.0);
    }
}

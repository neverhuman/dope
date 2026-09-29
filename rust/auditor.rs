use crate::data::Table;
use crate::error::{DopeError, Result};
use crate::model::Task;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PermutationImportance {
    pub feature_indices: Vec<usize>,
    pub baseline_predictions: Vec<f64>,
    pub baseline_loss: f64,
    pub importances: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FeatureImportanceConsistency {
    pub feature_count: usize,
    pub informative_feature_count: usize,
    pub spearman: Option<f64>,
    pub top_k: usize,
    pub top_k_agreement: f64,
    pub real_normalized_shares: Vec<(usize, f64)>,
    pub synthetic_normalized_shares: Vec<(usize, f64)>,
    pub mean_ratio_error: Option<f64>,
}

pub trait AuditorBackend: Send + Sync {
    fn id(&self) -> &'static str;
    fn predict(&self, train: &Table, test: &Table, task: Task, seed: u64) -> Result<Vec<f64>>;

    /// Fits once and predicts an arbitrary number of schema-identical
    /// holdouts. The default concatenation is lossless because every frozen
    /// auditor's prediction is row-independent after fitting.
    fn predict_many(
        &self,
        train: &Table,
        tests: &[&Table],
        task: Task,
        seed: u64,
    ) -> Result<Vec<Vec<f64>>> {
        let combined = combine_test_tables(train, tests)?;
        let predictions = self.predict(train, &combined, task, seed)?;
        split_predictions(predictions, tests)
    }

    fn predict_two(
        &self,
        train: &Table,
        first: &Table,
        second: &Table,
        task: Task,
        seed: u64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut predictions = self.predict_many(train, &[first, second], task, seed)?;
        let second = predictions.pop().expect("two auditor predictions");
        let first = predictions.pop().expect("two auditor predictions");
        Ok((first, second))
    }

    fn permutation_importance(
        &self,
        train: &Table,
        holdout: &Table,
        task: Task,
        seed: u64,
    ) -> Result<PermutationImportance> {
        permutation_importance_for(self, train, holdout, task, seed)
    }

    fn score(&self, train: &Table, test: &Table, task: Task, seed: u64) -> Result<f64> {
        let predictions = self.predict(train, test, task, seed)?;
        loss(task, &test.target, &predictions)
    }
}

pub fn implementation_hash(id: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(include_bytes!("auditor.rs"));
    hasher.update(id.as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn combine_test_tables(train: &Table, tests: &[&Table]) -> Result<Table> {
    if tests.is_empty() {
        return Err(DopeError::Data(
            "one-fit auditor prediction requires at least one holdout".into(),
        ));
    }
    for test in tests {
        validate_tables(train, test)?;
    }
    Ok(Table {
        rows: tests.iter().map(|test| test.rows).sum(),
        features: train.features,
        columns: (0..train.features)
            .map(|feature| {
                tests
                    .iter()
                    .flat_map(|test| test.columns[feature].iter().copied())
                    .collect()
            })
            .collect(),
        target: tests
            .iter()
            .flat_map(|test| test.target.iter().copied())
            .collect(),
    })
}

fn split_predictions(predictions: Vec<f64>, tests: &[&Table]) -> Result<Vec<Vec<f64>>> {
    if predictions.len() != tests.iter().map(|test| test.rows).sum::<usize>() {
        return Err(DopeError::Data(
            "one-fit auditor returned the wrong prediction count".into(),
        ));
    }
    let mut offset = 0usize;
    Ok(tests
        .iter()
        .map(|test| {
            let next = offset + test.rows;
            let values = predictions[offset..next].to_vec();
            offset = next;
            values
        })
        .collect())
}

fn permuted_holdout(holdout: &Table, feature: usize, seed: u64) -> Table {
    let mut order = (0..holdout.rows).collect::<Vec<_>>();
    let mut random = SplitMix64(seed ^ (feature as u64).wrapping_mul(0xd6e8_feb8_6659_fd93));
    for cursor in (1..order.len()).rev() {
        let replacement = (random.next() as usize) % (cursor + 1);
        order.swap(cursor, replacement);
    }
    let mut permuted = holdout.clone();
    permuted.columns[feature] = order
        .into_iter()
        .map(|row| holdout.columns[feature][row])
        .collect();
    permuted
}

fn permutation_importance_for<B: AuditorBackend + ?Sized>(
    backend: &B,
    train: &Table,
    holdout: &Table,
    task: Task,
    seed: u64,
) -> Result<PermutationImportance> {
    validate_tables(train, holdout)?;
    let mut baseline_predictions = None;
    let mut baseline_loss = None;
    let mut importances = Vec::with_capacity(holdout.features);
    for batch in (0..holdout.features).collect::<Vec<_>>().chunks(32) {
        let mut holdouts = Vec::with_capacity(batch.len() + 1);
        holdouts.push(holdout.clone());
        holdouts.extend(
            batch
                .iter()
                .map(|&feature| permuted_holdout(holdout, feature, seed ^ 0x91e1_0da5)),
        );
        let references = holdouts.iter().collect::<Vec<_>>();
        let mut predictions = backend.predict_many(train, &references, task, seed)?;
        if predictions.len() != holdouts.len() {
            return Err(DopeError::Data(
                "permutation auditor returned the wrong holdout count".into(),
            ));
        }
        let base = predictions.remove(0);
        let current_loss = loss(task, &holdout.target, &base)?;
        if baseline_loss.is_some_and(|prior: f64| (prior - current_loss).abs() > 1e-10) {
            return Err(DopeError::Data(
                "auditor baseline changed across importance batches".into(),
            ));
        }
        baseline_predictions.get_or_insert(base);
        baseline_loss.get_or_insert(current_loss);
        for prediction in predictions {
            importances.push(loss(task, &holdout.target, &prediction)? - current_loss);
        }
    }
    Ok(PermutationImportance {
        feature_indices: (0..holdout.features).collect(),
        baseline_predictions: baseline_predictions.unwrap_or_default(),
        baseline_loss: baseline_loss.unwrap_or(0.0),
        importances,
    })
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        values[*left]
            .total_cmp(&values[*right])
            .then_with(|| left.cmp(right))
    });
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let rank = (start + end - 1) as f64 / 2.0;
        for &index in &order[start..end] {
            ranks[index] = rank;
        }
        start = end;
    }
    ranks
}

fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() < 2 || left.len() != right.len() {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let numerator = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - left_mean) * (right - right_mean))
        .sum::<f64>();
    let left_scale = left
        .iter()
        .map(|value| (value - left_mean).powi(2))
        .sum::<f64>();
    let right_scale = right
        .iter()
        .map(|value| (value - right_mean).powi(2))
        .sum::<f64>();
    let denominator = (left_scale * right_scale).sqrt();
    (denominator > f64::EPSILON).then_some(numerator / denominator)
}

pub fn compare_permutation_importance(
    real: &PermutationImportance,
    synthetic: &PermutationImportance,
) -> Result<FeatureImportanceConsistency> {
    let indexed =
        |importance: &PermutationImportance| -> Result<std::collections::BTreeMap<usize, f64>> {
            if importance.feature_indices.len() != importance.importances.len()
                || importance
                    .importances
                    .iter()
                    .any(|value| !value.is_finite())
                || !importance.baseline_loss.is_finite()
            {
                return Err(DopeError::Data(
                    "invalid permutation-importance evidence".into(),
                ));
            }
            let values = importance
                .feature_indices
                .iter()
                .copied()
                .zip(importance.importances.iter().copied())
                .collect::<std::collections::BTreeMap<_, _>>();
            if values.len() != importance.feature_indices.len() {
                return Err(DopeError::Data(
                    "permutation-importance features are not unique".into(),
                ));
            }
            Ok(values)
        };
    let real = indexed(real)?;
    let synthetic = indexed(synthetic)?;
    let common = real
        .iter()
        .filter_map(|(feature, real)| {
            synthetic
                .get(feature)
                .map(|synthetic| (*feature, *real, *synthetic))
        })
        .collect::<Vec<_>>();
    if common.is_empty() {
        return Err(DopeError::Data(
            "permutation-importance comparisons have no common features".into(),
        ));
    }
    let informative = common
        .iter()
        .filter(|(_, real, _)| *real > 1e-12)
        .copied()
        .collect::<Vec<_>>();
    let real_ranks = average_ranks(
        &informative
            .iter()
            .map(|(_, value, _)| *value)
            .collect::<Vec<_>>(),
    );
    let synthetic_ranks = average_ranks(
        &informative
            .iter()
            .map(|(_, _, value)| *value)
            .collect::<Vec<_>>(),
    );
    let top_k = informative.len().div_ceil(5).min(10);
    let top_features = |position: usize| {
        let mut values = informative.clone();
        values.sort_by(|left, right| {
            let score = |value: &(usize, f64, f64)| {
                if position == 1 { value.1 } else { value.2 }
            };
            score(right)
                .total_cmp(&score(left))
                .then_with(|| left.0.cmp(&right.0))
        });
        values
            .into_iter()
            .take(top_k)
            .map(|value| value.0)
            .collect::<std::collections::BTreeSet<_>>()
    };
    let real_top = top_features(1);
    let synthetic_top = top_features(2);
    let intersection = real_top.intersection(&synthetic_top).count();
    let union = real_top.union(&synthetic_top).count();
    let shares = |position: usize| {
        let total = common
            .iter()
            .map(|(_, real, synthetic)| {
                if position == 1 {
                    real.max(0.0)
                } else {
                    synthetic.max(0.0)
                }
            })
            .sum::<f64>();
        common
            .iter()
            .map(|(feature, real, synthetic)| {
                let value = if position == 1 { *real } else { *synthetic };
                (
                    *feature,
                    if total > 0.0 {
                        value.max(0.0) / total
                    } else {
                        0.0
                    },
                )
            })
            .collect::<Vec<_>>()
    };
    let real_shares = shares(1);
    let synthetic_shares = shares(2);
    let mean_ratio_error = (!informative.is_empty()).then(|| {
        real_shares
            .iter()
            .zip(&synthetic_shares)
            .filter(|((feature, _), _)| informative.iter().any(|(index, _, _)| index == feature))
            .map(|((_, real), (_, synthetic))| (synthetic - real).abs() / real.max(1e-12))
            .sum::<f64>()
            / informative.len() as f64
    });
    Ok(FeatureImportanceConsistency {
        feature_count: common.len(),
        informative_feature_count: informative.len(),
        spearman: pearson(&real_ranks, &synthetic_ranks),
        top_k,
        top_k_agreement: if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        },
        real_normalized_shares: real_shares,
        synthetic_normalized_shares: synthetic_shares,
        mean_ratio_error,
    })
}

struct BoundedAuditor {
    inner: Box<dyn AuditorBackend>,
}

const AUDITOR_TRAIN_ROW_LIMIT: usize = 256;
const AUDITOR_FEATURE_LIMIT: usize = usize::MAX;

impl AuditorBackend for BoundedAuditor {
    fn id(&self) -> &'static str {
        self.inner.id()
    }

    fn predict(&self, train: &Table, test: &Table, task: Task, seed: u64) -> Result<Vec<f64>> {
        Ok(self.predict_many(train, &[test], task, seed)?.remove(0))
    }

    fn predict_many(
        &self,
        train: &Table,
        tests: &[&Table],
        task: Task,
        seed: u64,
    ) -> Result<Vec<Vec<f64>>> {
        let (bounded_train, bounded_tests, _) = bounded_auditor_tables(
            train,
            tests,
            seed,
            AUDITOR_TRAIN_ROW_LIMIT,
            AUDITOR_FEATURE_LIMIT,
        );
        let bounded_refs = bounded_tests.iter().collect::<Vec<_>>();
        self.inner
            .predict_many(&bounded_train, &bounded_refs, task, seed)
    }

    fn permutation_importance(
        &self,
        train: &Table,
        holdout: &Table,
        task: Task,
        seed: u64,
    ) -> Result<PermutationImportance> {
        let (bounded_train, mut bounded_tests, selected) = bounded_auditor_tables(
            train,
            &[holdout],
            seed,
            AUDITOR_TRAIN_ROW_LIMIT,
            AUDITOR_FEATURE_LIMIT,
        );
        let mut importance = permutation_importance_for(
            self.inner.as_ref(),
            &bounded_train,
            &bounded_tests.remove(0),
            task,
            seed,
        )?;
        importance.feature_indices = importance
            .feature_indices
            .into_iter()
            .map(|feature| selected[feature])
            .collect();
        Ok(importance)
    }
}

pub fn auditor_backend(id: &str) -> Result<Box<dyn AuditorBackend>> {
    let inner: Box<dyn AuditorBackend> = match id {
        "elastic_net_glm" => Box::new(ElasticNet),
        "ga2m" => Box::new(Ga2m),
        "extremely_randomized_trees" => Box::new(ExtraTrees),
        "histogram_gbdt" => Box::new(HistogramGbdt),
        #[cfg(feature = "gpu-training")]
        "gpu_residual_mlp_ensemble" => Box::new(GpuResidualMlpEnsemble),
        #[cfg(feature = "gpu-training")]
        "gpu_tabular_feature_transformer" => Box::new(GpuFeatureTransformer),
        #[cfg(not(feature = "gpu-training"))]
        "gpu_residual_mlp_ensemble" | "gpu_tabular_feature_transformer" => {
            return Err(DopeError::Unsupported(format!(
                "frozen GPU auditor {id} requires the tch-rs/libtorch evaluation binary"
            )));
        }
        _ => return Err(DopeError::Data(format!("unknown frozen auditor {id}"))),
    };
    Ok(Box::new(BoundedAuditor { inner }))
}

#[cfg(feature = "gpu-training")]
fn gpu_error(context: &str, error: tch::TchError) -> DopeError {
    DopeError::Data(format!("{context}: {error}"))
}

#[cfg(feature = "gpu-training")]
fn gpu_device() -> Result<tch::Device> {
    if !tch::Cuda::is_available() {
        return Err(DopeError::Unsupported(
            "frozen GPU auditor requires a CUDA device".into(),
        ));
    }
    if std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref() != Ok(":4096:8") {
        return Err(DopeError::Unsupported(
            "frozen GPU auditor requires CUBLAS_WORKSPACE_CONFIG=:4096:8".into(),
        ));
    }
    Ok(tch::Device::Cuda(0))
}

#[cfg(feature = "gpu-training")]
fn deterministic_split(rows: usize, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut fit = Vec::new();
    let mut validation = Vec::new();
    let mut rng = SplitMix64(seed ^ 0x6a09_e667_f3bc_c909);
    for row in 0..rows {
        if rng.next().is_multiple_of(5) {
            validation.push(row);
        } else {
            fit.push(row);
        }
    }
    if fit.is_empty() {
        fit.push(validation.pop().unwrap_or(0));
    }
    if validation.is_empty() {
        validation.push(*fit.last().unwrap_or(&0));
    }
    (fit, validation)
}

#[cfg(feature = "gpu-training")]
fn normalized_rows(
    table: &Table,
    indices: &[usize],
    means: &[f64],
    deviations: &[f64],
) -> Vec<f32> {
    let mut output = Vec::with_capacity(indices.len() * table.features);
    for &row in indices {
        for feature in 0..table.features {
            output.push(
                ((value(table, feature, row, means) - means[feature]) / deviations[feature]) as f32,
            );
        }
    }
    output
}

#[cfg(feature = "gpu-training")]
fn gpu_tensors(
    train: &Table,
    tests: &[&Table],
    seed: u64,
    device: tch::Device,
) -> Result<(
    tch::Tensor,
    tch::Tensor,
    tch::Tensor,
    tch::Tensor,
    Vec<tch::Tensor>,
)> {
    let means = column_means(train);
    let deviations = (0..train.features)
        .map(|feature| {
            let variance = (0..train.rows)
                .map(|row| {
                    let delta = value(train, feature, row, &means) - means[feature];
                    delta * delta
                })
                .sum::<f64>()
                / train.rows.max(1) as f64;
            variance.sqrt().max(1e-5)
        })
        .collect::<Vec<_>>();
    let (fit, validation) = deterministic_split(train.rows, seed);
    let x_fit = tch::Tensor::from_slice(&normalized_rows(train, &fit, &means, &deviations))
        .view([fit.len() as i64, train.features as i64])
        .to_device(device);
    let y_fit =
        tch::Tensor::from_slice(&fit.iter().map(|&row| train.target[row]).collect::<Vec<_>>())
            .view([fit.len() as i64, 1])
            .to_device(device);
    let x_validation =
        tch::Tensor::from_slice(&normalized_rows(train, &validation, &means, &deviations))
            .view([validation.len() as i64, train.features as i64])
            .to_device(device);
    let y_validation = tch::Tensor::from_slice(
        &validation
            .iter()
            .map(|&row| train.target[row])
            .collect::<Vec<_>>(),
    )
    .view([validation.len() as i64, 1])
    .to_device(device);
    let x_tests = tests
        .iter()
        .map(|test| {
            if test.features != train.features {
                return Err(DopeError::Data(
                    "GPU auditor train/test feature widths differ".into(),
                ));
            }
            let indices = (0..test.rows).collect::<Vec<_>>();
            Ok(
                tch::Tensor::from_slice(&normalized_rows(test, &indices, &means, &deviations))
                    .view([test.rows as i64, test.features as i64])
                    .to_device(device),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((x_fit, y_fit, x_validation, y_validation, x_tests))
}

#[cfg(feature = "gpu-training")]
fn tensor_predictions(output: tch::Tensor, task: Task, rows: usize) -> Result<Vec<f64>> {
    let output = if task == Task::Binary {
        output.sigmoid()
    } else {
        output.clamp(0.0, 1.0)
    }
    .to_device(tch::Device::Cpu)
    .view([rows as i64]);
    let values = Vec::<f32>::try_from(output).map_err(|error| gpu_error("GPU output", error))?;
    Ok(values.into_iter().map(f64::from).collect())
}

#[cfg(feature = "gpu-training")]
fn ridge_head(hidden: &tch::Tensor, target: &tch::Tensor, task: Task) -> Result<tch::Tensor> {
    let rows = hidden.size()[0];
    let width = hidden.size()[1];
    let ones = tch::Tensor::ones([rows, 1], (tch::Kind::Float, hidden.device()));
    let design = tch::Tensor::cat(&[&ones, hidden], 1);
    let fitted_target = if task == Task::Binary {
        target * 4.0 - 2.0
    } else {
        target.shallow_clone()
    };
    let transpose = design.transpose(0, 1);
    let gram = transpose.matmul(&design)
        + tch::Tensor::eye(width + 1, (tch::Kind::Float, hidden.device())) * 1e-3;
    let right = transpose.matmul(&fitted_target);
    tch::Tensor::f_linalg_solve(&gram, &right, true)
        .map_err(|error| gpu_error("GPU ridge solve", error))
}

#[cfg(feature = "gpu-training")]
fn ridge_predict(hidden: &tch::Tensor, head: &tch::Tensor) -> tch::Tensor {
    let ones = tch::Tensor::ones([hidden.size()[0], 1], (tch::Kind::Float, hidden.device()));
    tch::Tensor::cat(&[&ones, hidden], 1).matmul(head)
}

#[cfg(feature = "gpu-training")]
fn transform_row_batches(
    rows: &tch::Tensor,
    batch_rows: i64,
    transform: impl Fn(&tch::Tensor) -> tch::Tensor,
) -> tch::Tensor {
    let transformed = rows
        .split(batch_rows, 0)
        .into_iter()
        .map(|batch| transform(&batch))
        .collect::<Vec<_>>();
    tch::Tensor::cat(&transformed, 0)
}

#[cfg(feature = "gpu-training")]
fn residual_mlp_member(
    train: &Table,
    tests: &[&Table],
    task: Task,
    seed: u64,
) -> Result<Vec<Vec<f64>>> {
    use tch::nn::Module;

    let device = gpu_device()?;
    tch::manual_seed(seed as i64);
    tch::Cuda::manual_seed_all(seed);
    let (x_fit, y_fit, _x_validation, _y_validation, x_tests) =
        gpu_tensors(train, tests, seed, device)?;
    let store = tch::nn::VarStore::new(device);
    let root = store.root();
    let width = 128i64;
    let input = tch::nn::linear(
        &root / "input",
        train.features as i64,
        width,
        Default::default(),
    );
    let hidden1 = tch::nn::linear(&root / "hidden1", width, width, Default::default());
    let hidden2 = tch::nn::linear(&root / "hidden2", width, width, Default::default());
    let representation = |xs: &tch::Tensor| {
        let base = input.forward(xs).relu();
        let residual = hidden2.forward(&hidden1.forward(&base).relu()).relu();
        base + residual
    };
    let head = ridge_head(&representation(&x_fit), &y_fit, task)?;
    x_tests
        .into_iter()
        .zip(tests)
        .map(|(test_tensor, test)| {
            tensor_predictions(
                ridge_predict(&representation(&test_tensor), &head),
                task,
                test.rows,
            )
        })
        .collect()
}

#[cfg(feature = "gpu-training")]
struct GpuResidualMlpEnsemble;

#[cfg(feature = "gpu-training")]
impl AuditorBackend for GpuResidualMlpEnsemble {
    fn id(&self) -> &'static str {
        "gpu_residual_mlp_ensemble"
    }

    fn predict(&self, train: &Table, test: &Table, task: Task, seed: u64) -> Result<Vec<f64>> {
        Ok(self.predict_many(train, &[test], task, seed)?.remove(0))
    }

    fn predict_many(
        &self,
        train: &Table,
        tests: &[&Table],
        task: Task,
        seed: u64,
    ) -> Result<Vec<Vec<f64>>> {
        for test in tests {
            validate_tables(train, test)?;
        }
        crate::libtorch::with_seeded_libtorch(seed, || {
            let members = (0..3)
                .map(|member| {
                    residual_mlp_member(train, tests, task, seed.wrapping_add(member * 1_000_003))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(tests
                .iter()
                .enumerate()
                .map(|(test, table)| {
                    (0..table.rows)
                        .map(|row| {
                            members.iter().map(|values| values[test][row]).sum::<f64>() / 3.0
                        })
                        .collect()
                })
                .collect())
        })
    }

    fn predict_two(
        &self,
        train: &Table,
        first: &Table,
        second: &Table,
        task: Task,
        seed: u64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut predictions = self.predict_many(train, &[first, second], task, seed)?;
        let second = predictions.pop().expect("two residual-MLP outputs");
        let first = predictions.pop().expect("two residual-MLP outputs");
        Ok((first, second))
    }
}

#[cfg(feature = "gpu-training")]
struct GpuFeatureTransformer;

fn bounded_auditor_tables(
    train: &Table,
    tests: &[&Table],
    seed: u64,
    row_limit: usize,
    feature_limit: usize,
) -> (Table, Vec<Table>, Vec<usize>) {
    debug_assert!(row_limit > 0);
    debug_assert!(feature_limit > 0);
    let fit_rows = if train.rows > row_limit {
        let mut ranked_rows = (0..train.rows)
            .map(|row| {
                let mut generator =
                    SplitMix64(seed ^ (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                (generator.next(), row)
            })
            .collect::<Vec<_>>();
        ranked_rows.sort_unstable();
        ranked_rows.truncate(row_limit);
        ranked_rows
            .into_iter()
            .map(|(_, row)| row)
            .collect::<Vec<_>>()
    } else {
        (0..train.rows).collect::<Vec<_>>()
    };
    let means = train
        .columns
        .iter()
        .map(|column| {
            let (sum, count) = fit_rows.iter().fold((0.0, 0usize), |(sum, count), &row| {
                let value = column[row];
                if value.is_finite() {
                    (sum + f64::from(value), count + 1)
                } else {
                    (sum, count)
                }
            });
            sum / count.max(1) as f64
        })
        .collect::<Vec<_>>();
    let target_mean = fit_rows
        .iter()
        .map(|&row| f64::from(train.target[row]))
        .sum::<f64>()
        / fit_rows.len().max(1) as f64;
    let mut ranked = (0..train.features)
        .map(|feature| {
            let (covariance, variance) =
                fit_rows
                    .iter()
                    .fold((0.0f64, 0.0f64), |(covariance, variance), &row| {
                        let centered = value(train, feature, row, &means) - means[feature];
                        (
                            covariance + centered * (f64::from(train.target[row]) - target_mean),
                            variance + centered * centered,
                        )
                    });
            (covariance.abs() + variance.sqrt() * 1e-9, feature)
        })
        .collect::<Vec<_>>();
    if train.features > feature_limit {
        ranked.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
    } else {
        ranked.sort_by_key(|(_, feature)| *feature);
    }
    ranked.truncate(feature_limit);
    let selected = ranked
        .into_iter()
        .map(|(_, feature)| feature)
        .collect::<Vec<_>>();
    let project_test = |table: &Table| Table {
        rows: table.rows,
        features: selected.len(),
        columns: selected
            .iter()
            .map(|&feature| table.columns[feature].clone())
            .collect(),
        target: table.target.clone(),
    };
    let bounded_train = Table {
        rows: fit_rows.len(),
        features: selected.len(),
        columns: selected
            .iter()
            .map(|&feature| {
                fit_rows
                    .iter()
                    .map(|&row| train.columns[feature][row])
                    .collect()
            })
            .collect(),
        target: fit_rows.iter().map(|&row| train.target[row]).collect(),
    };
    (
        bounded_train,
        tests.iter().map(|table| project_test(table)).collect(),
        selected,
    )
}

#[cfg(feature = "gpu-training")]
fn feature_transformer_predictions(
    train: &Table,
    tests: &[&Table],
    task: Task,
    seed: u64,
) -> Result<Vec<Vec<f64>>> {
    use tch::nn::Module;

    let device = gpu_device()?;
    crate::libtorch::with_seeded_libtorch(seed, || {
        let (x_fit, y_fit, _x_validation, _y_validation, x_tests) =
            gpu_tensors(train, tests, seed, device)?;
        let store = tch::nn::VarStore::new(device);
        let root = store.root();
        let embedding = 32i64;
        let token = tch::nn::linear(&root / "token", 1, embedding, Default::default());
        let feature_embedding = root.var(
            "feature_embedding",
            &[1, train.features as i64, embedding],
            tch::nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );
        let query = root.var(
            "query",
            &[1, 1, embedding],
            tch::nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );
        let keys = tch::nn::linear(&root / "keys", embedding, embedding, Default::default());
        let values = tch::nn::linear(&root / "values", embedding, embedding, Default::default());
        let feed1 = tch::nn::linear(&root / "feed1", embedding, 64, Default::default());
        let feed2 = tch::nn::linear(&root / "feed2", 64, embedding, Default::default());
        let representation = |xs: &tch::Tensor| {
            let tokens = (token.forward(&xs.unsqueeze(-1)) + &feature_embedding).relu();
            let key = keys.forward(&tokens);
            let attention =
                ((key * &query).sum_dim_intlist([-1].as_slice(), false, tch::Kind::Float)
                    / (embedding as f64).sqrt())
                .softmax(-1, tch::Kind::Float);
            let pooled = (values.forward(&tokens) * attention.unsqueeze(-1)).sum_dim_intlist(
                [1].as_slice(),
                false,
                tch::Kind::Float,
            );
            let transformed = feed2.forward(&feed1.forward(&pooled).gelu("none"));
            pooled + transformed
        };
        // Wide tables otherwise materialize rows × features × embedding for the
        // entire table. Batching only the row-independent representation keeps
        // exact model weights and row order while bounding CUDA working memory.
        const ROW_BATCH: i64 = 128;
        let batched_representation =
            |xs: &tch::Tensor| transform_row_batches(xs, ROW_BATCH, representation);
        let head = ridge_head(&batched_representation(&x_fit), &y_fit, task)?;
        x_tests
            .into_iter()
            .zip(tests)
            .map(|(test_tensor, test)| {
                tensor_predictions(
                    ridge_predict(&batched_representation(&test_tensor), &head),
                    task,
                    test.rows,
                )
            })
            .collect()
    })
}

#[cfg(feature = "gpu-training")]
impl AuditorBackend for GpuFeatureTransformer {
    fn id(&self) -> &'static str {
        "gpu_tabular_feature_transformer"
    }

    fn predict(&self, train: &Table, test: &Table, task: Task, seed: u64) -> Result<Vec<f64>> {
        Ok(self.predict_many(train, &[test], task, seed)?.remove(0))
    }

    fn predict_many(
        &self,
        train: &Table,
        tests: &[&Table],
        task: Task,
        seed: u64,
    ) -> Result<Vec<Vec<f64>>> {
        for test in tests {
            validate_tables(train, test)?;
        }
        feature_transformer_predictions(train, tests, task, seed)
    }

    fn predict_two(
        &self,
        train: &Table,
        first: &Table,
        second: &Table,
        task: Task,
        seed: u64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        validate_tables(train, first)?;
        validate_tables(train, second)?;
        let mut predictions = self.predict_many(train, &[first, second], task, seed)?;
        let second = predictions.pop().expect("two feature-transformer outputs");
        let first = predictions.pop().expect("two feature-transformer outputs");
        Ok((first, second))
    }
}

fn validate_tables(train: &Table, test: &Table) -> Result<()> {
    if train.rows == 0 || test.rows == 0 || train.features != test.features {
        return Err(DopeError::Data(
            "auditor train/test tables must be nonempty and schema-aligned".into(),
        ));
    }
    Ok(())
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value.clamp(-30.0, 30.0)).exp())
}

fn loss(task: Task, target: &[f32], predictions: &[f64]) -> Result<f64> {
    if target.len() != predictions.len() || target.is_empty() {
        return Err(DopeError::Data("auditor prediction shape mismatch".into()));
    }
    Ok(match task {
        Task::Regression => {
            target
                .iter()
                .zip(predictions)
                .map(|(&target, &prediction)| (f64::from(target) - prediction).powi(2))
                .sum::<f64>()
                / target.len() as f64
        }
        Task::Binary => {
            target
                .iter()
                .zip(predictions)
                .map(|(&target, &prediction)| {
                    let probability = prediction.clamp(1e-7, 1.0 - 1e-7);
                    -f64::from(target) * probability.ln()
                        - (1.0 - f64::from(target)) * (1.0 - probability).ln()
                })
                .sum::<f64>()
                / target.len() as f64
        }
    })
}

fn column_means(table: &Table) -> Vec<f64> {
    table
        .columns
        .iter()
        .map(|column| {
            let observed = column.iter().copied().filter(|value| value.is_finite());
            let (sum, count) = observed.fold((0.0, 0usize), |(sum, count), value| {
                (sum + f64::from(value), count + 1)
            });
            sum / count.max(1) as f64
        })
        .collect()
}

fn value(table: &Table, feature: usize, row: usize, means: &[f64]) -> f64 {
    let value = table.columns[feature][row];
    if value.is_finite() {
        f64::from(value)
    } else {
        means[feature]
    }
}

struct ElasticNet;

impl AuditorBackend for ElasticNet {
    fn id(&self) -> &'static str {
        "elastic_net_glm"
    }

    #[allow(clippy::needless_range_loop)]
    fn predict(&self, train: &Table, test: &Table, task: Task, _seed: u64) -> Result<Vec<f64>> {
        validate_tables(train, test)?;
        let means = column_means(train);
        let target_mean = train
            .target
            .iter()
            .map(|&value| f64::from(value))
            .sum::<f64>()
            / train.rows as f64;
        let mut intercept = if task == Task::Binary {
            (target_mean.clamp(1e-5, 1.0 - 1e-5) / (1.0 - target_mean.clamp(1e-5, 1.0 - 1e-5))).ln()
        } else {
            target_mean
        };
        let mut weights = vec![0.0f64; train.features];
        let train_values = (0..train.rows)
            .map(|row| {
                (0..train.features)
                    .map(|feature| value(train, feature, row, &means))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let learning_rate = 0.15 / (train.features.max(1) as f64).sqrt();
        for _ in 0..240 {
            let mut intercept_gradient = 0.0;
            let mut gradients = vec![0.0; train.features];
            for row in 0..train.rows {
                let linear = intercept
                    + weights
                        .iter()
                        .zip(&train_values[row])
                        .map(|(weight, value)| weight * value)
                        .sum::<f64>();
                let prediction = if task == Task::Binary {
                    sigmoid(linear)
                } else {
                    linear
                };
                let residual = prediction - f64::from(train.target[row]);
                intercept_gradient += residual;
                for feature in 0..train.features {
                    gradients[feature] += residual * train_values[row][feature];
                }
            }
            intercept -= learning_rate * intercept_gradient / train.rows as f64;
            for feature in 0..train.features {
                let gradient = gradients[feature] / train.rows as f64 + 1e-3 * weights[feature];
                weights[feature] -= learning_rate * gradient;
                let shrinkage = learning_rate * 1e-4;
                weights[feature] =
                    weights[feature].signum() * (weights[feature].abs() - shrinkage).max(0.0);
            }
        }
        Ok((0..test.rows)
            .map(|row| {
                let linear = intercept
                    + (0..test.features)
                        .map(|feature| weights[feature] * value(test, feature, row, &means))
                        .sum::<f64>();
                if task == Task::Binary {
                    sigmoid(linear)
                } else {
                    linear.clamp(0.0, 1.0)
                }
            })
            .collect())
    }
}

fn bin(value: f64) -> usize {
    (value.clamp(0.0, 1.0) * 8.0).floor().min(7.0) as usize
}

struct Ga2m;

impl AuditorBackend for Ga2m {
    fn id(&self) -> &'static str {
        "ga2m"
    }

    #[allow(clippy::needless_range_loop)]
    fn predict(&self, train: &Table, test: &Table, task: Task, _seed: u64) -> Result<Vec<f64>> {
        validate_tables(train, test)?;
        let means = column_means(train);
        let mean = train
            .target
            .iter()
            .map(|&value| f64::from(value))
            .sum::<f64>()
            / train.rows as f64;
        let intercept = if task == Task::Binary {
            (mean.clamp(1e-5, 1.0 - 1e-5) / (1.0 - mean.clamp(1e-5, 1.0 - 1e-5))).ln()
        } else {
            mean
        };
        let mut main = vec![[0.0f64; 8]; train.features];
        let pairs = (0..train.features.saturating_sub(1).min(8))
            .map(|feature| (feature, feature + 1))
            .collect::<Vec<_>>();
        let train_bins = (0..train.features)
            .map(|feature| {
                (0..train.rows)
                    .map(|row| bin(value(train, feature, row, &means)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut interactions = vec![[0.0f64; 64]; pairs.len()];
        let mut linear = vec![intercept; train.rows];
        for _ in 0..12 {
            for (feature, feature_bins) in train_bins.iter().enumerate() {
                let mut sums = [0.0; 8];
                let mut counts = [0usize; 8];
                for row in 0..train.rows {
                    let slot = feature_bins[row];
                    let prediction = if task == Task::Binary {
                        sigmoid(linear[row])
                    } else {
                        linear[row]
                    };
                    sums[slot] += f64::from(train.target[row]) - prediction;
                    counts[slot] += 1;
                }
                let mut updates = [0.0; 8];
                for slot in 0..8 {
                    let update = 0.35 * sums[slot] / counts[slot].max(1) as f64;
                    main[feature][slot] += update;
                    updates[slot] = update;
                }
                for row in 0..train.rows {
                    linear[row] += updates[train_bins[feature][row]];
                }
            }
            for (pair_index, &(left, right)) in pairs.iter().enumerate() {
                let mut sums = [0.0; 64];
                let mut counts = [0usize; 64];
                for row in 0..train.rows {
                    let slot = train_bins[left][row] * 8 + train_bins[right][row];
                    let prediction = if task == Task::Binary {
                        sigmoid(linear[row])
                    } else {
                        linear[row]
                    };
                    sums[slot] += f64::from(train.target[row]) - prediction;
                    counts[slot] += 1;
                }
                for slot in 0..64 {
                    interactions[pair_index][slot] += 0.2 * sums[slot] / counts[slot].max(1) as f64;
                }
                for row in 0..train.rows {
                    let slot = train_bins[left][row] * 8 + train_bins[right][row];
                    linear[row] += 0.2 * sums[slot] / counts[slot].max(1) as f64;
                }
            }
        }
        Ok((0..test.rows)
            .map(|row| {
                let mut linear = intercept;
                for feature in 0..test.features {
                    linear += main[feature][bin(value(test, feature, row, &means))];
                }
                for (pair_index, &(left, right)) in pairs.iter().enumerate() {
                    let slot = bin(value(test, left, row, &means)) * 8
                        + bin(value(test, right, row, &means));
                    linear += interactions[pair_index][slot];
                }
                if task == Task::Binary {
                    sigmoid(linear)
                } else {
                    linear.clamp(0.0, 1.0)
                }
            })
            .collect())
    }
}

#[derive(Clone, Copy)]
struct Stump {
    feature: usize,
    threshold: f64,
    left: f64,
    right: f64,
}

impl Stump {
    fn predict(self, value: f64) -> f64 {
        if value < self.threshold {
            self.left
        } else {
            self.right
        }
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
}

fn fit_stump(
    table: &Table,
    means: &[f64],
    targets: &[f64],
    feature: usize,
    threshold: f64,
) -> Stump {
    let mut sums = [0.0; 2];
    let mut counts = [0usize; 2];
    for (row, &target) in targets.iter().enumerate().take(table.rows) {
        let side = usize::from(value(table, feature, row, means) > threshold);
        sums[side] += target;
        counts[side] += 1;
    }
    let fallback = targets.iter().sum::<f64>() / targets.len().max(1) as f64;
    Stump {
        feature,
        threshold,
        left: if counts[0] == 0 {
            fallback
        } else {
            sums[0] / counts[0] as f64
        },
        right: if counts[1] == 0 {
            fallback
        } else {
            sums[1] / counts[1] as f64
        },
    }
}

struct ExtraTrees;

impl AuditorBackend for ExtraTrees {
    fn id(&self) -> &'static str {
        "extremely_randomized_trees"
    }

    fn predict(&self, train: &Table, test: &Table, task: Task, seed: u64) -> Result<Vec<f64>> {
        validate_tables(train, test)?;
        let means = column_means(train);
        let targets = train
            .target
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>();
        let mut rng = SplitMix64(seed);
        let stumps = (0..192)
            .map(|_| {
                let feature = rng.next() as usize % train.features;
                let threshold = (rng.next() >> 11) as f64 / ((1u64 << 53) as f64);
                fit_stump(train, &means, &targets, feature, threshold)
            })
            .collect::<Vec<_>>();
        Ok((0..test.rows)
            .map(|row| {
                let prediction = stumps
                    .iter()
                    .map(|stump| stump.predict(value(test, stump.feature, row, &means)))
                    .sum::<f64>()
                    / stumps.len() as f64;
                if task == Task::Binary {
                    prediction.clamp(1e-7, 1.0 - 1e-7)
                } else {
                    prediction.clamp(0.0, 1.0)
                }
            })
            .collect())
    }
}

struct HistogramGbdt;

impl AuditorBackend for HistogramGbdt {
    fn id(&self) -> &'static str {
        "histogram_gbdt"
    }

    fn predict(&self, train: &Table, test: &Table, task: Task, _seed: u64) -> Result<Vec<f64>> {
        validate_tables(train, test)?;
        let means = column_means(train);
        let mean = train
            .target
            .iter()
            .map(|&value| f64::from(value))
            .sum::<f64>()
            / train.rows as f64;
        let intercept = if task == Task::Binary {
            (mean.clamp(1e-5, 1.0 - 1e-5) / (1.0 - mean.clamp(1e-5, 1.0 - 1e-5))).ln()
        } else {
            mean
        };
        let train_bins = (0..train.features)
            .map(|feature| {
                (0..train.rows)
                    .map(|row| {
                        (value(train, feature, row, &means).clamp(0.0, 1.0) * 16.0)
                            .floor()
                            .min(15.0) as usize
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut train_linear = vec![intercept; train.rows];
        let mut stumps = Vec::new();
        for _ in 0..96 {
            let residuals = (0..train.rows)
                .map(|row| {
                    f64::from(train.target[row])
                        - if task == Task::Binary {
                            sigmoid(train_linear[row])
                        } else {
                            train_linear[row]
                        }
                })
                .collect::<Vec<_>>();
            let mut best = None::<(f64, Stump)>;
            for (feature, feature_bins) in train_bins.iter().enumerate() {
                let mut sums = [0.0f64; 16];
                let mut squares = [0.0f64; 16];
                let mut counts = [0usize; 16];
                for (row, residual) in residuals.iter().copied().enumerate() {
                    let slot = feature_bins[row];
                    sums[slot] += residual;
                    squares[slot] += residual * residual;
                    counts[slot] += 1;
                }
                let total_sum = sums.iter().sum::<f64>();
                let total_square = squares.iter().sum::<f64>();
                let mut left_sum = 0.0;
                let mut left_square = 0.0;
                let mut left_count = 0usize;
                for threshold_bin in 1..16 {
                    let slot = threshold_bin - 1;
                    left_sum += sums[slot];
                    left_square += squares[slot];
                    left_count += counts[slot];
                    let right_sum = total_sum - left_sum;
                    let right_square = total_square - left_square;
                    let right_count = train.rows - left_count;
                    let left_mean = left_sum / left_count.max(1) as f64;
                    let right_mean = right_sum / right_count.max(1) as f64;
                    let error = left_square - left_sum * left_sum / left_count.max(1) as f64
                        + right_square
                        - right_sum * right_sum / right_count.max(1) as f64;
                    let stump = Stump {
                        feature,
                        threshold: threshold_bin as f64 / 16.0,
                        left: left_mean,
                        right: right_mean,
                    };
                    if best
                        .as_ref()
                        .is_none_or(|(best_error, _)| error < *best_error)
                    {
                        best = Some((error, stump));
                    }
                }
            }
            let mut stump = best.expect("nonempty feature search").1;
            stump.left *= 0.08;
            stump.right *= 0.08;
            for (row, linear) in train_linear.iter_mut().enumerate() {
                *linear += stump.predict(value(train, stump.feature, row, &means));
            }
            stumps.push(stump);
        }
        Ok((0..test.rows)
            .map(|row| {
                let linear = intercept
                    + stumps
                        .iter()
                        .map(|stump| stump.predict(value(test, stump.feature, row, &means)))
                        .sum::<f64>();
                if task == Task::Binary {
                    sigmoid(linear)
                } else {
                    linear.clamp(0.0, 1.0)
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[cfg(feature = "gpu-training")]
    static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn fixtures(task: Task) -> (Table, Table) {
        let rows = 64;
        let features = (0..rows)
            .flat_map(|row| {
                let x = row as f32 / (rows - 1) as f32;
                [x, 1.0 - x]
            })
            .collect::<Vec<_>>();
        let target = (0..rows)
            .map(|row| {
                let x = row as f32 / (rows - 1) as f32;
                if task == Task::Binary {
                    f32::from(x > 0.5)
                } else {
                    (0.8 * x + 0.1).clamp(0.0, 1.0)
                }
            })
            .collect::<Vec<_>>();
        let table = Table::from_arrays(&features, &target, rows, 2, task).unwrap();
        (table.clone(), table)
    }

    #[test]
    fn native_auditors_are_deterministic_and_beat_null_fixtures() {
        for task in [Task::Binary, Task::Regression] {
            let (train, test) = fixtures(task);
            let mean = train
                .target
                .iter()
                .map(|&value| f64::from(value))
                .sum::<f64>()
                / train.rows as f64;
            let null = loss(task, &test.target, &vec![mean; test.rows]).unwrap();
            for id in [
                "elastic_net_glm",
                "ga2m",
                "extremely_randomized_trees",
                "histogram_gbdt",
            ] {
                let backend = auditor_backend(id).unwrap();
                let first = backend.predict(&train, &test, task, 57721).unwrap();
                assert_eq!(first, backend.predict(&train, &test, task, 57721).unwrap());
                let paired = backend
                    .predict_two(&train, &test, &test, task, 57721)
                    .unwrap();
                assert_eq!(paired.0, paired.1, "{id} paired prediction mismatch");
                assert_eq!(first, paired.0, "{id} paired fit changed output");
                assert!(
                    loss(task, &test.target, &first).unwrap() < null,
                    "{id} {task:?}"
                );
                assert_eq!(implementation_hash(id).len(), 64);
            }
        }
    }

    struct CountingAuditor(AtomicUsize);

    impl AuditorBackend for CountingAuditor {
        fn id(&self) -> &'static str {
            "counting_fixture"
        }

        fn predict(
            &self,
            _train: &Table,
            test: &Table,
            _task: Task,
            _seed: u64,
        ) -> Result<Vec<f64>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(test.columns[0]
                .iter()
                .map(|value| f64::from(*value))
                .collect())
        }
    }

    #[test]
    fn permutation_importance_fits_once_for_all_holdouts() {
        let (train, test) = fixtures(Task::Regression);
        let auditor = CountingAuditor(AtomicUsize::new(0));
        let first = auditor
            .permutation_importance(&train, &test, Task::Regression, 57_721)
            .unwrap();
        assert_eq!(auditor.0.load(Ordering::SeqCst), 1);
        assert_eq!(first.feature_indices, vec![0, 1]);
        assert_eq!(first.importances.len(), 2);
        assert!(first.importances[0] > first.importances[1]);
        let second = auditor
            .permutation_importance(&train, &test, Task::Regression, 57_721)
            .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn importance_comparison_preserves_undefined_spearman() {
        let evidence = |importances: Vec<f64>| PermutationImportance {
            feature_indices: vec![0, 1, 2],
            baseline_predictions: vec![0.5],
            baseline_loss: 0.25,
            importances,
        };
        let informative = compare_permutation_importance(
            &evidence(vec![3.0, 2.0, 1.0]),
            &evidence(vec![6.0, 4.0, 2.0]),
        )
        .unwrap();
        assert_eq!(informative.spearman, Some(1.0));
        assert_eq!(informative.top_k, 1);
        assert_eq!(informative.top_k_agreement, 1.0);
        assert!(
            (informative
                .real_normalized_shares
                .iter()
                .map(|(_, share)| share)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-12
        );
        let changed_driver = compare_permutation_importance(
            &evidence(vec![3.0, 2.0, 1.0]),
            &evidence(vec![1.0, 2.0, 3.0]),
        )
        .unwrap();
        assert_eq!(changed_driver.top_k_agreement, 0.0);
        assert_eq!(changed_driver.spearman, Some(-1.0));

        let undefined = compare_permutation_importance(
            &evidence(vec![1.0, 1.0, 1.0]),
            &evidence(vec![2.0, 2.0, 2.0]),
        )
        .unwrap();
        assert_eq!(undefined.informative_feature_count, 3);
        assert_eq!(undefined.spearman, None);
    }

    #[test]
    fn importance_covers_high_index_features() {
        struct HighIndex;
        impl AuditorBackend for HighIndex {
            fn id(&self) -> &'static str {
                "high_index_fixture"
            }
            fn predict(
                &self,
                _train: &Table,
                test: &Table,
                _task: Task,
                _seed: u64,
            ) -> Result<Vec<f64>> {
                Ok(test.columns[69]
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect())
            }
        }
        let rows = 32;
        let mut values = vec![0.0; rows * 70];
        let mut target = Vec::new();
        for row in 0..rows {
            let value = row as f32 / (rows - 1) as f32;
            values[row * 70 + 69] = value;
            target.push(value);
        }
        let table = Table::from_arrays(&values, &target, rows, 70, Task::Regression).unwrap();
        let auditor = BoundedAuditor {
            inner: Box::new(HighIndex),
        };
        let importance = auditor
            .permutation_importance(&table, &table, Task::Regression, 57721)
            .unwrap();
        assert_eq!(importance.feature_indices.len(), 70);
        assert_eq!(importance.feature_indices[69], 69);
        assert!(importance.importances[69] > importance.importances[0]);
    }

    #[cfg(feature = "gpu-training")]
    #[test]
    fn gpu_feature_preprocessing_is_deterministic_and_bounded() {
        let rows = 8;
        let features = 300;
        let values = (0..rows * features)
            .map(|index| ((index * 37) % 101) as f32 / 100.0)
            .collect::<Vec<_>>();
        let target = (0..rows)
            .map(|row| row as f32 / (rows - 1) as f32)
            .collect::<Vec<_>>();
        let table = Table::from_arrays(&values, &target, rows, features, Task::Regression).unwrap();
        let first = bounded_auditor_tables(&table, &[&table], 57721, 256, 64);
        let second = bounded_auditor_tables(&table, &[&table], 57721, 256, 64);
        assert_eq!(first.0.features, 64);
        assert_eq!(first.0.columns, second.0.columns);
        assert_eq!(first.0.columns, first.1[0].columns);

        let tall_rows = 1_100;
        let tall_values = (0..tall_rows * 2)
            .map(|index| (index % 97) as f32 / 96.0)
            .collect::<Vec<_>>();
        let tall_target = (0..tall_rows)
            .map(|row| (row % 101) as f32 / 100.0)
            .collect::<Vec<_>>();
        let tall =
            Table::from_arrays(&tall_values, &tall_target, tall_rows, 2, Task::Regression).unwrap();
        let bounded = bounded_auditor_tables(&tall, &[&tall], 57721, 256, 64);
        assert_eq!(bounded.0.rows, 256);
        assert_eq!(bounded.1[0].rows, tall_rows);
    }

    #[cfg(feature = "gpu-training")]
    #[test]
    fn gpu_row_batching_preserves_order_and_values() {
        let _guard = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let device = gpu_device().unwrap();
        let rows = tch::Tensor::arange(1025, (tch::Kind::Float, device)).view([205, 5]);
        let expected = &rows * 1.25 + 3.0;
        let actual = transform_row_batches(&rows, 17, |batch| batch * 1.25 + 3.0);
        let difference = f64::try_from((expected - actual).abs().max()).unwrap();
        assert_eq!(difference, 0.0);
    }

    #[cfg(feature = "gpu-training")]
    #[test]
    fn gpu_auditors_train_on_binary_and_regression_fixtures() {
        let _guard = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            tch::Cuda::is_available(),
            "frozen CUDA runtime is unavailable"
        );
        for task in [Task::Binary, Task::Regression] {
            let (train, test) = fixtures(task);
            let mean = train
                .target
                .iter()
                .map(|&value| f64::from(value))
                .sum::<f64>()
                / train.rows as f64;
            let null = loss(task, &test.target, &vec![mean; test.rows]).unwrap();
            for id in [
                "gpu_residual_mlp_ensemble",
                "gpu_tabular_feature_transformer",
            ] {
                let prediction = auditor_backend(id)
                    .unwrap()
                    .predict(&train, &test, task, 57721)
                    .unwrap();
                let paired = auditor_backend(id)
                    .unwrap()
                    .predict_two(&train, &test, &test, task, 57721)
                    .unwrap();
                assert_eq!(paired.0, paired.1, "{id} paired prediction mismatch");
                assert_eq!(prediction, paired.0, "{id} paired fit changed output");
                assert!(
                    loss(task, &test.target, &prediction).unwrap() < null,
                    "{id} {task:?}"
                );
                assert_eq!(implementation_hash(id).len(), 64);
            }
        }
    }
}

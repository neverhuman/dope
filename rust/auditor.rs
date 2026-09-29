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
    hasher.update(include_bytes!("auditor/auditor_backends.rs"));
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

include!("auditor/auditor_backends.rs");

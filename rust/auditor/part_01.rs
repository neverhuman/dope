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
    hasher.update(include_bytes!("../auditor.rs"));
    hasher.update(include_bytes!("auditor_backends.rs"));
    // Include every moved auditor source section in the frozen auditor identity.
    hasher.update(include_bytes!("auditor_backends/part_01.rs"));
    hasher.update(include_bytes!("auditor_backends/part_02.rs"));
    hasher.update(include_bytes!("part_01.rs"));
    hasher.update(include_bytes!("part_02.rs"));
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
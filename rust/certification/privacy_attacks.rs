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

include!("privacy_attacks/certify.rs");

/// Scores one mapped triple without reading a path. `certify_kernel` is not used:
/// that entry requires an evaluator-owned `test.csv`.
pub fn score_holdout_tables(
    fit: &crate::data::Table,
    synthetic: &crate::data::Table,
    holdout: &crate::data::Table,
    generation_seed: u64,
) -> HoldoutTableScores {
    let feature_attack = membership_auc(synthetic, fit, holdout);
    let joint_attack = membership_auc(
        &joint_table(synthetic),
        &joint_table(fit),
        &joint_table(holdout),
    );
    let (marginal, dependence) = fidelity_components(fit, synthetic);
    let diagnostics = evaluate_fitness_diagnostics(holdout, synthetic, generation_seed);
    HoldoutTableScores {
        membership_auc: feature_attack.max(joint_attack),
        attribute_inference_advantage: attribute_inference_advantage(synthetic, fit, holdout),
        marginal_fidelity: marginal,
        dependence_fidelity: dependence,
        sliced_wasserstein_fidelity: diagnostics
            .as_ref()
            .map(|item| item.sliced_wasserstein_fidelity),
        mmd_fidelity: diagnostics.as_ref().map(|item| item.mmd_fidelity),
        coverage_realism: diagnostics.as_ref().map(|item| item.coverage_realism),
        driver_fidelity: driver_fidelity(fit, synthetic),
        driver_importance: "absolute_target_correlation",
    }
}

/// One-sided 95% lower bound used as the utility component before clamping.
pub fn retention_lower_bound(values: &[f64]) -> Option<f64> {
    crate::production::one_sided_lower(values, 0.95)
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct HoldoutTableScores {
    pub membership_auc: f64,
    pub attribute_inference_advantage: f64,
    pub marginal_fidelity: f64,
    pub dependence_fidelity: f64,
    pub sliced_wasserstein_fidelity: Option<f64>,
    pub mmd_fidelity: Option<f64>,
    pub coverage_realism: Option<f64>,
    pub driver_fidelity: Option<f64>,
    pub driver_importance: &'static str,
}

fn driver_fidelity(fit: &crate::data::Table, synthetic: &crate::data::Table) -> Option<f64> {
    if fit.features == 0 || fit.features != synthetic.features {
        return None;
    }
    let real: Vec<(usize, f64)> = (0..fit.features)
        .map(|column| {
            (
                column,
                correlation(&completed(fit, column), &fit.target).abs(),
            )
        })
        .filter(|(_, value)| value.is_finite() && *value > 1e-12)
        .collect();
    if real.len() < 2 {
        return None;
    }
    let mut paired = Vec::new();
    for (column, real_score) in real {
        let synthetic_score = correlation(&completed(synthetic, column), &synthetic.target).abs();
        if synthetic_score.is_finite() {
            paired.push((column, real_score, synthetic_score));
        }
    }
    if paired.len() < 2 {
        return None;
    }
    let real_ranks = average_rank_values(
        &paired
            .iter()
            .map(|(_, value, _)| *value)
            .collect::<Vec<_>>(),
    );
    let synthetic_ranks = average_rank_values(
        &paired
            .iter()
            .map(|(_, _, value)| *value)
            .collect::<Vec<_>>(),
    );
    let spearman = rank_correlation(&real_ranks, &synthetic_ranks)?;
    let top_k = paired.len().div_ceil(5).min(10);
    let top = |position: usize| {
        let mut values = paired.clone();
        values.sort_by(|left, right| {
            let score = |row: &(usize, f64, f64)| if position == 1 { row.1 } else { row.2 };
            score(right)
                .total_cmp(&score(left))
                .then_with(|| left.0.cmp(&right.0))
        });
        values
            .into_iter()
            .take(top_k)
            .map(|row| row.0)
            .collect::<std::collections::BTreeSet<_>>()
    };
    let real_top = top(1);
    let synthetic_top = top(2);
    let union = real_top.union(&synthetic_top).count();
    if union == 0 {
        return None;
    }
    let overlap = real_top.intersection(&synthetic_top).count() as f64 / union as f64;
    let rank = ((spearman + 1.0) / 2.0).clamp(0.0, 1.0);
    Some(0.5 * rank + 0.5 * overlap.clamp(0.0, 1.0))
}

fn average_rank_values(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| {
        values[left]
            .total_cmp(&values[right])
            .then_with(|| left.cmp(&right))
    });
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let rank = (start + end - 1) as f64 / 2.0 + 1.0;
        for index in &order[start..end] {
            ranks[*index] = rank;
        }
        start = end;
    }
    ranks
}

fn rank_correlation(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut numerator = 0.0;
    let mut left_scale = 0.0;
    let mut right_scale = 0.0;
    for (left, right) in left.iter().zip(right) {
        let left_delta = left - left_mean;
        let right_delta = right - right_mean;
        numerator += left_delta * right_delta;
        left_scale += left_delta * left_delta;
        right_scale += right_delta * right_delta;
    }
    let denominator = (left_scale * right_scale).sqrt();
    (denominator > f64::EPSILON).then_some(numerator / denominator)
}

#[cfg(test)]
include!("privacy_attacks/tests.rs");

#[cfg(test)]
mod holdout_table_score {
    use super::*;
    use crate::data::Table;
    use crate::model::Task;

    fn regression(columns: &[f32], target: &[f32], rows: usize) -> Table {
        Table::from_arrays(columns, target, rows, 1, Task::Regression).unwrap()
    }

    #[test]
    fn identical_fit_and_synthetic_keep_full_marginal_fidelity() {
        let column = [0.05, 0.15, 0.25, 0.35, 0.55, 0.65, 0.75, 0.85];
        let target = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80];
        let fit = regression(&column, &target, 8);
        let holdout = regression(
            &[0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82],
            &[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
            8,
        );
        let scores = score_holdout_tables(&fit, &fit, &holdout, 11);
        assert!((scores.marginal_fidelity - 1.0).abs() < 1e-9);
        assert!((scores.dependence_fidelity - 1.0).abs() < 1e-9);
        assert!(scores.membership_auc.is_finite());
        assert!(scores.attribute_inference_advantage.is_finite());
        assert_eq!(scores.driver_importance, "absolute_target_correlation");
        assert!(scores.sliced_wasserstein_fidelity.is_some());
        let lower = retention_lower_bound(&[0.2, 0.5, 0.8]).unwrap();
        assert!((lower + 0.0057563382544248975).abs() < 1e-9);
    }
}

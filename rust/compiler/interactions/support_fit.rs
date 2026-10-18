fn joint_mse(
    matrix: &ColumnMatrix,
    labels: &[u8],
    fold_count: u8,
    terms: &[InteractionProposal],
) -> f64 {
    let mut squared = 0.0;
    let mut count = 0usize;
    for fold in 0..fold_count {
        let train = rows_in(labels, fold, false);
        let eval = rows_in(labels, fold, true);
        if train.is_empty() || eval.is_empty() {
            continue;
        }
        let mut residual = gather_target(matrix, &train);
        let mean = mean(&residual);
        for value in &mut residual {
            *value -= mean;
        }
        let mut eval_prediction = vec![mean; eval.len()];
        for term in terms {
            let train_feature = gather(matrix, term.kind, term.left, term.right, &train);
            let (intercept, slope) = fit_line(&train_feature, &residual);
            for slot in 0..train.len() {
                residual[slot] -= intercept + slope * train_feature[slot];
            }
            let eval_feature = gather(matrix, term.kind, term.left, term.right, &eval);
            for (slot, feature) in eval_feature.iter().copied().enumerate() {
                eval_prediction[slot] += intercept + slope * feature;
            }
        }
        for (slot, row) in eval.iter().copied().enumerate() {
            let gap = eval_prediction[slot] - f64::from(matrix.target[row]);
            squared += gap * gap;
            count += 1;
        }
    }
    if count == 0 {
        f64::MAX
    } else {
        squared / count as f64
    }
}

fn rows_in(labels: &[u8], fold: u8, eval: bool) -> Vec<usize> {
    labels
        .iter()
        .enumerate()
        .filter_map(|(row, label)| ((*label == fold) == eval).then_some(row))
        .collect()
}

fn gather(
    matrix: &ColumnMatrix,
    kind: ProposalKind,
    left: u16,
    right: Option<u16>,
    rows: &[usize],
) -> Vec<f64> {
    rows.iter()
        .copied()
        .map(|row| feature_at(matrix, kind, left, right, row))
        .collect()
}

fn feature_at(
    matrix: &ColumnMatrix,
    kind: ProposalKind,
    left: u16,
    right: Option<u16>,
    row: usize,
) -> f64 {
    let left_value = f64::from(matrix.columns[usize::from(left)][row]);
    match kind {
        ProposalKind::Rank => left_value,
        ProposalKind::Product => {
            left_value * f64::from(matrix.columns[usize::from(right.unwrap_or(left))][row])
        }
        ProposalKind::Gate => {
            let right_value = matrix.columns[usize::from(right.unwrap_or(left))][row];
            if left_value > 0.0 && right_value > 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn gather_target(matrix: &ColumnMatrix, rows: &[usize]) -> Vec<f64> {
    rows.iter()
        .copied()
        .map(|row| f64::from(matrix.target[row]))
        .collect()
}

fn fit_line(feature: &[f64], target: &[f64]) -> (f64, f64) {
    let count = target.len() as f64;
    if count == 0.0 {
        return (0.0, 0.0);
    }
    let mean_feature = feature.iter().sum::<f64>() / count;
    let mean_target = target.iter().sum::<f64>() / count;
    let mut variance = 0.0;
    let mut covariance = 0.0;
    for (value, response) in feature.iter().zip(target) {
        let feature_gap = *value - mean_feature;
        variance += feature_gap * feature_gap;
        covariance += feature_gap * (response - mean_target);
    }
    let slope = if variance > 1.0e-12 {
        covariance / variance
    } else {
        0.0
    };
    (mean_target - slope * mean_feature, slope)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn abs_correlation(feature: &[f64], target: &[f64]) -> f64 {
    let (_, slope) = fit_line(feature, target);
    let feature_scale = scale(feature.iter().copied());
    let target_scale = scale(target.iter().copied());
    if feature_scale <= 1.0e-12 || target_scale <= 1.0e-12 {
        0.0
    } else {
        (slope * feature_scale / target_scale).abs()
    }
}

fn scale(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<f64> = values.collect();
    let center = mean(&values);
    let variance = values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>();
    (variance / values.len() as f64).sqrt()
}

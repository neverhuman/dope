//! Entrypoints (`prepare_table`, `prepare_table_cells`, `prepare_table_mixed`)
//! for constructing a [`PreparedTable`].

use super::descriptors::{compute_table_descriptors, Level};
use super::np::{argsort_i64, mean1d, median, pairwise_sum, quantiles, round6, sorted, std1d};
use super::scaling::{
    err, ClassVocabPolicy, ColumnKind, FitOnlyTableTransform, PrepTask, TargetScaler, TargetValue,
};
use super::scaling_fit::{
    fit_categorical_column, fit_numeric_column, infer_feature_kinds_and_parse, parse_float_cell,
    py_float_repr, py_strip, regression_array, safe_skew_kurt,
};
use super::scaling_table::{
    compute_native_target_stats, compute_observed_target_stats, hash_raw_cells, validate_row_ids,
    PreparedTable,
};
use crate::error::Result;

fn classes_of(t: &TargetScaler) -> usize {
    match t.task {
        PrepTask::Regression => 0,
        PrepTask::Classification => 2,
        PrepTask::Multiclass => t.class_labels.len(),
    }
}

/// Numeric-matrix preparation with explicit physical `row_ids` and [`ClassVocabPolicy`].
pub fn prepare_table_with_policy(
    x: &[Vec<f64>],
    y: &[f64],
    task: PrepTask,
    row_ids: Vec<u64>,
    class_vocab_policy: ClassVocabPolicy,
) -> Result<PreparedTable> {
    let n = x.len();
    let p = x.first().map_or(0, Vec::len);
    if x.iter().any(|r| r.len() != p) {
        return Err(err("prepare_ragged_rows"));
    }
    if y.len() != n {
        return Err(err("prepare_target_length_mismatch"));
    }
    validate_row_ids(&row_ids, n)?;
    let yv: Vec<TargetValue<'_>> = y.iter().map(|&v| TargetValue::Num(v)).collect();
    let target = fit_target_scaler_with_policy(&yv, task, class_vocab_policy)?;
    let mut numeric = Vec::with_capacity(p);
    let mut frame_columns = Vec::with_capacity(p);
    for j in 0..p {
        let col: Vec<f64> = x.iter().map(|r| r[j]).collect();
        numeric.push(Some(fit_numeric_column(j, &col)));
        frame_columns.push(
            col.into_iter()
                .map(|v| if v.is_finite() { v } else { f64::NAN })
                .collect(),
        );
    }
    let mut xs = vec![0.0; n * p];
    for j in 0..p {
        let sc = numeric[j].as_ref().expect("numeric");
        for i in 0..n {
            xs[i * p + j] = sc.transform_value(x[i][j]).0;
        }
    }
    let y_scaled = target.transform(&yv)?;
    let frame_y = if task == PrepTask::Regression {
        let fallback = if target.mean.is_finite() {
            target.mean
        } else {
            0.0
        };
        y.iter()
            .map(|&v| if v.is_finite() { v } else { fallback })
            .collect()
    } else {
        y_scaled.clone()
    };
    let observed_target_stats = compute_observed_target_stats(&y_scaled, task);
    let target_stats = compute_native_target_stats(&frame_y, &y_scaled, task);
    let high_table_descriptors = compute_table_descriptors(&xs, n, p, &y_scaled, Level::High);
    let transform = FitOnlyTableTransform {
        task,
        kinds: vec![ColumnKind::Numeric; p],
        numeric,
        categorical: vec![None; p],
        target,
        min_level_count: 5,
        prior_smoothing: 10.0,
        class_vocab_policy,
    };
    Ok(PreparedTable {
        task,
        rows: n,
        features: p,
        classes: classes_of(&transform.target),
        x_scaled: xs,
        y_scaled,
        transform,
        row_ids,
        categories: vec![None; p],
        category_tokens: None,
        frame_columns,
        frame_y,
        target_stats,
        observed_target_stats,
        high_table_descriptors,
        raw_cells_sha256: None,
        source_digest_override: None,
    })
}

/// Python `FitOnlyTableTransform.fit(X: float ndarray, y, task).transform(X, y)`
/// (all-numeric ndarray path) with explicit physical `row_ids` (`Cap32V1`).
pub fn prepare_table_with_ids(
    x: &[Vec<f64>],
    y: &[f64],
    task: PrepTask,
    row_ids: Vec<u64>,
) -> Result<PreparedTable> {
    prepare_table_with_policy(x, y, task, row_ids, ClassVocabPolicy::Cap32V1)
}

/// Python `FitOnlyTableTransform.fit(X: float ndarray, y, task).transform(X, y)`
/// (all-numeric ndarray path) using sequential `0..n as u64` row IDs (`Cap32V1`).
pub fn prepare_table(x: &[Vec<f64>], y: &[f64], task: PrepTask) -> Result<PreparedTable> {
    let row_ids = (0..x.len() as u64).collect();
    prepare_table_with_policy(x, y, task, row_ids, ClassVocabPolicy::Cap32V1)
}

/// Lossless FIT-vocabulary (`K >= 33` preserved) numeric table preparation.
pub fn prepare_table_lossless(x: &[Vec<f64>], y: &[f64], task: PrepTask) -> Result<PreparedTable> {
    let row_ids = (0..x.len() as u64).collect();
    prepare_table_with_policy(x, y, task, row_ids, ClassVocabPolicy::LosslessV1)
}

fn prepare_cells_with_yv(
    x: &[Vec<String>],
    yv: &[TargetValue<'_>],
    declared_kinds: Option<&[ColumnKind]>,
    task: PrepTask,
    row_ids: Vec<u64>,
    class_vocab_policy: ClassVocabPolicy,
) -> Result<PreparedTable> {
    let n = x.len();
    let p = x.first().map_or(0, Vec::len);
    if x.iter().any(|r| r.len() != p) {
        return Err(err("prepare_ragged_rows"));
    }
    if yv.len() != n {
        return Err(err("prepare_target_length_mismatch"));
    }
    validate_row_ids(&row_ids, n)?;
    let raw_cells_sha256 = Some(hash_raw_cells(x, yv));
    let min_level_count = 5usize;
    let prior_smoothing = 10.0f64;
    let (kinds, mut parsed_numeric_cols): (Vec<ColumnKind>, Vec<Option<Vec<f64>>>) =
        match declared_kinds {
            Some(dk) => {
                if dk.len() != p {
                    return Err(err("prepare_kinds_length_mismatch"));
                }
                (dk.to_vec(), vec![None; p])
            }
            None => infer_feature_kinds_and_parse(x, p),
        };
    let target = fit_target_scaler_with_policy(yv, task, class_vocab_policy)?;
    let y_num = target.transform(yv)?;
    let y_global_mean = if y_num.is_empty() {
        0.0
    } else {
        mean1d(&y_num)
    };
    let mut numeric = vec![None; p];
    let mut categorical = vec![None; p];
    for (j, kind) in kinds.iter().enumerate() {
        match kind {
            ColumnKind::Numeric => {
                let col = parsed_numeric_cols[j]
                    .get_or_insert_with(|| x.iter().map(|r| parse_float_cell(&r[j])).collect());
                numeric[j] = Some(fit_numeric_column(j, col));
            }
            ColumnKind::Categorical => {
                let col: Vec<&str> = x.iter().map(|r| r[j].as_str()).collect();
                categorical[j] = Some(fit_categorical_column(
                    j,
                    &col,
                    &y_num,
                    y_global_mean,
                    min_level_count,
                    prior_smoothing,
                ));
            }
        }
    }
    let mut xs = vec![0.0; n * p];
    let mut categories = vec![None; p];
    let mut frame_columns = vec![vec![f64::NAN; n]; p];
    for j in 0..p {
        match kinds[j] {
            ColumnKind::Numeric => {
                let sc = numeric[j].as_ref().expect("numeric");
                let col = parsed_numeric_cols[j].as_ref().expect("parsed numeric");
                for (i, &raw) in col.iter().enumerate() {
                    xs[i * p + j] = sc.transform_value(raw).0;
                    frame_columns[j][i] = if raw.is_finite() { raw } else { f64::NAN };
                }
            }
            ColumnKind::Categorical => {
                let ce = categorical[j].as_ref().expect("categorical");
                categories[j] = Some(ce.vocab_tokens.len());
                for i in 0..n {
                    let (z, idx, miss) = ce.transform_cell(&x[i][j]);
                    xs[i * p + j] = z;
                    frame_columns[j][i] = if miss > 0.0 { f64::NAN } else { idx as f64 };
                }
            }
        }
    }
    let category_tokens = if kinds.contains(&ColumnKind::Categorical) {
        Some(
            (0..p)
                .map(|j| {
                    categorical[j]
                        .as_ref()
                        .map_or_else(Vec::new, |ce| ce.vocab_tokens.clone())
                })
                .collect(),
        )
    } else {
        None
    };
    let frame_y = if task == PrepTask::Regression {
        let raw_y = regression_array(yv)?;
        let fallback = if target.mean.is_finite() {
            target.mean
        } else {
            0.0
        };
        raw_y
            .into_iter()
            .map(|v| if v.is_finite() { v } else { fallback })
            .collect()
    } else {
        y_num.clone()
    };
    let observed_target_stats = compute_observed_target_stats(&y_num, task);
    let target_stats = compute_native_target_stats(&frame_y, &y_num, task);
    let high_table_descriptors = compute_table_descriptors(&xs, n, p, &y_num, Level::High);
    let transform = FitOnlyTableTransform {
        task,
        kinds,
        numeric,
        categorical,
        target,
        min_level_count,
        prior_smoothing,
        class_vocab_policy,
    };
    Ok(PreparedTable {
        task,
        rows: n,
        features: p,
        classes: classes_of(&transform.target),
        x_scaled: xs,
        y_scaled: y_num,
        transform,
        row_ids,
        categories,
        category_tokens,
        frame_columns,
        frame_y,
        target_stats,
        observed_target_stats,
        high_table_descriptors,
        raw_cells_sha256,
        source_digest_override: None,
    })
}

/// String-cell table preparation with explicit `row_ids` and [`ClassVocabPolicy`].
pub fn prepare_table_cells_with_policy(
    x: &[Vec<String>],
    y: &[String],
    task: PrepTask,
    row_ids: Vec<u64>,
    class_vocab_policy: ClassVocabPolicy,
) -> Result<PreparedTable> {
    let yv: Vec<TargetValue<'_>> = y.iter().map(|s| TargetValue::Str(s.as_str())).collect();
    prepare_cells_with_yv(x, &yv, None, task, row_ids, class_vocab_policy)
}

/// Python `FitOnlyTableTransform.fit(X: list of str rows, y: list of str, task).transform(X, y)`
/// (kind inference + Bayesian-shrunk categorical encoding) with explicit physical `row_ids`.
pub fn prepare_table_cells_with_ids(
    x: &[Vec<String>],
    y: &[String],
    task: PrepTask,
    row_ids: Vec<u64>,
) -> Result<PreparedTable> {
    prepare_table_cells_with_policy(x, y, task, row_ids, ClassVocabPolicy::Cap32V1)
}

/// Mixed-column table preparation with explicit [`ClassVocabPolicy`].
pub fn prepare_table_mixed_with_policy(
    x_cells: &[Vec<String>],
    y: &[f64],
    declared_kinds: Option<&[ColumnKind]>,
    task: PrepTask,
    row_ids: Option<Vec<u64>>,
    class_vocab_policy: ClassVocabPolicy,
) -> Result<PreparedTable> {
    let yv: Vec<TargetValue<'_>> = y.iter().map(|&v| TargetValue::Num(v)).collect();
    let row_ids = row_ids.unwrap_or_else(|| (0..x_cells.len() as u64).collect());
    prepare_cells_with_yv(
        x_cells,
        &yv,
        declared_kinds,
        task,
        row_ids,
        class_vocab_policy,
    )
}

/// Mixed-column table preparation (`Cap32V1` default).
pub fn prepare_table_mixed(
    x_cells: &[Vec<String>],
    y: &[f64],
    declared_kinds: Option<&[ColumnKind]>,
    task: PrepTask,
    row_ids: Option<Vec<u64>>,
) -> Result<PreparedTable> {
    prepare_table_mixed_with_policy(
        x_cells,
        y,
        declared_kinds,
        task,
        row_ids,
        ClassVocabPolicy::Cap32V1,
    )
}

/// Python `FitOnlyTableTransform.fit(X: list of str rows, y: list of str, task).transform(X, y)`
/// using sequential `0..n as u64` row IDs (`Cap32V1`).
pub fn prepare_table_cells(
    x: &[Vec<String>],
    y: &[String],
    task: PrepTask,
) -> Result<PreparedTable> {
    let row_ids = (0..x.len() as u64).collect();
    prepare_table_cells_with_ids(x, y, task, row_ids)
}

pub(crate) fn fit_target_scaler_with_policy(
    y: &[TargetValue<'_>],
    task: PrepTask,
    class_vocab_policy: ClassVocabPolicy,
) -> Result<TargetScaler> {
    match task {
        PrepTask::Multiclass => {
            let raw: Vec<f64> = y
                .iter()
                .map(|v| match v {
                    TargetValue::Num(x) => *x,
                    TargetValue::Str(s) => parse_float_cell(s),
                })
                .collect();
            let finite: Vec<f64> = raw.into_iter().filter(|v| v.is_finite()).collect();
            if finite.is_empty() {
                let mut ts = TargetScaler::base(task, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
                ts.class_labels = vec![0.0, 0.5, 1.0];
                ts.class_priors = vec![0.34, 0.33, 0.33];
                return Ok(ts);
            }
            let rounded = sorted(&finite.iter().map(|&v| round6(v)).collect::<Vec<_>>());
            let mut u_vals: Vec<f64> = Vec::new();
            let mut u_counts: Vec<i64> = Vec::new();
            for v in rounded {
                if u_vals.last() == Some(&v) {
                    *u_counts.last_mut().expect("count") += 1;
                } else {
                    u_vals.push(v);
                    u_counts.push(1);
                }
            }
            let counts_f: Vec<f64> =
                if class_vocab_policy == ClassVocabPolicy::Cap32V1 && u_vals.len() > 32 {
                    let order = argsort_i64(&u_counts);
                    let top: Vec<f64> = order.iter().rev().take(32).map(|&i| u_vals[i]).collect();
                    u_vals = sorted(&top);
                    u_vals
                        .iter()
                        .map(|&v| finite.iter().filter(|&&f| (f - v).abs() <= 1e-5).count() as f64)
                        .collect()
                } else {
                    u_counts.iter().map(|&c| c as f64).collect()
                };
            let total = pairwise_sum(&counts_f).max(1.0);
            let priors: Vec<f64> = counts_f.iter().map(|c| c / total).collect();
            let mut ts = TargetScaler::base(
                task,
                mean1d(&finite),
                std1d(&finite).max(1e-6),
                median(&finite),
                1.0,
                0.0,
                0.0,
            );
            ts.class_prior = priors[0];
            ts.class_labels = u_vals;
            ts.class_priors = priors;
            Ok(ts)
        }
        PrepTask::Classification => {
            let tokens: Vec<String> = y
                .iter()
                .map(|v| match v {
                    TargetValue::Num(x) => py_float_repr(*x),
                    TargetValue::Str(s) => py_strip(s).to_string(),
                })
                .collect();
            let mut uniq = tokens.clone();
            uniq.sort();
            uniq.dedup();
            let pos = if uniq.iter().any(|u| u == "1" || u == "1.0") {
                if uniq.iter().any(|u| u == "1") {
                    "1".to_string()
                } else {
                    "1.0".to_string()
                }
            } else if let Some(t) = uniq.iter().find(|u| u.to_lowercase() == "true") {
                t.clone()
            } else {
                uniq.last().cloned().unwrap_or_else(|| "1".to_string())
            };
            let bin: Vec<f64> = tokens
                .iter()
                .map(|t| if *t == pos { 1.0 } else { 0.0 })
                .collect();
            let prior = if bin.is_empty() { 0.5 } else { mean1d(&bin) };
            let mut ts = TargetScaler::base(
                task,
                prior,
                (prior * (1.0 - prior)).sqrt().max(1e-6),
                if prior >= 0.5 { 1.0 } else { 0.0 },
                1.0,
                0.0,
                0.0,
            );
            ts.positive_label = Some(pos);
            ts.class_prior = prior;
            Ok(ts)
        }
        PrepTask::Regression => {
            let arr = regression_array(y)?;
            let finite: Vec<f64> = arr.into_iter().filter(|v| v.is_finite()).collect();
            if finite.is_empty() {
                return Ok(TargetScaler::base(task, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0));
            }
            let m = mean1d(&finite);
            let s = std1d(&finite).max(1e-6);
            let q = quantiles(&finite, &[0.25, 0.5, 0.75]);
            let (skew, kurt) = safe_skew_kurt(&finite);
            Ok(TargetScaler::base(
                task,
                m,
                s,
                q[1],
                (q[2] - q[0]).max(1e-6),
                skew,
                kurt,
            ))
        }
    }
}

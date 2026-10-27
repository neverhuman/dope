//! Entrypoints (`prepare_table`, `prepare_table_cells`, `prepare_table_mixed`)
//! for constructing a [`PreparedTable`].

use super::descriptors::{compute_table_descriptors, Level};
use super::np::mean1d;
use super::scaling::{
    err, ClassVocabPolicy, ColumnKind, FitOnlyTableTransform, PrepTask, PrepareStageTimings,
    TargetScaler, TargetValue,
};
use super::scaling_fit::{
    fit_categorical_column, fit_numeric_column, fit_target_scaler_with_policy, is_missing_token,
    parse_float_cell, py_float, py_strip, regression_array,
};
use super::scaling_table::{
    compute_native_target_stats, compute_observed_target_stats, hash_raw_cells, validate_row_ids,
    PreparedTable,
};
use crate::error::Result;
use std::time::Instant;

fn classes_of(t: &TargetScaler) -> usize {
    match t.task {
        PrepTask::Regression => 0,
        PrepTask::Classification => 2,
        PrepTask::Multiclass => t.class_labels.len(),
    }
}

/// Numeric-matrix preparation with explicit physical `row_ids`, [`ClassVocabPolicy`],
/// and non-overlapping stage timings.
pub fn prepare_table_with_policy_timed(
    x: &[Vec<f64>],
    y: &[f64],
    task: PrepTask,
    row_ids: Vec<u64>,
    class_vocab_policy: ClassVocabPolicy,
) -> Result<(PreparedTable, PrepareStageTimings)> {
    let t_total = Instant::now();
    let t_parse = Instant::now();
    let n = x.len();
    let p = x.first().map_or(0, Vec::len);
    if x.iter().any(|r| r.len() != p) {
        return Err(err("prepare_ragged_rows"));
    }
    if y.len() != n {
        return Err(err("prepare_target_length_mismatch"));
    }
    validate_row_ids(&row_ids, n)?;
    let mut frame_columns: Vec<Vec<f64>> = vec![vec![0.0f64; n]; p];
    for (i, row) in x.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            frame_columns[j][i] = v;
        }
    }
    let parse_and_infer_ms = t_parse.elapsed().as_secs_f64() * 1e3;

    let t_scale = Instant::now();
    let yv: Vec<TargetValue<'_>> = y.iter().map(|&v| TargetValue::Num(v)).collect();
    let target = fit_target_scaler_with_policy(&yv, task, class_vocab_policy)?;
    let mut numeric = Vec::with_capacity(p);
    let mut xs = vec![0.0; n * p];
    for (j, col) in frame_columns.iter_mut().enumerate() {
        let sc = fit_numeric_column(j, col);
        sc.transform_column_strided(col, &mut xs, j, p);
        for v in col.iter_mut() {
            if !v.is_finite() {
                *v = f64::NAN;
            }
        }
        numeric.push(Some(sc));
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
    let scaler_fit_transform_ms = t_scale.elapsed().as_secs_f64() * 1e3;

    let t_desc = Instant::now();
    let high_table_descriptors = compute_table_descriptors(&xs, n, p, &y_scaled, Level::High);
    let table_descriptors_ms = t_desc.elapsed().as_secs_f64() * 1e3;

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
    let timings = PrepareStageTimings {
        parse_and_infer_ms,
        scaler_fit_transform_ms,
        table_descriptors_ms,
        total_ms: t_total.elapsed().as_secs_f64() * 1e3,
    };
    Ok((
        PreparedTable {
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
        },
        timings,
    ))
}

/// Numeric-matrix preparation with explicit physical `row_ids` and [`ClassVocabPolicy`].
pub fn prepare_table_with_policy(
    x: &[Vec<f64>],
    y: &[f64],
    task: PrepTask,
    row_ids: Vec<u64>,
    class_vocab_policy: ClassVocabPolicy,
) -> Result<PreparedTable> {
    Ok(prepare_table_with_policy_timed(x, y, task, row_ids, class_vocab_policy)?.0)
}

/// Python `FitOnlyTableTransform.fit(X: float ndarray, y, task).transform(X, y)`
/// (all-numeric ndarray path) with explicit physical `row_ids` and stage timings (`Cap32V1`).
pub fn prepare_table_with_ids_timed(
    x: &[Vec<f64>],
    y: &[f64],
    task: PrepTask,
    row_ids: Vec<u64>,
) -> Result<(PreparedTable, PrepareStageTimings)> {
    prepare_table_with_policy_timed(x, y, task, row_ids, ClassVocabPolicy::Cap32V1)
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
/// (all-numeric ndarray path) using sequential `0..n as u64` row IDs, returning stage timings.
pub fn prepare_table_timed(
    x: &[Vec<f64>],
    y: &[f64],
    task: PrepTask,
) -> Result<(PreparedTable, PrepareStageTimings)> {
    let row_ids = (0..x.len() as u64).collect();
    prepare_table_with_ids_timed(x, y, task, row_ids)
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
) -> Result<(PreparedTable, PrepareStageTimings)> {
    let t_total = Instant::now();
    let t_parse = Instant::now();
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

    let (kinds, mut parsed_cols) = match declared_kinds {
        Some(dk) => {
            if dk.len() != p {
                return Err(err("prepare_kinds_length_mismatch"));
            }
            let kinds = dk.to_vec();
            let mut cols: Vec<Vec<f64>> = kinds
                .iter()
                .map(|&k| {
                    if k == ColumnKind::Numeric {
                        Vec::with_capacity(n)
                    } else {
                        Vec::new()
                    }
                })
                .collect();
            for row in x {
                for (j, &k) in kinds.iter().enumerate() {
                    if k == ColumnKind::Numeric {
                        cols[j].push(parse_float_cell(&row[j]));
                    }
                }
            }
            (kinds, cols)
        }
        None => {
            let mut is_num = vec![true; p];
            let mut cols: Vec<Vec<f64>> = (0..p).map(|_| Vec::with_capacity(n)).collect();
            for row in x {
                for j in 0..p {
                    if !is_num[j] {
                        continue;
                    }
                    let s = py_strip(&row[j]);
                    if is_missing_token(s) {
                        cols[j].push(f64::NAN);
                    } else if let Some(v) = py_float(s) {
                        cols[j].push(v);
                    } else {
                        is_num[j] = false;
                        cols[j] = Vec::new();
                    }
                }
            }
            let kinds: Vec<ColumnKind> = is_num
                .into_iter()
                .map(|ok| {
                    if ok {
                        ColumnKind::Numeric
                    } else {
                        ColumnKind::Categorical
                    }
                })
                .collect();
            (kinds, cols)
        }
    };
    let parse_and_infer_ms = t_parse.elapsed().as_secs_f64() * 1e3;

    let t_scale = Instant::now();
    let target = fit_target_scaler_with_policy(yv, task, class_vocab_policy)?;
    let y_num = target.transform(yv)?;
    let y_global_mean = if y_num.is_empty() {
        0.0
    } else {
        mean1d(&y_num)
    };
    let mut numeric = vec![None; p];
    let mut categorical = vec![None; p];
    let mut xs = vec![0.0; n * p];
    let mut categories = vec![None; p];
    let mut frame_columns = vec![Vec::new(); p];
    for (j, kind) in kinds.iter().enumerate() {
        match kind {
            ColumnKind::Numeric => {
                let col = &mut parsed_cols[j];
                let sc = fit_numeric_column(j, col);
                sc.transform_column_strided(col, &mut xs, j, p);
                for v in col.iter_mut() {
                    if !v.is_finite() {
                        *v = f64::NAN;
                    }
                }
                frame_columns[j] = std::mem::take(col);
                numeric[j] = Some(sc);
            }
            ColumnKind::Categorical => {
                let col: Vec<&str> = x.iter().map(|r| r[j].as_str()).collect();
                let ce = fit_categorical_column(
                    j,
                    &col,
                    &y_num,
                    y_global_mean,
                    min_level_count,
                    prior_smoothing,
                );
                categories[j] = Some(ce.vocab_tokens.len());
                let mut fcol = vec![f64::NAN; n];
                for i in 0..n {
                    let (z, idx, miss) = ce.transform_cell(&x[i][j]);
                    xs[i * p + j] = z;
                    fcol[i] = if miss > 0.0 { f64::NAN } else { idx as f64 };
                }
                frame_columns[j] = fcol;
                categorical[j] = Some(ce);
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
    let scaler_fit_transform_ms = t_scale.elapsed().as_secs_f64() * 1e3;

    let t_desc = Instant::now();
    let high_table_descriptors = compute_table_descriptors(&xs, n, p, &y_num, Level::High);
    let table_descriptors_ms = t_desc.elapsed().as_secs_f64() * 1e3;

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
    let timings = PrepareStageTimings {
        parse_and_infer_ms,
        scaler_fit_transform_ms,
        table_descriptors_ms,
        total_ms: t_total.elapsed().as_secs_f64() * 1e3,
    };
    Ok((
        PreparedTable {
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
        },
        timings,
    ))
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
    Ok(prepare_cells_with_yv(x, &yv, None, task, row_ids, class_vocab_policy)?.0)
}

/// Python `FitOnlyTableTransform.fit(X: list of str rows, y: list of str, task).transform(X, y)`
/// with explicit physical `row_ids`, returning non-overlapping stage timings (`Cap32V1`).
pub fn prepare_table_cells_with_ids_timed(
    x: &[Vec<String>],
    y: &[String],
    task: PrepTask,
    row_ids: Vec<u64>,
) -> Result<(PreparedTable, PrepareStageTimings)> {
    let yv: Vec<TargetValue<'_>> = y.iter().map(|s| TargetValue::Str(s.as_str())).collect();
    prepare_cells_with_yv(x, &yv, None, task, row_ids, ClassVocabPolicy::Cap32V1)
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
    Ok(prepare_cells_with_yv(
        x_cells,
        &yv,
        declared_kinds,
        task,
        row_ids,
        class_vocab_policy,
    )?
    .0)
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
/// using sequential `0..n as u64` row IDs (`Cap32V1`), returning non-overlapping stage timings.
pub fn prepare_table_cells_timed(
    x: &[Vec<String>],
    y: &[String],
    task: PrepTask,
) -> Result<(PreparedTable, PrepareStageTimings)> {
    let row_ids = (0..x.len() as u64).collect();
    prepare_table_cells_with_ids_timed(x, y, task, row_ids)
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

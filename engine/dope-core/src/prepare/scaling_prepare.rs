//! Entrypoints (`prepare_table`, `prepare_table_cells`, `prepare_table_mixed`)
//! for constructing a [`PreparedTable`].

use super::descriptors::{compute_table_descriptors, Level};
use super::np::mean1d;
use super::scaling::{
    err, ClassVocabPolicy, ColumnKind, FitOnlyTableTransform, PrepTask, PrepareStageTimings,
    TargetScaler, TargetValue,
};
use super::scaling_fit::{fit_target_scaler_with_policy, regression_array};
use super::scaling_par::{
    infer_parse_columns, interleave_columns, parse_declared_columns, scale_columns,
    scale_numeric_rows,
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
    let parse_and_infer_ms = t_parse.elapsed().as_secs_f64() * 1e3;

    let t_scale = Instant::now();
    let yv: Vec<TargetValue<'_>> = y.iter().map(|&v| TargetValue::Num(v)).collect();
    let target = fit_target_scaler_with_policy(&yv, task, class_vocab_policy)?;
    let scaled = scale_numeric_rows(x, p);
    let xs = interleave_columns(&scaled, n, p);
    let mut numeric = Vec::with_capacity(p);
    let mut frame_columns = Vec::with_capacity(p);
    for column in scaled {
        numeric.push(column.numeric);
        frame_columns.push(column.frame);
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
    if declared_kinds.is_some_and(|dk| dk.len() != p) {
        return Err(err("prepare_kinds_length_mismatch"));
    }
    let min_level_count = 5usize;
    let prior_smoothing = 10.0f64;

    // The raw-cell digest and the per-column parse are independent; run them together.
    let (raw_cells_sha256, (kinds, parsed_cols)) = rayon::join(
        || hash_raw_cells(x, yv),
        || match declared_kinds {
            Some(dk) => (dk.to_vec(), parse_declared_columns(x, dk)),
            None => infer_parse_columns(x, p),
        },
    );
    let raw_cells_sha256 = Some(raw_cells_sha256);
    let parse_and_infer_ms = t_parse.elapsed().as_secs_f64() * 1e3;

    let t_scale = Instant::now();
    let target = fit_target_scaler_with_policy(yv, task, class_vocab_policy)?;
    let y_num = target.transform(yv)?;
    let y_global_mean = if y_num.is_empty() {
        0.0
    } else {
        mean1d(&y_num)
    };
    let scaled = scale_columns(
        x,
        &kinds,
        parsed_cols,
        &y_num,
        y_global_mean,
        min_level_count,
        prior_smoothing,
    );
    let xs = interleave_columns(&scaled, n, p);
    let mut numeric = Vec::with_capacity(p);
    let mut categorical = Vec::with_capacity(p);
    let mut categories = Vec::with_capacity(p);
    let mut frame_columns = Vec::with_capacity(p);
    for column in scaled {
        categories.push(column.categorical.as_ref().map(|ce| ce.vocab_tokens.len()));
        numeric.push(column.numeric);
        categorical.push(column.categorical);
        frame_columns.push(column.frame);
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

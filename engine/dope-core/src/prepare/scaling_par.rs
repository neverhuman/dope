//! Column-parallel parse and scaler stages for [`super::scaling_prepare`].
//!
//! Each column runs exactly the serial per-column code it ran before (same cells, same
//! order, same arithmetic), and results are collected in column order. Prepared tables
//! are therefore bitwise identical for every Rayon thread count.

use super::scaling::{CategoricalColumnEncoder, ColumnKind, NumericColumnScaler};
use super::scaling_fit::{
    fit_categorical_column, fit_numeric_column, is_missing_token, parse_float_cell, py_float,
    py_strip,
};
use rayon::prelude::*;

/// Parse the declared numeric columns of row-major cells; other columns stay empty.
pub(super) fn parse_declared_columns(x: &[Vec<String>], kinds: &[ColumnKind]) -> Vec<Vec<f64>> {
    kinds
        .par_iter()
        .enumerate()
        .map(|(j, &k)| {
            if k == ColumnKind::Numeric {
                x.iter().map(|row| parse_float_cell(&row[j])).collect()
            } else {
                Vec::new()
            }
        })
        .collect()
}

/// Infer each column's kind: numeric while every cell is missing or parses as a float.
/// A categorical column keeps no parsed values.
pub(super) fn infer_parse_columns(x: &[Vec<String>], p: usize) -> (Vec<ColumnKind>, Vec<Vec<f64>>) {
    let parsed: Vec<Option<Vec<f64>>> = (0..p)
        .into_par_iter()
        .map(|j| {
            let mut col = Vec::with_capacity(x.len());
            for row in x {
                let s = py_strip(&row[j]);
                if is_missing_token(s) {
                    col.push(f64::NAN);
                } else if let Some(v) = py_float(s) {
                    col.push(v);
                } else {
                    return None;
                }
            }
            Some(col)
        })
        .collect();
    let kinds = parsed
        .iter()
        .map(|c| {
            if c.is_some() {
                ColumnKind::Numeric
            } else {
                ColumnKind::Categorical
            }
        })
        .collect();
    (
        kinds,
        parsed.into_iter().map(Option::unwrap_or_default).collect(),
    )
}

/// One fitted column: its scaled values, frame values and fitted transform.
pub(super) struct ScaledColumn {
    pub z: Vec<f64>,
    pub frame: Vec<f64>,
    pub numeric: Option<NumericColumnScaler>,
    pub categorical: Option<CategoricalColumnEncoder>,
}

/// Fit and apply a numeric scaler; non-finite frame values become NaN.
fn scale_numeric(j: usize, mut col: Vec<f64>) -> ScaledColumn {
    let sc = fit_numeric_column(j, &col);
    let mut z = vec![0.0; col.len()];
    sc.transform_column_strided(&col, &mut z, 0, 1);
    for v in col.iter_mut() {
        if !v.is_finite() {
            *v = f64::NAN;
        }
    }
    ScaledColumn {
        z,
        frame: col,
        numeric: Some(sc),
        categorical: None,
    }
}

/// Fit every column of a mixed table. Categorical columns are fit from their raw cells.
pub(super) fn scale_columns(
    x: &[Vec<String>],
    kinds: &[ColumnKind],
    parsed: Vec<Vec<f64>>,
    y_num: &[f64],
    y_global_mean: f64,
    min_level_count: usize,
    prior_smoothing: f64,
) -> Vec<ScaledColumn> {
    let n = x.len();
    parsed
        .into_par_iter()
        .zip(kinds.par_iter())
        .enumerate()
        .map(|(j, (col, kind))| match kind {
            ColumnKind::Numeric => scale_numeric(j, col),
            ColumnKind::Categorical => {
                let cells: Vec<&str> = x.iter().map(|r| r[j].as_str()).collect();
                let ce = fit_categorical_column(
                    j,
                    &cells,
                    y_num,
                    y_global_mean,
                    min_level_count,
                    prior_smoothing,
                );
                let mut z = vec![0.0; n];
                let mut fcol = vec![f64::NAN; n];
                for i in 0..n {
                    let (zi, idx, miss) = ce.transform_cell(&x[i][j]);
                    z[i] = zi;
                    fcol[i] = if miss > 0.0 { f64::NAN } else { idx as f64 };
                }
                ScaledColumn {
                    z,
                    frame: fcol,
                    numeric: None,
                    categorical: Some(ce),
                }
            }
        })
        .collect()
}

/// Fit every column of a numeric row-major matrix.
pub(super) fn scale_numeric_rows(x: &[Vec<f64>], p: usize) -> Vec<ScaledColumn> {
    (0..p)
        .into_par_iter()
        .map(|j| scale_numeric(j, x.iter().map(|row| row[j]).collect()))
        .collect()
}

/// The row-major `n × p` scaled matrix from fitted columns.
pub(super) fn interleave_columns(cols: &[ScaledColumn], n: usize, p: usize) -> Vec<f64> {
    let mut xs = vec![0.0; n * p];
    if p > 0 {
        xs.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
            for (j, c) in cols.iter().enumerate() {
                row[j] = c.z[i];
            }
        });
    }
    xs
}

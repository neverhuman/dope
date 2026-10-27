//! Rust port of `research/levels/scaling.py` (`FitOnlyTableTransform`).
//!
//! Train-only per-column scaling and categorical encoding producing the
//! `X_train_scaled` / `y_train_scaled` arrays consumed by the MLC-v3
//! conditioning builders. Statistics are fitted on the training rows only.

use super::np::{argsort_i64, clip, interp, mean1d, median, pairwise_sum, round6, sorted, std1d};
use crate::error::{DopeError, Result};
use std::collections::HashMap;

pub use super::scaling_fit::*;
pub use super::scaling_prepare::*;
pub use super::scaling_table::*;

/// Non-overlapping stage timings (in milliseconds) for [`prepare_table_cells_timed`]
/// and [`prepare_table_timed`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PrepareStageTimings {
    /// Raw cell hashing (`raw_cells_sha256`), string stripping, missing-token detection,
    /// column kind inference, and float parsing.
    pub parse_and_infer_ms: f64,
    /// Target and feature scaler fitting (`fit_target_scaler`, `fit_numeric_column`,
    /// `fit_categorical_column`) plus forward transformation into `x_scaled`, `y_scaled`,
    /// `frame_columns`, `frame_y`, and `target_stats`.
    pub scaler_fit_transform_ms: f64,
    /// Static [`super::descriptors::Level::High`] [`super::descriptors::TableDescriptors`]
    /// computation (`compute_table_descriptors`).
    pub table_descriptors_ms: f64,
    /// Total elapsed wall time inside `prepare_table*`.
    pub total_ms: f64,
}

/// `scaling.QUANTILE_GRID`.
pub const QUANTILE_GRID: [f64; 9] = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];
/// `scaling.NORMAL_GRID`.
pub const NORMAL_GRID: [f64; 9] = [
    -2.32634787,
    -1.64485363,
    -1.28155157,
    -0.67448975,
    0.0,
    0.67448975,
    1.28155157,
    1.64485363,
    2.32634787,
];

pub(crate) fn err(msg: &str) -> DopeError {
    DopeError::Input(msg.to_string())
}

/// Python task string accepted by `FitOnlyTableTransform.fit(task=...)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrepTask {
    Regression,
    Classification,
    Multiclass,
}

impl PrepTask {
    pub fn as_str(self) -> &'static str {
        match self {
            PrepTask::Regression => "regression",
            PrepTask::Classification => "classification",
            PrepTask::Multiclass => "multiclass",
        }
    }
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "regression" => Ok(PrepTask::Regression),
            "classification" | "binary" => Ok(PrepTask::Classification),
            "multiclass" => Ok(PrepTask::Multiclass),
            _ => Err(err("prepare_unknown_task")),
        }
    }
}

impl From<crate::schema::Task> for PrepTask {
    fn from(t: crate::schema::Task) -> Self {
        match t {
            crate::schema::Task::Regression => PrepTask::Regression,
            crate::schema::Task::Classification => PrepTask::Classification,
        }
    }
}

/// Multiclass target vocabulary retention policy during train-only preparation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClassVocabPolicy {
    /// Legacy `prepare-v1` / `full-report-v1` policy: retain at most the 32 most
    /// frequent `round6` classes (`mc_many` compatibility).
    #[default]
    Cap32V1,
    /// Lossless FIT-vocabulary policy: retain all distinct finite `round6` classes
    /// (`K >= 33` supported without truncation).
    LosslessV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnKind {
    Numeric,
    Categorical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformKind {
    Constant,
    Binary,
    SignedLog1pHybrid,
    AsinhHybrid,
    QuantileRobustHybrid,
}

impl TransformKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TransformKind::Constant => "constant",
            TransformKind::Binary => "binary",
            TransformKind::SignedLog1pHybrid => "signed_log1p_hybrid",
            TransformKind::AsinhHybrid => "asinh_hybrid",
            TransformKind::QuantileRobustHybrid => "quantile_robust_hybrid",
        }
    }
}

/// `scaling.NumericColumnScaler`.
#[derive(Clone, Debug, PartialEq)]
pub struct NumericColumnScaler {
    pub slot: usize,
    pub transform_kind: TransformKind,
    pub missing_rate: f64,
    pub center: f64,
    pub scale: f64,
    pub post_mean: f64,
    pub post_std: f64,
    pub quantiles: [f64; 9],
    pub skewness: f64,
    pub kurtosis: f64,
    pub blend_quantile_weight: f64,
}

impl NumericColumnScaler {
    #[inline]
    fn raw_transform_with_q(&self, v: f64, has_monotone_q: bool, iqr: f64) -> f64 {
        let z_log = match self.transform_kind {
            TransformKind::Constant => return 0.0,
            TransformKind::Binary => {
                return clip((v - self.center) / self.scale.max(1e-9), -6.0, 6.0);
            }
            TransformKind::SignedLog1pHybrid => {
                let shifted = v - self.center;
                let scaled = np_sign(shifted) * (shifted.abs() / self.scale.max(1e-9)).ln_1p();
                (scaled - self.post_mean) / self.post_std.max(1e-9)
            }
            TransformKind::AsinhHybrid => {
                let shifted = v - self.center;
                let scaled = (shifted / self.scale.max(1e-9)).asinh();
                (scaled - self.post_mean) / self.post_std.max(1e-9)
            }
            TransformKind::QuantileRobustHybrid => (v - self.post_mean) / self.post_std.max(1e-9),
        };
        let q = &self.quantiles;
        let z = if has_monotone_q {
            let z_q = if v < q[0] {
                NORMAL_GRID[0] + (v - q[0]) / iqr * 1.3489795
            } else if v > q[8] {
                NORMAL_GRID[8] + (v - q[8]) / iqr * 1.3489795
            } else {
                interp(v, q, &NORMAL_GRID)
            };
            let w = self.blend_quantile_weight;
            (1.0 - w) * z_log + w * z_q
        } else {
            z_log
        };
        6.0 * (z / 6.0).tanh()
    }

    fn raw_transform(&self, v: f64) -> f64 {
        let q = &self.quantiles;
        let has_monotone_q = q.windows(2).all(|w| w[1] - w[0] > 1e-12);
        let iqr = (q[5] - q[3]).max(1e-9);
        self.raw_transform_with_q(v, has_monotone_q, iqr)
    }

    /// Returns (z, missing indicator) for one value.
    pub fn transform_value(&self, v: f64) -> (f64, f64) {
        if v.is_finite() {
            (self.raw_transform(v), 0.0)
        } else {
            (0.0, 1.0)
        }
    }

    /// Transform a column of length `n` into `xs` (strided `j + i * p`), hoisting
    /// per-column quantile invariants out of the row loop.
    pub(crate) fn transform_column_strided(&self, col: &[f64], xs: &mut [f64], j: usize, p: usize) {
        let q = &self.quantiles;
        let has_monotone_q = q.windows(2).all(|w| w[1] - w[0] > 1e-12);
        let iqr = (q[5] - q[3]).max(1e-9);
        for (i, &v) in col.iter().enumerate() {
            xs[i * p + j] = if v.is_finite() {
                self.raw_transform_with_q(v, has_monotone_q, iqr)
            } else {
                0.0
            };
        }
    }
}

/// `scaling.CategoricalColumnEncoder`.
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalColumnEncoder {
    pub slot: usize,
    pub missing_rate: f64,
    pub vocab_tokens: Vec<String>,
    pub token_to_id: HashMap<String, usize>,
    pub id_target_encoding: Vec<f64>,
    pub id_frequencies: Vec<f64>,
    pub post_mean: f64,
    pub post_std: f64,
}

impl CategoricalColumnEncoder {
    /// Returns (z, id, missing) for one raw cell.
    pub fn transform_cell(&self, cell: &str) -> (f64, usize, f64) {
        let s = py_strip(cell);
        let (miss, idx) = if is_missing_token(s) {
            (1.0, 0)
        } else {
            (0.0, self.token_to_id.get(s).copied().unwrap_or(0))
        };
        let enc = self.id_target_encoding[idx];
        let z = 6.0 * (((enc - self.post_mean) / self.post_std.max(1e-9)) / 6.0).tanh();
        (z, idx, miss)
    }
}

/// `scaling.TargetScaler`.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetScaler {
    pub task: PrepTask,
    pub mean: f64,
    pub std: f64,
    pub median: f64,
    pub iqr: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub positive_label: Option<String>,
    pub class_prior: f64,
    pub class_labels: Vec<f64>,
    pub class_priors: Vec<f64>,
}

/// One raw target value: numeric (ndarray / python number) or string (CSV cell).
#[derive(Clone, Copy, Debug)]
pub enum TargetValue<'a> {
    Num(f64),
    Str(&'a str),
}

impl TargetScaler {
    pub(crate) fn base(
        task: PrepTask,
        mean: f64,
        std: f64,
        median: f64,
        iqr: f64,
        skewness: f64,
        kurtosis: f64,
    ) -> Self {
        Self {
            task,
            mean,
            std,
            median,
            iqr,
            skewness,
            kurtosis,
            positive_label: None,
            class_prior: 0.5,
            class_labels: Vec::new(),
            class_priors: Vec::new(),
        }
    }

    pub(crate) fn transform(&self, y: &[TargetValue<'_>]) -> Result<Vec<f64>> {
        if self.task == PrepTask::Multiclass && !self.class_labels.is_empty() {
            let labels = &self.class_labels;
            return Ok(y
                .iter()
                .map(|v| {
                    let mut a = match v {
                        TargetValue::Num(x) => *x,
                        TargetValue::Str(s) => parse_float_cell(s),
                    };
                    if a.is_nan() {
                        a = labels[0];
                    } else if a == f64::INFINITY {
                        a = f64::MAX;
                    } else if a == f64::NEG_INFINITY {
                        a = f64::MIN;
                    }
                    let mut best = 0usize;
                    let mut bd = (a - labels[0]).abs();
                    for (k, l) in labels.iter().enumerate().skip(1) {
                        let d = (a - l).abs();
                        if d < bd {
                            bd = d;
                            best = k;
                        }
                    }
                    best as f64
                })
                .collect());
        }
        if self.task == PrepTask::Classification {
            let pos = self.positive_label.as_deref().unwrap_or("1");
            return Ok(y
                .iter()
                .map(|v| match v {
                    TargetValue::Num(x) => {
                        if *x > 0.5 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    TargetValue::Str(s) => {
                        if py_strip(s) == pos {
                            1.0
                        } else {
                            0.0
                        }
                    }
                })
                .collect());
        }
        let arr = regression_array(y)?;
        Ok(arr
            .iter()
            .map(|&a| {
                if a.is_finite() {
                    8.0 * (((a - self.mean) / self.std.max(1e-9)) / 8.0).tanh()
                } else {
                    0.0
                }
            })
            .collect())
    }

    /// Strict target transform that rejects non-finite or unseen multiclass query labels
    /// outside the fitted `round6` FIT vocabulary (`1e-6` tolerance).
    pub fn transform_strict(&self, y: &[TargetValue<'_>]) -> Result<Vec<f64>> {
        if self.task == PrepTask::Multiclass && !self.class_labels.is_empty() {
            let labels = &self.class_labels;
            return y
                .iter()
                .map(|v| {
                    let a = match v {
                        TargetValue::Num(x) => *x,
                        TargetValue::Str(s) => parse_float_cell(s),
                    };
                    if !a.is_finite() {
                        return Err(err("unknown_multiclass_label"));
                    }
                    let r = super::np::round6(a);
                    let mut best = 0usize;
                    let mut bd = (r - labels[0]).abs();
                    for (k, l) in labels.iter().enumerate().skip(1) {
                        let d = (r - l).abs();
                        if d < bd {
                            bd = d;
                            best = k;
                        }
                    }
                    if bd > 1e-6 {
                        return Err(err("unknown_multiclass_label"));
                    }
                    Ok(best as f64)
                })
                .collect();
        }
        self.transform(y)
    }

    /// Map scaled target codes back to fitted raw class labels (multiclass), `{0.0, 1.0}`
    /// (binary classification), or raw regression scale.
    pub fn inverse_transform(&self, y_scaled: &[f64]) -> Vec<f64> {
        if self.task == PrepTask::Multiclass && !self.class_labels.is_empty() {
            let max_idx = self.class_labels.len() - 1;
            return y_scaled
                .iter()
                .map(|&v| {
                    let idx = (v.round() as isize).clamp(0, max_idx as isize) as usize;
                    self.class_labels[idx]
                })
                .collect();
        }
        if self.task == PrepTask::Classification {
            return y_scaled
                .iter()
                .map(|&v| if v >= 0.5 { 1.0 } else { 0.0 })
                .collect();
        }
        y_scaled
            .iter()
            .map(|&v| {
                let u = clip(v / 8.0, -0.999999, 0.999999);
                self.mean + self.std * 8.0 * u.atanh()
            })
            .collect()
    }
}

/// `scaling.FitOnlyTableTransform` (fitted state).
#[derive(Clone, Debug, PartialEq)]
pub struct FitOnlyTableTransform {
    pub task: PrepTask,
    pub kinds: Vec<ColumnKind>,
    pub numeric: Vec<Option<NumericColumnScaler>>,
    pub categorical: Vec<Option<CategoricalColumnEncoder>>,
    pub target: TargetScaler,
    pub min_level_count: usize,
    pub prior_smoothing: f64,
    pub class_vocab_policy: ClassVocabPolicy,
}

pub(crate) fn fit_multiclass_target_scaler(
    y: &[TargetValue<'_>],
    policy: ClassVocabPolicy,
) -> TargetScaler {
    let finite: Vec<f64> = y
        .iter()
        .map(|v| match v {
            TargetValue::Num(x) => *x,
            TargetValue::Str(s) => parse_float_cell(s),
        })
        .filter(|v| v.is_finite())
        .collect();
    if finite.is_empty() {
        let mut ts = TargetScaler::base(PrepTask::Multiclass, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
        ts.class_labels = vec![0.0, 0.5, 1.0];
        ts.class_priors = vec![0.34, 0.33, 0.33];
        return ts;
    }
    let rounded = sorted(&finite.iter().map(|&v| round6(v)).collect::<Vec<_>>());
    let (mut u_vals, mut u_counts) = (Vec::new(), Vec::new());
    for v in rounded {
        if u_vals.last() == Some(&v) {
            *u_counts.last_mut().expect("count") += 1;
        } else {
            u_vals.push(v);
            u_counts.push(1i64);
        }
    }
    let counts_f: Vec<f64> = if policy == ClassVocabPolicy::Cap32V1 && u_vals.len() > 32 {
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
    let (m, s, med) = (mean1d(&finite), std1d(&finite).max(1e-6), median(&finite));
    let mut ts = TargetScaler::base(PrepTask::Multiclass, m, s, med, 1.0, 0.0, 0.0);
    ts.class_prior = priors[0];
    ts.class_labels = u_vals;
    ts.class_priors = priors;
    ts
}

//! Intake: infer column types and fit the invertible map from raw cells to
//! DOPE's *model space*.
//!
//! * Numeric columns: optional `log1p` for heavily right-skewed non-negative
//!   data, then a robust affine map that sends the 0.5% and 99.5% quantiles to
//!   0 and 1 (tails are kept, not clipped, inside the model).
//! * Categorical columns: levels with at least `min_level_count` rows, ordered
//!   by their (shrunken) association with the target, coded on [0, 1]. Rare
//!   levels are pooled into `__other__`, so no singleton label is stored.
//! * The target is numeric (regression) or categorical (classification).

use crate::error::{DopeError, Result};
use crate::stats::{mean, quantile_sorted};
use crate::table::{is_missing, parse_number, RawTable};
use serde::{Deserialize, Serialize};

pub const OTHER: &str = "__other__";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    Regression,
    Classification,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NumericSpec {
    /// Model value is `(t(x) - lo) / (hi - lo)` with `t = log1p` when `log`.
    pub log: bool,
    pub lo: f64,
    pub hi: f64,
    /// All observed values were integers; decoded samples are rounded.
    pub integer: bool,
}

impl NumericSpec {
    pub fn to_model(&self, x: f64) -> f64 {
        let t = if self.log { x.max(0.0).ln_1p() } else { x };
        (t - self.lo) / (self.hi - self.lo)
    }
    pub fn to_raw(&self, u: f64) -> f64 {
        let t = self.lo + u * (self.hi - self.lo);
        let x = if self.log { t.exp_m1().max(0.0) } else { t };
        if self.integer {
            x.round()
        } else {
            x
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CategoricalSpec {
    /// Levels in model order; code of level k is k / (L - 1).
    pub levels: Vec<String>,
}

impl CategoricalSpec {
    pub fn code(&self, level: usize) -> f64 {
        if self.levels.len() <= 1 {
            0.5
        } else {
            level as f64 / (self.levels.len() - 1) as f64
        }
    }
    pub fn level_of(&self, cell: &str) -> Option<usize> {
        self.levels
            .iter()
            .position(|l| l == cell)
            .or_else(|| self.levels.iter().position(|l| l == OTHER))
    }
    pub fn level_from_code(&self, code: f64) -> usize {
        let l = self.levels.len();
        if l <= 1 {
            0
        } else {
            (code * (l - 1) as f64).round().clamp(0.0, (l - 1) as f64) as usize
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Kind {
    Numeric(NumericSpec),
    Categorical(CategoricalSpec),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColumnSpec {
    pub name: String,
    #[serde(flatten)]
    pub kind: Kind,
}

impl ColumnSpec {
    /// Raw cell -> model value (NaN when missing or unparseable).
    pub fn to_model(&self, cell: &str) -> f64 {
        if is_missing(cell) {
            return f64::NAN;
        }
        match &self.kind {
            Kind::Numeric(spec) => parse_number(cell).map_or(f64::NAN, |x| spec.to_model(x)),
            Kind::Categorical(spec) => spec.level_of(cell).map_or(f64::NAN, |k| spec.code(k)),
        }
    }

    /// Model value -> raw cell text (empty for missing).
    pub fn to_raw(&self, value: f64) -> String {
        if value.is_nan() {
            return String::new();
        }
        match &self.kind {
            Kind::Numeric(spec) => format_number(spec.to_raw(value)),
            Kind::Categorical(spec) => spec.levels[spec.level_from_code(value)].clone(),
        }
    }
}

pub fn format_number(x: f64) -> String {
    if x == x.trunc() && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        let s = format!("{x:.6}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Schema {
    pub task: Task,
    pub features: Vec<ColumnSpec>,
    pub target: ColumnSpec,
    /// Input columns excluded from the program (identifier-like or empty).
    pub dropped: Vec<String>,
}

impl Schema {
    pub fn classes(&self) -> usize {
        match &self.target.kind {
            Kind::Categorical(c) => c.levels.len(),
            Kind::Numeric(_) => 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IntakeOptions {
    pub task: Option<Task>,
    pub min_level_count: usize,
    pub max_levels: usize,
}

impl Default for IntakeOptions {
    fn default() -> Self {
        Self {
            task: None,
            min_level_count: 5,
            max_levels: 64,
        }
    }
}

/// Model-space data: feature columns (NaN = missing) and the target
/// (unit-scaled value for regression, class index for classification).
#[derive(Clone, Debug)]
pub struct ModelData {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
}

impl ModelData {
    pub fn rows(&self) -> usize {
        self.y.len()
    }
}

fn numeric_cells(cells: &[String]) -> (Vec<f64>, usize) {
    let mut values = Vec::with_capacity(cells.len());
    let mut present = 0;
    for c in cells {
        if !is_missing(c) {
            present += 1;
            if let Some(v) = parse_number(c) {
                values.push(v);
            }
        }
    }
    (values, present)
}

fn is_numeric(cells: &[String]) -> bool {
    let (values, present) = numeric_cells(cells);
    present > 0 && values.len() as f64 >= 0.98 * present as f64
}

fn skewness(v: &[f64]) -> f64 {
    let m = mean(v);
    let s2 = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len().max(1) as f64;
    if s2 <= 0.0 {
        return 0.0;
    }
    v.iter().map(|x| (x - m).powi(3)).sum::<f64>() / v.len() as f64 / s2.powf(1.5)
}

pub fn fit_numeric(values: &[f64]) -> NumericSpec {
    let integer = values.iter().all(|v| v.fract() == 0.0);
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let log = min >= 0.0 && skewness(values) > 2.0;
    let mut t: Vec<f64> = values
        .iter()
        .map(|&x| if log { x.ln_1p() } else { x })
        .collect();
    t.sort_by(f64::total_cmp);
    let (mut lo, mut hi) = (quantile_sorted(&t, 0.005), quantile_sorted(&t, 0.995));
    if hi - lo <= 1e-12 * (1.0 + lo.abs()) {
        lo = t[0];
        hi = t[t.len() - 1];
    }
    if hi - lo <= 1e-12 * (1.0 + lo.abs()) {
        hi = lo + 1.0;
    }
    NumericSpec {
        log,
        lo,
        hi,
        integer,
    }
}

/// Order categorical levels by a shrunken per-level target score.
fn fit_categorical(cells: &[String], score: &[f64], opts: &IntakeOptions) -> CategoricalSpec {
    let mut counts: std::collections::BTreeMap<&str, (usize, f64)> = Default::default();
    for (c, s) in cells.iter().zip(score) {
        if !is_missing(c) {
            let e = counts.entry(c.as_str()).or_insert((0, 0.0));
            e.0 += 1;
            if s.is_finite() {
                e.1 += s;
            }
        }
    }
    let mut by_count: Vec<(&str, usize, f64)> =
        counts.into_iter().map(|(k, (n, s))| (k, n, s)).collect();
    by_count.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let global = mean(
        &score
            .iter()
            .copied()
            .filter(|s| s.is_finite())
            .collect::<Vec<_>>(),
    );
    let mut kept = Vec::new();
    let (mut other_n, mut other_s) = (0usize, 0.0);
    for (i, (level, n, s)) in by_count.into_iter().enumerate() {
        if n >= opts.min_level_count && i < opts.max_levels {
            kept.push((level.to_string(), n, s));
        } else {
            other_n += n;
            other_s += s;
        }
    }
    if other_n > 0 {
        kept.push((OTHER.to_string(), other_n, other_s));
    }
    const PRIOR: f64 = 10.0;
    kept.sort_by(|a, b| {
        let sa = (a.2 + PRIOR * global) / (a.1 as f64 + PRIOR);
        let sb = (b.2 + PRIOR * global) / (b.1 as f64 + PRIOR);
        sa.total_cmp(&sb).then(a.0.cmp(&b.0))
    });
    CategoricalSpec {
        levels: kept.into_iter().map(|k| k.0).collect(),
    }
}

fn fit_target_classes(cells: &[String]) -> Result<CategoricalSpec> {
    let mut levels: Vec<String> = cells.iter().filter(|c| !is_missing(c)).cloned().collect();
    levels.sort();
    levels.dedup();
    if levels.len() < 2 {
        return Err(DopeError::Input(
            "classification target needs at least two classes".into(),
        ));
    }
    if levels.len() > 256 {
        return Err(DopeError::Input(
            "classification target has more than 256 classes".into(),
        ));
    }
    Ok(CategoricalSpec { levels })
}

/// Infer the schema of `raw` with `target` as the prediction column.
pub fn infer(raw: &RawTable, target: usize, opts: &IntakeOptions) -> Result<Schema> {
    let tcells = &raw.columns[target];
    let task = opts.task.unwrap_or_else(|| {
        let (values, _) = numeric_cells(tcells);
        let mut u = values.clone();
        u.sort_by(f64::total_cmp);
        u.dedup();
        if is_numeric(tcells) && u.len() > 2 {
            Task::Regression
        } else {
            Task::Classification
        }
    });
    let (target_spec, score): (ColumnSpec, Vec<f64>) = match task {
        Task::Regression => {
            let (values, _) = numeric_cells(tcells);
            if values.len() < 2 || !is_numeric(tcells) {
                return Err(DopeError::Input("regression target must be numeric".into()));
            }
            let spec = fit_numeric(&values);
            let score = tcells
                .iter()
                .map(|c| parse_number(c).map_or(f64::NAN, |x| spec.to_model(x)))
                .collect();
            let name = raw.headers[target].clone();
            (
                ColumnSpec {
                    name,
                    kind: Kind::Numeric(spec),
                },
                score,
            )
        }
        Task::Classification => {
            let spec = fit_target_classes(tcells)?;
            // Score levels by the indicator of the most frequent class.
            let mut freq = vec![0usize; spec.levels.len()];
            for c in tcells {
                if let Some(k) = spec.levels.iter().position(|l| l == c) {
                    freq[k] += 1;
                }
            }
            let major = (0..freq.len()).max_by_key(|&k| freq[k]).unwrap_or(0);
            let score = tcells
                .iter()
                .map(|c| match spec.levels.iter().position(|l| l == c) {
                    Some(k) => (k == major) as u8 as f64,
                    None => f64::NAN,
                })
                .collect();
            let name = raw.headers[target].clone();
            (
                ColumnSpec {
                    name,
                    kind: Kind::Categorical(spec),
                },
                score,
            )
        }
    };
    let rows = raw.rows();
    let mut features = Vec::new();
    let mut dropped = Vec::new();
    for (c, cells) in raw.columns.iter().enumerate() {
        if c == target {
            continue;
        }
        let name = raw.headers[c].clone();
        let present = cells.iter().filter(|x| !is_missing(x)).count();
        if present == 0 {
            dropped.push(name);
            continue;
        }
        if is_numeric(cells) {
            let (values, _) = numeric_cells(cells);
            features.push(ColumnSpec {
                name,
                kind: Kind::Numeric(fit_numeric(&values)),
            });
            continue;
        }
        let mut distinct: Vec<&str> = cells.iter().map(String::as_str).collect();
        distinct.sort_unstable();
        distinct.dedup();
        // Identifier-like text (mostly unique) carries no transferable structure.
        if distinct.len() > 100 && distinct.len() as f64 > 0.5 * rows as f64 {
            dropped.push(name);
            continue;
        }
        features.push(ColumnSpec {
            name,
            kind: Kind::Categorical(fit_categorical(cells, &score, opts)),
        });
    }
    if features.is_empty() {
        return Err(DopeError::Input("no usable feature columns".into()));
    }
    Ok(Schema {
        task,
        features,
        target: target_spec,
        dropped,
    })
}

impl Schema {
    /// Map raw rows to model space (features by name, so column order may differ).
    pub fn to_model(&self, raw: &RawTable) -> Result<ModelData> {
        let mut x = Vec::with_capacity(self.features.len());
        for f in &self.features {
            let c = raw
                .headers
                .iter()
                .position(|h| *h == f.name)
                .ok_or_else(|| {
                    DopeError::Input("a feature column is missing from the input".into())
                })?;
            x.push(raw.columns[c].iter().map(|cell| f.to_model(cell)).collect());
        }
        let y = match raw.headers.iter().position(|h| *h == self.target.name) {
            Some(c) => raw.columns[c]
                .iter()
                .map(|cell| match &self.target.kind {
                    Kind::Numeric(_) => self.target.to_model(cell),
                    Kind::Categorical(spec) => {
                        if is_missing(cell) {
                            f64::NAN
                        } else {
                            spec.levels
                                .iter()
                                .position(|l| l == cell)
                                .map_or(f64::NAN, |k| k as f64)
                        }
                    }
                })
                .collect(),
            None => vec![f64::NAN; raw.rows()],
        };
        Ok(ModelData { x, y })
    }

    pub fn target_to_raw(&self, value: f64) -> String {
        match &self.target.kind {
            Kind::Numeric(_) => self.target.to_raw(value),
            Kind::Categorical(spec) => {
                spec.levels[(value as usize).min(spec.levels.len() - 1)].clone()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(text: &str) -> RawTable {
        RawTable::from_reader(text.as_bytes()).unwrap()
    }

    #[test]
    fn infers_types_and_task() {
        let mut csv = String::from("size,color,price\n");
        for i in 0..60 {
            let color = ["red", "blue", "green"][i % 3];
            csv += &format!("{},{},{}\n", i, color, i * 10 + (i % 3) * 100);
        }
        let raw = table(&csv);
        let s = infer(&raw, 2, &IntakeOptions::default()).unwrap();
        assert_eq!(s.task, Task::Regression);
        assert!(matches!(s.features[0].kind, Kind::Numeric(_)));
        match &s.features[1].kind {
            Kind::Categorical(c) => assert_eq!(c.levels, vec!["red", "blue", "green"]),
            _ => panic!("color should be categorical"),
        }
        let m = s.to_model(&raw).unwrap();
        assert_eq!(m.x.len(), 2);
        assert_eq!(s.features[1].to_raw(m.x[1][1]), "blue");
    }

    #[test]
    fn numeric_roundtrip_with_log() {
        let values: Vec<f64> = (0..500)
            .map(|i| ((i as f64) / 40.0).exp().round())
            .collect();
        let spec = fit_numeric(&values);
        assert!(spec.log && spec.integer);
        for &v in values.iter().step_by(37) {
            assert_eq!(spec.to_raw(spec.to_model(v)), v);
        }
    }

    #[test]
    fn rare_levels_are_pooled() {
        let mut csv = String::from("c,y\n");
        for i in 0..40 {
            csv += &format!("{},{}\n", if i < 38 { "a" } else { "rare" }, i % 2);
        }
        let s = infer(&table(&csv), 1, &IntakeOptions::default()).unwrap();
        assert_eq!(s.task, Task::Classification);
        match &s.features[0].kind {
            Kind::Categorical(c) => assert!(c.levels.contains(&OTHER.to_string())),
            _ => panic!(),
        }
    }
}

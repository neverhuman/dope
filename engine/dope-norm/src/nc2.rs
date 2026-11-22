//! NC2 FIT-only scaling (`nc2-bulk-tails-v1`) beside frozen `nc1-minmax-onehot-v1`.
//!
//! - Numeric features: `asinh((x - median)/MAD)` skew-guard iff tail ratio
//!   `(q99 - q50)/(q50 - q01) not in [1/8, 8]`, then piecewise-linear `[q0.5%, q99.5%]`
//!   (clamped at `k`-th order stats, never raw min/max) -> `[0.02, 0.98]` with unclipped
//!   linear tail extension into `[0, 0.02]` / `[0.98, 1.0]` and beyond for `TEST`.
//! - Regression target: strictly affine (never piecewise, zero Jensen bias).
//! - Nominals: frequency-ordered opaque `u32` IDs, rare levels `< k=10` pooled to `unknown`,
//!   missingness always represented as a first-class state.

use crate::{digest, Cell, DeclaredSchema, FeatureKind, InputTable, NormError, Result, Task};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const RECIPE_NC2: &str = "nc2-bulk-tails-v1";
pub const NC2_SUPPORT_K: usize = 10;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Nc2NumericMap {
    pub use_asinh: bool,
    pub median: f64,
    pub mad: f64,
    pub t_lo_tail: f64,
    pub t_lo_bulk: f64,
    pub t_hi_bulk: f64,
    pub t_hi_tail: f64,
    pub constant: bool,
}

impl Nc2NumericMap {
    fn skew_transform(&self, x: f64) -> f64 {
        if self.use_asinh {
            ((x - self.median) / self.mad).asinh()
        } else {
            x
        }
    }

    fn skew_invert(&self, t: f64) -> f64 {
        if self.use_asinh {
            self.median + self.mad * t.sinh()
        } else {
            t
        }
    }

    /// Map raw value into canonical unit coordinates with unclipped linear tail extension.
    pub fn forward(&self, x: f64) -> f64 {
        if self.constant {
            return 0.5;
        }
        let t = self.skew_transform(x);
        if t < self.t_lo_bulk {
            let span = (self.t_lo_bulk - self.t_lo_tail).max(1e-9);
            0.02 - 0.02 * (self.t_lo_bulk - t) / span
        } else if t > self.t_hi_bulk {
            let span = (self.t_hi_tail - self.t_hi_bulk).max(1e-9);
            0.98 + 0.02 * (t - self.t_hi_bulk) / span
        } else {
            let span = (self.t_hi_bulk - self.t_lo_bulk).max(1e-9);
            0.02 + 0.96 * (t - self.t_lo_bulk) / span
        }
    }

    /// Exact inverse of `forward` across bulk AND unclipped out-of-range tails!
    pub fn inverse(&self, u: f64) -> f64 {
        if self.constant {
            return self.median;
        }
        let t = if u < 0.02 {
            let span = (self.t_lo_bulk - self.t_lo_tail).max(1e-9);
            self.t_lo_bulk - ((0.02 - u) / 0.02) * span
        } else if u > 0.98 {
            let span = (self.t_hi_tail - self.t_hi_bulk).max(1e-9);
            self.t_hi_bulk + ((u - 0.98) / 0.02) * span
        } else {
            let span = (self.t_hi_bulk - self.t_lo_bulk).max(1e-9);
            self.t_lo_bulk + ((u - 0.02) / 0.96) * span
        };
        self.skew_invert(t)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Nc2NominalMap {
    /// Frequency-ordered kept private categories (`count >= k`), mapped to opaque IDs `1..=K`.
    /// Rare levels `< k` and unseen TEST labels map to `0` (`unknown`); missing -> `None`.
    kept_levels: Vec<String>,
    pub support_k: usize,
}

impl Nc2NominalMap {
    pub fn encode(&self, label: &str) -> u32 {
        self.kept_levels
            .iter()
            .position(|l| l == label)
            .map_or(0, |idx| (idx + 1) as u32)
    }

    pub fn cardinality_with_unknown(&self) -> usize {
        self.kept_levels.len() + 1
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Nc2FeatureMap {
    Numeric(Nc2NumericMap),
    Nominal(Nc2NominalMap),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Nc2TargetMap {
    /// Strictly affine target map `(y - offset) / scale` (never piecewise).
    Affine {
        offset: f64,
        scale: f64,
    },
    Binary {
        positive_label: String,
    },
    Multiclass {
        labels: Vec<String>,
    },
}

/// Public, metadata-hidden NC2 schema (ZERO column names, category strings, or raw extrema!).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Nc2PublicGroup {
    pub group_id: usize,
    pub kind: String,
    pub width: usize,
    pub has_missing_slot: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Nc2PublicSchema {
    pub recipe: String,
    pub support_k: usize,
    pub feature_groups: Vec<Nc2PublicGroup>,
    pub target_kind: String,
    pub target_classes: Option<usize>,
    pub schema_sha256: String,
}

/// Owner-private NC2 ScaleCard (never leaves owner boundary).
#[derive(Clone, Serialize, Deserialize)]
pub struct Nc2ScaleCard {
    pub recipe: String,
    pub source_sha256: String,
    pub support_k: usize,
    feature_cols: Vec<usize>,
    features: Vec<Nc2FeatureMap>,
    target_col: usize,
    target: Nc2TargetMap,
    pub public_schema: Nc2PublicSchema,
}

#[derive(Clone, Debug)]
pub struct Nc2UnitTable {
    pub x_cols: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub row_ids: Vec<u64>,
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let pos = p.clamp(0.0, 1.0) * (n - 1) as f64;
    let i = pos.floor() as usize;
    let j = (i + 1).min(n - 1);
    let f = pos - i as f64;
    sorted[i] * (1.0 - f) + sorted[j] * f
}

fn fit_numeric_nc2(values: &[f64], support_k: usize) -> Nc2NumericMap {
    let mut s = values.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n == 0 {
        return Nc2NumericMap {
            use_asinh: false,
            median: 0.0,
            mad: 1.0,
            t_lo_tail: -1.0,
            t_lo_bulk: 0.0,
            t_hi_bulk: 1.0,
            t_hi_tail: 2.0,
            constant: true,
        };
    }
    let med = quantile(&s, 0.5);
    let mut abs_dev: Vec<f64> = s.iter().map(|&v| (v - med).abs()).collect();
    abs_dev.sort_by(f64::total_cmp);
    let mad = (quantile(&abs_dev, 0.5) * 1.4826).max(1e-9);
    let q01 = quantile(&s, 0.01);
    let q99 = quantile(&s, 0.99);
    let lo_arm = (med - q01).max(1e-12);
    let hi_arm = (q99 - med).max(1e-12);
    let tail_ratio = hi_arm / lo_arm;
    let use_asinh = !(0.125..=8.0).contains(&tail_ratio);

    let mut ts: Vec<f64> = s
        .iter()
        .map(|&x| {
            if use_asinh {
                ((x - med) / mad).asinh()
            } else {
                x
            }
        })
        .collect();
    ts.sort_by(f64::total_cmp);
    let k_ord = support_k.min(n / 10).max(1).min(n.saturating_sub(1));
    let t_lo_tail = ts[k_ord];
    let t_hi_tail = ts[n - 1 - k_ord];
    let t_lo_bulk = quantile(&ts, 0.005).max(t_lo_tail);
    let t_hi_bulk = quantile(&ts, 0.995).min(t_hi_tail);
    let constant = (t_hi_bulk - t_lo_bulk).abs() <= 1e-12 && (t_hi_tail - t_lo_tail).abs() <= 1e-12;
    let t_lo_tail = if (t_lo_bulk - t_lo_tail).abs() <= 1e-12 {
        t_lo_bulk - (t_hi_bulk - t_lo_bulk).max(1.0) * 0.02
    } else {
        t_lo_tail
    };
    let t_hi_tail = if (t_hi_tail - t_hi_bulk).abs() <= 1e-12 {
        t_hi_bulk + (t_hi_bulk - t_lo_bulk).max(1.0) * 0.02
    } else {
        t_hi_tail
    };
    Nc2NumericMap {
        use_asinh,
        median: med,
        mad,
        t_lo_tail,
        t_lo_bulk,
        t_hi_bulk,
        t_hi_tail,
        constant,
    }
}

impl Nc2ScaleCard {
    /// Fit NC2 strictly on owner-private `BUILD` rows (`support_k >= 10`).
    pub fn fit(table: &InputTable, schema: &DeclaredSchema, support_k: usize) -> Result<Self> {
        table.validate()?;
        let support_k = support_k.max(NC2_SUPPORT_K);
        let ncols = table.columns.len();
        if schema.target.column >= ncols {
            return Err(NormError::new("invalid_target_column"));
        }
        let mut feature_cols = Vec::with_capacity(schema.features.len());
        let mut features = Vec::with_capacity(schema.features.len());
        let mut public_groups = Vec::with_capacity(schema.features.len());

        for (gid, fspec) in schema.features.iter().enumerate() {
            if fspec.column >= ncols || fspec.column == schema.target.column {
                return Err(NormError::at("invalid_feature_column", gid));
            }
            feature_cols.push(fspec.column);
            let cells = &table.columns[fspec.column].values;
            match &fspec.kind {
                FeatureKind::Continuous | FeatureKind::Integer | FeatureKind::Boolean => {
                    let mut vals = Vec::new();
                    for cell in cells {
                        match cell {
                            Cell::Number(v) if v.is_finite() => vals.push(*v),
                            Cell::Boolean(b) => vals.push(if *b { 1.0 } else { 0.0 }),
                            Cell::Missing => {}
                            _ => return Err(NormError::at("type_mismatch", gid)),
                        }
                    }
                    features.push(Nc2FeatureMap::Numeric(fit_numeric_nc2(&vals, support_k)));
                    public_groups.push(Nc2PublicGroup {
                        group_id: gid,
                        kind: "numeric".to_string(),
                        width: 2, // unit value + first-class missingness indicator
                        has_missing_slot: true,
                    });
                }
                FeatureKind::Nominal | FeatureKind::Ordinal { .. } => {
                    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
                    for cell in cells {
                        match cell {
                            Cell::Category(s) => *counts.entry(s.as_str()).or_insert(0) += 1,
                            Cell::Missing => {}
                            _ => return Err(NormError::at("type_mismatch", gid)),
                        }
                    }
                    // Frequency-ordered opaque codes; rare levels `< support_k` pooled to unknown (0).
                    let mut freq: Vec<(&str, usize)> = counts
                        .into_iter()
                        .filter(|&(_, c)| c >= support_k)
                        .collect();
                    freq.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
                    let kept_levels: Vec<String> =
                        freq.into_iter().map(|(s, _)| s.to_string()).collect();
                    let width = kept_levels.len() + 2; // unknown + kept + missing slot
                    features.push(Nc2FeatureMap::Nominal(Nc2NominalMap {
                        kept_levels,
                        support_k,
                    }));
                    public_groups.push(Nc2PublicGroup {
                        group_id: gid,
                        kind: "nominal_onehot".to_string(),
                        width,
                        has_missing_slot: true,
                    });
                }
            }
        }

        // Fit strictly affine regression target or discrete classification target.
        let t_cells = &table.columns[schema.target.column].values;
        let (target_map, target_kind, target_classes) = match &schema.target.task {
            Task::Regression => {
                let mut ys = Vec::new();
                for cell in t_cells {
                    match cell {
                        Cell::Number(v) if v.is_finite() => ys.push(*v),
                        _ => return Err(NormError::new("invalid_regression_target")),
                    }
                }
                if ys.len() < 2 {
                    return Err(NormError::new("degenerate_target"));
                }
                let mean = ys.iter().sum::<f64>() / ys.len() as f64;
                let sd = (ys.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / ys.len() as f64)
                    .sqrt()
                    .max(1e-9);
                (
                    Nc2TargetMap::Affine {
                        offset: mean,
                        scale: sd,
                    },
                    "regression_affine".to_string(),
                    None,
                )
            }
            Task::Binary { positive_label } => (
                Nc2TargetMap::Binary {
                    positive_label: positive_label.clone(),
                },
                "binary".to_string(),
                Some(2),
            ),
            Task::Multiclass { labels } => (
                Nc2TargetMap::Multiclass {
                    labels: labels.clone(),
                },
                "multiclass".to_string(),
                Some(labels.len()),
            ),
        };

        let schema_sha256 = digest(&(&public_groups, &target_kind, &target_classes))?;
        let public_schema = Nc2PublicSchema {
            recipe: RECIPE_NC2.to_string(),
            support_k,
            feature_groups: public_groups,
            target_kind,
            target_classes,
            schema_sha256,
        };
        Ok(Self {
            recipe: RECIPE_NC2.to_string(),
            source_sha256: table.source_sha256.clone(),
            support_k,
            feature_cols,
            features,
            target_col: schema.target.column,
            target: target_map,
            public_schema,
        })
    }

    /// Transform any table (BUILD/SELECT/TEST) strictly using frozen FIT-only NC2 parameters.
    pub fn apply(&self, table: &InputTable) -> Result<Nc2UnitTable> {
        table.validate()?;
        let n = table.row_ids.len();
        let mut x_cols = Vec::new();
        for (gid, (&col_idx, fmap)) in self.feature_cols.iter().zip(&self.features).enumerate() {
            let cells = &table.columns[col_idx].values;
            match fmap {
                Nc2FeatureMap::Numeric(nm) => {
                    let mut vals = Vec::with_capacity(n);
                    let mut miss = Vec::with_capacity(n);
                    for cell in cells {
                        match cell {
                            Cell::Number(v) if v.is_finite() => {
                                vals.push(nm.forward(*v));
                                miss.push(0.0);
                            }
                            Cell::Boolean(b) => {
                                vals.push(nm.forward(if *b { 1.0 } else { 0.0 }));
                                miss.push(0.0);
                            }
                            Cell::Missing => {
                                vals.push(0.5);
                                miss.push(1.0);
                            }
                            _ => return Err(NormError::at("type_mismatch", gid)),
                        }
                    }
                    x_cols.push(vals);
                    x_cols.push(miss);
                }
                Nc2FeatureMap::Nominal(nom) => {
                    let k = nom.cardinality_with_unknown();
                    let mut onehot: Vec<Vec<f64>> = (0..(k + 1)).map(|_| vec![0.0; n]).collect();
                    for (r, cell) in cells.iter().enumerate() {
                        match cell {
                            Cell::Category(s) => {
                                let code = nom.encode(s) as usize;
                                onehot[code][r] = 1.0;
                            }
                            Cell::Missing => onehot[k][r] = 1.0,
                            _ => return Err(NormError::at("type_mismatch", gid)),
                        }
                    }
                    x_cols.extend(onehot);
                }
            }
        }
        let mut y = Vec::with_capacity(n);
        for cell in &table.columns[self.target_col].values {
            match (&self.target, cell) {
                (Nc2TargetMap::Affine { offset, scale }, Cell::Number(v)) if v.is_finite() => {
                    y.push((*v - offset) / scale);
                }
                (Nc2TargetMap::Binary { positive_label }, Cell::Category(s)) => {
                    y.push(if s == positive_label { 1.0 } else { 0.0 });
                }
                (Nc2TargetMap::Multiclass { labels }, Cell::Category(s)) => {
                    let idx = labels
                        .iter()
                        .position(|l| l == s)
                        .ok_or_else(|| NormError::new("unknown_target_class"))?;
                    y.push(idx as f64);
                }
                _ => return Err(NormError::new("invalid_target_cell")),
            }
        }
        Ok(Nc2UnitTable {
            x_cols,
            y,
            row_ids: table.row_ids.clone(),
        })
    }

    pub fn inverse_numeric(&self, feature_group: usize, u: f64) -> Option<f64> {
        match self.features.get(feature_group)? {
            Nc2FeatureMap::Numeric(nm) => Some(nm.inverse(u)),
            _ => None,
        }
    }

    pub fn inverse_target(&self, y_unit: f64) -> Option<f64> {
        match &self.target {
            Nc2TargetMap::Affine { offset, scale } => Some(offset + scale * y_unit),
            _ => None,
        }
    }
}

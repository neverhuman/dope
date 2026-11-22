//! Owner-side scale card: the map between raw table units and the unit space
//! the packet lives in. It is never part of the packet.
//!
//! `to_unit` reproduces `_fit_only_nc2_scale` of
//! `research/levels/s3_dynamic_augment.py`: per column `median` and
//! `scale = max(IQR / 1.349, population sd, 1e-6)` over the finite FIT values,
//! `z = (x - median) / scale`, `sign(z) * (3 + asinh(|z| - 3))` beyond `|z| > 3`,
//! clip to +-6, non-finite cells map to 0. A column with fewer than two finite
//! FIT values maps to 0 everywhere. `to_raw` is the exact inverse on (-6, 6);
//! a FIT-constant column returns its constant, and a column whose FIT values
//! all sit on a decimal grid is snapped back to that grid.

use super::StatError;
use serde::{Deserialize, Serialize};

pub const SCHEMA: &str = "dope.scale-card/v1";
const UNIT_CLIP: f64 = 6.0;
const TAIL_START: f64 = 3.0;
/// Candidate grid steps, tried from the coarsest.
const GRID_STEPS: [f64; 10] = [1000.0, 100.0, 10.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001];

/// Exact text form of an `f64` (shortest representation that parses back to the same bits).
mod f64_text {
    use serde::{de::Error, Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&format!("{v:?}"))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        let text = String::deserialize(d)?;
        text.parse()
            .map_err(|_| D::Error::custom("not a float literal"))
    }
}

mod f64_text_list {
    use serde::{de::Error, Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &[f64], s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(v.iter().map(|x| format!("{x:?}")))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
        let items = Vec::<String>::deserialize(d)?;
        items
            .iter()
            .map(|t| {
                t.parse()
                    .map_err(|_| D::Error::custom("not a float literal"))
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColumnScale {
    #[serde(with = "f64_text")]
    pub median: f64,
    /// 0 marks a column with fewer than two finite FIT values.
    #[serde(with = "f64_text")]
    pub scale: f64,
    /// Every finite FIT value is the same; `to_raw` returns `median`.
    pub constant: bool,
    /// Decimal grid step of the FIT values, 0 if none was found.
    #[serde(with = "f64_text")]
    pub grid: f64,
}

/// numpy's linear interpolation between two order statistics.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    if t >= 0.5 {
        b - (b - a) * (1.0 - t)
    } else {
        a + (b - a) * t
    }
}

/// `numpy.percentile(sorted, 100 * q)` with the default linear method.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let h = q * (sorted.len() - 1) as f64;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    lerp(sorted[lo], sorted[hi], h - lo as f64)
}

/// Spacing of `f32` values at `|x|`.
fn ulp32(x: f64) -> f64 {
    let f = x.abs() as f32;
    if !f.is_finite() {
        return f64::INFINITY;
    }
    (f32::from_bits(f.to_bits() + 1) - f) as f64
}

/// Coarsest candidate step on which every value sits (to within the `f32`
/// rounding of the source), provided the grid is resolvable at the largest value.
fn detect_grid(sorted: &[f64]) -> f64 {
    let top = sorted[0].abs().max(sorted[sorted.len() - 1].abs());
    for step in GRID_STEPS {
        if step < 8.0 * ulp32(top) {
            continue;
        }
        let on_grid = |x: &f64| {
            let g = (x / step).round() * step;
            (x - g).abs() <= (0.5001 * ulp32(*x)).max(8.0 * f64::EPSILON * x.abs())
        };
        if sorted.iter().all(on_grid) {
            return step;
        }
    }
    0.0
}

impl ColumnScale {
    /// Fits one column from its FIT cells (non-finite cells are ignored).
    pub fn fit(cells: impl Iterator<Item = f64>) -> Self {
        let mut fin: Vec<f64> = cells.filter(|v| v.is_finite()).collect();
        if fin.len() < 2 {
            return Self {
                median: fin.first().copied().unwrap_or(0.0),
                scale: 0.0,
                constant: true,
                grid: 0.0,
            };
        }
        fin.sort_by(f64::total_cmp);
        let n = fin.len();
        let median = if n % 2 == 1 {
            fin[n / 2]
        } else {
            (fin[n / 2 - 1] + fin[n / 2]) / 2.0
        };
        let iqr = percentile(&fin, 0.75) - percentile(&fin, 0.25);
        let mean = fin.iter().sum::<f64>() / n as f64;
        let var = fin.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        let constant = fin[0] == fin[n - 1];
        Self {
            median,
            scale: (iqr / 1.349).max(var.sqrt()).max(1e-6),
            constant,
            grid: if constant { 0.0 } else { detect_grid(&fin) },
        }
    }

    /// Raw value to unit space (non-finite input maps to 0, the median).
    pub fn to_unit(&self, x: f64) -> f64 {
        if self.scale == 0.0 || !x.is_finite() {
            return 0.0;
        }
        let z = (x - self.median) / self.scale;
        let u = if z.abs() > TAIL_START {
            z.signum() * (TAIL_START + (z.abs() - TAIL_START).asinh())
        } else {
            z
        };
        u.clamp(-UNIT_CLIP, UNIT_CLIP)
    }

    /// Unit value back to raw units.
    pub fn to_raw(&self, u: f64) -> f64 {
        if self.constant {
            return self.median;
        }
        let u = u.clamp(-UNIT_CLIP, UNIT_CLIP);
        let z = if u.abs() > TAIL_START {
            u.signum() * (TAIL_START + (u.abs() - TAIL_START).sinh())
        } else {
            u
        };
        let x = self.median + self.scale * z;
        if self.grid > 0.0 {
            (x / self.grid).round() * self.grid
        } else {
            x
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TargetScale {
    /// No target (feature-only use).
    None,
    /// `unit = clip((y - mean) / sd, +-6)`, `sd = max(population sd, 1e-5)`.
    Regression {
        #[serde(with = "f64_text")]
        mean: f64,
        #[serde(with = "f64_text")]
        sd: f64,
    },
    /// Class index `c` stands for raw label `labels[c]`.
    Classes {
        #[serde(with = "f64_text_list")]
        labels: Vec<f64>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScaleCard {
    pub schema: String,
    pub columns: Vec<ColumnScale>,
    pub target: TargetScale,
}

impl ScaleCard {
    /// Fits the feature part from raw FIT rows (`x` row-major `n x p`).
    pub fn fit(x: &[f64], n: usize, p: usize) -> Result<Self, StatError> {
        if p == 0 || n == 0 || x.len() != n * p {
            return Err(StatError::Invalid(
                "feature matrix shape does not match its dimensions",
            ));
        }
        let columns = (0..p)
            .map(|j| ColumnScale::fit((0..n).map(|i| x[i * p + j])))
            .collect();
        Ok(Self {
            schema: SCHEMA.to_string(),
            columns,
            target: TargetScale::None,
        })
    }

    /// Adds the regression target scale fitted on the FIT targets.
    pub fn with_regression_target(mut self, y: &[f64]) -> Result<Self, StatError> {
        if y.is_empty() || y.iter().any(|v| !v.is_finite()) {
            return Err(StatError::Invalid(
                "regression target is empty or not finite",
            ));
        }
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let var = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        self.target = TargetScale::Regression {
            mean,
            sd: var.sqrt().max(1e-5),
        };
        Ok(self)
    }

    /// Adds the class map: class index `c` stands for `labels[c]`.
    pub fn with_classes(mut self, labels: Vec<f64>) -> Self {
        self.target = TargetScale::Classes { labels };
        self
    }

    /// All rows to unit space (`x` row-major, any number of rows).
    pub fn to_unit_rows(&self, x: &[f64]) -> Result<Vec<f64>, StatError> {
        let p = self.columns.len();
        if p == 0 || x.len() % p != 0 {
            return Err(StatError::Invalid(
                "feature matrix shape does not match the scale card",
            ));
        }
        Ok(x.iter()
            .enumerate()
            .map(|(i, &v)| self.columns[i % p].to_unit(v))
            .collect())
    }

    /// Regression target to unit space; identity for other targets.
    pub fn target_to_unit(&self, y: f64) -> f64 {
        match &self.target {
            TargetScale::Regression { mean, sd } => ((y - mean) / sd).clamp(-UNIT_CLIP, UNIT_CLIP),
            _ => y,
        }
    }

    /// Unit target back to raw units: regression rescales, classes look up the label.
    pub fn target_to_raw(&self, t: f64) -> f64 {
        match &self.target {
            TargetScale::Regression { mean, sd } => mean + sd * t,
            TargetScale::Classes { labels } => {
                let c = (t.max(0.0).round() as usize).min(labels.len().saturating_sub(1));
                labels.get(c).copied().unwrap_or(t)
            }
            TargetScale::None => t,
        }
    }

    pub fn to_json(&self) -> Result<String, StatError> {
        serde_json::to_string_pretty(self).map_err(|_| StatError::Invalid("scale card encoding"))
    }

    pub fn from_json(text: &str) -> Result<Self, StatError> {
        let card: Self = serde_json::from_str(text)
            .map_err(|_| StatError::Invalid("scale card is not valid JSON of the schema"))?;
        if card.schema != SCHEMA {
            return Err(StatError::Invalid("unknown scale card schema"));
        }
        Ok(card)
    }
}

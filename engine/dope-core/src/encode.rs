//! ML-ready encoding: one-hot categoricals, numeric columns on [0, 1].
//!
//! * `Scaling::Unit` (default): log1p for heavily skewed non-negative
//!   columns, then the robust 0.5%–99.5% quantile range mapped to [0, 1] and
//!   clipped — linear structure is kept and one outlier cannot squash the
//!   column (plain min–max can).
//! * `Scaling::Quantile`: the probability integral transform F(x), uniform on
//!   [0, 1]; best for distance- and gradient-based learners on heavy tails.
//!
//! Missing values become the column median plus a `name=__missing__`
//! indicator (only for columns that had missing values).

use crate::marginal::Marginal;
use crate::program::Program;
use crate::schema::{Kind, Schema};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaling {
    Unit,
    Quantile,
}

impl Scaling {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "unit" | "robust" | "01" => Some(Self::Unit),
            "quantile" | "pit" | "uniform" => Some(Self::Quantile),
            _ => None,
        }
    }
}

pub struct MlEncoder<'a> {
    pub schema: &'a Schema,
    pub marginals: &'a [Marginal],
    pub missing: &'a [f64],
    pub scaling: Scaling,
}

impl<'a> MlEncoder<'a> {
    pub fn new(program: &'a Program, scaling: Scaling) -> Self {
        Self {
            schema: &program.schema,
            marginals: &program.marginals,
            missing: &program.missing,
            scaling,
        }
    }

    pub fn headers(&self) -> Vec<String> {
        let mut h = Vec::new();
        for (j, f) in self.schema.features.iter().enumerate() {
            match &f.kind {
                Kind::Numeric(_) => h.push(f.name.clone()),
                Kind::Categorical(c) => {
                    h.extend(c.levels.iter().map(|l| format!("{}={}", f.name, l)))
                }
            }
            if self.missing[j] > 0.0 {
                h.push(format!("{}=__missing__", f.name));
            }
        }
        h.push(self.schema.target.name.clone());
        h
    }

    fn numeric(&self, j: usize, u: f64) -> f64 {
        match self.scaling {
            Scaling::Unit => u.clamp(0.0, 1.0),
            Scaling::Quantile => self.marginals[j].cdf(u),
        }
    }

    /// Encode model-space rows (`x[feature][row]`, target `y`) row-major.
    pub fn rows(&self, x: &[Vec<f64>], y: &[f64]) -> Vec<Vec<f64>> {
        let n = y.len();
        let width = self.headers().len();
        (0..n)
            .map(|r| {
                let mut row = Vec::with_capacity(width);
                for (j, f) in self.schema.features.iter().enumerate() {
                    let v = x[j][r];
                    match &f.kind {
                        Kind::Numeric(_) => {
                            let u = if v.is_finite() {
                                v
                            } else {
                                self.marginals[j].at(0.0)
                            };
                            row.push(self.numeric(j, u));
                        }
                        Kind::Categorical(c) => {
                            let level = v.is_finite().then(|| c.level_from_code(v));
                            row.extend(
                                (0..c.levels.len()).map(|k| (level == Some(k)) as u8 as f64),
                            );
                        }
                    }
                    if self.missing[j] > 0.0 {
                        row.push((!v.is_finite()) as u8 as f64);
                    }
                }
                row.push(y[r]);
                row
            })
            .collect()
    }
}

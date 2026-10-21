//! A compiled DOPE program: everything sampling and prediction need, and
//! nothing else (no rows, no row references).

use crate::copula::Dependence;
use crate::forest::{code, Forest};
use crate::marginal::Marginal;
use crate::noise::Noise;
use crate::regimes::Regimes;
use crate::schema::{Kind, Schema, Task};
use crate::target::{Head, Net};

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub schema: Schema,
    /// One marginal per feature, in model space.
    pub marginals: Vec<Marginal>,
    pub missing: Vec<f64>,
    pub dependence: Dependence,
    /// Regime mixture over the shared dependence (K = 1: plain Gaussian copula).
    pub regimes: Regimes,
    /// Feature indices feeding the conditional model.
    pub screened: Vec<usize>,
    pub net: Net,
    /// Boosted oblivious trees added to the network's output (may be empty).
    pub forest: Forest,
    pub head: Head,
    pub noise: Noise,
    /// Regression draws are clamped to this model-space interval.
    pub target_range: (f64, f64),
}

impl Program {
    pub fn features(&self) -> usize {
        self.marginals.len()
    }

    /// Model-space value used for a missing input of the conditional model.
    pub fn impute(&self, feature: usize) -> f64 {
        self.marginals[feature].at(0.0)
    }

    /// Centred, imputed inputs of the conditional model for model-space rows
    /// (`x[feature][row]`).
    pub fn net_inputs(&self, x: &[Vec<f64>], rows: usize) -> Vec<f64> {
        let s = self.screened.len();
        let mut out = vec![0.0; rows * s];
        for (k, &f) in self.screened.iter().enumerate() {
            let fill = self.impute(f);
            for r in 0..rows {
                let v = x[f][r];
                out[r * s + k] = if v.is_finite() { v } else { fill } - 0.5;
            }
        }
        out
    }

    /// Raw outputs `[rows, o]` of the conditional model (network plus forest)
    /// for model-space rows: predictions for regression, logits otherwise.
    pub fn outputs(&self, x: &[Vec<f64>], rows: usize) -> Vec<f64> {
        let xc = self.net_inputs(x, rows);
        let mut out = self.net.forward(&xc, rows);
        if self.forest.trees() > 0 {
            let d = self.features();
            let o = self.net.outputs;
            let fill: Vec<f64> = (0..d).map(|f| self.impute(f)).collect();
            let mut row = vec![0u8; d];
            for r in 0..rows {
                for f in 0..d {
                    let v = x[f][r];
                    row[f] = code(if v.is_finite() { v } else { fill[f] });
                }
                self.forest.add(&row, &mut out[r * o..(r + 1) * o]);
            }
        }
        out
    }

    /// Conditional expectation (regression, model space) or class
    /// probabilities (classification) for model-space rows.
    pub fn predict_model(&self, x: &[Vec<f64>], rows: usize) -> Vec<Vec<f64>> {
        let out = self.outputs(x, rows);
        match self.head {
            Head::Regression => out.into_iter().map(|v| vec![v]).collect(),
            Head::Binary => {
                let tau = self.temperature();
                out.into_iter()
                    .map(|z| {
                        let p = 1.0 / (1.0 + (-z / tau).exp());
                        vec![1.0 - p, p]
                    })
                    .collect()
            }
            Head::Multiclass(k) => {
                let tau = self.temperature();
                out.chunks(k)
                    .map(|row| {
                        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let e: Vec<f64> = row.iter().map(|z| ((z - m) / tau).exp()).collect();
                        let total: f64 = e.iter().sum();
                        e.into_iter().map(|v| v / total).collect()
                    })
                    .collect()
            }
        }
    }

    pub fn temperature(&self) -> f64 {
        match self.noise {
            Noise::Temperature(t) => t,
            _ => 1.0,
        }
    }

    /// Raw-unit point prediction text for a regression output (no rounding).
    pub fn regression_to_raw(&self, value: f64) -> f64 {
        match &self.schema.target.kind {
            Kind::Numeric(spec) => {
                let t = spec.lo + value * (spec.hi - spec.lo);
                if spec.log {
                    t.exp_m1()
                } else {
                    t
                }
            }
            Kind::Categorical(_) => value,
        }
    }

    pub fn head_for(schema: &Schema) -> Head {
        match schema.task {
            Task::Regression => Head::Regression,
            Task::Classification if schema.classes() == 2 => Head::Binary,
            Task::Classification => Head::Multiclass(schema.classes()),
        }
    }
}

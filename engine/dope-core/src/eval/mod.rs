//! Retention: how much of a real-trained model's gain over a null survives
//! when the same auditor is trained only on synthetic rows.
//!
//! ```text
//! R = (L_null − L_syn) / (L_null − L_real)      on held-out real rows
//! ```
//!
//! Losses are MSE (regression, model-space target) or log-loss
//! (classification). Auditor families: `linear` (ridge / logistic /
//! softmax), `gbdt` (histogram boosted trees) and `mlp`. The robust score
//! is the minimum over informative families, as in the TDL definition.

mod audit;
mod frontier;
pub mod gbdt;
pub mod linear;
pub mod nn;

pub use audit::SELF_AUDIT_ROWS;
pub(crate) use audit::{audit_predict, SelfAudit};
pub use frontier::{frontier, FrontierOptions, FrontierPoint, FrontierResult};

use crate::artifact::decode;
use crate::copula::Dependence;
use crate::encode::{MlEncoder, Scaling};
use crate::error::{DopeError, Result};
use crate::marginal::Marginal;
use crate::noise::Noise;
use crate::program::Program;
use crate::sample::{sample, SampleOptions};
use crate::schema::{Kind, ModelData, Schema};
use crate::table::RawTable;
use crate::target::{Head, Net};
use candle_core::Device;
use serde::Serialize;

/// Auditors are trained on at most this many rows (real and synthetic alike).
pub const MAX_AUDIT_ROWS: usize = 50_000;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
    pub fn at(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }
}

/// Loss of raw scores (gbdt internals): regression values, binary logits,
/// multiclass logits.
pub(crate) fn loss_from_scores(head: Head, s: &[f64], y: &[f64]) -> f64 {
    crate::target::loss(head, s, y)
}

/// Loss of predictions: values (regression) or probabilities.
pub fn prediction_loss(head: Head, pred: &[f64], y: &[f64]) -> f64 {
    let n = y.len().max(1) as f64;
    let clip = |p: f64| p.clamp(1e-7, 1.0 - 1e-7);
    match head {
        Head::Regression => {
            pred.iter()
                .zip(y)
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / n
        }
        Head::Binary => {
            pred.iter()
                .zip(y)
                .map(|(p, t)| -(t * clip(*p).ln() + (1.0 - t) * (1.0 - clip(*p)).ln()))
                .sum::<f64>()
                / n
        }
        Head::Multiclass(k) => {
            y.iter()
                .enumerate()
                .map(|(r, t)| -clip(pred[r * k + (*t as usize).min(k - 1)]).ln())
                .sum::<f64>()
                / n
        }
    }
}

/// A program that only carries a schema and marginals (for encoding data
/// without a fitted artifact).
pub fn schema_program(schema: &Schema, table: &RawTable) -> Result<Program> {
    let data = schema.to_model(table)?;
    let d = schema.features.len();
    let marginals = (0..d)
        .map(|j| {
            let levels = match &schema.features[j].kind {
                Kind::Categorical(c) => Some(c.levels.len()),
                Kind::Numeric(_) => None,
            };
            Marginal::fit(&data.x[j], 33, levels)
        })
        .collect();
    let missing = data
        .x
        .iter()
        .map(|c| c.iter().filter(|v| !v.is_finite()).count() as f64 / c.len().max(1) as f64)
        .collect();
    let head = Program::head_for(schema);
    let o = head.outputs();
    Ok(Program {
        schema: schema.clone(),
        marginals,
        missing,
        dependence: Dependence::Independent,
        regimes: crate::regimes::Regimes::single(d),
        screened: vec![],
        net: Net {
            inputs: 0,
            hidden: 0,
            outputs: o,
            w: vec![],
            b: vec![],
            beta: vec![],
            v: vec![],
            a: vec![0.0; o],
        },
        forest: crate::forest::Forest::empty(o),
        head,
        noise: Noise::Temperature(1.0),
        target_range: (0.0, 1.0),
    })
}

#[derive(Clone, Debug)]
pub struct EvalOptions {
    pub auditors: Vec<String>,
    pub factor: f64,
    pub seed: u64,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            auditors: vec!["linear".into(), "gbdt".into(), "mlp".into()],
            factor: 4.0,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct AuditorResult {
    pub auditor: String,
    pub null_loss: f64,
    pub real_loss: f64,
    pub synth_loss: f64,
    pub retention: f64,
    pub informative: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct EvalResult {
    pub task: String,
    pub artifact_bytes: usize,
    pub train_rows: usize,
    pub test_rows: usize,
    pub synth_rows: usize,
    pub auditors: Vec<AuditorResult>,
    /// Minimum retention over informative auditor families.
    pub robust_retention: Option<f64>,
}

impl EvalResult {
    pub fn render(&self) -> String {
        let mut s = format!(
            "retention on {} held-out real rows ({} task, artifact {} bytes, {} synthetic vs {} real training rows)\n",
            self.test_rows, self.task, self.artifact_bytes, self.synth_rows, self.train_rows
        );
        s += "  auditor   null loss   real loss  synth loss  retention\n";
        for a in &self.auditors {
            s += &format!(
                "  {:<8} {:>10.5} {:>11.5} {:>11.5} {:>10}\n",
                a.auditor,
                a.null_loss,
                a.real_loss,
                a.synth_loss,
                if a.informative {
                    format!("{:.3}", a.retention)
                } else {
                    "n/a".into()
                }
            );
        }
        if let Some(r) = self.robust_retention {
            s += &format!("  robust retention (min over families): {r:.3}\n");
        }
        s
    }
}

pub(crate) fn to_matrix(
    enc: &MlEncoder<'_>,
    data: &ModelData,
    rows: &[usize],
) -> (Matrix, Vec<f64>) {
    let x: Vec<Vec<f64>> = data
        .x
        .iter()
        .map(|c| rows.iter().map(|&r| c[r]).collect())
        .collect();
    let y: Vec<f64> = rows.iter().map(|&r| data.y[r]).collect();
    let encoded = enc.rows(&x, &y);
    let cols = encoded.first().map_or(0, |r| r.len() - 1);
    let mut m = Vec::with_capacity(rows.len() * cols);
    for r in &encoded {
        m.extend_from_slice(&r[..cols]);
    }
    (
        Matrix {
            rows: rows.len(),
            cols,
            data: m,
        },
        y,
    )
}

fn observed(data: &ModelData, cap: usize) -> Vec<usize> {
    let rows: Vec<usize> = (0..data.rows())
        .filter(|&r| data.y[r].is_finite())
        .collect();
    let stride = rows.len().div_ceil(cap).max(1);
    rows.into_iter().step_by(stride).collect()
}

fn run_auditor(
    name: &str,
    head: Head,
    x: &Matrix,
    y: &[f64],
    test: &Matrix,
    seed: u64,
) -> Result<Vec<f64>> {
    let dev = Device::Cpu;
    Ok(match (name, head) {
        ("linear", Head::Multiclass(_)) => {
            nn::fit_predict(x, y, head, test, &nn::Params::softmax_linear(seed), &dev)?
        }
        ("linear", _) => linear::fit_predict(x, y, head, test),
        ("gbdt", _) => gbdt::fit_predict(
            x,
            y,
            head,
            test,
            &gbdt::Params {
                seed,
                ..Default::default()
            },
        ),
        ("mlp", _) => nn::fit_predict(x, y, head, test, &nn::Params::mlp(seed), &dev)?,
        _ => {
            return Err(DopeError::Input(
                "auditors must be linear, gbdt or mlp".into(),
            ))
        }
    })
}

/// Evaluate an artifact: real-train vs synthetic-train auditors on `test`.
pub fn evaluate(
    model: &[u8],
    train: &RawTable,
    test: &RawTable,
    synth: Option<&RawTable>,
    opts: &EvalOptions,
) -> Result<EvalResult> {
    let program = decode(model)?;
    let head = program.head;
    let enc = MlEncoder::new(&program, Scaling::Unit);
    let real = program.schema.to_model(train)?;
    let held = program.schema.to_model(test)?;
    let real_rows = observed(&real, MAX_AUDIT_ROWS);
    let test_rows = observed(&held, usize::MAX);
    let syn = match synth {
        Some(t) => program.schema.to_model(t)?,
        None => {
            let n =
                ((real_rows.len() as f64 * opts.factor).round() as usize).clamp(1, MAX_AUDIT_ROWS);
            let s = sample(
                &program,
                &SampleOptions {
                    rows: n,
                    seed: opts.seed ^ 0x5a,
                    ..Default::default()
                },
            )?;
            ModelData { x: s.x, y: s.y }
        }
    };
    let syn_rows = observed(&syn, MAX_AUDIT_ROWS);
    let (xr, yr) = to_matrix(&enc, &real, &real_rows);
    let (xs, ys) = to_matrix(&enc, &syn, &syn_rows);
    let (xt, yt) = to_matrix(&enc, &held, &test_rows);
    let null_pred: Vec<f64> = match head {
        Head::Regression => vec![yr.iter().sum::<f64>() / yr.len().max(1) as f64; yt.len()],
        Head::Binary => vec![yr.iter().sum::<f64>() / yr.len().max(1) as f64; yt.len()],
        Head::Multiclass(k) => {
            let mut c = vec![0.5; k];
            yr.iter().for_each(|t| c[(*t as usize).min(k - 1)] += 1.0);
            let s: f64 = c.iter().sum();
            (0..yt.len())
                .flat_map(|_| c.iter().map(move |v| v / s))
                .collect()
        }
    };
    let null_loss = prediction_loss(head, &null_pred, &yt);
    let mut auditors = Vec::new();
    for name in &opts.auditors {
        let real_pred = run_auditor(name, head, &xr, &yr, &xt, opts.seed)?;
        let syn_pred = run_auditor(name, head, &xs, &ys, &xt, opts.seed)?;
        let real_loss = prediction_loss(head, &real_pred, &yt);
        let synth_loss = prediction_loss(head, &syn_pred, &yt);
        let gain = null_loss - real_loss;
        auditors.push(AuditorResult {
            auditor: name.clone(),
            null_loss,
            real_loss,
            synth_loss,
            retention: if gain > 0.0 {
                (null_loss - synth_loss) / gain
            } else {
                f64::NAN
            },
            informative: gain >= 0.01 * null_loss.abs() && gain > 0.0,
        });
    }
    let informative: Vec<f64> = auditors
        .iter()
        .filter(|a| a.informative)
        .map(|a| a.retention)
        .collect();
    Ok(EvalResult {
        task: format!("{:?}", program.schema.task).to_lowercase(),
        artifact_bytes: model.len(),
        train_rows: real_rows.len(),
        test_rows: test_rows.len(),
        synth_rows: syn_rows.len(),
        robust_retention: informative.iter().copied().reduce(f64::min),
        auditors,
    })
}

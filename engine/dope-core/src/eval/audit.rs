//! Compile-time self-audit of candidate programs.

use super::{gbdt, prediction_loss, run_auditor, to_matrix, Matrix};
use crate::encode::{MlEncoder, Scaling};
use crate::error::Result;
use crate::program::Program;
use crate::sample::{sample, SampleOptions};
use crate::schema::ModelData;
use crate::target::Head;

/// Rows used by the compile-time self-audit (real and synthetic alike).
pub const SELF_AUDIT_ROWS: usize = 6000;

/// Compile-time self-audit: the robust retention of fast linear and boosted-
/// tree auditors trained on a candidate program's rows and scored on the
/// compiler's own validation rows (never on external test data). The real
/// reference is trained once on the fitting rows.
pub(crate) struct SelfAudit {
    head: Head,
    xv: Matrix,
    yv: Vec<f64>,
    null: f64,
    real: Vec<(&'static str, f64)>,
    rows: usize,
    seed: u64,
}

const AUDIT_FAMILIES: [&str; 2] = ["linear", "gbdt"];

fn audit_predict(
    name: &str,
    head: Head,
    x: &Matrix,
    y: &[f64],
    test: &Matrix,
    seed: u64,
) -> Result<Vec<f64>> {
    if name == "gbdt" {
        let p = gbdt::Params {
            rounds: 150,
            learning_rate: 0.12,
            seed,
            ..Default::default()
        };
        return Ok(gbdt::fit_predict(x, y, head, test, &p));
    }
    run_auditor(name, head, x, y, test, seed)
}

impl SelfAudit {
    pub(crate) fn new(
        program: &Program,
        data: &ModelData,
        fit_rows: &[usize],
        val_rows: &[usize],
        seed: u64,
    ) -> Result<Self> {
        let head = program.head;
        let enc = MlEncoder::new(program, Scaling::Unit);
        let stride = |n: usize| n.div_ceil(SELF_AUDIT_ROWS).max(1);
        let fit: Vec<usize> = fit_rows
            .iter()
            .step_by(stride(fit_rows.len()))
            .copied()
            .collect();
        let val: Vec<usize> = val_rows
            .iter()
            .step_by(stride(val_rows.len()))
            .copied()
            .collect();
        let (xr, yr) = to_matrix(&enc, data, &fit);
        let (xv, yv) = to_matrix(&enc, data, &val);
        let null_pred: Vec<f64> = match head {
            Head::Regression | Head::Binary => {
                vec![yr.iter().sum::<f64>() / yr.len().max(1) as f64; yv.len()]
            }
            Head::Multiclass(k) => {
                let mut c = vec![0.5; k];
                yr.iter().for_each(|t| c[(*t as usize).min(k - 1)] += 1.0);
                let s: f64 = c.iter().sum();
                (0..yv.len())
                    .flat_map(|_| c.iter().map(move |v| v / s))
                    .collect()
            }
        };
        let null = prediction_loss(head, &null_pred, &yv);
        let mut real = Vec::new();
        for name in AUDIT_FAMILIES {
            let pred = audit_predict(name, head, &xr, &yr, &xv, seed)?;
            real.push((name, prediction_loss(head, &pred, &yv)));
        }
        Ok(Self {
            head,
            xv,
            yv,
            null,
            real,
            rows: fit.len(),
            seed,
        })
    }

    /// Robust retention of `program` (1.0 when no family is informative).
    pub(crate) fn retention(&self, program: &Program) -> Result<f64> {
        let s = sample(
            program,
            &SampleOptions {
                rows: self.rows,
                seed: self.seed ^ 0xa0d1,
                ..Default::default()
            },
        )?;
        let syn = ModelData { x: s.x, y: s.y };
        let rows: Vec<usize> = (0..syn.y.len()).filter(|&r| syn.y[r].is_finite()).collect();
        let enc = MlEncoder::new(program, Scaling::Unit);
        let (xs, ys) = to_matrix(&enc, &syn, &rows);
        let mut robust = f64::INFINITY;
        for &(name, real) in &self.real {
            let gain = self.null - real;
            if !(gain > 0.0 && gain >= 0.01 * self.null.abs()) {
                continue;
            }
            let pred = audit_predict(name, self.head, &xs, &ys, &self.xv, self.seed)?;
            let r = (self.null - prediction_loss(self.head, &pred, &self.yv)) / gain;
            robust = robust.min(r);
        }
        Ok(if robust.is_finite() { robust } else { 1.0 })
    }
}

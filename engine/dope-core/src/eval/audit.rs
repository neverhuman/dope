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

/// Result of compile-time self-audit: distinguishes informative tasks from tasks where no
/// reference auditor improves over the null predictor by at least 1%.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AuditOutcome {
    Informative(f64),
    Uninformative,
}

impl AuditOutcome {
    pub fn score(self) -> Option<f64> {
        match self {
            AuditOutcome::Informative(v) if v.is_finite() => Some(v),
            _ => None,
        }
    }
}

const AUDIT_FAMILIES: [&str; 2] = ["linear-seeded-v1", "gbdt"];

pub(crate) fn audit_predict(
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
            group_duplicates: true,
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

    /// Robust retention of `program`, returning [`AuditOutcome::Uninformative`] instead of
    /// coercing to `1.0` when no reference family beats the null predictor by at least 1%.
    pub(crate) fn retention_outcome(&self, program: &Program) -> Result<AuditOutcome> {
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
        Ok(if robust.is_finite() {
            AuditOutcome::Informative(robust)
        } else {
            AuditOutcome::Uninformative
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{fit, FitOptions};
    use crate::eval::duplicate_grouped_split;
    use crate::rng::Rng;
    use crate::table::RawTable;
    use std::collections::BTreeSet;

    fn raw_from_rows(headers: &[&str], rows: Vec<Vec<String>>) -> RawTable {
        let mut csv = String::new();
        csv.push_str(&headers.join(","));
        csv.push('\n');
        for row in rows {
            csv.push_str(&row.join(","));
            csv.push('\n');
        }
        RawTable::from_reader(csv.as_bytes()).unwrap()
    }

    #[test]
    fn duplicate_grouped_split_never_straddles_exact_duplicates() {
        let mut data = Vec::new();
        let mut y = Vec::new();
        for g in 0..40usize {
            let xv = [g as f64 / 40.0, ((g * 7) % 13) as f64 / 13.0];
            let yv = (g % 2) as f64;
            for _ in 0..6 {
                data.extend_from_slice(&xv);
                y.push(yv);
            }
        }
        let m = Matrix {
            rows: y.len(),
            cols: 2,
            data,
        };
        let (val, fit_idx) = duplicate_grouped_split(&m, &y, 10, 100, 42);
        assert!(!val.is_empty() && !fit_idx.is_empty());
        assert_eq!(val.len() + fit_idx.len(), m.rows);
        let key = |r: usize| -> (u64, u64, u64) {
            (m.at(r, 0).to_bits(), m.at(r, 1).to_bits(), y[r].to_bits())
        };
        let val_keys: BTreeSet<_> = val.iter().map(|&r| key(r)).collect();
        for &r in &fit_idx {
            assert!(
                !val_keys.contains(&key(r)),
                "exact duplicate row straddled fit/val split"
            );
        }
    }

    #[test]
    fn self_audit_marks_pure_noise_uninformative_instead_of_one() {
        let mut rng = Rng::new(77);
        let mut rows = Vec::new();
        for _ in 0..240 {
            let x0 = rng.normal();
            let x1 = rng.normal();
            let y = rng.normal();
            rows.push(vec![
                format!("{x0:.6}"),
                format!("{x1:.6}"),
                format!("{y:.6}"),
            ]);
        }
        let raw = raw_from_rows(&["x0", "x1", "y"], rows);
        let opts = FitOptions {
            budget: Some(4096),
            seed: 13,
            max_steps: 100,
            ..Default::default()
        };
        let fitted = fit(&raw, 2, &opts).unwrap();
        assert_ne!(fitted.report.self_audit, Some(1.0));
        assert!(fitted.report.self_audit.is_none());
    }

    #[test]
    fn compile_is_bit_identical_across_reruns_for_reg_bin_and_multiclass() {
        let mut rng = Rng::new(2026);
        let mut reg_rows = Vec::new();
        let mut bin_rows = Vec::new();
        let mut multi_rows = Vec::new();
        for _ in 0..220 {
            let x0 = rng.normal();
            let x1 = rng.normal();
            let x2 = rng.normal();
            let signal = 1.5 * x0 - 0.8 * x1 + 0.6 * x0 * x2;
            let yr = signal + 0.1 * rng.normal();
            let yb = if signal > 0.0 { 1 } else { 0 };
            let ym = if signal < -0.5 {
                "A"
            } else if signal < 0.5 {
                "B"
            } else {
                "C"
            };
            reg_rows.push(vec![
                format!("{x0:.5}"),
                format!("{x1:.5}"),
                format!("{x2:.5}"),
                format!("{yr:.5}"),
            ]);
            bin_rows.push(vec![
                format!("{x0:.5}"),
                format!("{x1:.5}"),
                format!("{x2:.5}"),
                yb.to_string(),
            ]);
            multi_rows.push(vec![
                format!("{x0:.5}"),
                format!("{x1:.5}"),
                format!("{x2:.5}"),
                ym.to_string(),
            ]);
        }
        let opts = FitOptions {
            budget: Some(4096),
            seed: 19,
            max_steps: 120,
            ..Default::default()
        };
        for rows in [reg_rows, bin_rows, multi_rows] {
            let raw = raw_from_rows(&["x0", "x1", "x2", "y"], rows);
            let a = fit(&raw, 3, &opts).unwrap();
            let b = fit(&raw, 3, &opts).unwrap();
            assert_eq!(a.bytes, b.bytes, "compiled artifact bytes drifted on rerun");
            assert_eq!(a.report.self_audit, b.report.self_audit);
        }
    }
}

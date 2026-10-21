//! The retention-versus-bytes frontier and Transfer Description Length.

use super::{evaluate, EvalOptions};
use crate::compile::{fit, FitOptions};
use crate::error::{DopeError, Result};
use crate::table::RawTable;
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct FrontierOptions {
    pub budgets: Vec<usize>,
    pub tau: f64,
    pub eval: EvalOptions,
}

#[derive(Clone, Debug, Serialize)]
pub struct FrontierPoint {
    pub budget: usize,
    pub bytes: usize,
    pub robust_retention: Option<f64>,
    pub retention: Vec<(String, f64)>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FrontierResult {
    pub tau: f64,
    pub points: Vec<FrontierPoint>,
    /// Transfer Description Length: smallest tested budget whose robust
    /// retention reaches `tau` (None = not reached in the tested range).
    pub tdl_bytes: Option<usize>,
}

impl FrontierResult {
    pub fn render(&self) -> String {
        let mut s = String::from("  budget     bytes   robust retention   per auditor\n");
        for p in &self.points {
            let per: Vec<String> = p
                .retention
                .iter()
                .map(|(a, r)| format!("{a} {r:.3}"))
                .collect();
            s += &format!(
                "  {:>7} {:>8}   {:>16}   {}\n",
                p.budget,
                p.bytes,
                p.robust_retention
                    .map_or("n/a".into(), |r| format!("{r:.3}")),
                per.join(" · ")
            );
        }
        s += &match self.tdl_bytes {
            Some(b) => format!(
                "  TDL@{:.0}% = {} bytes (smallest tested budget reaching the threshold)\n",
                self.tau * 100.0,
                b
            ),
            None => format!(
                "  TDL@{:.0}% not reached in the tested range\n",
                self.tau * 100.0
            ),
        };
        s
    }
}

/// Compile at each budget, evaluate, and report the TDL.
pub fn frontier(
    train: &RawTable,
    target: usize,
    test: &RawTable,
    opts: &FrontierOptions,
) -> Result<FrontierResult> {
    let mut points = Vec::new();
    for &budget in &opts.budgets {
        let fitted = match fit(
            train,
            target,
            &FitOptions {
                budget: Some(budget),
                seed: opts.eval.seed,
                ..Default::default()
            },
        ) {
            Ok(f) => f,
            Err(DopeError::Budget(_)) => continue,
            Err(e) => return Err(e),
        };
        let r = evaluate(&fitted.bytes, train, test, None, &opts.eval)?;
        points.push(FrontierPoint {
            budget,
            bytes: fitted.bytes.len(),
            robust_retention: r.robust_retention,
            retention: r
                .auditors
                .iter()
                .filter(|a| a.informative)
                .map(|a| (a.auditor.clone(), a.retention))
                .collect(),
        });
    }
    let tdl_bytes = points
        .iter()
        .find(|p| p.robust_retention.is_some_and(|r| r >= opts.tau))
        .map(|p| p.bytes);
    Ok(FrontierResult {
        tau: opts.tau,
        points,
        tdl_bytes,
    })
}

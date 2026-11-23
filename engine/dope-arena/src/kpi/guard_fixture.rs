//! Hand-made ledger entries for the tests of `dope_kpi guards`, with gates
//! that can be computed by hand.
//!
//! Real rows in unit space (one feature and the target 0): A = {0, 2, 4, 6}
//! and B = {1, 3, 5, 7}. A synthetic copy of B passes privacy: every
//! synthetic row is nearer B (DCR share 0), every A row is farther from S
//! than every B row (MIA AUC 0, TPR 0) and none is an exact copy. A copy of A
//! fails it: DCR share 1, MIA AUC 1 and four exact copies of four rows.
//! `B_DUP` = {0, 3, 5, 7} shares one row with A: a copy of it holds one
//! exact copy of A, as the twin does, so its own excess is 0.
#![allow(dead_code)]

use dope_core::dvs::v3::bundle::{PredictionBundle, TaskKind};
use dope_core::dvs::v3::privacy::EmpiricalPrivacyInput;
use dope_core::dvs::v3::realism::RealismInput;
use dope_core::dvs::v3::{FidelityCellInput, FidelityCellRequest};

pub const REAL_A: [f64; 4] = [0.0, 2.0, 4.0, 6.0];
pub const REAL_B: [f64; 4] = [1.0, 3.0, 5.0, 7.0];
pub const B_DUP: [f64; 4] = [0.0, 3.0, 5.0, 7.0];

fn rows(values: &[f64]) -> Vec<Vec<f64>> {
    values.iter().map(|&v| vec![v, 0.0]).collect()
}

/// A privacy input sized as `dope_kpi score` sizes it.
pub fn privacy(a: &[f64], b: &[f64], s: &[f64]) -> EmpiricalPrivacyInput {
    let matched = a.len().min(b.len()) as u64;
    EmpiricalPrivacyInput {
        real_fit_a: rows(a),
        real_holdout_b: rows(b),
        synthetic: rows(s),
        min_emitted_support: 10,
        canary_rows: Vec::new(),
        max_distance_pairs: 4 * matched * s.len() as u64,
    }
}

/// A realism input whose C2ST AUC is 0.5 (`pass`) or 1.0 (above the 0.70 bound).
pub fn realism(pass: bool) -> RealismInput {
    let scores = if pass {
        vec![0.1, 0.1, 0.2, 0.2]
    } else {
        vec![0.1, 0.9, 0.2, 0.8]
    };
    RealismInput {
        c2st_labels: vec![0, 1, 0, 1],
        c2st_scores: scores,
        marginal_aucs: vec![0.55, 0.5],
        synth_rule_violation_rate: None,
        real_rule_violation_rate: None,
        illegal_rows: 0,
    }
}

fn bundle() -> PredictionBundle {
    PredictionBundle {
        task: TaskKind::Regression,
        null_losses: vec![1.0, 1.0],
        real_a_losses: vec![0.5, 0.5],
        real_b_losses: Some(vec![0.5, 0.5]),
        synth_losses: vec![0.6, 0.6],
        donor_ids: None,
        y_true: vec![0.0, 1.0],
        pred_a: vec![0.0, 1.0],
        pred_b: Some(vec![0.0, 1.0]),
        pred_s: vec![0.0, 1.0],
        num_outputs: 1,
    }
}

pub fn complete(
    realism: Option<RealismInput>,
    privacy: Option<EmpiricalPrivacyInput>,
) -> FidelityCellInput {
    FidelityCellInput::Complete {
        bundle: Box::new(bundle()),
        driver: None,
        deploy_roster: None,
        realism: realism.map(Box::new),
        privacy: privacy.map(Box::new),
    }
}

/// A complete entry whose synthetic rows are `s` (realism passes).
pub fn scored(s: &[f64]) -> FidelityCellInput {
    complete(Some(realism(true)), Some(privacy(&REAL_A, &REAL_B, s)))
}

pub fn failed() -> FidelityCellInput {
    FidelityCellInput::Failed {
        reason: "missing_file".to_string(),
    }
}

pub fn request(
    ledger_id: &str,
    family: &str,
    auditor: &str,
    input: FidelityCellInput,
) -> FidelityCellRequest {
    FidelityCellRequest {
        cell_id: ledger_id.to_string(),
        family_id: family.to_string(),
        auditor_id: auditor.to_string(),
        input,
    }
}

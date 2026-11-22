//! `dope-fidelity/v3`: hard-gate coverage evaluation (zero weighted composites).
//!
//! Evaluates synthetic artifacts in one frozen FIT-only unit space across:
//! 1. Validity & Empirical Privacy (explicitly non-DP / non-HIPAA)
//! 2. Parity Coverage vs fresh real twin `B` (unclipped retention + donor-aware harm UCB95)
//! 3. Equivalence Coverage (`kappa` for pred / lift / calib / driver + `S_deploy`)
//! 4. Realism Coverage (`C2ST`, `MargAUC`, rule violations)

pub mod bundle;
pub mod c2st;
pub mod deploy;
pub mod equiv;
pub mod groups;
pub mod harm;
pub mod privacy;
pub mod qualify;
pub mod realism;

use crate::dvs::{ensure, DvsError};
use bundle::PredictionBundle;
use deploy::{assess_deploy_selection, DeploySelectionAssessment, RosterConfigScore};
use equiv::{assess_equivalence, DriverEquivalenceInput, EquivalenceAssessment, KappaThresholds};
use harm::{assess_parity, ParityAssessment};
use privacy::{assess_empirical_privacy, EmpiricalPrivacyAssessment, EmpiricalPrivacyInput};
use realism::{assess_realism, RealismAssessment, RealismInput};
use serde::{Deserialize, Serialize};

pub const PROTOCOL_V3: &str = "dope-fidelity/v3";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum FidelityCellInput {
    Complete {
        bundle: Box<PredictionBundle>,
        driver: Option<Box<DriverEquivalenceInput>>,
        deploy_roster: Option<Vec<RosterConfigScore>>,
        realism: Option<Box<RealismInput>>,
        privacy: Option<Box<EmpiricalPrivacyInput>>,
    },
    Failed {
        reason: String,
    },
    TimedOut,
    Unsupported,
    OverBudget {
        actual_bytes: u64,
        cap_bytes: u64,
    },
    Missing,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FidelityCellRequest {
    pub cell_id: String,
    pub family_id: String,
    pub auditor_id: String,
    pub input: FidelityCellInput,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FidelityV3Request {
    pub protocol: String,
    pub mode: String,
    pub seed: u64,
    pub bootstrap_draws: usize,
    pub kappa_thresholds: KappaThresholds,
    /// When `false` (default, mandatory for qualification), absent `driver`, `realism`, or
    /// `privacy` evidence fails closed instead of silently passing.
    #[serde(default)]
    pub allow_partial_diagnostic: bool,
    pub cells: Vec<FidelityCellRequest>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FidelityCellReceipt {
    pub cell_id: String,
    pub family_id: String,
    pub auditor_id: String,
    pub status: &'static str,
    pub parity: Option<ParityAssessment>,
    pub equivalence: Option<EquivalenceAssessment>,
    pub deploy: Option<DeploySelectionAssessment>,
    pub realism: Option<RealismAssessment>,
    pub privacy: Option<EmpiricalPrivacyAssessment>,
    pub all_hard_gates_pass: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CoverageSummary {
    pub scheduled_cells: usize,
    pub informative_cells: usize,
    pub uninformative_cells: usize,
    pub failed_or_missing_cells: usize,
    /// Denominator includes ALL scheduled cells except legitimately uninformative real tasks (`T18`).
    pub eligible_denominator: usize,
    pub parity_coverage: f64,
    pub equivalence_coverage: f64,
    pub realism_coverage: f64,
    pub privacy_pass_coverage: f64,
    pub overall_pass_coverage: f64,
    pub median_unclipped_retention: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FidelityV3Receipt {
    pub protocol: &'static str,
    pub mode: String,
    pub composite_score: &'static str,
    pub formal_dp_state: &'static str,
    pub coverage: CoverageSummary,
    pub cells: Vec<FidelityCellReceipt>,
}

pub fn evaluate_v3(req: &FidelityV3Request) -> Result<FidelityV3Receipt, DvsError> {
    ensure(
        req.protocol == PROTOCOL_V3,
        "unsupported_fidelity_v3_protocol",
    )?;
    ensure(
        matches!(req.mode.as_str(), "faithful" | "distill" | "matched"),
        "unsupported_emission_mode",
    )?;
    ensure(!req.cells.is_empty(), "empty_fidelity_v3_schedule")?;
    let draws = req.bootstrap_draws.max(100);
    let mut cell_receipts = Vec::with_capacity(req.cells.len());

    for (idx, cell) in req.cells.iter().enumerate() {
        let cell_seed = req.seed ^ ((idx as u64 + 1).wrapping_mul(0x517c_c1b7_2722_0a95));
        let receipt = match &cell.input {
            FidelityCellInput::Complete {
                bundle,
                driver,
                deploy_roster,
                realism,
                privacy,
            } => {
                let parity = assess_parity(bundle, draws, cell_seed)?;
                let equiv = assess_equivalence(
                    bundle,
                    driver.as_deref(),
                    &req.kappa_thresholds,
                    draws,
                    cell_seed ^ 0x100,
                )?;
                let deploy = deploy_roster
                    .as_deref()
                    .map(|r| assess_deploy_selection(r, 0.01))
                    .transpose()?;
                let real = realism.as_deref().map(assess_realism).transpose()?;
                let priv_a = privacy
                    .as_deref()
                    .map(assess_empirical_privacy)
                    .transpose()?;
                let status = if parity.informative {
                    "complete"
                } else {
                    "uninformative"
                };
                let partial = req.allow_partial_diagnostic;
                let driver_ok = if partial {
                    equiv.driver.as_ref().is_none_or(|d| d.pass)
                } else {
                    equiv.driver.as_ref().is_some_and(|d| d.pass)
                };
                let equiv_ok = equiv.pred.pass
                    && equiv.lift.pass
                    && equiv.calib.pass
                    && driver_ok
                    && deploy.as_ref().is_none_or(|d| d.pass);
                let real_ok = if partial {
                    real.as_ref().is_none_or(|r| r.pass_standard)
                } else {
                    real.as_ref().is_some_and(|r| r.pass_standard)
                };
                let priv_ok = if partial {
                    priv_a.as_ref().is_none_or(|p| p.pass)
                } else {
                    priv_a.as_ref().is_some_and(|p| p.pass)
                };
                let all_ok = parity.parity_pass && equiv_ok && real_ok && priv_ok;
                FidelityCellReceipt {
                    cell_id: cell.cell_id.clone(),
                    family_id: cell.family_id.clone(),
                    auditor_id: cell.auditor_id.clone(),
                    status,
                    parity: Some(parity),
                    equivalence: Some(equiv),
                    deploy,
                    realism: real,
                    privacy: priv_a,
                    all_hard_gates_pass: all_ok,
                }
            }
            FidelityCellInput::Failed { .. } => terminal_cell(cell, "failed"),
            FidelityCellInput::TimedOut => terminal_cell(cell, "timed_out"),
            FidelityCellInput::Unsupported => terminal_cell(cell, "unsupported"),
            FidelityCellInput::OverBudget { .. } => terminal_cell(cell, "over_budget"),
            FidelityCellInput::Missing => terminal_cell(cell, "missing"),
        };
        cell_receipts.push(receipt);
    }

    let scheduled = cell_receipts.len();
    let informative = cell_receipts
        .iter()
        .filter(|c| c.status == "complete")
        .count();
    let uninformative = cell_receipts
        .iter()
        .filter(|c| c.status == "uninformative")
        .count();
    let failed_or_missing = scheduled - informative - uninformative;
    // T18: Failed, timed-out, over-budget, and missing cells remain in the denominator as fails!
    let eligible_denom = scheduled - uninformative;
    let frac = |hits: usize| -> f64 {
        if eligible_denom == 0 {
            0.0
        } else {
            hits as f64 / eligible_denom as f64
        }
    };
    let parity_hits = cell_receipts
        .iter()
        .filter(|c| c.status == "complete" && c.parity.as_ref().is_some_and(|p| p.parity_pass))
        .count();
    let equiv_hits = cell_receipts
        .iter()
        .filter(|c| {
            c.status == "complete"
                && c.equivalence.as_ref().is_some_and(|e| {
                    e.pred.pass
                        && e.lift.pass
                        && e.calib.pass
                        && if req.allow_partial_diagnostic {
                            e.driver.as_ref().is_none_or(|d| d.pass)
                        } else {
                            e.driver.as_ref().is_some_and(|d| d.pass)
                        }
                })
                && c.deploy.as_ref().is_none_or(|d| d.pass)
        })
        .count();
    let real_hits = cell_receipts
        .iter()
        .filter(|c| {
            c.status == "complete"
                && if req.allow_partial_diagnostic {
                    c.realism.as_ref().is_none_or(|r| r.pass_standard)
                } else {
                    c.realism.as_ref().is_some_and(|r| r.pass_standard)
                }
        })
        .count();
    let priv_hits = cell_receipts
        .iter()
        .filter(|c| {
            c.status == "complete"
                && if req.allow_partial_diagnostic {
                    c.privacy.as_ref().is_none_or(|p| p.pass)
                } else {
                    c.privacy.as_ref().is_some_and(|p| p.pass)
                }
        })
        .count();
    let overall_hits = cell_receipts
        .iter()
        .filter(|c| c.status == "complete" && c.all_hard_gates_pass)
        .count();
    let mut rets: Vec<f64> = cell_receipts
        .iter()
        .filter(|c| c.status == "complete")
        .filter_map(|c| c.parity.as_ref()?.unclipped_retention_s)
        .collect();
    rets.sort_by(f64::total_cmp);
    let median_ret = if rets.is_empty() {
        None
    } else {
        let m = rets.len() / 2;
        Some(if rets.len() % 2 == 0 {
            0.5 * (rets[m - 1] + rets[m])
        } else {
            rets[m]
        })
    };

    Ok(FidelityV3Receipt {
        protocol: PROTOCOL_V3,
        mode: req.mode.clone(),
        composite_score: "forbidden",
        formal_dp_state: "not_claimed",
        coverage: CoverageSummary {
            scheduled_cells: scheduled,
            informative_cells: informative,
            uninformative_cells: uninformative,
            failed_or_missing_cells: failed_or_missing,
            eligible_denominator: eligible_denom,
            parity_coverage: frac(parity_hits),
            equivalence_coverage: frac(equiv_hits),
            realism_coverage: frac(real_hits),
            privacy_pass_coverage: frac(priv_hits),
            overall_pass_coverage: frac(overall_hits),
            median_unclipped_retention: median_ret,
        },
        cells: cell_receipts,
    })
}

fn terminal_cell(cell: &FidelityCellRequest, status: &'static str) -> FidelityCellReceipt {
    FidelityCellReceipt {
        cell_id: cell.cell_id.clone(),
        family_id: cell.family_id.clone(),
        auditor_id: cell.auditor_id.clone(),
        status,
        parity: None,
        equivalence: None,
        deploy: None,
        realism: None,
        privacy: None,
        all_hard_gates_pass: false,
    }
}

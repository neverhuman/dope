//! Selection transfer (`S_deploy`): 24-config roster regret vs fresh real twin `B`.

use crate::dvs::{ensure, finite_nonnegative, DvsError};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RosterConfigScore {
    pub config_id: String,
    /// Validation loss measured on real query `T` when trained on `A` (true oracle ranking).
    pub true_real_loss: f64,
    /// Selection score measured on synthetic `S` (lower is better).
    pub synth_select_loss: f64,
    /// Optional selection score measured on twin `B` (lower is better).
    pub twin_b_select_loss: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeploySelectionAssessment {
    pub roster_size: usize,
    pub oracle_best_config: String,
    pub synth_selected_config: String,
    pub twin_b_selected_config: Option<String>,
    pub synth_regret: f64,
    pub twin_b_regret: Option<f64>,
    pub excess_regret_over_b: f64,
    pub max_excess_regret: f64,
    pub pass: bool,
}

/// Compute `S_deploy` selection regret and pass check (`regret_S <= regret_B + 0.01`).
pub fn assess_deploy_selection(
    roster: &[RosterConfigScore],
    max_excess_regret: f64,
) -> Result<DeploySelectionAssessment, DvsError> {
    ensure(roster.len() >= 2, "empty_selection_roster")?;
    finite_nonnegative(max_excess_regret)?;
    for r in roster {
        ensure(!r.config_id.is_empty(), "empty_config_id")?;
        finite_nonnegative(r.true_real_loss)?;
        finite_nonnegative(r.synth_select_loss)?;
        if let Some(b) = r.twin_b_select_loss {
            finite_nonnegative(b)?;
        }
    }
    let oracle = roster
        .iter()
        .min_by(|a, b| {
            a.true_real_loss
                .total_cmp(&b.true_real_loss)
                .then(a.config_id.cmp(&b.config_id))
        })
        .ok_or(DvsError("empty_selection_roster"))?;
    let synth_pick = roster
        .iter()
        .min_by(|a, b| {
            a.synth_select_loss
                .total_cmp(&b.synth_select_loss)
                .then(a.config_id.cmp(&b.config_id))
        })
        .ok_or(DvsError("empty_selection_roster"))?;
    let synth_regret = (synth_pick.true_real_loss - oracle.true_real_loss).max(0.0);

    let (twin_pick_id, twin_regret) = if roster.iter().all(|r| r.twin_b_select_loss.is_some()) {
        let b_pick = roster
            .iter()
            .min_by(|a, b| {
                a.twin_b_select_loss
                    .unwrap()
                    .total_cmp(&b.twin_b_select_loss.unwrap())
                    .then(a.config_id.cmp(&b.config_id))
            })
            .ok_or(DvsError("empty_selection_roster"))?;
        (
            Some(b_pick.config_id.clone()),
            Some((b_pick.true_real_loss - oracle.true_real_loss).max(0.0)),
        )
    } else {
        (None, None)
    };
    let excess = synth_regret - twin_regret.unwrap_or(0.0);
    Ok(DeploySelectionAssessment {
        roster_size: roster.len(),
        oracle_best_config: oracle.config_id.clone(),
        synth_selected_config: synth_pick.config_id.clone(),
        twin_b_selected_config: twin_pick_id,
        synth_regret,
        twin_b_regret: twin_regret,
        excess_regret_over_b: excess,
        max_excess_regret,
        pass: excess <= max_excess_regret,
    })
}

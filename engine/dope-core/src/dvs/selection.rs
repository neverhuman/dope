//! Selection on synthetic validation is distinct from transfer ranking.
use super::{ensure, finite_nonnegative, DvsError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionEvidence {
    /// All vectors use the fixed roster order bound by DataPins::roster_sha256.
    pub synthetic_validation_losses: Vec<f64>,
    pub real_fit_test_losses: Vec<f64>,
    pub synthetic_fit_test_losses: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SelectionResult {
    pub status: &'static str,
    pub reason: Option<&'static str>,
    pub selected_index: Option<usize>,
    pub real_best_index: Option<usize>,
    pub synthetic_selection_tied: bool,
    pub selection_regret: Option<f64>,
    pub transfer_rank: Value,
}

pub fn evaluate(e: &SelectionEvidence) -> Result<SelectionResult, DvsError> {
    let n = e.synthetic_validation_losses.len();
    ensure(
        n >= 2 && e.real_fit_test_losses.len() == n && e.synthetic_fit_test_losses.len() == n,
        "selection_shape_mismatch",
    )?;
    for values in [
        &e.synthetic_validation_losses,
        &e.real_fit_test_losses,
        &e.synthetic_fit_test_losses,
    ] {
        for &v in values {
            finite_nonnegative(v)?;
        }
    }
    let argmin = |values: &[f64]| -> usize {
        (0..values.len())
            .min_by(|&a, &b| values[a].total_cmp(&values[b]).then(a.cmp(&b)))
            .unwrap()
    };
    // No real-test value participates in the synthetic-validation argmin.
    let selected = argmin(&e.synthetic_validation_losses);
    let best = argmin(&e.real_fit_test_losses);
    let tied = e
        .synthetic_validation_losses
        .iter()
        .filter(|&&v| v == e.synthetic_validation_losses[selected])
        .count()
        > 1;
    let rank = super::drivers::rank_agreement_informative(
        &e.real_fit_test_losses,
        &e.synthetic_fit_test_losses,
    )
    .map_err(|_| DvsError("invalid_selection_evidence"))?;
    let regret = e.real_fit_test_losses[selected] - e.real_fit_test_losses[best];
    finite_nonnegative(regret)?;
    Ok(SelectionResult {
        status: if tied || rank["status"] != "complete" {
            "uninformative"
        } else {
            "complete"
        },
        reason: if tied {
            Some("synthetic_validation_tied")
        } else if rank["status"] != "complete" {
            Some("no_informative_selection_pairs")
        } else {
            None
        },
        selected_index: if tied { None } else { Some(selected) },
        real_best_index: Some(best),
        synthetic_selection_tied: tied,
        selection_regret: if tied { None } else { Some(regret) },
        transfer_rank: rank,
    })
}

pub fn not_run() -> Value {
    json!({"status": "not_run", "reason": "selection_not_supplied", "selection_regret": null, "transfer_rank": null})
}

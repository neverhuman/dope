//! Empirical privacy capacity backoff ladder when DCR/MIA/exact-copy/canary gates trip.
//!
//! Strictly empirical mitigation; ALWAYS reports `formal_dp_state: "not_claimed"`.

use crate::dvs::v3::privacy::EmpiricalPrivacyAssessment;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CapacityBackoffStep {
    pub step_index: usize,
    pub tree_fraction: f64,
    pub catf_tree_fraction: f64,
    pub support_k: usize,
}

pub const PRIVACY_BACKOFF_LADDER: [CapacityBackoffStep; 5] = [
    CapacityBackoffStep {
        step_index: 0,
        tree_fraction: 1.00,
        catf_tree_fraction: 1.00,
        support_k: 10,
    },
    CapacityBackoffStep {
        step_index: 1,
        tree_fraction: 0.75,
        catf_tree_fraction: 0.50,
        support_k: 15,
    },
    CapacityBackoffStep {
        step_index: 2,
        tree_fraction: 0.50,
        catf_tree_fraction: 0.25,
        support_k: 20,
    },
    CapacityBackoffStep {
        step_index: 3,
        tree_fraction: 0.25,
        catf_tree_fraction: 0.00,
        support_k: 30,
    },
    CapacityBackoffStep {
        step_index: 4,
        tree_fraction: 0.00,
        catf_tree_fraction: 0.00,
        support_k: 40,
    },
];

#[derive(Clone, Debug, Serialize)]
pub struct PrivacyBackoffOutcome {
    pub selected_step: CapacityBackoffStep,
    pub assessment: EmpiricalPrivacyAssessment,
    pub formal_dp_state: &'static str,
}

/// Walk the monotone capacity backoff ladder until empirical DCR/MIA/canary gates pass
/// (or return the most conservative step with explicit `pass = false`).
pub fn run_privacy_backoff<F>(mut eval_step: F) -> PrivacyBackoffOutcome
where
    F: FnMut(CapacityBackoffStep) -> EmpiricalPrivacyAssessment,
{
    let mut last = None;
    for step in PRIVACY_BACKOFF_LADDER {
        let assess = eval_step(step);
        if assess.pass {
            return PrivacyBackoffOutcome {
                selected_step: step,
                assessment: assess,
                formal_dp_state: "not_claimed",
            };
        }
        last = Some((step, assess));
    }
    let (step, assess) = last.unwrap();
    PrivacyBackoffOutcome {
        selected_step: step,
        assessment: assess,
        formal_dp_state: "not_claimed",
    }
}

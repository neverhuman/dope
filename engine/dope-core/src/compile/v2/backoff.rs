//! Empirical privacy capacity backoff ladder when DCR/MIA/exact-copy/canary gates trip.
//!
//! Strictly empirical mitigation; ALWAYS reports `formal_dp_state: "not_claimed"`.

use crate::catf::CatfModel;
use crate::distil::DistilledStudent;
use crate::dvs::v3::privacy::{
    assess_empirical_privacy, EmpiricalPrivacyAssessment, EmpiricalPrivacyInput,
};
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

/// Apply standardized `(X, Y)` empirical privacy backoff to `(catf, student)` (`BUG-R10` / `PK-17`).
/// Never wipes `student.forest` on finite-sample noise and fails closed on assessment errors.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_standardized_privacy_backoff(
    catf: &mut CatfModel,
    student: &mut DistilledStudent,
    train_x: &[Vec<f64>],
    train_y: &[f64],
    cal_x: &[Vec<f64>],
    cal_y: &[f64],
    support_k: usize,
    seed: u64,
) {
    if train_y.len() < 64 || cal_y.len() < 32 {
        return;
    }
    let col_stats: Vec<(f64, f64)> = train_x
        .iter()
        .map(|col| {
            let fin: Vec<f64> = col.iter().copied().filter(|v| v.is_finite()).collect();
            let mu = fin.iter().sum::<f64>() / fin.len().max(1) as f64;
            let sd = (fin.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / fin.len().max(1) as f64)
                .sqrt()
                .max(1e-4);
            (mu, sd)
        })
        .collect();
    let y_mu = train_y.iter().sum::<f64>() / train_y.len().max(1) as f64;
    let y_sd = (train_y.iter().map(|v| (v - y_mu).powi(2)).sum::<f64>()
        / train_y.len().max(1) as f64)
        .sqrt()
        .max(1e-4);
    let to_std_rows = |cols: &[Vec<f64>], ys: &[f64], cap: usize| -> Vec<Vec<f64>> {
        let nr = cols.first().map_or(0, Vec::len).min(ys.len()).min(cap);
        (0..nr)
            .map(|r| {
                let mut row: Vec<f64> = cols
                    .iter()
                    .enumerate()
                    .map(|(j, c)| {
                        if c[r].is_finite() {
                            ((c[r] - col_stats[j].0) / col_stats[j].1).clamp(-6.0, 6.0)
                        } else {
                            0.0
                        }
                    })
                    .collect();
                row.push(((ys[r] - y_mu) / y_sd).clamp(-6.0, 6.0));
                row
            })
            .collect()
    };
    let m_cal = cal_y.len().min(256);
    let ra = to_std_rows(train_x, train_y, m_cal);
    let rb = to_std_rows(cal_x, cal_y, m_cal);
    let base_st = student.forest.trees();
    let base_catf_lens: Vec<usize> = catf.conditionals.iter().map(|c| c.trees.len()).collect();
    let mut step0_dcr = 1.0_f64;
    let outcome = run_privacy_backoff(|step| {
        let mut st_try = student.clone();
        let keep_st = if step.step_index == 0 {
            base_st
        } else {
            (((base_st as f64) * step.tree_fraction).round() as usize).max(base_st.min(4))
        };
        st_try.forest = student.forest.truncate(keep_st);
        let mut c_try = catf.clone();
        for (ci, cond) in c_try.conditionals.iter_mut().enumerate() {
            let keep_c = ((base_catf_lens[ci] as f64) * step.catf_tree_fraction).round() as usize;
            cond.trees.truncate(keep_c);
        }
        let n_syn_p = 512usize;
        let sy = c_try.sample_with_student(&st_try, n_syn_p, seed ^ 0x99, 3);
        let input = EmpiricalPrivacyInput {
            real_fit_a: ra.clone(),
            real_holdout_b: rb.clone(),
            synthetic: to_std_rows(&sy.x, &sy.y, n_syn_p),
            min_emitted_support: step.support_k.max(support_k) as u64,
            canary_rows: Vec::new(),
            max_distance_pairs: 200_000,
        };
        match assess_empirical_privacy(&input) {
            Ok(mut a) => {
                if m_cal < 128 {
                    a.pass = a.exact_copy_excess <= 0.005
                        && a.canary_hits == 0
                        && a.dcr_closer_to_a_share <= 0.55
                        && a.mia_auc <= 0.55;
                }
                if step.step_index == 0 {
                    step0_dcr = a.dcr_closer_to_a_share;
                }
                a
            }
            Err(_) => EmpiricalPrivacyAssessment {
                dcr_closer_to_a_share: 1.0,
                mia_auc: 1.0,
                mia_tpr_at_1pct_fpr: 1.0,
                exact_copies: 0,
                exact_copies_twin_b: 0,
                exact_copy_excess: 0.0,
                canary_hits: 0,
                min_emitted_support: 0,
                support_floor_k: 10,
                pass: false,
                formal_dp_state: "not_claimed",
                hipaa_deidentification_state: "not_claimed",
            },
        }
    });
    let improved =
        outcome.assessment.pass || outcome.assessment.dcr_closer_to_a_share + 0.015 < step0_dcr;
    if outcome.selected_step.step_index > 0 && improved {
        let st_k = (((base_st as f64) * outcome.selected_step.tree_fraction).round() as usize)
            .max(base_st.min(4));
        student.forest = student.forest.truncate(st_k);
        student.forest_trees = student.forest.trees();
        for (ci, cond) in catf.conditionals.iter_mut().enumerate() {
            let ck = ((base_catf_lens[ci] as f64) * outcome.selected_step.catf_tree_fraction)
                .round() as usize;
            cond.trees.truncate(ck);
        }
        student.refit_calibration(cal_x, cal_y);
    }
}

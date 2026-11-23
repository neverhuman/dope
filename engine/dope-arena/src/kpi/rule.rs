//! The promotion rule of `contracts/dope-kpi.json` (`spec.rs`).
//!
//! PROMOTE iff (1) the 2.5th percentile of the paired delta is above 0,
//! (2) the Holm-adjusted p is below 0.05, (3) the delta is at least
//! `max(0.01, noise_sd)`, (4) every guard passes, and (5) for a seed group,
//! at least three training seeds each have the sign of the pooled delta.
//! REJECT iff the 97.5th percentile is below 0; HOLD otherwise. A guard
//! whose input is absent is "not evaluated", and the outcome is then never
//! PROMOTE.
//!
//! The byte guard reads the candidate's headline files: it fails when a
//! measured packet exceeds the headline budget, and is not evaluated when a
//! produced file carries no measured packet (a control) or nothing was
//! produced.

use super::boot::Interval;
use super::spec::PromotionSpec;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Outcome {
    Promote,
    Hold,
    Reject,
}

impl Outcome {
    pub fn word(self) -> &'static str {
        match self {
            Outcome::Promote => "PROMOTE",
            Outcome::Hold => "HOLD",
            Outcome::Reject => "REJECT",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Check {
    Pass,
    Fail,
    NotEvaluated,
}

/// One condition or guard of the rule with the value it was decided on.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Item {
    pub name: String,
    pub check: Check,
    pub value: Option<f64>,
    pub threshold: Option<f64>,
}

fn item(name: &str, value: Option<f64>, threshold: f64, pass: impl Fn(f64) -> bool) -> Item {
    Item {
        name: name.to_string(),
        check: match value {
            None => Check::NotEvaluated,
            Some(v) if pass(v) => Check::Pass,
            Some(_) => Check::Fail,
        },
        value,
        threshold: Some(threshold),
    }
}

/// Byte evidence of the candidate's headline files.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BytesEvidence {
    /// Files the candidate produced: scored, or carrying a measured packet.
    pub produced: usize,
    /// Files with a measured packet.
    pub measured: usize,
    /// Measured packets no longer than the budget.
    pub within: usize,
    pub max_bytes: u64,
}

impl BytesEvidence {
    fn check(self) -> Check {
        if self.within < self.measured {
            Check::Fail
        } else if self.measured == 0 || self.measured < self.produced {
            Check::NotEvaluated
        } else {
            Check::Pass
        }
    }
}

/// A seed group: training seeds and how many have the pooled delta's sign.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeedEvidence {
    pub seeds: usize,
    pub same_sign: usize,
}

/// Everything the rule reads for one comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct Evidence {
    pub delta: Interval,
    pub p_holm: f64,
    /// Lower bound of the paired delta of every task present.
    pub task_ci_lows: Vec<f64>,
    pub mlp_delta: Option<f64>,
    pub realism_delta: Option<f64>,
    pub privacy_delta: Option<f64>,
    pub exact_copy_excess: Option<f64>,
    pub bytes: BytesEvidence,
    pub seed_group: Option<SeedEvidence>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Decision {
    pub outcome: Outcome,
    pub conditions: Vec<Item>,
    pub guards: Vec<Item>,
}

pub const STATISTICAL_CONDITIONS: [&str; 3] = ["ci_low", "holm_p", "min_delta"];

impl Decision {
    /// Conditions (1)-(3): what the rule would decide if every guard passed.
    pub fn statistical_pass(&self) -> bool {
        self.conditions
            .iter()
            .filter(|c| STATISTICAL_CONDITIONS.contains(&c.name.as_str()))
            .all(|c| c.check == Check::Pass)
    }
}

pub fn decide(rule: &PromotionSpec, budget: u64, noise_sd: f64, ev: &Evidence) -> Decision {
    let min_delta = rule.min_delta.max(noise_sd);
    let mut conditions = vec![
        item("ci_low", Some(ev.delta.ci_low), rule.ci_low_gt, |v| {
            v > rule.ci_low_gt
        }),
        item("holm_p", Some(ev.p_holm), rule.holm_p_lt, |v| {
            v < rule.holm_p_lt
        }),
        item("min_delta", Some(ev.delta.value), min_delta, |v| {
            v >= min_delta
        }),
    ];
    if let Some(group) = ev.seed_group {
        let seeds = rule.min_training_seeds as f64;
        let all = group.seeds as f64;
        conditions.push(item("training_seeds", Some(all), seeds, |v| v >= seeds));
        conditions.push(item("seed_signs", Some(group.same_sign as f64), all, |v| {
            v == all
        }));
    }
    let lowest = ev.task_ci_lows.iter().copied().reduce(f64::min);
    let guards = vec![
        item("task_ci_low", lowest, rule.task_ci_low_gt, |v| {
            v > rule.task_ci_low_gt
        }),
        item("mlp_delta", ev.mlp_delta, rule.mlp_delta_ge, |v| {
            v >= rule.mlp_delta_ge
        }),
        item(
            "realism_coverage_delta",
            ev.realism_delta,
            rule.realism_coverage_delta_gt,
            |v| v > rule.realism_coverage_delta_gt,
        ),
        item(
            "privacy_coverage_delta",
            ev.privacy_delta,
            rule.privacy_coverage_delta_gt,
            |v| v > rule.privacy_coverage_delta_gt,
        ),
        item(
            "exact_copy_excess",
            ev.exact_copy_excess,
            rule.exact_copy_excess_le,
            |v| v <= rule.exact_copy_excess_le,
        ),
        Item {
            name: "bytes".to_string(),
            check: ev.bytes.check(),
            value: (ev.bytes.measured > 0).then_some(ev.bytes.max_bytes as f64),
            threshold: Some(budget as f64),
        },
    ];
    let promote = conditions
        .iter()
        .chain(&guards)
        .all(|i| i.check == Check::Pass);
    let outcome = if promote {
        Outcome::Promote
    } else if ev.delta.ci_high < rule.reject_ci_high_lt {
        Outcome::Reject
    } else {
        Outcome::Hold
    };
    Decision {
        outcome,
        conditions,
        guards,
    }
}

#[cfg(test)]
mod tests {
    use super::super::spec::Contract;
    use super::*;

    fn passing() -> Evidence {
        Evidence {
            delta: Interval {
                value: 0.05,
                ci_low: 0.02,
                ci_high: 0.08,
                p_one_sided: 0.001,
            },
            p_holm: 0.002,
            task_ci_lows: vec![0.01, -0.015, 0.0],
            mlp_delta: Some(0.0),
            realism_delta: Some(-0.029),
            privacy_delta: Some(0.0),
            exact_copy_excess: Some(0.005),
            bytes: BytesEvidence {
                produced: 10,
                measured: 10,
                within: 10,
                max_bytes: 2048,
            },
            seed_group: None,
        }
    }

    fn decision(ev: &Evidence, noise_sd: f64) -> Decision {
        let (c, _) = Contract::embedded().unwrap();
        decide(&c.promotion, c.bytes.headline, noise_sd, ev)
    }

    fn outcome(ev: &Evidence, noise_sd: f64) -> Outcome {
        decision(ev, noise_sd).outcome
    }

    #[test]
    fn every_condition_at_its_boundary() {
        assert_eq!(outcome(&passing(), 0.01), Outcome::Promote);
        // The noise SD raises the delta bar above the 0.01 floor.
        assert_eq!(outcome(&passing(), 0.06), Outcome::Hold);
        let mut ev = passing();
        ev.delta.ci_low = 0.0;
        assert_eq!(outcome(&ev, 0.01), Outcome::Hold);
        let mut ev = passing();
        ev.p_holm = 0.05;
        assert_eq!(outcome(&ev, 0.01), Outcome::Hold);
        let mut ev = passing();
        ev.delta.value = 0.009;
        assert!(!decision(&ev, 0.01).statistical_pass());
        assert_eq!(outcome(&ev, 0.01), Outcome::Hold);
    }

    type EvidenceEdit = fn(&mut Evidence);

    #[test]
    fn each_guard_flips_promote_to_hold() {
        let edits: [(EvidenceEdit, Check); 12] = [
            (|ev| ev.task_ci_lows[1] = -0.02, Check::Fail),
            (|ev| ev.mlp_delta = Some(-1e-9), Check::Fail),
            (|ev| ev.mlp_delta = None, Check::NotEvaluated),
            (|ev| ev.realism_delta = Some(-0.03), Check::Fail),
            (|ev| ev.realism_delta = None, Check::NotEvaluated),
            (|ev| ev.privacy_delta = Some(-0.031), Check::Fail),
            (|ev| ev.privacy_delta = None, Check::NotEvaluated),
            (|ev| ev.exact_copy_excess = Some(0.0051), Check::Fail),
            (|ev| ev.exact_copy_excess = None, Check::NotEvaluated),
            (|ev| ev.bytes.within = 9, Check::Fail),
            // One produced file without a measured packet.
            (
                |ev| (ev.bytes.measured, ev.bytes.within) = (9, 9),
                Check::NotEvaluated,
            ),
            (
                |ev| ev.bytes = BytesEvidence::default(),
                Check::NotEvaluated,
            ),
        ];
        for (index, (edit, check)) in edits.iter().enumerate() {
            let mut ev = passing();
            edit(&mut ev);
            let decision = decision(&ev, 0.01);
            assert_eq!(decision.outcome, Outcome::Hold, "edit {index}");
            assert!(decision.statistical_pass(), "edit {index}");
            let off: Vec<Check> = decision
                .guards
                .iter()
                .map(|g| g.check)
                .filter(|c| *c != Check::Pass)
                .collect();
            assert_eq!(off, [*check], "edit {index}");
        }
    }

    #[test]
    fn reject_needs_the_upper_bound_below_zero() {
        let mut ev = passing();
        ev.delta = Interval {
            value: -0.05,
            ci_low: -0.08,
            ci_high: -1e-6,
            p_one_sided: 0.99,
        };
        assert_eq!(outcome(&ev, 0.01), Outcome::Reject);
        ev.delta.ci_high = 0.0;
        assert_eq!(outcome(&ev, 0.01), Outcome::Hold);
        assert_eq!(Outcome::Reject.word(), "REJECT");
    }

    #[test]
    fn a_seed_group_needs_three_seeds_with_the_pooled_sign() {
        let mut ev = passing();
        ev.seed_group = Some(SeedEvidence {
            seeds: 3,
            same_sign: 3,
        });
        assert_eq!(outcome(&ev, 0.01), Outcome::Promote);
        ev.seed_group = Some(SeedEvidence {
            seeds: 3,
            same_sign: 2,
        });
        assert_eq!(outcome(&ev, 0.01), Outcome::Hold);
        ev.seed_group = Some(SeedEvidence {
            seeds: 2,
            same_sign: 2,
        });
        assert_eq!(outcome(&ev, 0.01), Outcome::Hold);
    }
}

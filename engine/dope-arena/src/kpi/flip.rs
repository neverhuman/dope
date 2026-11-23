//! Sign flips of one prepared comparison: the replicates `null` reports and
//! the calibration diagnostic `compare` reports beside each decision.
//!
//! Under H0 (no difference) a family's paired differences are as likely `+d`
//! as `-d`. A replicate multiplies every family's differences (TRG, each seed
//! of a seed group, the held-out MLP, realism and privacy coverage) by one
//! independent fair sign, reruns the bootstrap on the same family draws and
//! applies the decision rule with Holm over this one comparison. The
//! candidate's bytes and exact copies do not move under a flip.
//!
//! `flip_guard_rule` is the rule with the fixed guards and the unevaluated
//! guards taken as passing: every condition and every evaluated guard that a
//! flip moves. Guards only add requirements, so its rate bounds the
//! false-PROMOTE rate of the full rule for any value the fixed and the
//! unevaluated guards turn out to have. A Holm adjustment over more
//! comparisons only raises p, so the rate bounds a decision taken in a
//! receipt with several comparisons as well. The pre-registered decision
//! rule does not read it: it is reported, never decided on.

use super::boot::{holm, wilson95};
use super::compare::{evidence, Prepared};
use super::jsonl::canonical_json;
use super::receipt::Context;
use super::rule::{decide, Check, Decision, Outcome};
use super::seeds::KpiRng;
use dope_arena::ArenaResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Guards a sign flip does not move: properties of the candidate's own files.
pub const FIXED_GUARDS: [&str; 2] = ["exact_copy_excess", "bytes"];

/// Every condition passes and no guard that a flip moves fails.
pub fn flip_guard_rule(decision: &Decision) -> bool {
    decision.conditions.iter().all(|c| c.check == Check::Pass)
        && decision
            .guards
            .iter()
            .all(|g| FIXED_GUARDS.contains(&g.name.as_str()) || g.check != Check::Fail)
}

/// A replicate rate with its Wilson 95% interval, against the contract's maximum.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Rate {
    pub count: usize,
    pub rate: f64,
    pub wilson95_low: f64,
    pub wilson95_high: f64,
    pub acceptable: bool,
}

pub fn rate(count: usize, trials: usize, max: f64) -> Rate {
    let (low, high) = wilson95(count, trials);
    let rate = count as f64 / trials.max(1) as f64;
    Rate {
        count,
        rate,
        wilson95_low: low,
        wilson95_high: high,
        acceptable: rate <= max,
    }
}

/// Counts over the replicates of one comparison.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Flips {
    pub replicates: usize,
    pub seed: u64,
    /// Families that carry a sign.
    pub families: usize,
    /// Replicates the full rule promotes.
    pub promote: usize,
    pub flip_guard_rule: usize,
    /// Replicates passing conditions (1)-(3).
    pub statistical_pass: usize,
    /// Replicates passing each condition and guard, by name.
    pub passed: BTreeMap<String, usize>,
    pub outcomes: BTreeMap<String, usize>,
}

#[derive(Serialize)]
struct ReplicateLine {
    replicate: usize,
    delta: f64,
    ci_low: f64,
    ci_high: f64,
    p_one_sided: f64,
    outcome: Outcome,
    flip_guard_rule: bool,
    statistical_pass: bool,
}

/// `replicates` sign flips of `prepared` (seeded); `lines` gets one JSON record per replicate.
pub fn sign_flips(
    ctx: &Context,
    prepared: &Prepared,
    replicates: usize,
    seed: u64,
    noise_sd: f64,
    mut lines: Option<&mut Vec<String>>,
) -> ArenaResult<Flips> {
    let (rule, budget) = (&ctx.contract.promotion, ctx.contract.bytes.headline);
    let ids: Vec<&str> = prepared.paired.diffs.family_ids().into_iter().collect();
    let mut rng = KpiRng::new(seed);
    let mut flips = Flips {
        replicates,
        seed,
        families: ids.len(),
        ..Flips::default()
    };
    for replicate in 0..replicates {
        let signs: BTreeMap<&str, f64> = ids
            .iter()
            .map(|id| (*id, if rng.below(2) == 0 { 1.0 } else { -1.0 }))
            .collect();
        let diffs = prepared.paired.diffs.flipped(&|family| signs[family]);
        let stats = prepared.plan.run(&diffs.trg.values());
        let p_holm = holm(&[stats.overall.p_one_sided])[0];
        let ev = evidence(&stats, p_holm, &diffs, &prepared.fixed);
        let decision: Decision = decide(rule, budget, noise_sd, &ev);
        let flip_rule = flip_guard_rule(&decision);
        flips.promote += usize::from(decision.outcome == Outcome::Promote);
        flips.flip_guard_rule += usize::from(flip_rule);
        flips.statistical_pass += usize::from(decision.statistical_pass());
        *flips
            .outcomes
            .entry(decision.outcome.word().to_string())
            .or_default() += 1;
        for item in decision.conditions.iter().chain(&decision.guards) {
            *flips.passed.entry(item.name.clone()).or_default() +=
                usize::from(item.check == Check::Pass);
        }
        if let Some(lines) = lines.as_deref_mut() {
            lines.push(canonical_json(&ReplicateLine {
                replicate,
                delta: stats.overall.value,
                ci_low: stats.overall.ci_low,
                ci_high: stats.overall.ci_high,
                p_one_sided: stats.overall.p_one_sided,
                outcome: decision.outcome,
                flip_guard_rule: flip_rule,
                statistical_pass: decision.statistical_pass(),
            })?);
        }
    }
    Ok(flips)
}

/// The calibration of one comparison: the share of its sign-flip replicates
/// the rule promotes (a diagnostic; the decision does not read it).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Calibration {
    pub replicates: usize,
    pub seed: u64,
    pub families: usize,
    /// The sign-flip false-PROMOTE rate: replicates passing `flip_guard_rule`.
    pub flip_guard_rule: Rate,
    pub statistical_pass: Rate,
    pub max_false_promote_rate: f64,
}

impl Calibration {
    pub fn of(flips: &Flips, max: f64) -> Calibration {
        Calibration {
            replicates: flips.replicates,
            seed: flips.seed,
            families: flips.families,
            flip_guard_rule: rate(flips.flip_guard_rule, flips.replicates, max),
            statistical_pass: rate(flips.statistical_pass, flips.replicates, max),
            max_false_promote_rate: max,
        }
    }
}

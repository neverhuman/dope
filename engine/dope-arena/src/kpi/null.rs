//! `dope_kpi null`: the sign-flip null of one comparison.
//!
//! Under H0 (no difference) a family's paired difference is as likely to be
//! `+d` as `-d`. Each replicate multiplies every family's paired differences
//! (TRG, held-out MLP, realism and privacy coverage) by one independent fair
//! sign, reruns the bootstrap on the same family draws, and applies the full
//! decision rule (Holm over this one comparison). The candidate's own bytes
//! and exact-copy excess do not change under a flip. The share of replicates
//! that PROMOTE is the false-PROMOTE rate (acceptable at most 5%).
//!
//! A guard that is not evaluated, or a fixed guard the candidate fails,
//! blocks PROMOTE in every replicate, so the receipt reports two more rates:
//! the rule with the fixed and unevaluated guards taken as passing (every
//! condition and every evaluated flip-moved guard must pass), and conditions
//! (1)-(3) alone. Guards only add requirements, so the last rate bounds the
//! rule's false-PROMOTE rate from above whatever the guard inputs turn out to
//! be.

use super::boot::{holm, wilson95};
use super::compare::{evidence, prepare, result, ComparisonResult};
use super::emitlog::check_arm_id;
use super::jsonl::{canonical_json, write_lines};
use super::pairing::Spec;
use super::primary::summarize;
use super::receipt::{write_receipt, Context, RunArgs};
use super::rule::{decide, Check, Decision, Outcome};
use super::rundir::contract;
use super::seeds::KpiRng;
use dope_arena::ArenaResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(clap::Args, Debug)]
pub struct NullArgs {
    #[command(flatten)]
    pub run: RunArgs,
    #[arg(long)]
    pub candidate: String,
    #[arg(long)]
    pub baseline: String,
    /// Sign-flip replicates (default and minimum: the contract's).
    #[arg(long)]
    pub replicates: Option<usize>,
    /// Noise SD of the statistic (default: the contract's).
    #[arg(long)]
    pub noise_sd: Option<f64>,
    /// Seed of the sign stream (default: the contract's).
    #[arg(long)]
    pub null_seed: Option<u64>,
    /// Output receipt (JSON).
    #[arg(long)]
    pub out: PathBuf,
    /// Also write one JSON line per replicate (delta, bounds, p, outcome).
    #[arg(long)]
    pub replicates_out: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Rate {
    pub count: usize,
    pub rate: f64,
    pub wilson95_low: f64,
    pub wilson95_high: f64,
    pub acceptable: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NullResult {
    pub candidate: String,
    pub baseline: String,
    pub replicates: usize,
    pub seed: u64,
    /// Families that carry a sign.
    pub families: usize,
    pub max_false_promote_rate: f64,
    /// The comparison as observed (Holm over this one comparison).
    pub observed: ComparisonResult,
    /// Replicates the full rule promotes.
    pub false_promote: Rate,
    /// Replicates passing every condition and every guard a flip moves (task
    /// intervals, MLP, coverage) that could be evaluated: the false-PROMOTE
    /// rate for a candidate whose fixed and unevaluated guards pass.
    pub flip_guard_rule: Rate,
    /// Replicates passing conditions (1)-(3): an upper bound for any guards.
    pub statistical_pass: Rate,
    /// Replicates passing each condition and guard, by name.
    pub passed: BTreeMap<String, usize>,
    pub outcomes: BTreeMap<String, usize>,
    /// Guards the observed comparison could not evaluate (they block PROMOTE in every replicate).
    pub not_evaluated_guards: Vec<String>,
    pub replicates_sha256: Option<String>,
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

/// Guards a sign flip does not move: properties of the candidate's own files.
const FIXED_GUARDS: [&str; 2] = ["exact_copy_excess", "bytes"];

fn flip_guard_rule(decision: &Decision) -> bool {
    decision.conditions.iter().all(|c| c.check == Check::Pass)
        && decision
            .guards
            .iter()
            .all(|g| FIXED_GUARDS.contains(&g.name.as_str()) || g.check != Check::Fail)
}

fn rate(count: usize, trials: usize, max: f64) -> Rate {
    let (low, high) = wilson95(count, trials);
    let rate = count as f64 / trials as f64;
    Rate {
        count,
        rate,
        wilson95_low: low,
        wilson95_high: high,
        acceptable: rate <= max,
    }
}

/// The sign-flip null of `candidate` against `baseline` (`lines` gets one record per replicate).
pub fn sign_flip_null(
    ctx: &Context,
    candidate: &str,
    baseline: &str,
    replicates: usize,
    seed: u64,
    noise_sd: f64,
    lines: &mut Vec<String>,
) -> ArenaResult<NullResult> {
    let spec = Spec {
        id: candidate.to_string(),
        seed_group: false,
        members: vec![candidate.to_string()],
    };
    let prepared = prepare(ctx, &spec, baseline)?;
    let (rule, budget) = (&ctx.contract.promotion, ctx.contract.bytes.headline);
    let p = prepared.stats.overall.p_one_sided;
    let observed_decision = decide(
        rule,
        budget,
        noise_sd,
        &evidence(&prepared.stats, p, &prepared.paired.diffs, &prepared.fixed),
    );
    let not_evaluated_guards = observed_decision
        .guards
        .iter()
        .filter(|g| g.check == Check::NotEvaluated)
        .map(|g| g.name.clone())
        .collect();
    let observed = result(&prepared, baseline, p, observed_decision);
    let ids: Vec<&str> = prepared.paired.diffs.family_ids().into_iter().collect();
    let mut rng = KpiRng::new(seed);
    let (mut promote, mut flip_rule, mut statistical) = (0, 0, 0);
    let mut passed: BTreeMap<String, usize> = BTreeMap::new();
    let mut outcomes: BTreeMap<String, usize> = BTreeMap::new();
    for replicate in 0..replicates {
        let signs: BTreeMap<&str, f64> = ids
            .iter()
            .map(|id| (*id, if rng.below(2) == 0 { 1.0 } else { -1.0 }))
            .collect();
        let diffs = prepared.paired.diffs.flipped(&|family| signs[family]);
        let stats = prepared.plan.run(&diffs.trg.values());
        let p_holm = holm(&[stats.overall.p_one_sided])[0];
        let decision = decide(
            rule,
            budget,
            noise_sd,
            &evidence(&stats, p_holm, &diffs, &prepared.fixed),
        );
        promote += usize::from(decision.outcome == Outcome::Promote);
        flip_rule += usize::from(flip_guard_rule(&decision));
        statistical += usize::from(decision.statistical_pass());
        *outcomes
            .entry(decision.outcome.word().to_string())
            .or_default() += 1;
        for item in decision.conditions.iter().chain(&decision.guards) {
            *passed.entry(item.name.clone()).or_default() += usize::from(item.check == Check::Pass);
        }
        lines.push(canonical_json(&ReplicateLine {
            replicate,
            delta: stats.overall.value,
            ci_low: stats.overall.ci_low,
            ci_high: stats.overall.ci_high,
            p_one_sided: stats.overall.p_one_sided,
            outcome: decision.outcome,
            flip_guard_rule: flip_guard_rule(&decision),
            statistical_pass: decision.statistical_pass(),
        })?);
    }
    let max = ctx.contract.null.max_false_promote_rate;
    Ok(NullResult {
        candidate: candidate.to_string(),
        baseline: baseline.to_string(),
        replicates,
        seed,
        families: ids.len(),
        max_false_promote_rate: max,
        observed,
        false_promote: rate(promote, replicates, max),
        flip_guard_rule: rate(flip_rule, replicates, max),
        statistical_pass: rate(statistical, replicates, max),
        passed,
        outcomes,
        not_evaluated_guards,
        replicates_sha256: None,
    })
}

#[derive(Serialize)]
struct Summary {
    command: &'static str,
    replicates: usize,
    false_promote_rate: f64,
    flip_guard_rule_rate: f64,
    statistical_pass_rate: f64,
    receipt_sha256: String,
}

pub fn run(args: &NullArgs) -> ArenaResult<String> {
    check_arm_id(&args.candidate)?;
    check_arm_id(&args.baseline)?;
    let ctx = Context::open(
        &args.run,
        &[args.candidate.as_str(), args.baseline.as_str()],
    )?;
    let minimum = ctx.contract.null.min_replicates;
    let replicates = args.replicates.unwrap_or(minimum);
    if replicates < minimum {
        return contract(format!("--replicates must be at least {minimum}"));
    }
    let noise = ctx.noise_sd(args.noise_sd)?;
    let seed = args.null_seed.unwrap_or(ctx.contract.null.seed);
    let mut lines = Vec::new();
    let mut null = sign_flip_null(
        &ctx,
        &args.candidate,
        &args.baseline,
        replicates,
        seed,
        noise.value,
        &mut lines,
    )?;
    if let Some(path) = &args.replicates_out {
        null.replicates_sha256 = Some(write_lines(path, lines.iter().map(String::as_str))?);
    }
    let mut receipt = ctx.receipt("null");
    let mut ids = vec![args.candidate.as_str()];
    if args.baseline != args.candidate {
        ids.push(&args.baseline);
    }
    receipt.arms = ids.iter().map(|id| summarize(&ctx, ctx.arm(id))).collect();
    receipt.noise_sd = Some(noise);
    let false_promote_rate = null.false_promote.rate;
    let flip_guard_rule_rate = null.flip_guard_rule.rate;
    let statistical_pass_rate = null.statistical_pass.rate;
    receipt.null = Some(null);
    let receipt_sha256 = write_receipt(&args.out, &receipt)?;
    canonical_json(&Summary {
        command: "null",
        replicates,
        false_promote_rate,
        flip_guard_rule_rate,
        statistical_pass_rate,
        receipt_sha256,
    })
}

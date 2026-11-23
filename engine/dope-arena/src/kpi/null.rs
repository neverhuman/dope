//! `dope_kpi null`: the sign-flip null of one comparison.
//!
//! Under H0 (no difference) a family's paired difference is as likely to be
//! `+d` as `-d`. Each replicate (`flip::sign_flips`) multiplies every
//! family's paired differences (TRG, each seed of a seed group, held-out MLP,
//! realism and privacy coverage) by one independent fair sign, reruns the
//! bootstrap on the same family draws, and applies the full decision rule
//! (Holm over this one comparison). The candidate's own bytes and exact-copy
//! excess do not change under a flip.
//!
//! A guard that is not evaluated, or a fixed guard the candidate fails,
//! blocks PROMOTE in every replicate: the full rule's rate is then 0 whatever
//! the statistic does, and `false_promote` is absent (the guards that made it
//! so are listed). `flip_guard_rule` takes the fixed and unevaluated guards
//! as passing (every condition and every evaluated flip-moved guard must
//! pass): it equals the full rule's rate when that is informative and bounds
//! it from above otherwise, so `acceptable` reads it. Conditions (1)-(3)
//! alone are reported as well.

use super::compare::{evidence, prepare, result, ComparisonResult};
use super::emitlog::check_arm_id;
use super::flip::{rate, sign_flips, Rate, FIXED_GUARDS};
use super::jsonl::{canonical_json, write_lines};
use super::pairing::Spec;
use super::primary::summarize;
use super::receipt::{write_receipt, Context, RunArgs};
use super::rule::{decide, Check};
use super::rundir::contract;
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
    /// Replicates the full rule promotes; absent when a fixed guard fails or
    /// a guard is not evaluated, which blocks PROMOTE in every replicate.
    pub false_promote: Option<Rate>,
    /// Replicates passing every condition and every guard a flip moves (task
    /// intervals, MLP, coverage, seeds) that could be evaluated: the
    /// false-PROMOTE rate for a candidate whose fixed and unevaluated guards pass.
    pub flip_guard_rule: Rate,
    /// Replicates passing conditions (1)-(3): an upper bound for any guards.
    pub statistical_pass: Rate,
    /// `flip_guard_rule` is at most the contract's maximum.
    #[serde(default)]
    pub acceptable: bool,
    /// Replicates passing each condition and guard, by name.
    pub passed: BTreeMap<String, usize>,
    pub outcomes: BTreeMap<String, usize>,
    /// Guards the observed comparison could not evaluate (they block PROMOTE in every replicate).
    pub not_evaluated_guards: Vec<String>,
    /// Fixed guards the observed comparison fails (they block PROMOTE in every replicate).
    #[serde(default)]
    pub failing_fixed_guards: Vec<String>,
    pub replicates_sha256: Option<String>,
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
    let names = |keep: &dyn Fn(&super::rule::Item) -> bool| -> Vec<String> {
        observed_decision
            .guards
            .iter()
            .filter(|g| keep(g))
            .map(|g| g.name.clone())
            .collect()
    };
    let not_evaluated_guards = names(&|g| g.check == Check::NotEvaluated);
    let failing_fixed_guards =
        names(&|g| g.check == Check::Fail && FIXED_GUARDS.contains(&g.name.as_str()));
    let observed = result(&prepared, baseline, (p, 1), observed_decision, None);
    let flips = sign_flips(ctx, &prepared, replicates, seed, noise_sd, Some(lines))?;
    let max = ctx.contract.null.max_false_promote_rate;
    let informative = not_evaluated_guards.is_empty() && failing_fixed_guards.is_empty();
    let flip_guard_rule = rate(flips.flip_guard_rule, replicates, max);
    Ok(NullResult {
        candidate: candidate.to_string(),
        baseline: baseline.to_string(),
        replicates,
        seed,
        families: flips.families,
        max_false_promote_rate: max,
        observed,
        false_promote: informative.then(|| rate(flips.promote, replicates, max)),
        acceptable: flip_guard_rule.acceptable,
        flip_guard_rule,
        statistical_pass: rate(flips.statistical_pass, replicates, max),
        passed: flips.passed,
        outcomes: flips.outcomes,
        not_evaluated_guards,
        failing_fixed_guards,
        replicates_sha256: None,
    })
}

#[derive(Serialize)]
struct Summary {
    command: &'static str,
    replicates: usize,
    acceptable: bool,
    false_promote_rate: Option<f64>,
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
    let false_promote_rate = null.false_promote.as_ref().map(|r| r.rate);
    let acceptable = null.acceptable;
    let flip_guard_rule_rate = null.flip_guard_rule.rate;
    let statistical_pass_rate = null.statistical_pass.rate;
    receipt.null = Some(null);
    let receipt_sha256 = write_receipt(&args.out, &receipt)?;
    canonical_json(&Summary {
        command: "null",
        replicates,
        acceptable,
        false_promote_rate,
        flip_guard_rule_rate,
        statistical_pass_rate,
        receipt_sha256,
    })
}

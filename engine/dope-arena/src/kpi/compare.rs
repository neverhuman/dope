//! `dope_kpi compare`: candidates against one baseline, paired on the cells
//! applicable to every arm involved, and the promotion rule of the contract.
//!
//! Per comparison: the paired delta (candidate minus baseline) of the TRG,
//! its stratified family bootstrap interval (the same family draws for both
//! arms, i.e. a bootstrap of the per-family differences), the one-sided p
//! for `delta <= 0`, the Holm adjustment across every comparison of the
//! receipt, the per-task deltas with intervals, the matched-multiplier delta
//! (reported, not decided on), the guards and the outcome. Beside the
//! decision, and never read by it: the comparison's own sign-flip
//! false-PROMOTE rate (`flip::sign_flips` with the contract's null
//! replicates and seed) and the exact-copy maximum and share of files over
//! the bound (the guard reads the mean, as the contract defines it).
//!
//! A `--seed-group NAME=A,B,C` pools the training seeds of one model: per
//! cell the candidate value is the mean over the seeds, on cells applicable
//! to every seed and the baseline, so the pooled delta is the mean of the
//! seeds' deltas; each seed's own delta must have the pooled sign.

use super::boot::{holm, BootPlan, BootStats};
use super::emitlog::check_arm_id;
use super::flip::{sign_flips, Calibration};
use super::pairing::{pair, parse_specs, Diffs, Paired, Spec, States};
use super::primary::{exact_copy, headline_files, scored_above_budget, summarize, ExactCopy};
use super::receipt::{write_receipt, Context, RunArgs};
use super::rule::{decide, BytesEvidence, Decision, Evidence, SeedEvidence};
use super::rundir::{contract, Arm, STATUS_OK, TASKS};
use super::trg::{cell_states, Families};
use dope_arena::ArenaResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(clap::Args, Debug)]
pub struct CompareArgs {
    #[command(flatten)]
    pub run: RunArgs,
    /// Candidate arm (repeat for several; each is one comparison).
    #[arg(long = "candidate")]
    pub candidates: Vec<String>,
    /// Baseline arm every candidate is compared with.
    #[arg(long)]
    pub baseline: String,
    /// `NAME=ARM,ARM,ARM`: training seeds of one model, pooled into one comparison.
    #[arg(long = "seed-group")]
    pub seed_groups: Vec<String>,
    /// Noise SD of the statistic; the delta bar is `max(min_delta, noise_sd)` (default: the contract's).
    #[arg(long)]
    pub noise_sd: Option<f64>,
    /// Output receipt (JSON).
    #[arg(long)]
    pub out: PathBuf,
}

/// Candidate evidence that sign flips do not touch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fixed {
    pub exact_copy_excess: Option<f64>,
    pub bytes: BytesEvidence,
    /// The comparison pools training seeds (each seed's sign is read from `Diffs`).
    pub seed_group: bool,
}

pub fn evidence(stats: &BootStats, p_holm: f64, diffs: &Diffs, fixed: &Fixed) -> Evidence {
    let pooled = stats.overall.value;
    let same_sign = |d: &f64| *d != 0.0 && pooled != 0.0 && d.signum() == pooled.signum();
    let members = diffs.member_deltas();
    Evidence {
        delta: stats.overall,
        p_holm,
        task_ci_lows: stats.per_task.iter().map(|i| i.ci_low).collect(),
        mlp_delta: diffs.mlp.point(),
        realism_delta: diffs.realism.as_ref().and_then(Families::point),
        privacy_delta: diffs.privacy.as_ref().and_then(Families::point),
        exact_copy_excess: fixed.exact_copy_excess,
        bytes: fixed.bytes,
        seed_group: fixed.seed_group.then(|| SeedEvidence {
            seeds: members.len(),
            same_sign: members.iter().filter(|d| same_sign(d)).count(),
        }),
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TaskDelta {
    pub task: String,
    pub families: usize,
    pub delta: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MemberDelta {
    pub arm: String,
    pub delta: f64,
}

/// The paired delta at the matched multiplier (reported only).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Matched {
    pub multiplier: u32,
    pub cells: usize,
    pub families: usize,
    pub delta: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub p_one_sided: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ComparisonResult {
    pub id: String,
    /// `candidate` or `seed_group`.
    pub kind: String,
    pub members: Vec<String>,
    pub baseline: String,
    pub cells: usize,
    pub families: usize,
    pub mlp_cells: usize,
    /// Each side's TRG on the common cells.
    pub candidate_trg: Option<f64>,
    pub baseline_trg: Option<f64>,
    pub delta: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub p_one_sided: f64,
    pub p_holm: f64,
    /// Comparisons in the Holm family of `p_holm` (those of the receipt).
    #[serde(default)]
    pub holm_m: usize,
    pub tasks: Vec<TaskDelta>,
    pub member_deltas: Vec<MemberDelta>,
    pub matched: Option<Matched>,
    pub decision: Decision,
    /// Exact-copy excess of the candidate's scored headline files on the
    /// common cells: the guard reads the mean; max and share over the bound are diagnostics.
    #[serde(default)]
    pub exact_copy: Option<ExactCopy>,
    /// The comparison's sign-flip false-PROMOTE rate (diagnostic, not decided on).
    #[serde(default)]
    pub null_calibration: Option<Calibration>,
    /// The baseline's scored headline packets above the headline budget on
    /// the common cells (scored with a larger cap; diagnostic).
    #[serde(default)]
    pub baseline_above_budget: usize,
}

/// A comparison with its paired evidence and bootstrap, before Holm.
pub struct Prepared {
    pub spec: Spec,
    pub paired: Paired,
    pub fixed: Fixed,
    pub plan: BootPlan,
    pub stats: BootStats,
    pub matched: Option<Matched>,
    pub exact_copy: Option<ExactCopy>,
    pub baseline_above_budget: usize,
}

fn plan_for(ctx: &Context, families: &Families) -> Option<BootPlan> {
    let sizes: Vec<usize> = families.values().iter().map(Vec::len).collect();
    let b = &ctx.contract.bootstrap;
    (!sizes.is_empty()).then(|| BootPlan::new(&sizes, b.draws, ctx.seed, ctx.contract.quantiles()))
}

fn states(ctx: &Context, arm: &Arm, multiplier: u32) -> States {
    cell_states(&ctx.contract.protocol(), arm, multiplier, &|f| {
        ctx.panel.contains(f)
    })
}

/// Packet bytes of the candidate's headline files on the panel. A file was
/// produced when it was scored or carries a measured packet (an over-budget
/// file is produced but not scored).
fn bytes(ctx: &Context, members: &[&Arm]) -> BytesEvidence {
    let budget = ctx.contract.bytes.headline;
    let multiplier = ctx.contract.statistic.primary_multiplier;
    let mut out = BytesEvidence::default();
    for arm in members {
        for row in headline_files(ctx, arm, multiplier).values() {
            let measured = row.packet_bytes > 0;
            out.produced += usize::from(measured || row.status == STATUS_OK);
            out.measured += usize::from(measured);
            out.within += usize::from(measured && row.packet_bytes <= budget);
            out.max_bytes = out.max_bytes.max(row.packet_bytes);
        }
    }
    out
}

pub fn prepare(ctx: &Context, spec: &Spec, baseline: &str) -> ArenaResult<Prepared> {
    let s = &ctx.contract.statistic;
    let members: Vec<&Arm> = spec.members.iter().map(|m| ctx.arm(m)).collect();
    let at = |multiplier: u32| -> (Vec<States>, States) {
        let member_states = members
            .iter()
            .map(|arm| states(ctx, arm, multiplier))
            .collect();
        (member_states, states(ctx, ctx.arm(baseline), multiplier))
    };
    let (member_states, base_states) = at(s.primary_multiplier);
    let paired = pair(&member_states.iter().collect::<Vec<_>>(), &base_states);
    let Some(plan) = plan_for(ctx, &paired.diffs.trg) else {
        return contract(format!(
            "comparison {}: no informative cell is applicable to every arm",
            spec.id
        ));
    };
    let stats = plan.run(&paired.diffs.trg.values());
    let (matched_members, matched_base) = at(s.matched_multiplier);
    let matched_pair = pair(&matched_members.iter().collect::<Vec<_>>(), &matched_base);
    let matched = plan_for(ctx, &matched_pair.diffs.trg).map(|plan| {
        let overall = plan.run(&matched_pair.diffs.trg.values()).overall;
        Matched {
            multiplier: s.matched_multiplier,
            cells: matched_pair.common.len(),
            families: matched_pair.diffs.trg.count(),
            delta: overall.value,
            ci_low: overall.ci_low,
            ci_high: overall.ci_high,
            p_one_sided: overall.p_one_sided,
        }
    });
    // The candidate's scored headline files on the common cells.
    let common = |cell: &str| paired.common.contains(cell);
    let copies = exact_copy(ctx, &members, &common);
    let baseline_above_budget = scored_above_budget(ctx, ctx.arm(baseline), &common);
    let fixed = Fixed {
        exact_copy_excess: copies.as_ref().map(|c| c.mean),
        bytes: bytes(ctx, &members),
        seed_group: spec.seed_group,
    };
    Ok(Prepared {
        spec: spec.clone(),
        paired,
        fixed,
        plan,
        stats,
        matched,
        exact_copy: copies,
        baseline_above_budget,
    })
}

pub fn result(
    prepared: &Prepared,
    baseline: &str,
    (p_holm, holm_m): (f64, usize),
    decision: Decision,
    null_calibration: Option<Calibration>,
) -> ComparisonResult {
    let (spec, paired, stats) = (&prepared.spec, &prepared.paired, &prepared.stats);
    let tasks = paired
        .diffs
        .trg
        .present()
        .into_iter()
        .zip(&stats.per_task)
        .map(|(task, interval)| TaskDelta {
            task: TASKS[task].to_string(),
            families: paired.diffs.trg.by_task[task].len(),
            delta: interval.value,
            ci_low: interval.ci_low,
            ci_high: interval.ci_high,
        })
        .collect();
    ComparisonResult {
        id: spec.id.clone(),
        kind: if spec.seed_group {
            "seed_group"
        } else {
            "candidate"
        }
        .to_string(),
        members: spec.members.clone(),
        baseline: baseline.to_string(),
        cells: paired.common.len(),
        families: paired.diffs.trg.count(),
        mlp_cells: paired.mlp_cells,
        candidate_trg: paired.candidate.point(),
        baseline_trg: paired.baseline.point(),
        delta: stats.overall.value,
        ci_low: stats.overall.ci_low,
        ci_high: stats.overall.ci_high,
        p_one_sided: stats.overall.p_one_sided,
        p_holm,
        holm_m,
        tasks,
        member_deltas: if spec.seed_group {
            spec.members
                .iter()
                .zip(&paired.diffs.member_deltas())
                .map(|(arm, delta)| MemberDelta {
                    arm: arm.clone(),
                    delta: *delta,
                })
                .collect()
        } else {
            Vec::new()
        },
        matched: prepared.matched.clone(),
        decision,
        exact_copy: prepared.exact_copy.clone(),
        null_calibration,
        baseline_above_budget: prepared.baseline_above_budget,
    }
}

/// Every comparison of one receipt, Holm-adjusted together and decided.
pub fn compare_all(
    ctx: &Context,
    specs: &[Spec],
    baseline: &str,
    noise_sd: f64,
) -> ArenaResult<Vec<ComparisonResult>> {
    let prepared = specs
        .iter()
        .map(|spec| prepare(ctx, spec, baseline))
        .collect::<ArenaResult<Vec<_>>>()?;
    let raw: Vec<f64> = prepared
        .iter()
        .map(|p| p.stats.overall.p_one_sided)
        .collect();
    let (rule, budget) = (&ctx.contract.promotion, ctx.contract.bytes.headline);
    let null = &ctx.contract.null;
    let mut out = Vec::new();
    for (p, p_holm) in prepared.iter().zip(holm(&raw)) {
        let decision = decide(
            rule,
            budget,
            noise_sd,
            &evidence(&p.stats, p_holm, &p.paired.diffs, &p.fixed),
        );
        let flips = sign_flips(ctx, p, null.min_replicates, null.seed, noise_sd, None)?;
        let calibration = Calibration::of(&flips, null.max_false_promote_rate);
        let holm = (p_holm, prepared.len());
        out.push(result(p, baseline, holm, decision, Some(calibration)));
    }
    Ok(out)
}

#[derive(Serialize)]
struct Summary<'a> {
    command: &'static str,
    outcomes: BTreeMap<&'a str, super::rule::Outcome>,
    receipt_sha256: String,
}

pub fn run(args: &CompareArgs) -> ArenaResult<String> {
    check_arm_id(&args.baseline)?;
    let specs = parse_specs(&args.candidates, &args.seed_groups)?;
    let mut ids: Vec<&str> = Vec::new();
    for id in specs
        .iter()
        .flat_map(|s| s.members.iter())
        .chain([&args.baseline])
    {
        if !ids.contains(&id.as_str()) {
            ids.push(id.as_str());
        }
    }
    let ctx = Context::open(&args.run, &ids)?;
    let noise = ctx.noise_sd(args.noise_sd)?;
    let comparisons = compare_all(&ctx, &specs, &args.baseline, noise.value)?;
    let mut receipt = ctx.receipt("compare");
    receipt.arms = ids.iter().map(|id| summarize(&ctx, ctx.arm(id))).collect();
    receipt.noise_sd = Some(noise);
    receipt.comparisons = comparisons;
    let receipt_sha256 = write_receipt(&args.out, &receipt)?;
    super::jsonl::canonical_json(&Summary {
        command: "compare",
        outcomes: receipt
            .comparisons
            .iter()
            .map(|c| (c.id.as_str(), c.decision.outcome))
            .collect(),
        receipt_sha256,
    })
}

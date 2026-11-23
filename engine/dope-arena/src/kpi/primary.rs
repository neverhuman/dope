//! `dope_kpi primary`: the TRG of each arm with its stratified family
//! bootstrap interval, and the descriptive columns the README prints.
//!
//! Per arm: TRG at the primary multiplier (4) and, reported separately, at
//! the matched multiplier (1); per task the family count and value; the
//! held-out MLP value; packet bytes of the headline files (median, maximum,
//! share within the headline budget, over the files that carry a packet);
//! the unclipped median retention of each auditor over the scored
//! informative headline entries; the median C2ST AUC over the scored files
//! of the cells in the statistic; the cell accounting (scheduled =
//! uninformative + cells + not applicable, failed cells among the cells, and
//! cells informative for the held-out auditor only); and the inputs of the
//! realism, privacy and exact-copy guards on the cells in the statistic
//! (absent when a needed guard field is absent).
//!
//! The statistic reads the rows as `score` wrote them. An arm scored with a
//! byte cap above the headline budget also reports how many scored packets
//! exceed it and `trg_at_budget`, its TRG with those files over budget.
//! Diagnostics, never decided on: the exact-copy maximum and share of files
//! over the bound beside the mean the guard reads, over-budget files without
//! a measured size, and `trg_failed_at_floor` (a failed entry at
//! `clip_low - r_b` instead of the constant of record).

use super::boot::BootPlan;
use super::receipt::{write_receipt, Context, RunArgs};
use super::rundir::{Arm, KpiRow, STATUS_OK, STATUS_OVER_BUDGET, TASKS};
use super::trg::{
    cell_states, mean, scored, task_index, trg_families, CellScore, Families, TrgCellState,
};
use dope_arena::ArenaResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[derive(clap::Args, Debug)]
pub struct PrimaryArgs {
    #[command(flatten)]
    pub run: RunArgs,
    /// Arm to report (repeat for several).
    #[arg(long = "arm", required = true)]
    pub arms: Vec<String>,
    /// Output receipt (JSON).
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TaskStat {
    pub task: String,
    pub families: usize,
    pub value: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

/// The statistic of one arm at one multiplier.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Statistic {
    pub multiplier: u32,
    pub trg: Option<f64>,
    pub ci_low: Option<f64>,
    pub ci_high: Option<f64>,
    pub families: usize,
    /// Headline cells of the panel scheduled at this multiplier.
    #[serde(default)]
    pub scheduled_cells: usize,
    /// Scheduled cells where no statistic auditor is informative.
    #[serde(default)]
    pub uninformative_cells: usize,
    /// Of those, cells where only the held-out auditor is informative.
    #[serde(default)]
    pub mlp_only_cells: usize,
    /// Applicable informative cells in the statistic.
    pub cells: usize,
    /// Of those, cells with an informative entry that was not scored.
    pub failed_cells: usize,
    /// Cells of the statistic a control cannot exist for:
    /// `scheduled = uninformative + cells + not_applicable`.
    pub not_applicable_cells: usize,
    pub tasks: Vec<TaskStat>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Bytes {
    pub budget: u64,
    pub files: usize,
    pub with_packet: usize,
    pub median: Option<f64>,
    pub max: u64,
    pub within_budget: usize,
    /// `within_budget` over the files with a packet or over budget without one.
    pub share_within_budget: Option<f64>,
    /// Files `score` marked over budget, and those of them without a measured size.
    #[serde(default)]
    pub over_budget: usize,
    #[serde(default)]
    pub over_budget_unmeasured: usize,
    /// Scored files whose packet exceeds `budget` (scored with a larger cap).
    #[serde(default)]
    pub scored_above_budget: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AuditorMedian {
    pub auditor: String,
    pub entries: usize,
    pub median: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ArmSummary {
    pub arm_id: String,
    pub artifact_class: String,
    pub primary: Statistic,
    pub matched: Statistic,
    /// Primary TRG with every scored packet above the headline budget over
    /// budget; present when some scored headline packet exceeds it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trg_at_budget: Option<f64>,
    /// Held-out auditor, aggregated like the statistic.
    pub mlp_trg: Option<f64>,
    pub bytes: Bytes,
    pub median_retention: Vec<AuditorMedian>,
    pub c2st_median: Option<f64>,
    pub c2st_files: usize,
    /// Realism (privacy) coverage of the cells in the statistic, aggregated like the TRG.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub realism_coverage: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy_coverage: Option<f64>,
    /// Mean exact-copy excess of the scored headline files of the cells in
    /// the statistic (the value `promotion.exact_copy` defines).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_copy_excess: Option<f64>,
    /// The same files: mean, maximum and share over the bound (diagnostic).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_copy: Option<ExactCopy>,
    /// Primary TRG with a failed entry at `clip_low - r_b` instead of the
    /// constant of record (diagnostic, never decided on).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trg_failed_at_floor: Option<f64>,
}

/// Exact-copy excess over a set of scored headline files.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ExactCopy {
    pub files: usize,
    /// The guard's value (`promotion.exact_copy`).
    pub mean: f64,
    pub max: f64,
    /// Files above `promotion.exact_copy_excess_le`, and their share.
    pub over_bound: usize,
    pub share_over_bound: f64,
}

pub fn median(mut values: Vec<f64>) -> Option<f64> {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => Some(values[n / 2]),
        _ => Some((values[n / 2 - 1] + values[n / 2]) / 2.0),
    }
}

/// Stratified family bootstrap of `families` under the context's plan parameters.
pub fn bootstrap(ctx: &Context, families: &Families) -> Option<super::boot::BootStats> {
    let values = families.values();
    if values.is_empty() {
        return None;
    }
    let sizes: Vec<usize> = values.iter().map(Vec::len).collect();
    let plan = BootPlan::new(
        &sizes,
        ctx.contract.bootstrap.draws,
        ctx.seed,
        ctx.contract.quantiles(),
    );
    Some(plan.run(&values))
}

/// Headline cells of the panel scheduled at `multiplier`.
fn scheduled_cells(ctx: &Context, multiplier: u32) -> usize {
    let stratum = &ctx.contract.statistic.stratum;
    ctx.run
        .schedule
        .cells
        .values()
        .filter(|c| {
            &c.stratum == stratum
                && c.multipliers.contains(&multiplier)
                && task_index(&c.task).is_some()
                && ctx.panel.contains(&c.family_id)
        })
        .count()
}

pub fn statistic(ctx: &Context, arm: &Arm, multiplier: u32) -> Statistic {
    let protocol = ctx.contract.protocol();
    let states = cell_states(&protocol, arm, multiplier, &|f| ctx.panel.contains(f));
    let families = trg_families(&states);
    let boot = bootstrap(ctx, &families);
    let scored_cells: Vec<_> = states
        .values()
        .filter_map(TrgCellState::score)
        .filter(|c| c.trg.is_some())
        .collect();
    let in_statistic = states.values().filter(|s| s.in_statistic()).count();
    let scheduled = scheduled_cells(ctx, multiplier);
    let tasks = families
        .present()
        .into_iter()
        .zip(boot.iter().flat_map(|b| b.per_task.iter()))
        .map(|(task, interval)| TaskStat {
            task: TASKS[task].to_string(),
            families: families.by_task[task].len(),
            value: interval.value,
            ci_low: interval.ci_low,
            ci_high: interval.ci_high,
        })
        .collect();
    Statistic {
        multiplier,
        trg: boot.as_ref().map(|b| b.overall.value),
        ci_low: boot.as_ref().map(|b| b.overall.ci_low),
        ci_high: boot.as_ref().map(|b| b.overall.ci_high),
        families: families.count(),
        scheduled_cells: scheduled,
        uninformative_cells: scheduled.saturating_sub(in_statistic),
        mlp_only_cells: states.len() - in_statistic,
        cells: scored_cells.len(),
        failed_cells: scored_cells.iter().filter(|c| c.failed).count(),
        not_applicable_cells: in_statistic - scored_cells.len(),
        tasks,
    }
}

/// One row per headline file (cell, sample seed) of the panel at `multiplier`:
/// the first scored row of the file, else its first row.
pub fn headline_files<'a>(
    ctx: &Context,
    arm: &'a Arm,
    multiplier: u32,
) -> BTreeMap<(&'a str, u64), &'a KpiRow> {
    let stratum = &ctx.contract.statistic.stratum;
    let mut files: BTreeMap<(&str, u64), &KpiRow> = BTreeMap::new();
    for row in &arm.rows {
        if row.multiplier != multiplier
            || &row.stratum != stratum
            || !ctx.panel.contains(&row.family_id)
        {
            continue;
        }
        let held = files
            .entry((row.cell_id.as_str(), row.sample_seed))
            .or_insert(row);
        if held.status != STATUS_OK && row.status == STATUS_OK {
            *held = row;
        }
    }
    files
}

/// `exact_copy_excess` of the scored headline files of `arms` whose cell
/// `keep` accepts; absent when there is none or one lacks the value.
pub fn exact_copy(ctx: &Context, arms: &[&Arm], keep: &dyn Fn(&str) -> bool) -> Option<ExactCopy> {
    let multiplier = ctx.contract.statistic.primary_multiplier;
    let bound = ctx.contract.promotion.exact_copy_excess_le;
    let mut values = Vec::new();
    for arm in arms {
        for ((cell, _), row) in headline_files(ctx, arm, multiplier) {
            if keep(cell) && row.status == STATUS_OK {
                values.push(row.exact_copy_excess?);
            }
        }
    }
    let over_bound = values.iter().filter(|&&v| v > bound).count();
    Some(ExactCopy {
        files: values.len(),
        max: values.iter().copied().reduce(f64::max)?,
        share_over_bound: over_bound as f64 / values.len() as f64,
        over_bound,
        mean: mean(values)?,
    })
}

/// Scored headline files of `arm` whose cell `keep` accepts and whose packet
/// exceeds the headline budget.
pub fn scored_above_budget(ctx: &Context, arm: &Arm, keep: &dyn Fn(&str) -> bool) -> usize {
    let budget = ctx.contract.bytes.headline;
    let multiplier = ctx.contract.statistic.primary_multiplier;
    headline_files(ctx, arm, multiplier)
        .into_iter()
        .filter(|((cell, _), row)| {
            keep(cell) && row.status == STATUS_OK && row.packet_bytes > budget
        })
        .count()
}

/// A guard share of the cells in the statistic, aggregated like the TRG;
/// absent when some cell lacks it.
fn coverage(
    states: &BTreeMap<String, TrgCellState>,
    pick: fn(&CellScore) -> Option<f64>,
) -> Option<f64> {
    let cells: Option<Vec<(usize, &str, f64)>> = states
        .values()
        .filter_map(TrgCellState::score)
        .filter(|c| c.trg.is_some())
        .map(|c| pick(c).map(|share| (c.task, c.family_id.as_str(), share)))
        .collect();
    Families::from_cells(cells?).point()
}

fn bytes(ctx: &Context, files: &BTreeMap<(&str, u64), &KpiRow>) -> Bytes {
    let budget = ctx.contract.bytes.headline;
    let sizes: Vec<u64> = files
        .values()
        .map(|r| r.packet_bytes)
        .filter(|&b| b > 0)
        .collect();
    let within = sizes.iter().filter(|&&b| b <= budget).count();
    let over = |r: &&&KpiRow| r.status == STATUS_OVER_BUDGET;
    let unmeasured = files
        .values()
        .filter(over)
        .filter(|r| r.packet_bytes == 0)
        .count();
    let considered = sizes.len() + unmeasured;
    Bytes {
        budget,
        files: files.len(),
        with_packet: sizes.len(),
        median: median(sizes.iter().map(|&b| b as f64).collect()),
        max: sizes.iter().copied().max().unwrap_or(0),
        within_budget: within,
        share_within_budget: (considered > 0).then(|| within as f64 / considered as f64),
        over_budget: files.values().filter(over).count(),
        over_budget_unmeasured: unmeasured,
        scored_above_budget: files
            .values()
            .filter(|r| r.status == STATUS_OK && r.packet_bytes > budget)
            .count(),
    }
}

/// The arm's artifact class: that of its scored files, else of any file.
fn artifact_class(files: &BTreeMap<(&str, u64), &KpiRow>) -> String {
    let classes = |ok: bool| -> BTreeSet<&str> {
        files
            .values()
            .filter(|r| !ok || r.status == STATUS_OK)
            .map(|r| r.artifact_class.as_str())
            .filter(|c| !c.is_empty())
            .collect()
    };
    let mut found = classes(true);
    if found.is_empty() {
        found = classes(false);
    }
    match found.len() {
        0 => "unknown".to_string(),
        _ => found.into_iter().collect::<Vec<_>>().join("+"),
    }
}

pub fn summarize(ctx: &Context, arm: &Arm) -> ArmSummary {
    let s = &ctx.contract.statistic;
    let protocol = ctx.contract.protocol();
    let keep = |f: &str| ctx.panel.contains(f);
    let states = cell_states(&protocol, arm, s.primary_multiplier, &keep);
    let mlp = Families::from_cells(
        states
            .values()
            .filter_map(TrgCellState::score)
            .filter_map(|c| c.mlp.map(|v| (c.task, c.family_id.as_str(), v))),
    );
    let files = headline_files(ctx, arm, s.primary_multiplier);
    let in_statistic = |cell: &str| {
        states
            .get(cell)
            .and_then(TrgCellState::score)
            .is_some_and(|c| c.trg.is_some())
    };
    let c2st: Vec<f64> = files
        .iter()
        .filter(|((cell, _), row)| in_statistic(cell) && row.status == STATUS_OK)
        .filter_map(|(_, row)| row.c2st_auc)
        .collect();
    let auditors = s
        .auditors
        .iter()
        .chain(std::iter::once(&s.held_out_auditor));
    let median_retention = auditors
        .map(|auditor| {
            let values: Vec<f64> = arm
                .rows
                .iter()
                .filter(|r| {
                    &r.auditor == auditor
                        && r.multiplier == s.primary_multiplier
                        && r.stratum == s.stratum
                        && r.informative
                        && keep(&r.family_id)
                })
                .filter_map(scored)
                .collect();
            AuditorMedian {
                auditor: auditor.clone(),
                entries: values.len(),
                median: median(values),
            }
        })
        .collect();
    let bytes = bytes(ctx, &files);
    let at_budget = || {
        let capped = ctx.contract.protocol_at_headline();
        trg_families(&cell_states(&capped, arm, s.primary_multiplier, &keep)).point()
    };
    let copies = exact_copy(ctx, &[arm], &in_statistic);
    let floor = ctx.contract.protocol_failed_at_floor();
    let trg_failed_at_floor =
        trg_families(&cell_states(&floor, arm, s.primary_multiplier, &keep)).point();
    ArmSummary {
        arm_id: arm.id.clone(),
        artifact_class: artifact_class(&files),
        primary: statistic(ctx, arm, s.primary_multiplier),
        matched: statistic(ctx, arm, s.matched_multiplier),
        trg_at_budget: (bytes.scored_above_budget > 0).then(at_budget).flatten(),
        mlp_trg: mlp.point(),
        bytes,
        median_retention,
        c2st_files: c2st.len(),
        c2st_median: median(c2st),
        realism_coverage: coverage(&states, |c| c.realism),
        privacy_coverage: coverage(&states, |c| c.privacy),
        exact_copy_excess: copies.as_ref().map(|c| c.mean),
        exact_copy: copies,
        trg_failed_at_floor,
    }
}

#[derive(Serialize)]
struct Summary<'a> {
    command: &'static str,
    arms: BTreeMap<&'a str, Option<f64>>,
    receipt_sha256: String,
}

pub fn run(args: &PrimaryArgs) -> ArenaResult<String> {
    let ids: Vec<&str> = args.arms.iter().map(String::as_str).collect();
    let ctx = Context::open(&args.run, &ids)?;
    let mut receipt = ctx.receipt("primary");
    let mut seen = BTreeSet::new();
    for id in ids.iter().filter(|id| seen.insert(**id)) {
        receipt.arms.push(summarize(&ctx, ctx.arm(id)));
    }
    let receipt_sha256 = write_receipt(&args.out, &receipt)?;
    super::jsonl::canonical_json(&Summary {
        command: "primary",
        arms: receipt
            .arms
            .iter()
            .map(|a| (a.arm_id.as_str(), a.primary.trg))
            .collect(),
        receipt_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn medians_interpolate_the_middle_pair() {
        assert_eq!(median(vec![3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(vec![4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(median(Vec::new()), None);
    }
}

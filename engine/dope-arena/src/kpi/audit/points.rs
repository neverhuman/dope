use super::{check, same};
use crate::kpi::compare::{ComparisonResult, Matched};
use crate::kpi::primary::{Bytes as ReceiptBytes, Statistic};
use crate::kpi::receipt::Context;
use crate::kpi::rundir::{Arm, KpiRow, STATUS_OK, STATUS_OVER_BUDGET};
use crate::kpi::trg::{is_control_arm, not_applicable};
use dope_arena::ArenaResult;
use serde::Serialize;
use std::collections::BTreeMap;

fn average(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let (sum, count) = values
        .into_iter()
        .fold((0.0, 0), |(sum, n), v| (sum + v, n + 1));
    (count > 0).then(|| sum / f64::from(count))
}

type Cells<'a> = BTreeMap<&'a str, f64>;
type Group<'a> = (&'a str, &'a str);

#[derive(Default)]
pub(super) struct Points<'a> {
    families: BTreeMap<Group<'a>, Vec<f64>>,
}

impl<'a> Points<'a> {
    fn push(&mut self, group: Group<'a>, value: f64) {
        self.families.entry(group).or_default().push(value);
    }

    fn tasks(&self) -> BTreeMap<&'a str, Vec<f64>> {
        let mut tasks: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        for (&(task, _), values) in &self.families {
            if let Some(mean) = average(values.iter().copied()) {
                tasks.entry(task).or_default().push(mean);
            }
        }
        tasks
    }

    pub(super) fn point(&self) -> Option<f64> {
        average(self.tasks().into_values().filter_map(average))
    }

    fn verify_tasks<'b>(
        &self,
        expected: impl Iterator<Item = (&'b str, usize, f64)>,
    ) -> ArenaResult<()> {
        let mut tasks = self.tasks();
        for (task, families, value) in expected {
            check(
                tasks
                    .remove(task)
                    .is_some_and(|v| v.len() == families && same(average(v), Some(value))),
                "task point and denominator",
            )?;
        }
        check(tasks.is_empty(), "task roster")
    }
}

#[derive(Default)]
pub(super) struct Slice<'a> {
    pub(super) multiplier: u32,
    pub(super) trg: Cells<'a>,
    groups: BTreeMap<&'a str, Group<'a>>,
    floor_cells: Cells<'a>,
    unclipped_cells: Cells<'a>,
    mlp_cells: Cells<'a>,
    pub(super) points: Points<'a>,
    pub(super) floor: Points<'a>,
    pub(super) unclipped: Points<'a>,
    pub(super) mlp: Points<'a>,
    scheduled: usize,
    failed: usize,
    inapplicable: usize,
    uninformative: usize,
    mlp_only: usize,
}

impl<'a> Slice<'a> {
    pub(super) fn read(ctx: &'a Context, arm: &'a Arm, multiplier: u32, legacy: bool) -> Self {
        let spec = &ctx.contract.statistic;
        let failed_at_floor = ctx.contract.protocol().failed_at_floor;
        let mut out = Self {
            multiplier,
            ..Self::default()
        };
        let control = is_control_arm(&arm.id);
        let mut units: BTreeMap<&str, BTreeMap<&str, Vec<&KpiRow>>> = BTreeMap::new();
        for row in &arm.rows {
            if row.multiplier == multiplier
                && row.informative
                && row.stratum == spec.stratum
                && ctx.panel.contains(&row.family_id)
                && (row.auditor == spec.held_out_auditor || spec.auditors.contains(&row.auditor))
            {
                units
                    .entry(&row.cell_id)
                    .or_default()
                    .entry(&row.auditor)
                    .or_default()
                    .push(row);
            }
        }
        let entry = |row: &KpiRow, floor: bool, clip: bool| {
            if let (STATUS_OK, Some(r_s), Some(r_b)) = (row.status.as_str(), row.r_s, row.r_b) {
                let retention = if clip {
                    r_s.clamp(spec.clip_low, spec.clip_high)
                } else {
                    r_s
                };
                retention - r_b
            } else if floor {
                row.r_b.map_or(spec.failed_value, |r_b| spec.clip_low - r_b)
            } else {
                spec.failed_value
            }
        };
        for (id, cell) in &ctx.run.schedule.cells {
            if cell.stratum != spec.stratum
                || !ctx.panel.contains(&cell.family_id)
                || !cell.multipliers.contains(&multiplier)
            {
                continue;
            }
            out.scheduled += 1;
            let Some(mut units) = units.remove(id.as_str()) else {
                out.uninformative += 1;
                continue;
            };
            let held = units.remove(spec.held_out_auditor.as_str());
            let excluded = control
                && units
                    .values()
                    .chain(&held)
                    .flatten()
                    .any(|r| not_applicable(r));
            if units.is_empty() {
                out.mlp_only += 1;
            }
            if excluded && (legacy || !units.is_empty()) {
                out.inapplicable += 1;
                continue;
            }
            let group = (cell.task.as_str(), cell.family_id.as_str());
            out.groups.insert(id, group);
            if !excluded {
                if let Some(value) = held.and_then(|rows| {
                    average(rows.into_iter().map(|r| entry(r, failed_at_floor, true)))
                }) {
                    out.mlp_cells.insert(id, value);
                    out.mlp.push(group, value);
                }
            }
            let worst = |floor, clip| {
                units
                    .values()
                    .filter_map(|rows| average(rows.iter().map(|r| entry(r, floor, clip))))
                    .reduce(f64::min)
            };
            let Some(value) = worst(failed_at_floor, true) else {
                out.uninformative += 1;
                continue;
            };
            out.failed += usize::from(
                units
                    .values()
                    .flatten()
                    .any(|r| r.status != STATUS_OK || r.r_s.is_none() || r.r_b.is_none()),
            );
            out.trg.insert(id, value);
            out.points.push(group, value);
            if let Some(value) = worst(true, true) {
                out.floor_cells.insert(id, value);
                out.floor.push(group, value);
            }
            if let Some(value) = worst(failed_at_floor, false) {
                out.unclipped_cells.insert(id, value);
                out.unclipped.push(group, value);
            }
        }
        out
    }

    pub(super) fn verify(&self, stat: &Statistic, legacy: bool) -> ArenaResult<()> {
        check(
            stat.multiplier == self.multiplier
                && stat.cells == self.trg.len()
                && stat.families == self.points.families.len()
                && stat.not_applicable_cells == self.inapplicable
                && stat.failed_cells == self.failed
                && (legacy
                    || (stat.scheduled_cells == self.scheduled
                        && stat.uninformative_cells == self.uninformative
                        && stat.mlp_only_cells == self.mlp_only)),
            "arm denominator",
        )?;
        check(same(stat.trg, self.points.point()), "arm point")?;
        self.points.verify_tasks(
            stat.tasks
                .iter()
                .map(|t| (t.task.as_str(), t.families, t.value)),
        )
    }
}

#[derive(Default)]
pub(super) struct Paired<'a> {
    pub(super) cells: usize,
    pub(super) floor: Points<'a>,
    pub(super) unclipped: Points<'a>,
    mlp_cells: usize,
    pub(super) delta: Points<'a>,
    base: Points<'a>,
    candidate: Points<'a>,
    members: Vec<Points<'a>>,
}

pub(super) fn paired<'a>(base: &Slice<'a>, members: &[&Slice<'a>]) -> Paired<'a> {
    let mut out = Paired {
        members: (0..members.len()).map(|_| Points::default()).collect(),
        ..Paired::default()
    };
    for (&cell, &value) in &base.trg {
        let scores: Option<Vec<_>> = members.iter().map(|s| s.trg.get(cell).copied()).collect();
        let Some(scores) = scores else { continue };
        let group = base.groups[cell];
        let Some(candidate) = average(scores.iter().copied()) else {
            continue;
        };
        out.cells += 1;
        out.delta.push(group, candidate - value);
        out.base.push(group, value);
        out.candidate.push(group, candidate);
        for (member, score) in out.members.iter_mut().zip(scores) {
            member.push(group, score - value);
        }
        if let Some(value) = average(members.iter().map(|s| s.floor_cells[cell])) {
            out.floor.push(group, value - base.floor_cells[cell]);
        }
        if let Some(value) = average(members.iter().map(|s| s.unclipped_cells[cell])) {
            out.unclipped
                .push(group, value - base.unclipped_cells[cell]);
        }
    }
    out.mlp_cells = base
        .mlp_cells
        .keys()
        .filter(|cell| members.iter().all(|s| s.mlp_cells.contains_key(*cell)))
        .count();
    out
}

impl Paired<'_> {
    pub(super) fn verify(&self, comparison: &ComparisonResult) -> ArenaResult<()> {
        check(
            self.cells == comparison.cells
                && self.delta.families.len() == comparison.families
                && self.mlp_cells == comparison.mlp_cells,
            "comparison denominator",
        )?;
        check(
            same(self.delta.point(), Some(comparison.delta))
                && same(self.base.point(), comparison.baseline_trg)
                && same(self.candidate.point(), comparison.candidate_trg),
            "paired point",
        )?;
        let members_match = if comparison.kind == "seed_group" {
            comparison.member_deltas.len() == self.members.len()
                && self
                    .members
                    .iter()
                    .zip(comparison.members.iter().zip(&comparison.member_deltas))
                    .all(|(m, (id, want))| id == &want.arm && same(m.point(), Some(want.delta)))
        } else {
            comparison.member_deltas.is_empty()
        };
        check(members_match, "member points")?;
        self.delta.verify_tasks(
            comparison
                .tasks
                .iter()
                .map(|t| (t.task.as_str(), t.families, t.delta)),
        )
    }

    pub(super) fn verify_matched(
        &self,
        multiplier: u32,
        matched: Option<&Matched>,
    ) -> ArenaResult<()> {
        check(
            match matched {
                Some(want) => {
                    self.cells == want.cells
                        && self.delta.families.len() == want.families
                        && want.multiplier == multiplier
                        && same(self.delta.point(), Some(want.delta))
                }
                None => self.cells == 0,
            },
            "matched comparison",
        )
    }
}

#[derive(Serialize)]
pub(super) struct Factorial<'a> {
    multiplier: u32,
    cells: usize,
    x_main_effect: Effect<'a>,
    y_main_effect: Effect<'a>,
    interaction: Effect<'a>,
}

#[derive(Serialize)]
struct Effect<'a> {
    value: Option<f64>,
    tasks: BTreeMap<&'a str, TaskEffect>,
}

#[derive(Serialize)]
struct TaskEffect {
    families: usize,
    value: Option<f64>,
}

impl<'a> From<Points<'a>> for Effect<'a> {
    fn from(points: Points<'a>) -> Self {
        Self {
            value: points.point(),
            tasks: points
                .tasks()
                .into_iter()
                .map(|(task, values)| {
                    (
                        task,
                        TaskEffect {
                            families: values.len(),
                            value: average(values),
                        },
                    )
                })
                .collect(),
        }
    }
}

pub(super) fn factorial<'a>(
    pp: &Slice<'a>,
    rp: &Slice<'a>,
    pt: &Slice<'a>,
    rt: &Slice<'a>,
) -> Factorial<'a> {
    let (mut x, mut y, mut interaction) = (Points::default(), Points::default(), Points::default());
    let mut cells = 0;
    for (&cell, &pp_score) in &pp.trg {
        if let (Some(&rp), Some(&pt), Some(&rt)) =
            (rp.trg.get(cell), pt.trg.get(cell), rt.trg.get(cell))
        {
            let group = pp.groups[cell];
            cells += 1;
            x.push(group, ((rp - pp_score) + (rt - pt)) / 2.0);
            y.push(group, ((pt - pp_score) + (rt - rp)) / 2.0);
            interaction.push(group, (rt - pt) - (rp - pp_score));
        }
    }
    Factorial {
        multiplier: pp.multiplier,
        cells,
        x_main_effect: x.into(),
        y_main_effect: y.into(),
        interaction: interaction.into(),
    }
}

pub(super) struct Bytes(ReceiptBytes);

impl Bytes {
    pub(super) fn read(ctx: &Context, arm: &Arm) -> ArenaResult<Self> {
        let spec = &ctx.contract.statistic;
        let cap = ctx.contract.bytes.headline;
        let mut files = BTreeMap::new();
        for row in &arm.rows {
            if row.multiplier != spec.primary_multiplier
                || row.stratum != spec.stratum
                || !ctx.panel.contains(&row.family_id)
            {
                continue;
            }
            let held = files
                .entry((row.cell_id.as_str(), row.sample_seed))
                .or_insert(row);
            check(
                held.packet_bytes == row.packet_bytes,
                "ledger byte consistency",
            )?;
            if held.status != STATUS_OK && row.status == STATUS_OK {
                *held = row;
            }
        }
        let mut sizes: Vec<_> = files
            .values()
            .map(|r| r.packet_bytes)
            .filter(|&b| b > 0)
            .collect();
        sizes.sort_unstable();
        let within = sizes.partition_point(|&b| b <= cap);
        let over = |r: &&&KpiRow| r.status == STATUS_OVER_BUDGET;
        let unmeasured = files
            .values()
            .filter(over)
            .filter(|r| r.packet_bytes == 0)
            .count();
        let considered = sizes.len() + unmeasured;
        let median = (!sizes.is_empty()).then(|| {
            let low = sizes[(sizes.len() - 1) / 2] as f64;
            let high = sizes[sizes.len() / 2] as f64;
            (low + high) / 2.0
        });
        Ok(Self(ReceiptBytes {
            budget: cap,
            files: files.len(),
            with_packet: sizes.len(),
            median,
            max: sizes.last().copied().unwrap_or(0),
            within_budget: within,
            share_within_budget: (considered > 0).then(|| within as f64 / considered as f64),
            over_budget: files.values().filter(over).count(),
            over_budget_unmeasured: unmeasured,
            scored_above_budget: files
                .values()
                .filter(|r| r.status == STATUS_OK && r.packet_bytes > cap)
                .count(),
        }))
    }

    pub(super) fn verify(&self, expected: &ReceiptBytes, legacy: bool) -> ArenaResult<()> {
        let actual = &self.0;
        let mut comparable = actual.clone();
        if legacy {
            comparable.over_budget = expected.over_budget;
            comparable.over_budget_unmeasured = expected.over_budget_unmeasured;
            comparable.scored_above_budget = expected.scored_above_budget;
        }
        check(
            same(actual.share_within_budget, expected.share_within_budget),
            "byte share",
        )?;
        comparable.share_within_budget = expected.share_within_budget;
        check(comparable == *expected, "byte accounting")
    }
}

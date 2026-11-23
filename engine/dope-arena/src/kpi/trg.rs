//! The twin-referenced retention gap (TRG), the scoreboard's primary statistic.
//!
//! `R_x = (L0 - L_x) / (L0 - L_A)` is the retention of an auditor fitted on
//! `x`; rows carry `r_s` for the synthetic sample S and `r_b` for the real
//! twin B. For one informative (cell, auditor) unit of a slice (the headline
//! stratum at one multiplier):
//!
//! ```text
//! x = mean over sample seeds of v
//! v = clip(r_s, -1, 1.25) - r_b   for a scored entry
//! v = -1                          for an entry that was not scored
//! ```
//!
//! "Not scored" is failed, over budget, single class, or unsupported by a
//! generator: failures stay in the denominator. A cell takes the minimum of
//! `x` over `linear-seeded-v1` and `gbdt`; `mlp-seeded-v1` is computed the
//! same way and kept apart for the direction guard. Cells are averaged within
//! a family, families within a task, and the tasks present equally. Units
//! that are uninformative by the real-data rule are left out for every arm.
//!
//! Applicability: an informative entry with status `unsupported` from a
//! `control` arm marks a control that cannot exist for the cell (`twin_c` on a
//! table with fewer than 16 C rows). The cell then leaves that arm's
//! statistic, and a comparison keeps only cells applicable to every arm it
//! reads. A generator that declares a cell unsupported has failed it.

use super::emitlog::CLASS_CONTROL;
use super::rundir::{Arm, KpiRow, STATUS_OK, STATUS_UNSUPPORTED, TASKS};
use std::collections::{BTreeMap, BTreeSet};

/// Parameters of the statistic, read from the contract.
#[derive(Clone, Debug, PartialEq)]
pub struct Protocol {
    pub clip_low: f64,
    pub clip_high: f64,
    pub failed_value: f64,
    /// Auditors of the statistic; a cell takes the minimum over them.
    pub auditors: Vec<String>,
    /// Held out of the statistic; the direction guard reads it.
    pub held_out: String,
    pub stratum: String,
}

pub fn task_index(task: &str) -> Option<usize> {
    TASKS.iter().position(|t| *t == task)
}

/// A control that cannot exist for the entry's cell.
pub fn not_applicable(row: &KpiRow) -> bool {
    row.status == STATUS_UNSUPPORTED && row.artifact_class == CLASS_CONTROL
}

/// A scored entry: status `ok` with a retention.
pub fn scored(row: &KpiRow) -> Option<f64> {
    row.r_s.filter(|_| row.status == STATUS_OK)
}

pub fn mean(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let (mut sum, mut count) = (0.0, 0usize);
    for value in values {
        sum += value;
        count += 1;
    }
    (count > 0).then(|| sum / count as f64)
}

/// Passes of a guard flag over entries; `unknown` when a scored entry lacks the flag.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Tally {
    passed: usize,
    total: usize,
    unknown: bool,
}

impl Tally {
    fn add(&mut self, row: &KpiRow, flag: Option<bool>) {
        self.total += 1;
        if row.status == STATUS_OK {
            match flag {
                Some(pass) => self.passed += usize::from(pass),
                None => self.unknown = true,
            }
        }
    }

    fn merge(self, other: Tally) -> Tally {
        Tally {
            passed: self.passed + other.passed,
            total: self.total + other.total,
            unknown: self.unknown || other.unknown,
        }
    }

    fn share(self) -> Option<f64> {
        (!self.unknown && self.total > 0).then(|| self.passed as f64 / self.total as f64)
    }
}

/// One informative (cell, auditor) unit.
#[derive(Clone, Debug, PartialEq)]
struct Unit {
    value: f64,
    failed: usize,
    not_applicable: usize,
    realism: Tally,
    privacy: Tally,
}

/// The unit of `entries` (one cell, one auditor, one multiplier), if informative.
fn unit<'a>(p: &Protocol, entries: impl Iterator<Item = &'a KpiRow>) -> Option<Unit> {
    let mut out = Unit {
        value: 0.0,
        failed: 0,
        not_applicable: 0,
        realism: Tally::default(),
        privacy: Tally::default(),
    };
    let mut values = Vec::new();
    for row in entries.filter(|row| row.informative) {
        let value = match (scored(row), row.r_b) {
            (Some(r_s), Some(r_b)) => r_s.clamp(p.clip_low, p.clip_high) - r_b,
            _ => {
                out.failed += 1;
                out.not_applicable += usize::from(not_applicable(row));
                p.failed_value
            }
        };
        values.push(value);
        out.realism.add(row, row.realism_pass);
        out.privacy.add(row, row.privacy_pass);
    }
    out.value = mean(values)?;
    Some(out)
}

/// An informative cell of one arm in one slice.
#[derive(Clone, Debug, PartialEq)]
pub struct CellScore {
    pub family_id: String,
    pub task: usize,
    /// Minimum over the statistic's informative auditors (`None`: none is informative).
    pub trg: Option<f64>,
    /// The held-out auditor's unit value, when informative.
    pub mlp: Option<f64>,
    /// Some informative entry of the statistic was not scored.
    pub failed: bool,
    /// Share of the statistic's entries passing the realism gate (`None`: not recorded).
    pub realism: Option<f64>,
    pub privacy: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CellState {
    NotApplicable,
    Scored(CellScore),
}

impl CellState {
    pub fn score(&self) -> Option<&CellScore> {
        match self {
            CellState::Scored(score) => Some(score),
            CellState::NotApplicable => None,
        }
    }
}

/// The informative cells of `arm` in the protocol's stratum at `multiplier`,
/// restricted to the families `keep` accepts; uninformative cells are absent.
pub fn cell_states(
    p: &Protocol,
    arm: &Arm,
    multiplier: u32,
    keep: &dyn Fn(&str) -> bool,
) -> BTreeMap<String, CellState> {
    let mut out = BTreeMap::new();
    let rows = &arm.rows;
    let mut start = 0;
    while start < rows.len() {
        let cell_id = &rows[start].cell_id;
        let len = rows[start..]
            .iter()
            .take_while(|r| &r.cell_id == cell_id)
            .count();
        let group = &rows[start..start + len];
        start += len;
        let first = &group[0];
        let Some(task) = task_index(&first.task) else {
            continue;
        };
        if first.stratum != p.stratum || !keep(&first.family_id) {
            continue;
        }
        let unit_of = |auditor: &str| {
            unit(
                p,
                group
                    .iter()
                    .filter(|r| r.auditor == auditor && r.multiplier == multiplier),
            )
        };
        let stat: Vec<Unit> = p.auditors.iter().filter_map(|a| unit_of(a)).collect();
        let mlp = unit_of(&p.held_out);
        if stat.is_empty() && mlp.is_none() {
            continue;
        }
        let state = if stat.iter().chain(&mlp).any(|u| u.not_applicable > 0) {
            CellState::NotApplicable
        } else {
            let total = |pick: fn(&Unit) -> Tally| {
                stat.iter()
                    .map(pick)
                    .fold(Tally::default(), Tally::merge)
                    .share()
            };
            CellState::Scored(CellScore {
                family_id: first.family_id.clone(),
                task,
                trg: stat.iter().map(|u| u.value).reduce(f64::min),
                mlp: mlp.as_ref().map(|u| u.value),
                failed: stat.iter().any(|u| u.failed > 0),
                realism: total(|u| u.realism),
                privacy: total(|u| u.privacy),
            })
        };
        out.insert(cell_id.clone(), state);
    }
    out
}

/// Family values of one statistic, by task.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Families {
    /// Per task (`TASKS` order): family id -> mean of its cell values.
    pub by_task: [BTreeMap<String, f64>; 3],
}

impl Families {
    /// Mean of the cell values within each (task, family).
    pub fn from_cells<'a>(cells: impl IntoIterator<Item = (usize, &'a str, f64)>) -> Families {
        let mut sums: [BTreeMap<&str, (f64, usize)>; 3] = Default::default();
        for (task, family, value) in cells {
            let held = sums[task].entry(family).or_insert((0.0, 0));
            held.0 += value;
            held.1 += 1;
        }
        let mut out = Families::default();
        for (task, sums) in sums.into_iter().enumerate() {
            out.by_task[task] = sums
                .into_iter()
                .map(|(family, (sum, n))| (family.to_string(), sum / n as f64))
                .collect();
        }
        out
    }

    pub fn task_mean(&self, task: usize) -> Option<f64> {
        mean(self.by_task[task].values().copied())
    }

    /// Equal-weight mean of the task means over the tasks present.
    pub fn point(&self) -> Option<f64> {
        mean((0..TASKS.len()).filter_map(|task| self.task_mean(task)))
    }

    /// Tasks holding at least one family, in `TASKS` order.
    pub fn present(&self) -> Vec<usize> {
        (0..TASKS.len())
            .filter(|&t| !self.by_task[t].is_empty())
            .collect()
    }

    /// Family values of each present task, in family-id order.
    pub fn values(&self) -> Vec<Vec<f64>> {
        self.present()
            .into_iter()
            .map(|t| self.by_task[t].values().copied().collect())
            .collect()
    }

    pub fn count(&self) -> usize {
        self.by_task.iter().map(BTreeMap::len).sum()
    }

    pub fn ids(&self) -> BTreeSet<&str> {
        self.by_task
            .iter()
            .flat_map(|families| families.keys().map(String::as_str))
            .collect()
    }

    /// The same families with every value multiplied by the sign of its family.
    pub fn flipped(&self, sign: &dyn Fn(&str) -> f64) -> Families {
        let mut out = self.clone();
        for families in &mut out.by_task {
            for (family, value) in families.iter_mut() {
                *value *= sign(family);
            }
        }
        out
    }
}

/// Families of the TRG values of the scored cells.
pub fn trg_families(cells: &BTreeMap<String, CellState>) -> Families {
    Families::from_cells(
        cells
            .values()
            .filter_map(CellState::score)
            .filter_map(|c| c.trg.map(|value| (c.task, c.family_id.as_str(), value))),
    )
}

#[cfg(test)]
mod tests {
    use super::super::stat_testing::{arm, protocol, row, LINEAR, MLP};
    use super::*;

    fn all(_: &str) -> bool {
        true
    }

    #[test]
    fn units_clip_floor_and_take_the_minimum_over_auditors() {
        let mut rows = vec![
            row("t1", "f1", "regression", LINEAR, 2.0), // clip 1.25 - 1 = 0.25
            row("t1", "f1", "regression", "gbdt", 0.75), // -0.25 (minimum)
            row("t1", "f1", "regression", MLP, -3.0),   // clip -1 - 1 = -2
        ];
        let mut failed = row("t2", "f2", "binary", LINEAR, 0.0);
        failed.status = "failed".into();
        failed.r_s = None;
        rows.push(failed); // -1, not -1 - R_B
        rows.push(row("t2", "f2", "binary", "gbdt", -0.5)); // -1.5 is below the floor value
        let cells = cell_states(&protocol(), &arm(rows), 4, &all);
        let t1 = cells["p:t1:full:d0"].score().unwrap();
        assert_eq!(
            (t1.trg, t1.mlp, t1.failed),
            (Some(-0.25), Some(-2.0), false)
        );
        let t2 = cells["p:t2:full:d0"].score().unwrap();
        assert_eq!((t2.trg, t2.mlp, t2.failed), (Some(-1.5), None, true));
        assert_eq!((t1.realism, t1.privacy), (None, None));
    }

    #[test]
    fn ties_between_auditors_and_sample_seeds_average_first() {
        let mut second = row("t1", "f1", "binary", LINEAR, 0.0);
        second.sample_seed = 1;
        let rows = vec![
            row("t1", "f1", "binary", LINEAR, 1.0),
            second,                                 // linear: mean(0, -1) = -0.5
            row("t1", "f1", "binary", "gbdt", 0.5), // gbdt: -0.5, a tie
        ];
        let cells = cell_states(&protocol(), &arm(rows), 4, &all);
        assert_eq!(cells["p:t1:full:d0"].score().unwrap().trg, Some(-0.5));
    }

    #[test]
    fn uninformative_units_stratum_multiplier_and_panel_filter_drop_out() {
        let mut quiet = row("t1", "f1", "regression", LINEAR, 0.1);
        quiet.informative = false;
        let mut small = row("t2", "f2", "regression", LINEAR, 0.1);
        small.stratum = "small_n".into();
        let mut matched = row("t3", "f3", "regression", LINEAR, 0.1);
        matched.multiplier = 1;
        let rows = vec![
            quiet,
            row("t1", "f1", "regression", "gbdt", 1.0),
            small,
            matched,
            row("t4", "f4", "regression", LINEAR, 0.1),
        ];
        let arm = arm(rows);
        let cells = cell_states(&protocol(), &arm, 4, &all);
        let ids: Vec<&str> = cells.keys().map(String::as_str).collect();
        assert_eq!(ids, ["p:t1:full:d0", "p:t4:full:d0"]);
        assert_eq!(cells["p:t1:full:d0"].score().unwrap().trg, Some(0.0));
        assert_eq!(cell_states(&protocol(), &arm, 1, &all).len(), 1);
        let only_f4 = |family: &str| family == "f4";
        assert_eq!(cell_states(&protocol(), &arm, 4, &only_f4).len(), 1);
    }

    #[test]
    fn a_control_that_cannot_exist_is_not_applicable_but_a_generator_fails() {
        let mut twin = row("t1", "f1", "regression", LINEAR, 0.0);
        twin.status = STATUS_UNSUPPORTED.into();
        twin.artifact_class = CLASS_CONTROL.into();
        twin.r_s = None;
        let mut generator = twin.clone();
        generator.artifact_class = "source_free".into();
        let control = cell_states(&protocol(), &arm(vec![twin]), 4, &all);
        assert_eq!(control["p:t1:full:d0"], CellState::NotApplicable);
        let source_free = cell_states(&protocol(), &arm(vec![generator]), 4, &all);
        assert_eq!(source_free["p:t1:full:d0"].score().unwrap().trg, Some(-1.0));
    }

    #[test]
    fn guard_flags_are_shares_over_entries_or_unknown() {
        let mut a = row("t1", "f1", "regression", LINEAR, 1.0);
        a.realism_pass = Some(true);
        a.privacy_pass = Some(false);
        let mut b = row("t1", "f1", "regression", "gbdt", 1.0);
        b.realism_pass = Some(false);
        b.privacy_pass = Some(false);
        let mut c = row("t2", "f2", "regression", LINEAR, 1.0);
        c.status = "failed".into(); // counts as a miss without a flag
        let cells = cell_states(&protocol(), &arm(vec![a.clone(), b, c]), 4, &all);
        let t1 = cells["p:t1:full:d0"].score().unwrap();
        assert_eq!((t1.realism, t1.privacy), (Some(0.5), Some(0.0)));
        assert_eq!(cells["p:t2:full:d0"].score().unwrap().realism, Some(0.0));
        let unflagged = row("t1", "f1", "regression", "gbdt", 1.0);
        let cells = cell_states(&protocol(), &arm(vec![a, unflagged]), 4, &all);
        assert_eq!(cells["p:t1:full:d0"].score().unwrap().realism, None);
    }

    #[test]
    fn families_average_cells_then_tasks_present_equally() {
        let cells = [
            (0, "f1", 0.0),
            (0, "f1", -1.0), // f1 = -0.5
            (0, "f2", 0.5),  // regression = 0.0
            (2, "f3", -0.3), // multiclass = -0.3; binary absent
        ];
        let families = Families::from_cells(cells);
        assert_eq!(families.task_mean(0), Some(0.0));
        assert_eq!(families.task_mean(1), None);
        assert_eq!(families.point(), Some(-0.15));
        assert_eq!(families.present(), [0, 2]);
        assert_eq!(families.values(), [vec![-0.5, 0.5], vec![-0.3]]);
        assert_eq!((families.count(), families.ids().len()), (3, 3));
        let flipped = families.flipped(&|f| if f == "f2" { -1.0 } else { 1.0 });
        assert_eq!(flipped.values(), [vec![-0.5, -0.5], vec![-0.3]]);
        assert_eq!(Families::default().point(), None);
    }
}

//! Which arms a comparison reads, and its cell-by-cell pairing.
//!
//! A comparison is one candidate arm, or a `--seed-group NAME=A,B,C` whose
//! members are the training seeds of one model. It reads only the cells
//! applicable to every member and to the baseline (common cells); per cell
//! the candidate value is the mean over the members, so the pooled delta is
//! the mean of the members' deltas on the common cells. The per-family
//! differences, candidate minus baseline, are what the bootstrap resamples
//! and what the sign-flip null flips.

use super::emitlog::check_arm_id;
use super::rundir::contract;
use super::trg::{mean, CellScore, CellState, Families};
use dope_arena::ArenaResult;
use std::collections::{BTreeMap, BTreeSet};

/// One comparison: a candidate arm, or the seeds of a seed group.
#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    pub id: String,
    pub seed_group: bool,
    pub members: Vec<String>,
}

pub fn parse_specs(candidates: &[String], groups: &[String]) -> ArenaResult<Vec<Spec>> {
    let mut specs: Vec<Spec> = Vec::new();
    for candidate in candidates {
        check_arm_id(candidate)?;
        specs.push(Spec {
            id: candidate.clone(),
            seed_group: false,
            members: vec![candidate.clone()],
        });
    }
    for group in groups {
        let Some((name, members)) = group.split_once('=') else {
            return contract("--seed-group must be NAME=ARM,ARM,...");
        };
        check_arm_id(name)?;
        let members: Vec<String> = members.split(',').map(str::to_string).collect();
        for member in &members {
            check_arm_id(member)?;
        }
        if members.iter().collect::<BTreeSet<_>>().len() != members.len() {
            return contract("a --seed-group names an arm twice");
        }
        specs.push(Spec {
            id: name.to_string(),
            seed_group: true,
            members,
        });
    }
    if specs.is_empty() {
        return contract("compare needs --candidate or --seed-group");
    }
    if specs.iter().map(|s| &s.id).collect::<BTreeSet<_>>().len() != specs.len() {
        return contract("comparison ids repeat");
    }
    Ok(specs)
}

/// Per-family paired differences (candidate minus baseline) the rule reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Diffs {
    pub trg: Families,
    pub mlp: Families,
    /// `None` when a common cell lacks the realism flag in some arm.
    pub realism: Option<Families>,
    pub privacy: Option<Families>,
}

impl Diffs {
    pub fn family_ids(&self) -> BTreeSet<&str> {
        let mut ids = self.trg.ids();
        ids.extend(self.mlp.ids());
        for families in self.realism.iter().chain(&self.privacy) {
            ids.extend(families.ids());
        }
        ids
    }

    pub fn flipped(&self, sign: &dyn Fn(&str) -> f64) -> Diffs {
        Diffs {
            trg: self.trg.flipped(sign),
            mlp: self.mlp.flipped(sign),
            realism: self.realism.as_ref().map(|f| f.flipped(sign)),
            privacy: self.privacy.as_ref().map(|f| f.flipped(sign)),
        }
    }
}

pub type States = BTreeMap<String, CellState>;

/// One comparison paired on its common cells.
#[derive(Clone, Debug, PartialEq)]
pub struct Paired {
    /// Cells holding a TRG value in every arm.
    pub common: BTreeSet<String>,
    pub mlp_cells: usize,
    pub diffs: Diffs,
    pub candidate: Families,
    pub baseline: Families,
    /// Each member's own delta on the common cells.
    pub member_deltas: Vec<f64>,
}

/// Pair `members` (pooled by their mean) with `baseline` cell by cell.
pub fn pair(members: &[&States], baseline: &States) -> Paired {
    let (mut diff, mut cand, mut base, mut mlp) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut per_member: Vec<Vec<(usize, &str, f64)>> = vec![Vec::new(); members.len()];
    let (mut realism, mut privacy) = (Some(Vec::new()), Some(Vec::new()));
    let mut common = BTreeSet::new();
    for (cell, state) in baseline {
        let Some(b) = state.score() else {
            continue;
        };
        let scores: Option<Vec<&CellScore>> = members
            .iter()
            .map(|states| states.get(cell).and_then(CellState::score))
            .collect();
        let Some(scores) = scores else {
            continue;
        };
        let at = |value: f64| (b.task, b.family_id.as_str(), value);
        let all = |pick: fn(&CellScore) -> Option<f64>| -> Option<f64> {
            mean(
                scores
                    .iter()
                    .map(|c| pick(c))
                    .collect::<Option<Vec<f64>>>()?,
            )
        };
        if let (Some(b_trg), Some(c_trg)) = (b.trg, all(|c| c.trg)) {
            common.insert(cell.clone());
            diff.push(at(c_trg - b_trg));
            cand.push(at(c_trg));
            base.push(at(b_trg));
            for (held, score) in per_member.iter_mut().zip(&scores) {
                held.push(at(score.trg.unwrap_or(b_trg) - b_trg));
            }
            let shares = [
                (b.realism, all(|c| c.realism)),
                (b.privacy, all(|c| c.privacy)),
            ];
            for (list, (b_share, c_share)) in [&mut realism, &mut privacy].into_iter().zip(shares) {
                match (list.as_mut(), b_share.zip(c_share)) {
                    (Some(list), Some((b_share, c_share))) => list.push(at(c_share - b_share)),
                    _ => *list = None,
                }
            }
        }
        if let (Some(b_mlp), Some(c_mlp)) = (b.mlp, all(|c| c.mlp)) {
            mlp.push(at(c_mlp - b_mlp));
        }
    }
    Paired {
        common,
        mlp_cells: mlp.len(),
        diffs: Diffs {
            trg: Families::from_cells(diff),
            mlp: Families::from_cells(mlp),
            realism: realism.map(Families::from_cells),
            privacy: privacy.map(Families::from_cells),
        },
        candidate: Families::from_cells(cand),
        baseline: Families::from_cells(base),
        member_deltas: per_member
            .into_iter()
            .map(|cells| Families::from_cells(cells).point().unwrap_or(0.0))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::stat_testing::{arm_of, protocol, with_guard_fields};
    use super::super::trg::cell_states;
    use super::*;

    fn states(arm: &super::super::rundir::Arm) -> States {
        cell_states(&protocol(), arm, 4, &|_| true)
    }

    #[test]
    fn pairing_keeps_common_cells_and_pools_members_by_their_mean() {
        let base = states(&arm_of(
            "b",
            &[("t1", "f1", "binary", 0.5), ("t2", "f2", "binary", 0.5)],
        ));
        let s0 = states(&arm_of(
            "s0",
            &[("t1", "f1", "binary", 0.7), ("t2", "f2", "binary", 0.9)],
        ));
        // s1 lacks t2: t2 is not common to every member.
        let s1 = states(&arm_of("s1", &[("t1", "f1", "binary", 0.9)]));
        let paired = pair(&[&s0, &s1], &base);
        assert_eq!(paired.common.iter().collect::<Vec<_>>(), ["p:t1:full:d0"]);
        assert!((paired.diffs.trg.point().unwrap() - 0.3).abs() < 1e-12);
        assert!((paired.member_deltas[0] - 0.2).abs() < 1e-12);
        assert!((paired.member_deltas[1] - 0.4).abs() < 1e-12);
        assert_eq!(
            (paired.mlp_cells, paired.diffs.realism.is_none()),
            (1, true)
        );
        let flagged = states(&with_guard_fields(arm_of(
            "f",
            &[("t1", "f1", "binary", 0.7)],
        )));
        let paired = pair(
            &[&flagged],
            &states(&with_guard_fields(arm_of(
                "g",
                &[("t1", "f1", "binary", 0.5)],
            ))),
        );
        assert_eq!(paired.diffs.realism.and_then(|f| f.point()), Some(0.0));
        let flipped = pair(&[&s0], &base).diffs.flipped(&|_| -1.0);
        assert!((flipped.trg.point().unwrap() + 0.3).abs() < 1e-12);
    }
}

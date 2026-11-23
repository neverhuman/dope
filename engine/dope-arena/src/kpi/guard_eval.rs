//! Realism and empirical privacy gates of ledger entries (`dope_kpi guards`).
//!
//! Each gate is the one `dvs::v3::evaluate_v3` applies without partial
//! evidence: a complete entry passes realism iff it carries a realism input
//! and `assess_realism` sets `pass_standard`, and passes privacy iff it
//! carries a privacy input and `assess_empirical_privacy` sets `pass`. An
//! absent input, or one dope-core refuses to assess, fails its gate (the line
//! keeps the refusal code, a fixed string). An entry that is not complete
//! passes neither gate and holds no exact-copy counts. The privacy result is
//! empirical, never formal DP or HIPAA de-identification.
//!
//! The entries of one synthetic file (cell, sample seed, multiplier) form a
//! group: an input equal bit for bit to one already assessed in its group
//! reuses that assessment, so each distinct input is assessed once.

use super::score::status_of;
use dope_core::dvs::v3::privacy::{
    assess_empirical_privacy, EmpiricalPrivacyAssessment, EmpiricalPrivacyInput,
};
use dope_core::dvs::v3::realism::{assess_realism, RealismAssessment, RealismInput};
use dope_core::dvs::v3::{FidelityCellInput, FidelityCellRequest};
use serde::Serialize;

/// One line of `guards.jsonl`.
#[derive(Clone, Debug, Serialize)]
pub struct GuardLine {
    pub ledger_id: String,
    /// The scheduled cell of the entry.
    pub cell_id: String,
    /// Status of the entry as `rows.jsonl` writes it (`score::status_of`).
    pub status: &'static str,
    pub realism_pass: bool,
    pub privacy_pass: bool,
    /// Rows of the assessed privacy input: synthetic rows and those within
    /// 1e-12 of a matched A row; matched twin rows `min(|A|, |B|)` and those
    /// within 1e-12 of a matched A row. Absent when no privacy input was assessed.
    pub synthetic_rows: Option<u64>,
    pub exact_copies: Option<u64>,
    pub twin_rows: Option<u64>,
    pub twin_exact_copies: Option<u64>,
    pub realism: Option<RealismAssessment>,
    pub privacy: Option<EmpiricalPrivacyAssessment>,
    /// dope-core's refusal code for an input it could not assess.
    pub realism_refusal: Option<&'static str>,
    pub privacy_refusal: Option<&'static str>,
}

/// The lines of one group and how many distinct inputs it assessed.
#[derive(Debug)]
pub struct GroupResult {
    pub lines: Vec<GuardLine>,
    pub realism_assessments: usize,
    pub privacy_assessments: usize,
}

fn same_values(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn same_rows(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_values(x, y))
}

fn same_option(a: Option<f64>, b: Option<f64>) -> bool {
    a.map(f64::to_bits) == b.map(f64::to_bits)
}

/// Bit-for-bit equality; the destructuring fails to compile if a field is added.
fn same_realism(a: &RealismInput, b: &RealismInput) -> bool {
    let RealismInput {
        c2st_labels,
        c2st_scores,
        marginal_aucs,
        synth_rule_violation_rate,
        real_rule_violation_rate,
        illegal_rows,
    } = a;
    *c2st_labels == b.c2st_labels
        && same_values(c2st_scores, &b.c2st_scores)
        && same_values(marginal_aucs, &b.marginal_aucs)
        && same_option(*synth_rule_violation_rate, b.synth_rule_violation_rate)
        && same_option(*real_rule_violation_rate, b.real_rule_violation_rate)
        && *illegal_rows == b.illegal_rows
}

fn same_privacy(a: &EmpiricalPrivacyInput, b: &EmpiricalPrivacyInput) -> bool {
    let EmpiricalPrivacyInput {
        real_fit_a,
        real_holdout_b,
        synthetic,
        min_emitted_support,
        canary_rows,
        max_distance_pairs,
    } = a;
    *min_emitted_support == b.min_emitted_support
        && *max_distance_pairs == b.max_distance_pairs
        && same_rows(real_fit_a, &b.real_fit_a)
        && same_rows(real_holdout_b, &b.real_holdout_b)
        && same_rows(synthetic, &b.synthetic)
        && same_rows(canary_rows, &b.canary_rows)
}

type Assessed<A> = Result<A, &'static str>;

/// Inputs already assessed in one group, with their outcomes.
struct Memo<'a, I, A> {
    seen: Vec<(&'a I, Assessed<A>)>,
    same: fn(&I, &I) -> bool,
    assess: fn(&I) -> Assessed<A>,
}

impl<'a, I, A: Clone> Memo<'a, I, A> {
    fn new(same: fn(&I, &I) -> bool, assess: fn(&I) -> Assessed<A>) -> Self {
        Memo {
            seen: Vec::new(),
            same,
            assess,
        }
    }

    fn get(&mut self, input: &'a I) -> Assessed<A> {
        if let Some((_, outcome)) = self.seen.iter().find(|(seen, _)| (self.same)(seen, input)) {
            return outcome.clone();
        }
        let outcome = (self.assess)(input);
        self.seen.push((input, outcome.clone()));
        outcome
    }
}

fn realism_of(input: &RealismInput) -> Assessed<RealismAssessment> {
    assess_realism(input).map_err(|e| e.0)
}

fn privacy_of(input: &EmpiricalPrivacyInput) -> Assessed<EmpiricalPrivacyAssessment> {
    assess_empirical_privacy(input).map_err(|e| e.0)
}

/// A line for an entry that passes no gate and has no counts.
fn closed(ledger_id: &str, cell_id: &str, status: &'static str) -> GuardLine {
    GuardLine {
        ledger_id: ledger_id.to_string(),
        cell_id: cell_id.to_string(),
        status,
        realism_pass: false,
        privacy_pass: false,
        synthetic_rows: None,
        exact_copies: None,
        twin_rows: None,
        twin_exact_copies: None,
        realism: None,
        privacy: None,
        realism_refusal: None,
        privacy_refusal: None,
    }
}

/// Lines of the entries of one synthetic file of cell `cell_id`, in order.
pub fn assess_group(cell_id: &str, entries: &[&FidelityCellRequest]) -> GroupResult {
    let mut realism_memo = Memo::new(same_realism, realism_of);
    let mut privacy_memo = Memo::new(same_privacy, privacy_of);
    let mut lines = Vec::with_capacity(entries.len());
    for request in entries {
        let status = status_of(&request.input);
        let mut line = closed(&request.cell_id, cell_id, status);
        if let FidelityCellInput::Complete {
            realism, privacy, ..
        } = &request.input
        {
            match realism.as_deref().map(|input| realism_memo.get(input)) {
                Some(Ok(assessment)) => {
                    line.realism_pass = assessment.pass_standard;
                    line.realism = Some(assessment);
                }
                Some(Err(code)) => line.realism_refusal = Some(code),
                None => {}
            }
            if let Some(input) = privacy.as_deref() {
                match privacy_memo.get(input) {
                    Ok(assessment) => {
                        let twins = input.real_fit_a.len().min(input.real_holdout_b.len());
                        line.privacy_pass = assessment.pass;
                        line.synthetic_rows = Some(input.synthetic.len() as u64);
                        line.exact_copies = Some(assessment.exact_copies as u64);
                        line.twin_rows = Some(twins as u64);
                        line.twin_exact_copies = Some(assessment.exact_copies_twin_b as u64);
                        line.privacy = Some(assessment);
                    }
                    Err(code) => line.privacy_refusal = Some(code),
                }
            }
        }
        lines.push(line);
    }
    GroupResult {
        lines,
        realism_assessments: realism_memo.seen.len(),
        privacy_assessments: privacy_memo.seen.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::guard_fixture::{
        complete, failed, privacy, realism, request, scored, REAL_A, REAL_B,
    };
    use super::*;

    fn one(input: FidelityCellInput) -> GuardLine {
        let entry = request("c:s0:m4:gbdt", "f", "gbdt", input);
        assess_group("c", &[&entry]).lines.remove(0)
    }

    #[test]
    fn a_copy_of_the_twin_passes_and_a_copy_of_a_fails_privacy() {
        let twin = one(scored(&REAL_B));
        assert!(twin.realism_pass && twin.privacy_pass, "{twin:?}");
        let counts = (
            twin.synthetic_rows,
            twin.exact_copies,
            twin.twin_rows,
            twin.twin_exact_copies,
        );
        assert_eq!(counts, (Some(4), Some(0), Some(4), Some(0)));
        let p = twin.privacy.as_ref().unwrap();
        assert_eq!(
            (p.dcr_closer_to_a_share, p.mia_auc, p.mia_tpr_at_1pct_fpr),
            (0.0, 0.0, 0.0)
        );
        let copy = one(scored(&REAL_A));
        assert!(copy.realism_pass && !copy.privacy_pass);
        assert_eq!((copy.status, copy.exact_copies), ("ok", Some(4)));
        let p = copy.privacy.as_ref().unwrap();
        assert_eq!(
            (p.dcr_closer_to_a_share, p.mia_auc, p.exact_copy_excess),
            (1.0, 1.0, 1.0)
        );
        let unreal = one(complete(
            Some(realism(false)),
            Some(privacy(&REAL_A, &REAL_B, &REAL_B)),
        ));
        assert!(!unreal.realism_pass && unreal.privacy_pass);
        assert_eq!(unreal.realism.map(|r| r.c2st_auc), Some(1.0));
    }

    #[test]
    fn the_auditors_of_one_file_share_one_assessment() {
        let ids = [
            "c:s0:m4:gbdt",
            "c:s0:m4:linear-seeded-v1",
            "c:s0:m4:mlp-seeded-v1",
        ];
        let mut entries: Vec<FidelityCellRequest> = ids
            .iter()
            .map(|id| request(id, "f", "a", scored(&REAL_B)))
            .collect();
        let refs: Vec<&FidelityCellRequest> = entries.iter().collect();
        let out = assess_group("c", &refs);
        assert_eq!((out.realism_assessments, out.privacy_assessments), (1, 1));
        assert_eq!(out.lines[2].ledger_id, ids[2]);
        assert!(out.lines.iter().all(|l| l.privacy_pass && l.cell_id == "c"));
        // One synthetic value one bit away is a different input.
        let s = [1.0, 3.0, 5.0, f64::from_bits(7.0_f64.to_bits() + 1)];
        entries[1].input = complete(Some(realism(true)), Some(privacy(&REAL_A, &REAL_B, &s)));
        let refs: Vec<&FidelityCellRequest> = entries.iter().collect();
        let out = assess_group("c", &refs);
        assert_eq!((out.realism_assessments, out.privacy_assessments), (1, 2));
    }

    #[test]
    fn closed_absent_and_refused_inputs_fail_without_counts() {
        let no_marginals = RealismInput {
            marginal_aucs: Vec::new(),
            ..realism(true)
        };
        let over = FidelityCellInput::OverBudget {
            actual_bytes: 3000,
            cap_bytes: 2048,
        };
        let lines = [
            one(failed()),
            one(over),
            one(FidelityCellInput::Unsupported),
            one(complete(None, None)),
            one(complete(
                Some(no_marginals),
                Some(privacy(&REAL_A, &REAL_B, &[])),
            )),
        ];
        let statuses: Vec<&str> = lines.iter().map(|l| l.status).collect();
        assert_eq!(
            statuses,
            ["failed", "over_budget", "unsupported", "ok", "ok"]
        );
        for line in &lines {
            assert!(!line.realism_pass && !line.privacy_pass, "{line:?}");
            assert_eq!((line.exact_copies, line.twin_rows), (None, None));
            assert!(line.realism.is_none() && line.privacy.is_none());
        }
        assert_eq!(lines[3].realism_refusal, None);
        assert_eq!(lines[4].realism_refusal, Some("empty_marginal_aucs"));
        assert_eq!(lines[4].privacy_refusal, Some("empty_privacy_v3_input"));
    }
}

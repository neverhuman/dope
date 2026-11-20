//! Algorithm response panel.
//!
//! The evaluator still owns execution. One panel binds one task. A measured
//! value is admitted only for a non-confirmation fit scope with a disjoint
//! inner-query scope. Unknown statuses stay masked. Search and full records
//! of the same ranked program can be compared without treating that count as
//! a rank correlation.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{DopeError, Result};

pub const SCHEMA: &str = "dope-program-response/v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutcomeStatus {
    Measured,
    Invalid,
    Failed,
    TimedOut,
    Unsupported,
    NotRequested,
    RequestedNotObserved,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fidelity {
    Search,
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Acquisition {
    Ranked,
    Random,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PanelMode {
    FirstShot,
    Revealed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FitRole {
    Train,
    Dev,
    Recall,
    Confirm,
    Test,
    Query,
    Holdout,
}

impl FitRole {
    fn blocks_panel(self) -> bool {
        matches!(
            self,
            Self::Confirm | Self::Test | Self::Query | Self::Holdout
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetricDirection {
    LowerIsBetter,
    HigherIsBetter,
}

#[derive(Clone, Debug)]
pub struct AdmittedFit {
    pub fit_scope_sha256: String,
    pub row_count: u64,
    pub role: FitRole,
}

#[derive(Clone, Debug)]
pub struct OutcomeDraft {
    pub task_context_sha256: String,
    pub program_sha256: String,
    pub evaluator_sha256: String,
    pub fit_scope_sha256: String,
    pub inner_query_scope_sha256: String,
    pub budget_sha256: String,
    pub seed: u64,
    pub fidelity: Fidelity,
    pub acquisition: Acquisition,
    pub status: OutcomeStatus,
    pub direction: MetricDirection,
    pub raw_value: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct OutcomeRecord {
    pub task_context_sha256: String,
    pub program_sha256: String,
    pub evaluator_sha256: String,
    pub fit_scope_sha256: String,
    pub inner_query_scope_sha256: String,
    pub budget_sha256: String,
    pub seed: u64,
    pub fidelity: Fidelity,
    pub acquisition: Acquisition,
    pub status: OutcomeStatus,
    pub direction: MetricDirection,
    pub raw_value: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Panel {
    pub mode: PanelMode,
    pub task_context_sha256: String,
    pub direction: MetricDirection,
    pub records: Vec<OutcomeRecord>,
}

/// Empirical pool summary for one task. Regret is `u(oracle) - u(selected)`
/// on full-fit values. It is not regret against an unmeasured program.
#[derive(Clone, Debug, PartialEq)]
pub struct FidelityDiscrepancy {
    pub task_context_sha256: String,
    pub direction: MetricDirection,
    pub paired_program_count: u64,
    pub unpaired_program_count: u64,
    pub pairwise_comparisons: u64,
    pub pairwise_order_disagreements: u64,
    pub top_choice_disagrees: bool,
    pub oracle_program_sha256: String,
    pub selected_program_sha256: String,
    pub oracle_count: u64,
    pub selected_count: u64,
    pub oracle_full_value: f64,
    pub selected_full_value: f64,
    pub regret: f64,
    pub normalized_regret: f64,
}

pub fn build_panel(
    mode: PanelMode,
    admitted: &[AdmittedFit],
    drafts: &[OutcomeDraft],
) -> Result<Panel> {
    if drafts.is_empty() || admitted.is_empty() {
        return Err(DopeError::Data("response_panel_shape".into()));
    }
    let mut scopes = BTreeMap::<String, u64>::new();
    for fit in admitted {
        digest_text(&fit.fit_scope_sha256)?;
        if fit.role.blocks_panel() {
            return Err(DopeError::Data("response_panel_role".into()));
        }
        if fit.row_count == 0 {
            return Err(DopeError::Data("response_panel_scope".into()));
        }
        if let Some(prior) = scopes.insert(fit.fit_scope_sha256.clone(), fit.row_count) {
            if prior != fit.row_count {
                return Err(DopeError::Data("response_panel_scope".into()));
            }
        }
    }
    let mut records = Vec::with_capacity(drafts.len());
    for draft in drafts {
        for digest in [
            &draft.task_context_sha256,
            &draft.program_sha256,
            &draft.evaluator_sha256,
            &draft.fit_scope_sha256,
            &draft.inner_query_scope_sha256,
            &draft.budget_sha256,
        ] {
            digest_text(digest)?;
        }
        if !scopes.contains_key(&draft.fit_scope_sha256) {
            return Err(DopeError::Data("response_panel_untraced".into()));
        }
        if draft.inner_query_scope_sha256 == draft.fit_scope_sha256 {
            return Err(DopeError::Data("response_panel_query".into()));
        }
        match (mode, draft.status, draft.raw_value) {
            (PanelMode::FirstShot, _, Some(_))
            | (PanelMode::FirstShot, OutcomeStatus::Measured, _) => {
                return Err(DopeError::Data("response_panel_leak".into()));
            }
            (PanelMode::Revealed, OutcomeStatus::Measured, Some(value)) if value.is_finite() => {}
            (PanelMode::Revealed, OutcomeStatus::Measured, _) => {
                return Err(DopeError::Data("response_panel_value".into()));
            }
            (_, status, Some(_)) if !matches!(status, OutcomeStatus::Measured) => {
                return Err(DopeError::Data("response_panel_mask".into()));
            }
            _ => {}
        }
        records.push(record_from(draft));
    }
    let task = records[0].task_context_sha256.clone();
    let direction = records[0].direction;
    if records
        .iter()
        .any(|record| record.task_context_sha256 != task)
    {
        return Err(DopeError::Data("response_panel_task".into()));
    }
    if records.iter().any(|record| record.direction != direction) {
        return Err(DopeError::Data("response_panel_metric".into()));
    }
    Ok(Panel {
        mode,
        task_context_sha256: task,
        direction,
        records,
    })
}

pub fn fidelity_discrepancy(panel: &Panel) -> Result<FidelityDiscrepancy> {
    if panel.mode != PanelMode::Revealed {
        return Err(DopeError::Data("response_panel_leak".into()));
    }
    let mut search = BTreeMap::<&str, f64>::new();
    let mut full = BTreeMap::<&str, f64>::new();
    for record in &panel.records {
        if record.acquisition != Acquisition::Ranked || record.status != OutcomeStatus::Measured {
            continue;
        }
        let Some(value) = record.raw_value else {
            return Err(DopeError::Data("response_panel_value".into()));
        };
        let slot = match record.fidelity {
            Fidelity::Search => &mut search,
            Fidelity::Full => &mut full,
        };
        if slot.insert(record.program_sha256.as_str(), value).is_some() {
            return Err(DopeError::Data("response_panel_pair".into()));
        }
    }
    let mut names = BTreeSet::new();
    names.extend(search.keys().copied());
    names.extend(full.keys().copied());
    let mut paired = Vec::new();
    let mut unpaired_program_count = 0u64;
    for program in names {
        match (search.get(program), full.get(program)) {
            (Some(search_value), Some(full_value)) => {
                paired.push((program, *search_value, *full_value));
            }
            _ => unpaired_program_count += 1,
        }
    }
    if paired.is_empty() {
        return Err(DopeError::Data("response_panel_pair".into()));
    }
    let mut pairwise_comparisons = 0u64;
    let mut pairwise_order_disagreements = 0u64;
    for left in 0..paired.len() {
        for right in (left + 1)..paired.len() {
            pairwise_comparisons += 1;
            let search_order = utility(panel.direction, paired[left].1)
                .total_cmp(&utility(panel.direction, paired[right].1));
            let full_order = utility(panel.direction, paired[left].2)
                .total_cmp(&utility(panel.direction, paired[right].2));
            if search_order != full_order {
                pairwise_order_disagreements += 1;
            }
        }
    }
    let search_winners = winner_set(panel.direction, &paired, |row| row.1);
    let full_winners = winner_set(panel.direction, &paired, |row| row.2);
    let selected = search_winners
        .iter()
        .copied()
        .min()
        .ok_or_else(|| DopeError::Data("response_panel_pair".into()))?;
    let oracle = full_winners
        .iter()
        .copied()
        .min()
        .ok_or_else(|| DopeError::Data("response_panel_pair".into()))?;
    let selected_full = full_value(&paired, selected)?;
    let oracle_full = full_value(&paired, oracle)?;
    let regret = utility(panel.direction, oracle_full) - utility(panel.direction, selected_full);
    let mut lo = utility(panel.direction, paired[0].2);
    let mut hi = lo;
    for row in &paired {
        let score = utility(panel.direction, row.2);
        if score.total_cmp(&lo).is_lt() {
            lo = score;
        }
        if score.total_cmp(&hi).is_gt() {
            hi = score;
        }
    }
    let span = hi - lo;
    let normalized_regret = if span.total_cmp(&0.0).is_eq() {
        0.0
    } else {
        regret / span
    };
    Ok(FidelityDiscrepancy {
        task_context_sha256: panel.task_context_sha256.clone(),
        direction: panel.direction,
        paired_program_count: paired.len() as u64,
        unpaired_program_count,
        pairwise_comparisons,
        pairwise_order_disagreements,
        top_choice_disagrees: search_winners != full_winners,
        oracle_program_sha256: oracle.to_string(),
        selected_program_sha256: selected.to_string(),
        oracle_count: full_winners.len() as u64,
        selected_count: search_winners.len() as u64,
        oracle_full_value: oracle_full,
        selected_full_value: selected_full,
        regret,
        normalized_regret,
    })
}

fn record_from(draft: &OutcomeDraft) -> OutcomeRecord {
    OutcomeRecord {
        task_context_sha256: draft.task_context_sha256.clone(),
        program_sha256: draft.program_sha256.clone(),
        evaluator_sha256: draft.evaluator_sha256.clone(),
        fit_scope_sha256: draft.fit_scope_sha256.clone(),
        inner_query_scope_sha256: draft.inner_query_scope_sha256.clone(),
        budget_sha256: draft.budget_sha256.clone(),
        seed: draft.seed,
        fidelity: draft.fidelity,
        acquisition: draft.acquisition,
        status: draft.status,
        direction: draft.direction,
        raw_value: draft.raw_value,
    }
}

fn utility(direction: MetricDirection, value: f64) -> f64 {
    match direction {
        MetricDirection::HigherIsBetter => value,
        MetricDirection::LowerIsBetter => -value,
    }
}

fn winner_set<'a>(
    direction: MetricDirection,
    rows: &[(&'a str, f64, f64)],
    value: impl Fn(&(&'a str, f64, f64)) -> f64,
) -> BTreeSet<&'a str> {
    let mut best = utility(direction, value(&rows[0]));
    for row in rows.iter().skip(1) {
        let score = utility(direction, value(row));
        if score.total_cmp(&best).is_gt() {
            best = score;
        }
    }
    rows.iter()
        .filter(|row| utility(direction, value(row)).total_cmp(&best).is_eq())
        .map(|row| row.0)
        .collect()
}

fn full_value(rows: &[(&str, f64, f64)], program: &str) -> Result<f64> {
    rows.iter()
        .find(|row| row.0 == program)
        .map(|row| row.2)
        .ok_or_else(|| DopeError::Data("response_panel_pair".into()))
}

fn digest_text(value: &str) -> Result<()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err(DopeError::Data("response_panel_digest".into()))
    }
}

#[cfg(test)]
#[path = "response_panel_tests.rs"]
mod tests;

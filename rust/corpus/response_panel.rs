//! Algorithm response panel.
//!
//! The evaluator still owns execution. A measured value is admitted only when
//! its fit scope names admitted rows. Unmeasured statuses stay masked, a
//! search/full pair must share one task, and a first-shot panel carries no score.

use std::collections::BTreeMap;

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

#[derive(Clone, Debug)]
pub struct AdmittedFit {
    pub fit_scope_sha256: String,
    pub row_count: u64,
}

#[derive(Clone, Debug)]
pub struct OutcomeDraft {
    pub task_context_sha256: String,
    pub program_sha256: String,
    pub evaluator_sha256: String,
    pub fit_scope_sha256: String,
    pub budget_sha256: String,
    pub seed: u64,
    pub fidelity: Fidelity,
    pub acquisition: Acquisition,
    pub status: OutcomeStatus,
    pub raw_value: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct OutcomeRecord {
    pub task_context_sha256: String,
    pub program_sha256: String,
    pub evaluator_sha256: String,
    pub fit_scope_sha256: String,
    pub budget_sha256: String,
    pub seed: u64,
    pub fidelity: Fidelity,
    pub acquisition: Acquisition,
    pub status: OutcomeStatus,
    pub raw_value: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Panel {
    pub mode: PanelMode,
    pub records: Vec<OutcomeRecord>,
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
            &draft.budget_sha256,
        ] {
            digest_text(digest)?;
        }
        if !scopes.contains_key(&draft.fit_scope_sha256) {
            return Err(DopeError::Data("response_panel_untraced".into()));
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
        records.push(OutcomeRecord {
            task_context_sha256: draft.task_context_sha256.clone(),
            program_sha256: draft.program_sha256.clone(),
            evaluator_sha256: draft.evaluator_sha256.clone(),
            fit_scope_sha256: draft.fit_scope_sha256.clone(),
            budget_sha256: draft.budget_sha256.clone(),
            seed: draft.seed,
            fidelity: draft.fidelity,
            acquisition: draft.acquisition,
            status: draft.status,
            raw_value: draft.raw_value,
        });
    }
    let mut tasks_by_program: BTreeMap<&str, &str> = BTreeMap::new();
    for record in &records {
        if let Some(prior) = tasks_by_program.insert(
            record.program_sha256.as_str(),
            record.task_context_sha256.as_str(),
        ) {
            if prior != record.task_context_sha256 {
                return Err(DopeError::Data("response_panel_task".into()));
            }
        }
    }
    Ok(Panel { mode, records })
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
mod tests {
    use super::*;

    fn hex(fill: u8) -> String {
        format!("{fill:02x}").repeat(32)
    }

    fn admitted() -> Vec<AdmittedFit> {
        vec![AdmittedFit {
            fit_scope_sha256: hex(1),
            row_count: 32,
        }]
    }

    fn draft(status: OutcomeStatus, value: Option<f64>) -> OutcomeDraft {
        OutcomeDraft {
            task_context_sha256: hex(2),
            program_sha256: hex(3),
            evaluator_sha256: hex(4),
            fit_scope_sha256: hex(1),
            budget_sha256: hex(5),
            seed: 7,
            fidelity: Fidelity::Full,
            acquisition: Acquisition::Ranked,
            status,
            raw_value: value,
        }
    }

    #[test]
    fn measured_value_must_name_admitted_rows() {
        let panel = build_panel(
            PanelMode::Revealed,
            &admitted(),
            &[draft(OutcomeStatus::Measured, Some(0.25))],
        )
        .unwrap();
        assert_eq!(panel.records.len(), 1);
        assert_eq!(panel.records[0].raw_value, Some(0.25));
        let mut missing = draft(OutcomeStatus::Measured, Some(12_345.6));
        missing.fit_scope_sha256 = hex(9);
        let error = build_panel(PanelMode::Revealed, &admitted(), &[missing]).unwrap_err();
        let rendered = error.to_string();
        assert!(rendered.contains("response_panel_untraced"));
        assert!(!rendered.contains("12345"));
    }

    #[test]
    fn unknown_outcomes_are_masked() {
        let error = build_panel(
            PanelMode::Revealed,
            &admitted(),
            &[draft(OutcomeStatus::RequestedNotObserved, Some(0.0))],
        )
        .unwrap_err();
        assert!(error.to_string().contains("response_panel_mask"));
        let panel = build_panel(
            PanelMode::Revealed,
            &admitted(),
            &[draft(OutcomeStatus::RequestedNotObserved, None)],
        )
        .unwrap();
        assert_eq!(panel.records[0].raw_value, None);
        assert_eq!(panel.records[0].status, OutcomeStatus::RequestedNotObserved);
    }

    #[test]
    fn fidelity_pair_uses_one_task_and_keeps_nonwinners() {
        let mut search = draft(OutcomeStatus::Measured, Some(0.5));
        search.fidelity = Fidelity::Search;
        search.acquisition = Acquisition::Random;
        let mut full = draft(OutcomeStatus::Failed, None);
        full.fidelity = Fidelity::Full;
        let panel = build_panel(PanelMode::Revealed, &admitted(), &[search, full]).unwrap();
        assert_eq!(panel.records.len(), 2);
        assert_eq!(panel.records[0].acquisition, Acquisition::Random);
        assert_eq!(panel.records[1].status, OutcomeStatus::Failed);
        let mut other_task = draft(OutcomeStatus::Measured, Some(0.5));
        other_task.fidelity = Fidelity::Search;
        other_task.task_context_sha256 = hex(8);
        let mut full_ok = draft(OutcomeStatus::Measured, Some(0.25));
        full_ok.fidelity = Fidelity::Full;
        let error =
            build_panel(PanelMode::Revealed, &admitted(), &[other_task, full_ok]).unwrap_err();
        assert!(error.to_string().contains("response_panel_task"));
    }

    #[test]
    fn first_shot_panel_has_no_score() {
        let error = build_panel(
            PanelMode::FirstShot,
            &admitted(),
            &[draft(OutcomeStatus::Measured, Some(0.25))],
        )
        .unwrap_err();
        assert!(error.to_string().contains("response_panel_leak"));
        let panel = build_panel(
            PanelMode::FirstShot,
            &admitted(),
            &[draft(OutcomeStatus::NotRequested, None)],
        )
        .unwrap();
        assert_eq!(panel.mode, PanelMode::FirstShot);
        assert_eq!(panel.records[0].raw_value, None);
        assert_ne!(panel.records[0].status, OutcomeStatus::Measured);
    }
}

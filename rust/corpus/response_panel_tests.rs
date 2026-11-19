use super::*;

fn hex(fill: u8) -> String {
    format!("{fill:02x}").repeat(32)
}

fn admitted() -> Vec<AdmittedFit> {
    vec![AdmittedFit {
        fit_scope_sha256: hex(1),
        row_count: 32,
        role: FitRole::Train,
    }]
}

fn draft(status: OutcomeStatus, value: Option<f64>) -> OutcomeDraft {
    OutcomeDraft {
        task_context_sha256: hex(2),
        program_sha256: hex(3),
        evaluator_sha256: hex(4),
        fit_scope_sha256: hex(1),
        inner_query_scope_sha256: hex(6),
        budget_sha256: hex(5),
        seed: 7,
        fidelity: Fidelity::Full,
        acquisition: Acquisition::Ranked,
        status,
        direction: MetricDirection::HigherIsBetter,
        raw_value: value,
    }
}

fn program(fill: u8, search: f64, full: f64) -> [OutcomeDraft; 2] {
    let mut search_row = draft(OutcomeStatus::Measured, Some(search));
    search_row.program_sha256 = hex(fill);
    search_row.fidelity = Fidelity::Search;
    let mut full_row = draft(OutcomeStatus::Measured, Some(full));
    full_row.program_sha256 = hex(fill);
    full_row.fidelity = Fidelity::Full;
    [search_row, full_row]
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
    assert_eq!(panel.task_context_sha256, hex(2));
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
fn one_panel_binds_one_task_and_keeps_nonwinners() {
    let mut search = draft(OutcomeStatus::Measured, Some(0.5));
    search.fidelity = Fidelity::Search;
    search.acquisition = Acquisition::Random;
    let mut full = draft(OutcomeStatus::Failed, None);
    full.fidelity = Fidelity::Full;
    let panel = build_panel(PanelMode::Revealed, &admitted(), &[search, full]).unwrap();
    assert_eq!(panel.records.len(), 2);
    assert_eq!(panel.records[0].acquisition, Acquisition::Random);
    assert_eq!(panel.records[1].status, OutcomeStatus::Failed);
    let mut other_program = draft(OutcomeStatus::Measured, Some(0.5));
    other_program.program_sha256 = hex(8);
    other_program.task_context_sha256 = hex(9);
    let same_task = draft(OutcomeStatus::Measured, Some(0.25));
    let error = build_panel(
        PanelMode::Revealed,
        &admitted(),
        &[other_program, same_task],
    )
    .unwrap_err();
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
    let audit = fidelity_discrepancy(&panel).unwrap_err();
    assert!(audit.to_string().contains("response_panel_leak"));
}

#[test]
fn confirmation_role_and_fit_query_overlap_are_refused() {
    let mut confirm = admitted();
    confirm[0].role = FitRole::Confirm;
    let error = build_panel(
        PanelMode::Revealed,
        &confirm,
        &[draft(OutcomeStatus::Measured, Some(0.25))],
    )
    .unwrap_err();
    assert!(error.to_string().contains("response_panel_role"));
    let mut overlap = draft(OutcomeStatus::Measured, Some(9_999.0));
    overlap.inner_query_scope_sha256 = overlap.fit_scope_sha256.clone();
    let error = build_panel(PanelMode::Revealed, &admitted(), &[overlap]).unwrap_err();
    let rendered = error.to_string();
    assert!(rendered.contains("response_panel_query"));
    assert!(!rendered.contains("9999"));
}

#[test]
fn search_full_discrepancy_separates_oracle_and_choice() {
    let mut rows = Vec::new();
    rows.extend(program(0x21, 4.0, 1.0));
    rows.extend(program(0x22, 1.0, 4.0));
    let mut random = program(0x23, 8.0, 8.0);
    random[0].acquisition = Acquisition::Random;
    random[1].acquisition = Acquisition::Random;
    rows.extend(random);
    let panel = build_panel(PanelMode::Revealed, &admitted(), &rows).unwrap();
    let audit = fidelity_discrepancy(&panel).unwrap();
    assert_eq!(audit.paired_program_count, 2);
    assert_eq!(audit.pairwise_comparisons, 1);
    assert_eq!(audit.pairwise_order_disagreements, 1);
    assert!(audit.top_choice_disagrees);
    assert_eq!(audit.selected_program_sha256, hex(0x21));
    assert_eq!(audit.oracle_program_sha256, hex(0x22));
    assert_eq!(audit.selected_full_value, 1.0);
    assert_eq!(audit.oracle_full_value, 4.0);
    assert_eq!(audit.regret, 3.0);
    assert_eq!(audit.normalized_regret, 1.0);
    assert_ne!(audit.oracle_program_sha256, hex(0x23));
    assert_ne!(audit.selected_program_sha256, hex(0x23));
}

#[test]
fn lower_is_better_regret_uses_full_values() {
    let mut rows = Vec::new();
    rows.extend(program(0x31, 1.0, 4.0));
    rows.extend(program(0x32, 4.0, 1.0));
    for row in &mut rows {
        row.direction = MetricDirection::LowerIsBetter;
    }
    let panel = build_panel(PanelMode::Revealed, &admitted(), &rows).unwrap();
    let audit = fidelity_discrepancy(&panel).unwrap();
    assert_eq!(audit.direction, MetricDirection::LowerIsBetter);
    assert!(audit.top_choice_disagrees);
    assert_eq!(audit.selected_program_sha256, hex(0x31));
    assert_eq!(audit.oracle_program_sha256, hex(0x32));
    assert_eq!(audit.regret, 3.0);
    assert_eq!(audit.pairwise_order_disagreements, 1);
}

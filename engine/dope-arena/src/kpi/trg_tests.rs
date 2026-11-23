//! Unit tests of the TRG statistic (`trg.rs`).

use super::emitlog::CLASS_CONTROL;
use super::rundir::{Arm, KpiRow, STATUS_UNSUPPORTED};
use super::stat_testing::{arm, protocol, row, LINEAR, MLP};
use super::trg::*;

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
    let cells = cell_states(&protocol(), &arm(rows.clone()), 4, &all);
    let t1 = cells["p:t1:full:d0"].score().unwrap();
    assert_eq!(
        (t1.trg, t1.mlp, t1.failed),
        (Some(-0.25), Some(-2.0), false)
    );
    let t2 = cells["p:t2:full:d0"].score().unwrap();
    assert_eq!((t2.trg, t2.mlp, t2.failed), (Some(-1.5), None, true));
    assert_eq!((t1.realism, t1.privacy), (None, None));
    // The scored -1.5 is worse than the failure: withholding the file would pay.
    // `failed_entry: unit_floor` scores the failure at clip_low - R_B = -2 instead.
    let floor = Protocol {
        failed_at_floor: true,
        ..protocol()
    };
    let cells = cell_states(&floor, &arm(rows), 4, &all);
    assert_eq!(cells["p:t2:full:d0"].score().unwrap().trg, Some(-2.0));
    assert_eq!(cells["p:t1:full:d0"].score().unwrap().trg, Some(-0.25));
}

#[test]
fn a_byte_cap_counts_a_longer_scored_packet_as_over_budget() {
    let mut long = row("t1", "f1", "regression", LINEAR, 1.0);
    long.packet_bytes = 3000;
    let rows = vec![long, row("t1", "f1", "regression", "gbdt", 1.0)];
    let cells = cell_states(&protocol(), &arm(rows.clone()), 4, &all);
    assert_eq!(cells["p:t1:full:d0"].score().unwrap().trg, Some(0.0));
    let capped = Protocol {
        budget_cap: Some(2048),
        ..protocol()
    };
    let cells = cell_states(&capped, &arm(rows), 4, &all);
    let t1 = cells["p:t1:full:d0"].score().unwrap();
    assert_eq!((t1.trg, t1.failed), (Some(-1.0), true));
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

fn unsupported(table: &str, auditor: &str) -> KpiRow {
    let mut twin = row(table, "f1", "regression", auditor, 0.0);
    twin.status = STATUS_UNSUPPORTED.into();
    twin.artifact_class = CLASS_CONTROL.into();
    twin.r_s = None;
    twin
}

#[test]
fn only_a_native_control_has_cells_that_cannot_exist() {
    let twin = unsupported("t1", LINEAR);
    let control = Arm::sorted("twin_c", String::new(), vec![twin.clone()]);
    let states = cell_states(&protocol(), &control, 4, &all);
    let na = CellState::NotApplicable { in_statistic: true };
    assert_eq!(states["p:t1:full:d0"], na);
    assert!(states["p:t1:full:d0"].in_statistic());
    // Any other arm has failed the cell, whatever class its log claims.
    for (id, class) in [("gen", CLASS_CONTROL), ("twin_c", "source_free")] {
        let mut entry = twin.clone();
        entry.artifact_class = class.into();
        let other = Arm::sorted(id, String::new(), vec![entry]);
        let states = cell_states(&protocol(), &other, 4, &all);
        assert_eq!(
            states["p:t1:full:d0"].score().unwrap().trg,
            Some(-1.0),
            "{id}"
        );
    }
}

#[test]
fn a_cell_informative_only_for_the_held_out_auditor_is_not_a_statistic_cell() {
    let mut quiet = unsupported("t1", LINEAR);
    quiet.informative = false;
    let rows = vec![quiet, unsupported("t1", MLP)];
    let control = Arm::sorted("twin_c", String::new(), rows.clone());
    let states = cell_states(&protocol(), &control, 4, &all);
    let state = &states["p:t1:full:d0"];
    assert_eq!(
        *state,
        CellState::NotApplicable {
            in_statistic: false
        }
    );
    assert!(!state.in_statistic());
    let generator = Arm::sorted("gen", String::new(), rows);
    let states = cell_states(&protocol(), &generator, 4, &all);
    let score = states["p:t1:full:d0"].score().unwrap();
    assert_eq!((score.trg, score.mlp), (None, Some(-1.0)));
    assert!(!states["p:t1:full:d0"].in_statistic());
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

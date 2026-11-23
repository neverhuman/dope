//! Scenario tests of `primary`, `compare` and `null` on synthetic ledgers
//! with hand-computed values. Every cell of `arm_of` has `R_B = 1`, so a
//! scored cell with retention `r` has TRG `r - 1`.

use super::compare::{compare_all, ComparisonResult};
use super::null::sign_flip_null;
use super::pairing::parse_specs;
use super::primary::summarize;
use super::receipt::{receipt_bytes, Context};
use super::rule::{Check, Outcome};
use super::stat_testing::{arm_of, context, edit, with_guard_fields, MLP};
use super::trg::CellState;

const DRAWS: f64 = 10_000.0;

fn close(got: f64, want: f64) {
    assert!((got - want).abs() < 1e-12, "{got} != {want}");
}

/// One family per task at retention `r` (TRG `r - 1`).
fn three(id: &str, r: [f64; 3]) -> super::rundir::Arm {
    arm_of(
        id,
        &[
            ("t1", "f1", "regression", r[0]),
            ("t2", "f2", "binary", r[1]),
            ("t3", "f3", "multiclass", r[2]),
        ],
    )
}

fn compare(
    ctx: &Context,
    candidates: &[&str],
    groups: &[&str],
    baseline: &str,
    noise: f64,
) -> Vec<ComparisonResult> {
    let candidates: Vec<String> = candidates.iter().map(|s| s.to_string()).collect();
    let groups: Vec<String> = groups.iter().map(|s| s.to_string()).collect();
    compare_all(
        ctx,
        &parse_specs(&candidates, &groups).unwrap(),
        baseline,
        noise,
    )
    .unwrap()
}

fn not_passing(c: &ComparisonResult) -> Vec<&str> {
    c.decision
        .guards
        .iter()
        .filter(|g| g.check != Check::Pass)
        .map(|g| g.name.as_str())
        .collect()
}

#[test]
fn single_family_has_a_degenerate_interval_and_needs_every_guard_field() {
    let base = arm_of("base", &[("t1", "f1", "regression", 0.8)]);
    let cand = arm_of("cand", &[("t1", "f1", "regression", 0.9)]);
    let ctx = context("stat-single", &[base.clone(), cand.clone()]);
    let c = &compare(&ctx, &["cand"], &[], "base", 0.01)[0];
    close(c.delta, 0.1);
    close(c.ci_low, 0.1);
    close(c.ci_high, 0.1);
    assert_eq!(
        (c.p_one_sided, c.p_holm),
        (1.0 / (DRAWS + 1.0), 1.0 / (DRAWS + 1.0))
    );
    assert_eq!((c.cells, c.families, c.mlp_cells), (1, 1, 1));
    close(c.candidate_trg.unwrap(), -0.1);
    close(c.baseline_trg.unwrap(), -0.2);
    assert_eq!(c.decision.outcome, Outcome::Hold);
    assert_eq!(
        not_passing(c),
        [
            "realism_coverage_delta",
            "privacy_coverage_delta",
            "exact_copy_excess"
        ]
    );
    let ctx = context(
        "stat-single-fields",
        &[with_guard_fields(base), with_guard_fields(cand)],
    );
    let c = &compare(&ctx, &["cand"], &[], "base", 0.01)[0];
    assert_eq!(c.decision.outcome, Outcome::Promote, "{:?}", c.decision);
}

#[test]
fn unequal_family_sizes_average_families_before_tasks() {
    let a = arm_of(
        "a",
        &[
            ("t1", "f1", "regression", 1.0),
            ("t2", "f1", "regression", 0.5),
            ("t3", "f1", "regression", 0.0),
            ("t4", "f2", "regression", 0.8),
            ("t5", "f3", "binary", 0.9),
            ("t6", "f3", "binary", 0.7),
        ],
    );
    let ctx = context("stat-unequal", &[a]);
    let s = summarize(&ctx, ctx.arm("a"));
    // f1 = mean(0, -0.5, -1) = -0.5, f2 = -0.2 -> regression -0.35; f3 = -0.2.
    // TRG = mean(-0.35, -0.2) = -0.275 (the plain cell mean would be -0.35).
    close(s.primary.trg.unwrap(), -0.275);
    let tasks: Vec<(&str, usize)> = s
        .primary
        .tasks
        .iter()
        .map(|t| (t.task.as_str(), t.families))
        .collect();
    assert_eq!(tasks, [("regression", 2), ("binary", 1)]);
    close(s.primary.tasks[0].value, -0.35);
    assert_eq!(
        (s.primary.cells, s.primary.families, s.primary.failed_cells),
        (6, 3, 0)
    );
    // The schedule holds multiplier 4 only: the matched statistic is empty.
    assert_eq!((s.matched.trg, s.matched.cells), (None, 0));
    assert_eq!(
        (s.bytes.files, s.bytes.median, s.bytes.share_within_budget),
        (6, Some(1000.0), Some(1.0))
    );
    assert_eq!(s.median_retention[0].median, Some(0.75));
    assert_eq!(
        (s.c2st_median, s.c2st_files, s.artifact_class.as_str()),
        (Some(0.6), 6, "source_free")
    );
}

#[test]
fn an_arm_that_fails_everything_scores_minus_one_and_is_rejected() {
    let good = three("good", [0.9, 0.9, 0.9]);
    let mut failed = three("failed", [0.9, 0.9, 0.9]);
    for row in &mut failed.rows {
        row.status = "failed".into();
        row.r_s = None;
    }
    let ctx = context("stat-all-failed", &[good, failed]);
    let s = summarize(&ctx, ctx.arm("failed"));
    assert_eq!(
        (s.primary.trg, s.primary.failed_cells, s.primary.cells),
        (Some(-1.0), 3, 3)
    );
    assert_eq!((s.c2st_files, s.bytes.files), (0, 3));
    let c = &compare(&ctx, &["failed"], &[], "good", 0.01)[0];
    close(c.delta, -0.9);
    close(c.ci_high, -0.9);
    assert_eq!(c.p_one_sided, 1.0);
    assert_eq!(c.decision.outcome, Outcome::Reject);
}

#[test]
fn inapplicable_cells_leave_the_arm_and_every_comparison() {
    let cells = [
        ("t1", "f1", "regression", 0.9),
        ("t2", "f2", "regression", 0.7),
        ("t3", "f3", "binary", 0.8),
        ("t4", "f4", "binary", 0.6),
    ];
    let base = arm_of("base", &cells);
    let mut twin = arm_of("twin", &cells.map(|(t, f, task, _)| (t, f, task, 1.0)));
    for row in &mut twin.rows {
        row.artifact_class = "control".into();
        row.packet_bytes = 0;
    }
    let unsupported = |row: &mut super::rundir::KpiRow| {
        row.status = "unsupported".into();
        row.r_s = None;
    };
    let twin = edit(edit(twin, "t3", unsupported), "t4", unsupported);
    let ctx = context("stat-inapplicable", &[base, twin]);
    let s = summarize(&ctx, ctx.arm("twin"));
    assert_eq!(
        (
            s.primary.trg,
            s.primary.cells,
            s.primary.not_applicable_cells
        ),
        (Some(0.0), 2, 2)
    );
    assert_eq!(s.primary.tasks.len(), 1);
    let c = &compare(&ctx, &["twin"], &[], "base", 0.01)[0];
    // Common cells t1, t2: differences 0.1 and 0.3.
    assert_eq!((c.cells, c.families), (2, 2));
    close(c.delta, 0.2);
    close(c.baseline_trg.unwrap(), -0.2);
    // A control carries no packet: the byte guard cannot pass.
    assert!(not_passing(c).contains(&"bytes"));
    assert_ne!(c.decision.outcome, Outcome::Promote);
    let states = super::trg::cell_states(&ctx.contract.protocol(), ctx.arm("twin"), 4, &|_| true);
    assert_eq!(states["p:t3:full:d0"], CellState::NotApplicable);
}

#[test]
fn an_arm_against_itself_ties_at_zero_and_holds() {
    let base = three("base", [0.8, 0.9, 0.7]);
    let ctx = context("stat-tie", &[with_guard_fields(base)]);
    let c = &compare(&ctx, &["base"], &[], "base", 0.01)[0];
    assert_eq!(
        (c.delta, c.ci_low, c.ci_high, c.p_one_sided),
        (0.0, 0.0, 0.0, 1.0)
    );
    assert_eq!(c.decision.outcome, Outcome::Hold);
}

#[test]
fn holm_adjusts_across_the_comparisons_of_one_receipt() {
    let arms = [
        three("base", [0.8, 0.8, 0.8]),
        three("up", [0.9, 0.9, 0.9]),
        three("down", [0.7, 0.7, 0.7]),
        three("up2", [1.0, 1.0, 1.0]),
    ];
    let ctx = context("stat-holm", &arms);
    let results = compare(&ctx, &["up", "down", "up2"], &[], "base", 0.01);
    let p = 1.0 / (DRAWS + 1.0);
    let raw: Vec<f64> = results.iter().map(|c| c.p_one_sided).collect();
    assert_eq!(raw, [p, 1.0, p]);
    // Sorted: up (x3), up2 (x2), down (x1); the running maximum keeps 3p, 3p, 1.
    let holm: Vec<f64> = results.iter().map(|c| c.p_holm).collect();
    close(holm[0], 3.0 * p);
    close(holm[2], 3.0 * p);
    assert_eq!(holm[1], 1.0);
}

#[test]
fn each_guard_flips_a_promoted_comparison_to_hold() {
    let base = with_guard_fields(three("base", [0.8, 0.8, 0.8]));
    let cand = with_guard_fields(three("cand", [0.9, 0.9, 0.9]));
    let ctx = context("stat-guards-pass", &[base.clone(), cand.clone()]);
    assert_eq!(
        compare(&ctx, &["cand"], &[], "base", 0.01)[0]
            .decision
            .outcome,
        Outcome::Promote
    );
    // The noise SD raises the delta bar from 0.01 to 0.2.
    assert_eq!(
        compare(&ctx, &["cand"], &[], "base", 0.2)[0]
            .decision
            .outcome,
        Outcome::Hold
    );
    type Edit = fn(&mut super::rundir::KpiRow);
    let cases: [(&str, &str, Edit); 6] = [
        ("realism_coverage_delta", "t1", |r| {
            r.realism_pass = Some(false)
        }),
        ("privacy_coverage_delta", "t2", |r| {
            r.privacy_pass = Some(false)
        }),
        ("exact_copy_excess", "t1", |r| {
            r.exact_copy_excess = Some(0.05)
        }),
        ("bytes", "t3", |r| r.packet_bytes = 2049),
        ("mlp_delta", "t1", |r| {
            if r.auditor == MLP {
                r.r_s = Some(0.0)
            }
        }),
        // Multiclass delta -0.05: the overall delta (0.1 + 0.1 - 0.05) / 3 still passes.
        ("task_ci_low", "t3", |r| r.r_s = Some(0.75)),
    ];
    for (index, (guard, table, change)) in cases.into_iter().enumerate() {
        let edited = edit(cand.clone(), table, change);
        let ctx = context(&format!("stat-guards-{index}"), &[base.clone(), edited]);
        let c = &compare(&ctx, &["cand"], &[], "base", 0.01)[0];
        assert_eq!(c.decision.outcome, Outcome::Hold, "{guard}");
        assert_eq!(not_passing(c), [guard]);
        assert!(
            c.decision.conditions.iter().all(|i| i.check == Check::Pass),
            "{guard}"
        );
    }
}

#[test]
fn a_seed_group_pools_its_seeds_and_needs_their_signs() {
    let base = with_guard_fields(three("base", [0.8, 0.8, 0.8]));
    let seeds = |deltas: [f64; 3]| -> Vec<super::rundir::Arm> {
        deltas
            .iter()
            .enumerate()
            .map(|(i, d)| with_guard_fields(three(&format!("s{i}"), [0.8 + d; 3])))
            .collect()
    };
    let arms = [vec![base.clone()], seeds([0.1, 0.2, 0.15])].concat();
    let ctx = context("stat-seed-group", &arms);
    let c = &compare(&ctx, &[], &["model=s0,s1,s2"], "base", 0.01)[0];
    close(c.delta, 0.15);
    let members: Vec<f64> = c.member_deltas.iter().map(|m| m.delta).collect();
    assert_eq!(members.len(), 3);
    close(members[1], 0.2);
    assert_eq!(
        (c.kind.as_str(), c.decision.outcome),
        ("seed_group", Outcome::Promote)
    );
    let arms = [vec![base], seeds([0.1, 0.2, -0.05])].concat();
    let ctx = context("stat-seed-group-sign", &arms);
    let c = &compare(&ctx, &[], &["model=s0,s1,s2"], "base", 0.01)[0];
    let signs = c
        .decision
        .conditions
        .iter()
        .find(|i| i.name == "seed_signs")
        .unwrap();
    assert_eq!((signs.check, signs.value), (Check::Fail, Some(2.0)));
    assert_eq!(c.decision.outcome, Outcome::Hold);
    assert!(parse_specs(&[], &["model".into()]).is_err());
    assert!(parse_specs(&[], &["m=a,a".into()]).is_err());
}

#[test]
fn receipts_are_byte_identical_for_the_same_inputs() {
    let arms = [
        three("base", [0.8, 0.9, 0.7]),
        three("cand", [0.85, 0.95, 0.6]),
    ];
    let bytes = |name: &str| {
        let ctx = context(name, &arms);
        let mut receipt = ctx.receipt("compare");
        receipt.comparisons = compare(&ctx, &["cand"], &[], "base", 0.01);
        receipt.arms = vec![summarize(&ctx, ctx.arm("base"))];
        receipt_bytes(&receipt).unwrap()
    };
    let (first, second) = (bytes("stat-determinism-a"), bytes("stat-determinism-b"));
    assert_eq!(first, second);
    let text = String::from_utf8(first).unwrap();
    assert!(
        text.contains("\"schema\": \"dope.kpi-receipt/v2\"") && text.contains("\"rows_sha256\"")
    );
    assert!(
        !text.contains("stat-determinism"),
        "a receipt holds no path"
    );
}

#[test]
fn the_sign_flip_null_catches_a_three_family_panel() {
    // One family per task: a flip pattern with a positive majority passes conditions (1)-(3)
    // (probability 1/2); all three positive also passes the per-task guard (1/8).
    let arms = [
        with_guard_fields(three("base", [0.8; 3])),
        with_guard_fields(three("cand", [0.9; 3])),
    ];
    let ctx = context("stat-null", &arms);
    let mut lines = Vec::new();
    let null = sign_flip_null(&ctx, "cand", "base", 2000, 7, 0.01, &mut lines).unwrap();
    assert_eq!((null.families, lines.len()), (3, 2000));
    assert!(
        (null.false_promote.rate - 0.125).abs() < 0.03,
        "{:?}",
        null.false_promote
    );
    assert!(
        (null.statistical_pass.rate - 0.5).abs() < 0.05,
        "{:?}",
        null.statistical_pass
    );
    assert!(!null.false_promote.acceptable && null.not_evaluated_guards.is_empty());
    assert_eq!(null.flip_guard_rule, null.false_promote);
    assert_eq!(null.observed.decision.outcome, Outcome::Promote);
    let line: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(line["replicate"], 0);
    assert!(line["flip_guard_rule"].is_boolean() && line["delta"].is_f64());
    let again = sign_flip_null(&ctx, "cand", "base", 2000, 7, 0.01, &mut Vec::new()).unwrap();
    assert_eq!(again, null);
    // Without the guard fields nothing can be promoted; with them assumed to pass, 1/8 again.
    let ctx = context(
        "stat-null-unflagged",
        &[three("base", [0.8; 3]), three("cand", [0.9; 3])],
    );
    let unflagged = sign_flip_null(&ctx, "cand", "base", 2000, 7, 0.01, &mut Vec::new()).unwrap();
    assert_eq!(unflagged.false_promote.count, 0);
    assert_eq!(unflagged.not_evaluated_guards.len(), 3);
    assert_eq!(unflagged.flip_guard_rule, null.flip_guard_rule);
    assert_eq!(unflagged.statistical_pass, null.statistical_pass);
}

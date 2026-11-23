//! Run-directory tests of `dope_kpi guards` and of the merge of its files
//! into the rows (`guards.rs`, `guard_merge.rs`) on hand-made ledgers
//! (`guard_fixture.rs`), one file per cell. Arm `copy`: t1 is a copy of A
//! (privacy fails, four exact copies of four rows: file excess 1), t2 two
//! copies of a twin that shares one row with A (privacy passes, two copies of
//! eight rows against the twin's one of four: file excess 0), t3 failed (no
//! excess). Its headline mean is 0.5; pooled over the six complete entries,
//! rate(S in A) = 18/36 and rate(B in A) = 3/24 give 0.375, which no guard
//! reads. Arm `twin` is a copy of B everywhere (excess 0).

use super::compare::compare_all;
use super::guard_fixture::{
    complete, failed, privacy, realism, request, scored, B_DUP, REAL_A, REAL_B,
};
use super::guard_merge::{GUARDS_FILE, GUARDS_SUMMARY_FILE};
use super::guards::guard_arm;
use super::jsonl::{canonical_json, canonical_pretty, sha256_hex, write_lines};
use super::pairing::parse_specs;
use super::primary::summarize;
use super::receipt::Context;
use super::rule::Check;
use super::rundir::{table_of, Arm, KpiRow, RunDir, SCORE_DIR};
use super::score::{ledger_id, LEDGER_FILE};
use super::stat_testing::{arm_of, edit, run_args, write_run};
use dope_core::dvs::v3::FidelityCellInput;
use std::path::{Path, PathBuf};

const CELLS: [(&str, &str, &str, f64); 3] = [
    ("t1", "f1", "regression", 1.0),
    ("t2", "f2", "binary", 1.0),
    ("t3", "f3", "multiclass", 1.0),
];

fn with_ids(mut arm: Arm) -> Arm {
    for row in &mut arm.rows {
        let id = ledger_id(&row.cell_id, row.sample_seed, row.multiplier, &row.auditor);
        row.ledger_id = Some(id);
    }
    arm
}

fn arms() -> [Arm; 2] {
    let fail = |row: &mut KpiRow| {
        row.status = "failed".into();
        row.r_s = None;
    };
    let copy = edit(with_ids(arm_of("copy", &CELLS)), "t3", fail);
    [copy, with_ids(arm_of("twin", &CELLS))]
}

fn copy_input(row: &KpiRow) -> FidelityCellInput {
    match table_of(&row.cell_id) {
        Some("t1") => scored(&REAL_A),
        Some("t2") => {
            let twice = [B_DUP, B_DUP].concat();
            complete(Some(realism(true)), Some(privacy(&REAL_A, &B_DUP, &twice)))
        }
        _ => failed(),
    }
}

fn arm_dir(dir: &Path, arm: &str) -> PathBuf {
    dir.join(SCORE_DIR).join(arm)
}

/// `score/<arm>/ledger.jsonl` with `input(row)` for every row, sorted by entry
/// id as `score` writes it (the entries of one file are consecutive).
fn write_ledger(dir: &Path, arm: &Arm, input: fn(&KpiRow) -> FidelityCellInput) {
    let mut lines: Vec<(String, String)> = arm
        .rows
        .iter()
        .map(|row| {
            let id = row.ledger_id.clone().unwrap();
            let line = canonical_json(&request(&id, &row.family_id, &row.auditor, input(row)));
            (id, line.unwrap())
        })
        .collect();
    lines.sort();
    let path = arm_dir(dir, &arm.id).join(LEDGER_FILE);
    write_lines(&path, lines.iter().map(|(_, line)| line.as_str())).unwrap();
}

/// A run directory with both arms scored and their guards written.
fn guarded(name: &str) -> PathBuf {
    let [copy, twin] = arms();
    let dir = write_run(name, &[copy.clone(), twin.clone()]);
    write_ledger(&dir, &copy, copy_input);
    write_ledger(&dir, &twin, |_| scored(&REAL_B));
    for arm in ["copy", "twin"] {
        guard_arm(&dir, arm, 2).unwrap();
    }
    dir
}

fn read(dir: &Path, arm: &str, file: &str) -> Vec<u8> {
    std::fs::read(arm_dir(dir, arm).join(file)).unwrap()
}

#[test]
fn guards_pool_exact_copies_and_fail_closed_entries() {
    let dir = guarded("guards-pool");
    let summary: serde_json::Value =
        serde_json::from_slice(&read(&dir, "copy", GUARDS_SUMMARY_FILE)).unwrap();
    assert_eq!(summary["schema"], "dope.kpi-guards/v2");
    assert_eq!(
        (
            summary["entries"].as_u64(),
            summary["by_status"]["failed"].as_u64()
        ),
        (Some(9), Some(3))
    );
    let privacy = &summary["privacy"];
    // Three auditors share each file's input: one assessment per complete file.
    assert_eq!(
        (
            privacy["passed"].as_u64(),
            privacy["assessed"].as_u64(),
            privacy["assessments"].as_u64()
        ),
        (Some(3), Some(6), Some(2))
    );
    assert_eq!(privacy["pass_share_complete"], 0.5);
    assert_eq!(summary["realism"]["passed"], 6);
    // Each file pools its own complete entries; the failed file has no value.
    assert_eq!(
        summary["file_exact_copy_excess"],
        serde_json::json!({"p:t1:full:d0:s0:m4": 1.0, "p:t2:full:d0:s0:m4": 0.0})
    );
    let pool = &summary["exact_copy"];
    assert_eq!(
        (
            pool["exact_copies"].as_u64(),
            pool["synthetic_rows"].as_u64()
        ),
        (Some(18), Some(36))
    );
    assert_eq!(
        (
            pool["twin_exact_copies"].as_u64(),
            pool["twin_rows"].as_u64()
        ),
        (Some(3), Some(24))
    );
    assert_eq!(summary["pooled_exact_copy_excess"], 0.375);
    let twin: serde_json::Value =
        serde_json::from_slice(&read(&dir, "twin", GUARDS_SUMMARY_FILE)).unwrap();
    assert_eq!(
        (
            twin["pooled_exact_copy_excess"].as_f64(),
            twin["privacy"]["passed"].as_u64()
        ),
        (Some(0.0), Some(9))
    );
    let files = twin["file_exact_copy_excess"].as_object().unwrap();
    assert_eq!(files.len(), 3);
    assert!(files.values().all(|v| v.as_f64() == Some(0.0)));
    let text = String::from_utf8(read(&dir, "copy", GUARDS_FILE)).unwrap();
    let lines: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 9);
    let failed = lines.iter().find(|l| l["status"] == "failed").unwrap();
    assert_eq!(
        (
            failed["realism_pass"].as_bool(),
            failed["exact_copies"].as_u64()
        ),
        (Some(false), None)
    );
    assert_eq!(lines[0]["cell_id"], "p:t1:full:d0");
    assert_eq!(
        (
            lines[0]["privacy_pass"].as_bool(),
            lines[0]["exact_copies"].as_u64()
        ),
        (Some(false), Some(4))
    );
}

#[test]
fn rerunning_with_other_threads_writes_the_same_bytes() {
    let dir = guarded("guards-determinism");
    let files = |dir: &Path| {
        [
            read(dir, "copy", GUARDS_FILE),
            read(dir, "copy", GUARDS_SUMMARY_FILE),
        ]
    };
    let first = files(&dir);
    for threads in [1, 5] {
        guard_arm(&dir, "copy", threads).unwrap();
        assert_eq!(files(&dir), first, "threads {threads}");
    }
}

#[test]
fn merged_rows_feed_primary_and_compare() {
    let dir = guarded("guards-merge");
    let run = RunDir::open(&dir, None).unwrap();
    let copy = run.load_arm("copy").unwrap();
    let flags = |table: &str| {
        let rows: Vec<&KpiRow> = copy
            .rows
            .iter()
            .filter(|r| table_of(&r.cell_id) == Some(table))
            .collect();
        assert_eq!(rows.len(), 3);
        assert!(rows
            .iter()
            .all(|r| r.exact_copy_excess == rows[0].exact_copy_excess));
        (
            rows[0].realism_pass,
            rows[0].privacy_pass,
            rows[0].exact_copy_excess,
        )
    };
    assert_eq!(flags("t1"), (Some(true), Some(false), Some(1.0)));
    assert_eq!(flags("t2"), (Some(true), Some(true), Some(0.0)));
    assert_eq!(flags("t3"), (Some(false), Some(false), None));
    let guards = copy.guards.as_ref().unwrap();
    assert_eq!(
        guards.guards_sha256,
        sha256_hex(&read(&dir, "copy", GUARDS_FILE))
    );

    let ctx = Context::open(&run_args(&dir), &["copy", "twin"]).unwrap();
    let s = summarize(&ctx, ctx.arm("copy"));
    // Per task: f1 (1, 0), f2 (1, 1), f3 failed (0, 0).
    let third = 1.0 / 3.0;
    assert!((s.realism_coverage.unwrap() - 2.0 * third).abs() < 1e-12);
    assert!((s.privacy_coverage.unwrap() - third).abs() < 1e-12);
    assert_eq!(s.exact_copy_excess, Some(0.5));
    let receipt = ctx.receipt("primary");
    assert_eq!(receipt.inputs.guards_sha256.len(), 2);
    assert_eq!(
        receipt.inputs.guards_summary_sha256["copy"],
        guards.summary_sha256
    );
    let specs = parse_specs(&["copy".to_string()], &[]).unwrap();
    let c = &compare_all(&ctx, &specs, "twin", 0.01).unwrap()[0];
    let guard = |name: &str| c.decision.guards.iter().find(|g| g.name == name).unwrap();
    assert_eq!(guard("realism_coverage_delta").check, Check::Fail);
    assert_eq!(guard("privacy_coverage_delta").check, Check::Fail);
    // The mean of the two scored files on the common cells, not the pooled 0.375.
    let copies = guard("exact_copy_excess");
    assert_eq!((copies.check, copies.value), (Check::Fail, Some(0.5)));
    let twin = summarize(&ctx, ctx.arm("twin"));
    assert_eq!(
        (twin.privacy_coverage, twin.exact_copy_excess),
        (Some(1.0), Some(0.0))
    );
}

#[test]
fn without_guard_files_rows_and_receipts_stay_as_they_were() {
    let dir = guarded("guards-absent");
    for file in [GUARDS_FILE, GUARDS_SUMMARY_FILE] {
        std::fs::remove_file(arm_dir(&dir, "copy").join(file)).unwrap();
    }
    let arm = RunDir::open(&dir, None).unwrap().load_arm("copy").unwrap();
    assert!(arm.guards.is_none());
    assert!(arm
        .rows
        .iter()
        .all(|r| r.realism_pass.is_none() && r.exact_copy_excess.is_none()));
    let ctx = Context::open(&run_args(&dir), &["copy"]).unwrap();
    let s = summarize(&ctx, ctx.arm("copy"));
    assert_eq!((s.realism_coverage, s.exact_copy_excess), (None, None));
    let text = canonical_json(&ctx.receipt("primary")).unwrap();
    assert!(!text.contains("guards") && !canonical_json(&s).unwrap().contains("coverage"));
}

fn load_err(dir: &Path) -> String {
    RunDir::open(dir, None)
        .unwrap()
        .load_arm("copy")
        .unwrap_err()
        .to_string()
}

/// The load error after `copy`'s guard lines become `edit(original)`, the summary kept consistent.
fn with_lines(dir: &Path, original: &[String], edit: fn(&mut Vec<String>)) -> String {
    let mut lines = original.to_vec();
    edit(&mut lines);
    let path = arm_dir(dir, "copy").join(GUARDS_FILE);
    let sha = write_lines(&path, lines.iter().map(String::as_str)).unwrap();
    let summary_path = arm_dir(dir, "copy").join(GUARDS_SUMMARY_FILE);
    let mut summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&summary_path).unwrap()).unwrap();
    summary["guards_sha256"] = sha.into();
    std::fs::write(summary_path, canonical_pretty(&summary).unwrap()).unwrap();
    load_err(dir)
}

type LinesEdit = fn(&mut Vec<String>);

#[test]
fn every_row_needs_its_guard_line_and_every_line_its_row() {
    let dir = guarded("guards-lines");
    let text = String::from_utf8(read(&dir, "copy", GUARDS_FILE)).unwrap();
    let original: Vec<String> = text.lines().map(str::to_string).collect();
    let cases: [(LinesEdit, &str); 4] = [
        (
            |l| {
                l.pop();
            },
            "score/copy/rows.jsonl: line 9 has no line in score/copy/guards.jsonl",
        ),
        (
            |l| l[8] = l[8].replace("\"status\":\"failed\"", "\"status\":\"ok\""),
            "score/copy/guards.jsonl: line 9 does not match score/copy/rows.jsonl line 9",
        ),
        (
            |l| l.push(l[0].replace(":gbdt\"", ":other\"")),
            "score/copy/guards.jsonl: line 10 matches no row of score/copy/rows.jsonl",
        ),
        (
            |l| l.push(l[0].clone()),
            "score/copy/guards.jsonl: line 10 repeats an entry",
        ),
    ];
    for (edit, expected) in cases {
        let err = with_lines(&dir, &original, edit);
        assert!(err.contains(expected), "{err}");
        assert!(!err.contains("0.5") && !err.contains("0.625"), "{err}");
    }
}

#[test]
fn stale_or_foreign_guard_files_are_refused() {
    let dir = guarded("guards-stale");
    // Hand-editing the guards file without the summary.
    let path = arm_dir(&dir, "copy").join(GUARDS_FILE);
    let original = std::fs::read(&path).unwrap();
    std::fs::write(&path, [original.as_slice(), b"\n"].concat()).unwrap();
    assert!(
        load_err(&dir).contains("does not describe"),
        "{}",
        load_err(&dir)
    );
    std::fs::write(&path, &original).unwrap();
    // Rows rewritten after the guards were computed.
    let rows = arm_dir(&dir, "copy").join(super::rundir::ROWS_FILE);
    let bytes = std::fs::read(&rows).unwrap();
    std::fs::write(&rows, [bytes.as_slice(), b"\n"].concat()).unwrap();
    assert!(
        load_err(&dir).contains("written for another score/copy/rows.jsonl"),
        "{}",
        load_err(&dir)
    );
    std::fs::write(&rows, &bytes).unwrap();
    // A summary written for another arm.
    let other = read(&dir, "twin", GUARDS_SUMMARY_FILE);
    std::fs::write(arm_dir(&dir, "copy").join(GUARDS_SUMMARY_FILE), other).unwrap();
    assert!(
        load_err(&dir).contains("not a dope.kpi-guards/v2 summary"),
        "{}",
        load_err(&dir)
    );
    std::fs::remove_file(arm_dir(&dir, "copy").join(GUARDS_SUMMARY_FILE)).unwrap();
    assert!(
        load_err(&dir).contains("guards-summary.json"),
        "{}",
        load_err(&dir)
    );
}

#[test]
fn guards_refuse_rows_that_do_not_match_the_ledger() {
    let [copy, _] = arms();
    let dir = write_run("guards-join", std::slice::from_ref(&copy));
    // The ledger misses the entries of t3.
    let mut partial = copy.clone();
    partial.rows.retain(|r| table_of(&r.cell_id) != Some("t3"));
    write_ledger(&dir, &partial, copy_input);
    let err = guard_arm(&dir, "copy", 1).unwrap_err().to_string();
    assert!(
        err.contains("score/copy/rows.jsonl: line 7 has no entry in score/copy/ledger.jsonl"),
        "{err}"
    );
    // A complete ledger entry for a row that failed.
    write_ledger(&dir, &copy, |_| scored(&REAL_B));
    let err = guard_arm(&dir, "copy", 1).unwrap_err().to_string();
    assert!(
        err.contains("score/copy/ledger.jsonl: line 7 does not match score/copy/rows.jsonl line 7"),
        "{err}"
    );
    assert!(
        !arm_dir(&dir, "copy").join(GUARDS_FILE).exists(),
        "a refused arm writes nothing"
    );
}

#[test]
fn a_row_whose_ledger_id_names_another_entry_is_refused() {
    let [copy, _] = arms();
    let mut swapped = copy.clone();
    let first = swapped.rows[0].ledger_id.take();
    swapped.rows[0].ledger_id = swapped.rows[1].ledger_id.replace(first.unwrap());
    let dir = write_run("guards-swapped", &[swapped]);
    write_ledger(&dir, &copy, copy_input);
    guard_arm(&dir, "copy", 1).unwrap();
    let err = load_err(&dir);
    assert!(
        err.contains("rows.jsonl: line 1 has a ledger_id that is not its cell, seed, multiplier"),
        "{err}"
    );
}

#[test]
fn rows_that_carry_guard_fields_cannot_take_a_guards_file() {
    let [copy, _] = arms();
    let mut flagged = copy.clone();
    flagged.rows[0].realism_pass = Some(true);
    let dir = write_run("guards-double", &[flagged]);
    write_ledger(&dir, &copy, copy_input);
    guard_arm(&dir, "copy", 1).unwrap();
    assert!(
        load_err(&dir).contains("line 1 already carries guard fields"),
        "{}",
        load_err(&dir)
    );
}

//! Black-box tests of `dope_kpi emit-cli` with generators that misbehave:
//! every failure is a row with its documented reason code, the run itself
//! succeeds, and no file with real rows is left under the run directory.

mod kpi_ext_support;
mod kpi_support;

use kpi_ext_support::{
    contains, files_under, fit_row, fixture, stem, write_arm_file, write_script, ArmDef, Fixture,
    MAT_GENERATOR, TEXT_GENERATOR,
};
use kpi_support::{json_lines, path_str};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// (arm, fit mode, sample mode, reason for the binary cell, reason for the
/// regression cell); an empty reason means the rows are `ok`.
const CASES: [(&str, &str, &str, &str, &str); 12] = [
    ("fit-exit", "exit3", "copy", "fit_exit", "fit_exit"),
    (
        "no-packet",
        "nopacket",
        "copy",
        "packet_missing",
        "packet_missing",
    ),
    ("sample-exit", "copy", "exit4", "sample_exit", "sample_exit"),
    (
        "no-output",
        "copy",
        "noout",
        "output_missing",
        "output_missing",
    ),
    (
        "drop-column",
        "copy",
        "dropcol",
        "header_mismatch",
        "header_mismatch",
    ),
    (
        "extra-column",
        "copy",
        "extracol",
        "header_mismatch",
        "header_mismatch",
    ),
    ("text-cell", "copy", "text", "parse_error", "parse_error"),
    // The label K is out of range for a classifier and an ordinary value for a regressor.
    ("bad-label", "copy", "badlabel", "label_out_of_range", ""),
    (
        "no-target",
        "copy",
        "notarget",
        "non_finite_target",
        "non_finite_target",
    ),
    (
        "short",
        "copy",
        "short",
        "row_count_mismatch",
        "row_count_mismatch",
    ),
    (
        "long",
        "copy",
        "long",
        "row_count_mismatch",
        "row_count_mismatch",
    ),
    // A reordered header is not a failure: columns are matched by name.
    ("shuffled", "copy", "shuffle", "", ""),
];

/// All arms are declared `source_free` so that the rows show which failures
/// still report a packet; the generators are test doubles, not encodings.
fn arms(fx: &Fixture) -> PathBuf {
    let text = write_script(&fx.dir, "gen.sh", TEXT_GENERATOR);
    let mat = write_script(&fx.dir, "genmat.sh", MAT_GENERATOR);
    let arm = |arm_id, io, script, fit_mode, sample_mode| ArmDef {
        arm_id,
        class: "source_free",
        io,
        script,
        fit_mode,
        sample_mode,
    };
    let mut arms: Vec<ArmDef> = CASES
        .iter()
        .map(|case| arm(case.0, "csv", &text, case.1, case.2))
        .collect();
    arms.push(arm("fit-sleep", "csv", &text, "sleep", "copy"));
    arms.push(arm("sample-sleep", "csv", &text, "copy", "sleep"));
    arms.push(arm("stats", "csv", &text, "stats", "stats"));
    arms.push(arm("mat-fit-exit", "mat", &mat, "exit3", "copy"));
    arms.push(arm("copy-csv", "csv", &text, "copy", "copy"));
    arms.push(arm("copy-mat", "mat", &mat, "copy", "copy"));
    write_arm_file(&fx.dir, "arms.json", &arms)
}

fn text(value: &Value) -> &str {
    value.as_str().expect("string field")
}

/// Check the rows of one arm against the expected reason per cell (in
/// schedule order) and return them.
fn check_rows(fx: &Fixture, out: &str, arm: &str, reasons: &[&str]) -> Vec<Value> {
    let arm_dir = fx.dir.join(out).join(arm);
    let rows = json_lines(&arm_dir.join("emit.jsonl"));
    let files_per_cell = rows.len() / reasons.len();
    assert_eq!(rows.len(), reasons.len() * files_per_cell);
    for (index, row) in rows.iter().enumerate() {
        let (cell, reason) = (
            &fx.cells[index / files_per_cell],
            reasons[index / files_per_cell],
        );
        assert_eq!(row["cell_id"], cell["cell_id"]);
        assert_eq!(text(&row["reason"]), reason, "{arm}: {row}");
        let name = format!(
            "{}.s{}.m{}.mat",
            stem(cell),
            row["sample_seed"],
            row["multiplier"]
        );
        if reason.is_empty() {
            assert_eq!(row["status"], "ok");
            assert_eq!(text(&row["file"]), format!("{arm}/{name}"));
            assert!(arm_dir.join(&name).is_file());
            continue;
        }
        assert_eq!(row["status"], "failed", "{arm}: {row}");
        assert_eq!((&row["rows"], &row["file"]), (&json!(0), &json!("")));
        assert!(
            !arm_dir.join(&name).exists(),
            "{arm}: a failed row left {name}"
        );
        let packet = arm_dir.join(format!("packets/{}.packet", stem(cell)));
        if matches!(
            reason,
            "fit_spawn" | "fit_exit" | "fit_timeout" | "packet_missing"
        ) {
            // No packet: nothing to measure, and a half-written one is not kept.
            assert_eq!(
                (&row["packet_bytes"], &row["packet_sha256"]),
                (&json!(0), &json!(""))
            );
            assert!(!packet.exists() && !packet.with_extension("card").exists());
            assert_eq!(row["decode_ms"], 0.0);
        } else {
            assert_eq!(
                row["packet_bytes"],
                std::fs::metadata(&packet).unwrap().len()
            );
            assert_eq!(text(&row["packet_sha256"]).len(), 64);
            assert!(row["encode_ms"].as_f64().unwrap() > 0.0);
        }
    }
    assert!(
        !arm_dir.join("_work").exists(),
        "{arm}: scratch was left behind"
    );
    rows
}

#[test]
fn every_generator_failure_is_a_row_with_its_reason() {
    let fx = fixture("emit-failures", "0");
    let arms = arms(&fx);
    assert_eq!(fx.cells[0]["task"], "binary");
    assert_eq!(fx.cells[1]["task"], "regression");
    let limit = ["--limit-cells", "2", "--threads", "2"];
    // A run with failing generators still succeeds and says so in its summary.
    let summaries: Vec<Value> = std::thread::scope(|scope| {
        let emit = |arm| fx.emit_ok(&arms, arm, "run", &limit);
        let running = CASES.map(|case| scope.spawn(move || emit(case.0)));
        running.map(|run| run.join().unwrap()).into()
    });
    for ((arm, _, _, binary, regression), summary) in CASES.into_iter().zip(summaries) {
        check_rows(&fx, "run", arm, &[binary, regression]);
        let failed_cells = [binary, regression]
            .iter()
            .filter(|r| !r.is_empty())
            .count();
        assert_eq!(summary["schedule_cells"], 6);
        assert_eq!(
            (&summary["cells"], &summary["limit_cells"]),
            (&json!(2), &json!(2))
        );
        assert_eq!(summary["cells_failed"], failed_cells, "{arm}");
        assert_eq!(summary["cells_ok"], 2 - failed_cells);
        assert_eq!(summary["rows"], 4);
        assert_eq!(summary["rows_failed"], 2 * failed_cells);
        let mut by_reason = serde_json::Map::new();
        for reason in [binary, regression].iter().filter(|r| !r.is_empty()) {
            let seen = by_reason.get(*reason).and_then(Value::as_u64).unwrap_or(0);
            by_reason.insert(reason.to_string(), json!(seen + 2));
        }
        assert_eq!(summary["rows_failed_by_reason"], Value::Object(by_reason));
    }

    // A program that cannot be started is a row too, in either phase.
    let script = fx.dir.join("gen.sh");
    let spawn_arm = |arm_id: &str, fit: Value, sample: Value| {
        json!({"arm_id": arm_id, "artifact_class": "control", "io": "csv",
               "binary": path_str(&script), "fit": fit, "sample": sample})
    };
    let good_fit = json!([
        "sh",
        "{bin}",
        "fit",
        "copy",
        "{fit_rows}",
        "{packet}",
        "{card}",
        "0",
        "{task}",
        "{task2}",
        "{classes}"
    ]);
    let absent = "/nonexistent/kpi/generator";
    let file = json!({"schema": "dope.kpi-ext-arms/v1", "arms": [
        spawn_arm("no-fit-program", json!([absent, "{packet}"]), json!([absent, "{out}"])),
        spawn_arm("no-sample-program", good_fit, json!([absent, "{out}"])),
    ]});
    let spawn_arms = write_script(&fx.dir, "spawn-arms.json", &file.to_string());
    fx.emit_ok(&spawn_arms, "no-fit-program", "run", &limit);
    check_rows(&fx, "run", "no-fit-program", &["fit_spawn", "fit_spawn"]);
    let summary = fx.emit_ok(&spawn_arms, "no-sample-program", "run", &limit);
    let rows = json_lines(&fx.dir.join("run/no-sample-program/emit.jsonl"));
    assert_eq!(rows.len(), 4);
    assert!(rows
        .iter()
        .all(|row| row["reason"] == "sample_spawn" && row["decode_ms"] == 0.0));
    assert_eq!(summary["rows_failed_by_reason"], json!({"sample_spawn": 4}));
}

/// The text and binary forms of fit rows of `cell`: the first, a middle and the last one.
fn row_patterns(fx: &Fixture, cell: &Value) -> Vec<Vec<u8>> {
    let table = fx.table_of(cell);
    let fit_rows = cell["a_idx"].as_array().unwrap().len();
    let mut patterns = Vec::new();
    for position in [0, fit_rows / 2, fit_rows - 1] {
        let values = fit_row(table, cell, position);
        let (target, features) = values.split_last().unwrap();
        let mut line = String::new();
        for value in features {
            if !value.is_nan() {
                line.push_str(&value.to_string());
            }
            line.push(',');
        }
        if cell["task"] == "regression" {
            line.push_str(&target.to_string());
        } else {
            line.push_str(&(*target as u32).to_string());
        }
        patterns.push(format!("\n{line}\n").into_bytes());
        patterns.push(values.iter().flat_map(|v| v.to_le_bytes()).collect());
    }
    patterns
}

/// Files under `dir` that contain a fit row of one of the first `cells` cells.
fn files_with_real_rows(fx: &Fixture, dir: &Path, cells: usize) -> Vec<PathBuf> {
    let patterns: Vec<Vec<u8>> = fx.cells[..cells]
        .iter()
        .filter(|cell| cell["status"] == "ok")
        .flat_map(|cell| row_patterns(fx, cell))
        .collect();
    let holds_rows = |path: &PathBuf| {
        let bytes = std::fs::read(path).unwrap();
        patterns.iter().any(|pattern| contains(&bytes, pattern))
    };
    files_under(dir).into_iter().filter(holds_rows).collect()
}

#[test]
fn no_file_with_real_rows_is_left_after_success_or_failure() {
    let fx = fixture("emit-privacy", "0");
    let arms = arms(&fx);
    // Success with a generator that keeps no rows; fits that fail after copying
    // the rows into the packet, the card and their working directory.
    for arm in ["stats", "fit-exit", "no-packet", "mat-fit-exit"] {
        fx.emit_ok(&arms, arm, "run", &["--threads", "2"]);
    }
    let run = fx.dir.join("run");
    assert_eq!(files_with_real_rows(&fx, &run, 6), Vec::<PathBuf>::new());
    let failed = json_lines(&run.join("mat-fit-exit/emit.jsonl"));
    assert_eq!(failed.len(), 12);
    assert!(failed[..10].iter().all(|row| row["reason"] == "fit_exit"));
    for arm in ["stats", "fit-exit", "no-packet", "mat-fit-exit"] {
        assert!(!run.join(arm).join("_work").exists());
    }
    let names = files_under(&run);
    assert!(
        names.iter().all(|path| !path.ends_with("leak.tmp")),
        "{names:?}"
    );
    // 10 matrices, 5 packets and 5 cards of `stats`; per arm a log for each
    // supported cell, the row log and the summary; nothing else.
    assert_eq!(names.len(), 20 + 4 * (5 + 2));

    // The scan does find rows where a generator is allowed to keep them.
    for (arm, kind) in [("copy-csv", "text"), ("copy-mat", "binary")] {
        fx.emit_ok(&arms, arm, "control", &["--limit-cells", "1"]);
        let found = files_with_real_rows(&fx, &fx.dir.join("control").join(arm), 1);
        let packet = found
            .iter()
            .any(|path| path.extension().is_some_and(|e| e == "packet"));
        assert!(
            packet,
            "the scan missed the {kind} rows in a copied packet: {found:?}"
        );
    }
}

/// The fixture with one cell and one file per cell, so a timeout test waits once.
fn single_file_schedule(fx: Fixture) -> Fixture {
    let mut cell = fx.cells[0].clone();
    cell["multipliers"] = json!([1]);
    let schedule = fx.dir.join("schedule-one.jsonl");
    std::fs::write(&schedule, format!("{cell}\n")).unwrap();
    Fixture {
        schedule,
        cells: vec![cell],
        ..fx
    }
}

fn check_timeout(name: &str, arm: &str, reason: &str, timeout_s: &str) {
    let fx = single_file_schedule(fixture(name, "0"));
    let arms = arms(&fx);
    let started = Instant::now();
    let summary = fx.emit_ok(&arms, arm, "run", &["--timeout-s", timeout_s]);
    // The generator sleeps for a minute; the run returns soon after the limit.
    assert!(
        started.elapsed() < Duration::from_secs(40),
        "{:?}",
        started.elapsed()
    );
    let rows = check_rows(&fx, "run", arm, &[reason]);
    assert_eq!(rows.len(), 1);
    let limit_ms = 1000.0 * timeout_s.parse::<f64>().unwrap();
    let timed = if reason == "fit_timeout" {
        "encode_ms"
    } else {
        "decode_ms"
    };
    let measured = rows[0][timed].as_f64().unwrap();
    assert!(
        measured >= limit_ms && measured < limit_ms + 30_000.0,
        "{measured}"
    );
    assert_eq!(summary["rows_failed_by_reason"], json!({ reason: 1 }));
    assert_eq!(summary["cells_failed_by_reason"], json!({ reason: 1 }));
    // The killed command had copied the rows into its working directory.
    let run = fx.dir.join("run");
    let names = files_under(&run);
    assert!(
        names.iter().all(|path| !path.ends_with("leak.tmp")),
        "{names:?}"
    );
    if reason == "fit_timeout" {
        assert_eq!(files_with_real_rows(&fx, &run, 1), Vec::<PathBuf>::new());
        assert_eq!(
            names.len(),
            3,
            "log, emit.jsonl and summary only: {names:?}"
        );
    }
    let log = std::fs::read_to_string(
        run.join(arm)
            .join(format!("_logs/{}.log", stem(&fx.cells[0]))),
    );
    assert!(log.unwrap().ends_with(": timeout, killed\n"));
}

#[test]
fn a_fit_that_hangs_is_killed_and_its_rows_are_removed() {
    check_timeout("emit-fit-timeout", "fit-sleep", "fit_timeout", "1.5");
}

#[test]
fn a_sample_that_hangs_is_killed_and_reported() {
    check_timeout("emit-sample-timeout", "sample-sleep", "sample_timeout", "2");
}

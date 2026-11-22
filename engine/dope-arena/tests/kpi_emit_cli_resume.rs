//! Black-box tests of `dope_kpi emit-cli --resume` and of the checks that
//! stop a run before any command is started.

mod kpi_ext_support;
mod kpi_support;

use kpi_ext_support::{
    files_under, fixture, sha256_hex, stem, write_arm_file, write_script, ArmDef, Fixture,
    TEXT_GENERATOR,
};
use kpi_support::{json_lines, path_str};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// A generator that records every invocation in a file outside the run
/// directory. Its fit fails for classification tables until a `flag` file
/// exists, and only waits and then fails while a `hold` file exists.
fn flaky_generator(dir: &Path) -> String {
    format!(
        "set -eu\necho \"$1 $7\" >> '{calls}'\n\
         if [ \"$1\" = fit ] && [ -e '{hold}' ]; then sleep 4; exit 7; fi\n\
         if [ \"$1\" = fit ] && [ \"$7\" != regression ] && [ ! -e '{flag}' ]; then exit 5; fi\n\
         . '{base}'\n",
        calls = dir.join("calls.txt").display(),
        hold = dir.join("hold").display(),
        flag = dir.join("flag").display(),
        base = dir.join("gen.sh").display(),
    )
}

fn setup(name: &str) -> (Fixture, PathBuf) {
    let fx = fixture(name, "0");
    write_script(&fx.dir, "gen.sh", TEXT_GENERATOR);
    let flaky = write_script(&fx.dir, "flaky.sh", &flaky_generator(&fx.dir));
    let arms = write_arm_file(
        &fx.dir,
        "arms.json",
        &[ArmDef {
            arm_id: "flaky",
            class: "source_free",
            io: "csv",
            script: &flaky,
            fit_mode: "stats",
            sample_mode: "stats",
        }],
    );
    (fx, arms)
}

/// The recorded fit invocations (task names), sorted; the record is then cleared.
fn take_fits(fx: &Fixture) -> Vec<String> {
    let path = fx.dir.join("calls.txt");
    let calls = std::fs::read_to_string(&path).unwrap_or_default();
    let _ = std::fs::remove_file(&path);
    let fits = calls.lines().filter_map(|line| line.strip_prefix("fit "));
    let mut fits: Vec<String> = fits.map(str::to_string).collect();
    fits.sort();
    fits
}

fn lines(path: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap();
    text.lines().map(str::to_string).collect()
}

#[test]
fn resume_reruns_only_the_cells_that_are_not_ok() {
    let (fx, arms) = setup("emit-resume");
    let emit = fx.dir.join("run/flaky/emit.jsonl");
    let first = fx.emit_ok(&arms, "flaky", "run", &["--threads", "2"]);
    assert_eq!(first["cells_failed_by_reason"], json!({"fit_exit": 2}));
    assert_eq!(
        (&first["cells_ok"], &first["cells_kept"]),
        (&json!(3), &json!(0))
    );
    assert_eq!(
        take_fits(&fx),
        [
            "binary",
            "multiclass",
            "regression",
            "regression",
            "regression"
        ]
    );
    let before = lines(&emit);
    assert_eq!(before.len(), 12);

    // A second plain run would silently replace the log: refused, nothing touched.
    let refusal = fx.emit_err(&arms, "flaky", "run", &[]);
    assert!(
        refusal.contains("pass --resume or use a new --out"),
        "{refusal}"
    );
    assert_eq!((lines(&emit), take_fits(&fx)), (before.clone(), Vec::new()));

    // With the cause removed, --resume runs the two failed cells and nothing else.
    std::fs::write(fx.dir.join("flag"), b"").unwrap();
    let second = fx.emit_ok(&arms, "flaky", "run", &["--resume", "--threads", "2"]);
    assert_eq!(take_fits(&fx), ["binary", "multiclass"]);
    assert_eq!(
        (&second["cells_kept"], &second["cells_run"]),
        (&json!(3), &json!(2))
    );
    assert_eq!(
        (&second["cells_ok"], &second["cells_failed"]),
        (&json!(5), &json!(0))
    );
    assert_eq!(
        (&second["rows_ok"], &second["rows_unsupported"]),
        (&json!(10), &json!(2))
    );
    let after = lines(&emit);
    assert_eq!(after.len(), 12);
    for (old, new) in before.iter().zip(&after) {
        let (old_row, new_row): (Value, Value) = (
            serde_json::from_str(old).unwrap(),
            serde_json::from_str(new).unwrap(),
        );
        assert_eq!(old_row["cell_id"], new_row["cell_id"]);
        if old_row["status"] == "failed" {
            assert_eq!(
                (&new_row["status"], &new_row["reason"]),
                (&json!("ok"), &json!(""))
            );
        } else {
            // Kept rows are carried over verbatim, timings included.
            assert_eq!(old, new);
        }
    }

    // Nothing is left to do: no command runs and the log does not change.
    let third = fx.emit_ok(&arms, "flaky", "run", &["--resume"]);
    assert_eq!(
        (&third["cells_kept"], &third["cells_run"]),
        (&json!(5), &json!(0))
    );
    assert_eq!((lines(&emit), take_fits(&fx)), (after.clone(), Vec::new()));

    // A cell whose file is gone, or was cut short, is run again in full.
    let cell = &fx.cells[1];
    let lost = fx.dir.join(format!("run/flaky/{}.s0.m4.mat", stem(cell)));
    std::fs::remove_file(&lost).unwrap();
    let cut = fx
        .dir
        .join(format!("run/flaky/{}.s0.m1.mat", stem(&fx.cells[3])));
    let bytes = std::fs::read(&cut).unwrap();
    std::fs::write(&cut, &bytes[..bytes.len() - 4]).unwrap();
    let fourth = fx.emit_ok(&arms, "flaky", "run", &["--resume"]);
    assert_eq!(
        (&fourth["cells_kept"], &fourth["cells_run"]),
        (&json!(3), &json!(2))
    );
    assert_eq!(take_fits(&fx), ["regression", "regression"]);
    assert!(lost.is_file());
    assert_eq!(std::fs::read(&cut).unwrap(), bytes);
    assert_eq!(fourth["rows_ok"], 10);

    // Rows made by another binary are never kept.
    let script = fx.dir.join("flaky.sh");
    let mut text = std::fs::read_to_string(&script).unwrap();
    text.push_str("# a new build\n");
    std::fs::write(&script, &text).unwrap();
    let fifth = fx.emit_ok(&arms, "flaky", "run", &["--resume", "--limit-cells", "3"]);
    assert_eq!(
        (&fifth["cells_kept"], &fifth["cells_run"]),
        (&json!(0), &json!(3))
    );
    assert_eq!((&fifth["cells"], &fifth["rows"]), (&json!(3), &json!(6)));
    assert_eq!(take_fits(&fx), ["binary", "multiclass", "regression"]);
    let rows = json_lines(&emit);
    assert_eq!(rows.len(), 6);
    assert!(rows
        .iter()
        .all(|row| row["weights_sha256"] == sha256_hex(text.as_bytes())));
}

#[test]
fn bad_inputs_stop_the_run_before_any_command() {
    let (fx, arms) = setup("emit-refusals");
    let out = fx.dir.join("run");
    let refused = |arms: &Path, arm: &str, extra: &[&str], needle: &str| {
        let err = fx.emit_err(arms, arm, "run", extra);
        assert!(err.contains(needle), "{needle}: {err}");
        assert!(
            err.starts_with("error: ") && err.lines().count() == 1,
            "{err}"
        );
        assert!(!out.exists(), "{needle}: the run directory was created");
        assert_eq!(take_fits(&fx), Vec::<String>::new());
    };
    refused(&arms, "absent", &[], "no arm named absent");
    refused(&arms, "../flaky", &[], "--arm is not a valid arm id");
    refused(&arms, "flaky", &["--timeout-s", "0"], "--timeout-s must be");
    refused(
        &arms,
        "flaky",
        &["--timeout-s", "inf"],
        "--timeout-s must be",
    );
    refused(
        &arms,
        "flaky",
        &["--limit-cells", "0"],
        "--limit-cells must be",
    );
    refused(&fx.dir.join("none.json"), "flaky", &[], "none.json");

    // An arm file is checked as a whole, including arms that are not selected.
    let arm_text = std::fs::read_to_string(&arms).unwrap();
    let edited = |name: &str, from: &str, to: &str| {
        assert!(arm_text.contains(from), "{from}");
        write_script(&fx.dir, name, &arm_text.replace(from, to))
    };
    let typo = edited("typo.json", "{fit_rows}", "{fit_row}");
    refused(
        &typo,
        "flaky",
        &[],
        "arm flaky: fit[4]: unknown placeholder {fit_row}",
    );
    let phase = edited("phase.json", "\"{seed}\"", "\"{fit_rows}\"");
    refused(
        &phase,
        "flaky",
        &[],
        "sample[8]: {fit_rows} has no value in the sample command",
    );
    let pinned = edited(
        "pinned.json",
        "\"io\"",
        &format!("\"binary_sha256\":\"{}\",\"io\"", "0".repeat(64)),
    );
    refused(&pinned, "flaky", &[], "binary does not match binary_sha256");
    let moved = edited("moved.json", "flaky.sh", "absent.sh");
    refused(&moved, "flaky", &[], "absent.sh");

    // A schedule that does not describe this arena is refused with the cell
    // position and the field, never with a value.
    let schedule = std::fs::read_to_string(&fx.schedule).unwrap();
    let with_schedule = |path: PathBuf| Fixture {
        dir: fx.dir.clone(),
        arena: fx.arena.clone(),
        schedule: path,
        tables: Vec::new(),
        cells: Vec::new(),
    };
    let tampered = |name: &str, from: &str, to: &str| {
        assert!(schedule.contains(from), "{from}");
        with_schedule(write_script(&fx.dir, name, &schedule.replacen(from, to, 1)))
    };
    let cases = [
        (
            "\"dataset_hash\":\"00000000000000a1\"",
            "\"dataset_hash\":\"00000000000000a2\"",
            "cell 0: invalid dataset_hash",
        ),
        (
            "\"table_id\":0",
            "\"table_id\":99",
            "cell 0: invalid table_id",
        ),
        (
            "\"task\":\"binary\"",
            "\"task\":\"regression\"",
            "cell 0: invalid task",
        ),
        (
            "\"cols\":[0,1,2,3]",
            "\"cols\":[0,1,2,4]",
            "cell 0: invalid cols",
        ),
        (
            "\"multipliers\":[4,1]",
            "\"multipliers\":[4,0]",
            "cell 0: invalid multipliers",
        ),
        (
            "\"sample_seeds\":[0]",
            "\"sample_seeds\":[]",
            "cell 0: invalid sample_seeds",
        ),
        (
            "\"status\":\"ok\"",
            "\"status\":\"maybe\"",
            "cell 0: invalid status",
        ),
        ("\"a_idx\":[", "\"a_idx\":[100000,", "cell 0: invalid a_idx"),
        (":full:d0\"", ":full/../x:d0\"", "cell 0: invalid cell_id"),
    ];
    for (index, (from, to, needle)) in cases.into_iter().enumerate() {
        let bad = tampered(&format!("schedule-{index}.jsonl"), from, to);
        let err = bad.emit_err(&arms, "flaky", "run", &[]);
        assert_eq!(
            err,
            format!("error: arena contract error: schedule: {needle}\n")
        );
        assert!(!out.exists());
    }
    let first = schedule.lines().next().unwrap();
    let doubled = write_script(&fx.dir, "doubled.jsonl", &format!("{first}\n{schedule}"));
    let err = with_schedule(doubled).emit_err(&arms, "flaky", "run", &[]);
    assert!(err.contains("schedule: duplicate cell_id"), "{err}");

    // After all refusals the same inputs still run, and leave only documented files.
    let summary = fx.emit_ok(&arms, "flaky", "run", &["--limit-cells", "2"]);
    assert_eq!(summary["rows_failed_by_reason"], json!({"fit_exit": 2}));
    let names = files_under(&out);
    assert_eq!(names.len(), 2 + 2 + 2 + 2, "{names:?}");
    assert!(path_str(&names[0]).contains("/run/flaky/"));
}

/// Poll until `ready` holds; the limit only bounds a broken run.
fn wait_until(what: &str, ready: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while !ready() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn an_interrupted_run_leaves_a_complete_log_and_is_resumed() {
    let (fx, arms) = setup("emit-interrupted");
    let arm_dir = fx.dir.join("run/flaky");
    let (emit, work) = (arm_dir.join("emit.jsonl"), arm_dir.join("_work"));
    std::fs::write(fx.dir.join("hold"), b"").unwrap();
    let mut running = Command::new(env!("CARGO_BIN_EXE_dope_kpi"))
        .args(["emit-cli", "--arena", path_str(&fx.arena)])
        .args(["--schedule", path_str(&fx.schedule)])
        .args(["--out", path_str(&fx.dir.join("run"))])
        .args(["--arm-file", path_str(&arms), "--arm", "flaky"])
        .args(["--threads", "2"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let fit_rows_on_disk = || {
        let files = if work.is_dir() {
            files_under(&work)
        } else {
            Vec::new()
        };
        files.iter().any(|path| path.ends_with("fit_rows.csv"))
    };
    wait_until("a fit in flight", || emit.is_file() && fit_rows_on_disk());

    // While commands run, the log already has every row: nothing is ok yet,
    // and no row claims a file.
    let rows = json_lines(&emit);
    assert_eq!(rows.len(), 12);
    for (index, row) in rows.iter().enumerate() {
        let (status, reason) = if index < 10 {
            ("failed", "run_interrupted")
        } else {
            ("unsupported", "schedule_unsupported")
        };
        assert_eq!(
            (&row["status"], &row["reason"]),
            (&json!(status), &json!(reason))
        );
        assert_eq!((&row["rows"], &row["file"]), (&json!(0), &json!("")));
    }
    assert!(!arm_dir.join("emit-summary.json").exists());

    // Kill the run. It cannot clean up: the fit rows of the cells in flight
    // stay in the scratch directory until the arm is run again.
    running.kill().unwrap();
    running.wait().unwrap();
    assert!(fit_rows_on_disk());
    assert_eq!(json_lines(&emit), rows);
    let refusal = fx.emit_err(&arms, "flaky", "run", &[]);
    assert!(refusal.contains("pass --resume"), "{refusal}");
    assert!(
        fit_rows_on_disk(),
        "a refused run must not touch the directory"
    );

    // The resumed run removes the leftovers first and then runs every cell.
    std::fs::remove_file(fx.dir.join("hold")).unwrap();
    std::fs::write(fx.dir.join("flag"), b"").unwrap();
    take_fits(&fx);
    let summary = fx.emit_ok(&arms, "flaky", "run", &["--resume", "--threads", "2"]);
    assert_eq!(
        (&summary["cells_kept"], &summary["cells_run"]),
        (&json!(0), &json!(5))
    );
    assert_eq!(
        (&summary["rows_ok"], &summary["rows_failed"]),
        (&json!(10), &json!(0))
    );
    assert_eq!(take_fits(&fx).len(), 5);
    assert!(!work.exists());
    let names = files_under(&fx.dir.join("run"));
    assert!(
        names.iter().all(|path| !path.ends_with("fit_rows.csv")),
        "{names:?}"
    );
    assert_eq!(names.len(), 2 + 5 * (1 + 2 + 2), "{names:?}");
}

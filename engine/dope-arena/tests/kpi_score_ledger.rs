//! Ledger contract of `dope_kpi score` for an external arm: complete failure
//! accounting, byte-identical reruns for any `--threads`, and a ledger that
//! the existing reader and `evaluate_v3` accept.

mod kpi_signal;
mod kpi_support;

use dope_arena::ledger::read_ledger;
use dope_arena::report::reduce_cells;
use dope_core::dvs::v3::FidelityCellInput;
use kpi_signal::{
    append_emit_log, by_key, emit_row, mat_name, number, repeated_a_rows, score_rows, text,
    write_mat, Signal,
};
use kpi_support::{json_file, json_lines, kpi_ok, path_str};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const STEMS: [&str; 4] = ["sig_regression", "sig_binary", "sig_multiclass", "sig_tiny"];
const ROSTER: &str = "linear-seeded-v1,gbdt";

struct Prepared {
    signal: Signal,
    schedule: PathBuf,
    cells: Vec<Value>,
    emit: PathBuf,
}

fn prepare(name: &str, plan_extra: &[&str]) -> Prepared {
    let signal = Signal::build(name, plan_extra);
    let schedule = signal.subset("schedule.jsonl", &STEMS);
    let cells = json_lines(&schedule);
    let emit = signal.dir.join("emit");
    Prepared {
        signal,
        schedule,
        cells,
        emit,
    }
}

impl Prepared {
    /// Emit `arm` as the A rows repeated `multiplier` times, with a log row per file.
    /// `edit` may change the file values or return a non-ok log row instead.
    fn emit(&self, arm: &str, edit: impl Fn(&str, u64, u64, &mut Vec<f32>) -> Option<Value>) {
        let mut log = Vec::new();
        for cell in &self.cells {
            let cell_id = text(&cell["cell_id"]);
            for seed in cell["sample_seeds"].as_array().unwrap() {
                for multiplier in [4, 1] {
                    let seed = seed.as_u64().unwrap();
                    if text(&cell["status"]) != "ok" {
                        let reason = "schedule_unsupported";
                        log.push(emit_row(
                            arm,
                            cell_id,
                            (seed, multiplier),
                            "unsupported",
                            reason,
                            0,
                            0,
                        ));
                        continue;
                    }
                    let (_, cols, mut data) =
                        repeated_a_rows(&self.signal, cell, multiplier as usize);
                    if let Some(row) = edit(text(&cell["task"]), seed, multiplier, &mut data) {
                        if !row.is_null() {
                            log.push(row);
                        }
                        continue;
                    }
                    let rows = data.len() / (cols + 1);
                    let path = self.emit.join(mat_name(arm, cell_id, seed, multiplier));
                    write_mat(&path, rows, cols, &data);
                    log.push(emit_row(
                        arm,
                        cell_id,
                        (seed, multiplier),
                        "ok",
                        "",
                        rows,
                        1500,
                    ));
                }
            }
        }
        append_emit_log(&self.emit, &log);
    }

    fn score(&self, realref: &Path, arm: &str, out: &str, extra: &[&str]) -> (PathBuf, Value) {
        self.signal
            .score(&self.schedule, realref, &self.emit, arm, out, extra)
    }
}

fn bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap()
}

#[test]
fn reruns_and_thread_counts_give_identical_bytes_and_a_reducible_ledger() {
    let p = prepare("score-ledger-clean", &[]);
    let roster = ["--auditors", ROSTER];
    let (realref, _) = p.signal.real_ref(&p.schedule, "realref.jsonl", &roster);
    let (realref4, _) = p.signal.real_ref(
        &p.schedule,
        "realref-t4.jsonl",
        &["--auditors", ROSTER, "--threads", "4"],
    );
    assert_eq!(
        bytes(&realref),
        bytes(&realref4),
        "real-ref depends on --threads"
    );

    p.emit("ext", |_, _, _, _| None);
    let (one, summary) = p.score(&realref, "ext", "score-t1", &["--threads", "1"]);
    let (four, summary4) = p.score(&realref4, "ext", "score-t4", &["--threads", "4"]);
    let (again, _) = p.score(&realref, "ext", "score-again", &["--threads", "4"]);
    assert_eq!(summary, summary4);
    for file in ["ledger.jsonl", "rows.jsonl"] {
        assert_eq!(
            bytes(&one.join(file)),
            bytes(&four.join(file)),
            "{file}: --threads"
        );
        assert_eq!(
            bytes(&four.join(file)),
            bytes(&again.join(file)),
            "{file}: rerun"
        );
    }
    let leftovers: Vec<_> = std::fs::read_dir(&four).unwrap().collect();
    assert_eq!(leftovers.len(), 2, "worker part directories are removed");

    // Every scheduled cell exactly once per (auditor, sample seed, multiplier).
    assert_eq!(summary["entries"], 4 * 2 * 2);
    assert_eq!(summary["by_status"]["ok"], 12);
    assert_eq!(summary["by_status"]["unsupported"], 4);
    let rows = score_rows(&one);
    let mut expected = BTreeSet::new();
    for cell in &p.cells {
        for auditor in ROSTER.split(',') {
            for multiplier in [4, 1] {
                expected.insert(format!(
                    "{}:s0:m{multiplier}:{auditor}",
                    text(&cell["cell_id"])
                ));
            }
        }
    }
    let ids: Vec<&str> = rows.iter().map(|r| text(&r["ledger_id"])).collect();
    assert_eq!(
        ids.iter().map(|id| id.to_string()).collect::<BTreeSet<_>>(),
        expected
    );
    assert!(
        ids.windows(2).all(|w| w[0] < w[1]),
        "rows are sorted by ledger id"
    );

    // Repeating every A row is the same table to a duplicate-invariant auditor.
    for row in rows.iter().filter(|r| text(&r["status"]) == "ok") {
        let r_s = number(&row["r_s"]);
        assert!((r_s - 1.0).abs() < 0.02, "{}: {r_s}", row["ledger_id"]);
        assert_eq!(row["packet_bytes"], 1500);
        assert_eq!(text(&row["artifact_class"]), "source_free");
        assert_eq!(text(&row["reason"]), "");
        assert!(row.get("encode_ms").is_none() && row.get("decode_ms").is_none());
    }

    // The ledger is the existing `FidelityCellRequest` format, in the same order.
    let (cells, _) = read_ledger(&one.join("ledger.jsonl")).unwrap();
    assert_eq!(cells.len(), rows.len());
    for (cell, row) in cells.iter().zip(&rows) {
        assert_eq!(cell.cell_id, text(&row["ledger_id"]));
        assert_eq!(cell.auditor_id, text(&row["auditor"]));
        assert_eq!(cell.family_id, text(&row["family_id"]));
        match (&cell.input, text(&row["status"])) {
            (
                FidelityCellInput::Complete {
                    bundle,
                    realism,
                    privacy,
                    ..
                },
                "ok",
            ) => {
                bundle.validate().unwrap();
                assert!(bundle.real_b_losses.is_some() && bundle.pred_b.is_some());
                assert!((bundle.mean_synth() - number(&row["l_s"])).abs() < 1e-12);
                assert!((bundle.mean_null() - number(&row["l0"])).abs() < 1e-12);
                assert_eq!(bundle.donor_ids.is_some(), text(&row["task"]) == "binary");
                assert!(realism.is_some());
                let privacy = privacy.as_ref().expect("privacy input");
                assert_eq!(privacy.real_fit_a.len(), 1024);
                assert_eq!(privacy.min_emitted_support, 10);
            }
            (FidelityCellInput::Unsupported, "unsupported") => {}
            (other, status) => panic!("{status} row with input {other:?}"),
        }
    }

    // `evaluate_v3` runs on it; copies of A pass parity and fail the privacy gate.
    let receipt = reduce_cells(cells, "faithful".into(), 0, 200).unwrap();
    assert_eq!(receipt.coverage.scheduled_cells, 16);
    assert_eq!(receipt.coverage.failed_or_missing_cells, 4);
    assert_eq!(receipt.coverage.informative_cells, 12);
    assert_eq!(receipt.coverage.eligible_denominator, 16);
    assert_eq!(receipt.coverage.privacy_pass_coverage, 0.0);
    assert!((0.0..=0.75).contains(&receipt.coverage.realism_coverage));
    for cell in receipt.cells.iter().filter(|c| c.status == "complete") {
        let privacy = cell.privacy.as_ref().unwrap();
        assert!(!privacy.pass && privacy.exact_copies > 0);
        assert!(cell.realism.as_ref().unwrap().c2st_auc < 0.6);
    }

    // The scoreboard reducer: one multiplier, thresholds from a file, partial evidence allowed.
    let kappa = p.signal.dir.join("kappa.json");
    let thresholds = "{\"pred_kappa_star\":1.5,\"lift_kappa_star\":1.75,\
                      \"calib_kappa_star\":2.0,\"driver_kappa_star\":1.25}";
    std::fs::write(&kappa, thresholds).unwrap();
    let ledger = one.join("ledger.jsonl");
    let (receipt_path, calibrated) = (
        p.signal.dir.join("receipt.json"),
        p.signal.dir.join("k.json"),
    );
    let stdout = kpi_ok(&[
        "reduce-ledger",
        "--ledger",
        path_str(&ledger),
        "--out",
        path_str(&receipt_path),
        "--multiplier",
        "4",
        "--kappa-file",
        path_str(&kappa),
        "--allow-partial",
        "--bootstrap-draws",
        "200",
        "--kappa-out",
        path_str(&calibrated),
    ]);
    let reduced: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(reduced["ledger_entries"], 16);
    assert_eq!(reduced["reduced_entries"], 8);
    assert_eq!(reduced["coverage"]["scheduled_cells"], 8);
    assert_eq!(reduced["coverage"]["informative_cells"], 6);
    assert_eq!(reduced["allow_partial_diagnostic"], true);
    assert_eq!(reduced["kappa_thresholds"]["calib_kappa_star"], 2.0);
    let file = json_file(&receipt_path);
    assert_eq!(file["receipt"]["cells"].as_array().unwrap().len(), 8);
    let complete = file["receipt"]["cells"].as_array().unwrap().iter();
    let complete: Vec<&Value> = complete.filter(|c| c["status"] == "complete").collect();
    assert_eq!(complete.len(), 6);
    assert_eq!(complete[0]["equivalence"]["lift"]["kappa_star"], 1.75);
    assert_eq!(file["filters"]["multiplier"], 4);
    let calibrated = json_file(&calibrated);
    let star = number(&calibrated["pred_kappa_star"]);
    assert!((1.0..=2.5).contains(&star), "calibrated kappa {star}");
    std::fs::remove_dir_all(&p.signal.dir).unwrap();
}

#[test]
fn every_failure_is_a_ledger_entry_and_the_cell_count_never_changes() {
    let p = prepare("score-ledger-faults", &["--sample-seeds", "0,1"]);
    let (realref, _) = p.signal.real_ref(
        &p.schedule,
        "realref.jsonl",
        &["--auditors", ROSTER, "--threads", "3"],
    );
    let arm = "ext_bad";
    let log = |task: &str, seed, multiplier, status: &str, reason: &str, bytes| {
        let stem = format!("sig_{task}");
        emit_row(
            arm,
            &p.signal.cell_id(&stem),
            (seed, multiplier),
            status,
            reason,
            0,
            bytes,
        )
    };
    p.emit(arm, |task, seed, multiplier, data| {
        let width = 5;
        match (task, seed, multiplier) {
            // The emitter reported failures of its own.
            ("regression", 0, 1) => {
                return Some(log(
                    task,
                    seed,
                    multiplier,
                    "failed",
                    "encode_exception:ValueError",
                    0,
                ))
            }
            ("binary", 1, 1) => {
                return Some(log(
                    task,
                    seed,
                    multiplier,
                    "unsupported",
                    "too_many_columns",
                    0,
                ))
            }
            ("multiclass", 1, 1) => {
                return Some(log(task, seed, multiplier, "over_budget", "", 9000))
            }
            ("regression", 1, 1) => {
                return Some(log(
                    task,
                    seed,
                    multiplier,
                    "failed",
                    "it broke on value 3.75",
                    0,
                ))
            }
            // No log row at all.
            ("multiclass", 0, 4) => return Some(Value::Null),
            // Files with bad content.
            ("binary", 0, 1) => data.chunks_mut(width).for_each(|row| row[width - 1] = 1.0),
            ("multiclass", 0, 1) => data[width - 1] = f32::NAN,
            ("binary", 1, 4) => data[3] = f32::INFINITY,
            ("regression", 1, 4) => data.truncate(data.len() - width),
            _ => {}
        }
        None
    });
    // A logged file that is missing, and one that is not a DOPEMAT1 container.
    let file = |stem: &str, seed, multiplier| {
        p.emit
            .join(mat_name(arm, &p.signal.cell_id(stem), seed, multiplier))
    };
    std::fs::remove_file(file("sig_regression", 0, 4)).unwrap();
    std::fs::write(file("sig_multiclass", 1, 4), b"DOPEMAT1 but truncated").unwrap();

    let expect = |rows: &[Value], stem: &str, seed, multiplier, status: &str, reason: &str| {
        let rows = by_key(rows);
        for auditor in ROSTER.split(',') {
            let row = rows[&(
                p.signal.cell_id(stem),
                auditor.to_string(),
                multiplier,
                seed,
            )];
            let label = format!("{stem} s{seed} m{multiplier} {auditor}");
            assert_eq!(text(&row["status"]), status, "{label}");
            assert_eq!(text(&row["reason"]), reason, "{label}");
            if status != "ok" {
                assert!(row["l_s"].is_null() && row["r_s"].is_null(), "{label}");
                assert!(row["l0"].is_number() || status == "unsupported", "{label}");
            }
        }
    };
    let (out, summary) = p.score(&realref, arm, "score", &["--threads", "3"]);
    assert_eq!(summary["entries"], 4 * 2 * 2 * 2, "cell count is unchanged");
    let rows = score_rows(&out);
    expect(&rows, "sig_regression", 0, 4, "failed", "missing_file");
    expect(
        &rows,
        "sig_regression",
        0,
        1,
        "failed",
        "emit:encode_exception:ValueError",
    );
    expect(
        &rows,
        "sig_regression",
        1,
        1,
        "failed",
        "emit:nonconforming_reason",
    );
    expect(
        &rows,
        "sig_regression",
        1,
        4,
        "failed",
        "row_count_mismatch",
    );
    expect(&rows, "sig_binary", 0, 4, "ok", "");
    expect(&rows, "sig_binary", 0, 1, "failed", "single_class_target");
    expect(&rows, "sig_binary", 1, 4, "failed", "nonfinite_rows");
    expect(
        &rows,
        "sig_binary",
        1,
        1,
        "unsupported",
        "emit:too_many_columns",
    );
    expect(&rows, "sig_multiclass", 0, 4, "failed", "missing_emit_row");
    expect(&rows, "sig_multiclass", 0, 1, "failed", "nonfinite_rows");
    expect(&rows, "sig_multiclass", 1, 4, "failed", "malformed_file");
    expect(&rows, "sig_multiclass", 1, 1, "over_budget", "over_budget");
    for (seed, multiplier) in [(0, 4), (0, 1), (1, 4), (1, 1)] {
        expect(
            &rows,
            "sig_tiny",
            seed,
            multiplier,
            "unsupported",
            "schedule_unsupported",
        );
    }
    assert_eq!(summary["by_status"]["ok"], 2);
    assert_eq!(summary["by_status"]["failed"], 18);
    assert_eq!(summary["by_status"]["over_budget"], 2);
    assert_eq!(summary["by_status"]["unsupported"], 10);
    let text_of = std::fs::read_to_string(out.join("rows.jsonl")).unwrap()
        + &std::fs::read_to_string(out.join("ledger.jsonl")).unwrap();
    assert!(!text_of.contains("3.75") && !text_of.contains("it broke"));

    // The reader and `evaluate_v3` take the ledger: failures stay in the denominator.
    let (cells, _) = read_ledger(&out.join("ledger.jsonl")).unwrap();
    assert_eq!(cells.len(), 32);
    let over = cells.iter().find_map(|c| match c.input {
        FidelityCellInput::OverBudget {
            actual_bytes,
            cap_bytes,
        } => Some((actual_bytes, cap_bytes)),
        _ => None,
    });
    assert_eq!(over, Some((9000, 0)));
    let receipt = reduce_cells(cells, "faithful".into(), 0, 200).unwrap();
    assert_eq!(receipt.coverage.scheduled_cells, 32);
    assert_eq!(receipt.coverage.informative_cells, 2);
    assert_eq!(receipt.coverage.failed_or_missing_cells, 30);
    assert_eq!(receipt.coverage.eligible_denominator, 32);

    // A byte budget turns every longer packet into an over-budget entry, nothing else.
    let (capped, summary) = p.score(&realref, arm, "score-cap", &["--budget-bytes", "1024"]);
    assert_eq!(summary["entries"], 32);
    let rows = score_rows(&capped);
    expect(&rows, "sig_binary", 0, 4, "over_budget", "over_budget");
    expect(&rows, "sig_regression", 0, 4, "over_budget", "over_budget");
    expect(
        &rows,
        "sig_regression",
        0,
        1,
        "failed",
        "emit:encode_exception:ValueError",
    );
    expect(&rows, "sig_multiclass", 0, 4, "failed", "missing_emit_row");
    let (cells, _) = read_ledger(&capped.join("ledger.jsonl")).unwrap();
    let capped_at = cells.iter().any(|c| {
        matches!(
            c.input,
            FidelityCellInput::OverBudget {
                actual_bytes: 1500,
                cap_bytes: 1024
            }
        )
    });
    assert!(capped_at);
    std::fs::remove_dir_all(&p.signal.dir).unwrap();
}

//! Control-arm self-test of `dope_kpi real-ref` + `dope_kpi score` on tables
//! with a known signal: a copy of A retains the real gain, a bootstrap of A
//! agrees with the copy for every auditor, tables with the feature-target
//! link destroyed retain nothing, and a real twin scores like the twin B.

mod kpi_signal;
mod kpi_support;

use kpi_signal::{by_key, number, score_rows, text, Signal, AUDITORS};
use kpi_support::json_lines;
use serde_json::Value;
use std::collections::BTreeSet;

const SIGNAL_STEMS: [&str; 3] = ["sig_regression", "sig_binary", "sig_multiclass"];
const SEEDS: [u64; 3] = [0, 1, 2];

fn retention(row: &Value) -> f64 {
    assert_eq!(text(&row["status"]), "ok", "{}", row["ledger_id"]);
    number(&row["r_s"])
}

#[test]
fn copies_keep_the_signal_and_shuffles_destroy_it() {
    let signal = Signal::build("score-controls", &["--sample-seeds", "0,1,2"]);
    let schedule = signal.subset(
        "schedule-signal.jsonl",
        &["sig_regression", "sig_binary", "sig_multiclass", "sig_tiny"],
    );
    let (realref, summary) = signal.real_ref(&schedule, "realref.jsonl", &["--threads", "4"]);
    assert_eq!(summary["cells"], 4);
    assert_eq!(summary["rows"], 12);
    assert_eq!(summary["ok"], 9);
    assert_eq!(summary["unsupported_by_reason"]["schedule_unsupported"], 3);
    assert_eq!(
        summary["informative"], 9,
        "every signal cell is informative"
    );

    // Real reference: the twin B retains about as much as A, and A against B is not separable.
    for row in json_lines(&realref) {
        if text(&row["status"]) != "ok" {
            continue;
        }
        let label = format!("{} {}", row["cell_id"], row["auditor"]);
        let r_b = number(&row["r_b"]);
        assert!((0.9..=1.1).contains(&r_b), "{label}: twin retention {r_b}");
        assert!(number(&row["l_a"]) < 0.6 * number(&row["l0"]), "{label}");
        assert!(number(&row["c2st_real_ref"]) < 0.56, "{label}");
        let queries = row["y_true"].as_array().unwrap().len();
        assert!((879..=910).contains(&queries), "{label}: {queries} T rows");
        assert_eq!(row["null_losses"].as_array().unwrap().len(), queries);
    }

    let emit = signal.dir.join("emit");
    let run = |arm: &str| {
        let (out, summary) = signal.score(
            &schedule,
            &realref,
            &emit,
            arm,
            &format!("score-{arm}"),
            &["--threads", "4", "--privacy-rows", "64"],
        );
        // Every scheduled cell appears once per (auditor, sample seed, multiplier).
        assert_eq!(summary["entries"], 4 * 3 * 3 * 2);
        assert_eq!(
            summary["by_status"]["ok"],
            3 * 3 * 3 * 2,
            "{arm}: {summary}"
        );
        assert_eq!(summary["by_status"]["unsupported"], 3 * 3 * 2);
        let rows = score_rows(&out);
        let ids: BTreeSet<&str> = rows.iter().map(|r| text(&r["ledger_id"])).collect();
        assert_eq!(ids.len(), rows.len());
        rows
    };
    let copy = run("copy_a");
    let boot = run("bootstrap_a");
    let shuffle = run("label_shuffle");
    let indep = run("indep_marginals");
    let (copy, boot) = (by_key(&copy), by_key(&boot));
    let (shuffle, indep) = (by_key(&shuffle), by_key(&indep));

    for stem in SIGNAL_STEMS {
        for auditor in AUDITORS {
            for multiplier in [4, 1] {
                let key = |seed| (signal.cell_id(stem), auditor.to_string(), multiplier, seed);
                let label = format!("{stem} {auditor} m{multiplier}");
                let mean = |rows: &std::collections::BTreeMap<_, &Value>| {
                    SEEDS.iter().map(|&s| retention(rows[&key(s)])).sum::<f64>() / 3.0
                };
                for seed in SEEDS {
                    let r_copy = retention(copy[&key(seed)]);
                    assert!((r_copy - 1.0).abs() < 0.01, "{label}: copy_a {r_copy}");
                    let r_boot = retention(boot[&key(seed)]);
                    if multiplier == 4 {
                        // Rule A-1: four draws per row behave like the rows themselves.
                        assert!(
                            (r_boot - r_copy).abs() < 0.05,
                            "{label} s{seed}: bootstrap_a {r_boot} vs copy_a {r_copy}"
                        );
                    } else {
                        // One draw per row keeps about 63% of the rows: a little lower.
                        assert!((0.85..1.06).contains(&r_boot), "{label} s{seed}: {r_boot}");
                    }
                    for (arm, rows) in [("label_shuffle", &shuffle), ("indep_marginals", &indep)] {
                        let r = retention(rows[&key(seed)]);
                        assert!(r < 0.25, "{label} s{seed}: {arm} retained {r}");
                    }
                }
                if multiplier == 4 {
                    let gap = (mean(&boot) - mean(&copy)).abs();
                    assert!(gap < 0.025, "{label}: mean bootstrap gap {gap}");
                }
                assert!(
                    mean(&shuffle) < 0.1,
                    "{label}: label_shuffle {}",
                    mean(&shuffle)
                );
                assert!(
                    mean(&indep) < 0.1,
                    "{label}: indep_marginals {}",
                    mean(&indep)
                );

                let row = copy[&key(0)];
                assert_eq!(text(&row["artifact_class"]), "control");
                assert_eq!(row["packet_bytes"], 0);
                assert_eq!(row["informative"], true);
                assert_eq!(row["rows"], 1024);
                assert_eq!(boot[&key(1)]["rows"], 1024 * multiplier);
                // A copy of A against B is the real-vs-real reference itself.
                assert_eq!(row["c2st_auc"], row["c2st_real_ref"]);
                // Rows with the dependence removed are easy to tell from real rows.
                assert!(number(&indep[&key(2)]["c2st_auc"]) > 0.65, "{label}");
                assert!(number(&shuffle[&key(2)]["c2st_auc"]) > 0.65, "{label}");
            }
        }
    }
    // The table too small to split stays in the output as unsupported rows.
    let tiny = copy[&(signal.cell_id("sig_tiny"), "gbdt".to_string(), 4, 0)];
    assert_eq!(text(&tiny["status"]), "unsupported");
    assert_eq!(text(&tiny["reason"]), "schedule_unsupported");
    assert!(tiny["r_s"].is_null() && tiny["l0"].is_null());
    std::fs::remove_dir_all(&signal.dir).unwrap();
}

#[test]
fn a_real_twin_scores_like_the_twin_reference() {
    let signal = Signal::build("score-twin", &[]);
    let schedule = signal.subset("schedule-twin.jsonl", &["sig_twin", "sig_binary"]);
    let (realref, _) = signal.real_ref(&schedule, "realref.jsonl", &["--threads", "2"]);
    let emit = signal.dir.join("emit");
    let (out, summary) = signal.score(
        &schedule,
        &realref,
        &emit,
        "twin_c",
        "score-twin",
        &["--threads", "2", "--privacy-rows", "0"],
    );
    // The smaller table has three C rows: its entries are unsupported, not dropped.
    assert_eq!(summary["entries"], 2 * 3 * 2);
    assert_eq!(summary["by_status"]["ok"], 6);
    assert_eq!(summary["by_reason"]["emit:c_rows_below_minimum"], 6);
    let rows = score_rows(&out);
    let rows = by_key(&rows);
    for auditor in AUDITORS {
        let m4 = rows[&(signal.cell_id("sig_twin"), auditor.to_string(), 4, 0)];
        let m1 = rows[&(signal.cell_id("sig_twin"), auditor.to_string(), 1, 0)];
        let (r_s, r_b) = (retention(m4), number(&m4["r_b"]));
        assert!(
            (r_s - r_b).abs() < 0.06,
            "{auditor}: twin_c {r_s} vs B {r_b}"
        );
        assert!((r_s - 1.0).abs() < 0.08, "{auditor}: twin_c {r_s}");
        // C does not grow with the multiplier: both files hold the same 1,024 rows.
        assert_eq!(
            (m4["rows"].as_u64(), m1["rows"].as_u64()),
            (Some(1024), Some(1024))
        );
        assert_eq!(m4["r_s"], m1["r_s"]);
        assert!(
            number(&m4["c2st_auc"]) < 0.56,
            "real rows against real rows"
        );
        let none = rows[&(signal.cell_id("sig_binary"), auditor.to_string(), 4, 0)];
        assert_eq!(text(&none["status"]), "unsupported");
        assert_eq!(none["rows"], 0);
    }
    std::fs::remove_dir_all(&signal.dir).unwrap();
}

#[test]
fn the_format2_engine_is_scored_as_a_source_free_arm() {
    let signal = Signal::build("score-format2", &[]);
    let stems = ["sig_regression", "sig_binary", "sig_tiny"];
    let schedule = signal.subset("schedule-format2.jsonl", &stems);
    let roster = ["--auditors", "linear-seeded-v1,gbdt", "--threads", "2"];
    let (realref, _) = signal.real_ref(&schedule, "realref.jsonl", &roster);
    let emit = signal.dir.join("emit");
    let arm = "rust_format2_b4096";
    let (out, summary) = signal.score(
        &schedule,
        &realref,
        &emit,
        arm,
        "score-f2",
        &["--threads", "2"],
    );
    assert_eq!(summary["entries"], 3 * 2 * 2);
    assert_eq!(summary["by_status"]["ok"], 8, "{summary}");
    let rows = score_rows(&out);
    for row in rows.iter().filter(|r| text(&r["status"]) == "ok") {
        let label = text(&row["ledger_id"]);
        assert_eq!(text(&row["artifact_class"]), "source_free", "{label}");
        let bytes = row["packet_bytes"].as_u64().unwrap();
        assert!(bytes > 0 && bytes <= 4096, "{label}: {bytes} packet bytes");
        assert_eq!(
            row["rows"],
            1024 * row["multiplier"].as_u64().unwrap(),
            "{label}"
        );
        // A fitted generator keeps part of the signal and never beats real rows by much.
        let r_s = retention(row);
        assert!((0.2..1.1).contains(&r_s), "{label}: retention {r_s}");
        assert!(number(&row["c2st_auc"]) < 0.95, "{label}");
    }
    // The log declares the engine's support floor and the measured packet.
    let log = json_lines(&emit.join(arm).join("emit.jsonl"));
    assert_eq!(log.len(), 3 * 2);
    for row in log.iter().filter(|r| text(&r["status"]) == "ok") {
        assert_eq!(row["min_emitted_support"], 10);
        assert_eq!(text(&row["packet_sha256"]).len(), 64);
        assert!(emit.join(text(&row["file"])).is_file());
    }
    // One process or two: the same files, log and scores.
    let (single, _) = signal.score(&schedule, &realref, &emit, arm, "score-f2-t1", &[]);
    for file in ["ledger.jsonl", "rows.jsonl"] {
        let (a, b) = (
            std::fs::read(out.join(file)),
            std::fs::read(single.join(file)),
        );
        assert_eq!(a.unwrap(), b.unwrap(), "{file} depends on --threads");
    }
    assert_eq!(json_lines(&emit.join(arm).join("emit.jsonl")), log);
    std::fs::remove_dir_all(&signal.dir).unwrap();
}

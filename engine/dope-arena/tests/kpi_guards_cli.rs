//! Black-box test of `dope_kpi guards` and of `primary` / `compare` reading
//! its files, on a run directory written here from hand-made ledgers
//! (`src/kpi/guard_fixture.rs`): three cells (one family per task), one
//! sample seed, multipliers 4 and 1, three auditors, `R_B = 1` everywhere.
//!
//! * `twin`: a copy of the real twin B (every gate passes, no exact copy).
//! * `copy`: a copy of A (privacy fails; rate(S in A) = 1, excess 1).
//! * `gen`: retention 1.1 (TRG +0.1 over `twin`), a copy of B, 1500-byte
//!   packets; its matched file of the multiclass cell failed.
//!
//! Before `guards` the realism, privacy and exact-copy guards are not
//! evaluated and `gen` holds; after it `gen` is promoted and `copy` fails
//! the privacy and exact-copy guards.

#[path = "../src/kpi/guard_fixture.rs"]
mod guard_fixture;
mod kpi_support;

use guard_fixture::{failed, request, scored, REAL_A, REAL_B};
use kpi_support::{json_file, json_lines, kpi_err, kpi_ok, path_str, scratch};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const AUDITORS: [&str; 3] = ["gbdt", "linear-seeded-v1", "mlp-seeded-v1"];
const CELLS: [(&str, &str, &str); 3] = [
    ("ta", "famR", "regression"),
    ("tb", "famB", "binary"),
    ("tc", "famC", "multiclass"),
];

struct ArmSpec {
    id: &'static str,
    class: &'static str,
    packet_bytes: u64,
    r_s: f64,
    synthetic: [f64; 4],
    /// (table, multiplier) of a file that failed.
    failed: Option<(&'static str, u32)>,
}

const ARMS: [ArmSpec; 3] = [
    ArmSpec {
        id: "twin",
        class: "control",
        packet_bytes: 0,
        r_s: 1.0,
        synthetic: REAL_B,
        failed: None,
    },
    ArmSpec {
        id: "copy",
        class: "control",
        packet_bytes: 0,
        r_s: 1.0,
        synthetic: REAL_A,
        failed: None,
    },
    ArmSpec {
        id: "gen",
        class: "source_free",
        packet_bytes: 1500,
        r_s: 1.1,
        synthetic: REAL_B,
        failed: Some(("tc", 1)),
    },
];

fn cell_id(table: &str) -> String {
    format!("g:{table}:full:d0")
}

fn write_jsonl(path: &Path, lines: &[String]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, lines.join("\n") + "\n").unwrap();
}

/// Schedule, real reference and every arm's rows and ledger, sorted by entry id as `score` writes them.
fn write_run(dir: &Path) {
    let mut schedule = Vec::new();
    let mut realref = Vec::new();
    for (table, family, task) in CELLS {
        schedule.push(json!({"cell_id": cell_id(table), "panel_sha256": "ps", "dataset_hash": table,
            "family_id": family, "task": task, "view": "full", "data_seed": 0, "stratum": "headline",
            "sample_seeds": [0], "multipliers": [4, 1]}).to_string());
        for auditor in AUDITORS {
            realref.push(
                json!({"cell_id": cell_id(table), "auditor": auditor, "informative": true,
                "l0": 1.0, "l_a": 0.5, "l_b": 0.5, "r_b": 1.0, "c2st_real_ref": 0.5})
                .to_string(),
            );
        }
    }
    write_jsonl(&dir.join("schedule.jsonl"), &schedule);
    write_jsonl(&dir.join("realref.jsonl"), &realref);
    for arm in &ARMS {
        let mut entries: Vec<(String, String, String)> = Vec::new();
        for (table, family, task) in CELLS {
            for multiplier in [4u32, 1] {
                let failed_file = arm.failed == Some((table, multiplier));
                for auditor in AUDITORS {
                    let id = format!("{}:s0:m{multiplier}:{auditor}", cell_id(table));
                    let row = json!({"artifact_class": arm.class, "auditor": auditor, "c2st_auc": 0.5,
                        "c2st_real_ref": 0.5, "cell_id": cell_id(table), "data_seed": 0, "family_id": family,
                        "informative": true, "l0": 1.0, "l_a": 0.5, "l_b": 0.5, "ledger_id": id,
                        "multiplier": multiplier, "packet_bytes": arm.packet_bytes, "r_b": 1.0,
                        "r_s": if failed_file { Value::Null } else { arm.r_s.into() },
                        "reason": if failed_file { "missing_file" } else { "" }, "sample_seed": 0,
                        "status": if failed_file { "failed" } else { "ok" }, "stratum": "headline",
                        "task": task, "view": "full"});
                    let input = if failed_file {
                        failed()
                    } else {
                        scored(&arm.synthetic)
                    };
                    let ledger =
                        serde_json::to_string(&request(&id, family, auditor, input)).unwrap();
                    entries.push((id, row.to_string(), ledger));
                }
            }
        }
        entries.sort();
        let at = dir.join("score").join(arm.id);
        write_jsonl(
            &at.join("rows.jsonl"),
            &entries.iter().map(|e| e.1.clone()).collect::<Vec<_>>(),
        );
        write_jsonl(
            &at.join("ledger.jsonl"),
            &entries.iter().map(|e| e.2.clone()).collect::<Vec<_>>(),
        );
    }
}

fn run_dir(name: &str) -> PathBuf {
    let dir = scratch(name);
    write_run(&dir);
    dir
}

fn guard_files(dir: &Path, arm: &str) -> [Vec<u8>; 2] {
    let at = dir.join("score").join(arm);
    ["guards.jsonl", "guards-summary.json"].map(|f| std::fs::read(at.join(f)).unwrap())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn guards(dir: &Path, threads: &str) -> Value {
    let mut args = vec!["guards", "--run-dir", path_str(dir), "--threads", threads];
    for arm in &ARMS {
        args.extend(["--arm", arm.id]);
    }
    serde_json::from_str(&kpi_ok(&args)).unwrap()
}

fn statistic(dir: &Path, command: &str, args: &[&str], out: &str) -> Value {
    let out = dir.join(out);
    let mut full = vec![command, "--run-dir", path_str(dir)];
    full.extend_from_slice(args);
    full.extend(["--out", path_str(&out)]);
    kpi_ok(&full);
    json_file(&out)
}

const COMPARE: [&str; 6] = [
    "--candidate",
    "gen",
    "--candidate",
    "copy",
    "--baseline",
    "twin",
];
const PRIMARY: [&str; 6] = ["--arm", "twin", "--arm", "copy", "--arm", "gen"];

fn checks(comparison: &Value) -> Vec<String> {
    let guards = comparison["decision"]["guards"].as_array().unwrap();
    guards
        .iter()
        .map(|g| {
            format!(
                "{}={}",
                g["name"].as_str().unwrap(),
                g["check"].as_str().unwrap()
            )
        })
        .collect()
}

fn arm<'a>(receipt: &'a Value, id: &str) -> &'a Value {
    receipt["arms"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["arm_id"] == id)
        .unwrap()
}

#[test]
fn guards_turn_unevaluated_guards_into_decisions() {
    let dir = run_dir("guards-cli");
    let before = statistic(&dir, "compare", &COMPARE, "before.json");
    assert_eq!(before["comparisons"][0]["decision"]["outcome"], "HOLD");
    for name in [
        "realism_coverage_delta",
        "privacy_coverage_delta",
        "exact_copy_excess",
    ] {
        assert!(
            checks(&before["comparisons"][0]).contains(&format!("{name}=not_evaluated")),
            "{name}"
        );
    }
    assert!(arm(&before, "copy").get("privacy_coverage").is_none());

    let summary = guards(&dir, "2");
    let copy = &summary["arms"]["copy"];
    assert_eq!(
        (
            copy["entries"].as_u64(),
            copy["pooled_exact_copy_excess"].as_f64()
        ),
        (Some(18), Some(1.0))
    );
    assert_eq!(copy["privacy_pass_share_complete"], 0.0);
    assert_eq!(summary["arms"]["twin"]["pooled_exact_copy_excess"], 0.0);
    assert_eq!(summary["arms"]["gen"]["privacy_pass_share_complete"], 1.0);
    let gen_summary = json_file(&dir.join("score/gen/guards-summary.json"));
    assert_eq!(gen_summary["by_status"], json!({"failed": 3, "ok": 15}));
    assert_eq!(
        gen_summary["privacy"]["assessments"], 5,
        "one assessment per complete file"
    );
    // One excess per complete file; the failed file has none.
    let files = gen_summary["file_exact_copy_excess"].as_object().unwrap();
    assert_eq!(files.len(), 5);
    assert!(!files.contains_key(&format!("{}:s0:m1", cell_id("tc"))));
    assert!(files.values().all(|v| v.as_f64() == Some(0.0)));
    let lines = json_lines(&dir.join("score/gen/guards.jsonl"));
    assert_eq!(lines.len(), 18);
    let failed_line = lines.iter().find(|l| l["status"] == "failed").unwrap();
    assert_eq!(
        (
            failed_line["privacy_pass"].as_bool(),
            failed_line["synthetic_rows"].as_u64()
        ),
        (Some(false), None)
    );

    // Same bytes for another thread count.
    let first: Vec<[Vec<u8>; 2]> = ARMS.iter().map(|a| guard_files(&dir, a.id)).collect();
    guards(&dir, "1");
    assert_eq!(
        first,
        ARMS.iter()
            .map(|a| guard_files(&dir, a.id))
            .collect::<Vec<_>>()
    );

    let primary = statistic(&dir, "primary", &PRIMARY, "primary.json");
    let guarded = |id: &str| {
        let a = arm(&primary, id);
        (
            a["realism_coverage"].as_f64(),
            a["privacy_coverage"].as_f64(),
            a["exact_copy_excess"].as_f64(),
        )
    };
    assert_eq!(guarded("twin"), (Some(1.0), Some(1.0), Some(0.0)));
    assert_eq!(guarded("copy"), (Some(1.0), Some(0.0), Some(1.0)));
    assert_eq!(guarded("gen"), (Some(1.0), Some(1.0), Some(0.0)));
    let [lines_bytes, summary_bytes] = guard_files(&dir, "copy");
    assert_eq!(
        primary["inputs"]["guards_sha256"]["copy"],
        sha256(&lines_bytes)
    );
    assert_eq!(
        primary["inputs"]["guards_summary_sha256"]["copy"],
        sha256(&summary_bytes)
    );

    let after = statistic(&dir, "compare", &COMPARE, "after.json");
    let (gen, copy) = (&after["comparisons"][0], &after["comparisons"][1]);
    assert_eq!(gen["decision"]["outcome"], "PROMOTE", "{:?}", checks(gen));
    assert!(
        checks(gen).iter().all(|g| g.ends_with("=pass")),
        "{:?}",
        checks(gen)
    );
    assert_eq!(copy["decision"]["outcome"], "HOLD");
    let copy_checks = checks(copy);
    for expected in [
        "realism_coverage_delta=pass",
        "privacy_coverage_delta=fail",
        "exact_copy_excess=fail",
    ] {
        assert!(
            copy_checks.contains(&expected.to_string()),
            "{copy_checks:?}"
        );
    }
    let copies = copy["decision"]["guards"]
        .as_array()
        .unwrap()
        .iter()
        .find(|g| g["name"] == "exact_copy_excess");
    assert_eq!(copies.unwrap()["value"], 1.0);
}

#[test]
fn guard_files_that_do_not_cover_the_rows_are_refused_without_values() {
    let dir = run_dir("guards-cli-refusals");
    guards(&dir, "1");
    // Drop the last guard line and keep the summary consistent: the row is left without its line.
    let at = dir.join("score/copy");
    let text = std::fs::read_to_string(at.join("guards.jsonl")).unwrap();
    let kept: Vec<&str> = text.lines().take(17).collect();
    let bytes = kept.join("\n") + "\n";
    std::fs::write(at.join("guards.jsonl"), &bytes).unwrap();
    let mut summary = json_file(&at.join("guards-summary.json"));
    summary["guards_sha256"] = sha256(bytes.as_bytes()).into();
    std::fs::write(
        at.join("guards-summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();
    let out = dir.join("r.json");
    let err = kpi_err(&[
        "primary",
        "--run-dir",
        path_str(&dir),
        "--arm",
        "copy",
        "--out",
        path_str(&out),
    ]);
    assert!(
        err.contains("score/copy/rows.jsonl: line 18 has no line in score/copy/guards.jsonl"),
        "{err}"
    );
    assert!(!err.contains("1.1") && !err.contains("0.5"), "{err}");
    assert!(!out.exists(), "a refused command writes no receipt");

    // Rows without entry ids cannot be joined.
    let rows = dir.join("score/twin/rows.jsonl");
    let stripped: Vec<String> = json_lines(&rows)
        .into_iter()
        .map(|mut row| {
            row.as_object_mut().unwrap().remove("ledger_id");
            row.to_string()
        })
        .collect();
    write_jsonl(&rows, &stripped);
    let err = kpi_err(&["guards", "--run-dir", path_str(&dir), "--arm", "twin"]);
    assert!(
        err.contains("score/twin/rows.jsonl: line 1 has no ledger_id"),
        "{err}"
    );
    let err = kpi_err(&["guards", "--run-dir", path_str(&dir), "--arm", "../twin"]);
    assert!(err.contains("--arm must be"), "{err}");
}

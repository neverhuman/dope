//! Black-box test of `dope_kpi primary`, `compare`, `null` and
//! `render-readme` on the static run directory `tests/fixtures/kpi-stat`
//! (six cells, three auditors, multipliers 4 and 1, four arms). Every cell
//! has `R_B = 1` except `a3` (`R_B = 0.95`), so the expected values below are
//! computed by hand from the rows:
//!
//! * `base`: families famR1 (a1 -0.3, a2 -0.4), famR2 (a3 -0.1), famB1
//!   (b1: linear failed, -1), famC1 (c1 -0.3): TRG (-0.225 - 1 - 0.3) / 3.
//! * `cand`: famR1 (-0.15, -0.2), famR2 (+0.05, `R_S` 1.4 clipped to 1.25),
//!   famB1 -0.1, famC1 -0.15: TRG -0.3125 / 3, paired delta 1.2125 / 3.
//! * `twin_c`: a native control that cannot exist on c1 (multiclass leaves its
//!   statistic and every comparison it is in): TRG (-0.05 - 0.1) / 2.
//! * `noise`: retention -0.5 everywhere.

mod kpi_support;

use kpi_support::{json_file, json_lines, kpi_err, kpi_ok, path_str, scratch};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kpi-stat")
}

fn close(got: &Value, want: f64) {
    let got = got.as_f64().unwrap_or(f64::NAN);
    assert!((got - want).abs() < 1e-9, "{got} != {want}");
}

/// Run `command` on the fixture with `args` and `--out`; returns (receipt bytes, stdout summary).
fn receipt(dir: &Path, name: &str, command: &str, args: &[&str]) -> (Vec<u8>, Value) {
    let out = dir.join(name);
    let run_dir = fixture();
    let mut full = vec![command, "--run-dir", path_str(&run_dir)];
    full.extend_from_slice(args);
    full.extend(["--out", path_str(&out)]);
    let summary: Value = serde_json::from_str(&kpi_ok(&full)).unwrap();
    (std::fs::read(&out).unwrap(), summary)
}

fn arm<'a>(receipt: &'a Value, id: &str) -> &'a Value {
    let arms = receipt["arms"].as_array().unwrap();
    arms.iter().find(|a| a["arm_id"] == id).unwrap()
}

const ARMS: [&str; 8] = [
    "--arm", "base", "--arm", "cand", "--arm", "twin_c", "--arm", "noise",
];

#[test]
fn audit_reconstructs_partitioned_comparisons_and_rejects_altered_receipts() {
    let dir = scratch("audit-receipt");
    let dev = fixture().join("dev-tables.txt");
    let (bytes, _) = receipt(
        &dir,
        "compare.json",
        "compare",
        &[
            "--candidate",
            "cand",
            "--seed-group",
            "two=cand,noise",
            "--baseline",
            "base",
            "--panel",
            "dev",
            "--dev-tables",
            path_str(&dev),
        ],
    );
    let run_dir = fixture();
    let path = dir.join("compare.json");
    let args = [
        "audit",
        "--run-dir",
        path_str(&run_dir),
        "--receipt",
        path_str(&path),
        "--dev-tables",
        path_str(&dev),
    ];
    let audit: Value = serde_json::from_str(&kpi_ok(&args)).unwrap();
    assert_eq!(audit["schema"], "dope.kpi-audit/v1");
    assert_eq!(audit["entries"]["base"], 36);
    assert_eq!(audit["comparisons"]["cand"][0]["cells"], 3);
    close(
        &audit["arms"]["base"][0]["failed_at_floor"],
        (-0.35 - 2.0) / 2.0,
    );
    close(
        &audit["comparisons"]["cand"][0]["failed_at_floor"],
        (0.175 + 1.9) / 2.0,
    );
    assert_eq!(audit["comparisons"]["two"][1]["multiplier"], 1);

    for (pointer, value, reason) in [
        (
            "/inputs/realref_sha256",
            Value::from("wrong"),
            "input identities",
        ),
        (
            "/arms/0/bytes/with_packet",
            Value::from(0),
            "byte accounting",
        ),
        (
            "/comparisons/0/cells",
            Value::from(10),
            "comparison denominator",
        ),
        ("/comparisons/0/delta", Value::from(123.0), "paired point"),
    ] {
        let mut modified: Value = serde_json::from_slice(&bytes).unwrap();
        *modified.pointer_mut(pointer).unwrap() = value;
        std::fs::write(&path, serde_json::to_vec(&modified).unwrap()).unwrap();
        assert!(kpi_err(&args).contains(reason));
    }
}

#[test]
fn audit_factorial_uses_the_same_applicable_cells_for_all_four_arms() {
    let dir = scratch("audit-factorial");
    receipt(&dir, "primary.json", "primary", &ARMS);
    let source = fixture();
    let path = dir.join("primary.json");
    let output = kpi_ok(&[
        "audit",
        "--run-dir",
        path_str(&source),
        "--receipt",
        path_str(&path),
        "--factorial",
        "base",
        "cand",
        "noise",
        "twin_c",
    ]);
    let audit: Value = serde_json::from_str(&output).unwrap();
    assert_eq!(audit["factorial"]["packet_packet"], "base");
    assert_eq!(audit["factorial"]["real_tree"], "twin_c");
    let headline = &audit["factorial"]["readouts"][0];
    assert_eq!(headline["multiplier"], 4);
    assert_eq!(headline["cells"], 4);
    assert!(headline["interaction"]["tasks"]["multiclass"].is_null());
    close(
        &headline["x_main_effect"]["tasks"]["regression"]["value"],
        0.79375,
    );
    close(
        &headline["y_main_effect"]["tasks"]["regression"]["value"],
        -0.61875,
    );
    close(
        &headline["interaction"]["tasks"]["regression"]["value"],
        1.2625,
    );
    assert_eq!(audit["factorial"]["readouts"][1]["multiplier"], 1);
}

#[test]
fn primary_reports_hand_computed_values_and_repeats_byte_for_byte() {
    let dir = scratch("stat-primary");
    let (bytes, summary) = receipt(&dir, "primary.json", "primary", &ARMS);
    let (again, _) = receipt(&dir, "primary-again.json", "primary", &ARMS);
    assert_eq!(bytes, again, "same inputs, same receipt bytes");
    let r: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(r["schema"], "dope.kpi-receipt/v2");
    assert_eq!(r["contract"]["id"], "dope-kpi/v2");
    assert_eq!(r["contract"]["sha256"].as_str().unwrap().len(), 64);
    assert_eq!(r["inputs"]["rows_sha256"].as_object().unwrap().len(), 4);
    assert_eq!(
        (
            r["bootstrap"]["draws"].as_u64(),
            r["bootstrap"]["lower_rank"].as_u64()
        ),
        (Some(10_000), Some(249))
    );
    close(&summary["arms"]["base"], -1.525 / 3.0);
    let base = arm(&r, "base");
    close(&base["primary"]["trg"], -1.525 / 3.0);
    assert_eq!(
        (
            base["primary"]["cells"].as_u64(),
            base["primary"]["failed_cells"].as_u64()
        ),
        (Some(5), Some(1))
    );
    // Matched multiplier: every scored retention 0.1 lower.
    close(&base["matched"]["trg"], -1.725 / 3.0);
    close(&base["primary"]["tasks"][0]["value"], -0.225);
    assert_eq!(base["primary"]["tasks"][0]["families"], 2);
    close(&base["bytes"]["median"], 1500.0);
    close(&base["c2st_median"], 0.7);
    close(&base["median_retention"][0]["median"], 0.75);
    close(&arm(&r, "cand")["primary"]["trg"], -0.3125 / 3.0);
    let twin = arm(&r, "twin_c");
    close(&twin["primary"]["trg"], -0.075);
    assert_eq!(twin["primary"]["not_applicable_cells"], 1);
    assert_eq!(twin["primary"]["families"], 3);
    assert_eq!(twin["artifact_class"], "control");
    assert!(twin["bytes"]["median"].is_null());
    close(&arm(&r, "noise")["primary"]["trg"], -4.475 / 3.0);
    let text = String::from_utf8(bytes).unwrap();
    assert!(
        !text.contains(path_str(&fixture())),
        "receipts hold no path"
    );
}

#[test]
fn compare_promotes_holds_and_rejects_with_holm_over_the_receipt() {
    let dir = scratch("stat-compare");
    let args = [
        "--candidate",
        "cand",
        "--candidate",
        "twin_c",
        "--candidate",
        "noise",
        "--baseline",
        "base",
    ];
    let (bytes, summary) = receipt(&dir, "compare.json", "compare", &args);
    assert_eq!(summary["outcomes"]["cand"], "PROMOTE");
    assert_eq!(summary["outcomes"]["twin_c"], "HOLD");
    assert_eq!(summary["outcomes"]["noise"], "REJECT");
    let r: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(r["noise_sd"]["source"], "default");
    let c = &r["comparisons"];
    close(&c[0]["delta"], 1.2125 / 3.0);
    // Regression draws mix famR1 (+0.175) and famR2 (+0.15); one family elsewhere.
    close(&c[0]["ci_low"], 1.2 / 3.0);
    close(&c[0]["ci_high"], 1.225 / 3.0);
    close(&c[0]["p_one_sided"], 1.0 / 10_001.0);
    close(&c[0]["p_holm"], 3.0 / 10_001.0);
    close(&c[1]["p_holm"], 3.0 / 10_001.0);
    close(&c[2]["p_holm"], 1.0);
    // twin is paired on the four cells it can exist on.
    assert_eq!(
        (c[1]["cells"].as_u64(), c[1]["families"].as_u64()),
        (Some(4), Some(3))
    );
    close(&c[1]["delta"], (0.175 + 0.9) / 2.0);
    let guards = |i: usize| -> Vec<String> {
        let items = c[i]["decision"]["guards"].as_array().unwrap();
        items
            .iter()
            .map(|g| {
                format!(
                    "{}={}",
                    g["name"].as_str().unwrap(),
                    g["check"].as_str().unwrap()
                )
            })
            .collect()
    };
    assert!(
        guards(0).iter().all(|g| g.ends_with("=pass")),
        "{:?}",
        guards(0)
    );
    assert!(guards(1).contains(&"bytes=not_evaluated".to_string()));
    assert!(guards(2).contains(&"task_ci_low=fail".to_string()));
    close(&c[0]["matched"]["delta"], 1.1125 / 3.0);
    assert_eq!(r["arms"].as_array().unwrap().len(), 4);
    // A noise SD above the delta holds the candidate.
    let (bytes, summary) = receipt(
        &dir,
        "noisy.json",
        "compare",
        &[
            "--candidate",
            "cand",
            "--baseline",
            "base",
            "--noise-sd",
            "0.5",
        ],
    );
    assert_eq!(summary["outcomes"]["cand"], "HOLD");
    let r: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        (
            r["noise_sd"]["source"].as_str(),
            r["noise_sd"]["value"].as_f64()
        ),
        (Some("input"), Some(0.5))
    );
}

#[test]
fn dev_and_confirm_split_the_families() {
    let dir = scratch("stat-panels");
    let list = fixture().join("dev-tables.txt");
    let dev_args = [
        "--arm",
        "base",
        "--panel",
        "dev",
        "--dev-tables",
        path_str(&list),
    ];
    let (dev, _) = receipt(&dir, "dev.json", "primary", &dev_args);
    let dev: Value = serde_json::from_slice(&dev).unwrap();
    // a1 maps to famR1 through the schedule, e9 to famB1 through families.jsonl.
    close(&arm(&dev, "base")["primary"]["trg"], (-0.35 - 1.0) / 2.0);
    let p = &dev["panel"];
    assert_eq!(
        (
            p["id"].as_str(),
            p["dev_tables"].as_u64(),
            p["dev_tables_scheduled"].as_u64()
        ),
        (Some("dev"), Some(2), Some(1))
    );
    assert_eq!(
        (p["dev_families"].as_u64(), p["families"].as_u64()),
        (Some(2), Some(2))
    );
    let confirm_args = [
        "--arm",
        "base",
        "--panel",
        "confirm",
        "--dev-tables",
        path_str(&list),
    ];
    let (confirm, _) = receipt(&dir, "confirm.json", "primary", &confirm_args);
    let confirm: Value = serde_json::from_slice(&confirm).unwrap();
    close(&arm(&confirm, "base")["primary"]["trg"], (-0.1 - 0.3) / 2.0);
    assert_eq!(confirm["panel"]["families"], 3);
}

#[test]
fn null_and_readme_block_are_deterministic() {
    let dir = scratch("stat-null-render");
    let lines = dir.join("replicates.jsonl");
    let null_args = [
        "--candidate",
        "cand",
        "--baseline",
        "base",
        "--replicates-out",
        path_str(&lines),
    ];
    let (null_bytes, summary) = receipt(&dir, "null.json", "null", &null_args);
    assert_eq!(summary["replicates"], 2000);
    let n = json_file(&dir.join("null.json"))["null"].clone();
    assert_eq!(
        (n["families"].as_u64(), n["seed"].as_u64()),
        (Some(4), Some(20_261_124))
    );
    assert_eq!(n["observed"]["decision"]["outcome"], "PROMOTE");
    assert!(
        n["false_promote"]["count"].as_u64().unwrap()
            <= n["flip_guard_rule"]["count"].as_u64().unwrap()
    );
    assert!(
        n["flip_guard_rule"]["count"].as_u64().unwrap()
            <= n["statistical_pass"]["count"].as_u64().unwrap()
    );
    assert_eq!(json_lines(&lines).len(), 2000);
    assert_eq!(n["replicates_sha256"].as_str().unwrap().len(), 64);
    let (again, _) = receipt(&dir, "null-again.json", "null", &null_args);
    assert_eq!(null_bytes, again);
    receipt(&dir, "primary.json", "primary", &ARMS);
    receipt(
        &dir,
        "compare.json",
        "compare",
        &[
            "--candidate",
            "cand",
            "--candidate",
            "twin_c",
            "--baseline",
            "base",
        ],
    );
    let names = ["primary.json", "compare.json", "null.json"];
    let paths: Vec<PathBuf> = names.iter().map(|n| dir.join(n)).collect();
    let mut args = vec!["render-readme"];
    for path in &paths {
        args.extend(["--receipt", path_str(path)]);
    }
    let block = kpi_ok(&args);
    assert_eq!(
        block,
        kpi_ok(&args),
        "the block is a pure function of the receipts"
    );
    let lines: Vec<&str> = block.lines().collect();
    assert_eq!(lines.first(), Some(&"<!-- BEGIN DOPE1 SCOREBOARD -->"));
    assert_eq!(lines.last(), Some(&"<!-- END DOPE1 SCOREBOARD -->"));
    assert!(lines.contains(
        &"| `base` | source_free | 1500 / 1500 / 1.000 | -0.508 [-0.550, -0.467] | -0.225 (2) | -1.000 (1) | -0.300 (1) | 0.750 / 0.750 / 0.800 | 0.700 | 0.833 / 0.833 / 0.000 (0.000) | 1 / 5 |"
    ), "{block}");
    assert!(lines.contains(&"| `twin_c` | control | - | -0.075 [-0.075, -0.075] | -0.050 (2) | -0.100 (1) | - | 0.950 / 0.900 / 0.925 | 0.550 | - / - / - | 0 / 4 (1 n/a) |"), "{block}");
    // Four families: no sign-flip PROMOTE rate can be below 1/16, which the
    // diagnostic shows beside the contract's decision (still PROMOTE).
    assert!(lines.contains(&"| `cand` vs `base` | +0.404 [+0.400, +0.408] | <0.001 (2) | tasks +0.150 pass; mlp +0.163 pass; realism +0.167 pass; privacy +0.167 pass; copies 0.000 (0.000) pass; bytes 1800 pass | 0.062 (> 0.050) | **PROMOTE** |"), "{block}");
    assert!(
        block.contains("Sign-flip null, `cand` vs `base`, 2000 replicates (Wilson 95%): false PROMOTE with bytes, copies and unevaluated guards passing 0.062 [0.052, 0.073] (ABOVE the 0.050 maximum); full rule 0.062"),
        "{block}"
    );
    let footer = lines[lines.len() - 2];
    assert!(
        footer.starts_with("Contract `dope-kpi/v2` (embedded, candidate) sha256 `")
            && footer.contains("panel `all`"),
        "{footer}"
    );
    for path in &paths {
        let sha = format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()));
        assert!(footer.contains(&sha), "{footer}");
    }
    assert!(!block.contains("-0.000"));

    args.extend(["--ranked", "--label", "cand=Candidate"]);
    let ranked = kpi_ok(&args);
    let (ranking, full) = ranked.split_once("<details>").unwrap();
    assert!(ranking.contains("| Transfer Score ↑ |"), "{ranking}");
    let rows: Vec<_> = ranking
        .lines()
        .filter(|line| line.starts_with("| 🟩") || line.starts_with("| 🟦"))
        .collect();
    assert_eq!(rows.len(), 2, "{ranking}");
    assert!(rows[0].starts_with("| 🟩 1 | Candidate |"), "{ranking}");
    assert!(rows[1].starts_with("| 🟦 2 | `base` |"), "{ranking}");
    assert!(!ranking.contains("`twin_c`"), "{ranking}");
    assert!(full.contains("| `twin_c` | control |"), "{full}");
    assert!(full.contains("| `cand` vs `base` | +0.404"), "{full}");
}

#[test]
fn refusals_exit_non_zero_without_echoing_values() {
    let dir = scratch("stat-refusals");
    let run_dir = fixture();
    let run = path_str(&run_dir);
    let out = dir.join("r.json");
    let out = path_str(&out);
    let list = fixture().join("dev-tables.txt");
    let cases: [(Vec<&str>, &str); 6] = [
        (
            vec![
                "compare",
                "--run-dir",
                run,
                "--baseline",
                "base",
                "--out",
                out,
            ],
            "--candidate or --seed-group",
        ),
        (
            vec![
                "primary",
                "--run-dir",
                run,
                "--arm",
                "base",
                "--panel",
                "dev",
                "--out",
                out,
            ],
            "--dev-tables",
        ),
        (
            vec![
                "primary",
                "--run-dir",
                run,
                "--arm",
                "base",
                "--dev-tables",
                path_str(&list),
                "--out",
                out,
            ],
            "--panel dev or confirm",
        ),
        (
            vec![
                "null",
                "--run-dir",
                run,
                "--candidate",
                "cand",
                "--baseline",
                "base",
                "--replicates",
                "10",
                "--out",
                out,
            ],
            "at least 2000",
        ),
        (
            vec![
                "primary",
                "--run-dir",
                run,
                "--arm",
                "missing",
                "--out",
                out,
            ],
            "missing",
        ),
        (
            vec![
                "compare",
                "--run-dir",
                run,
                "--seed-group",
                "g=cand,cand",
                "--baseline",
                "base",
                "--out",
                out,
            ],
            "twice",
        ),
    ];
    for (args, expected) in &cases {
        let err = kpi_err(args);
        assert!(err.contains(expected), "{args:?}: {err}");
        assert!(!err.contains("0.95") && !err.contains("1500"), "{err}");
    }
    assert!(
        !Path::new(out).exists(),
        "a refused command writes no receipt"
    );
    // Receipts of different panels do not render together.
    let (_, _) = receipt(&dir, "all.json", "primary", &["--arm", "base"]);
    let (_, _) = receipt(
        &dir,
        "dev.json",
        "primary",
        &[
            "--arm",
            "base",
            "--panel",
            "dev",
            "--dev-tables",
            path_str(&list),
        ],
    );
    let (all, dev) = (dir.join("all.json"), dir.join("dev.json"));
    let err = kpi_err(&[
        "render-readme",
        "--receipt",
        path_str(&all),
        "--receipt",
        path_str(&dev),
    ]);
    assert!(err.contains("panel"), "{err}");
}

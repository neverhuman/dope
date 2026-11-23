//! Black-box test of `dope_kpi render-hero` on `primary` receipts of the
//! static run directory `tests/fixtures/kpi-stat`: `base` (1,500-byte
//! packets) and `cand` (1,800) are points, the control `twin_c` is a
//! reference line and `noise` (a control without bytes) stays unplotted.

mod kpi_support;

use kpi_support::{json_file, kpi_err, kpi_ok, path_str, scratch};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kpi-stat")
}

fn primary(out: &Path, arms: &[&str], panel: &[&str]) {
    let run_dir = fixture();
    let mut args = vec!["primary", "--run-dir", path_str(&run_dir)];
    for arm in arms {
        args.extend(["--arm", *arm]);
    }
    args.extend_from_slice(panel);
    args.extend(["--out", path_str(out)]);
    kpi_ok(&args);
}

fn sha8(path: &Path) -> String {
    let digest = Sha256::digest(std::fs::read(path).unwrap());
    digest.iter().take(4).map(|b| format!("{b:02x}")).collect()
}

/// `render-hero` arguments for `receipts` and `out`, then `extra`.
fn hero_args<'a>(receipts: &'a [PathBuf], out: &'a Path, extra: &[&'a str]) -> Vec<&'a str> {
    let mut args = vec!["render-hero"];
    for receipt in receipts {
        args.extend(["--receipt", path_str(receipt)]);
    }
    args.extend(["--out", path_str(out)]);
    args.extend_from_slice(extra);
    args
}

const ROLES: [&str; 12] = [
    "--highlight",
    "cand",
    "--label",
    "cand=Candidate",
    "--label",
    "base=Baseline",
    "--label",
    "twin_c=twin sample (C)",
    "--reference",
    "twin_c",
    "--note",
    "Fixture: {families} families, {cells} cells, panel {panel}.",
];

#[test]
fn render_hero_draws_a_primary_receipt_in_both_themes() {
    let dir = scratch("hero-themes");
    let receipt = dir.join("primary.json");
    primary(&receipt, &["base", "cand", "twin_c", "noise"], &[]);
    let r = json_file(&receipt);
    let cand = r["arms"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["arm_id"] == "cand")
        .unwrap();
    let receipts = vec![receipt.clone()];
    let mut svgs = Vec::new();
    for theme in ["light", "dark"] {
        let out = dir.join(format!("hero-{theme}.svg"));
        let mut extra = vec!["--theme", theme];
        extra.extend_from_slice(&ROLES);
        let summary: Value =
            serde_json::from_str(&kpi_ok(&hero_args(&receipts, &out, &extra))).unwrap();
        let svg = std::fs::read_to_string(&out).unwrap();
        assert_eq!(summary["command"], "render-hero");
        assert_eq!(summary["theme"], theme);
        assert_eq!(
            (summary["points"].as_u64(), summary["references"].as_u64()),
            (Some(2), Some(1))
        );
        assert_eq!(summary["unplotted"], serde_json::json!(["noise"]));
        let digest: String = Sha256::digest(svg.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(summary["svg_sha256"], digest.as_str());
        // Same receipt and flags, same bytes.
        kpi_ok(&hero_args(&receipts, &out, &extra));
        assert_eq!(std::fs::read_to_string(&out).unwrap(), svg);
        assert!(svg.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg "));
        assert!(svg.trim_end().ends_with("</svg>"));
        assert_eq!(svg.matches("class=\"pt ").count(), 2);
        assert_eq!(svg.matches("class=\"pt hl\"").count(), 1);
        assert_eq!(svg.matches("class=\"ref\"").count(), 1);
        assert!(svg.contains(">Candidate \u{b7} TRG "));
        assert!(svg.contains(">Baseline</text>") && svg.contains(">twin sample (C)</text>"));
        let note = format!(
            "Fixture: {} families, {} cells, panel all.",
            cand["primary"]["families"], cand["primary"]["cells"]
        );
        assert!(svg.contains(&note), "note filled from the receipt");
        let contract = &r["contract"]["sha256"].as_str().unwrap()[..8];
        let footer = format!(
            ">contract {contract} \u{b7} receipt {}</text>",
            sha8(&receipt)
        );
        assert!(svg.contains(&footer));
        assert!(!svg.contains(path_str(&dir)) && !svg.contains(path_str(&fixture())));
        svgs.push(svg);
    }
    assert_ne!(svgs[0], svgs[1], "the themes differ");
    assert!(svgs[1].contains("fill=\"#1a1a19\"") && !svgs[1].contains("#fcfcfb"));
}

#[test]
fn render_hero_merges_receipts_of_one_run_and_refuses_bad_roles() {
    let dir = scratch("hero-roles");
    let (one, two) = (dir.join("one.json"), dir.join("two.json"));
    primary(&one, &["base", "cand"], &[]);
    primary(&two, &["cand", "twin_c", "noise"], &[]);
    let receipts = vec![one.clone(), two.clone()];
    let out = dir.join("hero.svg");
    let mut extra = vec!["--theme", "light"];
    extra.extend_from_slice(&ROLES);
    let summary: Value =
        serde_json::from_str(&kpi_ok(&hero_args(&receipts, &out, &extra))).unwrap();
    assert_eq!(summary["points"], 2, "cand is drawn once");
    let svg = std::fs::read_to_string(&out).unwrap();
    assert!(svg.contains(&format!("receipts {}, {}</text>", sha8(&one), sha8(&two))));
    // Without --previous and --highlight every point is a plain method.
    let plain = dir.join("plain.svg");
    kpi_ok(&hero_args(&receipts, &plain, &["--theme", "dark"]));
    let plain = std::fs::read_to_string(&plain).unwrap();
    assert_eq!(plain.matches("class=\"pt ot\"").count(), 2);
    assert!(plain.contains("Panel all: "), "default note");

    let refused = |extra: &[&str], message: &str| {
        let mut args = vec!["--theme", "light"];
        args.extend_from_slice(extra);
        let err = kpi_err(&hero_args(&receipts, &out, &args));
        assert!(err.contains(message), "{err}");
    };
    refused(
        &["--highlight", "noise"],
        "--highlight arm noise needs a byte median and a TRG",
    );
    refused(
        &["--highlight", "ghost"],
        "--highlight names arm ghost, which no receipt holds",
    );
    refused(&["--reference", "ghost"], "--reference names arm ghost");
    refused(
        &["--highlight", "cand", "--reference", "cand"],
        "arm cand is given more than one role",
    );
    refused(&["--label", "cand"], "--label takes ARM=Text");
    refused(&["--label", "ghost=x"], "--label names arm ghost");
    refused(
        &["--label", "cand=a", "--label", "cand=b"],
        "--label names arm cand twice",
    );

    // Receipts of different panels do not draw together.
    let list = fixture().join("dev-tables.txt");
    let dev = dir.join("dev.json");
    primary(
        &dev,
        &["base"],
        &["--panel", "dev", "--dev-tables", path_str(&list)],
    );
    let mixed = vec![one.clone(), dev];
    let err = kpi_err(&hero_args(&mixed, &out, &["--theme", "light"]));
    assert!(
        err.contains("receipts differ in contract, schedule, real reference, panel or bootstrap")
    );
}

//! `dope fit --format auto` and `dope sample` on DPA1 files, end to end on synthetic CSVs.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_dope"))
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap()
}

fn ok(dir: &Path, args: &[&str]) -> Output {
    let out = run(dir, args);
    assert!(
        out.status.success(),
        "dope {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

/// A failure that is a clean `error:` exit, not a crash.
fn refused(dir: &Path, args: &[&str]) -> String {
    let out = run(dir, args);
    assert_eq!(out.status.code(), Some(1), "dope {args:?} did not exit 1");
    let err = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        err.starts_with("error: ") && !err.contains("panicked"),
        "{err}"
    );
    err
}

fn workdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("dope-auto-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Deterministic pseudo-random numbers in [0, 1) (no fixtures, no real data).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `p` numeric features `c0..`, target last, as the KPI harness writes them.
fn write_wide(dir: &Path, p: usize, n: usize) {
    let mut rng = Lcg(p as u64);
    let mut csv: String = (0..p).map(|j| format!("c{j},")).collect::<String>() + "target\n";
    for _ in 0..n {
        let x: Vec<f64> = (0..p).map(|_| rng.next() * 4.0 - 2.0).collect();
        let y = x[0] + 0.5 * x[1] - x[2] + 0.2 * rng.next();
        for v in &x {
            csv += &format!("{v:.4},");
        }
        csv += &format!("{y:.4}\n");
    }
    std::fs::write(dir.join("wide.csv"), csv).unwrap();
}

/// Mixed types with the class label in the middle of the header.
fn write_mixed(dir: &Path, n: usize) {
    let mut rng = Lcg(42);
    let mut csv = String::from("age,city,label,spend\n");
    for _ in 0..n {
        let age = 20 + (rng.next() * 50.0) as u32;
        let city = ["paris", "lima", "oslo"][(rng.next() * 3.0) as usize];
        let spend = rng.next() * 3.0 - 1.5;
        let z = 0.08 * (age as f64 - 45.0) + spend + if city == "oslo" { 1.0 } else { -0.4 };
        let label = if z + rng.next() - 0.5 > 0.0 {
            "yes"
        } else {
            "no"
        };
        csv += &format!("{age},{city},{label},{spend:.3}\n");
    }
    std::fs::write(dir.join("mixed.csv"), csv).unwrap();
}

fn lines(out: &Output) -> Vec<String> {
    String::from_utf8(out.stdout.clone())
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn zero_feature_tables_exit_cleanly_and_tiny_tables_fit() {
    let dir = workdir("zero");
    let target_only: String = (0..120).map(|i| format!("{}\n", i % 7)).collect();
    std::fs::write(dir.join("t.csv"), format!("y\n{target_only}")).unwrap();
    let ids: String = (0..300).map(|i| format!("row-{i},{}\n", i % 5)).collect();
    std::fs::write(dir.join("ids.csv"), format!("id,y\n{ids}")).unwrap();
    for csv in ["t.csv", "ids.csv"] {
        let err = refused(
            &dir,
            &["fit", csv, "-t", "y", "--format", "auto", "-o", "z.dope"],
        );
        assert!(err.contains("no usable feature columns"), "{err}");
        assert!(!err.contains("row-"), "{err}");
    }
    // One feature and 40 rows: too few rows for any dependence regime (allocate.rs panic).
    let mut rng = Lcg(5);
    let mut small = String::from("a,y\n");
    for _ in 0..40 {
        let a = rng.next();
        small += &format!("{a:.4},{:.4}\n", 2.0 * a + rng.next());
    }
    std::fs::write(dir.join("small.csv"), small).unwrap();
    ok(
        &dir,
        &[
            "fit",
            "small.csv",
            "-t",
            "y",
            "--size",
            "small",
            "-o",
            "f1.dope",
            "-q",
        ],
    );
    ok(
        &dir,
        &[
            "fit",
            "small.csv",
            "-t",
            "y",
            "--format",
            "auto",
            "-o",
            "a.dope",
            "-q",
        ],
    );
    let out = ok(&dir, &["sample", "a.dope", "-n", "30", "--seed", "1"]);
    assert_eq!(lines(&out).len(), 31);
    assert_eq!(lines(&out)[0], "a,y");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn wide_tables_fit_in_2kib_and_sample_target_last() {
    let dir = workdir("wide");
    write_wide(&dir, 200, 300);
    let fit = ok(
        &dir,
        &[
            "fit", "wide.csv", "--target", "target", "--format", "auto", "--seed", "7", "--json",
            "-o", "w.dope",
        ],
    );
    let report: serde_json::Value = serde_json::from_slice(&fit.stdout).unwrap();
    assert_eq!(report["encoder"], "v7");
    assert_eq!(report["format1"]["fitted"], false);
    let size = std::fs::metadata(dir.join("w.dope")).unwrap().len();
    assert!(size <= 2048, "artifact {size} bytes");
    assert_eq!(report["bytes"].as_u64(), Some(size));
    let summary = String::from_utf8_lossy(&fit.stderr).to_string();
    assert!(summary.starts_with("auto: chose v7 (") && summary.contains("format1 failed (budget)"));
    assert!(
        !summary.contains("c0") && !summary.contains("target"),
        "{summary}"
    );
    let out = ok(
        &dir,
        &["sample", "w.dope", "-n", "64", "--seed", "3", "-o", "s.csv"],
    );
    assert!(out.stdout.is_empty() || String::from_utf8_lossy(&out.stdout).contains("V7 packet"));
    let text = std::fs::read_to_string(dir.join("s.csv")).unwrap();
    let rows: Vec<&str> = text.lines().collect();
    assert_eq!(rows.len(), 65);
    let header: Vec<&str> = rows[0].split(',').collect();
    assert_eq!(header.len(), 201);
    assert_eq!(
        (header[0], header[199], header[200]),
        ("c0", "c199", "target")
    );
    assert!(rows[1..]
        .iter()
        .all(|r| r.split(',').count() == 201 && r.split(',').all(|v| v.parse::<f32>().is_ok())));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn round_trip_keeps_labels_moves_the_target_last_and_is_deterministic() {
    let dir = workdir("mixed");
    write_mixed(&dir, 400);
    let args = |out: &'static str, threads: &'static str| {
        vec![
            "--threads",
            threads,
            "fit",
            "mixed.csv",
            "-t",
            "label",
            "--format",
            "auto",
            "--seed",
            "3",
            "--report",
            "r.json",
            "-q",
            "-o",
            out,
        ]
    };
    ok(&dir, &args("a.dope", "4"));
    ok(&dir, &args("b.dope", "1"));
    let (a, b) = (
        std::fs::read(dir.join("a.dope")).unwrap(),
        std::fs::read(dir.join("b.dope")).unwrap(),
    );
    assert_eq!(
        a, b,
        "same seed must give the same bytes for any thread count"
    );
    assert!(a.len() <= 2048 && a.starts_with(b"DPA1"));
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("r.json")).unwrap()).unwrap();
    assert_eq!(report["check"]["ran"], true, "{report}");
    assert!(
        report["check"]["score_format1"].is_number() && report["check"]["score_v7"].is_number()
    );
    let text = report.to_string();
    for leaked in ["paris", "lima", "oslo", "spend", "\"age"] {
        assert!(!text.contains(leaked), "report leaks {leaked}");
    }
    let s1 = ok(&dir, &["sample", "a.dope", "-n", "50", "--seed", "9"]);
    let s2 = ok(
        &dir,
        &[
            "--threads",
            "1",
            "sample",
            "a.dope",
            "-n",
            "50",
            "--seed",
            "9",
        ],
    );
    let s3 = ok(&dir, &["sample", "a.dope", "-n", "50", "--seed", "10"]);
    assert_eq!(s1.stdout, s2.stdout);
    assert_ne!(s1.stdout, s3.stdout);
    let rows = lines(&s1);
    assert_eq!(rows.len(), 51);
    assert_eq!(rows[0], "age,city,spend,label");
    for r in &rows[1..] {
        let cells: Vec<&str> = r.split(',').collect();
        assert_eq!(cells.len(), 4);
        assert!(
            ["paris", "lima", "oslo", "__other__"].contains(&cells[1]),
            "{r}"
        );
        assert!(["yes", "no"].contains(&cells[3]), "{r}");
    }
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn foreign_weights_truncation_and_bad_headers_are_refused() {
    let dir = workdir("refuse");
    write_wide(&dir, 40, 200);
    let fit = ok(
        &dir,
        &[
            "fit", "wide.csv", "-t", "target", "--format", "v7", "-o", "w.dope",
        ],
    );
    let summary = String::from_utf8_lossy(&fit.stderr).to_string();
    assert!(
        summary.starts_with("v7 only: wrote v7 (") && summary.contains("format1 not requested"),
        "{summary}"
    );
    let good = std::fs::read(dir.join("w.dope")).unwrap();
    assert_eq!(good[4], 7);
    let sample = |bytes: &[u8]| {
        std::fs::write(dir.join("bad.dope"), bytes).unwrap();
        refused(&dir, &["sample", "bad.dope", "-n", "5"])
    };
    let mut foreign = good.clone();
    foreign[9] ^= 0xff;
    let err = sample(&foreign);
    assert!(err.contains("fingerprint mismatch"), "{err}");
    for cut in [13, 15, good.len() / 2, good.len() - 1] {
        let err = sample(&good[..cut]);
        assert!(err.contains("artifact"), "{err}");
    }
    let mut tag = good.clone();
    tag[4] = 3;
    assert!(sample(&tag).contains("unknown encoder tag"));
    std::fs::write(dir.join("w.dope"), &good).unwrap();
    let err = refused(&dir, &["sample", "w.dope", "--mode", "faithful"]);
    assert!(err.contains("raw rows only"), "{err}");

    // A Format 1 payload in DPA1 samples through the Format 1 path; truncation is refused.
    write_mixed(&dir, 300);
    ok(
        &dir,
        &[
            "fit",
            "mixed.csv",
            "-t",
            "label",
            "--budget",
            "2000",
            "-q",
            "-o",
            "f1.dope",
        ],
    );
    let f1 = std::fs::read(dir.join("f1.dope")).unwrap();
    let mut wrapped = b"DPA1\x01".to_vec();
    wrapped.extend_from_slice(&[0; 8]);
    wrapped.extend_from_slice(&f1);
    std::fs::write(dir.join("d1.dope"), &wrapped).unwrap();
    let direct = ok(&dir, &["sample", "f1.dope", "-n", "20", "--seed", "4"]);
    let via = ok(&dir, &["sample", "d1.dope", "-n", "20", "--seed", "4"]);
    assert_eq!(direct.stdout, via.stdout);
    assert!(sample(&wrapped[..wrapped.len() - 2]).contains("artifact"));
    let err = refused(
        &dir,
        &["sample", "d1.dope", "--v7-weights", "x.safetensors"],
    );
    assert!(err.contains("only to V7 artifacts"), "{err}");
    wrapped[5] = 1;
    assert!(sample(&wrapped).contains("weights fingerprint"));
    // An unknown --format is a usage error that lists the accepted values.
    let bad = run(&dir, &["fit", "mixed.csv", "-t", "label", "--format", "3"]);
    assert_eq!(bad.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&bad.stderr).contains("auto"));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn v7_only_fits_and_samples_with_embedded_or_external_weights() {
    let dir = workdir("v7w");
    write_mixed(&dir, 300);
    let embedded = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/single-enc-v7/model.safetensors"
    );
    // Same layout, one mantissa bit changed: a different checkpoint and fingerprint.
    let mut other = std::fs::read(embedded).unwrap();
    let at = other.len() - 4;
    other[at] ^= 0x01;
    std::fs::write(dir.join("other.safetensors"), &other).unwrap();
    let fit = |out: &str, extra: &[&str]| {
        let mut args = vec!["fit", "mixed.csv", "-t", "label", "--format", "v7"];
        args.extend_from_slice(extra);
        args.extend_from_slice(&["--seed", "5", "-q", "-o", out]);
        ok(&dir, &args);
        std::fs::read(dir.join(out)).unwrap()
    };
    let e = fit("e.dope", &[]);
    let x = fit("x.dope", &["--v7-weights", "other.safetensors"]);
    let same = fit("s.dope", &["--v7-weights", embedded]);
    assert_eq!(
        e, same,
        "the compiled weights passed as a file give the same artifact"
    );
    assert_eq!((e[4], x[4]), (7, 7));
    assert_ne!(e[5..13], x[5..13], "the fingerprint follows the weights");
    assert!(e.len() <= 2048 && x.len() <= 2048);
    let rows = |args: &[&str]| lines(&ok(&dir, args)).len();
    assert_eq!(rows(&["sample", "e.dope", "-n", "7"]), 8);
    let with = [
        "sample",
        "x.dope",
        "-n",
        "7",
        "--v7-weights",
        "other.safetensors",
    ];
    assert_eq!(rows(&with), 8);
    for args in [
        vec!["sample", "x.dope", "-n", "7"],
        vec![
            "sample",
            "e.dope",
            "-n",
            "7",
            "--v7-weights",
            "other.safetensors",
        ],
    ] {
        assert!(refused(&dir, &args).contains("fingerprint mismatch"));
    }
    // The zero-init (step-0) decoder needs no weights: both files give the same rows.
    let zero = |m: &str, seed: &str| {
        ok(
            &dir,
            &["sample", m, "-n", "9", "--seed", seed, "--v7-zero-init"],
        )
    };
    let z = zero("e.dope", "2");
    assert_eq!(z.stdout, zero("x.dope", "2").stdout);
    assert_ne!(z.stdout, zero("e.dope", "3").stdout);
    assert_eq!(lines(&z)[0], "age,city,spend,label");
    assert_eq!(lines(&z).len(), 10);
    // On a regression target the trained target head moves every decoded target value.
    write_wide(&dir, 5, 200);
    ok(
        &dir,
        &[
            "fit", "wide.csv", "-t", "target", "--format", "v7", "-q", "-o", "r.dope",
        ],
    );
    let trained = ok(&dir, &["sample", "r.dope", "-n", "9", "--seed", "2"]);
    assert_ne!(
        zero("r.dope", "2").stdout,
        trained.stdout,
        "zero-init differs from the trained decoder"
    );
    let both = run(
        &dir,
        &[
            "sample",
            "e.dope",
            "--v7-zero-init",
            "--v7-weights",
            "other.safetensors",
        ],
    );
    assert_eq!(both.status.code(), Some(2), "conflicting decoder flags");
    let err = refused(
        &dir,
        &[
            "fit",
            "mixed.csv",
            "-t",
            "label",
            "--v7-weights",
            "other.safetensors",
        ],
    );
    assert!(err.contains("--format auto or v7"), "{err}");
    ok(
        &dir,
        &["fit", "mixed.csv", "-t", "label", "-q", "-o", "plain.dope"],
    );
    let err = refused(&dir, &["sample", "plain.dope", "--v7-zero-init"]);
    assert!(err.contains("only to V7 artifacts"), "{err}");
    std::fs::write(dir.join("cut.safetensors"), &other[..other.len() / 3]).unwrap();
    let err = refused(
        &dir,
        &[
            "fit",
            "mixed.csv",
            "-t",
            "label",
            "--format",
            "v7",
            "--v7-weights",
            "cut.safetensors",
        ],
    );
    assert!(err.contains("--v7-weights is not SingleEncV7"), "{err}");
    std::fs::remove_dir_all(dir).ok();
}

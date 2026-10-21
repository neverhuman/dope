//! The `dope` binary end to end on a small mixed-type CSV.

use std::path::{Path, PathBuf};
use std::process::Command;

fn dope(dir: &Path, args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_dope"))
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "dope {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn workdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("dope-cli-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut csv = String::from("age,city,score\n");
    for i in 0..400u32 {
        let city = ["paris", "lima", "oslo"][(i % 3) as usize];
        let age = 20 + (i * 7919) % 50;
        let score =
            age as f64 * 0.5 + if city == "oslo" { 10.0 } else { 0.0 } + ((i * 31) % 7) as f64;
        csv += &format!("{age},{city},{score}\n");
    }
    std::fs::write(dir.join("train.csv"), csv).unwrap();
    dir
}

#[test]
fn fit_inspect_sample_predict_encode() {
    let dir = workdir("journey");
    let fit = dope(
        &dir,
        &[
            "fit",
            "train.csv",
            "--target",
            "score",
            "--budget",
            "2k",
            "-o",
            "m.dope",
            "--device",
            "cpu",
        ],
    );
    assert!(fit.contains("compiled train.csv -> m.dope"));
    let size = std::fs::metadata(dir.join("m.dope")).unwrap().len();
    assert!(size <= 2048, "artifact {size} bytes over budget");
    let inspect: serde_json::Value =
        serde_json::from_str(&dope(&dir, &["inspect", "m.dope", "--json"])).unwrap();
    assert_eq!(inspect["target"], "score");
    assert_eq!(inspect["bytes"]["total"].as_u64(), Some(size));
    let synth = dope(
        &dir,
        &[
            "sample", "m.dope", "-n", "20", "--seed", "3", "--device", "cpu",
        ],
    );
    let mut lines = synth.lines();
    assert_eq!(lines.next(), Some("age,city,score"));
    assert_eq!(lines.count(), 20);
    let ml = dope(
        &dir,
        &[
            "sample", "m.dope", "-n", "5", "--space", "ml", "--device", "cpu",
        ],
    );
    assert!(ml.lines().next().unwrap().contains("city=oslo"));
    let pred = dope(&dir, &["predict", "m.dope", "train.csv"]);
    assert_eq!(pred.lines().count(), 401);
    let enc = dope(&dir, &["encode", "train.csv", "--model", "m.dope"]);
    assert_eq!(enc.lines().next().unwrap(), ml.lines().next().unwrap());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn errors_name_the_problem_without_echoing_cells() {
    let dir = workdir("errors");
    let out = Command::new(env!("CARGO_BIN_EXE_dope"))
        .current_dir(&dir)
        .args(["fit", "train.csv", "--target", "missing_column"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("target column not found"));
    assert!(!err.contains("paris"));
    std::fs::remove_dir_all(dir).ok();
}

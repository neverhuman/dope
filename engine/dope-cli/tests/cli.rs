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

#[test]
fn size_parameter_caps_and_tier_catalog() {
    let dir = workdir("tiers");
    dope(
        &dir,
        &[
            "fit",
            "train.csv",
            "-t",
            "score",
            "--size",
            "small",
            "--max-parameters",
            "200",
            "--no-audit",
            "--report",
            "fit.json",
            "--device",
            "cpu",
        ],
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("fit.json")).unwrap()).unwrap();
    assert_eq!(report["budget"], 2048);
    assert!(report["parameters"]["total"].as_u64().unwrap() <= 200);
    let inspect: serde_json::Value =
        serde_json::from_str(&dope(&dir, &["inspect", "model.dope", "--json"])).unwrap();
    assert_eq!(report["parameters"], inspect["parameters"]);
    dope(
        &dir,
        &[
            "fit-tiers",
            "train.csv",
            "-t",
            "score",
            "-o",
            "zoo",
            "--dataset-id",
            "test-fixture",
            "--source",
            "fixture",
            "--data-license",
            "MIT",
            "--device",
            "cpu",
        ],
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("zoo/catalog.json")).unwrap()).unwrap();
    assert_eq!(manifest["kind"], "per-dataset-pretrained-generators");
    assert_eq!(manifest["release_authorized"], false);
    assert_eq!(manifest["anonymized"], false);
    assert_eq!(manifest["target_index"], 2);
    for item in manifest["artifacts"].as_array().unwrap() {
        let file = dir.join("zoo").join(item["file"].as_str().unwrap());
        let bytes = std::fs::read(&file).unwrap();
        assert_eq!(bytes.len() as u64, item["bytes"].as_u64().unwrap());
        assert!(item["bytes"].as_u64().unwrap() <= item["budget_bytes"].as_u64().unwrap());
        use sha2::{Digest, Sha256};
        assert_eq!(item["sha256"], format!("{:x}", Sha256::digest(&bytes)));
        let p = dope_core::decode(&bytes).unwrap();
        assert_eq!(
            item["parameters"],
            serde_json::to_value(p.parameters()).unwrap()
        );
    }
    let out = Command::new(env!("CARGO_BIN_EXE_dope"))
        .current_dir(&dir)
        .args([
            "fit-tiers",
            "train.csv",
            "-t",
            "score",
            "-o",
            "zoo",
            "--dataset-id",
            "fixture",
            "--source",
            "fixture",
            "--data-license",
            "MIT",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("already exists"));
    dope(
        &dir,
        &[
            "fit-tiers",
            "train.csv",
            "-t",
            "score",
            "-o",
            "zoo-anon",
            "--dataset-id",
            "test-fixture",
            "--source",
            "fixture",
            "--data-license",
            "MIT",
            "--device",
            "cpu",
            "--anonymize",
        ],
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("zoo-anon/catalog.json")).unwrap()).unwrap();
    assert_eq!(manifest["anonymized"], true);
    let p = dope_core::decode(&std::fs::read(dir.join("zoo-anon/small.dope")).unwrap()).unwrap();
    assert_ne!(p.schema.target.name, "score");
    std::fs::remove_dir_all(dir).ok();
}

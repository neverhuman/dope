//! Actual native CV, input-bound encoding, then recipient sampling without landmarks.
use dope_core::levels::digest;
use std::{path::PathBuf, process::Command};
#[test]
fn training_only_cv_features_are_inputs_and_not_recipient_dependencies() {
    let root = std::env::temp_dir().join(format!("dope-landmark-input-cli-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let csv = root.join("fit.csv");
    let text = String::from("PRIVATE_HEADER,c,target\n")
        + &(0..80)
            .map(|i| format!("{},PRIVATE_CATEGORY_{},{}\n", i, i % 2, i * 2))
            .collect::<String>();
    std::fs::write(&csv, &text).unwrap();
    let features = root.join("features");
    std::fs::create_dir(&features).unwrap();
    let measured = Command::new(env!("CARGO_BIN_EXE_dope"))
        .args(["--threads", "1", "landmarks", "real"])
        .arg(&csv)
        .args(["--target", "target", "--task", "regression", "--json"])
        .output()
        .unwrap();
    assert!(measured.status.success(), "{measured:?}");
    let report: serde_json::Value = serde_json::from_slice(&measured.stdout).unwrap();
    assert_eq!(report["rows"], 80);
    assert_eq!(report["options"]["folds"], 5);
    std::fs::write(features.join("report.json"), &measured.stdout).unwrap();
    let manifest = serde_json::json!({"format":"dope-fit-cv-landmark-input-v1",
        "feature_recipe":"fit-cv-gain23-mask-squash-v1","options":report["options"],
        "engine_sha256":digest(&std::fs::read(env!("CARGO_BIN_EXE_dope")).unwrap()),
        "report_sha256":digest(&measured.stdout),"producer":{"purpose":"actual native CV unit journey"},
        "scope":{"allowed_roles":["fit"],"split_policy":"iid","training_input_sha256":digest(text.as_bytes()),
        "column_kinds":["numeric","categorical","numeric"],"task":"regression","rows":80,
        "physical_rows_sha256":digest(b"owned-unit-rows"),"physical_table_id":digest(b"owned-unit-table")}});
    let raw = serde_json::to_vec(&manifest).unwrap();
    let expected = digest(&raw);
    std::fs::write(features.join("manifest.json"), raw).unwrap();
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../dope-core/tests/fixtures/learned-v3/input/model");
    let code = root.join("all.code");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_dope"));
    cmd.args(["--threads", "1", "learned", "encode", "--model"])
        .arg(&model)
        .arg("--csv")
        .arg(&csv)
        .args([
            "--target",
            "target",
            "--kinds",
            "numeric,categorical,numeric",
            "--task",
            "regression",
            "--levels",
            "low,medium,high,ultra",
            "--landmark-input",
        ])
        .arg(&features)
        .args(["--landmark-manifest-sha256", &expected, "--out"])
        .arg(&code);
    let output = cmd.output().unwrap();
    assert!(output.status.success(), "{output:?}");
    let bytes = std::fs::read(&code).unwrap();
    for private in [
        "PRIVATE_",
        "physical_rows_sha256",
        "training_input_sha256",
        "train_rows",
    ] {
        assert!(!String::from_utf8_lossy(&bytes).contains(private));
    }
    // A changed CSV cannot consume the old feature report.
    std::fs::write(&csv, text.replace("PRIVATE_CATEGORY_0", "CHANGED_CATEGORY")).unwrap();
    let refused = cmd.output().unwrap();
    assert!(!refused.status.success());
    assert!(!String::from_utf8_lossy(&refused.stderr).contains("PRIVATE_"));
    let recipient = root.join("recipient");
    std::fs::create_dir(&recipient).unwrap();
    for file in ["model.json", "decoder.safetensors"] {
        std::fs::copy(model.join(file), recipient.join(file)).unwrap();
    }
    std::fs::remove_file(&csv).unwrap();
    std::fs::remove_dir_all(features).unwrap();
    for tier in ["low", "medium", "high", "ultra"] {
        let sampled = Command::new(env!("CARGO_BIN_EXE_dope"))
            .env("PATH", "/nonexistent")
            .args(["--threads", "1", "learned", "sample", "--model"])
            .arg(&recipient)
            .arg("--code")
            .arg(&code)
            .args(["--level", tier, "--rows", "4", "--steps", "8", "--out"])
            .arg(root.join(format!("{tier}.csv")))
            .output()
            .unwrap();
        assert!(sampled.status.success(), "{tier}: {sampled:?}");
    }
    assert!(!recipient.join("encoder.safetensors").exists());
    std::fs::remove_dir_all(root).unwrap();
}

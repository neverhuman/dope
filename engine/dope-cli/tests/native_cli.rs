//! Customer commands with PATH empty: no Python, package manager or helper executable.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::{Command, Output};

fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_dope"))
        .current_dir(dir)
        .env_clear()
        .env("PATH", dir.join("no-tools"))
        .args(args)
        .output()
        .unwrap()
}
fn success(dir: &Path, args: &[&str]) -> String {
    let out = run(dir, args);
    assert!(
        out.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn native_customer_journey_and_machine_output_without_external_tools() {
    let dir = std::env::temp_dir().join(format!("dope-native-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut csv = "a,b,y\n".to_string();
    for i in 0..120 {
        let a = (i * 17 % 127) as f64 / 127.;
        let b = (i * 31 % 131) as f64 / 131.;
        csv.push_str(&format!("{a},{b},{}\n", a * 2. + b * b));
    }
    std::fs::write(dir.join("train.csv"), csv).unwrap();
    let doctor: Value =
        serde_json::from_str(&success(&dir, &["doctor", "--json", "--threads", "2"])).unwrap();
    assert_eq!(doctor["implementation"], "native_rust");
    assert_eq!(doctor["python_required"], false);
    assert_eq!(doctor["libtorch_required"], false);
    assert_eq!(doctor["threads"], 2);
    let binary = std::fs::read(env!("CARGO_BIN_EXE_dope")).unwrap();
    assert_eq!(
        doctor["executable_sha256"],
        format!("{:x}", Sha256::digest(&binary))
    );
    let fit: Value = serde_json::from_str(&success(
        &dir,
        &[
            "--threads",
            "2",
            "fit",
            "train.csv",
            "-t",
            "y",
            "--size",
            "small",
            "--max-parameters",
            "200",
            "--device",
            "cpu",
            "--plan",
            "3,2,0",
            "--no-trees",
            "--no-audit",
            "--json",
            "--report",
            "fit.json",
        ],
    ))
    .unwrap();
    assert_eq!(fit["budget"], 2048);
    assert_eq!(
        fit,
        serde_json::from_slice::<Value>(&std::fs::read(dir.join("fit.json")).unwrap()).unwrap()
    );
    let raw = success(
        &dir,
        &[
            "--threads",
            "2",
            "sample",
            "model.dope",
            "-n",
            "11",
            "--device",
            "cpu",
        ],
    );
    assert_eq!(raw.lines().count(), 12);
    assert_eq!(raw.lines().next(), Some("a,b,y"));
    let ml = success(
        &dir,
        &[
            "sample",
            "model.dope",
            "-n",
            "7",
            "--space",
            "ml",
            "--device",
            "cpu",
            "--threads",
            "2",
        ],
    );
    assert_eq!(ml.lines().count(), 8);
    let predictions = success(
        &dir,
        &["predict", "model.dope", "train.csv", "--threads", "2"],
    );
    assert_eq!(predictions.lines().count(), 121);
    let inspect: Value =
        serde_json::from_str(&success(&dir, &["inspect", "model.dope", "--json"])).unwrap();
    assert_eq!(inspect["parameters"], fit["parameters"]);
    let encoded = success(&dir, &["encode", "train.csv", "--model", "model.dope"]);
    assert_eq!(encoded.lines().count(), 121);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn invalid_thread_controls_fail_before_opening_input() {
    let dir = std::env::temp_dir();
    for threads in ["0", "1025", "-1", "invalid"] {
        let out = run(
            &dir,
            &["--threads", threads, "fit", "absent.csv", "-t", "y"],
        );
        assert_eq!(out.status.code(), Some(2));
    }
    let out = run(&dir, &["fit", "absent.csv", "-t", "y", "--json", "--quiet"]);
    assert_eq!(out.status.code(), Some(2));
}

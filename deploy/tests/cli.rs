use std::process::Command;

use dope_kernel::model::*;

#[test]
fn actual_cli_exports_inspects_samples_and_refuses_private_path_errors() {
    let root = std::env::current_dir().unwrap().join("target/cli-fixtures");
    std::fs::create_dir_all(&root).unwrap();
    let root = root.join(format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let native = root.join("private-source-marker.dpk");
    let carrier = root.join("model.safetensors");
    let samples = root.join("synthetic.csv");
    let kernel = Kernel {
        task: Task::Regression,
        rows_fitted: 64,
        features: 1,
        seed: 11,
        seed_policy: 0,
        quantization_bits: 8,
        compliant: false,
        schema: vec![ColumnSchema {
            kind: SchemaKind::Continuous,
            missing_probability: 0.0,
            impute: 0.5,
            transform: Transform::Identity,
        }],
        marginals: vec![Marginal::QuantileSpline {
            values: vec![0.0, 0.2, 0.5, 0.8, 1.0],
        }],
        program: KernelProgram::Symbolic {
            dependence: Dependence::Independent,
            target: Target::SparseLinear {
                intercept: 0.1,
                terms: vec![LinearTerm {
                    feature: 0,
                    coefficient: 0.5,
                }],
            },
            noise: Noise::Homoscedastic { sigma: 0.05 },
        },
    };
    std::fs::write(&native, dope_kernel::encode_kernel(&kernel).unwrap()).unwrap();
    let exe = env!("CARGO_BIN_EXE_dope-runtime");
    let packed = Command::new(exe)
        .arg("pack")
        .arg("--artifact")
        .arg(&native)
        .arg("--out")
        .arg(&carrier)
        .output()
        .unwrap();
    assert!(
        packed.status.success(),
        "{}",
        String::from_utf8_lossy(&packed.stderr)
    );
    let receipt: serde_json::Value = serde_json::from_slice(&packed.stdout).unwrap();
    assert_eq!(
        receipt["charged_bytes"].as_u64().unwrap(),
        std::fs::metadata(&carrier).unwrap().len()
    );
    let inspected = Command::new(exe)
        .arg("inspect")
        .arg("--artifact")
        .arg(&carrier)
        .output()
        .unwrap();
    assert!(inspected.status.success());
    let receipt: serde_json::Value = serde_json::from_slice(&inspected.stdout).unwrap();
    assert_eq!(receipt["cold_closure_bytes"], serde_json::Value::Null);
    assert_eq!(receipt["qualified"], false);
    let sampled = Command::new(exe)
        .arg("sample")
        .arg("--artifact")
        .arg(&carrier)
        .args(["--rows", "17", "--device", "auto"])
        .arg("--out")
        .arg(&samples)
        .output()
        .unwrap();
    assert!(sampled.status.success());
    let rows = std::fs::read_to_string(&samples).unwrap();
    assert_eq!(rows.lines().count(), 17);
    assert!(rows.lines().all(|line| line.split(',').count() == 2));
    let refused = Command::new(exe)
        .arg("pack")
        .arg("--artifact")
        .arg(&native)
        .arg("--out")
        .arg(&carrier)
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert!(!String::from_utf8_lossy(&refused.stderr).contains("private-source-marker"));
    let refused = Command::new(exe)
        .arg("sample")
        .arg("--artifact")
        .arg(&carrier)
        .args(["--rows", "1", "--device", "cuda"])
        .arg("--out")
        .arg(root.join("refused.csv"))
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert!(!root.join("refused.csv").exists());
    let _ = std::fs::remove_dir_all(root);
}

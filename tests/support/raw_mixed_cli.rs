use std::{fs, process::Command};

#[test]
fn mixed_cli_replays_tagged_masks_and_column_scoped_nominal_values_in_fresh_worker() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/raw-mixed-cli")
        .join(std::process::id().to_string());
    let empty = root.join("empty");
    fs::create_dir_all(&empty).unwrap();
    let schema = root.join("schema.json");
    let csv = root.join("fit.csv");
    let artifact = root.join("model.mixed");
    let output = root.join("samples.json");
    fs::write(
        &schema,
        serde_json::to_vec(&serde_json::json!({
            "format":"dope-table-schema","version":1,"columns":[
                {"id":"c0001","source_name":"private_anchor","role":"feature","dtype":"continuous"},
                {"id":"c0002","source_name":"private_category","role":"feature","dtype":"nominal",
                 "domain":{"kind":"categorical","values":["Aa","BB","missing"]},
                 "mask_tokens":[{"token":"M","state":"missing"}]},
                {"id":"c0003","source_name":"private_target","role":"target","dtype":"continuous"}
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    let mut rows = String::from("private_anchor,private_category,private_target\n");
    for row in 0..64 {
        rows.push_str(&format!(
            "{},{},{}\n",
            row,
            if row < 32 {
                "Aa"
            } else if row % 3 == 0 {
                "M"
            } else if row % 3 == 1 {
                "BB"
            } else {
                "undeclared_private"
            },
            row + 100
        ));
    }
    fs::write(&csv, rows).unwrap();
    let binary = env!("CARGO_BIN_EXE_dope-kernel");
    let result = Command::new(binary)
        .arg("compile-mixed")
        .arg("--csv")
        .arg(&csv)
        .arg("--schema")
        .arg(&schema)
        .arg("--out")
        .arg(&artifact)
        .args(["--profile", "numeric_split_v1", "--deadline-seconds", "30"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let inspected = Command::new(binary)
        .arg("inspect-mixed")
        .arg("--artifact")
        .arg(&artifact)
        .output()
        .unwrap();
    assert!(inspected.status.success());
    let public = String::from_utf8(inspected.stdout).unwrap();
    for private in [
        "private_anchor",
        "private_category",
        "Aa",
        "BB",
        "undeclared_private",
        "\"probabilities\"",
        "\"values\"",
    ] {
        assert!(!public.contains(private));
    }
    let result = Command::new(binary)
        .arg("sample-mixed")
        .arg("--artifact")
        .arg(&artifact)
        .arg("--out")
        .arg(&output)
        .args(["--rows", "256", "--seed", "11"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    let values: serde_json::Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert!(
        values["values"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v["kind"] == "mask" && v["value"] == "unknown_category")
    );
    assert!(
        values["values"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v["kind"] == "mask" && v["value"] == "missing")
    );
    let worker = Command::new(binary)
        .env_clear()
        .current_dir(&empty)
        .arg("sample-mixed")
        .arg("--artifact")
        .arg(&artifact)
        .args(["--rows", "256", "--seed", "11", "--isolated"])
        .output()
        .unwrap();
    assert!(
        worker.status.success(),
        "{}",
        String::from_utf8_lossy(&worker.stderr)
    );
    let replay: serde_json::Value = serde_json::from_slice(&worker.stdout).unwrap();
    assert_eq!(receipt["sample_sha256"], replay["sample_sha256"]);
    assert_eq!(
        replay["isolation"],
        "linux_x86_64_seccomp_tsync_default_deny"
    );
    assert!(
        replay["inspection"]["accounting"]["cold_sampling_closure_bytes"]
            .as_u64()
            .unwrap()
            > artifact.metadata().unwrap().len()
    );
    assert!(dope_kernel::typed_data::workflow::read_raw_artifact(&artifact).is_err());
    let forbidden = Command::new(binary)
        .arg("sample-mixed")
        .arg("--artifact")
        .arg(&artifact)
        .arg("--out")
        .arg(root.join("forbidden.json"))
        .args(["--rows", "10", "--isolated"])
        .output()
        .unwrap();
    assert!(!forbidden.status.success() && !root.join("forbidden.json").exists());
}

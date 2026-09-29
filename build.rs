use std::collections::BTreeMap;
use std::fs;

use serde_json::Value;
use sha2::{Digest, Sha256};

fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(entries) => Value::Object(
            entries
                .iter()
                .map(|(key, value)| (key.clone(), canonical(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(canonical).collect()),
        _ => value.clone(),
    }
}

fn main() {
    if std::env::var_os("CARGO_FEATURE_GPU_TRAINING").is_some() {
        println!("cargo:rustc-link-lib=cudart");
        // Keep libtorch_cuda in the final ELF even when the linker sees no
        // direct Rust symbol reference. Its static initializers register the
        // CUDA backend used by tch::Cuda::is_available().
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-arg=-ltorch_cuda");
        println!("cargo:rustc-link-arg=-lc10_cuda");
        println!("cargo:rustc-link-arg=-Wl,--as-needed");
    }
    let path = "production/kpi-contract.json";
    println!("cargo:rerun-if-changed={path}");
    let raw = fs::read(path).expect("production KPI contract must be readable");
    let parsed: Value = serde_json::from_slice(&raw).expect("production KPI contract must be JSON");
    let bytes = serde_json::to_vec(&canonical(&parsed)).expect("canonical KPI contract serializes");
    let normalized = raw.strip_suffix(b"\n").unwrap_or(&raw);
    assert_eq!(
        normalized, bytes,
        "production/kpi-contract.json must use canonical JSON encoding"
    );
    println!(
        "cargo:rustc-env=DOPE_KPI_CONTRACT_SHA256={:x}",
        Sha256::digest(&bytes)
    );
    println!(
        "cargo:rustc-env=DOPE_KPI_CONTRACT_BLAKE3={}",
        blake3::hash(&bytes).to_hex()
    );
    let v1_path = "production/kpi-contract-v1.json";
    println!("cargo:rerun-if-changed={v1_path}");
    let v1 = fs::read(v1_path).expect("historical KPI contract must be readable");
    let v1_parsed: Value =
        serde_json::from_slice(&v1).expect("historical KPI contract must be JSON");
    let v1_bytes =
        serde_json::to_vec(&canonical(&v1_parsed)).expect("historical KPI contract serializes");
    assert_eq!(v1.strip_suffix(b"\n").unwrap_or(&v1), v1_bytes);
    println!(
        "cargo:rustc-env=DOPE_KPI_CONTRACT_V1_SHA256={:x}",
        Sha256::digest(&v1_bytes)
    );
    println!(
        "cargo:rustc-env=DOPE_KPI_CONTRACT_V1_BLAKE3={}",
        blake3::hash(&v1_bytes).to_hex()
    );
}

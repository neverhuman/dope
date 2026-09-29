use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

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
    assert_eq!(
        parsed["version"].as_u64(),
        Some(2),
        "active contract must be v2"
    );
    let limits = parsed["tier_byte_limits"]
        .as_object()
        .expect("v2 tier limits must be an object");
    assert_eq!(limits.len(), 2, "only L2 and L3 have hard byte limits");
    let l2 = limits["l2"]
        .as_u64()
        .expect("L2 byte limit must be numeric");
    let l3 = limits["l3"]
        .as_u64()
        .expect("L3 byte limit must be numeric");
    assert_eq!((l2, l3), (32_768, 10_240), "frozen tier limits changed");
    let generated = format!(
        "// Generated from production/kpi-contract.json by build.rs.\n\
         pub const L2_ARTIFACT_LIMIT: usize = {l2};\n\
         pub const L3_ARTIFACT_LIMIT: usize = {l3};\n"
    );
    let generated_path = PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo OUT_DIR"))
        .join("kpi_contract_generated.rs");
    fs::write(generated_path, generated).expect("generated contract constants must be writable");
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

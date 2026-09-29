use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

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

fn libtorch_include_paths() -> Vec<PathBuf> {
    if std::env::var_os("LIBTORCH_USE_PYTORCH").is_some() {
        let python = if std::env::var_os("VIRTUAL_ENV").is_some() {
            "python"
        } else {
            "python3"
        };
        let script = "import torch\nfrom torch.utils import cpp_extension\nfor path in cpp_extension.include_paths(): print(path)";
        let output = Command::new(python)
            .args(["-c", script])
            .output()
            .expect("Python used by torch-sys must be executable");
        assert!(
            output.status.success(),
            "Python must report the installed PyTorch include paths: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return String::from_utf8(output.stdout)
            .expect("PyTorch include paths must be UTF-8")
            .lines()
            .map(PathBuf::from)
            .collect();
    }

    let root = std::env::var_os("LIBTORCH")
        .map(PathBuf::from)
        .or_else(|| {
            PathBuf::from("/usr/lib/libtorch.so")
                .exists()
                .then(|| PathBuf::from("/usr"))
        })
        .expect("gpu-training requires LIBTORCH or LIBTORCH_USE_PYTORCH=1");
    let include_root = std::env::var_os("LIBTORCH_INCLUDE")
        .map(PathBuf::from)
        .unwrap_or(root);
    vec![
        include_root.join("include"),
        include_root.join("include/torch/csrc/api/include"),
    ]
}

fn build_libtorch_determinism_bridge() {
    println!("cargo:rerun-if-changed=cpp/libtorch_determinism.cpp");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_INCLUDE");
    println!("cargo:rerun-if-env-changed=LIBTORCH_USE_PYTORCH");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("cpp/libtorch_determinism.cpp")
        .flag_if_supported("-std=c++17")
        // PyTorch's public headers emit unused-parameter warnings in their
        // fallback hook implementations; the bridge itself is warning-free.
        .warnings(false);
    for include in libtorch_include_paths() {
        build.include(include);
    }
    build.compile("dope_libtorch_determinism");
}

fn main() {
    if std::env::var_os("CARGO_FEATURE_GPU_TRAINING").is_some() {
        build_libtorch_determinism_bridge();
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

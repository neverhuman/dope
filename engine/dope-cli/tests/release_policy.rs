//! Release policy fixtures use a fake verifier to test invocation and fail-closed behavior.
//! Cryptographic/OIDC qualification belongs to the actual trusted tag workflow.
#![cfg(target_os = "linux")]
use sha2::{Digest, Sha256};
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
const TAG: &str = "dope-engine-v1.0.0";
const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn fixture(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("dope-release-{name}-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("tools")).unwrap();
    let gh = dir.join("tools/gh");
    std::fs::write(&gh, "#!/bin/bash\nprintf '%s\\n' \"$@\" >> \"$GH_TEST_LOG\"\n[[ \"${GH_TEST_MODE:-ok}\" == ok ]]\n").unwrap();
    std::fs::set_permissions(&gh, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut sums = String::new();
    for name in [
        "BUILDINFO.json",
        "Cargo.lock",
        "DEPENDENCIES.txt",
        "LICENSE",
        "THIRD_PARTY_NOTICES",
        "dope-linux-x86_64",
    ] {
        let bytes = b"fixture; never execute";
        std::fs::write(dir.join(name), bytes).unwrap();
        sums.push_str(&format!("{:x}  {name}\n", Sha256::digest(bytes)));
    }
    std::fs::write(dir.join("SHA256SUMS"), sums).unwrap();
    dir
}
fn verify(dir: &Path, tag: &str, sha: &str, mode: &str) -> std::process::Output {
    Command::new("/bin/bash")
        .arg(root().join("scripts/verify-engine-release.sh"))
        .args([dir.to_str().unwrap(), tag, sha])
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", dir.join("tools").display()),
        )
        .env("GH_TEST_LOG", dir.join("gh.log"))
        .env("GH_TEST_MODE", mode)
        .output()
        .unwrap()
}
#[test]
fn signer_policy_binds_binary_and_inventory_to_exact_source() {
    let dir = fixture("valid");
    let out = verify(&dir, TAG, SHA, "ok");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let log = std::fs::read_to_string(dir.join("gh.log")).unwrap();
    assert_eq!(log.matches("attestation\nverify\n").count(), 2);
    assert_eq!(
        log.matches("neverhuman/dope/.github/workflows/engine-release.yml")
            .count(),
        2
    );
    assert_eq!(log.matches(&format!("refs/tags/{TAG}")).count(), 2);
    assert_eq!(log.matches(SHA).count(), 2);
    assert!(log.contains("--repo\nneverhuman/dope\n"));
    std::fs::remove_dir_all(dir).unwrap();
}
#[test]
fn unsigned_tampered_linked_and_invalid_packages_are_refused() {
    let dir = fixture("negative");
    assert!(!verify(&dir, TAG, SHA, "reject").status.success());
    assert!(!verify(&dir, "other-tag", SHA, "ok").status.success());
    assert!(!verify(&dir, TAG, "invalid", "ok").status.success());
    std::fs::write(dir.join("dope-linux-x86_64"), "tampered").unwrap();
    assert!(!verify(&dir, TAG, SHA, "ok").status.success());
    std::fs::remove_file(dir.join("dope-linux-x86_64")).unwrap();
    symlink(dir.join("LICENSE"), dir.join("dope-linux-x86_64")).unwrap();
    assert!(!verify(&dir, TAG, SHA, "ok").status.success());
    std::fs::remove_dir_all(dir).unwrap();
    let dir = fixture("inventory");
    let sums = std::fs::read_to_string(dir.join("SHA256SUMS"))
        .unwrap()
        .replace("BUILDINFO.json", "../BUILDINFO.json");
    std::fs::write(dir.join("SHA256SUMS"), sums).unwrap();
    assert!(!verify(&dir, TAG, SHA, "ok").status.success());
    std::fs::remove_dir_all(dir).unwrap();
}
#[test]
fn cpu_graph_refuses_training_bindings_and_implicit_gpu_closures() {
    let dir = fixture("graph");
    for name in [
        "tch",
        "torch-sys",
        "pyo3",
        "pyo3-ffi",
        "cpython",
        "python3-sys",
        "numpy",
        "rustpython-vm",
        "libpython-clib",
        "ort",
        "cudarc",
        "candle-kernels",
        "metal",
    ] {
        std::fs::write(
            dir.join("graph"),
            format!("dope-cli v1.0.0\ndope-core v1.0.0\n{name} v1.0.0\n"),
        )
        .unwrap();
        let status = Command::new("/bin/bash")
            .arg(root().join("ops/ci/engine-runtime-graph.sh"))
            .arg(dir.join("graph"))
            .output()
            .unwrap()
            .status;
        assert!(!status.success(), "admitted {name}");
    }
    std::fs::write(
        dir.join("graph"),
        "dope-cli v1.0.0\ndope-core v1.0.0\nrayon v1.11.0\n",
    )
    .unwrap();
    assert!(Command::new("/bin/bash")
        .arg(root().join("ops/ci/engine-runtime-graph.sh"))
        .arg(dir.join("graph"))
        .status()
        .unwrap()
        .success());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn rust_application_gate_does_not_launch_research_python() {
    let dir = fixture("gate");
    for (name, script) in [
        ("cargo", "#!/bin/bash\nexit 0\n"),
        ("python3", "#!/bin/bash\nexit 99\n"),
    ] {
        let path = dir.join("tools").join(name);
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let out = Command::new("/bin/bash")
        .arg(root().join("ops/ci/engine.sh"))
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", dir.join("tools").display()),
        )
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::remove_dir_all(dir).unwrap();
}

//! Release policy fixtures use a fake verifier to test invocation and fail-closed behavior.
//! Cryptographic/OIDC qualification belongs to the actual trusted tag workflow.
#![cfg(target_os = "linux")]
use sha2::{Digest, Sha256};
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
const TAG: &str = "dope-engine-v1.0.0";
const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
fn installer_gh_fixture() -> &'static str {
    r#"#!/bin/bash
case "$*" in
  --version) printf 'gh version %s (policy fixture)\n' "${GH_TEST_VERSION:-2.102.0}"; exit 0 ;;
  'attestation verify --help') printf '%s\n' '--repo --signer-workflow --source-ref --source-digest --format --bundle'; exit 0 ;;
esac
printf '%s\n' "$@" >> "$GH_TEST_LOG"
[[ "${GH_TEST_MODE:-ok}" == ok ]]
"#
}
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
    std::fs::write(&gh, installer_gh_fixture()).unwrap();
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
    std::fs::write(dir.join("dope.sigstore.json"), b"{}\n").unwrap();
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
    assert_eq!(log.matches("--bundle\n").count(), 2);
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

#[test]
fn installer_verifies_release_integrity_and_attestation_policy_fail_closed() {
    let dir = std::env::temp_dir().join(format!("dope-installer-test-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("stage")).unwrap();
    std::fs::create_dir_all(dir.join("tools")).unwrap();
    let gh = dir.join("tools/gh");
    std::fs::write(&gh, installer_gh_fixture()).unwrap();
    std::fs::set_permissions(&gh, std::fs::Permissions::from_mode(0o755)).unwrap();

    let bin_bytes = std::fs::read(env!("CARGO_BIN_EXE_dope")).unwrap();
    let bin_sha = format!("{:x}", Sha256::digest(&bin_bytes));
    let buildinfo = serde_json::json!({
        "schema_version": 1,
        "product": "dope-engine",
        "version": "1.0.0",
        "source_commit": SHA,
        "binary_sha256": bin_sha,
        "signature": "external-sigstore-attestation-required"
    })
    .to_string();
    std::fs::write(dir.join("stage/dope-linux-x86_64"), &bin_bytes).unwrap();
    std::fs::write(dir.join("stage/BUILDINFO.json"), &buildinfo).unwrap();
    std::fs::write(dir.join("stage/LICENSE"), "MIT").unwrap();
    std::fs::write(dir.join("stage/THIRD_PARTY_NOTICES"), "notices").unwrap();
    let mut sums = String::new();
    for name in [
        "BUILDINFO.json",
        "LICENSE",
        "THIRD_PARTY_NOTICES",
        "dope-linux-x86_64",
    ] {
        let bytes = std::fs::read(dir.join("stage").join(name)).unwrap();
        sums.push_str(&format!("{:x}  {name}\n", Sha256::digest(&bytes)));
    }
    std::fs::write(dir.join("stage/SHA256SUMS"), sums).unwrap();
    std::fs::write(dir.join("stage/dope.sigstore.json"), b"{}\n").unwrap();

    // 1. Refuse floating 'latest' tag.
    let bad_tag = Command::new("/bin/bash")
        .arg(root().join("scripts/install.sh"))
        .args([
            "--tag",
            "latest",
            "--from-dir",
            dir.join("stage").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!bad_tag.status.success());

    // 2. Unsigned research tag v1.0.0 installs with explicit sha256-content-integrity-only disclosure.
    let ok_unsigned = Command::new("/bin/bash")
        .arg(root().join("scripts/install.sh"))
        .args([
            "--tag",
            "v1.0.0",
            "--from-dir",
            dir.join("stage").to_str().unwrap(),
            "--install-dir",
            dir.join("bin").to_str().unwrap(),
            "--share-dir",
            dir.join("share").to_str().unwrap(),
            "--no-weights",
        ])
        .output()
        .unwrap();
    assert!(
        ok_unsigned.status.success(),
        "{}",
        String::from_utf8_lossy(&ok_unsigned.stderr)
    );
    let stdout = String::from_utf8_lossy(&ok_unsigned.stdout);
    assert!(stdout.contains("sha256-content-integrity-only"));

    // 3. Signed dope-engine-v1.0.0 tag requires OIDC attestation and fails closed on reject.
    let reject_att = Command::new("/bin/bash")
        .arg(root().join("scripts/install.sh"))
        .args([
            "--tag",
            TAG,
            "--source-digest",
            SHA,
            "--from-dir",
            dir.join("stage").to_str().unwrap(),
            "--install-dir",
            dir.join("bin2").to_str().unwrap(),
            "--share-dir",
            dir.join("share2").to_str().unwrap(),
            "--no-weights",
        ])
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", dir.join("tools").display()),
        )
        .env("GH_TEST_LOG", dir.join("gh.log"))
        .env("GH_TEST_MODE", "reject")
        .output()
        .unwrap();
    assert!(!reject_att.status.success());
    assert!(!dir.join("bin2/dope").exists());

    // 4. Signed dope-engine-v1.0.0 tag succeeds when OIDC attestation passes.
    let ok_att = Command::new("/bin/bash")
        .arg(root().join("scripts/install.sh"))
        .args([
            "--tag",
            TAG,
            "--source-digest",
            SHA,
            "--from-dir",
            dir.join("stage").to_str().unwrap(),
            "--install-dir",
            dir.join("bin3").to_str().unwrap(),
            "--share-dir",
            dir.join("share3").to_str().unwrap(),
            "--no-weights",
        ])
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", dir.join("tools").display()),
        )
        .env("GH_TEST_LOG", dir.join("gh_ok.log"))
        .env("GH_TEST_MODE", "ok")
        .output()
        .unwrap();
    assert!(
        ok_att.status.success(),
        "{}",
        String::from_utf8_lossy(&ok_att.stderr)
    );
    let stdout_att = String::from_utf8_lossy(&ok_att.stdout);
    assert!(stdout_att.contains("sigstore-oidc-attestation + sha256-content-integrity"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn verifier_rejects_gh_below_2_102_0() {
    for version in ["2.45.0", "2.101.9"] {
        let dir = fixture(&format!("old-gh-{version}"));
        let out = Command::new("/bin/bash")
            .arg(root().join("scripts/verify-engine-release.sh"))
            .args([dir.to_str().unwrap(), TAG, SHA])
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", dir.join("tools").display()),
            )
            .env("GH_TEST_LOG", dir.join("gh.log"))
            .env("GH_TEST_MODE", "ok")
            .env("GH_TEST_VERSION", version)
            .output()
            .unwrap();
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{version}: {err}");
        assert!(err.contains("upgrade gh to 2.102.0"), "{version}: {err}");
        assert!(
            !String::from_utf8_lossy(&out.stdout).contains("Verified signed Rust engine release"),
            "{version}"
        );
        assert!(!dir.join("gh.log").exists(), "{version}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}

fn fixture_with_v8(name: &str) -> (PathBuf, String, usize) {
    let dir = std::env::temp_dir().join(format!("dope-release-v8-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("tools")).unwrap();
    let gh = dir.join("tools/gh");
    std::fs::write(
        &gh,
        r#"#!/bin/bash
case "$*" in
  --version) printf 'gh version %s (policy fixture)\n' "${GH_TEST_VERSION:-2.102.0}"; exit 0 ;;
  'attestation verify --help') printf '%s\n' '--repo --signer-workflow --source-ref --source-digest --format --bundle'; exit 0 ;;
esac
printf '%s\n' "$@" >> "$GH_TEST_LOG"
if [[ -n "${GH_REJECT_SUBJECT:-}" && "$*" == *"$GH_REJECT_SUBJECT"* ]]; then exit 1; fi
[[ "${GH_TEST_MODE:-ok}" == ok ]]
"#,
    )
    .unwrap();
    std::fs::set_permissions(&gh, std::fs::Permissions::from_mode(0o755)).unwrap();
    let weights = b"synthetic v8 safetensors fixture; never execute";
    let w_sha = format!("{:x}", Sha256::digest(weights));
    let w_len = weights.len();
    let manifest = serde_json::json!({
        "schema": "dope.single-enc-v8-weights/v1",
        "file": "model.safetensors",
        "arch": "single_enc_v8",
        "bytes": w_len,
        "sha256": w_sha
    })
    .to_string();
    let buildinfo = serde_json::json!({
        "schema_version": 1,
        "product": "dope-engine",
        "version": "3.1.0",
        "source_commit": SHA,
        "v8_weights_sha256": w_sha,
        "v8_weights_bytes": w_len
    })
    .to_string();
    std::fs::write(dir.join("BUILDINFO.json"), buildinfo).unwrap();
    for f in [
        "Cargo.lock",
        "DEPENDENCIES.txt",
        "LICENSE",
        "THIRD_PARTY_NOTICES",
        "dope-linux-x86_64",
    ] {
        std::fs::write(dir.join(f), b"fixture; never execute").unwrap();
    }
    std::fs::write(dir.join("V8_WEIGHTS_MANIFEST.json"), manifest).unwrap();
    std::fs::write(dir.join("dope-v8-weights.safetensors"), weights).unwrap();
    let mut sums = String::new();
    for f in [
        "BUILDINFO.json",
        "Cargo.lock",
        "DEPENDENCIES.txt",
        "LICENSE",
        "THIRD_PARTY_NOTICES",
        "V8_WEIGHTS_MANIFEST.json",
        "dope-linux-x86_64",
        "dope-v8-weights.safetensors",
    ] {
        let b = std::fs::read(dir.join(f)).unwrap();
        sums.push_str(&format!("{:x}  {f}\n", Sha256::digest(&b)));
    }
    std::fs::write(dir.join("SHA256SUMS"), sums).unwrap();
    std::fs::write(dir.join("dope.sigstore.json"), b"{}\n").unwrap();
    (dir, w_sha, w_len)
}

#[test]
fn signed_engine_with_v8_weights_verifies_four_subjects_and_fails_closed() {
    let (dir, _, _) = fixture_with_v8("ok");
    let out = verify(&dir, "dope-engine-v3.1.0", SHA, "ok");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let log = std::fs::read_to_string(dir.join("gh.log")).unwrap();
    assert_eq!(log.matches("attestation\nverify\n").count(), 4);
    assert!(log.contains("dope-v8-weights.safetensors"));
    assert!(log.contains("V8_WEIGHTS_MANIFEST.json"));
    // OIDC failure on weights subject must fail closed.
    let rej = Command::new("/bin/bash")
        .arg(root().join("scripts/verify-engine-release.sh"))
        .args([dir.to_str().unwrap(), "dope-engine-v3.1.0", SHA])
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", dir.join("tools").display()),
        )
        .env("GH_TEST_LOG", dir.join("gh_rej.log"))
        .env("GH_REJECT_SUBJECT", "dope-v8-weights.safetensors")
        .output()
        .unwrap();
    assert!(!rej.status.success());
    // Tampered weights, bad manifest, BUILDINFO mismatch, partial SHA256SUMS, and symlinks fail closed.
    std::fs::write(dir.join("dope-v8-weights.safetensors"), b"tampered").unwrap();
    assert!(!verify(&dir, "dope-engine-v3.1.0", SHA, "ok")
        .status
        .success());
    std::fs::remove_dir_all(&dir).unwrap();

    let (dir, w_sha, w_len) = fixture_with_v8("bad-buildinfo");
    std::fs::write(
        dir.join("BUILDINFO.json"),
        serde_json::json!({"v8_weights_sha256": "0".repeat(64), "v8_weights_bytes": w_len})
            .to_string(),
    )
    .unwrap();
    let mut sums = String::new();
    for f in [
        "BUILDINFO.json",
        "Cargo.lock",
        "DEPENDENCIES.txt",
        "LICENSE",
        "THIRD_PARTY_NOTICES",
        "V8_WEIGHTS_MANIFEST.json",
        "dope-linux-x86_64",
        "dope-v8-weights.safetensors",
    ] {
        let b = std::fs::read(dir.join(f)).unwrap();
        sums.push_str(&format!("{:x}  {f}\n", Sha256::digest(&b)));
    }
    std::fs::write(dir.join("SHA256SUMS"), &sums).unwrap();
    assert!(!verify(&dir, "dope-engine-v3.1.0", SHA, "ok")
        .status
        .success());
    // Partial 7-member inventory must fail closed.
    let partial: String = sums.lines().take(7).map(|l| format!("{l}\n")).collect();
    std::fs::write(dir.join("SHA256SUMS"), partial).unwrap();
    assert!(!verify(&dir, "dope-engine-v3.1.0", SHA, "ok")
        .status
        .success());
    let _ = w_sha;
    std::fs::remove_dir_all(&dir).unwrap();

    // Stray unindexed V8 weights alongside a 6-member weight-free release must fail closed.
    let stray = fixture("stray-v8");
    std::fs::write(stray.join("dope-v8-weights.safetensors"), b"stray").unwrap();
    let out = verify(&stray, TAG, SHA, "ok");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("stray_v8_weights"));
    std::fs::remove_dir_all(&stray).unwrap();
}

#[test]
fn engine_release_preflight_refuses_bad_v8_weights_overrides_before_cargo() {
    let dir = std::env::temp_dir().join(format!("dope-rel-preflight-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("tools")).unwrap();
    for tool in ["cargo", "rustc"] {
        let p = dir.join("tools").join(tool);
        std::fs::write(&p, "#!/bin/bash\nexit 99\n").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let path_env = format!("{}:/usr/bin:/bin", dir.join("tools").display());
    let out_dir = dir.join("out");
    let run = |extra: &[(&str, &str)]| {
        let mut cmd = Command::new("/bin/bash");
        cmd.arg(root().join("ops/ci/engine-release.sh"))
            .arg(&out_dir)
            .env("PATH", &path_env);
        for (k, v) in extra {
            cmd.env(k, v);
        }
        cmd.output().unwrap()
    };
    let on_tag = run(&[
        ("ENGINE_RELEASE_TAG", "dope-engine-v3.1.0"),
        ("DOPE_V8_WEIGHTS_SOURCE_DIR", dir.to_str().unwrap()),
    ]);
    assert_eq!(on_tag.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&on_tag.stderr).contains("override_forbidden_on_tag"));
    let missing = run(&[(
        "DOPE_V8_WEIGHTS_SOURCE_DIR",
        dir.join("missing").to_str().unwrap(),
    )]);
    assert_eq!(missing.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&missing.stderr).contains("override_missing"));
    let bad = dir.join("bad-v8");
    std::fs::create_dir_all(&bad).unwrap();
    std::fs::write(bad.join("manifest.json"), "{}").unwrap();
    let incomplete = run(&[("DOPE_V8_WEIGHTS_SOURCE_DIR", bad.to_str().unwrap())]);
    assert_eq!(incomplete.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&incomplete.stderr).contains("v8_weights_incomplete"));
    std::fs::write(bad.join("model.safetensors"), b"bytes").unwrap();
    let bad_manifest = run(&[("DOPE_V8_WEIGHTS_SOURCE_DIR", bad.to_str().unwrap())]);
    assert_eq!(bad_manifest.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&bad_manifest.stderr).contains("manifest_invalid"));
    std::fs::remove_dir_all(&dir).unwrap();
}

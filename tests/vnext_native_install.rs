use std::fs;
use std::process::Command;

#[test]
fn pinned_native_install_and_doctor_work_with_an_activated_v1_environment() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/native-install")
        .join(std::process::id().to_string());
    fs::create_dir_all(&root).unwrap();
    let binary = std::path::Path::new(env!("CARGO_BIN_EXE_dope-kernel"));
    let bytes = fs::read(binary).unwrap();
    let checksum = dope_kernel::bundle::content_fingerprint(&bytes);
    let prefix = root.join("native prefix with spaces");
    let installer =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/install-native.sh");
    let failed = Command::new(&installer)
        .arg(binary)
        .arg("0".repeat(64))
        .arg(&prefix)
        .output()
        .unwrap();
    assert!(!failed.status.success());
    assert!(!prefix.exists());
    let dry_run = Command::new(&installer)
        .arg(binary)
        .arg(&checksum)
        .arg(&prefix)
        .arg("--dry-run")
        .output()
        .unwrap();
    assert!(dry_run.status.success());
    assert!(!prefix.exists());
    let installed = Command::new(&installer)
        .arg(binary)
        .arg(&checksum)
        .arg(&prefix)
        .output()
        .unwrap();
    assert!(
        installed.status.success(),
        "{}",
        String::from_utf8_lossy(&installed.stderr)
    );
    let installed_binary = prefix.join("bin/dope-kernel");
    assert_eq!(fs::read(&installed_binary).unwrap(), bytes);
    let package = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let installed_license = prefix.join("share/dope-kernel/LICENSE");
    let installed_notices = prefix.join("share/dope-kernel/THIRD_PARTY_NOTICES");
    let installed_manifest = prefix.join("share/dope-kernel/installed.sha256");
    assert_eq!(
        fs::read(&installed_license).unwrap(),
        fs::read(package.join("LICENSE")).unwrap()
    );
    assert_eq!(
        fs::read(&installed_notices).unwrap(),
        fs::read(package.join("scripts/native-third-party.NOTICE")).unwrap()
    );
    assert_eq!(
        fs::read_to_string(&installed_manifest)
            .unwrap()
            .lines()
            .count(),
        4
    );
    let activated = root.join("v1-environment");
    fs::create_dir_all(activated.join("bin")).unwrap();
    fs::write(
        activated.join("bin/dope-kernel"),
        b"#!/bin/sh\necho old_python_console_collision\n",
    )
    .unwrap();
    for venv in [None, Some(&activated)] {
        let mut command = Command::new(&installed_binary);
        command.env_clear().arg("doctor");
        if let Some(path) = venv {
            command
                .env("VIRTUAL_ENV", path)
                .env("PATH", path.join("bin"));
        }
        let output = command.output().unwrap();
        assert!(output.status.success());
        let doctor: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(doctor["implementation"], "native_rust");
        assert_eq!(doctor["executable_sha256"], checksum);
        assert_eq!(doctor["weights_downloaded"], false);
    }
    let duplicate = Command::new(&installer)
        .arg(binary)
        .arg(&checksum)
        .arg(&prefix)
        .output()
        .unwrap();
    assert!(!duplicate.status.success());
    assert_eq!(fs::read(&installed_binary).unwrap(), bytes);
    fs::write(root.join("install-doctor.json"), installed.stdout).unwrap();
    let uninstaller =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/uninstall-native.sh");
    let unrelated = prefix.join("bin/unrelated");
    fs::write(&unrelated, b"preserved").unwrap();
    assert!(
        Command::new(&uninstaller)
            .arg(&prefix)
            .arg("--dry-run")
            .status()
            .unwrap()
            .success()
    );
    assert!(installed_binary.exists());
    // A changed notice or injected manifest path cannot turn uninstall into an
    // arbitrary file deletion. Validate every member before removing any.
    let notice_bytes = fs::read(&installed_notices).unwrap();
    fs::write(&installed_notices, b"modified_notice").unwrap();
    assert!(
        !Command::new(&uninstaller)
            .arg(&prefix)
            .status()
            .unwrap()
            .success()
    );
    assert!(installed_binary.exists());
    assert!(installed_license.exists());
    fs::write(&installed_notices, notice_bytes).unwrap();
    let manifest_bytes = fs::read(&installed_manifest).unwrap();
    fs::write(
        &installed_manifest,
        format!("{}  ../../bin/unrelated\n", "0".repeat(64)),
    )
    .unwrap();
    assert!(
        !Command::new(&uninstaller)
            .arg(&prefix)
            .status()
            .unwrap()
            .success()
    );
    assert_eq!(fs::read(&unrelated).unwrap(), b"preserved");
    assert!(installed_binary.exists());
    fs::write(&installed_manifest, manifest_bytes).unwrap();
    let receipt = prefix.join("share/dope-kernel/native.sha256");
    fs::write(&receipt, "0".repeat(64)).unwrap();
    assert!(
        !Command::new(&uninstaller)
            .arg(&prefix)
            .status()
            .unwrap()
            .success()
    );
    assert!(installed_binary.exists());
    fs::write(&receipt, format!("{checksum}\n")).unwrap();
    assert!(
        Command::new(&uninstaller)
            .arg(&prefix)
            .status()
            .unwrap()
            .success()
    );
    assert!(!installed_binary.exists());
    assert!(!receipt.exists());
    assert!(!installed_license.exists());
    assert!(!installed_notices.exists());
    assert!(!installed_manifest.exists());
    assert_eq!(fs::read(unrelated).unwrap(), b"preserved");
    assert!(activated.join("bin/dope-kernel").exists());
}

#[test]
fn tampered_package_notice_refuses_before_any_prefix_write() {
    let package = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = package
        .join("target/native-install")
        .join(format!("{}-tamper", std::process::id()));
    fs::create_dir_all(root.join("scripts")).unwrap();
    for relative in [
        "LICENSE",
        "Cargo.lock",
        "rust-toolchain.toml",
        "scripts/install-native.sh",
        "scripts/native-legal.sha256",
        "scripts/native-third-party.NOTICE",
    ] {
        fs::copy(package.join(relative), root.join(relative)).unwrap();
    }
    fs::write(root.join("scripts/native-third-party.NOTICE"), b"altered").unwrap();
    let binary = std::path::Path::new(env!("CARGO_BIN_EXE_dope-kernel"));
    let checksum = dope_kernel::bundle::content_fingerprint(&fs::read(binary).unwrap());
    let prefix = root.join("untouched prefix with spaces");
    let result = Command::new(root.join("scripts/install-native.sh"))
        .arg(binary)
        .arg(checksum)
        .arg(&prefix)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(
        String::from_utf8_lossy(&result.stderr).trim(),
        "native_install_legal_inputs_checksum_mismatch"
    );
    assert!(!prefix.exists());
}

#[test]
fn legacy_two_file_uninstall_preserves_unrelated_assets() {
    let package = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let prefix = package
        .join("target/native-install")
        .join(format!("{}-legacy", std::process::id()));
    fs::create_dir_all(prefix.join("bin")).unwrap();
    fs::create_dir_all(prefix.join("share/dope-kernel")).unwrap();
    let binary = std::path::Path::new(env!("CARGO_BIN_EXE_dope-kernel"));
    let checksum = dope_kernel::bundle::content_fingerprint(&fs::read(binary).unwrap());
    fs::copy(binary, prefix.join("bin/dope-kernel")).unwrap();
    fs::write(
        prefix.join("share/dope-kernel/native.sha256"),
        format!("{checksum}\n"),
    )
    .unwrap();
    fs::write(prefix.join("share/dope-kernel/unrelated"), b"keep").unwrap();
    assert!(
        Command::new(package.join("scripts/uninstall-native.sh"))
            .arg(&prefix)
            .status()
            .unwrap()
            .success()
    );
    assert!(!prefix.join("bin/dope-kernel").exists());
    assert_eq!(
        fs::read(prefix.join("share/dope-kernel/unrelated")).unwrap(),
        b"keep"
    );
}

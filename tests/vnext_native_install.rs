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
    assert_eq!(fs::read(unrelated).unwrap(), b"preserved");
    assert!(activated.join("bin/dope-kernel").exists());
}

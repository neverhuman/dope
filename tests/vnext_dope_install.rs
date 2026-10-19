use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::process::Command;

#[test]
fn facade_doctor_binds_actual_executable_build_inputs_and_cpu_profile() {
    let executable = env!("CARGO_BIN_EXE_dope");
    let output = Command::new(executable).arg("doctor").output().unwrap();
    assert!(output.status.success());
    let doctor: Value = serde_json::from_slice(&output.stdout).unwrap();
    let bytes = fs::read(executable).unwrap();
    assert_eq!(doctor["name"], "dope");
    assert_eq!(doctor["implementation"], "native_rust");
    assert_eq!(
        doctor["executable_sha256"],
        format!("{:x}", Sha256::digest(&bytes))
    );
    assert_eq!(doctor["executable_bytes"], bytes.len() as u64);
    assert_eq!(doctor["os"], std::env::consts::OS);
    assert_eq!(doctor["architecture"], std::env::consts::ARCH);
    assert_eq!(
        doctor["build_cargo_lock_sha256"],
        format!("{:x}", Sha256::digest(include_bytes!("../Cargo.lock")))
    );
    assert_eq!(
        doctor["declared_rust_toolchain_sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../rust-toolchain.toml"))
        )
    );
    assert_eq!(doctor["gpu_training"], cfg!(feature = "gpu-training"));
    assert_eq!(doctor["candidate_evaluation"], "unavailable");
    assert_eq!(doctor["source_free_isolation"], "not_claimed_by_facade");
    assert_eq!(doctor["formal_dp_state"], "unverified");
    assert_eq!(doctor["legal_release_state"], "pending");
    assert_eq!(doctor["weights_downloaded"], false);
}

#[test]
fn facade_identity_does_not_enable_network_access() {
    let output = Command::new(env!("CARGO_BIN_EXE_dope"))
        .args(["doctor", "--network"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(3));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("network_access_refused"));
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod installer {
    use super::*;
    use serde_json::json;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::path::{Path, PathBuf};
    use std::process::Output;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Fixture {
        root: PathBuf,
        bundle: PathBuf,
        prefix: PathBuf,
    }
    fn digest(path: &Path) -> String {
        format!("{:x}", Sha256::digest(fs::read(path).unwrap()))
    }
    fn fixture(name: &str) -> Fixture {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/dope-install")
            .join(format!(
                "{}-{}-{}",
                name,
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        let bundle = root.join("bundle");
        fs::create_dir_all(bundle.join("scripts")).unwrap();
        for (name, bytes) in [
            ("LICENSE", include_bytes!("../LICENSE").as_slice()),
            ("Cargo.lock", include_bytes!("../Cargo.lock").as_slice()),
            (
                "rust-toolchain.toml",
                include_bytes!("../rust-toolchain.toml").as_slice(),
            ),
            (
                "scripts/native-legal.sha256",
                include_bytes!("../scripts/native-legal.sha256").as_slice(),
            ),
            (
                "scripts/native-third-party.NOTICE",
                include_bytes!("../scripts/native-third-party.NOTICE").as_slice(),
            ),
            (
                "scripts/install-dope.sh",
                include_bytes!("../scripts/install-dope.sh").as_slice(),
            ),
            (
                "scripts/uninstall-dope.sh",
                include_bytes!("../scripts/uninstall-dope.sh").as_slice(),
            ),
        ] {
            fs::write(bundle.join(name), bytes).unwrap();
        }
        let executable = std::env::var_os("DOPE_INSTALL_TEST_BINARY")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_BIN_EXE_dope")));
        fs::copy(executable, bundle.join("dope")).unwrap();
        fs::set_permissions(bundle.join("dope"), fs::Permissions::from_mode(0o755)).unwrap();
        Fixture {
            prefix: root.join("prefix"),
            root,
            bundle,
        }
    }
    fn install(f: &Fixture, binary: &Path, hash: &str, dry: bool) -> Output {
        let mut c = Command::new("/bin/bash");
        c.arg(f.bundle.join("scripts/install-dope.sh"))
            .arg(binary)
            .arg(hash)
            .arg(&f.prefix);
        if dry {
            c.arg("--dry-run");
        }
        c.output().unwrap()
    }
    fn uninstall(f: &Fixture, dry: bool) -> Output {
        let mut c = Command::new("/bin/bash");
        c.arg(f.bundle.join("scripts/uninstall-dope.sh"))
            .arg(&f.prefix);
        if dry {
            c.arg("--dry-run");
        }
        c.output().unwrap()
    }
    fn accepted(output: Output) {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    fn installed(f: &Fixture) {
        let bin = f.bundle.join("dope");
        accepted(install(f, &bin, &digest(&bin), false));
    }

    #[test]
    fn installed_facade_bundle_runs_100_rows_and_preserves_unrelated_files() {
        let f = fixture("recipient");
        let bin = f.bundle.join("dope");
        accepted(install(&f, &bin, &digest(&bin), true));
        assert!(!f.prefix.exists(), "dry run mutated the prefix");
        fs::create_dir_all(&f.prefix).unwrap();
        fs::write(f.prefix.join("unrelated-evidence"), b"preserve").unwrap();
        installed(&f);
        let recipient = f.root.join("empty-recipient");
        let empty_path = f.root.join("empty-path");
        fs::create_dir_all(&recipient).unwrap();
        fs::create_dir_all(&empty_path).unwrap();
        let dope = f.prefix.join("bin/dope");
        let run = |args: &[&str]| {
            Command::new(&dope)
                .args(args)
                .current_dir(&recipient)
                .env_clear()
                .env("PATH", &empty_path)
                .env("RAYON_NUM_THREADS", "1")
                .env("CUDA_VISIBLE_DEVICES", "")
                .output()
                .unwrap()
        };
        let doctor = run(&["doctor"]);
        assert!(doctor.status.success());
        let identity: Value = serde_json::from_slice(&doctor.stdout).unwrap();
        assert_eq!(identity["executable_sha256"], digest(&dope));
        assert_eq!(identity["gpu_training"], false);
        accepted(run(&[
            "demo",
            "--out",
            "artifact.dpk",
            "--report",
            "demo.json",
        ]));
        let inspect = run(&["inspect", "--artifact", "artifact.dpk"]);
        assert!(inspect.status.success());
        let report: Value = serde_json::from_slice(&inspect.stdout).unwrap();
        assert!(report.as_object().is_some_and(|object| !object.is_empty()));
        let demo: Value =
            serde_json::from_slice(&fs::read(recipient.join("demo.json")).unwrap()).unwrap();
        assert_eq!(demo["format"], "dope-demo-receipt");
        assert_eq!(demo["fixture_only"], true);
        assert_eq!(demo["certified"], false);
        assert_eq!(demo["qualified"], false);
        assert_eq!(
            demo["artifact_bytes"],
            fs::metadata(recipient.join("artifact.dpk")).unwrap().len()
        );
        accepted(run(&[
            "sample",
            "--artifact",
            "artifact.dpk",
            "--rows",
            "100",
            "--seed",
            "7",
            "--out",
            "sample.csv",
        ]));
        let sample = fs::read_to_string(recipient.join("sample.csv")).unwrap();
        assert_eq!(sample.lines().count(), 100);
        for line in sample.lines() {
            for value in line.split(',') {
                assert!(value.parse::<f64>().unwrap().is_finite());
            }
        }
        fs::write(recipient.join("corrupt.dpk"), b"corrupt fixture").unwrap();
        assert!(
            !run(&[
                "sample",
                "--artifact",
                "corrupt.dpk",
                "--rows",
                "100",
                "--out",
                "rejected.csv"
            ])
            .status
            .success()
        );
        assert!(!recipient.join("rejected.csv").exists());
        let receipt = json!({"format":"dope-installed-facade-fixture-v1","binary_sha256":digest(&dope),
            "artifact_sha256":digest(&recipient.join("artifact.dpk")),"sample_sha256":digest(&recipient.join("sample.csv")),
            "rows":100,"empty_recipient_cwd":true,"python_absent_from_PATH":true,"gpu_training":false,
            "host_repository_unmounted":false,"corrupt_artifact_refused":true,"scientific_or_privacy_qualification":false});
        let receipt_path = std::env::var_os("DOPE_INSTALL_TEST_RECEIPT")
            .map(PathBuf::from)
            .unwrap_or_else(|| f.root.join("receipt.json"));
        use std::io::Write;
        let mut out = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&receipt_path)
            .unwrap();
        writeln!(out, "{}", serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
        accepted(uninstall(&f, true));
        assert!(dope.is_file());
        accepted(uninstall(&f, false));
        assert!(!dope.exists());
        assert_eq!(
            fs::read(f.prefix.join("unrelated-evidence")).unwrap(),
            b"preserve"
        );
        assert!(f.prefix.is_dir());
    }

    #[test]
    fn bad_digest_symlink_and_wrong_product_fail_before_destination_mutation() {
        let f = fixture("input-refusals");
        let bin = f.bundle.join("dope");
        assert!(!install(&f, &bin, &"0".repeat(64), false).status.success());
        assert!(!f.prefix.exists());
        let alias = f.root.join("binary-alias");
        symlink(&bin, &alias).unwrap();
        assert!(!install(&f, &alias, &digest(&bin), false).status.success());
        assert!(!f.prefix.exists());
        let kernel = f.root.join("wrong-product");
        fs::copy(env!("CARGO_BIN_EXE_dope-kernel"), &kernel).unwrap();
        assert!(
            !install(&f, &kernel, &digest(&kernel), false)
                .status
                .success()
        );
        assert!(!f.prefix.exists());
    }

    #[test]
    fn existing_destination_and_symlink_ancestor_are_preserved() {
        let f = fixture("destination");
        fs::create_dir_all(f.prefix.join("bin")).unwrap();
        let existing = f.prefix.join("bin/dope");
        fs::write(&existing, b"preserve-existing").unwrap();
        let bin = f.bundle.join("dope");
        assert!(!install(&f, &bin, &digest(&bin), false).status.success());
        assert_eq!(fs::read(existing).unwrap(), b"preserve-existing");
        assert!(!f.prefix.join("share").exists());
        let mut g = fixture("ancestor");
        let actual = g.root.join("actual");
        fs::create_dir(&actual).unwrap();
        let alias = g.root.join("alias");
        symlink(&actual, &alias).unwrap();
        g.prefix = alias.join("nested/prefix");
        let bin = g.bundle.join("dope");
        assert!(!install(&g, &bin, &digest(&bin), false).status.success());
        assert!(!actual.join("nested").exists());
    }

    #[test]
    fn untrusted_manifest_paths_and_oversized_manifests_fail_before_mutation() {
        let f = fixture("manifest-path");
        let external = f.root.join("external-sentinel");
        fs::write(&external, b"outside-package-sentinel").unwrap();
        let manifest = f.bundle.join("scripts/native-legal.sha256");
        let original = fs::read_to_string(&manifest).unwrap();
        let mut lines: Vec<String> = original.lines().map(str::to_owned).collect();
        lines[0] = format!("{}  {}", digest(&external), external.display());
        fs::write(&manifest, format!("{}\n", lines.join("\n"))).unwrap();
        let bin = f.bundle.join("dope");
        let rejected = install(&f, &bin, &digest(&bin), false);
        assert!(!rejected.status.success());
        assert!(
            String::from_utf8_lossy(&rejected.stderr)
                .contains("dope_install_legal_manifest_invalid")
        );
        assert!(!f.prefix.exists());
        assert_eq!(fs::read(&external).unwrap(), b"outside-package-sentinel");

        fs::write(&manifest, format!("{}{}", original, "x".repeat(2048))).unwrap();
        let rejected = install(&f, &bin, &digest(&bin), false);
        assert!(!rejected.status.success());
        assert!(
            String::from_utf8_lossy(&rejected.stderr)
                .contains("dope_install_legal_manifest_invalid")
        );
        assert!(!f.prefix.exists());
    }

    #[test]
    fn incomplete_legal_input_and_modified_installed_members_are_refused() {
        let f = fixture("legal-input");
        fs::write(f.bundle.join("LICENSE"), b"changed").unwrap();
        let bin = f.bundle.join("dope");
        assert!(!install(&f, &bin, &digest(&bin), false).status.success());
        assert!(!f.prefix.exists());
        let f = fixture("modified-legal");
        installed(&f);
        fs::write(f.prefix.join("share/dope/LICENSE"), b"modified").unwrap();
        assert!(!uninstall(&f, false).status.success());
        assert!(f.prefix.join("bin/dope").is_file());
        let f = fixture("missing-manifest");
        installed(&f);
        fs::remove_file(f.prefix.join("share/dope/installed.sha256")).unwrap();
        assert!(!uninstall(&f, false).status.success());
        assert!(f.prefix.join("bin/dope").is_file());
        let f = fixture("modified-binary");
        installed(&f);
        fs::write(f.prefix.join("bin/dope"), b"modified").unwrap();
        assert!(!uninstall(&f, false).status.success());
        assert_eq!(fs::read(f.prefix.join("bin/dope")).unwrap(), b"modified");
    }
}

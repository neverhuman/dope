//! Run the public install and walkthrough commands against the published v3.0.0 release.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const README: &str = include_str!("../../../README.md");
const GETTING_STARTED: &str = include_str!("../../../docs/getting-started.md");
const SIGNED_RELEASE: &str = include_str!("../../../docs/signed-engine-release.md");
const INSTALLER: &[u8] = include_bytes!("../../../scripts/install.sh");
const INSTALLER_URL: &str = "https://raw.githubusercontent.com/neverhuman/dope/51df4554bc292c3ad59ec2099fccbbde19cffadc/scripts/install.sh";
const RELEASE_DIGEST: &str = "8723955b1fcd59cab8996842a39b3a579c596c957ac1f89f291b0a9d1ebfaea0";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn success(output: Output) -> String {
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        output.status.success(),
        "{:?}\n{stdout}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    stdout
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct Workspace {
    root: PathBuf,
    home: PathBuf,
}

impl Workspace {
    fn new() -> Self {
        let target = repo().join("target");
        fs::create_dir_all(&target).unwrap();
        let dir = success(
            Command::new("mktemp")
                .args(["-d", "--"])
                .arg(target.join("readme-install-smoke.XXXXXX"))
                .output()
                .unwrap(),
        );
        let root = PathBuf::from(dir.trim());
        let home = root.join("home");
        fs::create_dir(&home).unwrap();
        println!("Smoke workspace {}", root.display());
        Self { root, home }
    }

    fn command(&self, program: impl AsRef<Path>) -> Command {
        let mut cmd = Command::new(program.as_ref());
        cmd.env_clear()
            .env("HOME", &self.home)
            .env("TMPDIR", &self.root)
            .env("PATH", "/usr/bin:/bin")
            .env("LC_ALL", "C")
            .current_dir(&self.root);
        cmd
    }

    fn binary(&self) -> PathBuf {
        self.home.join(".local/opt/dope/3.0.0/bin/dope")
    }
}

fn bash_blocks(section: &str) -> [&str; 2] {
    let blocks: Vec<_> = section
        .split("```bash\n")
        .skip(1)
        .map(|block| block.split_once("\n```").unwrap().0)
        .collect();
    blocks.try_into().unwrap()
}

fn assert_release_binary(workspace: &Workspace) {
    let binary = workspace.binary();
    assert_eq!(digest(&fs::read(&binary).unwrap()), RELEASE_DIGEST);
    assert_eq!(
        success(
            workspace
                .command(&binary)
                .arg("--version")
                .output()
                .unwrap()
        )
        .trim(),
        "dope 3.0.0"
    );
}

fn run_manual_install() {
    let workspace = Workspace::new();
    let archive = "gh_2.102.0_linux_amd64.tar.gz";
    let verifier = "gh_2.102.0_linux_amd64/bin/gh";
    success(
        workspace
            .command("/usr/bin/curl")
            .args([
                "-q",
                "-fsSL",
                "--proto",
                "=https",
                "--tlsv1.2",
                "-o",
                archive,
            ])
            .arg(format!(
                "https://github.com/cli/cli/releases/download/v2.102.0/{archive}"
            ))
            .output()
            .unwrap(),
    );
    assert_eq!(
        digest(&fs::read(workspace.root.join(archive)).unwrap()),
        "bb766f710eef8ede859c18578c72c327597cd4c8a85b06001b1f3843c6019386"
    );
    success(
        workspace
            .command("/bin/tar")
            .args(["-xzf", archive, verifier])
            .output()
            .unwrap(),
    );
    let verifier_path = workspace.root.join(verifier);
    assert_eq!(
        digest(&fs::read(&verifier_path).unwrap()),
        "7469124f706944133d6a169691dd1c6c3511b12e85878d255e044e2948df4c9b"
    );
    let path = std::env::join_paths([
        verifier_path.parent().unwrap(),
        Path::new("/usr/bin"),
        Path::new("/bin"),
    ])
    .unwrap();
    symlink(repo().join("scripts"), workspace.root.join("scripts")).unwrap();
    let (_, rest) = SIGNED_RELEASE
        .split_once("## Verify before install\n")
        .unwrap();
    let (section, _) = rest.split_once("## Checks\n").unwrap();
    let script = bash_blocks(section).join("\n");
    print!(
        "{}",
        success(
            workspace
                .command("/bin/bash")
                .env("PATH", path)
                .args(["-euo", "pipefail", "-c", &script])
                .output()
                .unwrap()
        )
    );
    assert_release_binary(&workspace);
}

fn rows(csv: &str, count: usize) -> HashSet<&str> {
    let mut lines = csv.lines();
    assert_eq!(lines.next(), Some("a,b,y"));
    let records: Vec<_> = lines.collect();
    assert_eq!(records.len(), count);
    assert!(records.iter().all(|line| line.split(',').count() == 3));
    records.into_iter().collect()
}

fn assert_evaluation(work: &Path, artifact_bytes: u64, synth_rows: usize) {
    let eval: Value =
        serde_json::from_slice(&fs::read(work.join("evaluation.json")).unwrap()).unwrap();
    assert_eq!(eval["artifact_bytes"], artifact_bytes);
    assert_eq!(eval["train_rows"], 160);
    assert_eq!(eval["test_rows"], 40);
    assert_eq!(eval["synth_rows"], synth_rows);
    assert_eq!(eval["auditors"].as_array().unwrap().len(), 1);
    let linear = &eval["auditors"][0];
    assert_eq!(linear["auditor"], "linear");
    for field in ["null_loss", "real_loss", "synth_loss", "retention"] {
        assert!(
            linear[field].as_f64().is_some_and(f64::is_finite),
            "{field}"
        );
    }
}

fn run_own_data_walkthrough(workspace: &Workspace, train: &str, test: &str) {
    let (_, rest) = GETTING_STARTED.split_once("## Fit a table\n").unwrap();
    let (section, _) = rest
        .split_once("## Historical kernel demonstration\n")
        .unwrap();
    let [fit_and_sample, evaluate] = bash_blocks(section);
    let work = workspace.root.join("own-data");
    fs::create_dir(&work).unwrap();
    fs::write(work.join("train.csv"), train).unwrap();
    fs::write(work.join("test.csv"), test).unwrap();
    // Run both blocks in the same shell, just as a reader would paste them.
    let script = [fit_and_sample, evaluate].join("\n");
    print!(
        "{}",
        success(
            workspace
                .command("/bin/bash")
                .current_dir(&work)
                .env("RAYON_NUM_THREADS", "1")
                .args(["-euo", "pipefail", "-c", &script])
                .output()
                .unwrap()
        )
    );

    let model = fs::read(work.join("model.dope")).unwrap();
    assert!(model.starts_with(b"DPA1"));
    assert!((1..=2048).contains(&model.len()));
    let audit_schema = fs::read(work.join("audit-schema.dope")).unwrap();
    assert!(!audit_schema.starts_with(b"DPA1"));
    assert!((1..=2048).contains(&audit_schema.len()));
    let synthetic = fs::read_to_string(work.join("synthetic.csv")).unwrap();
    rows(&synthetic, 1000);
    assert_evaluation(&work, audit_schema.len() as u64, 1000);
}

#[test]
#[ignore = "requires public GitHub and Sigstore access on Linux x86-64 with glibc 2.39+"]
fn published_readme_quick_start() {
    let workspace = Workspace::new();
    let (_, rest) = README.split_once("## Quick start\n").unwrap();
    let (section, _) = rest.split_once("## Results\n").unwrap();
    let [install, demo] = bash_blocks(section);
    assert!(install.contains(INSTALLER_URL));
    assert!(install.contains(&digest(INSTALLER)));
    for block in [install, demo] {
        print!(
            "{}",
            success(
                workspace
                    .command("/bin/bash")
                    .args(["-c", block])
                    .output()
                    .unwrap()
            )
        );
    }

    assert_release_binary(&workspace);
    let binary = workspace.binary();
    let work = fs::read_dir(&workspace.root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("dope-quick-start.")
        })
        .unwrap();
    let artifact = work.join("model.dope");
    let size = fs::metadata(&artifact).unwrap().len();
    assert!((1..=2048).contains(&size));
    let train = fs::read_to_string(work.join("train.csv")).unwrap();
    let test = fs::read_to_string(work.join("test.csv")).unwrap();
    assert!(rows(&train, 160).is_disjoint(&rows(&test, 40)));
    let synthetic = fs::read_to_string(work.join("synthetic.csv")).unwrap();
    rows(&synthetic, 640);

    assert_evaluation(&work, size, 640);

    let replay = workspace.root.join("replay");
    fs::create_dir(&replay).unwrap();
    fs::copy(artifact, replay.join("model.dope")).unwrap();
    assert_eq!(fs::read_dir(&replay).unwrap().count(), 1);
    success(
        workspace
            .command(&binary)
            .current_dir(&replay)
            .args([
                "--threads",
                "1",
                "sample",
                "model.dope",
                "-n",
                "640",
                "--seed",
                "7",
                "-o",
                "replay.csv",
            ])
            .output()
            .unwrap(),
    );
    assert_eq!(
        fs::read(replay.join("replay.csv")).unwrap(),
        synthetic.as_bytes()
    );

    run_own_data_walkthrough(&workspace, &train, &test);
    run_manual_install();
}

#[test]
#[ignore = "run on Ubuntu 22.04 with actual glibc 2.35"]
fn ubuntu_22_04_refuses_before_download() {
    let workspace = Workspace::new();
    let libc = success(
        workspace
            .command("getconf")
            .arg("GNU_LIBC_VERSION")
            .output()
            .unwrap(),
    );
    assert_eq!(libc.trim(), "glibc 2.35");
    let curl = workspace.root.join("curl");
    fs::write(
        &curl,
        "#!/bin/bash\nprintf attempted > \"$TMPDIR/download-attempted\"\nexit 91\n",
    )
    .unwrap();
    fs::set_permissions(curl, fs::Permissions::from_mode(0o755)).unwrap();
    let output = workspace
        .command("/bin/bash")
        .arg(repo().join("scripts/install.sh"))
        .arg("--standalone-prefix")
        .arg(workspace.binary().parent().unwrap().parent().unwrap())
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", workspace.root.display()),
        )
        .output()
        .unwrap();
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(error.contains("requires glibc 2.39 or newer"), "{error}");
    assert!(!workspace.root.join("download-attempted").exists());
    assert!(!workspace.home.join(".local").exists());
    print!("{error}");
}

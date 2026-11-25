//! Run the verbatim README commands against the published v3.0.0 release.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const README: &str = include_str!("../../../README.md");
const INSTALLER: &[u8] = include_bytes!("../../../scripts/install.sh");
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

fn quick_start() -> [&'static str; 2] {
    let (_, rest) = README.split_once("## Quick start\n").unwrap();
    let (section, _) = rest.split_once("## Results\n").unwrap();
    let blocks: Vec<_> = section
        .split("```bash\n")
        .skip(1)
        .map(|block| block.split_once("\n```").unwrap().0)
        .collect();
    blocks.try_into().unwrap()
}

fn rows(csv: &str, count: usize) -> HashSet<&str> {
    let mut lines = csv.lines();
    assert_eq!(lines.next(), Some("a,b,y"));
    let records: Vec<_> = lines.collect();
    assert_eq!(records.len(), count);
    assert!(records.iter().all(|line| line.split(',').count() == 3));
    records.into_iter().collect()
}

#[test]
#[ignore = "requires public GitHub and Sigstore access on Linux x86-64 with glibc 2.39+"]
fn published_readme_quick_start() {
    let workspace = Workspace::new();
    let [install, demo] = quick_start();
    let blob = success(
        Command::new("git")
            .args(["hash-object", "scripts/install.sh"])
            .current_dir(repo())
            .output()
            .unwrap(),
    );
    assert!(install.contains(&format!(
        "https://api.github.com/repos/neverhuman/dope/git/blobs/{}",
        blob.trim()
    )));
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

    let eval: Value =
        serde_json::from_slice(&fs::read(work.join("evaluation.json")).unwrap()).unwrap();
    assert_eq!(eval["artifact_bytes"], size);
    assert_eq!(eval["train_rows"], 160);
    assert_eq!(eval["test_rows"], 40);
    assert_eq!(eval["synth_rows"], 640);
    assert_eq!(eval["auditors"].as_array().unwrap().len(), 1);
    let linear = &eval["auditors"][0];
    assert_eq!(linear["auditor"], "linear");
    for field in ["null_loss", "real_loss", "synth_loss", "retention"] {
        assert!(
            linear[field].as_f64().is_some_and(f64::is_finite),
            "{field}"
        );
    }

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

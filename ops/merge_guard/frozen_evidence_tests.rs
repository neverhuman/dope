use super::*;
use serde_json::{Value, json};
use std::ffi::OsString;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

const PATHS: [&str; 6] = [
    "research/benchmark/results/fixture/panel.json",
    "docs/whitepaper/generated/fixture.tex",
    "research/benchmark/methods.lock.json",
    "research/benchmark/review_fixes/predeclare.json",
    "research/benchmark/review_fixes/predeclare_v2.json",
    "ops/ci/paper-review-inputs.json",
];
const ORIGINAL: &[u8] = b"{\"host\":\"historical-host\",\"retention\":0.9}\n";
const SCRUBBED: &[u8] = b"{\"host\":\"public-placeholder\",\"retention\":0.9}\n";

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

struct SyntheticRepo {
    root: PathBuf,
}

impl SyntheticRepo {
    fn new() -> Self {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("target/frozen-evidence-rust-tests");
        fs::create_dir_all(&base).expect("create ignored synthetic-test parent");
        let base = base
            .canonicalize()
            .expect("canonical synthetic-test parent");
        let root = loop {
            let name = format!(
                "repo-{}-{}",
                std::process::id(),
                NEXT_ID.fetch_add(1, Ordering::Relaxed)
            );
            let candidate = base.join(name);
            match fs::create_dir(&candidate) {
                Ok(()) => break candidate,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create synthetic repo: {error}"),
            }
        };
        let repo = Self { root };
        repo.git(&["init", "-q"]);
        for path in PATHS {
            repo.write(path, ORIGINAL);
        }
        repo.write(".gitignore", b"__pycache__/\n*.pyc\n");
        repo.git(&["add", "."]);
        repo
    }

    fn git(&self, args: &[&str]) {
        let output = Command::new("git")
            .args(args)
            .current_dir(&self.root)
            .output()
            .expect("start git for synthetic repository");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn write(&self, name: &str, bytes: &[u8]) {
        let path = self.root.join(name);
        fs::create_dir_all(path.parent().expect("fixture parent")).expect("create fixture parent");
        fs::write(path, bytes).expect("write synthetic fixture");
    }

    fn baseline(&self) -> Hashes {
        PATHS
            .iter()
            .map(|name| {
                (
                    (*name).to_owned(),
                    format!("{:x}", Sha256::digest(ORIGINAL)),
                )
            })
            .collect()
    }

    fn check(&self, allowlist: Option<&Value>) -> Vec<String> {
        let allowlist = allowlist.cloned().unwrap_or_else(|| json!({}));
        check_tree(
            &self.root,
            &self.baseline(),
            &replacement_hashes(&allowlist).unwrap(),
        )
        .expect("synthetic inventory and file reads")
    }
}

impl Drop for SyntheticRepo {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn has_failure(failures: &[String], fragment: &str) -> bool {
    failures.iter().any(|failure| failure.contains(fragment))
}

#[test]
fn unchanged_baseline_and_ignored_python_caches_pass() {
    let repo = SyntheticRepo::new();
    repo.write(
        "research/benchmark/results/__pycache__/fixture.pyc",
        b"ignored cache",
    );
    assert!(repo.check(None).is_empty());
}

#[test]
fn host_only_scrub_in_every_protected_area_fails() {
    let repo = SyntheticRepo::new();
    for name in PATHS {
        repo.write(name, SCRUBBED);
        let failures = repo.check(None);
        assert_eq!(
            failures.len(),
            1,
            "unexpected failures for {name}: {failures:?}"
        );
        assert!(has_failure(
            &failures,
            &format!("{name}: frozen SHA256 drift")
        ));
        repo.write(name, ORIGINAL);
    }
}

#[test]
fn missing_file_rename_and_unrecorded_new_files_are_rejected() {
    let repo = SyntheticRepo::new();
    let old = repo.root.join(PATHS[0]);
    fs::rename(&old, old.with_file_name("renamed.json")).unwrap();
    let failures = repo.check(None);
    assert!(has_failure(
        &failures,
        &format!("{}: frozen file missing", PATHS[0])
    ));
    assert!(has_failure(&failures, "renamed.json: unrecorded"));

    for name in [
        "research/benchmark/results/new.json",
        "docs/whitepaper/generated/new.tex",
        "research/benchmark/review_fixes/predeclare_new.json",
    ] {
        let repo = SyntheticRepo::new();
        repo.write(name, ORIGINAL);
        assert!(has_failure(
            &repo.check(None),
            &format!("{name}: unrecorded")
        ));
    }
}

#[test]
fn allowlist_requires_exact_replacement_bytes_and_file_presence() {
    let repo = SyntheticRepo::new();
    let name = PATHS[0];
    repo.write(name, SCRUBBED);
    let allowed = json!({
        (name): {
            "sha256": format!("{:x}", Sha256::digest(SCRUBBED)),
            "reason": "Reviewed synthetic replacement"
        }
    });
    assert!(repo.check(Some(&allowed)).is_empty());
    repo.write(
        name,
        b"{\"host\":\"public-placeholder\",\"retention\":0.9} \n",
    );
    assert!(has_failure(
        &repo.check(Some(&allowed)),
        "frozen SHA256 drift"
    ));
    fs::remove_file(repo.root.join(name)).unwrap();
    assert!(has_failure(
        &repo.check(Some(&allowed)),
        "frozen file missing"
    ));

    let new_path = "research/benchmark/results/new.json";
    let allow_new = json!({
        (new_path): {
            "sha256": format!("{:x}", Sha256::digest(ORIGINAL)),
            "reason": "Reviewed new synthetic fixture"
        }
    });
    let repo = SyntheticRepo::new();
    assert!(has_failure(
        &repo.check(Some(&allow_new)),
        "frozen file missing"
    ));
    repo.write(new_path, ORIGINAL);
    assert!(repo.check(Some(&allow_new)).is_empty());
}

#[test]
fn invalid_allowlists_fail_closed_and_blank_reason_matches_python_strip() {
    let repo = SyntheticRepo::new();
    let entry = json!({"sha256": "a".repeat(64), "reason": "fixture"});
    let blank_reasons = [
        "",
        " \t\n",
        "\u{1c}",
        "\u{1d}\u{1e}\u{1f}",
        " \u{1c} \u{1f}\t",
    ];
    for reason in blank_reasons {
        let value = json!({ (PATHS[0]): {"sha256": "a".repeat(64), "reason": reason} });
        assert!(
            replacement_hashes(&value).is_err(),
            "accepted blank reason {reason:?}"
        );
    }
    for value in [
        json!([]),
        json!({"research/benchmark/results/*": entry.clone()}),
        json!({"research/benchmark/results/../methods.lock.json": entry.clone()}),
        json!({"/research/benchmark/results/panel.json": entry.clone()}),
        json!({"README.md": entry.clone()}),
        json!({(PATHS[0]): {"reason": "missing hash"}}),
        json!({(PATHS[0]): {"sha256": serde_json::Value::Null, "reason": "fixture"}}),
        json!({(PATHS[0]): {"sha256": "not a hash", "reason": "fixture"}}),
        json!({(PATHS[0]): {"sha256": "a".repeat(64), "reason": "fixture", "extra": true}}),
    ] {
        assert!(replacement_hashes(&value).is_err(), "accepted {value}");
    }
    assert!(repo.check(None).is_empty());
}

#[cfg(unix)]
#[test]
fn matching_bytes_through_file_or_directory_symlink_fail() {
    use std::os::unix::fs::symlink;

    let repo = SyntheticRepo::new();
    let path = repo.root.join(PATHS[0]);
    let saved = repo.root.join("saved.json");
    fs::rename(&path, &saved).unwrap();
    symlink(&saved, &path).unwrap();
    assert!(has_failure(
        &repo.check(None),
        &format!("{}: frozen file missing", PATHS[0])
    ));

    fs::remove_file(&path).unwrap();
    fs::rename(&saved, &path).unwrap();
    let parent = path.parent().unwrap();
    let saved_parent = repo.root.join("saved-directory");
    fs::rename(parent, &saved_parent).unwrap();
    symlink(&saved_parent, parent).unwrap();
    assert!(has_failure(
        &repo.check(None),
        &format!("{}: frozen file missing", PATHS[0])
    ));
}

#[test]
fn tracked_files_stay_protected_when_a_new_ignore_rule_matches() {
    let repo = SyntheticRepo::new();
    repo.write(".gitignore", b"*.json\n");
    repo.write(PATHS[0], SCRUBBED);
    assert!(has_failure(
        &repo.check(None),
        &format!("{}: frozen SHA256 drift", PATHS[0])
    ));
}

#[test]
fn only_declared_protected_paths_are_frozen() {
    let repo = SyntheticRepo::new();
    repo.write("README.md", SCRUBBED);
    assert!(repo.check(None).is_empty());
}

#[cfg(unix)]
#[test]
fn non_utf8_git_paths_refuse_inventory() {
    use std::os::unix::ffi::OsStringExt;

    let repo = SyntheticRepo::new();
    let parent = repo.root.join("research/benchmark/results");
    fs::create_dir_all(&parent).unwrap();
    let path = parent.join(OsString::from_vec(b"invalid-\xff.json".to_vec()));
    fs::write(path, ORIGINAL).unwrap();
    assert_eq!(
        inventory(&repo.root).unwrap_err(),
        "repository contains a non-UTF-8 path"
    );
}

#[test]
fn manifest_baseline_and_recorded_hash_lock_refuse_mutation() {
    // This reads only manifest metadata, never protected evidence file contents.
    let manifest = fs::read(manifest_path()).expect("read frozen manifest metadata");
    let original: Value = serde_json::from_slice(&manifest).unwrap();
    let first = original["files"]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap()
        .clone();
    let mut changed_hash = original.clone();
    changed_hash["files"][&first] = Value::String("0".repeat(64));
    reject_manifest_mutation(&changed_hash);
    let mut removed_entry = original.clone();
    removed_entry["files"]
        .as_object_mut()
        .unwrap()
        .remove(&first);
    reject_manifest_mutation(&removed_entry);
    let mut changed_baseline = original.clone();
    changed_baseline["baseline_commit"] = Value::String("f".repeat(40));
    reject_manifest_mutation(&changed_baseline);
    let mut changed_allowlist = original.clone();
    changed_allowlist["allowlist"] = json!([]);
    reject_manifest_mutation(&changed_allowlist);
    assert!(load_manifest(&manifest_path()).is_ok());
}

fn reject_manifest_mutation(value: &Value) {
    let path = temporary_manifest(value);
    assert!(load_manifest(&path).is_err());
    fs::remove_file(path).unwrap();
}

fn temporary_manifest(value: &Value) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("target")
        .join(format!(
            "frozen-manifest-{}-{}.json",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
    path
}

#[test]
fn cli_parser_preserves_default_split_and_equals_roots() {
    let default = parse_args(std::iter::empty());
    assert_eq!(default, CliRequest::Run(repository_root()));
    assert_eq!(
        parse_args(["--root", "relative/root"].map(OsString::from)),
        CliRequest::Run(PathBuf::from("relative/root"))
    );
    assert_eq!(
        parse_args(["--root=/absolute/root"].map(OsString::from)),
        CliRequest::Run(PathBuf::from("/absolute/root"))
    );
    // Python Path("") resolves to the current directory.
    assert_eq!(
        parse_args(["--root", ""].map(OsString::from)),
        CliRequest::Run(PathBuf::from("."))
    );
    assert_eq!(
        parse_args(["--root="].map(OsString::from)),
        CliRequest::Run(PathBuf::from("."))
    );
}
#[test]
fn cli_parser_preserves_help_and_usage_refusals() {
    assert_eq!(parse_args(["--help"].map(OsString::from)), CliRequest::Help);
    assert_eq!(
        parse_args(["-h", "--extra"].map(OsString::from)),
        CliRequest::Help
    );
    for arguments in [
        vec!["--root"],
        vec!["--root", "x", "extra"],
        vec!["--root=x", "extra"],
        vec!["--unknown"],
        vec!["--rootx"],
    ] {
        assert_eq!(
            parse_args(arguments.into_iter().map(OsString::from)),
            CliRequest::UsageError
        );
    }
}
#[test]
fn path_scope_matches_python_pureposix_validation() {
    assert!(valid_path("docs/whitepaper/generated/figure-one.svg"));
    assert!(valid_path("research/benchmark/results/run/report.json"));
    assert!(valid_path(
        "research/benchmark/review_fixes/predeclare.json"
    ));
    assert!(valid_path("ops/ci/paper-review-inputs.json"));
    for path in [
        "../docs/whitepaper/generated/x",
        "docs//whitepaper/generated/x",
        "docs/whitepaper/generated/../secret",
        "docs/whitepaper/generated/x?.json",
        "docs/whitepaper/generated/x\\y",
        "docs/whitepaper/generated/x/",
        "README.md",
    ] {
        assert!(!valid_path(path), "accepted {path}");
    }
    assert!(!protected("ops/ci/frozen-evidence.json"));
}

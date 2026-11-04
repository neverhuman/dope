//! Rust parity candidate for `ops/ci/check-frozen-evidence.py`.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

const BASELINE: &str = "b8340ec41799ccfe18d7aeee25bc6a23c2e0a093";
const RECORDED_HASHES_SHA256: &str =
    "99c09343219da7cd7175fe4ccc567e67ba6db6da15ee00f22c9824577992a82b";
type Hashes = BTreeMap<String, String>;

fn protected(name: &str) -> bool {
    name.starts_with("research/benchmark/results/")
        || name.starts_with("docs/whitepaper/generated/")
        || name.starts_with("research/benchmark/review_fixes/predeclare")
        || matches!(
            name,
            "research/benchmark/methods.lock.json" | "ops/ci/paper-review-inputs.json"
        )
}

fn valid_path(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('/')
        && !name.contains(['*', '?', '[', ']', '\\'])
        && name
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
        && protected(name)
}

fn valid_hash(value: &Value) -> bool {
    value.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    })
}

// Python str.strip() also treats U+001C..U+001F as whitespace, unlike Rust trim().
fn valid_reason(reason: &str) -> bool {
    !reason.is_empty()
        && !reason
            .chars()
            .all(|character| character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}'))
}

fn replacement_hashes(allowlist: &Value) -> Result<Hashes, String> {
    let entries = allowlist.as_object().ok_or_else(|| {
        "allowlist must map exact paths to replacement hashes and reasons".to_owned()
    })?;
    let mut result = BTreeMap::new();
    for (name, entry) in entries {
        let fields = entry
            .as_object()
            .ok_or_else(|| format!("invalid frozen-evidence allowlist entry: {name}"))?;
        if !valid_path(name)
            || fields.len() != 2
            || !fields.contains_key("sha256")
            || !fields.contains_key("reason")
            || !valid_hash(&fields["sha256"])
            || fields["reason"]
                .as_str()
                .is_none_or(|reason| !valid_reason(reason))
        {
            return Err(format!("invalid frozen-evidence allowlist entry: {name}"));
        }
        result.insert(name.clone(), fields["sha256"].as_str().unwrap().to_owned());
    }
    Ok(result)
}

fn load_manifest(path: &Path) -> Result<(Hashes, Hashes), String> {
    let data: Value = serde_json::from_slice(
        &fs::read(path)
            .map_err(|error| format!("cannot read frozen-evidence manifest: {error}"))?,
    )
    .map_err(|error| format!("cannot parse frozen-evidence manifest: {error}"))?;
    let root = data
        .as_object()
        .ok_or_else(|| "invalid frozen-evidence manifest or baseline commit".to_owned())?;
    if root.len() != 3
        || !root.contains_key("baseline_commit")
        || !root.contains_key("files")
        || !root.contains_key("allowlist")
        || root["baseline_commit"] != BASELINE
    {
        return Err("invalid frozen-evidence manifest or baseline commit".to_owned());
    }
    let files = root["files"]
        .as_object()
        .ok_or_else(|| "invalid frozen-evidence manifest or baseline commit".to_owned())?;
    let canonical = serde_json::to_vec(&root["files"])
        .map_err(|_| "invalid frozen-evidence manifest or baseline commit".to_owned())?;
    let digest = format!("{:x}", Sha256::digest(canonical));
    if digest != RECORDED_HASHES_SHA256 {
        return Err(
            "recorded frozen hashes changed; use an explicit allowlist exception".to_owned(),
        );
    }
    let mut recorded = BTreeMap::new();
    for (name, value) in files {
        if !valid_hash(value) {
            return Err("invalid frozen-evidence manifest or baseline commit".to_owned());
        }
        recorded.insert(name.clone(), value.as_str().unwrap().to_owned());
    }
    let allowlist = replacement_hashes(&root["allowlist"])?;
    Ok((recorded, allowlist))
}

fn inventory(root: &Path) -> Result<BTreeSet<String>, String> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot inspect repository root: {error}"))?;
    if !output.status.success() {
        return Err("--root must be the Git repository root".to_owned());
    }
    let top = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    if top.canonicalize().ok().as_deref() != Some(root) {
        return Err("--root must be the Git repository root".to_owned());
    }
    let output = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot inventory repository files: {error}"))?;
    if !output.status.success() {
        return Err("cannot inventory repository files".to_owned());
    }
    let mut paths = BTreeSet::new();
    for path in output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
    {
        let path = std::str::from_utf8(path)
            .map_err(|_| "repository contains a non-UTF-8 path".to_owned())?;
        if protected(path) {
            paths.insert(path.to_owned());
        }
    }
    Ok(paths)
}

fn check_tree(root: &Path, recorded: &Hashes, allowlist: &Hashes) -> Result<Vec<String>, String> {
    let mut expected = recorded.clone();
    expected.extend(allowlist.clone());
    let mut names = inventory(root)?;
    names.extend(expected.keys().cloned());
    let mut failures = Vec::new();
    for name in names {
        let Some(digest) = expected.get(&name) else {
            failures.push(format!(
                "{name}: unrecorded frozen-evidence file; requires an allowlist entry"
            ));
            continue;
        };
        let path = root.join(&name);
        if !valid_path(&name)
            || path.canonicalize().ok().as_deref() != Some(path.as_path())
            || !path.is_file()
        {
            failures.push(format!(
                "{name}: frozen file missing, not regular, or reached through a symlink"
            ));
            continue;
        }
        let mut file =
            fs::File::open(&path).map_err(|error| format!("cannot read {name}: {error}"))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0; 64 * 1024];
        loop {
            let count = file
                .read(&mut buffer)
                .map_err(|error| format!("cannot read {name}: {error}"))?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        let actual = format!("{:x}", hasher.finalize());
        if &actual != digest {
            failures.push(format!(
                "{name}: frozen SHA256 drift (expected {digest}, got {actual})"
            ));
        }
    }
    Ok(failures)
}

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../ci/frozen-evidence.json")
}

fn run(root: &Path) -> Result<(usize, usize), String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot resolve --root: {error}"))?;
    let (recorded, allowlist) = load_manifest(&manifest_path())?;
    let failures = check_tree(&root, &recorded, &allowlist)?;
    if !failures.is_empty() {
        for failure in &failures {
            eprintln!("{failure}");
        }
        return Err(format!(
            "{} file(s). Restore recorded bytes or add a reviewed exact-path/hash/reason exception.",
            failures.len()
        ));
    }
    Ok((recorded.len(), allowlist.len()))
}

#[derive(Debug, PartialEq, Eq)]
enum CliRequest {
    Run(PathBuf),
    Help,
    UsageError,
}
// Match the current internal CLI forms and Python argparse's --root=PATH spelling.
// The equals spelling accepts UTF-8 values; split --root PATH preserves OsString paths.
fn parse_args<I>(arguments: I) -> CliRequest
where
    I: IntoIterator<Item = std::ffi::OsString>,
{
    let mut args = arguments.into_iter();
    match args.next() {
        None => CliRequest::Run(repository_root()),
        Some(flag) if flag == "--help" || flag == "-h" => CliRequest::Help,
        Some(flag) if flag == "--root" => match args.next() {
            Some(path) if args.next().is_none() => {
                let path = if path.is_empty() {
                    std::ffi::OsString::from(".")
                } else {
                    path
                };
                CliRequest::Run(PathBuf::from(path))
            }
            _ => CliRequest::UsageError,
        },
        Some(flag) => match flag
            .to_str()
            .and_then(|value| value.strip_prefix("--root="))
        {
            Some(path) if args.next().is_none() => {
                CliRequest::Run(PathBuf::from(if path.is_empty() { "." } else { path }))
            }
            _ => CliRequest::UsageError,
        },
    }
}
fn main() {
    let root = match parse_args(env::args_os().skip(1)) {
        CliRequest::Run(root) => root,
        CliRequest::Help => {
            println!("Usage: check-frozen-evidence [--root REPOSITORY_ROOT]");
            return;
        }
        CliRequest::UsageError => {
            eprintln!("usage: check-frozen-evidence [--root REPOSITORY_ROOT]");
            std::process::exit(2);
        }
    };
    match run(&root) {
        Ok((files, exceptions)) => println!(
            "Frozen evidence verified: {files} recorded files; {exceptions} explicit exceptions (baseline {BASELINE})."
        ),
        Err(error) => {
            eprintln!("Frozen evidence check failed: {error}");
            std::process::exit(1);
        }
    }
}
#[cfg(test)]
#[path = "frozen_evidence_tests.rs"]
mod tests;

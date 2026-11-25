//! Pure-Rust public-repository hygiene checker.
//!
//! Enforces four structural invariants without Python dependencies:
//! 1. No internal infrastructure hostnames, private bucket URIs, or local user
//!    home paths in public-facing repository surfaces.
//! 2. No tracked scratch directories or duplicated governance copies at root.
//! 3. No stale entries in `agent/owner-map.json` or `agent/test-map.json` that
//!    match zero repository files.
//! 4. No broken relative Markdown file links across primary public docs.
use serde_json::Value;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

const PUBLIC_DOCS_WITH_LINKS: &[&str] = &[
    "README.md",
    "ROADMAP.md",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "AGENTS.md",
    "docs/README.md",
    "docs/REPO_MAP.md",
    "docs/getting-started.md",
    "docs/training.md",
    "docs/briefs/ceo-brief.md",
    "docs/briefs/cto-brief.md",
    "docs/whitepaper-v3/ARXIV_UPLOAD.md",
    "examples/v3-quickstart/README.md",
];

const FORBIDDEN_ROOT_FILES: &[&str] = &["PYTHON_EXCEPTIONS.md", "PY_TO_RUST_PLAN.md"];
const FORBIDDEN_ROOT_PREFIXES: &[&str] = &[
    "receipts/",
    "tips/",
    "todo/",
    "catboost_info/",
    ".agent/",
    "target/",
];

const LEAK_LITERALS: &[(&str, &str)] = &[
    ("s3://veox-", "private S3 bucket URI"),
    ("veox-jepson", "personal AWS profile name"),
    ("/home/ubuntu/", "local Ubuntu user home path"),
    ("/root/dope", "local root checkout path"),
    ("/Users/", "local macOS user home path"),
];

fn git_files(root: &Path, include_untracked: bool) -> Result<BTreeSet<String>, String> {
    let mut args = vec!["ls-files", "-z"];
    if include_untracked {
        args.extend(["--cached", "--others", "--exclude-standard"]);
    }
    let output = Command::new("git")
        .args(&args)
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot run git ls-files: {error}"))?;
    if !output.status.success() {
        return Err("git ls-files failed; pass --root pointing to a Git repository".to_owned());
    }
    let mut files = BTreeSet::new();
    for entry in output.stdout.split(|b| *b == 0).filter(|s| !s.is_empty()) {
        let name = std::str::from_utf8(entry)
            .map_err(|_| "repository contains a non-UTF-8 path".to_owned())?;
        files.insert(name.to_owned());
    }
    Ok(files)
}

fn is_wave_scratch_dir(path: &str) -> bool {
    let Some((first, _)) = path.split_once('/') else {
        return false;
    };
    first
        .strip_prefix('H')
        .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
}

fn check_root_cleanliness(tracked: &BTreeSet<String>) -> Vec<String> {
    let mut failures = Vec::new();
    for path in tracked {
        if FORBIDDEN_ROOT_FILES.contains(&path.as_str()) {
            failures.push(format!(
                "root-hygiene: duplicate root file '{path}' must live under docs/governance/"
            ));
            continue;
        }
        if FORBIDDEN_ROOT_PREFIXES
            .iter()
            .any(|prefix| path.starts_with(prefix))
        {
            failures.push(format!(
                "root-hygiene: forbidden tracked scratch/orphan path '{path}'"
            ));
            continue;
        }
        if is_wave_scratch_dir(path) {
            failures.push(format!(
                "root-hygiene: root wave directory '{path}' must be archived under docs/archive/waves/"
            ));
        }
    }
    failures
}

fn is_public_surface(path: &str) -> bool {
    if matches!(
        path,
        "README.md" | "ROADMAP.md" | "CONTRIBUTING.md" | "GOVERNANCE.md" | "AGENTS.md"
    ) {
        return true;
    }
    if let Some(rest) = path.strip_prefix("docs/") {
        if !rest.contains('/') && path.ends_with(".md") {
            return true;
        }
        return path.starts_with("docs/briefs/")
            || path.starts_with("docs/archive/")
            || path.starts_with("docs/governance/");
    }
    path.starts_with("engine/")
        || path.starts_with("scripts/")
        || path.starts_with("tools/")
        || path.starts_with("examples/")
        || path.starts_with(".github/")
}

fn is_text_candidate(path: &str) -> bool {
    !matches!(
        Path::new(path)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some(
            "png"
                | "jpg"
                | "jpeg"
                | "gif"
                | "pdf"
                | "zip"
                | "gz"
                | "tar"
                | "bin"
                | "dope"
                | "dpk"
                | "safetensors"
                | "ico"
                | "woff2"
        )
    )
}

fn contains_internal_host(line: &str) -> bool {
    let bytes = line.as_bytes();
    let needle = b"xbabe";
    if bytes.len() <= needle.len() {
        return false;
    }
    for index in 0..=(bytes.len() - needle.len() - 1) {
        let matched = bytes[index..index + needle.len()]
            .iter()
            .zip(needle.iter())
            .all(|(a, b)| a.to_ascii_lowercase() == *b);
        if matched && bytes[index + needle.len()].is_ascii_digit() {
            return true;
        }
    }
    false
}

fn scan_content_for_leaks(path: &str, content: &str) -> Vec<String> {
    let mut failures = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        let lineno = idx + 1;
        if contains_internal_host(line) {
            failures.push(format!(
                "leak-scan: {path}:{lineno}: internal cluster hostname (xbabe<N>)"
            ));
        }
        for &(literal, label) in LEAK_LITERALS {
            if line.contains(literal) {
                failures.push(format!("leak-scan: {path}:{lineno}: {label} ({literal})"));
            }
        }
    }
    failures
}

fn check_public_leaks(root: &Path, files: &BTreeSet<String>) -> Vec<String> {
    let mut failures = Vec::new();
    for rel in files {
        if !is_public_surface(rel) || !is_text_candidate(rel) {
            continue;
        }
        let full = root.join(rel);
        if !full.is_file() {
            continue;
        }
        if let Ok(content) = fs::read_to_string(&full) {
            failures.extend(scan_content_for_leaks(rel, &content));
        }
    }
    failures
}

fn check_map_file(
    root: &Path,
    rel_map: &str,
    top_key: &str,
    files: &BTreeSet<String>,
) -> Vec<String> {
    let path = root.join(rel_map);
    let raw = match fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(error) => return vec![format!("map-check: cannot read {rel_map}: {error}")],
    };
    let parsed: Value = match serde_json::from_str(&raw) {
        Ok(parsed) => parsed,
        Err(error) => return vec![format!("map-check: invalid JSON in {rel_map}: {error}")],
    };
    let Some(entries) = parsed.get(top_key).and_then(Value::as_object) else {
        return vec![format!("map-check: {rel_map} missing object '{top_key}'")];
    };
    let mut failures = Vec::new();
    for pattern in entries.keys() {
        let matched = if pattern.ends_with('/') {
            files.iter().any(|file| file.starts_with(pattern))
        } else {
            files
                .iter()
                .any(|file| file == pattern || file.starts_with(pattern))
        };
        if !matched {
            failures.push(format!(
                "map-check: {rel_map}: stale entry '{pattern}' matches zero repository files"
            ));
        }
    }
    failures
}

fn extract_markdown_targets(content: &str) -> Vec<(usize, String)> {
    let mut results = Vec::new();
    let mut in_fence = false;
    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        let bytes = line.as_bytes();
        let mut i = 0;
        let mut in_code = false;
        while i < bytes.len() {
            if bytes[i] == b'`' {
                in_code = !in_code;
                i += 1;
                continue;
            }
            if !in_code && i + 1 < bytes.len() && bytes[i] == b']' && bytes[i + 1] == b'(' {
                let start = i + 2;
                let mut depth = 1usize;
                let mut j = start;
                while j < bytes.len() && depth > 0 {
                    match bytes[j] {
                        b'(' => depth += 1,
                        b')' => depth -= 1,
                        _ => {}
                    }
                    if depth > 0 {
                        j += 1;
                    }
                }
                if depth == 0 {
                    let raw = line[start..j].trim();
                    let target = raw
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .trim_matches(|c| c == '<' || c == '>');
                    if !target.is_empty() {
                        results.push((idx + 1, target.to_owned()));
                    }
                    i = j + 1;
                    continue;
                }
            }
            i += 1;
        }
    }
    results
}

fn normalize_relative(base: &Path, target: &str) -> Option<PathBuf> {
    let mut out = base.to_path_buf();
    for comp in Path::new(target).components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    return None;
                }
            }
            Component::Normal(part) => out.push(part),
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    Some(out)
}

fn check_markdown_links(root: &Path) -> Vec<String> {
    let mut failures = Vec::new();
    for rel_doc in PUBLIC_DOCS_WITH_LINKS {
        let doc_path = root.join(rel_doc);
        let Ok(content) = fs::read_to_string(&doc_path) else {
            failures.push(format!(
                "link-check: missing required public doc '{rel_doc}'"
            ));
            continue;
        };
        let parent = Path::new(rel_doc).parent().unwrap_or_else(|| Path::new(""));
        for (lineno, target) in extract_markdown_targets(&content) {
            if target.starts_with("http://")
                || target.starts_with("https://")
                || target.starts_with("mailto:")
                || target.starts_with('#')
            {
                continue;
            }
            let path_part = target.split('#').next().unwrap_or("").trim();
            if path_part.is_empty() {
                continue;
            }
            let Some(resolved) = normalize_relative(parent, path_part) else {
                failures.push(format!(
                    "link-check: {rel_doc}:{lineno}: target '{target}' escapes repository root"
                ));
                continue;
            };
            if !root.join(&resolved).exists() {
                failures.push(format!(
                    "link-check: {rel_doc}:{lineno}: broken link '{target}' (resolved '{}')",
                    resolved.display()
                ));
            }
        }
    }
    failures
}

fn run(root: &Path) -> Result<Vec<String>, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot resolve --root: {error}"))?;
    let tracked = git_files(&root, false)?;
    let all_non_ignored = git_files(&root, true)?;
    let mut failures = Vec::new();
    failures.extend(check_root_cleanliness(&tracked));
    failures.extend(check_public_leaks(&root, &all_non_ignored));
    failures.extend(check_map_file(
        &root,
        "agent/owner-map.json",
        "owners",
        &all_non_ignored,
    ));
    failures.extend(check_map_file(
        &root,
        "agent/test-map.json",
        "tests",
        &all_non_ignored,
    ));
    failures.extend(check_markdown_links(&root));
    Ok(failures)
}

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn parse_root() -> Result<Option<PathBuf>, String> {
    let mut args = env::args_os().skip(1);
    match args.next() {
        None => Ok(Some(repository_root())),
        Some(flag) if flag == "-h" || flag == "--help" => Ok(None),
        Some(flag) if flag == "--root" => match args.next() {
            Some(path) if args.next().is_none() => Ok(Some(PathBuf::from(path))),
            _ => Err("usage: repo-hygiene [--root REPOSITORY_ROOT]".to_owned()),
        },
        Some(flag) => match flag.to_str().and_then(|s| s.strip_prefix("--root=")) {
            Some(path) if args.next().is_none() => Ok(Some(PathBuf::from(path))),
            _ => Err("usage: repo-hygiene [--root REPOSITORY_ROOT]".to_owned()),
        },
    }
}

fn main() {
    let root = match parse_root() {
        Ok(Some(root)) => root,
        Ok(None) => {
            println!("Usage: repo-hygiene [--root REPOSITORY_ROOT]");
            return;
        }
        Err(usage) => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    };
    match run(&root) {
        Ok(failures) if failures.is_empty() => {
            println!("Repository hygiene checks passed (root, leaks, maps, links).");
        }
        Ok(failures) => {
            for failure in &failures {
                eprintln!("{failure}");
            }
            eprintln!(
                "Repository hygiene failed with {} issue(s).",
                failures.len()
            );
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("Repository hygiene error: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
#[path = "repo_hygiene_tests.rs"]
mod tests;

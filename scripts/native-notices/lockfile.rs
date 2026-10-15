//! Strict readers for the three text inputs of the notice inventory.
//!
//! `Cargo.lock` versions 3 and 4 as Cargo writes them, the `[package]` version
//! of `Cargo.toml`, and `cargo tree --prefix none --format {p}` lines. Anything
//! outside that subset (other tables, escapes, unknown keys, duplicate
//! packages, unmatched tree lines) is refused rather than guessed.

use std::collections::{BTreeSet, HashMap};

use crate::NoticeError;

pub type Locked = HashMap<(String, String), Option<String>>;

const LOCK: NoticeError = NoticeError("native_notice_lock_unsupported");
const MANIFEST: NoticeError = NoticeError("native_notice_manifest_unsupported");
const TREE: NoticeError = NoticeError("native_notice_tree_unsupported");

fn string(value: &str, error: NoticeError) -> Result<String, NoticeError> {
    let inner = value
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .ok_or(error)?;
    if inner
        .chars()
        .any(|character| character == '"' || character == '\\' || character.is_control())
    {
        return Err(error);
    }
    Ok(inner.to_string())
}

#[derive(Default)]
struct Package {
    name: Option<String>,
    version: Option<String>,
    source: bool,
    checksum: Option<String>,
}

fn finish(package: Option<Package>, locked: &mut Locked) -> Result<(), NoticeError> {
    if let Some(package) = package {
        let key = (package.name.ok_or(LOCK)?, package.version.ok_or(LOCK)?);
        if locked.insert(key, package.checksum).is_some() {
            return Err(LOCK);
        }
    }
    Ok(())
}

fn set(slot: &mut Option<String>, value: &str) -> Result<(), NoticeError> {
    if slot.is_some() {
        return Err(LOCK);
    }
    *slot = Some(string(value, LOCK)?);
    Ok(())
}

/// (name, version) to checksum for every `[[package]]` of a version 3 or 4 lock file.
pub fn parse_lock(text: &str) -> Result<Locked, NoticeError> {
    let mut locked = Locked::new();
    let mut current: Option<Package> = None;
    let mut format: Option<u8> = None;
    let mut in_array = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if in_array {
            if trimmed == "]" {
                in_array = false;
            } else {
                string(trimmed.strip_suffix(',').unwrap_or(trimmed), LOCK)?;
            }
            continue;
        }
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed == "[[package]]" {
            finish(current.take(), &mut locked)?;
            current = Some(Package::default());
            continue;
        }
        let (key, value) = trimmed.split_once(" = ").ok_or(LOCK)?;
        match current.as_mut() {
            None => match (key, value, format) {
                ("version", "3", None) => format = Some(3),
                ("version", "4", None) => format = Some(4),
                _ => return Err(LOCK),
            },
            Some(package) => match key {
                "name" => set(&mut package.name, value)?,
                "version" => set(&mut package.version, value)?,
                "source" if !package.source => {
                    string(value, LOCK)?;
                    package.source = true;
                }
                "checksum" => set(&mut package.checksum, value)?,
                "dependencies" if value == "[" => in_array = true,
                "dependencies" if value == "[]" => {}
                _ => return Err(LOCK),
            },
        }
    }
    finish(current, &mut locked)?;
    if in_array || format.is_none() || locked.is_empty() {
        return Err(LOCK);
    }
    Ok(locked)
}

/// The single `version` string of the `[package]` table.
pub fn package_version(manifest: &str) -> Result<String, NoticeError> {
    let mut table = "";
    let mut found = None;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            table = trimmed;
            continue;
        }
        if table != "[package]" {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("version") {
            if let Some(value) = rest.trim_start().strip_prefix('=') {
                if found.replace(string(value.trim(), MANIFEST)?).is_some() {
                    return Err(MANIFEST);
                }
            }
        }
    }
    found.ok_or(MANIFEST)
}

/// Distinct (name, version) pairs of `cargo tree --prefix none --format {p}` output.
pub fn tree_packages(output: &str) -> Result<BTreeSet<(String, String)>, NoticeError> {
    let mut packages = BTreeSet::new();
    for line in output.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let split = line
            .find(|character: char| {
                !(character.is_ascii_alphanumeric() || character == '_' || character == '-')
            })
            .ok_or(TREE)?;
        let (name, rest) = line.split_at(split);
        let version = rest.strip_prefix(" v").ok_or(TREE)?;
        let version = version.split(char::is_whitespace).next().unwrap_or("");
        if name.is_empty() || version.is_empty() {
            return Err(TREE);
        }
        packages.insert((name.to_string(), version.to_string()));
    }
    Ok(packages)
}

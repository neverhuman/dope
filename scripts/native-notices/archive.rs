//! Registry `.crate` members, read through the system `tar` with literal argv.
//!
//! One `--list --verbose` call per archive proves that every requested legal
//! member exists exactly once as a regular file; each member's bytes then come
//! from `--extract --to-stdout`. Names are passed after `--`, wildcards are
//! off and quoting is literal, so a name is matched exactly. A missing,
//! repeated, linked or non-regular member is refused. `tar` reads the gzip
//! stream itself; nothing is written to disk.

use std::path::Path;
use std::process::{Command, Stdio};

use crate::NoticeError;

const MEMBER: NoticeError = NoticeError("native_notice_registry_license_mismatch");
const TOOL: NoticeError = NoticeError("native_notice_tool_unavailable");

/// Run `tar <modes> --gzip --no-wildcards --quoting-style=literal --file <archive> -- <names>`.
fn tar(archive: &Path, modes: [&str; 2], names: &[&str]) -> Result<Vec<u8>, NoticeError> {
    if names
        .iter()
        .any(|name| name.is_empty() || name.chars().any(char::is_control))
    {
        return Err(MEMBER);
    }
    let output = Command::new("tar")
        .args(modes)
        .args([
            "--gzip",
            "--no-wildcards",
            "--quoting-style=literal",
            "--file",
        ])
        .arg(archive)
        .arg("--")
        .args(names)
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map_err(|_| TOOL)?;
    if !output.status.success() {
        return Err(MEMBER);
    }
    Ok(output.stdout)
}

/// The member name of a `tar --list --verbose` line for a regular file, if it is one.
///
/// The line is `mode owner/group size date time name`; only a mode starting
/// with `-` (regular file) is accepted.
fn regular_name(line: &str) -> Option<&str> {
    if !line.starts_with('-') {
        return None;
    }
    let mut rest = line;
    for _ in 0..5 {
        let start = rest.find(|character: char| !character.is_ascii_whitespace())?;
        rest = &rest[start..];
        let end = rest.find(|character: char| character.is_ascii_whitespace())?;
        rest = &rest[end..];
    }
    rest.strip_prefix(' ')
}

/// Refuse unless each name occurs exactly once in the archive, as a regular file.
pub fn require_regular(archive: &Path, names: &[&str]) -> Result<(), NoticeError> {
    let listing = tar(archive, ["--list", "--verbose"], names)?;
    let listing = String::from_utf8(listing).map_err(|_| MEMBER)?;
    let listing = listing.strip_suffix('\n').unwrap_or(&listing);
    let lines: Vec<&str> = if listing.is_empty() {
        Vec::new()
    } else {
        listing.split('\n').collect()
    };
    if lines.len() != names.len() {
        return Err(MEMBER);
    }
    for name in names {
        let found = lines
            .iter()
            .filter(|line| regular_name(line) == Some(*name))
            .count();
        if found != 1 {
            return Err(MEMBER);
        }
    }
    Ok(())
}

/// Bytes of one member already proven unique and regular by [`require_regular`].
pub fn member(archive: &Path, name: &str) -> Result<Vec<u8>, NoticeError> {
    tar(archive, ["--extract", "--to-stdout"], &[name])
}

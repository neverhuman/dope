//! Freeze existing local legal inputs for the locked default Linux native package.
//!
//! This Rust tool replaces the retired `scripts/native-notices.py` and writes the
//! same two files: `scripts/native-third-party.NOTICE` and
//! `scripts/native-legal.sha256`. `--check` compares instead of writing.
//!
//! No fetch, acceptance or dependency update is performed. The inventory is the
//! offline `cargo tree` normal/build graph of the default CPU-only Linux
//! x86-64 target. Each package must have exactly one unpacked registry
//! source, its cached `.crate` must match the `Cargo.lock` checksum, and every
//! license-like file of the source must equal the same regular member of that
//! archive byte for byte; archive members are read with the system `tar`
//! (literal argv, no wildcards). The Rust library notices come from the
//! toolchain sysroot. Symbolic links, special files, unsupported lock formats
//! and missing, repeated or non-regular members are refused. Errors are static codes; on failure the
//! tool prints one fixed line and exits 1. This is a conservative package
//! notice inventory, not a signed release qualification.

#[path = "native-notices/archive.rs"]
mod archive;
#[path = "native-notices/lockfile.rs"]
mod lockfile;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use sha2::{Digest, Sha256};

/// A static diagnostic. Source values never enter the message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoticeError(pub &'static str);

const NOTICE: &str = "scripts/native-third-party.NOTICE";
const PINS: &str = "scripts/native-legal.sha256";
const TARGET: &str = "x86_64-unknown-linux-gnu";
const ROOT_PACKAGE: &str = "dope-kernel";
const LEGAL_PREFIXES: [&str; 5] = ["LICENSE", "LICENCE", "COPYING", "NOTICE", "COPYRIGHT"];
const FAILURE: &str = "native_notice_inputs_unavailable_or_drift";
const USAGE: &str = "usage: dope-native-notices [--check]";
const HEADER: [&str; 4] = [
    "DOPE locked default Linux x86-64 native dependency notices\n",
    "Inventory: normal/build Cargo packages and installed Rust library notices.\n",
    "No GPU/research adapters or separately provisioned OS runtime are distributed here.\n",
    "License texts retain upstream bytes; this inventory does not qualify a release.\n",
];

const IO: NoticeError = NoticeError("native_notice_io_unavailable");
const TOOL: NoticeError = NoticeError("native_notice_tool_unavailable");
const LOCK_MISMATCH: NoticeError = NoticeError("native_notice_lock_mismatch");
const CACHE: NoticeError = NoticeError("native_notice_cache_unavailable");
const PACKAGE: NoticeError = NoticeError("native_notice_registry_package_mismatch");
const LICENSE_SOURCE: NoticeError = NoticeError("native_notice_license_source_unavailable");
const LICENSE_MEMBER: NoticeError = NoticeError("native_notice_registry_license_mismatch");
const SPECIAL: NoticeError = NoticeError("native_notice_source_link_or_special_file");
const RUST_LIBRARY: NoticeError = NoticeError("native_notice_rust_library_source_unavailable");
const DRIFT: NoticeError = NoticeError("native_notice_frozen_inputs_drift");

/// Inputs that come from the local toolchain rather than from the repository.
pub struct Environment {
    pub tree: String,
    pub cargo_home: PathBuf,
    pub sysroot: PathBuf,
}

fn digest(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn read(path: &Path) -> Result<Vec<u8>, NoticeError> {
    fs::read(path).map_err(|_| IO)
}

/// Regular file bytes; a symbolic link or anything else is refused.
fn read_regular(path: &Path, missing: NoticeError) -> Result<Vec<u8>, NoticeError> {
    let metadata = fs::symlink_metadata(path).map_err(|_| missing)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(if metadata.file_type().is_symlink() {
            SPECIAL
        } else {
            missing
        });
    }
    read(path)
}

/// Repository root: two levels above this source file, as the Python tool resolved it.
fn root() -> Result<PathBuf, NoticeError> {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join(file!());
    let root = source.parent().and_then(Path::parent).ok_or(IO)?;
    fs::canonicalize(root).map_err(|_| IO)
}

fn tool(root: &Path, program: &str, arguments: &[&str]) -> Result<String, NoticeError> {
    let output = Command::new(program)
        .args(arguments)
        .current_dir(root)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map_err(|_| TOOL)?;
    if !output.status.success() {
        return Err(TOOL);
    }
    String::from_utf8(output.stdout).map_err(|_| TOOL)
}

fn environment(root: &Path) -> Result<Environment, NoticeError> {
    let tree = tool(
        root,
        "cargo",
        &[
            "tree",
            "--locked",
            "--offline",
            "--target",
            TARGET,
            "-e",
            "normal,build",
            "--prefix",
            "none",
            "--format",
            "{p}",
        ],
    )?;
    let cargo_home = match env::var_os("CARGO_HOME") {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        Some(_) => return Err(CACHE),
        None => PathBuf::from(
            env::var_os("HOME")
                .filter(|home| !home.is_empty())
                .ok_or(CACHE)?,
        )
        .join(".cargo"),
    };
    let sysroot = PathBuf::from(tool(root, "rustc", &["--print", "sysroot"])?.trim());
    Ok(Environment {
        tree,
        cargo_home,
        sysroot,
    })
}

/// The single unpacked registry source `<registry>/<name>-<version>`.
fn unique_source(sources: &Path, directory: &str) -> Result<PathBuf, NoticeError> {
    let mut found = Vec::new();
    for entry in fs::read_dir(sources).map_err(|_| CACHE)? {
        let candidate = entry.map_err(|_| CACHE)?.path().join(directory);
        if fs::symlink_metadata(&candidate).is_ok() {
            found.push(candidate);
        }
    }
    let [source] = found.as_slice() else {
        return Err(CACHE);
    };
    let metadata = fs::symlink_metadata(source).map_err(|_| CACHE)?;
    if metadata.file_type().is_symlink() {
        return Err(SPECIAL);
    }
    if !metadata.is_dir() {
        return Err(CACHE);
    }
    Ok(source.clone())
}

fn is_legal(name: &str) -> bool {
    let upper = name.to_uppercase();
    LEGAL_PREFIXES
        .iter()
        .any(|prefix| upper.starts_with(prefix))
}

fn walk(
    directory: &Path,
    parts: &[String],
    found: &mut Vec<Vec<String>>,
) -> Result<(), NoticeError> {
    for entry in fs::read_dir(directory).map_err(|_| IO)? {
        let entry = entry.map_err(|_| IO)?;
        let name = entry.file_name().into_string().map_err(|_| SPECIAL)?;
        let metadata = fs::symlink_metadata(entry.path()).map_err(|_| IO)?;
        let mut path = parts.to_vec();
        path.push(name.clone());
        if metadata.file_type().is_symlink() {
            return Err(SPECIAL);
        } else if metadata.is_dir() {
            walk(&entry.path(), &path, found)?;
        } else if metadata.is_file() {
            if is_legal(&name) {
                found.push(path);
            }
        } else {
            return Err(SPECIAL);
        }
    }
    Ok(())
}

/// License-like files of a source tree, ordered component by component like Python paths.
fn legal_files(source: &Path) -> Result<Vec<Vec<String>>, NoticeError> {
    let mut found = Vec::new();
    walk(source, &[], &mut found)?;
    found.sort();
    Ok(found)
}

fn section(payload: &mut Vec<u8>, display: &str, content: &[u8]) {
    payload.extend_from_slice(format!("--- {display} sha256 {} ---\n", digest(content)).as_bytes());
    payload.extend_from_slice(content);
    payload.push(b'\n');
}

/// Render the notice and the pins from the repository root and the local environment.
pub fn render(root: &Path, environment: &Environment) -> Result<(Vec<u8>, Vec<u8>), NoticeError> {
    let lock = read(&root.join("Cargo.lock"))?;
    let locked = lockfile::parse_lock(std::str::from_utf8(&lock).map_err(|_| IO)?)?;
    let manifest = fs::read_to_string(root.join("Cargo.toml")).map_err(|_| IO)?;
    let own = (
        ROOT_PACKAGE.to_string(),
        lockfile::package_version(&manifest)?,
    );
    let mut packages = lockfile::tree_packages(&environment.tree)?;
    packages.remove(&own);
    let registry = environment.cargo_home.join("registry");
    let mut payload: Vec<u8> = HEADER.concat().into_bytes();
    payload.extend_from_slice(format!("Cargo.lock sha256 {}\n", digest(&lock)).as_bytes());
    for (name, version) in &packages {
        let checksum = locked
            .get(&(name.clone(), version.clone()))
            .ok_or(LOCK_MISMATCH)?;
        let directory = format!("{name}-{version}");
        let source = unique_source(&registry.join("src"), &directory)?;
        let index = source.parent().and_then(Path::file_name).ok_or(CACHE)?;
        let archive_path = registry
            .join("cache")
            .join(index)
            .join(format!("{directory}.crate"));
        let archive_bytes = read_regular(&archive_path, PACKAGE)?;
        if checksum.as_deref() != Some(digest(&archive_bytes).as_str()) {
            return Err(PACKAGE);
        }
        let legal = legal_files(&source)?;
        if legal.is_empty() {
            return Err(LICENSE_SOURCE);
        }
        payload.extend_from_slice(format!("\n===== {name} {version} =====\n").as_bytes());
        let members: Vec<String> = legal
            .iter()
            .map(|parts| format!("{directory}/{}", parts.join("/")))
            .collect();
        let names: Vec<&str> = members.iter().map(String::as_str).collect();
        archive::require_regular(&archive_path, &names)?;
        for (parts, member) in legal.iter().zip(&names) {
            let display = parts.join("/");
            let content = read_regular(&source.join(&display), LICENSE_SOURCE)?;
            if archive::member(&archive_path, member)? != content {
                return Err(LICENSE_MEMBER);
            }
            section(&mut payload, &display, &content);
        }
    }
    let documents = environment.sysroot.join("share/doc/rust");
    let copyright = read_regular(&documents.join("COPYRIGHT-library.html"), RUST_LIBRARY)?;
    let mut licenses = Vec::new();
    for entry in fs::read_dir(documents.join("licenses")).map_err(|_| RUST_LIBRARY)? {
        let name = entry
            .map_err(|_| RUST_LIBRARY)?
            .file_name()
            .into_string()
            .map_err(|_| SPECIAL)?;
        if name.ends_with(".txt") {
            licenses.push(name);
        }
    }
    licenses.sort();
    payload
        .extend_from_slice(b"\n===== Rust library upstream notice and license inventory =====\n");
    section(&mut payload, "COPYRIGHT-library.html", &copyright);
    for name in &licenses {
        let content = read_regular(&documents.join("licenses").join(name), RUST_LIBRARY)?;
        section(&mut payload, &format!("licenses/{name}"), &content);
    }
    let mut pins = String::new();
    for (name, content) in [
        ("LICENSE", read(&root.join("LICENSE"))?),
        (NOTICE, payload.clone()),
        ("Cargo.lock", lock),
        (
            "rust-toolchain.toml",
            read(&root.join("rust-toolchain.toml"))?,
        ),
    ] {
        pins.push_str(&format!("{}  {name}\n", digest(&content)));
    }
    Ok((payload, pins.into_bytes()))
}

/// Compare with, or write, the frozen notice and pins.
fn freeze(root: &Path, notice: &[u8], pins: &[u8], check: bool) -> Result<(), NoticeError> {
    if check {
        if read(&root.join(NOTICE))? != notice || read(&root.join(PINS))? != pins {
            return Err(DRIFT);
        }
        return Ok(());
    }
    fs::write(root.join(NOTICE), notice).map_err(|_| IO)?;
    fs::write(root.join(PINS), pins).map_err(|_| IO)
}

fn run(check: bool) -> Result<(), NoticeError> {
    let root = root()?;
    let environment = environment(&root)?;
    let (notice, pins) = render(&root, &environment)?;
    freeze(&root, &notice, &pins, check)
}

fn main() -> ExitCode {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    let check = match arguments.as_slice() {
        [] => false,
        [flag] if flag == "--check" => true,
        [flag] if flag == "-h" || flag == "--help" => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(check) {
        Ok(()) => ExitCode::SUCCESS,
        Err(_) => {
            eprintln!("{FAILURE}");
            ExitCode::FAILURE
        }
    }
}

include!("native-notices/tests.rs");

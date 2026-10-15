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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    /// A fresh directory under the system temporary directory; no environment variable is changed.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            let path = env::temp_dir().join(format!(
                "dope-native-notices-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::SeqCst)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn write(&self, relative: &str, content: &[u8]) -> PathBuf {
            let path = self.0.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, content).unwrap();
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// A gzip `.crate` built by the system tar from files (and links) in the given order.
    fn crate_archive(files: &[(&str, &[u8])], links: &[(&str, &str)], order: &[&str]) -> Vec<u8> {
        let staging = Scratch::new();
        for (name, content) in files {
            staging.write(name, content);
        }
        for (name, target) in links {
            std::os::unix::fs::symlink(target, staging.0.join(name)).unwrap();
        }
        let output = staging.0.join("out.crate");
        let status = Command::new("tar")
            .args(["--create", "--gzip", "--file"])
            .arg(&output)
            .arg("--directory")
            .arg(&staging.0)
            .arg("--")
            .args(order)
            .env("LC_ALL", "C")
            .status()
            .unwrap();
        assert!(status.success());
        fs::read(output).unwrap()
    }

    const SOURCE: [(&str, &[u8]); 3] = [
        ("licenses/LICENSE-MIT", b"mit body\n"),
        ("licenses-extra/LICENSE", b"extra body\n"),
        ("src/lib.rs", b"pub fn demo() {}\n"),
    ];

    fn members() -> Vec<(String, &'static [u8])> {
        SOURCE
            .iter()
            .map(|(name, content)| (format!("demo-1.0.0/{name}"), *content))
            .collect()
    }

    fn standard_archive() -> Vec<u8> {
        let members = members();
        let files: Vec<(&str, &[u8])> = members
            .iter()
            .map(|(name, content)| (name.as_str(), *content))
            .collect();
        let order: Vec<&str> = members.iter().map(|(name, _)| name.as_str()).collect();
        crate_archive(&files, &[], &order)
    }

    struct Fixture {
        root: Scratch,
        home: Scratch,
        sysroot: Scratch,
        tree: String,
    }

    fn fixture(archive: &[u8], source: &[(&str, &[u8])]) -> Fixture {
        let root = Scratch::new();
        let home = Scratch::new();
        let sysroot = Scratch::new();
        root.write(
            "Cargo.lock",
            format!(
                "version = 4\n\n[[package]]\nname = \"demo\"\nversion = \"1.0.0\"\nchecksum = \"{}\"\n\n\
                 [[package]]\nname = \"dope-kernel\"\nversion = \"0.1.0\"\n",
                digest(archive)
            )
            .as_bytes(),
        );
        root.write(
            "Cargo.toml",
            b"[package]\nname = \"dope-kernel\"\nversion = \"0.1.0\"\n",
        );
        root.write("LICENSE", b"root license\n");
        root.write(
            "rust-toolchain.toml",
            b"[toolchain]\nchannel = \"1.88.0\"\n",
        );
        home.write("registry/cache/index.test/demo-1.0.0.crate", archive);
        for (path, content) in source {
            home.write(
                &format!("registry/src/index.test/demo-1.0.0/{path}"),
                content,
            );
        }
        sysroot.write("share/doc/rust/COPYRIGHT-library.html", b"<p>rust</p>\n");
        sysroot.write("share/doc/rust/licenses/MIT.txt", b"mit\n");
        sysroot.write("share/doc/rust/licenses/Apache-2.0.txt", b"apache\n");
        sysroot.write(
            "share/doc/rust/licenses/README.md",
            b"not a license list entry\n",
        );
        let tree = "dope-kernel v0.1.0 (/repo)\ndemo v1.0.0\n".to_string();
        Fixture {
            root,
            home,
            sysroot,
            tree,
        }
    }

    fn environment_of(fixture: &Fixture) -> Environment {
        Environment {
            tree: fixture.tree.clone(),
            cargo_home: fixture.home.0.clone(),
            sysroot: fixture.sysroot.0.clone(),
        }
    }

    fn render_of(fixture: &Fixture) -> Result<(), NoticeError> {
        render(&fixture.root.0, &environment_of(fixture)).map(|_| ())
    }

    #[test]
    fn archive_members_must_be_unique_regular_files() {
        let regular = crate_archive(
            &[
                ("demo-1.0.0/LICENSE", b"text\n"),
                ("demo-1.0.0/odd name/COPYING", b"odd\n"),
            ],
            &[("demo-1.0.0/NOTICE", "LICENSE")],
            &[
                "demo-1.0.0/LICENSE",
                "demo-1.0.0/odd name/COPYING",
                "demo-1.0.0/NOTICE",
            ],
        );
        let scratch = Scratch::new();
        let path = scratch.write("demo.crate", &regular);
        archive::require_regular(
            &path,
            &["demo-1.0.0/LICENSE", "demo-1.0.0/odd name/COPYING"],
        )
        .unwrap();
        assert_eq!(
            archive::member(&path, "demo-1.0.0/LICENSE").unwrap(),
            b"text\n"
        );
        assert_eq!(
            archive::member(&path, "demo-1.0.0/odd name/COPYING").unwrap(),
            b"odd\n"
        );
        for refused in [
            vec!["demo-1.0.0/NOTICE"],
            vec!["demo-1.0.0/absent"],
            vec!["demo-1.0.0/LICENS*"],
            vec!["demo-1.0.0"],
            vec!["demo-1.0.0/LICENSE\nx"],
        ] {
            assert_eq!(
                archive::require_regular(&path, &refused),
                Err(LICENSE_MEMBER)
            );
        }
        let twice = crate_archive(
            &[("demo-1.0.0/LICENSE", b"text\n")],
            &[],
            &["demo-1.0.0/LICENSE", "demo-1.0.0/LICENSE"],
        );
        let path = scratch.write("twice.crate", &twice);
        assert_eq!(
            archive::require_regular(&path, &["demo-1.0.0/LICENSE"]),
            Err(LICENSE_MEMBER)
        );
        let broken = scratch.write("broken.crate", b"not a gzip stream");
        assert_eq!(
            archive::require_regular(&broken, &["demo-1.0.0/LICENSE"]),
            Err(LICENSE_MEMBER)
        );
    }

    #[test]
    fn lock_manifest_and_tree_inputs_are_strict() {
        let lock = "# generated\nversion = 4\n\n[[package]]\nname = \"demo\"\nversion = \"1.0.0\"\n\
                    source = \"registry+https://github.com/rust-lang/crates.io-index\"\nchecksum = \"ab\"\n\
                    dependencies = [\n \"other\",\n]\n\n[[package]]\nname = \"dope-kernel\"\nversion = \"0.1.0\"\n";
        let locked = lockfile::parse_lock(lock).unwrap();
        assert_eq!(
            locked[&("demo".into(), "1.0.0".into())].as_deref(),
            Some("ab")
        );
        assert_eq!(locked[&("dope-kernel".into(), "0.1.0".into())], None);
        let refused = Err(NoticeError("native_notice_lock_unsupported"));
        for bad in [
            lock.replace("version = 4", "version = 1"),
            lock.replace("checksum = \"ab\"", "checksum = \"a\\\\b\""),
            lock.replace("checksum = \"ab\"", "replace = \"x\""),
            format!("{lock}[metadata]\n"),
            format!("{lock}\n[[package]]\nname = \"demo\"\nversion = \"1.0.0\"\n"),
            lock.replace("]\n\n[[package]]", "\n\n[[package]]"),
        ] {
            assert_eq!(lockfile::parse_lock(&bad).map(|_| ()), refused);
        }
        let manifest = "[package]\nname = \"dope-kernel\"\nversion = \"0.1.0\"\n\n[dependencies.tch]\nversion = \"=9\"\n";
        assert_eq!(lockfile::package_version(manifest).unwrap(), "0.1.0");
        let unsupported = Err(NoticeError("native_notice_manifest_unsupported"));
        assert_eq!(
            lockfile::package_version("[package]\nversion.workspace = true\n"),
            unsupported
        );
        assert_eq!(
            lockfile::package_version("[package]\nversion = \"1\"\nversion = \"2\"\n"),
            unsupported
        );
        let tree = "dope-kernel v0.1.0 (/repo)\nserde v1.0.0\nserde_derive v1.0.0 (proc-macro)\nserde v1.0.0 (*)\n\n";
        let packages = lockfile::tree_packages(tree).unwrap();
        assert_eq!(packages.len(), 3);
        assert!(packages.contains(&("serde_derive".into(), "1.0.0".into())));
        let tree_refused = Err(NoticeError("native_notice_tree_unsupported"));
        assert_eq!(
            lockfile::tree_packages("warning: something\n").map(|_| ()),
            tree_refused
        );
        assert_eq!(
            lockfile::tree_packages("serde 1.0.0\n").map(|_| ()),
            tree_refused
        );
    }

    #[test]
    fn render_writes_the_python_layout_and_check_detects_drift() {
        let archive = standard_archive();
        let fixture = fixture(&archive, &SOURCE);
        let (notice, pins) = render(&fixture.root.0, &environment_of(&fixture)).unwrap();
        let lock = fs::read(fixture.root.0.join("Cargo.lock")).unwrap();
        let mut expected = HEADER.concat().into_bytes();
        expected.extend_from_slice(format!("Cargo.lock sha256 {}\n", digest(&lock)).as_bytes());
        expected.extend_from_slice(b"\n===== demo 1.0.0 =====\n");
        section(&mut expected, "licenses/LICENSE-MIT", b"mit body\n");
        section(&mut expected, "licenses-extra/LICENSE", b"extra body\n");
        expected.extend_from_slice(
            b"\n===== Rust library upstream notice and license inventory =====\n",
        );
        section(&mut expected, "COPYRIGHT-library.html", b"<p>rust</p>\n");
        section(&mut expected, "licenses/Apache-2.0.txt", b"apache\n");
        section(&mut expected, "licenses/MIT.txt", b"mit\n");
        assert_eq!(notice, expected);
        let expected_pins = format!(
            "{}  LICENSE\n{}  {NOTICE}\n{}  Cargo.lock\n{}  rust-toolchain.toml\n",
            digest(b"root license\n"),
            digest(&expected),
            digest(&lock),
            digest(b"[toolchain]\nchannel = \"1.88.0\"\n")
        );
        assert_eq!(String::from_utf8(pins.clone()).unwrap(), expected_pins);
        fs::create_dir_all(fixture.root.0.join("scripts")).unwrap();
        freeze(&fixture.root.0, &notice, &pins, false).unwrap();
        freeze(&fixture.root.0, &notice, &pins, true).unwrap();
        fixture.root.write(NOTICE, b"altered");
        assert_eq!(freeze(&fixture.root.0, &notice, &pins, true), Err(DRIFT));
    }

    #[test]
    fn tampered_missing_ambiguous_and_linked_inputs_fail_closed() {
        let archive = standard_archive();

        let altered = fixture(&archive, &[("licenses/LICENSE-MIT", b"changed\n")]);
        assert_eq!(render_of(&altered), Err(LICENSE_MEMBER));

        let linked_member = crate_archive(
            &[("demo-1.0.0/src/lib.rs", b"x\n")],
            &[("demo-1.0.0/LICENSE", "src/lib.rs")],
            &["demo-1.0.0/src/lib.rs", "demo-1.0.0/LICENSE"],
        );
        let linked = fixture(&linked_member, &[("LICENSE", b"x\n")]);
        assert_eq!(render_of(&linked), Err(LICENSE_MEMBER));

        let absent_member = fixture(
            &archive,
            &[
                ("licenses/LICENSE-MIT", b"mit body\n"),
                ("COPYRIGHT", b"extra\n"),
            ],
        );
        assert_eq!(render_of(&absent_member), Err(LICENSE_MEMBER));

        let wrong_checksum = fixture(&archive, &SOURCE);
        fs::write(
            wrong_checksum
                .home
                .0
                .join("registry/cache/index.test/demo-1.0.0.crate"),
            b"other",
        )
        .unwrap();
        assert_eq!(render_of(&wrong_checksum), Err(PACKAGE));

        let twice = fixture(&archive, &SOURCE);
        twice
            .home
            .write("registry/src/index.other/demo-1.0.0/LICENSE", b"copy\n");
        assert_eq!(render_of(&twice), Err(CACHE));

        let unlicensed = fixture(&archive, &SOURCE[2..]);
        assert_eq!(render_of(&unlicensed), Err(LICENSE_SOURCE));

        let mut unlocked = fixture(&archive, &SOURCE);
        unlocked.tree.push_str("absent v9.9.9\n");
        assert_eq!(render_of(&unlocked), Err(LOCK_MISMATCH));

        let no_copyright = fixture(&archive, &SOURCE);
        fs::remove_file(
            no_copyright
                .sysroot
                .0
                .join("share/doc/rust/COPYRIGHT-library.html"),
        )
        .unwrap();
        assert_eq!(render_of(&no_copyright), Err(RUST_LIBRARY));

        let link = fixture(&archive, &SOURCE);
        std::os::unix::fs::symlink(
            link.home
                .0
                .join("registry/src/index.test/demo-1.0.0/src/lib.rs"),
            link.home
                .0
                .join("registry/src/index.test/demo-1.0.0/COPYING"),
        )
        .unwrap();
        assert_eq!(render_of(&link), Err(SPECIAL));
    }
}

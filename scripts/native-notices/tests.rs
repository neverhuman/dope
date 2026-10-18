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

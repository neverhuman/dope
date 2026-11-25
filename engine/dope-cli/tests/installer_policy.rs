//! Installer boundary fixtures use fake gh/curl and executable bytes, never real model weights.
//! These prove invocation, staging and fail-closed policy; actual OIDC/release proof is separate.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::os::unix::fs::{symlink, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

const ENGINE: &str = "dope-engine-v3.0.0";
const MODEL: &str = "v1.0.0";
const SOURCE: &str = "e15c415bad9554f7c419dd240aa670d1a79a27fe";
const MEMBERS: [&str; 3] = [
    "MODEL_MANIFEST.json",
    "dope_mlc_v3_multitask_best.safetensors",
    "packet_high_regression.dml3",
];
fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn executable(path: &Path, text: &str) {
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
fn sums(dir: &Path, names: &[&str]) {
    let content = names
        .iter()
        .map(|name| format!("{}  {name}\n", digest(&fs::read(dir.join(name)).unwrap())))
        .collect::<String>();
    fs::write(dir.join("SHA256SUMS"), content).unwrap();
}
struct Fixture {
    root: PathBuf,
    engine: PathBuf,
    models: PathBuf,
    prefix: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let base = repo().join("target/installer-policy");
        fs::create_dir_all(&base).unwrap();
        let root = base.join(format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let f = Self {
            engine: root.join("engine release"),
            models: root.join("model release"),
            prefix: root.join("installed prefix"),
            root,
        };
        for path in [
            &f.engine,
            &f.models,
            &f.root.join("tools"),
            &f.root.join("scratch"),
        ] {
            fs::create_dir(path).unwrap();
        }
        executable(
            &f.root.join("tools/gh"),
            r#"#!/bin/bash
case "$*" in
  --version) printf 'gh version %s (policy fixture)\n' "${GH_FIXTURE_VERSION:-2.102.0}"; exit 0 ;;
  'attestation verify --help')
    printf '%s\n' '--bundle --hostname --repo --signer-workflow --source-digest --format'
    [[ "${GH_FIXTURE_NO_SOURCE_REF:-0}" == 1 ]] || printf '%s\n' '--source-ref'
    exit 0 ;;
esac
printf '%s\n' "$@" >> "$GH_FIXTURE_LOG"
[[ $4 == --bundle && $(<"$5") == fixture-bundle ]] || exit 1
[[ "${GH_FIXTURE_MODE:-ok}" == ok ]]
"#,
        );
        executable(
            &f.root.join("tools/curl"),
            r#"#!/bin/bash
printf '%s\n' "$@" >> "$CURL_FIXTURE_LOG"
[[ "${CURL_FIXTURE_MODE:-refuse}" == assets ]] || exit 99
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) destination=$2; shift 2 ;;
    https://*) url=$1; shift ;;
    *) shift ;;
  esac
done
case "$url" in
  https://github.com/cli/cli/releases/download/v2.102.0/gh_2.102.0_linux_amd64.tar.gz)
    printf 'invalid verifier archive' > "$destination"; exit 0 ;;
  https://github.com/neverhuman/dope/releases/download/dope-engine-v3.0.0/*) source="$ENGINE_FIXTURE_SOURCE/${url##*/}" ;;
  https://github.com/neverhuman/dope/releases/download/v1.0.0/*) source="$MODEL_FIXTURE_SOURCE/${url##*/}" ;;
  *) exit 98 ;;
esac
cp -- "$source" "$destination"
"#,
        );
        let binary = r#"#!/bin/bash
printf '%s\n' "$*" >> "$DOPE_FIXTURE_RUN_LOG"
exit 0
"#;
        fs::write(f.engine.join("dope-linux-x86_64"), binary).unwrap();
        fs::write(
            f.engine.join("BUILDINFO.json"),
            serde_json::to_vec(&serde_json::json!({
                "source_commit":SOURCE,"binary_sha256":digest(binary.as_bytes())
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(f.engine.join("LICENSE"), b"policy fixture license").unwrap();
        for name in ["Cargo.lock", "DEPENDENCIES.txt", "dope.sigstore.json"] {
            fs::write(f.engine.join(name), b"fixture-bundle").unwrap();
        }
        fs::write(
            f.engine.join("THIRD_PARTY_NOTICES"),
            b"policy fixture notices",
        )
        .unwrap();
        f.engine_sums(false);
        let weights = b"synthetic policy fixture; not safetensors";
        let wire = b"synthetic policy fixture; not DML3";
        fs::write(f.models.join(MEMBERS[1]), weights).unwrap();
        fs::write(f.models.join(MEMBERS[2]), wire).unwrap();
        fs::write(f.models.join(MEMBERS[0]), serde_json::to_vec(&serde_json::json!({
            "release_tag":MODEL,
            "models":[{"filename":MEMBERS[1],"sha256":digest(weights),"bytes":weights.len()}],
            "sample_wire_fixture":{"filename":MEMBERS[2],"sha256":digest(wire),"bytes":wire.len()}
        })).unwrap()).unwrap();
        sums(&f.models, &MEMBERS);
        f
    }
    fn engine_sums(&self, models: bool) {
        let mut names = vec![
            "BUILDINFO.json",
            "Cargo.lock",
            "DEPENDENCIES.txt",
            "LICENSE",
            "THIRD_PARTY_NOTICES",
            "dope-linux-x86_64",
        ];
        if models {
            names.extend(MEMBERS);
        }
        sums(&self.engine, &names);
    }
    fn base_command(&self) -> Command {
        let mut c = Command::new("/bin/bash");
        c.arg(repo().join("scripts/install.sh"))
            .arg("--color")
            .arg("never")
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("tools").display()),
            )
            .env("TMPDIR", self.root.join("scratch"))
            .env("GH_FIXTURE_LOG", self.root.join("gh.log"))
            .env("GH_FIXTURE_VERSION", "2.102.0")
            .env("GH_FIXTURE_NO_SOURCE_REF", "0")
            .env("GH_FIXTURE_MODE", "ok")
            .env("CURL_FIXTURE_LOG", self.root.join("curl.log"))
            .env("CURL_FIXTURE_MODE", "refuse")
            .env("ENGINE_FIXTURE_SOURCE", &self.engine)
            .env("MODEL_FIXTURE_SOURCE", &self.models)
            .env("DOPE_FIXTURE_RUN_LOG", self.root.join("executed.log"))
            .env_remove("DOPE_RELEASE_TAG")
            .env_remove("DOPE_MODEL_TAG")
            .env_remove("DOPE_SOURCE_COMMIT")
            .env_remove("DOPE_INSTALL_V8_WEIGHTS");
        c
    }
    fn command(&self) -> Command {
        let mut c = self.base_command();
        c.arg("--install-dir")
            .arg(self.prefix.join("bin"))
            .arg("--share-dir")
            .arg(self.prefix.join("share"))
            .arg("--weights-dir")
            .arg(self.prefix.join("weights"));
        c
    }
    fn standalone(&self) -> Command {
        let mut c = self.base_command();
        c.arg("--standalone-prefix").arg(&self.prefix);
        c
    }
    fn split(&self) -> Command {
        let mut c = self.command();
        c.args(["--tag", ENGINE, "--source-digest", SOURCE])
            .arg("--from-dir")
            .arg(&self.engine)
            .arg("--model-from-dir")
            .arg(&self.models);
        c
    }
    fn refused(&self, out: Output, message: &str) {
        assert!(!out.status.success(), "unexpected success");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(message),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !self.prefix.exists(),
            "verification failure installed a destination"
        );
        assert!(
            !self.root.join("executed.log").exists(),
            "verification failure executed a binary"
        );
    }
}
fn succeeded(out: Output) -> String {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn signed_engine_and_independent_model_release_use_separate_inventories() {
    let f = Fixture::new();
    let text = succeeded(f.split().output().unwrap());
    assert!(text.contains(ENGINE) && text.contains(MODEL));
    assert!(text.contains("sigstore-oidc-attestation + sha256-content-integrity"));
    assert!(text.contains("model verification") && text.contains("sha256-content-integrity-only"));
    assert_eq!(
        fs::read(f.prefix.join("weights").join(MEMBERS[1])).unwrap(),
        fs::read(f.models.join(MEMBERS[1])).unwrap()
    );
    let log = fs::read_to_string(f.root.join("gh.log")).unwrap();
    assert_eq!(log.matches("attestation\nverify\n").count(), 2);
    for expected in [
        "--repo\nneverhuman/dope\n",
        "--hostname\ngithub.com\n",
        "--bundle\n",
        "--signer-workflow\nneverhuman/dope/.github/workflows/engine-release.yml\n",
        &format!("--source-ref\nrefs/tags/{ENGINE}\n"),
        &format!("--source-digest\n{SOURCE}\n"),
    ] {
        assert_eq!(log.matches(expected).count(), 2, "{expected}");
    }
    assert!(!f.root.join("curl.log").exists());
}

#[test]
fn default_engine_uses_pinned_signed_source_and_skips_independent_models() {
    let f = Fixture::new();
    let text = succeeded(
        f.command()
            .env("CURL_FIXTURE_MODE", "assets")
            .output()
            .unwrap(),
    );
    assert!(text.contains(&format!("concrete release tag {ENGINE}")));
    let requests = fs::read_to_string(f.root.join("curl.log")).unwrap();
    assert!(requests.contains(&format!("/download/{ENGINE}/dope-linux-x86_64")));
    assert!(!requests.contains(&format!("/download/{MODEL}/")));
    assert!(!requests.contains(&format!("/download/{ENGINE}/MODEL_MANIFEST.json")));
    let log = fs::read_to_string(f.root.join("gh.log")).unwrap();
    assert!(!log.contains("api\n"));
    assert_eq!(
        log.matches(&format!("--source-ref\nrefs/tags/{ENGINE}\n"))
            .count(),
        2
    );
    assert_eq!(
        log.matches(&format!("--source-digest\n{SOURCE}\n")).count(),
        2
    );
    assert!(!f.prefix.join("weights").exists());
}

#[test]
fn online_split_fetch_uses_the_engine_tag_and_independent_model_tag() {
    let f = Fixture::new();
    let out = f
        .command()
        .args(["--tag", ENGINE, "--source-digest", SOURCE])
        .args(["--with-weights", "--model-tag", MODEL])
        .env("CURL_FIXTURE_MODE", "assets")
        .output()
        .unwrap();
    succeeded(out);
    let requests = fs::read_to_string(f.root.join("curl.log")).unwrap();
    assert!(requests.contains(&format!("/download/{ENGINE}/dope-linux-x86_64")));
    assert!(requests.contains(&format!("/download/{MODEL}/MODEL_MANIFEST.json")));
    assert!(requests.contains(&format!(
        "/download/{MODEL}/dope_mlc_v3_multitask_best.safetensors"
    )));
    assert!(!requests.contains(&format!("/download/{ENGINE}/MODEL_MANIFEST.json")));
    assert!(f.prefix.join("weights").join(MEMBERS[1]).is_file());
}

#[test]
fn explicit_legacy_tag_and_one_directory_engine_only_offline_behavior_remain_supported() {
    for (engine_tag, with_models) in [(false, true), (false, false), (true, false)] {
        let f = Fixture::new();
        if with_models {
            for name in MEMBERS {
                fs::copy(f.models.join(name), f.engine.join(name)).unwrap();
            }
            f.engine_sums(true);
        }
        let mut command = f.command();
        command.arg("--from-dir").arg(&f.engine);
        if engine_tag {
            command.args(["--tag", ENGINE, "--source-digest", SOURCE]);
        } else {
            command
                .args(["--tag", MODEL])
                .env("GH_FIXTURE_VERSION", "2.62.0");
        }
        let text = succeeded(command.output().unwrap());
        assert!(text.contains(if engine_tag {
            ENGINE
        } else {
            "concrete release tag v1.0.0"
        }));
        assert!(f.prefix.join("bin/dope").is_file());
        assert_eq!(f.prefix.join("weights").exists(), with_models);
        assert!(!f.root.join("curl.log").exists());
    }
}

#[test]
fn old_or_missing_policy_flags_refuse_before_any_fetch_or_install() {
    for incomplete in [false, true] {
        let f = Fixture::new();
        let mut c = f.command();
        c.args(["--source-digest", SOURCE, "--no-weights"]);
        if incomplete {
            c.env("GH_FIXTURE_NO_SOURCE_REF", "1");
        } else {
            c.env("GH_FIXTURE_VERSION", "2.62.0");
        }
        f.refused(c.output().unwrap(), "upgrade gh to 2.102.0 or newer");
        assert!(!f.root.join("curl.log").exists());
        assert!(!f.root.join("gh.log").exists());
    }
}

#[test]
fn wrong_model_release_and_tampered_model_refuse_before_install_or_execution() {
    for wrong_tag in [true, false] {
        let f = Fixture::new();
        if wrong_tag {
            let path = f.models.join(MEMBERS[0]);
            let mut manifest: serde_json::Value =
                serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            manifest["release_tag"] = serde_json::json!("v9.9.9");
            fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
            sums(&f.models, &MEMBERS); // Reach the tag check after valid content checks.
        } else {
            fs::write(f.models.join(MEMBERS[1]), b"tampered").unwrap();
        }
        f.refused(
            f.split().output().unwrap(),
            if wrong_tag {
                "requested model tag"
            } else {
                "SHA-256 mismatch"
            },
        );
    }
}

#[test]
fn oidc_refusal_has_no_unsigned_fallback_and_never_executes_or_installs() {
    let f = Fixture::new();
    let out = f.split().env("GH_FIXTURE_MODE", "reject").output().unwrap();
    assert!(!out.status.success());
    assert!(!f.prefix.exists() && !f.root.join("executed.log").exists());
    assert!(fs::read_to_string(f.root.join("gh.log"))
        .unwrap()
        .contains("attestation\nverify\n"));
}

#[test]
fn explicit_no_weights_never_opens_the_independent_model_source() {
    let f = Fixture::new();
    let text = succeeded(
        f.command()
            .args(["--tag", ENGINE, "--source-digest", SOURCE, "--no-weights"])
            .arg("--from-dir")
            .arg(&f.engine)
            .arg("--model-from-dir")
            .arg(f.root.join("missing model source"))
            .output()
            .unwrap(),
    );
    assert!(text.contains("Installed"));
    assert!(!f.prefix.join("weights").exists() && !f.root.join("curl.log").exists());
}

#[test]
fn explicit_offline_split_request_does_not_fall_back_to_a_download_or_relabel() {
    let f = Fixture::new();
    let out = f
        .command()
        .args([
            "--tag",
            ENGINE,
            "--source-digest",
            SOURCE,
            "--model-tag",
            MODEL,
        ])
        .arg("--from-dir")
        .arg(&f.engine)
        .output()
        .unwrap();
    f.refused(out, "--model-from-dir");
    assert!(!f.root.join("curl.log").exists());
}

#[test]
fn partial_offline_models_and_empty_model_selectors_fail_without_fetching() {
    for (flag, value) in [
        ("--model-tag", ""),
        ("--model-from-dir", ""),
        ("--model-tag", "latest"),
    ] {
        let f = Fixture::new();
        let out = f.command().args([flag, value]).output().unwrap();
        assert!(!out.status.success());
        assert!(!f.prefix.exists() && !f.root.join("executed.log").exists());
        assert!(!f.root.join("curl.log").exists());
    }
    let f = Fixture::new();
    fs::copy(f.models.join(MEMBERS[0]), f.engine.join(MEMBERS[0])).unwrap();
    f.refused(
        f.command()
            .args(["--tag", MODEL])
            .arg("--from-dir")
            .arg(&f.engine)
            .output()
            .unwrap(),
        "release asset",
    );
    assert!(!f.root.join("curl.log").exists());
}

#[test]
fn missing_tools_and_old_glibc_refuse_before_download() {
    let f = Fixture::new();
    f.refused(
        f.standalone().env("PATH", "").output().unwrap(),
        "required tool missing: uname",
    );
    executable(
        &f.root.join("tools/getconf"),
        "#!/bin/bash\nprintf '%s\\n' 'glibc 2.35'\n",
    );
    f.refused(
        f.standalone().output().unwrap(),
        "requires glibc 2.39 or newer",
    );
    assert!(!f.root.join("curl.log").exists());
    let limited = f.root.join("limited");
    fs::create_dir(&limited).unwrap();
    for name in ["uname", "getconf", "sha256sum"] {
        let system = ["/usr/bin", "/bin"]
            .iter()
            .map(|dir| Path::new(dir).join(name))
            .find(|path| path.is_file())
            .unwrap();
        symlink(system, limited.join(name)).unwrap();
    }
    f.refused(
        f.standalone().env("PATH", limited).output().unwrap(),
        "required tool missing: jq",
    );
}

#[test]
fn wrong_source_missing_bundle_invalid_bundle_and_bad_checksums_refuse() {
    let f = Fixture::new();
    f.refused(
        f.standalone()
            .args([
                "--source-digest",
                "0123456789abcdef0123456789abcdef01234567",
            ])
            .output()
            .unwrap(),
        "pinned dope-engine-v3.0.0 release commit",
    );
    assert!(!f.root.join("curl.log").exists());
    let bundle = f.engine.join("dope.sigstore.json");
    fs::remove_file(&bundle).unwrap();
    let offline = || {
        f.standalone()
            .arg("--from-dir")
            .arg(&f.engine)
            .output()
            .unwrap()
    };
    f.refused(offline(), "dope.sigstore.json");
    fs::write(&bundle, b"invalid fixture bundle").unwrap();
    f.refused(offline(), "attestation verification failed");
    fs::write(&bundle, b"fixture-bundle").unwrap();
    fs::write(f.engine.join("Cargo.lock"), b"tampered").unwrap();
    f.refused(offline(), "SHA-256 mismatch for Cargo.lock");
    let buildinfo = f.engine.join("BUILDINFO.json");
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(&buildinfo).unwrap()).unwrap();
    metadata["source_commit"] = serde_json::json!("0123456789abcdef0123456789abcdef01234567");
    fs::write(&buildinfo, serde_json::to_vec(&metadata).unwrap()).unwrap();
    f.engine_sums(false);
    f.refused(offline(), "BUILDINFO.json source_commit differs");
}

#[test]
fn rejects_bootstrap_archive_before_fetching_engine() {
    let f = Fixture::new();
    f.refused(
        f.standalone()
            .env("GH_FIXTURE_VERSION", "2.62.0")
            .env("CURL_FIXTURE_MODE", "assets")
            .output()
            .unwrap(),
        "GitHub CLI archive SHA-256 mismatch",
    );
    let downloads = fs::read_to_string(f.root.join("curl.log")).unwrap();
    assert!(downloads.contains("/cli/cli/releases/download/v2.102.0/"));
    assert!(!downloads.contains("/neverhuman/dope/releases/"));
}

#[test]
fn standalone_reinstall_preserves_files_and_refuses_conflicts() {
    let f = Fixture::new();
    let install = || {
        f.standalone()
            .arg("--from-dir")
            .arg(&f.engine)
            .output()
            .unwrap()
    };
    fs::create_dir(&f.prefix).unwrap();
    let blocked = f.standalone().output().unwrap();
    assert!(String::from_utf8_lossy(&blocked.stderr).contains("conflicting or symlinked"));
    assert!(!f.root.join("curl.log").exists());
    assert!(!f.root.join("gh.log").exists());
    assert!(!f.root.join("executed.log").exists());
    fs::remove_dir(&f.prefix).unwrap();
    symlink(&f.engine, &f.prefix).unwrap();
    assert!(!install().status.success());
    assert!(!f.root.join("gh.log").exists());
    fs::remove_file(&f.prefix).unwrap();
    let binary = f.prefix.join("bin/dope");
    let text = succeeded(install());
    assert!(text.contains(binary.canonicalize().unwrap().to_str().unwrap()));
    assert!(text.contains("export PATH="));
    assert!(f.prefix.join("share/dope-engine/LICENSE").is_file());
    let before = fs::metadata(&binary).unwrap();
    fs::set_permissions(&f.root, fs::Permissions::from_mode(0o555)).unwrap();
    let reinstall = install();
    fs::set_permissions(&f.root, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(succeeded(reinstall).contains("Already installed"));
    let after = fs::metadata(&binary).unwrap();
    assert_eq!(
        (before.ino(), before.modified().unwrap()),
        (after.ino(), after.modified().unwrap())
    );
    assert_eq!(
        fs::read_to_string(f.root.join("executed.log")).unwrap(),
        "doctor --json\n"
    );
    let license = f.prefix.join("share/dope-engine/LICENSE");
    fs::write(&license, "conflicting notice").unwrap();
    assert!(!install().status.success());
    assert_eq!(fs::read_to_string(license).unwrap(), "conflicting notice");
    assert!(!f.root.join("curl.log").exists());
}

#[test]
fn path_hint_is_printed_only_when_the_install_dir_is_absent() {
    let f = Fixture::new();
    let text = succeeded(
        f.command()
            .env(
                "PATH",
                format!(
                    "{}:{}:/usr/bin:/bin",
                    f.root.join("tools").display(),
                    f.prefix.join("bin").display()
                ),
            )
            .env("CURL_FIXTURE_MODE", "assets")
            .output()
            .unwrap(),
    );
    assert!(!text.contains("export PATH="), "{text}");
    assert!(text.contains("dope --version"));
}

#[test]
fn help_names_with_weights_and_no_unpublished_engine_tag() {
    let f = Fixture::new();
    let text = succeeded(f.command().arg("--help").output().unwrap());
    assert!(text.contains("--with-weights"));
    assert!(!text.contains("dope-engine-v3.1.0"));
    assert!(!f.prefix.exists());
}

#[test]
fn truncated_installer_prefix_does_not_install() {
    let source = fs::read_to_string(repo().join("scripts/install.sh")).unwrap();
    let lines: Vec<&str> = source.lines().collect();
    for count in [50, 150, 300, lines.len() - 1] {
        let f = Fixture::new();
        let mut child = Command::new("/bin/bash")
            .args(["-s", "--", "--standalone-prefix"])
            .arg(&f.prefix)
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", f.root.join("tools").display()),
            )
            .env("CURL_FIXTURE_LOG", f.root.join("curl.log"))
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let prefix = lines[..count].join("\n") + "\n";
        child
            .stdin
            .take()
            .unwrap()
            .write_all(prefix.as_bytes())
            .unwrap();
        child.wait().unwrap();
        assert!(!f.prefix.exists(), "{count}-line prefix installed");
        assert!(
            !f.root.join("curl.log").exists(),
            "{count}-line prefix fetched"
        );
    }
}

#[test]
fn uninstall_removes_owned_files_and_keeps_siblings() {
    let f = Fixture::new();
    let installed = succeeded(
        f.command()
            .args(["--tag", ENGINE, "--source-digest", SOURCE, "--from-dir"])
            .arg(&f.engine)
            .output()
            .unwrap(),
    );
    assert!(installed.contains("Installed"), "{installed}");
    let sibling = f.prefix.join("bin/keep-me");
    fs::write(&sibling, b"keep").unwrap();
    let removed = succeeded(
        f.command()
            .env("PATH", "/usr/bin:/bin")
            .arg("--uninstall")
            .output()
            .unwrap(),
    );
    assert!(
        removed.contains("uninstalled owned dope files"),
        "{removed}"
    );
    assert!(!f.prefix.join("bin/dope").exists());
    assert!(!f.prefix.join("share/LICENSE").exists());
    assert!(!f.prefix.join("share/THIRD_PARTY_NOTICES").exists());
    assert!(!f.prefix.join("share/BUILDINFO.json").exists());
    assert_eq!(fs::read(&sibling).unwrap(), b"keep");
    let again = succeeded(
        f.command()
            .env("PATH", "/usr/bin:/bin")
            .arg("--uninstall")
            .output()
            .unwrap(),
    );
    assert!(again.contains("uninstalled owned dope files"), "{again}");
    assert_eq!(fs::read(&sibling).unwrap(), b"keep");
}

#[test]
fn uninstall_refuses_a_symlinked_binary() {
    let f = Fixture::new();
    let install = f.prefix.join("bin");
    fs::create_dir_all(&install).unwrap();
    let real = f.root.join("real-dope");
    fs::write(&real, b"real").unwrap();
    symlink(&real, install.join("dope")).unwrap();
    let sibling = install.join("keep-me");
    fs::write(&sibling, b"keep").unwrap();
    let out = f
        .command()
        .env("PATH", "/usr/bin:/bin")
        .arg("--uninstall")
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{err}");
    assert!(err.contains("refusing to remove symlink"), "{err}");
    assert_eq!(fs::read(&real).unwrap(), b"real");
    assert_eq!(fs::read(&sibling).unwrap(), b"keep");
    assert!(install
        .join("dope")
        .symlink_metadata()
        .unwrap()
        .file_type()
        .is_symlink());
}

#[test]
fn symlink_install_dir_is_refused() {
    let f = Fixture::new();
    let real = f.root.join("real-bin");
    fs::create_dir(&real).unwrap();
    fs::create_dir_all(&f.prefix).unwrap();
    symlink(&real, f.prefix.join("bin")).unwrap();
    let out = f
        .command()
        .args(["--tag", ENGINE, "--source-digest", SOURCE, "--from-dir"])
        .arg(&f.engine)
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{err}");
    assert!(err.contains("refusing to install through symlink"), "{err}");
    assert!(!real.join("dope").exists());
    assert!(!f.prefix.join("share").exists());
}

#[test]
fn read_only_share_dir_leaves_the_existing_binary_and_no_temp_files() {
    let f = Fixture::new();
    let install = f.prefix.join("bin");
    let share = f.prefix.join("share");
    fs::create_dir_all(&install).unwrap();
    fs::create_dir_all(&share).unwrap();
    let existing = b"existing-dope";
    fs::write(install.join("dope"), existing).unwrap();
    let mut perms = fs::metadata(&share).unwrap().permissions();
    perms.set_mode(0o555);
    fs::set_permissions(&share, perms).unwrap();
    let out = f
        .command()
        .args(["--tag", ENGINE, "--source-digest", SOURCE, "--from-dir"])
        .arg(&f.engine)
        .output()
        .unwrap();
    let mut perms = fs::metadata(&share).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&share, perms).unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{err}");
    assert!(err.contains("destination is not writable"), "{err}");
    assert_eq!(fs::read(install.join("dope")).unwrap(), existing);
    for dir in [&install, &share] {
        for entry in fs::read_dir(dir).unwrap() {
            let name = entry.unwrap().file_name();
            let name = name.to_string_lossy();
            assert!(!name.contains(".tmp."), "{name}");
            assert!(!name.starts_with(".dope-write-probe."), "{name}");
        }
    }
}

#[test]
fn installer_and_verifier_use_shared_gh_policy() {
    let install = fs::read_to_string(repo().join("scripts/install.sh")).unwrap();
    let verify = fs::read_to_string(repo().join("scripts/verify-engine-release.sh")).unwrap();
    assert!(verify.contains("source ") && verify.contains("/install.sh"));
    for name in ["gh_policy_supported", "verify_attestation"] {
        assert!(install.contains(&format!("{name}()")));
        assert!(verify.contains(&format!("{name} ")));
        assert!(!verify.contains(&format!("{name}()")));
    }
}

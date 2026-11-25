//! Installer policy tests for optional attested `SingleEncV8` engine release weights.
//! Uses synthetic fixtures and fake `gh`/`curl`/binary stubs; real OIDC proof belongs to tag runs.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
use sha2::{Digest, Sha256};
use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const ENGINE_V31: &str = "dope-engine-v3.1.0";
const ENGINE_V30: &str = "dope-engine-v3.0.0";
const SOURCE: &str = "fedcba9876543210fedcba9876543210fedcba98";
const V30_SOURCE: &str = "e15c415bad9554f7c419dd240aa670d1a79a27fe";
const CORE: [&str; 6] = [
    "BUILDINFO.json",
    "Cargo.lock",
    "DEPENDENCIES.txt",
    "LICENSE",
    "THIRD_PARTY_NOTICES",
    "dope-linux-x86_64",
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
fn write_sums(dir: &Path, names: &[&str]) {
    let body: String = names
        .iter()
        .map(|n| format!("{}  {n}\n", digest(&fs::read(dir.join(n)).unwrap())))
        .collect();
    fs::write(dir.join("SHA256SUMS"), body).unwrap();
}

struct V8Fixture {
    root: PathBuf,
    engine: PathBuf,
    prefix: PathBuf,
    weights_bytes: Vec<u8>,
}

impl Drop for V8Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

impl V8Fixture {
    fn new(with_v8: bool) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "dope-v8-inst-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&root);
        let engine = root.join("engine");
        let prefix = root.join("prefix");
        for d in [&engine, &root.join("tools"), &root.join("scratch")] {
            fs::create_dir_all(d).unwrap();
        }
        executable(
            &root.join("tools/gh"),
            r#"#!/bin/bash
case "$*" in
  --version) printf '%s\n' 'gh version 2.102.0 (policy fixture)'; exit 0 ;;
  'attestation verify --help') printf '%s\n' '--bundle --hostname --repo --signer-workflow --source-ref --source-digest --format'; exit 0 ;;
esac
printf '%s\n' "$@" >> "$GH_LOG"
if [[ -n "${GH_REJECT_SUBJECT:-}" && "$*" == *"$GH_REJECT_SUBJECT"* ]]; then exit 1; fi
[[ "${GH_MODE:-ok}" == ok ]]
"#,
        );
        executable(
            &root.join("tools/curl"),
            r#"#!/bin/bash
printf '%s\n' "$@" >> "$CURL_LOG"
[[ "${CURL_MODE:-refuse}" == assets ]] || exit 99
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) dst=$2; shift 2 ;;
    https://*) url=$1; shift ;;
    *) shift ;;
  esac
done
case "$url" in
  https://github.com/neverhuman/dope/releases/download/dope-engine-v3.1.0/*) src="$ENGINE_SRC/${url##*/}" ;;
  *) exit 98 ;;
esac
cp -- "$src" "$dst"
"#,
        );
        let bin = "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$RUN_LOG\"\nexit 0\n";
        fs::write(engine.join("dope-linux-x86_64"), bin).unwrap();
        fs::write(engine.join("LICENSE"), b"MIT").unwrap();
        fs::write(engine.join("THIRD_PARTY_NOTICES"), b"notices").unwrap();
        for name in ["Cargo.lock", "DEPENDENCIES.txt", "dope.sigstore.json"] {
            fs::write(engine.join(name), b"policy fixture").unwrap();
        }
        let weights_bytes = b"synthetic SingleEncV8 safetensors payload".to_vec();
        let w_sha = digest(&weights_bytes);
        if with_v8 {
            fs::write(
                engine.join("BUILDINFO.json"),
                serde_json::to_vec(&serde_json::json!({
                    "source_commit": SOURCE,
                    "binary_sha256": digest(bin.as_bytes()),
                    "v8_weights_sha256": w_sha,
                    "v8_weights_bytes": weights_bytes.len()
                }))
                .unwrap(),
            )
            .unwrap();
            fs::write(
                engine.join("V8_WEIGHTS_MANIFEST.json"),
                serde_json::to_vec(&serde_json::json!({
                    "schema": "dope.single-enc-v8-weights/v1",
                    "file": "model.safetensors",
                    "arch": "single_enc_v8",
                    "bytes": weights_bytes.len(),
                    "sha256": w_sha
                }))
                .unwrap(),
            )
            .unwrap();
            fs::write(engine.join("dope-v8-weights.safetensors"), &weights_bytes).unwrap();
            write_sums(
                &engine,
                &[
                    "BUILDINFO.json",
                    "Cargo.lock",
                    "DEPENDENCIES.txt",
                    "LICENSE",
                    "THIRD_PARTY_NOTICES",
                    "V8_WEIGHTS_MANIFEST.json",
                    "dope-linux-x86_64",
                    "dope-v8-weights.safetensors",
                ],
            );
        } else {
            fs::write(
                engine.join("BUILDINFO.json"),
                serde_json::to_vec(&serde_json::json!({
                    "source_commit": V30_SOURCE,
                    "binary_sha256": digest(bin.as_bytes())
                }))
                .unwrap(),
            )
            .unwrap();
            write_sums(&engine, &CORE);
        }
        Self {
            root,
            engine,
            prefix,
            weights_bytes,
        }
    }

    fn cmd(&self) -> Command {
        let mut c = Command::new("/bin/bash");
        c.arg(repo().join("scripts/install.sh"))
            .arg("--install-dir")
            .arg(self.prefix.join("bin"))
            .arg("--share-dir")
            .arg(self.prefix.join("share"))
            .arg("--weights-dir")
            .arg(self.prefix.join("weights"))
            .arg("--color")
            .arg("never")
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("tools").display()),
            )
            .env("TMPDIR", self.root.join("scratch"))
            .env("GH_LOG", self.root.join("gh.log"))
            .env("CURL_LOG", self.root.join("curl.log"))
            .env("ENGINE_SRC", &self.engine)
            .env("RUN_LOG", self.root.join("executed.log"))
            .env_remove("DOPE_RELEASE_TAG")
            .env_remove("DOPE_MODEL_TAG")
            .env_remove("DOPE_SOURCE_COMMIT")
            .env_remove("DOPE_INSTALL_V8_WEIGHTS");
        c
    }

    fn assert_refused(&self, out: Output, needle: &str) {
        assert!(!out.status.success(), "unexpected success");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains(needle), "stderr missing {needle:?}: {err}");
        assert!(
            !self.prefix.join("bin/dope").exists(),
            "refusal installed dope binary"
        );
        assert!(
            !self
                .prefix
                .join("weights/single-enc-v8/model.safetensors")
                .exists(),
            "refusal installed v8 weights"
        );
        assert!(
            !self.root.join("executed.log").exists(),
            "refusal executed unverified binary"
        );
    }
}

#[test]
fn online_and_offline_v8_weights_verify_four_subjects_and_skip_mlc() {
    for flags in [
        &["--no-weights", "--v8-weights"][..],
        &["--v8-weights", "--no-weights"][..],
    ] {
        let f = V8Fixture::new(true);
        let out = f
            .cmd()
            .args(["--tag", ENGINE_V31, "--source-digest", SOURCE])
            .args(flags)
            .env("CURL_MODE", "assets")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(text.contains("v8 verification"));
        assert!(text.contains("not quality or privacy certification"));
        assert_eq!(
            fs::read(f.prefix.join("weights/single-enc-v8/model.safetensors")).unwrap(),
            f.weights_bytes
        );
        assert!(f
            .prefix
            .join("weights/single-enc-v8/manifest.json")
            .is_file());
        assert!(!f.prefix.join("weights/MODEL_MANIFEST.json").exists());
        let gh_log = fs::read_to_string(f.root.join("gh.log")).unwrap();
        assert_eq!(gh_log.matches("attestation\nverify\n").count(), 4);
        assert!(gh_log.contains("dope-v8-weights.safetensors"));
        assert!(gh_log.contains("V8_WEIGHTS_MANIFEST.json"));
        let curl_log = fs::read_to_string(f.root.join("curl.log")).unwrap();
        assert!(curl_log.contains(&format!(
            "/download/{ENGINE_V31}/dope-v8-weights.safetensors"
        )));
        assert!(!curl_log.contains("MODEL_MANIFEST.json"));
    }

    // One-directory offline v3.1.0 auto-installs bundled SingleEncV8 weights without HTTPS.
    let f = V8Fixture::new(true);
    let out = f
        .cmd()
        .args(["--tag", ENGINE_V31, "--source-digest", SOURCE, "--from-dir"])
        .arg(&f.engine)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(f
        .prefix
        .join("weights/single-enc-v8/model.safetensors")
        .is_file());
    assert!(!f.root.join("curl.log").exists());

    // --no-v8-weights skips bundled SingleEncV8 weights and verifies only 2 core subjects.
    let f2 = V8Fixture::new(true);
    let out2 = f2
        .cmd()
        .args([
            "--tag",
            ENGINE_V31,
            "--source-digest",
            SOURCE,
            "--no-v8-weights",
            "--from-dir",
        ])
        .arg(&f2.engine)
        .output()
        .unwrap();
    assert!(
        out2.status.success(),
        "{}",
        String::from_utf8_lossy(&out2.stderr)
    );
    assert!(!f2.prefix.join("weights/single-enc-v8").exists());
    let gh_log2 = fs::read_to_string(f2.root.join("gh.log")).unwrap();
    assert_eq!(gh_log2.matches("attestation\nverify\n").count(), 2);
}

#[test]
fn v8_weights_failures_refuse_before_execution_or_install() {
    // 1. Requiring --v8-weights on weight-free dope-engine-v3.0.0 fails closed.
    let wf = V8Fixture::new(false);
    let out = wf
        .cmd()
        .args([
            "--tag",
            ENGINE_V30,
            "--source-digest",
            V30_SOURCE,
            "--no-weights",
            "--v8-weights",
            "--from-dir",
        ])
        .arg(&wf.engine)
        .output()
        .unwrap();
    wf.assert_refused(out, "does not ship SingleEncV8 weights");

    // 2. Partial SHA256SUMS (manifest without model) fails closed.
    let partial = V8Fixture::new(true);
    write_sums(
        &partial.engine,
        &[
            "BUILDINFO.json",
            "Cargo.lock",
            "DEPENDENCIES.txt",
            "LICENSE",
            "THIRD_PARTY_NOTICES",
            "V8_WEIGHTS_MANIFEST.json",
            "dope-linux-x86_64",
        ],
    );
    let out = partial
        .cmd()
        .args(["--tag", ENGINE_V31, "--source-digest", SOURCE, "--from-dir"])
        .arg(&partial.engine)
        .output()
        .unwrap();
    partial.assert_refused(out, "partial or duplicate SingleEncV8");

    // 3. Tampered SingleEncV8 weights fail SHA-256 check before execution.
    let tampered = V8Fixture::new(true);
    fs::write(tampered.engine.join("dope-v8-weights.safetensors"), b"evil").unwrap();
    let out = tampered
        .cmd()
        .args(["--tag", ENGINE_V31, "--source-digest", SOURCE, "--from-dir"])
        .arg(&tampered.engine)
        .output()
        .unwrap();
    tampered.assert_refused(out, "SHA-256 mismatch");

    // 4. BUILDINFO v8_weights_sha256 mismatch fails closed.
    let bad_bi = V8Fixture::new(true);
    let mut bi: serde_json::Value =
        serde_json::from_slice(&fs::read(bad_bi.engine.join("BUILDINFO.json")).unwrap()).unwrap();
    bi["v8_weights_sha256"] = serde_json::json!("0".repeat(64));
    fs::write(
        bad_bi.engine.join("BUILDINFO.json"),
        serde_json::to_vec(&bi).unwrap(),
    )
    .unwrap();
    write_sums(
        &bad_bi.engine,
        &[
            "BUILDINFO.json",
            "Cargo.lock",
            "DEPENDENCIES.txt",
            "LICENSE",
            "THIRD_PARTY_NOTICES",
            "V8_WEIGHTS_MANIFEST.json",
            "dope-linux-x86_64",
            "dope-v8-weights.safetensors",
        ],
    );
    let out = bad_bi
        .cmd()
        .args(["--tag", ENGINE_V31, "--source-digest", SOURCE, "--from-dir"])
        .arg(&bad_bi.engine)
        .output()
        .unwrap();
    bad_bi.assert_refused(out, "BUILDINFO.json v8_weights metadata");

    // 5. OIDC attestation refusal on dope-v8-weights.safetensors fails closed.
    let oidc = V8Fixture::new(true);
    let out = oidc
        .cmd()
        .args(["--tag", ENGINE_V31, "--source-digest", SOURCE, "--from-dir"])
        .arg(&oidc.engine)
        .env("GH_REJECT_SUBJECT", "dope-v8-weights.safetensors")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(!oidc.prefix.join("bin/dope").exists());
    assert!(!oidc.root.join("executed.log").exists());

    // 6. Symlinked destination directory fails before binary execution or partial install.
    let sym = V8Fixture::new(true);
    let decoy = sym.root.join("decoy");
    fs::create_dir_all(&decoy).unwrap();
    fs::create_dir_all(sym.prefix.join("weights")).unwrap();
    symlink(&decoy, sym.prefix.join("weights/single-enc-v8")).unwrap();
    let out = sym
        .cmd()
        .args(["--tag", ENGINE_V31, "--source-digest", SOURCE, "--from-dir"])
        .arg(&sym.engine)
        .output()
        .unwrap();
    sym.assert_refused(out, "symlink or non-directory path");
    assert!(!decoy.join("model.safetensors").exists());

    // 7. Legacy non-engine tags cannot install SingleEncV8 weights without attestation.
    let leg_req = V8Fixture::new(true);
    let out = leg_req
        .cmd()
        .args(["--tag", "v1.0.0", "--no-weights", "--v8-weights"])
        .output()
        .unwrap();
    leg_req.assert_refused(out, "attested dope-engine-v* release tag");
    assert!(!leg_req.root.join("curl.log").exists());
    assert!(!leg_req.root.join("gh.log").exists());

    let leg_auto = V8Fixture::new(true);
    let out = leg_auto
        .cmd()
        .args(["--tag", "v1.0.0", "--from-dir"])
        .arg(&leg_auto.engine)
        .output()
        .unwrap();
    leg_auto.assert_refused(out, "attested dope-engine-v* release tag");
    assert!(!leg_auto.root.join("gh.log").exists());

    // Legacy tag with --no-v8-weights skips bundled V8 weights and installs engine-only cleanly.
    let leg_skip = V8Fixture::new(true);
    let out = leg_skip
        .cmd()
        .args(["--tag", "v1.0.0", "--no-v8-weights", "--from-dir"])
        .arg(&leg_skip.engine)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(leg_skip.prefix.join("bin/dope").is_file());
    assert!(!leg_skip.prefix.join("weights/single-enc-v8").exists());

    // 8. Existing directory at V8 destination file paths (or file at single-enc-v8 dir) fails closed before execution.
    for collision in ["model.safetensors", "manifest.json"] {
        let dir_col = V8Fixture::new(true);
        let col_path = dir_col.prefix.join("weights/single-enc-v8").join(collision);
        fs::create_dir_all(&col_path).unwrap();
        let out = dir_col
            .cmd()
            .args(["--tag", ENGINE_V31, "--source-digest", SOURCE, "--from-dir"])
            .arg(&dir_col.engine)
            .output()
            .unwrap();
        assert!(
            !out.status.success(),
            "unexpected success on {collision} dir"
        );
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("non-regular path"),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(!dir_col.prefix.join("bin/dope").exists());
        assert!(!dir_col.root.join("executed.log").exists());
        assert!(col_path.is_dir());
        assert_eq!(fs::read_dir(&col_path).unwrap().count(), 0);
    }

    let file_dir = V8Fixture::new(true);
    fs::create_dir_all(file_dir.prefix.join("weights")).unwrap();
    fs::write(file_dir.prefix.join("weights/single-enc-v8"), b"not-a-dir").unwrap();
    let out = file_dir
        .cmd()
        .args(["--tag", ENGINE_V31, "--source-digest", SOURCE, "--from-dir"])
        .arg(&file_dir.engine)
        .output()
        .unwrap();
    file_dir.assert_refused(out, "non-directory path");
}

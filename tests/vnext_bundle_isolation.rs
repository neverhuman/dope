use dope_kernel::bundle::{BundleLimits, content_fingerprint, verify_bundle};
use dope_kernel::codec;
use dope_kernel::model::{
    ColumnSchema, Dependence, Kernel, KernelProgram, Marginal, Noise, SchemaKind, Target, Task,
    Transform,
};
use serde_json::{Value, json};
use std::fs;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

// A real main, rather than a libtest thread: the production boundary rejects
// workers with pre-existing threads. Compile against this lane's native rlib.
const WORKER: &str = r#"
use dope_kernel::bundle::{BundleLimits, source_free_closure::*};
use dope_kernel::typed_data::isolation;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a[1] == "identity" { println!("{}", decoder_fingerprint().unwrap()); return; }
    let root = std::path::Path::new(&a[2]);
    let outside = std::path::Path::new(&a[5]);
    let inherited: i32 = a[6].parse().unwrap();
    let result = if a[1] == "probe" {
        isolation::confine_bundle_directory(root).unwrap();
        let directory_blocked = std::fs::read(outside).is_err();
        assert!(std::fs::read_dir(root.parent().unwrap()).is_err());
        let path = std::ffi::CString::new(a[5].as_bytes()).unwrap();
        assert_eq!(unsafe { libc::open(path.as_ptr(), libc::O_PATH) }, -1);
        let socket_blocked = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) } == -1;
        let mut byte = [0u8];
        let fd_closed = unsafe { libc::read(inherited, byte.as_mut_ptr().cast(), 1) } == -1;
        let env_cleared = std::env::var_os("DOPE_PRIVATE_SOURCE").is_none();
        let verified = dope_kernel::bundle::verify_bundle(root, &a[3], &BundleLimits::default()).unwrap();
        isolation::seal_sampling_process().unwrap();
        assert!(std::fs::read(root.join("payload.dpk")).is_err());
        assert!(std::fs::read(outside).is_err());
        assert_eq!(unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) }, -1);
        assert!(directory_blocked && socket_blocked && fd_closed && env_cleared);
        verified.sample(8, Some(17), &BundleLimits::default()).map(|v| v.len())
    } else {
        sample_source_free(root, &a[3], &a[4], 8, 17, &BundleLimits::default()).map(|s| {
            let encoded: Vec<u8> = s.values.iter().flat_map(|v| v.to_le_bytes()).collect();
            println!("{}", serde_json::json!({"sample_sha256":dope_kernel::bundle::content_fingerprint(&encoded),
                "charged_bundle_bytes":s.accounting.charged_sampling_bytes,
                "cold_sampling_closure_bytes":s.cold_sampling_closure_bytes,
                "decoder_bytes":s.decoder_files.iter().map(|f| f.bytes).sum::<usize>()}));
            s.values.len()
        })
    };
    match result {
        Ok(n) => println!("rows_values={n}"),
        Err(e) => { println!("{}", e.code()); std::process::exit(2); }
    }
}
"#;

fn suite() -> &'static Path {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/vnext/b01-isolation")
            .join(std::process::id().to_string());
        fs::create_dir_all(&root).unwrap();
        root
    })
}

fn worker() -> &'static Path {
    static WORKER_PATH: OnceLock<PathBuf> = OnceLock::new();
    WORKER_PATH.get_or_init(|| {
        let deps = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let find = |prefix: &str| {
            fs::read_dir(&deps)
                .unwrap()
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name().unwrap().to_str().unwrap().starts_with(prefix)
                        && p.extension().is_some_and(|x| x == "rlib")
                })
                .max_by_key(|p| fs::metadata(p).unwrap().modified().unwrap())
                .unwrap()
        };
        let source = suite().join("worker.rs");
        fs::write(&source, WORKER).unwrap();
        let out = suite().join("worker");
        let status = Command::new("rustc")
            .args(["--edition=2024", "-C", "debuginfo=0"])
            .arg(&source)
            .arg("-o")
            .arg(&out)
            .arg("-L")
            .arg(format!("dependency={}", deps.display()))
            .arg("--extern")
            .arg(format!("dope_kernel={}", find("libdope_kernel-").display()))
            .arg("--extern")
            .arg(format!("serde_json={}", find("libserde_json-").display()))
            .arg("--extern")
            .arg(format!("libc={}", find("liblibc-").display()))
            .status()
            .unwrap();
        assert!(status.success());
        out
    })
}

fn kernel() -> Vec<u8> {
    codec::encode_kernel(&Kernel {
        task: Task::Regression,
        rows_fitted: 8,
        features: 1,
        seed: 9,
        seed_policy: 0,
        quantization_bits: 8,
        compliant: false,
        schema: vec![ColumnSchema {
            kind: SchemaKind::Continuous,
            missing_probability: 0.0,
            impute: 0.5,
            transform: Transform::Identity,
        }],
        marginals: vec![Marginal::QuantileSpline {
            values: vec![0.0, 0.2, 0.5, 0.8, 1.0],
        }],
        program: KernelProgram::Symbolic {
            dependence: Dependence::Independent,
            target: Target::SparseLinear {
                intercept: 0.2,
                terms: vec![],
            },
            noise: Noise::Homoscedastic { sigma: 0.05 },
        },
    })
    .unwrap()
}

fn stage(name: &str) -> (PathBuf, Value) {
    let root = suite().join(name);
    fs::create_dir_all(&root).unwrap();
    let bytes = kernel();
    fs::write(root.join("payload.dpk"), &bytes).unwrap();
    let manifest = json!({"format":"dope.generator-bundle/v1","version":1,
        "bundle_id":"closure","amortization_count":1,"sampling_handler":"native-inline-dpk",
        "transport_framing":"included","capabilities":["byte-inventory","native-dpk-sample"],
        "components":[{"id":"payload","role":"sampling_payload","path":"payload.dpk",
            "sha256":content_fingerprint(&bytes),"bytes":bytes.len(),"kind":"dpk3","required":true,
            "consumed_by_sampler":true,"depends_on":[],"executable":false,"handler":"none"}]});
    (root, manifest)
}

fn manifest(root: &Path, value: &Value) -> String {
    let bytes = serde_json::to_vec(value).unwrap();
    fs::write(root.join("manifest.json"), &bytes).unwrap();
    content_fingerprint(&bytes)
}

fn run(mode: &str, root: &Path, digest: &str, pin: Option<&str>) -> Output {
    let decoder = Command::new(worker()).arg("identity").output().unwrap();
    assert!(decoder.status.success());
    let identity = String::from_utf8(decoder.stdout).unwrap();
    let outside = suite().join("private-source.csv");
    fs::write(&outside, b"private-header,private-value,999999\n").unwrap();
    let fd = fs::File::open(&outside).unwrap();
    // SAFETY: clear CLOEXEC on this owned synthetic-source fixture descriptor.
    assert_eq!(unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, 0) }, 0);
    let mut command = Command::new(worker());
    if root.is_dir() {
        command.current_dir(root);
    }
    command
        .env_clear()
        .env("DOPE_PRIVATE_SOURCE", &outside)
        .arg(mode)
        .arg(root)
        .arg(digest)
        .arg(pin.unwrap_or(identity.trim()))
        .arg(outside)
        .arg(fd.as_raw_fd().to_string())
        .output()
        .unwrap()
}

fn rejected(output: Output, expected: &str) {
    assert_eq!(
        output.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap().trim(), expected);
    assert!(output.stderr.is_empty());
}

#[test]
fn deleted_source_replay_matches_direct_sample_and_closes_all_dependency_access() {
    let (root, value) = stage("positive");
    let digest = manifest(&root, &value);
    let sample = verify_bundle(&root, &digest, &BundleLimits::default())
        .unwrap()
        .sample(8, Some(17), &BundleLimits::default())
        .unwrap();
    let expected = content_fingerprint(
        &sample
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let source = suite().join("deleted-fit-partition");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("rows.csv"), b"private-header,private-value\n").unwrap();
    fs::write(source.join("projection.json"), b"private-transform").unwrap();
    let clean = suite().join("recipient");
    fs::create_dir(&clean).unwrap();
    for name in ["manifest.json", "payload.dpk"] {
        fs::copy(root.join(name), clean.join(name)).unwrap();
    }
    // Only generated fixtures exist here; retain the directory and test evidence.
    fs::remove_file(root.join("payload.dpk")).unwrap();
    fs::remove_file(root.join("manifest.json")).unwrap();
    fs::remove_file(source.join("rows.csv")).unwrap();
    fs::remove_file(source.join("projection.json")).unwrap();
    fs::remove_dir(&source).unwrap();
    assert!(!source.exists());
    let output = run("sample", &clean, &digest, None);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let receipt: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert_eq!(receipt["sample_sha256"], expected);
    let charged = fs::read(clean.join("payload.dpk")).unwrap().len()
        + fs::read(clean.join("manifest.json")).unwrap().len();
    assert_eq!(receipt["charged_bundle_bytes"], charged);
    assert!(receipt["decoder_bytes"].as_u64().unwrap() > 0);
    assert_eq!(
        receipt["cold_sampling_closure_bytes"].as_u64().unwrap(),
        charged as u64 + receipt["decoder_bytes"].as_u64().unwrap()
    );
    assert!(run("probe", &clean, &digest, None).status.success());
    fs::write(
        suite().join("access-proof.json"),
        serde_json::to_vec(&json!({
            "native_sample": receipt, "source_deleted": true, "external_reads_denied": true,
            "sockets_denied": true, "inherited_source_fd_closed": true,
            "environment_cleared": true, "bundle_reads_denied_during_sampling": true
        }))
        .unwrap(),
    )
    .unwrap();
    rejected(
        run("sample", &clean, &digest, Some(&"0".repeat(64))),
        "decoder_identity_mismatch",
    );
}

#[test]
fn external_dependencies_traversal_and_environment_references_fail_before_sampling() {
    for (i, path) in [
        "../projection.json",
        "/private/source.csv",
        "$DOPE_PRIVATE_SOURCE",
        "${DOPE_PRIVATE_SOURCE}/rows.csv",
        "a/../../prior.bin",
    ]
    .iter()
    .enumerate()
    {
        let (root, mut value) = stage(&format!("bad-path-{i}"));
        value["components"][0]["path"] = json!(path);
        let digest = manifest(&root, &value);
        rejected(run("sample", &root, &digest, None), "unsafe_path");
    }
    let (root, mut value) = stage("missing-projection");
    value["components"][0]["depends_on"] = json!(["projection"]);
    let digest = manifest(&root, &value);
    rejected(run("sample", &root, &digest, None), "unbound_dependency");
    for name in [
        "projection.json",
        "private-priors.bin",
        "checkpoint.bin",
        "source.csv",
    ] {
        let (root, value) = stage(&name.replace('.', "-"));
        fs::write(root.join(name), b"private-value").unwrap();
        let digest = manifest(&root, &value);
        rejected(run("sample", &root, &digest, None), "undeclared_component");
    }
    let (root, mut value) = stage("optional-uncharged");
    let mut optional = value["components"][0].clone();
    optional["id"] = json!("optional");
    optional["role"] = json!("materialized_rows");
    optional["path"] = json!("private-rows.bin");
    optional["required"] = json!(false);
    optional["consumed_by_sampler"] = json!(false);
    optional["kind"] = json!("opaque");
    value["components"]
        .as_array_mut()
        .unwrap()
        .insert(0, optional);
    // If component I/O ran first, this would fail as a symlink instead.
    std::os::unix::fs::symlink("/private/rows.csv", root.join("private-rows.bin")).unwrap();
    let digest = manifest(&root, &value);
    rejected(run("sample", &root, &digest, None), "uncharged_dependency");
}

#[test]
fn symlinked_files_directories_and_ancestors_are_refused() {
    let (root, value) = stage("symlink-payload");
    let outside = suite().join("outside.dpk");
    fs::write(&outside, kernel()).unwrap();
    fs::remove_file(root.join("payload.dpk")).unwrap();
    std::os::unix::fs::symlink(&outside, root.join("payload.dpk")).unwrap();
    let digest = manifest(&root, &value);
    rejected(run("sample", &root, &digest, None), "symlink_component");
    let (nested, mut value) = stage("symlink-directory");
    let outside_directory = suite().join("outside-directory");
    fs::create_dir(&outside_directory).unwrap();
    fs::write(outside_directory.join("payload.dpk"), kernel()).unwrap();
    fs::remove_file(nested.join("payload.dpk")).unwrap();
    std::os::unix::fs::symlink(outside_directory, nested.join("weights")).unwrap();
    value["components"][0]["path"] = json!("weights/payload.dpk");
    let digest = manifest(&nested, &value);
    rejected(run("sample", &nested, &digest, None), "symlink_component");
    let (root, value) = stage("ancestor");
    let digest = manifest(&root, &value);
    let alias = suite().join("alias");
    std::os::unix::fs::symlink(&root, &alias).unwrap();
    rejected(
        run("sample", &alias, &digest, None),
        "bundle_isolation_refused",
    );
    rejected(
        run("sample", &alias.join("nested"), &digest, None),
        "bundle_isolation_refused",
    );
    rejected(
        run("sample", &root.join("../ancestor"), &digest, None),
        "bundle_isolation_refused",
    );
}

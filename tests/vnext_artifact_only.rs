use std::fs;
use std::process::{Command, Stdio};

use dope_kernel::compiler::CompileOptions;
use dope_kernel::schema::TableSchemaV1;
use dope_kernel::typed_data::workflow::{compile_raw, sample_raw};
use dope_kernel::typed_data::{InputLimits, TypedTable};
use serde_json::json;

#[test]
fn denied_access_worker() {
    let Some(source) = std::env::var_os("DOPE_DENIAL_SOURCE") else {
        return;
    };
    let fd: i32 = std::env::var("DOPE_DENIAL_FD").unwrap().parse().unwrap();
    dope_kernel::typed_data::isolation::seal_sampling_process().unwrap();
    assert_eq!(
        fs::read(source).unwrap_err().raw_os_error(),
        Some(libc::EPERM)
    );
    let mut buffer = [0u8; 1];
    // SAFETY: valid one-byte destination; inherited descriptor must be closed.
    assert_eq!(unsafe { libc::read(fd, buffer.as_mut_ptr().cast(), 1) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::EBADF)
    );
    assert_eq!(
        std::net::TcpStream::connect("127.0.0.1:9")
            .unwrap_err()
            .raw_os_error(),
        Some(libc::EPERM)
    );
    // Direct exec syscall must fail before attempting to resolve any path.
    let path = c"/bin/true";
    let argv = [path.as_ptr(), std::ptr::null()];
    // SAFETY: both arrays are null-terminated; denied by the installed filter.
    assert_eq!(
        unsafe { libc::execve(path.as_ptr(), argv.as_ptr(), [std::ptr::null()].as_ptr()) },
        -1
    );
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::EPERM)
    );
    println!("source_read=denied inherited_fd=closed network=denied exec=denied");
    std::process::exit(0);
}

#[test]
fn source_path_inherited_descriptor_network_and_exec_are_actually_denied() {
    use std::os::fd::AsRawFd;
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/artifact-only-denials");
    fs::create_dir_all(&root).unwrap();
    let source = root.join("source.csv");
    fs::write(&source, b"synthetic-private-fixture\n").unwrap();
    let file = fs::File::open(&source).unwrap();
    // SAFETY: obtain an intentionally inheritable duplicate for the negative test.
    let inherited = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_DUPFD, 3) };
    assert!(inherited >= 3);
    let output = Command::new(std::env::current_exe().unwrap())
        .env_clear()
        .env("DOPE_DENIAL_SOURCE", &source)
        .env("DOPE_DENIAL_FD", inherited.to_string())
        .args(["--exact", "denied_access_worker", "--nocapture"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    // SAFETY: parent owns this duplicate and closes it exactly once.
    unsafe {
        libc::close(inherited);
    }
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let receipt = String::from_utf8(output.stdout).unwrap();
    assert!(receipt.contains("source_read=denied inherited_fd=closed network=denied exec=denied"));
    fs::write(root.join("receipt.txt"), receipt).unwrap();
}

#[test]
fn fresh_worker_samples_with_empty_environment_and_no_new_access() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/artifact-only");
    fs::create_dir_all(&root).unwrap();
    let schema = TableSchemaV1::from_private_json(
        &serde_json::to_vec(&json!({
            "format":"dope-table-schema","version":1,"columns":[
                {"id":"c0001","source_name":"x","role":"feature","dtype":"continuous"},
                {"id":"c0002","source_name":"y","role":"target","dtype":"continuous"}
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    let table = TypedTable::read_csv(
        "x,y\n0,10\n1,20\n0.5,15\n".as_bytes(),
        schema,
        InputLimits::default(),
    )
    .unwrap();
    let compiled = compile_raw(
        &table,
        &[0, 1, 2],
        &CompileOptions {
            seed: Some(11),
            backend_id: Some("independent_quantile".into()),
            ..CompileOptions::default()
        },
        32768,
    )
    .unwrap();
    let artifact = root.join("artifact.raw");
    fs::write(&artifact, &compiled.artifact).unwrap();
    let expected = sample_raw(&compiled.artifact, 32, Some(99)).unwrap();
    let digest = dope_kernel::bundle::content_fingerprint(
        &expected
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    drop(table);
    for (rows, seed) in [
        (32, 99),
        (32, 100),
        (32, 101),
        (128, 99),
        (128, 100),
        (128, 101),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_dope-kernel"))
            .env_clear()
            .current_dir(&root)
            .args(["sample-raw-isolated", "--artifact"])
            .arg(&artifact)
            .args(["--rows", &rows.to_string(), "--seed", &seed.to_string()])
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let receipt: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(output.stderr.is_empty());
        assert!(receipt.get("private_sample_stream").is_none());
        assert!(receipt.get("values").is_none());
        let samples = sample_raw(&compiled.artifact, rows, Some(seed)).unwrap();
        let expected_bytes = samples
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let expected_digest = dope_kernel::bundle::content_fingerprint(&expected_bytes);
        assert_eq!(receipt["sample_sha256"], expected_digest);
        if rows == 32 && seed == 99 {
            assert_eq!(receipt["sample_sha256"], digest);
        }
        let runtime_bytes: u64 = receipt["runtime_inventory"]
            .as_array()
            .unwrap()
            .iter()
            .map(|file| file["bytes"].as_u64().unwrap())
            .sum();
        assert_eq!(
            receipt["inspection"]["accounting"]["generic_runtime_bytes"],
            runtime_bytes
        );
        assert_eq!(
            receipt["inspection"]["accounting"]["cold_sampling_closure_bytes"],
            runtime_bytes + compiled.artifact.len() as u64
        );
        assert_eq!(receipt["filesystem_network_exec"], "denied");
        fs::write(root.join("receipt.json"), output.stdout).unwrap();

        let private = Command::new(env!("CARGO_BIN_EXE_dope-kernel"))
            .env_clear()
            .current_dir(&root)
            .args(["sample-raw-isolated", "--artifact"])
            .arg(&artifact)
            .args(["--rows", &rows.to_string(), "--seed", &seed.to_string()])
            .arg("--private-samples-f64le")
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert!(private.status.success(), "{:?}", private.status);
        assert_eq!(private.stdout, expected_bytes);
        let receipt: serde_json::Value = serde_json::from_slice(&private.stderr).unwrap();
        assert_eq!(receipt["sample_sha256"], expected_digest);
        assert_eq!(receipt["state"], "sampled");
        assert_eq!(
            receipt["private_sample_stream"],
            json!({
                "format":"f64le", "layout":"row_major", "column_order":"features_then_target",
                "rows":rows, "columns":2, "bytes":rows * 2 * 8
            })
        );
        assert_eq!(receipt["filesystem_network_exec"], "denied");
        assert_eq!(receipt["inherited_source_fds"], "closed");
        assert!(receipt.get("values").is_none());
        let public_text = String::from_utf8(private.stderr).unwrap();
        assert!(!public_text.contains(root.to_str().unwrap()));
        assert!(!public_text.contains("source_name"));
    }
    let malformed = root.join("malformed.raw");
    fs::write(&malformed, b"PRIVATE_MALFORMED_ARTIFACT").unwrap();
    for (input, rows) in [
        (&malformed, 32),
        (&artifact, 0),
        (&artifact, 100_001),
        (&artifact, usize::MAX),
    ] {
        for private_mode in [false, true] {
            let mut command = Command::new(env!("CARGO_BIN_EXE_dope-kernel"));
            command
                .env_clear()
                .args(["sample-raw-isolated", "--artifact"])
                .arg(input)
                .args(["--rows", &rows.to_string()])
                .stdin(Stdio::null());
            if private_mode {
                command.arg("--private-samples-f64le");
            }
            let output = command.output().unwrap();
            assert!(!output.status.success());
            assert!(output.stdout.is_empty());
            let error = String::from_utf8(output.stderr).unwrap();
            assert!(!error.contains("PRIVATE_MALFORMED_ARTIFACT"));
            assert!(!error.contains(root.to_str().unwrap()));
            assert!(!error.contains("\"state\":\"sampled\""));
        }
    }
    let full = fs::OpenOptions::new()
        .write(true)
        .open("/dev/full")
        .unwrap();
    let failed_write = Command::new(env!("CARGO_BIN_EXE_dope-kernel"))
        .env_clear()
        .args(["sample-raw-isolated", "--artifact"])
        .arg(&artifact)
        .args(["--rows", "32", "--private-samples-f64le"])
        .stdin(Stdio::null())
        .stdout(Stdio::from(full))
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(!failed_write.status.success());
    let error = String::from_utf8(failed_write.stderr).unwrap();
    assert!(error.contains("sampling_output_write_failed"));
    assert!(!error.contains("\"state\":\"sampled\""));
    assert!(!error.contains(root.to_str().unwrap()));
}

#[test]
fn raw_carrier_rejects_symlinks_hardlinks_special_files_and_oversized_files() {
    use dope_kernel::typed_data::workflow::read_raw_artifact;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/raw-carrier")
        .join(std::process::id().to_string());
    fs::create_dir_all(&root).unwrap();
    let invalid = root.join("invalid.raw");
    fs::write(&invalid, b"bad").unwrap();
    let alias = root.join("alias.raw");
    std::os::unix::fs::symlink(&invalid, &alias).unwrap();
    assert!(read_raw_artifact(&alias).is_err());
    let link = root.join("hardlink.raw");
    fs::hard_link(&invalid, &link).unwrap();
    assert!(read_raw_artifact(&link).is_err());
    let huge = root.join("huge.raw");
    fs::File::create(&huge)
        .unwrap()
        .set_len(32 * 1024 * 1024 + 1)
        .unwrap();
    assert!(read_raw_artifact(&huge).is_err());
    assert!(read_raw_artifact(&root).is_err());
    assert!(read_raw_artifact(std::path::Path::new("/dev/zero")).is_err());
}

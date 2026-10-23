//! Source text must not escape through parser or child-process diagnostics.
use dope_probes::inputs::{split_table, Inputs, SplitParams};
use dope_probes::landmarks::Tstr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

const PRIVATE: &str = "PRIVATE_SOURCE_CANARY_59612";

struct PrivateDir(PathBuf);
static NEXT_DIR: AtomicU64 = AtomicU64::new(0);
impl PrivateDir {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        Self::at_clock_tick(nonce)
    }
    fn at_clock_tick(nonce: u128) -> Self {
        // Guest clocks can give concurrent tests the same timestamp. Exclusive mkdir
        // plus a process-local sequence also tolerates an existing stale directory.
        for _ in 0..1024 {
            let sequence = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "dope-private-{}-{nonce}-{sequence}",
                std::process::id()
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("cannot create private test directory: {error}"),
            }
        }
        panic!("private test directory allocation exhausted");
    }
}
impl Drop for PrivateDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn temporary_dirs_do_not_collide_at_same_clock_tick() {
    let workers: Vec<_> = (0..32)
        .map(|_| std::thread::spawn(|| PrivateDir::at_clock_tick(0)))
        .collect();
    let directories: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
    let paths: std::collections::BTreeSet<_> = directories.iter().map(|d| &d.0).collect();
    assert_eq!(paths.len(), 32);
    assert!(paths.iter().all(|p| p.is_dir()));
}

#[test]
fn receipt_parser_redacts_source_strings() {
    let dir = PrivateDir::new();
    let path = dir.0.join("receipt.jsonl");
    std::fs::write(&path, format!("{{\"real\":\"{PRIVATE}\"}}\n")).unwrap();
    let error = dope_probes::load_jsonl::<Tstr>(&path)
        .unwrap_err()
        .to_string();
    assert!(error.contains("invalid JSON at column"));
    assert!(!error.contains(PRIVATE));
}

#[test]
fn manifest_parser_and_missing_target_redact_source_strings() {
    let dir = PrivateDir::new();
    let path = dir.0.join("inputs.json");
    std::fs::write(
        &path,
        format!("{{\"pmlb_base\":\"x\",\"split\":{{\"seed\":\"{PRIVATE}\"}}}}"),
    )
    .unwrap();
    let error = Inputs::load(&path).unwrap_err().to_string();
    assert!(error.contains("invalid JSON"));
    assert!(!error.contains(PRIVATE));
    let params = SplitParams {
        seed: 0,
        test_share: 0.2,
        max_test_rows: 10,
        max_train_rows: 10,
        target: PRIVATE.into(),
    };
    let error = split_table(b"feature\ttarget\n1\t2\n", &params, &dir.0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("target column is missing"));
    assert!(!error.contains(PRIVATE));
}

#[cfg(unix)]
#[test]
fn engine_failure_redacts_private_stderr() {
    use std::os::unix::fs::PermissionsExt;
    let dir = PrivateDir::new();
    let executable = dir.0.join("engine");
    std::fs::write(
        &executable,
        format!("#!/bin/sh\nprintf '%s' '{PRIVATE}' >&2\nexit 7\n"),
    )
    .unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let error =
        dope_probes::levels::probe_table(&executable, &dir.0.join("table"), &dir.0.join("out"))
            .unwrap_err()
            .to_string();
    assert!(error.contains("status") && error.contains('7'));
    assert!(!error.contains(PRIVATE));
}

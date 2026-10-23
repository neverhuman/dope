//! Source text must not escape through parser or child-process diagnostics.
use dope_probes::inputs::{split_table, Inputs, SplitParams};
use dope_probes::landmarks::Tstr;
use std::path::PathBuf;

const PRIVATE: &str = "PRIVATE_SOURCE_CANARY_59612";

struct PrivateDir(PathBuf);
impl PrivateDir {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("dope-private-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for PrivateDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
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

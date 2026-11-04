//! Focused review regressions. These execute real owned synthetic children via
//! the same isolated supervisor entrypoint; no native qualification is claimed yet.
use super::{io, tests::Case};
use serde_json::{Value, json};
use std::fs;
use std::os::unix::fs::DirBuilderExt;

fn request(case: &Case, entry: &str) -> Value {
    let mut request = case.request("payload_small", 3000, 256 << 20);
    request["argv"][3] = json!(format!("correction_tests::{entry}"));
    request
}
#[test]
fn renamed_capture_replacement_cannot_diverge_receipt_from_output_document_path() {
    for entry in ["replace_stdout", "replace_stderr"] {
        let case = Case::new();
        let result = case.run(request(&case, entry), "normal");
        assert_eq!(result["error"], "native_file_unavailable");
        assert_eq!(result["owned_children_after"], 0);
        assert!(!result.to_string().contains("PRIVATE_REPLACEMENT"));
        let suffix = if entry == "replace_stdout" {
            "stdout"
        } else {
            "stderr"
        };
        assert_eq!(
            fs::read_to_string(case.0.join(format!("measured.{suffix}"))).unwrap(),
            "PRIVATE_REPLACEMENT"
        );
        assert!(case.0.join(format!("renamed.{suffix}")).is_file());
    }
}
#[test]
fn renamed_whole_root_replacement_refuses_even_when_original_capture_inodes_remain() {
    let case = Case::new();
    let result = case.run(request(&case, "replace_root"), "normal");
    assert_eq!(result["error"], "native_file_unavailable");
    assert_eq!(result["owned_children_after"], 0);
    let moved = case.0.with_extension("moved");
    assert!(moved.join("measured.stdout").is_file());
    assert!(moved.join("measured.stderr").is_file());
    assert_eq!(
        fs::read_to_string(case.0.join("measured.stdout")).unwrap(),
        "PRIVATE_REPLACEMENT"
    );
    assert!(!result.to_string().contains("PRIVATE_REPLACEMENT"));
    fs::remove_dir_all(moved).unwrap();
}
#[test]
fn persistent_cleanup_observer_failure_still_reaps_the_known_real_child() {
    let case = Case::new();
    let result = case.run(
        case.request("payload_sleep", 3000, 256 << 20),
        "persistent_observer_failure",
    );
    assert_eq!(result["error"], "native_process_unavailable");
    assert_eq!(result["owned_children_after"], 0);
    let pid = result["spawned_leader_pid"].as_i64().unwrap();
    assert!(
        pid > 0,
        "fault must occur after a real owned command was spawned"
    );
    assert!(
        !std::path::Path::new(&format!("/proc/{pid}")).exists(),
        "persistent observer errors must not leave the killed child unreaped"
    );
}
#[test]
fn caller_validated_longer_reservation_parses_and_quick_command_finishes() {
    let case = Case::new();
    let request = case.request("payload_small", 120001, 256 << 20);
    assert_eq!(
        io::parse_request(request.to_string().as_bytes())
            .unwrap()
            .wall_ms,
        120001
    );
    let result = case.run(request, "normal");
    assert_eq!(result["receipt"]["status"], "completed");
    assert_eq!(result["receipt"]["returncode"], 0);
    assert!(result["receipt"]["wall_ms"].as_u64().unwrap() < 120001);
    assert_eq!(result["owned_children_after"], 0);
}
fn replace_capture(suffix: &str) {
    fs::rename(format!("measured.{suffix}"), format!("renamed.{suffix}")).unwrap();
    fs::write(format!("measured.{suffix}"), "PRIVATE_REPLACEMENT").unwrap();
}
#[test]
#[ignore = "owned rename/replacement command payload"]
fn replace_stdout() {
    replace_capture("stdout");
}
#[test]
#[ignore = "owned rename/replacement command payload"]
fn replace_stderr() {
    replace_capture("stderr");
}
#[test]
#[ignore = "owned whole-root rename/replacement command payload"]
fn replace_root() {
    let root = std::env::current_dir().unwrap();
    fs::rename(&root, root.with_extension("moved")).unwrap();
    fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
    fs::write(root.join("measured.stdout"), "PRIVATE_REPLACEMENT").unwrap();
    fs::write(root.join("measured.stderr"), "PRIVATE_REPLACEMENT").unwrap();
}

//! Public synthetic process fixtures. Ignored entrypoints run only in isolated child
//! test processes; ordinary tests invoke them explicitly, never share subreaper state.
use super::{io, supervisor, tree};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
static SERIAL: AtomicU64 = AtomicU64::new(0);
pub(super) struct Case(pub(super) std::path::PathBuf);
impl Case {
    pub(super) fn new() -> Self {
        let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/native-process-tests");
        fs::create_dir_all(&parent).unwrap();
        let path = parent.join(format!(
            "{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap();
        Self(path.canonicalize().unwrap())
    }
    pub(super) fn request(&self, payload: &str, wall: u64, ram: u64) -> Value {
        json!({"argv":[std::env::current_exe().unwrap(),"--ignored","--exact",
            format!("tests::{payload}"),"--nocapture"],"root":self.0,
            "label":"measured","wall_ms":wall,"ram_bytes":ram})
    }
    pub(super) fn run(&self, request: Value, mode: &str) -> Value {
        let response = self.0.join("test-response.json");
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "tests::isolated_supervisor_entry",
                "--nocapture",
            ])
            .env("NATIVE_TEST_REQUEST", request.to_string())
            .env("NATIVE_TEST_RESPONSE", &response)
            .env("NATIVE_TEST_MODE", mode)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated supervisor fixture failed"
        );
        serde_json::from_slice(&fs::read(response).unwrap()).unwrap()
    }
}
impl Drop for Case {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn request_shape_and_bounds_refuse_without_source_echo() {
    let case = Case::new();
    let valid = case.request("payload_small", 2000, 256 << 20);
    assert!(io::parse_request(valid.to_string().as_bytes()).is_ok());
    for (key, value) in [
        ("label", json!("../CANARY")),
        ("wall_ms", json!(0)),
        ("ram_bytes", json!(-1)),
        ("argv", json!([])),
        ("root", json!("relative")),
        ("unknown", json!("PRIVATE_CANARY")),
    ] {
        let mut bad = valid.clone();
        bad[key] = value;
        assert_eq!(
            io::parse_request(bad.to_string().as_bytes()).err(),
            Some("native_process_unavailable")
        );
    }
}
#[test]
fn prior_large_child_cannot_contaminate_later_small_command_wait4_peak() {
    let case = Case::new();
    let result = case.run(case.request("payload_small", 3000, 64 << 20), "prior_high");
    assert!(result["prior_children_peak"].as_u64().unwrap() >= 96 << 20);
    assert_eq!(result["receipt"]["status"], "completed");
    assert!(result["receipt"]["peak_ram_bytes"].as_u64().unwrap() < 64 << 20);
    assert_eq!(
        result["receipt"]["rss_measurement"],
        "per_command_polled_owned_tree_and_wait4_child_rusage"
    );
}
#[test]
fn nonzero_and_missing_executable_refuse_and_retain_private_capture_hashes() {
    let case = Case::new();
    let mut request = case.request("payload_small", 2000, 256 << 20);
    request["argv"] = json!(["/bin/sh", "-c", "printf hello; printf private >&2; exit 7"]);
    let result = case.run(request, "normal");
    assert_eq!(result["receipt"]["status"], "failed");
    assert_eq!(result["receipt"]["returncode"], 7);
    for suffix in ["stdout", "stderr"] {
        let path = case.0.join(format!("measured.{suffix}"));
        let bytes = fs::read(&path).unwrap();
        assert_eq!(
            result["receipt"][format!("{suffix}_sha256")],
            format!("{:x}", Sha256::digest(&bytes))
        );
        assert_eq!(fs::metadata(path).unwrap().mode() & 0o777, 0o600);
    }
    assert!(!result["receipt"].to_string().contains("private"));
    let missing = Case::new();
    let mut request = missing.request("payload_small", 2000, 256 << 20);
    request["argv"] = json!(["/PRIVATE_CANARY/missing"]);
    let result = missing.run(request, "normal");
    assert_eq!(result["error"], "native_process_unavailable");
    assert_eq!(result["owned_children_after"], 0);
    assert!(!result.to_string().contains("CANARY"));
}
#[test]
fn denied_allocation_cannot_be_completed_or_claim_under_budget_success() {
    let case = Case::new();
    let result = case.run(case.request("payload_high", 3000, 32 << 20), "normal");
    assert_ne!(result["receipt"]["status"], "completed");
    assert_ne!(result["receipt"]["returncode"], 0);
    assert_eq!(result["owned_children_after"], 0);
}
#[test]
fn timeout_and_leader_exit_cleanup_owned_descendants() {
    for (payload, expected) in [("payload_tree", "timeout"), ("payload_orphan", "failed")] {
        let case = Case::new();
        let result = case.run(case.request(payload, 800, 256 << 20), "normal");
        assert_eq!(result["receipt"]["status"], expected);
        assert_eq!(result["owned_children_after"], 0);
        let pid = fs::read_to_string(case.0.join("descendant.pid")).unwrap();
        assert!(!std::path::Path::new(&format!("/proc/{}", pid.trim())).exists());
    }
}
#[test]
fn actual_post_spawn_monitor_failure_reaps_owned_child_and_never_returns_success() {
    let case = Case::new();
    let result = case.run(
        case.request("payload_sleep", 3000, 256 << 20),
        "monitor_fault",
    );
    assert_eq!(result["error"], "native_process_unavailable");
    assert_eq!(result["owned_children_after"], 0);
}
#[test]
fn child_environment_affinity_nice_limits_and_capture_are_real() {
    let case = Case::new();
    let result = case.run(case.request("payload_limits", 2300, 256 << 20), "normal");
    assert_eq!(result["receipt"]["status"], "completed");
    let text = fs::read_to_string(case.0.join("measured.stdout")).unwrap();
    let raw = text
        .lines()
        .find_map(|line| line.strip_prefix("LIMITS_JSON:"))
        .unwrap();
    let proof: Value = serde_json::from_str(raw).unwrap();
    assert_eq!(proof["cpu_count"], 1);
    assert!(proof["nice"].as_i64().unwrap() >= 10);
    assert_eq!(proof["AS"], 256 << 20);
    assert_eq!(proof["CPU"], 4);
    assert_eq!(proof["FSIZE"], 32 << 20);
    assert_eq!(proof["extra_fds"], 0);
    assert_eq!(proof["stdout_access_mode"], libc::O_WRONLY);
    assert_eq!(proof["stderr_access_mode"], libc::O_WRONLY);
    assert_eq!(proof["cwd"], case.0.to_str().unwrap());
    for key in [
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ] {
        assert_eq!(proof["env"][key], "1");
    }
    assert_eq!(proof["env"]["CUDA_VISIBLE_DEVICES"], "");
    assert_eq!(proof["env"]["PATH"], "/bin:/usr/bin");
    assert_eq!(proof["env"]["LC_ALL"], "C");
}
#[test]
fn exact_32mib_capture_limit_still_blocks_oversized_output() {
    let case = Case::new();
    let result = case.run(
        case.request("payload_large_output", 3000, 256 << 20),
        "normal",
    );
    assert_eq!(result["error"], "native_file_unavailable");
    assert_eq!(result["owned_children_after"], 0);
    assert!(fs::metadata(case.0.join("measured.stdout")).unwrap().len() <= 32 << 20);
}
#[test]
fn existing_8mib_hash_admission_remains_distinct_from_32mib_write_limit() {
    let case = Case::new();
    let result = case.run(
        case.request("payload_hash_limit", 3000, 256 << 20),
        "normal",
    );
    assert_eq!(result["error"], "native_file_unavailable");
    let size = fs::metadata(case.0.join("measured.stdout")).unwrap().len();
    assert!(size > 8 << 20 && size < 32 << 20);
    assert_eq!(result["owned_children_after"], 0);
}
#[test]
#[ignore = "isolated fixture entry, explicitly executed by supervisor tests"]
fn isolated_supervisor_entry() {
    let request =
        io::parse_request(std::env::var("NATIVE_TEST_REQUEST").unwrap().as_bytes()).unwrap();
    let mode = std::env::var("NATIVE_TEST_MODE").unwrap();
    let mut prior = 0;
    if mode == "prior_high" {
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "tests::payload_high", "--nocapture"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(status.success());
        // SAFETY: initialized output struct and valid pointer for this test process.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: this is a test-only observation of intentionally completed prior children.
        assert_eq!(
            unsafe { libc::getrusage(libc::RUSAGE_CHILDREN, &mut usage) },
            0
        );
        prior = usage.ru_maxrss as u64 * 1024;
    }
    let result = if mode == "monitor_fault" {
        supervisor::invoke_with_monitor_fault(request)
    } else if mode == "persistent_observer_failure" {
        supervisor::invoke_with_persistent_observer_failure(request)
    } else {
        supervisor::invoke(request)
    };
    let children = tree::snapshot(std::process::id() as libc::pid_t)
        .unwrap()
        .0
        .len();
    let value = match result {
        Ok(receipt) => {
            json!({"receipt":receipt,"prior_children_peak":prior,"owned_children_after":children})
        }
        Err(error) => {
            json!({"error":error,"owned_children_after":children,"spawned_leader_pid":supervisor::last_spawned_pid()})
        }
    };
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(std::env::var("NATIVE_TEST_RESPONSE").unwrap())
        .unwrap();
    std::io::Write::write_all(&mut file, value.to_string().as_bytes()).unwrap();
    file.sync_all().unwrap();
}
#[test]
#[ignore = "owned synthetic command payload"]
fn payload_small() {
    println!("bounded-native-process-fixture");
}
#[test]
#[ignore = "owned synthetic command payload"]
fn payload_high() {
    let mut bytes = vec![0u8; 128 << 20];
    for byte in &mut bytes {
        *byte = 7;
    }
    std::hint::black_box(&bytes);
    std::thread::sleep(Duration::from_millis(20));
}
#[test]
#[ignore = "owned synthetic command payload"]
fn payload_sleep() {
    std::thread::sleep(Duration::from_secs(10));
}
fn descendant() -> std::process::Child {
    let child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "tests::payload_sleep",
            "--nocapture",
        ])
        .spawn()
        .unwrap();
    fs::write("descendant.pid", child.id().to_string()).unwrap();
    child
}
#[test]
#[ignore = "owned synthetic command payload"]
fn payload_tree() {
    let mut child = descendant();
    let _ = child.wait();
}
#[test]
#[ignore = "owned synthetic command payload"]
fn payload_orphan() {
    let _child = descendant();
    std::process::exit(0);
}
#[test]
#[ignore = "owned synthetic command payload"]
fn payload_large_output() {
    let data = vec![b'x'; 33 << 20];
    std::io::Write::write_all(&mut std::io::stdout(), &data).unwrap();
}
#[test]
#[ignore = "owned synthetic command payload"]
fn payload_limits() {
    // SAFETY: initialized CPU mask/output limits and valid pointers for self-observation.
    unsafe {
        let mut mask: libc::cpu_set_t = std::mem::zeroed();
        assert_eq!(
            libc::sched_getaffinity(0, std::mem::size_of_val(&mask), &mut mask),
            0
        );
        let count = (0..libc::CPU_SETSIZE as usize)
            .filter(|i| libc::CPU_ISSET(*i, &mask))
            .count();
        let mut limits = serde_json::Map::new();
        for (name, kind) in [
            ("AS", libc::RLIMIT_AS),
            ("CPU", libc::RLIMIT_CPU),
            ("FSIZE", libc::RLIMIT_FSIZE),
        ] {
            let mut limit: libc::rlimit = std::mem::zeroed();
            assert_eq!(libc::getrlimit(kind, &mut limit), 0);
            limits.insert(name.to_owned(), json!(limit.rlim_cur));
        }
        let fds = fs::read_dir("/proc/self/fd")
            .unwrap()
            .count()
            .saturating_sub(4);
        limits.insert("extra_fds".to_owned(), json!(fds));
        for (name, fd) in [
            ("stdout_access_mode", libc::STDOUT_FILENO),
            ("stderr_access_mode", libc::STDERR_FILENO),
        ] {
            let flags = libc::fcntl(fd, libc::F_GETFL);
            assert!(flags >= 0);
            limits.insert(name.to_owned(), json!(flags & libc::O_ACCMODE));
        }
        limits.insert("cpu_count".to_owned(), json!(count));
        limits.insert(
            "nice".to_owned(),
            json!(libc::getpriority(libc::PRIO_PROCESS, 0)),
        );
        limits.insert("cwd".to_owned(), json!(std::env::current_dir().unwrap()));
        limits.insert(
            "env".to_owned(),
            json!(std::env::vars().collect::<std::collections::BTreeMap<_, _>>()),
        );
        println!("\nLIMITS_JSON:{}", Value::Object(limits));
    }
}

#[test]
#[ignore = "owned synthetic command payload"]
fn payload_hash_limit() {
    let data = vec![b'x'; 9 << 20];
    std::io::Write::write_all(&mut std::io::stdout(), &data).unwrap();
}

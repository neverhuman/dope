use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt, symlink};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dope_kernel::compiler::CompileOptions;
use dope_kernel::contract::ReleasePolicy;
use dope_kernel::evaluation::{
    EvaluationCapability, EvaluationDeclaration, EvaluationEvent, FrozenSelection, SelectionView,
    evaluate_frozen_batch, frozen_batch_hash, serve_frozen_batch,
};
use dope_kernel::evidence::{canonical_json, parse_json};
use dope_kernel::model::Task;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn fit_csv() -> String {
    (0..32)
        .map(|i| {
            let x = i as f64 / 31.0;
            format!("{x},{},{},{}\n", x * x, 1.0 - x, 0.2 + 0.6 * x)
        })
        .collect()
}
fn query_csv(permuted: bool) -> String {
    (0..32)
        .map(|i| {
            let x = (i as f64 + 0.25) / 32.0;
            let y = if permuted { 1.0 - x } else { x };
            format!("{x},{},{},{}\n", x * x, 1.0 - x, 0.2 + 0.6 * y)
        })
        .collect()
}
fn protocol(frozen: bool) -> Value {
    let mut p = parse_json(include_bytes!("../production/behavior-contract-v1.json")).unwrap();
    if frozen {
        p["status"] = "frozen".into();
        p["protocol_frozen"] = true.into();
        p["confirmation_allowed"] = true.into();
        for v in p["freeze_requirements"]
            .as_object_mut()
            .unwrap()
            .values_mut()
        {
            *v = json!("f".repeat(64));
        }
        for v in p["margins"].as_object_mut().unwrap().values_mut() {
            *v = json!(0.1);
        }
        for v in p["budgets"].as_object_mut().unwrap().values_mut() {
            *v = json!(1);
        }
        for k in [
            "table_schema_hash",
            "task_manifest_hash",
            "population_history_hash",
        ] {
            p["workload"][k] = json!("f".repeat(64));
        }
        p["learners"]["held_out_families"] = json!(["held_out"]);
        p["disclosure"]["protected_unit"] = "row".into();
        p["disclosure"]["auxiliary_information_hash"] = json!("f".repeat(64));
        p["disclosure"]["repeated_release_policy_hash"] = json!("f".repeat(64));
        p["disclosure"]["privacy_route"] = "scoped".into();
        p["disclosure"]["legal_route"] = "scoped".into();
        p["disclosure"]["purpose_grants"] = json!(["customer_modeling"]);
    }
    p
}
struct Fixture {
    root: PathBuf,
    frozen: FrozenSelection,
}
impl Fixture {
    fn new(permuted: bool, frozen_protocol: bool, budget: u32) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "target/custody/{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("train.csv"), fit_csv()).unwrap();
        let options = CompileOptions {
            seed: Some(1729),
            beam_width: Some(2),
            quantization_profiles: vec![8],
            ..Default::default()
        };
        let view = SelectionView::open(
            &root.join("train.csv"),
            Task::Regression,
            &options.release_policy,
        )
        .unwrap();
        let compiled = view.select(&options).unwrap();
        let frozen = view.freeze(&compiled, &options, 1729, 0, 1).unwrap();
        fs::write(
            root.join("frozen-selection.json"),
            canonical_json(&serde_json::to_value(&frozen).unwrap()).unwrap(),
        )
        .unwrap();
        let protocol = canonical_json(&protocol(frozen_protocol)).unwrap();
        // All data here is synthetic. Setup is consumed by the privileged fixture
        // process, which creates owner-only files before any evaluation request.
        let declaration = EvaluationDeclaration {
            confirmation_id: hash(root.to_string_lossy().as_bytes()),
            batch_hash: frozen_batch_hash(std::slice::from_ref(&frozen)).unwrap(),
            protocol_hash: hash(&protocol),
            fit_hash: hash(fit_csv().as_bytes()),
            query_hash: hash(query_csv(permuted).as_bytes()),
            producer_uid: unsafe { libc::geteuid() },
            max_attempts: 2,
            query_open_budget: budget,
        };
        fs::write(root.join("setup.json"),serde_json::to_vec(&json!({"declaration":declaration,"protocol":protocol,"fit":fit_csv(),"query":query_csv(permuted)})).unwrap()).unwrap();
        Self { root, frozen }
    }
    fn start(&self, index: usize, crash: bool) -> (Child, PathBuf) {
        let socket = self.root.join(format!("s{index}.sock"));
        let child = Command::new("sudo")
            .args([
                "-n",
                "--preserve-env=DOPE_CUSTODY_CASE,DOPE_CUSTODY_SOCKET,DOPE_CUSTODY_CRASH",
                "--",
            ])
            .arg(std::env::current_exe().unwrap())
            .args(["--exact", "custody_owner", "--ignored", "--nocapture"])
            .env("DOPE_CUSTODY_CASE", &self.root)
            .env("DOPE_CUSTODY_SOCKET", &socket)
            .env("DOPE_CUSTODY_CRASH", if crash { "1" } else { "0" })
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        while !socket.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(socket.exists(), "owner fixture endpoint did not start");
        (child, socket)
    }
    fn finish(&self, child: Child, index: usize) -> Value {
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "owner failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&fs::read(self.root.join(format!("s{index}.summary"))).unwrap())
            .unwrap()
    }
}

/// Fixture subprocess entry, launched under UID 0 by the non-root test producer.
#[test]
#[ignore]
fn custody_owner() {
    let root = PathBuf::from(std::env::var_os("DOPE_CUSTODY_CASE").unwrap());
    assert!(
        root.canonicalize()
            .unwrap()
            .starts_with(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/custody"))
    );
    let socket = PathBuf::from(std::env::var_os("DOPE_CUSTODY_SOCKET").unwrap());
    assert_eq!(socket.parent(), Some(root.as_path()));
    assert_eq!(unsafe { libc::geteuid() }, 0);
    let private = root.join("private");
    let setup: Value = serde_json::from_slice(&fs::read(root.join("setup.json")).unwrap()).unwrap();
    if !private.exists() {
        fs::create_dir(&private).unwrap();
        fs::set_permissions(&private, fs::Permissions::from_mode(0o700)).unwrap();
        for (name, bytes) in [
            (
                "declaration.json",
                serde_json::to_vec(&setup["declaration"]).unwrap(),
            ),
            (
                "protocol.json",
                serde_json::from_value::<Vec<u8>>(setup["protocol"].clone()).unwrap(),
            ),
            (
                "train.csv",
                setup["fit"].as_str().unwrap().as_bytes().to_vec(),
            ),
            (
                "test.csv",
                setup["query"].as_str().unwrap().as_bytes().to_vec(),
            ),
        ] {
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(private.join(name))
                .unwrap();
            f.write_all(&bytes).unwrap();
            f.sync_all().unwrap();
        }
        for name in ["renamed.csv", "labels.tar", "array.json"] {
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o400)
                .open(root.join(name))
                .unwrap();
            f.write_all(setup["query"].as_str().unwrap().as_bytes())
                .unwrap();
        }
    }
    let listener = UnixListener::bind(&socket).unwrap();
    fs::set_permissions(&socket, fs::Permissions::from_mode(0o666)).unwrap();
    let ledger = private.join(format!(
        "{}.jsonl",
        setup["declaration"]["confirmation_id"].as_str().unwrap()
    ));
    if std::env::var("DOPE_CUSTODY_CRASH").unwrap() == "1" {
        let watched = ledger.clone();
        std::thread::spawn(move || {
            loop {
                if fs::read_to_string(&watched).is_ok_and(|s| s.contains("query_opened")) {
                    // SAFETY: this fixture intentionally simulates process loss after access.
                    unsafe { libc::_exit(74) }
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        });
    }
    let root_selection_denied = SelectionView::open(
        &private.join("test.csv"),
        Task::Regression,
        &ReleasePolicy::default(),
    )
    .is_err();
    let result = serve_frozen_batch(&listener, &private);
    let events = fs::read_to_string(ledger)
        .unwrap_or_default()
        .lines()
        .map(|line| serde_json::from_str::<EvaluationEvent>(line).unwrap())
        .collect::<Vec<_>>();
    let network_denied = std::net::TcpStream::connect("127.0.0.1:9")
        .is_err_and(|error| error.raw_os_error() == Some(libc::EPERM));
    let exec_denied = Command::new("/bin/true")
        .status()
        .is_err_and(|error| error.raw_os_error() == Some(libc::EPERM));
    let mut summary = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o644)
        .custom_flags(libc::O_NOFOLLOW)
        .open(socket.with_extension("summary"))
        .unwrap();
    summary.write_all(&serde_json::to_vec(&json!({"served":result.is_ok(),"root_selection_denied":root_selection_denied,"error":result.err().map(|e|e.to_string()),"events":events,"network_denied":network_denied,"exec_denied":exec_denied})).unwrap()).unwrap();
}

#[test]
fn final_label_permutation_cannot_change_development_selection() {
    let a = Fixture::new(false, true, 2);
    let b = Fixture::new(true, true, 2);
    assert_eq!(a.frozen.artifact(), b.frozen.artifact());
    assert_eq!(a.frozen.candidate_id(), b.frozen.candidate_id());
    for fixture in [&a, &b] {
        let (child, socket) = fixture.start(0, false);
        let outcome = evaluate_frozen_batch(
            EvaluationCapability::connect(&socket, 0).unwrap(),
            std::slice::from_ref(&fixture.frozen),
        )
        .unwrap();
        assert_eq!(outcome.sealed_test_open_count, 1);
        assert_eq!(outcome.query_opens_charged, 1);
        assert!(
            outcome
                .events
                .iter()
                .position(|e| e.kind == "query_access")
                .unwrap()
                < outcome
                    .events
                    .iter()
                    .position(|e| e.kind == "query_opened")
                    .unwrap()
        );
        assert!(!outcome.reports[0].certified);
        assert!(
            outcome.reports[0]
                .failed_gates
                .contains(&"auditor_family_coverage".into())
        );
        let summary = fixture.finish(child, 0);
        assert_eq!(summary["root_selection_denied"], true);
        assert_eq!(summary["network_denied"], true);
        assert_eq!(summary["exec_denied"], true);
    }
}

#[test]
fn protected_aliases_copies_archives_arrays_and_callbacks_are_denied() {
    let fixture = Fixture::new(false, true, 2);
    let (child, socket) = fixture.start(0, false);
    symlink(
        fixture.root.join("private/test.csv"),
        fixture.root.join("alias.csv"),
    )
    .unwrap();
    for name in [
        "private/test.csv",
        "alias.csv",
        "renamed.csv",
        "labels.tar",
        "array.json",
    ] {
        assert!(fs::read(fixture.root.join(name)).is_err());
        let result = SelectionView::open(
            &fixture.root.join(name),
            Task::Regression,
            &ReleasePolicy::default(),
        );
        assert!(result.is_err());
        assert!(
            !result
                .err()
                .unwrap()
                .to_string()
                .contains(&query_csv(false))
        );
    }
    let mut request = serde_json::to_value(vec![fixture.frozen.clone()]).unwrap();
    request[0]["optimizer_callback"] = json!("must_not_run");
    let mut stream = UnixStream::connect(socket).unwrap();
    stream
        .write_all(&serde_json::to_vec(&request).unwrap())
        .unwrap();
    stream.shutdown(std::net::Shutdown::Write).unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    assert!(response.is_empty());
    let summary = fixture.finish(child, 0);
    assert_eq!(summary["served"], false);
    assert_eq!(summary["events"], json!([]));
}

#[test]
fn completed_failed_audit_cannot_retry_or_try_another_candidate() {
    let fixture = Fixture::new(false, true, 2);
    let (child, socket) = fixture.start(0, false);
    let outcome = evaluate_frozen_batch(
        EvaluationCapability::connect(&socket, 0).unwrap(),
        std::slice::from_ref(&fixture.frozen),
    )
    .unwrap();
    assert!(!outcome.reports[0].certified);
    fixture.finish(child, 0);
    let (child, socket) = fixture.start(1, false);
    assert!(
        EvaluationCapability::connect(&socket, 0)
            .and_then(|capability| evaluate_frozen_batch(
                capability,
                std::slice::from_ref(&fixture.frozen)
            ))
            .is_err()
    );
    let summary = fixture.finish(child, 1);
    assert_eq!(
        summary["events"],
        serde_json::to_value(&outcome.events).unwrap()
    );
    let (child, socket) = fixture.start(2, false);
    let mut changed = serde_json::to_value(vec![fixture.frozen.clone()]).unwrap();
    changed[0]["seed"] = json!(13);
    let mut stream = UnixStream::connect(socket).unwrap();
    stream
        .write_all(&serde_json::to_vec(&changed).unwrap())
        .unwrap();
    stream.shutdown(std::net::Shutdown::Write).unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    assert!(response.is_empty());
    let summary = fixture.finish(child, 2);
    assert_eq!(
        summary["events"],
        serde_json::to_value(&outcome.events).unwrap()
    );
}

#[test]
fn interrupted_identical_batch_retains_access_and_budget() {
    let fixture = Fixture::new(false, true, 2);
    let (child, socket) = fixture.start(0, true);
    assert!(
        evaluate_frozen_batch(
            EvaluationCapability::connect(&socket, 0).unwrap(),
            std::slice::from_ref(&fixture.frozen)
        )
        .is_err()
    );
    assert_eq!(child.wait_with_output().unwrap().status.code(), Some(74));
    let (child, socket) = fixture.start(1, false);
    let outcome = evaluate_frozen_batch(
        EvaluationCapability::connect(&socket, 0).unwrap(),
        std::slice::from_ref(&fixture.frozen),
    )
    .unwrap();
    assert_eq!(outcome.query_opens_charged, 2);
    assert_eq!(outcome.sealed_test_open_count, 2);
    assert!(
        outcome
            .events
            .iter()
            .any(|e| e.attempt == 1 && e.kind == "query_access")
    );
    assert!(
        outcome
            .events
            .iter()
            .any(|e| e.attempt == 2 && e.kind == "completed")
    );
    fixture.finish(child, 1);
    let bounded = Fixture::new(false, true, 1);
    let (child, socket) = bounded.start(0, true);
    assert!(
        evaluate_frozen_batch(
            EvaluationCapability::connect(&socket, 0).unwrap(),
            std::slice::from_ref(&bounded.frozen)
        )
        .is_err()
    );
    assert_eq!(child.wait_with_output().unwrap().status.code(), Some(74));
    let (child, socket) = bounded.start(1, false);
    assert!(
        evaluate_frozen_batch(
            EvaluationCapability::connect(&socket, 0).unwrap(),
            std::slice::from_ref(&bounded.frozen)
        )
        .is_err()
    );
    let summary = bounded.finish(child, 1);
    assert_eq!(
        summary["events"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| e["kind"] == "query_access")
            .count(),
        1
    );
}

#[test]
fn unresolved_protocol_and_same_principal_cannot_mint_capability() {
    let fixture = Fixture::new(false, false, 2);
    let (child, socket) = fixture.start(0, false);
    assert!(
        EvaluationCapability::connect(&socket, 0)
            .and_then(|capability| evaluate_frozen_batch(
                capability,
                std::slice::from_ref(&fixture.frozen)
            ))
            .is_err()
    );
    assert_eq!(fixture.finish(child, 0)["events"], json!([]));
    let reused = Fixture::new(false, true, 2);
    let mut setup: Value =
        serde_json::from_slice(&fs::read(reused.root.join("setup.json")).unwrap()).unwrap();
    setup["query"] = setup["fit"].clone();
    setup["declaration"]["query_hash"] = setup["declaration"]["fit_hash"].clone();
    fs::write(
        reused.root.join("setup.json"),
        serde_json::to_vec(&setup).unwrap(),
    )
    .unwrap();
    let (child, socket) = reused.start(0, false);
    assert!(
        EvaluationCapability::connect(&socket, 0)
            .and_then(|capability| evaluate_frozen_batch(
                capability,
                std::slice::from_ref(&reused.frozen)
            ))
            .is_err()
    );
    assert_eq!(reused.finish(child, 0)["events"], json!([]));
    let corrupt = Fixture::new(false, true, 2);
    let mut setup: Value =
        serde_json::from_slice(&fs::read(corrupt.root.join("setup.json")).unwrap()).unwrap();
    setup["declaration"]["query_hash"] = json!("e".repeat(64));
    fs::write(
        corrupt.root.join("setup.json"),
        serde_json::to_vec(&setup).unwrap(),
    )
    .unwrap();
    let (child, socket) = corrupt.start(0, false);
    assert!(
        EvaluationCapability::connect(&socket, 0)
            .and_then(|capability| evaluate_frozen_batch(
                capability,
                std::slice::from_ref(&corrupt.frozen)
            ))
            .is_err()
    );
    let summary = corrupt.finish(child, 0);
    assert_eq!(
        summary["events"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| e["kind"] == "query_opened")
            .count(),
        1
    );
    assert_eq!(
        summary["events"].as_array().unwrap().last().unwrap()["kind"],
        "rejected"
    );
    let own = fixture.root.join("own.sock");
    let _listener = UnixListener::bind(&own).unwrap();
    assert!(EvaluationCapability::connect(&own, unsafe { libc::geteuid() }).is_err());
    let alias = fixture.root.join("alias.sock");
    symlink(own, &alias).unwrap();
    assert!(EvaluationCapability::connect(&alias, 0).is_err());
    let value = json!({"labels":[0.1,0.2],"artifact":fixture.frozen.artifact()});
    assert!(serde_json::from_value::<FrozenSelection>(value).is_err());
}

#[test]
fn predeclared_batch_is_evaluated_in_full_without_filtering_failed_cells() {
    let fixture = Fixture::new(false, true, 2);
    let batch = vec![fixture.frozen.clone(), fixture.frozen.clone()];
    let mut setup: Value =
        serde_json::from_slice(&fs::read(fixture.root.join("setup.json")).unwrap()).unwrap();
    setup["declaration"]["batch_hash"] = json!(frozen_batch_hash(&batch).unwrap());
    fs::write(
        fixture.root.join("setup.json"),
        serde_json::to_vec(&setup).unwrap(),
    )
    .unwrap();
    let (child, socket) = fixture.start(0, false);
    let outcome =
        evaluate_frozen_batch(EvaluationCapability::connect(&socket, 0).unwrap(), &batch).unwrap();
    assert_eq!(outcome.reports.len(), 2);
    assert!(outcome.reports.iter().all(|report| !report.certified));
    assert_eq!(outcome.sealed_test_open_count, 2);
    assert_eq!(outcome.query_opens_charged, 2);
    fixture.finish(child, 0);
}

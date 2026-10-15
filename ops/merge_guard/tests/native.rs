//! Language and OS boundary cases beyond the original Python scenarios.
use super::super::*;
use github::{Api, Transport};
use serde_json::json;
use std::os::unix::fs::PermissionsExt;
use std::time::{Duration, Instant};

fn scratch(label: &str) -> std::path::PathBuf {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target/vnext/20261014/guard-native-port/process-fixtures");
    let dir = root.join(format!(
        "{label}-{}-{}",
        std::process::id(),
        timestamp::now().unwrap().replace(':', "_")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
fn executable(dir: &std::path::Path, source: &str) -> std::path::PathBuf {
    let path = dir.join("fake gh with spaces");
    std::fs::write(&path, source).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
    path
}
// The stderr fixtures read their body through an immutable interpreter rather
// than immediately executing the newly written file. The real bounded process
// transport still captures both streams; no shell command string is built.
struct ScriptTransport;
impl Transport for ScriptTransport {
    fn run(&mut self, argv: &[String]) -> Result<github::Response> {
        let mut interpreted = vec!["/bin/sh".into()];
        interpreted.extend_from_slice(argv);
        github::ProcessTransport::default().run(&interpreted)
    }
}
#[test]
fn version_one_canonical_digests_match_python_unicode_and_numbers() {
    let rows: Value = serde_json::from_str(include_str!("../fixtures/canonical-v1.json")).unwrap();
    for (i, row) in array(&rows).unwrap().iter().enumerate() {
        let input: Value = serde_json::from_str(text(&row["input_json"]).unwrap()).unwrap();
        assert_eq!(
            json::canonical(&input).unwrap(),
            row["canonical"].as_str().unwrap(),
            "case{i}"
        );
        assert_eq!(
            json::digest(&input).unwrap(),
            row["sha256"].as_str().unwrap(),
            "case{i}"
        );
    }
}
#[test]
fn actual_transport_preserves_literal_argv_without_a_shell_command() {
    let dir = scratch("argv");
    // Select an immutable executable through a path containing spaces. The
    // fixture does not need to publish and immediately execute a writable shim.
    let fake = dir.join("fake gh with spaces");
    std::os::unix::fs::symlink("/usr/bin/printf", &fake).unwrap();
    let endpoint = "repos/fixture/pulls?value=literal; quoted ' é";
    let argv = vec![
        fake.to_string_lossy().into_owned(),
        "%s\n".into(),
        "api".into(),
        "--method".into(),
        "GET".into(),
        endpoint.into(),
        "--paginate".into(),
        "--slurp".into(),
    ];
    let response = github::ProcessTransport::default().run(&argv).unwrap();
    assert_eq!(response.code, 0);
    assert_eq!(
        response.stdout.lines().collect::<Vec<_>>(),
        ["api", "--method", "GET", endpoint, "--paginate", "--slurp"]
    );
}
#[test]
fn request_timeout_terminates_only_its_owned_child_group() {
    let dir = scratch("timeout");
    let fake = executable(&dir, "#!/bin/sh\nexec sleep 2\n");
    let mut unrelated = std::process::Command::new("sleep")
        .arg("2")
        .spawn()
        .unwrap();
    let mut transport = github::ProcessTransport {
        limit: Duration::from_millis(40),
    };
    let start = Instant::now();
    let result = transport.run(&[
        fake.to_string_lossy().into_owned(),
        "api".into(),
        "--method".into(),
        "GET".into(),
        "user".into(),
    ]);
    assert_eq!(result.err(), Some(Refused("api_unavailable")));
    assert!(start.elapsed() < Duration::from_secs(1));
    assert!(unrelated.try_wait().unwrap().is_none());
    unrelated.kill().unwrap();
    unrelated.wait().unwrap();
    assert_eq!(
        github::ProcessTransport::default().limit,
        Duration::from_secs(45)
    );
}
#[test]
fn actual_malformed_response_never_qualifies_or_echoes_payload() {
    let dir = scratch("malformed");
    let fake = executable(&dir, "#!/bin/sh\nprintf '%s' 'malformed payload marker'\n");
    let mut api = github::GitHub::new(
        fake.to_string_lossy().into_owned(),
        github::ProcessTransport::default(),
    );
    assert_eq!(
        api.get("user", false, false),
        Err(Refused("api_unavailable"))
    );
    assert_eq!(api.calls()[0]["method"], "GET");
    assert!(!api.calls().to_string().contains("payload marker"));
}
#[test]
fn bounded_stderr_is_retained_internally_and_omitted_from_github_receipts() {
    let dir = scratch("stderr");
    let fake = executable(
        &dir,
        "#!/bin/sh\nprintf '%s' '{\"fixture\":true}'\nprintf 'private diagnostic marker\\377' >&2\n",
    );
    let response = ScriptTransport
        .run(&[fake.to_string_lossy().into_owned()])
        .unwrap();
    assert_eq!(response.code, 0);
    assert_eq!(response.stderr, "private diagnostic marker\u{fffd}");
    let mut api = github::GitHub::new(fake.to_string_lossy().into_owned(), ScriptTransport);
    assert_eq!(api.get("user", false, false), Ok(json!({"fixture": true})));
    assert!(!api.calls().to_string().contains("diagnostic marker"));
    let fake = executable(
        &dir,
        "#!/bin/sh\nprintf '%s' '{\"status\":500}'\nprintf '%s' 'cargo metadata diagnostic marker' >&2\nexit 101\n",
    );
    let response = ScriptTransport
        .run(&[fake.to_string_lossy().into_owned()])
        .unwrap();
    assert_eq!(response.code, 101);
    assert_eq!(response.stderr, "cargo metadata diagnostic marker");
    let mut api = github::GitHub::new(fake.to_string_lossy().into_owned(), ScriptTransport);
    assert_eq!(
        api.get("user", false, false),
        Err(Refused("api_unavailable"))
    );
    assert!(!api.calls().to_string().contains("diagnostic marker"));
}
#[test]
fn stderr_exceeding_the_existing_capture_bound_is_refused() {
    let dir = scratch("stderr-bound");
    let fake = executable(
        &dir,
        "#!/bin/sh\nprintf '%s' '{}'\nhead -c 65537 /dev/zero >&2\n",
    );
    assert_eq!(
        ScriptTransport
            .run(&[fake.to_string_lossy().into_owned()])
            .err(),
        Some(Refused("api_unavailable"))
    );
}
#[test]
fn selected_gh_runs_when_rtk_is_absent_from_path() {
    const CHILD: &str = "DOPE_GUARD_RTK_ABSENT_CHILD";
    if let Some(fake) = std::env::var_os(CHILD) {
        let path = std::env::var_os("PATH").unwrap();
        assert!(std::env::split_paths(&path).all(|p| !p.join("rtk").exists()));
        assert!(std::process::Command::new("rtk").output().is_err());
        let mut api = github::GitHub::new(
            fake.to_str().unwrap().into(),
            github::ProcessTransport::default(),
        );
        assert_eq!(
            api.get("repos/fixture/pulls?value=literal; quoted ' é", false, true),
            Ok(json!({"fixture": true}))
        );
        assert_eq!(api.calls().as_array().unwrap().len(), 1);
        assert_eq!(api.calls()[0]["method"], "GET");
        return;
    }
    let dir = scratch("rtk-absent");
    let fake = executable(
        &dir,
        r#"#!/bin/sh
[ "$#" -eq 6 ] || exit 90
[ "$1" = api ] && [ "$2" = --method ] && [ "$3" = GET ] || exit 91
[ "$4" = "repos/fixture/pulls?value=literal; quoted ' é" ] || exit 92
[ "$5" = --paginate ] && [ "$6" = --slurp ] || exit 93
printf '%s' '{"fixture":true}'
"#,
    );
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "guard::tests::native::selected_gh_runs_when_rtk_is_absent_from_path",
        ])
        .env("PATH", &dir)
        .env(CHILD, fake)
        .output()
        .unwrap();
    assert!(child.status.success(), "{child:?}");
    assert!(
        String::from_utf8(child.stdout)
            .unwrap()
            .contains("1 passed")
    );
}
#[test]
fn malformed_receipt_file_is_a_safe_json_refusal() {
    use clap::Parser;
    let dir = scratch("receipt");
    let path = dir.join("malformed.json");
    std::fs::write(&path, "private malformed payload marker").unwrap();
    let args = crate::Cli::try_parse_from([
        "guard",
        "--base",
        "b",
        "--pr",
        "1",
        "--source",
        "a",
        "--review-receipt",
        path.to_str().unwrap(),
    ])
    .unwrap();
    let result = crate::execute(args);
    assert_eq!(result["reason"], "receipt_unavailable");
    assert!(!result.to_string().contains("payload marker"));
}
fn baseline() -> (Value, Value, Value) {
    let cases: Value = serde_json::from_str(include_str!(
        "../fixtures/test_qualified_classic_and_ruleset_controls.json"
    ))
    .unwrap();
    (
        cases[0]["args"][0].clone(),
        cases[0]["args"][1].clone(),
        cases[0]["args"][2].clone(),
    )
}
#[test]
fn missing_or_mistyped_remote_fields_fail_closed_without_panics() {
    let (original, pins, receipt) = baseline();
    for replacement in [Value::Null, json!([]), json!("bad"), json!(false), json!(1)] {
        for key in [
            "before",
            "source_commit",
            "reviews",
            "runs",
            "commits",
            "workflow_definitions",
        ] {
            let mut evidence = original.clone();
            evidence[key] = replacement.clone();
            assert!(
                evidence::evaluate_preflight(&evidence, &pins, &receipt).is_err(),
                "{key}: {replacement}"
            );
        }
    }
    for app in [json!(true), json!(15368.0), json!("15368"), Value::Null] {
        let mut evidence = original.clone();
        evidence["before"]["controls"]["protection"]["required_status_checks"]["checks"][0]["app_id"] =
            app;
        evidence["after"] = evidence["before"].clone();
        assert_eq!(
            evidence::evaluate_preflight(&evidence, &pins, &receipt).err(),
            Some(Refused("control_unavailable"))
        );
    }
}
#[test]
fn review_body_hash_uses_utf8_and_unicode_word_boundaries() {
    let (mut evidence, pins, mut receipt) = baseline();
    let body = format!(
        "{}\nVerified café 雪 🚀; source {}.",
        evidence["reviews"][0]["body"].as_str().unwrap(),
        pins["source_sha"].as_str().unwrap().to_uppercase()
    );
    receipt["body_sha256"] = json!(json::hash_bytes(body.as_bytes()));
    evidence["reviews"][0]["body"] = json!(body);
    assert!(evidence::evaluate_preflight(&evidence, &pins, &receipt).is_ok());
    let body = format!(
        "{}\nOld source BBBBBBBBBBBB.",
        evidence["reviews"][0]["body"].as_str().unwrap()
    );
    receipt["body_sha256"] = json!(json::hash_bytes(body.as_bytes()));
    evidence["reviews"][0]["body"] = json!(body);
    assert_eq!(
        evidence::evaluate_preflight(&evidence, &pins, &receipt).err(),
        Some(Refused("approval_body_mismatch"))
    );
}

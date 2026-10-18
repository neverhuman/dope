use super::*;
fn app(root: PathBuf) -> App {
    App {
        root,
        token: "aabbccddeeff00112233445566778899".into(),
        origin: "http://127.0.0.1:1234".into(),
        capabilities: HashMap::new(),
        jobs: HashMap::new(),
        active_jobs: Arc::new(AtomicUsize::new(0)),
    }
}
fn req(method: &str, path: &str, body: Value) -> Request {
    let mut headers = HashMap::new();
    headers.insert("host".into(), "127.0.0.1:1234".into());
    headers.insert("origin".into(), "http://127.0.0.1:1234".into());
    headers.insert(
        "authorization".into(),
        "Bearer aabbccddeeff00112233445566778899".into(),
    );
    headers.insert(
        "x-dope-csrf".into(),
        "aabbccddeeff00112233445566778899".into(),
    );
    if method == "POST" {
        headers.insert("content-type".into(), "application/json".into());
    }
    Request {
        method: method.into(),
        path: path.into(),
        headers,
        body: serde_json::to_vec(&body).unwrap(),
    }
}
#[test]
fn remote_bind_rejected() {
    assert_eq!(
        "loopback_only",
        run(vec!["--bind".into(), "0.0.0.0:80".into()]).unwrap_err()
    );
}
#[test]
fn origin_and_token_required() {
    let root = std::env::temp_dir();
    let mut a = app(root);
    let mut r = req("GET", "/v1/capabilities", json!({}));
    r.headers
        .insert("origin".into(), "https://evil.invalid".into());
    assert_eq!(403, a.handle(r).status);
}
#[test]
fn traversal_symlink_and_upload_fields_refused() {
    let base = std::env::temp_dir().join(format!("dope-service-{}", std::process::id()));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir(&base).unwrap();
    fs::write(base.join("ok.json"), b"{}").unwrap();
    let mut a = app(fs::canonicalize(&base).unwrap());
    let mut r = req(
        "POST",
        "/v1/capabilities",
        json!({"relative_path":"../outside"}),
    );
    assert_eq!(400, a.handle(r).status);
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(base.join("ok.json"), base.join("link.json")).unwrap();
        r = req(
            "POST",
            "/v1/capabilities",
            json!({"relative_path":"link.json"}),
        );
        assert_eq!(400, a.handle(r).status);
    }
    r = req(
        "POST",
        "/v1/capabilities",
        json!({"relative_path":"ok.json","upload":"bad"}),
    );
    assert_eq!(400, a.handle(r).status);
    let _ = fs::remove_dir_all(base);
}
#[test]
fn missing_token_fails_closed() {
    let root = std::env::temp_dir();
    let mut a = app(root);
    let mut r = req("GET", "/v1/capabilities", json!({}));
    r.headers.remove("authorization");
    assert_eq!(403, a.handle(r).status);
}

#[test]
fn csrf_and_capability_replacement_fail_closed() {
    let base = std::env::temp_dir().join(format!("dope-service-cap-{}", std::process::id()));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir(&base).unwrap();
    fs::write(base.join("input.json"), b"source-free fixture").unwrap();
    fs::write(base.join("outside.json"), b"outside").unwrap();
    let mut a = app(fs::canonicalize(&base).unwrap());
    let mut no_csrf = req(
        "POST",
        "/v1/capabilities",
        json!({"relative_path":"input.json"}),
    );
    no_csrf.headers.remove("x-dope-csrf");
    assert_eq!(403, a.handle(no_csrf).status);
    let capability = a.handle(req(
        "POST",
        "/v1/capabilities",
        json!({"relative_path":"input.json"}),
    ));
    let id = capability.body["capability_id"]
        .as_str()
        .unwrap()
        .to_owned();
    fs::remove_file(base.join("input.json")).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(base.join("outside.json"), base.join("input.json")).unwrap();
    let job = req(
        "POST",
        "/v1/jobs",
        json!({"format":"dope-local-job","version":1,"kind":"inspect","capability_id":id}),
    );
    assert_eq!(400, a.handle(job).status);
    let _ = fs::remove_dir_all(base);
}

#[test]
fn cancellation_state_is_preserved_without_qualification() {
    let mut a = app(std::env::temp_dir());
    let id = "0123456789abcdef0123456789abcdef".to_owned();
    let receipt = Receipt {
        format: "dope-local-job-receipt",
        version: 1,
        job_id: id.clone(),
        job_state: "running",
        kind: "inspect".into(),
        input_sha256: "0".repeat(64),
        input_bytes: 1,
        report_sha256: None,
        report: None,
        utility_state: "unassessed",
        empirical_privacy_state: "unassessed",
        formal_dp_state: "not_claimed",
        release_authorization_state: "not_authorized",
        qualified: false,
    };
    a.jobs.insert(
        id.clone(),
        Arc::new(JobEntry {
            receipt: Mutex::new(receipt),
            cancelled: Arc::new(AtomicBool::new(false)),
        }),
    );
    let response = a.handle(req("POST", &format!("/v1/jobs/{id}/cancel"), json!({})));
    assert_eq!(200, response.status);
    assert_eq!("cancelled", a.jobs[&id].receipt.lock().unwrap().job_state);
    assert!(!a.jobs[&id].receipt.lock().unwrap().qualified);
}

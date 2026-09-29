
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ValidationAuthorizationEvent {
    pub format: String,
    pub version: u8,
    pub source_commit: String,
    pub router_bundle_sha256: String,
    pub authorization_token_sha256: String,
    pub opened_unix_seconds: u64,
    pub validation_cert_open_count: usize,
    pub sealed_test_open_count: usize,
}

pub fn open_validation_cert(state_dir: &Path, authorization: &Path) -> Result<CampaignState> {
    require_host("xbabe3", "validation-cert authorization")?;
    let mut state = load_state(state_dir)?;
    if state.phase != CampaignPhase::RouterFrozen || state.validation_cert_open_count != 0 {
        return Err(DopeError::Data(
            "validation-cert requires a frozen router and a single unopened authorization".into(),
        ));
    }
    let token = fs::read(authorization).map_err(|error| io_error(authorization, error))?;
    if token.is_empty() {
        return Err(DopeError::Data("validation authorization is empty".into()));
    }
    let event = ValidationAuthorizationEvent {
        format: "dope-validation-cert-authorization".into(),
        version: 1,
        source_commit: state.source_commit.clone(),
        router_bundle_sha256: state
            .router_bundle
            .as_ref()
            .expect("validated router state")
            .sha256
            .clone(),
        authorization_token_sha256: hashes(&token).sha256,
        opened_unix_seconds: unix_now(),
        validation_cert_open_count: 1,
        sealed_test_open_count: 0,
    };
    write_canonical(
        &state_dir.join("validation-cert-authorization.json"),
        &event,
    )?;
    state.phase = CampaignPhase::ValidationCertOpen;
    state.validation_cert_open_count = 1;
    state.validate()?;
    write_canonical(&state_path(state_dir), &state)?;
    Ok(state)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ControllerRequest {
    Lease { worker: String },
    MarkRunning { job_id: String, worker: String },
    Heartbeat { job_id: String, worker: String },
    Finish { receipt: Box<JobReceipt> },
    Status,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ControllerResponse {
    pub ok: bool,
    pub error: Option<String>,
    pub leased: Option<LeasedJob>,
    pub status: Option<crate::ledger::LedgerStatus>,
}

impl ControllerResponse {
    fn ok() -> Self {
        Self {
            ok: true,
            error: None,
            leased: None,
            status: None,
        }
    }

    fn error(error: impl ToString) -> Self {
        Self {
            ok: false,
            error: Some(error.to_string()),
            leased: None,
            status: None,
        }
    }
}

fn handle_request(
    state_dir: &Path,
    ledger: &CampaignLedger,
    key: &[u8; 32],
    request: ControllerRequest,
) -> Result<ControllerResponse> {
    match request {
        ControllerRequest::Lease { worker } => {
            require_durable_space(state_dir)?;
            ledger.reclaim_expired(unix_now())?;
            let state = load_state(state_dir)?;
            let phases: &[&str] = if worker == "xbabe3" {
                if state.phase != CampaignPhase::ValidationCertOpen {
                    &[]
                } else {
                    &["validation_cert"]
                }
            } else {
                &["training_gold", "validation_select"]
            };
            let mut response = ControllerResponse::ok();
            response.leased = ledger.lease_next_for(&worker, LEASE_SECONDS, phases)?;
            Ok(response)
        }
        ControllerRequest::MarkRunning { job_id, worker } => {
            ledger.mark_running(&job_id, &worker)?;
            Ok(ControllerResponse::ok())
        }
        ControllerRequest::Heartbeat { job_id, worker } => {
            ledger.heartbeat(&job_id, &worker, LEASE_SECONDS)?;
            Ok(ControllerResponse::ok())
        }
        ControllerRequest::Finish { receipt } => {
            ledger.finish_signed(&receipt, key)?;
            Ok(ControllerResponse::ok())
        }
        ControllerRequest::Status => {
            let mut response = ControllerResponse::ok();
            response.status = Some(ledger.status()?);
            Ok(response)
        }
    }
}

pub fn serve(
    state_dir: &Path,
    bind: &str,
    receipt_key: &Path,
    max_requests: Option<usize>,
) -> Result<()> {
    let state = load_state(state_dir)?;
    let key = load_receipt_key(receipt_key)?;
    if hashes(&key).sha256 != state.receipt_key_sha256 {
        return Err(DopeError::Data(
            "controller receipt key differs from freeze".into(),
        ));
    }
    let ledger = CampaignLedger::open(&ledger_path(state_dir))?;
    let listener = TcpListener::bind(bind)
        .map_err(|error| DopeError::Data(format!("cannot bind controller {bind}: {error}")))?;
    for (index, stream) in listener.incoming().enumerate() {
        let mut stream = stream.map_err(|error| DopeError::Data(error.to_string()))?;
        let mut bytes = Vec::new();
        stream
            .read_to_end(&mut bytes)
            .map_err(|error| DopeError::Data(error.to_string()))?;
        let response = match serde_json::from_slice::<ControllerRequest>(&bytes) {
            Ok(request) => handle_request(state_dir, &ledger, &key, request)
                .unwrap_or_else(ControllerResponse::error),
            Err(error) => ControllerResponse::error(error),
        };
        stream
            .write_all(&canonical_json(&response)?)
            .map_err(|error| DopeError::Data(error.to_string()))?;
        if max_requests.is_some_and(|limit| index + 1 >= limit) {
            break;
        }
    }
    Ok(())
}

fn controller_call(address: &str, request: &ControllerRequest) -> Result<ControllerResponse> {
    let mut stream = TcpStream::connect(address).map_err(|error| {
        DopeError::Data(format!("controller {address} is unreachable: {error}"))
    })?;
    stream
        .write_all(&canonical_json(request)?)
        .map_err(|error| DopeError::Data(error.to_string()))?;
    stream
        .shutdown(Shutdown::Write)
        .map_err(|error| DopeError::Data(error.to_string()))?;
    let mut bytes = Vec::new();
    stream
        .read_to_end(&mut bytes)
        .map_err(|error| DopeError::Data(error.to_string()))?;
    let response: ControllerResponse = serde_json::from_slice(&bytes)?;
    if !response.ok {
        return Err(DopeError::Data(
            response
                .error
                .unwrap_or_else(|| "controller rejected request".into()),
        ));
    }
    Ok(response)
}

fn execute_job(
    controller: &str,
    worker: &str,
    work_dir: &Path,
    leased: &LeasedJob,
    key: &[u8; 32],
) -> Result<JobReceipt> {
    let started = unix_now();
    controller_call(
        controller,
        &ControllerRequest::MarkRunning {
            job_id: leased.job_id.clone(),
            worker: worker.into(),
        },
    )?;
    let mut command = Command::new(&leased.spec.command[0]);
    command
        .args(&leased.spec.command[1..])
        .current_dir(work_dir)
        .envs(&leased.spec.environment)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    let mut failure_class = None;
    let mut failure = None;
    let mut exit_code = None;
    let mut state = JobState::Failed;
    let mut output_hashes = BTreeMap::new();
    let mut evidence = None;
    match command.spawn() {
        Err(error) => {
            failure_class = Some(FailureClass::Infrastructure);
            failure = Some(format!("failed to spawn frozen command: {error}"));
        }
        Ok(mut child) => {
            let deadline = Instant::now() + Duration::from_secs(leased.spec.timeout_seconds);
            let mut next_heartbeat = Instant::now() + Duration::from_secs(HEARTBEAT_SECONDS);
            loop {
                if let Some(status) = child
                    .try_wait()
                    .map_err(|error| DopeError::Data(error.to_string()))?
                {
                    exit_code = status.code();
                    if status.success() {
                        let outputs = leased
                            .spec
                            .expected_outputs
                            .iter()
                            .map(|(name, path)| {
                                let path = if path.is_absolute() {
                                    path.clone()
                                } else {
                                    work_dir.join(path)
                                };
                                Ok((name.clone(), path.clone(), file_hashes(&path)?))
                            })
                            .collect::<Result<Vec<_>>>();
                        match outputs {
                            Ok(outputs) => {
                                for (name, _, digest) in &outputs {
                                    output_hashes.insert(name.clone(), digest.clone());
                                }
                                let metric_path = outputs
                                    .iter()
                                    .find(|(name, _, _)| name == "metrics" || name == "evidence")
                                    .map(|(_, path, _)| path);
                                match metric_path
                                    .map(|path| read_json::<JobEvidence>(path))
                                    .transpose()
                                {
                                    Ok(Some(value))
                                        if value.validate_against(&leased.spec).is_ok() =>
                                    {
                                        evidence = Some(value);
                                        state = JobState::Succeeded;
                                    }
                                    Ok(_) => {
                                        failure_class = Some(FailureClass::ContractMismatch);
                                        failure = Some("successful command did not emit typed metrics evidence".into());
                                    }
                                    Err(error) => {
                                        failure_class = Some(FailureClass::ContractMismatch);
                                        failure = Some(error.to_string());
                                    }
                                }
                            }
                            Err(error) => {
                                failure_class = Some(FailureClass::ContractMismatch);
                                failure = Some(error.to_string());
                            }
                        }
                    } else {
                        failure_class = Some(FailureClass::DeterministicModel);
                        failure = Some(format!("frozen command exited with status {status}"));
                    }
                    break;
                }
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    state = JobState::TimedOut;
                    failure_class = Some(FailureClass::Timeout);
                    failure = Some("frozen command exceeded its timeout".into());
                    break;
                }
                if Instant::now() >= next_heartbeat {
                    controller_call(
                        controller,
                        &ControllerRequest::Heartbeat {
                            job_id: leased.job_id.clone(),
                            worker: worker.into(),
                        },
                    )?;
                    next_heartbeat = Instant::now() + Duration::from_secs(HEARTBEAT_SECONDS);
                }
                thread::sleep(Duration::from_millis(100));
            }
        }
    }
    let mut receipt = JobReceipt {
        format: "dope-job-receipt".into(),
        version: 1,
        job_id: leased.job_id.clone(),
        attempt: leased.attempt,
        worker: worker.into(),
        state,
        started_unix_seconds: started,
        finished_unix_seconds: unix_now(),
        source_commit: leased.spec.source_commit.clone(),
        environment_lock_sha256: leased.spec.environment_lock_sha256.clone(),
        corpus_manifest_sha256: leased.spec.corpus_manifest_sha256.clone(),
        kpi_contract_sha256: leased.spec.kpi_contract_sha256.clone(),
        output_hashes,
        failure_class,
        failure,
        command_exit_code: exit_code,
        evidence,
        signature: None,
    };
    receipt.sign(key)?;
    Ok(receipt)
}

pub fn worker(
    controller: &str,
    worker: &str,
    work_dir: &Path,
    receipt_key: &Path,
    once: bool,
) -> Result<usize> {
    let key = load_receipt_key(receipt_key)?;
    let mut completed = 0;
    loop {
        let response = controller_call(
            controller,
            &ControllerRequest::Lease {
                worker: worker.into(),
            },
        )?;
        let Some(leased) = response.leased else { break };
        let receipt = execute_job(controller, worker, work_dir, &leased, &key)?;
        controller_call(
            controller,
            &ControllerRequest::Finish {
                receipt: Box::new(receipt),
            },
        )?;
        completed += 1;
        if once {
            break;
        }
    }
    Ok(completed)
}

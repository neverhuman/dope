// Owner-only custody operations; included in the parent module and code hash.
fn table(bytes: &[u8], task: Task, policy: &ReleasePolicy) -> Result<Table> {
    let mut flat = Vec::new();
    let mut target = Vec::new();
    let mut features = None;
    for record in csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_reader(bytes)
        .records()
    {
        let record = record.map_err(|_| denied())?;
        if record.len() < 2 || features.is_some_and(|n| n != record.len() - 1) {
            return Err(denied());
        }
        features = Some(record.len() - 1);
        for field in record.iter().take(record.len() - 1) {
            flat.push(
                if field.trim().is_empty() || field.eq_ignore_ascii_case("nan") {
                    f32::NAN
                } else {
                    field.parse().map_err(|_| denied())?
                },
            );
        }
        target.push(record[record.len() - 1].parse().map_err(|_| denied())?);
    }
    Table::from_arrays_with_policy(
        &flat,
        &target,
        target.len(),
        features.ok_or_else(denied)?,
        task,
        policy,
    )
    .map_err(|_| denied())
}

fn event(
    ledger: &mut File,
    declaration: &EvaluationDeclaration,
    attempt: u32,
    kind: &str,
    charge: u32,
) -> Result<EvaluationEvent> {
    let value = EvaluationEvent {
        confirmation_id: declaration.confirmation_id.clone(),
        batch_hash: declaration.batch_hash.clone(),
        declaration_hash: sha(&encoded(declaration)?),
        attempt,
        kind: kind.into(),
        query_opens_charged: charge,
    };
    ledger
        .write_all(&encoded(&value)?)
        .and_then(|_| ledger.write_all(b"\n"))
        .and_then(|_| ledger.sync_all())
        .map_err(|_| denied())?;
    Ok(value)
}

/// Owner process only. Keep the root private (0700), files owner-only, and the
/// declaration immutable before starting the endpoint. The socket may be shared.
/// This installs a process-wide network/exec deny filter; run in a dedicated process.
pub fn serve_frozen_batch(listener: &UnixListener, root: &Path) -> Result<()> {
    let uid = effective_uid();
    use std::os::unix::fs::MetadataExt;
    let metadata = fs::symlink_metadata(root).map_err(|_| denied())?;
    if !metadata.is_dir() || metadata.uid() != uid || metadata.mode() & 0o777 != 0o700 {
        return Err(denied());
    }
    let declaration: EvaluationDeclaration = serde_json::from_value(parse_json(&read_regular(
        &root.join("declaration.json"),
        Some(uid),
        true,
    )?)?)
    .map_err(|_| denied())?;
    if declaration.producer_uid == uid
        || declaration.producer_uid == 0
        || declaration.fit_hash == declaration.query_hash
        || declaration.max_attempts == 0
        || declaration.query_open_budget == 0
        || [
            &declaration.confirmation_id,
            &declaration.batch_hash,
            &declaration.protocol_hash,
            &declaration.fit_hash,
            &declaration.query_hash,
        ]
        .iter()
        .any(|v| !digest_valid(v))
    {
        return Err(denied());
    }
    let protocol_bytes = read_regular(&root.join("protocol.json"), Some(uid), true)?;
    let protocol = parse_json(&protocol_bytes)?;
    validate_behavior_contract(&protocol)?;
    if sha(&protocol_bytes) != declaration.protocol_hash
        || protocol["protocol_frozen"] != true
        || protocol["confirmation_allowed"] != true
    {
        return Err(denied());
    }
    let (mut connection, _) = listener.accept().map_err(|_| denied())?;
    if peer_uid(&connection)? != declaration.producer_uid {
        return Err(denied());
    }
    connection
        .set_read_timeout(Some(Duration::from_secs(60)))
        .map_err(|_| denied())?;
    let mut request = Vec::new();
    (&mut connection)
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut request)
        .map_err(|_| denied())?;
    let batch: Vec<FrozenSelection> =
        serde_json::from_value(parse_json(&request)?).map_err(|_| denied())?;
    if batch.is_empty() || batch.len() > 64 || frozen_batch_hash(&batch)? != declaration.batch_hash
    {
        return Err(denied());
    }
    for frozen in &batch {
        frozen.validate()?;
        if frozen.fit_hash != declaration.fit_hash {
            return Err(denied());
        }
    }
    deny_evaluator_network()?;
    // The owner controls the ledger directory, independently of producer paths.
    let path = root.join(format!("{}.jsonl", declaration.confirmation_id));
    let mut ledger = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
        .map_err(|_| denied())?;
    let metadata = ledger.metadata().map_err(|_| denied())?;
    if !metadata.is_file()
        || metadata.uid() != uid
        || metadata.mode() & 0o077 != 0
        || metadata.nlink() != 1
        || metadata.len() > 16 * 1024 * 1024
    {
        return Err(denied());
    }
    // SAFETY: flock operates on this live file descriptor; the lock is held until drop.
    if unsafe { libc::flock(ledger.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(denied());
    }
    File::open(root)
        .and_then(|f| f.sync_all())
        .map_err(|_| denied())?;
    let mut history = String::new();
    ledger.read_to_string(&mut history).map_err(|_| denied())?;
    let previous = history
        .lines()
        .map(|line| serde_json::from_str::<EvaluationEvent>(line).map_err(|_| denied()))
        .collect::<Result<Vec<_>>>()?;
    let declaration_hash = sha(&encoded(&declaration)?);
    if previous.iter().any(|e| {
        e.batch_hash != declaration.batch_hash
            || e.declaration_hash != declaration_hash
            || e.confirmation_id != declaration.confirmation_id
            || e.attempt == 0
            || !matches!(
                e.kind.as_str(),
                "started" | "query_access" | "query_opened" | "interrupted"
            )
            || e.query_opens_charged != u32::from(e.kind == "query_access")
    }) {
        return Err(denied());
    }
    let attempt = previous
        .iter()
        .map(|e| e.attempt)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(denied)?;
    let charged = previous.iter().try_fold(0u32, |total, e| {
        total.checked_add(e.query_opens_charged).ok_or_else(denied)
    })?;
    let cost = u32::try_from(batch.len()).map_err(|_| denied())?;
    if attempt > declaration.max_attempts
        || charged
            .checked_add(cost)
            .is_none_or(|n| n > declaration.query_open_budget)
    {
        return Err(denied());
    }
    let mut events = previous;
    events.push(event(&mut ledger, &declaration, attempt, "started", 0)?);
    let mut reports = Vec::new();
    for (index, frozen) in batch.iter().enumerate() {
        // Charge and fsync BEFORE opening labels, including failed/interrupted attempts.
        events.push(event(
            &mut ledger,
            &declaration,
            attempt,
            "query_access",
            1,
        )?);
        let result = (|| {
            let fit_bytes = read_regular(&root.join("train.csv"), Some(uid), true)?;
            let query_bytes = read_regular(&root.join("test.csv"), Some(uid), true)?;
            events.push(event(
                &mut ledger,
                &declaration,
                attempt,
                "query_opened",
                0,
            )?);
            if sha(&fit_bytes) != declaration.fit_hash
                || sha(&query_bytes) != declaration.query_hash
            {
                return Err(denied());
            }
            let task = decode_kernel(frozen.artifact())?.task;
            let fit = table(&fit_bytes, task, &frozen.policy)?;
            let query = table(&query_bytes, task, &frozen.policy)?;
            let artifact_path = root.join(format!(
                "{}-{attempt}-{index}.dpk",
                declaration.confirmation_id
            ));
            let mut artifact = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&artifact_path)
                .map_err(|_| denied())?;
            artifact
                .write_all(frozen.artifact())
                .and_then(|_| artifact.sync_all())
                .map_err(|_| denied())?;
            let out = root.join(format!(
                "{}-{attempt}-{index}.certification.json",
                declaration.confirmation_id
            ));
            certify_frozen_tables(frozen, &artifact_path, &out, fit, query)
        })();
        match result {
            Ok(report) => reports.push(report),
            Err(error) => {
                let kind = match &error {
                    DopeError::Io { source, .. }
                        if matches!(
                            source.kind(),
                            std::io::ErrorKind::Interrupted
                                | std::io::ErrorKind::TimedOut
                                | std::io::ErrorKind::WouldBlock
                        ) =>
                    {
                        "interrupted"
                    }
                    _ => "rejected",
                };
                event(&mut ledger, &declaration, attempt, kind, 0)?;
                return Err(denied());
            }
        }
    }
    events.push(event(&mut ledger, &declaration, attempt, "completed", 0)?);
    let opens = events
        .iter()
        .filter(|event| event.kind == "query_opened")
        .count();
    let outcome = EvaluationOutcome {
        batch_hash: declaration.batch_hash,
        reports,
        events,
        query_opens_charged: charged + cost,
        sealed_test_open_count: opens,
    };
    connection
        .write_all(&encoded(&outcome)?)
        .map_err(|_| denied())?;
    Ok(())
}

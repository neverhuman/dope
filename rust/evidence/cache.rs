pub fn commit_completed_receipt(cache: &Path, receipt: &VerifiedEvaluationReceipt) -> Result<()> {
    let key = scientific_cache_key(receipt.identity())?;
    let envelope = json!({"format":"dope-scientific-cache", "version":1, "state":"complete", "cache_key":key, "receipt_hash":receipt.receipt_hash, "receipt":receipt.receipt});
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(cache)
        .map_err(|_| error("cache_commit_failed"))?;
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let stage = cache.join(format!(
        ".staging-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&stage)
        .map_err(|_| error("cache_commit_failed"))?;
    file.write_all(&canonical_json(&envelope)?)
        .and_then(|_| file.sync_all())
        .map_err(|_| error("cache_commit_failed"))?;
    fs::rename(stage, cache.join(format!("{key}.json")))
        .map_err(|_| error("cache_commit_failed"))?;
    File::open(cache)
        .and_then(|f| f.sync_all())
        .map_err(|_| error("cache_commit_failed"))?;
    Ok(())
}
pub fn load_completed_receipt(
    cache: &Path,
    root: &Path,
    policy: &VerificationPolicy,
) -> Result<VerifiedEvaluationReceipt> {
    let key = scientific_cache_key(&policy.identity)?;
    let (_, _, data) = read_file(cache, &format!("{key}.json"), true)?;
    let envelope = parse_json(&data)?;
    closed(
        &envelope,
        &[
            "format",
            "version",
            "state",
            "cache_key",
            "receipt_hash",
            "receipt",
        ],
    )?;
    require(
        version(&envelope, "dope-scientific-cache")
            && envelope["state"] == "complete"
            && envelope["cache_key"] == key
            && envelope["receipt_hash"] == policy.receipt_hash,
        "cache_incomplete_or_mismatched",
    )?;
    verify_receipt(UnverifiedReceipt(envelope["receipt"].clone()), root, policy)
}

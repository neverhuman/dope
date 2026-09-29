

pub fn freeze_router(
    state_dir: &Path,
    bundle_path: &Path,
    evidence_path: &Path,
    allow_failed_frontier: bool,
) -> Result<CampaignState> {
    let mut state = load_state(state_dir)?;
    if state.phase != CampaignPhase::Frozen {
        return Err(DopeError::Data("router may be frozen exactly once".into()));
    }
    let bundle: RouterBundle = read_json(bundle_path)?;
    bundle.validate()?;
    if !bundle.trained {
        return Err(DopeError::Data("untrained router cannot be frozen".into()));
    }
    let evidence: RouterEvidence = read_json(evidence_path)?;
    if evidence.format != "dope-router-evidence" || evidence.version != 1 {
        return Err(DopeError::Data("invalid router evidence format".into()));
    }
    let failed = evidence.failed_gates();
    if !failed.is_empty() && !allow_failed_frontier {
        return Err(DopeError::Data(format!(
            "router freeze gates failed (use the explicit measured failed-frontier path to continue certification without promotion): {}",
            failed.join(",")
        )));
    }
    let bundle_hash = copy_file(bundle_path, &state_dir.join("router.bundle"))?;
    let evidence_hash = copy_file(evidence_path, &state_dir.join("router-evidence.json"))?;
    if bundle.training_evidence_sha256.as_deref() != Some(&evidence.training_evidence_sha256) {
        return Err(DopeError::Data(
            "router bundle and training evidence hashes differ".into(),
        ));
    }
    state.phase = CampaignPhase::RouterFrozen;
    state.router_bundle = Some(bundle_hash);
    state.router_evidence = Some(evidence_hash);
    state.validate()?;
    write_canonical(&state_path(state_dir), &state)?;
    CampaignLedger::open(&ledger_path(state_dir))?.put_metadata("router_evidence", &evidence)?;
    Ok(state)
}

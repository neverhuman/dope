fn run_campaign_group_03(command: CampaignCommand) -> Result<()> {
    match command {
        CampaignCommand::BuildRc {
                state_dir,
                metrics,
                release_dir,
                gate_evidence,
                coverage,
                out,
                source_commit,
                source_tag_object_sha256,
                signature_kind,
                public_key_sha256,
            } => {
                let metrics: dope_kernel::campaign::MetricsBundle = serde_json::from_slice(
                    &fs::read(&metrics)
                        .map_err(|error| dope_kernel::error::io_error(&metrics, error))?,
                )?;
                if !metrics.failed_gates.is_empty() {
                    let decision = failed_release_decision(&state_dir, &metrics)?;
                    dope_kernel::production::write_canonical(&out, &decision)?;
                    println!("{}", serde_json::to_string(&decision)?);
                } else {
                    let evidence: GateEvidence =
                        serde_json::from_slice(&fs::read(&gate_evidence).map_err(|error| {
                            dope_kernel::error::io_error(&gate_evidence, error)
                        })?)?;
                    let coverage: CoverageReport = serde_json::from_slice(
                        &fs::read(&coverage)
                            .map_err(|error| dope_kernel::error::io_error(&coverage, error))?,
                    )?;
                    let manifest = build_rc_manifest(
                        &release_dir,
                        &evidence,
                        &coverage,
                        &BuildRcOptions {
                            source_commit,
                            source_tag_object_sha256,
                            signature_kind,
                            public_key_sha256,
                        },
                    )?;
                    dope_kernel::production::write_canonical(&out, &manifest)?;
                    println!("{}", serde_json::to_string(&manifest)?);
                }
            },
        _ => return Err(DopeError::Unsupported("campaign command routing mismatch".into())),
    };
    Ok(())
}

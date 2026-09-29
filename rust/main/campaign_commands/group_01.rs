fn run_campaign_group_01(command: CampaignCommand) -> Result<()> {
    match command {
        CampaignCommand::FreezeDeepImplementation {
                source_tree,
                binary,
                out,
            } => {
                let manifest = freeze_deep_implementation(&source_tree, &binary)?;
                dope_kernel::production::write_canonical(&out, &manifest)?;
                println!("{}", serde_json::to_string(&manifest)?);
            },
        CampaignCommand::CreateReceiptKey { out } => {
                let hashes = create_receipt_key(&out)?;
                println!("{}", serde_json::to_string(&hashes)?);
            },
        CampaignCommand::Freeze {
                state_dir,
                specs,
                source_commit,
                source_tree,
                environment_lock,
                host_inventory,
                source_hashes,
                sbom,
                notices,
                checkpoint_manifest,
                corpus_manifest,
                validation_manifest,
                receipt_key,
            } => {
                let state = freeze(&FreezeOptions {
                    state_dir: &state_dir,
                    specs: &specs,
                    source_commit: &source_commit,
                    source_tree: &source_tree,
                    environment_lock: &environment_lock,
                    host_inventory: &host_inventory,
                    source_hashes: &source_hashes,
                    sbom: &sbom,
                    notices: &notices,
                    checkpoint_manifest: &checkpoint_manifest,
                    corpus_manifest: &corpus_manifest,
                    validation_manifest: &validation_manifest,
                    receipt_key: &receipt_key,
                })?;
                println!("{}", serde_json::to_string(&state)?);
            },
        CampaignCommand::Serve {
                state_dir,
                bind,
                receipt_key,
                max_requests,
            } => {
                serve(&state_dir, &bind, &receipt_key, max_requests)?;
            },
        CampaignCommand::Worker {
                controller,
                worker: worker_id,
                work_dir,
                receipt_key,
                once,
            } => {
                let completed = worker(&controller, &worker_id, &work_dir, &receipt_key, once)?;
                println!(
                    "{}",
                    serde_json::json!({"completed": completed, "worker": worker_id})
                );
            },
        CampaignCommand::VerifyCorpus { manifest } => {
                let verification = dope_kernel::campaign::verify_corpus_sample(&manifest)?;
                println!("{}", serde_json::to_string(&verification)?);
            },
        CampaignCommand::ExportCorpusSample { manifest, out } => {
                let plan = corpus_sample_plan(&manifest)?;
                dope_kernel::production::write_canonical(&out, &plan)?;
                println!(
                    "{}",
                    serde_json::json!({"paths": plan.paths.len(), "out": out})
                );
            },
        CampaignCommand::PlanCohort {
                manifest,
                validation_manifest,
                kind,
                out,
            } => {
                let cohort = plan_cohort(&manifest, &validation_manifest, &kind)?;
                dope_kernel::production::write_canonical(&out, &cohort)?;
                println!(
                    "{}",
                    serde_json::json!({"kind": cohort.kind, "records": cohort.records.len(), "exclusions": cohort.exclusions.len(), "out": out})
                );
            },
        CampaignCommand::PlanDeepCohorts { training_gold, out } => {
                let cohort: CohortPlan = serde_json::from_slice(
                    &fs::read(&training_gold)
                        .map_err(|error| dope_kernel::error::io_error(&training_gold, error))?,
                )?;
                let plan = plan_deep_cohorts(&cohort)?;
                dope_kernel::production::write_canonical(&out, &plan)?;
                println!(
                    "{}",
                    serde_json::json!({"discovery": plan.discovery.len(), "confirmation": plan.confirmation.len(), "out": out})
                );
            },
        CampaignCommand::MaterializeDeepCohort { plan, stage, out } => {
                let plan: DeepCohortPlan = serde_json::from_slice(
                    &fs::read(&plan).map_err(|error| dope_kernel::error::io_error(&plan, error))?,
                )?;
                let cohort = plan.stage_cohort(&stage)?;
                dope_kernel::production::write_canonical(&out, &cohort)?;
                println!(
                    "{}",
                    serde_json::json!({"stage": stage, "records": cohort.records.len(), "out": out})
                );
            },
        CampaignCommand::BuildDeepEvidence { cohort, block, out } => {
                let cohort_plan: CohortPlan = serde_json::from_slice(
                    &fs::read(&cohort)
                        .map_err(|error| dope_kernel::error::io_error(&cohort, error))?,
                )?;
                let outcomes = rebuild_deep_outcomes(&cohort_plan, &block)?;
                dope_kernel::production::write_canonical(&out, &outcomes)?;
                println!(
                    "{}",
                    serde_json::json!({"outcomes": outcomes.len(), "out": out})
                );
            },
        CampaignCommand::SelectDeepDiscovery { evidence, out } => {
                let outcomes: Vec<DeepOutcome> = serde_json::from_slice(
                    &fs::read(&evidence)
                        .map_err(|error| dope_kernel::error::io_error(&evidence, error))?,
                )?;
                let selection = select_discovery_configurations(&outcomes)?;
                dope_kernel::production::write_canonical(&out, &selection)?;
                println!("{}", serde_json::to_string(&selection)?);
            },
        CampaignCommand::QualifyDeepConfirmation {
                candidate,
                evidence,
                scheduled_cells,
                out,
            } => {
                let outcomes: Vec<DeepOutcome> = serde_json::from_slice(
                    &fs::read(&evidence)
                        .map_err(|error| dope_kernel::error::io_error(&evidence, error))?,
                )?;
                let qualification =
                    qualify_confirmation_candidate(&candidate, &outcomes, scheduled_cells)?;
                dope_kernel::production::write_canonical(&out, &qualification)?;
                println!("{}", serde_json::to_string(&qualification)?);
            },
        CampaignCommand::SelectDeepFinal {
                evidence,
                qualification,
                out,
            } => {
                let outcomes: Vec<DeepOutcome> = serde_json::from_slice(
                    &fs::read(&evidence)
                        .map_err(|error| dope_kernel::error::io_error(&evidence, error))?,
                )?;
                let qualifications = qualification
                    .iter()
                    .map(|path| {
                        fs::read(path)
                            .map_err(|error| dope_kernel::error::io_error(path, error))
                            .and_then(|bytes| {
                                serde_json::from_slice::<Qualification>(&bytes).map_err(Into::into)
                            })
                    })
                    .collect::<dope_kernel::Result<Vec<_>>>()?;
                let selection = select_final_family(&outcomes, &qualifications)?;
                dope_kernel::production::write_canonical(&out, &selection)?;
                println!("{}", serde_json::to_string(&selection)?);
            },
        CampaignCommand::BuildDeepArtifactManifest {
                selection,
                artifact,
                representative_dataset_id,
                out,
            } => {
                let selection: FinalFamilySelection = serde_json::from_slice(
                    &fs::read(&selection)
                        .map_err(|error| dope_kernel::error::io_error(&selection, error))?,
                )?;
                let artifacts = artifact
                    .iter()
                    .map(|entry| {
                        let (dataset_id, path) = entry.split_once('=').ok_or_else(|| {
                            dope_kernel::DopeError::Data(
                                "deep artifact must use DATASET_ID=PATH".into(),
                            )
                        })?;
                        Ok((dataset_id.to_string(), PathBuf::from(path)))
                    })
                    .collect::<dope_kernel::Result<Vec<_>>>()?;
                let manifest = build_deep_artifact_manifest(
                    &selection,
                    &artifacts,
                    &representative_dataset_id,
                )?;
                dope_kernel::production::write_canonical(&out, &manifest)?;
                println!("{}", serde_json::to_string(&manifest)?);
            },
        CampaignCommand::BuildDeepModelCard {
                selection,
                report,
                out,
            } => {
                let selection: FinalFamilySelection = serde_json::from_slice(
                    &fs::read(&selection)
                        .map_err(|error| dope_kernel::error::io_error(&selection, error))?,
                )?;
                let report: DeepCampaignReport = serde_json::from_slice(
                    &fs::read(&report)
                        .map_err(|error| dope_kernel::error::io_error(&report, error))?,
                )?;
                let card = build_deep_model_card(&selection, &report)?;
                fs::write(&out, card).map_err(|error| dope_kernel::error::io_error(&out, error))?;
                println!("{}", serde_json::json!({"out": out}));
            },
        CampaignCommand::ReportDeepCampaign { evidence, out } => {
                let outcomes: Vec<DeepOutcome> = serde_json::from_slice(
                    &fs::read(&evidence)
                        .map_err(|error| dope_kernel::error::io_error(&evidence, error))?,
                )?;
                let report = build_deep_campaign_report(&outcomes)?;
                dope_kernel::production::write_canonical(&out, &report)?;
                println!(
                    "{}",
                    serde_json::json!({"slices": report.slices.len(), "failures": report.failures.len(), "out": out})
                );
            },
        CampaignCommand::PlanValidationCert {
                state_dir,
                manifest,
                validation_manifest,
                out,
            } => {
                let cohort =
                    plan_validation_cert_cohort(&state_dir, &manifest, &validation_manifest)?;
                dope_kernel::production::write_canonical(&out, &cohort)?;
                println!(
                    "{}",
                    serde_json::json!({"kind": cohort.kind, "records": cohort.records.len(), "out": out})
                );
            },
        CampaignCommand::RunBaseline {
                cohort,
                out,
                candidate,
                auditor,
                shard_index,
                shard_count,
                primary_only,
            } => {
                let summary = run_baseline(
                    &cohort,
                    &out,
                    &candidate,
                    &auditor,
                    shard_index,
                    shard_count,
                    primary_only,
                )?;
                println!("{}", serde_json::to_string(&summary)?);
            },
        CampaignCommand::SummarizeEvidence { metrics, out } => {
                let summary = summarize_evidence(&metrics, &out)?;
                println!("{}", serde_json::to_string(&summary)?);
            },
        CampaignCommand::RunGoldMatrix {
                cohort,
                out,
                candidate,
                auditor,
                size_multiplier,
                generation_seed,
                auditor_seed,
                shard_index,
                shard_count,
                primary_only,
                blocks_only,
                neural_target_weight,
                neural_structural_penalty,
            } => {
                let summary = run_gold_matrix(&GoldMatrixOptions {
                    cohort_path: &cohort,
                    out: &out,
                    candidate_ids: &candidate,
                    auditor_ids: &auditor,
                    size_multipliers: &size_multiplier,
                    generation_seeds: &generation_seed,
                    auditor_seeds: &auditor_seed,
                    shard_index,
                    shard_count,
                    primary_only,
                    blocks_only,
                    neural_target_weight,
                    neural_structural_penalty,
                })?;
                println!("{}", serde_json::to_string(&summary)?);
            },
        _ => return Err(DopeError::Unsupported("campaign command routing mismatch".into())),
    };
    Ok(())
}

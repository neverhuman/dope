

/// Materializes validation-cert paths only after the single durable
/// authorization event. This command is intended to execute on xbabe3; the
/// resulting cohort never participates in router training.
pub fn plan_validation_cert_cohort(
    state_dir: &Path,
    manifest_path: &Path,
    validation_path: &Path,
) -> Result<CohortPlan> {
    require_host("xbabe3", "validation-cert cohort planning")?;
    let state = load_state(state_dir)?;
    if state.phase != CampaignPhase::ValidationCertOpen
        || state.validation_cert_open_count != 1
        || state.sealed_test_open_count != 0
    {
        return Err(DopeError::Data(
            "validation-cert cohort requires the one durable authorization event".into(),
        ));
    }
    let manifest: SplitManifest = read_json(manifest_path)?;
    validate_split_manifest(&manifest)?;
    let validation: ValidationSubmanifest = read_json(validation_path)?;
    validate_validation_submanifest(&validation)?;
    if validation.source_manifest_checksum != manifest.checksum
        || file_hashes(manifest_path)? != state.corpus_manifest
        || file_hashes(validation_path)? != state.validation_manifest
    {
        return Err(DopeError::Data(
            "validation-cert manifests differ from the frozen campaign".into(),
        ));
    }
    let broker = ValidationAccessBroker::new(&validation, AccessRole::FinalReleaseEvaluator, true)?;
    let assignments = validation
        .assignments
        .iter()
        .map(|assignment| (assignment.dataset_id.as_str(), assignment))
        .collect::<BTreeMap<_, _>>();
    let mut canonical = BTreeMap::<String, ([u8; 32], CohortRecord)>::new();
    for record in &manifest.datasets {
        let Some(assignment) = assignments.get(record.dataset_id.as_str()) else {
            continue;
        };
        if record.split != "validation" || assignment.partition != "validation-cert" {
            continue;
        }
        broker.authorize_lineage(&validation, &assignment.lineage_group_id)?;
        let features = record.cols.saturating_sub(1);
        let profile = StructuralProfile::from_shape(record.task, record.rows, features)?.id();
        let order = cohort_order(
            manifest.seed,
            &profile,
            &assignment.lineage_group_id,
            &record.dataset_id,
        );
        let value = CohortRecord {
            dataset_id: record.dataset_id.clone(),
            dataset_path: record.path.clone(),
            task: record.task.as_str().into(),
            rows: record.rows,
            features,
            lineage_group_id: assignment.lineage_group_id.clone(),
            structural_profile: profile,
            partition: "validation_cert".into(),
        };
        let entry = canonical
            .entry(assignment.lineage_group_id.clone())
            .or_insert_with(|| (order, value.clone()));
        if order < entry.0 {
            *entry = (order, value);
        }
    }
    let mut records = canonical
        .into_values()
        .map(|(_, record)| record)
        .collect::<Vec<_>>();
    records.sort_by(|left, right| {
        left.structural_profile
            .cmp(&right.structural_profile)
            .then_with(|| left.lineage_group_id.cmp(&right.lineage_group_id))
    });
    if records.len() != state.validation_cert_lineage_groups
        || records
            .iter()
            .map(|record| &record.lineage_group_id)
            .collect::<BTreeSet<_>>()
            .len()
            != state.validation_cert_lineage_groups
    {
        return Err(DopeError::Data(
            "canonical validation-cert cohort lost required lineages".into(),
        ));
    }
    Ok(CohortPlan {
        format: "dope-campaign-cohort".into(),
        version: 1,
        kind: "validation-cert".into(),
        records,
        exclusions: Vec::new(),
    })
}

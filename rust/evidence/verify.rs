pub fn verify_receipt(
    unverified: UnverifiedReceipt,
    root: &Path,
    policy: &VerificationPolicy,
) -> Result<VerifiedEvaluationReceipt> {
    let receipt = unverified.0;
    validate(&receipt, &schema(RUN_SCHEMA)?)?;
    let receipt_hash = digest(&receipt)?;
    require(
        is_digest(&policy.executor_hash)
            && is_digest(&policy.receipt_hash)
            && policy.freeze_hash.as_deref().is_none_or(is_digest),
        "invalid_trust_pin",
    )?;
    require(
        receipt_hash == policy.receipt_hash && receipt["identity"] == policy.identity.0,
        "receipt_identity_mismatch",
    )?;
    require(
        receipt["executor_hash"] == policy.executor_hash
            && receipt["freeze_hash"].as_str() == policy.freeze_hash.as_deref(),
        "executor_or_freeze_mismatch",
    )?;
    require(receipt["execution"] == "complete", "evaluation_incomplete")?;
    require(
        receipt["projection"] == "private",
        "private_evidence_required",
    )?;
    let identity = &receipt["identity"];
    let produced = uint(&identity["produced_rows"])?;
    let consumed = uint(&identity["consumed_rows"])?;
    require(
        produced == uint(&identity["requested_rows"])?
            && consumed > 0
            && consumed <= produced
            && uint(&identity["context_rows"])? <= consumed,
        "row_count_mismatch",
    )?;
    let mut nodes = BTreeMap::new();
    let mut paths = BTreeSet::new();
    for node in receipt["dependencies"]
        .as_array()
        .ok_or_else(|| error("schema_mismatch"))?
    {
        require(
            nodes.insert(text(&node["sha256"])?, node).is_none()
                && paths.insert(text(&node["path"])?),
            "dependency_alias",
        )?;
    }
    let mut roots = BTreeSet::new();
    for (field, value) in identity
        .as_object()
        .ok_or_else(|| error("schema_mismatch"))?
    {
        if field.ends_with("_hash") {
            roots.insert(text(value)?);
        }
    }
    roots.insert(text(&receipt["executor_hash"])?);
    roots.insert(text(&receipt["paired_evidence_hash"])?);
    if let Some(f) = receipt["freeze_hash"].as_str() {
        roots.insert(f);
    }
    let mut visited = BTreeSet::new();
    let mut active = BTreeSet::new();
    for key in roots {
        visit(key, &nodes, &mut visited, &mut active)?;
    }
    require(visited.len() == nodes.len(), "unbound_dependency")?;
    let mut manifests = BTreeMap::new();
    for (key, node) in &nodes {
        let manifest = matches!(
            text(&node["kind"])?,
            "protocol" | "split" | "fit" | "artifact_inventory" | "paired_evidence" | "freeze"
        );
        let (actual, size, data) = read_file(root, text(&node["path"])?, manifest)?;
        require(
            actual == *key && size == uint(&node["bytes"])?,
            "dependency_bytes_mismatch",
        )?;
        if manifest {
            manifests.insert(*key, parse_json(&data)?);
        }
    }
    for (field, key) in identity
        .as_object()
        .ok_or_else(|| error("schema_mismatch"))?
    {
        if let Some(kind) = field.strip_suffix("_hash") {
            require(
                nodes[text(key)?]["kind"] == kind,
                "dependency_kind_mismatch",
            )?;
        }
    }
    let executor = text(&receipt["executor_hash"])?;
    let paired_key = text(&receipt["paired_evidence_hash"])?;
    require(
        nodes[executor]["kind"] == "executor" && nodes[paired_key]["kind"] == "paired_evidence",
        "dependency_kind_mismatch",
    )?;
    let protocol_key = text(&identity["protocol_hash"])?;
    let split_key = text(&identity["split_hash"])?;
    let split = &manifests[split_key];
    closed(
        split,
        &[
            "format",
            "version",
            "data_hash",
            "query_hash",
            "partition_role",
            "fit_rows",
            "query_rows",
        ],
    )?;
    require(
        version(split, "dope-split-manifest")
            && [
                "data_hash",
                "query_hash",
                "partition_role",
                "fit_rows",
                "query_rows",
            ]
            .iter()
            .all(|k| split[k] == identity[k]),
        "split_identity_mismatch",
    )?;
    require(
        children(nodes[split_key])?
            == BTreeSet::from([
                text(&identity["data_hash"])?,
                text(&identity["query_hash"])?,
            ]),
        "split_dependency_mismatch",
    )?;
    validate_behavior_contract(&manifests[protocol_key])?;
    let protocol = &manifests[protocol_key];
    for (declared, actual) in [
        ("table_schema_hash", "schema_hash"),
        ("task_manifest_hash", "task_hash"),
        ("population_history_hash", "source_family_hash"),
    ] {
        require(
            protocol["workload"][declared].is_null()
                || protocol["workload"][declared] == identity[actual],
            "protocol_workload_mismatch",
        )?;
    }
    let mut commitments = BTreeSet::new();
    for section in ["freeze_requirements", "workload", "disclosure"] {
        for (field, value) in protocol[section]
            .as_object()
            .ok_or_else(|| error("schema_mismatch"))?
        {
            if (section == "freeze_requirements" || field.ends_with("_hash")) && !value.is_null() {
                commitments.insert(text(value)?);
            }
        }
    }
    require(
        commitments.is_subset(&children(nodes[protocol_key])?),
        "protocol_commitment_unbound",
    )?;
    let inventory_key = text(&identity["artifact_inventory_hash"])?;
    let inventory = &manifests[inventory_key];
    closed(inventory, &["format", "version", "files"])?;
    require(
        version(inventory, "dope-artifact-inventory"),
        "invalid_artifact_inventory",
    )?;
    let files: Vec<&str> = inventory["files"]
        .as_array()
        .ok_or_else(|| error("invalid_artifact_inventory"))?
        .iter()
        .map(text)
        .collect::<Result<_>>()?;
    let artifacts: BTreeSet<&str> = nodes
        .iter()
        .filter(|(_, n)| n["kind"] == "artifact")
        .map(|(k, _)| *k)
        .collect();
    require(
        !artifacts.is_empty()
            && files.len() == artifacts.len()
            && files.iter().copied().collect::<BTreeSet<_>>() == artifacts
            && children(nodes[inventory_key])? == artifacts,
        "artifact_inventory_mismatch",
    )?;
    let bytes = artifacts.iter().try_fold(0_u64, |s, k| {
        s.checked_add(uint(&nodes[k]["bytes"])?)
            .ok_or_else(|| error("sampling_byte_sum_mismatch"))
    })?;
    require(
        bytes == uint(&receipt["sampling_payload_bytes"])?,
        "sampling_byte_sum_mismatch",
    )?;
    let fit_key = text(&identity["fit_hash"])?;
    let fit = &manifests[fit_key];
    let mut fit_keys = vec!["format", "version", "state_hash"];
    fit_keys.extend(FIT_FIELDS);
    closed(fit, &fit_keys)?;
    require(
        version(fit, "dope-generator-fit") && FIT_FIELDS.iter().all(|k| fit[k] == identity[k]),
        "fit_identity_mismatch",
    )?;
    let state_key = text(&fit["state_hash"])?;
    let mut fit_requires = BTreeSet::from([state_key]);
    for field in FIT_FIELDS {
        if field.ends_with("_hash") {
            fit_requires.insert(text(&fit[field])?);
        }
    }
    require(
        nodes
            .get(state_key)
            .is_some_and(|n| n["kind"] == "fit_state")
            && children(nodes[fit_key])? == fit_requires,
        "fit_state_unbound",
    )?;
    let paired = &manifests[paired_key];
    closed(
        paired,
        &[
            "format",
            "version",
            "query_hash",
            "metric_hash",
            "loss",
            "loss_space",
            "loss_unit",
            "real",
            "synthetic",
            "null",
        ],
    )?;
    require(
        version(paired, "dope-paired-losses")
            && ["loss", "loss_space", "loss_unit"]
                .iter()
                .all(|k| paired[k] == receipt[k])
            && ["query_hash", "metric_hash"]
                .iter()
                .all(|k| paired[k] == identity[k]),
        "paired_identity_mismatch",
    )?;
    require(
        children(nodes[paired_key])?
            == BTreeSet::from([
                text(&identity["query_hash"])?,
                text(&identity["metric_hash"])?,
            ]),
        "paired_dependency_mismatch",
    )?;
    for label in ["real", "synthetic", "null"] {
        let losses = paired[label]
            .as_array()
            .ok_or_else(|| error("invalid_paired_losses"))?;
        require(
            losses.len() as u64 == uint(&identity["query_rows"])?,
            "invalid_paired_losses",
        )?;
        let mut mean = 0.0;
        let mut compensation = 0.0;
        for loss in losses {
            let x = loss
                .as_f64()
                .ok_or_else(|| error("invalid_paired_losses"))?;
            require(x.is_finite() && x >= 0.0, "invalid_paired_losses")?;
            let term = x / losses.len() as f64 - compensation;
            let sum = mean + term;
            compensation = (sum - mean) - term;
            mean = sum;
        }
        let supplied = receipt["losses"][label]
            .as_f64()
            .ok_or_else(|| error("invalid_paired_losses"))?;
        require(
            mean.is_finite()
                && (mean - supplied).abs() <= 1e-12_f64.max(1e-12 * mean.abs().max(supplied.abs())),
            "paired_loss_sum_mismatch",
        )?;
    }
    if let Some(key) = receipt["freeze_hash"].as_str() {
        require(nodes[key]["kind"] == "freeze", "dependency_kind_mismatch")?;
        let freeze = &manifests[key];
        validate(freeze, &schema(FREEZE_SCHEMA)?)?;
        require(
            [
                "protocol_hash",
                "artifact_inventory_hash",
                "configuration_hash",
                "split_hash",
                "evaluation_code_hash",
            ]
            .iter()
            .all(|k| freeze[k] == identity[k])
                && freeze["executor_hash"] == policy.executor_hash
                && freeze["cell_hash"] == scientific_cache_key(&policy.identity)?,
            "frozen_selection_mismatch",
        )?;
        for (field, kind) in [
            ("selection_receipt_hash", "selection_receipt"),
            ("access_ledger_hash", "access_ledger"),
        ] {
            let dependency = text(&freeze[field])?;
            require(
                nodes.get(dependency).is_some_and(|n| n["kind"] == kind)
                    && children(nodes[key])?.contains(dependency),
                "freeze_dependency_missing",
            )?;
        }
        if let Some(parent) = freeze["parent_freeze_hash"].as_str() {
            require(
                nodes.get(parent).is_some_and(|n| n["kind"] == "freeze")
                    && children(nodes[key])?.contains(parent),
                "freeze_parent_missing",
            )?;
        }
        require(
            manifests[protocol_key]["protocol_frozen"] == true,
            "protocol_unfrozen",
        )?;
    }
    require(
        identity["partition_role"] != "confirmation" || receipt["freeze_hash"].is_string(),
        "confirmation_freeze_required",
    )?;
    let identity = EvaluationIdentity::new(identity.clone())?;
    Ok(VerifiedEvaluationReceipt {
        receipt,
        identity,
        receipt_hash,
    })
}

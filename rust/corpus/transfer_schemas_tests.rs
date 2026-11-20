use super::*;

fn hex64(fill: u8) -> String {
    format!("{fill:02x}").repeat(32)
}

fn hex40(fill: u8) -> String {
    format!("{fill:02x}").repeat(20)
}

fn valid_view() -> TrainingViewV1 {
    TrainingViewV1 {
        schema: SCHEMA_VIEW.into(),
        view_id: "view-alpha-01".into(),
        base_dataset_id: "base-0001".into(),
        partition_source_id: "part-seed-0001".into(),
        target_declaration_sha256: hex64(0x10),
        task_context_sha256: hex64(0x11),
        metric_and_budget_sha256: hex64(0x12),
        fit_row_commitment: hex64(0x13),
        gradient_query_row_commitment: hex64(0x14),
        physical_membership_disjoint: true,
        normalizer_sha256: hex64(0x15),
        feature_group_map_sha256: hex64(0x16),
        fit_scope_sha256: hex64(0x17),
        observed_mask: 0b0111,
        consumed_measured_mask: 0b0011,
        predicted_mask: 0b1000,
        predictor_sha256: Some(hex64(0x18)),
        task_weight: 1.0,
    }
}

fn valid_receipt() -> TransferExperimentReceiptV1 {
    TransferExperimentReceiptV1 {
        schema: SCHEMA_EXPERIMENT.into(),
        experiment_id: "exp-transfer-001".into(),
        status: "completed".into(),
        protocol: "budgeted_search".into(),
        exposure_class: "unseen_source_family".into(),
        native_qualification: "PASS".into(),
        scientific_verdict: "PROMOTE".into(),
        dope_commit: hex40(0xaa),
        jopedime_commit: hex40(0xbb),
        graph_config_sha256: hex64(0x21),
        model_weights_sha256: hex64(0x22),
        checkpoint_bundle_manifest_sha256: hex64(0x23),
        parameter_inventory_sha256: hex64(0x24),
        initialization_lineage_sha256: hex64(0x25),
        corpus_root_sha256: hex64(0x26),
        committed_exposure_ledger_sha256: hex64(0x27),
        objective_recipe_sha256: hex64(0x28),
        teacher_panel_sha256: hex64(0x29),
        preprocessing_sha256: hex64(0x2a),
        projector_and_consumer_sha256: hex64(0x2b),
        preregistration_sha256: hex64(0x2c),
        frozen_before_sha256: hex64(0x2d),
        frozen_after_sha256: hex64(0x2d),
        resource_receipt_sha256: hex64(0x2e),
        per_task_results_sha256: Some(hex64(0x2f)),
        accepted_base_count: 1000,
        committed_unique_base_count: 950,
    }
}

#[test]
fn training_view_accepts_measured_and_predicted_variants() {
    let view = valid_view();
    validate_training_view(&view).unwrap();
    let encoded = serde_json::to_vec(&view).unwrap();
    assert_eq!(parse_training_view_json(&encoded).unwrap(), view);

    let mut no_predicted = valid_view();
    no_predicted.predicted_mask = 0;
    no_predicted.predictor_sha256 = None;
    validate_training_view(&no_predicted).unwrap();
}

#[test]
fn training_view_rejects_fit_query_collision_and_partition_alias() {
    let mut non_disjoint = valid_view();
    non_disjoint.physical_membership_disjoint = false;
    assert!(
        validate_training_view(&non_disjoint)
            .unwrap_err()
            .to_string()
            .contains("transfer_view_disjoint")
    );

    let mut same_commitment = valid_view();
    same_commitment.gradient_query_row_commitment = same_commitment.fit_row_commitment.clone();
    assert!(
        validate_training_view(&same_commitment)
            .unwrap_err()
            .to_string()
            .contains("transfer_view_disjoint")
    );

    let mut partition_alias = valid_view();
    partition_alias.partition_source_id = partition_alias.base_dataset_id.clone();
    assert!(
        validate_training_view(&partition_alias)
            .unwrap_err()
            .to_string()
            .contains("transfer_view_partition_source")
    );

    let mut empty_partition = valid_view();
    empty_partition.partition_source_id.clear();
    assert!(
        validate_training_view(&empty_partition)
            .unwrap_err()
            .to_string()
            .contains("transfer_view_partition_source")
    );
}

#[test]
fn training_view_rejects_landmark_mask_and_predictor_violations() {
    let mut overflow = valid_view();
    overflow.observed_mask = LANDMARK_MASK_LIMIT;
    assert!(
        validate_training_view(&overflow)
            .unwrap_err()
            .to_string()
            .contains("transfer_view_landmark_mask")
    );

    let mut unmeasured = valid_view();
    unmeasured.observed_mask = 0b0010;
    unmeasured.consumed_measured_mask = 0b0011;
    assert!(
        validate_training_view(&unmeasured)
            .unwrap_err()
            .to_string()
            .contains("transfer_view_landmark_subset")
    );

    let mut overlap = valid_view();
    overlap.consumed_measured_mask = 0b0011;
    overlap.predicted_mask = 0b0010;
    assert!(
        validate_training_view(&overlap)
            .unwrap_err()
            .to_string()
            .contains("transfer_view_landmark_disjoint")
    );

    let mut missing_predictor = valid_view();
    missing_predictor.predicted_mask = 0b1000;
    missing_predictor.predictor_sha256 = None;
    assert!(
        validate_training_view(&missing_predictor)
            .unwrap_err()
            .to_string()
            .contains("transfer_view_predictor_presence")
    );

    let mut orphan_predictor = valid_view();
    orphan_predictor.predicted_mask = 0;
    orphan_predictor.predictor_sha256 = Some(hex64(0x44));
    assert!(
        validate_training_view(&orphan_predictor)
            .unwrap_err()
            .to_string()
            .contains("transfer_view_predictor_presence")
    );
}

#[test]
fn training_view_rejects_bad_digests_weights_and_echoes_nothing() {
    let mut bad_digest = valid_view();
    bad_digest.normalizer_sha256 = "A".repeat(64);
    let err = validate_training_view(&bad_digest).unwrap_err().to_string();
    assert!(err.contains("transfer_view_digest"));
    assert!(!err.contains("AAAA"));

    for bad_weight in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut weighted = valid_view();
        weighted.task_weight = bad_weight;
        assert!(
            validate_training_view(&weighted)
                .unwrap_err()
                .to_string()
                .contains("transfer_view_weight")
        );
    }

    let raw_err = parse_training_view_json(br#"{"secret_column_header": 9999}"#)
        .unwrap_err()
        .to_string();
    assert!(raw_err.contains("transfer_view_schema"));
    assert!(!raw_err.contains("secret_column_header"));
}

#[test]
fn experiment_receipt_accepts_valid_promote_and_pending_states() {
    let receipt = valid_receipt();
    validate_experiment_receipt(&receipt).unwrap();
    let encoded = serde_json::to_vec(&receipt).unwrap();
    assert_eq!(parse_experiment_receipt_json(&encoded).unwrap(), receipt);

    let mut pending = valid_receipt();
    pending.status = "running".into();
    pending.exposure_class = "in_corpus".into();
    pending.native_qualification = "NOT_RUN".into();
    pending.scientific_verdict = "PENDING".into();
    pending.per_task_results_sha256 = None;
    validate_experiment_receipt(&pending).unwrap();
}

#[test]
fn experiment_receipt_rejects_frozen_drift_base_overflow_and_illegal_promote() {
    let mut drifted = valid_receipt();
    drifted.frozen_after_sha256 = hex64(0x99);
    assert!(
        validate_experiment_receipt(&drifted)
            .unwrap_err()
            .to_string()
            .contains("transfer_experiment_frozen_drift")
    );

    let mut over_committed = valid_receipt();
    over_committed.committed_unique_base_count = over_committed.accepted_base_count + 1;
    assert!(
        validate_experiment_receipt(&over_committed)
            .unwrap_err()
            .to_string()
            .contains("transfer_experiment_base_count")
    );

    for bad_exposure in ["in_corpus", "recall"] {
        let mut exposed = valid_receipt();
        exposed.exposure_class = bad_exposure.into();
        assert!(
            validate_experiment_receipt(&exposed)
                .unwrap_err()
                .to_string()
                .contains("transfer_experiment_promote_gate")
        );
    }

    for bad_qual in ["FAIL", "NOT_RUN"] {
        let mut unqualified = valid_receipt();
        unqualified.native_qualification = bad_qual.into();
        assert!(
            validate_experiment_receipt(&unqualified)
                .unwrap_err()
                .to_string()
                .contains("transfer_experiment_promote_gate")
        );
    }

    for bad_status in ["running", "failed", "stopped"] {
        let mut incomplete = valid_receipt();
        incomplete.status = bad_status.into();
        assert!(
            validate_experiment_receipt(&incomplete)
                .unwrap_err()
                .to_string()
                .contains("transfer_experiment_promote_gate")
        );
    }

    let mut missing_results = valid_receipt();
    missing_results.per_task_results_sha256 = None;
    assert!(
        validate_experiment_receipt(&missing_results)
            .unwrap_err()
            .to_string()
            .contains("transfer_experiment_promote_gate")
    );
}

#[test]
fn experiment_receipt_rejects_unknown_enums_and_malformed_commits() {
    let mut bad_proto = valid_receipt();
    bad_proto.protocol = "unbounded_grid".into();
    assert!(
        validate_experiment_receipt(&bad_proto)
            .unwrap_err()
            .to_string()
            .contains("transfer_experiment_protocol")
    );

    let mut bad_commit = valid_receipt();
    bad_commit.dope_commit = "abcd1234".into();
    assert!(
        validate_experiment_receipt(&bad_commit)
            .unwrap_err()
            .to_string()
            .contains("transfer_experiment_commit")
    );

    let raw_err = parse_experiment_receipt_json(br#"{"private_dataset_name": "secret"}"#)
        .unwrap_err()
        .to_string();
    assert!(raw_err.contains("transfer_experiment_schema"));
    assert!(!raw_err.contains("private_dataset_name"));
}

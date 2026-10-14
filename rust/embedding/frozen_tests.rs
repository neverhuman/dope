fn frozen_root(label: &str) -> std::path::PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/frozen-embedding-fixtures")
        .join(format!(
            "{label}-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
    fs::create_dir_all(&root).unwrap();
    root
}

#[test]
fn frozen_ranges_and_target_encoding_ignore_query_extrema_and_column_order() {
    let root = frozen_root("ranges");
    let fit = root.join("fit.csv");
    let query = root.join("query.csv");
    let augmented = root.join("augmented.csv");
    fs::write(&fit, "private_feature,private_target\n0,100\n10,200\n").unwrap();
    fs::write(&query, "private_target,private_feature\n150,5\n").unwrap();
    fs::write(
        &augmented,
        "private_feature,private_target\n5,150\n1000,9999\n",
    )
    .unwrap();
    let frozen =
        FrozenNumericTransform::fit_csv(&fit, TargetSelector::Index(1), Task::Regression).unwrap();
    let one = frozen.apply_csv(&query).unwrap();
    let many = frozen.apply_csv(&augmented).unwrap();
    assert_eq!(one.table.columns[0], [0.5]);
    assert_eq!(one.table.target, [0.5]);
    assert_eq!(many.table.columns[0], [0.5, 1.0]);
    assert_eq!(many.table.target, [0.5, 1.0]);
    // The historical loader refits; it remains unchanged and is deliberately
    // distinct from the new frozen-reference API.
    let historical =
        load_numeric_csv(&augmented, TargetSelector::Index(1), Task::Regression).unwrap();
    assert_eq!(historical.table.columns[0][0], 0.0);
    let public = frozen.descriptor().to_string();
    assert!(!public.contains("private_feature"));
    assert!(!public.contains("private_target"));
    assert!(!public.contains("minimum"));
    assert!(!public.contains("maximum"));
    assert!(public.contains("reference_minmax_clamped_v1"));
    let id = frozen.descriptor()["state_digest"].clone();
    fs::write(&fit, "private_feature,private_target\n0,100\n12,200\n").unwrap();
    let changed =
        FrozenNumericTransform::fit_csv(&fit, TargetSelector::Index(1), Task::Regression).unwrap();
    assert_ne!(id, changed.descriptor()["state_digest"]);
}

#[test]
fn frozen_binary_codes_and_refusals_do_not_echo_source_values_or_headers() {
    let root = frozen_root("binary");
    let fit = root.join("fit.csv");
    let query = root.join("query.csv");
    fs::write(&fit, "x,y\n0,-7\n10,42\n").unwrap();
    let frozen =
        FrozenNumericTransform::fit_csv(&fit, TargetSelector::Name("y"), Task::Binary).unwrap();
    fs::write(&query, "x,y\n5,42\n5,-7\n").unwrap();
    assert_eq!(frozen.apply_csv(&query).unwrap().table.target, [1.0, 0.0]);
    for (csv, code) in [
        ("x,y\n5,999999\n", "frozen_binary_label_outside_reference"),
        ("private_header,y\n5,42\n", "frozen_input_schema_mismatch"),
        ("x,y\nprivate_value,42\n", "frozen_numeric_input_invalid"),
        ("x,y\n5,\n", "frozen_target_missing"),
    ] {
        fs::write(&query, csv).unwrap();
        let error = match frozen.apply_csv(&query) {
            Err(error) => error,
            Ok(_) => panic!("invalid fixture admitted"),
        };
        assert_eq!(error.to_string(), format!("invalid data: {code}"));
    }
}

#[test]
fn frozen_action_basis_metadata_and_predictions_use_actual_bound_router() {
    let root = frozen_root("action");
    let (bundle, evidence, _) = write_router(&root, 128);
    let fit = root.join("fit.csv");
    let reordered = root.join("reordered.csv");
    fs::write(&fit, "x,y\n0,100\n5,150\n10,200\n").unwrap();
    fs::write(&reordered, "y,x\n100,0\n150,5\n200,10\n").unwrap();
    let transform =
        FrozenNumericTransform::fit_csv(&fit, TargetSelector::Name("y"), Task::Regression).unwrap();
    let embedder = DatasetEmbedder::load(&bundle, &evidence).unwrap();
    let first = embedder
        .embed_csv_frozen(&fit, &transform, MetadataMode::Public)
        .unwrap();
    let second = embedder
        .embed_csv_frozen(&reordered, &transform, MetadataMode::Public)
        .unwrap();
    assert_eq!(first.embedding.vector, second.embedding.vector);
    assert_eq!(first.basis_sha256, second.basis_sha256);
    assert_eq!(first.embedding.predictions, second.embedding.predictions);
    assert_eq!(first.embedding.version, ACTION_EMBEDDING_VERSION);
    assert_eq!(first.embedding.dimension, ACTION_EMBEDDING_DIMENSION);
    assert_eq!(
        first.capability.checkpoint_sha256.as_ref(),
        Some(&first.embedding.router_bundle_sha256)
    );
    assert_eq!(
        first.capability.evidence_sha256.as_ref(),
        Some(&first.embedding.router_evidence_sha256)
    );
    assert_eq!(first.capability.width, Some(4168));
    assert_eq!(first.capability.context_rows, Some(3));
    assert_eq!(first.capability.target_included, Some(true));
    assert!(!first.capability.qualifies_transfer);
    assert!(first
        .embedding
        .normalization
        .features
        .iter()
        .all(|r| r.minimum.is_none() && r.maximum.is_none()));
}

#[test]
fn missing_official_activation_preserves_declared_prediction_capability() {
    let capability = EncoderCapability::unavailable(
        "fixture_official_encoder".into(),
        GeometrySpace::RowActivation,
        CapabilityAvailability::Available,
        EmbeddingUnavailableReason::OfficialActivationNotExposed,
    )
    .unwrap();
    assert_eq!(capability.predictive, CapabilityAvailability::Available);
    assert_eq!(capability.embedding, CapabilityAvailability::Unavailable);
    assert_eq!(capability.width, None);
    assert_eq!(capability.layer, None);
    assert_eq!(capability.checkpoint_sha256, None);
    assert!(!capability.qualifies_transfer);
    assert!(!capability.qualifies_privacy);
}

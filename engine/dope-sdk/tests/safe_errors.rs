use dope_sdk::v0::{
    prepare::{prepare_table_mixed_with_policy, ClassVocabPolicy, ColumnKind, PrepTask},
    DopeError,
};

#[test]
fn malformed_weights_return_stable_artifact_error_without_echoing_input() {
    let canary = b"d03-private-weight-canary-29b4";
    let error = dope_sdk::v0::mlc_v3::MlcV3Model::from_safetensors_bytes(canary)
        .err()
        .expect("malformed safetensors bytes were accepted");

    assert!(matches!(&error, DopeError::Artifact(_)));
    let display = error.to_string();
    assert_eq!(display, "artifact: mlc_v3 safetensors deserialize failed");
    assert!(!display.contains(std::str::from_utf8(canary).unwrap()));
}

#[test]
fn mismatched_preparation_schema_returns_stable_input_error_without_echoing_cells() {
    let canary = "d03-private-schema-canary-74a1";
    let x = vec![
        vec![canary.to_owned()],
        vec!["ordinary-category".to_owned()],
    ];
    let y = [0.0, 1.0];
    let no_declared_columns: [ColumnKind; 0] = [];

    let error = prepare_table_mixed_with_policy(
        &x,
        &y,
        Some(&no_declared_columns),
        PrepTask::Regression,
        None,
        ClassVocabPolicy::Cap32V1,
    )
    .expect_err("schema with the wrong declared column count was accepted");

    assert!(matches!(&error, DopeError::Input(_)));
    let display = error.to_string();
    assert_eq!(display, "input: prepare_kinds_length_mismatch");
    assert!(!display.contains(canary));
}

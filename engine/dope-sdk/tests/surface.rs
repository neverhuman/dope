use dope_sdk::v0::prepare::{prepare_table_with_ids, PrepTask};

#[test]
fn facade_preserves_preparation_and_physical_source_identity() {
    let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, (i % 3) as f64]).collect();
    let y: Vec<f64> = (0..20).map(|i| (i * i) as f64).collect();
    let ids: Vec<u64> = (100..120).collect();
    let upstream =
        dope_core::prepare::prepare_table_with_ids(&x, &y, PrepTask::Regression, ids.clone())
            .unwrap();
    let sdk = prepare_table_with_ids(&x, &y, PrepTask::Regression, ids).unwrap();
    assert_eq!(sdk.x_scaled, upstream.x_scaled);
    assert_eq!(sdk.y_scaled, upstream.y_scaled);
    assert_eq!(sdk.source_rows_digest(), upstream.source_rows_digest());
    let other = prepare_table_with_ids(&x, &y, PrepTask::Regression, (200..220).collect()).unwrap();
    assert_ne!(sdk.source_rows_digest(), other.source_rows_digest());
}

#[test]
fn facade_refuses_duplicate_rows_and_target_shape_without_source_echo() {
    let x = vec![vec![1234567.0]; 20];
    let y = vec![0.0; 20];
    let err = prepare_table_with_ids(&x, &y, PrepTask::Regression, vec![7; 20])
        .err()
        .unwrap();
    assert!(!err.to_string().contains("1234567"));
    assert!(prepare_table_with_ids(&x, &y[..19], PrepTask::Regression, (0..20).collect()).is_err());
}

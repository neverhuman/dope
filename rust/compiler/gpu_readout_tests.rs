use super::*;

#[test]
fn gpu_conditional_readout_preserves_small_target_variance() {
    let rows = 384;
    let features: Vec<_> = (0..rows).map(|row| row as f32 / rows as f32).collect();
    let target: Vec<_> = features.iter().map(|value| 0.54 + 0.02 * value).collect();
    let table = Table::from_arrays(&features, &target, rows, 1, Task::Regression).unwrap();
    let fit = || fit_compact_neural_target(&table, &table.columns, &[(0, 1.0)], 11).unwrap();
    let model = fit();
    assert_eq!(
        serde_json::to_value(&model).unwrap(),
        serde_json::to_value(fit()).unwrap()
    );
    let mse = (0..100)
        .map(|index| {
            let x = (index as f32 + 0.5) / 100.0;
            let (prediction, logistic) = evaluate_target(&model, &[x]);
            assert!(!logistic);
            (prediction - (0.54 + 0.02 * x)).powi(2)
        })
        .sum::<f32>()
        / 100.0;
    assert!(
        mse < 1e-7,
        "GPU readout lost the small target signal: {mse}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[cfg(feature = "gpu-training")]
    static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn fixtures(task: Task) -> (Table, Table) {
        let rows = 64;
        let features = (0..rows)
            .flat_map(|row| {
                let x = row as f32 / (rows - 1) as f32;
                [x, 1.0 - x]
            })
            .collect::<Vec<_>>();
        let target = (0..rows)
            .map(|row| {
                let x = row as f32 / (rows - 1) as f32;
                if task == Task::Binary {
                    f32::from(x > 0.5)
                } else {
                    (0.8 * x + 0.1).clamp(0.0, 1.0)
                }
            })
            .collect::<Vec<_>>();
        let table = Table::from_arrays(&features, &target, rows, 2, task).unwrap();
        (table.clone(), table)
    }

    #[test]
    fn native_auditors_are_deterministic_and_beat_null_fixtures() {
        for task in [Task::Binary, Task::Regression] {
            let (train, test) = fixtures(task);
            let mean = train
                .target
                .iter()
                .map(|&value| f64::from(value))
                .sum::<f64>()
                / train.rows as f64;
            let null = loss(task, &test.target, &vec![mean; test.rows]).unwrap();
            for id in [
                "elastic_net_glm",
                "ga2m",
                "extremely_randomized_trees",
                "histogram_gbdt",
            ] {
                let backend = auditor_backend(id).unwrap();
                let first = backend.predict(&train, &test, task, 57721).unwrap();
                assert_eq!(first, backend.predict(&train, &test, task, 57721).unwrap());
                let paired = backend
                    .predict_two(&train, &test, &test, task, 57721)
                    .unwrap();
                assert_eq!(paired.0, paired.1, "{id} paired prediction mismatch");
                assert_eq!(first, paired.0, "{id} paired fit changed output");
                assert!(
                    loss(task, &test.target, &first).unwrap() < null,
                    "{id} {task:?}"
                );
                assert_eq!(implementation_hash(id).len(), 64);
            }
        }
    }

    struct CountingAuditor(AtomicUsize);

    impl AuditorBackend for CountingAuditor {
        fn id(&self) -> &'static str {
            "counting_fixture"
        }

        fn predict(
            &self,
            _train: &Table,
            test: &Table,
            _task: Task,
            _seed: u64,
        ) -> Result<Vec<f64>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(test.columns[0]
                .iter()
                .map(|value| f64::from(*value))
                .collect())
        }
    }

    #[test]
    fn permutation_importance_fits_once_for_all_holdouts() {
        let (train, test) = fixtures(Task::Regression);
        let auditor = CountingAuditor(AtomicUsize::new(0));
        let first = auditor
            .permutation_importance(&train, &test, Task::Regression, 57_721)
            .unwrap();
        assert_eq!(auditor.0.load(Ordering::SeqCst), 1);
        assert_eq!(first.feature_indices, vec![0, 1]);
        assert_eq!(first.importances.len(), 2);
        assert!(first.importances[0] > first.importances[1]);
        let second = auditor
            .permutation_importance(&train, &test, Task::Regression, 57_721)
            .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn importance_comparison_preserves_undefined_spearman() {
        let evidence = |importances: Vec<f64>| PermutationImportance {
            feature_indices: vec![0, 1, 2],
            baseline_predictions: vec![0.5],
            baseline_loss: 0.25,
            importances,
        };
        let informative = compare_permutation_importance(
            &evidence(vec![3.0, 2.0, 1.0]),
            &evidence(vec![6.0, 4.0, 2.0]),
        )
        .unwrap();
        assert_eq!(informative.spearman, Some(1.0));
        assert_eq!(informative.top_k, 1);
        assert_eq!(informative.top_k_agreement, 1.0);
        assert!(
            (informative
                .real_normalized_shares
                .iter()
                .map(|(_, share)| share)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-12
        );
        let changed_driver = compare_permutation_importance(
            &evidence(vec![3.0, 2.0, 1.0]),
            &evidence(vec![1.0, 2.0, 3.0]),
        )
        .unwrap();
        assert_eq!(changed_driver.top_k_agreement, 0.0);
        assert_eq!(changed_driver.spearman, Some(-1.0));

        let undefined = compare_permutation_importance(
            &evidence(vec![1.0, 1.0, 1.0]),
            &evidence(vec![2.0, 2.0, 2.0]),
        )
        .unwrap();
        assert_eq!(undefined.informative_feature_count, 3);
        assert_eq!(undefined.spearman, None);
    }

    #[test]
    fn importance_covers_high_index_features() {
        struct HighIndex;
        impl AuditorBackend for HighIndex {
            fn id(&self) -> &'static str {
                "high_index_fixture"
            }
            fn predict(
                &self,
                _train: &Table,
                test: &Table,
                _task: Task,
                _seed: u64,
            ) -> Result<Vec<f64>> {
                Ok(test.columns[69]
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect())
            }
        }
        let rows = 32;
        let mut values = vec![0.0; rows * 70];
        let mut target = Vec::new();
        for row in 0..rows {
            let value = row as f32 / (rows - 1) as f32;
            values[row * 70 + 69] = value;
            target.push(value);
        }
        let table = Table::from_arrays(&values, &target, rows, 70, Task::Regression).unwrap();
        let auditor = BoundedAuditor {
            inner: Box::new(HighIndex),
        };
        let importance = auditor
            .permutation_importance(&table, &table, Task::Regression, 57721)
            .unwrap();
        assert_eq!(importance.feature_indices.len(), 70);
        assert_eq!(importance.feature_indices[69], 69);
        assert!(importance.importances[69] > importance.importances[0]);
    }

    #[cfg(feature = "gpu-training")]
    #[test]
    fn gpu_feature_preprocessing_is_deterministic_and_bounded() {
        let rows = 8;
        let features = 300;
        let values = (0..rows * features)
            .map(|index| ((index * 37) % 101) as f32 / 100.0)
            .collect::<Vec<_>>();
        let target = (0..rows)
            .map(|row| row as f32 / (rows - 1) as f32)
            .collect::<Vec<_>>();
        let table = Table::from_arrays(&values, &target, rows, features, Task::Regression).unwrap();
        let first = bounded_auditor_tables(&table, &[&table], 57721, 256, 64);
        let second = bounded_auditor_tables(&table, &[&table], 57721, 256, 64);
        assert_eq!(first.0.features, 64);
        assert_eq!(first.0.columns, second.0.columns);
        assert_eq!(first.0.columns, first.1[0].columns);

        let tall_rows = 1_100;
        let tall_values = (0..tall_rows * 2)
            .map(|index| (index % 97) as f32 / 96.0)
            .collect::<Vec<_>>();
        let tall_target = (0..tall_rows)
            .map(|row| (row % 101) as f32 / 100.0)
            .collect::<Vec<_>>();
        let tall =
            Table::from_arrays(&tall_values, &tall_target, tall_rows, 2, Task::Regression).unwrap();
        let bounded = bounded_auditor_tables(&tall, &[&tall], 57721, 256, 64);
        assert_eq!(bounded.0.rows, 256);
        assert_eq!(bounded.1[0].rows, tall_rows);
    }

    #[cfg(feature = "gpu-training")]
    #[test]
    fn gpu_row_batching_preserves_order_and_values() {
        let _guard = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let device = gpu_device().unwrap();
        let rows = tch::Tensor::arange(1025, (tch::Kind::Float, device)).view([205, 5]);
        let expected = &rows * 1.25 + 3.0;
        let actual = transform_row_batches(&rows, 17, |batch| batch * 1.25 + 3.0);
        let difference = f64::try_from((expected - actual).abs().max()).unwrap();
        assert_eq!(difference, 0.0);
    }

    #[cfg(feature = "gpu-training")]
    #[test]
    fn gpu_auditors_train_on_binary_and_regression_fixtures() {
        let _guard = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            tch::Cuda::is_available(),
            "frozen CUDA runtime is unavailable"
        );
        for task in [Task::Binary, Task::Regression] {
            let (train, test) = fixtures(task);
            let mean = train
                .target
                .iter()
                .map(|&value| f64::from(value))
                .sum::<f64>()
                / train.rows as f64;
            let null = loss(task, &test.target, &vec![mean; test.rows]).unwrap();
            for id in [
                "gpu_residual_mlp_ensemble",
                "gpu_tabular_feature_transformer",
            ] {
                let prediction = auditor_backend(id)
                    .unwrap()
                    .predict(&train, &test, task, 57721)
                    .unwrap();
                let paired = auditor_backend(id)
                    .unwrap()
                    .predict_two(&train, &test, &test, task, 57721)
                    .unwrap();
                assert_eq!(paired.0, paired.1, "{id} paired prediction mismatch");
                assert_eq!(prediction, paired.0, "{id} paired fit changed output");
                assert!(
                    loss(task, &test.target, &prediction).unwrap() < null,
                    "{id} {task:?}"
                );
                assert_eq!(implementation_hash(id).len(), 64);
            }
        }
    }
}

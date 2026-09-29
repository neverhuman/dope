#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::production::write_canonical;
    use crate::router::{QuantizedLayer, ROUTER_OUTPUTS};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "dope-embedding-{name}-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn test_bundle(hidden: usize) -> RouterBundle {
        let sketch = 856;
        let embedding = 1;
        let first_hidden = 3;
        let candidate_ids = (0..ACTION_EMBEDDING_CANDIDATES)
            .map(|index| match index {
                0 => "independent_quantile".into(),
                1 => "chow_liu".into(),
                _ => format!("candidate-{index:02}"),
            })
            .collect::<Vec<_>>();
        RouterBundle {
            format: "dope-distilled-router".into(),
            version: 1,
            sketch_mean: vec![0.0; sketch],
            sketch_std: vec![1.0; sketch],
            input_scale: 0.01,
            candidate_embeddings: candidate_ids
                .iter()
                .enumerate()
                .map(|(index, _)| vec![index as f32 / ACTION_EMBEDDING_CANDIDATES as f32])
                .collect(),
            candidate_ids,
            layers: vec![
                QuantizedLayer {
                    input: sketch + embedding,
                    output: first_hidden,
                    weights: vec![1; (sketch + embedding) * first_hidden],
                    biases: vec![0; first_hidden],
                    multiplier: 0.01,
                },
                QuantizedLayer {
                    input: first_hidden,
                    output: hidden,
                    weights: vec![1; first_hidden * hidden],
                    biases: vec![1; hidden],
                    multiplier: 0.5,
                },
                QuantizedLayer {
                    input: hidden,
                    output: ROUTER_OUTPUTS,
                    weights: vec![1; hidden * ROUTER_OUTPUTS],
                    biases: vec![0; ROUTER_OUTPUTS],
                    multiplier: 0.5,
                },
            ],
            output_scale: vec![0.1; ROUTER_OUTPUTS],
            output_bias: vec![0.2; ROUTER_OUTPUTS],
            ga2m_router: None,
            trained: true,
            training_evidence_sha256: Some("a".repeat(64)),
        }
    }

    fn passing_evidence(bundle: &RouterBundle) -> RouterEvidence {
        RouterEvidence {
            format: "dope-router-evidence".into(),
            version: 1,
            top_four_oracle_recall: 1.0,
            maximum_profile_regret_upper: 0.01,
            beats_random: true,
            beats_best_fixed: true,
            beats_ga2m: true,
            paired_hypervolume_improvement: 0.1,
            paired_hypervolume_ci_lower: 0.01,
            student_max_abs_difference: 0.0001,
            top_choice_agreement: 1.0,
            bundle_bytes: serde_json::to_vec(bundle).unwrap().len() as u64,
            inference_p95_ms: 1.0,
            sketch_p95_ms: 10.0,
            training_evidence_sha256: "a".repeat(64),
            best_fixed_candidate_id: "independent_quantile".into(),
            student_mean_regret: 0.01,
            random_mean_regret: 0.02,
            best_fixed_mean_regret: 0.02,
            ga2m_mean_regret: 0.02,
        }
    }

    fn write_router(
        root: &Path,
        hidden: usize,
    ) -> (std::path::PathBuf, std::path::PathBuf, RouterBundle) {
        fs::create_dir_all(root).unwrap();
        let bundle = test_bundle(hidden);
        let evidence = passing_evidence(&bundle);
        let bundle_path = root.join("router.bundle.json");
        let evidence_path = root.join("router-evidence.json");
        write_canonical(&bundle_path, &bundle).unwrap();
        write_canonical(&evidence_path, &evidence).unwrap();
        (bundle_path, evidence_path, bundle)
    }

    #[test]
    fn loader_normalizes_features_targets_and_binary_labels() {
        let root = temp_dir("loader");
        let csv = root.join("train.csv");
        fs::write(
            &csv,
            "constant,missing,value,outcome\n7,,1,20\n7,NaN,3,10\n7,,5,20\n",
        )
        .unwrap();
        let binary = load_numeric_csv(&csv, TargetSelector::Name("outcome"), Task::Binary).unwrap();
        assert_eq!(binary.table.target, vec![1.0, 0.0, 1.0]);
        assert_eq!(binary.table.columns[0], vec![0.0; 3]);
        assert!(binary.table.columns[1].iter().all(|value| value.is_nan()));
        assert_eq!(binary.table.columns[2], vec![0.0, 0.5, 1.0]);
        assert_eq!(binary.normalization.features[1].minimum, None);

        let regression =
            load_numeric_csv(&csv, TargetSelector::Index(3), Task::Regression).unwrap();
        assert_eq!(regression.table.target, vec![1.0, 0.0, 1.0]);
        assert_eq!(regression.normalization.target.minimum, Some(10.0));
        assert_eq!(regression.normalization.target.maximum, Some(20.0));

        let extreme = root.join("extreme.csv");
        fs::write(&extreme, "value,outcome\n0,-1e308\n1,0\n2,1e308\n").unwrap();
        let extreme =
            load_numeric_csv(&extreme, TargetSelector::Index(1), Task::Regression).unwrap();
        assert_eq!(extreme.table.target, vec![0.0, 0.5, 1.0]);

        let target_only = root.join("target-only.csv");
        fs::write(&target_only, "outcome\n10\n20\n15\n").unwrap();
        let target_only =
            load_numeric_csv(&target_only, TargetSelector::Index(0), Task::Regression).unwrap();
        assert_eq!(target_only.table.rows, 3);
        assert_eq!(target_only.table.features, 0);
        assert!(target_only.table.columns.is_empty());
        assert!(target_only.normalization.features.is_empty());
        assert_eq!(target_only.table.target, vec![0.0, 1.0, 0.5]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn embedding_is_invariant_to_rows_columns_and_affine_rescaling() {
        let root = temp_dir("invariance");
        let base = root.join("base.csv");
        let transformed = root.join("transformed.csv");
        fs::write(&base, "a,b,y\n0,10,-1\n1,20,2\n2,NaN,-1\n3,40,2\n").unwrap();
        fs::write(&transformed, "b,y,a\n116,9,11\nNaN,4,9\n56,9,7\n26,4,5\n").unwrap();
        let (bundle, evidence, _) = write_router(&root.join("router"), 128);
        let original = embed_dataset(
            &base,
            TargetSelector::Name("y"),
            Task::Binary,
            &bundle,
            &evidence,
        )
        .unwrap();
        let changed = embed_dataset(
            &transformed,
            TargetSelector::Name("y"),
            Task::Binary,
            &bundle,
            &evidence,
        )
        .unwrap();
        assert_eq!(original.vector, changed.vector);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn malformed_csv_inputs_fail_closed() {
        let root = temp_dir("malformed");
        let cases = [
            ("empty", ""),
            ("header-only", "a,y\n"),
            ("duplicate", "a,a,y\n1,2,0\n"),
            ("ragged", "a,b,y\n1,2,0\n1,0\n"),
            ("invalid", "a,y\nnope,0\n"),
            ("infinite", "a,y\ninf,0\n"),
            ("missing-target", "a,y\n1,NaN\n"),
            ("one-binary-label", "a,y\n1,4\n2,4\n"),
            ("three-binary-labels", "a,y\n1,4\n2,5\n3,6\n"),
        ];
        for (name, contents) in cases {
            let path = root.join(format!("{name}.csv"));
            fs::write(&path, contents).unwrap();
            assert!(
                load_numeric_csv(&path, TargetSelector::Name("y"), Task::Binary).is_err(),
                "{name} unexpectedly loaded"
            );
        }
        let valid = root.join("valid.csv");
        fs::write(&valid, "a,y\n1,0\n2,1\n").unwrap();
        assert!(load_numeric_csv(&valid, TargetSelector::Name("absent"), Task::Binary).is_err());
        assert!(load_numeric_csv(&valid, TargetSelector::Index(2), Task::Binary).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn qualified_embedding_has_expected_offsets_and_prediction_parity() {
        let root = temp_dir("shape");
        let csv = root.join("train.csv");
        fs::write(&csv, "a,b,y\n1,4,10\n2,8,20\n3,12,15\n").unwrap();
        let hidden = 128;
        let (bundle_path, evidence_path, bundle) =
            write_router(&root.join(hidden.to_string()), hidden);
        let embedding = embed_dataset(
            &csv,
            TargetSelector::Index(2),
            Task::Regression,
            &bundle_path,
            &evidence_path,
        )
        .unwrap();
        assert_eq!(
            embedding.dimension,
            856 + ACTION_EMBEDDING_CANDIDATES * (hidden + ROUTER_OUTPUTS)
        );
        assert_eq!(embedding.component_names.len(), embedding.dimension);
        assert_eq!(embedding.components.standardized_sketch.offset, 0);
        assert_eq!(embedding.components.standardized_sketch.length, 856);
        assert_eq!(embedding.components.candidates[0].hidden_state.offset, 856);
        assert_eq!(
            embedding.components.candidates[0].hidden_state.length,
            hidden
        );

        let loaded = load_numeric_csv(&csv, TargetSelector::Index(2), Task::Regression).unwrap();
        let sketch = DatasetSketch::from_train(&loaded.table, Task::Regression);
        let predictions = QuantizedRouter::new(bundle)
            .unwrap()
            .predict(&sketch)
            .unwrap();
        assert_eq!(embedding.predictions, predictions);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn output_is_bit_reproducible_and_csv_matches_json_vector() {
        let root = temp_dir("output");
        let csv = root.join("train.csv");
        fs::write(&csv, "a,y\n1,0\n2,1\n").unwrap();
        let (bundle, evidence, _) = write_router(&root.join("router"), 128);
        let first = embed_dataset(
            &csv,
            TargetSelector::Name("y"),
            Task::Binary,
            &bundle,
            &evidence,
        )
        .unwrap();
        let second = embed_dataset(
            &csv,
            TargetSelector::Name("y"),
            Task::Binary,
            &bundle,
            &evidence,
        )
        .unwrap();
        assert_eq!(
            canonical_json(&first).unwrap(),
            canonical_json(&second).unwrap()
        );
        let vector_path = root.join("embedding.csv");
        write_embedding_csv(&vector_path, &first).unwrap();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(vector_path)
            .unwrap();
        assert_eq!(
            reader.headers().unwrap().iter().collect::<Vec<_>>(),
            first
                .component_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
        let row = reader.records().next().unwrap().unwrap();
        let parsed = row
            .iter()
            .map(|value| value.parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(parsed, first.vector);
        assert!(reader.records().next().is_none());
        let json_vector = serde_json::to_string(&first.vector).unwrap();
        let csv_vector = fs::read_to_string(root.join("embedding.csv"))
            .unwrap()
            .lines()
            .nth(1)
            .unwrap()
            .to_string();
        assert_eq!(csv_vector, json_vector[1..json_vector.len() - 1]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn promotion_failures_are_admitted_and_integrity_failures_are_rejected() {
        let root = temp_dir("qualification");
        let csv = root.join("train.csv");
        fs::write(&csv, "a,y\n1,0\n2,1\n").unwrap();
        let bundle_path = root.join("bundle.json");
        let evidence_path = root.join("evidence.json");

        let bundle = test_bundle(128);
        write_canonical(&bundle_path, &bundle).unwrap();
        let mut evidence = passing_evidence(&bundle);
        evidence.beats_random = false;
        write_canonical(&evidence_path, &evidence).unwrap();
        let qualification = qualify_embedding_router(&bundle_path, &evidence_path).unwrap();
        assert!(qualification.embedding_eligible);
        assert!(!qualification.promotion_qualified);
        assert_eq!(qualification.promotion_failed_gates, vec!["beats_random"]);
        assert_eq!(qualification.version, 2);
        assert_eq!(qualification.dimension, ACTION_EMBEDDING_DIMENSION);
        embed_dataset(
            &csv,
            TargetSelector::Name("y"),
            Task::Binary,
            &bundle_path,
            &evidence_path,
        )
        .unwrap();

        let mut evidence = passing_evidence(&bundle);
        evidence.bundle_bytes += 1;
        write_canonical(&evidence_path, &evidence).unwrap();
        assert!(
            embed_dataset(
                &csv,
                TargetSelector::Name("y"),
                Task::Binary,
                &bundle_path,
                &evidence_path,
            )
            .is_err()
        );

        let mut evidence = passing_evidence(&bundle);
        evidence.training_evidence_sha256 = "b".repeat(64);
        write_canonical(&evidence_path, &evidence).unwrap();
        assert!(
            embed_dataset(
                &csv,
                TargetSelector::Name("y"),
                Task::Binary,
                &bundle_path,
                &evidence_path,
            )
            .is_err()
        );

        let wrong_dimension = test_bundle(127);
        write_canonical(&bundle_path, &wrong_dimension).unwrap();
        write_canonical(&evidence_path, &passing_evidence(&wrong_dimension)).unwrap();
        assert!(
            qualify_embedding_router(&bundle_path, &evidence_path)
                .unwrap_err()
                .to_string()
                .contains("expected 4168")
        );

        let mut corrupted = bundle.clone();
        corrupted.layers[0].weights.pop();
        write_canonical(&bundle_path, &corrupted).unwrap();
        write_canonical(&evidence_path, &passing_evidence(&corrupted)).unwrap();
        assert!(
            embed_dataset(
                &csv,
                TargetSelector::Name("y"),
                Task::Binary,
                &bundle_path,
                &evidence_path,
            )
            .is_err()
        );

        let mut untrained = bundle;
        untrained.trained = false;
        untrained.training_evidence_sha256 = None;
        write_canonical(&bundle_path, &untrained).unwrap();
        write_canonical(&evidence_path, &passing_evidence(&untrained)).unwrap();
        assert!(
            embed_dataset(
                &csv,
                TargetSelector::Name("y"),
                Task::Binary,
                &bundle_path,
                &evidence_path,
            )
            .unwrap_err()
            .to_string()
            .contains("untrained router")
        );

        let bundle = test_bundle(128);
        write_canonical(&bundle_path, &bundle).unwrap();
        fs::write(&evidence_path, b"{malformed").unwrap();
        assert!(qualify_embedding_router(&bundle_path, &evidence_path).is_err());

        let finite_evidence = canonical_json(&passing_evidence(&bundle)).unwrap();
        let non_finite_evidence = String::from_utf8(finite_evidence).unwrap().replace(
            "\"top_four_oracle_recall\":1.0",
            "\"top_four_oracle_recall\":1e999",
        );
        fs::write(&evidence_path, non_finite_evidence).unwrap();
        assert!(qualify_embedding_router(&bundle_path, &evidence_path).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}

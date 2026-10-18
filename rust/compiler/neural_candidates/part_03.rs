#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_small_cap_reports_smallest_observed_encoded_candidate() {
        let options = CompileOptions {
            release_policy: ReleasePolicy::new(AnonymizationTier::L3, Some(1), false).unwrap(),
            beam_width: Some(1),
            ..Default::default()
        };
        let error = compile_kernel_from_arrays(
            &[0.0, 0.25, 0.75, 1.0],
            &[0.0, 0.25, 0.75, 1.0],
            4,
            1,
            Task::Regression,
            &options,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("smallest observed size:"));
    }

    #[test]
    fn near_equal_discrete_values_are_counted_without_exact_search_panics() {
        let fit = fit_column(&[0.1, 0.100_000_05, 0.2, 0.2]);
        assert!(fit.discrete.is_some());
    }

    #[test]
    fn forced_baseline_consumes_the_requested_beam_budget() {
        let features = (0..64).map(|row| row as f32 / 63.0).collect::<Vec<_>>();
        let targets = features.iter().map(|x| 0.2 + 0.6 * x).collect::<Vec<_>>();
        for beam in [1, 18] {
            let options = CompileOptions {
                seed: Some(1729),
                beam_width: Some(beam),
                quantization_profiles: vec![8],
                ..Default::default()
            };
            let result =
                compile_kernel_from_arrays(&features, &targets, 64, 1, Task::Regression, &options)
                    .unwrap();
            assert!(
                result
                    .candidates
                    .iter()
                    .any(|candidate| candidate.candidate_id == "k5-independence-sparse_linear-q8")
            );
            assert_eq!(result.report.search_history.len(), beam);
            assert_eq!(result.report.candidate_count, beam);
        }
    }
    use crate::codec::decode_kernel;
    use std::collections::BTreeSet;

    #[test]
    fn compiles_deterministically_and_keeps_positions() {
        let rows = 100;
        let features: Vec<f32> = (0..rows)
            .flat_map(|row| [0.1, (row % 2) as f32, row as f32 / rows as f32])
            .collect();
        let target: Vec<f32> = (0..rows)
            .map(|row| 0.2 + 0.6 * row as f32 / rows as f32)
            .collect();
        let options = CompileOptions {
            seed: Some(7),
            ..Default::default()
        };
        let first =
            compile_kernel_from_arrays(&features, &target, rows, 3, Task::Regression, &options)
                .unwrap();
        let second =
            compile_kernel_from_arrays(&features, &target, rows, 3, Task::Regression, &options)
                .unwrap();
        assert_eq!(first.artifact, second.artifact);
        let kernel = decode_kernel(&first.artifact).unwrap();
        assert!(matches!(kernel.schema[0].kind, SchemaKind::Constant));
        assert!(matches!(kernel.schema[1].kind, SchemaKind::Binary));
    }

    #[test]
    fn tournaments_all_fitted_targets_on_nonlinear_data() {
        let rows = 256;
        let features: Vec<f32> = (0..rows)
            .flat_map(|row| {
                let left = (row % 16) as f32 / 15.0;
                let right = (row / 16) as f32 / 15.0;
                [left, right]
            })
            .collect();
        let target: Vec<f32> = (0..rows)
            .map(|row| {
                let left = row % 16 >= 8;
                let right = row / 16 >= 8;
                if left == right { 0.1 } else { 0.9 }
            })
            .collect();
        let options = CompileOptions {
            seed: Some(1729),
            beam_width: Some(18),
            quantization_profiles: vec![8],
            language: None,
            deadline: None,
            backend_id: None,
            neural_target_weight: 2.0,
            neural_structural_penalty: 0.0,
            adaptive_allocation: false,
            development_residual: None,
            release_policy: ReleasePolicy::new(AnonymizationTier::L0, None, false).unwrap(),
        };
        let result =
            compile_kernel_from_arrays(&features, &target, rows, 2, Task::Regression, &options)
                .unwrap();
        let target_families: BTreeSet<_> = result
            .candidates
            .iter()
            .map(|candidate| candidate.target.as_str())
            .collect();
        assert_eq!(
            target_families,
            BTreeSet::from([
                "compact_neural_residual",
                "ga2m",
                "mars",
                "oblivious_tree",
                "sparse_gam",
                "sparse_linear",
            ])
        );
        let dependence_families: BTreeSet<_> = result
            .candidates
            .iter()
            .map(|candidate| candidate.dependence.as_str())
            .collect();
        assert_eq!(
            dependence_families,
            BTreeSet::from([
                "chow_liu",
                "independence",
                "poet_factor",
                "sparse_gaussian_copula",
                "triangular_autoregressive",
                "truncated_c_vine",
                "two_component_copula_mixture",
            ])
        );
        let baseline = result
            .candidates
            .iter()
            .find(|candidate| candidate.target == "sparse_linear")
            .unwrap();
        let tree = result
            .candidates
            .iter()
            .find(|candidate| candidate.target == "oblivious_tree")
            .unwrap();
        assert!(tree.stored_target_fit > baseline.stored_target_fit + 0.5);
        assert_eq!(result.report.candidate_count, 18);
        let mut wire = serde_json::to_value(baseline).unwrap();
        assert!(wire.get("utility_retention").is_none());
        let value = wire
            .as_object_mut()
            .unwrap()
            .remove("stored_target_fit")
            .unwrap();
        wire["utility_retention"] = value;
        let old: CandidateReport = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(&old, baseline);
        wire["stored_target_fit"] = serde_json::json!(0.0);
        assert!(serde_json::from_value::<CandidateReport>(wire).is_err());
        assert!(
            result.candidates.iter().all(|candidate| {
                candidate.encoded_numbers > 0
                    && (0.0..=1.0).contains(&candidate.knowledge_compression)
            })
        );
        assert!(result.report.encoded_numbers > 0);
        assert!((0.0..=1.0).contains(&result.report.knowledge_compression));
        decode_kernel(&result.artifact).unwrap();
    }
}

#[cfg(test)]
mod staged_search_tests {
    use super::*;
    use crate::schema::{
        Horizon, HorizonUnit, PreparedGroup, PreparedRights, PreparedTarget, RightsState,
        TableSchemaV1, TaskKind,
    };
    use crate::typed_data::prepared::{PrepareSpec, PreparedView, prepare};
    use crate::typed_data::{InputLimits, TypedTable};

    fn views() -> (PreparedView, PreparedView) {
        let schema = TableSchemaV1::from_private_json(
            br#"{"format":"dope-table-schema","version":1,"columns":[
            {"id":"c0001","source_name":"private_header","role":"feature","dtype":"continuous"},
            {"id":"c0002","source_name":"private_target","role":"target","dtype":"continuous","units":"private_units"}]}"#,
        ).unwrap();
        let table = TypedTable::read_csv(
            b"private_header,private_target\n10,100\n20,200\n12,120\n18,180\n".as_slice(),
            schema,
            InputLimits::default(),
        )
        .unwrap();
        let fit = prepare(
            &table,
            &[0, 1],
            PrepareSpec {
                task: TaskKind::Regression,
                horizon: Horizon {
                    steps: 1,
                    unit: HorizonUnit::Rows,
                },
                group_map: vec![PreparedGroup {
                    id: 1,
                    column_ids: vec![1],
                }],
                target: PreparedTarget::Regression {
                    units: "private_units".into(),
                },
                entity_digest: "a".repeat(64),
                split_digest: "b".repeat(64),
                rights: PreparedRights {
                    state: RightsState::ResearchUnassessed,
                    evidence_digest: None,
                },
            },
        )
        .unwrap();
        let selection = fit.freeze().unwrap().apply(&table, &[2, 3]).unwrap();
        (fit, selection)
    }

    fn options() -> DevelopmentSearchOptions {
        DevelopmentSearchOptions {
            identity: CacheIdentity {
                source_digest: [1; 32],
                fit_row_digest: [2; 32],
                prepared_view_digest: [3; 32],
                compiler_version: [4; 32],
                codec_version: [5; 32],
                candidate_spec: [6; 32],
                worker_abi_version: [7; 32],
                model_panel_digest: [8; 32],
                context_window: 128,
                seed_schedule: vec![1729, 42],
                loss_id: [9; 32],
                metric_id: [10; 32],
            },
            policy: ReleasePolicy::new(AnonymizationTier::L0, None, false).unwrap(),
            sample_rows: 128,
            max_candidates: 2,
        }
    }

    fn proposal(coefficient: f32) -> Kernel {
        Kernel {
            task: Task::Regression,
            rows_fitted: 2,
            features: 1,
            seed: 7,
            seed_policy: 0,
            quantization_bits: 8,
            compliant: false,
            schema: vec![ColumnSchema {
                kind: SchemaKind::Continuous,
                missing_probability: 0.0,
                impute: 0.5,
                transform: crate::model::Transform::Identity,
            }],
            marginals: vec![Marginal::QuantileSpline {
                values: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            }],
            program: KernelProgram::Symbolic {
                dependence: Dependence::Independent,
                target: Target::SparseLinear {
                    intercept: 0.0,
                    terms: vec![LinearTerm {
                        feature: 0,
                        coefficient,
                    }],
                },
                noise: Noise::Homoscedastic { sigma: 0.0 },
            },
        }
    }

    #[derive(Default)]
    struct LinearPanel {
        calls: usize,
        fail: bool,
    }

    impl DevelopmentEvaluator for LinearPanel {
        fn evaluate(
            &mut self,
            synthetic: &[f32],
            features: usize,
            selection: &PreparedCompileInput<'_>,
            _seed: u64,
        ) -> Result<DevelopmentMetrics> {
            self.calls += 1;
            if self.fail {
                return Err(DopeError::Data(
                    "private_header private_value 123456789".into(),
                ));
            }
            assert_eq!(features, 1);
            let x = synthetic.chunks_exact(2).map(|r| r[0]).collect::<Vec<_>>();
            let y = synthetic.chunks_exact(2).map(|r| r[1]).collect::<Vec<_>>();
            let train = Table::from_arrays(&x, &y, y.len(), 1, Task::Regression)?;
            let screen = target_screen(&train.columns, &train.target);
            let (target, _, _) = fit_linear_target(&train, &train.columns, &screen);
            let view = selection.view();
            let contract = view.freeze().unwrap();
            let loss = view
                .rows()
                .iter()
                .map(|row| {
                    let prediction = evaluate_target(&target, &[row[0] as f32]).0;
                    let original = contract
                        .regression_prediction(f64::from(prediction))
                        .unwrap();
                    (row[1] - original).powi(2)
                })
                .sum::<f64>()
                / view.rows().len() as f64;
            let mut fitted = proposal(1.0);
            if let KernelProgram::Symbolic { target: model, .. } = &mut fitted.program {
                *model = target;
            }
            let selection_x = vec![view.rows().iter().map(|r| r[0] as f32).collect()];
            let selection_y = view.rows().iter().map(|r| r[1] as f32).collect::<Vec<_>>();
            Ok(DevelopmentMetrics {
                tstr_loss: loss,
                driver_agreement: quantized_driver_agreement(
                    &fitted,
                    &target_screen(&selection_x, &selection_y),
                ),
            })
        }
    }

    #[test]
    fn real_reload_sample_tstr_selects_and_freezes_without_confirmation_labels() {
        let (fit, select) = views();
        let contract = fit.freeze().unwrap();
        let input = contract.compile_input(&select).unwrap();
        let selection =
            DevelopmentSelection::new(&input, SearchPartition::DevelopmentSelection).unwrap();
        let mut cache = DevelopmentCache::new(8).unwrap();
        let mut panel = LinearPanel::default();
        let proposals = [proposal(0.0), proposal(1.0)];
        let options = options();
        let first =
            search_development_candidates(&proposals, &selection, &options, &mut cache, &mut panel)
                .unwrap();
        assert_eq!(panel.calls, 4);
        assert!(first.observations[1].metrics.tstr_loss < first.observations[0].metrics.tstr_loss);
        assert_eq!(
            first.recipe.artifact(),
            encode_kernel(&proposals[1]).unwrap()
        );
        for (observation, proposal) in first.observations.iter().zip(&proposals) {
            assert_eq!(
                observation.wire_bytes,
                encode_kernel(proposal).unwrap().len()
            );
        }
        let second =
            search_development_candidates(&proposals, &selection, &options, &mut cache, &mut panel)
                .unwrap();
        assert_eq!(panel.calls, 4);
        assert!(second.observations.iter().all(|o| o.cache_hit));
        assert_eq!(first.recipe.digest(), second.recipe.digest());
        let confirmation = first.recipe.confirm(first.recipe.digest()).unwrap();
        assert!(!confirmation.qualified);
        assert_eq!(
            confirmation,
            first.recipe.confirm(first.recipe.digest()).unwrap()
        );
        assert_eq!(panel.calls, 4);
        assert!(first.recipe.confirm([0; 32]).is_err());
        let mut changed = options;
        changed.identity.worker_abi_version[0] ^= 1;
        search_development_candidates(&proposals, &selection, &changed, &mut cache, &mut panel)
            .unwrap();
        assert_eq!(panel.calls, 8);
        // Same declared pins with changed D_select labels cannot reuse a score.
        let mut poisoned = serde_json::to_value(&select).unwrap();
        poisoned["rows"][0][1] = serde_json::json!(135.0);
        let poisoned =
            PreparedView::from_private_json(&serde_json::to_vec(&poisoned).unwrap()).unwrap();
        let poisoned_input = contract.compile_input(&poisoned).unwrap();
        let poisoned_selection =
            DevelopmentSelection::new(&poisoned_input, SearchPartition::DevelopmentSelection)
                .unwrap();
        let changed_selection = search_development_candidates(
            &proposals,
            &poisoned_selection,
            &changed,
            &mut cache,
            &mut panel,
        )
        .unwrap();
        assert!(changed_selection.observations.iter().all(|o| !o.cache_hit));
        assert_eq!(panel.calls, 12);
        let x = (0..64).map(|row| row as f32 / 63.0).collect::<Vec<_>>();
        let compiled = compile_kernel_from_arrays(
            &x,
            &x,
            64,
            1,
            Task::Regression,
            &CompileOptions {
                beam_width: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        let bridged =
            select_compiled_development(&compiled, &selection, &changed, &mut cache, &mut panel)
                .unwrap();
        assert_eq!(
            bridged.observations.len(),
            compiled.candidate_artifacts.len()
        );
        assert_eq!(bridged.recipe.artifact(), compiled.artifact);
    }

    #[test]
    fn caps_sealed_access_invalid_proposals_and_evaluator_errors_fail_closed() {
        let (fit, select) = views();
        let contract = fit.freeze().unwrap();
        let input = contract.compile_input(&select).unwrap();
        for partition in [
            SearchPartition::SealedConfirmationQueryFeatures,
            SearchPartition::HeldOutTestLabels,
        ] {
            assert!(
                matches!(DevelopmentSelection::new(&input, partition), Err(DopeError::Data(code)) if code == "sealed_partition_access_denied")
            );
        }
        let selection =
            DevelopmentSelection::new(&input, SearchPartition::DevelopmentSelection).unwrap();
        let mut cache = DevelopmentCache::new(8).unwrap();
        let mut panel = LinearPanel::default();
        let mut options = options();
        let proposals = [proposal(1.0)];
        options.policy = ReleasePolicy::new(AnonymizationTier::L3, Some(1), false).unwrap();
        assert!(
            search_development_candidates(&proposals, &selection, &options, &mut cache, &mut panel)
                .is_err()
        );
        assert_eq!(panel.calls, 0);
        let exact = encode_kernel(&proposals[0]).unwrap().len();
        options.policy = ReleasePolicy::new(AnonymizationTier::L3, Some(exact), false).unwrap();
        let result =
            search_development_candidates(&proposals, &selection, &options, &mut cache, &mut panel)
                .unwrap();
        assert_eq!(result.observations[0].wire_bytes, exact);
        let mut invalid = proposal(1.0);
        invalid.schema[0].impute = f32::NAN;
        assert!(
            search_development_candidates(&[invalid], &selection, &options, &mut cache, &mut panel)
                .is_err()
        );
        options.identity.model_panel_digest[0] ^= 1;
        panel.fail = true;
        let error =
            search_development_candidates(&proposals, &selection, &options, &mut cache, &mut panel)
                .err()
                .unwrap();
        let text = error.to_string();
        assert!(text.contains("development_evaluation_failed"));
        for private in ["private_header", "private_value", "123456789"] {
            assert!(!text.contains(private));
        }
        panel.fail = false;
        let retry =
            search_development_candidates(&proposals, &selection, &options, &mut cache, &mut panel)
                .unwrap();
        assert!(!retry.observations[0].cache_hit);
    }
}

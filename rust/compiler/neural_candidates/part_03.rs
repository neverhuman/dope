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
        assert!(tree.utility_retention > baseline.utility_retention + 0.5);
        assert_eq!(result.report.candidate_count, 18);
        decode_kernel(&result.artifact).unwrap();
    }
}

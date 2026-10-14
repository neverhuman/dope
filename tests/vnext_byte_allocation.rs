use dope_kernel::compiler::{CompileOptions, compile_kernel_from_arrays};
use dope_kernel::contract::{AnonymizationTier, ReleasePolicy};
use dope_kernel::model::Task;

#[test]
fn actual_adaptive_search_respects_encoded_cap_and_same_offer_budget() {
    let rows = 64;
    let features = (0..rows)
        .flat_map(|row| [(row % 8) as f32 / 7.0, (row / 8) as f32 / 7.0])
        .collect::<Vec<_>>();
    let targets = (0..rows)
        .map(|row| {
            if (row % 8 >= 4) == (row / 8 >= 4) {
                0.1
            } else {
                0.9
            }
        })
        .collect::<Vec<_>>();
    for adaptive in [false, true] {
        let options = CompileOptions {
            seed: Some(1729),
            beam_width: Some(52),
            quantization_profiles: vec![4, 8],
            adaptive_allocation: adaptive,
            release_policy: ReleasePolicy::new(AnonymizationTier::L0, Some(512), false).unwrap(),
            ..Default::default()
        };
        let result =
            compile_kernel_from_arrays(&features, &targets, rows, 2, Task::Regression, &options)
                .unwrap();
        assert_eq!(result.report.search_history.len(), 52);
        assert!(result.artifact.len() <= 512);
        for (candidate, artifact) in &result.candidate_artifacts {
            assert!(artifact.len() <= 512, "over-cap promotion: {candidate}");
            dope_kernel::codec::decode_kernel(artifact).unwrap();
        }
        assert!(result.report.search_preparation_seconds.unwrap() >= 0.0);
        if adaptive {
            assert!(result.report.refinement_fit_count > 0);
            assert!(result.report.refinement_fit_count <= 18);
            assert!(result.report.search_history.iter().any(|attempt| {
                attempt.capacity_refinement.is_some() && attempt.actual_artifact_bytes.is_some()
            }));
        } else {
            assert_eq!(result.report.refinement_fit_count, 0);
            assert!(
                result
                    .report
                    .search_history
                    .iter()
                    .all(|attempt| attempt.capacity_refinement.is_none())
            );
            assert!(result.candidates.iter().all(|candidate| !matches!(
                candidate.candidate_id.rsplit('-').next(),
                Some("c1" | "c2" | "c3")
            )));
        }
        assert!(!result.report.unvisited_beam_candidates.is_empty());
        let families = result
            .candidates
            .iter()
            .map(|candidate| (candidate.dependence.clone(), candidate.target.clone()))
            .collect::<std::collections::BTreeSet<_>>();
        // Cap refusals remain in the trace and do not receive fictitious quality.
        assert!(families.len() >= 20);
        if adaptive {
            assert!(
                result
                    .report
                    .search_history
                    .iter()
                    .any(|attempt| attempt.parent_candidate_id.is_some())
            );
            assert!(result.report.allocation_policy.contains("uncalibrated"));
            assert!(
                result
                    .report
                    .search_history
                    .iter()
                    .any(|attempt| attempt.allocation == "adaptive_encoded_racing_uncalibrated")
            );
        }
    }
}

#[test]
fn refitting_deadline_spends_no_additional_offer_or_artifact_budget() {
    let features = (0..64).map(|row| row as f32 / 63.0).collect::<Vec<_>>();
    let options = CompileOptions {
        adaptive_allocation: true,
        seed: Some(1729),
        beam_width: Some(9),
        quantization_profiles: vec![8],
        deadline: Some(std::time::Duration::ZERO),
        ..Default::default()
    };
    let result =
        compile_kernel_from_arrays(&features, &features, 64, 1, Task::Regression, &options)
            .unwrap();
    assert_eq!(result.report.search_history.len(), 9);
    assert_eq!(result.report.refinement_fit_count, 0);
    assert_eq!(result.candidate_artifacts.len(), 1);
    assert_eq!(
        result
            .report
            .search_history
            .iter()
            .filter(|attempt| attempt.disposition == "unvisited_deadline")
            .count(),
        8
    );
    assert!(!result.report.compliant);
}

#[test]
fn frozen_backend_cannot_silently_become_an_adaptive_campaign() {
    let options = CompileOptions {
        adaptive_allocation: true,
        backend_id: Some("chow_liu".into()),
        ..Default::default()
    };
    let error = compile_kernel_from_arrays(
        &[0.0, 0.5, 1.0],
        &[0.0, 0.5, 1.0],
        3,
        1,
        Task::Regression,
        &options,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("adaptive_allocation_requires_unrestricted_symbolic_tournament")
    );
}

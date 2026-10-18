//! Bounded pre-repair observations; these fixtures do not qualify a release.
use super::*;
use crate::codec::{account_artifact, decode_kernel};
use crate::model::repro_tests::{draws, fixture, mse};
use serde_json::{Value, json};

fn rank_correlation(samples: &[f32], width: usize) -> f32 {
    let left = samples
        .chunks_exact(width + 1)
        .map(|r| r[0])
        .collect::<Vec<_>>();
    let right = samples
        .chunks_exact(width + 1)
        .map(|r| r[1])
        .collect::<Vec<_>>();
    dot(&rank_normalize(&left), &rank_normalize(&right))
}

pub(crate) fn copula_observation() -> Value {
    let ranks = [
        rank_normalize(&[0., 1., 2., 3., 4.]),
        rank_normalize(&[2., 1., 0., 4., 3.]),
    ];
    let matrix = full_correlation(&ranks);
    assert!((matrix[1] - 0.5).abs() < 1e-6, "analytic Spearman fixture");
    let gaussian = 2.0 * (std::f32::consts::PI * matrix[1] / 6.0).sin();
    assert!(
        (gaussian - matrix[1]).abs() > 0.017,
        "rank and latent differ"
    );
    assert!(
        1.0 - gaussian * gaussian > 0.0,
        "two-variable reference is PSD"
    );

    // Seeded Gaussian source with population Spearman 0.5, then exercise the
    // actual rank fitter and candidate dependence path, rather than a formula only.
    let reference = Dependence::GaussianCopula {
        cholesky: vec![1.0, 0.0, gaussian, (1.0 - gaussian * gaussian).sqrt()],
    };
    let source = draws(fixture(reference, 2), 4096);
    let empirical_rank = rank_correlation(&source, 2);
    assert!(
        (empirical_rank - 0.5).abs() < 0.05,
        "seeded Gaussian fixture"
    );
    let matrix = vec![1.0, empirical_rank, empirical_rank, 1.0];
    let (legacy, _, _) = candidate_dependence(3, 2, Some(&matrix), &[], &[]);
    let Dependence::SparseGraph { edges } = &legacy else {
        panic!("sparse profile")
    };
    assert!(
        (edges[0].correlation - empirical_rank).abs() < 1e-7,
        "legacy propagates rank dot product as latent rho"
    );
    let mapped = 2.0 * (std::f32::consts::PI * empirical_rank / 6.0).sin();
    let corrected = Dependence::SparseGraph {
        edges: vec![CopulaEdge {
            parent: 0,
            child: 1,
            correlation: mapped,
        }],
    };
    let legacy_samples = draws(fixture(legacy, 2), 4096);
    let mapped_samples = draws(fixture(corrected, 2), 4096);
    let legacy_rank = rank_correlation(&legacy_samples, 2);
    let mapped_rank = rank_correlation(&mapped_samples, 2);
    assert!(
        mapped_rank - legacy_rank > 0.01,
        "measured downstream rank effect"
    );
    json!({
        "id": "P02-REPRO-COPULA", "status": "reproduced_before_repair",
        "rows": 4096, "width": 2, "seed": 1729,
        "analytic_spearman": 0.5, "analytic_gaussian_latent": gaussian,
        "fitted_rank_dot_product": empirical_rank, "legacy_latent": empirical_rank,
        "gaussian_mapped_latent": mapped, "legacy_sample_spearman": legacy_rank,
        "mapped_sample_spearman": mapped_rank,
        "paired_sample_mse": mse(&legacy_samples, &mapped_samples),
        "psd": "analytic two-variable determinant positive; no high-dimensional PSD repair tested",
        "assumption": "continuous Gaussian copula; conversion is not universal"
    })
}

pub(crate) fn vine_fit_observation() -> Value {
    let width = 8;
    let mut matrix = vec![0.2; width * width];
    for i in 0..width {
        matrix[i * width + i] = 1.0;
    }
    let Dependence::Vine { edges } = vine_dependence(Some(&matrix), &[], width) else {
        panic!("vine profile")
    };
    assert_eq!(edges.len(), width * 3);
    assert!(
        edges
            .iter()
            .enumerate()
            .all(|(i, e)| e.conditioning_depth == (i % 4) as u8),
        "depth is edge-index modulo four"
    );
    let depth_counts = (0..4)
        .map(|d| edges.iter().filter(|e| e.conditioning_depth == d).count())
        .collect::<Vec<_>>();
    json!({"width": width, "available_pairs": width * (width - 1) / 2,
        "retained_edges": edges.len(), "depth_counts": depth_counts,
        "selection": "strongest 3*width unconditional pairs; depth=edge_index%4",
        "tree_conditioning_sets_present": false})
}

fn factor_covariance(dependence: &Dependence, width: usize) -> Vec<f64> {
    let Dependence::Poet { rank, loadings, .. } = dependence else {
        panic!("factor profile")
    };
    let mut dense = vec![0.0; width * usize::from(*rank)];
    for l in loadings {
        dense[l.feature as usize * usize::from(*rank) + usize::from(l.factor)] =
            f64::from(l.loading);
    }
    (0..width * width)
        .map(|i| {
            (0..usize::from(*rank))
                .map(|k| {
                    dense[i / width * usize::from(*rank) + k]
                        * dense[i % width * usize::from(*rank) + k]
                })
                .sum()
        })
        .collect()
}

pub(crate) fn poet_observation() -> Value {
    let width = 8;
    let matrix = (0..width * width)
        .map(|i| {
            let (r, c) = (i / width, i % width);
            if r == c {
                1.0
            } else if r >= 4 && c >= 4 {
                0.8
            } else {
                0.0
            }
        })
        .collect::<Vec<f32>>();
    let original = poet_dependence(Some(&matrix), &[], width);
    let Dependence::Poet {
        rank,
        loadings,
        residual_edges,
    } = &original
    else {
        panic!("factor profile")
    };
    assert_eq!(*rank, 4);
    assert!(
        loadings.iter().all(|l| (l.loading
            - matrix[l.feature as usize * width + usize::from(l.factor)] / 2.0)
            .abs()
            < 1e-7),
        "loadings are first correlation columns"
    );
    assert!(
        loadings
            .iter()
            .filter(|l| l.feature >= 4)
            .all(|l| l.loading == 0.0),
        "dominant final-block eigenspace has no factor loading"
    );
    assert!(
        residual_edges
            .iter()
            .all(|e| e.correlation == matrix[e.parent as usize * width + e.child as usize]),
        "residual edges retain raw correlations, without factor subtraction"
    );
    let permutation = [4, 5, 6, 7, 0, 1, 2, 3];
    let permuted_matrix = (0..width * width)
        .map(|i| matrix[permutation[i / width] * width + permutation[i % width]])
        .collect::<Vec<_>>();
    let permuted = poet_dependence(Some(&permuted_matrix), &[], width);
    let covariance = factor_covariance(&original, width);
    let permuted_covariance = factor_covariance(&permuted, width);
    let equivariance_error = (0..width * width)
        .map(|i| {
            (permuted_covariance[i]
                - covariance[permutation[i / width] * width + permutation[i % width]])
                .powi(2)
        })
        .sum::<f64>()
        .sqrt();
    assert!(
        equivariance_error > 1.0,
        "factor covariance depends on column order"
    );
    // Analytic dominant eigenpair: eigenvalue 3.4, unit vector 1/2 on last block.
    let dominant_error = (0..width * width)
        .map(|i| {
            let reference = if i / width >= 4 && i % width >= 4 {
                3.4 / 4.0
            } else {
                0.0
            };
            (covariance[i] - reference).powi(2)
        })
        .sum::<f64>()
        .sqrt();
    let factor_only = Dependence::Poet {
        rank: *rank,
        loadings: loadings.clone(),
        residual_edges: vec![],
    };
    let residual_mse = mse(
        &draws(fixture(original.clone(), width), 256),
        &draws(fixture(factor_only, width), 256),
    );
    assert!(
        residual_mse > 1e-4,
        "residuals affect downstream samples separately"
    );
    json!({
        "id": "P02-REPRO-POET", "audit_alias": "P02-REPRO-FACTOR",
        "status": "reproduced_before_repair", "width": width, "rank": rank,
        "seed": 1729, "rows": 256, "loading_count": loadings.len(),
        "dominant_block_nonzero_loadings": 0, "analytic_dominant_eigenvalue": 3.4,
        "factor_covariance_permutation_frobenius": equivariance_error,
        "factor_covariance_dominant_reference_frobenius": dominant_error,
        "residual_edge_count": residual_edges.len(), "residuals_use_raw_correlations": true,
        "with_vs_without_residual_sample_mse": residual_mse,
        "scope": "first-column factor heuristic; residual edges measured separately; no general POET repair claim"
    })
}

pub(crate) fn common_profile_observation() -> Value {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("one-thread pool");
    pool.install(|| {
        let features = (0..64).flat_map(|r|
            [r as f32 / 63.0, ((r * 17) % 64) as f32 / 63.0, ((r * 7) % 64) as f32 / 63.0]
        ).collect::<Vec<_>>();
        let target = features.chunks_exact(3).map(|r| 0.2 + 0.4 * r[0] + 0.2 * r[1]).collect::<Vec<_>>();
        let options = CompileOptions {
            seed: Some(1729), backend_id: Some("chow_liu".into()),
            quantization_profiles: vec![8], beam_width: Some(1), ..Default::default()
        };
        let run = || compile_kernel_from_arrays(&features, &target, 64, 3, Task::Regression, &options)
            .expect("common symbolic profile compiles");
        let first = run();
        let second = run();
        assert!(first.artifact == second.artifact, "same-pool deterministic artifact");
        assert_eq!(first.candidates.len(), 1);
        assert!(first.report.probe_failures.is_empty());
        let kernel = decode_kernel(&first.artifact).expect("common profile decodes");
        assert!(matches!(&kernel.program, KernelProgram::Symbolic {
            dependence: Dependence::ChowLiu { .. }, .. }), "explicit symbolic candidate");
        let predictions = features.chunks_exact(3).map(|row|
            evaluate_target(kernel.symbolic().expect("symbolic target").1, row).0
        ).collect::<Vec<_>>();
        let prediction_bytes = predictions.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let input_bytes = features.iter().chain(&target).flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let accounting = account_artifact(&first.artifact).expect("section accounting");
        let mut offset = accounting.header_bytes;
        let sections = accounting.sections.iter().map(|s| {
            let end = offset + s.framing_bytes + s.encoded_bytes;
            let hash = blake3::hash(&first.artifact[offset..end]).to_hex().to_string();
            offset = end;
            json!({"tag": s.tag, "framed_bytes": s.framing_bytes + s.encoded_bytes, "blake3": hash})
        }).collect::<Vec<_>>();
        json!({
            "id": "P02-REPRO-CPU-GPU", "status": "common_profile_baseline_measured",
            "features": if cfg!(feature = "gpu-training") { "gpu-training" } else { "default" },
            "backend": "chow_liu", "candidate_id": first.report.selected_candidate,
            "release_policy_hash": first.report.release_policy_hash,
            "fixture_blake3": blake3::hash(&input_bytes).to_hex().to_string(),
            "candidate_count": 1, "seed": 1729, "quantization_profiles": [8],
            "beam_width": 1, "adaptive_allocation": false, "development_residual": null,
            "rows": 64, "width": 3, "threads": 1, "device": "cpu_symbolic",
            "artifact_blake3": blake3::hash(&first.artifact).to_hex().to_string(),
            "artifact_bytes": first.artifact.len(), "sections": sections,
            "predictions_blake3": blake3::hash(&prediction_bytes).to_hex().to_string(),
            "repeat_artifact_equal": true, "cross_build_parity": "blocked_pending_admitted_gpu_build",
            "comparison_rule": "same source, seed, explicit profile and environment; exact section and prediction hashes",
            "unrestricted_tournament_parity": "not_claimed; gpu-training may add L2/L3 compact neural probes",
            "gpu_resource_admission": "not_granted; no neural training or GPU job executed"
        })
    })
}

pub(crate) fn q4_levels(bits: u8) -> Vec<f32> {
    (0..16)
        .map(|i| quantize(i as f32 / 15.0, bits, true))
        .collect()
}

#[test]
fn p02_gaussian_copula_rank_mapping() {
    copula_observation();
}

#[test]
fn p02_vine_truncation_is_pair_count_not_tree_depth() {
    vine_fit_observation();
}

#[test]
fn p02_factor_dominant_subspace() {
    poet_observation();
}

#[test]
fn p02_cpu_gpu_common_profile() {
    common_profile_observation();
}

#[cfg(feature = "gpu-training")]
#[test]
fn p02_forced_symbolic_profile_skips_neural_probes() {
    let table = Table::from_arrays_with_policy(
        &[0.0, 0.3, 0.6, 1.0],
        &[0.0, 0.3, 0.6, 1.0],
        4,
        1,
        Task::Regression,
        &CompileOptions::default().release_policy,
    )
    .expect("synthetic table");
    let options = CompileOptions {
        backend_id: Some("chow_liu".into()),
        release_policy: ReleasePolicy::new(AnonymizationTier::L2, None, false).expect("L2 policy"),
        ..Default::default()
    };
    let symbolic =
        compile_symbolic_table(&table, Task::Regression, &options).expect("symbolic compile");
    let expected = symbolic.artifact.clone();
    let probed =
        probe_compact_neural(&table, Task::Regression, &options, symbolic).expect("probe bypass");
    assert!(
        probed.artifact == expected,
        "forced pool bypasses GPU tournament"
    );
    assert!(probed.report.probe_failures.is_empty());
}

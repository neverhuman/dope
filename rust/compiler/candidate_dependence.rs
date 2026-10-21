fn candidate_dependence(
    dependence_family: u8,
    width: usize,
    correlation: Option<&Vec<f32>>,
    edges: &[CopulaEdge],
    triangular_terms: &[TriangularTerm],
) -> (Dependence, &'static str, f64) {
    match dependence_family {
        0 => {
            let fidelity = correlation.map_or(0.85, |matrix| {
                let mean = matrix
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| index / width != index % width)
                    .map(|(_, value)| f64::from(value.abs()))
                    .sum::<f64>()
                    / (width * width - width).max(1) as f64;
                (1.0 - mean / 0.5).clamp(0.0, 1.0)
            });
            (Dependence::Independent, "independence", fidelity)
        }
        1 => (
            Dependence::ChowLiu {
                root: 0,
                edges: edges.to_vec(),
            },
            "chow_liu",
            0.97,
        ),
        2 => (
            Dependence::Triangular {
                terms: triangular_terms.to_vec(),
            },
            "triangular_autoregressive",
            0.98,
        ),
        3 => (
            Dependence::SparseGraph {
                edges: strongest_sparse_edges(correlation, edges, width, width.saturating_mul(4)),
            },
            "sparse_gaussian_copula",
            0.985,
        ),
        4 => (
            poet_dependence(correlation, edges, width),
            "poet_factor",
            0.985,
        ),
        5 => (
            vine_dependence(correlation, edges, width),
            "truncated_c_vine",
            0.99,
        ),
        6 => (
            Dependence::Mixture {
                weights: vec![0.5, 0.5],
                components: vec![
                    Dependence::ChowLiu {
                        root: 0,
                        edges: edges.to_vec(),
                    },
                    Dependence::Triangular {
                        terms: triangular_terms.to_vec(),
                    },
                ],
            },
            "two_component_copula_mixture",
            0.99,
        ),
        7 => (
            gaussian_copula_dependence(correlation, edges, width),
            "gaussian_copula",
            0.995,
        ),
        _ => unreachable!("bounded dependence family"),
    }
}

#[cfg(test)]
mod public_prior_pilot {
    use serde_json::Value;

    #[test]
    fn frozen_prior_pilot_compares_controls_and_refuses_unproven_public_prior() {
        let lock: Value = serde_json::from_str(include_str!(
            "../../research/transfer/locks/public-prior.json"
        )).unwrap();
        let pilot = &lock["source_model_holdouts"]["pilot"];
        assert_eq!(pilot["qualification"], "fixture_only_not_a_source_or_model_holdout");
        let candidates = pilot["candidate_outcomes"].as_array().unwrap();
        assert_eq!(candidates.len(), 3);
        let no_prior = candidates.iter().min_by_key(|item| item["no_prior_order"].as_u64().unwrap()).unwrap();
        let hand_coded = candidates.iter().max_by(|a, b| {
            a["hand_coded_score"].as_f64().unwrap().total_cmp(&b["hand_coded_score"].as_f64().unwrap())
        }).unwrap();
        let loss = |item: &Value| item["synthetic_holdout_loss"].as_f64().unwrap();
        assert_eq!(no_prior["candidate"], "linear");
        assert_eq!(hand_coded["candidate"], "linear");
        assert_eq!(loss(no_prior) - loss(hand_coded), 0.0);

        let accounting = &lock["independence_and_accounting"];
        assert_eq!(accounting["existing_language_calibration_is_public_prior"], false);
        assert!(accounting["public_training_source_roster_sha256"].is_null());
        assert!(accounting["public_prior_payload_sha256"].is_null());
        assert!(lock["source_model_holdouts"]["development"]["model_roster_sha256"].is_null());
        assert_eq!(lock["production_wiring"], "disabled_until_all_admission_gates_pass");
        eprintln!("R07 pilot: no-prior loss={}, hand-coded loss={}, difference=0; public-trained unsupported (no disjoint roster/payload); complete cold bundle bytes unavailable; private adaptation not run", loss(no_prior), loss(hand_coded));
    }
}

/// Dependence family index of the exact normal-score Gaussian copula.
const GAUSSIAN_COPULA_FAMILY: u8 = 7;
/// Widest table stored as a dense Cholesky factor; wider tables keep the
/// Chow-Liu tree so the factor's p^2 bytes stay bounded.
const GAUSSIAN_COPULA_MAX_WIDTH: usize = 32;

/// Exact Gaussian copula of normal scores. Spearman correlations of the fit
/// columns are mapped to latent Pearson correlations with r = 2 sin(pi rho / 6)
/// (exact for a Gaussian copula), shrunk toward the identity only as far as a
/// Cholesky factorization needs, and stored as a dense lower-triangular factor.
/// Unlike the sequential edge updates of families 3 to 6, sampling L z keeps
/// every latent variance at one and reproduces all pairwise correlations.
fn gaussian_copula_dependence(
    correlation: Option<&Vec<f32>>,
    edges: &[CopulaEdge],
    width: usize,
) -> Dependence {
    let tree = || Dependence::ChowLiu {
        root: 0,
        edges: edges.to_vec(),
    };
    let Some(matrix) = correlation.filter(|_| (2..=GAUSSIAN_COPULA_MAX_WIDTH).contains(&width))
    else {
        return tree();
    };
    let latent = (0..width * width)
        .map(|index| {
            if index / width == index % width {
                1.0
            } else {
                2.0 * (std::f64::consts::PI * f64::from(matrix[index]) / 6.0).sin()
            }
        })
        .collect::<Vec<f64>>();
    for shrinkage in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5] {
        let shrunk = latent
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if index / width == index % width {
                    1.0
                } else {
                    (1.0 - shrinkage) * value
                }
            })
            .collect::<Vec<_>>();
        if let Some(factor) = cholesky_lower(&shrunk, width) {
            return Dependence::GaussianCopula {
                cholesky: factor.into_iter().map(|value| value.clamp(-1.0, 1.0) as f32).collect(),
            };
        }
    }
    tree()
}

fn cholesky_lower(matrix: &[f64], width: usize) -> Option<Vec<f64>> {
    let mut factor = vec![0.0; width * width];
    for row in 0..width {
        for column in 0..=row {
            let mut sum = matrix[row * width + column];
            for inner in 0..column {
                sum -= factor[row * width + inner] * factor[column * width + inner];
            }
            if row == column {
                if sum <= 1e-6 {
                    return None;
                }
                factor[row * width + column] = sum.sqrt();
            } else {
                factor[row * width + column] = sum / factor[column * width + column];
            }
        }
    }
    Some(factor)
}

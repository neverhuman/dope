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
        _ => unreachable!("bounded dependence family"),
    }
}

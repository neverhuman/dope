/// Executed development capacity controls. Counts are budgets, not measured
/// parameter importance, confidence bounds, or a conditional-mask capability.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapacityRefinement {
    pub dependence_rank: usize,
    pub edge_multiplier: usize,
    pub target_support: usize,
    pub target_width: usize,
    pub interaction_terms: usize,
    pub residual_support: usize,
}

fn capacity_refinement(profile: u8) -> Option<CapacityRefinement> {
    let (rank, edges, support, width, pairs, residual) = match profile {
        0 => return None,
        1 => (1, 1, 4, 4, 2, 0),
        2 => (2, 2, 12, 8, 4, 2),
        3 => (4, 4, 24, 16, 8, 8),
        _ => unreachable!("internal bounded capacity profile"),
    };
    Some(CapacityRefinement {
        dependence_rank: rank,
        edge_multiplier: edges,
        target_support: support,
        target_width: width,
        interaction_terms: pairs,
        residual_support: residual,
    })
}

fn capacity_order(profile: u8) -> u8 {
    if profile == 0 { 3 } else { profile - 1 }
}

fn capacity_candidate_id(id: String, profile: u8) -> String {
    if profile == 0 {
        id
    } else {
        format!("{id}-c{profile}")
    }
}

fn refined_target(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
    task: Task,
    target_index: usize,
    capacity: CapacityRefinement,
) -> (Target, Noise, &'static str) {
    let target = refined_target_model(table, completed, screen, task, target_index, capacity);
    let (noise, rule) = fit_noise_bounded(
        table,
        completed,
        screen,
        &target,
        task,
        capacity.residual_support,
        Some(capacity),
    );
    (target, noise, rule)
}

fn refined_target_model(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
    task: Task,
    target_index: usize,
    capacity: CapacityRefinement,
) -> Target {
    let limited = &screen[..screen.len().min(capacity.target_support)];
    match target_index {
        0 => match task {
            Task::Regression => fit_linear_target(table, completed, limited).0,
            Task::Binary => fit_logistic_target(table, completed, limited).0,
        },
        1 => fit_gam_target_bounded(table, completed, screen, task, capacity.target_support),
        2 => fit_ga2m_target_bounded(
            table,
            completed,
            screen,
            task,
            capacity.target_support,
            capacity.interaction_terms,
        ),
        3 => fit_mars_target(table, completed, limited, task),
        4 => fit_oblivious_tree_target(table, completed, limited, task),
        5 => fit_compact_neural_target_native_bounded(
            table,
            completed,
            limited,
            task,
            capacity.target_width,
        ),
        _ => unreachable!("bounded target family"),
    }
}

fn refine_dependence(dependence: &mut Dependence, capacity: CapacityRefinement, width: usize) {
    let limit = width.saturating_mul(capacity.edge_multiplier);
    match dependence {
        Dependence::ChowLiu { edges, .. } | Dependence::SparseGraph { edges } => {
            edges.truncate(limit)
        }
        Dependence::Poet {
            rank,
            loadings,
            residual_edges,
        } => {
            let previous_rank = *rank;
            *rank = (*rank).min(capacity.dependence_rank as u8);
            loadings.retain(|loading| loading.factor < *rank);
            let rescale = (f32::from(previous_rank) / f32::from(*rank)).sqrt();
            for loading in loadings {
                loading.loading = (loading.loading * rescale).clamp(-1.0, 1.0);
            }
            residual_edges.truncate(limit);
        }
        Dependence::Vine { edges } => edges.truncate(limit),
        Dependence::Triangular { terms } => terms.truncate(limit),
        Dependence::Mixture { components, .. } => {
            for component in components {
                refine_dependence(component, capacity, width);
            }
        }
        Dependence::Independent | Dependence::GaussianCopula { .. } => {}
    }
}

#[cfg(test)]
mod capacity_refinement_tests {
    use super::*;

    #[test]
    fn larger_support_and_width_are_actually_fitted_for_diffuse_signal() {
        let rows = 128;
        let width = 24;
        let columns = (0..width)
            .map(|feature| {
                (0..rows)
                    .map(|row| {
                        (splitmix64(row as u64 ^ ((feature as u64 + 1) << 24)) % 1000) as f32
                            / 999.0
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let target = (0..rows)
            .map(|row| columns.iter().map(|column| column[row]).sum::<f32>() / width as f32)
            .collect();
        let table = Table {
            rows,
            features: width,
            columns: columns.clone(),
            target,
        };
        let screen = target_screen(&columns, &table.target);
        for (profile, expected_support, expected_width) in [(1, 4, 4), (3, 24, 16)] {
            let capacity = capacity_refinement(profile).unwrap();
            let gam =
                refined_target_model(&table, &columns, &screen, Task::Regression, 1, capacity);
            let Target::SparseGam { terms, .. } = gam else {
                panic!("GAM refinement")
            };
            assert_eq!(terms.len(), expected_support);
            let neural =
                refined_target_model(&table, &columns, &screen, Task::Regression, 5, capacity);
            let Target::CompactNeuralResidual {
                hidden_features,
                hidden_width,
                input_weights,
                ..
            } = neural
            else {
                panic!("native neural refinement")
            };
            assert_eq!(hidden_features.len(), expected_support);
            assert_eq!(usize::from(hidden_width), expected_width);
            assert_eq!(input_weights.len(), expected_support * expected_width);
        }
    }

    #[test]
    fn rank_and_edges_change_retained_state_with_valid_factor_indices() {
        let width = 8;
        let correlation = (0..width * width)
            .map(|at| if at / width == at % width { 1.0 } else { 0.2 })
            .collect();
        let (mut small, _, _) = candidate_dependence(4, width, Some(&correlation), &[], &[]);
        let (mut large, _, _) = candidate_dependence(4, width, Some(&correlation), &[], &[]);
        refine_dependence(&mut small, capacity_refinement(1).unwrap(), width);
        refine_dependence(&mut large, capacity_refinement(3).unwrap(), width);
        let Dependence::Poet {
            rank,
            loadings,
            residual_edges,
        } = small
        else {
            panic!("Poet")
        };
        assert_eq!(rank, 1);
        assert_eq!(loadings.len(), 8);
        assert_eq!(residual_edges.len(), 8);
        assert!(loadings.iter().all(|loading| loading.factor < rank));
        let Dependence::Poet {
            rank,
            loadings,
            residual_edges,
        } = large
        else {
            panic!("Poet")
        };
        assert_eq!(rank, 4);
        assert_eq!(loadings.len(), 32);
        assert_eq!(residual_edges.len(), 16);
    }
}

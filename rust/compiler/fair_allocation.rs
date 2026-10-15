// The final coordinate chooses an executed capacity profile; zero preserves
// the historical fixed fits. Nonzero profiles are development-only refinements.
type CandidateSpec = (usize, u8, usize, u8, f64, u8);

/// Keep the same beam/encode budget but give target/dependence families a
/// round-robin opportunity before more precision/knots within a family.
fn fair_candidate_specs(specs: Vec<CandidateSpec>, beam: usize, seed: u64) -> Vec<CandidateSpec> {
    let mut groups = BTreeMap::<(u8, usize), std::collections::VecDeque<CandidateSpec>>::new();
    for spec in specs {
        groups.entry((spec.1, spec.2)).or_default().push_back(spec);
    }
    let mut families = groups.keys().copied().collect::<Vec<_>>();
    families.sort_by_key(|(dependence, target)| {
        splitmix64(seed ^ u64::from(*dependence) ^ ((*target as u64) << 8))
    });
    let mut selected = Vec::new();
    while selected.len() < beam {
        let before = selected.len();
        for family in &families {
            if let Some(spec) = groups.get_mut(family).and_then(|group| group.pop_front()) {
                selected.push(spec);
            }
            if selected.len() == beam {
                break;
            }
        }
        if selected.len() == before {
            break;
        }
    }
    selected
}

#[cfg(test)]
mod allocation_tests {
    use super::*;
    #[test]
    fn round_robin_preserves_families_under_input_order_permutations() {
        let mut specs = Vec::new();
        for family in 0..7 {
            for target in 0..6 {
                for bits in [4, 8, 16] {
                    specs.push((0, family, target, bits, 0.0, 0));
                }
            }
        }
        let selected = fair_candidate_specs(specs.clone(), 42, 1729);
        let families = selected
            .iter()
            .map(|s| (s.1, s.2))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(families.len(), 42);
        specs.reverse();
        let reversed = fair_candidate_specs(specs, 42, 1729);
        assert_eq!(
            reversed
                .iter()
                .map(|s| (s.1, s.2))
                .collect::<std::collections::BTreeSet<_>>(),
            families
        );
        assert_eq!(selected.len(), 42);
    }
}

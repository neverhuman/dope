// Bounded development search, using encoded observations only. Its exploration
// bonus is a heuristic, never a calibrated confidence interval or certification.

type AllocationFamily = (u8, usize);

#[derive(Clone)]
struct AllocationPoint {
    score: f64,
    bytes: usize,
    id: String,
}

struct AdaptiveAllocation {
    groups: BTreeMap<AllocationFamily, std::collections::VecDeque<CandidateSpec>>,
    families: Vec<AllocationFamily>,
    offered: BTreeMap<AllocationFamily, usize>,
    frontier: BTreeMap<AllocationFamily, Vec<AllocationPoint>>,
    count: usize,
    beam: usize,
    exploration: bool,
}

impl AdaptiveAllocation {
    fn new(mut specs: Vec<CandidateSpec>, beam: usize, seed: u64, baseline: CandidateSpec) -> Self {
        specs.sort_by(|a, b| {
            b.4.total_cmp(&a.4)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.3.cmp(&b.3))
        });
        let mut groups =
            BTreeMap::<AllocationFamily, std::collections::VecDeque<CandidateSpec>>::new();
        for spec in specs {
            groups.entry((spec.1, spec.2)).or_default().push_back(spec);
        }
        let baseline_family = (baseline.1, baseline.2);
        if let Some(group) = groups.get_mut(&baseline_family) {
            group.retain(|spec| (spec.0, spec.3) != (baseline.0, baseline.3));
            group.push_front(baseline);
        }
        let mut families = groups.keys().copied().collect::<Vec<_>>();
        families.sort_by_key(|(dependence, target)| {
            (
                (*dependence, *target) != baseline_family,
                splitmix64(seed ^ u64::from(*dependence) ^ ((*target as u64) << 8)),
            )
        });
        Self {
            groups,
            families,
            offered: BTreeMap::new(),
            frontier: BTreeMap::new(),
            count: 0,
            beam,
            exploration: true,
        }
    }

    fn next(&mut self) -> Option<CandidateSpec> {
        if self.count >= self.beam {
            return None;
        }
        let available = self
            .families
            .iter()
            .copied()
            .filter(|family| {
                self.groups
                    .get(family)
                    .is_some_and(|group| !group.is_empty())
            })
            .collect::<Vec<_>>();
        let unvisited = available
            .iter()
            .copied()
            .find(|family| !self.offered.contains_key(family));
        // Every family gets one opportunity before racing. Thereafter one in
        // four offers explores the least visited available family.
        self.exploration = unvisited.is_some() || self.count.is_multiple_of(4);
        let family = if let Some(family) = unvisited {
            family
        } else if self.exploration {
            *available
                .iter()
                .min_by_key(|family| self.offered.get(*family).copied().unwrap_or(0))?
        } else {
            let value = |family: &AllocationFamily| {
                let best = self
                    .frontier
                    .get(family)
                    .into_iter()
                    .flatten()
                    .map(|point| point.score)
                    .max_by(f64::total_cmp)
                    .unwrap_or(-1.0);
                best + 0.1 / (self.offered.get(family).copied().unwrap_or(0).max(1) as f64).sqrt()
            };
            *available
                .iter()
                .max_by(|a, b| value(a).total_cmp(&value(b)).then_with(|| b.cmp(a)))?
        };
        self.count += 1;
        *self.offered.entry(family).or_default() += 1;
        self.groups.get_mut(&family)?.pop_front()
    }

    fn parent(&self, spec: CandidateSpec) -> Option<String> {
        self.frontier
            .get(&(spec.1, spec.2))?
            .iter()
            .max_by(|a, b| {
                a.score
                    .total_cmp(&b.score)
                    .then_with(|| b.bytes.cmp(&a.bytes))
            })
            .map(|point| point.id.clone())
    }

    fn observe(&mut self, spec: CandidateSpec, id: String, score: f64, bytes: usize) {
        let frontier = self.frontier.entry((spec.1, spec.2)).or_default();
        if frontier
            .iter()
            .any(|point| point.score >= score && point.bytes <= bytes)
        {
            return;
        }
        frontier.retain(|point| !(score >= point.score && bytes <= point.bytes));
        frontier.push(AllocationPoint { score, bytes, id });
    }
}

// Remove only exact zero support after quantization. This changes serialized
// capacity without dropping a nonzero fitted signal or using final outcomes.
fn prune_encoded_zero_support(target: &mut Target, noise: &mut Noise) -> usize {
    fn retain<T>(terms: &mut Vec<T>, nonzero: impl FnMut(&T) -> bool) -> usize {
        let before = terms.len();
        terms.retain(nonzero);
        before - terms.len()
    }
    let mut removed = match target {
        Target::SparseLinear { terms, .. } | Target::SparseLogistic { terms, .. } => {
            retain(terms, |term| term.coefficient != 0.0)
        }
        Target::SparseGam { terms, .. } => {
            retain(terms, |term| term.values.iter().any(|value| *value != 0.0))
        }
        Target::Ga2m {
            main_terms,
            interactions,
            ..
        } => {
            retain(main_terms, |term| {
                term.values.iter().any(|value| *value != 0.0)
            }) + retain(interactions, |term| {
                term.values.iter().any(|value| *value != 0.0)
            })
        }
        Target::Mars { terms, .. } => retain(terms, |term| term.coefficient != 0.0),
        _ => 0,
    };
    if let Noise::Heteroscedastic { terms, .. } = noise {
        removed += retain(terms, |term| term.coefficient != 0.0);
    }
    removed
}

#[cfg(test)]
mod adaptive_allocation_tests {
    use super::*;

    fn specs() -> Vec<CandidateSpec> {
        (0..3)
            .flat_map(|family| (0..3).map(move |knot| (knot, family, 0, 8, 0.0)))
            .collect()
    }

    #[test]
    fn encoded_feedback_changes_refinement_without_extra_offers_or_starvation() {
        let mut search = AdaptiveAllocation::new(specs(), 6, 1729, (0, 0, 0, 8, 0.0));
        let mut first = std::collections::BTreeSet::new();
        for _ in 0..3 {
            let spec = search.next().unwrap();
            first.insert(spec.1);
            search.observe(
                spec,
                format!("parent-{}", spec.1),
                if spec.1 == 2 { 0.95 } else { 0.1 },
                100,
            );
        }
        assert_eq!(first, std::collections::BTreeSet::from([0, 1, 2]));
        let refined = search.next().unwrap();
        assert_eq!(refined.1, 2);
        assert_eq!(search.parent(refined).as_deref(), Some("parent-2"));
        assert!(!search.exploration);
        assert!(search.next().is_some());
        assert!(search.exploration);
        assert!(search.next().is_some());
        assert!(search.next().is_none());
    }

    #[test]
    fn proposal_permutation_keeps_sequence_and_encoded_frontier_retains_tradeoffs() {
        let mut reversed = specs();
        reversed.reverse();
        let mut left = AdaptiveAllocation::new(specs(), 9, 1729, (0, 0, 0, 8, 0.0));
        let mut right = AdaptiveAllocation::new(reversed, 9, 1729, (0, 0, 0, 8, 0.0));
        for _ in 0..9 {
            assert_eq!(left.next(), right.next());
        }
        let spec = (0, 0, 0, 8, 0.0);
        left.observe(spec, "small".into(), 0.5, 50);
        left.observe(spec, "better".into(), 0.8, 100);
        left.observe(spec, "dominated".into(), 0.4, 60);
        assert_eq!(left.frontier[&(0, 0)].len(), 2);
        assert_eq!(left.parent(spec).as_deref(), Some("better"));
    }

    #[test]
    fn pruning_encoded_zeros_preserves_nonzero_mean_and_residual_support() {
        let mut target = Target::SparseLinear {
            intercept: 0.2,
            terms: vec![
                LinearTerm {
                    feature: 0,
                    coefficient: 0.0,
                },
                LinearTerm {
                    feature: 1,
                    coefficient: 0.3,
                },
            ],
        };
        let mut noise = Noise::Homoscedastic { sigma: 0.1 };
        let before = evaluate_target(&target, &[0.1, 0.8]);
        assert_eq!(prune_encoded_zero_support(&mut target, &mut noise), 1);
        assert_eq!(before, evaluate_target(&target, &[0.1, 0.8]));
        let Target::SparseLinear { terms, .. } = target else {
            panic!("linear fixture")
        };
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].feature, 1);
        assert_eq!(terms[0].coefficient, 0.3);
    }
}

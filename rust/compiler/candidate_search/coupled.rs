//! Deterministic selector comparison for coupled development edits.
//!
//! This module is a bounded experiment harness. Metrics are supplied by the
//! caller; outcomes are not production qualification or privacy certification.
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum EditOperator {
    Knot,
    Edge,
    Lowrank,
    Ga2m,
    Noise,
    Precision,
    Subexpression,
    Category,
    Calibration,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CoupledEdit {
    pub section: String,
    pub operator: EditOperator,
    pub payload: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SelectorMetrics {
    pub aggregate_tstr: f64,
    pub critical_tail_tstr: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selector {
    ExistingSelector,
    Greedy,
    Beam,
    Exchange,
    Lookahead,
    IndependentSectionDp,
}

impl Selector {
    pub const ALL: [Self; 6] = [
        Self::ExistingSelector,
        Self::Greedy,
        Self::Beam,
        Self::Exchange,
        Self::Lookahead,
        Self::IndependentSectionDp,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::ExistingSelector => "existing_selector",
            Self::Greedy => "greedy",
            Self::Beam => "beam",
            Self::Exchange => "exchange",
            Self::Lookahead => "lookahead",
            Self::IndependentSectionDp => "independent_section_dp",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectorResult {
    pub selector: Selector,
    pub selected: Vec<usize>,
    pub metrics: SelectorMetrics,
    pub whole_file_bytes: usize,
    pub evaluations: usize,
    pub stopped_for_critical_tail: bool,
}

#[derive(Clone, Copy)]
struct Candidate {
    metrics: SelectorMetrics,
    bytes: usize,
}

/// Compare selectors under one candidate pool, a common evaluation cap, and
/// actual serialized whole-file size. The evaluator must be deterministic.
pub fn compare<F>(
    base: &[u8],
    edits: &[CoupledEdit],
    byte_cap: usize,
    critical_tail_floor: f64,
    evaluation_budget: usize,
    mut evaluate: F,
) -> Vec<SelectorResult>
where
    F: FnMut(&[usize]) -> SelectorMetrics,
{
    Selector::ALL
        .into_iter()
        .map(|selector| {
            run_one(
                selector,
                base,
                edits,
                byte_cap,
                critical_tail_floor,
                evaluation_budget,
                &mut evaluate,
            )
        })
        .collect()
}

fn whole_file_bytes(base: &[u8], edits: &[CoupledEdit], selected: &[usize]) -> usize {
    let chosen = selected.iter().map(|&index| &edits[index]).collect::<Vec<_>>();
    serde_json::to_vec(&(base, chosen)).expect("serializable selector input").len()
}

fn normalized(mut selected: Vec<usize>) -> Vec<usize> {
    selected.sort_unstable();
    selected.dedup();
    selected
}

struct SelectorSearch<'a, F> {
    base: &'a [u8],
    edits: &'a [CoupledEdit],
    byte_cap: usize,
    tail_floor: f64,
    budget: usize,
    evaluate: &'a mut F,
    evaluations: usize,
    stopped: bool,
    cache: std::collections::BTreeMap<Vec<usize>, Option<Candidate>>,
    best_set: Vec<usize>,
    best: Candidate,
}

impl<F: FnMut(&[usize]) -> SelectorMetrics> SelectorSearch<'_, F> {
    fn score(&mut self, selected: Vec<usize>) -> Option<Candidate> {
        let selected = normalized(selected);
        if let Some(cached) = self.cache.get(&selected) { return *cached; }
        if self.evaluations >= self.budget { return None; }
        let bytes = whole_file_bytes(self.base, self.edits, &selected);
        let candidate = if bytes > self.byte_cap {
            None
        } else {
            self.evaluations += 1;
            let metrics = (self.evaluate)(&selected);
            if !metrics.aggregate_tstr.is_finite() || !metrics.critical_tail_tstr.is_finite() {
                None
            } else if metrics.critical_tail_tstr < self.tail_floor {
                self.stopped = true;
                None
            } else {
                Some(Candidate { metrics, bytes })
            }
        };
        self.cache.insert(selected, candidate);
        candidate
    }

    fn consider(&mut self, selected: Vec<usize>) {
        if let Some(candidate) = self.score(selected.clone()) {
            if candidate.metrics.aggregate_tstr > self.best.metrics.aggregate_tstr {
                self.best = candidate;
                self.best_set = normalized(selected);
            }
        }
    }
}

fn run_one<F>(selector: Selector, base: &[u8], edits: &[CoupledEdit], byte_cap: usize,
    tail_floor: f64, budget: usize, evaluate: &mut F) -> SelectorResult
where F: FnMut(&[usize]) -> SelectorMetrics {
    let empty = Candidate { metrics: SelectorMetrics::default(), bytes: whole_file_bytes(base, edits, &[]) };
    let mut search = SelectorSearch { base, edits, byte_cap, tail_floor, budget, evaluate,
        evaluations: 0, stopped: false, cache: std::collections::BTreeMap::new(),
        best_set: Vec::new(), best: empty };
    search.best = search.score(Vec::new()).unwrap_or(empty);
    match selector {
        Selector::ExistingSelector => {}
        Selector::Greedy => {
            let mut current = Vec::new();
            loop {
                let before = search.score(current.clone()).unwrap_or(search.best);
                let mut winner: Option<(usize, Candidate)> = None;
                for index in 0..edits.len() {
                    if current.contains(&index) { continue; }
                    let mut proposal = current.clone(); proposal.push(index);
                    if let Some(candidate) = search.score(proposal) {
                        if candidate.metrics.aggregate_tstr > before.metrics.aggregate_tstr
                            && winner.is_none_or(|(_, prior)| candidate.metrics.aggregate_tstr > prior.metrics.aggregate_tstr) {
                            winner = Some((index, candidate));
                        }
                    }
                }
                let Some((index, candidate)) = winner else { break };
                current.push(index); current.sort_unstable();
                search.best = candidate; search.best_set = current.clone();
                if search.evaluations >= budget { break; }
            }
        }
        Selector::Beam => {
            let mut frontier = vec![Vec::<usize>::new()];
            for _ in 0..edits.len() {
                let mut next = Vec::new();
                for state in &frontier {
                    for index in 0..edits.len() {
                        if state.contains(&index) { continue; }
                        let mut proposal = state.clone(); proposal.push(index); proposal = normalized(proposal);
                        if search.cache.contains_key(&proposal) { continue; }
                        if let Some(candidate) = search.score(proposal.clone()) {
                            search.consider(proposal.clone());
                            next.push((proposal, candidate.metrics.aggregate_tstr));
                        }
                        if search.evaluations >= budget { break; }
                    }
                    if search.evaluations >= budget { break; }
                }
                next.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
                next.dedup_by(|left, right| left.0 == right.0);
                frontier = next.into_iter().take(2).map(|entry| entry.0).collect();
                if frontier.is_empty() || search.evaluations >= budget { break; }
            }
        }
        Selector::Exchange | Selector::Lookahead => {
            for left in 0..edits.len() {
                search.consider(vec![left]);
                for right in left + 1..edits.len() {
                    search.consider(vec![left, right]);
                    if selector == Selector::Lookahead && search.evaluations >= budget { break; }
                }
                if selector == Selector::Lookahead && search.evaluations >= budget { break; }
            }
            if selector == Selector::Exchange {
                let start = search.best_set.clone();
                for remove in &start {
                    for add in 0..edits.len() {
                        if !start.contains(&add) {
                            let mut proposal = start.clone(); proposal.retain(|item| item != remove); proposal.push(add);
                            search.consider(proposal);
                        }
                        if search.evaluations >= budget { break; }
                    }
                }
            }
        }
        Selector::IndependentSectionDp => {
            let mut by_section = std::collections::BTreeMap::<&str, Vec<usize>>::new();
            for (index, edit) in edits.iter().enumerate() { by_section.entry(&edit.section).or_default().push(index); }
            let baseline = search.best.metrics.aggregate_tstr;
            let empty_bytes = search.best.bytes;
            let capacity = byte_cap.saturating_sub(empty_bytes);
            let mut dp = std::collections::BTreeMap::from([(0usize, (0.0f64, Vec::new()))]);
            for indices in by_section.values() {
                let mut options = Vec::new();
                for &index in indices {
                    if let Some(candidate) = search.score(vec![index]) {
                        let utility = candidate.metrics.aggregate_tstr - baseline;
                        let bytes = candidate.bytes.saturating_sub(empty_bytes);
                        if utility > 0.0 && bytes <= capacity { options.push((index, bytes, utility)); }
                    }
                }
                let mut next = dp.clone();
                for (used, (utility, selected)) in &dp {
                    for &(index, cost, gain) in &options {
                        let total = used.saturating_add(cost);
                        if total <= capacity {
                            let mut proposal = selected.clone();
                            proposal.push(index);
                            let candidate = (utility + gain, proposal);
                            if next.get(&total).is_none_or(|prior| candidate.0 > prior.0 ||
                                (candidate.0 == prior.0 && candidate.1 < prior.1)) {
                                next.insert(total, candidate);
                            }
                        }
                    }
                }
                dp = next;
                if search.evaluations >= budget { break; }
            }
            let selected = dp.into_values().max_by(|left, right| left.0.total_cmp(&right.0)
                .then_with(|| right.1.cmp(&left.1))).map_or_else(Vec::new, |(_, selected)| selected);
            search.consider(selected);
        }
    }
    SelectorResult { selector, selected: search.best_set, metrics: search.best.metrics,
        whole_file_bytes: search.best.bytes, evaluations: search.evaluations,
        stopped_for_critical_tail: search.stopped }
}

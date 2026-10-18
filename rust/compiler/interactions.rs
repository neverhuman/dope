//! Development comparison of interaction support under one term-tariff budget.
//!
//! Coefficients are fit on each fold's complement; sign gates use a fixed zero
//! threshold. Widths 12, 24, 48, and adaptive share fixed term-field charges,
//! which do not measure a complete encoded kernel. Fold assignments hash the
//! input and target values, so these scores are development comparisons rather
//! than independent generalization estimates. This helper does not promote
//! production candidates or qualify an artifact tier.

use std::fmt;

const INDEX_BYTES: u32 = 2;
const THRESHOLD_BYTES: u32 = 4;
const COEFFICIENT_BYTES: u32 = 4;
const ADAPTIVE_GAIN: f64 = 0.05;
const MAX_PAIR_PARENTS: usize = 8;
const RANDOM_PAIRS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteractionError {
    InvalidShape,
    InvalidBudget,
}

impl fmt::Display for InteractionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidShape => "invalid_shape",
            Self::InvalidBudget => "invalid_budget",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposalKind {
    Rank,
    Product,
    Gate,
}

impl ProposalKind {
    pub fn bytes(self) -> u32 {
        match self {
            Self::Rank => INDEX_BYTES + THRESHOLD_BYTES + 2 * COEFFICIENT_BYTES,
            Self::Product => 2 * INDEX_BYTES + 2 * COEFFICIENT_BYTES,
            Self::Gate => 2 * INDEX_BYTES + 2 * THRESHOLD_BYTES + 2 * COEFFICIENT_BYTES,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InteractionProposal {
    pub kind: ProposalKind,
    pub left: u16,
    pub right: Option<u16>,
    pub held_out_mse: f64,
    pub bytes: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CrossFittedSupport {
    pub proposals: Vec<InteractionProposal>,
    pub fold_count: u8,
    pub train_counts: Vec<usize>,
    pub eval_counts: Vec<usize>,
    pub charged_bytes: u32,
    pub held_out_mse: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WidthReport {
    pub label: &'static str,
    pub terms: Vec<InteractionProposal>,
    pub charged_bytes: u32,
    pub held_out_mse: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureBudgetSearch {
    pub support: CrossFittedSupport,
    pub widths: Vec<WidthReport>,
    pub selected: usize,
    pub randomized_pairs: Vec<(u16, u16)>,
}

#[derive(Clone, Debug)]
pub struct ColumnMatrix {
    columns: Vec<Vec<f32>>,
    target: Vec<f32>,
}

impl ColumnMatrix {
    pub fn new(columns: Vec<Vec<f32>>, target: Vec<f32>) -> Result<Self, InteractionError> {
        if columns.is_empty() || target.len() < 4 {
            return Err(InteractionError::InvalidShape);
        }
        let rows = target.len();
        let same_shape = columns.iter().all(|column| column.len() == rows);
        let finite = columns.iter().flatten().all(|value| value.is_finite())
            && target.iter().all(|value| value.is_finite());
        if !same_shape || !finite || columns.len() > usize::from(u16::MAX) {
            return Err(InteractionError::InvalidShape);
        }
        Ok(Self { columns, target })
    }

    pub fn rows(&self) -> usize {
        self.target.len()
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }
}

pub fn feature_budget_search(
    matrix: &ColumnMatrix,
    byte_budget: u32,
    seed: u64,
) -> Result<FeatureBudgetSearch, InteractionError> {
    if byte_budget == 0 {
        return Err(InteractionError::InvalidBudget);
    }
    let folds = fold_labels(matrix)?;
    let fold_count = folds.iter().copied().max().unwrap_or(0) + 1;
    let (train_counts, eval_counts) = fold_counts(&folds, fold_count);
    let ranked = rank_columns(matrix, &folds, fold_count);
    let parents: Vec<u16> = ranked
        .iter()
        .take(MAX_PAIR_PARENTS)
        .map(|(index, _)| *index)
        .collect();
    let mut pair_keys = Vec::new();
    for left_at in 0..parents.len() {
        for right_at in left_at + 1..parents.len() {
            pair_keys.push(ordered_pair(parents[left_at], parents[right_at]));
        }
    }
    let randomized_pairs = propose_pairs(matrix.column_count(), seed, RANDOM_PAIRS, &pair_keys);
    let mut proposals = Vec::new();
    for (index, _) in &ranked {
        proposals.push(singleton(
            matrix,
            &folds,
            fold_count,
            ProposalKind::Rank,
            *index,
            None,
        ));
    }
    for (left, right) in pair_keys
        .iter()
        .copied()
        .chain(randomized_pairs.iter().copied())
    {
        proposals.push(singleton(
            matrix,
            &folds,
            fold_count,
            ProposalKind::Product,
            left,
            Some(right),
        ));
        proposals.push(singleton(
            matrix,
            &folds,
            fold_count,
            ProposalKind::Gate,
            left,
            Some(right),
        ));
    }
    proposals.sort_by(|left, right| {
        left.held_out_mse
            .total_cmp(&right.held_out_mse)
            .then(left.bytes.cmp(&right.bytes))
            .then(left.left.cmp(&right.left))
            .then(left.right.cmp(&right.right))
            .then((left.kind as u8).cmp(&(right.kind as u8)))
    });
    let specs = [
        ("12", 12, false),
        ("24", 24, false),
        ("48", 48, false),
        ("adaptive", 48, true),
    ];
    let mut widths = Vec::with_capacity(specs.len());
    for (label, max_terms, adaptive) in specs {
        let terms = if adaptive {
            take_adaptive(
                matrix,
                &folds,
                fold_count,
                &proposals,
                max_terms,
                byte_budget,
            )
        } else {
            take_prefix(&proposals, max_terms, byte_budget)
        };
        let charged_bytes = charged(&terms);
        let held_out_mse = joint_mse(matrix, &folds, fold_count, &terms);
        widths.push(WidthReport {
            label,
            terms,
            charged_bytes,
            held_out_mse,
        });
    }
    let selected = (0..widths.len())
        .min_by(|&left, &right| {
            widths[left]
                .held_out_mse
                .total_cmp(&widths[right].held_out_mse)
                .then(widths[left].charged_bytes.cmp(&widths[right].charged_bytes))
                .then(left.cmp(&right))
        })
        .unwrap_or(0);
    let chosen = &widths[selected];
    Ok(FeatureBudgetSearch {
        support: CrossFittedSupport {
            proposals: chosen.terms.clone(),
            fold_count,
            train_counts,
            eval_counts,
            charged_bytes: chosen.charged_bytes,
            held_out_mse: chosen.held_out_mse,
        },
        widths,
        selected,
        randomized_pairs,
    })
}

fn ordered_pair(left: u16, right: u16) -> (u16, u16) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
}

fn charged(terms: &[InteractionProposal]) -> u32 {
    terms
        .iter()
        .fold(0, |sum, term| sum.saturating_add(term.bytes))
}

fn fold_labels(matrix: &ColumnMatrix) -> Result<Vec<u8>, InteractionError> {
    let mut folds = if matrix.rows() >= 64 { 4 } else { 2 };
    loop {
        let labels: Vec<u8> = (0..matrix.rows())
            .map(|row| row_fold(matrix, row, folds))
            .collect();
        let mut counts = vec![0usize; usize::from(folds)];
        for label in &labels {
            counts[usize::from(*label)] += 1;
        }
        if counts.iter().all(|count| *count > 0) {
            return Ok(labels);
        }
        if folds == 2 {
            return Err(InteractionError::InvalidShape);
        }
        folds -= 1;
    }
}

fn row_fold(matrix: &ColumnMatrix, row: usize, folds: u8) -> u8 {
    let mut hash = 0xcbf29ce484222325u64;
    for column in &matrix.columns {
        hash ^= u64::from(column[row].to_bits());
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash ^= u64::from(matrix.target[row].to_bits());
    hash = hash.wrapping_mul(0x100000001b3);
    (splitmix(hash) % u64::from(folds)) as u8
}

fn fold_counts(labels: &[u8], fold_count: u8) -> (Vec<usize>, Vec<usize>) {
    let mut eval_counts = vec![0usize; usize::from(fold_count)];
    for label in labels {
        eval_counts[usize::from(*label)] += 1;
    }
    let rows = labels.len();
    let train_counts = eval_counts.iter().map(|eval| rows - eval).collect();
    (train_counts, eval_counts)
}

fn rank_columns(matrix: &ColumnMatrix, labels: &[u8], fold_count: u8) -> Vec<(u16, f64)> {
    let mut ranked = Vec::with_capacity(matrix.column_count());
    for index in 0..matrix.column_count() {
        let mut total = 0.0;
        for fold in 0..fold_count {
            let rows = rows_in(labels, fold, false);
            let values = gather(matrix, ProposalKind::Rank, index as u16, None, &rows);
            let target = gather_target(matrix, &rows);
            total += abs_correlation(&values, &target);
        }
        ranked.push((index as u16, total / f64::from(fold_count)));
    }
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    ranked
}

fn singleton(
    matrix: &ColumnMatrix,
    labels: &[u8],
    fold_count: u8,
    kind: ProposalKind,
    left: u16,
    right: Option<u16>,
) -> InteractionProposal {
    let proposal = InteractionProposal {
        kind,
        left,
        right,
        held_out_mse: 0.0,
        bytes: kind.bytes(),
    };
    let held_out_mse = joint_mse(matrix, labels, fold_count, &[proposal.clone()]);
    InteractionProposal {
        held_out_mse,
        ..proposal
    }
}

fn take_prefix(
    sorted: &[InteractionProposal],
    max_terms: usize,
    budget: u32,
) -> Vec<InteractionProposal> {
    let mut chosen = Vec::new();
    let mut bytes = 0u32;
    for proposal in sorted {
        if chosen.len() == max_terms {
            break;
        }
        let Some(next) = bytes.checked_add(proposal.bytes) else {
            continue;
        };
        if next > budget {
            continue;
        }
        bytes = next;
        chosen.push(proposal.clone());
    }
    chosen
}

fn take_adaptive(
    matrix: &ColumnMatrix,
    labels: &[u8],
    fold_count: u8,
    sorted: &[InteractionProposal],
    max_terms: usize,
    budget: u32,
) -> Vec<InteractionProposal> {
    let mut chosen = Vec::new();
    let mut bytes = 0u32;
    let mut current = joint_mse(matrix, labels, fold_count, &chosen);
    let mut used = vec![false; sorted.len()];
    // Ranks take budget before pairwise gates. A sign gate of two additive
    // signals otherwise wins the first cross-fit step, and the remaining
    // main effects never clear the gain.
    for ranks_only in [true, false] {
        while chosen.len() < max_terms {
            let mut best: Option<(usize, u32, f64)> = None;
            for (index, proposal) in sorted.iter().enumerate() {
                let is_rank = proposal.kind == ProposalKind::Rank;
                if used[index] || is_rank != ranks_only {
                    continue;
                }
                if !ranks_only {
                    let right = proposal.right.unwrap_or(proposal.left);
                    let kept = |column: u16| {
                        chosen
                            .iter()
                            .any(|term| term.kind == ProposalKind::Rank && term.left == column)
                    };
                    if !kept(proposal.left) || !kept(right) {
                        continue;
                    }
                }
                let Some(next_bytes) = bytes.checked_add(proposal.bytes) else {
                    continue;
                };
                if next_bytes > budget {
                    continue;
                }
                let mut trial = chosen.clone();
                trial.push(proposal.clone());
                let mse = joint_mse(matrix, labels, fold_count, &trial);
                let improves = current - mse >= ADAPTIVE_GAIN;
                let better = best.map(|(_, _, best_mse)| mse < best_mse).unwrap_or(true);
                if improves && better {
                    best = Some((index, next_bytes, mse));
                }
            }
            let Some((index, next_bytes, mse)) = best else {
                break;
            };
            used[index] = true;
            bytes = next_bytes;
            chosen.push(sorted[index].clone());
            current = mse;
        }
    }
    chosen
}

include!("interactions/support_fit.rs");

fn propose_pairs(
    column_count: usize,
    seed: u64,
    limit: usize,
    existing: &[(u16, u16)],
) -> Vec<(u16, u16)> {
    if column_count < 2 || limit == 0 {
        return Vec::new();
    }
    let mut state = splitmix(seed);
    let mut found = Vec::new();
    let mut guard = 0usize;
    while found.len() < limit && guard < 64 {
        guard += 1;
        state = xorshift(state);
        let left = (state % column_count as u64) as u16;
        state = xorshift(state);
        let right = (state % column_count as u64) as u16;
        if left == right {
            continue;
        }
        let key = ordered_pair(left, right);
        if existing.contains(&key) || found.contains(&key) {
            continue;
        }
        found.push(key);
    }
    found
}

fn splitmix(seed: u64) -> u64 {
    let mut mixed = seed.wrapping_add(0x9E3779B97F4A7C15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D049BB133111EB);
    mixed ^ (mixed >> 31)
}

fn xorshift(mut state: u64) -> u64 {
    state ^= state << 7;
    state ^= state >> 9;
    state ^= state << 8;
    state
}

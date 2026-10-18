//! Deterministic, fixture-only study primitives for diverse out-of-fold teachers.
//! These do not train product models or admit a transfer mechanism.

use std::collections::BTreeSet;

const MANIFEST: &str = include_str!("../../../research/transfer/ablations/matched_teacher.json");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TargetKind {
    Mean,
    Probability,
    Residual,
    GroupResponse,
}

impl TargetKind {
    fn wire(self) -> u8 {
        match self {
            Self::Mean => 0,
            Self::Probability => 1,
            Self::Residual => 2,
            Self::GroupResponse => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TeacherCount {
    None,
    Single,
    Multiple,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CostPlan {
    student_fit: u32,
    teacher_fit: u32,
    selection: u32,
}

impl CostPlan {
    fn equal_total(count: TeacherCount) -> Self {
        let teacher_fit = match count {
            TeacherCount::None => 0,
            TeacherCount::Single => 12,
            TeacherCount::Multiple => 24,
        };
        Self {
            student_fit: 120 - teacher_fit,
            teacher_fit,
            selection: 24,
        }
    }

    fn total(self) -> u32 {
        self.student_fit + self.teacher_fit + self.selection
    }
}

fn fold_for(row: usize, folds: usize, seed: u64) -> usize {
    // Stable arithmetic partition: no hash-map or runtime RNG dependency.
    (row.wrapping_mul(17).wrapping_add(seed as usize)) % folds
}

fn oof_mean(labels: &[f64], row: usize, folds: usize, seed: u64) -> f64 {
    let heldout = fold_for(row, folds, seed);
    let mut sum = 0.0;
    let mut count = 0usize;
    for (index, label) in labels.iter().enumerate() {
        if fold_for(index, folds, seed) != heldout {
            sum += label;
            count += 1;
        }
    }
    assert!(count > 0);
    sum / count as f64
}

fn canonical_target(kind: TargetKind, values: &[f64]) -> Vec<u8> {
    let mut bytes = vec![kind.wire()];
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn fit_teacher_artifact(labels: &[f64], count: TeacherCount, seed: u64) -> Vec<u8> {
    let teacher_count = match count {
        TeacherCount::None => 0,
        TeacherCount::Single => 1,
        TeacherCount::Multiple => 2,
    };
    let mut bytes = vec![1, teacher_count, seed as u8];
    for _ in 0..teacher_count {
        let folds = 3;
        for row in 0..labels.len() {
            bytes.extend_from_slice(&oof_mean(labels, row, folds, seed).to_bits().to_le_bytes());
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_preregisters_holdouts_cost_seeds_and_stop_rule() {
        let value: serde_json::Value = serde_json::from_str(MANIFEST).unwrap();
        let study = &value["study"];
        assert_eq!(study["format"], "dope-diverse-oof-teacher-study-v1");
        assert_eq!(study["candidate_set"].as_array().unwrap().len(), 3);
        assert_eq!(study["cost"]["total_fit_and_selection_units_per_candidate"], 144);
        assert_eq!(study["cost"]["max_teacher_state_bytes"], 16_384);
        assert_eq!(study["cost"]["max_total_study_operations"], 1_728);
        assert_eq!(study["seeds"]["pilot"][0], 11);
        assert_eq!(study["seeds"]["development"].as_array().unwrap().len(), 3);
        assert_eq!(study["exposed_auditors_confirmatory_reuse"], false);
        assert!(study["stop_rule"].as_str().unwrap().contains("Stop immediately"));
        let source_holdouts = study["heldout_source_families"].as_array().unwrap();
        assert_eq!(source_holdouts.len(), 2);
        assert!(source_holdouts.iter().any(|source| source == "pmlb_lineage"));
        assert_eq!(study["heldout_teacher_architectures"][0], "tabular_diffusion");
        assert_eq!(study["mechanism_wiring"], "none_until_all_admission_gates_pass_and_a_candidate_wins; negative_or_unsupported_result_is_retained");
    }

    #[test]
    fn teacher_counts_reconcile_equal_total_fit_and_selection_cost() {
        let plans = [TeacherCount::None, TeacherCount::Single, TeacherCount::Multiple]
            .map(CostPlan::equal_total);
        assert!(plans.iter().all(|plan| plan.total() == 144));
        assert!(plans.windows(2).all(|pair| pair[0].student_fit > pair[1].student_fit));
        assert!(plans.iter().all(|plan| plan.selection == 24));
    }

    #[test]
    fn one_priced_pilot_then_three_seed_development_fixture_has_no_campaign() {
        let labels = [0.1, 0.4, 0.2, 0.8, 0.7, 0.3, 0.9, 0.5, 0.6, 0.15, 0.35, 0.95];
        let candidates = [TeacherCount::None, TeacherCount::Single, TeacherCount::Multiple];
        let pilot_units: u32 = candidates.iter().map(|kind| CostPlan::equal_total(*kind).total()).sum();
        assert_eq!(pilot_units, 432);
        let pilot_bytes = candidates
            .iter()
            .map(|kind| fit_teacher_artifact(&labels, *kind, 11).len())
            .max()
            .unwrap();
        assert!(pilot_bytes <= 16_384);
        let mut development_runs = 0;
        let mut development_units = 0;
        let mut development_max_bytes = 0;
        for seed in [11, 23, 37] {
            for kind in candidates {
                let artifact = fit_teacher_artifact(&labels, kind, seed);
                assert!(artifact.len() <= 16_384);
                assert_eq!(CostPlan::equal_total(kind).total(), 144);
                development_units += CostPlan::equal_total(kind).total();
                development_max_bytes = development_max_bytes.max(artifact.len());
                development_runs += 1;
            }
        }
        assert_eq!(development_runs, 9);
        assert_eq!(development_units, 1_296);
        assert_eq!(development_max_bytes, 195);
        // This fixture has no admitted external holdout runtimes, so it reports
        // no efficacy winner and never starts an automated campaign.
        let manifest: serde_json::Value = serde_json::from_str(MANIFEST).unwrap();
        assert_eq!(manifest["study"]["pilot_record"]["priced_units"], pilot_units);
        assert_eq!(manifest["study"]["pilot_record"]["max_teacher_state_bytes_observed"], pilot_bytes);
        assert_eq!(pilot_bytes, 195);
        assert_eq!(manifest["study"]["development_record"]["priced_operations"], development_units);
        assert_eq!(manifest["study"]["development_record"]["winner"], serde_json::Value::Null);
        assert_eq!(manifest["study"]["campaign"], "none; no automatic campaign or external data access");
    }

    #[test]
    fn poisoned_label_cannot_change_its_own_oof_teacher_target() {
        let labels = [0.1, 0.4, 0.2, 0.8, 0.7, 0.3, 0.9, 0.5, 0.6, 0.15, 0.35, 0.95];
        let poison_row = 5;
        let mut poisoned = labels;
        poisoned[poison_row] = 1_000_000.0;
        let before = oof_mean(&labels, poison_row, 3, 11);
        let after = oof_mean(&poisoned, poison_row, 3, 11);
        assert_eq!(before.to_bits(), after.to_bits());
        // The same label is available to other folds, proving the poison fixture
        // would expose leakage if the held-out fold were included.
        assert_ne!(oof_mean(&labels, 0, 3, 11), oof_mean(&poisoned, 0, 3, 11));
    }

    #[test]
    fn teacher_target_kinds_remain_distinct_and_artifacts_are_bounded() {
        let values = [0.2, 0.5, 0.7];
        let encodings = [
            TargetKind::Mean,
            TargetKind::Probability,
            TargetKind::Residual,
            TargetKind::GroupResponse,
        ]
        .map(|kind| canonical_target(kind, &values));
        assert_eq!(encodings.iter().map(|bytes| bytes[0]).collect::<BTreeSet<_>>().len(), 4);
        assert!(encodings.windows(2).all(|pair| pair[0] != pair[1]));
        for seed in [11, 23, 37] {
            let labels = [0.1, 0.4, 0.2, 0.8, 0.7, 0.3, 0.9, 0.5, 0.6, 0.15, 0.35, 0.95];
            for count in [TeacherCount::None, TeacherCount::Single, TeacherCount::Multiple] {
                let artifact = fit_teacher_artifact(&labels, count, seed);
                assert!(artifact.len() <= 16_384);
                assert_eq!(CostPlan::equal_total(count).total(), 144);
            }
        }
    }
}


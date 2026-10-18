//! Minimum observed qualified bytes and full-roster cumulative availability.
//!
//! `qualified` is the evaluator's joint T90/D80/integrity/task-gate decision
//! for one frozen method/seed. This arithmetic does not authenticate or calibrate
//! that decision, certify a release, or establish a global minimum.

use std::collections::BTreeMap;

use super::MetricError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactTrial {
    pub task_index: usize,
    pub search_cap_bytes: u64,
    pub artifact_bytes: Option<u64>,
    pub qualified: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RosterCoverage {
    pub minimum_bytes: Vec<Option<u64>>,
    pub qualified_tasks: usize,
    pub admitted_tasks: usize,
    pub coverage: f64,
}

fn validate(trials: &[ArtifactTrial]) -> Result<(), MetricError> {
    for trial in trials {
        if trial.search_cap_bytes == 0
            || trial
                .artifact_bytes
                .is_some_and(|bytes| bytes == 0 || bytes > trial.search_cap_bytes)
            || (trial.qualified && trial.artifact_bytes.is_none())
        {
            return Err(MetricError::InvalidEvidence);
        }
    }
    Ok(())
}

/// A raw search-cap success followed by a cap with no success is non-monotone.
/// Multiple precommitted candidates at one cap are combined by availability.
pub fn non_monotone(trials: &[ArtifactTrial], task_index: usize) -> bool {
    let mut caps = BTreeMap::new();
    for trial in trials.iter().filter(|trial| trial.task_index == task_index) {
        *caps.entry(trial.search_cap_bytes).or_insert(false) |= trial.qualified;
    }
    let mut seen_success = false;
    for qualified in caps.values() {
        if seen_success && !qualified {
            return true;
        }
        seen_success |= qualified;
    }
    false
}

/// Minimize charged artifact bytes, independent of discovery order/search cap.
/// Reuse survives failed larger searches unless a monotone regime is required.
pub fn minimum_qualified_bytes(
    trials: &[ArtifactTrial],
    task_index: usize,
    byte_cap: u64,
    require_monotone: bool,
) -> Result<Option<u64>, MetricError> {
    validate(trials)?;
    if require_monotone && non_monotone(trials, task_index) {
        return Ok(None);
    }
    Ok(trials
        .iter()
        .filter(|trial| trial.task_index == task_index && trial.qualified)
        .filter_map(|trial| trial.artifact_bytes)
        .filter(|bytes| *bytes <= byte_cap)
        .min())
}

/// The count is the precommitted roster size, never the number of observed rows.
/// Missing, failed, unsupported and unevaluated tasks contribute zero.
pub fn roster_coverage(
    admitted_tasks: usize,
    trials: &[ArtifactTrial],
    byte_cap: u64,
    require_monotone: bool,
) -> Result<RosterCoverage, MetricError> {
    if admitted_tasks == 0
        || trials
            .iter()
            .any(|trial| trial.task_index >= admitted_tasks)
    {
        return Err(MetricError::InvalidEvidence);
    }
    let minimum_bytes = (0..admitted_tasks)
        .map(|task| minimum_qualified_bytes(trials, task, byte_cap, require_monotone))
        .collect::<Result<Vec<_>, _>>()?;
    let qualified_tasks = minimum_bytes.iter().filter(|bytes| bytes.is_some()).count();
    Ok(RosterCoverage {
        minimum_bytes,
        qualified_tasks,
        admitted_tasks,
        coverage: qualified_tasks as f64 / admitted_tasks as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial(task_index: usize, cap: u64, bytes: Option<u64>, qualified: bool) -> ArtifactTrial {
        ArtifactTrial {
            task_index,
            search_cap_bytes: cap,
            artifact_bytes: bytes,
            qualified,
        }
    }

    #[test]
    fn minimum_uses_bytes_even_when_smaller_artifact_is_found_later() {
        let mut trials = vec![
            trial(0, 1000, Some(900), true),
            trial(0, 1000, Some(800), true),
        ];
        assert_eq!(
            minimum_qualified_bytes(&trials, 0, 1000, false),
            Ok(Some(800))
        );
        trials.reverse();
        assert_eq!(
            minimum_qualified_bytes(&trials, 0, 800, false),
            Ok(Some(800))
        );
        assert_eq!(minimum_qualified_bytes(&trials, 0, 799, false), Ok(None));
        trials.push(trial(0, 2000, Some(700), true));
        assert_eq!(
            minimum_qualified_bytes(&trials, 0, 700, false),
            Ok(Some(700))
        );
    }

    #[test]
    fn later_failure_does_not_erase_an_artifact() {
        let trials = vec![trial(0, 900, Some(900), true), trial(0, 1200, None, false)];
        for cap in [900, 1200, 2000] {
            assert_eq!(
                minimum_qualified_bytes(&trials, 0, cap, false),
                Ok(Some(900))
            );
        }
        assert!(non_monotone(&trials, 0));
        assert_eq!(minimum_qualified_bytes(&trials, 0, 2000, true), Ok(None));
    }

    #[test]
    fn full_roster_counts_missing_failed_and_unevaluated_tasks() {
        let trials = vec![trial(0, 1000, Some(900), true), trial(1, 1200, None, false)];
        let result = roster_coverage(5, &trials, 1000, false).unwrap();
        assert_eq!(
            result.minimum_bytes,
            vec![Some(900), None, None, None, None]
        );
        assert_eq!(
            (
                result.qualified_tasks,
                result.admitted_tasks,
                result.coverage
            ),
            (1, 5, 0.2)
        );
        assert_eq!(roster_coverage(5, &[], 1000, false).unwrap().coverage, 0.0);
        assert!(roster_coverage(1, &trials, 1000, false).is_err());
        assert!(roster_coverage(0, &[], 1000, false).is_err());
    }

    #[test]
    fn malformed_trial_evidence_is_rejected() {
        for invalid in [
            trial(0, 100, None, true),
            trial(0, 100, Some(101), true),
            trial(0, 0, None, false),
            trial(0, 100, Some(0), true),
        ] {
            assert_eq!(
                minimum_qualified_bytes(&[invalid], 0, 100, false),
                Err(MetricError::InvalidEvidence)
            );
        }
    }

    #[test]
    fn metrics_frontier_v2_golden_parity_and_redaction() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/frontier");
        let mut count = 0;
        for entry in std::fs::read_dir(root).unwrap() {
            let path = entry.unwrap().path();
            if !path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("v2_")
            {
                continue;
            }
            let case: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let actual = crate::metrics::frontier::evaluate(&case["input"]);
            assert_eq!(actual, case["expected"], "{}", case["case_id"]);
            let mut invalid = case["input"].clone();
            invalid["cells"] = serde_json::json!([{"task_id":"PRIVATE_SOURCE_HEADER_EXTREMA"}]);
            let failed = crate::metrics::frontier::evaluate(&invalid);
            assert_eq!(failed["status"], "failed");
            assert!(!failed.to_string().contains("PRIVATE_SOURCE_HEADER_EXTREMA"));
            count += 1;
        }
        assert!(count >= 3);
    }
}

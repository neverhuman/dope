

impl DeepMetrics {
    fn finite(&self) -> bool {
        [
            self.null_loss,
            self.trtr_loss,
            self.tstr_loss,
            self.bounded_retention,
            self.absolute_lift,
            self.regret,
            self.oracle_recall,
            self.calibration_degradation,
            self.query_p95_error,
            self.type_i_error,
            self.membership_auc,
            self.attribute_inference_advantage,
            self.driver_agreement,
            self.joint_fidelity,
            self.rare_tail_retention,
            self.supported_subgroup_retention,
            self.nominal_coverage,
            self.feature_importance_spearman,
            self.feature_importance_top_k_agreement,
            self.feature_importance_informative_groups,
        ]
        .into_iter()
        .flatten()
        .all(f64::is_finite)
    }

    fn add_assign(&mut self, other: &Self) {
        let add = |left: &mut Option<f64>, right: Option<f64>| {
            *left = left.zip(right).map(|(left, right)| left + right);
        };
        add(&mut self.null_loss, other.null_loss);
        add(&mut self.trtr_loss, other.trtr_loss);
        add(&mut self.tstr_loss, other.tstr_loss);
        add(&mut self.bounded_retention, other.bounded_retention);
        add(&mut self.absolute_lift, other.absolute_lift);
        add(&mut self.regret, other.regret);
        add(&mut self.oracle_recall, other.oracle_recall);
        add(
            &mut self.calibration_degradation,
            other.calibration_degradation,
        );
        add(&mut self.query_p95_error, other.query_p95_error);
        add(&mut self.type_i_error, other.type_i_error);
        add(&mut self.membership_auc, other.membership_auc);
        add(
            &mut self.attribute_inference_advantage,
            other.attribute_inference_advantage,
        );
        add(&mut self.driver_agreement, other.driver_agreement);
        add(&mut self.joint_fidelity, other.joint_fidelity);
        add(&mut self.rare_tail_retention, other.rare_tail_retention);
        add(
            &mut self.supported_subgroup_retention,
            other.supported_subgroup_retention,
        );
        add(&mut self.nominal_coverage, other.nominal_coverage);
        add(
            &mut self.feature_importance_spearman,
            other.feature_importance_spearman,
        );
        add(
            &mut self.feature_importance_top_k_agreement,
            other.feature_importance_top_k_agreement,
        );
        add(
            &mut self.feature_importance_informative_groups,
            other.feature_importance_informative_groups,
        );
        let add_count = |left: &mut Option<u64>, right: Option<u64>| {
            *left = left.zip(right).map(|(left, right)| left + right);
        };
        add_count(&mut self.exact_copies, other.exact_copies);
        add_count(&mut self.near_copies, other.near_copies);
        add_count(&mut self.canary_extractions, other.canary_extractions);
        add_count(&mut self.lineage_leaks, other.lineage_leaks);
    }

    fn divide(&mut self, denominator: f64) {
        for value in [
            &mut self.null_loss,
            &mut self.trtr_loss,
            &mut self.tstr_loss,
            &mut self.bounded_retention,
            &mut self.absolute_lift,
            &mut self.regret,
            &mut self.oracle_recall,
            &mut self.calibration_degradation,
            &mut self.query_p95_error,
            &mut self.type_i_error,
            &mut self.membership_auc,
            &mut self.attribute_inference_advantage,
            &mut self.driver_agreement,
            &mut self.joint_fidelity,
            &mut self.rare_tail_retention,
            &mut self.supported_subgroup_retention,
            &mut self.nominal_coverage,
            &mut self.feature_importance_spearman,
            &mut self.feature_importance_top_k_agreement,
            &mut self.feature_importance_informative_groups,
        ]
        .into_iter()
        .flatten()
        {
            *value /= denominator;
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DeepOutcome {
    pub task: String,
    pub lineage_group_id: String,
    pub structural_profile: String,
    pub candidate_id: String,
    pub configuration: Option<LossConfiguration>,
    pub generation_seed: u64,
    pub auditor_seed: u64,
    pub auditor_id: String,
    pub size_multiplier: usize,
    pub state: EvidenceState,
    /// Must be absent unless state is `succeeded`.
    pub metrics: Option<DeepMetrics>,
    #[serde(default)]
    pub failure_reason: Option<String>,
    pub artifact_bytes: Option<u64>,
    pub runtime_ms: Option<u64>,
    pub fitting_time_ms: Option<u64>,
    pub sampling_time_ms: Option<u64>,
    pub auditor_time_ms: Option<u64>,
    pub peak_cpu_memory_bytes: Option<u64>,
    pub peak_gpu_memory_bytes: Option<u64>,
    pub artifact_cache_status: Option<CacheStatus>,
    pub real_auditor_cache_status: Option<CacheStatus>,
    pub ancillary_cache_status: Option<CacheStatus>,
    pub invalid_rows: usize,
    pub schema_violations: usize,
    pub nondeterministic_output: bool,
}

impl DeepOutcome {
    pub fn validate(&self) -> Result<()> {
        if (self.state == EvidenceState::Succeeded)
            != (self.metrics.is_some()
                && self.artifact_bytes.is_some()
                && self.runtime_ms.is_some()
                && self.fitting_time_ms.is_some()
                && self.sampling_time_ms.is_some()
                && self.auditor_time_ms.is_some()
                && self.peak_cpu_memory_bytes.is_some()
                && self.artifact_cache_status.is_some()
                && self.real_auditor_cache_status.is_some()
                && self.ancillary_cache_status.is_some()
                && self.failure_reason.is_none())
            || (self.state != EvidenceState::Succeeded
                && self.failure_reason.as_deref().is_none_or(str::is_empty))
            || self
                .metrics
                .as_ref()
                .is_some_and(|metrics| !metrics.finite())
            || self.artifact_bytes.is_some_and(|bytes| bytes == 0)
            || !matches!(self.task.as_str(), "binary" | "regression")
        {
            return Err(DopeError::Data(
                "deep outcome has an inconsistent measured/failed state".into(),
            ));
        }
        Ok(())
    }
}

fn bounded_retention(evidence: &JobEvidence) -> f64 {
    let improvement = evidence.null_loss - evidence.trtr_loss;
    if improvement >= 0.01 * evidence.null_loss.abs() && improvement.abs() > f64::EPSILON {
        ((evidence.null_loss - evidence.tstr_loss) / improvement).clamp(0.0, 1.0)
    } else {
        f64::from(evidence.tstr_loss <= evidence.trtr_loss + 0.01 * evidence.null_loss.abs())
    }
}

fn failure_state(error: &str) -> EvidenceState {
    let error = error.to_ascii_lowercase();
    if error.contains("timeout") || error.contains("deadline") {
        EvidenceState::TimedOut
    } else if error.contains("out of memory") || error.contains("oom") {
        EvidenceState::Oom
    } else if error.contains("malformed") || error.contains("schema") || error.contains("corrupt") {
        EvidenceState::Malformed
    } else {
        EvidenceState::Failed
    }
}

fn block_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for path in paths {
        if path.is_dir() {
            for entry in fs::read_dir(path).map_err(|error| crate::error::io_error(path, error))? {
                let entry = entry.map_err(|error| crate::error::io_error(path, error))?;
                let candidate = entry.path();
                if candidate.extension().and_then(|value| value.to_str()) == Some("json") {
                    files.push(candidate);
                }
            }
        } else {
            files.push(path.clone());
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

type DeepOutcomeKey = (String, String, String, String, String, usize, u64, u64);
type DeepOutcomeMap = BTreeMap<DeepOutcomeKey, DeepOutcome>;

/// Reconstructs seed-addressed deep evidence from durable gold blocks. Failed
/// cells retain their exact matrix coordinates and every derived comparison is
/// computed from measured peers in the same cell.
pub fn rebuild_deep_outcomes(
    cohort: &CohortPlan,
    block_paths: &[PathBuf],
) -> Result<Vec<DeepOutcome>> {
    let records = cohort
        .records
        .iter()
        .map(|record| (record.dataset_id.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    let mut outcomes = DeepOutcomeMap::new();
    for path in block_files(block_paths)? {
        let block: GoldRecordBlock = read_json(&path)?;
        if block.format != "dope-gold-record-block" || block.version != 2 {
            return Err(DopeError::Data(format!(
                "invalid gold block for deep evidence: {}",
                path.display()
            )));
        }
        let record = records
            .get(block.dataset_id.as_str())
            .ok_or_else(|| DopeError::Data("gold block dataset is absent from cohort".into()))?;
        let configuration = LossConfiguration {
            target_weight: block.neural_target_weight,
            structural_penalty: block.neural_structural_penalty,
        };
        if !LossConfiguration::GRID.contains(&configuration) {
            return Err(DopeError::Data(
                "gold block contains an unfrozen neural loss configuration".into(),
            ));
        }
        for evidence in block.evidence {
            if !(2..=JOB_EVIDENCE_VERSION).contains(&evidence.version) {
                return Err(DopeError::Data(
                    "gold block mixes superseded job evidence with the neural campaign".into(),
                ));
            }
            let is_new = NEW_CANDIDATES.contains(&evidence.candidate_id.as_str());
            let metrics = DeepMetrics {
                null_loss: Some(evidence.null_loss),
                trtr_loss: Some(evidence.trtr_loss),
                tstr_loss: Some(evidence.tstr_loss),
                bounded_retention: Some(bounded_retention(&evidence)),
                absolute_lift: None,
                regret: None,
                oracle_recall: None,
                calibration_degradation: evidence.calibration_degradation,
                query_p95_error: evidence.query_p95_normalized_error,
                type_i_error: evidence.type_i_error,
                membership_auc: evidence.membership_auc,
                attribute_inference_advantage: evidence.attribute_inference_advantage,
                driver_agreement: evidence.driver_agreement,
                joint_fidelity: evidence.joint_fidelity,
                rare_tail_retention: evidence.rare_class_or_tail_retention,
                supported_subgroup_retention: evidence.supported_subgroup_retention,
                nominal_coverage: evidence.nominal_95_coverage,
                feature_importance_spearman: evidence.feature_importance_spearman,
                feature_importance_top_k_agreement: Some(
                    evidence.feature_importance_top_k_agreement,
                ),
                feature_importance_informative_groups: Some(
                    evidence.feature_importance_informative_count as f64,
                ),
                exact_copies: Some(evidence.exact_copies as u64),
                near_copies: Some(evidence.near_copies as u64),
                canary_extractions: Some(evidence.canary_extractions as u64),
                lineage_leaks: Some(evidence.lineage_leaks as u64),
            };
            let outcome = DeepOutcome {
                task: evidence.task,
                lineage_group_id: evidence.lineage_group_id,
                structural_profile: evidence.structural_profile,
                candidate_id: evidence.candidate_id,
                configuration: is_new.then_some(configuration),
                generation_seed: evidence.generation_seed,
                auditor_seed: evidence.auditor_seed,
                auditor_id: evidence.auditor_id,
                size_multiplier: evidence.size_multiplier,
                state: EvidenceState::Succeeded,
                metrics: Some(metrics),
                failure_reason: None,
                artifact_bytes: Some(evidence.artifact_bytes),
                runtime_ms: Some(evidence.runtime_ms),
                fitting_time_ms: Some(evidence.fitting_time_ms),
                sampling_time_ms: Some(evidence.sampling_time_ms),
                auditor_time_ms: Some(evidence.auditor_time_ms),
                peak_cpu_memory_bytes: Some(evidence.peak_cpu_memory_bytes),
                peak_gpu_memory_bytes: evidence.peak_gpu_memory_bytes,
                artifact_cache_status: Some(evidence.artifact_cache_status),
                real_auditor_cache_status: Some(evidence.real_auditor_cache_status),
                ancillary_cache_status: Some(evidence.ancillary_cache_status),
                invalid_rows: evidence.invalid_rows,
                schema_violations: evidence.schema_violations,
                nondeterministic_output: evidence.nondeterministic_output,
            };
            insert_deep_outcome(&mut outcomes, outcome)?;
        }
        for failure in block.failures {
            let size_multiplier = failure.size_multiplier.ok_or_else(|| {
                DopeError::Data("gold failure is missing its size multiplier".into())
            })?;
            let generation_seed = failure.generation_seed.ok_or_else(|| {
                DopeError::Data("gold failure is missing its generation seed".into())
            })?;
            let auditor_seed = failure.auditor_seed.ok_or_else(|| {
                DopeError::Data("gold failure is missing its auditor seed".into())
            })?;
            let is_new = NEW_CANDIDATES.contains(&failure.candidate_id.as_str());
            let outcome = DeepOutcome {
                task: record.task.clone(),
                lineage_group_id: record.lineage_group_id.clone(),
                structural_profile: record.structural_profile.clone(),
                candidate_id: failure.candidate_id,
                configuration: is_new.then_some(configuration),
                generation_seed,
                auditor_seed,
                auditor_id: failure.auditor_id,
                size_multiplier,
                state: failure_state(&failure.error),
                metrics: None,
                failure_reason: Some(failure.error),
                artifact_bytes: None,
                runtime_ms: None,
                fitting_time_ms: None,
                sampling_time_ms: None,
                auditor_time_ms: None,
                peak_cpu_memory_bytes: None,
                peak_gpu_memory_bytes: None,
                artifact_cache_status: None,
                real_auditor_cache_status: None,
                ancillary_cache_status: None,
                invalid_rows: 0,
                schema_violations: 0,
                nondeterministic_output: false,
            };
            insert_deep_outcome(&mut outcomes, outcome)?;
        }
    }
    let mut outcomes = outcomes.into_values().collect::<Vec<_>>();
    add_cell_comparisons(&mut outcomes);
    outcomes.sort_by(|left, right| {
        left.lineage_group_id
            .cmp(&right.lineage_group_id)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
            .then_with(|| left.auditor_id.cmp(&right.auditor_id))
            .then_with(|| left.size_multiplier.cmp(&right.size_multiplier))
            .then_with(|| left.generation_seed.cmp(&right.generation_seed))
            .then_with(|| left.auditor_seed.cmp(&right.auditor_seed))
    });
    Ok(outcomes)
}

fn insert_deep_outcome(outcomes: &mut DeepOutcomeMap, outcome: DeepOutcome) -> Result<()> {
    let configuration = outcome
        .configuration
        .map(LossConfiguration::id)
        .unwrap_or_default();
    let key = (
        outcome.lineage_group_id.clone(),
        outcome.structural_profile.clone(),
        outcome.candidate_id.clone(),
        configuration,
        outcome.auditor_id.clone(),
        outcome.size_multiplier,
        outcome.generation_seed,
        outcome.auditor_seed,
    );
    if let Some(existing) = outcomes.insert(key, outcome.clone())
        && existing != outcome
    {
        return Err(DopeError::Data(
            "duplicate gold cells contain inconsistent deep evidence".into(),
        ));
    }
    Ok(())
}

include!("cell_comparison.rs");
include!("slice_reports.rs");

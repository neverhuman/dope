
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DeepSliceReport {
    pub dimension: String,
    pub key: String,
    pub candidate_id: String,
    pub attempted_cells: usize,
    pub succeeded_cells: usize,
    pub failed_cells: usize,
    pub timed_out_cells: usize,
    pub lineage_groups: usize,
    pub metrics: DeepMetrics,
    pub bounded_retention_interval: Option<PairedInterval>,
    pub absolute_lift_interval: Option<PairedInterval>,
    pub mean_runtime_ms: Option<f64>,
    pub mean_fitting_time_ms: Option<f64>,
    pub mean_sampling_time_ms: Option<f64>,
    pub mean_auditor_time_ms: Option<f64>,
    pub peak_cpu_memory_bytes: Option<u64>,
    pub peak_gpu_memory_bytes: Option<u64>,
    pub maximum_artifact_bytes: Option<u64>,
    pub artifact_cache_statuses: BTreeMap<String, usize>,
    pub real_auditor_cache_statuses: BTreeMap<String, usize>,
    pub ancillary_cache_statuses: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DeepCampaignReport {
    pub format: String,
    pub version: u8,
    pub slices: Vec<DeepSliceReport>,
    pub failures: Vec<DeepOutcome>,
    pub paired_pareto_hypervolume: BTreeMap<String, Option<PairedInterval>>,
}

#[derive(Default)]
struct SliceAccumulator {
    attempted: usize,
    succeeded: usize,
    failed: usize,
    timed_out: usize,
    runtimes: Vec<u64>,
    fitting_times: Vec<u64>,
    sampling_times: Vec<u64>,
    auditor_times: Vec<u64>,
    cpu_memory: Vec<u64>,
    gpu_memory: Vec<u64>,
    artifact_bytes: Vec<u64>,
    artifact_cache_statuses: BTreeMap<String, usize>,
    real_auditor_cache_statuses: BTreeMap<String, usize>,
    ancillary_cache_statuses: BTreeMap<String, usize>,
}

/// Builds the publication slices without substituting zero for a missing
/// metric. Seed/auditor replications are averaged inside each lineage before
/// lineages receive equal weight.
pub fn build_deep_campaign_report(outcomes: &[DeepOutcome]) -> Result<DeepCampaignReport> {
    for candidate in NEW_CANDIDATES {
        let configurations = outcomes
            .iter()
            .filter(|outcome| outcome.candidate_id == candidate)
            .filter_map(|outcome| outcome.configuration.map(LossConfiguration::id))
            .collect::<BTreeSet<_>>();
        if configurations.len() > 1 {
            return Err(DopeError::Data(format!(
                "campaign report mixes frozen configurations for {candidate}"
            )));
        }
    }
    let overall_averaged = averaged_outcomes(outcomes)?;
    let mut cells = BTreeMap::<(String, String, String), SliceAccumulator>::new();
    let mut lineage_metrics =
        BTreeMap::<(String, String, String, String, String), (DeepMetrics, usize)>::new();
    let mut failures = Vec::new();
    for outcome in outcomes {
        outcome.validate()?;
        if outcome.state != EvidenceState::Succeeded {
            failures.push(outcome.clone());
        }
        let dimensions = [
            ("overall", "all".to_string()),
            ("task", outcome.task.clone()),
            ("profile", outcome.structural_profile.clone()),
            ("auditor", outcome.auditor_id.clone()),
            ("size", outcome.size_multiplier.to_string()),
        ];
        for (dimension, key) in dimensions {
            let cell_key = (
                dimension.to_string(),
                key.clone(),
                outcome.candidate_id.clone(),
            );
            let cell = cells.entry(cell_key.clone()).or_default();
            cell.attempted += 1;
            match outcome.state {
                EvidenceState::Succeeded => cell.succeeded += 1,
                EvidenceState::TimedOut => cell.timed_out += 1,
                _ => cell.failed += 1,
            }
            if let Some(value) = outcome.runtime_ms {
                cell.runtimes.push(value);
            }
            if let Some(value) = outcome.fitting_time_ms {
                cell.fitting_times.push(value);
            }
            if let Some(value) = outcome.sampling_time_ms {
                cell.sampling_times.push(value);
            }
            if let Some(value) = outcome.auditor_time_ms {
                cell.auditor_times.push(value);
            }
            if let Some(value) = outcome.peak_cpu_memory_bytes {
                cell.cpu_memory.push(value);
            }
            if let Some(value) = outcome.peak_gpu_memory_bytes {
                cell.gpu_memory.push(value);
            }
            if let Some(value) = outcome.artifact_bytes {
                cell.artifact_bytes.push(value);
            }
            let record_cache = |counts: &mut BTreeMap<String, usize>,
                                status: Option<CacheStatus>| {
                if let Some(status) = status {
                    *counts
                        .entry(format!("{status:?}").to_ascii_lowercase())
                        .or_default() += 1;
                }
            };
            record_cache(
                &mut cell.artifact_cache_statuses,
                outcome.artifact_cache_status,
            );
            record_cache(
                &mut cell.real_auditor_cache_statuses,
                outcome.real_auditor_cache_status,
            );
            record_cache(
                &mut cell.ancillary_cache_statuses,
                outcome.ancillary_cache_status,
            );
            if let Some(metrics) = &outcome.metrics {
                let lineage_key = (
                    cell_key.0,
                    cell_key.1,
                    cell_key.2,
                    outcome.structural_profile.clone(),
                    outcome.lineage_group_id.clone(),
                );
                match lineage_metrics.entry(lineage_key) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert((metrics.clone(), 1));
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        entry.get_mut().0.add_assign(metrics);
                        entry.get_mut().1 += 1;
                    }
                }
            }
        }
    }
    let mut by_slice = BTreeMap::<(String, String, String), Vec<DeepMetrics>>::new();
    for ((dimension, key, candidate, _, _), (mut metrics, count)) in lineage_metrics {
        metrics.divide(count as f64);
        by_slice
            .entry((dimension, key, candidate))
            .or_default()
            .push(metrics);
    }
    let mean = |values: &[u64]| {
        (!values.is_empty()).then(|| values.iter().sum::<u64>() as f64 / values.len() as f64)
    };
    let mut slices = Vec::new();
    for (key, cell) in cells {
        let lineage_values = by_slice.remove(&key).unwrap_or_default();
        let mut metrics = lineage_values.first().cloned().unwrap_or_default();
        for value in lineage_values.iter().skip(1) {
            metrics.add_assign(value);
        }
        if !lineage_values.is_empty() {
            metrics.divide(lineage_values.len() as f64);
        }
        let bounded_retention = lineage_values
            .iter()
            .filter_map(|metrics| metrics.bounded_retention)
            .collect::<Vec<_>>();
        let absolute_lift = lineage_values
            .iter()
            .filter_map(|metrics| metrics.absolute_lift)
            .collect::<Vec<_>>();
        slices.push(DeepSliceReport {
            dimension: key.0,
            key: key.1,
            candidate_id: key.2,
            attempted_cells: cell.attempted,
            succeeded_cells: cell.succeeded,
            failed_cells: cell.failed,
            timed_out_cells: cell.timed_out,
            lineage_groups: lineage_values.len(),
            metrics,
            bounded_retention_interval: paired_interval(&bounded_retention),
            absolute_lift_interval: paired_interval(&absolute_lift),
            mean_runtime_ms: mean(&cell.runtimes),
            mean_fitting_time_ms: mean(&cell.fitting_times),
            mean_sampling_time_ms: mean(&cell.sampling_times),
            mean_auditor_time_ms: mean(&cell.auditor_times),
            peak_cpu_memory_bytes: cell.cpu_memory.into_iter().max(),
            peak_gpu_memory_bytes: cell.gpu_memory.into_iter().max(),
            maximum_artifact_bytes: cell.artifact_bytes.into_iter().max(),
            artifact_cache_statuses: cell.artifact_cache_statuses,
            real_auditor_cache_statuses: cell.real_auditor_cache_statuses,
            ancillary_cache_statuses: cell.ancillary_cache_statuses,
        });
    }
    slices.sort_by(|left, right| {
        left.dimension
            .cmp(&right.dimension)
            .then_with(|| left.key.cmp(&right.key))
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    let hypervolume = |metrics: &DeepMetrics| {
        metrics
            .bounded_retention
            .zip(metrics.joint_fidelity)
            .zip(metrics.membership_auc)
            .map(|((retention, fidelity), membership)| {
                retention.clamp(0.0, 1.0)
                    * fidelity.clamp(0.0, 1.0)
                    * (1.0 - membership).clamp(0.0, 1.0)
            })
    };
    let paired_pareto_hypervolume = NEW_CANDIDATES
        .into_iter()
        .map(|candidate| {
            let differences = metric_differences(&overall_averaged, candidate, hypervolume);
            (candidate.to_string(), paired_interval(&differences))
        })
        .collect();
    Ok(DeepCampaignReport {
        format: "dope-deep-joint-campaign-report".into(),
        version: 1,
        slices,
        failures,
        paired_pareto_hypervolume,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DatasetArtifactEntry {
    pub dataset_id: String,
    pub file_name: String,
    pub bytes: u64,
    pub sha256: String,
    pub blake3: String,
    pub rows_fitted: u64,
    pub features: u32,
    pub training_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DeepArtifactManifest {
    pub format: String,
    pub version: u8,
    pub family: String,
    pub configuration: LossConfiguration,
    pub implementation_hash: String,
    pub compilation_scope: String,
    pub representative_dataset_id: String,
    pub artifacts: Vec<DatasetArtifactEntry>,
}

pub fn build_deep_artifact_manifest(
    selection: &FinalFamilySelection,
    artifacts: &[(String, PathBuf)],
    representative_dataset_id: &str,
) -> Result<DeepArtifactManifest> {
    let winner = selection.winner.as_ref().ok_or_else(|| {
        DopeError::Data("cannot publish deep artifacts without a passing final family".into())
    })?;
    if artifacts.is_empty() {
        return Err(DopeError::Data("deep artifact manifest is empty".into()));
    }
    let mut seen = BTreeSet::new();
    let mut entries = Vec::new();
    for (dataset_id, path) in artifacts {
        if dataset_id.is_empty() || !seen.insert(dataset_id.clone()) {
            return Err(DopeError::Data(
                "deep artifact manifest contains an invalid dataset ID".into(),
            ));
        }
        if path.extension().and_then(|value| value.to_str()) != Some("dpk") {
            return Err(DopeError::Data(
                "deep artifact manifest requires .dpk filenames".into(),
            ));
        }
        let bytes = fs::read(path).map_err(|error| crate::error::io_error(path, error))?;
        if &bytes[..bytes.len().min(6)] != b"DPK3\x03\x02" {
            return Err(DopeError::Data(format!(
                "deep artifact is not DPK3.2: {}",
                path.display()
            )));
        }
        let kernel = decode_kernel(&bytes)?;
        let KernelProgram::NeuralJoint(generator) = &kernel.program else {
            return Err(DopeError::Data(
                "deep artifact manifest contains a non-neural kernel".into(),
            ));
        };
        if generator.implementation_hash != winner.implementation_hash
            || generator.architecture
                != match winner.candidate_id.as_str() {
                    "tvae" => crate::model::NeuralArchitecture::Tvae,
                    "single_table_autoregressive_transformer" => {
                        crate::model::NeuralArchitecture::MaskedAutoregressiveTransformer
                    }
                    "tabsyn" => crate::model::NeuralArchitecture::TabSyn,
                    _ => {
                        return Err(DopeError::Data(
                            "final family is not a frozen neural candidate".into(),
                        ));
                    }
                }
        {
            return Err(DopeError::Data(
                "artifact implementation does not match the frozen winner".into(),
            ));
        }
        let content = hashes(&bytes);
        entries.push(DatasetArtifactEntry {
            dataset_id: dataset_id.clone(),
            file_name: path
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| DopeError::Data("artifact filename is not UTF-8".into()))?
                .into(),
            bytes: bytes.len() as u64,
            sha256: content.sha256,
            blake3: content.blake3,
            rows_fitted: kernel.rows_fitted,
            features: kernel.features,
            training_hash: generator.training_hash.clone(),
        });
    }
    entries.sort_by(|left, right| left.dataset_id.cmp(&right.dataset_id));
    if !seen.contains(representative_dataset_id) {
        return Err(DopeError::Data(
            "representative artifact is absent from the manifest".into(),
        ));
    }
    Ok(DeepArtifactManifest {
        format: "dope-deep-dpk-artifact-manifest".into(),
        version: 1,
        family: winner.candidate_id.clone(),
        configuration: winner.configuration,
        implementation_hash: winner.implementation_hash.clone(),
        compilation_scope: "one independent fit and DPK3.2 artifact per dataset".into(),
        representative_dataset_id: representative_dataset_id.into(),
        artifacts: entries,
    })
}

fn display_metric(value: Option<f64>) -> String {
    value.map_or_else(|| "missing".into(), |value| format!("{value:.6}"))
}
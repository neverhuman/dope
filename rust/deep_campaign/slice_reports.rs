
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

pub fn build_deep_model_card(
    selection: &FinalFamilySelection,
    report: &DeepCampaignReport,
) -> Result<String> {
    if report.format != "dope-deep-joint-campaign-report" {
        return Err(DopeError::Data("invalid deep campaign report".into()));
    }
    let (family, configuration, implementation, decision) = selection.winner.as_ref().map_or_else(
        || {
            (
                "none".to_string(),
                "none".to_string(),
                "none".to_string(),
                "No deep family passed every frozen gate; retain the current frontier.".to_string(),
            )
        },
        |winner| {
            (
                winner.candidate_id.clone(),
                winner.configuration.id(),
                winner.implementation_hash.clone(),
                format!(
                    "Promote {} only as a compiler for dataset-specific DPK3.2 artifacts.",
                    winner.candidate_id
                ),
            )
        },
    );
    let overall = selection.winner.as_ref().and_then(|winner| {
        report.slices.iter().find(|slice| {
            slice.dimension == "overall"
                && slice.key == "all"
                && slice.candidate_id == winner.candidate_id
        })
    });
    let metrics = overall.map(|slice| &slice.metrics);
    let metric = |field: fn(&DeepMetrics) -> Option<f64>| display_metric(metrics.and_then(field));
    let runtime = overall
        .and_then(|slice| slice.mean_runtime_ms)
        .map_or_else(|| "missing".into(), |value| format!("{value:.3} ms"));
    let artifact = overall
        .and_then(|slice| slice.maximum_artifact_bytes)
        .map_or_else(|| "missing".into(), |value| format!("{value} bytes"));
    Ok(format!(
        "# Deep joint generator model card\n\n\
         ## Decision\n\n{decision}\n\n\
         Family: `{family}`  \nConfiguration: `{configuration}`  \nImplementation hash: `{implementation}`\n\n\
         ## Architecture and data accounting\n\n\
         The frozen candidates are TVAE, a masked autoregressive transformer, and TabSyn. \
         Discovery contains 206 datasets and 108,523 rows; confirmation contains 601 disjoint \
         datasets and 409,686 rows; validation-select contains 11,892 lineages and 11,620,492 rows. \
         These rows are distributed across independent per-dataset fits. They are never pooled \
         into one cross-dataset network. The selected family, if any, compiles one DPK3.2 artifact \
         for each dataset and sampling remains pure Rust.\n\n\
         ## Synthetic-to-real transfer\n\n\
         Mean bounded retention: {}  \nAbsolute lift over legacy TVAE: {}  \nRegret: {}  \nOracle recall: {}\n\n\
         ## Feature-importance consistency\n\n\
         Permutation-importance Spearman: {}  \nTop-k agreement: {}  \nInformative feature groups: {}  \nUndefined Spearman values remain missing and are not replaced by zero.\n\n\
         ## Fidelity, privacy, and safeguards\n\n\
         Driver agreement: {}  \nJoint fidelity: {}  \nMembership AUC: {}  \nAttribute-inference advantage: {}  \nExact copies: {}  \nCanary extractions: {}\n\n\
         ## Operations\n\n\
         Mean per-cell runtime: {runtime}  \nMaximum artifact: {artifact}\n\n\
         ## Limitations\n\n\
         Promotion is valid only for the frozen source, binary, configuration, cohort split, \
         auditors, seeds, coverage rules, and artifact/parity gates. Validation-cert and sealed-test \
         are outside family selection. Missing, failed, timed-out, OOM, or inadequately informative \
         cells cannot be imputed as successful evidence.\n",
        metric(|m| m.bounded_retention),
        metric(|m| m.absolute_lift),
        metric(|m| m.regret),
        metric(|m| m.oracle_recall),
        metric(|m| m.feature_importance_spearman),
        metric(|m| m.feature_importance_top_k_agreement),
        metric(|m| m.feature_importance_informative_groups),
        metric(|m| m.driver_agreement),
        metric(|m| m.joint_fidelity),
        metric(|m| m.membership_auc),
        metric(|m| m.attribute_inference_advantage),
        metrics
            .and_then(|m| m.exact_copies)
            .map_or_else(|| "missing".into(), |v| v.to_string()),
        metrics
            .and_then(|m| m.canary_extractions)
            .map_or_else(|| "missing".into(), |v| v.to_string()),
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpansionDecision {
    None,
    Ctgan,
    Tabddpm,
    PublishFailedFrontier,
}

pub fn bounded_expansion(
    maximum_safe_discovery_lift: f64,
    largest_deficit: &str,
) -> ExpansionDecision {
    if maximum_safe_discovery_lift < 0.05 {
        return ExpansionDecision::None;
    }
    match largest_deficit {
        "joint_fidelity" | "query_error" => ExpansionDecision::Ctgan,
        "rare_tail_retention" | "multimodal_coverage" => ExpansionDecision::Tabddpm,
        "privacy" | "copying" | "calibration" => ExpansionDecision::PublishFailedFrontier,
        _ => ExpansionDecision::None,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CompressionEvidence {
    pub artifact_bytes: u64,
    pub retention_before: f64,
    pub retention_after: f64,
    pub profile_conclusions_unchanged: bool,
    pub safeguard_gates_pass: bool,
    pub deterministic_draws: usize,
    pub maximum_libtorch_rust_difference: f64,
    pub discrete_decision_agreement: f64,
}

impl CompressionEvidence {
    pub fn validate(&self) -> Result<()> {
        if self.artifact_bytes > ROUTER_ARTIFACT_LIMIT_BYTES
            || !self.retention_before.is_finite()
            || !self.retention_after.is_finite()
            || self.retention_before - self.retention_after > 0.01
            || !self.profile_conclusions_unchanged
            || !self.safeguard_gates_pass
            || self.deterministic_draws != 100_000
            || !self.maximum_libtorch_rust_difference.is_finite()
            || self.maximum_libtorch_rust_difference > 1e-3
            || !self.discrete_decision_agreement.is_finite()
            || self.discrete_decision_agreement < 0.999
        {
            return Err(DopeError::Data(
                "compressed joint generator does not satisfy router handoff gates".into(),
            ));
        }
        Ok(())
    }
}

/// Produces the exact router-handoff bytes once the already-int8 joint model
/// fits the compressed ceiling. Models over the ceiling remain failed evidence
/// until an offline structured-pruning pass emits a new frozen kernel.
pub fn prepare_router_artifact(kernel: &Kernel) -> Result<Vec<u8>> {
    if !matches!(kernel.program, KernelProgram::NeuralJoint(_)) {
        return Err(DopeError::Data(
            "router deep handoff requires a neural joint kernel".into(),
        ));
    }
    let artifact = encode_kernel(kernel)?;
    if artifact.len() as u64 > ROUTER_ARTIFACT_LIMIT_BYTES {
        return Err(DopeError::Data(
            "joint artifact still exceeds 4 MiB after int8 export; structured pruning is required"
                .into(),
        ));
    }
    Ok(artifact)
}

const COMPRESSION_SCHEDULE: [(f64, usize); 4] = [(0.10, 129), (0.20, 65), (0.35, 33), (0.50, 17)];

fn sparsify_output_channels(layer: &mut QuantizedLinear, fraction: f64) {
    let input = layer.input_dim as usize;
    let output = layer.output_dim as usize;
    let prune = ((output as f64 * fraction).floor() as usize).min(output.saturating_sub(1));
    let mut channels = (0..output)
        .map(|channel| {
            let salience = layer.weights[channel * input..(channel + 1) * input]
                .iter()
                .map(|weight| f64::from(weight.unsigned_abs()))
                .sum::<f64>()
                * f64::from(layer.scales[channel])
                + f64::from(layer.biases[channel].abs());
            (salience, channel)
        })
        .collect::<Vec<_>>();
    channels.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    for (_, channel) in channels.into_iter().take(prune) {
        layer.weights[channel * input..(channel + 1) * input].fill(0);
        layer.biases[channel] = 0.0;
    }
}

fn reduce_marginal_knots(marginal: &mut Marginal, maximum: usize) {
    match marginal {
        Marginal::QuantileSpline { values } if values.len() > maximum => {
            let last = values.len() - 1;
            *values = (0..maximum)
                .map(|index| values[index * last / (maximum - 1)])
                .collect();
        }
        Marginal::ZeroInflated { base, .. } => reduce_marginal_knots(base, maximum),
        _ => {}
    }
}

fn compressed_kernel(kernel: &Kernel, fraction: f64, maximum_knots: usize) -> Kernel {
    let mut compressed = kernel.clone();
    for marginal in &mut compressed.marginals {
        reduce_marginal_knots(marginal, maximum_knots);
    }
    let KernelProgram::NeuralJoint(generator) = &mut compressed.program else {
        return compressed;
    };
    reduce_marginal_knots(&mut generator.target_marginal, maximum_knots);
    match &mut generator.network {
        JointNetwork::Tvae { decoder } => {
            sparsify_output_channels(&mut decoder.hidden_1, fraction);
            sparsify_output_channels(&mut decoder.hidden_2, fraction);
        }
        JointNetwork::MaskedAutoregressiveTransformer { transformer } => {
            for block in &mut transformer.blocks {
                sparsify_output_channels(&mut block.feed_forward_1, fraction);
            }
        }
        JointNetwork::TabSyn {
            decoder, denoiser, ..
        } => {
            sparsify_output_channels(&mut decoder.hidden_1, fraction);
            sparsify_output_channels(&mut decoder.hidden_2, fraction);
            sparsify_output_channels(&mut denoiser.hidden_1, fraction);
            sparsify_output_channels(&mut denoiser.hidden_2, fraction);
            sparsify_output_channels(&mut denoiser.hidden_3, fraction);
        }
        JointNetwork::TabDdpm { denoiser, .. } => {
            sparsify_output_channels(&mut denoiser.hidden_1, fraction);
            sparsify_output_channels(&mut denoiser.hidden_2, fraction);
            sparsify_output_channels(&mut denoiser.hidden_3, fraction);
        }
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"joint-router-compression-v1");
    hasher.update(generator.training_hash.as_bytes());
    hasher.update(&fraction.to_bits().to_le_bytes());
    hasher.update(&(maximum_knots as u64).to_le_bytes());
    generator.training_hash = hasher.finalize().to_hex().to_string();
    compressed
}

/// Applies the frozen train-only compression schedule. The first artifact that
/// reaches 4 MiB is returned only when independent compression/parity evidence
/// satisfies every frozen gate and reconciles to the exact byte count.
pub fn prepare_router_artifact_with_compression(
    kernel: &Kernel,
    evidence: &CompressionEvidence,
) -> Result<Vec<u8>> {
    if !matches!(kernel.program, KernelProgram::NeuralJoint(_)) {
        return Err(DopeError::Data(
            "router deep compression requires a neural joint kernel".into(),
        ));
    }
    let artifact = prepare_router_artifact(kernel);
    if let Ok(artifact) = artifact {
        return Ok(artifact);
    }
    evidence.validate()?;
    for (fraction, maximum_knots) in COMPRESSION_SCHEDULE {
        let compressed = compressed_kernel(kernel, fraction, maximum_knots);
        compressed.validate().map_err(DopeError::Data)?;
        let artifact = encode_kernel(&compressed)?;
        if artifact.len() as u64 <= ROUTER_ARTIFACT_LIMIT_BYTES {
            if artifact.len() as u64 != evidence.artifact_bytes {
                return Err(DopeError::Data(
                    "compression evidence does not reconcile to the emitted artifact".into(),
                ));
            }
            return Ok(artifact);
        }
    }
    Err(DopeError::Data(
        "joint artifact exceeds 4 MiB after the complete frozen compression schedule".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn record(profile: &str, lineage: usize) -> CohortRecord {
        CohortRecord {
            dataset_id: format!("d{lineage}"),
            dataset_path: PathBuf::from(format!("/{lineage}")),
            task: "regression".into(),
            rows: 128,
            features: 8,
            lineage_group_id: format!("{profile}-l{lineage}"),
            structural_profile: profile.into(),
            partition: "training_gold".into(),
        }
    }

    fn final_outcome(
        candidate: &str,
        profile: &str,
        retention: f64,
        regret: f64,
        artifact_bytes: u64,
        runtime_ms: u64,
    ) -> DeepOutcome {
        DeepOutcome {
            task: "regression".into(),
            lineage_group_id: "shared-lineage".into(),
            structural_profile: profile.into(),
            candidate_id: candidate.into(),
            configuration: Some(LossConfiguration::GRID[0]),
            generation_seed: 1_829,
            auditor_seed: 57_721,
            auditor_id: "elastic_net_glm".into(),
            size_multiplier: 1,
            state: EvidenceState::Succeeded,
            metrics: Some(DeepMetrics {
                bounded_retention: Some(retention),
                regret: Some(regret),
                ..Default::default()
            }),
            failure_reason: None,
            artifact_bytes: Some(artifact_bytes),
            runtime_ms: Some(runtime_ms),
            fitting_time_ms: Some(10),
            sampling_time_ms: Some(5),
            auditor_time_ms: Some(runtime_ms.saturating_sub(15)),
            peak_cpu_memory_bytes: Some(1_024),
            peak_gpu_memory_bytes: None,
            artifact_cache_status: Some(CacheStatus::Miss),
            real_auditor_cache_status: Some(CacheStatus::Miss),
            ancillary_cache_status: Some(CacheStatus::Disabled),
            invalid_rows: 0,
            schema_violations: 0,
            nondeterministic_output: false,
        }
    }

    fn passing(candidate: &str) -> Qualification {
        Qualification {
            candidate_id: candidate.into(),
            implementation_hash: candidate_implementation_hash(candidate),
            qualifies: true,
            overall_lift: None,
            oracle_lift: Some(0.0),
            success_coverage: 1.0,
            minimum_profile_coverage: 1.0,
            feature_importance_coverage: 1.0,
            feature_importance_spearman_delta: None,
            feature_importance_top_k_delta: None,
            maximum_artifact_bytes: Some(1_024),
            failed_gates: Vec::new(),
        }
    }

    #[test]
    fn cohorts_take_first_eight_then_next_thirty_two_without_overlap() {
        let plan = CohortPlan {
            format: "dope-campaign-cohort".into(),
            version: 1,
            kind: "training-gold".into(),
            records: (0..100)
                .map(|lineage| record(if lineage < 50 { "a" } else { "b" }, lineage))
                .collect(),
            exclusions: Vec::new(),
        };
        let deep = plan_deep_cohorts(&plan).unwrap();
        assert_eq!(deep.discovery.len(), 16);
        assert_eq!(deep.confirmation.len(), 64);
        let discovery = deep
            .discovery
            .iter()
            .map(|record| &record.lineage_group_id)
            .collect::<BTreeSet<_>>();
        assert!(
            deep.confirmation
                .iter()
                .all(|record| !discovery.contains(&record.lineage_group_id))
        );
        assert_eq!(deep.confirmation_cells(14), 64 * 14 * 108);
        assert_eq!(
            EXPECTED_CONFIRMATION_LINEAGES * 14 * 108,
            MAX_CONFIRMATION_CELLS
        );
        assert_eq!(
            validation_select_cells(3).unwrap(),
            MAX_VALIDATION_SELECT_CELLS
        );
    }

    #[test]
    fn bounded_expansion_follows_the_frozen_deficit_rule() {
        assert_eq!(
            bounded_expansion(0.049, "query_error"),
            ExpansionDecision::None
        );
        assert_eq!(
            bounded_expansion(0.05, "query_error"),
            ExpansionDecision::Ctgan
        );
        assert_eq!(
            bounded_expansion(0.08, "rare_tail_retention"),
            ExpansionDecision::Tabddpm
        );
        assert_eq!(
            bounded_expansion(0.08, "privacy"),
            ExpansionDecision::PublishFailedFrontier
        );
    }

    #[test]
    fn final_selector_uses_profile_lineage_units_and_frozen_tiebreaks() {
        let outcomes = vec![
            final_outcome("tvae", "a", 0.8, 0.2, 1_000, 30),
            final_outcome("tvae", "b", 0.8, 0.2, 1_000, 30),
            final_outcome("tabsyn", "a", 0.8, 0.1, 2_000, 20),
            final_outcome("tabsyn", "b", 0.8, 0.1, 2_000, 20),
        ];
        let selection =
            select_final_family(&outcomes, &[passing("tvae"), passing("tabsyn")]).unwrap();
        let winner = selection.winner.unwrap();
        assert_eq!(winner.candidate_id, "tabsyn");
        assert_eq!(winner.cohort_units, 2);
    }

    #[test]
    fn final_selector_recommends_frontier_when_no_candidate_passes() {
        let selection = select_final_family(&[], &[]).unwrap();
        assert_eq!(selection.winner, None);
        assert_eq!(selection.recommendation, "retain_current_frontier");
        let card = build_deep_model_card(
            &selection,
            &DeepCampaignReport {
                format: "dope-deep-joint-campaign-report".into(),
                version: 1,
                slices: Vec::new(),
                failures: Vec::new(),
                paired_pareto_hypervolume: BTreeMap::new(),
            },
        )
        .unwrap();
        assert!(card.contains("retain the current frontier"));
        assert!(card.contains("never pooled"));
    }
}

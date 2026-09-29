

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
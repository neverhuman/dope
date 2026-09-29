
fn compile_neural_table(
    table: &Table,
    task: Task,
    options: &CompileOptions,
    architecture: NeuralArchitecture,
    seed: u64,
    started: Instant,
) -> Result<CompileResult> {
    if architecture == NeuralArchitecture::TabDdpm
        && !matches!(
            options.release_policy.tier,
            AnonymizationTier::L0 | AnonymizationTier::L1
        )
    {
        return Err(DopeError::Data(
            "direct rank TabDDPM is eligible only for L1 or research L0".into(),
        ));
    }
    if !matches!(options.neural_target_weight, 2.0 | 4.0)
        || !matches!(options.neural_structural_penalty, 0.0 | 0.1)
    {
        return Err(DopeError::Data(
            "neural loss configuration is outside the frozen 2x2 grid".into(),
        ));
    }
    let fits = table
        .columns
        .iter()
        .map(|column| fit_column(column))
        .collect::<Vec<_>>();
    let data = neural_training_data(table, &fits, seed);
    let candidate_id = options
        .backend_id
        .clone()
        .expect("restricted neural candidate");
    let candidate_seed = {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&seed.to_le_bytes());
        hasher.update(candidate_id.as_bytes());
        u64::from_le_bytes(hasher.finalize().as_bytes()[..8].try_into().unwrap())
    };
    let generator: JointGenerator = crate::neural_train::fit_joint_generator(
        &data,
        crate::neural_train::NeuralTrainingConfig {
            architecture,
            profile: neural_profile(&candidate_id),
            target_weight: options.neural_target_weight,
            structural_penalty: options.neural_structural_penalty,
            seed: candidate_seed,
            deadline: started
                + options
                    .deadline
                    .unwrap_or(Duration::from_secs(600))
                    .min(Duration::from_secs(600)),
        },
    )?;
    let marginals = fits
        .iter()
        .map(|fit| {
            fit.discrete
                .clone()
                .unwrap_or_else(|| Marginal::QuantileSpline {
                    values: fit.quantiles[4].clone(),
                })
        })
        .collect::<Vec<_>>();
    let kernel = Kernel {
        task,
        rows_fitted: table.rows as u64,
        features: table.features as u32,
        seed,
        seed_policy: u8::from(options.seed.is_none()),
        quantization_bits: 8,
        compliant: false,
        schema: fits.iter().map(|fit| fit.schema.clone()).collect(),
        marginals,
        program: KernelProgram::NeuralJoint(generator),
    };
    let artifact = encode_kernel(&kernel)?;
    if !options
        .release_policy
        .accepts_artifact_bytes(artifact.len())
    {
        return Err(DopeError::Data(format!(
            "no encoded neural candidate fits the release policy; smallest observed size: {} bytes",
            artifact.len()
        )));
    }
    let report = CandidateReport {
        candidate_id: candidate_id.clone(),
        artifact_bytes: artifact.len(),
        bits_per_original_cell: 8.0 * artifact.len() as f64
            / (table.rows * (table.features + 1)).max(1) as f64,
        score: 0.0,
        compliant: false,
        quantization_bits: 8,
        marginal_knots: 65,
        dependence: "neural_joint".into(),
        target: match architecture {
            NeuralArchitecture::Tvae => "tvae",
            NeuralArchitecture::MaskedAutoregressiveTransformer => {
                "masked_autoregressive_transformer"
            }
            NeuralArchitecture::TabSyn => "tabsyn",
            NeuralArchitecture::TabDdpm => "tabddpm",
        }
        .into(),
        utility_retention: 0.0,
        driver_agreement: 0.0,
        proxy_joint_fidelity: 0.0,
        proxy_membership_auc: 0.5,
        failed_gates: vec!["production_evidence_unmeasured".into()],
    };
    let compile_report = CompileReport {
        format: "dope-kernel-compile-report".into(),
        version: 3,
        release_policy: options.release_policy.clone(),
        release_policy_hash: options.release_policy.hash(),
        task: task.as_str().into(),
        rows: table.rows,
        positional_features: table.features,
        selected_candidate: candidate_id.clone(),
        artifact_bytes: artifact.len(),
        effective_bytes: artifact.len(),
        bits_per_original_cell: report.bits_per_original_cell,
        compliant: false,
        failed_gates: report.failed_gates.clone(),
        candidate_count: 1,
        frontier_count: 1,
        beam_width: 1,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        contains_row_payloads: false,
        contains_row_references: false,
        probe_failures: BTreeMap::new(),
    };
    Ok(CompileResult {
        candidate_artifacts: BTreeMap::from([(candidate_id, artifact.clone())]),
        artifact,
        report: compile_report,
        candidates: vec![report.clone()],
        pareto_frontier: vec![report],
    })
}

fn compile_table(table: &Table, task: Task, options: &CompileOptions) -> Result<CompileResult> {
    if let Some(architecture) = options.backend_id.as_deref().and_then(neural_architecture) {
        let started = Instant::now();
        return compile_neural_table(
            table,
            task,
            options,
            architecture,
            options.seed.unwrap_or_else(|| stable_seed(table)),
            started,
        );
    }
    let symbolic = compile_symbolic_table(table, task, options)?;
    #[cfg(feature = "gpu-training")]
    {
        probe_compact_neural(table, task, options, symbolic)
    }
    #[cfg(not(feature = "gpu-training"))]
    {
        Ok(symbolic)
    }
}

#[cfg(feature = "gpu-training")]
fn probe_compact_neural(
    table: &Table,
    task: Task,
    options: &CompileOptions,
    mut result: CompileResult,
) -> Result<CompileResult> {
    if options.backend_id.is_some() {
        return Ok(result);
    }
    let probes: &[&str] = match options.release_policy.tier {
        AnonymizationTier::L3 => &["micro_tvae_4_16", "micro_tvae_8_24"],
        AnonymizationTier::L2 => &["micro_tvae_4_16", "micro_tvae_8_24", "tiny_mat_16_2_32"],
        AnonymizationTier::L0 | AnonymizationTier::L1 => &[],
    };
    if probes.is_empty() {
        return Ok(result);
    }
    let started = Instant::now();
    let seed = options.seed.unwrap_or_else(|| stable_seed(table));
    for &id in probes {
        if options
            .deadline
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            break;
        }
        let mut restricted = options.clone();
        restricted.backend_id = Some(id.into());
        if let Some(limit) = options.deadline {
            restricted.deadline = Some(limit.saturating_sub(started.elapsed()));
        }
        let architecture = neural_architecture(id).expect("frozen compact neural candidate");
        // Training and proxy predictions never establish size eligibility. Only
        // the final encoded DPK passes the policy byte cap in this function.
        match compile_neural_table(table, task, &restricted, architecture, seed, Instant::now()) {
            Ok(neural) => {
                result
                    .candidate_artifacts
                    .extend(neural.candidate_artifacts);
                result.candidates.extend(neural.candidates);
            }
            Err(error) => {
                result
                    .report
                    .probe_failures
                    .insert(id.into(), error.to_string());
            }
        }
    }
    result.pareto_frontier = pareto(&result.candidates);
    let selected = result
        .candidates
        .iter()
        .min_by(|left, right| {
            maximum_normalized_gate_shortfall(left)
                .total_cmp(&maximum_normalized_gate_shortfall(right))
                .then_with(|| left.artifact_bytes.cmp(&right.artifact_bytes))
                .then_with(|| left.candidate_id.cmp(&right.candidate_id))
        })
        .expect("symbolic baseline exists");
    result.artifact = result.candidate_artifacts[&selected.candidate_id].clone();
    result.report.selected_candidate = selected.candidate_id.clone();
    result.report.artifact_bytes = result.artifact.len();
    result.report.effective_bytes = result.artifact.len();
    result.report.bits_per_original_cell = selected.bits_per_original_cell;
    result.report.compliant = selected.compliant;
    result.report.failed_gates = selected.failed_gates.clone();
    result.report.candidate_count = result.candidates.len();
    result.report.frontier_count = result.pareto_frontier.len();
    result.report.elapsed_seconds += started.elapsed().as_secs_f64();
    Ok(result)
}
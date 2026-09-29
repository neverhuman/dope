
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

fn compile_symbolic_table(
    table: &Table,
    task: Task,
    options: &CompileOptions,
) -> Result<CompileResult> {
    let started = Instant::now();
    let seed = options.seed.unwrap_or_else(|| stable_seed(table));
    let restricted_profile = options
        .backend_id
        .as_deref()
        .map(|id| {
            backend_profile(id)
                .ok_or_else(|| DopeError::Data(format!("unknown frozen empirical backend {id}")))
        })
        .transpose()?;
    if options.quantization_profiles.is_empty()
        || options
            .quantization_profiles
            .iter()
            .any(|profile| !QUANTIZATION_LADDER.contains(profile))
    {
        return Err(DopeError::Data(
            "invalid quantization profile selection".into(),
        ));
    }
    let seed_policy = options.seed.is_none() as u8;
    let width = table.features;
    let beam_width = if restricted_profile.is_some() {
        1
    } else {
        options
            .beam_width
            .unwrap_or(if width > 256 { 72 } else { 108 })
            .max(1)
    };
    let deadline = options.deadline.map(|duration| started + duration);
    let fits: Vec<ColumnFit> = table
        .columns
        .par_iter()
        .map(|column| fit_column(column))
        .collect();
    let schema: Vec<_> = fits.iter().map(|fit| fit.schema.clone()).collect();
    let completed: Vec<_> = fits.iter().map(|fit| fit.completed.clone()).collect();
    let ranks: Vec<_> = completed
        .par_iter()
        .map(|column| rank_normalize(column))
        .collect();
    let correlation = (width <= 512).then(|| full_correlation(&ranks));
    let edges = correlation.as_ref().map_or_else(
        || wide_tree(&ranks),
        |matrix| maximum_spanning_tree(matrix, width),
    );
    let triangular_terms = wide_tree(&ranks)
        .into_iter()
        .map(|edge| TriangularTerm {
            parent: edge.parent,
            child: edge.child,
            coefficient: edge.correlation,
        })
        .collect::<Vec<_>>();
    let screen = target_screen(&completed, &table.target);
    let (target_model, noise, _fitted_utility) = match task {
        Task::Regression => fit_linear_target(table, &completed, &screen),
        Task::Binary => fit_logistic_target(table, &completed, &screen),
    };
    let baseline_name = if task == Task::Regression {
        "sparse_linear"
    } else {
        "sparse_logistic"
    };
    let mut target_models = vec![(baseline_name, target_model.clone(), noise.clone())];
    let gam = fit_gam_target(table, &completed, &screen);
    target_models.push((
        "sparse_gam",
        gam.clone(),
        fit_noise(table, &completed, &screen, &gam),
    ));
    let ga2m = fit_ga2m_target(table, &completed, &screen);
    target_models.push((
        "ga2m",
        ga2m.clone(),
        fit_noise(table, &completed, &screen, &ga2m),
    ));
    let mars = fit_mars_target(table, &completed, &screen);
    target_models.push((
        "mars",
        mars.clone(),
        fit_noise(table, &completed, &screen, &mars),
    ));
    let oblivious_tree = fit_oblivious_tree_target(table, &completed, &screen);
    target_models.push((
        "oblivious_tree",
        oblivious_tree.clone(),
        fit_noise(table, &completed, &screen, &oblivious_tree),
    ));
    let compact_neural = if restricted_profile.is_some_and(|profile| profile.2 != 5) {
        target_model.clone()
    } else {
        fit_compact_neural_target(table, &completed, &screen, seed)?
    };
    target_models.push((
        "compact_neural_residual",
        compact_neural.clone(),
        fit_noise(table, &completed, &screen, &compact_neural),
    ));
    let seed_kernel = Kernel {
        task,
        rows_fitted: table.rows as u64,
        features: width as u32,
        seed,
        seed_policy,
        quantization_bits: 8,
        compliant: false,
        schema,
        marginals: Vec::new(),
        program: KernelProgram::Symbolic {
            dependence: Dependence::Independent,
            target: target_model,
            noise,
        },
    };

    let mut artifacts = BTreeMap::<String, Vec<u8>>::new();
    let mut candidates = Vec::<CandidateReport>::new();
    let dataset_descriptor = describe_table(table, task);
    let mut specs: Vec<(usize, u8, usize, u8, f64)> = Vec::new();
    for (knot_index, &knots) in KNOT_LADDER.iter().enumerate() {
        for dependence_family in 0..7 {
            for (target_index, (target_name, target_model, _)) in target_models.iter().enumerate() {
                for &bits in &options.quantization_profiles {
                    if restricted_profile.is_some_and(|profile| {
                        profile != (knot_index, dependence_family, target_index, bits)
                    }) {
                        continue;
                    }
                    let dependence_name = dependence_name(dependence_family);
                    let descriptor = CandidateDescriptor {
                        quantization_bits: bits,
                        marginal_knots: knots,
                        dependence: dependence_name.into(),
                        target: (*target_name).into(),
                        effective_bytes: 56
                            + width * (knots * 2 + 8)
                            + usize::from(dependence_family != 0) * width.saturating_sub(1) * 12
                            + target_parameter_bytes(target_model),
                    };
                    let priority = options.language.as_ref().map_or(0.0, |language| {
                        language.candidate_priority(&dataset_descriptor, &descriptor)
                    });
                    specs.push((knot_index, dependence_family, target_index, bits, priority));
                }
            }
        }
    }
    specs.sort_by(|left, right| {
        right
            .4
            .total_cmp(&left.4)
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.3.cmp(&right.3))
    });
    let mut selected_specs: Vec<_> = specs.into_iter().take(beam_width).collect();
    if restricted_profile.is_none() {
        let baseline = (0, 0, 0, options.quantization_profiles[0]);
        if !selected_specs
            .iter()
            .any(|spec| (spec.0, spec.1, spec.2, spec.3) == baseline)
        {
            selected_specs.push((baseline.0, baseline.1, baseline.2, baseline.3, 0.0));
        }
    }
    let mut smallest_observed = None::<usize>;
    for (knot_index, dependence_family, target_index, bits, _) in selected_specs {
        if deadline.is_some_and(|limit| Instant::now() > limit) && !candidates.is_empty() {
            break;
        }
        let knots = KNOT_LADDER[knot_index];
        let marginals: Vec<Marginal> = fits
            .iter()
            .map(|fit| {
                fit.discrete
                    .clone()
                    .unwrap_or_else(|| Marginal::QuantileSpline {
                        values: fit.quantiles[knot_index].clone(),
                    })
            })
            .collect();
        let marginal_fidelity = 1.0 - 0.035 / (knots as f64).sqrt();
        let (dependence, dependence_name, dependence_fidelity) = match dependence_family {
            0 => {
                let fidelity = correlation.as_ref().map_or(0.85, |matrix| {
                    let mean = matrix
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| index / width != index % width)
                        .map(|(_, value)| f64::from(value.abs()))
                        .sum::<f64>()
                        / (width * width - width).max(1) as f64;
                    (1.0 - mean / 0.5).clamp(0.0, 1.0)
                });
                (Dependence::Independent, "independence", fidelity)
            }
            1 => (
                Dependence::ChowLiu {
                    root: 0,
                    edges: edges.clone(),
                },
                "chow_liu",
                0.97,
            ),
            2 => (
                Dependence::Triangular {
                    terms: triangular_terms.clone(),
                },
                "triangular_autoregressive",
                0.98,
            ),
            3 => (
                Dependence::SparseGraph {
                    edges: strongest_sparse_edges(
                        correlation.as_ref(),
                        &edges,
                        width,
                        width.saturating_mul(4),
                    ),
                },
                "sparse_gaussian_copula",
                0.985,
            ),
            4 => (
                poet_dependence(correlation.as_ref(), &edges, width),
                "poet_factor",
                0.985,
            ),
            5 => (
                vine_dependence(correlation.as_ref(), &edges, width),
                "truncated_c_vine",
                0.99,
            ),
            6 => (
                Dependence::Mixture {
                    weights: vec![0.5, 0.5],
                    components: vec![
                        Dependence::ChowLiu {
                            root: 0,
                            edges: edges.clone(),
                        },
                        Dependence::Triangular {
                            terms: triangular_terms.clone(),
                        },
                    ],
                },
                "two_component_copula_mixture",
                0.99,
            ),
            _ => unreachable!("bounded dependence family"),
        };
        let (target_name, target_model, noise) = &target_models[target_index];
        let mut kernel = seed_kernel.clone();
        kernel.marginals = marginals.clone();
        kernel.program = KernelProgram::Symbolic {
            dependence: dependence.clone(),
            target: target_model.clone(),
            noise: noise.clone(),
        };
        kernel = quantized_kernel(&kernel, bits);
        let utility = quantized_target_utility(&kernel, table, &completed);
        let driver_agreement = quantized_driver_agreement(&kernel, &screen);
        let quantization_penalty = 0.5 / ((1u32 << bits) - 1) as f64;
        let proxy_joint_fidelity = 0.55
            * (marginal_fidelity - quantization_penalty).clamp(0.0, 1.0)
            + 0.45 * dependence_fidelity;
        let proxy_membership_auc = 0.5;
        let mut failed_gates = Vec::new();
        if utility < 0.99 {
            failed_gates.push("utility".to_string());
        }
        if driver_agreement < 0.95 {
            failed_gates.push("driver".to_string());
        }
        if proxy_joint_fidelity < 0.90 {
            failed_gates.push("proxy_joint_fidelity".to_string());
        }
        if proxy_membership_auc > 0.60 {
            failed_gates.push("proxy_anti_memorization".to_string());
        }
        failed_gates.push("production_evidence_unmeasured".to_string());
        // Compilation sees train.csv only. No train-only heuristic may certify an artifact.
        kernel.compliant = false;
        let artifact = encode_kernel(&kernel)?;
        smallest_observed =
            Some(smallest_observed.map_or(artifact.len(), |size| size.min(artifact.len())));
        if !options
            .release_policy
            .accepts_artifact_bytes(artifact.len())
        {
            continue;
        }
        let candidate_id = options
            .backend_id
            .clone()
            .unwrap_or_else(|| format!("k{knots}-{dependence_name}-{target_name}-q{bits}"));
        let score = 0.45 * utility + 0.25 * driver_agreement + 0.30 * proxy_joint_fidelity;
        let record = CandidateReport {
            candidate_id: candidate_id.clone(),
            artifact_bytes: artifact.len(),
            bits_per_original_cell: 8.0 * artifact.len() as f64 / (table.rows * (width + 1)) as f64,
            score,
            compliant: kernel.compliant,
            quantization_bits: bits,
            marginal_knots: knots,
            dependence: dependence_name.into(),
            target: (*target_name).into(),
            utility_retention: utility,
            driver_agreement,
            proxy_joint_fidelity,
            proxy_membership_auc,
            failed_gates,
        };
        artifacts.insert(candidate_id, artifact);
        candidates.push(record);
    }
    if candidates.is_empty() {
        return Err(DopeError::Data(match smallest_observed {
            Some(size) => format!(
                "no encoded candidate fits the release policy; smallest observed size: {size} bytes"
            ),
            None => "deadline expired before baseline encoding".into(),
        }));
    }
    let frontier = pareto(&candidates);
    let selected = candidates
        .iter()
        .min_by(|left, right| {
            maximum_normalized_gate_shortfall(left)
                .total_cmp(&maximum_normalized_gate_shortfall(right))
                .then_with(|| left.artifact_bytes.cmp(&right.artifact_bytes))
                .then_with(|| left.candidate_id.cmp(&right.candidate_id))
        })
        .unwrap()
        .clone();
    let artifact = artifacts.get(&selected.candidate_id).unwrap().clone();
    let report = CompileReport {
        format: "dope-kernel-compile-report".into(),
        version: 3,
        release_policy: options.release_policy.clone(),
        release_policy_hash: options.release_policy.hash(),
        task: task.as_str().into(),
        rows: table.rows,
        positional_features: width,
        selected_candidate: selected.candidate_id.clone(),
        artifact_bytes: artifact.len(),
        effective_bytes: artifact.len(),
        bits_per_original_cell: selected.bits_per_original_cell,
        compliant: selected.compliant,
        failed_gates: selected.failed_gates.clone(),
        candidate_count: candidates.len(),
        frontier_count: frontier.len(),
        beam_width,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        contains_row_payloads: false,
        contains_row_references: false,
        probe_failures: BTreeMap::new(),
    };
    Ok(CompileResult {
        artifact,
        candidate_artifacts: artifacts,
        report,
        candidates,
        pareto_frontier: frontier,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_small_cap_reports_smallest_observed_encoded_candidate() {
        let options = CompileOptions {
            release_policy: ReleasePolicy::new(AnonymizationTier::L3, Some(1), false).unwrap(),
            beam_width: Some(1),
            ..Default::default()
        };
        let error = compile_kernel_from_arrays(
            &[0.0, 0.25, 0.75, 1.0],
            &[0.0, 0.25, 0.75, 1.0],
            4,
            1,
            Task::Regression,
            &options,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("smallest observed size:"));
    }

    #[test]
    fn near_equal_discrete_values_are_counted_without_exact_search_panics() {
        let fit = fit_column(&[0.1, 0.100_000_05, 0.2, 0.2]);
        assert!(fit.discrete.is_some());
    }
    use crate::codec::decode_kernel;
    use std::collections::BTreeSet;

    #[test]
    fn compiles_deterministically_and_keeps_positions() {
        let rows = 100;
        let features: Vec<f32> = (0..rows)
            .flat_map(|row| [0.1, (row % 2) as f32, row as f32 / rows as f32])
            .collect();
        let target: Vec<f32> = (0..rows)
            .map(|row| 0.2 + 0.6 * row as f32 / rows as f32)
            .collect();
        let options = CompileOptions {
            seed: Some(7),
            ..Default::default()
        };
        let first =
            compile_kernel_from_arrays(&features, &target, rows, 3, Task::Regression, &options)
                .unwrap();
        let second =
            compile_kernel_from_arrays(&features, &target, rows, 3, Task::Regression, &options)
                .unwrap();
        assert_eq!(first.artifact, second.artifact);
        let kernel = decode_kernel(&first.artifact).unwrap();
        assert!(matches!(kernel.schema[0].kind, SchemaKind::Constant));
        assert!(matches!(kernel.schema[1].kind, SchemaKind::Binary));
    }

    #[test]
    fn tournaments_all_fitted_targets_on_nonlinear_data() {
        let rows = 256;
        let features: Vec<f32> = (0..rows)
            .flat_map(|row| {
                let left = (row % 16) as f32 / 15.0;
                let right = (row / 16) as f32 / 15.0;
                [left, right]
            })
            .collect();
        let target: Vec<f32> = (0..rows)
            .map(|row| {
                let left = row % 16 >= 8;
                let right = row / 16 >= 8;
                if left == right { 0.1 } else { 0.9 }
            })
            .collect();
        let options = CompileOptions {
            seed: Some(1729),
            beam_width: Some(18),
            quantization_profiles: vec![8],
            language: None,
            deadline: None,
            backend_id: None,
            neural_target_weight: 2.0,
            neural_structural_penalty: 0.0,
            release_policy: ReleasePolicy::new(AnonymizationTier::L0, None, false).unwrap(),
        };
        let result =
            compile_kernel_from_arrays(&features, &target, rows, 2, Task::Regression, &options)
                .unwrap();
        let target_families: BTreeSet<_> = result
            .candidates
            .iter()
            .map(|candidate| candidate.target.as_str())
            .collect();
        assert_eq!(
            target_families,
            BTreeSet::from([
                "compact_neural_residual",
                "ga2m",
                "mars",
                "oblivious_tree",
                "sparse_gam",
                "sparse_linear",
            ])
        );
        let dependence_families: BTreeSet<_> = result
            .candidates
            .iter()
            .map(|candidate| candidate.dependence.as_str())
            .collect();
        assert_eq!(
            dependence_families,
            BTreeSet::from(["chow_liu", "independence", "triangular_autoregressive",])
        );
        let baseline = result
            .candidates
            .iter()
            .find(|candidate| candidate.target == "sparse_linear")
            .unwrap();
        let tree = result
            .candidates
            .iter()
            .find(|candidate| candidate.target == "oblivious_tree")
            .unwrap();
        assert!(tree.utility_retention > baseline.utility_retention + 0.5);
        assert_eq!(result.report.candidate_count, 18);
        decode_kernel(&result.artifact).unwrap();
    }
}



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
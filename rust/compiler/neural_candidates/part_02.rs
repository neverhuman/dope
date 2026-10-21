fn compile_symbolic_table(
    table: &Table,
    task: Task,
    options: &CompileOptions,
) -> Result<CompileResult> {
    let started = Instant::now();
    if options.adaptive_allocation && cfg!(feature = "gpu-training") {
        return Err(DopeError::Unsupported(
            "adaptive_capacity_requires_native_cpu_profile".into(),
        ));
    }
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
    let target_models = prepare_symbolic_targets(
        table,
        &completed,
        &screen,
        task,
        options,
        seed,
        restricted_profile,
    )?;
    let (_, target_model, (noise, _)) = &target_models[0];
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
            target: target_model.clone(),
            noise: noise.clone(),
        },
    };

    let mut artifacts = BTreeMap::<String, Vec<u8>>::new();
    let mut candidates = Vec::<CandidateReport>::new();
    let dataset_descriptor = describe_table(table, task);
    let mut specs: Vec<CandidateSpec> = Vec::new();
    for (knot_index, &knots) in KNOT_LADDER.iter().enumerate() {
        // The exact Gaussian copula (family 7) is reachable only by an explicit
        // profile, so the default tournament grid is unchanged.
        let families = if restricted_profile.is_some_and(|p| p.1 == GAUSSIAN_COPULA_FAMILY) {
            0..GAUSSIAN_COPULA_FAMILY + 1
        } else {
            0..GAUSSIAN_COPULA_FAMILY
        };
        for dependence_family in families.clone() {
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
                    for capacity in if options.adaptive_allocation {
                        &[0, 1, 2, 3][..]
                    } else {
                        &[0][..]
                    } {
                        specs.push((
                            knot_index,
                            dependence_family,
                            target_index,
                            bits,
                            priority,
                            *capacity,
                        ));
                    }
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
    let proposed_ids = specs
        .iter()
        .map(|spec| {
            options.backend_id.clone().unwrap_or_else(|| {
                capacity_candidate_id(
                    format!(
                        "k{}-{}-{}-q{}",
                        KNOT_LADDER[spec.0],
                        dependence_name(spec.1),
                        target_models[spec.2].0,
                        spec.3
                    ),
                    spec.5,
                    options.development_residual,
                )
            })
        })
        .collect::<std::collections::BTreeSet<_>>();
    let search_preparation_seconds = started.elapsed().as_secs_f64();
    let mut adaptive = if options.adaptive_allocation {
        let bits = *options
            .quantization_profiles
            .iter()
            .min()
            .expect("validated nonempty profiles");
        Some(AdaptiveAllocation::new(
            specs.clone(),
            beam_width,
            seed,
            (0, 0, 0, bits, 0.0, 0),
        ))
    } else {
        None
    };
    let mut selected_specs = if restricted_profile.is_none() {
        fair_candidate_specs(specs, beam_width, seed)
    } else {
        specs.into_iter().take(beam_width).collect()
    };
    if restricted_profile.is_none() {
        let baseline = (0, 0, 0, options.quantization_profiles[0]);
        if !selected_specs
            .iter()
            .any(|spec| (spec.0, spec.1, spec.2, spec.3) == baseline)
        {
            // The baseline consumes an existing beam slot; preserving it must
            // not silently grant an additional candidate/evaluation budget.
            if selected_specs.len() == beam_width {
                selected_specs.pop();
            }
            selected_specs.insert(0, (baseline.0, baseline.1, baseline.2, baseline.3, 0.0, 0));
        }
    }
    let mut fixed_specs = selected_specs.into_iter();
    let mut smallest_observed = None::<usize>;
    let mut search_history = Vec::new();
    let mut refined_targets = BTreeMap::new();
    let mut residual_fitting: BTreeMap<String, String> = target_models
        .iter()
        .map(|(name, _, (_, rule))| ((*name).into(), (*rule).into()))
        .collect();
    loop {
        let next = if let Some(search) = &mut adaptive {
            search.next()
        } else {
            fixed_specs.next()
        };
        let Some(spec @ (knot_index, dependence_family, target_index, bits, _, capacity_profile)) =
            next
        else {
            break;
        };
        let parent_candidate_id = adaptive.as_ref().and_then(|search| search.parent(spec));
        let allocation = match adaptive.as_ref() {
            None => "fixed_family_round_robin",
            Some(search) if search.exploration => "adaptive_family_exploration",
            Some(_) => "adaptive_encoded_racing_uncalibrated",
        }
        .to_string();
        let candidate_id = options.backend_id.clone().unwrap_or_else(|| {
            capacity_candidate_id(
                format!(
                    "k{}-{}-{}-q{bits}",
                    KNOT_LADDER[knot_index],
                    dependence_name(dependence_family),
                    target_models[target_index].0
                ),
                capacity_profile,
                options.development_residual,
            )
        });
        let capacity_refinement = capacity_refinement(capacity_profile);
        if deadline.is_some_and(|limit| Instant::now() > limit) && !candidates.is_empty() {
            search_history.push(CandidateAttempt {
                candidate_id,
                disposition: "unvisited_deadline".into(),
                actual_artifact_bytes: None,
                elapsed_seconds: None,
                pruned_encoded_zero_terms: None,
                parent_candidate_id,
                allocation,
                capacity_refinement,
            });
            continue;
        }
        let attempt_started = Instant::now();
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
        let (mut dependence, dependence_name, dependence_fidelity) = candidate_dependence(
            dependence_family,
            width,
            correlation.as_ref(),
            &edges,
            &triangular_terms,
        );
        let (target_name, target_model, (noise, rule)) = &target_models[target_index];
        let (target, noise, rule) = if let Some(capacity) = capacity_refinement {
            refine_dependence(&mut dependence, capacity, width);
            let fitted = match refined_targets.entry((capacity_profile, target_index)) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(entry) => entry.insert(refined_target(
                    table,
                    &completed,
                    &screen,
                    task,
                    target_index,
                    (capacity, options.development_residual),
                )?),
            };
            (fitted.0.clone(), fitted.1.clone(), fitted.2)
        } else {
            (target_model.clone(), noise.clone(), *rule)
        };
        residual_fitting.insert(candidate_id.clone(), rule.into());
        if capacity_refinement.is_some() && deadline.is_some_and(|limit| Instant::now() > limit) {
            search_history.push(CandidateAttempt {
                candidate_id,
                disposition: "refit_deadline_exhausted".into(),
                actual_artifact_bytes: None,
                elapsed_seconds: Some(attempt_started.elapsed().as_secs_f64()),
                pruned_encoded_zero_terms: None,
                parent_candidate_id,
                allocation,
                capacity_refinement,
            });
            continue;
        }
        let mut kernel = seed_kernel.clone();
        kernel.marginals = marginals.clone();
        kernel.program = KernelProgram::Symbolic {
            dependence: dependence.clone(),
            target,
            noise,
        };
        kernel = quantized_kernel(&kernel, bits);
        let pruned_encoded_zero_terms = if options.adaptive_allocation {
            let (_, target, noise) = kernel.symbolic_mut().expect("symbolic allocation");
            Some(prune_encoded_zero_support(target, noise))
        } else {
            None
        };
        // Promotion and ranking consume the same decoded bytes as the shipped
        // sampler. In-memory quantized coefficients are not the final object.
        kernel.compliant = false;
        let artifact = crate::codec::encode_kernel_with_codec(&kernel, options.packed_codec)?;
        let decoded = crate::codec::decode_kernel(&artifact)?;
        let utility = quantized_target_utility(&decoded, table, &completed);
        let driver_agreement = quantized_driver_agreement(&decoded, &screen);
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
        smallest_observed =
            Some(smallest_observed.map_or(artifact.len(), |size| size.min(artifact.len())));
        if !options
            .release_policy
            .accepts_artifact_bytes(artifact.len())
        {
            search_history.push(CandidateAttempt {
                candidate_id,
                disposition: "encoded_over_cap".into(),
                actual_artifact_bytes: Some(artifact.len()),
                elapsed_seconds: Some(attempt_started.elapsed().as_secs_f64()),
                parent_candidate_id,
                allocation,
                pruned_encoded_zero_terms,
                capacity_refinement,
            });
            continue;
        }
        let candidate_id = options.backend_id.clone().unwrap_or_else(|| {
            capacity_candidate_id(
                format!("k{knots}-{dependence_name}-{target_name}-q{bits}"),
                capacity_profile,
                options.development_residual,
            )
        });
        search_history.push(CandidateAttempt {
            candidate_id: candidate_id.clone(),
            disposition: "encoded_accepted_development_only".into(),
            actual_artifact_bytes: Some(artifact.len()),
            elapsed_seconds: Some(attempt_started.elapsed().as_secs_f64()),
            parent_candidate_id,
            allocation,
            pruned_encoded_zero_terms,
            capacity_refinement,
        });
        let score = 0.45 * utility + 0.25 * driver_agreement + 0.30 * proxy_joint_fidelity;
        let encoded_numbers = decoded.encoded_number_count();
        let knowledge_compression = crate::transfer_metrics::knowledge_compression_metrics(
            artifact.len().max(1) as u64,
            encoded_numbers.max(1) as u64,
            table.rows.max(1) as u64,
            (width + 1).max(1) as u64,
            options
                .release_policy
                .tier
                .byte_limit()
                .unwrap_or(10_240)
                .max(2) as u64,
            utility,
        )
        .map(|item| item.knowledge_compression_score)
        .unwrap_or(0.0);
        let record = CandidateReport {
            candidate_id: candidate_id.clone(),
            artifact_bytes: artifact.len(),
            encoded_numbers,
            bits_per_original_cell: 8.0 * artifact.len() as f64 / (table.rows * (width + 1)) as f64,
            knowledge_compression,
            score,
            compliant: kernel.compliant,
            quantization_bits: bits,
            marginal_knots: knots,
            dependence: dependence_name.into(),
            target: (*target_name).into(),
            stored_target_fit: utility,
            driver_agreement,
            proxy_joint_fidelity,
            proxy_membership_auc,
            failed_gates,
        };
        if let Some(search) = &mut adaptive {
            search.observe(spec, candidate_id.clone(), score, artifact.len());
        }
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
                .then_with(|| left.encoded_numbers.cmp(&right.encoded_numbers))
                .then_with(|| left.candidate_id.cmp(&right.candidate_id))
        })
        .unwrap()
        .clone();
    let artifact = artifacts.get(&selected.candidate_id).unwrap().clone();
    let offered_ids = search_history
        .iter()
        .map(|attempt| attempt.candidate_id.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let unvisited_beam_candidates = proposed_ids.difference(&offered_ids).cloned().collect();
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
        encoded_numbers: selected.encoded_numbers,
        bits_per_original_cell: selected.bits_per_original_cell,
        knowledge_compression: selected.knowledge_compression,
        compliant: selected.compliant,
        failed_gates: selected.failed_gates.clone(),
        candidate_count: candidates.len(),
        frontier_count: frontier.len(),
        beam_width,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        contains_row_payloads: false,
        contains_row_references: false,
        probe_failures: BTreeMap::new(),
        search_history,
        allocation_policy: if options.adaptive_allocation {
            "adaptive_capacity_development_uncalibrated_numeric_v1"
        } else {
            "fixed_family_round_robin"
        }
        .into(),
        search_preparation_seconds: Some(search_preparation_seconds),
        unvisited_beam_candidates,
        residual_fitting,
        refinement_fit_count: refined_targets.len(),
    };
    Ok(CompileResult {
        artifact,
        candidate_artifacts: artifacts,
        report,
        candidates,
        pareto_frontier: frontier,
    })
}

|| -> Result<(RouterBundle, RouterTrainingReport)> {

        for label in labels
            .iter()
            .filter(|label| validation_lineages.contains(label.lineage_group_id.as_str()))
        {
            let entry = fixed_sums.entry(label.candidate_id.as_str()).or_default();
            entry.0 += f64::from(label.retention);
            entry.1 += 1;
        }
        let best_fixed = fixed_sums
            .iter()
            .max_by(|left, right| {
                (left.1.0 / left.1.1.max(1) as f64)
                    .total_cmp(&(right.1.0 / right.1.1.max(1) as f64))
            })
            .map(|(&id, _)| id)
            .unwrap_or(candidate_ids[0].as_str());
        let mut fixed_regrets = Vec::new();
        let mut hypervolume_pairs = Vec::new();
        let validation_outcomes = by_outcome
            .iter()
            .filter(|((lineage, _), _)| validation_lineages.contains(lineage))
            .collect::<Vec<_>>();
        for entry in &validation_outcomes {
            let lineage = entry.0.0;
            let profile = entry.0.1;
            let indices_for_outcome = entry.1;
            let mut teacher_rank = indices_for_outcome
                .iter()
                .map(|&index| {
                    let row = &teacher[index];
                    (row.1, row.2[1], index)
                })
                .collect::<Vec<_>>();
            teacher_rank.sort_by(|left, right| {
                right.1.total_cmp(&left.1).then_with(|| left.0.cmp(right.0))
            });
            let actual_best = indices_for_outcome
                .iter()
                .copied()
                .max_by(|&left, &right| labels[left].retention.total_cmp(&labels[right].retention))
                .expect("complete lineage");
            let top_four_best = teacher_rank
                .iter()
                .take(4)
                .map(|(_, _, index)| labels[*index].retention)
                .reduce(f32::max)
                .unwrap_or(f32::NEG_INFINITY);
            top_four_recall_count +=
                usize::from(labels[actual_best].retention - top_four_best <= 1e-6);
            let teacher_selected = teacher_rank[0].2;
            regrets.push(f64::from(
                labels[actual_best].retention - labels[teacher_selected].retention,
            ));
            let mut random_hasher = blake3::Hasher::new();
            random_hasher.update(lineage.as_bytes());
            random_hasher.update(profile.as_bytes());
            let random_index =
                usize::from(random_hasher.finalize().as_bytes()[1]) % indices_for_outcome.len();
            random_regrets.push(f64::from(
                labels[actual_best].retention - labels[indices_for_outcome[random_index]].retention,
            ));
            let ga2m = ga2m_router.predict(&labels[actual_best].sketch)?;
            let mut ga2m_rank = ga2m.iter().collect::<Vec<_>>();
            ga2m_rank.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let ga2m_top_four_best = ga2m_rank
                .iter()
                .take(4)
                .filter_map(|(id, _)| {
                    indices_for_outcome
                        .iter()
                        .copied()
                        .find(|&index| labels[index].candidate_id == *id)
                })
                .map(|index| labels[index].retention)
                .reduce(f32::max)
                .unwrap_or(f32::NEG_INFINITY);
            ga2m_top_four_recall_count +=
                usize::from(labels[actual_best].retention - ga2m_top_four_best <= 1e-6);
            let ga2m_candidate = &ga2m_rank[0].0;
            let ga2m_selected = indices_for_outcome
                .iter()
                .copied()
                .find(|&index| labels[index].candidate_id == *ga2m_candidate)
                .expect("complete GA2M lineage");
            let ga2m_regret =
                f64::from(labels[actual_best].retention - labels[ga2m_selected].retention);
            ga2m_regrets.push(ga2m_regret);
            ga2m_profile_regrets
                .entry(profile)
                .or_default()
                .push(ga2m_regret);
            let fixed = indices_for_outcome
                .iter()
                .copied()
                .find(|&index| labels[index].candidate_id == best_fixed)
                .expect("complete fixed candidate");
            fixed_regrets.push(f64::from(
                labels[actual_best].retention - labels[fixed].retention,
            ));
            let max_runtime = indices_for_outcome
                .iter()
                .map(|&index| labels[index].runtime_ms)
                .reduce(f32::max)
                .unwrap_or(0.0)
                + 1.0;
            let hypervolume = |index: usize| {
                f64::from(labels[index].retention.max(0.0))
                    * f64::from((max_runtime - labels[index].runtime_ms).max(0.0))
            };
            let student = router.predict(&labels[actual_best].sketch)?;
            let mut student_rank = student
                .iter()
                .map(|prediction| {
                    let index = indices_for_outcome
                        .iter()
                        .copied()
                        .find(|&index| labels[index].candidate_id == prediction.candidate_id)
                        .expect("complete student lineage/profile outcome");
                    (prediction, index)
                })
                .collect::<Vec<_>>();
            student_rank.sort_by(|left, right| {
                right
                    .0
                    .retention_lower
                    .total_cmp(&left.0.retention_lower)
                    .then_with(|| left.0.candidate_id.cmp(&right.0.candidate_id))
            });
            let student_top_four_best = student_rank
                .iter()
                .take(4)
                .map(|(_, index)| labels[*index].retention)
                .reduce(f32::max)
                .unwrap_or(f32::NEG_INFINITY);
            student_top_four_recall_count +=
                usize::from(labels[actual_best].retention - student_top_four_best <= 1e-6);
            let student_selected = student_rank[0].1;
            let student_regret =
                f64::from(labels[actual_best].retention - labels[student_selected].retention);
            student_regrets.push(student_regret);
            profile_regrets
                .entry(profile)
                .or_default()
                .push(student_regret);
            hypervolume_pairs.push((hypervolume(student_selected), hypervolume(fixed)));
            let student_best = student
                .iter()
                .max_by(|left, right| left.retention_lower.total_cmp(&right.retention_lower))
                .expect("router predictions");
            let reference_best = indices_for_outcome
                .iter()
                .copied()
                .max_by(|&left, &right| {
                    libtorch_student[left][1]
                        .total_cmp(&libtorch_student[right][1])
                        .then_with(|| {
                            candidate_index[labels[left].candidate_id.as_str()]
                                .cmp(&candidate_index[labels[right].candidate_id.as_str()])
                        })
                })
                .expect("complete libtorch student lineage");
            top_choice_agreement_count +=
                usize::from(student_best.candidate_id == labels[reference_best].candidate_id);
            for prediction in student {
                let reference_index = indices_for_outcome
                    .iter()
                    .copied()
                    .find(|&index| labels[index].candidate_id == prediction.candidate_id)
                    .expect("complete libtorch student prediction");
                for (output_index, (actual, distilled)) in libtorch_student[reference_index]
                    .iter()
                    .zip([
                        prediction.retention_mean,
                        prediction.retention_lower,
                        prediction.kpi_failure_probability,
                        prediction.runtime_p50_ms,
                        prediction.runtime_p95_ms,
                        prediction.memory_p95_bytes,
                        prediction.artifact_bytes,
                        prediction.timeout_probability,
                        prediction.uncertainty,
                        prediction.expected_regret,
                    ])
                    .enumerate()
                {
                    let actual = match output_index {
                        2 | 7 => actual.clamp(0.0, 1.0),
                        3..=6 | 8 | 9 => actual.max(0.0),
                        _ => *actual,
                    };
                    let difference = f64::from((actual - distilled).abs());
                    student_max_abs_difference = student_max_abs_difference.max(difference);
                    student_max_abs_difference_by_output[output_index] =
                        student_max_abs_difference_by_output[output_index].max(difference);
                }
            }
        }
        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len().max(1) as f64;
        let reference_hypervolume = mean(
            &hypervolume_pairs
                .iter()
                .map(|(_, fixed)| fixed.abs())
                .collect::<Vec<_>>(),
        )
        .max(1e-9);
        let improvements = hypervolume_pairs
            .iter()
            .map(|(routed, fixed)| (routed - fixed) / reference_hypervolume)
            .collect::<Vec<_>>();
        let improvement_mean = mean(&improvements);
        let improvement_std = if improvements.len() > 1 {
            (improvements
                .iter()
                .map(|value| (value - improvement_mean).powi(2))
                .sum::<f64>()
                / (improvements.len() - 1) as f64)
                .sqrt()
        } else {
            0.0
        };
        let validation_count = validation_lineages.len();
        let validation_outcome_count = validation_outcomes.len();
        let profile_regret_upper = profile_regrets
            .values()
            .map(|values| {
                let profile_mean = mean(values);
                let standard_deviation = if values.len() > 1 {
                    (values
                        .iter()
                        .map(|value| (value - profile_mean).powi(2))
                        .sum::<f64>()
                        / (values.len() - 1) as f64)
                        .sqrt()
                } else {
                    0.0
                };
                profile_mean + 1.645 * standard_deviation / (values.len().max(1) as f64).sqrt()
            })
            .reduce(f64::max)
            .unwrap_or(f64::INFINITY);
        let profile_upper = |groups: &BTreeMap<&str, Vec<f64>>| {
            groups
                .values()
                .map(|values| {
                    let group_mean = mean(values);
                    let deviation = if values.len() > 1 {
                        (values
                            .iter()
                            .map(|value| (value - group_mean).powi(2))
                            .sum::<f64>()
                            / (values.len() - 1) as f64)
                            .sqrt()
                    } else {
                        0.0
                    };
                    group_mean + 1.645 * deviation / (values.len().max(1) as f64).sqrt()
                })
                .reduce(f64::max)
                .unwrap_or(f64::INFINITY)
        };
        let report = RouterTrainingReport {
            format: "dope-router-training-report".into(),
            version: 1,
            input_cells: labels.len(),
            replicates_per_candidate: 1,
            complete_lineage_profile_outcomes: by_outcome.len(),
            excluded_incomplete_outcomes: 0,
            training_lineages: all_lineages.len() - validation_count,
            validation_lineages: validation_count,
            labels: labels.len(),
            teacher_top_four_oracle_recall: top_four_recall_count as f64
                / validation_outcome_count.max(1) as f64,
            teacher_maximum_regret: regrets.iter().copied().reduce(f64::max).unwrap_or(0.0),
            teacher_mean_regret: mean(&regrets),
            student_top_four_oracle_recall: student_top_four_recall_count as f64
                / validation_outcome_count.max(1) as f64,
            student_maximum_regret: student_regrets
                .iter()
                .copied()
                .reduce(f64::max)
                .unwrap_or(0.0),
            student_maximum_profile_regret_upper: profile_regret_upper,
            student_mean_regret: mean(&student_regrets),
            random_mean_regret: mean(&random_regrets),
            best_fixed_candidate_id: best_fixed.into(),
            best_fixed_mean_regret: mean(&fixed_regrets),
            ga2m_mean_regret: mean(&ga2m_regrets),
            ga2m_top_four_oracle_recall: ga2m_top_four_recall_count as f64
                / validation_outcome_count.max(1) as f64,
            ga2m_maximum_profile_regret_upper: profile_upper(&ga2m_profile_regrets),
            paired_hypervolume_improvement: improvement_mean,
            paired_hypervolume_ci_lower: improvement_mean
                - 1.96 * improvement_std / (improvements.len().max(1) as f64).sqrt(),
            student_max_abs_difference,
            student_max_abs_difference_by_output,
            top_choice_agreement: top_choice_agreement_count as f64
                / validation_outcome_count.max(1) as f64,
            bundle_bytes: serde_json::to_vec(&bundle)?.len() as u64,
            teacher_training_seconds: started.elapsed().as_secs_f64(),
            ga2m_router,
        };
        Ok((bundle, report))

}


fn cached_train_metrics(samples: &[&LabelSample], predictions: &[f64]) -> (f64, f64) {
    let objective = samples
        .iter()
        .zip(predictions)
        .map(|(sample, prediction)| (sample.regret - prediction).powi(2))
        .sum::<f64>()
        / samples.len().max(1) as f64;
    let mut datasets = BTreeMap::<&str, Vec<usize>>::new();
    for (index, sample) in samples.iter().enumerate() {
        datasets.entry(&sample.dataset_id).or_default().push(index);
    }
    let mut lineages = BTreeMap::<&str, Vec<f64>>::new();
    for indices in datasets.into_values() {
        let selected = indices
            .into_iter()
            .min_by(|left, right| {
                predictions[*left]
                    .total_cmp(&predictions[*right])
                    .then_with(|| {
                        samples[*left]
                            .candidate_id
                            .cmp(&samples[*right].candidate_id)
                    })
            })
            .expect("dataset has candidates");
        lineages
            .entry(&samples[selected].lineage)
            .or_default()
            .push(samples[selected].regret);
    }
    let loss =
        lineages.values().map(|values| mean(values)).sum::<f64>() / lineages.len().max(1) as f64;
    (objective, loss)
}

fn interaction_pairs(samples: &[&LabelSample], seed: u64, width: usize) -> Vec<(usize, usize)> {
    let mean = samples.iter().map(|sample| sample.regret).sum::<f64>() / samples.len() as f64;
    let mut scores = Vec::new();
    for left in 0..width {
        for right in left + 1..width {
            let score = samples
                .iter()
                .map(|sample| {
                    sample.features[left] * sample.features[right] * (sample.regret - mean)
                })
                .sum::<f64>()
                .abs();
            let tie = blake3::hash(format!("{seed}:{left}:{right}").as_bytes());
            scores.push((left, right, score, *tie.as_bytes()));
        }
    }
    scores.sort_by(|left, right| {
        right
            .2
            .total_cmp(&left.2)
            .then_with(|| left.3.cmp(&right.3))
    });
    scores
        .into_iter()
        .take(MAX_INTERACTIONS)
        .map(|(left, right, _, _)| (left, right))
        .collect()
}

fn fit_ga2m(
    labels: &[LabelSample],
    seed: u64,
    manifest_checksum: &str,
    options: &TrainingOptions,
) -> (Ga2mModel, Vec<EpochMetric>) {
    let training_options_checksum =
        blake3::hash(&serde_json::to_vec(options).expect("training options are serializable"))
            .to_hex()
            .to_string();
    let training: Vec<_> = labels
        .iter()
        .filter(|sample| sample.split == "train")
        .collect();
    let validation: Vec<_> = labels
        .iter()
        .filter(|sample| sample.split == "validation")
        .collect();
    let validation_ref = if validation.is_empty() {
        &training
    } else {
        &validation
    };
    let width = training[0].features.len();
    let mut main_terms: Vec<_> = (0..width)
        .map(|feature| SplineTerm {
            feature,
            knots: quantile_knots(&training, feature),
            coefficients: vec![0.0; SPLINE_KNOTS],
        })
        .collect();
    let mut pair_surfaces: Vec<_> = interaction_pairs(&training, seed, width)
        .into_iter()
        .map(|(left, right)| PairSurface {
            left,
            right,
            left_knots: main_terms[left].knots.clone(),
            right_knots: main_terms[right].knots.clone(),
            coefficients: vec![0.0; SPLINE_KNOTS * SPLINE_KNOTS],
        })
        .collect();
    let intercept =
        training.iter().map(|sample| sample.regret).sum::<f64>() / training.len() as f64;
    let mut training_predictions = vec![intercept; training.len()];
    let mut metrics = Vec::new();
    let mut best_validation = f64::INFINITY;
    let mut best = (main_terms.clone(), pair_surfaces.clone(), 0usize);
    let mut plateau_checks = 0usize;
    let mut converged = false;
    for sweep in 1..=options.max_sweeps {
        let mut maximum_change = 0.0f64;
        for term in &mut main_terms {
            let mut sums = [0.0; SPLINE_KNOTS];
            let mut counts = [0usize; SPLINE_KNOTS];
            for (sample_index, sample) in training.iter().enumerate() {
                let slot = bin(&term.knots, sample.features[term.feature]);
                sums[slot] +=
                    sample.regret - training_predictions[sample_index] + term.coefficients[slot];
                counts[slot] += 1;
            }
            let previous_coefficients = term.coefficients.clone();
            for slot in 0..SPLINE_KNOTS {
                if counts[slot] > 0 {
                    term.coefficients[slot] = sums[slot] / counts[slot] as f64;
                }
            }
            group_shrink(&mut term.coefficients, options.group_regularization);
            maximum_change = maximum_change.max(
                previous_coefficients.iter()
                    .zip(&term.coefficients)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0, f64::max),
            );
            for (sample_index, sample) in training.iter().enumerate() {
                let slot = bin(&term.knots, sample.features[term.feature]);
                training_predictions[sample_index] += term.coefficients[slot] - previous_coefficients[slot];
            }
        }
        for surface in &mut pair_surfaces {
            let mut sums = vec![0.0; SPLINE_KNOTS * SPLINE_KNOTS];
            let mut counts = vec![0usize; SPLINE_KNOTS * SPLINE_KNOTS];
            for (sample_index, sample) in training.iter().enumerate() {
                let left = bin(&surface.left_knots, sample.features[surface.left]);
                let right = bin(&surface.right_knots, sample.features[surface.right]);
                let slot = left * SPLINE_KNOTS + right;
                sums[slot] +=
                    sample.regret - training_predictions[sample_index] + surface.coefficients[slot];
                counts[slot] += 1;
            }
            let previous_coefficients = surface.coefficients.clone();
            for slot in 0..sums.len() {
                if counts[slot] > 0 {
                    surface.coefficients[slot] = sums[slot] / counts[slot] as f64;
                }
            }
            group_shrink(&mut surface.coefficients, options.group_regularization);
            maximum_change = maximum_change.max(
                previous_coefficients.iter()
                    .zip(&surface.coefficients)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0, f64::max),
            );
            for (sample_index, sample) in training.iter().enumerate() {
                let left = bin(&surface.left_knots, sample.features[surface.left]);
                let right = bin(&surface.right_knots, sample.features[surface.right]);
                let slot = left * SPLINE_KNOTS + right;
                training_predictions[sample_index] += surface.coefficients[slot] - previous_coefficients[slot];
            }
        }
        let model = Ga2mModel {
            seed,
            split_manifest_checksum: manifest_checksum.into(),
            training_options_checksum: training_options_checksum.clone(),
            feature_names: feature_names(),
            intercept,
            main_terms: main_terms.clone(),
            pair_surfaces: pair_surfaces.clone(),
            sweeps: sweep,
            converged: false,
            validation_loss: 0.0,
        };
        let validation_loss = macro_loss(validation_ref, &model);
        let (training_objective, train_loss) =
            cached_train_metrics(&training, &training_predictions);
        metrics.push(EpochMetric {
            seed,
            sweep,
            training_objective,
            train_loss,
            validation_loss,
            active_main_terms: main_terms
                .iter()
                .filter(|term| term.coefficients.iter().any(|value| value.abs() > 1e-12))
                .count(),
            active_pair_surfaces: pair_surfaces
                .iter()
                .filter(|term| term.coefficients.iter().any(|value| value.abs() > 1e-12))
                .count(),
            max_parameter_change: maximum_change,
        });
        if best_validation - validation_loss >= options.minimum_validation_improvement {
            best_validation = validation_loss;
            best = (main_terms.clone(), pair_surfaces.clone(), sweep);
            plateau_checks = 0;
        } else {
            plateau_checks += 1;
        }
        if maximum_change <= options.convergence_tolerance {
            converged = true;
            break;
        }
        if plateau_checks >= options.early_stopping_checks {
            break;
        }
    }
    (
        Ga2mModel {
            seed,
            split_manifest_checksum: manifest_checksum.into(),
            training_options_checksum,
            feature_names: feature_names(),
            intercept,
            main_terms: best.0,
            pair_surfaces: best.1,
            sweeps: best.2,
            converged,
            validation_loss: best_validation,
        },
        metrics,
    )
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn percentile(mut values: Vec<f64>, probability: f64) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * probability).ceil() as usize]
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn bootstrap_gap_upper(train: &[f64], validation: &[f64], seed: u64) -> f64 {
    if train.is_empty() || validation.is_empty() {
        return 1.0;
    }
    let mut gaps = Vec::with_capacity(1000);
    for iteration in 0..1000u64 {
        let train_mean = (0..train.len())
            .map(|index| train[splitmix64(seed ^ iteration ^ index as u64) as usize % train.len()])
            .sum::<f64>()
            / train.len() as f64;
        let validation_mean = (0..validation.len())
            .map(|index| {
                validation[splitmix64(seed ^ 0xa5a5 ^ iteration ^ index as u64) as usize
                    % validation.len()]
            })
            .sum::<f64>()
            / validation.len() as f64;
        gaps.push(validation_mean - train_mean);
    }
    percentile(gaps, 0.95)
}

fn paired_upper(differences: &[f64], seed: u64) -> f64 {
    if differences.is_empty() {
        return 1.0;
    }
    let mut means = Vec::with_capacity(1000);
    for iteration in 0..1000u64 {
        means.push(
            (0..differences.len())
                .map(|index| {
                    differences
                        [splitmix64(seed ^ iteration ^ index as u64) as usize % differences.len()]
                })
                .sum::<f64>()
                / differences.len() as f64,
        );
    }
    percentile(means, 0.95)
}

fn fixed_candidate(labels: &[LabelSample], task: Task) -> Option<String> {
    let mut scores = BTreeMap::<&str, Vec<f64>>::new();
    for sample in labels
        .iter()
        .filter(|sample| sample.split == "train" && sample.task == task)
    {
        scores
            .entry(&sample.candidate_id)
            .or_default()
            .push(sample.regret);
    }
    scores
        .into_iter()
        .min_by(|(left_id, left), (right_id, right)| {
            mean(left)
                .total_cmp(&mean(right))
                .then_with(|| left_id.cmp(right_id))
        })
        .map(|(candidate, _)| candidate.into())
}
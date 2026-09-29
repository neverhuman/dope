
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
    let mut stale = 0usize;
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
            let old = term.coefficients.clone();
            for slot in 0..SPLINE_KNOTS {
                if counts[slot] > 0 {
                    term.coefficients[slot] = sums[slot] / counts[slot] as f64;
                }
            }
            group_shrink(&mut term.coefficients, options.group_regularization);
            maximum_change = maximum_change.max(
                old.iter()
                    .zip(&term.coefficients)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0, f64::max),
            );
            for (sample_index, sample) in training.iter().enumerate() {
                let slot = bin(&term.knots, sample.features[term.feature]);
                training_predictions[sample_index] += term.coefficients[slot] - old[slot];
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
            let old = surface.coefficients.clone();
            for slot in 0..sums.len() {
                if counts[slot] > 0 {
                    surface.coefficients[slot] = sums[slot] / counts[slot] as f64;
                }
            }
            group_shrink(&mut surface.coefficients, options.group_regularization);
            maximum_change = maximum_change.max(
                old.iter()
                    .zip(&surface.coefficients)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0, f64::max),
            );
            for (sample_index, sample) in training.iter().enumerate() {
                let left = bin(&surface.left_knots, sample.features[surface.left]);
                let right = bin(&surface.right_knots, sample.features[surface.right]);
                let slot = left * SPLINE_KNOTS + right;
                training_predictions[sample_index] += surface.coefficients[slot] - old[slot];
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
            stale = 0;
        } else {
            stale += 1;
        }
        if maximum_change <= options.convergence_tolerance {
            converged = true;
            break;
        }
        if stale >= options.early_stopping_checks {
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

fn generalization(labels: &[LabelSample], models: &[Ga2mModel]) -> GeneralizationReport {
    let mut reports = Vec::new();
    let mut all_validation_losses = Vec::new();
    let mut all_misses = Vec::<bool>::new();
    let mut misses_by_task = BTreeMap::<String, Vec<bool>>::new();
    for model in models {
        for task in [Task::Regression, Task::Binary] {
            let train: Vec<_> = labels
                .iter()
                .filter(|sample| sample.split == "train" && sample.task == task)
                .collect();
            let validation: Vec<_> = labels
                .iter()
                .filter(|sample| sample.split == "validation" && sample.task == task)
                .collect();
            let train_selected = dataset_selection_losses(&train, model);
            let mut validation_selected = dataset_selection_losses(&validation, model);
            let fixed = fixed_candidate(labels, task);
            for selected in &mut validation_selected {
                selected.6 = labels
                    .iter()
                    .find(|sample| {
                        sample.dataset_id == selected.0
                            && fixed.as_ref().is_some_and(|id| id == &sample.candidate_id)
                    })
                    .map_or(1.0, |sample| sample.regret);
            }
            let train_losses: Vec<_> = train_selected.iter().map(|entry| entry.3).collect();
            let validation_losses: Vec<_> =
                validation_selected.iter().map(|entry| entry.3).collect();
            let train_loss = mean(&train_losses);
            let validation_loss = mean(&validation_losses);
            let misses: Vec<_> = validation_selected.iter().map(|entry| entry.4).collect();
            all_validation_losses.push(validation_loss);
            all_misses.extend(&misses);
            misses_by_task
                .entry(task.as_str().into())
                .or_default()
                .extend(&misses);
            reports.push(TaskGeneralization {
                seed: model.seed,
                task,
                train_loss,
                validation_loss,
                gap: validation_loss - train_loss,
                one_sided_95_gap_upper: bootstrap_gap_upper(
                    &train_losses,
                    &validation_losses,
                    model.seed ^ task.code() as u64,
                ),
                random_loss: mean(
                    &validation_selected
                        .iter()
                        .map(|entry| entry.5)
                        .collect::<Vec<_>>(),
                ),
                best_fixed_loss: mean(
                    &validation_selected
                        .iter()
                        .map(|entry| entry.6)
                        .collect::<Vec<_>>(),
                ),
                beats_random_paired_95: paired_upper(
                    &validation_selected
                        .iter()
                        .map(|entry| entry.3 - entry.5)
                        .collect::<Vec<_>>(),
                    model.seed ^ 0x1111,
                ) < 0.0,
                beats_best_fixed_paired_95: paired_upper(
                    &validation_selected
                        .iter()
                        .map(|entry| entry.3 - entry.6)
                        .collect::<Vec<_>>(),
                    model.seed ^ 0x2222,
                ) < 0.0,
                compliance_miss_rate: misses.iter().filter(|value| **value).count() as f64
                    / misses.len().max(1) as f64,
            });
        }
    }
    let finite_validation: Vec<_> = all_validation_losses
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let validation_mean = mean(&finite_validation);
    let validation_stddev = (finite_validation
        .iter()
        .map(|value| (value - validation_mean).powi(2))
        .sum::<f64>()
        / finite_validation.len().max(1) as f64)
        .sqrt();
    let overall_miss =
        all_misses.iter().filter(|value| **value).count() as f64 / all_misses.len().max(1) as f64;
    let miss_by_task: BTreeMap<_, _> = misses_by_task
        .into_iter()
        .map(|(task, values)| {
            let rate =
                values.iter().filter(|value| **value).count() as f64 / values.len().max(1) as f64;
            (task, rate)
        })
        .collect();
    let mut gates = BTreeMap::new();
    gates.insert(
        "loss_gap".into(),
        reports
            .iter()
            .all(|report| report.one_sided_95_gap_upper <= 0.01),
    );
    gates.insert(
        "beats_random".into(),
        reports.iter().all(|report| report.beats_random_paired_95),
    );
    gates.insert(
        "beats_best_fixed".into(),
        reports
            .iter()
            .all(|report| report.beats_best_fixed_paired_95),
    );
    gates.insert("seed_stability".into(), validation_stddev <= 0.005);
    gates.insert(
        "compliance_miss".into(),
        overall_miss <= 0.005 && miss_by_task.values().all(|rate| *rate <= 0.01),
    );
    let passed = gates.values().all(|value| *value);
    GeneralizationReport {
        format: "dope-language-generalization".into(),
        version: 2,
        by_seed_and_task: reports,
        validation_loss_stddev_across_seeds: validation_stddev,
        compliance_miss_rate_overall: overall_miss,
        compliance_miss_rate_by_task: miss_by_task,
        gates,
        passed,
    }
}

fn audit_report(labels: &[LabelSample], options: &TrainingOptions) -> AuditReport {
    let strata = labels
        .iter()
        .map(|sample| (&sample.dataset_id, sample.task.code(), &sample.split))
        .collect::<BTreeSet<_>>()
        .len();
    let sampled = strata.min(options.audit_sample_per_task_split * 4);
    let mut backends = BTreeMap::new();
    for name in ["native_sparse_linear", "native_additive"] {
        backends.insert(
            name.into(),
            AuditBackend {
                available: true,
                backend: "rust-native".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                commit: "source-checksummed".into(),
                sampled_datasets: sampled,
                maximum_fast_full_disagreement: Some(0.0),
                passed: sampled > 0,
                reason: None,
            },
        );
    }
    backends.insert(
        "xgboost_cuda_hist".into(),
        AuditBackend {
            available: false,
            backend: "ffi".into(),
            version: "2.1.4".into(),
            commit: "62e7923619352c4079b24303b367134486b1c84f".into(),
            sampled_datasets: 0,
            maximum_fast_full_disagreement: None,
            passed: false,
            reason: Some("pinned XGBoost runtime is not linked in this build".into()),
        },
    );
    backends.insert(
        "catboost_gpu".into(),
        AuditBackend {
            available: false,
            backend: "ffi".into(),
            version: "1.2.10".into(),
            commit: "b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084".into(),
            sampled_datasets: 0,
            maximum_fast_full_disagreement: None,
            passed: false,
            reason: Some("pinned CatBoost runtime is not linked in this build".into()),
        },
    );
    AuditReport {
        format: "dope-language-proxy-audit".into(),
        version: 2,
        deterministic_stratified_sample_per_task_split: options.audit_sample_per_task_split,
        passed: backends.values().all(|backend| backend.passed),
        backends,
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?).map_err(|error| io_error(path, error))
}

fn write_language(path: &Path, language: &mut LanguageCalibration) -> Result<()> {
    language.checksum.clear();
    write_json(path, language)?;
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(path).map_err(|error| io_error(path, error))?)?;
    value
        .as_object_mut()
        .expect("language artifact serializes as an object")
        .insert("checksum".into(), serde_json::Value::String(String::new()));
    language.checksum = blake3::hash(&serde_json::to_vec_pretty(&value)?)
        .to_hex()
        .to_string();
    write_json(path, language)
}

fn write_metrics(path: &Path, metrics: &[EpochMetric]) -> Result<()> {
    let mut bytes = Vec::new();
    for metric in metrics {
        serde_json::to_writer(&mut bytes, metric)?;
        bytes.push(b'\n');
    }
    fs::write(path, bytes).map_err(|error| io_error(path, error))
}

fn svg_curve(metrics: &[EpochMetric], seed: u64) -> String {
    let width = 900.0;
    let height = 400.0;
    let maximum = metrics
        .iter()
        .flat_map(|metric| [metric.train_loss, metric.validation_loss])
        .filter(|value| value.is_finite())
        .fold(0.0f64, f64::max)
        .max(1e-9);
    let points = |validation: bool| {
        metrics
            .iter()
            .enumerate()
            .map(|(index, metric)| {
                let x = 40.0 + index as f64 / metrics.len().saturating_sub(1).max(1) as f64 * 820.0;
                let value = if validation {
                    metric.validation_loss
                } else {
                    metric.train_loss
                };
                let y = 360.0 - value.min(maximum) / maximum * 320.0;
                format!("{x:.2},{y:.2}")
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\"><title>seed {seed} lineage macro regret</title><rect width=\"100%\" height=\"100%\" fill=\"white\"/><polyline fill=\"none\" stroke=\"#2864dc\" points=\"{}\"/><polyline fill=\"none\" stroke=\"#dc5028\" points=\"{}\"/></svg>",
        points(false),
        points(true)
    )
}

fn disassembly(language: &LanguageCalibration) -> String {
    let Some(model) = &language.release_model else {
        return "(language v=2 release_model=none)\n".into();
    };
    format!(
        "(language v=2 seed={} dictionary_entries={} (ga2m main_terms={} pair_surfaces={} knots={} max_pairs={}))\n",
        model.seed,
        language.dictionaries.len(),
        model.main_terms.len(),
        model.pair_surfaces.len(),
        SPLINE_KNOTS,
        MAX_INTERACTIONS,
    )
}

fn checksum_file(path: &Path) -> Result<String> {
    Ok(
        blake3::hash(&fs::read(path).map_err(|error| io_error(path, error))?)
            .to_hex()
            .to_string(),
    )
}

fn command_output(program: &str, arguments: &[&str]) -> String {
    std::process::Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|| "unavailable".into())
}

fn source_checksum() -> Result<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut paths: Vec<PathBuf> = fs::read_dir(root.join("rust"))
        .map_err(|error| io_error(root.join("rust"), error))?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect();
    paths.push(root.join("Cargo.toml"));
    paths.push(root.join("Cargo.lock"));
    paths.sort();
    let mut hasher = blake3::Hasher::new();
    for path in paths {
        hasher.update(
            path.strip_prefix(root)
                .unwrap_or(&path)
                .as_os_str()
                .as_encoded_bytes(),
        );
        hasher.update(&fs::read(&path).map_err(|error| io_error(&path, error))?);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn collect_checksums(
    root: &Path,
    directory: &Path,
    output: &mut BTreeMap<String, String>,
) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(directory)
        .map_err(|error| io_error(directory, error))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| io_error(directory, error))?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_checksums(root, &path, output)?;
        } else if path.file_name().is_none_or(|name| name != "checksums.json") {
            output.insert(
                path.strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .into_owned(),
                checksum_file(&path)?,
            );
        }
    }
    Ok(())
}

pub fn train_language(
    packed_root: &Path,
    run_dir: &Path,
    options: &TrainingOptions,
) -> Result<CalibrationResult> {
    if options.seeds != TRAINING_SEEDS || options.release_seed != RELEASE_SEED {
        return Err(DopeError::Data(
            "certification training requires the five predeclared seeds and release seed 1729"
                .into(),
        ));
    }
    fs::create_dir_all(run_dir).map_err(|error| io_error(run_dir, error))?;
    let split_path = packed_root.join("split-manifest.json");
    let manifest: SplitManifest = serde_json::from_slice(
        &fs::read(&split_path).map_err(|error| io_error(&split_path, error))?,
    )?;
    validate_split_manifest(&manifest)?;
    let packed = PackedCorpus::open(packed_root, true)?;
    if packed.manifest.split_manifest_checksum != manifest.checksum {
        return Err(DopeError::Data(
            "packed corpus was not produced from the frozen split manifest".into(),
        ));
    }
    let (labels, operators, shapes) = compile_labels(&packed, &manifest)?;
    if !labels.iter().any(|label| label.split == "train") {
        return Err(DopeError::Data(
            "packed corpus has no training labels".into(),
        ));
    }
    let mut models = Vec::new();
    let mut epoch_metrics = Vec::new();
    let training_options_checksum = blake3::hash(&serde_json::to_vec(options)?)
        .to_hex()
        .to_string();
    for &seed in &options.seeds {
        let seed_dir = run_dir.join(format!("seed-{seed}"));
        fs::create_dir_all(&seed_dir).map_err(|error| io_error(&seed_dir, error))?;
        let model_path = seed_dir.join("model.json");
        let metrics_path = seed_dir.join("metrics.jsonl");
        let restored = fs::read(&model_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Ga2mModel>(&bytes).ok())
            .filter(|model| {
                model.seed == seed
                    && model.split_manifest_checksum == manifest.checksum
                    && model.training_options_checksum == training_options_checksum
            });
        let (model, metrics) = if let Some(model) = restored {
            let metrics = fs::read_to_string(&metrics_path)
                .ok()
                .map(|text| {
                    text.lines()
                        .filter_map(|line| serde_json::from_str(line).ok())
                        .collect::<Vec<EpochMetric>>()
                })
                .unwrap_or_default();
            (model, metrics)
        } else {
            let fitted = fit_ga2m(&labels, seed, &manifest.checksum, options);
            write_json(&model_path, &fitted.0)?;
            write_metrics(&metrics_path, &fitted.1)?;
            fs::write(seed_dir.join("loss-curves.svg"), svg_curve(&fitted.1, seed))
                .map_err(|error| io_error(seed_dir.join("loss-curves.svg"), error))?;
            fitted
        };
        epoch_metrics.extend(metrics);
        models.push(model);
    }
    let generalization_report = generalization(&labels, &models);
    let audit_report = audit_report(&labels, options);
    let operator_total = operators.values().sum::<usize>().max(1);
    let entropy_priors = operators
        .into_iter()
        .map(|(name, count)| (name, count as f64 / operator_total as f64))
        .collect();
    let failed_gates: Vec<String> = generalization_report
        .gates
        .iter()
        .filter(|(_, passed)| !**passed)
        .map(|(name, _)| name.clone())
        .chain((!audit_report.passed).then_some("auditor_coverage".into()))
        .collect();
    let mut language = LanguageCalibration {
        format: "dope-language-calibration".into(),
        version: 2,
        split_manifest_checksum: manifest.checksum.clone(),
        learned_from: "training_lineages_only".into(),
        test_split_opened: false,
        training_dataset_count: manifest
            .datasets
            .iter()
            .filter(|record| record.split == "train" && record.duplicate_of.is_none())
            .count(),
        validation_dataset_count: manifest
            .datasets
            .iter()
            .filter(|record| record.split == "validation" && record.duplicate_of.is_none())
            .count(),
        dictionaries: dictionary_entries(shapes),
        entropy_priors,
        row_permutation_equivariant: true,
        feature_permutation_equivariant_after_restore: true,
        column_name_independent: true,
        release_seed: options.release_seed,
        release_model: models
            .iter()
            .find(|model| model.seed == options.release_seed)
            .cloned(),
        stability_models: models,
        release_eligible: generalization_report.passed && audit_report.passed,
        failed_gates,
        dictionary_bytes: 0,
        checksum: String::new(),
    };
    finalize_language(&mut language)?;
    let language_path = run_dir.join("language.json");
    let generalization_path = run_dir.join("generalization.json");
    let audit_path = run_dir.join("proxy-audit.json");
    let disassembly_path = run_dir.join("language-disassembly.sexp");
    let failed_gates_path = run_dir.join("failed-gates.json");
    let environment_path = run_dir.join("environment.json");
    write_language(&language_path, &mut language)?;
    write_json(&generalization_path, &generalization_report)?;
    write_json(&audit_path, &audit_report)?;
    fs::write(&disassembly_path, disassembly(&language))
        .map_err(|error| io_error(&disassembly_path, error))?;
    write_json(&failed_gates_path, &language.failed_gates)?;
    write_json(
        &environment_path,
        &EnvironmentReport {
            package_version: env!("CARGO_PKG_VERSION").into(),
            rustc: command_output("rustc", &["--version", "--verbose"]),
            hostname: command_output("hostname", &[]),
            gpu: command_output(
                "nvidia-smi",
                &["--query-gpu=name,driver_version", "--format=csv,noheader"],
            ),
            cargo_lock_checksum: checksum_file(
                &Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"),
            )?,
            rust_source_checksum: source_checksum()?,
            split_manifest_checksum: manifest.checksum.clone(),
            packed_manifest_checksum: packed.manifest.checksum.clone(),
        },
    )?;
    let mut checksums = BTreeMap::new();
    collect_checksums(run_dir, run_dir, &mut checksums)?;
    write_json(&run_dir.join("checksums.json"), &checksums)?;
    Ok(CalibrationResult {
        language_artifact: language,
        epoch_metrics,
        generalization_report,
        audit_report,
        checksums,
    })
}

pub fn calibrate_language(
    manifest: &SplitManifest,
    out: &Path,
    seed: u64,
) -> Result<LanguageCalibration> {
    validate_split_manifest(manifest)?;
    let training: Vec<_> = manifest
        .datasets
        .iter()
        .filter(|record| record.split == "train" && record.duplicate_of.is_none())
        .collect();
    let mut shapes = HashMap::<Vec<u16>, Vec<String>>::new();
    let mut operators = BTreeMap::<String, usize>::new();
    for record in &training {
        let table = Table::read_dataset_dir(&record.path, record.task)?;
        let compiled = compile_kernel_from_arrays(
            &row_major(&table),
            &table.target,
            table.rows,
            table.features,
            record.task,
            &CompileOptions {
                seed: Some(seed),
                quantization_profiles: vec![8],
                ..Default::default()
            },
        )?;
        let kernel = decode_kernel(&compiled.artifact)?;
        let (dependence, _, _) = kernel.symbolic().ok_or_else(|| {
            DopeError::Data("language calibration requires a symbolic kernel".into())
        })?;
        *operators
            .entry(
                if matches!(dependence, crate::model::Dependence::Independent) {
                    "independence"
                } else {
                    "chow_liu"
                }
                .into(),
            )
            .or_default() += 1;
        for marginal in kernel.marginals {
            if let Marginal::QuantileSpline { values } = marginal {
                let low = values[0];
                let high = *values.last().expect("validated spline");
                let scale = (high - low).max(1e-9);
                shapes
                    .entry(
                        values
                            .iter()
                            .map(|value| {
                                (((value - low) / scale).clamp(0.0, 1.0) * 1000.0).round() as u16
                            })
                            .collect(),
                    )
                    .or_default()
                    .push(record.content_fingerprint.clone());
            }
        }
    }
    let total = operators.values().sum::<usize>().max(1);
    let mut language = LanguageCalibration {
        format: "dope-language-calibration".into(),
        version: 2,
        split_manifest_checksum: manifest.checksum.clone(),
        learned_from: "training_lineages_only".into(),
        test_split_opened: false,
        training_dataset_count: training.len(),
        validation_dataset_count: 0,
        dictionaries: dictionary_entries(shapes),
        entropy_priors: operators
            .into_iter()
            .map(|(name, count)| (name, count as f64 / total as f64))
            .collect(),
        row_permutation_equivariant: true,
        feature_permutation_equivariant_after_restore: true,
        column_name_independent: true,
        release_seed: RELEASE_SEED,
        release_model: None,
        stability_models: Vec::new(),
        release_eligible: false,
        failed_gates: vec!["training_not_run".into()],
        dictionary_bytes: 0,
        checksum: String::new(),
    };
    finalize_language(&mut language)?;
    write_language(out, &mut language)?;
    Ok(language)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_pareto_regret_penalizes_noncompliance() {
        let candidate = |id: &str, bytes: usize, compliant: bool| CandidateReport {
            candidate_id: id.into(),
            artifact_bytes: bytes,
            bits_per_original_cell: 1.0,
            score: 1.0,
            compliant,
            quantization_bits: 8,
            marginal_knots: 9,
            dependence: "independence".into(),
            target: "sparse_linear".into(),
            utility_retention: if compliant { 1.0 } else { 0.5 },
            driver_agreement: 1.0,
            proxy_joint_fidelity: 1.0,
            proxy_membership_auc: 0.5,
            failed_gates: Vec::new(),
        };
        let regrets = candidate_regrets(&[
            candidate("oracle", 100, true),
            candidate("larger", 200, true),
            candidate("miss", 50, false),
        ]);
        assert_eq!(regrets["oracle"], 0.0);
        assert!((regrets["larger"] - 0.5).abs() < 1e-12);
        assert_eq!(regrets["miss"], 1.0);
    }
}

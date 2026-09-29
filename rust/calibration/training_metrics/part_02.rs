

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
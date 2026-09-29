

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

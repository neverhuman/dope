fn run_release(command: Command) -> Result<()> {
    match command {
        Command::Convert {
            dataset_dir,
            real_holdout_dir,
            task,
            out,
            seed,
            runtime_dictionary_bytes,
            supported_datasets,
            synthetic_seed,
            tier,
            max_artifact_bytes,
            require_formal_dp,
        } => {
            let release_policy = parse_policy(&tier, max_artifact_bytes, require_formal_dp)?;
            let result = convert_release(
                &dataset_dir,
                &real_holdout_dir,
                parse_task(&task)?,
                &out,
                &ConversionOptions {
                    seed,
                    runtime_dictionary_bytes,
                    supported_datasets,
                    synthetic_csv_seeds: synthetic_seed,
                    compile_options: CompileOptions {
                        beam_width: match release_policy.tier {
                            AnonymizationTier::L3 => Some(4),
                            AnonymizationTier::L2 => Some(8),
                            AnonymizationTier::L1 | AnonymizationTier::L0 => Some(16),
                        },
                        release_policy,
                        ..Default::default()
                    },
                },
            )?;
            println!(
                "{{\"certified\":{},\"artifact_bytes\":{},\"out\":{}}}",
                result.report.certified,
                result.report.byte_accounting.artifact_bytes,
                serde_json::to_string(&out)?
            );
        }
        Command::Compile {
            dataset_dir,
            task,
            out,
            report,
            pareto_frontier,
            seed,
            deadline_seconds,
            language,
            candidate,
            neural_target_weight,
            neural_structural_penalty,
            adaptive_allocation,
            native_target_profile,
            packed_codec,
            tier,
            max_artifact_bytes,
            require_formal_dp,
        } => {
            let native_target_profile = native_target_profile
                .as_deref()
                .map(dope_kernel::compiler::NativeTargetProfile::parse)
                .transpose()?;
            let options = CompileOptions {
                release_policy: parse_policy(&tier, max_artifact_bytes, require_formal_dp)?,
                seed,
                deadline: deadline_seconds.map(Duration::from_secs_f64),
                language: language.as_deref().map(load_language).transpose()?,
                backend_id: candidate,
                neural_target_weight,
                neural_structural_penalty,
                adaptive_allocation,
                native_target_profile,
                packed_codec,
                ..Default::default()
            };
            let result = compile_kernel_from_dir(&dataset_dir, parse_task(&task)?, &options)?;
            fs::write(&out, &result.artifact)
                .map_err(|error| dope_kernel::error::io_error(&out, error))?;
            let report = match report {
                Some(path) => path,
                None => PathBuf::from(format!("{}.report.json", out.display())),
            };
            let frontier = match pareto_frontier {
                Some(path) => path,
                None => PathBuf::from(format!("{}.pareto.json", out.display())),
            };
            write_json(&report, &result.report)?;
            write_json(&frontier, &result.pareto_frontier)?;
            println!(
                "{{\"artifact_bytes\":{},\"compliant\":{},\"out\":{}}}",
                result.artifact.len(),
                result.report.compliant,
                serde_json::to_string(&out)?
            );
        }
        Command::Inspect { kernel, out } => {
            let raw_bytes =
                fs::read(&kernel).map_err(|error| dope_kernel::error::io_error(&kernel, error))?;
            if raw_bytes.starts_with(b"DOPERAW1") || raw_bytes.starts_with(b"DOPERAW2") {
                let bytes = dope_kernel::typed_data::workflow::read_raw_artifact(&kernel)?;
                let inspection = dope_kernel::typed_data::workflow::inspect_raw(&bytes)?;
                if let Some(path) = out {
                    write_json(&path, &inspection)?;
                } else {
                    println!("{}", serde_json::to_string_pretty(&inspection)?);
                }
            } else if raw_bytes.starts_with(b"DOPMIX01") {
                let bytes = dope_kernel::typed_data::mixed::read_mixed_artifact(&kernel)?;
                let inspection = dope_kernel::typed_data::mixed::inspect_mixed(&bytes)?;
                if let Some(path) = out {
                    write_json(&path, &inspection)?;
                } else {
                    println!("{}", serde_json::to_string_pretty(&inspection)?);
                }
            } else {
                let inspection = inspect_kernel(&kernel)?;
                if let Some(path) = out {
                    write_json(&path, &inspection)?;
                } else {
                    println!("{}", serde_json::to_string_pretty(&inspection)?);
                }
            }
        }
        Command::Sample {
            kernel,
            rows,
            out,
            seed,
        } => {
            let raw_bytes =
                fs::read(&kernel).map_err(|error| dope_kernel::error::io_error(&kernel, error))?;
            if raw_bytes.starts_with(b"DOPERAW1") || raw_bytes.starts_with(b"DOPERAW2") {
                let bytes = dope_kernel::typed_data::workflow::read_raw_artifact(&kernel)?;
                let inspection = dope_kernel::typed_data::workflow::inspect_raw(&bytes)?;
                let columns = inspection
                    .schema
                    .iter()
                    .filter(|c| {
                        matches!(
                            c.role,
                            dope_kernel::schema::ColumnRole::Feature
                                | dope_kernel::schema::ColumnRole::Target
                        )
                    })
                    .count();
                let samples =
                    dope_kernel::typed_data::workflow::sample_raw_detailed(&bytes, rows, seed)?;
                let file =
                    fs::File::create(&out).map_err(|e| dope_kernel::error::io_error(&out, e))?;
                let mut writer = csv::WriterBuilder::new()
                    .has_headers(false)
                    .from_writer(file);
                for row in samples.values.chunks_exact(columns) {
                    writer.write_record(row.iter().map(ToString::to_string))?;
                }
                writer
                    .flush()
                    .map_err(|e| dope_kernel::error::io_error(&out, e))?;
                println!(
                    "{{\"out\":{},\"rows\":{rows},\"complete_sampling_bytes\":{}}}",
                    serde_json::to_string(&out)?,
                    inspection.accounting.complete_sampling_bytes
                );
            } else {
                let loaded = load_kernel(&kernel)?;
                sample_kernel_to_csv(&loaded, SampleOptions { rows, seed }, &out)?;
                println!(
                    "{{\"out\":{},\"rows\":{rows}}}",
                    serde_json::to_string(&out)?
                );
            }
        }
        Command::PackCorpus {
            corpus,
            out,
            seed,
            shard_mib,
            archive_zstd,
        } => {
            fs::create_dir_all(&out).map_err(|error| dope_kernel::error::io_error(&out, error))?;
            let (inventory, manifest) = build_split_manifest_multi(&corpus, seed)?;
            write_inventory(&inventory, &out.join("inventory.json"))?;
            write_exclusions(&inventory, &out.join("exclusions.jsonl"))?;
            write_split_manifest(&manifest, &out.join("split-manifest.json"))?;
            let packed = dope_kernel::packed::pack_corpus(
                &manifest,
                &out,
                (shard_mib * 1024.0 * 1024.0) as usize,
                archive_zstd,
            )?;
            println!("{}", serde_json::to_string(&packed)?);
        }
        Command::InventoryCorpus { corpus, out, seed } => {
            fs::create_dir_all(&out).map_err(|error| dope_kernel::error::io_error(&out, error))?;
            let (inventory, manifest) = build_split_manifest_multi(&corpus, seed)?;
            write_inventory(&inventory, &out.join("inventory.json"))?;
            write_exclusions(&inventory, &out.join("exclusions.jsonl"))?;
            write_split_manifest(&manifest, &out.join("split-manifest.json"))?;
            println!(
                "{{\"discovered_paths\":{},\"eligible_paths\":{},\"excluded_paths\":{},\"manifest_checksum\":{}}}",
                inventory.discovered_paths,
                inventory.eligible_paths,
                inventory.exclusions.len(),
                serde_json::to_string(&manifest.checksum)?,
            );
        }
        Command::FinalizeInventory {
            inventory,
            out,
            seed,
        } => {
            fs::create_dir_all(&out).map_err(|error| dope_kernel::error::io_error(&out, error))?;
            let inventory = refresh_inventory_lineages(&load_inventory(&inventory)?)?;
            let manifest = split_inventory(&inventory, seed)?;
            write_inventory(&inventory, &out.join("inventory.json"))?;
            write_exclusions(&inventory, &out.join("exclusions.jsonl"))?;
            write_split_manifest(&manifest, &out.join("split-manifest.json"))?;
            println!(
                "{{\"discovered_paths\":{},\"eligible_paths\":{},\"excluded_paths\":{},\"inventory_checksum\":{},\"manifest_checksum\":{}}}",
                inventory.discovered_paths,
                inventory.eligible_paths,
                inventory.exclusions.len(),
                serde_json::to_string(&inventory.checksum)?,
                serde_json::to_string(&manifest.checksum)?,
            );
        }
        Command::SplitValidation {
            manifest,
            out,
            seed,
        } => {
            let bytes = fs::read(&manifest)
                .map_err(|error| dope_kernel::error::io_error(&manifest, error))?;
            let source = serde_json::from_slice(&bytes)?;
            let submanifest = split_validation_manifest(&source, seed)?;
            write_json(&out, &submanifest)?;
            println!(
                "{{\"assignments\":{},\"validation_cert_sealed\":true,\"out\":{}}}",
                submanifest.assignments.len(),
                serde_json::to_string(&out)?
            );
        }
        Command::CalibrateLanguage { corpus, out, seed } => {
            if seed != 1729 {
                return Err(DopeError::Data(
                    "the release run predeclares seed 1729; stability seeds are fixed internally"
                        .into(),
                ));
            }
            let calibration = train_language(&corpus, &out, &TrainingOptions::default())?;
            println!(
                "{{\"dictionary_bytes\":{},\"release_eligible\":{},\"out\":{}}}",
                calibration.language_artifact.dictionary_bytes,
                calibration.language_artifact.release_eligible,
                serde_json::to_string(&out)?
            );
        }
        Command::Certify {
            real_dir,
            kernel,
            out,
            seed,
            runtime_dictionary_bytes,
            supported_datasets,
            tier,
            max_artifact_bytes,
            require_formal_dp,
        } => {
            let policy = parse_policy(&tier, max_artifact_bytes, require_formal_dp)?;
            let report = certify_kernel_with_policy(
                &real_dir,
                &kernel,
                &out,
                seed,
                runtime_dictionary_bytes,
                supported_datasets,
                &policy,
            )?;
            println!(
                "{{\"certified\":{},\"out\":{}}}",
                report.certified,
                serde_json::to_string(&out)?
            );
        }
        _ => return Err(DopeError::Unsupported("legacy_dispatch_invariant".into())),
    }
    Ok(())
}

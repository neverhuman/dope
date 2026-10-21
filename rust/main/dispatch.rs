fn run(command: Command) -> Result<()> {
    match command {
        command @ (Command::FitTable { .. }
        | Command::SampleTable { .. }
        | Command::InspectTable { .. }) => return run_table(command),
        Command::CompileMixed {
            csv,
            schema,
            out,
            profile,
            seed,
            max_complete_bytes,
            deadline_seconds,
        } => {
            compile_mixed_command(
                csv,
                schema,
                out,
                profile,
                seed,
                max_complete_bytes,
                deadline_seconds,
            )?;
        }
        Command::InspectMixed { artifact } => {
            let bytes = dope_kernel::typed_data::mixed::read_mixed_artifact(&artifact)?;
            println!(
                "{}",
                serde_json::to_string(&dope_kernel::typed_data::mixed::inspect_mixed(&bytes)?)?
            );
        }
        Command::SampleMixed {
            artifact,
            rows,
            out,
            seed,
            isolated,
        } => {
            sample_mixed_command(artifact, rows, out, seed, isolated)?;
        }
        Command::Doctor => {
            let executable = std::env::current_exe()
                .map_err(|_| DopeError::Data("native_identity_unavailable".into()))?;
            let bytes = fs::read(executable)
                .map_err(|_| DopeError::Data("native_identity_unavailable".into()))?;
            println!(
                "{}",
                serde_json::json!({
                    "name":"dope-kernel","implementation":"native_rust","version":DEVELOPMENT_VERSION,
                    "executable_sha256":format!("{:x}", sha2::Sha256::digest(&bytes)),
                    "executable_bytes":bytes.len(),"os":std::env::consts::OS,"architecture":std::env::consts::ARCH,
                    "build_cargo_lock_sha256":format!("{:x}", sha2::Sha256::digest(include_bytes!("../../Cargo.lock"))),
                    "declared_rust_toolchain_sha256":format!("{:x}", sha2::Sha256::digest(include_bytes!("../../rust-toolchain.toml"))),
                    "cpu_sampling":"available","gpu_training":cfg!(feature="gpu-training"),
                "raw_profile":"raw-numeric-symbolic-v1","raw_residual_profile":"raw-numeric-residual-v1","v1_artifact_decode":"read_only_compatible",
                    "source_free_isolation":if cfg!(all(target_os="linux",target_arch="x86_64")) {"linux_x86_64_fresh_worker"} else {"unavailable"},
                    "certification":"requires_measured_external_gates","weights_downloaded":false
                })
            );
        }
        Command::SampleRawIsolated {
            artifact,
            rows,
            seed,
            private_samples_f64le,
        } => {
            let bytes = dope_kernel::typed_data::workflow::read_raw_artifact(&artifact)?;
            let mut inspection = dope_kernel::typed_data::workflow::inspect_raw(&bytes)?;
            let runtime = dope_kernel::typed_data::isolation::runtime_inventory()?;
            let runtime_bytes = runtime
                .iter()
                .try_fold(0usize, |sum, file| sum.checked_add(file.bytes))
                .ok_or_else(|| DopeError::Data("sampling_runtime_size_overflow".into()))?;
            inspection.accounting.generic_runtime_bytes = Some(runtime_bytes);
            inspection.accounting.cold_sampling_closure_bytes = Some(
                runtime_bytes
                    .checked_add(bytes.len())
                    .ok_or_else(|| DopeError::Data("sampling_runtime_size_overflow".into()))?,
            );
            dope_kernel::typed_data::isolation::seal_sampling_process()?;
            let samples =
                dope_kernel::typed_data::workflow::sample_raw_detailed(&bytes, rows, Some(seed))?;
            let columns = inspection.schema.len();
            let output_bytes = rows
                .checked_mul(columns)
                .filter(|count| *count == samples.values.len())
                .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
                .ok_or_else(|| DopeError::Data("sampling_output_shape_invalid".into()))?;
            let encoded = samples
                .values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            if encoded.len() != output_bytes {
                return Err(DopeError::Data("sampling_output_shape_invalid".into()));
            }
            let mut receipt = serde_json::json!({"state":"sampled","isolation":"linux_x86_64_seccomp_tsync_default_deny","inherited_source_fds":"closed","filesystem_network_exec":"denied","rows":rows,"seed":seed,"sample_sha256":dope_kernel::bundle::content_fingerprint(&encoded),"runtime_inventory":runtime,"inspection":inspection,"integer_rounding_repairs":samples.integer_rounding_repairs,"boolean_rounding_repairs":samples.boolean_rounding_repairs,"physical_domain_rejections":samples.physical_domain_rejections});
            if private_samples_f64le {
                use std::io::Write;
                receipt["private_sample_stream"] = serde_json::json!({
                    "format":"f64le", "layout":"row_major",
                    "column_order":"features_then_target", "rows":rows,
                    "columns":columns, "bytes":output_bytes
                });
                let mut public_bytes = serde_json::to_vec(&receipt)?;
                public_bytes.push(b'\n');
                let write_failed = |_| DopeError::Data("sampling_output_write_failed".into());
                // A partial stream has no completion receipt and is never valid.
                // Consumers must also require a successful exit, length and digest.
                let mut private = std::io::stdout().lock();
                private.write_all(&encoded).map_err(write_failed)?;
                private.flush().map_err(write_failed)?;
                let mut public = std::io::stderr().lock();
                public.write_all(&public_bytes).map_err(write_failed)?;
                public.flush().map_err(write_failed)?;
            } else {
                println!("{receipt}");
            }
        }
        Command::InspectDevelopmentPilot {
            plan,
            expected_plan_digest,
            attempts,
        } => {
            let inspection = dope_kernel::corpus::source_pilot::inspect_source_pilot_v2(
                &plan,
                attempts.as_deref(),
                &expected_plan_digest,
            )?;
            println!("{}", serde_json::to_string(&inspection)?);
        }
        Command::CompileCsv {
            csv,
            target,
            out,
            seed,
            candidate,
            max_complete_bytes,
            deadline_seconds,
        } => {
            compile_csv_command(
                csv,
                target,
                out,
                seed,
                candidate,
                max_complete_bytes,
                deadline_seconds,
            )?;
        }
        Command::CompileRaw {
            csv,
            schema,
            out,
            report,
            seed,
            candidate,
            residual_profile,
            max_complete_bytes,
            deadline_seconds,
        } => {
            if !deadline_seconds.is_finite()
                || !(0.0..=600.0).contains(&deadline_seconds)
                || deadline_seconds == 0.0
            {
                return Err(DopeError::Data("raw_deadline_invalid".into()));
            }
            let residual = residual_profile
                .map(|profile| {
                    dope_kernel::model::ResidualProfile::parse(&profile)
                        .ok_or_else(|| DopeError::Data("raw_residual_profile_invalid".into()))
                })
                .transpose()?;
            if (candidate == "development_symbolic") != residual.is_some() {
                return Err(DopeError::Unsupported(
                    "raw_residual_requires_explicit_development_candidate".into(),
                ));
            }
            let schema_bytes =
                fs::read(&schema).map_err(|_| DopeError::Data("raw_schema_read_failed".into()))?;
            let declaration = dope_kernel::schema::TableSchemaV1::from_private_json(&schema_bytes)
                .map_err(|e| DopeError::Data(e.to_string()))?;
            let input = fs::File::open(&csv)
                .map_err(|_| DopeError::Data("raw_input_read_failed".into()))?;
            let table = dope_kernel::typed_data::TypedTable::read_csv(
                input,
                declaration,
                dope_kernel::typed_data::InputLimits::default(),
            )
            .map_err(|e| DopeError::Data(e.to_string()))?;
            let options = CompileOptions {
                seed: Some(seed),
                backend_id: residual.is_none().then_some(candidate),
                beam_width: residual.map(|_| 42),
                deadline: Some(Duration::from_secs_f64(deadline_seconds)),
                ..CompileOptions::default()
            };
            let rows = (0..table.rows()).collect::<Vec<_>>();
            let result = if let Some(profile) = residual {
                dope_kernel::typed_data::workflow::compile_raw_residual(
                    &table,
                    &rows,
                    &options,
                    max_complete_bytes,
                    profile,
                )?
            } else {
                dope_kernel::typed_data::workflow::compile_raw(
                    &table,
                    &rows,
                    &options,
                    max_complete_bytes,
                )?
            };
            fs::write(&out, &result.artifact).map_err(|e| dope_kernel::error::io_error(&out, e))?;
            if let Some(path) = report {
                write_json(
                    &path,
                    &serde_json::json!({"inspection":result.inspection, "development_compiler":result.report}),
                )?;
            }
            println!("{}", serde_json::to_string(&result.inspection)?);
        }
        Command::SampleRaw {
            artifact,
            rows,
            out,
            seed,
        } => {
            let bytes = dope_kernel::typed_data::workflow::read_raw_artifact(&artifact)?;
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
            let file = fs::File::create(&out).map_err(|e| dope_kernel::error::io_error(&out, e))?;
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
                "{}",
                serde_json::json!({"rows":rows,"profile":inspection.profile,"complete_sampling_bytes":inspection.accounting.complete_sampling_bytes,"privacy":"unmeasured","integer_rounding_repairs":samples.integer_rounding_repairs,"boolean_rounding_repairs":samples.boolean_rounding_repairs,"physical_domain_rejections":samples.physical_domain_rejections})
            );
        }
        Command::InspectRaw { artifact } => {
            let bytes = dope_kernel::typed_data::workflow::read_raw_artifact(&artifact)?;
            println!(
                "{}",
                serde_json::to_string(&dope_kernel::typed_data::workflow::inspect_raw(&bytes)?)?
            );
        }
        command => return run_metadata(command),
    }
    Ok(())
}

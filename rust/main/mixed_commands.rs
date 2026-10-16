fn compile_mixed_command(
    csv: PathBuf,
    schema: PathBuf,
    out: PathBuf,
    profile: String,
    seed: u64,
    max_complete_bytes: usize,
    deadline_seconds: f64,
) -> Result<()> {
    use std::io::Read;
    if !deadline_seconds.is_finite()
        || !(0.0..=600.0).contains(&deadline_seconds)
        || deadline_seconds == 0.0
    {
        return Err(DopeError::Data("mixed_deadline_invalid".into()));
    }
    let profile = dope_kernel::typed_data::mixed::MixedProfile::parse(&profile)
        .ok_or_else(|| DopeError::Data("mixed_profile_invalid".into()))?;
    let mut schema_bytes = Vec::new();
    fs::File::open(&schema)
        .map_err(|_| DopeError::Data("mixed_schema_read_failed".into()))?
        .take(dope_kernel::schema::MAX_SCHEMA_BYTES as u64 + 1)
        .read_to_end(&mut schema_bytes)
        .map_err(|_| DopeError::Data("mixed_schema_read_failed".into()))?;
    let schema = dope_kernel::schema::TableSchemaV1::from_private_json(&schema_bytes)
        .map_err(|e| DopeError::Data(e.to_string()))?;
    let input =
        fs::File::open(csv).map_err(|_| DopeError::Data("mixed_input_read_failed".into()))?;
    let table = dope_kernel::typed_data::TypedTable::read_csv(
        input,
        schema,
        dope_kernel::typed_data::InputLimits::default(),
    )
    .map_err(|e| DopeError::Data(e.to_string()))?;
    let options = CompileOptions {
        seed: Some(seed),
        backend_id: Some("chow_liu".into()),
        deadline: Some(Duration::from_secs_f64(deadline_seconds)),
        ..CompileOptions::default()
    };
    let result = dope_kernel::typed_data::mixed::compile_mixed(
        &table,
        &(0..table.rows()).collect::<Vec<_>>(),
        &options,
        max_complete_bytes,
        profile,
    )?;
    fs::write(out, &result.artifact)
        .map_err(|_| DopeError::Data("mixed_artifact_write_failed".into()))?;
    println!(
        "{}",
        serde_json::json!({
            "inspection":result.inspection,"development_compiler":result.report,
            "preparation_seconds":result.preparation_seconds,"complete_elapsed_seconds":result.complete_elapsed_seconds,
        })
    );
    Ok(())
}

fn sample_mixed_command(
    artifact: PathBuf,
    rows: usize,
    out: Option<PathBuf>,
    seed: u64,
    isolated: bool,
) -> Result<()> {
    if isolated == out.is_some() {
        return Err(DopeError::Data(
            "mixed_sample_requires_output_or_isolated_digest".into(),
        ));
    }
    let bytes = dope_kernel::typed_data::mixed::read_mixed_artifact(&artifact)?;
    let mut inspection = dope_kernel::typed_data::mixed::inspect_mixed(&bytes)?;
    let runtime = if isolated {
        let runtime = dope_kernel::typed_data::isolation::runtime_inventory()?;
        let total = runtime
            .iter()
            .try_fold(0usize, |n, f| n.checked_add(f.bytes))
            .ok_or_else(|| DopeError::Data("sampling_runtime_size_overflow".into()))?;
        inspection.accounting.generic_runtime_bytes = Some(total);
        inspection.accounting.cold_sampling_closure_bytes = Some(
            total
                .checked_add(bytes.len())
                .ok_or_else(|| DopeError::Data("sampling_runtime_size_overflow".into()))?,
        );
        dope_kernel::typed_data::isolation::seal_sampling_process()?;
        Some(runtime)
    } else {
        None
    };
    let samples = dope_kernel::typed_data::mixed::sample_mixed(&bytes, rows, Some(seed))?;
    let encoded = serde_json::to_vec(&samples.values)?;
    if let Some(path) = out {
        fs::write(path, serde_json::to_vec(&samples)?)
            .map_err(|_| DopeError::Data("mixed_samples_write_failed".into()))?;
    }
    println!(
        "{}",
        serde_json::json!({
            "state":"sampled","rows":rows,"seed":seed,
            "sample_sha256":dope_kernel::bundle::content_fingerprint(&encoded),
            "inspection":inspection,"runtime_inventory":runtime,
            "isolation":if isolated {"linux_x86_64_seccomp_tsync_default_deny"} else {"ordinary_native_process"},
            "integer_rounding_repairs":samples.integer_rounding_repairs,
            "boolean_rounding_repairs":samples.boolean_rounding_repairs,
            "masks_emitted":samples.masks_emitted,"physical_domain_rejections":samples.physical_domain_rejections,
        })
    );
    Ok(())
}

fn compile_csv_command(
    csv_path: PathBuf,
    target: Option<String>,
    out: PathBuf,
    seed: u64,
    candidate: String,
    max_complete_bytes: usize,
    deadline_seconds: f64,
) -> Result<()> {
    if !deadline_seconds.is_finite()
        || !(0.0..=600.0).contains(&deadline_seconds)
        || deadline_seconds == 0.0
    {
        return Err(DopeError::Data("csv_deadline_invalid".into()));
    }
    let file = fs::File::open(&csv_path).map_err(|_| DopeError::Data("csv_input_read_failed".into()))?;
    let mut reader = csv::ReaderBuilder::new().has_headers(true).from_reader(file);
    let headers = reader
        .headers()
        .map_err(|_| DopeError::Data("csv_header_read_failed".into()))?
        .iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if headers.len() < 2 || headers.len() > dope_kernel::schema::MAX_COLUMNS {
        return Err(DopeError::Data("csv_column_count_invalid".into()));
    }
    let target_idx = match target.as_deref() {
        None => headers.len() - 1,
        Some(spec) => headers
            .iter()
            .position(|h| h == spec)
            .or_else(|| spec.parse::<usize>().ok().filter(|&idx| idx < headers.len()))
            .ok_or_else(|| DopeError::Data("csv_target_column_not_found".into()))?,
    };
    let mut records = Vec::new();
    for record in reader.records() {
        let row = record.map_err(|_| DopeError::Data("csv_row_read_failed".into()))?;
        if row.len() != headers.len() {
            return Err(DopeError::Data("csv_row_width_mismatch".into()));
        }
        records.push(row.iter().map(str::to_owned).collect::<Vec<_>>());
    }
    if records.len() < 2 {
        return Err(DopeError::Data("csv_insufficient_rows".into()));
    }

    let max_exact_int: i64 = (1i64 << 53) - 1;
    let mut columns = Vec::with_capacity(headers.len());
    let mut has_nominal_or_masked = false;

    for (col_idx, header) in headers.iter().enumerate() {
        let id = format!("c{:04}", col_idx + 1);
        let is_target = col_idx == target_idx;
        let role = if is_target { "target" } else { "feature" };
        let cells = records.iter().map(|r| r[col_idx].as_str()).collect::<Vec<_>>();
        let has_empty = cells.iter().any(|c| c.is_empty());
        let all_numeric = !has_empty
            && cells
                .iter()
                .all(|c| c.parse::<f64>().is_ok_and(f64::is_finite));
        if is_target {
            if !all_numeric {
                return Err(DopeError::Data("csv_target_requires_finite_numeric".into()));
            }
            columns.push(serde_json::json!({
                "id": id,
                "source_name": header,
                "role": role,
                "dtype": "continuous"
            }));
        } else if all_numeric {
            let all_exact_int = cells.iter().all(|c| {
                !c.contains('.')
                    && !c.contains('e')
                    && !c.contains('E')
                    && c.parse::<i64>()
                        .is_ok_and(|v| (-max_exact_int..=max_exact_int).contains(&v))
            });
            let dtype = if all_exact_int { "integer" } else { "continuous" };
            columns.push(serde_json::json!({
                "id": id,
                "source_name": header,
                "role": role,
                "dtype": dtype
            }));
        } else {
            has_nominal_or_masked = true;
            let mut values = std::collections::BTreeSet::new();
            for cell in &cells {
                if !cell.is_empty() {
                    values.insert((*cell).to_owned());
                }
            }
            if values.is_empty() {
                return Err(DopeError::Data("csv_nominal_column_empty".into()));
            }
            let mask_tokens = if has_empty {
                vec![serde_json::json!({"token": "", "state": "missing"})]
            } else {
                Vec::new()
            };
            columns.push(serde_json::json!({
                "id": id,
                "source_name": header,
                "role": role,
                "dtype": "nominal",
                "domain": {
                    "kind": "categorical",
                    "values": values.into_iter().collect::<Vec<_>>()
                },
                "mask_tokens": mask_tokens
            }));
        }
    }

    let schema_json = serde_json::to_vec(&serde_json::json!({
        "format": "dope-table-schema",
        "version": 1,
        "metadata_visibility": "opaque_ids",
        "columns": columns
    }))?;
    let declaration = dope_kernel::schema::TableSchemaV1::from_private_json(&schema_json)
        .map_err(|e| DopeError::Data(e.to_string()))?;
    let input = fs::File::open(&csv_path).map_err(|_| DopeError::Data("csv_input_read_failed".into()))?;
    let table = dope_kernel::typed_data::TypedTable::read_csv(
        input,
        declaration,
        dope_kernel::typed_data::InputLimits::default(),
    )
    .map_err(|e| DopeError::Data(e.to_string()))?;
    let fit_rows = (0..table.rows()).collect::<Vec<_>>();
    let options = CompileOptions {
        seed: Some(seed),
        backend_id: Some(candidate),
        deadline: Some(Duration::from_secs_f64(deadline_seconds)),
        ..CompileOptions::default()
    };

    if has_nominal_or_masked {
        let result = dope_kernel::typed_data::mixed::compile_mixed(
            &table,
            &fit_rows,
            &options,
            max_complete_bytes,
            dope_kernel::typed_data::mixed::MixedProfile::NumericSplitV1,
        )?;
        fs::write(&out, &result.artifact).map_err(|e| dope_kernel::error::io_error(&out, e))?;
        println!("{}", serde_json::to_string(&result.inspection)?);
    } else {
        let result = dope_kernel::typed_data::workflow::compile_raw(
            &table,
            &fit_rows,
            &options,
            max_complete_bytes,
        )?;
        fs::write(&out, &result.artifact).map_err(|e| dope_kernel::error::io_error(&out, e))?;
        println!("{}", serde_json::to_string(&result.inspection)?);
    }
    Ok(())
}

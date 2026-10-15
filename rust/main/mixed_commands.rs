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

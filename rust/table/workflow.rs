/// Largest table the workflow reads (cells); larger inputs are refused.
pub const MAX_TABLE_CELLS: usize = 20_000_000;

/// Read a headered CSV into its header and string rows.
pub fn read_csv_table(path: &std::path::Path) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_path(path)
        .map_err(|error| DopeError::Data(format!("table_csv_unreadable: {error}")))?;
    let header = reader
        .headers()
        .map_err(|error| DopeError::Data(format!("table_csv_header_unreadable: {error}")))?
        .iter()
        .map(|name| name.trim().to_string())
        .collect::<Vec<_>>();
    if header.len() < 2 || header.iter().collect::<BTreeSet<_>>().len() != header.len() {
        return Err(DopeError::Data("table_header_needs_two_unique_names".into()));
    }
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|error| DopeError::Data(format!("table_csv_row_unreadable: {error}")))?;
        rows.push(record.iter().map(str::to_string).collect::<Vec<_>>());
        if rows.len().saturating_mul(header.len()) > MAX_TABLE_CELLS {
            return Err(DopeError::Data("table_exceeds_cell_limit".into()));
        }
    }
    Ok((header, rows))
}

/// Settings for compiling a planned table.
#[derive(Clone, Debug)]
pub struct FitTableOptions {
    pub table: TableOptions,
    pub seed: u64,
    /// Symbolic candidate; the default is the published headline recipe.
    pub candidate: String,
    pub native_target_profile: Option<crate::compiler::NativeTargetProfile>,
    pub deadline: Option<std::time::Duration>,
}

impl FitTableOptions {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            table: TableOptions::new(target),
            seed: 1729,
            candidate: "compact_neural_residual_symbolic".into(),
            native_target_profile: Some(crate::compiler::NativeTargetProfile::HEADLINE),
            deadline: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitTableReport {
    pub task: Task,
    pub rows_fitted: usize,
    pub rows_without_target: usize,
    pub columns: Vec<ColumnSummary>,
    pub dropped: Vec<DroppedColumn>,
    pub bytes: TableAccounting,
    pub candidate: String,
    pub native_target_profile: Option<String>,
}

/// A planned column, described without its name or values.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColumnSummary {
    pub position: usize,
    pub kind: String,
    pub levels: Option<usize>,
    pub is_target: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DroppedColumn {
    pub name: String,
    pub reason: String,
}

fn summaries(plan: &TablePlan) -> Vec<ColumnSummary> {
    plan.columns
        .iter()
        .enumerate()
        .map(|(position, column)| ColumnSummary {
            position,
            kind: column.encoding.kind().into(),
            levels: match &column.encoding {
                Encoding::Category { levels } => Some(levels.len()),
                _ => None,
            },
            is_target: position == plan.target,
        })
        .collect()
}

/// Plan, encode and compile a headered table into one table artifact.
pub fn fit_table(
    header: &[String],
    rows: &[Vec<String>],
    options: &FitTableOptions,
) -> Result<(Vec<u8>, FitTableReport)> {
    let planned = plan_table(header, rows, &options.table)?;
    let compile_options = crate::compiler::CompileOptions {
        seed: Some(options.seed),
        deadline: options.deadline,
        backend_id: Some(options.candidate.clone()),
        native_target_profile: options.native_target_profile,
        packed_codec: true,
        ..Default::default()
    };
    let compiled = crate::compiler::compile_kernel_from_arrays(
        &planned.features,
        &planned.target,
        planned.rows,
        planned.plan.feature_count(),
        planned.plan.task,
        &compile_options,
    )?;
    let (artifact, bytes) = write_table_artifact(&planned.plan, &compiled.artifact);
    let report = FitTableReport {
        task: planned.plan.task,
        rows_fitted: planned.rows,
        rows_without_target: planned.rows_without_target,
        columns: summaries(&planned.plan),
        dropped: planned
            .dropped
            .iter()
            .map(|(name, reason)| DroppedColumn { name: name.clone(), reason: (*reason).into() })
            .collect(),
        bytes,
        candidate: options.candidate.clone(),
        native_target_profile: options.native_target_profile.map(|profile| profile.name()),
    };
    Ok((artifact, report))
}

/// Sample `rows` synthetic rows in the source header, categories and units.
pub fn sample_table(artifact: &[u8], rows: usize, seed: u64) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let (plan, kernel, _) = read_table_artifact(artifact)?;
    let loaded = crate::codec::LoadedKernel::V3(Box::new(kernel));
    let samples = crate::sample::sample_kernel(&loaded, crate::sample::SampleOptions { rows, seed: Some(seed) })?;
    let header = plan.header().into_iter().map(str::to_string).collect();
    Ok((header, decode_rows(&plan, &samples)))
}

/// Byte accounting and column kinds; no names, levels or extrema.
pub fn inspect_table(artifact: &[u8]) -> Result<serde_json::Value> {
    let (plan, kernel, bytes) = read_table_artifact(artifact)?;
    let Some((dependence, target, noise)) = kernel.symbolic() else {
        return Err(DopeError::Codec("table_kernel_not_symbolic".into()));
    };
    let (dependence, target, noise) = (component_kind(dependence)?, component_kind(target)?, component_kind(noise)?);
    Ok(serde_json::json!({
        "format": "dope-table-v1",
        "task": plan.task,
        "bytes": bytes,
        "packed_values": crate::codec::is_packed(&artifact[artifact.len() - bytes.kernel_bytes..]),
        "rows_fitted": kernel.rows_fitted,
        "columns": summaries(&plan),
        "kernel": {
            "dependence": dependence,
            "target": target,
            "noise": noise,
            "quantization_bits": kernel.quantization_bits,
        },
    }))
}

/// The serde tag of a kernel component (dependence, target or noise model).
fn component_kind<T: Serialize>(component: &T) -> Result<serde_json::Value> {
    match serde_json::to_value(component)? {
        serde_json::Value::Object(mut fields) => fields
            .remove("kind")
            .ok_or_else(|| DopeError::Codec("table_kernel_component_untagged".into())),
        _ => Err(DopeError::Codec("table_kernel_component_untagged".into())),
    }
}

/// Write rows as a headered CSV.
pub fn write_csv_table(path: &std::path::Path, header: &[String], rows: &[Vec<String>]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|error| DopeError::Data(format!("table_csv_unwritable: {error}")))?;
    writer.write_record(header)?;
    for row in rows {
        writer.write_record(row)?;
    }
    writer.flush().map_err(|error| crate::error::io_error(path, error))
}

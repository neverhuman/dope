use super::exit_codes::{Code, Failure, Result};
use super::io;
use super::registry::{Compile, Input, Limits};
use dope_kernel::compiler::CompileOptions;
use dope_kernel::contract::{AnonymizationTier, ReleasePolicy};
use dope_kernel::schema::TableSchemaV1;
use dope_kernel::typed_data::{InputLimits, TypedTable};
use serde_json::{Value, json};
use std::path::Path;
use std::time::{Duration, Instant};

pub fn table(input: &Input, limits: &Limits) -> Result<(TypedTable, Value)> {
    let schema_bytes = io::read(&input.schema, limits.max_input_bytes.min(2_097_152))?;
    let csv = io::read(&input.csv, limits.max_input_bytes)?;
    let mut schema = TableSchemaV1::from_private_json_with_field_limit(
        &schema_bytes,
        limits.max_field_bytes as usize,
    )
    .map_err(|_| Failure::usage("schema_invalid"))?;
    let mut header_reader = csv::Reader::from_reader(csv.as_slice());
    let headers = header_reader
        .headers()
        .map_err(|_| Failure::usage("invalid_csv_header"))?;
    if headers
        .iter()
        .any(|header| header.len() as u64 > limits.max_field_bytes)
    {
        return Err(Failure::budget());
    }
    // User-approved labels and units remain private at this CLI boundary.
    schema.metadata_visibility = dope_kernel::schema::MetadataVisibility::OpaqueIds;
    let table = TypedTable::read_csv(
        csv.as_slice(),
        schema,
        InputLimits {
            max_bytes: limits.max_input_bytes as usize,
            max_rows: limits.max_rows as usize,
            max_cells: limits.max_cells as usize,
            max_field_bytes: limits.max_field_bytes as usize,
        },
    )
    .map_err(|error| match error.code {
        "input_bytes_exceeded" | "input_dimensions_exceeded" | "field_bytes_exceeded" => {
            Failure::budget()
        }
        _ => Failure::usage("input_schema_validation_failed"),
    })?;
    let receipt = json!({
        "format":"dope-input-receipt", "version":1, "state":"input_validated",
        "rows":table.rows(), "columns":table.schema().columns.len(),
        "input_sha256":dope_kernel::bundle::content_fingerprint(&csv),
        "schema_sha256":dope_kernel::bundle::content_fingerprint(&schema_bytes),
        "fit_transform_state":"not_fitted", "prepared_view":"unavailable",
        "qualified":false, "offline":true,
    });
    Ok((table, receipt))
}

pub fn numeric_csv(bytes: &[u8], limits: &Limits) -> Result<(Vec<f32>, Vec<f32>, usize)> {
    let mut csv = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(bytes);
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut width = None;
    for row in csv.records() {
        let row = row.map_err(|_| Failure::usage("numeric_csv_invalid"))?;
        let n = row.len();
        if !(2..=dope_kernel::schema::MAX_COLUMNS).contains(&n) || width.is_some_and(|w| w != n) {
            return Err(Failure::usage("numeric_csv_shape_invalid"));
        }
        width = Some(n);
        if y.len() as u64 >= limits.max_rows || (y.len() as u64 + 1) * n as u64 > limits.max_cells {
            return Err(Failure::budget());
        }
        for (column, value) in row.iter().enumerate() {
            if value.len() as u64 > limits.max_field_bytes {
                return Err(Failure::budget());
            }
            let value: f32 = value
                .parse()
                .map_err(|_| Failure::usage("numeric_csv_invalid"))?;
            if !value.is_finite() {
                return Err(Failure::usage("numeric_csv_invalid"));
            }
            if column + 1 == n {
                y.push(value);
            } else {
                x.push(value);
            }
        }
    }
    let features = width.ok_or_else(|| Failure::usage("numeric_csv_empty"))? - 1;
    Ok((x, y, features))
}

pub fn compile(args: Compile, limits: &Limits) -> Result<()> {
    let mut outputs = vec![args.out.as_path()];
    outputs.extend(args.report.as_deref());
    io::outputs(&outputs)?;
    if args.require_formal_dp {
        return Err(Failure::new(
            Code::Privacy,
            "formal_dp_mechanism_unavailable",
        ));
    }
    let started = Instant::now();
    let tier = args
        .tier
        .parse::<AnonymizationTier>()
        .map_err(|_| Failure::usage("tier_invalid"))?;
    let payload_cap = tier
        .byte_limit()
        .map_or(args.max_complete_bytes as usize, |ceiling| {
            ceiling.min(args.max_complete_bytes as usize)
        });
    let options = CompileOptions {
        seed: args.seed,
        deadline: Some(Duration::from_secs(args.deadline_seconds)),
        backend_id: Some(args.candidate),
        release_policy: ReleasePolicy::new(tier, Some(payload_cap), false)?,
        ..CompileOptions::default()
    };
    let artifact = match (args.csv, args.schema, args.dataset_dir) {
        (Some(csv), Some(schema), None) => {
            if args.task != "regression" {
                return Err(Failure::unsupported());
            }
            let (table, _) = table(&Input { csv, schema }, limits)?;
            dope_kernel::typed_data::workflow::compile_raw(
                &table,
                &(0..table.rows()).collect::<Vec<_>>(),
                &options,
                args.max_complete_bytes as usize,
            )?
            .artifact
        }
        (None, None, Some(directory)) => {
            io::local(&directory)?;
            eprintln!(
                "migration: --dataset-dir uses the legacy numeric compiler; prefer --csv with --schema"
            );
            let bytes = io::read(&directory.join("train.csv"), limits.max_input_bytes)?;
            let (x, y, features) = numeric_csv(&bytes, limits)?;
            let task = dope_kernel::model::Task::parse(&args.task)
                .ok_or_else(|| Failure::usage("task_invalid"))?;
            dope_kernel::compiler::compile_kernel_from_arrays(
                &x,
                &y,
                y.len(),
                features,
                task,
                &options,
            )?
            .artifact
        }
        _ => return Err(Failure::usage("select_csv_and_schema_or_dataset_dir")),
    };
    if started.elapsed() > Duration::from_secs(args.deadline_seconds)
        || artifact.len() as u64 > args.max_complete_bytes.min(limits.max_output_bytes)
    {
        return Err(Failure::budget());
    }
    let receipt = json!({"state":"compiled","artifact_bytes":artifact.len(),
        "artifact_sha256":dope_kernel::bundle::content_fingerprint(&artifact),
        "qualified":false,"utility_qualification_state":"unassessed",
        "empirical_privacy_state":"unassessed","formal_dp_state":"unverified",
        "legal_release_state":"pending","offline":true});
    // Check both serialized sizes before creating either output.
    if args.report.is_some()
        && serde_json::to_vec_pretty(&receipt).unwrap().len() as u64 > limits.max_output_bytes
    {
        return Err(Failure::budget());
    }
    io::write(&args.out, &artifact, limits)?;
    if let Some(report) = args.report {
        io::emit(&receipt, Some(&report), limits)?;
    }
    io::emit(&receipt, None, limits)
}

pub fn inspect(path: &Path, limits: &Limits) -> Result<Value> {
    let bytes = io::read(path, limits.max_input_bytes)?;
    inspect_bytes(&bytes, limits)
}

fn inspect_bytes(bytes: &[u8], limits: &Limits) -> Result<Value> {
    super::artifacts::budget(bytes, limits)?;
    let value = if bytes.starts_with(b"DOPERAW1") || bytes.starts_with(b"DOPERAW2") {
        serde_json::to_value(dope_kernel::typed_data::workflow::inspect_raw(bytes)?).unwrap()
    } else if bytes.starts_with(b"DOPMIX01") {
        serde_json::to_value(dope_kernel::typed_data::mixed::inspect_mixed(bytes)?).unwrap()
    } else {
        let (_file, captured) = io::snapshot(bytes)?;
        serde_json::to_value(dope_kernel::inspect_kernel(&captured)?).unwrap()
    };
    // Raw artifact metadata can contain approved labels/units from older producers.
    let mut value = value;
    if value["version"] == 1 && value["format"] == "dope-kernel" {
        for field in [
            "task",
            "marginals",
            "dependence",
            "target",
            "noise",
            "sexpr",
        ] {
            value.as_object_mut().unwrap().remove(field);
        }
    }
    if let Some(columns) = value["schema"].as_array_mut() {
        for column in columns {
            if let Some(object) = column.as_object_mut() {
                object.remove("label");
                object.remove("units");
            }
        }
    }
    Ok(value)
}

pub fn sample(
    path: &Path,
    rows: u64,
    seed: Option<u64>,
    out: &Path,
    limits: &Limits,
) -> Result<()> {
    io::outputs(&[out])?;
    let bytes = io::read(path, limits.max_input_bytes)?;
    let inspection = inspect_bytes(&bytes, limits)?;
    let columns = if let Some(features) = inspection["positional_features"].as_u64() {
        features
            .checked_add(1)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(Failure::budget)?
    } else {
        inspection["schema"].as_array().map_or(0, |columns| {
            columns
                .iter()
                .filter(|c| matches!(c["role"].as_str(), Some("feature" | "target")))
                .count()
        })
    };
    let cells = rows
        .checked_mul(columns as u64)
        .ok_or_else(Failure::budget)?;
    if columns == 0
        || rows > limits.max_rows
        || cells > limits.max_cells
        || cells > limits.max_output_bytes / 32
    {
        return Err(Failure::budget());
    }
    let values: Vec<f64> = if bytes.starts_with(b"DOPERAW1") || bytes.starts_with(b"DOPERAW2") {
        dope_kernel::typed_data::workflow::sample_raw(&bytes, rows as usize, seed)?
    } else if bytes.starts_with(b"DOPMIX01") {
        return Err(Failure::unsupported()); // Mixed cells require a distinct output contract.
    } else {
        let (_file, captured) = io::snapshot(&bytes)?;
        super::artifacts::sampling(&bytes, limits)?;
        let loaded = dope_kernel::codec::load_kernel(&captured)?;
        dope_kernel::sample::sample_kernel(
            &loaded,
            dope_kernel::sample::SampleOptions {
                rows: rows as usize,
                seed,
            },
        )?
        .into_iter()
        .map(f64::from)
        .collect()
    };
    if values.len() as u64 != cells {
        return Err(Failure::usage("sample_shape_invalid"));
    }
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(Vec::new());
    for row in values.chunks_exact(columns) {
        writer
            .write_record(row.iter().map(ToString::to_string))
            .map_err(|_| Failure::new(Code::Internal, "sample_encode_failed"))?;
    }
    let bytes = writer
        .into_inner()
        .map_err(|_| Failure::new(Code::Internal, "sample_encode_failed"))?;
    io::write(out, &bytes, limits)?;
    io::emit(
        &json!({"state":"sampled","rows":rows,"privacy":"unassessed","isolation":"ordinary_native_process","offline":true}),
        None,
        limits,
    )
}

//! Versioned original-unit workflow. This is development fitting and sampling,
//! never a certification or protected-query capability.

use serde::Serialize;

use super::{FittedTransforms, TypedTable, fit_transform};
use crate::compiler::{CompileOptions, CompileReport, compile_kernel_from_arrays};
use crate::error::{DopeError, Result};
use crate::model::{KernelProgram, Noise, ResidualProfile, Task};
use crate::sample::{SampleOptions, sample_symbolic_support};
use crate::schema::{ColumnRole, ColumnType, PublicColumn};

const MAGIC: &[u8; 8] = b"DOPERAW1";
const RESIDUAL_MAGIC: &[u8; 8] = b"DOPERAW2";
const HEADER: usize = 56;
const MAX_BYTES: usize = 32 * 1024 * 1024;

/// Bound the carrier before allocation and reject indirect/special-file inputs.
pub fn read_raw_artifact(path: &std::path::Path) -> Result<Vec<u8>> {
    use std::io::Read;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|_| rejected("raw_artifact_read_failed"))?;
    let metadata = file
        .metadata()
        .map_err(|_| rejected("raw_artifact_read_failed"))?;
    if !metadata.is_file() || metadata.nlink() != 1 || metadata.len() > MAX_BYTES as u64 {
        return Err(rejected("raw_artifact_carrier_invalid"));
    }
    let mut bytes = Vec::new();
    file.take(MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| rejected("raw_artifact_read_failed"))?;
    if bytes.len() > MAX_BYTES {
        return Err(rejected("raw_artifact_carrier_invalid"));
    }
    unpack(&bytes)?;
    Ok(bytes)
}

#[derive(Debug, Serialize)]
pub struct RawAccounting {
    pub byte_profile_id: &'static str,
    pub framing_bytes: usize,
    pub transform_schema_bytes: usize,
    pub native_payload_bytes: usize,
    pub complete_sampling_bytes: usize,
    pub transport_archive_bytes: usize,
    pub generic_runtime_bytes: Option<usize>,
    pub cold_sampling_closure_bytes: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct RawInspection {
    pub format: &'static str,
    pub version: u32,
    pub profile: &'static str,
    pub capability: &'static str,
    pub feature_support: &'static str,
    pub target_support: &'static str,
    pub constant_target_rule: &'static str,
    pub privacy: &'static str,
    pub certification: &'static str,
    pub schema: Vec<PublicColumn>,
    pub accounting: RawAccounting,
    pub native_sections: crate::codec::ArtifactAccounting,
    pub decoded_native_float32_values: usize,
    pub decoded_native_integer_values: usize,
    pub retained_numeric_fields: usize,
    pub mathematical_parameter_count: Option<usize>,
}

pub struct RawCompileResult {
    pub artifact: Vec<u8>,
    pub report: CompileReport,
    pub inspection: RawInspection,
}

fn rejected(code: &'static str) -> DopeError {
    DopeError::Data(code.into())
}

fn schema_error(error: crate::schema::SchemaError) -> DopeError {
    DopeError::Data(error.to_string())
}

/// Only explicitly supported, fully observed numeric columns enter the first
/// native raw profile. Nominal dictionaries and conditional masks stay typed
/// unsupported rather than being silently coerced into independent numeric data.
fn ordered_columns(state: &FittedTransforms) -> Result<Vec<PublicColumn>> {
    let mut columns = state.public_projection().map_err(schema_error)?;
    if columns
        .iter()
        .any(|c| !matches!(c.role, ColumnRole::Feature | ColumnRole::Target))
    {
        return Err(rejected("raw_sampler_unconsumed_state"));
    }
    if columns.iter().any(|c| !c.dtype.numeric()) {
        return Err(DopeError::Unsupported(
            "raw_profile_requires_numeric_columns".into(),
        ));
    }
    columns.sort_by_key(|c| c.role == ColumnRole::Target);
    if columns
        .last()
        .is_none_or(|c| c.dtype != ColumnType::Continuous)
    {
        return Err(DopeError::Unsupported(
            "raw_regression_target_requires_continuous_type".into(),
        ));
    }
    Ok(columns)
}

pub fn compile_raw(
    table: &TypedTable,
    fit_rows: &[usize],
    options: &CompileOptions,
    max_complete_bytes: usize,
) -> Result<RawCompileResult> {
    if options.development_residual.is_some() {
        return Err(DopeError::Unsupported(
            "raw_v1_refuses_development_residual".into(),
        ));
    }
    compile_raw_profile(table, fit_rows, options, max_complete_bytes, MAGIC)
}

/// Explicit fully observed numeric successor. Residual parameters enter the
/// native payload before every encoded cap check; the v1 entry stays unchanged.
pub fn compile_raw_residual(
    table: &TypedTable,
    fit_rows: &[usize],
    options: &CompileOptions,
    max_complete_bytes: usize,
    profile: ResidualProfile,
) -> Result<RawCompileResult> {
    let mut options = options.clone();
    options.development_residual = Some(profile);
    compile_raw_profile(
        table,
        fit_rows,
        &options,
        max_complete_bytes,
        RESIDUAL_MAGIC,
    )
}

fn compile_raw_profile(
    table: &TypedTable,
    fit_rows: &[usize],
    options: &CompileOptions,
    max_complete_bytes: usize,
    magic: &[u8; 8],
) -> Result<RawCompileResult> {
    if max_complete_bytes == 0 || max_complete_bytes > MAX_BYTES {
        return Err(rejected("raw_artifact_cap_invalid"));
    }
    let mut state = fit_transform(table, fit_rows).map_err(schema_error)?;
    // Split/group/entity fields never enter either the fit tensor or the
    // successor sampling closure. Preserve the historical transform API.
    state
        .columns
        .retain(|column| matches!(column.role, ColumnRole::Feature | ColumnRole::Target));
    let columns = ordered_columns(&state)?;
    let features = columns.len() - 1;
    let mut x = Vec::with_capacity(fit_rows.len() * features);
    let mut y = Vec::with_capacity(fit_rows.len());
    for &row in fit_rows {
        for column in &columns {
            let value = table
                .cell(row, &column.id)
                .map_err(schema_error)?
                .number()
                .ok_or_else(|| {
                    DopeError::Unsupported("raw_profile_requires_observed_cells".into())
                })?;
            let value = state
                .transform_numeric(&column.id, &[value])
                .map_err(schema_error)?[0];
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(rejected("raw_fit_normalization_invalid"));
            }
            if column.role == ColumnRole::Target {
                y.push(value as f32);
            } else {
                x.push(value as f32);
            }
        }
    }
    let retained = state.sampling_state_bytes().map_err(schema_error)?;
    let payload_cap = max_complete_bytes
        .checked_sub(HEADER + retained.len())
        .filter(|cap| *cap > 0)
        .ok_or_else(|| rejected("raw_complete_byte_cap_exceeded"))?;
    let mut bounded_options = options.clone();
    bounded_options.release_policy.maximum_artifact_bytes = Some(
        options
            .release_policy
            .maximum_artifact_bytes
            .map_or(payload_cap, |cap| cap.min(payload_cap)),
    );
    bounded_options.release_policy.validate()?;
    let compiled = compile_kernel_from_arrays(
        &x,
        &y,
        fit_rows.len(),
        features,
        Task::Regression,
        &bounded_options,
    )?;
    let kernel = crate::codec::decode_kernel(&compiled.artifact)?;
    if !matches!(kernel.program, KernelProgram::Symbolic { .. }) {
        return Err(DopeError::Unsupported(
            "raw_profile_requires_symbolic_backend".into(),
        ));
    }
    let state = retained;
    let total = HEADER
        .checked_add(state.len())
        .and_then(|n| n.checked_add(compiled.artifact.len()))
        .ok_or_else(|| rejected("raw_artifact_size_overflow"))?;
    if total > max_complete_bytes {
        return Err(rejected("raw_complete_byte_cap_exceeded"));
    }
    let mut artifact = Vec::with_capacity(total);
    artifact.extend_from_slice(magic);
    artifact.extend_from_slice(&(state.len() as u64).to_le_bytes());
    artifact.extend_from_slice(&(compiled.artifact.len() as u64).to_le_bytes());
    let mut content = state;
    content.extend_from_slice(&compiled.artifact);
    artifact.extend_from_slice(blake3::hash(&content).as_bytes());
    artifact.extend_from_slice(&content);
    let inspection = inspect_raw(&artifact)?;
    Ok(RawCompileResult {
        artifact,
        report: compiled.report,
        inspection,
    })
}

fn unpack(artifact: &[u8]) -> Result<(FittedTransforms, &[u8])> {
    if artifact.len() < HEADER
        || artifact.len() > MAX_BYTES
        || (&artifact[..8] != MAGIC && &artifact[..8] != RESIDUAL_MAGIC)
    {
        return Err(rejected("raw_artifact_invalid"));
    }
    let read_length = |offset| -> Result<usize> {
        let bytes = artifact[offset..offset + 8]
            .try_into()
            .expect("bounded header");
        usize::try_from(u64::from_le_bytes(bytes))
            .map_err(|_| rejected("raw_artifact_size_overflow"))
    };
    let state_bytes = read_length(8)?;
    let payload_bytes = read_length(16)?;
    if HEADER
        .checked_add(state_bytes)
        .and_then(|n| n.checked_add(payload_bytes))
        != Some(artifact.len())
        || blake3::hash(&artifact[HEADER..]).as_bytes() != &artifact[24..56]
    {
        return Err(rejected("raw_artifact_integrity_failed"));
    }
    let state =
        FittedTransforms::reload(&artifact[HEADER..HEADER + state_bytes]).map_err(schema_error)?;
    let payload = &artifact[HEADER + state_bytes..];
    crate::bundle::preflight_dpk(payload, &crate::bundle::BundleLimits::default())
        .map_err(|_| rejected("raw_payload_limits_invalid"))?;
    let kernel = crate::codec::decode_kernel(payload)?;
    let residual = matches!(
        kernel.program,
        KernelProgram::Symbolic {
            noise: Noise::ResidualV1 { .. },
            ..
        }
    );
    if residual != (&artifact[..8] == RESIDUAL_MAGIC) {
        return Err(rejected("raw_residual_profile_mismatch"));
    }
    let columns = ordered_columns(&state)?;
    if kernel.features as usize + 1 != columns.len()
        || kernel.task != Task::Regression
        || !matches!(kernel.program, KernelProgram::Symbolic { .. })
    {
        return Err(rejected("raw_sampler_closure_mismatch"));
    }
    Ok((state, payload))
}

pub fn inspect_raw(artifact: &[u8]) -> Result<RawInspection> {
    let (state, payload) = unpack(artifact)?;
    let kernel = crate::codec::decode_kernel(payload)?;
    let native = serde_json::to_value(&kernel)?;
    let (native_floats, native_integers) = numeric_census(&native);
    let retained = serde_json::to_value(&state)?;
    let (state_floats, state_integers) = numeric_census(&retained);
    let residual = &artifact[..8] == RESIDUAL_MAGIC;
    Ok(RawInspection {
        format: "dope-raw-sampling-artifact",
        version: if residual { 2 } else { 1 },
        profile: if residual {
            "raw-numeric-residual-v1"
        } else {
            "raw-numeric-symbolic-v1"
        },
        capability: "fully_observed_numeric_features_continuous_regression",
        feature_support: "fit_range_bounded_historical_native_marginals",
        target_support: "unclipped_affine_regression_not_tail_qualified",
        constant_target_rule: "unit_scale_with_fitted_offset",
        privacy: "unmeasured_private_transforms_retained",
        certification: "unqualified_development_artifact",
        schema: state.public_projection().map_err(schema_error)?,
        accounting: RawAccounting {
            byte_profile_id: if residual {
                "native-raw-complete-residual-v1"
            } else {
                "native-raw-complete-v1"
            },
            framing_bytes: HEADER,
            transform_schema_bytes: artifact.len() - HEADER - payload.len(),
            native_payload_bytes: payload.len(),
            complete_sampling_bytes: artifact.len(),
            transport_archive_bytes: artifact.len(),
            generic_runtime_bytes: None,
            cold_sampling_closure_bytes: None,
        },
        native_sections: crate::codec::account_artifact(payload)?,
        decoded_native_float32_values: native_floats,
        decoded_native_integer_values: native_integers,
        retained_numeric_fields: state_floats + state_integers,
        mathematical_parameter_count: None,
    })
}

fn numeric_census(value: &serde_json::Value) -> (usize, usize) {
    match value {
        serde_json::Value::Number(n) => {
            if n.is_f64() {
                (1, 0)
            } else {
                (0, 1)
            }
        }
        serde_json::Value::Array(values) => values
            .iter()
            .map(numeric_census)
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1)),
        serde_json::Value::Object(values) => values
            .values()
            .map(numeric_census)
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1)),
        _ => (0, 0),
    }
}

/// Reloaded bytes alone determine samples and their original units. No table,
/// source-name mapping, or query values are retained or consulted here.
pub fn sample_raw(artifact: &[u8], rows: usize, seed: Option<u64>) -> Result<Vec<f64>> {
    Ok(sample_raw_detailed(artifact, rows, seed)?.values)
}

#[derive(Serialize)]
pub struct RawSamples {
    pub values: Vec<f64>,
    pub integer_rounding_repairs: usize,
    pub boolean_rounding_repairs: usize,
    pub physical_domain_rejections: usize,
}

pub fn sample_raw_detailed(artifact: &[u8], rows: usize, seed: Option<u64>) -> Result<RawSamples> {
    let (state, payload) = unpack(artifact)?;
    let columns = ordered_columns(&state)?;
    if rows == 0
        || rows > 100_000
        || rows
            .checked_mul(columns.len())
            .is_none_or(|n| n > 1_000_000)
    {
        return Err(rejected("raw_sample_cap_exceeded"));
    }
    let kernel = crate::codec::decode_kernel(payload)?;
    let normalized = sample_symbolic_support(&kernel, SampleOptions { rows, seed }, false)?;
    let mut values = Vec::with_capacity(normalized.len());
    let mut integer_repairs = 0;
    let mut boolean_repairs = 0;
    for row in normalized.chunks_exact(columns.len()) {
        for (column, value) in columns.iter().zip(row) {
            let mut original = state
                .inverse_transform(&column.id, &[f64::from(*value)])
                .map_err(schema_error)?[0];
            if matches!(column.dtype, ColumnType::Integer | ColumnType::Ordinal) {
                let rounded = original.round();
                integer_repairs += usize::from(rounded != original);
                original = rounded;
            } else if column.dtype == ColumnType::Boolean {
                let rounded = u8::from(original >= 0.5) as f64;
                boolean_repairs += usize::from(rounded != original);
                original = rounded;
            }
            // Physical declarations are separate from observed extrema. Never
            // silently clamp or retry a draw that violates the declared domain.
            let retained = state.column(&column.id).map_err(schema_error)?;
            super::validate_numeric_cell(original, column.dtype, &retained.domain, |_| {
                crate::schema::SchemaError::new("raw_sample_physical_domain_violation", None, None)
            })
            .map_err(schema_error)?;
            values.push(original);
        }
    }
    Ok(RawSamples {
        values,
        integer_rounding_repairs: integer_repairs,
        boolean_rounding_repairs: boolean_repairs,
        physical_domain_rejections: 0,
    })
}

/// Point prediction in original target units; query values never refit the
/// transforms and may extend beyond observed fit extrema. This is a predictor
/// control, not proof of learner transfer or calibrated conditional uncertainty.
pub fn predict_raw(artifact: &[u8], features: &[f64]) -> Result<Vec<f64>> {
    let (state, payload) = unpack(artifact)?;
    let columns = ordered_columns(&state)?;
    let width = columns.len() - 1;
    if features.is_empty() || features.len() % width != 0 || features.len() > 1_000_000 {
        return Err(rejected("raw_query_dimensions_invalid"));
    }
    let kernel = crate::codec::decode_kernel(payload)?;
    let KernelProgram::Symbolic { target, .. } = &kernel.program else {
        return Err(DopeError::Unsupported(
            "raw_profile_requires_symbolic_backend".into(),
        ));
    };
    let mut predictions = Vec::with_capacity(features.len() / width);
    for row in features.chunks_exact(width) {
        let normalized = columns[..width]
            .iter()
            .zip(row)
            .map(|(column, value)| {
                let value = state
                    .transform_numeric(&column.id, &[*value])
                    .map_err(schema_error)?[0] as f32;
                if !value.is_finite() {
                    return Err(rejected("raw_query_precision_overflow"));
                }
                Ok(value)
            })
            .collect::<Result<Vec<_>>>()?;
        let (prediction, _) = crate::sample::evaluate_target(target, &normalized);
        predictions.push(
            state
                .inverse_transform(&columns[width].id, &[f64::from(prediction)])
                .map_err(schema_error)?[0],
        );
    }
    Ok(predictions)
}

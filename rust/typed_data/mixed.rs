//! Explicit bounded mixed-feature development carrier. Private dictionaries,
//! completions and conditional probabilities are part of the sampling closure.
//! This does not certify conditional uncertainty, disclosure or transfer gains.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::{Cell, FittedTransforms, TypedTable, fit_transform};
use crate::compiler::{CompileOptions, CompileReport, compile_kernel_from_arrays};
use crate::error::{DopeError, Result};
use crate::model::{KernelProgram, Task};
use crate::schema::{ColumnRole, ColumnType, MaskState, PublicColumn};

const MAGIC: &[u8; 8] = b"DOPMIX01";
const HEADER: usize = 56;
const MAX_BYTES: usize = 32 * 1024 * 1024;
const MAX_LATENT_FEATURES: usize = 128;
const MAX_CATEGORIES: usize = 32;
const MASKS: [MaskState; 5] = [
    MaskState::Missing,
    MaskState::UnknownCategory,
    MaskState::StructuralAbsence,
    MaskState::Padding,
    MaskState::Censored,
];

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MixedProfile {
    IndependentV1,
    NumericSplitV1,
}

impl MixedProfile {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "independent_v1" => Some(Self::IndependentV1),
            "numeric_split_v1" => Some(Self::NumericSplitV1),
            _ => None,
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Split {
    anchor: usize,
    threshold: f32,
    left: Vec<f64>,
    right: Vec<f64>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Feature {
    id: String,
    start: usize,
    // Empty for a numeric column. Nominal values are exact, column scoped and
    // one-hot encoded; hashes or ordinal category numbers never identify them.
    categories: Vec<String>,
    completion: f32,
    probabilities: Vec<f64>,
    split: Option<Split>,
}

impl Feature {
    fn observed_classes(&self) -> usize {
        self.categories.len().max(1)
    }

    fn width(&self) -> usize {
        self.categories.len().max(1)
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct State {
    format: String,
    version: u32,
    profile: MixedProfile,
    constant_target: bool,
    affine: FittedTransforms,
    features: Vec<Feature>,
}

#[derive(Serialize)]
pub struct MixedInspection {
    pub format: &'static str,
    pub version: u32,
    pub profile: MixedProfile,
    pub capability: &'static str,
    pub support: &'static str,
    pub privacy: &'static str,
    pub certification: &'static str,
    pub schema: Vec<PublicColumn>,
    pub latent_features: usize,
    pub conditional_columns: usize,
    pub accounting: super::workflow::RawAccounting,
    pub native_sections: crate::codec::ArtifactAccounting,
    pub mathematical_parameter_count: Option<usize>,
}

pub struct MixedCompileResult {
    pub artifact: Vec<u8>,
    pub report: CompileReport,
    pub inspection: MixedInspection,
    pub preparation_seconds: f64,
    pub complete_elapsed_seconds: f64,
}

fn rejected(code: &'static str) -> DopeError {
    DopeError::Data(code.into())
}

fn unavailable(code: &'static str) -> DopeError {
    DopeError::Unsupported(code.into())
}

fn schema_error(error: crate::schema::SchemaError) -> DopeError {
    DopeError::Data(error.to_string())
}

fn deadline(started: Instant, limit: Option<Duration>) -> Result<()> {
    if limit.is_some_and(|limit| started.elapsed() >= limit) {
        return Err(unavailable("mixed_preparation_deadline_exhausted"));
    }
    Ok(())
}

/// A separate entry: old raw profiles never silently acquire masks/categories.
pub fn compile_mixed(
    table: &TypedTable,
    fit_rows: &[usize],
    options: &CompileOptions,
    max_complete_bytes: usize,
    profile: MixedProfile,
) -> Result<MixedCompileResult> {
    let started = Instant::now();
    let limit = options.deadline.unwrap_or(Duration::from_secs(600));
    if limit.is_zero() || limit > Duration::from_secs(600) {
        return Err(unavailable("mixed_preparation_deadline_exhausted"));
    }
    if max_complete_bytes == 0 || max_complete_bytes > MAX_BYTES {
        return Err(rejected("mixed_artifact_cap_invalid"));
    }
    deadline(started, Some(limit))?;
    let (state, x, y) = fit_state(table, fit_rows, profile, started, Some(limit))?;
    let mut retained = super::BoundedStateBytes(Vec::new());
    serde_json::to_writer(&mut retained, &state)
        .map_err(|_| rejected("mixed_state_bytes_exceeded"))?;
    let payload_cap = max_complete_bytes
        .checked_sub(HEADER + retained.0.len())
        .filter(|cap| *cap > 0)
        .ok_or_else(|| rejected("mixed_complete_byte_cap_exceeded"))?;
    deadline(started, Some(limit))?;
    let preparation_seconds = started.elapsed().as_secs_f64();
    let mut bounded = options.clone();
    bounded.deadline = Some(limit.saturating_sub(started.elapsed()));
    bounded.release_policy.maximum_artifact_bytes = Some(
        options
            .release_policy
            .maximum_artifact_bytes
            .map_or(payload_cap, |cap| cap.min(payload_cap)),
    );
    bounded.release_policy.validate()?;
    let compiled = compile_kernel_from_arrays(
        &x,
        &y,
        fit_rows.len(),
        state.width(),
        Task::Regression,
        &bounded,
    )?;
    deadline(started, Some(limit))?;
    let artifact = pack(&retained.0, &compiled.artifact);
    if artifact.len() > max_complete_bytes {
        return Err(rejected("mixed_complete_byte_cap_exceeded"));
    }
    let inspection = inspect_mixed(&artifact)?;
    Ok(MixedCompileResult {
        artifact,
        report: compiled.report,
        inspection,
        preparation_seconds,
        complete_elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

include!("mixed/fit.rs");
include!("mixed/codec.rs");
include!("mixed/sample.rs");

#[cfg(test)]
mod tests;

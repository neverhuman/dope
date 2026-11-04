//! NC1 FIT-only normalization. Real matrices and ScaleCards remain private.
//! Public schemas expose shape and kinds only. No neural or device dependency.
#![forbid(unsafe_code)]

mod apply;
mod card;
mod fit;
mod public_schema;

pub use card::{ApplyReceipt, ScaleCard, UnitTable};
pub use public_schema::{GroupKind, GroupSpec, TargetKind, UnitSchema};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Frozen successor recipe; released v3 preparation is a different contract.
pub const RECIPE: &str = "nc1-minmax-onehot-v1";
/// Parser and allocation safety bound, not a scientifically chosen hyperparameter.
pub const MAX_JSON_BYTES: usize = 16 * 1024 * 1024;
/// Maximum supported expanded feature width.
pub const MAX_UNIT_COLUMNS: usize = 256;
/// Maximum accepted source columns.
pub const MAX_SOURCE_COLUMNS: usize = 4096;
/// Maximum rows accepted in a single in-memory view.
pub const MAX_ROWS: usize = 1_000_000;
/// Maximum source cells accepted in a single in-memory view.
pub const MAX_CELLS: usize = 16_000_000;

/// Safe error codes. Never contains a source name, label, value, or parser detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormError {
    /// Stable error identifier.
    pub code: &'static str,
    /// Optional zero-based source/group position.
    pub column: Option<usize>,
}
impl NormError {
    pub(crate) fn new(code: &'static str) -> Self {
        Self { code, column: None }
    }
    pub(crate) fn at(code: &'static str, column: usize) -> Self {
        Self {
            code,
            column: Some(column),
        }
    }
}
impl fmt::Display for NormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.column {
            Some(i) => write!(f, "{}:column={i}", self.code),
            None => f.write_str(self.code),
        }
    }
}
impl std::error::Error for NormError {}
/// NC1 operation result.
pub type Result<T> = std::result::Result<T, NormError>;

/// Explicit typed cell. Missingness is never inferred from category spelling.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum Cell {
    Number(f64),
    Category(String),
    Boolean(bool),
    Missing,
}
impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Cell(<private>)")
    }
}
/// Private source column.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputColumn {
    pub name: String,
    pub values: Vec<Cell>,
}
impl fmt::Debug for InputColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("InputColumn(<private>)")
    }
}
/// Private owner input with stable physical row IDs and caller-verified source identity.
/// The source identity is immutable provenance, not recomputed from query values.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputTable {
    pub source_sha256: String,
    pub row_ids: Vec<u64>,
    pub columns: Vec<InputColumn>,
}
impl fmt::Debug for InputTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("InputTable(<private>)")
    }
}
impl InputTable {
    /// Construct and validate rectangular, bounded source data.
    pub fn new(
        source_sha256: String,
        row_ids: Vec<u64>,
        columns: Vec<InputColumn>,
    ) -> Result<Self> {
        let out = Self {
            source_sha256,
            row_ids,
            columns,
        };
        out.validate()?;
        Ok(out)
    }
    /// Validate shape, source identity, header uniqueness, and physical-row uniqueness.
    pub fn validate(&self) -> Result<()> {
        if !is_digest(&self.source_sha256) {
            return Err(NormError::new("invalid_source_identity"));
        }
        let n = self.row_ids.len();
        if n == 0
            || n > MAX_ROWS
            || self.columns.is_empty()
            || self.columns.len() > MAX_SOURCE_COLUMNS
            || n.checked_mul(self.columns.len())
                .is_none_or(|v| v > MAX_CELLS)
        {
            return Err(NormError::new("input_shape_limit"));
        }
        let ids: std::collections::BTreeSet<_> = self.row_ids.iter().collect();
        if ids.len() != n {
            return Err(NormError::new("duplicate_row_id"));
        }
        let mut names = std::collections::BTreeSet::new();
        for (i, col) in self.columns.iter().enumerate() {
            if col.name.is_empty() || !names.insert(&col.name) || col.values.len() != n {
                return Err(NormError::at("invalid_source_column", i));
            }
        }
        Ok(())
    }
}
/// Declared feature semantics. No kind is inferred from a target or cardinality.
#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeatureKind {
    Continuous,
    Integer,
    Boolean,
    Ordinal { levels: Vec<String> },
    Nominal,
}
/// Explicit source feature role.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureSpec {
    pub column: usize,
    pub kind: FeatureKind,
}
/// Explicit target task and stable binary class semantics.
#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "task", rename_all = "snake_case", deny_unknown_fields)]
pub enum Task {
    Regression,
    Binary { positive_label: String },
    Multiclass { labels: Vec<String> },
}
/// Explicit target column role.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSpec {
    pub column: usize,
    pub task: Task,
}
/// Private declared source roles. Unlisted columns are excluded.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeclaredSchema {
    pub features: Vec<FeatureSpec>,
    pub target: TargetSpec,
}
/// Strict vocabulary safety limits. Rare-level pooling is not implemented.
/// min_level_count must be 1; max_levels is a refusal cap, never a truncation.
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NormOptions {
    pub min_level_count: usize,
    pub max_levels: usize,
}
impl Default for NormOptions {
    fn default() -> Self {
        Self {
            min_level_count: 1,
            max_levels: 255,
        }
    }
}
pub(crate) fn is_digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub(crate) fn digest<T: Serialize>(value: &T) -> Result<String> {
    use sha2::{Digest, Sha256};
    let bytes = serde_json::to_vec(value).map_err(|_| NormError::new("serialization_failed"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
pub(crate) fn parse_json<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    if bytes.len() > MAX_JSON_BYTES {
        return Err(NormError::new("json_size_limit"));
    }
    serde_json::from_slice(bytes).map_err(|_| NormError::new("invalid_json"))
}

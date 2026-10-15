//! Declared raw-table schema and metadata projections.
//! Source names, domains and units are private input metadata. The public view
//! never publishes source names, domains or fitted numeric parameters.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

pub const SCHEMA_FORMAT: &str = "dope-table-schema";
pub const MAX_COLUMNS: usize = 4096;
pub const MAX_SCHEMA_BYTES: usize = 2 * 1024 * 1024;
pub const DEFAULT_MAX_FIELD_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchemaError {
    pub code: &'static str,
    pub row: Option<usize>,
    pub column: Option<usize>,
}

impl SchemaError {
    pub fn new(code: &'static str, row: Option<usize>, column: Option<usize>) -> Self {
        Self { code, row, column }
    }
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(row) = self.row {
            write!(f, "row {row}, ")?;
        }
        if let Some(column) = self.column {
            write!(f, "column {column}: ")?;
        }
        f.write_str(self.code)
    }
}

impl std::error::Error for SchemaError {}

pub type Result<T> = std::result::Result<T, SchemaError>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ColumnRole {
    Feature,
    Target,
    EntityKey,
    Timestamp,
    GroupSplitKey,
    ExcludedSensitiveIdentifier,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ColumnType {
    Continuous,
    Integer,
    Ordinal,
    Nominal,
    Boolean,
    DateTime,
    Text,
}

impl ColumnType {
    pub fn numeric(self) -> bool {
        matches!(
            self,
            Self::Continuous | Self::Integer | Self::Ordinal | Self::Boolean
        )
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MetadataVisibility {
    #[default]
    OpaqueIds,
    ApprovedSemanticLabels,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum MaskState {
    Missing,
    UnknownCategory,
    StructuralAbsence,
    Padding,
    Censored,
}

#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ColumnDomain {
    Numeric {
        lower: Option<f64>,
        upper: Option<f64>,
    },
    Categorical {
        values: Vec<String>,
    },
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MaskToken {
    pub token: String,
    pub state: MaskState,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnSchemaV1 {
    pub id: String,
    pub source_name: String,
    pub role: ColumnRole,
    pub dtype: ColumnType,
    #[serde(default)]
    pub units: Option<String>,
    #[serde(default)]
    pub semantic_label: Option<String>,
    #[serde(default)]
    pub domain: Option<ColumnDomain>,
    #[serde(default)]
    pub mask_tokens: Vec<MaskToken>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableSchemaV1 {
    pub format: String,
    pub version: u32,
    pub columns: Vec<ColumnSchemaV1>,
    #[serde(default)]
    pub metadata_visibility: MetadataVisibility,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct PublicColumn {
    pub id: String,
    pub role: ColumnRole,
    pub dtype: ColumnType,
    pub mask_states: Vec<MaskState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub units: Option<String>,
}

pub(crate) fn opaque_id(id: &str) -> bool {
    id.len() == 5 && id.starts_with('c') && id.as_bytes()[1..].iter().all(u8::is_ascii_digit)
}

pub(crate) fn validate_domain(
    dtype: ColumnType,
    domain: &Option<ColumnDomain>,
    max_field_bytes: usize,
) -> bool {
    match domain {
        None => true,
        Some(ColumnDomain::Numeric { lower, upper }) => {
            dtype.numeric()
                && lower.iter().chain(upper).all(|x| x.is_finite())
                && !matches!((lower, upper), (Some(a), Some(b)) if a > b)
        }
        Some(ColumnDomain::Categorical { values }) => {
            dtype == ColumnType::Nominal
                && !values.is_empty()
                && values.len() <= 4096
                && values
                    .iter()
                    .all(|x| !x.is_empty() && x.len() <= max_field_bytes)
                && values.iter().collect::<BTreeSet<_>>().len() == values.len()
        }
    }
}

impl TableSchemaV1 {
    pub fn from_private_json(bytes: &[u8]) -> Result<Self> {
        Self::from_private_json_with_field_limit(bytes, DEFAULT_MAX_FIELD_BYTES)
    }

    /// Parse private metadata using the same byte limit as the table's input limits.
    pub fn from_private_json_with_field_limit(
        bytes: &[u8],
        max_field_bytes: usize,
    ) -> Result<Self> {
        if bytes.len() > MAX_SCHEMA_BYTES {
            return Err(SchemaError::new("schema_bytes_exceeded", None, None));
        }
        let schema: Self = serde_json::from_slice(bytes)
            .map_err(|_| SchemaError::new("invalid_schema_json", None, None))?;
        schema.validate_with_field_limit(max_field_bytes)?;
        Ok(schema)
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_with_field_limit(DEFAULT_MAX_FIELD_BYTES)
    }

    /// Validate categorical tokens in bytes, including declared values absent from the CSV.
    pub fn validate_with_field_limit(&self, max_field_bytes: usize) -> Result<()> {
        if max_field_bytes == 0 {
            return Err(SchemaError::new("invalid_input_limits", None, None));
        }
        if self.format != SCHEMA_FORMAT || self.version != 1 {
            return Err(SchemaError::new("unknown_schema_version", None, None));
        }
        if self.columns.len() < 2 || self.columns.len() > MAX_COLUMNS {
            return Err(SchemaError::new("invalid_column_count", None, None));
        }
        let mut ids = BTreeSet::new();
        let mut names = BTreeSet::new();
        let mut metadata_bytes = 256usize;
        for (index, c) in self.columns.iter().enumerate() {
            let error = |code| SchemaError::new(code, None, Some(index + 1));
            if !opaque_id(&c.id) || !ids.insert(&c.id) {
                return Err(error("ambiguous_column_id"));
            }
            if c.source_name.trim().is_empty()
                || c.source_name.len() > 256
                || !names.insert(&c.source_name)
            {
                return Err(error("ambiguous_source_name"));
            }
            if c.units
                .iter()
                .chain(&c.semantic_label)
                .any(|x| x.trim().is_empty() || x.len() > 128)
            {
                return Err(error("invalid_metadata_label"));
            }
            if matches!(c.dtype, ColumnType::DateTime | ColumnType::Text) {
                return Err(error("unsupported_logical_type"));
            }
            if !validate_domain(c.dtype, &c.domain, max_field_bytes) {
                return Err(error("invalid_declared_domain"));
            }
            let mut tokens = BTreeSet::new();
            if c.mask_tokens.len() > 32
                || c.mask_tokens
                    .iter()
                    .any(|x| x.token.len() > 128 || !tokens.insert(&x.token))
            {
                return Err(error("ambiguous_mask_tokens"));
            }
            let strings = std::iter::once(c.source_name.as_str())
                .chain(c.units.iter().chain(&c.semantic_label).map(String::as_str))
                .chain(c.mask_tokens.iter().map(|x| x.token.as_str()));
            for value in strings {
                metadata_bytes = metadata_bytes.saturating_add(value.len() + 32);
            }
            if let Some(ColumnDomain::Categorical { values }) = &c.domain {
                for value in values {
                    metadata_bytes = metadata_bytes.saturating_add(value.len() + 32);
                }
            }
            metadata_bytes = metadata_bytes.saturating_add(256);
            if metadata_bytes > MAX_SCHEMA_BYTES {
                return Err(error("schema_bytes_exceeded"));
            }
        }
        if self
            .columns
            .iter()
            .filter(|c| c.role == ColumnRole::Target)
            .count()
            != 1
            || !self.columns.iter().any(|c| c.role == ColumnRole::Feature)
        {
            return Err(SchemaError::new(
                "explicit_feature_and_target_roles_required",
                None,
                None,
            ));
        }
        Ok(())
    }

    pub fn public_projection(&self) -> Result<Vec<PublicColumn>> {
        self.public_projection_with_field_limit(DEFAULT_MAX_FIELD_BYTES)
    }

    /// Project custom-limit metadata without publishing source names or domains.
    pub fn public_projection_with_field_limit(
        &self,
        max_field_bytes: usize,
    ) -> Result<Vec<PublicColumn>> {
        self.validate_with_field_limit(max_field_bytes)?;
        let approved = self.metadata_visibility == MetadataVisibility::ApprovedSemanticLabels;
        Ok(self
            .columns
            .iter()
            .map(|c| PublicColumn {
                id: c.id.clone(),
                role: c.role,
                dtype: c.dtype,
                mask_states: c.mask_states(),
                label: approved.then(|| c.semantic_label.clone()).flatten(),
                units: approved.then(|| c.units.clone()).flatten(),
            })
            .collect())
    }
}

impl ColumnSchemaV1 {
    pub(crate) fn mask_states(&self) -> Vec<MaskState> {
        let mut states: BTreeSet<_> = self.mask_tokens.iter().map(|x| x.state).collect();
        if matches!(&self.domain, Some(ColumnDomain::Categorical { .. })) {
            states.insert(MaskState::UnknownCategory);
        }
        states.into_iter().collect()
    }
}

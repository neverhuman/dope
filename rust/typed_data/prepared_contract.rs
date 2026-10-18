//! Private PreparedView metadata. Native validation supplements the JSON schema.
use serde::{Deserialize, Serialize};

use super::{MAX_COLUMNS, Result, SchemaError};

pub const PREPARED_VIEW_FORMAT: &str = "prepared-view-v1";

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Regression,
    Binary,
    Multiclass,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Horizon {
    pub steps: u32,
    pub unit: HorizonUnit,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HorizonUnit {
    Rows,
    Seconds,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PreparedType {
    Continuous,
    Integer,
    Boolean,
    CategoryCode,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PreparedRole {
    Feature,
    Target,
}

#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PreparedTransform {
    Affine { offset: f64, scale: f64 },
    Identity {},
    CategoryIndex { categories: Vec<String> },
}

#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PreparedBounds {
    #[serde(deserialize_with = "required_nullable")]
    pub lower: Option<f64>,
    #[serde(deserialize_with = "required_nullable")]
    pub upper: Option<f64>,
}

fn required_nullable<'de, D, T>(deserializer: D) -> std::result::Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::deserialize(deserializer)
}

#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PreparedColumn {
    pub id: u32,
    pub source_id: String,
    pub dtype: PreparedType,
    pub role: PreparedRole,
    pub transform: PreparedTransform,
    pub bounds: PreparedBounds,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreparedGroup {
    pub id: u32,
    pub column_ids: Vec<u32>,
}

#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PreparedTarget {
    Regression { units: String },
    Discrete { class_order: Vec<u32> },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreparedPolicies {
    pub missing: String,
    pub category: String,
    pub oov: String,
    pub support: String,
    pub saturation: String,
}

impl Default for PreparedPolicies {
    fn default() -> Self {
        Self {
            missing: "reject".into(),
            category: "fit_dictionary".into(),
            oov: "reject".into(),
            support: "declared_bounds".into(),
            saturation: "none".into(),
        }
    }
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreparedProvenance {
    pub fit_digest: String,
    pub entity_digest: String,
    pub split_digest: String,
    pub column_digest: String,
    pub implementation_hash: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RightsState {
    ResearchUnassessed,
    Approved,
    Blocked,
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreparedRights {
    pub state: RightsState,
    #[serde(deserialize_with = "required_nullable")]
    pub evidence_digest: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreparedProjection {
    pub state: String,
    pub public: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskCapability {
    pub schema: bool,
    pub adapter: bool,
    pub qualified: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreparedCapabilities {
    pub regression: TaskCapability,
    pub binary: TaskCapability,
    pub multiclass: TaskCapability,
}

impl Default for PreparedCapabilities {
    fn default() -> Self {
        let initial = TaskCapability {
            schema: true,
            adapter: true,
            qualified: true,
        };
        Self {
            regression: initial,
            binary: initial,
            multiclass: TaskCapability {
                qualified: false,
                ..initial
            },
        }
    }
}

#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PreparedMetadata {
    pub task: TaskKind,
    pub horizon: Horizon,
    pub columns: Vec<PreparedColumn>,
    pub group_map: Vec<PreparedGroup>,
    pub target: PreparedTarget,
    pub policies: PreparedPolicies,
    pub provenance: PreparedProvenance,
    pub rights: PreparedRights,
    pub projection: PreparedProjection,
    pub capabilities: PreparedCapabilities,
}

include!("prepared_contract_validation.rs");

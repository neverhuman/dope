//! Explicit fit-only preparation and frozen prepared-input admission.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{Cell, InputLimits, TypedTable, fit_transform};
use crate::schema::*;

#[derive(Clone, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedView {
    format: String,
    version: u32,
    metadata: PreparedMetadata,
    rows: Vec<Vec<f64>>,
}

/// Migration spelling only; its wire format is still `prepared-view-v1`.
#[deprecated(note = "use PreparedView; PreparedTable is a migration alias only")]
pub type PreparedTable = PreparedView;

impl std::fmt::Debug for PreparedView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedView")
            .field("rows", &self.rows.len())
            .finish_non_exhaustive()
    }
}

/// A trusted frozen map, retained from explicit preparation or an admitted manifest.
#[derive(Clone)]
pub struct PreparedContract {
    metadata: PreparedMetadata,
}

impl std::fmt::Debug for PreparedContract {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedContract").finish_non_exhaustive()
    }
}

/// Borrowed only after frozen-contract validation and task-scope admission.
#[derive(Debug)]
pub struct PreparedCompileInput<'a>(&'a PreparedView);

impl<'a> PreparedCompileInput<'a> {
    pub fn view(&self) -> &'a PreparedView {
        self.0
    }
}

#[derive(Clone)]
pub struct PrepareSpec {
    pub task: TaskKind,
    pub horizon: Horizon,
    pub group_map: Vec<PreparedGroup>,
    pub target: PreparedTarget,
    pub entity_digest: String,
    pub split_digest: String,
    pub rights: PreparedRights,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedPublicProjection {
    pub format: &'static str,
    pub version: u32,
    pub state: &'static str,
    pub task: TaskKind,
    pub columns: Vec<PreparedPublicColumn>,
    pub group_map: Vec<PreparedGroup>,
    pub capabilities: PreparedCapabilities,
    pub rights_state: RightsState,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedPublicColumn {
    pub id: u32,
    pub dtype: PreparedType,
    pub role: PreparedRole,
}

fn error(code: &'static str) -> SchemaError {
    SchemaError::new(code, None, None)
}

fn json_digest(value: &impl Serialize) -> Result<String> {
    let bytes = serde_json::to_vec(value).map_err(|_| error("invalid_prepared_contract"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub fn implementation_hash() -> String {
    let mut hash = Sha256::new();
    for source in [
        include_str!("../schema.rs"),
        include_str!("../typed_data.rs"),
        include_str!("transforms.rs"),
        include_str!("transforms_replay.rs"),
        include_str!("prepared.rs"),
        include_str!("prepared_contract.rs"),
        include_str!("prepared_contract_validation.rs"),
        include_str!("prepared_fit.rs"),
        include_str!("prepared_adapter.rs"),
        include_str!("prepared_replay.rs"),
        include_str!("../../schemas/prepared-view-v1.schema.json"),
    ] {
        hash.update((source.len() as u64).to_le_bytes());
        hash.update(source.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

impl PreparedView {
    /// Structural/semantic validation alone does not establish trusted provenance.
    pub fn from_private_json(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > InputLimits::default().max_bytes {
            return Err(error("prepared_bytes_exceeded"));
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            format: String,
            version: u32,
            metadata: PreparedMetadata,
            rows: Vec<Vec<f64>>,
        }
        let wire: Wire =
            serde_json::from_slice(bytes).map_err(|_| error("invalid_prepared_json"))?;
        let view = Self {
            format: wire.format,
            version: wire.version,
            metadata: wire.metadata,
            rows: wire.rows,
        };
        view.validate()?;
        Ok(view)
    }

    pub fn validate(&self) -> Result<()> {
        if self.format != PREPARED_VIEW_FORMAT || self.version != 1 {
            return Err(error("prepared_input_required"));
        }
        self.metadata.validate()?;
        let limits = InputLimits::default();
        if self.rows.is_empty()
            || self.rows.len() > limits.max_rows
            || self
                .rows
                .len()
                .checked_mul(self.metadata.columns.len())
                .is_none_or(|n| n > limits.max_cells)
        {
            return Err(error("prepared_dimensions_exceeded"));
        }
        for row in &self.rows {
            if row.len() != self.metadata.columns.len() {
                return Err(error("prepared_column_mismatch"));
            }
            for (value, column) in row.iter().zip(&self.metadata.columns) {
                validate_prepared_number(*value, column, &self.metadata.target)?;
            }
        }
        Ok(())
    }

    pub fn private_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|_| error("invalid_prepared_json"))?;
        if bytes.len() > InputLimits::default().max_bytes {
            return Err(error("prepared_bytes_exceeded"));
        }
        Ok(bytes)
    }

    /// Trust this map only after explicit prepare or independent manifest admission.
    pub fn freeze(&self) -> Result<PreparedContract> {
        self.validate()?;
        Ok(PreparedContract {
            metadata: self.metadata.clone(),
        })
    }

    pub fn metadata(&self) -> &PreparedMetadata {
        &self.metadata
    }
    pub fn rows(&self) -> &[Vec<f64>] {
        &self.rows
    }

    pub fn public_projection(&self) -> Result<PreparedPublicProjection> {
        self.validate()?;
        Ok(PreparedPublicProjection {
            format: PREPARED_VIEW_FORMAT,
            version: 1,
            state: "public",
            task: self.metadata.task,
            columns: self
                .metadata
                .columns
                .iter()
                .map(|c| PreparedPublicColumn {
                    id: c.id,
                    dtype: c.dtype,
                    role: c.role,
                })
                .collect(),
            group_map: self.metadata.group_map.clone(),
            capabilities: self.metadata.capabilities.clone(),
            rights_state: self.metadata.rights.state,
        })
    }
}

impl PreparedContract {
    pub fn validate_view(&self, view: &PreparedView) -> Result<()> {
        view.validate()?;
        if view.metadata.columns != self.metadata.columns {
            return Err(error("prepared_column_mismatch"));
        }
        if view.metadata.group_map != self.metadata.group_map {
            return Err(error("prepared_group_mismatch"));
        }
        if view.metadata.target != self.metadata.target {
            return Err(error("prepared_class_mismatch"));
        }
        if view.metadata != self.metadata {
            return Err(error("prepared_contract_mismatch"));
        }
        if self.metadata.provenance.implementation_hash != implementation_hash() {
            return Err(error("prepared_implementation_mismatch"));
        }
        Ok(())
    }

    /// Prepared-only compile admission; this never fits or transforms input.
    pub fn compile_input<'a>(&self, view: &'a PreparedView) -> Result<PreparedCompileInput<'a>> {
        self.validate_view(view)?;
        if self.metadata.task == TaskKind::Multiclass {
            return Err(error("prepared_task_not_qualified"));
        }
        if self.metadata.rights.state == RightsState::Blocked {
            return Err(error("prepared_rights_blocked"));
        }
        Ok(PreparedCompileInput(view))
    }
}

/// Validate prepared bytes against the same frozen map for every arm.
pub fn validate_input(bytes: &[u8], contract: &PreparedContract) -> Result<PreparedView> {
    let view = PreparedView::from_private_json(bytes)?;
    contract.validate_view(&view)?;
    Ok(view)
}

include!("prepared_fit.rs");
include!("prepared_adapter.rs");
include!("prepared_replay.rs");

#[cfg(test)]
#[path = "prepared_tests.rs"]
mod tests;

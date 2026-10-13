use serde::{Deserialize, Serialize};

use crate::schema::{
    MAX_COLUMNS, MAX_SCHEMA_BYTES, MetadataVisibility, PublicColumn, opaque_id, validate_domain,
};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RetainedColumn {
    id: String,
    role: ColumnRole,
    dtype: ColumnType,
    mask_states: Vec<MaskState>,
    units: Option<String>,
    label: Option<String>,
    domain: Option<ColumnDomain>,
    offset: f64,
    scale: f64,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FittedTransforms {
    format: String,
    version: u32,
    fit_rows: usize,
    visibility: MetadataVisibility,
    columns: Vec<RetainedColumn>,
}

impl fmt::Debug for FittedTransforms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FittedTransforms")
            .field("fit_rows", &self.fit_rows)
            .field("columns", &self.columns.len())
            .finish()
    }
}

/// Fit only the caller's declared fitting rows. Other rows do not contribute.
pub fn fit_transform(table: &TypedTable, fit_rows: &[usize]) -> Result<FittedTransforms> {
    if fit_rows.is_empty() || fit_rows.len() > table.rows() {
        return Err(SchemaError::new("invalid_fit_rows", None, None));
    }
    let mut seen = BTreeSet::new();
    if fit_rows
        .iter()
        .any(|row| *row >= table.rows() || !seen.insert(*row))
    {
        return Err(SchemaError::new("invalid_fit_rows", None, None));
    }
    let mut columns = Vec::new();
    for (index, column) in table.schema.columns.iter().enumerate() {
        let input_column = table.input_positions[index] + 1;
        if column.role == ColumnRole::Target {
            if !column.dtype.numeric() {
                return Err(SchemaError::new(
                    "fit_target_not_numeric",
                    None,
                    Some(input_column),
                ));
            }
            for row in fit_rows {
                if table.columns[index][*row].number().is_none() {
                    return Err(SchemaError::new(
                        "fit_target_not_observed",
                        Some(row + 2),
                        Some(input_column),
                    ));
                }
            }
        }
        let learned = matches!(column.role, ColumnRole::Feature | ColumnRole::Target)
            && matches!(
                column.dtype,
                ColumnType::Continuous | ColumnType::Integer | ColumnType::Ordinal
            );
        let (offset, scale) = if learned {
            let mut low = f64::INFINITY;
            let mut high = f64::NEG_INFINITY;
            for row in fit_rows {
                if let Some(value) = table.columns[index][*row].number() {
                    low = low.min(value);
                    high = high.max(value);
                } else if column.role == ColumnRole::Target {
                    return Err(SchemaError::new(
                        "fit_target_not_observed",
                        Some(row + 2),
                        Some(input_column),
                    ));
                }
            }
            if !low.is_finite() || !high.is_finite() {
                return Err(SchemaError::new(
                    "fit_column_not_observed",
                    None,
                    Some(input_column),
                ));
            }
            let scale = if low == high { 1.0 } else { high - low };
            if !scale.is_finite() || scale <= 0.0 {
                return Err(SchemaError::new(
                    "affine_range_not_representable",
                    None,
                    Some(input_column),
                ));
            }
            (low, scale)
        } else {
            (0.0, 1.0)
        };
        columns.push(RetainedColumn {
            id: column.id.clone(),
            role: column.role,
            dtype: column.dtype,
            mask_states: column.mask_states(),
            units: column.units.clone(),
            label: (table.schema.metadata_visibility == MetadataVisibility::ApprovedSemanticLabels)
                .then(|| column.semantic_label.clone())
                .flatten(),
            domain: column.domain.clone(),
            offset,
            scale,
        });
    }
    let state = FittedTransforms {
        format: "dope-affine-state".into(),
        version: 1,
        fit_rows: fit_rows.len(),
        visibility: table.schema.metadata_visibility,
        columns,
    };
    state.validate()?;
    Ok(state)
}

impl FittedTransforms {
    fn validate(&self) -> Result<()> {
        if self.format != "dope-affine-state"
            || self.version != 1
            || self.fit_rows == 0
            || self.columns.len() < 2
            || self.columns.len() > MAX_COLUMNS
        {
            return Err(SchemaError::new("invalid_transform_state", None, None));
        }
        let mut ids = BTreeSet::new();
        let mut metadata_bytes = 256usize;
        for (index, column) in self.columns.iter().enumerate() {
            let learned = matches!(column.role, ColumnRole::Feature | ColumnRole::Target)
                && matches!(
                    column.dtype,
                    ColumnType::Continuous | ColumnType::Integer | ColumnType::Ordinal
                );
            if !opaque_id(&column.id)
                || !ids.insert(&column.id)
                || !column.offset.is_finite()
                || !column.scale.is_finite()
                || column.scale <= 0.0
                || !validate_domain(column.dtype, &column.domain)
                || column
                    .units
                    .iter()
                    .chain(&column.label)
                    .any(|x| x.is_empty() || x.len() > 128)
                || (self.visibility == MetadataVisibility::OpaqueIds && column.label.is_some())
                || matches!(column.dtype, ColumnType::DateTime | ColumnType::Text)
                || column.mask_states.iter().collect::<BTreeSet<_>>().len()
                    != column.mask_states.len()
                || (!learned && (column.offset != 0.0 || column.scale != 1.0))
            {
                return Err(SchemaError::new(
                    "invalid_transform_state",
                    None,
                    Some(index + 1),
                ));
            }
            for value in column.units.iter().chain(&column.label) {
                metadata_bytes = metadata_bytes.saturating_add(value.len() + 32);
            }
            if let Some(ColumnDomain::Categorical { values }) = &column.domain {
                for value in values {
                    metadata_bytes = metadata_bytes.saturating_add(value.len() + 32);
                }
            }
            metadata_bytes = metadata_bytes.saturating_add(256);
            if metadata_bytes > MAX_SCHEMA_BYTES {
                return Err(SchemaError::new("state_bytes_exceeded", None, None));
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
            return Err(SchemaError::new("invalid_transform_state", None, None));
        }
        Ok(())
    }

    fn column(&self, id: &str) -> Result<&RetainedColumn> {
        self.validate()?;
        self.columns
            .iter()
            .find(|c| c.id == id)
            .ok_or_else(|| SchemaError::new("unknown_column_id", None, None))
    }

    pub fn transform_numeric(&self, id: &str, values: &[f64]) -> Result<Vec<f64>> {
        self.numeric_map(id, values, false)
    }

    /// Restore predictions to their declared original units without clipping.
    pub fn inverse_transform(&self, id: &str, values: &[f64]) -> Result<Vec<f64>> {
        self.numeric_map(id, values, true)
    }

    fn numeric_map(&self, id: &str, values: &[f64], inverse: bool) -> Result<Vec<f64>> {
        let column = self.column(id)?;
        if !column.dtype.numeric() {
            return Err(SchemaError::new(
                "numeric_transform_unavailable",
                None,
                None,
            ));
        }
        values
            .iter()
            .enumerate()
            .map(|(row, value)| {
                let result = if inverse {
                    value * column.scale + column.offset
                } else {
                    (value - column.offset) / column.scale
                };
                if !value.is_finite() || !result.is_finite() {
                    return Err(SchemaError::new(
                        "nonfinite_transform_value",
                        Some(row + 1),
                        None,
                    ));
                }
                Ok(result)
            })
            .collect()
    }

    pub fn column_units(&self, id: &str) -> Result<Option<&str>> {
        Ok(self.column(id)?.units.as_deref())
    }

    pub fn transform_cell(&self, id: &str, cell: &Cell) -> Result<Cell> {
        let column = self.column(id)?;
        match cell {
            Cell::Mask(state) if column.mask_states.contains(state) => Ok(cell.clone()),
            Cell::Category(value)
                if column.dtype == ColumnType::Nominal
                    && !value.is_empty()
                    && value.len() <= 4096 =>
            {
                if let Some(ColumnDomain::Categorical { values }) = &column.domain {
                    if !values.contains(value) {
                        return Ok(Cell::Mask(MaskState::UnknownCategory));
                    }
                }
                Ok(cell.clone())
            }
            Cell::Boolean(_) if column.dtype == ColumnType::Boolean => Ok(cell.clone()),
            Cell::Number(value)
                if matches!(
                    column.dtype,
                    ColumnType::Continuous | ColumnType::Integer | ColumnType::Ordinal
                ) =>
            {
                Ok(Cell::Number(self.transform_numeric(id, &[*value])?[0]))
            }
            _ => Err(SchemaError::new("cell_type_or_mask_mismatch", None, None)),
        }
    }

    /// Required private sampling state: schema, units and affine inverses.
    /// Source-name maps and original rows are absent from this payload.
    pub fn sampling_state_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut output = BoundedStateBytes(Vec::new());
        serde_json::to_writer(&mut output, self)
            .map_err(|_| SchemaError::new("state_bytes_exceeded", None, None))?;
        Ok(output.0)
    }

    pub fn required_sampling_bytes(&self) -> Result<usize> {
        Ok(self.sampling_state_bytes()?.len())
    }

    pub fn reload(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_SCHEMA_BYTES {
            return Err(SchemaError::new("state_bytes_exceeded", None, None));
        }
        let state: Self = serde_json::from_slice(bytes)
            .map_err(|_| SchemaError::new("invalid_transform_state", None, None))?;
        state.validate()?;
        Ok(state)
    }

    pub fn public_projection(&self) -> Result<Vec<PublicColumn>> {
        self.validate()?;
        let approved = self.visibility == MetadataVisibility::ApprovedSemanticLabels;
        Ok(self
            .columns
            .iter()
            .map(|c| PublicColumn {
                id: c.id.clone(),
                role: c.role,
                dtype: c.dtype,
                mask_states: c.mask_states.clone(),
                label: approved.then(|| c.label.clone()).flatten(),
                units: approved.then(|| c.units.clone()).flatten(),
            })
            .collect())
    }
}

struct BoundedStateBytes(Vec<u8>);

impl Write for BoundedStateBytes {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self
            .0
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > MAX_SCHEMA_BYTES)
        {
            return Err(std::io::Error::other("state_bytes_exceeded"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

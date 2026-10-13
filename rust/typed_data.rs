//! Headered raw input. This path preserves original values and mask semantics;
//! it does not invoke the historical unit-interval Table clipping path.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::{Read, Write};
use std::path::Path;

use crate::schema::{
    ColumnDomain, ColumnRole, ColumnSchemaV1, ColumnType, MaskState, Result, SchemaError,
    TableSchemaV1,
};

#[derive(Clone, Debug, PartialEq)]
pub enum Cell {
    Number(f64),
    Boolean(bool),
    Category(String),
    Mask(MaskState),
}

impl Cell {
    pub(crate) fn number(&self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(*value),
            Self::Boolean(value) => Some(u8::from(*value) as f64),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InputLimits {
    pub max_bytes: usize,
    pub max_rows: usize,
    pub max_cells: usize,
    pub max_field_bytes: usize,
}

impl Default for InputLimits {
    fn default() -> Self {
        Self {
            max_bytes: 16 * 1024 * 1024,
            max_rows: 100_000,
            max_cells: 1_000_000,
            max_field_bytes: 4096,
        }
    }
}

pub struct TypedTable {
    pub(crate) schema: TableSchemaV1,
    pub(crate) columns: Vec<Vec<Cell>>,
    pub(crate) input_positions: Vec<usize>,
    rows: usize,
}

impl fmt::Debug for TypedTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TypedTable")
            .field("rows", &self.rows)
            .field("columns", &self.columns.len())
            .finish()
    }
}

impl TypedTable {
    pub fn read_csv(reader: impl Read, schema: TableSchemaV1, limits: InputLimits) -> Result<Self> {
        schema.validate()?;
        if limits.max_bytes == 0
            || limits.max_rows == 0
            || limits.max_cells == 0
            || limits.max_field_bytes == 0
        {
            return Err(SchemaError::new("invalid_input_limits", None, None));
        }
        let cap = limits
            .max_bytes
            .checked_add(1)
            .ok_or_else(|| SchemaError::new("invalid_input_limits", None, None))?;
        let mut bytes = Vec::new();
        reader
            .take(cap as u64)
            .read_to_end(&mut bytes)
            .map_err(|_| SchemaError::new("input_io_failed", None, None))?;
        if bytes.len() > limits.max_bytes {
            return Err(SchemaError::new("input_bytes_exceeded", None, None));
        }
        let mut csv = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_reader(bytes.as_slice());
        let headers = csv
            .headers()
            .map_err(|_| SchemaError::new("invalid_csv_header", Some(1), None))?
            .clone();
        if headers.len() != schema.columns.len() {
            return Err(SchemaError::new("schema_header_mismatch", Some(1), None));
        }
        let mut unique = BTreeSet::new();
        for (index, header) in headers.iter().enumerate() {
            if header.trim().is_empty() || !unique.insert(header) {
                return Err(SchemaError::new(
                    "ambiguous_csv_header",
                    Some(1),
                    Some(index + 1),
                ));
            }
        }
        let positions: Vec<usize> = schema
            .columns
            .iter()
            .enumerate()
            .map(|(index, column)| {
                headers
                    .iter()
                    .position(|header| header == column.source_name)
                    .ok_or_else(|| {
                        SchemaError::new("schema_header_mismatch", Some(1), Some(index + 1))
                    })
            })
            .collect::<Result<_>>()?;
        let mut columns = vec![Vec::new(); schema.columns.len()];
        let mut rows = 0;
        for record in csv.records() {
            let row = rows + 2;
            let record =
                record.map_err(|_| SchemaError::new("invalid_csv_record", Some(row), None))?;
            if record.len() != headers.len() {
                return Err(SchemaError::new("wrong_field_count", Some(row), None));
            }
            if rows >= limits.max_rows
                || (rows + 1)
                    .checked_mul(columns.len())
                    .is_none_or(|n| n > limits.max_cells)
            {
                return Err(SchemaError::new(
                    "input_dimensions_exceeded",
                    Some(row),
                    None,
                ));
            }
            for (index, column) in schema.columns.iter().enumerate() {
                let position = positions[index];
                let value = &record[position];
                if value.len() > limits.max_field_bytes {
                    return Err(SchemaError::new(
                        "field_bytes_exceeded",
                        Some(row),
                        Some(position + 1),
                    ));
                }
                columns[index].push(parse_cell(value, column, row, position + 1)?);
            }
            rows += 1;
        }
        if rows == 0 {
            return Err(SchemaError::new("empty_table", None, None));
        }
        Ok(Self {
            schema,
            columns,
            input_positions: positions,
            rows,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn schema(&self) -> &TableSchemaV1 {
        &self.schema
    }

    pub fn feature_cells(&self, row: usize) -> Result<Vec<(&str, &Cell)>> {
        if row >= self.rows {
            return Err(SchemaError::new("row_out_of_range", None, None));
        }
        Ok(self
            .schema
            .columns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.role == ColumnRole::Feature)
            .map(|(index, c)| (c.id.as_str(), &self.columns[index][row]))
            .collect())
    }

    pub fn cell(&self, row: usize, column_id: &str) -> Result<&Cell> {
        let index = self
            .schema
            .columns
            .iter()
            .position(|c| c.id == column_id)
            .ok_or_else(|| SchemaError::new("unknown_column_id", None, None))?;
        self.columns[index]
            .get(row)
            .ok_or_else(|| SchemaError::new("row_out_of_range", None, None))
    }

    pub fn target_id(&self) -> &str {
        &self
            .schema
            .columns
            .iter()
            .find(|c| c.role == ColumnRole::Target)
            .expect("validated target role")
            .id
    }

    /// Create an owner-only local source-name map. Existing paths are refused.
    #[cfg(unix)]
    pub fn write_private_source_map(&self, path: &Path) -> Result<usize> {
        use std::os::unix::fs::OpenOptionsExt;
        let names: BTreeMap<_, _> = self
            .schema
            .columns
            .iter()
            .map(|c| (&c.id, &c.source_name))
            .collect();
        let bytes = serde_json::to_vec(&names)
            .map_err(|_| SchemaError::new("metadata_encoding_failed", None, None))?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .map_err(|_| SchemaError::new("private_map_create_failed", None, None))?;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|_| SchemaError::new("private_map_write_failed", None, None))?;
        Ok(bytes.len())
    }
}

fn parse_cell(value: &str, column: &ColumnSchemaV1, row: usize, index: usize) -> Result<Cell> {
    let error = |code| SchemaError::new(code, Some(row), Some(index));
    if let Some(token) = column.mask_tokens.iter().find(|token| token.token == value) {
        return Ok(Cell::Mask(token.state));
    }
    if value.is_empty() {
        return Err(error("undeclared_missing_value"));
    }
    match column.dtype {
        ColumnType::Continuous | ColumnType::Integer | ColumnType::Ordinal => {
            let number: f64 = if column.dtype == ColumnType::Continuous {
                value
                    .trim()
                    .parse()
                    .map_err(|_| error("invalid_numeric_value"))?
            } else {
                let integer: i64 = value
                    .trim()
                    .parse()
                    .map_err(|_| error("integer_precision_or_type_mismatch"))?;
                if integer.unsigned_abs() > 9_007_199_254_740_991 {
                    return Err(error("integer_precision_or_type_mismatch"));
                }
                integer as f64
            };
            validate_numeric_cell(number, column.dtype, &column.domain, error)?;
            Ok(Cell::Number(number))
        }
        ColumnType::Boolean => {
            let boolean = match value {
                "0" | "false" => false,
                "1" | "true" => true,
                _ => return Err(error("invalid_boolean_value")),
            };
            check_numeric_domain(u8::from(boolean) as f64, &column.domain, error)?;
            Ok(Cell::Boolean(boolean))
        }
        ColumnType::Nominal => {
            if let Some(ColumnDomain::Categorical { values }) = &column.domain {
                if !values.iter().any(|allowed| allowed == value) {
                    return Ok(Cell::Mask(MaskState::UnknownCategory));
                }
            }
            Ok(Cell::Category(value.to_owned()))
        }
        _ => Err(error("unsupported_logical_type")),
    }
}

fn validate_numeric_cell(
    number: f64,
    dtype: ColumnType,
    domain: &Option<ColumnDomain>,
    error: impl Fn(&'static str) -> SchemaError,
) -> Result<()> {
    if !number.is_finite() {
        return Err(error("nonfinite_numeric_value"));
    }
    if dtype == ColumnType::Boolean && number != 0.0 && number != 1.0 {
        return Err(error("invalid_boolean_value"));
    }
    if matches!(dtype, ColumnType::Integer | ColumnType::Ordinal)
        && (number.fract() != 0.0 || number.abs() > 9_007_199_254_740_991.0)
    {
        return Err(error("integer_precision_or_type_mismatch"));
    }
    check_numeric_domain(number, domain, error)
}

fn check_numeric_domain(
    number: f64,
    domain: &Option<ColumnDomain>,
    error: impl Fn(&'static str) -> SchemaError,
) -> Result<()> {
    if let Some(ColumnDomain::Numeric { lower, upper }) = domain {
        if lower.is_some_and(|bound| number < bound) || upper.is_some_and(|bound| number > bound) {
            return Err(error("outside_declared_domain"));
        }
    }
    Ok(())
}

include!("typed_data/transforms.rs");

//! Private consumer view; no transform, source names, or serialization is added.
use dope_norm::{NormError, Result, UnitSchema, UnitTable, MAX_CELLS, MAX_ROWS};

/// Owner-private canonical rows, with the declared target last and physical IDs intact.
/// This type intentionally implements neither Serialize nor source-bearing Debug.
pub struct TargetLastView {
    row_ids: Vec<u64>,
    schema: UnitSchema,
    rows: Vec<Vec<f64>>,
    target_column: usize,
    source_sha256: String,
    fit_scope_sha256: String,
    transform_sha256: String,
}

impl std::fmt::Debug for TargetLastView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TargetLastView(<private>)")
    }
}

impl TargetLastView {
    pub fn row_ids(&self) -> &[u64] {
        &self.row_ids
    }
    /// Structural feature schema; columns excludes the target.
    pub fn schema(&self) -> &UnitSchema {
        &self.schema
    }
    pub fn rows(&self) -> &[Vec<f64>] {
        &self.rows
    }
    pub fn target_column(&self) -> usize {
        self.target_column
    }
    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }
    pub fn fit_scope_sha256(&self) -> &str {
        &self.fit_scope_sha256
    }
    pub fn transform_sha256(&self) -> &str {
        &self.transform_sha256
    }
}

/// Import this trait from `dope_sdk::nc1` to expose the native consumer conversion.
pub trait TargetLastViewExt {
    fn to_target_last_view(&self) -> Result<TargetLastView>;
}

fn error(code: &'static str) -> NormError {
    NormError { code, column: None }
}

fn checked_width(rows: usize, columns: usize) -> Result<usize> {
    let width = columns
        .checked_add(1)
        .ok_or(error("target_last_shape_limit"))?;
    if rows == 0
        || rows > MAX_ROWS
        || columns == 0
        || columns > dope_norm::MAX_UNIT_COLUMNS
        || rows.checked_mul(width).is_none_or(|n| n > MAX_CELLS)
    {
        return Err(error("target_last_shape_limit"));
    }
    Ok(width)
}

impl TargetLastViewExt for UnitTable {
    fn to_target_last_view(&self) -> Result<TargetLastView> {
        self.schema().validate()?;
        let y = self.y().ok_or(error("target_last_target_absent"))?;
        let width = checked_width(self.x().len(), self.schema().columns)?;
        if self.row_ids().len() != self.x().len()
            || y.len() != self.x().len()
            || y.iter().any(|v| !v.is_finite())
            || self
                .x()
                .iter()
                .any(|row| row.len() != self.schema().columns || row.iter().any(|v| !v.is_finite()))
        {
            return Err(error("target_last_invalid_table"));
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(self.x().len())
            .map_err(|_| error("target_last_allocation_refused"))?;
        for (x, &target) in self.x().iter().zip(y) {
            let mut row = Vec::new();
            row.try_reserve_exact(width)
                .map_err(|_| error("target_last_allocation_refused"))?;
            row.extend_from_slice(x);
            row.push(target);
            rows.push(row);
        }
        let mut row_ids = Vec::new();
        row_ids
            .try_reserve_exact(self.row_ids().len())
            .map_err(|_| error("target_last_allocation_refused"))?;
        row_ids.extend_from_slice(self.row_ids());
        Ok(TargetLastView {
            row_ids,
            schema: self.schema().clone(),
            rows,
            target_column: self.schema().columns,
            source_sha256: self.source_sha256().to_owned(),
            fit_scope_sha256: self.fit_scope_sha256().to_owned(),
            transform_sha256: self.transform_sha256().to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_last_budget_includes_target_and_rejects_overflow_before_allocation() {
        assert_eq!(checked_width(MAX_CELLS / 257, 256).unwrap(), 257);
        assert_eq!(
            checked_width(MAX_CELLS / 257 + 1, 256).unwrap_err().code,
            "target_last_shape_limit"
        );
        for (rows, columns) in [(0, 1), (1, 0), (MAX_ROWS + 1, 1), (1, usize::MAX)] {
            assert_eq!(
                checked_width(rows, columns).unwrap_err().code,
                "target_last_shape_limit"
            );
        }
    }
}

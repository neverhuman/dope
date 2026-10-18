/// Explicit replay state; a synthetic arm already in prepared coordinates must
/// enter `validate_input`/`compile_input`, never the raw transform path.
#[derive(Debug)]
pub enum ReplayInput<'a> {
    Raw(&'a TypedTable, &'a [usize]),
    Prepared(&'a PreparedView),
}

impl PreparedContract {
    pub fn replay(&self, input: ReplayInput<'_>) -> Result<PreparedView> {
        match input {
            ReplayInput::Raw(table, rows) => self.apply(table, rows),
            ReplayInput::Prepared(_) => Err(error("prepared_already_transformed")),
        }
    }

    /// Recover private original-unit cells in the view's declared column order.
    /// Features are inverted exactly once; target rows already use raw units.
    /// Numeric inverse replay is unclipped, as with regression_prediction.
    pub fn original_rows(&self, view: &PreparedView) -> Result<Vec<Vec<Cell>>> {
        self.validate_view(view)?;
        view.rows
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&self.metadata.columns)
                    .map(|(value, column)| {
                        if column.role == PreparedRole::Feature {
                            match &column.transform {
                                PreparedTransform::Affine { offset, scale } => {
                                    let raw = value * scale + offset;
                                    if !raw.is_finite() {
                                        return Err(error("prepared_nonfinite_value"));
                                    }
                                    return Ok(Cell::Number(raw));
                                }
                                PreparedTransform::CategoryIndex { categories } => {
                                    return Ok(Cell::Category(categories[*value as usize].clone()));
                                }
                                _ => {}
                            }
                        }
                        Ok(if column.dtype == PreparedType::Boolean {
                            Cell::Boolean(*value == 1.0)
                        } else {
                            Cell::Number(*value)
                        })
                    })
                    .collect()
            })
            .collect()
    }
}

fn validate_prepared_number(
    value: f64,
    column: &PreparedColumn,
    target: &PreparedTarget,
) -> Result<()> {
    if !value.is_finite() {
        return Err(error("prepared_nonfinite_value"));
    }
    if column.role == PreparedRole::Feature {
        if let PreparedTransform::Affine { offset, scale } = column.transform {
            if !(value * scale + offset).is_finite() {
                return Err(error("prepared_nonfinite_value"));
            }
        }
    }
    let bound = |b: f64| {
        if column.role == PreparedRole::Feature {
            if let PreparedTransform::Affine { offset, scale } = column.transform {
                return (b - offset) / scale;
            }
        }
        b
    };
    if column.bounds.lower.is_some_and(|b| value < bound(b))
        || column.bounds.upper.is_some_and(|b| value > bound(b))
    {
        return Err(error("prepared_support_mismatch"));
    }
    match column.dtype {
        PreparedType::Integer if value.fract() != 0.0 || value.abs() > 9_007_199_254_740_991.0 => {
            return Err(error("prepared_type_mismatch"));
        }
        PreparedType::Boolean if value != 0.0 && value != 1.0 => {
            return Err(error("prepared_type_mismatch"));
        }
        PreparedType::CategoryCode => {
            if let PreparedTransform::CategoryIndex { categories } = &column.transform {
                if value.fract() != 0.0 || value < 0.0 || value >= categories.len() as f64 {
                    return Err(error("prepared_oov_rejected"));
                }
            }
        }
        _ => {}
    }
    if column.role == PreparedRole::Target {
        if let PreparedTarget::Discrete { class_order } = target {
            if !class_order.iter().any(|id| *id as f64 == value) {
                return Err(error("prepared_class_mismatch"));
            }
        }
    }
    Ok(())
}

impl PreparedContract {
    /// Apply the retained map to any arm; no scaler or dictionary is fitted here.
    pub fn apply(&self, table: &TypedTable, rows: &[usize]) -> Result<PreparedView> {
        selected_rows(table, rows)?;
        self.metadata.validate()?;
        if column_digest(table)? != self.metadata.provenance.column_digest {
            return Err(error("prepared_column_mismatch"));
        }
        if self.metadata.provenance.implementation_hash != implementation_hash() {
            return Err(error("prepared_implementation_mismatch"));
        }
        let mut output = Vec::new();
        for row in rows {
            let mut values = Vec::new();
            for column in &self.metadata.columns {
                let cell = table.cell(*row, &column.source_id)?;
                let raw = match cell {
                    Cell::Number(value) => *value,
                    Cell::Boolean(value) => u8::from(*value) as f64,
                    Cell::Category(value) => {
                        if let PreparedTransform::CategoryIndex { categories } = &column.transform {
                            categories
                                .iter()
                                .position(|c| c == value)
                                .ok_or_else(|| error("prepared_oov_rejected"))?
                                as f64
                        } else {
                            return Err(error("prepared_type_mismatch"));
                        }
                    }
                    Cell::Mask(_) => return Err(error("prepared_missing_rejected")),
                };
                let value = if column.role == PreparedRole::Feature {
                    match column.transform {
                        PreparedTransform::Affine { offset, scale } => (raw - offset) / scale,
                        _ => raw,
                    }
                } else {
                    raw
                };
                validate_prepared_number(value, column, &self.metadata.target)?;
                values.push(value);
            }
            output.push(values);
        }
        let view = PreparedView {
            format: PREPARED_VIEW_FORMAT.into(),
            version: 1,
            metadata: self.metadata.clone(),
            rows: output,
        };
        view.validate()?;
        Ok(view)
    }

    /// Inverse for workers that use the recorded regression target affine map.
    /// PreparedView target rows themselves always remain in original units.
    pub fn regression_prediction(&self, prepared_prediction: f64) -> Result<f64> {
        if self.metadata.task != TaskKind::Regression {
            return Err(error("prepared_type_mismatch"));
        }
        let column = self
            .metadata
            .columns
            .last()
            .ok_or_else(|| error("invalid_prepared_contract"))?;
        if let PreparedTransform::Affine { offset, scale } = column.transform {
            let value = prepared_prediction * scale + offset;
            if value.is_finite() && prepared_prediction.is_finite() {
                return Ok(value);
            }
        }
        Err(error("prepared_nonfinite_value"))
    }
}

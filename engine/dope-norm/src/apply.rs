use crate::card::{FeatureState, TargetState, UnitData};
use crate::fit::{normalize, row_positions};
use crate::{ApplyReceipt, Cell, InputTable, NormError, Result, ScaleCard, UnitTable};

impl ScaleCard {
    /// Apply the frozen map to features and target of explicit owner rows.
    /// Query source identity may differ from FIT; it is bound separately in the matrix.
    pub fn apply(&self, table: &InputTable, rows: &[u64]) -> Result<(UnitTable, ApplyReceipt)> {
        self.apply_inner(table, rows, true)
    }
    /// Apply features only. Target cells are never read and target column may be absent.
    /// Retained feature source positions/names must match the private declared schema.
    pub fn apply_features(
        &self,
        table: &InputTable,
        rows: &[u64],
    ) -> Result<(UnitTable, ApplyReceipt)> {
        self.apply_inner(table, rows, false)
    }
    fn apply_inner(
        &self,
        table: &InputTable,
        rows: &[u64],
        with_target: bool,
    ) -> Result<(UnitTable, ApplyReceipt)> {
        self.validate()?;
        table.validate()?;
        let feature_positions: Vec<usize> = self
            .data
            .features
            .iter()
            .map(|feat| {
                table
                    .columns
                    .iter()
                    .position(|column| column.name == feat.name)
                    .ok_or(NormError::at("source_schema_mismatch", feat.column))
            })
            .collect::<Result<_>>()?;
        let target_col = self.data.declared.target.column;
        let target_position = if with_target {
            Some(
                table
                    .columns
                    .iter()
                    .position(|column| column.name == self.data.names[target_col])
                    .ok_or(NormError::at("target_schema_mismatch", target_col))?,
            )
        } else {
            None
        };
        if rows
            .len()
            .checked_mul(self.data.public.columns)
            .is_none_or(|n| n > crate::MAX_CELLS)
        {
            return Err(NormError::new("output_shape_limit"));
        }
        let positions = row_positions(table, rows, false)?;
        let g = self.data.features.len();
        let mut receipt = ApplyReceipt {
            rows: rows.len(),
            source_sha256: table.source_sha256.clone(),
            fit_source_sha256: self.data.source_sha256.clone(),
            unobserved_fit_classes: self
                .data
                .fit_class_counts
                .iter()
                .enumerate()
                .filter_map(|(i, n)| (*n == 0).then_some(i))
                .collect(),
            out_of_range: vec![0; g],
            unseen_levels: vec![0; g],
            unexpected_missing: vec![0; g],
            schema_sha256: self.data.public.sha256()?,
            fit_scope_sha256: self.data.fit_scope_sha256.clone(),
            transform_sha256: self.data.transform_sha256.clone(),
        };
        let mut x = Vec::with_capacity(rows.len());
        let mut y = with_target.then(|| Vec::with_capacity(rows.len()));
        for &i in &positions {
            let mut row = Vec::with_capacity(self.data.public.columns);
            for (group_id, feat) in self.data.features.iter().enumerate() {
                let value = &table.columns[feature_positions[group_id]].values[i];
                let missing = matches!(value, Cell::Missing);
                if missing && !feat.has_missing {
                    receipt.unexpected_missing[group_id] += 1;
                }
                match &feat.state {
                    FeatureState::Numeric { numeric: n } => {
                        let u = match value {
                            Cell::Missing => n.median,
                            Cell::Number(v) => {
                                if self.data.public.groups[group_id].kind
                                    == crate::GroupKind::Integer
                                    && v.fract() != 0.0
                                {
                                    return Err(NormError::at("invalid_integer", feat.column));
                                }
                                if *v < n.lo || *v > n.hi {
                                    receipt.out_of_range[group_id] += 1;
                                }
                                normalize(*v, n).map_err(|e| NormError::at(e.code, feat.column))?
                            }
                            _ => return Err(NormError::at("invalid_numeric", feat.column)),
                        };
                        row.push(u);
                    }
                    FeatureState::Boolean => row.push(match value {
                        Cell::Boolean(v) => f64::from(*v),
                        Cell::Missing => 0.0,
                        _ => return Err(NormError::at("invalid_boolean", feat.column)),
                    }),
                    FeatureState::Ordinal { levels, median } => row.push(match value {
                        Cell::Missing => *median,
                        Cell::Category(label) => levels
                            .iter()
                            .position(|v| v == label)
                            .map(|rank| rank as f64 / (levels.len() - 1) as f64)
                            .ok_or(NormError::at("unknown_ordinal", feat.column))?,
                        _ => return Err(NormError::at("invalid_ordinal", feat.column)),
                    }),
                    FeatureState::Nominal { levels } => {
                        let width = self.data.public.groups[group_id].width;
                        let selected = match value {
                            Cell::Missing if feat.has_missing => levels.len() + 1,
                            Cell::Missing => levels.len(),
                            Cell::Category(label) => match levels.iter().position(|v| v == label) {
                                Some(k) => k,
                                None => {
                                    receipt.unseen_levels[group_id] += 1;
                                    levels.len()
                                }
                            },
                            _ => return Err(NormError::at("invalid_nominal", feat.column)),
                        };
                        row.extend((0..width).map(|k| f64::from(k == selected)));
                    }
                }
                if feat.has_missing && !matches!(feat.state, FeatureState::Nominal { .. }) {
                    row.push(f64::from(missing));
                }
            }
            x.push(row);
            if let Some(targets) = &mut y {
                let value = &table.columns
                    [target_position.ok_or(NormError::new("missing_target_position"))?]
                .values[i];
                let target = match &self.data.target {
                    TargetState::Regression { numeric } => {
                        let Cell::Number(v) = value else {
                            return Err(NormError::at("invalid_target", target_col));
                        };
                        normalize(*v, numeric).map_err(|e| NormError::at(e.code, target_col))?
                    }
                    TargetState::Classification { labels } => {
                        let Cell::Category(label) = value else {
                            return Err(NormError::at("invalid_target", target_col));
                        };
                        labels
                            .iter()
                            .position(|s| s == label)
                            .map(|k| k as f64)
                            .ok_or(NormError::at("unknown_target_class", target_col))?
                    }
                };
                targets.push(target);
            }
        }
        let out = UnitTable {
            data: UnitData {
                schema: self.data.public.clone(),
                source_sha256: table.source_sha256.clone(),
                fit_scope_sha256: self.data.fit_scope_sha256.clone(),
                transform_sha256: self.data.transform_sha256.clone(),
                row_ids: rows.to_vec(),
                x,
                y,
            },
        };
        out.validate()?;
        Ok((out, receipt))
    }
}

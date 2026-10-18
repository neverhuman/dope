fn selected_rows(table: &TypedTable, rows: &[usize]) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    if rows.is_empty()
        || rows.len() > table.rows()
        || rows.iter().any(|r| *r >= table.rows() || !seen.insert(*r))
    {
        return Err(error("invalid_fit_rows"));
    }
    Ok(())
}

fn column_digest(table: &TypedTable) -> Result<String> {
    // Includes private declarations, never exported in the public projection.
    let columns: Vec<_> = table
        .schema
        .columns
        .iter()
        .map(|c| {
            (
                &c.id,
                &c.source_name,
                c.dtype,
                c.role,
                &c.domain,
                &c.units,
                &c.semantic_label,
                &c.mask_tokens,
            )
        })
        .collect();
    json_digest(&(table.schema.metadata_visibility, columns))
}

/// Explicit raw operation. Only `fit_rows` determine maps and output rows.
/// Entity/split digests are caller attestations; no sealed source is opened.
pub fn prepare(table: &TypedTable, fit_rows: &[usize], spec: PrepareSpec) -> Result<PreparedView> {
    selected_rows(table, fit_rows)?;
    let fitted = fit_transform(table, fit_rows)?;
    let mut selected: Vec<_> = table
        .schema
        .columns
        .iter()
        .enumerate()
        .filter(|(_, c)| c.role == ColumnRole::Feature)
        .collect();
    selected.extend(
        table
            .schema
            .columns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.role == ColumnRole::Target),
    );
    let mut columns = Vec::new();
    for (index, (source_index, source)) in selected.iter().enumerate() {
        let target = source.role == ColumnRole::Target;
        let state = fitted.column(&source.id)?;
        let (dtype, transform) = if target && spec.task != TaskKind::Regression {
            (PreparedType::Integer, PreparedTransform::Identity {})
        } else if source.dtype == ColumnType::Nominal {
            let mut categories = Vec::new();
            for row in fit_rows {
                if let Cell::Category(value) = &table.columns[*source_index][*row] {
                    if !categories.contains(value) {
                        categories.push(value.clone());
                    }
                } else {
                    return Err(error("prepared_missing_rejected"));
                }
            }
            (
                PreparedType::CategoryCode,
                PreparedTransform::CategoryIndex { categories },
            )
        } else if source.dtype == ColumnType::Boolean && !target {
            (PreparedType::Boolean, PreparedTransform::Identity {})
        } else {
            (
                PreparedType::Continuous,
                PreparedTransform::Affine {
                    offset: state.offset,
                    scale: state.scale,
                },
            )
        };
        let bounds = match &source.domain {
            Some(ColumnDomain::Numeric { lower, upper }) => PreparedBounds {
                lower: *lower,
                upper: *upper,
            },
            _ => PreparedBounds {
                lower: None,
                upper: None,
            },
        };
        columns.push(PreparedColumn {
            id: (index + 1) as u32,
            source_id: source.id.clone(),
            dtype,
            role: if target {
                PreparedRole::Target
            } else {
                PreparedRole::Feature
            },
            transform,
            bounds,
        });
    }
    if let PreparedTarget::Regression { units } = &spec.target {
        let source = selected
            .last()
            .ok_or_else(|| error("invalid_prepared_contract"))?
            .1;
        if units != source.units.as_deref().unwrap_or("unspecified") {
            return Err(error("prepared_units_mismatch"));
        }
    }
    let metadata = PreparedMetadata {
        task: spec.task,
        horizon: spec.horizon,
        columns,
        group_map: spec.group_map,
        target: spec.target,
        policies: PreparedPolicies::default(),
        provenance: PreparedProvenance {
            fit_digest: fitted
                .fit_row_digest()
                .ok_or_else(|| error("fit_row_digest_required"))?
                .into(),
            entity_digest: spec.entity_digest,
            split_digest: spec.split_digest,
            column_digest: column_digest(table)?,
            implementation_hash: implementation_hash(),
        },
        rights: spec.rights,
        projection: PreparedProjection {
            state: "private".into(),
            public: "opaque_only".into(),
        },
        capabilities: PreparedCapabilities::default(),
    };
    metadata.validate()?;
    PreparedContract { metadata }.apply(table, fit_rows)
}

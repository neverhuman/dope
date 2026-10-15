fn class(cell: &Cell, feature: &Feature) -> Result<usize> {
    match cell {
        Cell::Mask(mask) => Ok(feature.observed_classes()
            + MASKS
                .iter()
                .position(|value| value == mask)
                .expect("all mask variants")),
        Cell::Category(value) => feature
            .categories
            .binary_search(value)
            .map_err(|_| unavailable("mixed_fit_category_unavailable")),
        Cell::Number(_) | Cell::Boolean(_) if feature.categories.is_empty() => Ok(0),
        _ => Err(rejected("mixed_cell_type_mismatch")),
    }
}

fn probabilities(counts: &[usize]) -> Vec<f64> {
    let total = counts.iter().sum::<usize>() as f64;
    counts.iter().map(|count| *count as f64 / total).collect()
}

fn entropy(counts: &[usize]) -> f64 {
    probabilities(counts)
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.ln())
        .sum()
}

fn fit_state(
    table: &TypedTable,
    rows: &[usize],
    profile: MixedProfile,
    started: Instant,
    limit: Option<Duration>,
) -> Result<(State, Vec<f32>, Vec<f32>)> {
    let mut affine = fit_transform(table, rows).map_err(schema_error)?;
    affine
        .columns
        .retain(|c| matches!(c.role, ColumnRole::Feature | ColumnRole::Target));
    affine.columns.sort_by_key(|c| c.role == ColumnRole::Target);
    if affine
        .columns
        .last()
        .is_none_or(|c| c.dtype != ColumnType::Continuous)
    {
        return Err(unavailable("mixed_target_requires_observed_continuous"));
    }
    let mut features = Vec::new();
    let mut width = 0;
    for column in affine
        .columns
        .iter()
        .filter(|c| c.role == ColumnRole::Feature)
    {
        deadline(started, limit)?;
        if !column.dtype.numeric() && column.dtype != ColumnType::Nominal {
            return Err(unavailable("mixed_feature_type_unavailable"));
        }
        let mut categories = std::collections::BTreeSet::new();
        let mut sum = 0.0;
        let mut observed = 0;
        for &row in rows {
            match table.cell(row, &column.id).map_err(schema_error)? {
                Cell::Category(value) => {
                    categories.insert(value.clone());
                    if categories.len() > MAX_CATEGORIES {
                        return Err(unavailable("mixed_category_cap_exceeded"));
                    }
                    observed += 1;
                }
                Cell::Mask(_) => {}
                cell => {
                    let value = cell
                        .number()
                        .ok_or_else(|| rejected("mixed_cell_type_mismatch"))?;
                    sum += (value - column.offset) / column.scale;
                    observed += 1;
                }
            }
        }
        if observed == 0 {
            return Err(unavailable("mixed_feature_not_observed"));
        }
        let categories = categories.into_iter().collect::<Vec<_>>();
        let mut feature = Feature {
            id: column.id.clone(),
            start: width,
            categories,
            completion: if column.dtype == ColumnType::Nominal {
                0.0
            } else {
                (sum / observed as f64) as f32
            },
            probabilities: vec![],
            split: None,
        };
        width += feature.width();
        if width > MAX_LATENT_FEATURES
            || rows.len().checked_mul(width).is_none_or(|n| n > 1_000_000)
        {
            return Err(unavailable("mixed_latent_dimension_cap_exceeded"));
        }
        let mut counts = vec![0; feature.observed_classes() + MASKS.len()];
        for &row in rows {
            counts[class(table.cell(row, &column.id).map_err(schema_error)?, &feature)?] += 1;
        }
        feature.probabilities = probabilities(&counts);
        if !feature.categories.is_empty() {
            feature.completion = counts[..feature.categories.len()]
                .iter()
                .enumerate()
                .max_by_key(|(index, count)| (**count, std::cmp::Reverse(*index)))
                .expect("observed category")
                .0 as f32;
        }
        features.push(feature);
    }
    let mut x = Vec::with_capacity(rows.len() * width);
    let mut y = Vec::with_capacity(rows.len());
    for &row in rows {
        deadline(started, limit)?;
        for (index, feature) in features.iter().enumerate() {
            let cell = table.cell(row, &feature.id).map_err(schema_error)?;
            if feature.categories.is_empty() {
                let column = &affine.columns[index];
                let normalized = cell
                    .number()
                    .map(|v| ((v - column.offset) / column.scale) as f32)
                    .unwrap_or(feature.completion);
                if !normalized.is_finite() || !(0.0..=1.0).contains(&normalized) {
                    return Err(rejected("mixed_fit_normalization_invalid"));
                }
                x.push(normalized);
            } else {
                let at = match cell {
                    Cell::Category(value) => feature
                        .categories
                        .binary_search(value)
                        .expect("fit dictionary"),
                    _ => feature.completion as usize,
                };
                x.extend((0..feature.width()).map(|index| u8::from(index == at) as f32));
            }
        }
        let target = affine.columns.last().expect("validated target");
        let value = table
            .cell(row, &target.id)
            .map_err(schema_error)?
            .number()
            .ok_or_else(|| unavailable("mixed_target_requires_observed_continuous"))?;
        y.push(((value - target.offset) / target.scale) as f32);
    }
    if profile == MixedProfile::NumericSplitV1 {
        // Anchors are wholly observed numeric features; at most eight are
        // considered in opaque-id order. No target/held-out values enter splits.
        let mut anchors = features
            .iter()
            .enumerate()
            .filter(|(_, f)| f.categories.is_empty() && f.probabilities[0] == 1.0)
            .map(|(index, f)| (index, f.id.clone(), f.start))
            .collect::<Vec<_>>();
        anchors.sort_by(|a, b| a.1.cmp(&b.1));
        anchors.truncate(8);
        for feature in &mut features {
            let variable = feature.probabilities.iter().filter(|p| **p > 0.0).count() > 1;
            if !variable || (feature.categories.is_empty() && feature.probabilities[0] == 1.0) {
                continue;
            }
            let classes = rows
                .iter()
                .map(|row| {
                    class(
                        table.cell(*row, &feature.id).map_err(schema_error)?,
                        feature,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let mut best: Option<(f64, Split)> = None;
            for (anchor, _, start) in &anchors {
                deadline(started, limit)?;
                let mut values = x.chunks_exact(width).map(|r| r[*start]).collect::<Vec<_>>();
                values.sort_by(f32::total_cmp);
                let threshold = values[(values.len() - 1) / 2];
                let mut counts = [
                    vec![0usize; feature.probabilities.len()],
                    vec![0usize; feature.probabilities.len()],
                ];
                for (row, class) in x.chunks_exact(width).zip(&classes) {
                    counts[usize::from(row[*start] > threshold)][*class] += 1;
                }
                let n0 = counts[0].iter().sum::<usize>();
                let n1 = counts[1].iter().sum::<usize>();
                if n0 < 4 || n1 < 4 {
                    continue;
                }
                let loss = (n0 as f64 * entropy(&counts[0]) + n1 as f64 * entropy(&counts[1]))
                    / rows.len() as f64;
                if best.as_ref().is_none_or(|(previous, _)| loss < *previous) {
                    best = Some((
                        loss,
                        Split {
                            anchor: *anchor,
                            threshold,
                            left: probabilities(&counts[0]),
                            right: probabilities(&counts[1]),
                        },
                    ));
                }
            }
            feature.split = Some(
                best.ok_or_else(|| unavailable("mixed_conditional_anchor_unavailable"))?
                    .1,
            );
        }
    }
    let state = State {
        format: "dope-mixed-state".into(),
        version: 1,
        profile,
        constant_target: y.iter().all(|value| *value == 0.0),
        affine,
        features,
    };
    state.validate()?;
    Ok((state, x, y))
}

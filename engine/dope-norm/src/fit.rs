use crate::card::{CardData, FeatureState, FittedFeature, NumericState, TargetState};
use crate::{
    digest, Cell, DeclaredSchema, FeatureKind, GroupKind, GroupSpec, InputTable, NormError,
    NormOptions, Result, ScaleCard, TargetKind, Task, UnitSchema, MAX_UNIT_COLUMNS, RECIPE,
};
use std::collections::{BTreeMap, BTreeSet};

impl ScaleCard {
    /// Fit exclusively on explicit physical FIT rows, reduced in sorted physical-ID order.
    pub fn fit(
        table: &InputTable,
        declared: &DeclaredSchema,
        fit_rows: &[u64],
        options: NormOptions,
    ) -> Result<Self> {
        table.validate()?;
        if options.min_level_count != 1 || options.max_levels == 0 || options.max_levels > 255 {
            return Err(NormError::new("invalid_norm_options"));
        }
        if declared.features.is_empty()
            || declared.features.len() > MAX_UNIT_COLUMNS
            || declared.target.column >= table.columns.len()
        {
            return Err(NormError::new("invalid_declared_schema"));
        }
        let positions = row_positions(table, fit_rows, true)?;
        let mut used = BTreeSet::from([declared.target.column]);
        let mut features = Vec::new();
        let mut groups = Vec::new();
        let mut columns = 0usize;
        for spec in &declared.features {
            let col = spec.column;
            if col >= table.columns.len() || !used.insert(col) {
                return Err(NormError::at("invalid_feature_role", col));
            }
            let values: Vec<_> = positions
                .iter()
                .map(|&i| &table.columns[col].values[i])
                .collect();
            let has_missing = values.iter().any(|v| matches!(v, Cell::Missing));
            let (state, kind, width) = match &spec.kind {
                FeatureKind::Continuous | FeatureKind::Integer => {
                    let integer = spec.kind == FeatureKind::Integer;
                    let numeric = numeric_state(&values, col, integer, false)?;
                    (
                        FeatureState::Numeric { numeric },
                        if integer {
                            GroupKind::Integer
                        } else {
                            GroupKind::Numeric
                        },
                        1 + usize::from(has_missing),
                    )
                }
                FeatureKind::Boolean => {
                    if values
                        .iter()
                        .any(|v| !matches!(v, Cell::Boolean(_) | Cell::Missing))
                    {
                        return Err(NormError::at("invalid_boolean", col));
                    }
                    (
                        FeatureState::Boolean,
                        GroupKind::Boolean,
                        1 + usize::from(has_missing),
                    )
                }
                FeatureKind::Ordinal { levels } => {
                    validate_levels(levels, 4096, col)?;
                    if levels.len() < 2 {
                        return Err(NormError::at("invalid_ordinal", col));
                    }
                    let mut encoded = Vec::new();
                    for value in &values {
                        match value {
                            Cell::Missing => {}
                            Cell::Category(label) => {
                                let rank = levels
                                    .iter()
                                    .position(|v| v == label)
                                    .ok_or(NormError::at("unknown_ordinal", col))?;
                                encoded.push(rank as f64 / (levels.len() - 1) as f64);
                            }
                            _ => return Err(NormError::at("invalid_ordinal", col)),
                        }
                    }
                    encoded.sort_by(f64::total_cmp);
                    let median = median(&encoded);
                    (
                        FeatureState::Ordinal {
                            levels: levels.clone(),
                            median,
                        },
                        GroupKind::Ordinal,
                        1 + usize::from(has_missing),
                    )
                }
                FeatureKind::Nominal => {
                    let mut counts = BTreeMap::<String, usize>::new();
                    for value in &values {
                        match value {
                            Cell::Missing => {}
                            Cell::Category(label) => {
                                *counts.entry(label.clone()).or_default() += 1;
                            }
                            _ => return Err(NormError::at("invalid_nominal", col)),
                        }
                    }
                    if counts.len() > options.max_levels {
                        return Err(NormError::at("categorical_width_limit", col));
                    }
                    let mut ranked: Vec<_> = counts.into_iter().collect();
                    ranked.sort_by(|(a, an), (b, bn)| {
                        bn.cmp(an).then_with(|| a.as_bytes().cmp(b.as_bytes()))
                    });
                    let levels: Vec<_> = ranked.into_iter().map(|(s, _)| s).collect();
                    let width = levels.len() + 1 + usize::from(has_missing);
                    (FeatureState::Nominal { levels }, GroupKind::Nominal, width)
                }
            };
            let next = columns
                .checked_add(width)
                .ok_or(NormError::new("shape_overflow"))?;
            if next > MAX_UNIT_COLUMNS {
                return Err(NormError::new("expanded_width_limit"));
            }
            groups.push(GroupSpec {
                id: groups.len(),
                kind,
                start: columns,
                width,
                has_missing,
            });
            columns = next;
            features.push(FittedFeature {
                column: col,
                name: table.columns[col].name.clone(),
                has_missing,
                state,
            });
        }
        let target_col = declared.target.column;
        let target_values: Vec<_> = positions
            .iter()
            .map(|&i| &table.columns[target_col].values[i])
            .collect();
        let (target, target_kind, classes) = match &declared.target.task {
            Task::Regression => (
                TargetState::Regression {
                    numeric: numeric_state(&target_values, target_col, false, true)?,
                },
                TargetKind::Regression,
                0,
            ),
            Task::Binary { positive_label } => {
                let mut labels = BTreeSet::new();
                for value in target_values {
                    let Cell::Category(label) = value else {
                        return Err(NormError::at("invalid_target", target_col));
                    };
                    labels.insert(label.clone());
                }
                if labels.len() != 2 || !labels.contains(positive_label) {
                    return Err(NormError::at("invalid_binary_target", target_col));
                }
                let negative = labels
                    .into_iter()
                    .find(|s| s != positive_label)
                    .ok_or(NormError::at("invalid_binary_target", target_col))?;
                (
                    TargetState::Classification {
                        labels: vec![negative, positive_label.clone()],
                    },
                    TargetKind::Binary,
                    2,
                )
            }
            Task::Multiclass { labels } => {
                validate_levels(labels, 256, target_col)?;
                if labels.len() < 2 {
                    return Err(NormError::at("invalid_class_vocabulary", target_col));
                }
                for value in target_values {
                    let Cell::Category(label) = value else {
                        return Err(NormError::at("invalid_target", target_col));
                    };
                    if !labels.contains(label) {
                        return Err(NormError::at("unknown_target_class", target_col));
                    }
                }
                (
                    TargetState::Classification {
                        labels: labels.clone(),
                    },
                    TargetKind::Multiclass,
                    labels.len(),
                )
            }
        };
        let fit_class_counts=match &target {
            TargetState::Regression { .. }=>Vec::new(),
            TargetState::Classification { labels }=>labels.iter().map(|label| positions.iter()
                .filter(|&&i| matches!(&table.columns[target_col].values[i], Cell::Category(v) if v==label)).count()).collect(),
        };
        let public = UnitSchema {
            schema: "dope.nc1-public/v1".into(),
            recipe: RECIPE.into(),
            columns,
            groups,
            target: target_kind,
            classes,
        };
        public.validate()?;
        let canonical_ids: Vec<_> = positions.iter().map(|&i| table.row_ids[i]).collect();
        // FIT payload only: query values/labels never enter this digest.
        let fit_values: Vec<Vec<_>> = positions
            .iter()
            .map(|&i| {
                declared
                    .features
                    .iter()
                    .map(|s| table.columns[s.column].values[i].clone())
                    .chain(std::iter::once(table.columns[target_col].values[i].clone()))
                    .collect()
            })
            .collect();
        let fit_scope_sha256 = digest(&(
            RECIPE,
            &table.source_sha256,
            &canonical_ids,
            &fit_values,
            declared,
            options,
        ))?;
        let mut data = CardData {
            schema: "dope.nc1-private/v1".into(),
            recipe: RECIPE.into(),
            source_sha256: table.source_sha256.clone(),
            fit_scope_sha256,
            transform_sha256: String::new(),
            options,
            declared: declared.clone(),
            names: table.columns.iter().map(|c| c.name.clone()).collect(),
            features,
            target,
            fit_class_counts,
            public,
        };
        data.transform_sha256 = digest(&data)?;
        let out = Self { data };
        out.validate()?;
        Ok(out)
    }
}
pub(crate) fn row_positions(
    table: &InputTable,
    rows: &[u64],
    canonical: bool,
) -> Result<Vec<usize>> {
    if rows.is_empty() || rows.len() > crate::MAX_ROWS {
        return Err(NormError::new("invalid_row_selection"));
    }
    let lookup: BTreeMap<_, _> = table
        .row_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();
    let mut seen = BTreeSet::new();
    let mut selected = Vec::with_capacity(rows.len());
    for id in rows {
        if !seen.insert(*id) {
            return Err(NormError::new("duplicate_selected_row"));
        }
        selected.push(*lookup.get(id).ok_or(NormError::new("unknown_row_id"))?);
    }
    if canonical {
        selected.sort_by_key(|&i| table.row_ids[i]);
    }
    Ok(selected)
}
pub(crate) fn normalize(value: f64, n: &NumericState) -> Result<f64> {
    if !value.is_finite() {
        return Err(NormError::new("nonfinite_numeric"));
    }
    if n.hi == n.lo {
        return Ok(0.0);
    }
    if value == n.lo {
        return Ok(0.0);
    }
    if value == n.hi {
        return Ok(1.0);
    }
    let span = n.hi - n.lo;
    let difference = value - n.lo;
    let u = if span.is_finite() && difference.is_finite() {
        difference / span
    } else {
        let scale = n.lo.abs().max(n.hi.abs()).max(value.abs());
        ((value / scale) - (n.lo / scale)) / ((n.hi / scale) - (n.lo / scale))
    };
    if u.is_finite() {
        Ok(u)
    } else {
        Err(NormError::new("normalization_overflow"))
    }
}
fn numeric_state(
    values: &[&Cell],
    col: usize,
    integer: bool,
    target: bool,
) -> Result<NumericState> {
    let mut finite = Vec::new();
    for value in values {
        match value {
            Cell::Number(x) if x.is_finite() && (!integer || x.fract() == 0.0) => finite.push(*x),
            Cell::Missing if !target => {}
            _ => return Err(NormError::at("invalid_numeric", col)),
        }
    }
    finite.sort_by(f64::total_cmp);
    if target && (finite.is_empty() || finite.first() == finite.last()) {
        return Err(NormError::at("constant_or_missing_target", col));
    }
    let mut n = NumericState {
        lo: finite.first().copied().unwrap_or(0.0),
        hi: finite.last().copied().unwrap_or(0.0),
        median: 0.0,
        all_missing: finite.is_empty(),
    };
    let normalized: Vec<_> = finite
        .iter()
        .map(|&v| normalize(v, &n))
        .collect::<Result<_>>()?;
    n.median = median(&normalized);
    Ok(n)
}
fn median(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        values[n / 2]
    } else {
        values[n / 2 - 1] * 0.5 + values[n / 2] * 0.5
    }
}
pub(crate) fn validate_levels(levels: &[String], cap: usize, col: usize) -> Result<()> {
    if levels.is_empty()
        || levels.len() > cap
        || levels.iter().any(|v| v.len() > 65536)
        || levels.iter().collect::<BTreeSet<_>>().len() != levels.len()
    {
        Err(NormError::at("invalid_declared_levels", col))
    } else {
        Ok(())
    }
}

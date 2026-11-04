use crate::{
    digest, is_digest, DeclaredSchema, NormError, NormOptions, Result, UnitSchema, MAX_JSON_BYTES,
    MAX_ROWS, RECIPE,
};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NumericState {
    pub lo: f64,
    pub hi: f64,
    pub median: f64,
    pub all_missing: bool,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum FeatureState {
    Numeric { numeric: NumericState },
    Boolean,
    Ordinal { levels: Vec<String>, median: f64 },
    Nominal { levels: Vec<String> },
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FittedFeature {
    pub column: usize,
    pub name: String,
    pub has_missing: bool,
    pub state: FeatureState,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum TargetState {
    Regression { numeric: NumericState },
    Classification { labels: Vec<String> },
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CardData {
    pub schema: String,
    pub recipe: String,
    pub source_sha256: String,
    pub fit_scope_sha256: String,
    pub transform_sha256: String,
    pub options: NormOptions,
    pub declared: DeclaredSchema,
    pub names: Vec<String>,
    pub features: Vec<FittedFeature>,
    pub target: TargetState,
    pub fit_class_counts: Vec<usize>,
    pub public: UnitSchema,
}
/// Private fitted owner state. Serialization is explicitly named private.
#[derive(Clone)]
pub struct ScaleCard {
    pub(crate) data: CardData,
}
impl fmt::Debug for ScaleCard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ScaleCard(<private>)")
    }
}
impl ScaleCard {
    /// Public structural projection only.
    pub fn public_schema(&self) -> &UnitSchema {
        &self.data.public
    }
    pub fn source_sha256(&self) -> &str {
        &self.data.source_sha256
    }
    pub fn fit_scope_sha256(&self) -> &str {
        &self.data.fit_scope_sha256
    }
    pub fn transform_sha256(&self) -> &str {
        &self.data.transform_sha256
    }
    /// Explicit private export. Never attach this to a public packet or receipt.
    pub fn to_private_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let bytes =
            serde_json::to_vec(&self.data).map_err(|_| NormError::new("serialization_failed"))?;
        if bytes.len() > MAX_JSON_BYTES {
            return Err(NormError::new("json_size_limit"));
        }
        Ok(bytes)
    }
    /// Validate bounded private state and its content-derived transform identity.
    pub fn from_private_json(bytes: &[u8]) -> Result<Self> {
        let out = Self {
            data: crate::parse_json(bytes)?,
        };
        out.validate()?;
        Ok(out)
    }
    pub(crate) fn validate(&self) -> Result<()> {
        let d = &self.data;
        d.public.validate()?;
        if d.schema != "dope.nc1-private/v1"
            || d.recipe != RECIPE
            || !is_digest(&d.source_sha256)
            || !is_digest(&d.fit_scope_sha256)
            || !is_digest(&d.transform_sha256)
            || d.features.len() != d.public.groups.len()
            || d.features.len() != d.declared.features.len()
            || d.names.is_empty()
            || d.names.len() > crate::MAX_SOURCE_COLUMNS
            || d.declared.target.column >= d.names.len()
            || d.options.min_level_count != 1
            || d.options.max_levels == 0
            || d.options.max_levels > 255
        {
            return Err(NormError::new("invalid_private_card"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for (i, feat) in d.features.iter().enumerate() {
            let group = &d.public.groups[i];
            if feat.column >= d.names.len()
                || !seen.insert(feat.column)
                || feat.column == d.declared.target.column
                || feat.name != d.names[feat.column]
                || feat.column != d.declared.features[i].column
                || feat.has_missing != group.has_missing
            {
                return Err(NormError::at("invalid_private_card", i));
            }
            let semantics_match = matches!(
                (&d.declared.features[i].kind, &feat.state, &group.kind),
                (
                    crate::FeatureKind::Continuous,
                    FeatureState::Numeric { .. },
                    crate::GroupKind::Numeric
                ) | (
                    crate::FeatureKind::Integer,
                    FeatureState::Numeric { .. },
                    crate::GroupKind::Integer
                ) | (
                    crate::FeatureKind::Boolean,
                    FeatureState::Boolean,
                    crate::GroupKind::Boolean
                ) | (
                    crate::FeatureKind::Ordinal { .. },
                    FeatureState::Ordinal { .. },
                    crate::GroupKind::Ordinal
                ) | (
                    crate::FeatureKind::Nominal,
                    FeatureState::Nominal { .. },
                    crate::GroupKind::Nominal
                )
            );
            if !semantics_match {
                return Err(NormError::at("invalid_private_card", i));
            }
            match &feat.state {
                FeatureState::Numeric { numeric: n } => {
                    if !n.lo.is_finite()
                        || !n.hi.is_finite()
                        || !n.median.is_finite()
                        || n.lo > n.hi
                        || !(0.0..=1.0).contains(&n.median)
                    {
                        return Err(NormError::at("invalid_private_card", i));
                    }
                }
                FeatureState::Ordinal { levels, median } => {
                    if levels.len() < 2
                        || levels.len() > 4096
                        || !median.is_finite()
                        || !(0.0..=1.0).contains(median)
                        || unique(levels) != levels.len()
                    {
                        return Err(NormError::at("invalid_private_card", i));
                    }
                }
                FeatureState::Nominal { levels } => {
                    if levels.len() > d.options.max_levels
                        || unique(levels) != levels.len()
                        || group.width != levels.len() + 1 + usize::from(feat.has_missing)
                    {
                        return Err(NormError::at("invalid_private_card", i));
                    }
                }
                FeatureState::Boolean => {}
            }
        }
        match &d.target {
            TargetState::Regression { numeric: n } => {
                if !n.lo.is_finite()
                    || !n.hi.is_finite()
                    || n.lo >= n.hi
                    || d.public.target != crate::TargetKind::Regression
                    || !d.fit_class_counts.is_empty()
                {
                    return Err(NormError::new("invalid_private_card"));
                }
            }
            TargetState::Classification { labels } => {
                if labels.len() != d.public.classes
                    || unique(labels) != labels.len()
                    || labels.len() < 2
                    || labels.len() > 256
                    || d.fit_class_counts.len() != labels.len()
                {
                    return Err(NormError::new("invalid_private_card"));
                }
            }
        }
        match (&d.declared.target.task, &d.target) {
            (crate::Task::Regression, TargetState::Regression { .. }) => {}
            (crate::Task::Binary { positive_label }, TargetState::Classification { labels })
                if labels.len() == 2
                    && &labels[1] == positive_label
                    && d.public.target == crate::TargetKind::Binary => {}
            (
                crate::Task::Multiclass { labels: declared },
                TargetState::Classification { labels },
            ) if labels == declared && d.public.target == crate::TargetKind::Multiclass => {}
            _ => return Err(NormError::new("invalid_private_card")),
        }
        let mut unsigned = d.clone();
        unsigned.transform_sha256.clear();
        if digest(&unsigned)? != d.transform_sha256 {
            return Err(NormError::new("card_digest_mismatch"));
        }
        Ok(())
    }
    /// Invert a regression prediction in owner custody. f64 affine; no clipping.
    pub fn inverse_target(&self, unit: f64) -> Result<f64> {
        if !unit.is_finite() {
            return Err(NormError::new("nonfinite_prediction"));
        }
        let TargetState::Regression { numeric: n } = &self.data.target else {
            return Err(NormError::new("target_not_regression"));
        };
        let span = n.hi - n.lo;
        let value = if span.is_finite() {
            n.lo + unit * span
        } else {
            (1.0 - unit) * n.lo + unit * n.hi
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err(NormError::new("inverse_overflow"))
        }
    }
}
fn unique(values: &[String]) -> usize {
    values
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}

/// Aggregate, header-free apply diagnostics. Treat real-data receipts as private.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApplyReceipt {
    pub rows: usize,
    pub source_sha256: String,
    pub fit_source_sha256: String,
    pub unobserved_fit_classes: Vec<usize>,
    pub out_of_range: Vec<usize>,
    pub unseen_levels: Vec<usize>,
    pub unexpected_missing: Vec<usize>,
    pub schema_sha256: String,
    pub fit_scope_sha256: String,
    pub transform_sha256: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UnitData {
    pub schema: UnitSchema,
    pub source_sha256: String,
    pub fit_scope_sha256: String,
    pub transform_sha256: String,
    pub row_ids: Vec<u64>,
    pub x: Vec<Vec<f64>>,
    pub y: Option<Vec<f64>>,
}
/// Private normalized real rows. Not Serialize: exports require explicit private call.
#[derive(Clone)]
pub struct UnitTable {
    pub(crate) data: UnitData,
}
impl fmt::Debug for UnitTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("UnitTable(<private rows>)")
    }
}
impl UnitTable {
    pub fn x(&self) -> &[Vec<f64>] {
        &self.data.x
    }
    pub fn y(&self) -> Option<&[f64]> {
        self.data.y.as_deref()
    }
    pub fn row_ids(&self) -> &[u64] {
        &self.data.row_ids
    }
    pub fn schema(&self) -> &UnitSchema {
        &self.data.schema
    }
    pub fn fit_scope_sha256(&self) -> &str {
        &self.data.fit_scope_sha256
    }
    pub fn transform_sha256(&self) -> &str {
        &self.data.transform_sha256
    }
    pub fn source_sha256(&self) -> &str {
        &self.data.source_sha256
    }
    /// Content identity of private matrix, not permission to publish its rows.
    pub fn sha256(&self) -> Result<String> {
        self.validate()?;
        digest(&self.data)
    }
    pub fn to_private_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let bytes =
            serde_json::to_vec(&self.data).map_err(|_| NormError::new("serialization_failed"))?;
        if bytes.len() > MAX_JSON_BYTES {
            return Err(NormError::new("json_size_limit"));
        }
        Ok(bytes)
    }
    pub fn from_private_json(bytes: &[u8]) -> Result<Self> {
        let out = Self {
            data: crate::parse_json(bytes)?,
        };
        out.validate()?;
        Ok(out)
    }
    pub(crate) fn validate(&self) -> Result<()> {
        let d = &self.data;
        d.schema.validate()?;
        if !is_digest(&d.source_sha256)
            || !is_digest(&d.fit_scope_sha256)
            || !is_digest(&d.transform_sha256)
            || d.x.is_empty()
            || d.x.len() > MAX_ROWS
            || d.row_ids.len() != d.x.len()
            || d.x
                .len()
                .checked_mul(d.schema.columns)
                .is_none_or(|n| n > crate::MAX_CELLS)
            || d.row_ids
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != d.row_ids.len()
            || d.y
                .as_ref()
                .is_some_and(|y| y.len() != d.x.len() || y.iter().any(|v| !v.is_finite()))
            || d.x
                .iter()
                .any(|row| row.len() != d.schema.columns || row.iter().any(|v| !v.is_finite()))
        {
            return Err(NormError::new("invalid_unit_table"));
        }
        if d.schema.target != crate::TargetKind::Regression
            && d.y.as_ref().is_some_and(|values| {
                values
                    .iter()
                    .any(|v| v.fract() != 0.0 || *v < 0.0 || *v >= d.schema.classes as f64)
            })
        {
            return Err(NormError::new("invalid_unit_target"));
        }
        for row in &d.x {
            for group in &d.schema.groups {
                let slice = &row[group.start..group.start + group.width];
                if group.has_missing
                    && group.kind != crate::GroupKind::Nominal
                    && slice.last().is_some_and(|v| *v != 0.0 && *v != 1.0)
                {
                    return Err(NormError::at("invalid_missing_indicator", group.id));
                }
                if group.kind == crate::GroupKind::Nominal
                    && (slice.iter().any(|v| *v != 0.0 && *v != 1.0)
                        || slice.iter().filter(|v| **v == 1.0).count() != 1)
                {
                    return Err(NormError::at("invalid_onehot", group.id));
                }
            }
        }
        Ok(())
    }
}

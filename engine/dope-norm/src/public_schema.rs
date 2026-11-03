use crate::{digest, NormError, Result, MAX_UNIT_COLUMNS, RECIPE};
use serde::{Deserialize, Serialize};

/// Public structural group kind; never contains labels or extrema.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupKind {
    Numeric,
    Integer,
    Boolean,
    Ordinal,
    Nominal,
}
/// Public structural group. Missing indicator is last within numeric groups.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupSpec {
    pub id: usize,
    pub kind: GroupKind,
    pub start: usize,
    pub width: usize,
    pub has_missing: bool,
}
/// Public target layout.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetKind {
    Regression,
    Binary,
    Multiclass,
}
/// Names, raw units, vocabulary and class labels are excluded by construction.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitSchema {
    pub schema: String,
    pub recipe: String,
    pub columns: usize,
    pub groups: Vec<GroupSpec>,
    pub target: TargetKind,
    pub classes: usize,
}
impl UnitSchema {
    /// Validate contiguous bounded grouping, exact recipe and target shape.
    pub fn validate(&self) -> Result<()> {
        if self.schema != "dope.nc1-public/v1"
            || self.recipe != RECIPE
            || self.columns == 0
            || self.columns > MAX_UNIT_COLUMNS
            || self.groups.is_empty()
            || self.groups.len() > MAX_UNIT_COLUMNS
        {
            return Err(NormError::new("invalid_public_schema"));
        }
        let mut start = 0usize;
        for (i, g) in self.groups.iter().enumerate() {
            let numeric_width = if g.has_missing { 2 } else { 1 };
            if g.id != i
                || g.start != start
                || g.width == 0
                || (g.kind != GroupKind::Nominal && g.width != numeric_width)
                || (g.kind == GroupKind::Nominal && g.width < 1 + usize::from(g.has_missing))
            {
                return Err(NormError::at("invalid_group", i));
            }
            start = start
                .checked_add(g.width)
                .ok_or(NormError::new("shape_overflow"))?;
        }
        if start != self.columns
            || match self.target {
                TargetKind::Regression => self.classes != 0,
                TargetKind::Binary => self.classes != 2,
                TargetKind::Multiclass => !(2..=256).contains(&self.classes),
            }
        {
            return Err(NormError::new("invalid_public_schema"));
        }
        Ok(())
    }
    /// Canonical structural schema identity.
    pub fn sha256(&self) -> Result<String> {
        self.validate()?;
        digest(self)
    }
    /// Parse a bounded structural projection, refusing unknown fields.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let out: Self = crate::parse_json(bytes)?;
        out.validate()?;
        Ok(out)
    }
    /// Serialize only public shape/kinds; private rows never appear.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|_| NormError::new("serialization_failed"))
    }
}

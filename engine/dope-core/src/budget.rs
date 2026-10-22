//! Size presets and a versioned count of the deployed generator's scalar state.
use crate::{copula::Dependence, marginal::Marginal, noise::Noise, schema::Kind, Program};
use serde::Serialize;
use std::str::FromStr;

/// Per-table compilation presets, not cross-table pretrained foundation models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSize {
    Small,
    Medium,
    Large,
}

impl ModelSize {
    pub fn budget(self) -> usize {
        match self {
            Self::Small => 2048,
            Self::Medium => 4096,
            Self::Large => 16384,
        }
    }
}

impl FromStr for ModelSize {
    type Err = &'static str;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            _ => Err("model size must be small, medium or large"),
        }
    }
}

/// Logical scalar slots of the decoded program (definition v1). This is not
/// degrees of freedom, RAM, or bytes. Quantizer scales reconstructing net
/// coefficients are not counted twice; forest scales are explicit state.
/// Names, labels, indices, flags and fixed matrices are charged by bytes only.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ParameterCount {
    pub version: u8,
    pub network: usize,
    pub forest: usize,
    pub marginals: usize,
    pub dependence: usize,
    pub regimes: usize,
    pub noise: usize,
    pub schema_and_missing: usize,
    pub total: usize,
}

impl Program {
    /// Count the complete generator, rather than only its conditional network.
    pub fn parameters(&self) -> ParameterCount {
        let network = self.net.parameter_count();
        let forest =
            self.forest.thresholds.len() + self.forest.codes.len() + self.forest.scales.len();
        let marginals = self
            .marginals
            .iter()
            .zip(&self.schema.features)
            .map(|(m, c)| match m {
                Marginal::Continuous { knots, atoms } => knots.len() + 3 * atoms.len(),
                Marginal::Discrete { values, probs } => {
                    probs.len().saturating_sub(1)
                        + if matches!(c.kind, Kind::Numeric(_)) {
                            values.len()
                        } else {
                            0
                        }
                }
            })
            .sum();
        let d = self.features();
        let dependence = match &self.dependence {
            Dependence::Independent => 0,
            Dependence::Tree { parent, .. } => parent.iter().filter(|p| p.is_some()).count(),
            Dependence::Factor { loadings, .. } => loadings.len(),
            Dependence::Full { .. } => d * d.saturating_sub(1) / 2,
        };
        let regimes = if self.regimes.is_trivial() {
            0
        } else {
            self.regimes.weights.len() + self.regimes.means.len() + self.regimes.scales.len()
        };
        let noise = match &self.noise {
            Noise::Gaussianish { coef, knots, .. } => coef.len() + knots.len() + 2,
            Noise::Temperature(_) => 1,
        };
        let schema_and_missing = self.missing.len()
            + 2
            + 2 * self
                .schema
                .features
                .iter()
                .chain(std::iter::once(&self.schema.target))
                .filter(|c| matches!(c.kind, Kind::Numeric(_)))
                .count();
        ParameterCount {
            version: 1,
            network,
            forest,
            marginals,
            dependence,
            regimes,
            noise,
            schema_and_missing,
            total: network + forest + marginals + dependence + regimes + noise + schema_and_missing,
        }
    }
}

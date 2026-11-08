//! Additive DML4 reference contract. DML3 and released v3 remain unchanged.
//! Numeric models and group indices only; no owner ScaleCard or source rows.
#![forbid(unsafe_code)]
pub mod basis;
pub mod rd_pack;
pub mod sample;
pub mod wire;

pub use basis::GateValues;
use dope_norm::{GroupKind, TargetKind, UnitSchema};
pub use sample::{sample, sample_with_draws, SyntheticTable};
use serde::{Deserialize, Serialize};
use std::fmt;
pub use wire::{decode, encode, Level};

/// Safe DML4 error; never stores source values or parser text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WireError {
    pub code: &'static str,
}
impl WireError {
    pub(crate) fn new(code: &'static str) -> Self {
        Self { code }
    }
}
impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code)
    }
}
impl std::error::Error for WireError {}
pub type Result<T> = std::result::Result<T, WireError>;

/// Independent hash domains. Weights identify declared shared sampler closure.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PacketIdentity {
    pub weights_sha256: String,
    pub encoder_recipe_sha256: String,
    pub conditioning_sha256: String,
    pub fit_scope_sha256: String,
}
/// Unit-space marginal distributions. Position-only structure; no vocabulary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GroupModel {
    Numeric {
        knots: Vec<f64>,
        missing_probability: f64,
        imputation: f64,
    },
    Discrete {
        values: Vec<f64>,
        probabilities: Vec<f64>,
        missing_probability: f64,
        imputation: f64,
    },
    Categorical {
        probabilities: Vec<f64>,
    },
}
/// Every fitted basis coordinate and target parameter is carried on wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "task", rename_all = "snake_case", deny_unknown_fields)]
pub enum TargetModel {
    Regression {
        intercept: f64,
        coefficients: Vec<f64>,
        noise_scale: f64,
    },
    Classification {
        intercepts: Vec<f64>,
        coefficients: Vec<Vec<f64>>,
    },
}
/// Fully self-describing reference packet body. Never accepts arbitrary strings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Packet {
    pub contract: String,
    pub level: Level,
    pub schema: UnitSchema,
    pub schema_sha256: String,
    pub identity: PacketIdentity,
    pub gates: GateValues,
    pub groups: Vec<GroupModel>,
    pub target: TargetModel,
}
impl Packet {
    /// Validate bounded shape and every active graph parameter before encode/sample.
    pub fn validate(&self) -> Result<()> {
        self.schema
            .validate()
            .map_err(|_| WireError::new("invalid_public_schema"))?;
        if self.contract != "dope.dml4-reference/v1"
            || self
                .schema
                .sha256()
                .map_err(|_| WireError::new("invalid_public_schema"))?
                != self.schema_sha256
            || self.groups.len() != self.schema.groups.len()
            || ![
                &self.identity.weights_sha256,
                &self.identity.encoder_recipe_sha256,
                &self.identity.conditioning_sha256,
                &self.identity.fit_scope_sha256,
            ]
            .iter()
            .all(|v| digest_valid(v))
        {
            return Err(WireError::new("invalid_packet_identity"));
        }
        self.gates.validate()?;
        for (group, model) in self.schema.groups.iter().zip(&self.groups) {
            match (model, &group.kind) {
                (
                    GroupModel::Numeric {
                        knots,
                        missing_probability,
                        imputation,
                    },
                    GroupKind::Numeric,
                ) => {
                    if !(2..=65).contains(&knots.len())
                        || !knots.iter().all(|v| unit(*v))
                        || knots.windows(2).any(|v| v[0] > v[1])
                        || !unit(*imputation)
                        || !missing_valid(*missing_probability, group.has_missing)
                    {
                        return Err(WireError::new("invalid_numeric_model"));
                    }
                }
                (
                    GroupModel::Discrete {
                        values,
                        probabilities,
                        missing_probability,
                        imputation,
                    },
                    kind,
                ) if matches!(
                    kind,
                    GroupKind::Boolean | GroupKind::Integer | GroupKind::Ordinal
                ) =>
                {
                    if values.is_empty()
                        || values.len() > 256
                        || values.len() != probabilities.len()
                        || !values.iter().all(|v| unit(*v))
                        || values.windows(2).any(|v| v[0] >= v[1])
                        || !probabilities_valid(probabilities)
                        || !unit(*imputation)
                        || !missing_valid(*missing_probability, group.has_missing)
                        || (*kind == GroupKind::Boolean
                            && (values.iter().any(|v| *v != 0.0 && *v != 1.0)
                                || (*imputation != 0.0 && *imputation != 1.0)))
                    {
                        return Err(WireError::new("invalid_discrete_model"));
                    }
                }
                (GroupModel::Categorical { probabilities }, GroupKind::Nominal) => {
                    if probabilities.len() != group.width || !probabilities_valid(probabilities) {
                        return Err(WireError::new("invalid_categorical_model"));
                    }
                }
                _ => return Err(WireError::new("group_model_mismatch")),
            }
        }
        let width = basis::width(self.schema.columns)?;
        match (&self.target, &self.schema.target) {
            (
                TargetModel::Regression {
                    intercept,
                    coefficients,
                    noise_scale,
                },
                TargetKind::Regression,
            ) => {
                if !parameter(*intercept)
                    || coefficients.len() != width
                    || !coefficients.iter().all(|v| parameter(*v))
                    || !noise_scale.is_finite()
                    || *noise_scale < 0.0
                    || *noise_scale > 1e6
                {
                    return Err(WireError::new("invalid_target_model"));
                }
            }
            (
                TargetModel::Classification {
                    intercepts,
                    coefficients,
                },
                TargetKind::Binary | TargetKind::Multiclass,
            ) => {
                if intercepts.len() != self.schema.classes
                    || coefficients.len() != self.schema.classes
                    || !intercepts.iter().all(|v| parameter(*v))
                    || coefficients
                        .iter()
                        .any(|w| w.len() != width || !w.iter().all(|v| parameter(*v)))
                {
                    return Err(WireError::new("invalid_target_model"));
                }
            }
            _ => return Err(WireError::new("target_model_mismatch")),
        }
        Ok(())
    }
}
pub(crate) fn unit(v: f64) -> bool {
    v.is_finite() && (0.0..=1.0).contains(&v)
}
fn parameter(v: f64) -> bool {
    v.is_finite() && v.abs() <= 1e6
}
fn digest_valid(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn missing_valid(v: f64, has: bool) -> bool {
    unit(v) && (has || v == 0.0)
}
pub(crate) fn probabilities_valid(p: &[f64]) -> bool {
    !p.is_empty() && p.len() <= 256 && p.iter().all(|v| v.is_finite() && *v >= 0.0) && {
        let sum: f64 = p.iter().sum();
        sum.is_finite() && sum > 0.0
    }
}

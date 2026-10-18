//! Independent release axes. Receipt identities require a trusted verifier;
//! serialized state names and byte counts never grant release authority.
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{DopeError, Result};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    Utility,
    EmpiricalPrivacy,
    FormalDp,
    LegalRelease,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UtilityQualificationState {
    #[default]
    Unassessed,
    Pending,
    Passed,
    Failed,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmpiricalPrivacyState {
    #[default]
    Unassessed,
    Pending,
    Screened,
    Inconclusive,
    Failed,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormalDpState {
    #[default]
    Unassessed,
    Pending,
    Verified,
    Failed,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegalReleaseState {
    #[default]
    Unassessed,
    Pending,
    Authorized,
    Denied,
}

/// Commitments cover every independently source-derived observable, including
/// failed statuses, lengths, metrics and earlier/repeated artifacts. Even an
/// empty channel needs an explicit commitment. These are not custody proofs.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseAccounting {
    pub artifact_inventory_sha256: String,
    pub release_history_sha256: String,
    pub source_derived_lengths_sha256: String,
    pub source_derived_statuses_sha256: String,
    pub source_derived_metrics_sha256: String,
    pub release_count: u32,
}

impl ReleaseAccounting {
    pub fn identity(&self) -> Result<String> {
        let value = serde_json::to_value(self)?;
        check(
            (1..=100_000).contains(&self.release_count)
                && value.as_object().unwrap().iter().all(|(key, value)| {
                    key == "release_count" || value.as_str().is_some_and(is_pin)
                }),
        )?;
        Ok(format!(
            "{:x}",
            Sha256::digest(crate::evidence::canonical_json(&value)?)
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "axis", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvidenceIdentity {
    Utility {
        receipt_sha256: String,
        accounting_sha256: String,
    },
    EmpiricalPrivacy {
        receipt_sha256: String,
        accounting_sha256: String,
        threat_model_sha256: String,
        attack_power_sha256: String,
    },
    FormalDp {
        receipt_sha256: String,
        accounting_sha256: String,
        mechanism_sha256: String,
        accountant_sha256: String,
        proof_sha256: String,
        epsilon: f64,
        delta: f64,
    },
    LegalRelease {
        receipt_sha256: String,
        accounting_sha256: String,
        rights_owner_sha256: String,
    },
}

impl EvidenceIdentity {
    pub fn axis(&self) -> Axis {
        match self {
            Self::Utility { .. } => Axis::Utility,
            Self::EmpiricalPrivacy { .. } => Axis::EmpiricalPrivacy,
            Self::FormalDp { .. } => Axis::FormalDp,
            Self::LegalRelease { .. } => Axis::LegalRelease,
        }
    }
}

/// Implement only in the independently trusted evidence/rights-owner boundary.
/// Verification must resolve the exact receipt and all its identities, scope,
/// coverage, power, composition and authority. A hash or caller flag is no proof.
pub trait ReleaseEvidenceVerifier {
    fn verify(&self, evidence: &EvidenceIdentity) -> bool;
}

pub struct NoReleaseEvidence;
impl ReleaseEvidenceVerifier for NoReleaseEvidence {
    fn verify(&self, _: &EvidenceIdentity) -> bool {
        false
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseProfile {
    format: String,
    version: u8,
    utility_qualification_state: UtilityQualificationState,
    empirical_privacy_state: EmpiricalPrivacyState,
    formal_dp_state: FormalDpState,
    legal_release_state: LegalReleaseState,
    #[serde(deserialize_with = "deserialize_accounting")]
    accounting: Option<ReleaseAccounting>,
    evidence: Vec<EvidenceIdentity>,
}

impl Default for ReleaseProfile {
    fn default() -> Self {
        Self {
            format: "dope-release-profile".into(),
            version: 1,
            utility_qualification_state: UtilityQualificationState::Unassessed,
            empirical_privacy_state: EmpiricalPrivacyState::Unassessed,
            formal_dp_state: FormalDpState::Unassessed,
            legal_release_state: LegalReleaseState::Unassessed,
            accounting: None,
            evidence: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseSurface {
    Cli,
    Service,
    Ui,
    Report,
    Export,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReleaseView {
    pub profile: ReleaseProfile,
    pub release_authorized: bool,
    pub label: &'static str,
}

impl ReleaseProfile {
    /// A changed inventory, observable channel or history invalidates all axes.
    pub fn with_accounting(accounting: ReleaseAccounting) -> Result<Self> {
        accounting.identity()?;
        Ok(Self {
            accounting: Some(accounting),
            ..Self::default()
        })
    }

    pub fn decode(bytes: &[u8], verifier: &dyn ReleaseEvidenceVerifier) -> Result<Self> {
        check(bytes.len() <= 65_536)?;
        let value = crate::evidence::parse_json(bytes).map_err(|_| invalid())?;
        let profile: Self = serde_json::from_value(value).map_err(|_| invalid())?;
        profile.validate(verifier)?;
        Ok(profile)
    }

    pub fn begin(&self, axis: Axis) -> Self {
        let mut next = self.clone();
        next.evidence.retain(|evidence| evidence.axis() != axis);
        match axis {
            Axis::Utility => next.utility_qualification_state = UtilityQualificationState::Pending,
            Axis::EmpiricalPrivacy => next.empirical_privacy_state = EmpiricalPrivacyState::Pending,
            Axis::FormalDp => next.formal_dp_state = FormalDpState::Pending,
            Axis::LegalRelease => next.legal_release_state = LegalReleaseState::Pending,
        }
        next
    }

    pub fn complete(
        &self,
        evidence: EvidenceIdentity,
        verifier: &dyn ReleaseEvidenceVerifier,
    ) -> Result<Self> {
        check(self.pending(evidence.axis()))?;
        let mut next = self.clone();
        match evidence.axis() {
            Axis::Utility => next.utility_qualification_state = UtilityQualificationState::Passed,
            Axis::EmpiricalPrivacy => {
                next.empirical_privacy_state = EmpiricalPrivacyState::Screened
            }
            Axis::FormalDp => next.formal_dp_state = FormalDpState::Verified,
            Axis::LegalRelease => next.legal_release_state = LegalReleaseState::Authorized,
        }
        next.evidence.push(evidence);
        next.validate(verifier)?;
        Ok(next)
    }

    pub fn reject(&self, axis: Axis) -> Result<Self> {
        check(self.pending(axis))?;
        let mut next = self.clone();
        match axis {
            Axis::Utility => next.utility_qualification_state = UtilityQualificationState::Failed,
            Axis::EmpiricalPrivacy => next.empirical_privacy_state = EmpiricalPrivacyState::Failed,
            Axis::FormalDp => next.formal_dp_state = FormalDpState::Failed,
            Axis::LegalRelease => next.legal_release_state = LegalReleaseState::Denied,
        }
        Ok(next)
    }

    pub fn empirical_inconclusive(&self) -> Result<Self> {
        check(self.pending(Axis::EmpiricalPrivacy))?;
        let mut next = self.clone();
        next.empirical_privacy_state = EmpiricalPrivacyState::Inconclusive;
        Ok(next)
    }

    fn pending(&self, axis: Axis) -> bool {
        match axis {
            Axis::Utility => self.utility_qualification_state == UtilityQualificationState::Pending,
            Axis::EmpiricalPrivacy => {
                self.empirical_privacy_state == EmpiricalPrivacyState::Pending
            }
            Axis::FormalDp => self.formal_dp_state == FormalDpState::Pending,
            Axis::LegalRelease => self.legal_release_state == LegalReleaseState::Pending,
        }
    }

    fn positive_axes(&self) -> BTreeSet<Axis> {
        [
            (
                Axis::Utility,
                self.utility_qualification_state == UtilityQualificationState::Passed,
            ),
            (
                Axis::EmpiricalPrivacy,
                self.empirical_privacy_state == EmpiricalPrivacyState::Screened,
            ),
            (
                Axis::FormalDp,
                self.formal_dp_state == FormalDpState::Verified,
            ),
            (
                Axis::LegalRelease,
                self.legal_release_state == LegalReleaseState::Authorized,
            ),
        ]
        .into_iter()
        .filter_map(|(axis, positive)| positive.then_some(axis))
        .collect()
    }

    pub fn validate(&self, verifier: &dyn ReleaseEvidenceVerifier) -> Result<()> {
        check(self.format == "dope-release-profile" && self.version == 1)?;
        let scope = self.accounting.as_ref().map(|a| a.identity()).transpose()?;
        let mut axes = BTreeSet::new();
        for evidence in &self.evidence {
            let value = serde_json::to_value(evidence)?;
            check(axes.insert(evidence.axis()))?;
            check(value["accounting_sha256"].as_str() == scope.as_deref() && scope.is_some())?;
            check(value.as_object().unwrap().iter().all(|(key, value)| {
                matches!(key.as_str(), "axis" | "epsilon" | "delta")
                    || value.as_str().is_some_and(is_pin)
            }))?;
            if let EvidenceIdentity::FormalDp { epsilon, delta, .. } = evidence {
                check(
                    epsilon.is_finite()
                        && *epsilon > 0.
                        && delta.is_finite()
                        && (0.0..1.0).contains(delta),
                )?;
            }
            check(verifier.verify(evidence))?;
        }
        check(axes == self.positive_axes())
    }

    /// All presentation surfaces and export use the same verification and gate.
    /// Formal DP is optional unless requested; a failed/pending DP assessment
    /// still blocks export. Legal authorization is always independently required.
    pub fn view(
        &self,
        surface: ReleaseSurface,
        require_formal_dp: bool,
        verifier: &dyn ReleaseEvidenceVerifier,
    ) -> Result<ReleaseView> {
        self.validate(verifier)?;
        let authorized = self.utility_qualification_state == UtilityQualificationState::Passed
            && self.empirical_privacy_state == EmpiricalPrivacyState::Screened
            && self.legal_release_state == LegalReleaseState::Authorized
            && (self.formal_dp_state == FormalDpState::Verified
                || (!require_formal_dp && self.formal_dp_state == FormalDpState::Unassessed));
        if surface == ReleaseSurface::Export && !authorized {
            return Err(DopeError::Data("release_authorization_required".into()));
        }
        Ok(ReleaseView {
            profile: self.clone(),
            release_authorized: authorized,
            label: if authorized {
                "scoped_release"
            } else {
                "local_research"
            },
        })
    }

    pub fn check_claim(
        &self,
        label: &str,
        epsilon: Option<f64>,
        verifier: &dyn ReleaseEvidenceVerifier,
    ) -> Result<()> {
        let view = self.view(ReleaseSurface::Report, epsilon.is_some(), verifier)?;
        check(epsilon.is_none_or(|value| {
            value.is_finite() && self.evidence.iter().any(|evidence| {
                matches!(evidence, EvidenceIdentity::FormalDp { epsilon, .. } if *epsilon == value)
            })
        }))?;
        check(match label {
            "local_research" => true,
            "utility_qualified" => {
                self.utility_qualification_state == UtilityQualificationState::Passed
            }
            "empirical_screened" => self.empirical_privacy_state == EmpiricalPrivacyState::Screened,
            "formal_dp_verified" => self.formal_dp_state == FormalDpState::Verified,
            "organization_authorized" => self.legal_release_state == LegalReleaseState::Authorized,
            "scoped_release" => view.release_authorized,
            _ => false,
        })
    }
}

fn is_pin(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn invalid() -> DopeError {
    DopeError::Data("invalid_release_profile".into())
}

fn deserialize_accounting<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<ReleaseAccounting>, D::Error> {
    Option::deserialize(deserializer)
}

fn check(condition: bool) -> Result<()> {
    if condition { Ok(()) } else { Err(invalid()) }
}

#[cfg(test)]
#[path = "profile_tests.rs"]
mod tests;

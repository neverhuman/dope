//! Caller-declared DVS view roles with a deterministic structural binding.
//!
//! The digest detects malformed or inconsistent declarations. It does not
//! authenticate who supplied a hash or prove row origin/disjointness.
use super::{ensure, pin, DataPins, DvsError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const ROLE_SCHEMA: &str = "dope.dvs.declared-evidence-roles/v1";
pub const REAL_FIT_USAGE: &str = "fit_only";
pub const SYNTHETIC_VALIDATION_USAGE: &str = "model_choice";
pub const REAL_TEST_USAGE: &str = "assessment_only";
const DIGEST_DOMAIN: &[u8] = b"dope-dvs-role-binding-v1\0";

/// Required declaration for every newly issued DVS receipt.
///
/// `common_pins` duplicates the request's five common identities. The three
/// view pins bind FIT to `fit_scope_sha256`, TEST to `real_test_sha256`, and
/// explicitly identify the synthetic-validation view used for model choice.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeclaredEvidenceRoles {
    pub schema: String,
    pub real_fit_usage: String,
    pub synthetic_validation_usage: String,
    pub real_test_usage: String,
    pub real_fit_sha256: String,
    pub synthetic_validation_sha256: String,
    pub real_test_sha256: String,
    pub common_pins: DataPins,
    pub binding_sha256: String,
}

impl DeclaredEvidenceRoles {
    /// Build the fixed role declaration from the common pins and explicit TUNE view hash.
    pub fn new(
        common_pins: DataPins,
        synthetic_validation_sha256: String,
    ) -> Result<Self, DvsError> {
        let mut declaration = Self {
            schema: ROLE_SCHEMA.to_owned(),
            real_fit_usage: REAL_FIT_USAGE.to_owned(),
            synthetic_validation_usage: SYNTHETIC_VALIDATION_USAGE.to_owned(),
            real_test_usage: REAL_TEST_USAGE.to_owned(),
            real_fit_sha256: common_pins.fit_scope_sha256.clone(),
            synthetic_validation_sha256,
            real_test_sha256: common_pins.real_test_sha256.clone(),
            common_pins,
            binding_sha256: String::new(),
        };
        declaration.binding_sha256 = declaration.recompute_binding_sha256()?;
        Ok(declaration)
    }

    /// Canonical digest: fixed domain; each role label, usage tag and 64-byte
    /// lowercase ASCII pin in FIT/TUNE/TEST order; then the five common pins in
    /// DataPins field order. NUL terminators separate all labels and tags.
    pub fn recompute_binding_sha256(&self) -> Result<String, DvsError> {
        let mut hash = Sha256::new();
        hash.update(DIGEST_DOMAIN);
        for (label, usage, value) in [
            (
                "real_fit",
                self.real_fit_usage.as_str(),
                self.real_fit_sha256.as_str(),
            ),
            (
                "synthetic_validation",
                self.synthetic_validation_usage.as_str(),
                self.synthetic_validation_sha256.as_str(),
            ),
            (
                "real_test",
                self.real_test_usage.as_str(),
                self.real_test_sha256.as_str(),
            ),
        ] {
            role_pin(value)?;
            hash.update(label.as_bytes());
            hash.update([0]);
            hash.update(usage.as_bytes());
            hash.update([0]);
            hash.update(value.as_bytes());
            hash.update([0]);
        }
        for (label, value) in [
            (
                "fit_scope_sha256",
                self.common_pins.fit_scope_sha256.as_str(),
            ),
            (
                "real_test_sha256",
                self.common_pins.real_test_sha256.as_str(),
            ),
            (
                "prepared_view_sha256",
                self.common_pins.prepared_view_sha256.as_str(),
            ),
            ("groups_sha256", self.common_pins.groups_sha256.as_str()),
            ("roster_sha256", self.common_pins.roster_sha256.as_str()),
        ] {
            role_pin(value)?;
            hash.update(label.as_bytes());
            hash.update([0]);
            hash.update(value.as_bytes());
            hash.update([0]);
        }
        Ok(format!("{:x}", hash.finalize()))
    }

    pub fn validate(&self, expected: &DataPins) -> Result<RoleBindingReceipt, DvsError> {
        ensure(
            self.schema == ROLE_SCHEMA
                && self.real_fit_usage == REAL_FIT_USAGE
                && self.synthetic_validation_usage == SYNTHETIC_VALIDATION_USAGE
                && self.real_test_usage == REAL_TEST_USAGE,
            "invalid_dvs_role_binding",
        )?;
        self.common_pins
            .validate()
            .map_err(|_| DvsError("invalid_dvs_role_binding"))?;
        expected
            .validate()
            .map_err(|_| DvsError("invalid_dvs_role_binding"))?;
        ensure(
            self.common_pins == *expected
                && self.real_fit_sha256 == self.common_pins.fit_scope_sha256
                && self.real_test_sha256 == self.common_pins.real_test_sha256
                && self.real_fit_sha256 != self.real_test_sha256
                && self.synthetic_validation_sha256 != self.real_test_sha256,
            "invalid_dvs_role_binding",
        )?;
        ensure(
            self.binding_sha256 == self.recompute_binding_sha256()?,
            "invalid_dvs_role_binding",
        )?;
        Ok(RoleBindingReceipt {
            state: "caller_declared_hash_checked".to_owned(),
            schema: self.schema.clone(),
            real_fit_usage: self.real_fit_usage.clone(),
            synthetic_validation_usage: self.synthetic_validation_usage.clone(),
            real_test_usage: self.real_test_usage.clone(),
            real_fit_sha256: self.real_fit_sha256.clone(),
            synthetic_validation_sha256: self.synthetic_validation_sha256.clone(),
            real_test_sha256: self.real_test_sha256.clone(),
            common_pins: self.common_pins.clone(),
            binding_sha256: self.binding_sha256.clone(),
        })
    }
}

fn role_pin(value: &str) -> Result<(), DvsError> {
    pin(value).map_err(|_| DvsError("invalid_dvs_role_binding"))
}

/// New receipts retain the declaration but label its limited verification scope.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RoleBindingReceipt {
    pub state: String,
    pub schema: String,
    pub real_fit_usage: String,
    pub synthetic_validation_usage: String,
    pub real_test_usage: String,
    pub real_fit_sha256: String,
    pub synthetic_validation_sha256: String,
    pub real_test_sha256: String,
    pub common_pins: DataPins,
    pub binding_sha256: String,
}

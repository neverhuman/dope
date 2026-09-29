use serde::{Deserialize, Serialize};

use crate::corpus::{
    DatasetRecord, SplitManifest, ValidationSubmanifest, validate_split_manifest,
    validate_validation_submanifest,
};
use crate::error::{DopeError, Result};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AccessRole {
    Generator,
    SurfaceTrainer,
    GuardedEvaluator,
    FinalReleaseEvaluator,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccessBroker {
    pub manifest_checksum: String,
    pub role: AccessRole,
    pub final_release_open: bool,
}

impl AccessBroker {
    pub fn new(
        manifest: &SplitManifest,
        role: AccessRole,
        final_release_open: bool,
    ) -> Result<Self> {
        validate_split_manifest(manifest)?;
        if final_release_open && role != AccessRole::FinalReleaseEvaluator {
            return Err(DopeError::Data(
                "only the final release evaluator may open sealed lineages".into(),
            ));
        }
        Ok(Self {
            manifest_checksum: manifest.checksum.clone(),
            role,
            final_release_open,
        })
    }

    pub fn authorize(&self, record: &DatasetRecord, partition: &str) -> Result<()> {
        let sealed = record.split == "test";
        let allowed = match self.role {
            AccessRole::Generator | AccessRole::SurfaceTrainer => !sealed && partition == "train",
            AccessRole::GuardedEvaluator => !sealed && matches!(partition, "train" | "test"),
            AccessRole::FinalReleaseEvaluator => {
                (!sealed || self.final_release_open) && matches!(partition, "train" | "test")
            }
        };
        if allowed {
            Ok(())
        } else {
            Err(DopeError::Data(format!(
                "access denied for role {:?}, split {}, partition {partition}",
                self.role, record.split
            )))
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationAccessBroker {
    pub manifest_checksum: String,
    pub role: AccessRole,
    pub validation_cert_open: bool,
}

impl ValidationAccessBroker {
    pub fn new(
        manifest: &ValidationSubmanifest,
        role: AccessRole,
        validation_cert_open: bool,
    ) -> Result<Self> {
        validate_validation_submanifest(manifest)?;
        if validation_cert_open && role != AccessRole::FinalReleaseEvaluator {
            return Err(DopeError::Data(
                "only the guarded final evaluator may open validation-cert".into(),
            ));
        }
        Ok(Self {
            manifest_checksum: manifest.checksum.clone(),
            role,
            validation_cert_open,
        })
    }

    pub fn authorize_lineage(
        &self,
        manifest: &ValidationSubmanifest,
        lineage_group_id: &str,
    ) -> Result<()> {
        validate_validation_submanifest(manifest)?;
        if manifest.checksum != self.manifest_checksum {
            return Err(DopeError::Data(
                "validation access manifest changed after authorization".into(),
            ));
        }
        let partition = manifest
            .assignments
            .iter()
            .find(|assignment| assignment.lineage_group_id == lineage_group_id)
            .map(|assignment| assignment.partition.as_str())
            .ok_or_else(|| DopeError::Data("lineage is absent from validation manifest".into()))?;
        let allowed = match self.role {
            AccessRole::Generator | AccessRole::SurfaceTrainer => partition == "validation-select",
            AccessRole::GuardedEvaluator => partition == "validation-select",
            AccessRole::FinalReleaseEvaluator => {
                partition == "validation-select"
                    || (partition == "validation-cert" && self.validation_cert_open)
            }
        };
        if allowed {
            Ok(())
        } else {
            Err(DopeError::Data(format!(
                "access denied for role {:?} to {partition}",
                self.role
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{ValidationAssignment, ValidationSubmanifest};

    fn manifest() -> ValidationSubmanifest {
        let mut manifest = ValidationSubmanifest {
            format: "dope-validation-submanifest".into(),
            version: 1,
            seed: 1729,
            source_manifest_checksum: "source".into(),
            selection_numerator: 3,
            selection_denominator: 5,
            validation_cert_sealed: true,
            validation_select_lineage_groups: 1,
            validation_cert_lineage_groups: 1,
            assignments: vec![
                ValidationAssignment {
                    dataset_id: "a".into(),
                    lineage_group_id: "select".into(),
                    partition: "validation-select".into(),
                },
                ValidationAssignment {
                    dataset_id: "b".into(),
                    lineage_group_id: "cert".into(),
                    partition: "validation-cert".into(),
                },
            ],
            checksum: String::new(),
        };
        // This mirrors the private canonical checksum rule used by corpus.
        manifest.checksum = blake3::hash(&serde_json::to_vec(&manifest).unwrap())
            .to_hex()
            .to_string();
        manifest
    }

    #[test]
    fn train_roles_cannot_open_validation_cert() {
        let manifest = manifest();
        let trainer =
            ValidationAccessBroker::new(&manifest, AccessRole::SurfaceTrainer, false).unwrap();
        assert!(trainer.authorize_lineage(&manifest, "select").is_ok());
        assert!(trainer.authorize_lineage(&manifest, "cert").is_err());
        assert!(ValidationAccessBroker::new(&manifest, AccessRole::SurfaceTrainer, true).is_err());
        let evaluator =
            ValidationAccessBroker::new(&manifest, AccessRole::FinalReleaseEvaluator, true)
                .unwrap();
        assert!(evaluator.authorize_lineage(&manifest, "cert").is_ok());
    }
}

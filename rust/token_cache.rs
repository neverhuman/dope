//! Identity-bound token cache for a new encoder study.
//!
//! The key is the canonical hash of the producing commitments in the
//! 2026-11-17 cache contract. A matching key does not authorize a forbidden
//! source role. Historical scientific-cache readers are a separate module
//! and stay available for exact replay.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::error::{DopeError, Result};

pub const SCHEMA: &str = "dope.token-cache/v1";
const KEY_SCHEMA: &str = "dope.token-cache-key/v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceRole {
    Train,
    Test,
    Confirmation,
    Query,
    Holdout,
}

impl SourceRole {
    pub fn parse(raw: &str) -> Result<Self> {
        match raw {
            "train" => Ok(Self::Train),
            "test" => Ok(Self::Test),
            "confirmation" => Ok(Self::Confirmation),
            "query" => Ok(Self::Query),
            "holdout" => Ok(Self::Holdout),
            _ => Err(DopeError::Data("token_cache_role".into())),
        }
    }

    fn admits_cache(self) -> bool {
        matches!(self, Self::Train)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenCacheIdentity {
    pub encoder_weights_sha256: String,
    pub encoder_graph_and_readout_sha256: String,
    pub preprocessing_and_normalization_sha256: String,
    pub source_content_commitment: String,
    pub ordered_physical_fit_row_commitment: String,
    pub target_declaration_and_fit_target_commitment: String,
    pub task_metric_budget_constraints_sha256: String,
    pub feature_group_identity_and_permutation_sha256: String,
    pub landmark_recipe_values_masks_and_fit_scope_sha256: String,
    pub view_seed_and_row_column_sampling_policy: String,
    pub projector_normalizer_and_token_layout_sha256: String,
    pub dtype_device_numeric_policy: String,
}

impl TokenCacheIdentity {
    pub fn checked(self) -> Result<Self> {
        self.check_digests()?;
        Ok(self)
    }

    fn check_digests(&self) -> Result<()> {
        for field in [
            &self.encoder_weights_sha256,
            &self.encoder_graph_and_readout_sha256,
            &self.preprocessing_and_normalization_sha256,
            &self.source_content_commitment,
            &self.ordered_physical_fit_row_commitment,
            &self.target_declaration_and_fit_target_commitment,
            &self.task_metric_budget_constraints_sha256,
            &self.feature_group_identity_and_permutation_sha256,
            &self.landmark_recipe_values_masks_and_fit_scope_sha256,
            &self.view_seed_and_row_column_sampling_policy,
            &self.projector_normalizer_and_token_layout_sha256,
            &self.dtype_device_numeric_policy,
        ] {
            digest_text(field)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenPayload {
    pub tensor_shapes: Vec<Vec<u64>>,
    pub bytes: Vec<u8>,
    pub producing_run: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenCacheEnvelope {
    pub schema: &'static str,
    pub cache_key: String,
    pub tensor_shapes: Vec<Vec<u64>>,
    pub payload_sha256: String,
    pub producing_run: String,
}

#[derive(Clone, Debug)]
struct Stored {
    envelope: TokenCacheEnvelope,
    payload: Vec<u8>,
}

#[derive(Debug, Default)]
pub struct TokenCache {
    entries: BTreeMap<String, Stored>,
}

impl TokenCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn identity_key(identity: &TokenCacheIdentity) -> Result<String> {
        identity.check_digests()?;
        let document = serde_json::json!({
            "schema": KEY_SCHEMA,
            "dtype_device_numeric_policy": identity.dtype_device_numeric_policy,
            "encoder_graph_and_readout_sha256": identity.encoder_graph_and_readout_sha256,
            "encoder_weights_sha256": identity.encoder_weights_sha256,
            "feature_group_identity_and_permutation_sha256": identity.feature_group_identity_and_permutation_sha256,
            "landmark_recipe_values_masks_and_fit_scope_sha256": identity.landmark_recipe_values_masks_and_fit_scope_sha256,
            "ordered_physical_fit_row_commitment": identity.ordered_physical_fit_row_commitment,
            "preprocessing_and_normalization_sha256": identity.preprocessing_and_normalization_sha256,
            "projector_normalizer_and_token_layout_sha256": identity.projector_normalizer_and_token_layout_sha256,
            "source_content_commitment": identity.source_content_commitment,
            "target_declaration_and_fit_target_commitment": identity.target_declaration_and_fit_target_commitment,
            "task_metric_budget_constraints_sha256": identity.task_metric_budget_constraints_sha256,
            "view_seed_and_row_column_sampling_policy": identity.view_seed_and_row_column_sampling_policy,
        });
        let bytes = serde_json::to_vec(&document)?;
        Ok(sha256(&bytes))
    }

    /// Role is checked before any cache lookup. A forbidden role is not a miss.
    pub fn serve(&self, identity: &TokenCacheIdentity, role: SourceRole) -> Result<&[u8]> {
        if !role.admits_cache() {
            return Err(DopeError::Data("token_cache_role".into()));
        }
        let key = Self::identity_key(identity)?;
        self.entries
            .get(&key)
            .map(|stored| stored.payload.as_slice())
            .ok_or_else(|| DopeError::Data("token_cache_miss".into()))
    }

    pub fn read_or_store(
        &mut self,
        identity: &TokenCacheIdentity,
        role: SourceRole,
        produce: impl FnOnce() -> Result<TokenPayload>,
    ) -> Result<Vec<u8>> {
        if !role.admits_cache() {
            return Err(DopeError::Data("token_cache_role".into()));
        }
        let key = Self::identity_key(identity)?;
        if let Some(stored) = self.entries.get(&key) {
            return Ok(stored.payload.clone());
        }
        let payload = produce()?;
        let stored = store_record(&key, payload)?;
        let bytes = stored.payload.clone();
        self.entries.insert(key, stored);
        Ok(bytes)
    }

    pub fn envelope(
        &self,
        identity: &TokenCacheIdentity,
        role: SourceRole,
    ) -> Result<TokenCacheEnvelope> {
        if !role.admits_cache() {
            return Err(DopeError::Data("token_cache_role".into()));
        }
        let key = Self::identity_key(identity)?;
        self.entries
            .get(&key)
            .map(|stored| stored.envelope.clone())
            .ok_or_else(|| DopeError::Data("token_cache_miss".into()))
    }
}

fn store_record(key: &str, payload: TokenPayload) -> Result<Stored> {
    if payload.bytes.is_empty() {
        return Err(DopeError::Data("token_cache_payload".into()));
    }
    check_shapes(&payload.tensor_shapes)?;
    digest_text(&payload.producing_run)?;
    let payload_sha256 = sha256(&payload.bytes);
    Ok(Stored {
        envelope: TokenCacheEnvelope {
            schema: SCHEMA,
            cache_key: key.to_string(),
            tensor_shapes: payload.tensor_shapes,
            payload_sha256,
            producing_run: payload.producing_run,
        },
        payload: payload.bytes,
    })
}

fn check_shapes(shapes: &[Vec<u64>]) -> Result<()> {
    let ok = !shapes.is_empty()
        && shapes.iter().all(|shape| {
            !shape.is_empty()
                && shape.len() <= 8
                && shape.iter().all(|dim| (1..=1_048_576).contains(dim))
        });
    if ok {
        Ok(())
    } else {
        Err(DopeError::Data("token_cache_shape".into()))
    }
}

fn digest_text(value: &str) -> Result<()> {
    let ok = value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase());
    if ok {
        Ok(())
    } else {
        Err(DopeError::Data("token_cache_digest".into()))
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(fill: u8) -> String {
        format!("{fill:02x}").repeat(32)
    }

    fn identity() -> TokenCacheIdentity {
        TokenCacheIdentity {
            encoder_weights_sha256: hex(0x11),
            encoder_graph_and_readout_sha256: hex(0x22),
            preprocessing_and_normalization_sha256: hex(0x33),
            source_content_commitment: hex(0x44),
            ordered_physical_fit_row_commitment: hex(0x55),
            target_declaration_and_fit_target_commitment: hex(0x66),
            task_metric_budget_constraints_sha256: hex(0x77),
            feature_group_identity_and_permutation_sha256: hex(0x88),
            landmark_recipe_values_masks_and_fit_scope_sha256: hex(0x99),
            view_seed_and_row_column_sampling_policy: hex(0xaa),
            projector_normalizer_and_token_layout_sha256: hex(0xbb),
            dtype_device_numeric_policy: hex(0xcc),
        }
        .checked()
        .unwrap()
    }

    fn payload() -> TokenPayload {
        TokenPayload {
            tensor_shapes: vec![vec![4, 16]],
            bytes: b"token-bytes".to_vec(),
            producing_run: hex(0xdd),
        }
    }

    #[test]
    fn one_producing_input_invalidates_the_key() {
        let base = identity();
        let key = TokenCache::identity_key(&base).unwrap();
        let replacements = [
            (
                "weights",
                TokenCacheIdentity {
                    encoder_weights_sha256: hex(0x12),
                    ..base.clone()
                },
            ),
            (
                "target",
                TokenCacheIdentity {
                    target_declaration_and_fit_target_commitment: hex(0x67),
                    ..base.clone()
                },
            ),
            (
                "support-rows",
                TokenCacheIdentity {
                    ordered_physical_fit_row_commitment: hex(0x56),
                    ..base.clone()
                },
            ),
            (
                "normalizer",
                TokenCacheIdentity {
                    preprocessing_and_normalization_sha256: hex(0x34),
                    ..base.clone()
                },
            ),
            (
                "masks",
                TokenCacheIdentity {
                    landmark_recipe_values_masks_and_fit_scope_sha256: hex(0x9a),
                    ..base.clone()
                },
            ),
            (
                "projector",
                TokenCacheIdentity {
                    projector_normalizer_and_token_layout_sha256: hex(0xbc),
                    ..base.clone()
                },
            ),
        ];
        let mut cache = TokenCache::new();
        cache
            .read_or_store(&base, SourceRole::Train, || Ok(payload()))
            .unwrap();
        for (name, changed) in replacements {
            assert_ne!(key, TokenCache::identity_key(&changed).unwrap(), "{name}");
            assert_eq!(
                cache
                    .serve(&changed, SourceRole::Train)
                    .unwrap_err()
                    .to_string(),
                "invalid data: token_cache_miss",
                "{name}"
            );
        }
        assert_eq!(
            cache.serve(&base, SourceRole::Train).unwrap(),
            b"token-bytes"
        );
    }

    #[test]
    fn forbidden_roles_are_refused_before_a_hit() {
        let mut cache = TokenCache::new();
        let identity = identity();
        cache
            .read_or_store(&identity, SourceRole::Train, || Ok(payload()))
            .unwrap();
        for role in [
            SourceRole::Test,
            SourceRole::Confirmation,
            SourceRole::Query,
            SourceRole::Holdout,
        ] {
            assert_eq!(
                cache.serve(&identity, role).unwrap_err().to_string(),
                "invalid data: token_cache_role"
            );
            assert!(cache.envelope(&identity, role).is_err());
        }
        assert_eq!(
            SourceRole::parse("confirmation").unwrap(),
            SourceRole::Confirmation
        );
        assert!(SourceRole::parse("eval").is_err());
        assert_eq!(
            cache.serve(&identity, SourceRole::Train).unwrap(),
            b"token-bytes"
        );
    }

    #[test]
    fn cold_and_warm_reads_agree_and_the_producer_runs_once() {
        let mut cache = TokenCache::new();
        let identity = identity();
        let mut produced = 0;
        let cold = cache
            .read_or_store(&identity, SourceRole::Train, || {
                produced += 1;
                Ok(payload())
            })
            .unwrap();
        let warm = cache
            .read_or_store(&identity, SourceRole::Train, || {
                produced += 1;
                Ok(payload())
            })
            .unwrap();
        assert_eq!(cold, warm);
        assert_eq!(produced, 1);
        let envelope = cache.envelope(&identity, SourceRole::Train).unwrap();
        assert_eq!(envelope.schema, SCHEMA);
        assert_eq!(envelope.payload_sha256, sha256(&warm));
        assert_eq!(envelope.tensor_shapes, vec![vec![4, 16]]);
        assert_eq!(envelope.producing_run, hex(0xdd));
        assert_eq!(
            envelope.cache_key,
            TokenCache::identity_key(&identity).unwrap()
        );
    }

    #[test]
    fn a_forbidden_role_does_not_store() {
        let mut cache = TokenCache::new();
        let mut produced = 0;
        let error = cache
            .read_or_store(&identity(), SourceRole::Test, || {
                produced += 1;
                Ok(payload())
            })
            .unwrap_err();
        assert_eq!(error.to_string(), "invalid data: token_cache_role");
        assert_eq!(produced, 0);
        assert!(cache.entries.is_empty());
    }
}

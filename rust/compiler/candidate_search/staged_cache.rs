//! Bounded in-memory development cache. Keys and values contain only digests
//! and aggregate measurements, never tables, headers, paths, or extrema.
use std::collections::BTreeMap;

use serde::Serialize;

use crate::error::{DopeError, Result};

fn denied() -> DopeError {
    DopeError::Data("sealed_partition_access_denied".into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchPartition {
    Fit,
    DevelopmentSelection,
    SealedConfirmationQueryFeatures,
    HeldOutTestLabels,
}

impl SearchPartition {
    pub(crate) fn require_selection(self) -> Result<()> {
        match self {
            Self::DevelopmentSelection => Ok(()),
            Self::SealedConfirmationQueryFeatures | Self::HeldOutTestLabels => Err(denied()),
            Self::Fit => Err(DopeError::Data("development_selection_required".into())),
        }
    }
}

/// All external identities are cryptographic digests, including candidate spec,
/// loss and metric protocol identities. Caller pins must describe the actual
/// worker/model implementation used by DevelopmentEvaluator. Selection bytes,
/// exact artifact bytes and sample count are additionally bound by the search.
#[derive(Clone, Serialize)]
pub struct CacheIdentity {
    pub source_digest: [u8; 32],
    pub fit_row_digest: [u8; 32],
    pub prepared_view_digest: [u8; 32],
    pub compiler_version: [u8; 32],
    pub codec_version: [u8; 32],
    pub candidate_spec: [u8; 32],
    pub worker_abi_version: [u8; 32],
    pub model_panel_digest: [u8; 32],
    pub context_window: usize,
    pub seed_schedule: Vec<u64>,
    pub loss_id: [u8; 32],
    pub metric_id: [u8; 32],
}

impl CacheIdentity {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.context_window == 0
            || self.seed_schedule.is_empty()
            || self.seed_schedule.len() > 64
            || self
                .seed_schedule
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.seed_schedule.len()
        {
            return Err(DopeError::Data("invalid_development_cache_identity".into()));
        }
        Ok(())
    }

    pub fn key(
        &self,
        partition: SearchPartition,
        selection_digest: [u8; 32],
        artifact_digest: [u8; 32],
        sample_rows: usize,
    ) -> Result<[u8; 32]> {
        partition.require_selection()?;
        self.validate()?;
        let encoded = serde_json::to_vec(&(self, selection_digest, artifact_digest, sample_rows))
            .map_err(|_| DopeError::Data("invalid_development_cache_identity".into()))?;
        let mut hash = blake3::Hasher::new();
        hash.update(b"dope-development-candidate-cache-v1\0");
        // Local search changes invalidate entries even when a caller reuses a
        // stale compiler version pin. Length framing prevents concatenation aliases.
        for source in [
            include_str!("../candidate_search.rs"),
            include_str!("staged_cache.rs"),
            include_str!("../adaptive_allocation.rs"),
        ] {
            hash.update(&(source.len() as u64).to_le_bytes());
            hash.update(source.as_bytes());
        }
        hash.update(&encoded);
        Ok(*hash.finalize().as_bytes())
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq)]
pub struct DevelopmentMetrics {
    pub tstr_loss: f64,
    pub driver_agreement: f64,
}

impl DevelopmentMetrics {
    pub(crate) fn validate(self) -> Result<()> {
        if !self.tstr_loss.is_finite()
            || self.tstr_loss < 0.0
            || !self.driver_agreement.is_finite()
            || !(0.0..=1.0).contains(&self.driver_agreement)
        {
            return Err(DopeError::Data("invalid_development_metrics".into()));
        }
        Ok(())
    }
}

pub struct DevelopmentCache {
    entries: BTreeMap<[u8; 32], DevelopmentMetrics>,
    capacity: usize,
}

impl DevelopmentCache {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > 4096 {
            return Err(DopeError::Data("development_cache_budget_exceeded".into()));
        }
        Ok(Self {
            entries: BTreeMap::new(),
            capacity,
        })
    }

    pub(crate) fn get(
        &self,
        partition: SearchPartition,
        key: [u8; 32],
    ) -> Result<Option<DevelopmentMetrics>> {
        partition.require_selection()?;
        Ok(self.entries.get(&key).copied())
    }

    pub(crate) fn insert(
        &mut self,
        partition: SearchPartition,
        key: [u8; 32],
        metrics: DevelopmentMetrics,
    ) -> Result<()> {
        partition.require_selection()?;
        metrics.validate()?;
        if self.entries.len() == self.capacity && !self.entries.contains_key(&key) {
            self.entries.pop_first();
        }
        self.entries.insert(key, metrics);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn identity() -> CacheIdentity {
        CacheIdentity {
            source_digest: [1; 32],
            fit_row_digest: [2; 32],
            prepared_view_digest: [3; 32],
            compiler_version: [4; 32],
            codec_version: [5; 32],
            candidate_spec: [6; 32],
            worker_abi_version: [7; 32],
            model_panel_digest: [8; 32],
            context_window: 64,
            seed_schedule: vec![1729, 42],
            loss_id: [9; 32],
            metric_id: [10; 32],
        }
    }

    #[test]
    fn every_identity_dimension_and_exact_evaluation_input_invalidates_cache() {
        let base = identity();
        let key = |id: &CacheIdentity| {
            id.key(
                SearchPartition::DevelopmentSelection,
                [11; 32],
                [12; 32],
                64,
            )
            .unwrap()
        };
        let expected = key(&base);
        for field in 0..12 {
            let mut changed = base.clone();
            match field {
                0 => changed.source_digest[0] ^= 1,
                1 => changed.fit_row_digest[0] ^= 1,
                2 => changed.prepared_view_digest[0] ^= 1,
                3 => changed.compiler_version[0] ^= 1,
                4 => changed.codec_version[0] ^= 1,
                5 => changed.candidate_spec[0] ^= 1,
                6 => changed.worker_abi_version[0] ^= 1,
                7 => changed.model_panel_digest[0] ^= 1,
                8 => changed.context_window += 1,
                9 => changed.seed_schedule.reverse(),
                10 => changed.loss_id[0] ^= 1,
                11 => changed.metric_id[0] ^= 1,
                _ => unreachable!(),
            }
            assert_ne!(key(&changed), expected);
        }
        for (selection, artifact, rows) in [
            ([13; 32], [12; 32], 64),
            ([11; 32], [13; 32], 64),
            ([11; 32], [12; 32], 256),
        ] {
            assert_ne!(
                base.key(
                    SearchPartition::DevelopmentSelection,
                    selection,
                    artifact,
                    rows
                )
                .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn sealed_reads_writes_and_conditioning_fail_before_lookup() {
        let mut cache = DevelopmentCache::new(1).unwrap();
        let metrics = DevelopmentMetrics {
            tstr_loss: 0.2,
            driver_agreement: 0.9,
        };
        for partition in [
            SearchPartition::SealedConfirmationQueryFeatures,
            SearchPartition::HeldOutTestLabels,
        ] {
            for error in [
                cache.get(partition, [0; 32]).unwrap_err(),
                cache.insert(partition, [0; 32], metrics).unwrap_err(),
                identity().key(partition, [0; 32], [0; 32], 64).unwrap_err(),
            ] {
                assert_eq!(
                    error.to_string(),
                    "invalid data: sealed_partition_access_denied"
                );
            }
        }
        cache
            .insert(SearchPartition::DevelopmentSelection, [1; 32], metrics)
            .unwrap();
        cache
            .insert(SearchPartition::DevelopmentSelection, [2; 32], metrics)
            .unwrap();
        assert_eq!(cache.entries.len(), 1);
        assert!(
            cache
                .get(SearchPartition::DevelopmentSelection, [1; 32])
                .unwrap()
                .is_none()
        );
        assert!(
            cache
                .insert(
                    SearchPartition::DevelopmentSelection,
                    [3; 32],
                    DevelopmentMetrics {
                        tstr_loss: f64::NAN,
                        driver_agreement: 1.0
                    }
                )
                .is_err()
        );
        let mut invalid = identity();
        invalid.seed_schedule = vec![1, 1];
        assert!(invalid.validate().is_err());
    }
}

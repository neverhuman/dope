//! Fresh-worker reload and sampling using only charged bundle state and a
//! caller-pinned, source-independent native decoder. This is Linux evidence,
//! not an OS image inventory or a claim about arbitrary decoder semantics.

use super::{
    BundleError, BundleLimits, SamplingAccounting, content_fingerprint, err, verify_bundle,
};
use crate::typed_data::isolation::{self, RuntimeFile};
use std::path::Path;

fn check_charged_closure(
    root: &Path,
    digest: &str,
    limits: &BundleLimits,
) -> Result<(), BundleError> {
    // Inventory-only verification may legitimately read optional, uncharged
    // evidence. A sampling access proof must reject it before component I/O.
    let bytes = super::read_relative(root, super::MANIFEST_NAME, limits.max_manifest_bytes as u64)?;
    if content_fingerprint(&bytes) != digest {
        return Err(err("manifest_digest_mismatch"));
    }
    let manifest = super::parse_json_subset(&bytes)?;
    let (handler, _, _, components) = super::parse_manifest(&manifest, limits)?;
    if handler != super::SamplingHandler::Native {
        return Err(err("sampling_handler_refused"));
    }
    let edges = components
        .iter()
        .map(|c| (c.id.clone(), c.depends_on.clone()))
        .collect();
    let mut reachable = std::collections::BTreeSet::new();
    for component in &components {
        if component.role == super::Role::Payload {
            reachable.extend(super::closure(&component.id, &edges));
        }
    }
    if components.iter().any(|c| {
        matches!(c.role, super::Role::Evidence | super::Role::Rows)
            && !c.required
            && !reachable.contains(&c.id)
    }) {
        return Err(err("uncharged_dependency"));
    }
    Ok(())
}

pub struct SourceFreeSample {
    pub values: Vec<f32>,
    pub accounting: SamplingAccounting,
    pub decoder_files: Vec<RuntimeFile>,
    pub cold_sampling_closure_bytes: u64,
}

/// Pin the mapped executable and libraries before entering the worker. All
/// mapped files are counted once; learned state must reside in the bundle.
pub fn decoder_fingerprint() -> Result<String, BundleError> {
    let files = isolation::runtime_inventory().map_err(|_| err("decoder_inventory_unavailable"))?;
    fingerprint(&files)
}

fn fingerprint(files: &[RuntimeFile]) -> Result<String, BundleError> {
    let bytes = serde_json::to_vec(files).map_err(|_| err("decoder_inventory_unavailable"))?;
    Ok(content_fingerprint(&bytes))
}

/// Irreversibly confine a fresh worker before reloading the bundle, then deny
/// all filesystem access before decoding/sampling retained bytes. The caller
/// supplies the independently pinned shared decoder identity and explicit seed.
/// No source rows or learned priors may already be resident in this worker.
pub fn sample_source_free(
    root: &Path,
    expected_manifest_sha256: &str,
    expected_decoder_sha256: &str,
    rows: usize,
    seed: u64,
    limits: &BundleLimits,
) -> Result<SourceFreeSample, BundleError> {
    let decoder_files =
        isolation::runtime_inventory().map_err(|_| err("decoder_inventory_unavailable"))?;
    if fingerprint(&decoder_files)? != expected_decoder_sha256 {
        return Err(err("decoder_identity_mismatch"));
    }
    isolation::confine_bundle_directory(root).map_err(|_| err("bundle_isolation_refused"))?;
    check_charged_closure(root, expected_manifest_sha256, limits)?;
    let verified = verify_bundle(root, expected_manifest_sha256, limits)?;
    let accounting = verified.accounting().clone();
    let cold_sampling_closure_bytes = decoder_files
        .iter()
        .try_fold(accounting.charged_sampling_bytes, |total, file| {
            super::checked_add(total, file.bytes as u64)
        })?;
    isolation::seal_sampling_process().map_err(|_| err("sampling_isolation_refused"))?;
    let values = verified.sample(rows, Some(seed), limits)?;
    Ok(SourceFreeSample {
        values,
        accounting,
        decoder_files,
        cold_sampling_closure_bytes,
    })
}

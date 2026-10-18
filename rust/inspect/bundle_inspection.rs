use super::*;
use crate::bundle::{BundleLimits, verify_bundle};
use serde_json::json;

/// Inspect exactly the declared inventory, under an explicit manifest identity.
/// Identity is not evidence that a decoder is public, independent or licensed.
/// Inventory-only bundles do not establish executable sampling closure.
pub fn inspect_bundle(
    root: &Path,
    expected_manifest_sha256: &str,
    limits: &BundleLimits,
) -> Result<Value> {
    let verified = verify_bundle(root, expected_manifest_sha256, limits)
        .map_err(|error| DopeError::Codec(error.code().into()))?;
    let a = verified.accounting();
    let r = verified
        .recipient_accounting()
        .map_err(|error| DopeError::Codec(error.code().into()))?;
    Ok(json!({
        "format": "dope.recipient-inventory/v1",
        "status": "inventory_verified",
        "sampling_closure_qualification": "unmeasured",
        "scope": "declared_recipient_bundle",
        "amortization_status": "unverified_full_charge",
        "dataset_specific_bytes": r.dataset_specific_bytes,
        "declared_runtime_bytes": r.declared_runtime_bytes,
        "shared_source_independent_decoder_bytes": r.shared_source_independent_decoder_bytes,
        "cold_closure_bytes": r.cold_closure_bytes,
        "amortized_k_closure_bytes": r.amortized_k_closure_bytes,
        "amortization_count": r.amortization_count,
        "generated_samples_bytes": r.generated_samples_bytes,
        "component_bytes": r.component_bytes,
        "manifest_framing_bytes": r.manifest_framing_bytes,
        "physical_file_bytes": r.physical_file_bytes,
        "nonrecipient_delivery_bytes": r.nonrecipient_delivery_bytes,
        "complete_delivery_bytes": a.complete_delivery_bytes,
        "section_bytes": {
            "sampling_payload": a.sampling_payload_bytes,
            "shared_learned_state": a.shared_learned_state_bytes,
            "runtime": a.runtime_package_bytes,
            "evidence_authorization": a.evidence_authorization_bytes,
            "materialized_rows": a.materialized_rows_bytes,
            "transform": a.transform_bytes,
            "dictionary": a.dictionary_bytes
        }
    }))
}

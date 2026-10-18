/// Recipient closure of the verified directory inventory, using measured wire
/// bytes. Role labels, hashes and K are not evidence of source independence or
/// licensing: all learned state and unqualified runtime remain in R(A,H).
/// This is inventory accounting, not qualification of an external sampler.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecipientAccounting {
    pub dataset_specific_bytes: u64,
    pub declared_runtime_bytes: u64,
    pub shared_source_independent_decoder_bytes: Option<u64>,
    pub cold_closure_bytes: u64,
    pub amortized_k_closure_bytes: u64,
    pub amortization_count: u64,
    pub generated_samples_bytes: u64,
    pub component_bytes: u64,
    pub manifest_framing_bytes: u64,
    pub physical_file_bytes: u64,
    pub nonrecipient_delivery_bytes: u64,
}

impl VerifiedBundle {
    pub fn recipient_accounting(&self) -> Result<RecipientAccounting, BundleError> {
        recipient_accounting(&self.accounting)
    }
}

fn recipient_accounting(a: &SamplingAccounting) -> Result<RecipientAccounting, BundleError> {
    let component_bytes = [
        a.sampling_payload_bytes,
        a.shared_learned_state_bytes,
        a.runtime_package_bytes,
        a.evidence_authorization_bytes,
        a.materialized_rows_bytes,
        a.transform_bytes,
        a.dictionary_bytes,
    ]
    .into_iter()
    .try_fold(0u64, checked_add)?;
    let physical_file_bytes = checked_add(component_bytes, a.manifest_bytes)?;
    let nonrecipient_delivery_bytes = physical_file_bytes
        .checked_sub(a.charged_sampling_bytes)
        .ok_or_else(|| err("inconsistent_byte_ledger"))?;
    Ok(RecipientAccounting {
        dataset_specific_bytes: a.charged_sampling_bytes,
        declared_runtime_bytes: a.runtime_package_bytes,
        shared_source_independent_decoder_bytes: None,
        cold_closure_bytes: a.charged_sampling_bytes,
        amortized_k_closure_bytes: a.charged_sampling_bytes,
        amortization_count: a.amortization_count,
        generated_samples_bytes: a.materialized_rows_bytes,
        component_bytes,
        manifest_framing_bytes: a.manifest_bytes,
        physical_file_bytes,
        nonrecipient_delivery_bytes,
    })
}

/// Identity of a bounded manifest read, for inspection only. This is not a
/// trusted publication pin: callers still verify every file against the bytes.
pub fn manifest_fingerprint(root: &Path, limits: &BundleLimits) -> Result<String, BundleError> {
    read_relative(root, MANIFEST_NAME, limits.max_manifest_bytes as u64)
        .map(|bytes| content_fingerprint(&bytes))
}

#[cfg(test)]
mod recipient_tests {
    use super::*;

    #[test]
    fn recipient_ledger_rejects_overflow() {
        let mut a = SamplingAccounting {
            sampling_payload_bytes: u64::MAX,
            shared_learned_state_bytes: 1,
            standalone_learned_state_bytes: 0,
            amortized_learned_state_bytes: 0,
            runtime_package_bytes: 0,
            evidence_authorization_bytes: 0,
            materialized_rows_bytes: 0,
            transform_bytes: 0,
            dictionary_bytes: 0,
            manifest_bytes: 0,
            charged_sampling_bytes: 0,
            amortization_count: 1,
            transport_framing: TransportFraming::Unmeasured,
            complete_delivery_bytes: None,
        };
        assert_eq!(
            recipient_accounting(&a).unwrap_err().code(),
            "integer_overflow"
        );
        a.shared_learned_state_bytes = 0;
        a.manifest_bytes = 1;
        assert_eq!(
            recipient_accounting(&a).unwrap_err().code(),
            "integer_overflow"
        );
        a.sampling_payload_bytes = 0;
        a.charged_sampling_bytes = 2;
        assert_eq!(
            recipient_accounting(&a).unwrap_err().code(),
            "inconsistent_byte_ledger"
        );
    }
}

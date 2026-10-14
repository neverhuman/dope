#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GeometrySpace {
    RowActivation,
    PooledEmbedding,
    PredictionFingerprint,
    RefitStatistics,
    ActionRepresentation,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityAvailability {
    Available,
    Unavailable,
    Unmeasured,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingUnavailableReason {
    OfficialActivationNotExposed,
    CheckpointNotAdmitted,
    UnsupportedPlatform,
}

/// Declared capability metadata. Missing activations do not erase an available
/// prediction interface, and available coordinates do not qualify transfer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EncoderCapability {
    pub encoder_id: String,
    pub space: GeometrySpace,
    pub predictive: CapabilityAvailability,
    pub embedding: CapabilityAvailability,
    pub unavailable_reason: Option<EmbeddingUnavailableReason>,
    pub checkpoint_sha256: Option<String>,
    pub evidence_sha256: Option<String>,
    pub layer: Option<String>,
    pub pooling: Option<String>,
    pub normalization: Option<String>,
    pub target_included: Option<bool>,
    pub context_rows: Option<usize>,
    pub width: Option<usize>,
    pub qualifies_transfer: bool,
    pub qualifies_privacy: bool,
}

impl EncoderCapability {
    pub fn unavailable(
        encoder_id: String,
        space: GeometrySpace,
        predictive: CapabilityAvailability,
        reason: EmbeddingUnavailableReason,
    ) -> Result<Self> {
        if encoder_id.is_empty() {
            return Err(frozen_input_error("encoder_identity_missing"));
        }
        Ok(Self {
            encoder_id,
            space,
            predictive,
            embedding: CapabilityAvailability::Unavailable,
            unavailable_reason: Some(reason),
            checkpoint_sha256: None,
            evidence_sha256: None,
            layer: None,
            pooling: None,
            normalization: None,
            target_included: None,
            context_rows: None,
            width: None,
            qualifies_transfer: false,
            qualifies_privacy: false,
        })
    }

    fn native_action(embedding: &DatasetActionEmbedding) -> Self {
        Self {
            encoder_id: "dope_quantized_router_action_v1".into(),
            space: GeometrySpace::ActionRepresentation,
            predictive: CapabilityAvailability::Available,
            embedding: CapabilityAvailability::Available,
            unavailable_reason: None,
            checkpoint_sha256: Some(embedding.router_bundle_sha256.clone()),
            evidence_sha256: Some(embedding.router_evidence_sha256.clone()),
            layer: Some("candidate_penultimate_plus_standardized_actions".into()),
            pooling: Some("dataset_sketch_then_candidate_concatenation".into()),
            normalization: Some("reference_minmax_clamped_then_frozen_router_zscore".into()),
            target_included: Some(true),
            context_rows: Some(embedding.rows),
            width: Some(embedding.dimension),
            qualifies_transfer: false,
            qualifies_privacy: false,
        }
    }
}

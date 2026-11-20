//! Transfer training-view and experiment-receipt validators.
//!
//! Enforces disjoint physical FIT vs. gradient-query row commitments, prevents
//! query-bearing base dataset IDs from seeding FIT partitions, separates
//! measured vs. predicted 23-slot landmark bitsets, and blocks scientific
//! promotion unless execution completed on an unseen source family with
//! passing native qualification and intact frozen state.

use crate::error::{DopeError, Result};

pub const SCHEMA_VIEW: &str = "dope-training-view/v1";
pub const SCHEMA_EXPERIMENT: &str = "dope-transfer-experiment/v1";
pub const LANDMARK_SLOT_COUNT: u32 = 23;
pub const LANDMARK_MASK_LIMIT: u32 = 1 << LANDMARK_SLOT_COUNT;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingViewV1 {
    pub schema: String,
    pub view_id: String,
    pub base_dataset_id: String,
    pub partition_source_id: String,
    pub target_declaration_sha256: String,
    pub task_context_sha256: String,
    pub metric_and_budget_sha256: String,
    pub fit_row_commitment: String,
    pub gradient_query_row_commitment: String,
    pub physical_membership_disjoint: bool,
    pub normalizer_sha256: String,
    pub feature_group_map_sha256: String,
    pub fit_scope_sha256: String,
    pub observed_mask: u32,
    pub consumed_measured_mask: u32,
    pub predicted_mask: u32,
    pub predictor_sha256: Option<String>,
    pub task_weight: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransferExperimentReceiptV1 {
    pub schema: String,
    pub experiment_id: String,
    pub status: String,
    pub protocol: String,
    pub exposure_class: String,
    pub native_qualification: String,
    pub scientific_verdict: String,
    pub dope_commit: String,
    pub jopedime_commit: String,
    pub graph_config_sha256: String,
    pub model_weights_sha256: String,
    pub checkpoint_bundle_manifest_sha256: String,
    pub parameter_inventory_sha256: String,
    pub initialization_lineage_sha256: String,
    pub corpus_root_sha256: String,
    pub committed_exposure_ledger_sha256: String,
    pub objective_recipe_sha256: String,
    pub teacher_panel_sha256: String,
    pub preprocessing_sha256: String,
    pub projector_and_consumer_sha256: String,
    pub preregistration_sha256: String,
    pub frozen_before_sha256: String,
    pub frozen_after_sha256: String,
    pub resource_receipt_sha256: String,
    pub per_task_results_sha256: Option<String>,
    pub accepted_base_count: u64,
    pub committed_unique_base_count: u64,
}

pub fn validate_training_view(view: &TrainingViewV1) -> Result<()> {
    if view.schema != SCHEMA_VIEW {
        return Err(DopeError::Data("transfer_view_schema".into()));
    }
    if view.view_id.is_empty() || view.base_dataset_id.is_empty() {
        return Err(DopeError::Data("transfer_view_id".into()));
    }
    if view.partition_source_id.is_empty() || view.partition_source_id == view.base_dataset_id {
        return Err(DopeError::Data("transfer_view_partition_source".into()));
    }
    for digest in [
        &view.target_declaration_sha256,
        &view.task_context_sha256,
        &view.metric_and_budget_sha256,
        &view.fit_row_commitment,
        &view.gradient_query_row_commitment,
        &view.normalizer_sha256,
        &view.feature_group_map_sha256,
        &view.fit_scope_sha256,
    ] {
        hex_lower(digest, 64, "transfer_view_digest")?;
    }
    if !view.physical_membership_disjoint
        || view.fit_row_commitment == view.gradient_query_row_commitment
    {
        return Err(DopeError::Data("transfer_view_disjoint".into()));
    }
    if view.observed_mask >= LANDMARK_MASK_LIMIT
        || view.consumed_measured_mask >= LANDMARK_MASK_LIMIT
        || view.predicted_mask >= LANDMARK_MASK_LIMIT
    {
        return Err(DopeError::Data("transfer_view_landmark_mask".into()));
    }
    if (view.consumed_measured_mask & !view.observed_mask) != 0 {
        return Err(DopeError::Data("transfer_view_landmark_subset".into()));
    }
    if (view.consumed_measured_mask & view.predicted_mask) != 0 {
        return Err(DopeError::Data("transfer_view_landmark_disjoint".into()));
    }
    if view.predictor_sha256.is_some() != (view.predicted_mask != 0) {
        return Err(DopeError::Data("transfer_view_predictor_presence".into()));
    }
    if let Some(ref digest) = view.predictor_sha256 {
        hex_lower(digest, 64, "transfer_view_digest")?;
    }
    if !view.task_weight.is_finite() || view.task_weight <= 0.0 {
        return Err(DopeError::Data("transfer_view_weight".into()));
    }
    Ok(())
}

pub fn parse_training_view_json(bytes: &[u8]) -> Result<TrainingViewV1> {
    let view: TrainingViewV1 = serde_json::from_slice(bytes)
        .map_err(|_| DopeError::Data("transfer_view_schema".into()))?;
    validate_training_view(&view)?;
    Ok(view)
}

pub fn validate_experiment_receipt(receipt: &TransferExperimentReceiptV1) -> Result<()> {
    if receipt.schema != SCHEMA_EXPERIMENT {
        return Err(DopeError::Data("transfer_experiment_schema".into()));
    }
    if receipt.experiment_id.is_empty() {
        return Err(DopeError::Data("transfer_experiment_id".into()));
    }
    if !matches!(
        receipt.status.as_str(),
        "running" | "completed" | "failed" | "stopped"
    ) {
        return Err(DopeError::Data("transfer_experiment_status".into()));
    }
    if !matches!(
        receipt.protocol.as_str(),
        "fixed_pool" | "first_program" | "budgeted_search"
    ) {
        return Err(DopeError::Data("transfer_experiment_protocol".into()));
    }
    if !matches!(
        receipt.exposure_class.as_str(),
        "unseen_source_family" | "in_corpus" | "recall"
    ) {
        return Err(DopeError::Data("transfer_experiment_exposure".into()));
    }
    if !matches!(
        receipt.native_qualification.as_str(),
        "PASS" | "FAIL" | "NOT_RUN"
    ) {
        return Err(DopeError::Data("transfer_experiment_qualification".into()));
    }
    if !matches!(
        receipt.scientific_verdict.as_str(),
        "PROMOTE" | "NO_LIFT" | "INCONCLUSIVE" | "PENDING"
    ) {
        return Err(DopeError::Data("transfer_experiment_verdict".into()));
    }
    for commit in [&receipt.dope_commit, &receipt.jopedime_commit] {
        hex_lower(commit, 40, "transfer_experiment_commit")?;
    }
    for digest in [
        &receipt.graph_config_sha256,
        &receipt.model_weights_sha256,
        &receipt.checkpoint_bundle_manifest_sha256,
        &receipt.parameter_inventory_sha256,
        &receipt.initialization_lineage_sha256,
        &receipt.corpus_root_sha256,
        &receipt.committed_exposure_ledger_sha256,
        &receipt.objective_recipe_sha256,
        &receipt.teacher_panel_sha256,
        &receipt.preprocessing_sha256,
        &receipt.projector_and_consumer_sha256,
        &receipt.preregistration_sha256,
        &receipt.frozen_before_sha256,
        &receipt.frozen_after_sha256,
        &receipt.resource_receipt_sha256,
    ] {
        hex_lower(digest, 64, "transfer_experiment_digest")?;
    }
    if let Some(ref results_sha) = receipt.per_task_results_sha256 {
        hex_lower(results_sha, 64, "transfer_experiment_digest")?;
    }
    if receipt.frozen_before_sha256 != receipt.frozen_after_sha256 {
        return Err(DopeError::Data("transfer_experiment_frozen_drift".into()));
    }
    if receipt.committed_unique_base_count > receipt.accepted_base_count {
        return Err(DopeError::Data("transfer_experiment_base_count".into()));
    }
    if receipt.scientific_verdict == "PROMOTE"
        && (receipt.status != "completed"
            || receipt.native_qualification != "PASS"
            || receipt.exposure_class != "unseen_source_family"
            || receipt.per_task_results_sha256.is_none())
    {
        return Err(DopeError::Data("transfer_experiment_promote_gate".into()));
    }
    Ok(())
}

pub fn parse_experiment_receipt_json(bytes: &[u8]) -> Result<TransferExperimentReceiptV1> {
    let receipt: TransferExperimentReceiptV1 = serde_json::from_slice(bytes)
        .map_err(|_| DopeError::Data("transfer_experiment_schema".into()))?;
    validate_experiment_receipt(&receipt)?;
    Ok(receipt)
}

fn hex_lower(value: &str, expected_len: usize, code: &'static str) -> Result<()> {
    if value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err(DopeError::Data(code.into()))
    }
}

#[cfg(test)]
#[path = "transfer_schemas_tests.rs"]
mod tests;

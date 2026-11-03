//! Source-free public DVS receipts; no raw rows, names or extrema.
use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Debug, Serialize)]
pub struct CellReceipt {
    pub cell_sha256: String,
    pub family_id: u32,
    pub status: &'static str,
    pub reason: Option<&'static str>,
    pub query_rows: usize,
    pub null_loss: Option<f64>,
    pub real_loss: Option<f64>,
    pub synthetic_loss: Option<f64>,
    pub raw_retention: Option<f64>,
    pub driver: Value,
    pub wall_seconds: Option<f64>,
    pub fit_count: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PartReceipt {
    pub scope: &'static str,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ByteReceipt {
    pub payload_bytes: u64,
    pub shared_bytes: u64,
    pub sampling_closure_bytes: u64,
    pub owner_private_dependency_bytes: u64,
    pub parts: Vec<PartReceipt>,
    pub clean_room_sampling_state: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct FamilyReceipt {
    pub family_id: u32,
    pub scheduled_cells: usize,
    pub informative_cells: usize,
    pub descriptive_mean_retention: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DvsReceipt {
    pub protocol: &'static str,
    pub preparation_recipe: String,
    pub status: &'static str,
    pub scheduled_cells: usize,
    pub complete_cells: usize,
    pub uninformative_cells: usize,
    pub failed_or_missing_cells: usize,
    pub cells: Vec<CellReceipt>,
    pub families: Vec<FamilyReceipt>,
    pub robust_retention: Option<f64>,
    pub selection: Value,
    pub bytes: ByteReceipt,
    pub privacy: Value,
    pub pins: super::DataPins,
    pub qualification_state: &'static str,
    pub formal_dp_state: &'static str,
    pub release_authorization_state: &'static str,
}

/// Render only computed fields; no independent formula or success badge.
pub fn audit_card(r: &DvsReceipt) -> String {
    let value =
        |v: Option<f64>| v.map_or_else(|| "not available".to_string(), |n| format!("{n:.6}"));
    format!("# DOPE Audit Card\n\nProtocol: {}\nPreparation: {}\nState: {}\n\nScheduled cells: {}\nComplete: {}\nUninformative: {}\nFailed or missing: {}\nRobust retention: {}\nPayload bytes: {}\nSampling closure bytes: {}\nOwner-private dependency bytes: {}\n\nSelection: {}\nPrivacy diagnostics: {}\n\nQualification: {}\nFormal DP: {}\nRelease authorization: {}\n\nEmpirical diagnostics establish no formal DP or de-identification claim. No byte-based memorization bound is asserted.\n",
        r.protocol, r.preparation_recipe, r.status, r.scheduled_cells, r.complete_cells,
        r.uninformative_cells, r.failed_or_missing_cells, value(r.robust_retention),
        r.bytes.payload_bytes, r.bytes.sampling_closure_bytes, r.bytes.owner_private_dependency_bytes,
        r.selection, r.privacy, r.qualification_state, r.formal_dp_state, r.release_authorization_state)
}

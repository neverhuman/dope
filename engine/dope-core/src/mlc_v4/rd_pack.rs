//! Deterministic bounded candidate choice on caller-supplied FIT/TUNE evidence.
use super::{wire, Level, Packet, Result, WireError};
/// Candidate evidence must originate from declared FIT/TUNE, never real TEST.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub packet: Packet,
    pub fit_tune_loss: f64,
}
/// Pick smallest valid packet within additive tolerance of best supplied loss.
/// Ties use exact byte identity, avoiding input-order-dependent decisions.
pub fn select(candidates: &[Candidate], level: Level, tolerance: f64) -> Result<Vec<u8>> {
    if candidates.is_empty() || candidates.len() > 128 || !tolerance.is_finite() || tolerance < 0.0
    {
        return Err(WireError::new("invalid_candidates"));
    }
    let first = &candidates[0].packet;
    if candidates
        .iter()
        .any(|c| c.packet.identity != first.identity || c.packet.schema != first.schema)
    {
        return Err(WireError::new("candidate_scope_mismatch"));
    }
    let mut admitted = Vec::new();
    for candidate in candidates {
        if candidate.packet.level != level
            || !candidate.fit_tune_loss.is_finite()
            || candidate.fit_tune_loss < 0.0
        {
            return Err(WireError::new("invalid_candidate_evidence"));
        }
        match wire::encode(&candidate.packet) {
            Ok(bytes) => admitted.push((candidate.fit_tune_loss, bytes)),
            Err(e) if e.code == "over_budget" => {}
            Err(e) => return Err(e),
        }
    }
    let best = admitted
        .iter()
        .map(|(loss, _)| *loss)
        .min_by(f64::total_cmp)
        .ok_or(WireError::new("over_budget"))?;
    admitted.retain(|(loss, _)| *loss <= best + tolerance);
    admitted.sort_by(|(_, a), (_, b)| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    admitted
        .into_iter()
        .next()
        .map(|(_, bytes)| bytes)
        .ok_or(WireError::new("over_budget"))
}

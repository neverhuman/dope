//! Transitive bucketed lexicographic candidate selector strictly on decoded bytes (`T20`).

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateScore {
    pub id: String,
    pub bytes: usize,
    /// Worst hard-gate / witness violation (`0.02` bucket width).
    pub v_max: f64,
    /// Mean hard-gate / witness violation (`0.02` bucket width).
    pub v_mean: f64,
    /// Predictive SELECT loss (`0.002` bucket width).
    pub l_pred: f64,
    pub refit_lineage_sha256: String,
}

impl CandidateScore {
    /// Transitive lexicographic key: `(floor(V_max/0.02), floor(V_mean/0.02), floor(L_pred/0.002), bytes, id)`.
    pub fn bucket_key(&self) -> (i64, i64, i64, usize, &str) {
        let b_max = if self.v_max.is_finite() {
            (self.v_max.max(0.0) / 0.02).floor() as i64
        } else {
            i64::MAX
        };
        let b_mean = if self.v_mean.is_finite() {
            (self.v_mean.max(0.0) / 0.02).floor() as i64
        } else {
            i64::MAX
        };
        let b_pred = if self.l_pred.is_finite() {
            (self.l_pred / 0.002).floor() as i64
        } else {
            i64::MAX
        };
        (b_max, b_mean, b_pred, self.bytes, self.id.as_str())
    }

    pub fn cmp_lex(&self, other: &Self) -> Ordering {
        self.bucket_key().cmp(&other.bucket_key())
    }
}

/// Select the globally best decoded candidate under the strictly transitive bucketed key (`T20`).
pub fn select_best_decoded(candidates: &[CandidateScore]) -> Option<&CandidateScore> {
    candidates.iter().min_by(|a, b| a.cmp_lex(b))
}

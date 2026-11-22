//! Frozen prediction bundles and unclipped retention arithmetic for `dope-fidelity/v3`.

use crate::dvs::{ensure, finite_nonnegative, DvsError};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Regression,
    Binary,
    Multiclass,
}

/// Per-query predictions and exact losses evaluated in one frozen FIT-only unit space.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionBundle {
    pub task: TaskKind,
    /// Per-query losses of the constant/prior null baseline `L_0(i)`.
    pub null_losses: Vec<f64>,
    /// Per-query losses of the auditor trained on real `A`: `L_A(i)`.
    pub real_a_losses: Vec<f64>,
    /// Optional per-query losses of the auditor trained on fresh real twin `B`: `L_B(i)`.
    pub real_b_losses: Option<Vec<f64>>,
    /// Per-query losses of the auditor trained on synthetic `S`: `L_S(i)`.
    pub synth_losses: Vec<f64>,
    /// Optional physical donor/cluster ID per query row for donor-aware bootstrap (`T29`).
    pub donor_ids: Option<Vec<u64>>,
    /// Ground-truth targets on sealed query rows (regression value, {0,1}, or class index).
    pub y_true: Vec<f64>,
    /// Predictions from auditor trained on `A`: scalar per row (reg/binary prob) or flat `n * k` probs.
    pub pred_a: Vec<f64>,
    /// Optional predictions from auditor trained on twin `B`.
    pub pred_b: Option<Vec<f64>>,
    /// Predictions from auditor trained on `S`.
    pub pred_s: Vec<f64>,
    /// Number of classes for multiclass (1 for regression/binary scalar probabilities).
    pub num_outputs: usize,
}

impl PredictionBundle {
    pub fn n_query(&self) -> usize {
        self.null_losses.len()
    }

    pub fn validate(&self) -> Result<(), DvsError> {
        let n = self.null_losses.len();
        ensure(n >= 2, "empty_or_degenerate_query_bundle")?;
        ensure(self.num_outputs >= 1, "invalid_num_outputs")?;
        ensure(
            self.real_a_losses.len() == n
                && self.synth_losses.len() == n
                && self.y_true.len() == n
                && self.pred_a.len() == n * self.num_outputs
                && self.pred_s.len() == n * self.num_outputs,
            "bundle_length_mismatch",
        )?;
        if let Some(ref b) = self.real_b_losses {
            ensure(b.len() == n, "bundle_twin_loss_mismatch")?;
        }
        if let Some(ref pb) = self.pred_b {
            ensure(
                pb.len() == n * self.num_outputs,
                "bundle_twin_pred_mismatch",
            )?;
        }
        if let Some(ref d) = self.donor_ids {
            ensure(d.len() == n, "bundle_donor_len_mismatch")?;
        }
        for v in self
            .null_losses
            .iter()
            .chain(&self.real_a_losses)
            .chain(&self.synth_losses)
        {
            finite_nonnegative(*v)?;
        }
        if let Some(ref b) = self.real_b_losses {
            for &v in b {
                finite_nonnegative(v)?;
            }
        }
        for v in self.y_true.iter().chain(&self.pred_a).chain(&self.pred_s) {
            ensure(v.is_finite(), "nonfinite_bundle_prediction")?;
        }
        if let Some(ref pb) = self.pred_b {
            for &v in pb {
                ensure(v.is_finite(), "nonfinite_bundle_prediction")?;
            }
        }
        // T17: refuse degenerate constant target vectors on query set.
        let first = self.y_true[0];
        ensure(
            self.y_true.iter().any(|&v| (v - first).abs() > 1e-12),
            "degenerate_constant_query_target",
        )?;
        Ok(())
    }

    pub fn mean_null(&self) -> f64 {
        mean_slice(&self.null_losses)
    }

    pub fn mean_real_a(&self) -> f64 {
        mean_slice(&self.real_a_losses)
    }

    pub fn mean_real_b(&self) -> Option<f64> {
        self.real_b_losses.as_deref().map(mean_slice)
    }

    pub fn mean_synth(&self) -> f64 {
        mean_slice(&self.synth_losses)
    }

    /// Unclipped retention `R_f(S) = (L_0 - L_S) / (L_0 - L_A)` (`T22`).
    pub fn unclipped_retention_s(&self) -> Option<f64> {
        let l0 = self.mean_null();
        let la = self.mean_real_a();
        let ls = self.mean_synth();
        let gain = l0 - la;
        if gain <= 0.0 || !gain.is_finite() {
            return None;
        }
        let r = (l0 - ls) / gain;
        r.is_finite().then_some(r)
    }

    /// Unclipped twin retention `R_f(B) = (L_0 - L_B) / (L_0 - L_A)`.
    pub fn unclipped_retention_b(&self) -> Option<f64> {
        let lb = self.mean_real_b()?;
        let l0 = self.mean_null();
        let la = self.mean_real_a();
        let gain = l0 - la;
        if gain <= 0.0 || !gain.is_finite() {
            return None;
        }
        let r = (l0 - lb) / gain;
        r.is_finite().then_some(r)
    }
}

pub(crate) fn mean_slice(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let inv = 1.0 / xs.len() as f64;
    xs.iter().map(|&v| v * inv).sum()
}

//! Research-only Ultra/High 629D view beside `MlcV3Model::encode_features_629`.
//!
//! Keeps `MlcV3Model::encode_features_629` (`High` product contract) untouched while providing
//! pure-Rust, level-verified `Ultra` routing (`WP1`).

use super::{bad, MlcV3Model, MLC_V3_FEATURES_629_DIM, MLC_V3_GATE_KEYS_14, MLC_V3_MAX_COLS};
use crate::Result;
use candle_core::{DType, Tensor};

pub const RESEARCH_ENCODER_VIEW_V1_RECIPE: &str = "research-encoder-view/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResearchConditioningLevel {
    High,
    Ultra,
}

impl ResearchConditioningLevel {
    pub fn parse(raw: &str) -> Result<Self> {
        match raw {
            "high" => Ok(Self::High),
            "ultra" => Ok(Self::Ultra),
            _ => Err(bad("research encoder view accepts only high or ultra")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Ultra => "ultra",
        }
    }

    pub fn code(self) -> f64 {
        match self {
            Self::High => 0.0,
            Self::Ultra => 1.0,
        }
    }
}

impl MlcV3Model {
    /// Level-bound research `[629]` feature view. `High` delegates to `encode_features_629`;
    /// `Ultra` enforces matching `conditioning_level == Ultra` and finite inputs before routing.
    pub fn encode_research_view_629(
        &self,
        dataset_features: &[f32],
        col_landmarks: &[Vec<f32>],
        conditioning_level: ResearchConditioningLevel,
        requested_level: ResearchConditioningLevel,
    ) -> Result<Vec<f32>> {
        if conditioning_level != requested_level {
            return Err(bad("research encoder view conditioning_level mismatch"));
        }
        if requested_level == ResearchConditioningLevel::High {
            return self.encode_features_629(dataset_features, col_landmarks, "high");
        }
        if dataset_features.len() != 256
            || dataset_features.iter().any(|v| !v.is_finite())
            || col_landmarks.len() != MLC_V3_MAX_COLS
            || col_landmarks
                .iter()
                .any(|row| row.len() != 17 || row.iter().any(|v| !v.is_finite()))
        {
            return Err(bad(
                "ultra research view requires finite [256] and [64, 17] inputs",
            ));
        }
        let routed = self.router.forward(dataset_features, "ultra")?;
        let col_gates =
            self.target_net
                .compute_column_gates(&routed.task_latent, Some(col_landmarks), None)?;
        let mut out = Vec::with_capacity(MLC_V3_FEATURES_629_DIM);
        out.extend_from_slice(dataset_features);
        out.extend_from_slice(&routed.task_latent);
        out.extend_from_slice(&routed.dist_latent);
        for key in MLC_V3_GATE_KEYS_14 {
            let g = *routed
                .gates
                .get(key)
                .ok_or_else(|| bad("629D gate schema mismatch"))?;
            out.push(g as f32);
        }
        out.extend_from_slice(&routed.landmark_pred);
        out.extend_from_slice(&routed.bb_pred);
        out.extend_from_slice(&col_gates);
        for v in &mut out {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        Ok(out)
    }

    /// Device-resident batched `[B, 629]` research view for `High` or `Ultra`.
    pub fn encode_research_view_629_tensor(
        &self,
        dataset_features: &Tensor,
        col_landmarks: &Tensor,
        conditioning_level: ResearchConditioningLevel,
        requested_level: ResearchConditioningLevel,
    ) -> Result<Tensor> {
        if conditioning_level != requested_level {
            return Err(bad("research encoder view conditioning_level mismatch"));
        }
        if requested_level == ResearchConditioningLevel::High {
            return self.encode_features_629_tensor(dataset_features, col_landmarks, "high");
        }
        let (batch, width) = dataset_features.dims2()?;
        if batch == 0
            || width != 256
            || dataset_features.dtype() != DType::F32
            || col_landmarks.dims() != [batch, MLC_V3_MAX_COLS, 17]
            || col_landmarks.dtype() != DType::F32
            || !dataset_features.device().same_device(self.device())
            || !col_landmarks.device().same_device(self.device())
        {
            return Err(bad(
                "ultra research tensor view requires matching FP32 [B,256] and [B,64,17]",
            ));
        }
        let desc_host = dataset_features
            .to_device(&candle_core::Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let col_host = col_landmarks
            .to_device(&candle_core::Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if desc_host.iter().chain(&col_host).any(|v| !v.is_finite()) {
            return Err(bad("ultra research tensor view refuses non-finite inputs"));
        }
        let routed = self.router.forward_tensor(dataset_features, "ultra")?;
        let column_gates = self
            .target_net
            .compute_column_gates_tensor(&routed.task_latent, col_landmarks)?;
        let output = Tensor::cat(
            &[
                dataset_features,
                &routed.task_latent,
                &routed.dist_latent,
                &routed.gates,
                &routed.landmark_pred,
                &routed.bb_pred,
                &column_gates,
            ],
            1,
        )?;
        let finite = output.abs()?.le(f32::MAX as f64)?;
        Ok(finite.where_cond(&output, &output.zeros_like()?)?)
    }
}

pub(super) struct TreeScanStats {
    pub split_cnt: Vec<usize>,
    pub depth_w: Vec<f64>,
    pub root_cnt: Vec<usize>,
    pub col_thresh: Vec<Vec<f64>>,
    pub all_thresh: Vec<f64>,
    pub all_leaves: Vec<f64>,
    pub depth_sum: usize,
    pub first_rms: f64,
    pub last_rms: f64,
}

impl TreeScanStats {
    pub fn new(p_use: usize) -> Self {
        Self {
            split_cnt: vec![0; p_use],
            depth_w: vec![0.0; p_use],
            root_cnt: vec![0; p_use],
            col_thresh: vec![Vec::new(); p_use],
            all_thresh: Vec::new(),
            all_leaves: Vec::new(),
            depth_sum: 0,
            first_rms: 0.0,
            last_rms: 0.0,
        }
    }

    pub fn collect(&mut self, b: &super::TargetBundle, p_use: usize) {
        if !b.mc_trees.is_empty() {
            for (ti, t) in b.mc_trees.iter().enumerate() {
                self.record_splits(&t.features, &t.thresholds, p_use);
                let flat: Vec<f64> = t.leaf_values.iter().flatten().copied().collect();
                self.record_leaves(ti, b.mc_trees.len(), &flat);
            }
        } else {
            for (ti, t) in b.trees.iter().enumerate() {
                self.record_splits(&t.features, &t.thresholds, p_use);
                self.record_leaves(ti, b.trees.len(), &t.leaf_values);
            }
        }
    }

    fn record_splits(&mut self, features: &[usize], thresholds: &[f64], p_use: usize) {
        self.depth_sum += features.len();
        for (d, (&f, &th)) in features.iter().zip(thresholds).enumerate() {
            self.all_thresh.push(th);
            if f < p_use {
                self.split_cnt[f] += 1;
                self.depth_w[f] += 2.0_f64.powi(-(d as i32));
                if d == 0 {
                    self.root_cnt[f] += 1;
                }
                self.col_thresh[f].push(th);
            }
        }
    }

    fn record_leaves(&mut self, ti: usize, total: usize, leaves: &[f64]) {
        let rms = rms_slice(leaves);
        if ti == 0 {
            self.first_rms = rms;
        }
        if ti + 1 == total {
            self.last_rms = rms;
        }
        self.all_leaves.extend_from_slice(leaves);
    }
}

pub(super) fn basis_at(rows: &[f64], idx: usize) -> f64 {
    rows.get(idx).copied().unwrap_or(0.0)
}

pub(super) fn rms_slice(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    (vals.iter().map(|v| v * v).sum::<f64>() / (vals.len() as f64)).sqrt()
}

pub(super) fn mean_std_max(vals: &[f64]) -> (f64, f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let n = vals.len() as f64;
    let m = vals.iter().sum::<f64>() / n;
    let v = vals.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
    let mx = vals.iter().fold(f64::NEG_INFINITY, |a, &x| a.max(x));
    (m, v.max(0.0).sqrt(), if mx.is_finite() { mx } else { 0.0 })
}

pub(super) fn resample_knots_9(knots: &[f64]) -> [f64; 9] {
    if knots.is_empty() {
        return [0.0; 9];
    }
    if knots.len() == 1 {
        return [(knots[0].clamp(-6.0, 6.0)) / 6.0; 9];
    }
    let last = (knots.len() - 1) as f64;
    std::array::from_fn(|m| {
        let pos = (m as f64 / 8.0) * last;
        let lo = (pos.floor() as usize).min(knots.len() - 1);
        let hi = (lo + 1).min(knots.len() - 1);
        let frac = pos - (lo as f64);
        let v = knots[lo] * (1.0 - frac) + knots[hi] * frac;
        v.clamp(-6.0, 6.0) / 6.0
    })
}

pub(super) fn write_latent_stats(vals: &[f64], dst: &mut [f64]) {
    if vals.is_empty() {
        dst.fill(0.0);
        return;
    }
    let abs_vals: Vec<f64> = vals.iter().map(|v| v.abs()).collect();
    let (m_abs, _, max_abs) = mean_std_max(&abs_vals);
    dst[0] = rms_slice(vals);
    dst[1] = m_abs;
    dst[2] = max_abs;
    dst[3] = mean_std_max(vals).1;
}

//! Tensor inference for resident FP32 CPU/CUDA inputs, with no intermediate host readback.
use super::{bad, LandmarkHyperRouter, MlcV3Model};
use crate::Result;
use candle_core::{DType, Module, Tensor};

/// Batched routing products on the model device. Gate columns follow `MLC_V3_GATE_KEYS_14`.
pub struct MlcV3RoutedTensors {
    pub gates: Tensor,
    pub task_latent: Tensor,
    pub dist_latent: Tensor,
    pub landmark_pred: Tensor,
    pub bb_pred: Tensor,
}

impl LandmarkHyperRouter {
    /// Route `[B,256]` FP32 conditioning without synchronizing or copying outputs to the host.
    pub fn forward_tensor(&self, conditioning: &Tensor, level: &str) -> Result<MlcV3RoutedTensors> {
        let (batch, width) = conditioning.dims2()?;
        if batch == 0
            || width != 256
            || conditioning.dtype() != DType::F32
            || !conditioning.device().same_device(self.small_embed.device())
        {
            return Err(bad("routing requires FP32 [B,256] on the model device"));
        }
        let level_index = match level {
            "low" | "small" | "medium" => 0u32,
            "high" => 1u32,
            "ultra" => 2u32,
            _ => return Err(bad("unsupported mlc_v3 level")),
        };
        let small = matches!(level, "low" | "small");
        let select = |sm, md, hi, ul| {
            if small {
                sm
            } else if level == "medium" {
                md
            } else if level == "high" {
                hi
            } else {
                ul
            }
        };
        let high = if matches!(level, "high" | "ultra") {
            1.0
        } else {
            0.0
        };
        let x140 = self
            .input_norm
            .forward(&conditioning.narrow(1, 0, 140)?.contiguous()?)?;
        let extra = self
            .v3_inter_norm
            .forward(&conditioning.narrow(1, 140, 116)?.contiguous()?)?;
        let x256 = Tensor::cat(&[&x140, &extra], 1)?;
        let mut embedding = self.level_embed.forward(&Tensor::full(
            level_index,
            (batch,),
            conditioning.device(),
        )?)?;
        if small {
            embedding = embedding.broadcast_add(&self.small_embed.unsqueeze(0)?)?;
        }
        let h0 = self.bb_ln2.forward(&self.bb_fc0.forward(&x140)?.silu()?)?;
        let backbone = self.bb_ln5.forward(&self.bb_fc3.forward(&h0)?.silu()?)?;
        let hidden = ((backbone + embedding)? + self.inter_attn.forward(&x256)?)?;
        let raw = self.gate_head.forward(&hidden)?;
        let raw_v3 = self.v3_gate_head.forward(&hidden)?;
        let sig = candle_nn::ops::sigmoid(&raw)?;
        let sig_v3 = candle_nn::ops::sigmoid(&raw_v3)?;
        let gate =
            |index, scale| -> Result<Tensor> { Ok(sig.narrow(1, index, 1)?.affine(scale, 0.0)?) };
        let v3 = |index, scale| -> Result<Tensor> {
            Ok(sig_v3.narrow(1, index, 1)?.affine(scale, 0.0)?)
        };
        // Stable softplus; unlike exp(x), this stays finite for large positive logits.
        let ridge = raw.narrow(1, 0, 1)?;
        let softplus = (ridge.relu()? + ridge.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?)?;
        let gates = Tensor::cat(
            &[
                v3(0, if small { 0.35 } else { 0.55 })?,
                gate(7, select(0.60, 0.68, 0.90, 0.99))?,
                v3(3, 0.40 * high)?,
                gate(2, 0.55 * high)?,
                gate(1, select(0.50, 0.55, 0.65, 0.65))?,
                gate(6, select(0.25, 0.35, 0.65, 1.00))?,
                gate(9, select(0.35, 0.32, 0.20, 0.12))?,
                gate(3, 0.60 * high)?,
                v3(2, 0.45 * high)?,
                gate(8, select(0.25, 0.35, 0.75, 1.00))?,
                gate(4, 0.45 * high)?,
                softplus.affine(1.0, select(0.45, 0.35, 0.15, 0.08))?,
                v3(1, if small { 0.30 } else { 0.50 })?,
                gate(5, select(0.15, 0.20, 0.70, 1.00))?,
            ],
            1,
        )?;
        Ok(MlcV3RoutedTensors {
            gates,
            task_latent: self.task_latent_head.forward(&hidden)?.tanh()?,
            dist_latent: self.dist_latent_head.forward(&hidden)?.tanh()?,
            landmark_pred: self.landmark_recon_head.forward(&hidden)?.tanh()?,
            bb_pred: self.bb_recon_head.forward(&hidden)?.tanh()?,
        })
    }
}

impl MlcV3Model {
    /// `([B,256], [B,64,17]) -> [B,629]`, FP32/High only, on the model device.
    ///
    /// Inputs and all routing/column-attention intermediates stay on that device. Callers
    /// may retain the returned tensor for downstream GPU fusion or explicitly copy it to CPU.
    /// Non-finite output coordinates become positive zero, as in the host feature API.
    pub fn encode_features_629_tensor(
        &self,
        dataset_features: &Tensor,
        col_landmarks: &Tensor,
        level: &str,
    ) -> Result<Tensor> {
        if level != "high" {
            return Err(bad("629D encoding requires High tier"));
        }
        let routed = self.router.forward_tensor(dataset_features, level)?;
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

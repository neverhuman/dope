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
    fn forward_hidden_tensor(&self, conditioning: &Tensor, level: &str) -> Result<Tensor> {
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
        Ok(((backbone + embedding)? + self.inter_attn.forward(&x256)?)?)
    }

    /// Route `[B,256]` FP32 conditioning to `(task_latent, dist_latent)` (`([B,128], [B,128])`)
    /// without evaluating the gate or reconstruction heads.
    pub fn forward_latents_tensor(
        &self,
        conditioning: &Tensor,
        level: &str,
    ) -> Result<(Tensor, Tensor)> {
        let hidden = self.forward_hidden_tensor(conditioning, level)?;
        Ok((
            self.task_latent_head.forward(&hidden)?.tanh()?,
            self.dist_latent_head.forward(&hidden)?.tanh()?,
        ))
    }

    /// Route `[B,256]` FP32 conditioning without synchronizing or copying outputs to the host.
    pub fn forward_tensor(&self, conditioning: &Tensor, level: &str) -> Result<MlcV3RoutedTensors> {
        let hidden = self.forward_hidden_tensor(conditioning, level)?;
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
    /// `[B,256] -> [B,512]`, FP32/High only, on the model device.
    ///
    /// Computes the canonical 512-coordinate prefix
    /// (`[dataset_features (256), task_latent (128), dist_latent (128)]`) bit-identically to
    /// `&encode_features_629_tensor(...)[.., ..512]`, without evaluating the router
    /// gate/reconstruction heads or `target_net` column attention.
    pub fn encode_descriptor_512_tensor(
        &self,
        dataset_features: &Tensor,
        level: &str,
    ) -> Result<Tensor> {
        if level != "high" {
            return Err(bad("512D encoding requires High tier"));
        }
        let (task_latent, dist_latent) = self
            .router
            .forward_latents_tensor(dataset_features, level)?;
        let output = Tensor::cat(&[dataset_features, &task_latent, &dist_latent], 1)?;
        let finite = output.abs()?.le(f32::MAX as f64)?;
        Ok(finite.where_cond(&output, &output.zeros_like()?)?)
    }

    /// `([B,256], [B,64,17]) -> [B,512]`, FP32/High only, on the model device.
    ///
    /// Validates the exact two-input `[B,256]` and `[B,64,17]` contract of
    /// [`Self::encode_features_629_tensor`] and returns the `[B,512]` prefix
    /// (`[dataset_features (256), task_latent (128), dist_latent (128)]`) without evaluating
    /// the discarded trailing 117 coordinates.
    pub fn encode_features_512_tensor(
        &self,
        dataset_features: &Tensor,
        col_landmarks: &Tensor,
        level: &str,
    ) -> Result<Tensor> {
        if level != "high" {
            return Err(bad("512D encoding requires High tier"));
        }
        let (batch, width) = dataset_features.dims2()?;
        if batch == 0
            || width != 256
            || dataset_features.dtype() != DType::F32
            || !dataset_features.device().same_device(self.device())
        {
            return Err(bad("routing requires FP32 [B,256] on the model device"));
        }
        if col_landmarks.dims() != [batch, 64, 17]
            || col_landmarks.dtype() != DType::F32
            || !col_landmarks
                .device()
                .same_device(dataset_features.device())
        {
            return Err(bad(
                "512D encoding requires matching FP32 [B,256] and [B,64,17] tensors on the model device",
            ));
        }
        self.encode_descriptor_512_tensor(dataset_features, level)
    }

    /// Validate `dataset_features: [256]` and `col_landmarks: [64, 17]`, upload only
    /// `dataset_features: [1, 256]` onto `self.device()`, execute resident
    /// [`Self::encode_descriptor_512_tensor`], and copy the final `[512]` vector back to the host.
    pub fn encode_conditioning_512(
        &self,
        dataset_features: &[f32],
        col_landmarks: &[[f32; 17]],
        level: &str,
    ) -> Result<Vec<f32>> {
        if dataset_features.len() != 256 || col_landmarks.len() != 64 {
            return Err(bad("conditioning requires [256] and [64, 17] inputs"));
        }
        let descriptor = Tensor::from_slice(dataset_features, (1, 256), self.device())?;
        let out = self.encode_descriptor_512_tensor(&descriptor, level)?;
        Ok(out
            .to_device(&candle_core::Device::Cpu)?
            .squeeze(0)?
            .to_vec1::<f32>()?)
    }

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

    /// Upload `dataset_features: [1, 256]` and `col_landmarks: [1, 64, 17]` onto `self.device()`,
    /// execute resident [`Self::encode_features_629_tensor`], and copy the final `[629]` vector
    /// back to the host once.
    pub fn encode_conditioning_629(
        &self,
        dataset_features: &[f32],
        col_landmarks: &[[f32; 17]],
        level: &str,
    ) -> Result<Vec<f32>> {
        if dataset_features.len() != 256 || col_landmarks.len() != 64 {
            return Err(bad("conditioning requires [256] and [64, 17] inputs"));
        }
        let dev = self.device();
        let descriptor = Tensor::from_slice(dataset_features, (1, 256), dev)?;
        let mut columns = Vec::with_capacity(64 * 17);
        for row in col_landmarks {
            columns.extend_from_slice(row);
        }
        let columns = Tensor::from_vec(columns, (1, 64, 17), dev)?;
        let out = self.encode_features_629_tensor(&descriptor, &columns, level)?;
        Ok(out
            .to_device(&candle_core::Device::Cpu)?
            .squeeze(0)?
            .to_vec1::<f32>()?)
    }
}

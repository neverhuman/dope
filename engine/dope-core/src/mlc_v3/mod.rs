//! 100% Pure-Rust SafeTensors inference engine for `DOPE-MLC-v3` (`MultiLevelLandmarkModel`).
//!
//! Loads all 56 `F32` tensors (`444,168` parameters) of `LandmarkHyperRouter`,
//! `ConditionalHyperTargetNet`, and `CovariateRefinementNet` from a `.safetensors` file
//! without Python, and executes routing, target conditioning, covariate refinement,
//! and `DML3`/`CSC1` wire decoding and synthetic table generation on CPU.

pub mod inflate;
pub mod nets;
pub mod sample;
pub mod scaler;
pub mod wire;

pub use nets::{
    ConditionalHyperTargetNet, CovariateRefinementNet, MlcV3ForwardCheckInput,
    MlcV3ForwardCheckOutput,
};
pub use sample::{
    build_enriched_basis, sample_synthetic_table, sample_synthetic_table_with_draws,
    FixedSampleDraws, MlcV3SampledTable,
};
pub use scaler::{Csc1Sidecar, NumericScalerSpec, TargetScalerSpec};
pub use wire::{
    decode_dml2_core, Dml3Envelope, MlcV3Encoding, ObliviousTree, TargetBundle, VectorLeafTree,
    TIER_BYTE_CAPS, TIER_COORD_WIDTHS,
};

use crate::levels::digest;
use crate::{DopeError, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

pub const MLC_V3_EXPECTED_TENSORS: usize = 56;
pub const MLC_V3_EXPECTED_PARAMETERS: usize = 444_168;
pub const MLC_V3_MAX_COLS: usize = 64;

fn bad(msg: &str) -> DopeError {
    DopeError::Artifact(msg.into())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlcV3WeightInventory {
    pub tensor_count: usize,
    pub parameter_count: usize,
    pub byte_len: usize,
    pub sha256: String,
}

pub(crate) struct MlcV3Weights {
    tensors: HashMap<String, Tensor>,
    used: BTreeSet<String>,
    pub inventory: MlcV3WeightInventory,
}

impl MlcV3Weights {
    pub(crate) fn from_bytes(raw: &[u8]) -> Result<Self> {
        if raw.is_empty() || raw.len() > 33_554_432 {
            return Err(bad("mlc_v3 safetensors byte size out of bounds"));
        }
        let st = safetensors::SafeTensors::deserialize(raw)
            .map_err(|_| bad("mlc_v3 safetensors deserialize failed"))?;
        if st.len() != MLC_V3_EXPECTED_TENSORS {
            return Err(bad("mlc_v3 safetensors tensor count mismatch"));
        }
        let mut parameters = 0usize;
        for (_, view) in st.tensors() {
            if view.dtype() != safetensors::Dtype::F32 || view.shape().len() > 2 {
                return Err(bad("mlc_v3 safetensors tensor dtype or rank invalid"));
            }
            if !view
                .data()
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite())
            {
                return Err(bad("mlc_v3 safetensors contains non-finite weight"));
            }
            parameters = parameters.saturating_add(view.data().len() / 4);
        }
        if parameters != MLC_V3_EXPECTED_PARAMETERS {
            return Err(bad("mlc_v3 safetensors parameter count mismatch"));
        }
        let tensor_count = st.len();
        let tensors = candle_core::safetensors::load_buffer(raw, &Device::Cpu)?;
        Ok(Self {
            tensors,
            used: BTreeSet::new(),
            inventory: MlcV3WeightInventory {
                tensor_count,
                parameter_count: parameters,
                byte_len: raw.len(),
                sha256: digest(raw),
            },
        })
    }

    pub(crate) fn get(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| bad("missing mlc_v3 tensor"))?;
        if t.dims() != shape || t.dtype() != DType::F32 {
            return Err(bad("mlc_v3 tensor shape mismatch"));
        }
        self.used.insert(name.to_string());
        Ok(t.clone())
    }

    pub(crate) fn linear(&mut self, prefix: &str, in_dim: usize, out_dim: usize) -> Result<Linear> {
        Ok(Linear::new(
            self.get(&format!("{prefix}.weight"), &[out_dim, in_dim])?,
            Some(self.get(&format!("{prefix}.bias"), &[out_dim])?),
        ))
    }

    pub(crate) fn norm(&mut self, prefix: &str, dim: usize) -> Result<LayerNorm> {
        Ok(LayerNorm::new(
            self.get(&format!("{prefix}.weight"), &[dim])?,
            self.get(&format!("{prefix}.bias"), &[dim])?,
            1e-5,
        ))
    }

    pub(crate) fn finish(self) -> Result<MlcV3WeightInventory> {
        if self.used.len() != self.tensors.len() {
            return Err(bad("unaccounted mlc_v3 safetensors weights"));
        }
        Ok(self.inventory)
    }
}

pub struct InterLandmarkAttentionBlock {
    token_proj: Linear,
    qkv: Linear,
    bilinear_u: Linear,
    bilinear_v: Linear,
    out_proj: Linear,
    norm: LayerNorm,
}

impl InterLandmarkAttentionBlock {
    fn load(w: &mut MlcV3Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            token_proj: w.linear(&format!("{prefix}.token_proj"), 32, 32)?,
            qkv: w.linear(&format!("{prefix}.qkv"), 32, 96)?,
            bilinear_u: w.linear(&format!("{prefix}.bilinear_u"), 256, 128)?,
            bilinear_v: w.linear(&format!("{prefix}.bilinear_v"), 256, 128)?,
            out_proj: w.linear(&format!("{prefix}.out_proj"), 384, 256)?,
            norm: w.norm(&format!("{prefix}.norm"), 256)?,
        })
    }

    fn forward(&self, x256: &Tensor) -> Result<Tensor> {
        let b = x256.dim(0)?;
        let tok_in = x256.reshape((b, 8, 32))?;
        let tok = self.token_proj.forward(&tok_in)?.silu()?;
        let qkv = self.qkv.forward(&tok)?;
        let q = qkv.narrow(2, 0, 32)?.contiguous()?;
        let k = qkv.narrow(2, 32, 32)?.contiguous()?;
        let v = qkv.narrow(2, 64, 32)?.contiguous()?;
        let scores = q
            .matmul(&k.transpose(1, 2)?.contiguous()?)?
            .affine(1.0 / (32.0f64).sqrt(), 0.0)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended = (attn.matmul(&v)? + &tok)?.reshape((b, 256))?;
        let bilin =
            (self.bilinear_u.forward(x256)?.tanh()? * self.bilinear_v.forward(x256)?.silu()?)?;
        let cat = Tensor::cat(&[&attended, &bilin], 1)?;
        Ok(self.norm.forward(&self.out_proj.forward(&cat)?)?)
    }
}

pub struct LandmarkHyperRouter {
    level_embed: Embedding,
    small_embed: Tensor,
    input_norm: LayerNorm,
    v3_inter_norm: LayerNorm,
    bb_fc0: Linear,
    bb_ln2: LayerNorm,
    bb_fc3: Linear,
    bb_ln5: LayerNorm,
    inter_attn: InterLandmarkAttentionBlock,
    gate_head: Linear,
    v3_gate_head: Linear,
    task_latent_head: Linear,
    dist_latent_head: Linear,
    landmark_recon_head: Linear,
    bb_recon_head: Linear,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlcV3RoutedOutput {
    pub level: String,
    pub gates: BTreeMap<String, f64>,
    pub task_latent: Vec<f32>,
    pub dist_latent: Vec<f32>,
    pub landmark_pred: Vec<f32>,
    pub bb_pred: Vec<f32>,
}

impl LandmarkHyperRouter {
    fn load(w: &mut MlcV3Weights) -> Result<Self> {
        Ok(Self {
            level_embed: Embedding::new(w.get("router.level_embed.weight", &[3, 256])?, 256),
            small_embed: w.get("router.small_embed", &[256])?,
            input_norm: w.norm("router.input_norm", 140)?,
            v3_inter_norm: w.norm("router.v3_inter_norm", 116)?,
            bb_fc0: w.linear("router.backbone.0", 140, 256)?,
            bb_ln2: w.norm("router.backbone.2", 256)?,
            bb_fc3: w.linear("router.backbone.3", 256, 256)?,
            bb_ln5: w.norm("router.backbone.5", 256)?,
            inter_attn: InterLandmarkAttentionBlock::load(w, "router.inter_attn")?,
            gate_head: w.linear("router.gate_head", 256, 10)?,
            v3_gate_head: w.linear("router.v3_gate_head", 256, 4)?,
            task_latent_head: w.linear("router.task_latent_head", 256, 128)?,
            dist_latent_head: w.linear("router.dist_latent_head", 256, 128)?,
            landmark_recon_head: w.linear("router.landmark_recon_head", 256, 23)?,
            bb_recon_head: w.linear("router.bb_recon_head", 256, 16)?,
        })
    }

    pub fn forward(&self, desc_vec: &[f32], level: &str) -> Result<MlcV3RoutedOutput> {
        if desc_vec.len() < 140 {
            return Err(bad("conditioning vector requires at least 140 coordinates"));
        }
        let lvl_idx = match level {
            "low" | "small" | "medium" => 0u32,
            "high" => 1u32,
            "ultra" => 2u32,
            _ => return Err(bad("unsupported mlc_v3 level")),
        };
        let x140 = self.input_norm.forward(&Tensor::from_slice(
            &desc_vec[..140],
            (1, 140),
            &Device::Cpu,
        )?)?;
        let x_extra = if desc_vec.len() >= 256 {
            self.v3_inter_norm.forward(&Tensor::from_slice(
                &desc_vec[140..256],
                (1, 116),
                &Device::Cpu,
            )?)?
        } else {
            Tensor::zeros((1, 116), DType::F32, &Device::Cpu)?
        };
        let x256 = Tensor::cat(&[&x140, &x_extra], 1)?;
        let mut lvl_emb =
            self.level_embed
                .forward(&Tensor::from_slice(&[lvl_idx], (1,), &Device::Cpu)?)?;
        let is_small = matches!(level, "low" | "small");
        if is_small {
            lvl_emb = (lvl_emb + self.small_embed.unsqueeze(0)?)?;
        }
        let h0 = self.bb_ln2.forward(&self.bb_fc0.forward(&x140)?.silu()?)?;
        let h_bb = self.bb_ln5.forward(&self.bb_fc3.forward(&h0)?.silu()?)?;
        let h = ((h_bb + lvl_emb)? + self.inter_attn.forward(&x256)?)?;
        let rg = self.gate_head.forward(&h)?.squeeze(0)?.to_vec1::<f32>()?;
        let rv = self
            .v3_gate_head
            .forward(&h)?
            .squeeze(0)?
            .to_vec1::<f32>()?;
        let sig = |x: f32| 1.0 / (1.0 + (-x as f64).exp());
        let softplus = |x: f32| (1.0 + (x as f64).exp()).ln();
        let is_high = if matches!(level, "high" | "ultra") {
            1.0
        } else {
            0.0
        };
        let sel = |sm: f64, md: f64, hi: f64, ul: f64| {
            if is_small {
                sm
            } else if level == "medium" {
                md
            } else if level == "high" {
                hi
            } else {
                ul
            }
        };
        let mut gates = BTreeMap::new();
        gates.insert(
            "ridge_alpha".into(),
            softplus(rg[0]) + sel(0.45, 0.35, 0.15, 0.08),
        );
        gates.insert(
            "monotone_w".into(),
            sig(rg[1]) * sel(0.50, 0.55, 0.65, 0.65),
        );
        gates.insert("hinge_w".into(), sig(rg[2]) * 0.55 * is_high);
        gates.insert("pair_w".into(), sig(rg[3]) * 0.60 * is_high);
        gates.insert("rff_w".into(), sig(rg[4]) * 0.45 * is_high);
        gates.insert("tree_w".into(), sig(rg[5]) * sel(0.15, 0.20, 0.70, 1.00));
        gates.insert("neural_w".into(), sig(rg[6]) * sel(0.25, 0.35, 0.65, 1.00));
        gates.insert(
            "copula_retention".into(),
            sig(rg[7]) * sel(0.60, 0.68, 0.90, 0.99),
        );
        gates.insert(
            "regime_sharpness".into(),
            sig(rg[8]) * sel(0.25, 0.35, 0.75, 1.00),
        );
        gates.insert(
            "noise_scale".into(),
            sig(rg[9]) * sel(0.35, 0.32, 0.20, 0.12),
        );
        gates.insert(
            "asinh_atan_w".into(),
            sig(rv[0]) * if is_small { 0.35 } else { 0.55 },
        );
        gates.insert(
            "row_summary_w".into(),
            sig(rv[1]) * if is_small { 0.30 } else { 0.50 },
        );
        gates.insert("periodic_fract_w".into(), sig(rv[2]) * 0.45 * is_high);
        gates.insert("het_scale_w".into(), sig(rv[3]) * 0.40 * is_high);
        let head_vec = |lin: &Linear| -> Result<Vec<f32>> {
            Ok(lin.forward(&h)?.tanh()?.squeeze(0)?.to_vec1()?)
        };
        Ok(MlcV3RoutedOutput {
            level: level.to_string(),
            gates,
            task_latent: head_vec(&self.task_latent_head)?,
            dist_latent: head_vec(&self.dist_latent_head)?,
            landmark_pred: head_vec(&self.landmark_recon_head)?,
            bb_pred: head_vec(&self.bb_recon_head)?,
        })
    }
}

pub struct MlcV3Model {
    pub router: LandmarkHyperRouter,
    pub target_net: ConditionalHyperTargetNet,
    pub cov_net: CovariateRefinementNet,
    pub inventory: MlcV3WeightInventory,
}

impl MlcV3Model {
    pub fn load_safetensors(path: &Path) -> Result<Self> {
        let meta = std::fs::symlink_metadata(path)?;
        if !meta.is_file() || meta.file_type().is_symlink() {
            return Err(bad("mlc_v3 safetensors path must be a regular file"));
        }
        let raw = std::fs::read(path)?;
        Self::from_safetensors_bytes(&raw)
    }

    pub fn from_safetensors_bytes(raw: &[u8]) -> Result<Self> {
        let mut w = MlcV3Weights::from_bytes(raw)?;
        let router = LandmarkHyperRouter::load(&mut w)?;
        let target_net = ConditionalHyperTargetNet::load(&mut w)?;
        let cov_net = CovariateRefinementNet::load(&mut w)?;
        let inventory = w.finish()?;
        Ok(Self {
            router,
            target_net,
            cov_net,
            inventory,
        })
    }
}

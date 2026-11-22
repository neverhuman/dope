//! Fail-closed pure-Rust F32 SafeTensors parser (`rank <= 2`, `<= 16 MiB` hard cap).

use crate::error::{DopeError, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

pub const MAX_SAFETENSORS_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_HEADER_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug)]
pub(crate) struct Linear {
    pub(crate) out_dim: usize,
    pub(crate) in_dim: usize,
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}

impl Linear {
    pub(crate) fn forward_vec(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_dim);
        let mut out = self.bias.clone();
        for o in 0..self.out_dim {
            let row = &self.weight[o * self.in_dim..(o + 1) * self.in_dim];
            let mut acc = out[o];
            for i in 0..self.in_dim {
                acc += row[i] * x[i];
            }
            out[o] = acc;
        }
        out
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LayerNorm {
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}

impl LayerNorm {
    pub(crate) fn forward_vec(&self, x: &[f32]) -> Vec<f32> {
        let d = x.len();
        let mean: f32 = x.iter().sum::<f32>() / d.max(1) as f32;
        let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d.max(1) as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        (0..d)
            .map(|i| (x[i] - mean) * inv * self.weight[i] + self.bias[i])
            .collect()
    }
}

#[derive(Deserialize)]
struct StTensorMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub(crate) type TensorMap = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

pub(crate) fn parse_f32_rank2_safetensors(bytes: &[u8]) -> Result<TensorMap> {
    if bytes.len() < 8 || bytes.len() > MAX_SAFETENSORS_BYTES {
        return Err(DopeError::Artifact(
            "single_enc_v7 safetensors size out of bounds".into(),
        ));
    }
    let hdr_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if hdr_len == 0 || hdr_len > MAX_HEADER_BYTES || 8 + hdr_len > bytes.len() {
        return Err(DopeError::Artifact(
            "single_enc_v7 safetensors header bounds invalid".into(),
        ));
    }
    let hdr_str = std::str::from_utf8(&bytes[8..8 + hdr_len])
        .map_err(|_| DopeError::Artifact("single_enc_v7 safetensors non-utf8 header".into()))?;
    let raw_map: BTreeMap<String, serde_json::Value> = serde_json::from_str(hdr_str)
        .map_err(|_| DopeError::Artifact("single_enc_v7 safetensors invalid json".into()))?;
    let data = &bytes[8 + hdr_len..];
    let mut tensors = BTreeMap::new();
    for (k, v) in raw_map {
        if k == "__metadata__" {
            continue;
        }
        let meta: StTensorMeta = serde_json::from_value(v)
            .map_err(|_| DopeError::Artifact(format!("invalid tensor meta: {k}")))?;
        if meta.dtype != "F32" {
            return Err(DopeError::Artifact(format!(
                "single_enc_v7 requires F32 dtype, got {}",
                meta.dtype
            )));
        }
        if meta.shape.len() > 2 {
            return Err(DopeError::Artifact(format!(
                "single_enc_v7 tensor rank > 2: {k}"
            )));
        }
        let [lo, hi] = meta.data_offsets;
        let n_elem: usize = meta.shape.iter().copied().product();
        if lo > hi || hi > data.len() || hi - lo != n_elem.saturating_mul(4) {
            return Err(DopeError::Artifact(format!(
                "single_enc_v7 tensor byte range mismatch: {k}"
            )));
        }
        let mut vals = Vec::with_capacity(n_elem);
        for chunk in data[lo..hi].chunks_exact(4) {
            let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if !f.is_finite() {
                return Err(DopeError::Artifact(format!(
                    "single_enc_v7 non-finite weight in {k}"
                )));
            }
            vals.push(f);
        }
        tensors.insert(k, (meta.shape, vals));
    }
    Ok(tensors)
}

pub(crate) fn take_linear(
    tensors: &TensorMap,
    prefix: &str,
    exp_in: Option<usize>,
    exp_out: Option<usize>,
) -> Result<Linear> {
    let (ws, w) = tensors
        .get(&format!("{prefix}.weight"))
        .ok_or_else(|| DopeError::Artifact(format!("missing {prefix}.weight")))?;
    let (bs, b) = tensors
        .get(&format!("{prefix}.bias"))
        .ok_or_else(|| DopeError::Artifact(format!("missing {prefix}.bias")))?;
    if ws.len() != 2 || bs.len() != 1 || ws[0] != bs[0] {
        return Err(DopeError::Artifact(format!("bad linear shape: {prefix}")));
    }
    if let Some(ei) = exp_in {
        if ws[1] != ei {
            return Err(DopeError::Artifact(format!("bad in_dim for {prefix}")));
        }
    }
    if let Some(eo) = exp_out {
        if ws[0] != eo {
            return Err(DopeError::Artifact(format!("bad out_dim for {prefix}")));
        }
    }
    Ok(Linear {
        out_dim: ws[0],
        in_dim: ws[1],
        weight: w.clone(),
        bias: b.clone(),
    })
}

pub(crate) fn take_layernorm(tensors: &TensorMap, prefix: &str, d: usize) -> Result<LayerNorm> {
    let (ws, w) = tensors
        .get(&format!("{prefix}.weight"))
        .ok_or_else(|| DopeError::Artifact(format!("missing {prefix}.weight")))?;
    let (bs, b) = tensors
        .get(&format!("{prefix}.bias"))
        .ok_or_else(|| DopeError::Artifact(format!("missing {prefix}.bias")))?;
    if ws != &[d] || bs != &[d] {
        return Err(DopeError::Artifact(format!(
            "bad layernorm shape: {prefix}"
        )));
    }
    Ok(LayerNorm {
        weight: w.clone(),
        bias: b.clone(),
    })
}

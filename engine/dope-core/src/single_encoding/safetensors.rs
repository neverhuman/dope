//! Fail-closed pure-Rust SafeTensors parser (`rank <= 2`; V7 strictly F32 `<= 16 MiB`,
//! V8 supports F32 and BF16 up to `1 GiB` for large neural encoders/decoders).

use crate::error::{DopeError, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

pub const MAX_SAFETENSORS_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_V8_SAFETENSORS_BYTES: usize = 1024 * 1024 * 1024;
pub const MAX_HEADER_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_TENSORS: usize = 4096;
/// 1-byte `U8` SafeTensors capability guard tensor emitted for `active-rank-v1` checkpoints.
/// Older `v3.0.0` loaders ignore `__metadata__` but reject non-F32 tensors, forcing fail-closed refusal.
pub const ACTIVE_RANK_V1_MARKER_TENSOR: &str = "__dope_dse8_decoder_active_rank_v1__";

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
pub(crate) type SafeTensorsMetadata = BTreeMap<String, String>;

pub(crate) fn parse_f32_rank2_safetensors(bytes: &[u8]) -> Result<TensorMap> {
    parse_rank2_safetensors_inner(bytes, MAX_SAFETENSORS_BYTES, false, "single_enc_v7")
        .map(|(t, _, _)| t)
}

/// Fail-closed `SingleEncV8` SafeTensors parser (`F32` or `BF16`, `rank <= 2`, `<= 1 GiB`)
/// returning tensors, `__metadata__` map, and whether the `active-rank-v1` `U8` guard tensor is present.
pub(crate) fn parse_rank2_safetensors_v8_with_meta(
    bytes: &[u8],
) -> Result<(TensorMap, SafeTensorsMetadata, bool)> {
    parse_rank2_safetensors_inner(bytes, MAX_V8_SAFETENSORS_BYTES, true, "single_enc_v8")
}

struct ValidTensorPlan {
    name: String,
    shape: Vec<usize>,
    elem_bytes: usize,
    lo: usize,
    hi: usize,
    n_elem: usize,
}

fn parse_rank2_safetensors_inner(
    bytes: &[u8],
    max_bytes: usize,
    allow_bf16: bool,
    tag: &str,
) -> Result<(TensorMap, SafeTensorsMetadata, bool)> {
    if bytes.len() < 8 || bytes.len() > max_bytes {
        return Err(DopeError::Artifact(format!(
            "{tag} safetensors size out of bounds"
        )));
    }
    let hdr_len_u64 = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let hdr_len = usize::try_from(hdr_len_u64)
        .map_err(|_| DopeError::Artifact(format!("{tag} safetensors header bounds invalid")))?;
    let payload_start = 8usize
        .checked_add(hdr_len)
        .ok_or_else(|| DopeError::Artifact(format!("{tag} safetensors header bounds invalid")))?;
    if hdr_len == 0 || hdr_len > MAX_HEADER_BYTES || payload_start > bytes.len() {
        return Err(DopeError::Artifact(format!(
            "{tag} safetensors header bounds invalid"
        )));
    }
    let hdr_str = std::str::from_utf8(&bytes[8..payload_start])
        .map_err(|_| DopeError::Artifact(format!("{tag} safetensors non-utf8 header")))?;
    let raw_map: BTreeMap<String, serde_json::Value> = serde_json::from_str(hdr_str)
        .map_err(|_| DopeError::Artifact(format!("{tag} safetensors invalid json")))?;
    if raw_map.len() > MAX_TENSORS + 1 {
        return Err(DopeError::Artifact(format!(
            "{tag} safetensors tensor count exceeds limit"
        )));
    }
    let data = &bytes[payload_start..];
    // Pass 1: validate every dtype, rank, shape, non-overlapping byte range, and total decoded
    // element budget BEFORE allocating any tensor buffers.
    let mut plans = Vec::with_capacity(raw_map.len());
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(raw_map.len());
    let mut metadata: SafeTensorsMetadata = BTreeMap::new();
    let mut has_active_rank_marker = false;
    let mut total_elems: usize = 0;
    for (k, v) in raw_map {
        if k == "__metadata__" {
            let obj = v.as_object().ok_or_else(|| {
                DopeError::Artifact(format!("{tag} safetensors metadata must be an object"))
            })?;
            for (mk, mv) in obj {
                let s = mv.as_str().ok_or_else(|| {
                    DopeError::Artifact(format!("{tag} safetensors metadata value must be string"))
                })?;
                metadata.insert(mk.clone(), s.to_string());
            }
            continue;
        }
        let meta: StTensorMeta = serde_json::from_value(v)
            .map_err(|_| DopeError::Artifact(format!("invalid tensor meta: {k}")))?;
        if allow_bf16 && k == ACTIVE_RANK_V1_MARKER_TENSOR {
            let [lo, hi] = meta.data_offsets;
            if meta.dtype != "U8"
                || meta.shape != [1]
                || lo >= hi
                || hi > data.len()
                || hi - lo != 1
                || data[lo] != 1u8
            {
                return Err(DopeError::Artifact(format!(
                    "{tag} invalid active-rank-v1 capability marker"
                )));
            }
            ranges.push((lo, hi));
            has_active_rank_marker = true;
            continue;
        }
        if k.starts_with("__dope_dse8_decoder_") {
            return Err(DopeError::Artifact(format!(
                "{tag} unknown decoder capability marker: {k}"
            )));
        }
        let elem_bytes = match meta.dtype.as_str() {
            "F32" => 4usize,
            "BF16" if allow_bf16 => 2usize,
            other => {
                let expect = if allow_bf16 { "F32 or BF16" } else { "F32" };
                return Err(DopeError::Artifact(format!(
                    "{tag} requires {expect} dtype, got {other}"
                )));
            }
        };
        if meta.shape.len() > 2 {
            return Err(DopeError::Artifact(format!("{tag} tensor rank > 2: {k}")));
        }
        let [lo, hi] = meta.data_offsets;
        let n_elem: usize = meta
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| DopeError::Artifact(format!("{tag} tensor shape overflow: {k}")))?;
        let exp_bytes = n_elem
            .checked_mul(elem_bytes)
            .ok_or_else(|| DopeError::Artifact(format!("{tag} tensor size overflow: {k}")))?;
        if lo > hi || hi > data.len() || hi - lo != exp_bytes {
            return Err(DopeError::Artifact(format!(
                "{tag} tensor byte range mismatch: {k}"
            )));
        }
        total_elems = total_elems
            .checked_add(n_elem)
            .ok_or_else(|| DopeError::Artifact(format!("{tag} total tensor size overflow")))?;
        if hi > lo {
            ranges.push((lo, hi));
        }
        plans.push(ValidTensorPlan {
            name: k,
            shape: meta.shape,
            elem_bytes,
            lo,
            hi,
            n_elem,
        });
    }
    if plans.len() > MAX_TENSORS {
        return Err(DopeError::Artifact(format!(
            "{tag} safetensors tensor count exceeds limit"
        )));
    }
    ranges.sort_unstable();
    for w in ranges.windows(2) {
        if w[0].1 > w[1].0 {
            return Err(DopeError::Artifact(format!(
                "{tag} overlapping safetensors data_offsets"
            )));
        }
    }
    let max_decoded_bytes = if allow_bf16 {
        max_bytes.saturating_mul(2)
    } else {
        max_bytes
    };
    let decoded_bytes = total_elems
        .checked_mul(4)
        .ok_or_else(|| DopeError::Artifact(format!("{tag} decoded buffer size overflow")))?;
    if decoded_bytes > max_decoded_bytes {
        return Err(DopeError::Artifact(format!(
            "{tag} decoded tensor bytes exceed limit"
        )));
    }

    // Pass 2: fallible per-tensor allocation (`try_reserve_exact`) + finite-value decoding.
    let mut tensors = BTreeMap::new();
    for plan in plans {
        let mut vals: Vec<f32> = Vec::new();
        vals.try_reserve_exact(plan.n_elem).map_err(|_| {
            DopeError::Artifact(format!("{tag} allocation failed for tensor: {}", plan.name))
        })?;
        if plan.elem_bytes == 4 {
            for chunk in data[plan.lo..plan.hi].chunks_exact(4) {
                let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if !f.is_finite() {
                    return Err(DopeError::Artifact(format!(
                        "{tag} non-finite weight in {}",
                        plan.name
                    )));
                }
                vals.push(f);
            }
        } else {
            for chunk in data[plan.lo..plan.hi].chunks_exact(2) {
                let bits = (u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16;
                let f = f32::from_bits(bits);
                if !f.is_finite() {
                    return Err(DopeError::Artifact(format!(
                        "{tag} non-finite BF16 weight in {}",
                        plan.name
                    )));
                }
                vals.push(f);
            }
        }
        tensors.insert(plan.name, (plan.shape, vals));
    }
    Ok((tensors, metadata, has_active_rank_marker))
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

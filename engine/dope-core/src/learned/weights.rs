//! Read only fixed local weight members and account for every tensor.
use super::manifest::{bad, check, WeightFile};
use crate::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

pub struct Weights {
    tensors: HashMap<String, Tensor>,
    used: BTreeSet<String>,
}
impl Weights {
    pub fn load(root: &Path, component: &str, record: &WeightFile) -> Result<Self> {
        check(
            record.file == format!("{component}.safetensors")
                && record.bytes <= 536870912
                && record.parameters <= 100000000,
            "learned weight envelope exceeded",
        )?;
        let path = root.join(&record.file);
        let metadata = std::fs::symlink_metadata(&path)?;
        check(
            metadata.is_file()
                && !metadata.file_type().is_symlink()
                && metadata.len() == record.bytes as u64,
            "learned weight file mismatch",
        )?;
        let raw = std::fs::read(path)?;
        check(
            crate::levels::digest(&raw) == record.sha256,
            "learned weight digest mismatch",
        )?;
        let (_, header) = safetensors::SafeTensors::read_metadata(&raw)
            .map_err(|_| bad("learned weight header"))?;
        check(
            header.metadata().as_ref().is_none_or(|m| m.is_empty()),
            "unexpected learned weight metadata",
        )?;
        let st = safetensors::SafeTensors::deserialize(&raw)
            .map_err(|_| bad("learned weight tensor table"))?;
        check(st.len() <= 2048, "too many learned weight tensors")?;
        let mut parameters = 0usize;
        for (_, tensor) in st.tensors() {
            check(
                tensor.dtype() == safetensors::Dtype::F32 && tensor.shape().len() <= 2,
                "learned weight dtype or rank",
            )?;
            parameters = parameters
                .checked_add(tensor.data().len() / 4)
                .ok_or_else(|| bad("learned parameter overflow"))?;
            check(
                parameters <= 100000000
                    && tensor
                        .data()
                        .chunks_exact(4)
                        .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
                "nonfinite or excessive learned weights",
            )?;
        }
        check(
            parameters == record.parameters,
            "learned parameter inventory mismatch",
        )?;
        let tensors = candle_core::safetensors::load_buffer(&raw, &Device::Cpu)?;
        Ok(Self {
            tensors,
            used: BTreeSet::new(),
        })
    }
    pub fn get(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| bad("missing learned weight"))?;
        check(
            tensor.dims() == shape && tensor.dtype() == DType::F32,
            "learned weight shape mismatch",
        )?;
        self.used.insert(name.into());
        Ok(tensor.clone())
    }
    pub fn linear(&mut self, prefix: &str, input: usize, output: usize) -> Result<Linear> {
        Ok(Linear::new(
            self.get(&format!("{prefix}.weight"), &[output, input])?,
            Some(self.get(&format!("{prefix}.bias"), &[output])?),
        ))
    }
    pub fn norm(&mut self, prefix: &str, width: usize) -> Result<LayerNorm> {
        Ok(LayerNorm::new(
            self.get(&format!("{prefix}.weight"), &[width])?,
            self.get(&format!("{prefix}.bias"), &[width])?,
            1e-5,
        ))
    }
    pub fn embedding(&mut self, prefix: &str, count: usize, width: usize) -> Result<Embedding> {
        Ok(Embedding::new(
            self.get(&format!("{prefix}.weight"), &[count, width])?,
            width,
        ))
    }
    pub fn finish(self) -> Result<()> {
        check(
            self.used.len() == self.tensors.len(),
            "unaccounted learned weight tensors",
        )
    }
}

//! `CSC1` compact scaler sidecar decoder and raw-domain inverse transforms for `DOPE-MLC-v3`.

use crate::{DopeError, Result};
use half::f16;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub(crate) fn bad(msg: &str) -> DopeError {
    DopeError::Artifact(msg.into())
}

pub(crate) struct Cursor<'a> {
    pub(crate) buf: &'a [u8],
    pub(crate) off: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, off: 0 }
    }
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .off
            .checked_add(n)
            .ok_or_else(|| bad("wire cursor overflow"))?;
        if end > self.buf.len() {
            return Err(bad("truncated mlc_v3 wire stream"));
        }
        let s = &self.buf[self.off..end];
        self.off = end;
        Ok(s)
    }
    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub(crate) fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub(crate) fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub(crate) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(crate) fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(crate) fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(crate) fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub(crate) fn f16_vec(&mut self, count: usize) -> Result<Vec<f64>> {
        let bytes = self.take(
            count
                .checked_mul(2)
                .ok_or_else(|| bad("f16 count overflow"))?,
        )?;
        Ok(bytes
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f64())
            .collect())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NumericScalerSpec {
    pub slot: usize,
    pub transform_kind: String,
    pub center: f64,
    pub scale: f64,
    pub post_mean: f64,
    pub post_std: f64,
    pub missing_rate: f64,
    pub blend_quantile_weight: f64,
    pub quantiles: [f64; 9],
}

impl NumericScalerSpec {
    pub fn inverse_transform_scalar(&self, z: f64) -> f64 {
        let z_clamped = (z / 6.0).clamp(-0.999_999, 0.999_999);
        let z_unclipped = 6.0 * z_clamped.atanh();
        match self.transform_kind.as_str() {
            "constant" => self.center,
            "binary" => {
                let raw = self.center + z_unclipped * self.scale;
                let (q0, q1) = (self.quantiles[0], self.quantiles[8]);
                if raw >= 0.5 * (q0 + q1) {
                    q1
                } else {
                    q0
                }
            }
            "signed_log1p_hybrid" => {
                let scaled = self.post_mean + z_unclipped * self.post_std;
                self.center + scaled.signum() * scaled.abs().exp_m1() * self.scale
            }
            "asinh_hybrid" => {
                let scaled = self.post_mean + z_unclipped * self.post_std;
                self.center + scaled.sinh() * self.scale
            }
            _ => self.post_mean + z_unclipped * self.post_std,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TargetScalerSpec {
    pub task: String,
    pub mean: f64,
    pub std: f64,
    pub median: f64,
    pub iqr: f64,
    pub class_prior: f64,
    pub class_labels: Vec<f64>,
    pub class_priors: Vec<f64>,
}

impl TargetScalerSpec {
    pub fn inverse_transform_scalar(&self, y_scaled: f64) -> f64 {
        if self.task == "multiclass" && !self.class_labels.is_empty() {
            let idx = y_scaled
                .round()
                .clamp(0.0, (self.class_labels.len() - 1) as f64) as usize;
            return self.class_labels[idx];
        }
        if self.task == "classification" {
            return if y_scaled >= 0.5 { 1.0 } else { 0.0 };
        }
        y_scaled * self.std + self.mean
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Csc1Sidecar {
    pub task: String,
    pub p_use: usize,
    pub kinds: Vec<String>,
    pub numeric_scalers: BTreeMap<usize, NumericScalerSpec>,
    pub target_scaler: TargetScalerSpec,
}

impl Csc1Sidecar {
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        let mut c = Cursor::new(buf);
        let t_id = c.u8()?;
        let k_cls = c.u8()? as usize;
        let p_use = c.u16()? as usize;
        let is_compact = c.u8()? != 0;
        let task = match t_id {
            1 => "classification",
            2 => "multiclass",
            _ => "regression",
        }
        .to_string();
        let (mean, std, median, iqr, c_prior) = (
            c.f32()? as f64,
            (c.f32()? as f64).max(1e-9),
            c.f32()? as f64,
            (c.f32()? as f64).max(1e-9),
            c.f32()? as f64,
        );
        let mut class_labels = Vec::with_capacity(k_cls);
        for _ in 0..k_cls {
            class_labels.push(c.f32()? as f64);
        }
        let class_priors = if k_cls > 0 {
            c.f16_vec(k_cls)?
        } else {
            Vec::new()
        };
        let mut kinds = vec!["numeric".to_string(); p_use];
        let mut numeric_scalers = BTreeMap::new();
        for (j, kind_slot) in kinds.iter_mut().enumerate().take(p_use) {
            let k_id = c.u8()?;
            let s6 = c.f16_vec(6)?;
            let q9 = if is_compact {
                let q3 = c.f16_vec(3)?;
                let xs = [0.01, 0.50, 0.99];
                let grid = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];
                let mut arr = [0.0f64; 9];
                for (idx, &xq) in grid.iter().enumerate() {
                    arr[idx] = if xq <= xs[1] {
                        q3[0] + (xq - xs[0]) / (xs[1] - xs[0]) * (q3[1] - q3[0])
                    } else {
                        q3[1] + (xq - xs[1]) / (xs[2] - xs[1]) * (q3[2] - q3[1])
                    };
                }
                arr
            } else {
                let qv = c.f16_vec(9)?;
                qv.try_into().unwrap()
            };
            if k_id == 5 {
                *kind_slot = "categorical".to_string();
            } else {
                let tk = match k_id {
                    0 => "constant",
                    1 => "binary",
                    2 => "signed_log1p_hybrid",
                    4 => "asinh_hybrid",
                    _ => "quantile_robust_hybrid",
                }
                .to_string();
                numeric_scalers.insert(
                    j,
                    NumericScalerSpec {
                        slot: j,
                        transform_kind: tk,
                        center: s6[0],
                        scale: s6[1].max(1e-9),
                        post_mean: s6[2],
                        post_std: s6[3].max(1e-9),
                        missing_rate: s6[4],
                        blend_quantile_weight: s6[5],
                        quantiles: q9,
                    },
                );
            }
        }
        Ok(Self {
            task: task.clone(),
            p_use,
            kinds,
            numeric_scalers,
            target_scaler: TargetScalerSpec {
                task,
                mean,
                std,
                median,
                iqr,
                class_prior: c_prior,
                class_labels,
                class_priors,
            },
        })
    }

    pub fn inverse_transform(
        &self,
        x_scaled: &[Vec<f64>],
        y_scaled: &[f64],
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let x_raw = x_scaled
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, &val)| {
                        self.numeric_scalers
                            .get(&j)
                            .map_or(val, |sc| sc.inverse_transform_scalar(val))
                    })
                    .collect()
            })
            .collect();
        let y_raw = y_scaled
            .iter()
            .map(|&yv| self.target_scaler.inverse_transform_scalar(yv))
            .collect();
        (x_raw, y_raw)
    }
}

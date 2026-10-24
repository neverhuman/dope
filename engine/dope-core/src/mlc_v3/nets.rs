//! `ConditionalHyperTargetNet` and `CovariateRefinementNet` for `DOPE-MLC-v3` SafeTensors inference.

use super::{MlcV3RoutedOutput, MlcV3Weights, MLC_V3_MAX_COLS};
use crate::Result;
use candle_core::{Device, IndexOp, Module, Tensor};
use candle_nn::{LayerNorm, Linear};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub struct ConditionalHyperTargetNet {
    film_gamma: Linear,
    film_beta: Linear,
    fc1: Linear,
    fc2: Linear,
    out_head: Linear,
    col_gate_head: Linear,
}

impl ConditionalHyperTargetNet {
    pub(crate) fn load(w: &mut MlcV3Weights) -> Result<Self> {
        Ok(Self {
            film_gamma: w.linear("target_net.film_gamma", 128, 64)?,
            film_beta: w.linear("target_net.film_beta", 128, 64)?,
            fc1: w.linear("target_net.fc1", 192, 128)?,
            fc2: w.linear("target_net.fc2", 128, 128)?,
            out_head: w.linear("target_net.out_head", 128, 1)?,
            col_gate_head: w.linear("target_net.col_gate_head", 145, 1)?,
        })
    }

    pub fn compute_column_gates(
        &self,
        task_lat: &[f32],
        col_lm: Option<&[Vec<f32>]>,
        x_pad: Option<&[Vec<f32>]>,
    ) -> Result<Vec<f32>> {
        let mut c_pad = vec![0.0f32; MLC_V3_MAX_COLS * 17];
        if let Some(lm) = col_lm {
            for (j, row) in lm.iter().take(MLC_V3_MAX_COLS).enumerate() {
                for (k, &v) in row.iter().take(17).enumerate() {
                    c_pad[j * 17 + k] = v;
                }
            }
        } else if let Some(xp) = x_pad {
            let n = xp.len();
            if n > 1 {
                let nf = n as f32;
                for j in 0..MLC_V3_MAX_COLS {
                    let xm = xp.iter().map(|r| r[j]).sum::<f32>() / nf;
                    let xs = (xp.iter().map(|r| (r[j] - xm).powi(2)).sum::<f32>() / nf)
                        .sqrt()
                        .max(1e-6);
                    let mut acc = [0.0f32; 17];
                    acc[0] = xm;
                    acc[1] = xs;
                    let (mut zmin, mut zmax) = (f32::INFINITY, f32::NEG_INFINITY);
                    for r in xp {
                        let z = ((r[j] - xm) / xs).clamp(-6.0, 6.0);
                        acc[2] += z.powi(3);
                        acc[3] += z.powi(4);
                        acc[4] += z.abs();
                        acc[5] += if z > 0.0 { 1.0 } else { 0.0 };
                        acc[6] += z.asinh();
                        acc[7] += z.atan();
                        acc[8] += z.tanh();
                        acc[9] += (z - 0.5).max(0.0);
                        acc[10] += (-z - 0.5).max(0.0);
                        acc[11] += (z * 1.5).sin();
                        acc[12] += (z * 1.5).cos();
                        acc[13] += z - z.floor();
                        zmin = zmin.min(z);
                        zmax = zmax.max(z);
                        acc[16] += if z.abs() < 0.1 { 1.0 } else { 0.0 };
                    }
                    acc[2] = (acc[2] / nf) / 6.0;
                    acc[3] = ((acc[3] / nf) - 3.0).clamp(-3.0, 20.0) / 10.0;
                    for item in acc.iter_mut().take(14).skip(4) {
                        *item /= nf;
                    }
                    acc[14] = zmin / 6.0;
                    acc[15] = zmax / 6.0;
                    acc[16] /= nf;
                    c_pad[j * 17..(j + 1) * 17].copy_from_slice(&acc);
                }
            }
        }
        let mut cat_in = Vec::with_capacity(MLC_V3_MAX_COLS * 145);
        for j in 0..MLC_V3_MAX_COLS {
            cat_in.extend_from_slice(&task_lat[..128]);
            cat_in.extend_from_slice(&c_pad[j * 17..(j + 1) * 17]);
        }
        let tin = Tensor::from_vec(cat_in, (MLC_V3_MAX_COLS, 145), &Device::Cpu)?;
        let raw_g = self
            .col_gate_head
            .forward(&tin)?
            .squeeze(1)?
            .to_vec1::<f32>()?;
        Ok(raw_g.into_iter().map(|g| 1.0 + 0.25 * g.tanh()).collect())
    }

    pub fn forward(
        &self,
        x_pad: &[Vec<f32>],
        task_lat: &[f32],
        p_use: usize,
        col_lm: Option<&[Vec<f32>]>,
    ) -> Result<Vec<f32>> {
        let n = x_pad.len();
        let c_gate = self.compute_column_gates(task_lat, col_lm, Some(x_pad))?;
        let t_ten = Tensor::from_slice(&task_lat[..128], (1, 128), &Device::Cpu)?;
        let gamma = self
            .film_gamma
            .forward(&t_ten)?
            .tanh()?
            .squeeze(0)?
            .to_vec1::<f32>()?;
        let beta = self
            .film_beta
            .forward(&t_ten)?
            .tanh()?
            .squeeze(0)?
            .to_vec1::<f32>()?;
        let mut fc1_in = Vec::with_capacity(n * 192);
        for row in x_pad {
            for j in 0..MLC_V3_MAX_COLS {
                let v = if j < p_use {
                    row[j] * c_gate[j] * (1.0 + 0.3 * gamma[j]) + 0.1 * beta[j]
                } else {
                    0.0
                };
                fc1_in.push(v);
            }
            fc1_in.extend_from_slice(&task_lat[..128]);
        }
        let h = self
            .fc1
            .forward(&Tensor::from_vec(fc1_in, (n, 192), &Device::Cpu)?)?
            .silu()?;
        let h2 = (self.fc2.forward(&h)?.silu()? + &h)?;
        Ok(self.out_head.forward(&h2)?.squeeze(1)?.to_vec1::<f32>()?)
    }
}

pub struct CovariateRefinementNet {
    fc0: Linear,
    ln2: LayerNorm,
    fc3: Linear,
}

impl CovariateRefinementNet {
    pub(crate) fn load(w: &mut MlcV3Weights) -> Result<Self> {
        Ok(Self {
            fc0: w.linear("cov_net.net.0", 192, 128)?,
            ln2: w.norm("cov_net.net.2", 128)?,
            fc3: w.linear("cov_net.net.3", 128, 64)?,
        })
    }

    pub fn forward(
        &self,
        x_pad: &[Vec<f32>],
        dist_lat: &[f32],
        p_use: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let n = x_pad.len();
        let mut net_in = Vec::with_capacity(n * 192);
        for row in x_pad {
            net_in.extend_from_slice(&row[..MLC_V3_MAX_COLS]);
            net_in.extend_from_slice(&dist_lat[..128]);
        }
        let h = self.ln2.forward(
            &self
                .fc0
                .forward(&Tensor::from_vec(net_in, (n, 192), &Device::Cpu)?)?
                .silu()?,
        )?;
        let delta = self.fc3.forward(&h)?.tanh()?;
        let mut out = Vec::with_capacity(n);
        for (i, row) in x_pad.iter().enumerate().take(n) {
            let d_row = delta.i(i)?.to_vec1::<f32>()?;
            let mut r = vec![0.0f32; MLC_V3_MAX_COLS];
            for j in 0..p_use.min(MLC_V3_MAX_COLS) {
                r[j] = row[j] + 0.15 * d_row[j];
            }
            out.push(r);
        }
        Ok(out)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlcV3ForwardCheckInput {
    pub p_use: usize,
    pub x_pad: Vec<Vec<f32>>,
    pub col_landmarks: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlcV3ForwardCheckOutput {
    pub level: String,
    pub gates: BTreeMap<String, f64>,
    pub task_latent: Vec<f32>,
    pub dist_latent: Vec<f32>,
    pub landmark_pred: Vec<f32>,
    pub bb_pred: Vec<f32>,
    pub col_gates_from_landmarks: Vec<f32>,
    pub col_gates_from_xpad: Vec<f32>,
    pub target_net_out_with_landmarks: Vec<f32>,
    pub target_net_out_from_xpad: Vec<f32>,
    pub cov_net_out: Vec<Vec<f32>>,
}

impl MlcV3ForwardCheckOutput {
    pub fn evaluate(
        routed: MlcV3RoutedOutput,
        target_net: &ConditionalHyperTargetNet,
        cov_net: &CovariateRefinementNet,
        inp: &MlcV3ForwardCheckInput,
    ) -> Result<Self> {
        let col_gates_from_landmarks = target_net.compute_column_gates(
            &routed.task_latent,
            Some(&inp.col_landmarks),
            Some(&inp.x_pad),
        )?;
        let col_gates_from_xpad =
            target_net.compute_column_gates(&routed.task_latent, None, Some(&inp.x_pad))?;
        let target_net_out_with_landmarks = target_net.forward(
            &inp.x_pad,
            &routed.task_latent,
            inp.p_use,
            Some(&inp.col_landmarks),
        )?;
        let target_net_out_from_xpad =
            target_net.forward(&inp.x_pad, &routed.task_latent, inp.p_use, None)?;
        let cov_net_out = cov_net.forward(&inp.x_pad, &routed.dist_latent, inp.p_use)?;
        Ok(Self {
            level: routed.level,
            gates: routed.gates,
            task_latent: routed.task_latent,
            dist_latent: routed.dist_latent,
            landmark_pred: routed.landmark_pred,
            bb_pred: routed.bb_pred,
            col_gates_from_landmarks,
            col_gates_from_xpad,
            target_net_out_with_landmarks,
            target_net_out_from_xpad,
            cov_net_out,
        })
    }
}

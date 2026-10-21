//! Neural auditor (MLP) and the multiclass softmax-linear auditor, trained
//! with candle on any device.

use super::Matrix;
use crate::error::Result;
use crate::rng::Rng;
use crate::target::Head;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{linear, AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

#[derive(Clone, Debug)]
pub struct Params {
    pub hidden: Vec<usize>,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub batch: usize,
    pub max_epochs: usize,
    pub patience: usize,
    pub seed: u64,
}

impl Params {
    pub fn mlp(seed: u64) -> Self {
        Self {
            hidden: vec![128, 128],
            learning_rate: 1e-3,
            weight_decay: 1e-5,
            batch: 256,
            max_epochs: 200,
            patience: 15,
            seed,
        }
    }
    pub fn softmax_linear(seed: u64) -> Self {
        Self {
            hidden: vec![],
            learning_rate: 1e-2,
            weight_decay: 1e-4,
            batch: 512,
            max_epochs: 200,
            patience: 10,
            seed,
        }
    }
}

struct Mlp {
    layers: Vec<candle_nn::Linear>,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for (i, l) in self.layers.iter().enumerate() {
            h = l.forward(&h)?;
            if i + 1 < self.layers.len() {
                h = h.relu()?;
            }
        }
        Ok(h)
    }
}

fn loss(head: Head, out: &Tensor, y: &Tensor) -> Result<Tensor> {
    Ok(match head {
        Head::Regression => (out - y)?.sqr()?.mean_all()?,
        Head::Binary => {
            let sp = (out.relu()? + ((out.abs()?.neg()?.exp()? + 1.0)?.log()?))?;
            (sp - (y * out)?)?.mean_all()?
        }
        Head::Multiclass(_) => candle_nn::loss::cross_entropy(out, y)?,
    })
}

fn tensors(
    x: &Matrix,
    rows: &[usize],
    y: &[f64],
    head: Head,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut xv = Vec::with_capacity(rows.len() * x.cols);
    for &r in rows {
        xv.extend(x.row(r).iter().map(|v| *v as f32));
    }
    let xt = Tensor::from_vec(xv, (rows.len(), x.cols), dev)?;
    let yt = match head {
        Head::Multiclass(_) => Tensor::from_vec(
            rows.iter().map(|&r| y[r] as u32).collect::<Vec<_>>(),
            rows.len(),
            dev,
        )?,
        _ => Tensor::from_vec(
            rows.iter().map(|&r| y[r] as f32).collect::<Vec<_>>(),
            (rows.len(), 1),
            dev,
        )?,
    };
    Ok((xt, yt))
}

/// Fit and predict like `gbdt::fit_predict`.
pub fn fit_predict(
    x: &Matrix,
    y: &[f64],
    head: Head,
    test: &Matrix,
    p: &Params,
    dev: &Device,
) -> Result<Vec<f64>> {
    let o = head.outputs();
    let mut idx: Vec<usize> = (0..x.rows).collect();
    let mut rng = Rng::new(p.seed ^ 0x3a1);
    rng.shuffle(&mut idx);
    let n_val = if x.rows >= 100 { x.rows / 10 } else { 0 };
    let (val, fit) = idx.split_at(n_val);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let mut dims = vec![x.cols];
    dims.extend(&p.hidden);
    dims.push(o);
    // Deterministic initialisation: candle's init uses the device RNG.
    dev.set_seed(p.seed).ok();
    let layers = (0..dims.len() - 1)
        .map(|i| linear(dims[i], dims[i + 1], vb.pp(format!("l{i}"))))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let model = Mlp { layers };
    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: p.learning_rate,
            weight_decay: p.weight_decay,
            ..Default::default()
        },
    )?;
    let (xv, yv) = if val.is_empty() {
        tensors(x, fit, y, head, dev)?
    } else {
        tensors(x, val, y, head, dev)?
    };
    let mut best = (f64::INFINITY, Vec::<Tensor>::new());
    let mut since = 0;
    let mut order: Vec<usize> = fit.to_vec();
    for _epoch in 0..p.max_epochs {
        rng.shuffle(&mut order);
        for chunk in order.chunks(p.batch) {
            let (xb, yb) = tensors(x, chunk, y, head, dev)?;
            let l = loss(head, &model.forward(&xb)?, &yb)?;
            opt.backward_step(&l)?;
        }
        let l = loss(head, &model.forward(&xv)?, &yv)?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()? as f64;
        if l < best.0 - 1e-7 {
            best = (
                l,
                varmap
                    .all_vars()
                    .iter()
                    .map(|v| v.as_tensor().copy())
                    .collect::<candle_core::Result<_>>()?,
            );
            since = 0;
        } else {
            since += 1;
            if since >= p.patience {
                break;
            }
        }
    }
    for (v, t) in varmap.all_vars().iter().zip(&best.1) {
        v.set(t)?;
    }
    let all: Vec<usize> = (0..test.rows).collect();
    let mut out = Vec::with_capacity(test.rows * o);
    for chunk in all.chunks(8192) {
        let (xt, _) = tensors(test, chunk, &vec![0.0; test.rows], Head::Regression, dev)?;
        let z = model.forward(&xt)?;
        let z = match head {
            Head::Regression => z,
            Head::Binary => candle_nn::ops::sigmoid(&z)?,
            Head::Multiclass(_) => candle_nn::ops::softmax(&z, D::Minus1)?,
        };
        out.extend(
            z.flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(|v| v as f64),
        );
    }
    Ok(out)
}

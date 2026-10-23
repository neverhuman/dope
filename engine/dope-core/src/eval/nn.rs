//! Neural auditor (MLP) and the multiclass softmax-linear auditor, trained
//! with candle on any device.

use super::Matrix;
use crate::error::{DopeError, Result};
use crate::rng::Rng;
use crate::target::Head;
use candle_core::{DType, Device, Tensor, Var, D};
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

/// Historical device-initialized auditor. CPU device seeding is unsupported;
/// use `fit_predict_seeded_cpu` for new reproducible development controls.
pub fn fit_predict(
    x: &Matrix,
    y: &[f64],
    head: Head,
    test: &Matrix,
    p: &Params,
    dev: &Device,
) -> Result<Vec<f64>> {
    fit_predict_impl(x, y, head, test, p, dev, false)
}

/// Version of the explicit host-generated initialization and ordered parameter recipe.
pub const SEEDED_CPU_RECIPE: &str = "candle-mlp-host-init-v1";

/// A bounded CPU-only auditor initialized independently of Candle's global device RNG.
/// Pin the recipe, parameters, source, representation and runtime in comparison receipts.
/// Bitwise replay is tested on one runtime, not promised across CPU architectures.
pub fn fit_predict_seeded_cpu(
    x: &Matrix,
    y: &[f64],
    head: Head,
    test: &Matrix,
    p: &Params,
) -> Result<Vec<f64>> {
    let shape_ok = |m: &Matrix| {
        m.rows > 0
            && m.rows <= 50_000
            && m.cols > 0
            && m.cols <= 1024
            && m.rows.checked_mul(m.cols) == Some(m.data.len())
            && m.data.len() <= 2_000_000
            && m.data
                .iter()
                .all(|v| v.is_finite() && (*v as f32).is_finite())
    };
    let labels_ok = y.iter().all(|v| {
        v.is_finite()
            && match head {
                Head::Regression => (*v as f32).is_finite(),
                Head::Binary => *v == 0.0 || *v == 1.0,
                Head::Multiclass(k) => v.fract() == 0.0 && *v >= 0.0 && *v < k as f64,
            }
    });
    if !shape_ok(x)
        || !shape_ok(test)
        || x.cols != test.cols
        || y.len() != x.rows
        || !labels_ok
        || head.outputs() == 0
        || head.outputs() > 64
        || p.hidden.len() > 4
        || p.hidden.iter().any(|&n| n == 0 || n > 512)
        || p.batch == 0
        || p.batch > 4096
        || p.max_epochs == 0
        || p.max_epochs > 200
        || p.patience == 0
        || p.patience > 200
        || !p.learning_rate.is_finite()
        || p.learning_rate <= 0.0
        || p.learning_rate > 0.1
        || !p.weight_decay.is_finite()
        || !(0.0..=1.0).contains(&p.weight_decay)
    {
        return Err(DopeError::Input(
            "seeded neural auditor contract invalid".into(),
        ));
    }
    let predictions = fit_predict_impl(x, y, head, test, p, &Device::Cpu, true)?;
    if predictions.iter().any(|v| !v.is_finite()) {
        return Err(DopeError::Input(
            "seeded neural auditor nonfinite result".into(),
        ));
    }
    Ok(predictions)
}

fn seeded_layers(dims: &[usize], vars: &VarMap, seed: u64) -> Result<Vec<candle_nn::Linear>> {
    let mut layers = Vec::new();
    for (i, pair) in dims.windows(2).enumerate() {
        let (input, output) = (pair[0], pair[1]);
        let scale = (2.0 / input as f64).sqrt();
        let weight: Vec<f32> = (0..input * output)
            .map(|j| (crate::rng::normal(seed, 2 * i as u64, j as u64) * scale) as f32)
            .collect();
        let bound = 1.0 / (input as f64).sqrt();
        let bias: Vec<f32> = (0..output)
            .map(|j| {
                ((2.0 * crate::rng::uniform(seed, 2 * i as u64 + 1, j as u64) - 1.0) * bound) as f32
            })
            .collect();
        let weight = Var::from_vec(weight, (output, input), &Device::Cpu)?;
        let bias = Var::from_vec(bias, output, &Device::Cpu)?;
        layers.push(candle_nn::Linear::new(
            weight.as_tensor().clone(),
            Some(bias.as_tensor().clone()),
        ));
        let mut map = vars
            .data()
            .lock()
            .map_err(|_| DopeError::Device("auditor variables unavailable".into()))?;
        map.insert(format!("l{i}.weight"), weight);
        map.insert(format!("l{i}.bias"), bias);
    }
    Ok(layers)
}

fn fit_predict_impl(
    x: &Matrix,
    y: &[f64],
    head: Head,
    test: &Matrix,
    p: &Params,
    dev: &Device,
    seeded: bool,
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
    let layers = if seeded {
        seeded_layers(&dims, &varmap, p.seed)?
    } else {
        // Preserve historical device initialization, including unsupported CPU seeding.
        dev.set_seed(p.seed).ok();
        (0..dims.len() - 1)
            .map(|i| linear(dims[i], dims[i + 1], vb.pp(format!("l{i}"))))
            .collect::<candle_core::Result<Vec<_>>>()?
    };
    let model = Mlp { layers };
    let variables = if seeded {
        let map = varmap
            .data()
            .lock()
            .map_err(|_| DopeError::Device("auditor variables unavailable".into()))?;
        let mut names: Vec<_> = map.keys().collect();
        names.sort();
        names.into_iter().map(|name| map[name].clone()).collect()
    } else {
        varmap.all_vars()
    };
    let mut opt = AdamW::new(
        variables.clone(),
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
        if seeded && !l.is_finite() {
            return Err(DopeError::Input(
                "seeded neural auditor nonfinite objective".into(),
            ));
        }
        if l < best.0 - 1e-7 {
            best = (
                l,
                variables
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
    for (v, t) in variables.iter().zip(&best.1) {
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

//! Training of the residual network with candle (CPU, CUDA or Metal).
//!
//! AdamW on all parameters, with the readout periodically replaced by its
//! closed-form optimum for the current hidden layer (variable projection).
//! Early stopping on an internal validation split picks the step count, so
//! no external validation data is ever consulted.

use crate::error::Result;
use crate::rng::Rng;
use crate::target::{loss, solve_readout, Head, Net};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

#[derive(Clone, Debug)]
pub struct TrainOptions {
    pub hidden: usize,
    pub max_steps: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub batch: usize,
    pub eval_every: usize,
    pub patience: usize,
    pub seed: u64,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            hidden: 16,
            max_steps: 1500,
            learning_rate: 1e-2,
            weight_decay: 1e-4,
            batch: 2048,
            eval_every: 50,
            patience: 400,
            seed: 0,
        }
    }
}

pub(crate) fn init_net(s: usize, h: usize, o: usize, seed: u64) -> Net {
    let mut rng = Rng::new(seed ^ 0x7a11);
    let scale = 5.0 / (s.max(1) as f64).sqrt();
    Net {
        inputs: s,
        hidden: h,
        outputs: o,
        w: (0..h * s).map(|_| rng.normal() * scale).collect(),
        b: (0..h).map(|_| rng.uniform() * 2.0 - 1.0).collect(),
        beta: vec![0.0; o * s],
        v: vec![0.0; o * h],
        a: vec![0.0; o],
    }
}

/// Cosine decay from `learning_rate` to `learning_rate / 20` over the run.
pub fn learning_rate(opts: &TrainOptions, step: usize) -> f64 {
    let t = (step as f64 / opts.max_steps.max(1) as f64).min(1.0);
    let floor = opts.learning_rate / 20.0;
    floor + 0.5 * (opts.learning_rate - floor) * (1.0 + (std::f64::consts::PI * t).cos())
}

fn to_tensor(v: &[f64], shape: (usize, usize), dev: &Device) -> Result<Tensor> {
    let f: Vec<f32> = v.iter().map(|x| *x as f32).collect();
    Ok(Tensor::from_vec(f, shape, dev)?)
}

fn read(var: &Var) -> Result<Vec<f64>> {
    Ok(var
        .as_tensor()
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|v| v as f64)
        .collect())
}

struct Params {
    w: Var,
    b: Var,
    beta: Var,
    v: Var,
    a: Var,
}

impl Params {
    fn new(net: &Net, dev: &Device) -> Result<Self> {
        let (s, h, o) = (net.inputs, net.hidden, net.outputs);
        Ok(Self {
            w: Var::from_tensor(&to_tensor(&net.w, (h, s), dev)?)?,
            b: Var::from_tensor(&to_tensor(&net.b, (1, h), dev)?)?,
            beta: Var::from_tensor(&to_tensor(&net.beta, (o, s), dev)?)?,
            v: Var::from_tensor(&to_tensor(&net.v, (o, h), dev)?)?,
            a: Var::from_tensor(&to_tensor(&net.a, (1, o), dev)?)?,
        })
    }

    fn snapshot(&self, template: &Net) -> Result<Net> {
        Ok(Net {
            w: read(&self.w)?,
            b: read(&self.b)?,
            beta: read(&self.beta)?,
            v: read(&self.v)?,
            a: read(&self.a)?,
            ..template.clone()
        })
    }

    fn load_readout(&self, net: &Net, dev: &Device) -> Result<()> {
        let (s, h, o) = (net.inputs, net.hidden, net.outputs);
        self.beta.set(&to_tensor(&net.beta, (o, s), dev)?)?;
        self.v.set(&to_tensor(&net.v, (o, h), dev)?)?;
        self.a.set(&to_tensor(&net.a, (1, o), dev)?)?;
        Ok(())
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = x
            .matmul(&self.w.as_tensor().t()?)?
            .broadcast_add(self.b.as_tensor())?
            .tanh()?;
        let linear = x
            .matmul(&self.beta.as_tensor().t()?)?
            .broadcast_add(self.a.as_tensor())?;
        Ok((linear + hidden.matmul(&self.v.as_tensor().t()?)?)?)
    }

    fn vars(&self) -> Vec<Var> {
        vec![
            self.w.clone(),
            self.b.clone(),
            self.beta.clone(),
            self.v.clone(),
            self.a.clone(),
        ]
    }
}

fn batch_loss(head: Head, out: &Tensor, y: &Tensor) -> Result<Tensor> {
    Ok(match head {
        Head::Regression => (out - y)?.sqr()?.mean_all()?,
        Head::Binary => {
            // softplus(z) − y z, written stably.
            let sp = (out.relu()? + ((out.abs()?.neg()?.exp()? + 1.0)?.log()?))?;
            (sp - (y * out)?)?.mean_all()?
        }
        Head::Multiclass(_) => candle_nn::loss::cross_entropy(out, y)?,
    })
}

fn targets(head: Head, y: &[f64], dev: &Device) -> Result<Tensor> {
    Ok(match head {
        Head::Multiclass(_) => Tensor::from_vec(
            y.iter().map(|v| *v as u32).collect::<Vec<_>>(),
            y.len(),
            dev,
        )?,
        _ => to_tensor(y, (y.len(), 1), dev)?,
    })
}

/// Train on `(xc, y)` (row-major, centred inputs) with early stopping on
/// `val`: the hand-vectorized trainer on CPU, candle on CUDA/Metal.
pub fn train(
    head: Head,
    inputs: usize,
    fit: (&[f64], &[f64]),
    val: (&[f64], &[f64]),
    opts: &TrainOptions,
    dev: &Device,
) -> Result<Net> {
    if dev.is_cpu() {
        Ok(crate::train_cpu::train_cpu(head, inputs, fit, val, opts))
    } else {
        train_device(head, inputs, fit, val, opts, dev)
    }
}

/// The candle trainer (any device). On CPU it is the parity reference.
pub fn train_device(
    head: Head,
    inputs: usize,
    fit: (&[f64], &[f64]),
    val: (&[f64], &[f64]),
    opts: &TrainOptions,
    dev: &Device,
) -> Result<Net> {
    let (fx, fy) = fit;
    let n = fy.len();
    let mut net = init_net(inputs, opts.hidden, head.outputs(), opts.seed);
    if n == 0 || inputs == 0 {
        return Ok(net);
    }
    solve_readout(&mut net, head, fx, fy);
    if opts.hidden == 0 || opts.max_steps == 0 {
        return Ok(net);
    }
    let params = Params::new(&net, dev)?;
    let x_all = to_tensor(fx, (n, inputs), dev)?;
    let y_all = targets(head, fy, dev)?;
    let mut opt = AdamW::new(
        params.vars(),
        ParamsAdamW {
            lr: opts.learning_rate,
            weight_decay: opts.weight_decay,
            ..Default::default()
        },
    )?;
    let (vx, vy) = if val.1.is_empty() { fit } else { val };
    let score = |net: &Net| loss(head, &net.forward(vx, vy.len()), vy);
    let mut best = (score(&net), net.clone());
    let mut since_best = 0;
    let mut rng = Rng::new(opts.seed ^ 0xba7c);
    for step in 1..=opts.max_steps {
        let (xb, yb) = if n > opts.batch {
            let ids: Vec<u32> = (0..opts.batch).map(|_| rng.below(n) as u32).collect();
            let ids = Tensor::from_vec(ids, opts.batch, dev)?;
            (x_all.index_select(&ids, 0)?, y_all.index_select(&ids, 0)?)
        } else {
            (x_all.clone(), y_all.clone())
        };
        let out = params.forward(&xb)?;
        opt.set_learning_rate(learning_rate(opts, step));
        opt.backward_step(&batch_loss(head, &out, &yb)?)?;
        if step % opts.eval_every == 0 || step == opts.max_steps {
            let mut current = params.snapshot(&net)?;
            if head != Head::Regression || step % (2 * opts.eval_every) == 0 {
                solve_readout(&mut current, head, fx, fy);
                params.load_readout(&current, dev)?;
            }
            let l = score(&current);
            if l < best.0 - 1e-9 {
                best = (l, current);
                since_best = 0;
            } else {
                since_best += opts.eval_every;
                if since_best >= opts.patience {
                    break;
                }
            }
        }
    }
    net = best.1;
    Ok(net)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::null_loss;

    #[test]
    fn learns_a_nonlinear_function_on_cpu() {
        let n = 600;
        let mut rng = Rng::new(1);
        let xc: Vec<f64> = (0..n * 3).map(|_| rng.uniform() - 0.5).collect();
        let y: Vec<f64> = (0..n)
            .map(|r| (4.0 * xc[3 * r]).sin() * xc[3 * r + 1] + 0.3 * xc[3 * r + 2])
            .collect();
        let (fit_n, val_n) = (500, 100);
        let opts = TrainOptions {
            hidden: 8,
            max_steps: 600,
            ..Default::default()
        };
        let net = train(
            Head::Regression,
            3,
            (&xc[..fit_n * 3], &y[..fit_n]),
            (&xc[fit_n * 3..], &y[fit_n..]),
            &opts,
            &Device::Cpu,
        )
        .unwrap();
        let out = net.forward(&xc[fit_n * 3..], val_n);
        let r2 = 1.0
            - loss(Head::Regression, &out, &y[fit_n..]) / null_loss(Head::Regression, &y[fit_n..]);
        assert!(r2 > 0.8, "r2 {r2}");
    }

    #[test]
    fn candle_and_native_trainers_agree() {
        let n = 800;
        let mut rng = Rng::new(2);
        let xc: Vec<f64> = (0..n * 2).map(|_| rng.uniform() - 0.5).collect();
        let y: Vec<f64> = (0..n)
            .map(|r| (3.0 * xc[2 * r]).tanh() + xc[2 * r + 1].powi(2))
            .collect();
        let opts = TrainOptions {
            hidden: 6,
            max_steps: 400,
            batch: 256,
            ..Default::default()
        };
        let fit = (&xc[..1400], &y[..700]);
        let val = (&xc[1400..], &y[700..]);
        let a = train_device(Head::Regression, 2, fit, val, &opts, &Device::Cpu).unwrap();
        let b = crate::train_cpu::train_cpu(Head::Regression, 2, fit, val, &opts);
        let la = loss(Head::Regression, &a.forward(val.0, 100), val.1);
        let lb = loss(Head::Regression, &b.forward(val.0, 100), val.1);
        let null = null_loss(Head::Regression, val.1);
        assert!(
            (la - lb).abs() < 0.05 * null,
            "candle {la} vs native {lb} (null {null})"
        );
    }
}

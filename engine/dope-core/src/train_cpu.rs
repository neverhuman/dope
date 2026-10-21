//! Hand-vectorized CPU trainer for the residual network.
//!
//! Same algorithm, initialisation, batches, AdamW update and schedule as
//! the candle trainer in `train.rs` (which runs on CUDA/Metal), with
//! closed-form gradients for the one-hidden-layer network; ~10–20× faster
//! than a generic autograd graph at these tiny sizes. A parity test keeps
//! the two in agreement.

use crate::rng::Rng;
use crate::target::{loss, solve_readout, Head, Net};
use crate::train::{init_net, TrainOptions};
use rayon::prelude::*;

struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
}

impl Adam {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }
    /// candle-compatible AdamW (decoupled weight decay).
    fn step(&mut self, theta: &mut [f64], grad: &[f64], lr: f64, wd: f64) {
        const B1: f64 = 0.9;
        const B2: f64 = 0.999;
        const EPS: f64 = 1e-8;
        self.t += 1;
        let (c1, c2) = (1.0 - B1.powi(self.t), 1.0 - B2.powi(self.t));
        for i in 0..theta.len() {
            theta[i] *= 1.0 - lr * wd;
            self.m[i] = B1 * self.m[i] + (1.0 - B1) * grad[i];
            self.v[i] = B2 * self.v[i] + (1.0 - B2) * grad[i] * grad[i];
            theta[i] -= lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + EPS);
        }
    }
}

/// Flattened parameter layout: w | b | beta | v | a.
fn flatten(net: &Net) -> Vec<f64> {
    [&net.w, &net.b, &net.beta, &net.v, &net.a]
        .iter()
        .flat_map(|t| t.iter().copied())
        .collect()
}

fn unflatten(net: &mut Net, theta: &[f64]) {
    let mut o = 0;
    for t in [
        &mut net.w,
        &mut net.b,
        &mut net.beta,
        &mut net.v,
        &mut net.a,
    ] {
        let n = t.len();
        t.copy_from_slice(&theta[o..o + n]);
        o += n;
    }
}

/// Fast f32 tanh (Eigen's rational approximation, |error| < 1e-6 after the
/// clamp); vectorizes, unlike libm's tanh.
#[inline(always)]
pub fn fast_tanh(x: f32) -> f32 {
    let x = x.clamp(-7.905_311, 7.905_311);
    let x2 = x * x;
    let p = ((((((-2.760_768_5e-16f32 * x2 + 2.000_187_9e-13) * x2 - 8.604_672e-11) * x2
        + 5.122_297e-8)
        * x2
        + 1.485_722_4e-5)
        * x2
        + 6.372_619_3e-4)
        * x2
        + 4.893_524_6e-3)
        * x;
    let q = ((1.198_258_4e-6f32 * x2 + 1.185_347_1e-4) * x2 + 2.268_434_6e-3) * x2 + 4.893_525e-3;
    p / q
}

/// Mean-loss gradient over `rows` of row-major `x` (n × s), in f32 with
/// transposed weights so the inner loops run over contiguous hidden units.
fn gradient(net: &Net, head: Head, x: &[f32], y: &[f64], rows: &[usize]) -> Vec<f64> {
    let (s, h, o) = (net.inputs, net.hidden, net.outputs);
    let np = net.parameter_count();
    let inv = 1.0 / rows.len().max(1) as f32;
    let (ow, ob, obeta, ov, oa) = (
        0,
        h * s,
        h * s + h,
        h * s + h + o * s,
        h * s + h + o * s + o * h,
    );
    // wt[i*h + j] = W[j][i]; vt[j*o + k] = v[k][j]; bt[i*o + k] = beta[k][i]
    let wt: Vec<f32> = (0..s * h)
        .map(|q| net.w[(q % h) * s + q / h] as f32)
        .collect();
    let vt: Vec<f32> = (0..h * o)
        .map(|q| net.v[(q % o) * h + q / o] as f32)
        .collect();
    let bt: Vec<f32> = (0..s * o)
        .map(|q| net.beta[(q % o) * s + q / o] as f32)
        .collect();
    let b: Vec<f32> = net.b.iter().map(|v| *v as f32).collect();
    let a: Vec<f32> = net.a.iter().map(|v| *v as f32).collect();
    let partial = rows
        .par_chunks(512)
        .map(|chunk| {
            let mut gwt = vec![0f32; s * h];
            let mut gb = vec![0f32; h];
            let mut gbt = vec![0f32; s * o];
            let mut gvt = vec![0f32; h * o];
            let mut ga = vec![0f32; o];
            let mut hid = vec![0f32; h];
            let mut dpre = vec![0f32; h];
            let mut out = vec![0f32; o];
            for &r in chunk {
                let xr = &x[r * s..(r + 1) * s];
                hid.copy_from_slice(&b);
                for i in 0..s {
                    let xi = xr[i];
                    let wrow = &wt[i * h..(i + 1) * h];
                    for j in 0..h {
                        hid[j] += xi * wrow[j];
                    }
                }
                for v in hid.iter_mut() {
                    *v = fast_tanh(*v);
                }
                out.copy_from_slice(&a);
                for i in 0..s {
                    for k in 0..o {
                        out[k] += xr[i] * bt[i * o + k];
                    }
                }
                for j in 0..h {
                    for k in 0..o {
                        out[k] += hid[j] * vt[j * o + k];
                    }
                }
                match head {
                    Head::Regression => out[0] = 2.0 * (out[0] - y[r] as f32),
                    Head::Binary => out[0] = 1.0 / (1.0 + (-out[0]).exp()) - y[r] as f32,
                    Head::Multiclass(_) => {
                        let m = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let total: f32 = out.iter().map(|z| (z - m).exp()).sum();
                        let t = y[r] as usize;
                        for (k, v) in out.iter_mut().enumerate() {
                            *v = (*v - m).exp() / total - (t == k) as u8 as f32;
                        }
                    }
                }
                // `out` now holds dLoss/dout (unscaled).
                for k in 0..o {
                    out[k] *= inv;
                    ga[k] += out[k];
                }
                for i in 0..s {
                    for k in 0..o {
                        gbt[i * o + k] += xr[i] * out[k];
                    }
                }
                for j in 0..h {
                    let mut dh = 0f32;
                    for k in 0..o {
                        gvt[j * o + k] += hid[j] * out[k];
                        dh += out[k] * vt[j * o + k];
                    }
                    dpre[j] = dh * (1.0 - hid[j] * hid[j]);
                    gb[j] += dpre[j];
                }
                for i in 0..s {
                    let xi = xr[i];
                    let grow = &mut gwt[i * h..(i + 1) * h];
                    for j in 0..h {
                        grow[j] += xi * dpre[j];
                    }
                }
            }
            (gwt, gb, gbt, gvt, ga)
        })
        .collect::<Vec<_>>();
    let mut g = vec![0.0; np];
    for (gwt, gb, gbt, gvt, ga) in partial {
        for q in 0..s * h {
            g[ow + (q % h) * s + q / h] += gwt[q] as f64;
        }
        for j in 0..h {
            g[ob + j] += gb[j] as f64;
        }
        for q in 0..s * o {
            g[obeta + (q % o) * s + q / o] += gbt[q] as f64;
        }
        for q in 0..h * o {
            g[ov + (q % o) * h + q / o] += gvt[q] as f64;
        }
        for k in 0..o {
            g[oa + k] += ga[k] as f64;
        }
    }
    g
}

/// CPU counterpart of `train::train_device`.
pub fn train_cpu(
    head: Head,
    inputs: usize,
    fit: (&[f64], &[f64]),
    val: (&[f64], &[f64]),
    opts: &TrainOptions,
) -> Net {
    let (fx, fy) = fit;
    let n = fy.len();
    let mut net = init_net(inputs, opts.hidden, head.outputs(), opts.seed);
    if n == 0 || inputs == 0 {
        return net;
    }
    solve_readout(&mut net, head, fx, fy);
    if opts.hidden == 0 || opts.max_steps == 0 {
        return net;
    }
    let (vx, vy) = if val.1.is_empty() { fit } else { val };
    let score = |net: &Net| loss(head, &net.forward(vx, vy.len()), vy);
    let mut best = (score(&net), net.clone());
    let mut since_best = 0;
    let mut rng = Rng::new(opts.seed ^ 0xba7c);
    let mut theta = flatten(&net);
    let mut adam = Adam::new(theta.len());
    let all: Vec<usize> = (0..n).collect();
    let fx32: Vec<f32> = fx.iter().map(|v| *v as f32).collect();
    for step in 1..=opts.max_steps {
        let batch: Vec<usize> = if n > opts.batch {
            (0..opts.batch).map(|_| rng.below(n)).collect()
        } else {
            all.clone()
        };
        let g = gradient(&net, head, &fx32, fy, &batch);
        adam.step(
            &mut theta,
            &g,
            crate::train::learning_rate(opts, step),
            opts.weight_decay,
        );
        unflatten(&mut net, &theta);
        if step % opts.eval_every == 0 || step == opts.max_steps {
            if head != Head::Regression || step % (2 * opts.eval_every) == 0 {
                solve_readout(&mut net, head, fx, fy);
                theta = flatten(&net);
            }
            let l = score(&net);
            if l < best.0 - 1e-9 {
                best = (l, net.clone());
                since_best = 0;
            } else {
                since_best += opts.eval_every;
                if since_best >= opts.patience {
                    break;
                }
            }
        }
    }
    best.1
}

#[cfg(test)]
mod tests {
    use super::fast_tanh;

    #[test]
    fn fast_tanh_is_accurate() {
        for i in -2000..=2000 {
            let x = i as f32 / 100.0;
            assert!((fast_tanh(x) - x.tanh()).abs() < 2e-6, "{x}");
        }
    }
}

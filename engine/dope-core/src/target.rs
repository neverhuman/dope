//! The stored conditional model `f(x)`: a residual tanh network
//!
//! ```text
//! out(x) = a + β·x_c + v·tanh(W x_c + b),   x_c = x_S − ½
//! ```
//!
//! over a screened set `S` of inputs. Its readout `(a, β, v)` is linear in a
//! fixed basis, so it is always re-solved in closed form (variable
//! projection) — during training, and again after the hidden layer has been
//! quantized, which cancels most of the hidden-layer rounding error.

use crate::quant::QTensor;
use crate::stats::solve_spd;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Head {
    Regression,
    Binary,
    Multiclass(usize),
}

impl Head {
    pub fn outputs(self) -> usize {
        match self {
            Head::Regression | Head::Binary => 1,
            Head::Multiclass(k) => k,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Net {
    pub inputs: usize,
    pub hidden: usize,
    pub outputs: usize,
    pub w: Vec<f64>,
    pub b: Vec<f64>,
    pub beta: Vec<f64>,
    pub v: Vec<f64>,
    pub a: Vec<f64>,
}

impl Net {
    /// `[n, 1 + s + h]` basis: constant, centred inputs, hidden units.
    pub fn basis(&self, xc: &[f64], n: usize) -> Vec<f64> {
        let (s, h) = (self.inputs, self.hidden);
        let p = 1 + s + h;
        let mut out = vec![0.0; n * p];
        for r in 0..n {
            let x = &xc[r * s..(r + 1) * s];
            let row = &mut out[r * p..(r + 1) * p];
            row[0] = 1.0;
            row[1..1 + s].copy_from_slice(x);
            for j in 0..h {
                let mut z = self.b[j];
                for i in 0..s {
                    z += self.w[j * s + i] * x[i];
                }
                row[1 + s + j] = z.tanh();
            }
        }
        out
    }

    /// Raw outputs `[n, o]` (prediction for regression, logits otherwise).
    pub fn forward(&self, xc: &[f64], n: usize) -> Vec<f64> {
        let (s, h, o) = (self.inputs, self.hidden, self.outputs);
        let basis = self.basis(xc, n);
        let p = 1 + s + h;
        let mut out = vec![0.0; n * o];
        for r in 0..n {
            let row = &basis[r * p..(r + 1) * p];
            for k in 0..o {
                let mut z = self.a[k];
                for i in 0..s {
                    z += self.beta[k * s + i] * row[1 + i];
                }
                for j in 0..h {
                    z += self.v[k * h + j] * row[1 + s + j];
                }
                out[r * o + k] = z;
            }
        }
        out
    }

    pub fn parameter_count(&self) -> usize {
        self.w.len() + self.b.len() + self.beta.len() + self.v.len() + self.a.len()
    }
}

/// Mean loss: MSE (regression), logistic (binary), softmax CE (multiclass).
pub fn loss(head: Head, out: &[f64], y: &[f64]) -> f64 {
    let n = y.len().max(1);
    match head {
        Head::Regression => out.iter().zip(y).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / n as f64,
        Head::Binary => {
            out.iter()
                .zip(y)
                .map(|(z, t)| softplus(*z) - t * z)
                .sum::<f64>()
                / n as f64
        }
        Head::Multiclass(k) => {
            let mut total = 0.0;
            for (r, t) in y.iter().enumerate() {
                let row = &out[r * k..(r + 1) * k];
                total += log_sum_exp(row) - row[(*t as usize).min(k - 1)];
            }
            total / n as f64
        }
    }
}

/// Loss of the best constant predictor (the "null").
pub fn null_loss(head: Head, y: &[f64]) -> f64 {
    let n = y.len().max(1) as f64;
    match head {
        Head::Regression => {
            let m = y.iter().sum::<f64>() / n;
            y.iter().map(|t| (t - m).powi(2)).sum::<f64>() / n
        }
        Head::Binary => {
            let p = (y.iter().sum::<f64>() / n).clamp(1e-6, 1.0 - 1e-6);
            -(p * p.ln() + (1.0 - p) * (1.0 - p).ln())
        }
        Head::Multiclass(k) => {
            let mut c = vec![0.0; k];
            for t in y {
                c[(*t as usize).min(k - 1)] += 1.0;
            }
            -c.iter()
                .filter(|v| **v > 0.0)
                .map(|v| (v / n) * (v / n).ln())
                .sum::<f64>()
        }
    }
}

pub fn softplus(z: f64) -> f64 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}

pub fn log_sum_exp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

fn gram(basis: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut g = vec![0.0; p * p];
    for r in 0..n {
        let row = &basis[r * p..(r + 1) * p];
        for i in 0..p {
            let ri = row[i];
            if ri == 0.0 {
                continue;
            }
            for j in i..p {
                g[i * p + j] += ri * row[j];
            }
        }
    }
    for i in 0..p {
        for j in 0..i {
            g[i * p + j] = g[j * p + i];
        }
    }
    g
}

fn ridge_penalty(g: &mut [f64], p: usize, rel: f64) {
    let avg = (1..p).map(|i| g[i * p + i]).sum::<f64>() / (p - 1).max(1) as f64;
    for i in 1..p {
        g[i * p + i] += rel * avg.max(1e-12);
    }
}

/// Re-solve the readout in closed form for the current hidden layer.
/// Regression: ridge least squares. Binary: ridge logistic by Newton steps.
/// Multiclass readouts are left as trained.
pub fn solve_readout(net: &mut Net, head: Head, xc: &[f64], y: &[f64]) {
    if y.is_empty() {
        return;
    }
    let (s, h) = (net.inputs, net.hidden);
    // The readout has at most a few dozen coefficients: a deterministic
    // stride of READOUT_ROWS rows determines it to well below quantization noise.
    let stride = y.len().div_ceil(READOUT_ROWS).max(1);
    let rows: Vec<usize> = (0..y.len()).step_by(stride).collect();
    let n = rows.len();
    let xs: Vec<f64> = rows
        .iter()
        .flat_map(|&r| xc[r * s..(r + 1) * s].iter().copied())
        .collect();
    let y: Vec<f64> = rows.iter().map(|&r| y[r]).collect();
    let p = 1 + s + h;
    let basis = net.basis(&xs, n);
    let theta = match head {
        Head::Regression => {
            let mut g = gram(&basis, n, p);
            ridge_penalty(&mut g, p, 1e-4);
            let mut rhs = vec![0.0; p];
            for r in 0..n {
                for i in 0..p {
                    rhs[i] += basis[r * p + i] * y[r];
                }
            }
            solve_spd(&g, &rhs, p)
        }
        Head::Binary => {
            let mut theta = vec![0.0; p];
            theta[0] = net.a[0];
            theta[1..1 + s].copy_from_slice(&net.beta);
            theta[1 + s..].copy_from_slice(&net.v);
            for _ in 0..6 {
                let mut g = vec![0.0; p * p];
                let mut grad = vec![0.0; p];
                for r in 0..n {
                    let row = &basis[r * p..(r + 1) * p];
                    let z: f64 = row.iter().zip(&theta).map(|(a, b)| a * b).sum();
                    let mu = 1.0 / (1.0 + (-z).exp());
                    let wgt = (mu * (1.0 - mu)).max(1e-6);
                    for i in 0..p {
                        grad[i] += (mu - y[r]) * row[i];
                        for j in i..p {
                            g[i * p + j] += wgt * row[i] * row[j];
                        }
                    }
                }
                for i in 0..p {
                    for j in 0..i {
                        g[i * p + j] = g[j * p + i];
                    }
                }
                let avg = (1..p).map(|i| g[i * p + i]).sum::<f64>() / (p - 1).max(1) as f64;
                for i in 1..p {
                    g[i * p + i] += 1e-4 * avg.max(1e-12);
                    grad[i] += 1e-4 * avg.max(1e-12) * theta[i];
                }
                let step = solve_spd(&g, &grad, p);
                for i in 0..p {
                    theta[i] -= step[i];
                }
            }
            theta
        }
        Head::Multiclass(_) => return,
    };
    net.a[0] = theta[0];
    net.beta.copy_from_slice(&theta[1..1 + s]);
    net.v.copy_from_slice(&theta[1 + s..]);
}

/// Rows used by a closed-form readout solve.
pub const READOUT_ROWS: usize = 8192;

/// A net whose weight tensors are stored as packed integer codes.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantNet {
    pub inputs: usize,
    pub hidden: usize,
    pub outputs: usize,
    pub w: QTensor,
    pub b: QTensor,
    pub beta: QTensor,
    pub v: QTensor,
    /// Intercepts are kept at full precision (they shift every prediction).
    pub a: Vec<f32>,
}

impl QuantNet {
    pub fn net(&self) -> Net {
        Net {
            inputs: self.inputs,
            hidden: self.hidden,
            outputs: self.outputs,
            w: self.w.values(),
            b: self.b.values(),
            beta: self.beta.values(),
            v: self.v.values(),
            a: self.a.iter().map(|v| *v as f64).collect(),
        }
    }

    pub fn payload_bytes(&self) -> usize {
        let t = |q: &QTensor| crate::quant::packed_len(q.codes.len(), q.bits) + 4;
        t(&self.w) + t(&self.b) + t(&self.beta) + t(&self.v) + 4 * self.a.len()
    }
}

fn quantize_with(
    net: &Net,
    bits: [u8; 4],
    head: Head,
    xc: &[f64],
    y: &[f64],
    cache: &mut std::collections::HashMap<(u8, u8), Net>,
) -> QuantNet {
    let w = QTensor::quantize(&net.w, bits[0]);
    let b = QTensor::quantize(&net.b, bits[1]);
    // Error feedback: the readout absorbs the hidden layer's rounding.
    let refit = cache
        .entry((bits[0], bits[1]))
        .or_insert_with(|| {
            let mut r = net.clone();
            r.w = w.values();
            r.b = b.values();
            solve_readout(&mut r, head, xc, y);
            r
        })
        .clone();
    QuantNet {
        inputs: net.inputs,
        hidden: net.hidden,
        outputs: net.outputs,
        w,
        b,
        beta: QTensor::quantize(&refit.beta, bits[2]),
        v: QTensor::quantize(&refit.v, bits[3]),
        a: refit.a.iter().map(|v| *v as f32).collect(),
    }
}

/// Measured rate–distortion bit allocation.
///
/// Starting from 8 bits everywhere, repeatedly take the one-bit reduction
/// with the best bytes-saved per unit of validation loss, while the loss
/// stays within `tolerance × (null − float)` of the float model — i.e. the
/// quantized model keeps at least `1 − tolerance` of the stored gain.
pub fn allocate_bits(
    net: &Net,
    head: Head,
    fit: (&[f64], &[f64]),
    val: (&[f64], &[f64]),
    tolerance: f64,
) -> QuantNet {
    let n_val = val.1.len();
    let base = loss(head, &net.forward(val.0, n_val), val.1);
    let null = null_loss(head, val.1);
    let allowed = base + tolerance * (null - base).max(0.0) + 1e-12;
    let evaluate = |q: &QuantNet| loss(head, &q.net().forward(val.0, n_val), val.1);
    let mut bits = [8u8; 4];
    let mut cache = std::collections::HashMap::new();
    let mut best = quantize_with(net, bits, head, fit.0, fit.1, &mut cache);
    loop {
        let current = best.payload_bytes();
        let mut choice: Option<(f64, [u8; 4], QuantNet)> = None;
        for g in 0..4 {
            if bits[g] <= 2 {
                continue;
            }
            let mut trial = bits;
            trial[g] -= 1;
            let q = quantize_with(net, trial, head, fit.0, fit.1, &mut cache);
            let l = evaluate(&q);
            if l > allowed {
                continue;
            }
            let saved = current.saturating_sub(q.payload_bytes()) as f64;
            let score = saved / ((l - base).max(0.0) + 1e-9);
            if saved > 0.0 && choice.as_ref().is_none_or(|c| score > c.0) {
                choice = Some((score, trial, q));
            }
        }
        match choice {
            Some((_, trial, q)) => {
                bits = trial;
                best = q;
            }
            None => return best,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> (Vec<f64>, Vec<f64>, Net) {
        let n = 400;
        let xc: Vec<f64> = (0..n * 2)
            .map(|i| ((i * 37 % 101) as f64 / 100.0) - 0.5)
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|r| (3.0 * xc[2 * r]).sin() + 0.5 * xc[2 * r + 1])
            .collect();
        let net = Net {
            inputs: 2,
            hidden: 4,
            outputs: 1,
            w: vec![3.0, 0.1, -3.0, 0.2, 1.5, -1.0, 0.3, 2.0],
            b: vec![0.1, -0.2, 0.3, 0.0],
            beta: vec![0.0, 0.0],
            v: vec![0.0; 4],
            a: vec![0.0],
        };
        (xc, y, net)
    }

    #[test]
    fn closed_form_readout_reduces_loss() {
        let (xc, y, mut net) = toy();
        let before = loss(Head::Regression, &net.forward(&xc, y.len()), &y);
        solve_readout(&mut net, Head::Regression, &xc, &y);
        let after = loss(Head::Regression, &net.forward(&xc, y.len()), &y);
        assert!(after < 0.2 * before, "{before} -> {after}");
    }

    #[test]
    fn allocation_respects_tolerance() {
        let (xc, y, mut net) = toy();
        solve_readout(&mut net, Head::Regression, &xc, &y);
        let base = loss(Head::Regression, &net.forward(&xc, y.len()), &y);
        let null = null_loss(Head::Regression, &y);
        let q = allocate_bits(&net, Head::Regression, (&xc, &y), (&xc, &y), 0.01);
        let l = loss(Head::Regression, &q.net().forward(&xc, y.len()), &y);
        assert!(l <= base + 0.01 * (null - base) + 1e-9);
        assert!(q.w.bits < 8 || q.v.bits < 8);
    }
}

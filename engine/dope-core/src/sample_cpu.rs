//! Native CPU sampler: the same draw as the tensor program in `sample.rs`,
//! evaluated row by row in parallel. Every row reads only the program
//! (a few kilobytes, cache-resident) and its own counter-based random
//! inputs, so throughput scales with cores and needs no batching. A parity
//! test holds it to the candle path within floating-point rounding.

use crate::forest::code;
use crate::marginal::{interp_grid, Marginal, Z_GRID};
use crate::noise::Noise;
use crate::program::Program;
use crate::rng::{normal, uniform};
use crate::sample::{
    Samples, LABEL_STREAM, MISSING_STREAM, NOISE_STREAM, REGIME_STREAM, SCORE_TABLE,
};
use crate::target::Head;
use rayon::prelude::*;

/// Program pieces precomputed once per call.
struct Plan<'a> {
    p: &'a Program,
    mix: Vec<f64>,
    m: usize,
    /// Concatenated column index of each feature (continuous first).
    col: Vec<usize>,
    thresholds: Vec<Vec<f64>>,
    cum: Vec<f64>,
    tables: Vec<(f64, f64, Vec<f64>)>,
    impute: Vec<f64>,
}

fn table(lo: f64, hi: f64, t: &[f64], z: f64) -> f64 {
    let k = t.len();
    let pos = ((z - lo) / (hi - lo) * (k - 1) as f64).clamp(0.0, (k - 1) as f64);
    let i = (pos.floor() as usize).min(k - 2);
    let f = pos - i as f64;
    t[i] + f * (t[i + 1] - t[i])
}

impl<'a> Plan<'a> {
    fn new(p: &'a Program) -> Self {
        let d = p.features();
        let (mix, m) = p.dependence.mixing(d);
        let cont: Vec<usize> = (0..d)
            .filter(|&j| matches!(p.marginals[j], Marginal::Continuous { .. }))
            .collect();
        let disc: Vec<usize> = (0..d)
            .filter(|&j| !matches!(p.marginals[j], Marginal::Continuous { .. }))
            .collect();
        let mut col = vec![0; d];
        for (c, &j) in cont.iter().chain(disc.iter()).enumerate() {
            col[j] = c;
        }
        let thresholds = p
            .marginals
            .iter()
            .map(|mg| match mg {
                Marginal::Discrete { probs, .. } => Marginal::thresholds(probs),
                _ => Vec::new(),
            })
            .collect();
        let r = &p.regimes;
        let mut acc = 0.0;
        let cum = r.weights[..r.k() - 1].iter().map(|w| {
            acc += w;
            acc
        });
        let tables = if r.is_trivial() {
            Vec::new()
        } else {
            (0..d).map(|j| r.score_table(j, SCORE_TABLE)).collect()
        };
        Self {
            p,
            mix,
            m,
            col,
            thresholds,
            cum: cum.collect(),
            tables,
            impute: (0..d).map(|j| p.impute(j)).collect(),
        }
    }

    fn scratch(&self) -> Scratch {
        let p = self.p;
        Scratch {
            eps: vec![0.0; self.m],
            xc: vec![0.0; p.screened.len()],
            out: vec![0.0; p.net.outputs],
            codes: vec![0; p.features()],
        }
    }

    fn row(&self, seed: u64, g: u64, x: &mut [f64], s: &mut Scratch) -> f64 {
        let p = self.p;
        let d = p.features();
        let eps = &mut s.eps;
        for (k, e) in eps.iter_mut().enumerate() {
            *e = normal(seed, k as u64, g);
        }
        let comp = if self.tables.is_empty() {
            0
        } else {
            // Single-precision cumulative weights, as in the tensor path.
            let u = uniform(seed, REGIME_STREAM, g) as f32;
            self.cum.iter().filter(|&&c| u > c as f32).count()
        };
        for j in 0..d {
            let mut z: f64 = (0..self.m).map(|k| self.mix[j * self.m + k] * eps[k]).sum();
            if let Some((lo, hi, t)) = self.tables.get(j) {
                let r = &p.regimes;
                z = table(
                    *lo,
                    *hi,
                    t,
                    r.means[comp * d + j] + r.scales[comp * d + j] * z,
                );
            }
            x[j] = match &p.marginals[j] {
                Marginal::Continuous { knots, atoms } => atoms
                    .iter()
                    .find(|a| z >= a.z_lo && z <= a.z_hi)
                    .map_or_else(|| interp_grid(knots, z), |a| a.value),
                Marginal::Discrete { values, .. } => {
                    values[self.thresholds[j].iter().filter(|&&t| z > t).count()]
                }
            };
            if uniform(seed, MISSING_STREAM + self.col[j] as u64, g) < p.missing[j] {
                x[j] = f64::NAN;
            }
        }
        self.target(seed, g, x, s)
    }

    fn target(&self, seed: u64, g: u64, x: &[f64], scratch: &mut Scratch) -> f64 {
        let p = self.p;
        let net = &p.net;
        let (s, h, o) = (net.inputs, net.hidden, net.outputs);
        let Scratch { xc, out, codes, .. } = scratch;
        for (c, &f) in xc.iter_mut().zip(&p.screened) {
            *c = if x[f].is_finite() {
                x[f]
            } else {
                self.impute[f]
            } - 0.5;
        }
        out.copy_from_slice(&net.a);
        for k in 0..o {
            for i in 0..s {
                out[k] += net.beta[k * s + i] * xc[i];
            }
        }
        for jh in 0..h {
            let mut a = net.b[jh];
            for i in 0..s {
                a += net.w[jh * s + i] * xc[i];
            }
            let t = a.tanh();
            for k in 0..o {
                out[k] += net.v[k * h + jh] * t;
            }
        }
        if p.forest.trees() > 0 {
            for ((c, v), fill) in codes.iter_mut().zip(x).zip(&self.impute) {
                *c = code(if v.is_finite() { *v } else { *fill });
            }
            p.forest.add(codes, out);
        }
        match (&p.noise, p.head) {
            (Noise::Gaussianish { knots, .. }, Head::Regression) => {
                let sigma = p.noise.sigma(xc);
                let e = interp_grid(knots, normal(seed, NOISE_STREAM, g).clamp(-Z_GRID, Z_GRID));
                (out[0] + sigma * e).clamp(p.target_range.0, p.target_range.1)
            }
            (_, Head::Binary) => {
                let prob = 1.0 / (1.0 + (-out[0] / p.temperature()).exp());
                (uniform(seed, LABEL_STREAM, g) < prob) as u8 as f64
            }
            (_, Head::Multiclass(k)) => {
                let tau = p.temperature();
                (0..k)
                    .map(|c| out[c] / tau - (-uniform(seed, LABEL_STREAM + c as u64, g).ln()).ln())
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .map_or(0.0, |(c, _)| c as f64)
            }
            _ => f64::NAN,
        }
    }
}

/// Per-thread buffers reused across rows.
struct Scratch {
    eps: Vec<f64>,
    xc: Vec<f64>,
    out: Vec<f64>,
    codes: Vec<u8>,
}

/// Draw rows `0..rows` (global row indices) in parallel.
pub fn sample_cpu(program: &Program, rows: usize, seed: u64) -> Samples {
    let plan = Plan::new(program);
    let d = program.features();
    let mut flat = vec![0.0; rows * d];
    let mut out_y = vec![0.0; rows];
    if d == 0 {
        out_y
            .par_iter_mut()
            .enumerate()
            .with_min_len(1024)
            .for_each_init(
                || plan.scratch(),
                |s, (r, y)| *y = plan.row(seed, r as u64, &mut [], s),
            );
        return Samples {
            x: Vec::new(),
            y: out_y,
        };
    }
    flat.par_chunks_mut(d)
        .zip(out_y.par_iter_mut())
        .enumerate()
        .with_min_len(1024)
        .for_each_init(
            || plan.scratch(),
            |s, (r, (x, y))| *y = plan.row(seed, r as u64, x, s),
        );
    let out_x = (0..d)
        .into_par_iter()
        .map(|j| (0..rows).map(|r| flat[r * d + j]).collect())
        .collect();
    Samples { x: out_x, y: out_y }
}

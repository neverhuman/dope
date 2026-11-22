//! Stateless row-keyed emission contract (`faithful | distill | matched`).
//!
//! Every emitted row depends solely on `(namespace_128, seed, table_id, row_id, group_id)`
//! and NEVER on batch size, thread count (`T11`), or packet digest (`T25` CRN).

use crate::error::{DopeError, Result};
use crate::forest::code;
use crate::marginal::{interp_grid, Marginal, Z_GRID};
use crate::noise::Noise;
use crate::program::Program;
use crate::rng::{
    row_keyed_normal, row_keyed_uniform, DEFAULT_NAMESPACE_128, PURPOSE_COPULA_EPS, PURPOSE_LABEL,
    PURPOSE_MISSING, PURPOSE_NOISE, PURPOSE_REGIME,
};
use crate::sample::{Samples, SCORE_TABLE};
use crate::target::Head;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmissionMode {
    /// Historical Format-1 counter sampler (preserved bit-identically).
    #[default]
    Legacy,
    /// Headline mode: stateless row-keyed independent synthetic draws (default 4n).
    Faithful,
    /// Utility-only variance-reduced mode: antithetic regression pairs / Rao-Blackwellized
    /// class-weighted rows emitting `weight` and `pair_id` sidecar metadata.
    Distill,
    /// Matched-count stateless row-keyed independent synthetic draws (default 1n).
    Matched,
}

impl EmissionMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "legacy" => Ok(Self::Legacy),
            "faithful" => Ok(Self::Faithful),
            "distill" => Ok(Self::Distill),
            "matched" => Ok(Self::Matched),
            _ => Err(DopeError::Input(
                "--mode must be one of legacy, faithful, distill, matched".into(),
            )),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Faithful => "faithful",
            Self::Distill => "distill",
            Self::Matched => "matched",
        }
    }

    pub fn default_row_multiplier(&self) -> usize {
        match self {
            Self::Faithful | Self::Distill => 4,
            Self::Legacy | Self::Matched => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EmitOptions {
    pub mode: EmissionMode,
    pub rows: usize,
    pub seed: u64,
    pub table_id: u64,
    pub row_offset: u64,
    pub group_id: u64,
    pub namespace_128: [u8; 16],
}

impl Default for EmitOptions {
    fn default() -> Self {
        Self {
            mode: EmissionMode::Faithful,
            rows: 1000,
            seed: 0,
            table_id: 0,
            row_offset: 0,
            group_id: 0,
            namespace_128: DEFAULT_NAMESPACE_128,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EmittedSamples {
    pub samples: Samples,
    pub mode: EmissionMode,
    pub table_id: u64,
    /// Present in `Distill` mode: per-row training weight (each `pair_id` sums to 1.0).
    pub weights: Option<Vec<f64>>,
    /// Present in `Distill` mode: pair/group ID tying paired rows together for Rule A-1.
    pub pair_ids: Option<Vec<u64>>,
}

struct EmitPlan<'a> {
    p: &'a Program,
    mix: Vec<f64>,
    m: usize,
    thresholds: Vec<Vec<f64>>,
    cum: Vec<f64>,
    tables: Vec<(f64, f64, Vec<f64>)>,
    impute: Vec<f64>,
}

fn interp_table(lo: f64, hi: f64, t: &[f64], z: f64) -> f64 {
    let k = t.len();
    let pos = ((z - lo) / (hi - lo) * (k - 1) as f64).clamp(0.0, (k - 1) as f64);
    let i = (pos.floor() as usize).min(k - 2);
    let f = pos - i as f64;
    t[i] + f * (t[i + 1] - t[i])
}

impl<'a> EmitPlan<'a> {
    fn new(p: &'a Program) -> Self {
        let d = p.features();
        let (mix, m) = p.dependence.mixing(d);
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
        let cum = r.weights[..r.k() - 1]
            .iter()
            .map(|w| {
                acc += w;
                acc
            })
            .collect();
        let tables = if r.is_trivial() {
            Vec::new()
        } else {
            (0..d).map(|j| r.score_table(j, SCORE_TABLE)).collect()
        };
        Self {
            p,
            mix,
            m,
            thresholds,
            cum,
            tables,
            impute: (0..d).map(|j| p.impute(j)).collect(),
        }
    }

    fn draw_features(&self, opts: &EmitOptions, row_id: u64, x_out: &mut [f64], eps: &mut [f64]) {
        let p = self.p;
        let d = p.features();
        for (k, e) in eps.iter_mut().enumerate() {
            *e = row_keyed_normal(
                &opts.namespace_128,
                opts.seed,
                opts.table_id,
                row_id,
                opts.group_id,
                PURPOSE_COPULA_EPS,
                k as u32,
            );
        }
        let comp = if self.tables.is_empty() {
            0
        } else {
            let u = row_keyed_uniform(
                &opts.namespace_128,
                opts.seed,
                opts.table_id,
                row_id,
                opts.group_id,
                PURPOSE_REGIME,
                0,
            );
            self.cum.iter().filter(|&&c| u > c).count()
        };
        for j in 0..d {
            let mut z: f64 = (0..self.m).map(|k| self.mix[j * self.m + k] * eps[k]).sum();
            if let Some((lo, hi, t)) = self.tables.get(j) {
                let r = &p.regimes;
                z = interp_table(
                    *lo,
                    *hi,
                    t,
                    r.means[comp * d + j] + r.scales[comp * d + j] * z,
                );
            }
            x_out[j] = match &p.marginals[j] {
                Marginal::Continuous { knots, atoms } => atoms
                    .iter()
                    .find(|a| z >= a.z_lo && z <= a.z_hi)
                    .map_or_else(|| interp_grid(knots, z), |a| a.value),
                Marginal::Discrete { values, .. } => {
                    values[self.thresholds[j].iter().filter(|&&t| z > t).count()]
                }
            };
            let um = row_keyed_uniform(
                &opts.namespace_128,
                opts.seed,
                opts.table_id,
                row_id,
                opts.group_id,
                PURPOSE_MISSING,
                j as u32,
            );
            if um < p.missing[j] {
                x_out[j] = f64::NAN;
            }
        }
    }

    fn eval_conditional(&self, x: &[f64], xc: &mut [f64], out: &mut [f64], codes: &mut [u8]) {
        let p = self.p;
        let net = &p.net;
        let (s, h, o) = (net.inputs, net.hidden, net.outputs);
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
    }

    fn draw_target_faithful(
        &self,
        opts: &EmitOptions,
        row_id: u64,
        xc: &[f64],
        out: &[f64],
    ) -> f64 {
        let p = self.p;
        match (&p.noise, p.head) {
            (Noise::Gaussianish { knots, .. }, Head::Regression) => {
                let sigma = p.noise.sigma(xc);
                let z = row_keyed_normal(
                    &opts.namespace_128,
                    opts.seed,
                    opts.table_id,
                    row_id,
                    opts.group_id,
                    PURPOSE_NOISE,
                    0,
                )
                .clamp(-Z_GRID, Z_GRID);
                let e = interp_grid(knots, z);
                (out[0] + sigma * e).clamp(p.target_range.0, p.target_range.1)
            }
            (_, Head::Binary) => {
                let prob = 1.0 / (1.0 + (-out[0] / p.temperature()).exp());
                let u = row_keyed_uniform(
                    &opts.namespace_128,
                    opts.seed,
                    opts.table_id,
                    row_id,
                    opts.group_id,
                    PURPOSE_LABEL,
                    0,
                );
                (u < prob) as u8 as f64
            }
            (_, Head::Multiclass(k)) => {
                let tau = p.temperature();
                (0..k)
                    .map(|c| {
                        let u = row_keyed_uniform(
                            &opts.namespace_128,
                            opts.seed,
                            opts.table_id,
                            row_id,
                            opts.group_id,
                            PURPOSE_LABEL,
                            c as u32,
                        );
                        out[c] / tau - (-u.ln()).ln()
                    })
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .map_or(0.0, |(c, _)| c as f64)
            }
            _ => f64::NAN,
        }
    }
}

/// Emit rows under the Phase P2 row-keyed emission contract (`faithful | distill | matched | legacy`).
pub fn sample_with_mode(program: &Program, opts: &EmitOptions) -> Result<EmittedSamples> {
    if opts.mode == EmissionMode::Legacy {
        let s = crate::sample_cpu::sample_cpu(program, opts.rows, opts.seed);
        return Ok(EmittedSamples {
            samples: s,
            mode: EmissionMode::Legacy,
            table_id: opts.table_id,
            weights: None,
            pair_ids: None,
        });
    }
    let plan = EmitPlan::new(program);
    let d = program.features();
    if matches!(opts.mode, EmissionMode::Faithful | EmissionMode::Matched) {
        let mut flat = vec![0.0; opts.rows * d];
        let mut out_y = vec![0.0; opts.rows];
        if d == 0 {
            out_y.par_iter_mut().enumerate().for_each_init(
                || {
                    (
                        vec![0.0; program.screened.len()],
                        vec![0.0; program.net.outputs],
                        vec![0u8; d],
                    )
                },
                |(xc, out, codes), (r, y)| {
                    let rid = opts.row_offset + r as u64;
                    plan.eval_conditional(&[], xc, out, codes);
                    *y = plan.draw_target_faithful(opts, rid, xc, out);
                },
            );
            return Ok(EmittedSamples {
                samples: Samples {
                    x: Vec::new(),
                    y: out_y,
                },
                mode: opts.mode,
                table_id: opts.table_id,
                weights: None,
                pair_ids: None,
            });
        }
        flat.par_chunks_mut(d)
            .zip(out_y.par_iter_mut())
            .enumerate()
            .for_each_init(
                || {
                    (
                        vec![0.0; plan.m],
                        vec![0.0; program.screened.len()],
                        vec![0.0; program.net.outputs],
                        vec![0u8; d],
                    )
                },
                |(eps, xc, out, codes), (r, (x_row, y))| {
                    let rid = opts.row_offset + r as u64;
                    plan.draw_features(opts, rid, x_row, eps);
                    plan.eval_conditional(x_row, xc, out, codes);
                    *y = plan.draw_target_faithful(opts, rid, xc, out);
                },
            );
        let out_x = (0..d)
            .into_par_iter()
            .map(|j| (0..opts.rows).map(|r| flat[r * d + j]).collect())
            .collect();
        return Ok(EmittedSamples {
            samples: Samples { x: out_x, y: out_y },
            mode: opts.mode,
            table_id: opts.table_id,
            weights: None,
            pair_ids: None,
        });
    }

    // `Distill` mode: variance-reduced antithetic regression pairs or Rao-Blackwellized
    // exact probability-weighted class rows, emitting `weight` and `pair_id`.
    let n_pairs = match program.head {
        Head::Regression => opts.rows.div_ceil(2).max(1),
        Head::Binary => opts.rows.div_ceil(2).max(1),
        Head::Multiclass(k) => opts.rows.div_ceil(k.max(2)).max(1),
    };
    let mut x_cols = vec![Vec::new(); d];
    let mut y_vals = Vec::new();
    let mut weights = Vec::new();
    let mut pair_ids = Vec::new();
    let mut x_row = vec![0.0; d];
    let mut eps = vec![0.0; plan.m];
    let mut xc = vec![0.0; program.screened.len()];
    let mut out = vec![0.0; program.net.outputs];
    let mut codes = vec![0u8; d];

    for p_idx in 0..n_pairs {
        let pid = opts.row_offset + p_idx as u64;
        plan.draw_features(opts, pid, &mut x_row, &mut eps);
        plan.eval_conditional(&x_row, &mut xc, &mut out, &mut codes);
        match (&program.noise, program.head) {
            (Noise::Gaussianish { knots, .. }, Head::Regression) => {
                let sigma = program.noise.sigma(&xc);
                let z = row_keyed_normal(
                    &opts.namespace_128,
                    opts.seed,
                    opts.table_id,
                    pid,
                    opts.group_id,
                    PURPOSE_NOISE,
                    0,
                )
                .clamp(-Z_GRID, Z_GRID);
                let e_pos = interp_grid(knots, z);
                let e_neg = interp_grid(knots, -z);
                for delta in [e_pos, e_neg] {
                    for j in 0..d {
                        x_cols[j].push(x_row[j]);
                    }
                    y_vals.push(
                        (out[0] + sigma * delta)
                            .clamp(program.target_range.0, program.target_range.1),
                    );
                    weights.push(0.5);
                    pair_ids.push(pid);
                }
            }
            (_, Head::Binary) => {
                let p1 = 1.0 / (1.0 + (-out[0] / program.temperature()).exp());
                let probs = [1.0 - p1, p1];
                for (c, &w) in probs.iter().enumerate() {
                    for j in 0..d {
                        x_cols[j].push(x_row[j]);
                    }
                    y_vals.push(c as f64);
                    weights.push(w);
                    pair_ids.push(pid);
                }
            }
            (_, Head::Multiclass(k)) => {
                let tau = program.temperature();
                let max_l = out[..k].iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = (0..k).map(|c| ((out[c] - max_l) / tau).exp()).collect();
                let sum: f64 = exps.iter().sum::<f64>().max(1e-12);
                for (c, &e) in exps.iter().enumerate() {
                    for j in 0..d {
                        x_cols[j].push(x_row[j]);
                    }
                    y_vals.push(c as f64);
                    weights.push(e / sum);
                    pair_ids.push(pid);
                }
            }
            _ => {}
        }
    }

    Ok(EmittedSamples {
        samples: Samples {
            x: x_cols,
            y: y_vals,
        },
        mode: EmissionMode::Distill,
        table_id: opts.table_id,
        weights: Some(weights),
        pair_ids: Some(pair_ids),
    })
}

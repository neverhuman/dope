//! Sampling on any candle device (CPU, CUDA, Metal).
//!
//! Per chunk of rows the whole draw is a short tensor program:
//!
//! ```text
//! z  = ε Mᵀ                                   (one GEMM, all dependence families)
//! x  = gather(knots, ⌊a z + b⌋) + linear interp  (continuous marginals)
//! x  = gather(values, Σ_k [z > t_k])          (discrete marginals)
//! m  = U < p_missing
//! ŷ  = a + β x_c + v tanh(W x_c + b)          (conditional model: network)
//!    + Σ_t leaf_t[Σ_l 2^l·(code(x_f) > c)]     (and oblivious trees)
//! y  = ŷ + σ(x) Q_e(ξ)  |  Gumbel-argmax(ŷ/τ)  (noise)
//! ```
//!
//! Random inputs come from the portable counter RNG, so a seed gives the
//! same rows on every device up to floating-point rounding.

use crate::error::Result;
use crate::marginal::{Marginal, Z_GRID};
use crate::noise::{Noise, RESIDUAL_KNOTS};
use crate::program::Program;
use crate::rng::{normal_block, uniform_block};
use crate::target::Head;
use candle_core::{DType, Device, Tensor, D};

mod forest;
use forest::DeviceForest;

#[derive(Clone, Debug)]
pub struct SampleOptions {
    pub rows: usize,
    pub seed: u64,
    pub chunk: usize,
    pub device: Device,
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            rows: 1000,
            seed: 0,
            chunk: 32_768,
            device: Device::Cpu,
        }
    }
}

/// Synthetic rows in model space: `x[feature][row]` (NaN = missing) and the
/// target (model-space value for regression, class index for classification).
#[derive(Clone, Debug)]
pub struct Samples {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
}

pub(crate) const MISSING_STREAM: u64 = 1 << 32;
pub(crate) const NOISE_STREAM: u64 = 2 << 32;
pub(crate) const LABEL_STREAM: u64 = 3 << 32;
pub(crate) const REGIME_STREAM: u64 = 4 << 32;
/// Nodes of the per-column latent-to-score tables used with regimes.
pub(crate) const SCORE_TABLE: usize = 257;

fn t1(v: Vec<f32>, dev: &Device) -> Result<Tensor> {
    let n = v.len();
    Ok(Tensor::from_vec(v, n, dev)?)
}
fn t2(v: Vec<f32>, r: usize, c: usize, dev: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(v, (r, c), dev)?)
}
fn f32s(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}

/// Piecewise-linear interpolation of `z` ([n, c]) on per-column uniform
/// grids: column j has `k` nodes starting at `lo[j]` with spacing
/// `1 / inv_step[j]`; `table` stores the node values row-wise ([c·k]).
fn table_lookup(
    z: &Tensor,
    table: &Tensor,
    k: usize,
    lo: &Tensor,
    inv_step: &Tensor,
) -> Result<Tensor> {
    let (n, c) = z.dims2()?;
    let pos = z
        .broadcast_sub(lo)?
        .broadcast_mul(inv_step)?
        .clamp(0f32, (k - 1) as f32)?;
    let i = pos.floor()?.clamp(0f32, (k - 2) as f32)?;
    let t = (&pos - &i)?;
    let offs = t2((0..c).map(|j| (j * k) as f32).collect(), 1, c, z.device())?;
    let flat = i.broadcast_add(&offs)?;
    let ids0 = flat.to_dtype(DType::U32)?.flatten_all()?;
    let ids1 = (flat + 1.0)?.to_dtype(DType::U32)?.flatten_all()?;
    let a = table.index_select(&ids0, 0)?.reshape((n, c))?;
    let b = table.index_select(&ids1, 0)?.reshape((n, c))?;
    Ok((&a + (t * (b - &a)?)?)?)
}

/// Piecewise-linear interpolation of `z` ([n, c]) on uniform z-grids stored
/// row-wise in `table` ([c·k]).
fn grid_lookup(z: &Tensor, table: &Tensor, k: usize, c: usize) -> Result<Tensor> {
    let n = z.dim(0)?;
    let a = (k - 1) as f64 / (2.0 * Z_GRID);
    let pos = z.affine(a, Z_GRID * a)?.clamp(0f32, (k - 1) as f32)?;
    let i = pos.floor()?.clamp(0f32, (k - 2) as f32)?;
    let t = (&pos - &i)?;
    let offs: Vec<f32> = (0..c).map(|j| (j * k) as f32).collect();
    let offs = t2(offs, 1, c, z.device())?;
    let flat = i.broadcast_add(&offs)?;
    let ids0 = flat.to_dtype(DType::U32)?.flatten_all()?;
    let ids1 = (flat + 1.0)?.to_dtype(DType::U32)?.flatten_all()?;
    let lo = table.index_select(&ids0, 0)?.reshape((n, c))?;
    let hi = table.index_select(&ids1, 0)?.reshape((n, c))?;
    Ok((&lo + (t * (hi - &lo)?)?)?)
}

/// Draw `opts.rows` synthetic rows from `program`: the native row-parallel
/// sampler on CPU, the tensor program on CUDA/Metal.
pub fn sample(program: &Program, opts: &SampleOptions) -> Result<Samples> {
    if opts.device.is_cpu() {
        return Ok(crate::sample_cpu::sample_cpu(program, opts.rows, opts.seed));
    }
    sample_device(program, opts)
}

/// The tensor-program sampler (any candle device; on CPU it is the parity
/// reference for the native sampler).
pub fn sample_device(program: &Program, opts: &SampleOptions) -> Result<Samples> {
    let dev = &opts.device;
    let d = program.features();
    let (mix, m) = program.dependence.mixing(d);
    // Mᵀ as [m, d] so that z = ε · Mᵀ.
    let mut mt = vec![0f32; m * d];
    for i in 0..d {
        for k in 0..m {
            mt[k * d + i] = mix[i * m + k] as f32;
        }
    }
    let mt = t2(mt, m, d, dev)?;
    let cont: Vec<usize> = (0..d)
        .filter(|&j| matches!(program.marginals[j], Marginal::Continuous { .. }))
        .collect();
    let disc: Vec<usize> = (0..d)
        .filter(|&j| !matches!(program.marginals[j], Marginal::Continuous { .. }))
        .collect();
    let k = cont.first().map_or(3, |&j| match &program.marginals[j] {
        Marginal::Continuous { knots, .. } => knots.len(),
        _ => 3,
    });
    let mut knots = Vec::with_capacity(cont.len() * k);
    let mut atoms = Vec::new();
    for (c, &j) in cont.iter().enumerate() {
        if let Marginal::Continuous {
            knots: kn,
            atoms: at,
        } = &program.marginals[j]
        {
            knots.extend(f32s(kn));
            atoms.extend(at.iter().map(|a| (c, a.value, a.z_lo, a.z_hi)));
        }
    }
    let knots = t1(knots, dev)?;
    let lmax = disc
        .iter()
        .map(|&j| match &program.marginals[j] {
            Marginal::Discrete { probs, .. } => probs.len(),
            _ => 1,
        })
        .max()
        .unwrap_or(1);
    let (mut thr, mut vals) = (Vec::new(), Vec::new());
    for &j in &disc {
        if let Marginal::Discrete { values, probs } = &program.marginals[j] {
            let t = Marginal::thresholds(probs);
            thr.extend(
                (0..lmax.saturating_sub(1).max(1)).map(|q| t.get(q).map_or(1e30, |v| *v as f32)),
            );
            vals.extend((0..lmax).map(|q| values.get(q).map_or(0.0, |v| *v as f32)));
        }
    }
    let lt = lmax.saturating_sub(1).max(1);
    let thr = t2(thr, disc.len(), lt, dev)?;
    let vals = t1(vals, dev)?;
    let doffs = t2(
        (0..disc.len()).map(|j| (j * lmax) as f32).collect(),
        1,
        disc.len(),
        dev,
    )?;
    // Concatenated layout: continuous columns, then discrete columns.
    let layout: Vec<usize> = cont.iter().chain(disc.iter()).copied().collect();
    let pos_of: Vec<usize> = {
        let mut p = vec![0; d];
        for (c, &j) in layout.iter().enumerate() {
            p[j] = c;
        }
        p
    };
    let pmiss = t2(
        layout.iter().map(|&j| program.missing[j] as f32).collect(),
        1,
        d,
        dev,
    )?;
    let ids = |v: Vec<u32>| -> Result<Tensor> {
        let n = v.len();
        Ok(Tensor::from_vec(v, n, dev)?)
    };
    let cont_ids = ids(cont.iter().map(|&j| j as u32).collect())?;
    let disc_ids = ids(disc.iter().map(|&j| j as u32).collect())?;
    let net = &program.net;
    let (s, h, o) = (net.inputs, net.hidden, net.outputs);
    let screen_ids = ids(program.screened.iter().map(|&j| pos_of[j] as u32).collect())?;
    let impute = t2(
        program
            .screened
            .iter()
            .map(|&j| program.impute(j) as f32)
            .collect(),
        1,
        s,
        dev,
    )?;
    let w = t2(f32s(&net.w), h, s, dev)?;
    let b = t2(f32s(&net.b), 1, h, dev)?;
    let beta = t2(f32s(&net.beta), o, s, dev)?;
    let v = t2(f32s(&net.v), o, h, dev)?;
    let a = t2(f32s(&net.a), 1, o, dev)?;

    // Regime mixture: per-row component, location/scale, and the per-column
    // map from the latent mixture coordinate back to a standard score.
    let regimes = &program.regimes;
    let regime = if regimes.is_trivial() {
        None
    } else {
        let k = regimes.k();
        let mut cum = 0.0;
        let thresholds: Vec<f32> = regimes.weights[..k - 1]
            .iter()
            .map(|w| {
                cum += w;
                cum as f32
            })
            .collect();
        let (mut table, mut lo, mut inv) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..d {
            let (a, b, t) = regimes.score_table(j, SCORE_TABLE);
            table.extend(f32s(&t));
            lo.push(a as f32);
            inv.push(((SCORE_TABLE - 1) as f64 / (b - a)) as f32);
        }
        Some((
            t2(thresholds, 1, k - 1, dev)?,
            t2(f32s(&regimes.means), k, d, dev)?,
            t2(f32s(&regimes.scales), k, d, dev)?,
            t1(table, dev)?,
            t2(lo, 1, d, dev)?,
            t2(inv, 1, d, dev)?,
        ))
    };
    let forest = if program.forest.trees() > 0 {
        Some(DeviceForest::new(program, &pos_of, &layout, dev)?)
    } else {
        None
    };
    let mut out_x = vec![Vec::with_capacity(opts.rows); d];
    let mut out_y = Vec::with_capacity(opts.rows);
    let mut row0 = 0usize;
    while row0 < opts.rows {
        let n = opts.chunk.min(opts.rows - row0);
        let trace = std::env::var_os("DOPE_TRACE").is_some();
        let t0 = std::time::Instant::now();
        let eps = t2(normal_block(opts.seed, 0, row0 as u64, n, m), n, m, dev)?;
        let t_rng = t0.elapsed();
        let mut z = eps.matmul(&mt)?;
        if let Some((thr, mu, sd, table, lo, inv)) = &regime {
            let u = t2(
                uniform_block(opts.seed, REGIME_STREAM, row0 as u64, n, 1),
                n,
                1,
                dev,
            )?;
            let comp = u
                .broadcast_gt(thr)?
                .to_dtype(DType::F32)?
                .sum(1)?
                .to_dtype(DType::U32)?;
            let (m, s) = (mu.index_select(&comp, 0)?, sd.index_select(&comp, 0)?);
            z = ((z * s)? + m)?;
            z = table_lookup(&z, table, SCORE_TABLE, lo, inv)?;
        }
        let mut parts = Vec::new();
        if !cont.is_empty() {
            let zc = z.index_select(&cont_ids, 1)?;
            let mut xc = grid_lookup(&zc, &knots, k, cont.len())?;
            for &(c, value, lo, hi) in &atoms {
                let col = zc.narrow(1, c, 1)?;
                let hit = (col.ge(lo as f32)?.to_dtype(DType::F32)?
                    * col.le(hi as f32)?.to_dtype(DType::F32)?)?;
                let mut sel = vec![0f32; cont.len()];
                sel[c] = 1.0;
                let sel = hit.broadcast_mul(&t2(sel, 1, cont.len(), dev)?)?;
                xc = (&xc + (sel * xc.affine(-1.0, value)?)?)?;
            }
            parts.push(xc);
        }
        if !disc.is_empty() {
            let zd = z.index_select(&disc_ids, 1)?;
            let level = zd
                .unsqueeze(2)?
                .broadcast_gt(&thr.unsqueeze(0)?)?
                .to_dtype(DType::F32)?
                .sum(2)?;
            let flat = level
                .broadcast_add(&doffs)?
                .to_dtype(DType::U32)?
                .flatten_all()?;
            parts.push(vals.index_select(&flat, 0)?.reshape((n, disc.len()))?);
        }
        let x = Tensor::cat(&parts, 1)?;
        let t_marg = t0.elapsed();
        let u = t2(
            uniform_block(opts.seed, MISSING_STREAM, row0 as u64, n, d),
            n,
            d,
            dev,
        )?;
        let miss = u.broadcast_lt(&pmiss)?;
        let (y, xc_s) = if s == 0 {
            (a.broadcast_as((n, o))?.contiguous()?, None)
        } else {
            let xs = x.index_select(&screen_ids, 1)?;
            let ms = miss.index_select(&screen_ids, 1)?;
            let xs = ms
                .where_cond(&impute.broadcast_as((n, s))?.contiguous()?, &xs)?
                .affine(1.0, -0.5)?;
            let mut out = xs.matmul(&beta.t()?)?.broadcast_add(&a)?;
            if h > 0 {
                let hid = xs.matmul(&w.t()?)?.broadcast_add(&b)?.tanh()?;
                out = (out + hid.matmul(&v.t()?)?)?;
            }
            (out, Some(xs))
        };
        let y = match &forest {
            Some(fd) => (&y + fd.outputs(&x, &miss, n)?)?,
            None => y,
        };
        let yv = draw_target(program, xc_s.as_ref(), &y, opts, row0, n)?;
        let t_target = t0.elapsed();
        let xv = x.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let mv = miss.to_dtype(DType::U8)?.to_vec2::<u8>()?;
        for r in 0..n {
            for (c, &j) in layout.iter().enumerate() {
                out_x[j].push(if mv[r][c] == 1 {
                    f64::NAN
                } else {
                    xv[r][c] as f64
                });
            }
        }
        out_y.extend(yv);
        if trace {
            eprintln!(
                "chunk {n}: rng {:?} marginals {:?} target {:?} output {:?}",
                t_rng,
                t_marg - t_rng,
                t_target - t_marg,
                t0.elapsed() - t_target
            );
        }
        row0 += n;
    }
    Ok(Samples { x: out_x, y: out_y })
}

fn draw_target(
    program: &Program,
    xc: Option<&Tensor>,
    out: &Tensor,
    opts: &SampleOptions,
    row0: usize,
    n: usize,
) -> Result<Vec<f64>> {
    let dev = &opts.device;
    match (&program.noise, program.head) {
        (
            Noise::Gaussianish {
                coef,
                lo,
                hi,
                knots,
            },
            Head::Regression,
        ) => {
            let t = coef.len() - 1;
            let mut log_sigma = Tensor::full(coef[0] as f32, (n, 1), dev)?;
            if let (true, Some(xc)) = (t > 0, xc) {
                let gamma = Tensor::from_vec(
                    coef[1..].iter().map(|c| *c as f32).collect::<Vec<_>>(),
                    (t, 1),
                    dev,
                )?;
                log_sigma = log_sigma.broadcast_add(&xc.narrow(1, 0, t)?.matmul(&gamma)?)?;
            }
            let sigma = log_sigma.exp()?.clamp(*lo as f32, *hi as f32)?;
            let xi = Tensor::from_vec(
                normal_block(opts.seed, NOISE_STREAM, row0 as u64, n, 1),
                (n, 1),
                dev,
            )?;
            let table = Tensor::from_vec(
                knots.iter().map(|v| *v as f32).collect::<Vec<_>>(),
                RESIDUAL_KNOTS,
                dev,
            )?;
            let e = grid_lookup(&xi, &table, RESIDUAL_KNOTS, 1)?;
            let (rlo, rhi) = program.target_range;
            let y = (out + (sigma * e)?)?.clamp(rlo as f32, rhi as f32)?;
            Ok(y.flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(|v| v as f64)
                .collect())
        }
        (_, Head::Binary) => {
            let tau = program.temperature();
            let p = (out.affine(-1.0 / tau, 0.0)?.exp()? + 1.0)?.recip()?;
            let u = Tensor::from_vec(
                uniform_block(opts.seed, LABEL_STREAM, row0 as u64, n, 1),
                (n, 1),
                dev,
            )?;
            let y = u.lt(&p)?.to_dtype(DType::F32)?;
            Ok(y.flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(|v| v as f64)
                .collect())
        }
        (_, Head::Multiclass(k)) => {
            let tau = program.temperature();
            let u = Tensor::from_vec(
                uniform_block(opts.seed, LABEL_STREAM, row0 as u64, n, k),
                (n, k),
                dev,
            )?;
            let gumbel = u.log()?.neg()?.log()?.neg()?;
            let y = (out.affine(1.0 / tau, 0.0)? + gumbel)?.argmax_keepdim(D::Minus1)?;
            Ok(y.flatten_all()?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?
                .into_iter()
                .map(|v| v as f64)
                .collect())
        }
        _ => Ok(vec![f64::NAN; n]),
    }
}

//! Decoding: rebuild the exact program from artifact bytes.

use super::{limits, FORMAT_VERSION};
use crate::copula::{clamp_rows, Dependence};
use crate::error::{DopeError, Result};
use crate::forest::Forest;
use crate::marginal::{Atom, Marginal};
use crate::noise::{Noise, RESIDUAL_KNOTS};
use crate::program::Program;
use crate::quant::{unit_value, QTensor};
use crate::regimes::{Regimes, MEAN_RANGE, SCALE_MAX, SCALE_MIN};
use crate::schema::{CategoricalSpec, ColumnSpec, Kind, NumericSpec, Schema, Task};
use crate::target::{Head, Net, QuantNet};
use half::f16;
use serde_json::Value;

fn read_f16(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f64())
        .collect()
}
pub(super) fn read_f32(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect()
}

fn column_from_meta(v: &Value, affine: &mut std::slice::Iter<'_, f64>) -> Result<ColumnSpec> {
    let bad = || DopeError::Artifact("malformed column metadata".into());
    let a = v.as_array().ok_or_else(bad)?;
    let name = a
        .first()
        .and_then(Value::as_str)
        .ok_or_else(bad)?
        .to_string();
    let kind = match a.get(1).and_then(Value::as_str) {
        Some("n") => {
            let flags = a.get(2).and_then(Value::as_u64).ok_or_else(bad)?;
            let lo = *affine.next().ok_or_else(bad)?;
            let hi = *affine.next().ok_or_else(bad)?;
            Kind::Numeric(NumericSpec {
                log: flags & 1 == 1,
                integer: flags & 2 == 2,
                lo,
                hi,
            })
        }
        Some("c") => {
            let levels = a.get(2).and_then(Value::as_array).ok_or_else(bad)?;
            let levels = levels
                .iter()
                .map(|l| l.as_str().map(str::to_string).ok_or_else(bad));
            Kind::Categorical(CategoricalSpec {
                levels: levels.collect::<Result<_>>()?,
            })
        }
        _ => return Err(bad()),
    };
    Ok(ColumnSpec { name, kind })
}

/// Rebuild a program from artifact bytes.
pub fn decode(bytes: &[u8]) -> Result<Program> {
    let bad = |what: &str| DopeError::Artifact(format!("not a DOPE artifact ({what})"));
    let parts = super::unpack(bytes)?;
    let meta = &parts.meta;
    if meta["v"].as_u64() != Some(FORMAT_VERSION) {
        return Err(bad("unsupported format version"));
    }
    limits::validate(&parts)?;
    let tensor = |name: &str| parts.sections.get(name).map(Vec::as_slice);
    let affine = tensor("schema.affine").map(read_f32).unwrap_or_default();
    let mut aff = affine.iter();
    let cols = meta["cols"].as_array().ok_or_else(|| bad("columns"))?;
    let features = cols
        .iter()
        .map(|c| column_from_meta(c, &mut aff))
        .collect::<Result<Vec<_>>>()?;
    let target = column_from_meta(&meta["y"], &mut aff)?;
    let task = match meta["task"].as_str() {
        Some("regression") => Task::Regression,
        Some("classification") => Task::Classification,
        _ => return Err(bad("task")),
    };
    let dropped = meta["drop"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| bad("dropped columns"))
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    let schema = Schema {
        task,
        features,
        target,
        dropped,
    };
    let d = schema.features.len();
    let knots = limits::integer(&meta["k"])?;
    let missing = match tensor("features.missing") {
        Some(b) => b.iter().map(|c| (*c as f64 / 255.0).powi(2)).collect(),
        None => vec![0.0; d],
    };
    let ranges = tensor("marginal.range").map(read_f16).unwrap_or_default();
    let codes = tensor("marginal.knots").unwrap_or(&[]);
    let atoms = tensor("marginal.atoms").map(read_f16).unwrap_or_default();
    let cum: Vec<f64> = tensor("discrete.cum")
        .map(|b| {
            b.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]) as f64 / 65535.0)
                .collect()
        })
        .unwrap_or_default();
    let dvalues = tensor("discrete.values").map(read_f32).unwrap_or_default();
    let (mut ci, mut cc, mut dv) = (0usize, 0usize, 0usize);
    let mut marginals = Vec::with_capacity(d);
    for (j, col) in schema.features.iter().enumerate() {
        // A feature is continuous iff it owns a range row, in order.
        let is_cont = is_continuous(j, meta, d);
        if is_cont {
            let (lo, hi) = (
                *ranges
                    .get(limits::mul(2, ci)?)
                    .ok_or_else(|| bad("ranges"))?,
                *ranges
                    .get(limits::add(limits::mul(2, ci)?, 1)?)
                    .ok_or_else(|| bad("ranges"))?,
            );
            let mut k = vec![lo];
            for m in 0..knots.saturating_sub(2) {
                let code = *codes
                    .get(limits::add(limits::mul(ci, knots - 2)?, m)?)
                    .ok_or_else(|| bad("knots"))?;
                k.push(unit_value(code as u32, lo, hi, 8));
            }
            k.push(hi);
            let a = atoms
                .chunks_exact(4)
                .filter(|a| a[0] as usize == j)
                .map(|a| Atom {
                    value: a[1],
                    z_lo: a[2],
                    z_hi: a[3],
                })
                .collect();
            marginals.push(Marginal::Continuous { knots: k, atoms: a });
            ci += 1;
        } else {
            let levels = discrete_levels(j, meta)?;
            let mut probs = Vec::with_capacity(levels);
            let mut prev = 0.0;
            for _ in 0..levels - 1 {
                let c = *cum.get(cc).ok_or_else(|| bad("discrete"))?;
                probs.push((c - prev).max(0.0));
                prev = c;
                cc += 1;
            }
            probs.push((1.0 - prev).max(0.0));
            let values = match &col.kind {
                Kind::Categorical(s) => (0..levels).map(|k| s.code(k)).collect(),
                Kind::Numeric(_) => {
                    let v = dvalues
                        .get(dv..limits::add(dv, levels)?)
                        .ok_or_else(|| bad("discrete values"))?
                        .to_vec();
                    dv = limits::add(dv, levels)?;
                    v
                }
            };
            marginals.push(Marginal::Discrete { values, probs });
        }
    }
    let i8v = |b: u8| (b as i8) as f64 / 127.0;
    let dep = meta["dep"].as_str().unwrap_or("independent");
    let dependence = if dep == "full" {
        let upper = tensor("copula.corr").ok_or_else(|| bad("copula"))?;
        let mut corr = vec![0.0; limits::mul(d, d)?];
        let mut it = upper.iter();
        for i in 0..d {
            corr[limits::add(limits::mul(i, d)?, i)?] = 1.0;
            for j in i + 1..d {
                let v = i8v(*it.next().ok_or_else(|| bad("copula"))?);
                corr[limits::add(limits::mul(i, d)?, j)?] = v;
                corr[limits::add(limits::mul(j, d)?, i)?] = v;
            }
        }
        Dependence::Full { corr }
    } else if let Some(rank) = dep
        .strip_prefix("factor")
        .and_then(|r| r.parse::<usize>().ok())
    {
        let raw = tensor("copula.loadings").ok_or_else(|| bad("copula"))?;
        let mut loadings: Vec<f64> = raw.iter().map(|b| i8v(*b)).collect();
        clamp_rows(&mut loadings, d, rank);
        Dependence::Factor { loadings, rank }
    } else if dep == "tree" {
        let par = tensor("copula.parent").ok_or_else(|| bad("copula"))?;
        let rho = tensor("copula.rho").ok_or_else(|| bad("copula"))?;
        let parent = par
            .chunks_exact(2)
            .map(|c| match u16::from_le_bytes([c[0], c[1]]) {
                0 => None,
                p => Some(p as usize - 1),
            })
            .collect();
        Dependence::Tree {
            parent,
            rho: rho.iter().map(|b| i8v(*b)).collect(),
        }
    } else {
        Dependence::Independent
    };
    let regimes = match tensor("regime.weights") {
        None => Regimes::single(d),
        Some(w) => {
            let k = w.len();
            let means = tensor("regime.means").ok_or_else(|| bad("regimes"))?;
            let scales = tensor("regime.scales").ok_or_else(|| bad("regimes"))?;
            if means.len() != limits::mul(k, d)? || scales.len() != limits::mul(k, d)? {
                return Err(bad("regimes"));
            }
            let total: f64 = w.iter().map(|c| *c as f64).sum();
            let (lmin, lmax) = (SCALE_MIN.ln(), SCALE_MAX.ln());
            Regimes {
                weights: w.iter().map(|c| *c as f64 / total).collect(),
                means: means
                    .iter()
                    .map(|c| (*c as i8) as f64 / 127.0 * MEAN_RANGE)
                    .collect(),
                scales: scales
                    .iter()
                    .map(|c| (lmin + *c as f64 / 255.0 * (lmax - lmin)).exp())
                    .collect(),
            }
        }
    };
    let screened = limits::integers(&meta["s"])?;
    let dims = limits::integers(&meta["net"])?;
    if dims.len() != 7 || dims[0] != screened.len() || screened.iter().any(|&f| f >= d) {
        return Err(bad("net dimensions"));
    }
    let (s, h, o) = (dims[0], dims[1], dims[2]);
    let floats = tensor("net.params")
        .map(read_f32)
        .ok_or_else(|| bad("net params"))?;
    let codes = tensor("net.codes").unwrap_or(&[]);
    let mut offset = 0;
    let mut group = |n: usize, bits: usize, scale: f64| -> Result<QTensor> {
        let len = limits::packed(n, bits)?;
        let end = limits::add(offset, len)?;
        let slice = codes.get(offset..end).ok_or_else(|| bad("net codes"))?;
        offset = end;
        Ok(QTensor::unpack(slice, n, bits as u8, scale as f32))
    };
    if floats.len() != limits::add(4, o)? {
        return Err(bad("net params"));
    }
    let q = QuantNet {
        inputs: s,
        hidden: h,
        outputs: o,
        w: group(limits::mul(h, s)?, dims[3], floats[0])?,
        b: group(h, dims[4], floats[1])?,
        beta: group(limits::mul(o, s)?, dims[5], floats[2])?,
        v: group(limits::mul(o, h)?, dims[6], floats[3])?,
        a: floats[4..].iter().map(|v| *v as f32).collect(),
    };
    let net: Net = q.net();
    let head = Program::head_for(&schema);
    let forest = match meta["forest"].as_array() {
        None => Forest::empty(o),
        Some(a) => {
            let v = limits::integers(&Value::Array(a.clone()))?;
            let blob = tensor("forest").ok_or_else(|| bad("forest"))?;
            match v[..] {
                [trees, depth, bits] if (1..=8).contains(&depth) && (2..=8).contains(&bits) => {
                    Forest::from_bytes(blob, d, trees, depth, o, bits as u8)
                        .ok_or_else(|| bad("forest"))?
                }
                _ => return Err(bad("forest")),
            }
        }
    };
    let nv = tensor("noise").map(read_f16).ok_or_else(|| bad("noise"))?;
    let (noise, target_range) = match head {
        Head::Regression => {
            let t = limits::integer(&meta["t"])?;
            let coef_end = limits::add(1, t)?;
            let knots_start = limits::add(coef_end, 2)?;
            let knots_end = limits::add(knots_start, RESIDUAL_KNOTS)?;
            if nv.len() != limits::add(knots_end, 2)? {
                return Err(bad("noise"));
            }
            let coef = nv[..coef_end].to_vec();
            let (lo, hi) = (nv[coef_end], nv[limits::add(coef_end, 1)?]);
            let knots_e = nv[knots_start..knots_end].to_vec();
            let r = (nv[knots_end], nv[limits::add(knots_end, 1)?]);
            (
                Noise::Gaussianish {
                    coef,
                    lo,
                    hi,
                    knots: knots_e,
                },
                r,
            )
        }
        _ => (
            Noise::Temperature(*nv.first().ok_or_else(|| bad("noise"))?),
            (0.0, 1.0),
        ),
    };
    Ok(Program {
        schema,
        marginals,
        missing,
        dependence,
        regimes,
        screened,
        net,
        forest,
        head,
        noise,
        target_range,
    })
}

/// Continuous/discrete layout is stored implicitly: discrete features are
/// listed in metadata `"disc"` with their level counts.
fn is_continuous(j: usize, meta: &Value, _d: usize) -> bool {
    !meta["disc"]
        .as_object()
        .is_some_and(|m| m.contains_key(&j.to_string()))
}

fn discrete_levels(j: usize, meta: &Value) -> Result<usize> {
    limits::integer(&meta["disc"][j.to_string()])
}

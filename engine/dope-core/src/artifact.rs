//! The on-disk artifact: a standard **safetensors** file with one `U8`
//! tensor, `dope`, holding the program as a single brotli stream.
//!
//! Inflated, the stream is the program's metadata JSON (schema, structure and
//! bit widths) followed by its named sections: integer codes (bit-packed where
//! noted), half- and single-precision scales and a few constants. Any
//! safetensors reader can open the file; `decode` rebuilds the exact program
//! that was evaluated at compile time, on any device. See [`container`].

use crate::copula::Dependence;
use crate::error::Result;
use crate::marginal::Marginal;
use crate::noise::Noise;
use crate::program::Program;
use crate::quant::unit_code;
use crate::regimes::{MEAN_RANGE, SCALE_MAX, SCALE_MIN};
use crate::schema::{ColumnSpec, Kind, Task};
use crate::target::QuantNet;
use half::f16;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub const FORMAT_VERSION: u64 = 1;

pub mod container;
mod decode;
pub use container::{unpack, Parts};
pub use decode::decode;

fn f16_bytes(v: &[f64]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| f16::from_f64(*x).to_le_bytes())
        .collect()
}
fn f32_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect()
}
fn round16(v: f64) -> f64 {
    f16::from_f64(v).to_f64()
}

fn column_meta(c: &ColumnSpec) -> Value {
    match &c.kind {
        Kind::Numeric(s) => json!([c.name, "n", (s.log as u8) | ((s.integer as u8) << 1)]),
        Kind::Categorical(s) => json!([c.name, "c", s.levels]),
    }
}

/// Serialize a program (whose net is given in quantized form) and return the
/// bytes of a valid safetensors file.
pub fn encode(p: &Program, q: &QuantNet, knots: usize) -> Result<Vec<u8>> {
    let mut t: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
    let d = p.features();
    // Schema affine maps (raw units) for numeric features, then a numeric target.
    let mut affine = Vec::new();
    for c in p
        .schema
        .features
        .iter()
        .chain(std::iter::once(&p.schema.target))
    {
        if let Kind::Numeric(s) = &c.kind {
            affine.extend([s.lo, s.hi]);
        }
    }
    if !affine.is_empty() {
        t.insert("schema.affine", f32_bytes(&affine));
    }
    if p.missing.iter().any(|m| *m > 0.0) {
        let codes = p
            .missing
            .iter()
            .map(|m| (m.clamp(0.0, 1.0).sqrt() * 255.0).round() as u8)
            .collect();
        t.insert("features.missing", codes);
    }
    // Marginals.
    let (mut ranges, mut codes, mut atoms, mut cum, mut values) =
        (vec![], vec![], vec![], vec![], vec![]);
    let (mut cont, mut disc) = (vec![], vec![]);
    for (j, m) in p.marginals.iter().enumerate() {
        match m {
            Marginal::Continuous { knots: k, atoms: a } => {
                cont.push(j);
                let (lo, hi) = (round16(k[0]), round16(k[k.len() - 1]));
                ranges.extend([lo, hi]);
                codes.extend(
                    k[1..k.len() - 1]
                        .iter()
                        .map(|v| unit_code(*v, lo, hi, 8) as u8),
                );
                for at in a {
                    atoms.extend([j as f64, at.value, at.z_lo, at.z_hi]);
                }
            }
            Marginal::Discrete { values: v, probs } => {
                disc.push(j);
                let mut c = 0.0;
                for pr in &probs[..probs.len() - 1] {
                    c += pr;
                    cum.extend(((c.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes());
                }
                if matches!(p.schema.features[j].kind, Kind::Numeric(_)) {
                    values.extend(v.iter().copied());
                }
            }
        }
    }
    if !cont.is_empty() {
        t.insert("marginal.range", f16_bytes(&ranges));
        if knots > 2 {
            t.insert("marginal.knots", codes);
        }
    }
    if !atoms.is_empty() {
        t.insert("marginal.atoms", f16_bytes(&atoms));
    }
    if !cum.is_empty() {
        t.insert("discrete.cum", cum);
    }
    if !values.is_empty() {
        t.insert("discrete.values", f32_bytes(&values));
    }
    // Dependence.
    let i8c = |v: f64| ((v.clamp(-1.0, 1.0) * 127.0).round() as i8) as u8;
    match &p.dependence {
        Dependence::Independent => {}
        Dependence::Full { corr } => {
            let upper: Vec<u8> = (0..d)
                .flat_map(|i| (i + 1..d).map(move |j| (i, j)))
                .map(|(i, j)| i8c(corr[i * d + j]))
                .collect();
            t.insert("copula.corr", upper);
        }
        Dependence::Factor { loadings, .. } => {
            t.insert(
                "copula.loadings",
                loadings.iter().map(|v| i8c(*v)).collect(),
            );
        }
        Dependence::Tree { parent, rho } => {
            let par: Vec<u8> = parent
                .iter()
                .flat_map(|p| (p.map_or(0, |x| x + 1) as u16).to_le_bytes())
                .collect();
            t.insert("copula.parent", par);
            t.insert("copula.rho", rho.iter().map(|v| i8c(*v)).collect());
        }
    }
    // Regime mixture (absent = one regime, the plain Gaussian copula).
    if !p.regimes.is_trivial() {
        let r = &p.regimes;
        let (lmin, lmax) = (SCALE_MIN.ln(), SCALE_MAX.ln());
        t.insert(
            "regime.weights",
            r.weights
                .iter()
                .map(|w| ((w * 255.0).round() as u8).max(1))
                .collect(),
        );
        t.insert(
            "regime.means",
            r.means
                .iter()
                .map(|m| ((m / MEAN_RANGE).clamp(-1.0, 1.0) * 127.0).round() as i8 as u8)
                .collect(),
        );
        t.insert(
            "regime.scales",
            r.scales
                .iter()
                .map(|v| (((v.ln() - lmin) / (lmax - lmin)).clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect(),
        );
    }
    // Conditional model.
    let mut net_codes = Vec::new();
    for g in [&q.w, &q.b, &q.beta, &q.v] {
        net_codes.extend(g.packed());
    }
    if !net_codes.is_empty() {
        t.insert("net.codes", net_codes);
    }
    let mut floats: Vec<f64> = [&q.w, &q.b, &q.beta, &q.v]
        .iter()
        .map(|g| g.scale as f64)
        .collect();
    floats.extend(q.a.iter().map(|v| *v as f64));
    t.insert("net.params", f32_bytes(&floats));
    // Boosted trees: one U8 blob (see forest::Forest::to_bytes).
    let f = &p.forest;
    if f.trees() > 0 {
        let blob = f.to_bytes(d);
        t.insert("forest", blob);
    }
    let mut noise_inputs = 0;
    let noise = match &p.noise {
        Noise::Gaussianish {
            coef,
            lo,
            hi,
            knots: e,
        } => {
            noise_inputs = coef.len() - 1;
            let mut v = coef.clone();
            v.extend([*lo, *hi]);
            v.extend(e.iter().copied());
            v.extend([p.target_range.0, p.target_range.1]);
            v
        }
        Noise::Temperature(tau) => vec![*tau],
    };
    t.insert("noise", f16_bytes(&noise));

    let mut meta = json!({
        "v": FORMAT_VERSION,
        "task": match p.schema.task { Task::Regression => "regression", Task::Classification => "classification" },
        "cols": p.schema.features.iter().map(column_meta).collect::<Vec<_>>(),
        "y": column_meta(&p.schema.target),
        "k": knots,
        "dep": p.dependence.name(),
        "regimes": p.regimes.k(),
        "s": p.screened,
        "net": [q.inputs, q.hidden, q.outputs, q.w.bits, q.b.bits, q.beta.bits, q.v.bits],
        "t": noise_inputs,
    });
    if f.trees() > 0 {
        meta["forest"] = json!([f.trees(), f.depth, f.bits]);
    }
    if !p.schema.dropped.is_empty() {
        meta["drop"] = json!(p.schema.dropped);
    }
    let disc = discrete_layout(p);
    if disc.as_object().is_some_and(|m| !m.is_empty()) {
        meta["disc"] = disc;
    }
    container::pack(&meta, &t)
}

/// Add the discrete layout to the metadata (called by `encode`).
pub(crate) fn discrete_layout(p: &Program) -> Value {
    let mut m = serde_json::Map::new();
    for (j, marg) in p.marginals.iter().enumerate() {
        if let Marginal::Discrete { probs, .. } = marg {
            m.insert(j.to_string(), json!(probs.len()));
        }
    }
    Value::Object(m)
}

/// Byte accounting of an artifact: the file, its safetensors header, the
/// stored (compressed) program, and what the program holds before compression.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ByteReport {
    pub total: usize,
    /// Safetensors framing: the 8-byte length and the header JSON.
    pub header: usize,
    /// The brotli stream (everything sampling reads besides the framing).
    pub stored: usize,
    /// Inflated metadata JSON (schema, structure, bit widths).
    pub metadata: usize,
    /// Inflated sections, by name.
    pub tensors: BTreeMap<String, usize>,
}

pub fn byte_report(bytes: &[u8]) -> Result<ByteReport> {
    let parts = unpack(bytes)?;
    Ok(ByteReport {
        total: bytes.len(),
        header: parts.header,
        stored: parts.stored,
        metadata: parts.meta_bytes,
        tensors: parts
            .sections
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect(),
    })
}

//! Pure-Rust bounds-checked binary primitives and section codecs for Format 2 (`Schema` and
//! `ColumnMarginalV2`) — zero JSON inside the Format 2 payload cap.

use crate::error::{DopeError, Result};
use crate::marginal::atoms::AtomV2;
use crate::marginal::lattice::{HeapingSpec, LatticeSpec};
use crate::marginal::v2::ColumnMarginalV2;
use crate::schema::{CategoricalSpec, ColumnSpec, Kind, NumericSpec, Schema, Task};

const MAX_SEQ: usize = 1 << 20;

pub(crate) struct BinWriter(pub Vec<u8>);

impl BinWriter {
    pub fn new() -> Self {
        Self(Vec::new())
    }
    pub fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    pub fn u16(&mut self, v: u16) {
        self.0.extend(v.to_le_bytes());
    }
    pub fn u32(&mut self, v: u32) {
        self.0.extend(v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.0.extend(v.to_le_bytes());
    }
    pub fn i64(&mut self, v: i64) {
        self.0.extend(v.to_le_bytes());
    }
    pub fn f32(&mut self, v: f64) {
        self.0.extend((v as f32).to_le_bytes());
    }
    pub fn f64(&mut self, v: f64) {
        self.0.extend(v.to_le_bytes());
    }
    pub fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.0.extend(s.as_bytes());
    }
    pub fn f32_slice(&mut self, xs: &[f64]) {
        self.u32(xs.len() as u32);
        for &v in xs {
            self.f32(v);
        }
    }
    pub fn f64_slice(&mut self, xs: &[f64]) {
        self.u32(xs.len() as u32);
        for &v in xs {
            self.f64(v);
        }
    }
    pub fn usize_slice(&mut self, xs: &[usize]) {
        self.u32(xs.len() as u32);
        for &v in xs {
            self.u32(v as u32);
        }
    }
}

pub(crate) struct BinReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> BinReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos.checked_add(n).is_none_or(|e| e > self.buf.len()) {
            return Err(DopeError::Artifact(
                "format2 truncated binary section".into(),
            ));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn f32(&mut self) -> Result<f64> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64)
    }
    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn len(&mut self) -> Result<usize> {
        let n = self.u32()? as usize;
        if n > MAX_SEQ {
            return Err(DopeError::Artifact("format2 sequence exceeds limit".into()));
        }
        Ok(n)
    }
    pub fn str(&mut self) -> Result<String> {
        let n = self.len()?;
        let b = self.take(n)?;
        String::from_utf8(b.to_vec())
            .map_err(|_| DopeError::Artifact("format2 invalid utf8 string".into()))
    }
    pub fn f32_vec(&mut self) -> Result<Vec<f64>> {
        let n = self.len()?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.f32()?);
        }
        Ok(out)
    }
    pub fn f64_vec(&mut self) -> Result<Vec<f64>> {
        let n = self.len()?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.f64()?);
        }
        Ok(out)
    }
    pub fn usize_vec(&mut self) -> Result<Vec<usize>> {
        let n = self.len()?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.u32()? as usize);
        }
        Ok(out)
    }
}

fn write_col_spec(w: &mut BinWriter, c: &ColumnSpec) {
    w.str(&c.name);
    match &c.kind {
        Kind::Numeric(n) => {
            w.u8(0);
            w.u8((u8::from(n.log)) | (u8::from(n.integer) << 1));
            w.f64(n.lo);
            w.f64(n.hi);
        }
        Kind::Categorical(cat) => {
            w.u8(1);
            w.u32(cat.levels.len() as u32);
            for lv in &cat.levels {
                w.str(lv);
            }
        }
    }
}

fn read_col_spec(r: &mut BinReader<'_>) -> Result<ColumnSpec> {
    let name = r.str()?;
    let kind = match r.u8()? {
        0 => {
            let fl = r.u8()?;
            let lo = r.f64()?;
            let hi = r.f64()?;
            Kind::Numeric(NumericSpec {
                log: (fl & 1) != 0,
                integer: (fl & 2) != 0,
                lo,
                hi,
            })
        }
        1 => {
            let n = r.len()?;
            let mut levels = Vec::with_capacity(n);
            for _ in 0..n {
                levels.push(r.str()?);
            }
            Kind::Categorical(CategoricalSpec { levels })
        }
        _ => return Err(DopeError::Artifact("format2 invalid column kind".into())),
    };
    Ok(ColumnSpec { name, kind })
}

pub(crate) fn encode_schema_section(schema: &Schema, n_fit: u64, budget_bytes: u32) -> Vec<u8> {
    let mut w = BinWriter::new();
    w.u64(n_fit);
    w.u32(budget_bytes);
    w.u8(match schema.task {
        Task::Regression => 0,
        Task::Classification => 1,
    });
    w.u32(schema.features.len() as u32);
    for f in &schema.features {
        write_col_spec(&mut w, f);
    }
    write_col_spec(&mut w, &schema.target);
    w.0
}

pub(crate) fn decode_schema_section(bytes: &[u8]) -> Result<(Schema, u64, u32)> {
    let mut r = BinReader::new(bytes);
    let n_fit = r.u64()?;
    let budget_bytes = r.u32()?;
    let task = match r.u8()? {
        0 => Task::Regression,
        1 => Task::Classification,
        _ => return Err(DopeError::Artifact("format2 invalid task".into())),
    };
    let n_feat = r.len()?;
    let mut features = Vec::with_capacity(n_feat);
    for _ in 0..n_feat {
        features.push(read_col_spec(&mut r)?);
    }
    let target = read_col_spec(&mut r)?;
    Ok((
        Schema {
            features,
            target,
            task,
            dropped: Vec::new(),
        },
        n_fit,
        budget_bytes,
    ))
}

pub(crate) fn encode_marginals_section(marginals: &[ColumnMarginalV2]) -> Vec<u8> {
    let mut w = BinWriter::new();
    w.u32(marginals.len() as u32);
    for m in marginals {
        match m {
            ColumnMarginalV2::Discrete {
                values,
                probs,
                p_missing,
                support_k,
            } => {
                w.u8(0);
                w.f64(*p_missing);
                w.u32(*support_k as u32);
                w.f64_slice(values);
                w.f32_slice(probs);
            }
            ColumnMarginalV2::Continuous {
                atoms,
                lattice,
                heaping,
                smooth_u,
                smooth_x,
                offgrid_u,
                offgrid_x,
                p_missing,
                support_k,
            } => {
                w.u8(1);
                w.f64(*p_missing);
                w.u32(*support_k as u32);
                w.u32(atoms.len() as u32);
                for a in atoms {
                    w.f64(a.value);
                    w.f32(a.prob);
                    w.f32(a.u_lo);
                    w.f32(a.u_hi);
                    w.f32(a.z_lo);
                    w.f32(a.z_hi);
                }
                match lattice {
                    None => w.u8(0),
                    Some(l) => {
                        w.u8(1);
                        w.f64(l.origin);
                        w.f64(l.mantissa);
                        w.u8(l.dec_exp as u8);
                        w.f64(l.step);
                        w.u8(u8::from(l.f32_heritage));
                        w.f32(l.pi_offgrid);
                    }
                }
                match heaping {
                    None => w.u8(0),
                    Some(h) => {
                        w.u8(1);
                        w.u32(h.multiple);
                        w.f32(h.observed_share);
                        w.f32(h.excess);
                    }
                }
                w.f32_slice(smooth_u);
                w.f32_slice(smooth_x);
                w.f32_slice(offgrid_u);
                w.f32_slice(offgrid_x);
            }
        }
    }
    w.0
}

pub(crate) fn decode_marginals_section(bytes: &[u8]) -> Result<Vec<ColumnMarginalV2>> {
    let mut r = BinReader::new(bytes);
    let n = r.len()?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        match r.u8()? {
            0 => {
                let p_missing = r.f64()?;
                let support_k = r.u32()? as usize;
                let values = r.f64_vec()?;
                let probs = r.f32_vec()?;
                out.push(ColumnMarginalV2::Discrete {
                    values,
                    probs,
                    p_missing,
                    support_k,
                });
            }
            1 => {
                let p_missing = r.f64()?;
                let support_k = r.u32()? as usize;
                let na = r.len()?;
                let mut atoms = Vec::with_capacity(na);
                for _ in 0..na {
                    atoms.push(AtomV2 {
                        value: r.f64()?,
                        prob: r.f32()?,
                        u_lo: r.f32()?,
                        u_hi: r.f32()?,
                        z_lo: r.f32()?,
                        z_hi: r.f32()?,
                    });
                }
                let lattice = if r.u8()? == 1 {
                    Some(LatticeSpec {
                        origin: r.f64()?,
                        mantissa: r.f64()?,
                        dec_exp: r.u8()? as i8,
                        step: r.f64()?,
                        f32_heritage: r.u8()? != 0,
                        pi_offgrid: r.f32()?,
                    })
                } else {
                    None
                };
                let heaping = if r.u8()? == 1 {
                    Some(HeapingSpec {
                        multiple: r.u32()?,
                        observed_share: r.f32()?,
                        excess: r.f32()?,
                    })
                } else {
                    None
                };
                out.push(ColumnMarginalV2::Continuous {
                    atoms,
                    lattice,
                    heaping,
                    smooth_u: r.f32_vec()?,
                    smooth_x: r.f32_vec()?,
                    offgrid_u: r.f32_vec()?,
                    offgrid_x: r.f32_vec()?,
                    p_missing,
                    support_k,
                });
            }
            _ => return Err(DopeError::Artifact("format2 invalid marginal tag".into())),
        }
    }
    Ok(out)
}

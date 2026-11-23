//! Column schema stored in front of the DSE7 packet of a `DPA1` V7 payload.
//!
//! The packet holds numbers only; this section names the columns, keeps integer flags and
//! maps categorical codes and class indices back to their labels. With `--anonymize` only
//! the shape is stored and labels are positional (`x0`, `L0`, `y`, `C0`), as in Format 1.
//!
//! ```text
//! section = codec u8 (0 raw, 1 brotli) | stored length u16 LE | stored bytes
//! plain   = version u8 | flags u8 | n_features varint | kinds u8 x (n_features + 1)
//!           | level counts varint per categorical column (features, then target)
//!           | unless anonymized: names (features, target), then labels, as varint-length UTF-8
//! ```

use crate::error::{DopeError, Result};
use crate::schema::{Kind, Schema};
use std::io::Read;

const VERSION: u8 = 1;
const FLAG_ANONYMIZED: u8 = 1;
const CODEC_RAW: u8 = 0;
const CODEC_BROTLI: u8 = 1;
const KIND_REAL: u8 = 0;
const KIND_INTEGER: u8 = 1;
const KIND_CATEGORICAL: u8 = 2;
/// Inflated schema cap (fail closed against decompression bombs).
const MAX_PLAIN: usize = 1 << 20;
const MAX_FEATURES: usize = u16::MAX as usize;
const MAX_LEVELS: usize = 4096;
const MAX_TOTAL_LEVELS: usize = 1 << 18;

/// How one column's numbers are turned back into cells.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum V7Kind {
    Real,
    /// Decoded values are rounded.
    Integer,
    /// The packet carries the level index; `labels[k]` is level `k`.
    Categorical(Vec<String>),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct V7Column {
    pub(crate) name: String,
    pub(crate) kind: V7Kind,
}

/// Features in output order, then the target (categorical = classification).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct V7Schema {
    pub(crate) features: Vec<V7Column>,
    pub(crate) target: V7Column,
}

impl V7Schema {
    /// The same columns, kinds and labels as the Format 1 intake schema.
    pub(crate) fn from_schema(schema: &Schema) -> Self {
        let column = |name: &str, kind: &Kind| V7Column {
            name: name.to_string(),
            kind: match kind {
                Kind::Numeric(spec) if spec.integer => V7Kind::Integer,
                Kind::Numeric(_) => V7Kind::Real,
                Kind::Categorical(spec) => V7Kind::Categorical(spec.levels.clone()),
            },
        };
        Self {
            features: schema
                .features
                .iter()
                .map(|f| column(&f.name, &f.kind))
                .collect(),
            target: column(&schema.target.name, &schema.target.kind),
        }
    }

    /// Number of target classes (0 for regression).
    pub(crate) fn classes(&self) -> usize {
        match &self.target.kind {
            V7Kind::Categorical(labels) => labels.len(),
            _ => 0,
        }
    }

    /// CSV header: features, then the target last.
    pub(crate) fn headers(&self) -> Vec<String> {
        let mut h: Vec<String> = self.features.iter().map(|c| c.name.clone()).collect();
        h.push(self.target.name.clone());
        h
    }

    fn columns(&self) -> impl Iterator<Item = &V7Column> {
        self.features.iter().chain(std::iter::once(&self.target))
    }
}

fn put_varint(out: &mut Vec<u8>, mut v: usize) {
    while v >= 0x80 {
        out.push((v as u8 & 0x7f) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    put_varint(out, s.len());
    out.extend_from_slice(s.as_bytes());
}

fn plain(schema: &V7Schema, anonymize: bool) -> Vec<u8> {
    let mut out = vec![VERSION, if anonymize { FLAG_ANONYMIZED } else { 0 }];
    put_varint(&mut out, schema.features.len());
    for c in schema.columns() {
        out.push(match c.kind {
            V7Kind::Real => KIND_REAL,
            V7Kind::Integer => KIND_INTEGER,
            V7Kind::Categorical(_) => KIND_CATEGORICAL,
        });
    }
    for c in schema.columns() {
        if let V7Kind::Categorical(labels) = &c.kind {
            put_varint(&mut out, labels.len());
        }
    }
    if !anonymize {
        for c in schema.columns() {
            put_str(&mut out, &c.name);
        }
        for c in schema.columns() {
            if let V7Kind::Categorical(labels) = &c.kind {
                labels.iter().for_each(|l| put_str(&mut out, l));
            }
        }
    }
    out
}

fn brotli(plain: &[u8]) -> Result<Vec<u8>> {
    let bits = usize::BITS - plain.len().max(1).leading_zeros();
    let params = brotli::enc::BrotliEncoderParams {
        quality: 11,
        lgwin: (bits as i32).clamp(10, 24),
        size_hint: plain.len(),
        ..Default::default()
    };
    let mut z = Vec::new();
    brotli::BrotliCompress(&mut &plain[..], &mut z, &params)
        .map_err(|_| DopeError::Artifact("V7 schema compression failed".into()))?;
    Ok(z)
}

/// The stored schema section (the smaller of raw and brotli).
pub(crate) fn encode_section(schema: &V7Schema, anonymize: bool) -> Result<Vec<u8>> {
    let raw = plain(schema, anonymize);
    let z = brotli(&raw)?;
    let (codec, stored) = if z.len() < raw.len() {
        (CODEC_BROTLI, z)
    } else {
        (CODEC_RAW, raw)
    };
    let len = u16::try_from(stored.len()).map_err(|_| {
        DopeError::Budget("the V7 column schema is too large (try --anonymize)".into())
    })?;
    let mut out = Vec::with_capacity(3 + stored.len());
    out.push(codec);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&stored);
    Ok(out)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

fn bad(what: &str) -> DopeError {
    DopeError::Artifact(format!("V7 column schema is invalid ({what})"))
}

impl Cursor<'_> {
    fn byte(&mut self) -> Result<u8> {
        let b = *self.bytes.get(self.at).ok_or_else(|| bad("truncated"))?;
        self.at += 1;
        Ok(b)
    }

    fn varint(&mut self, max: usize) -> Result<usize> {
        let mut v = 0usize;
        for shift in (0..35).step_by(7) {
            let b = self.byte()?;
            v |= ((b & 0x7f) as usize) << shift;
            if b & 0x80 == 0 {
                return if v <= max {
                    Ok(v)
                } else {
                    Err(bad("count out of range"))
                };
            }
        }
        Err(bad("varint too long"))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.varint(self.bytes.len())?;
        let end = self.at.checked_add(n).ok_or_else(|| bad("truncated"))?;
        let s = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| bad("truncated"))?;
        self.at = end;
        String::from_utf8(s.to_vec()).map_err(|_| bad("label encoding"))
    }
}

fn parse_plain(bytes: &[u8]) -> Result<V7Schema> {
    let mut c = Cursor { bytes, at: 0 };
    if c.byte()? != VERSION {
        return Err(bad("version"));
    }
    let flags = c.byte()?;
    if flags & !FLAG_ANONYMIZED != 0 {
        return Err(bad("flags"));
    }
    let anonymized = flags & FLAG_ANONYMIZED != 0;
    let n = c.varint(MAX_FEATURES)?;
    if n == 0 {
        return Err(bad("no feature columns"));
    }
    let kinds: Vec<u8> = (0..=n).map(|_| c.byte()).collect::<Result<_>>()?;
    if kinds.iter().any(|&k| k > KIND_CATEGORICAL) {
        return Err(bad("column kind"));
    }
    let mut counts = Vec::new();
    for (i, &k) in kinds.iter().enumerate() {
        if k == KIND_CATEGORICAL {
            let levels = c.varint(MAX_LEVELS)?;
            let min = if i == n { 2 } else { 1 };
            if levels < min {
                return Err(bad("level count"));
            }
            counts.push(levels);
        }
    }
    if counts.iter().sum::<usize>() > MAX_TOTAL_LEVELS {
        return Err(bad("level count"));
    }
    let names: Vec<String> = if anonymized {
        (0..n)
            .map(|j| format!("x{j}"))
            .chain(["y".into()])
            .collect()
    } else {
        (0..=n).map(|_| c.string()).collect::<Result<_>>()?
    };
    let mut counts = counts.into_iter();
    let mut columns = Vec::with_capacity(n + 1);
    for (i, (name, &k)) in names.into_iter().zip(&kinds).enumerate() {
        let kind = match k {
            KIND_REAL => V7Kind::Real,
            KIND_INTEGER => V7Kind::Integer,
            _ => {
                let levels = counts.next().ok_or_else(|| bad("level count"))?;
                V7Kind::Categorical(if !anonymized {
                    (0..levels).map(|_| c.string()).collect::<Result<_>>()?
                } else if i == n {
                    (0..levels).map(|k| format!("C{k}")).collect()
                } else {
                    (0..levels).map(|k| format!("L{k}")).collect()
                })
            }
        };
        columns.push(V7Column { name, kind });
    }
    if c.at != bytes.len() {
        return Err(bad("trailing bytes"));
    }
    let target = columns.pop().ok_or_else(|| bad("no target"))?;
    Ok(V7Schema {
        features: columns,
        target,
    })
}

/// Parse a section at the start of `bytes`; returns the schema and the bytes it used.
pub(crate) fn decode_section(bytes: &[u8]) -> Result<(V7Schema, usize)> {
    let head = bytes.get(..3).ok_or_else(|| bad("truncated"))?;
    let len = u16::from_le_bytes([head[1], head[2]]) as usize;
    let stored = bytes.get(3..3 + len).ok_or_else(|| bad("truncated"))?;
    let plain = match head[0] {
        CODEC_RAW => stored.to_vec(),
        CODEC_BROTLI => {
            let mut out = Vec::new();
            brotli::Decompressor::new(stored, 4096)
                .take(MAX_PLAIN as u64 + 1)
                .read_to_end(&mut out)
                .map_err(|_| bad("compressed stream"))?;
            if out.len() > MAX_PLAIN {
                return Err(bad("inflation limit"));
            }
            out
        }
        _ => return Err(bad("codec")),
    };
    Ok((parse_plain(&plain)?, 3 + len))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> V7Schema {
        let col = |name: &str, kind: V7Kind| V7Column {
            name: name.into(),
            kind,
        };
        V7Schema {
            features: vec![
                col("age", V7Kind::Integer),
                col(
                    "city, region",
                    V7Kind::Categorical(vec!["a".into(), "é".into()]),
                ),
                col("score", V7Kind::Real),
            ],
            target: col(
                "label",
                V7Kind::Categorical(vec!["no".into(), "yes".into()]),
            ),
        }
    }

    #[test]
    fn section_round_trips_plain_and_anonymized() {
        let s = schema();
        let section = encode_section(&s, false).unwrap();
        let mut file = section.clone();
        file.extend_from_slice(b"packet");
        let (back, used) = decode_section(&file).unwrap();
        assert_eq!((back, used), (s.clone(), section.len()));
        let anon = encode_section(&s, true).unwrap();
        assert!(anon.len() < section.len());
        let (a, _) = decode_section(&anon).unwrap();
        assert_eq!(a.headers(), ["x0", "x1", "x2", "y"]);
        assert_eq!(
            a.features[1].kind,
            V7Kind::Categorical(vec!["L0".into(), "L1".into()])
        );
        assert_eq!(
            a.target.kind,
            V7Kind::Categorical(vec!["C0".into(), "C1".into()])
        );
        assert_eq!(a.classes(), 2);
    }

    #[test]
    fn wide_positional_names_compress_and_malformed_sections_fail_closed() {
        let features = (0..256)
            .map(|j| V7Column {
                name: format!("c{j}"),
                kind: V7Kind::Real,
            })
            .collect();
        let wide = V7Schema {
            features,
            target: V7Column {
                name: "target".into(),
                kind: V7Kind::Real,
            },
        };
        let section = encode_section(&wide, false).unwrap();
        assert_eq!(section[0], CODEC_BROTLI);
        assert!(section.len() < plain(&wide, false).len());
        assert_eq!(decode_section(&section).unwrap().0, wide);

        let good = encode_section(&schema(), false).unwrap();
        for cut in 0..good.len() {
            assert!(
                decode_section(&good[..cut]).is_err(),
                "accepted {cut} bytes"
            );
        }
        let mut codec = good.clone();
        codec[0] = 9;
        assert!(decode_section(&codec).is_err());
        let mut raw = vec![CODEC_RAW, 0, 0];
        let body = [VERSION, 0, 0];
        raw[1] = body.len() as u8;
        raw.extend_from_slice(&body);
        assert!(decode_section(&raw).is_err(), "zero features accepted");
    }
}

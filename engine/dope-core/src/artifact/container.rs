//! The program container: one brotli stream, stored as the only tensor of a
//! standard safetensors file.
//!
//! Inflated, the container is a little-endian `u32` length, that many bytes of
//! metadata JSON (schema, structure, bit widths and the section list
//! `"sec": [[name, bytes], ...]`), then each section's bytes in list order.
//! Compressing the schema text and the quantized codes together keeps column
//! names, category labels and the tensor table cheap: they would otherwise be
//! most of a few-kilobyte file.

use super::limits::{MAX_METADATA, MAX_SECTIONS};
use crate::error::{DopeError, Result};
use safetensors::{Dtype, SafeTensors, View};
use serde_json::{json, Value};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::io::Read;

/// `__metadata__["dope"]` of every artifact: the container format version.
pub const FORMAT: &str = "1";
/// Name of the file's one tensor (`U8`, the brotli stream).
pub const TENSOR: &str = "dope";
/// Largest inflated container accepted (a guard against decompression bombs).
pub const MAX_INFLATED: usize = 64 << 20;
/// Brotli quality (the densest setting; programs are a few kilobytes, so it is fast).
const QUALITY: i32 = 11;

// `shape` must return a borrowed slice; keep the length next to the bytes.
struct Stored {
    len: [usize; 1],
    bytes: Vec<u8>,
}

impl View for &Stored {
    fn dtype(&self) -> Dtype {
        Dtype::U8
    }
    fn shape(&self) -> &[usize] {
        &self.len
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.bytes)
    }
    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

/// Smallest brotli window (log2 bytes) that holds the whole input.
fn window(len: usize) -> i32 {
    let bits = usize::BITS - len.max(1).leading_zeros();
    (bits as i32).clamp(10, 24)
}

/// Compress `meta` and `sections` into one safetensors file.
pub(crate) fn pack(meta: &Value, sections: &BTreeMap<&str, Vec<u8>>) -> Result<Vec<u8>> {
    let mut meta = meta.clone();
    meta["sec"] = Value::Array(sections.iter().map(|(k, v)| json!([k, v.len()])).collect());
    let text = serde_json::to_vec(&meta)?;
    let mut plain =
        Vec::with_capacity(4 + text.len() + sections.values().map(Vec::len).sum::<usize>());
    plain.extend((text.len() as u32).to_le_bytes());
    plain.extend(&text);
    for v in sections.values() {
        plain.extend(v);
    }
    seal(&plain)
}

/// Compress an inflated container and wrap it as the file's one tensor.
fn seal(plain: &[u8]) -> Result<Vec<u8>> {
    let params = brotli::enc::BrotliEncoderParams {
        quality: QUALITY,
        lgwin: window(plain.len()),
        size_hint: plain.len(),
        ..Default::default()
    };
    let mut z = Vec::new();
    brotli::BrotliCompress(&mut &plain[..], &mut z, &params)
        .map_err(|e| DopeError::Artifact(format!("brotli compression: {e}")))?;
    let stored = Stored {
        len: [z.len()],
        bytes: z,
    };
    let info = HashMap::from([("dope".to_string(), FORMAT.to_string())]);
    safetensors::serialize([(TENSOR, &stored)], Some(info))
        .map_err(|e| DopeError::Artifact(format!("safetensors serialization: {e:?}")))
}

/// An unpacked artifact.
#[derive(Clone, Debug)]
pub struct Parts {
    /// Program structure and schema.
    pub meta: Value,
    /// Named sections (quantized parameters), inflated.
    pub sections: BTreeMap<String, Vec<u8>>,
    /// Bytes of the safetensors header (8-byte length plus JSON).
    pub header: usize,
    /// Bytes of the stored (compressed) program.
    pub stored: usize,
    /// Bytes of the inflated metadata JSON.
    pub meta_bytes: usize,
}

/// Read a safetensors file and inflate its program container.
pub fn unpack(bytes: &[u8]) -> Result<Parts> {
    let bad = |what: &str| DopeError::Artifact(format!("not a DOPE artifact ({what})"));
    let header_len = bytes.get(..8).ok_or_else(|| bad("header length"))?;
    let header_len = u64::from_le_bytes(header_len.try_into().map_err(|_| bad("header length"))?);
    if header_len > MAX_METADATA as u64 || bytes.len() > MAX_INFLATED + MAX_METADATA + 8 {
        return Err(bad("stored artifact limit"));
    }
    let (n, header) = SafeTensors::read_metadata(bytes).map_err(|_| bad("safetensors header"))?;
    match header.metadata().as_ref().and_then(|m| m.get("dope")) {
        Some(v) if v == FORMAT => {}
        Some(_) => return Err(bad("unsupported format version")),
        None => return Err(bad("missing dope metadata")),
    }
    let st = SafeTensors::deserialize(bytes).map_err(|_| bad("tensor table"))?;
    if st.len() != 1 {
        return Err(bad("one program tensor required"));
    }
    let tensor = st.tensor(TENSOR).map_err(|_| bad("program tensor"))?;
    let z = tensor.data();
    if z.len() > MAX_INFLATED || tensor.dtype() != Dtype::U8 || tensor.shape() != [z.len()] {
        return Err(bad("program tensor shape or dtype"));
    }
    let header_bytes = n.checked_add(8).ok_or_else(|| bad("header length"))?;
    if header_bytes.checked_add(z.len()) != Some(bytes.len()) {
        return Err(bad("unaccounted artifact bytes"));
    }
    let mut plain = Vec::new();
    brotli::Decompressor::new(z, 4096)
        .take(MAX_INFLATED as u64 + 1)
        .read_to_end(&mut plain)
        .map_err(|_| bad("compressed program"))?;
    if plain.len() > MAX_INFLATED {
        return Err(bad("program larger than the inflation limit"));
    }
    let m = plain
        .get(..4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        .ok_or_else(|| bad("container length"))?;
    if m > MAX_METADATA {
        return Err(bad("metadata limit"));
    }
    let metadata_end = m.checked_add(4).ok_or_else(|| bad("metadata length"))?;
    let text = plain
        .get(4..metadata_end)
        .ok_or_else(|| bad("container metadata"))?;
    let meta: Value = serde_json::from_slice(text).map_err(|_| bad("container metadata"))?;
    let list = meta["sec"].as_array().ok_or_else(|| bad("section list"))?;
    if list.len() > MAX_SECTIONS {
        return Err(bad("section count limit"));
    }
    let mut at = metadata_end;
    let mut sections = BTreeMap::new();
    for s in list {
        if s.as_array().is_none_or(|a| a.len() != 2) {
            return Err(bad("section list"));
        }
        let (name, len) = match (s[0].as_str(), s[1].as_u64()) {
            (Some(name), Some(len)) if name.len() <= 128 => (
                name.to_string(),
                usize::try_from(len).map_err(|_| bad("section length"))?,
            ),
            _ => return Err(bad("section list")),
        };
        let end = at.checked_add(len).ok_or_else(|| bad("section length"))?;
        let data = plain.get(at..end).ok_or_else(|| bad("section length"))?;
        if sections.insert(name, data.to_vec()).is_some() {
            return Err(bad("duplicate section"));
        }
        at = end;
    }
    if at != plain.len() {
        return Err(bad("trailing container bytes"));
    }
    Ok(Parts {
        meta,
        sections,
        header: header_bytes,
        stored: z.len(),
        meta_bytes: m,
    })
}

/// Write `parts` back into an artifact (tests and tooling: tamper, then re-pack).
#[doc(hidden)]
pub fn repack(meta: &Value, sections: &BTreeMap<String, Vec<u8>>) -> Result<Vec<u8>> {
    let borrowed: BTreeMap<&str, Vec<u8>> = sections
        .iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect();
    pack(meta, &borrowed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_round_trips_and_refuses_damage() {
        let mut s = BTreeMap::new();
        s.insert("a", vec![1u8, 2, 3]);
        s.insert("b", vec![]);
        s.insert("c", vec![9u8; 40]);
        let meta = json!({"v": 1, "cols": [["x", "n", 0]]});
        let bytes = pack(&meta, &s).unwrap();
        let p = unpack(&bytes).unwrap();
        assert_eq!(p.meta["cols"], meta["cols"]);
        assert_eq!(p.sections["a"], vec![1, 2, 3]);
        assert_eq!(p.sections["c"].len(), 40);
        assert_eq!(p.header + p.stored, bytes.len());
        // Truncated or bit-flipped streams are refused, never panicked on.
        for cut in [bytes.len() - 1, bytes.len() / 2, 9] {
            assert!(unpack(&bytes[..cut]).is_err());
        }
        let mut flipped = bytes.clone();
        let last = flipped.len() - 2;
        flipped[last] ^= 0x5a;
        let _ = unpack(&flipped); // may decode to other bytes, must not panic
                                  // A section list that disagrees with the bytes is refused.
        let container = |meta: &Value, body: &[u8]| {
            let text = serde_json::to_vec(meta).unwrap();
            let mut plain = (text.len() as u32).to_le_bytes().to_vec();
            plain.extend(text);
            plain.extend(body);
            seal(&plain).unwrap()
        };
        let body: Vec<u8> = [vec![1u8, 2, 3], vec![9u8; 40]].concat();
        assert!(unpack(&container(&p.meta, &body)).is_ok());
        let mut wrong = p.meta.clone();
        wrong["sec"][0][1] = json!(4);
        assert!(unpack(&container(&wrong, &body)).is_err());
        assert!(unpack(&container(&p.meta, &body[..10])).is_err());
        let mut extra = body.clone();
        extra.push(0);
        assert!(unpack(&container(&p.meta, &extra)).is_err());
        let mut dup = p.meta.clone();
        dup["sec"][1][0] = json!("a");
        assert!(unpack(&container(&dup, &body)).is_err());
        for entry in [
            json!(["a", u64::MAX]),
            json!(["a", -1]),
            json!(["a", "3"]),
            json!(["a", 3, 0]),
            json!({"0":"a", "1":3}),
        ] {
            let mut malformed = p.meta.clone();
            malformed["sec"][0] = entry;
            assert!(unpack(&container(&malformed, &body)).is_err());
        }
    }

    #[test]
    fn window_covers_the_input() {
        assert_eq!(window(0), 10);
        assert_eq!(window(3000), 12);
        assert_eq!(window(1 << 30), 24);
    }
}

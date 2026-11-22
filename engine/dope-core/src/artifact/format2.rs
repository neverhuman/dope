//! `.dope` Format 2 binary section container (O-2): length-prefixed binary sections,
//! nested tier prefix table, IEEE CRC32, zero JSON inside payload cap, hard parser bounds
//! (`<= 64 MiB` compressed / `<= 256 MiB` decoded RAM), and `f64 -> f32 -> i16` quantization
//! (`T13`, `T14`, `T15`, `T24`).

use crate::error::{DopeError, Result};
use brotli::{CompressorWriter, Decompressor};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

pub const FORMAT2_MAGIC: &[u8; 8] = b"DOPE2\0\0\0";
pub const FORMAT2_VERSION: u16 = 2;
pub const MAX_FORMAT2_COMPRESSED_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_FORMAT2_DECODED_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_FORMAT2_SECTIONS: usize = 64;
pub const MAX_FORMAT2_TIERS: usize = 16;

pub const TAG_UNIT_SCHEMA: u16 = 1;
pub const TAG_MARGINALS_V2: u16 = 2;
pub const TAG_CATF: u16 = 3;
pub const TAG_RULES: u16 = 4;
pub const TAG_STUDENT_MAIN: u16 = 5;
pub const TAG_STUDENT_PAIRS: u16 = 6;
pub const TAG_STUDENT_FOREST: u16 = 7;
pub const TAG_NOISE_TEMP: u16 = 8;
pub const TAG_EMBED_SUMMARY: u16 = 9;

pub const CODEC_RAW: u8 = 0;
pub const CODEC_BROTLI: u8 = 1;
pub const CODEC_Q16: u8 = 2;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierPrefixEntry {
    pub tier_budget_bytes: u32,
    pub section_prefix_count: u32,
    pub forest_trees_prefix: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Format2Section {
    pub tag: u16,
    pub codec: u8,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Format2Packet {
    pub flags: u16,
    pub tiers: Vec<TierPrefixEntry>,
    pub sections: Vec<Format2Section>,
}

/// Pure-Rust IEEE 802.3 CRC32 (zero extra crate dependency).
pub fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffff_u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

pub fn is_format2(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[..8] == FORMAT2_MAGIC
}

/// Encode a `Format2Packet` into a self-contained brotli-compressed binary stream with
/// CRC32 integrity check and nested tier prefix table (zero JSON inside payload cap!).
pub fn encode_format2(pkt: &Format2Packet) -> Result<Vec<u8>> {
    if pkt.sections.len() > MAX_FORMAT2_SECTIONS || pkt.tiers.len() > MAX_FORMAT2_TIERS {
        return Err(DopeError::Artifact(
            "format2 header count exceeds limits".into(),
        ));
    }
    let mut body = Vec::new();
    for t in &pkt.tiers {
        body.extend(t.tier_budget_bytes.to_le_bytes());
        body.extend(t.section_prefix_count.to_le_bytes());
        body.extend(t.forest_trees_prefix.to_le_bytes());
    }
    for s in &pkt.sections {
        let len = u32::try_from(s.data.len())
            .map_err(|_| DopeError::Artifact("format2 section exceeds u32".into()))?;
        body.extend(s.tag.to_le_bytes());
        body.push(s.codec);
        body.push(0u8); // reserved alignment byte
        body.extend(len.to_le_bytes());
        body.extend(&s.data);
    }
    if body.len() > MAX_FORMAT2_DECODED_BYTES {
        return Err(DopeError::Artifact(
            "format2 decoded size exceeds 256 MiB".into(),
        ));
    }
    let crc = crc32_ieee(&body);
    let mut compressed = Vec::new();
    {
        let mut enc = CompressorWriter::new(&mut compressed, 4096, 6, 22);
        enc.write_all(&body)
            .map_err(|_| DopeError::Artifact("format2 compression failed".into()))?;
        enc.flush()
            .map_err(|_| DopeError::Artifact("format2 compression flush failed".into()))?;
    }
    if compressed.len() + 32 > MAX_FORMAT2_COMPRESSED_BYTES {
        return Err(DopeError::Artifact(
            "format2 compressed size exceeds 64 MiB".into(),
        ));
    }
    let mut out = Vec::with_capacity(32 + compressed.len());
    out.extend(FORMAT2_MAGIC);
    out.extend(FORMAT2_VERSION.to_le_bytes());
    out.extend(pkt.flags.to_le_bytes());
    out.extend((pkt.sections.len() as u32).to_le_bytes());
    out.extend((pkt.tiers.len() as u32).to_le_bytes());
    out.extend(crc.to_le_bytes());
    out.extend((body.len() as u64).to_le_bytes());
    out.extend(&compressed);
    Ok(out)
}

/// Fail-closed Format 2 decoder enforcing hard byte/RAM/count limits and CRC32 (`T13`, `T14`).
pub fn decode_format2(bytes: &[u8]) -> Result<Format2Packet> {
    if bytes.len() < 32 || bytes.len() > MAX_FORMAT2_COMPRESSED_BYTES {
        return Err(DopeError::Artifact(
            "format2 file size out of bounds".into(),
        ));
    }
    if &bytes[..8] != FORMAT2_MAGIC {
        return Err(DopeError::Artifact("invalid format2 magic".into()));
    }
    let version = u16::from_le_bytes([bytes[8], bytes[9]]);
    if version != FORMAT2_VERSION {
        return Err(DopeError::Artifact("unsupported format2 version".into()));
    }
    let flags = u16::from_le_bytes([bytes[10], bytes[11]]);
    let section_count = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let tier_count = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
    let expected_crc = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    let declared_decoded = u64::from_le_bytes([
        bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
    ]);
    if section_count > MAX_FORMAT2_SECTIONS
        || tier_count > MAX_FORMAT2_TIERS
        || declared_decoded > MAX_FORMAT2_DECODED_BYTES as u64
    {
        return Err(DopeError::Artifact(
            "format2 header exceeds safety bounds".into(),
        ));
    }
    let declared_len = declared_decoded as usize;
    // Bounded decompression preventing decompression bombs (`T14`).
    let mut dec = Decompressor::new(&bytes[32..], 4096);
    let mut body = Vec::with_capacity(declared_len.min(1 << 20));
    let mut chunk = [0u8; 8192];
    loop {
        let n = dec
            .read(&mut chunk)
            .map_err(|_| DopeError::Artifact("format2 brotli decode failed".into()))?;
        if n == 0 {
            break;
        }
        if body.len().saturating_add(n) > declared_len {
            return Err(DopeError::Artifact(
                "format2 inflated bytes exceed declared bound".into(),
            ));
        }
        body.extend_from_slice(&chunk[..n]);
    }
    if body.len() != declared_len {
        return Err(DopeError::Artifact(
            "format2 decoded length mismatch".into(),
        ));
    }
    if crc32_ieee(&body) != expected_crc {
        return Err(DopeError::Artifact(
            "format2 CRC32 checksum mismatch".into(),
        ));
    }

    let mut at = 0usize;
    let tier_bytes = tier_count
        .checked_mul(12)
        .ok_or_else(|| DopeError::Artifact("format2 tier overflow".into()))?;
    if at + tier_bytes > body.len() {
        return Err(DopeError::Artifact("format2 truncated tier table".into()));
    }
    let mut tiers = Vec::with_capacity(tier_count);
    for _ in 0..tier_count {
        let tier_budget_bytes = u32::from_le_bytes(body[at..at + 4].try_into().unwrap());
        let section_prefix_count = u32::from_le_bytes(body[at + 4..at + 8].try_into().unwrap());
        let forest_trees_prefix = u32::from_le_bytes(body[at + 8..at + 12].try_into().unwrap());
        if section_prefix_count as usize > section_count {
            return Err(DopeError::Artifact(
                "format2 tier prefix out of range".into(),
            ));
        }
        tiers.push(TierPrefixEntry {
            tier_budget_bytes,
            section_prefix_count,
            forest_trees_prefix,
        });
        at += 12;
    }

    let mut sections = Vec::with_capacity(section_count);
    for _ in 0..section_count {
        if at + 8 > body.len() {
            return Err(DopeError::Artifact(
                "format2 truncated section header".into(),
            ));
        }
        let tag = u16::from_le_bytes([body[at], body[at + 1]]);
        let codec = body[at + 2];
        let reserved = body[at + 3];
        if reserved != 0 || codec > CODEC_Q16 {
            return Err(DopeError::Artifact("format2 invalid section header".into()));
        }
        let len = u32::from_le_bytes(body[at + 4..at + 8].try_into().unwrap()) as usize;
        at += 8;
        if at.checked_add(len).is_none_or(|end| end > body.len()) {
            return Err(DopeError::Artifact(
                "format2 truncated section payload".into(),
            ));
        }
        sections.push(Format2Section {
            tag,
            codec,
            data: body[at..at + len].to_vec(),
        });
        at += len;
    }
    if at != body.len() {
        return Err(DopeError::Artifact(
            "format2 trailing bytes after sections".into(),
        ));
    }
    Ok(Format2Packet {
        flags,
        tiers,
        sections,
    })
}

/// Measured `f64 -> f32 -> i16` affine quantization (`T15`).
pub fn quantize_f64_i16(vals: &[f64]) -> Result<Vec<u8>> {
    if vals.iter().any(|v| !v.is_finite()) {
        return Err(DopeError::Artifact(
            "non-finite value in q16 quantization".into(),
        ));
    }
    let f32s: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
    let lo = f32s.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = f32s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (lo, span) = if f32s.is_empty() || hi <= lo {
        (if f32s.is_empty() { 0.0f32 } else { lo }, 0.0f32)
    } else {
        (lo, hi - lo)
    };
    let mut out = Vec::with_capacity(12 + vals.len() * 2);
    out.extend((vals.len() as u32).to_le_bytes());
    out.extend(lo.to_le_bytes());
    out.extend(span.to_le_bytes());
    for v in f32s {
        let q: i16 = if span > 0.0 {
            let norm = ((v - lo) / span).clamp(0.0, 1.0);
            ((norm * 2.0 - 1.0) * 32767.0)
                .round()
                .clamp(-32767.0, 32767.0) as i16
        } else {
            0
        };
        out.extend(q.to_le_bytes());
    }
    Ok(out)
}

pub fn dequantize_i16_f64(bytes: &[u8]) -> Result<Vec<f64>> {
    if bytes.len() < 12 {
        return Err(DopeError::Artifact("truncated q16 header".into()));
    }
    let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let lo = f32::from_le_bytes(bytes[4..8].try_into().unwrap()) as f64;
    let span = f32::from_le_bytes(bytes[8..12].try_into().unwrap()) as f64;
    if !lo.is_finite() || !span.is_finite() || span < 0.0 || bytes.len() != 12 + count * 2 {
        return Err(DopeError::Artifact("invalid q16 payload".into()));
    }
    let mut out = Vec::with_capacity(count);
    for c in bytes[12..].chunks_exact(2) {
        let q = i16::from_le_bytes([c[0], c[1]]);
        let v = if span > 0.0 {
            lo + ((q as f64 / 32767.0) + 1.0) * 0.5 * span
        } else {
            lo
        };
        out.push(v);
    }
    Ok(out)
}

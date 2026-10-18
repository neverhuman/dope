//! Candidate DPK4.0 wire format; legacy writers/readers remain unchanged.
//!
//! All integers and f32 tensors are little endian. The 64-byte fixed header is
//! magic, major/minor, header size (u16), required features (u32), packet count
//! (u32), payload byte/bit lengths (u64 each), then SHA-256 of the first 32
//! header bytes followed by the payload. Each 32-byte packet header contains
//! type/encoding (u8 each), flags (u16), element count (u32), byte/bit lengths
//! (u64 each), CRC32 (IEEE), and reserved (u32). Flags/reserved must be zero.
//! Q4 puts the first value in the low nibble; an unused high nibble is zero.
//! Integrity checks detect corruption; they do not authenticate a producer.

use sha2::{Digest, Sha256};

use crate::error::{DopeError, Result};

#[path = "dpk4_packets.rs"]
mod packets;
pub use packets::{decode_kernel, encode_kernel};

pub const MAGIC: &[u8; 4] = b"DPK4";
pub const HEADER_BYTES: usize = 64;
pub const PACKET_HEADER_BYTES: usize = 32;
pub const MAX_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_VALUES: usize = 1_048_576;
pub const MAX_PACKETS: usize = 64;
const SUPPORTED_FEATURES: u32 = 15;

/// Numeric type tags are also their required-feature bit positions.
#[derive(Clone, Debug, PartialEq)]
pub enum Packet {
    Q4(Vec<u8>),
    Q8(Vec<u8>),
    F32(Vec<f32>),
    /// Canonical model structure with numeric tensor placeholders.
    Model(Vec<u8>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PacketAccounting {
    pub tag: u8,
    pub elements: usize,
    pub framing_bytes: usize,
    pub payload_bytes: usize,
    pub payload_bits: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Accounting {
    pub header_bytes: usize,
    pub packets: Vec<PacketAccounting>,
    pub total_bytes: usize,
}

fn invalid(reason: &'static str) -> DopeError {
    DopeError::Codec(reason.into())
}

/// IEEE CRC32, with no table or optional entropy dependency.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & 0u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}

fn lengths(tag: u8, count: usize) -> Result<(usize, usize)> {
    if count > MAX_VALUES && tag != 4 || count > MAX_BYTES {
        return Err(invalid("DPK4 element count exceeds its bound"));
    }
    let width = match tag {
        1 => 4,
        2 | 4 => 8,
        3 => 32,
        _ => return Err(invalid("DPK4 unknown packet type")),
    };
    let bits = count
        .checked_mul(width)
        .ok_or_else(|| invalid("DPK4 length overflow"))?;
    Ok((bits.div_ceil(8), bits))
}

/// Pack already quantized integer levels, rejecting out-of-range q4 values.
pub fn pack_levels(values: &[u8], bits: u8) -> Result<Vec<u8>> {
    let tag = match bits {
        4 => 1,
        8 => 2,
        _ => return Err(invalid("DPK4 unsupported precision")),
    };
    let (bytes, _) = lengths(tag, values.len())?;
    if bits == 8 {
        return Ok(values.to_vec());
    }
    if values.iter().any(|&value| value > 15) {
        return Err(invalid("DPK4 invalid quantized level"));
    }
    let mut packed = vec![0; bytes];
    for (index, &value) in values.iter().enumerate() {
        packed[index / 2] |= value << ((index % 2) * 4);
    }
    Ok(packed)
}

pub fn unpack_levels(payload: &[u8], count: usize, bits: u8) -> Result<Vec<u8>> {
    let tag = match bits {
        4 => 1,
        8 => 2,
        _ => return Err(invalid("DPK4 unsupported precision")),
    };
    let (bytes, _) = lengths(tag, count)?;
    if payload.len() != bytes {
        return Err(invalid("DPK4 numeric payload length mismatch"));
    }
    if bits == 8 {
        return Ok(payload.to_vec());
    }
    if count % 2 == 1 && payload.last().is_some_and(|value| value & 0xf0 != 0) {
        return Err(invalid("DPK4 nonzero padding bits"));
    }
    Ok((0..count)
        .map(|index| (payload[index / 2] >> ((index % 2) * 4)) & 15)
        .collect())
}

impl Packet {
    fn wire(&self) -> Result<(u8, usize, Vec<u8>)> {
        let (tag, count) = match self {
            Self::Q4(values) => (1, values.len()),
            Self::Q8(values) => (2, values.len()),
            Self::F32(values) => (3, values.len()),
            Self::Model(bytes) => (4, bytes.len()),
        };
        lengths(tag, count)?;
        let payload = match self {
            Self::Q4(values) => pack_levels(values, 4)?,
            Self::Q8(values) => pack_levels(values, 8)?,
            Self::F32(values) => {
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(invalid("DPK4 non-finite numeric tensor"));
                }
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect()
            }
            Self::Model(bytes) => bytes.clone(),
        };
        Ok((tag, count, payload))
    }
}

pub fn encode(packets: &[Packet]) -> Result<Vec<u8>> {
    if packets.is_empty() || packets.len() > MAX_PACKETS {
        return Err(invalid("DPK4 packet count exceeds its bound"));
    }
    let mut payload = Vec::new();
    let mut features = 0u32;
    let mut total_values = 0usize;
    for packet in packets {
        let (tag, count, bytes) = packet.wire()?;
        if tag != 4 {
            total_values += count;
            if total_values > MAX_VALUES {
                return Err(invalid("DPK4 numeric allocation exceeds its bound"));
            }
        }
        let (_, bits) = lengths(tag, count)?;
        if HEADER_BYTES + payload.len() + PACKET_HEADER_BYTES + bytes.len() > MAX_BYTES {
            return Err(invalid("DPK4 artifact exceeds its bound"));
        }
        features |= 1 << (tag - 1);
        payload.extend_from_slice(&[tag, 0, 0, 0]);
        payload.extend_from_slice(&(count as u32).to_le_bytes());
        payload.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        payload.extend_from_slice(&(bits as u64).to_le_bytes());
        payload.extend_from_slice(&crc32(&bytes).to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&bytes);
    }
    let mut artifact = Vec::with_capacity(HEADER_BYTES + payload.len());
    artifact.extend_from_slice(MAGIC);
    artifact.extend_from_slice(&[4, 0]);
    artifact.extend_from_slice(&(HEADER_BYTES as u16).to_le_bytes());
    artifact.extend_from_slice(&features.to_le_bytes());
    artifact.extend_from_slice(&(packets.len() as u32).to_le_bytes());
    artifact.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    artifact.extend_from_slice(&((payload.len() as u64) * 8).to_le_bytes());
    let mut digest = Sha256::new();
    digest.update(&artifact);
    digest.update(&payload);
    artifact.extend_from_slice(&digest.finalize());
    artifact.extend_from_slice(&payload);
    Ok(artifact)
}

fn number(bytes: &[u8], range: std::ops::Range<usize>) -> Result<usize> {
    let raw = bytes
        .get(range)
        .ok_or_else(|| invalid("DPK4 truncated header"))?;
    let mut value = [0u8; 8];
    value[..raw.len()].copy_from_slice(raw);
    usize::try_from(u64::from_le_bytes(value)).map_err(|_| invalid("DPK4 length exceeds host size"))
}

fn scan(artifact: &[u8]) -> Result<(Vec<Packet>, Accounting)> {
    if !(HEADER_BYTES..=MAX_BYTES).contains(&artifact.len()) {
        return Err(invalid("DPK4 artifact violates size bounds"));
    }
    if &artifact[..4] != MAGIC {
        return Err(invalid("DPK4 invalid magic"));
    }
    if artifact[4..8] != [4, 0, HEADER_BYTES as u8, 0] {
        return Err(invalid("DPK4 unsupported version or header size"));
    }
    let features = number(artifact, 8..12)? as u32;
    let count = number(artifact, 12..16)?;
    if features & !SUPPORTED_FEATURES != 0 || count == 0 || count > MAX_PACKETS {
        return Err(invalid("DPK4 unsupported features or packet count"));
    }
    let payload_bytes = number(artifact, 16..24)?;
    if payload_bytes != artifact.len() - HEADER_BYTES
        || number(artifact, 24..32)? != payload_bytes * 8
    {
        return Err(invalid("DPK4 payload length mismatch"));
    }
    let mut digest = Sha256::new();
    digest.update(&artifact[..32]);
    digest.update(&artifact[HEADER_BYTES..]);
    if digest.finalize()[..] != artifact[32..64] {
        return Err(invalid("DPK4 SHA-256 mismatch"));
    }
    let mut offset = HEADER_BYTES;
    let mut seen = 0u32;
    let mut packets = Vec::with_capacity(count);
    let mut accounting = Vec::with_capacity(count);
    let mut total_values = 0usize;
    for _ in 0..count {
        let header = artifact
            .get(offset..offset + PACKET_HEADER_BYTES)
            .ok_or_else(|| invalid("DPK4 truncated packet header"))?;
        let tag = header[0];
        if header[1..4] != [0; 3] || header[28..32] != [0; 4] {
            return Err(invalid("DPK4 unsupported packet encoding or flags"));
        }
        let elements = number(header, 4..8)?;
        let (bytes, bits) = lengths(tag, elements)?;
        if number(header, 8..16)? != bytes || number(header, 16..24)? != bits {
            return Err(invalid("DPK4 packet length mismatch"));
        }
        if tag != 4 {
            total_values += elements;
            if total_values > MAX_VALUES {
                return Err(invalid("DPK4 numeric allocation exceeds its bound"));
            }
        }
        offset += PACKET_HEADER_BYTES;
        let payload = artifact
            .get(offset..offset + bytes)
            .ok_or_else(|| invalid("DPK4 truncated packet payload"))?;
        let expected_crc = u32::from_le_bytes(header[24..28].try_into().unwrap());
        if crc32(payload) != expected_crc {
            return Err(invalid("DPK4 CRC32 mismatch"));
        }
        let packet = match tag {
            1 => Packet::Q4(unpack_levels(payload, elements, 4)?),
            2 => Packet::Q8(unpack_levels(payload, elements, 8)?),
            3 => {
                let values: Vec<f32> = payload
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                    .collect();
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(invalid("DPK4 non-finite numeric tensor"));
                }
                Packet::F32(values)
            }
            4 => Packet::Model(payload.to_vec()),
            _ => return Err(invalid("DPK4 unknown packet type")),
        };
        packets.push(packet);
        accounting.push(PacketAccounting {
            tag,
            elements,
            framing_bytes: PACKET_HEADER_BYTES,
            payload_bytes: bytes,
            payload_bits: bits,
        });
        seen |= 1 << (tag - 1);
        offset += bytes;
    }
    if offset != artifact.len() || seen != features {
        return Err(invalid("DPK4 trailing bytes or feature mismatch"));
    }
    Ok((
        packets,
        Accounting {
            header_bytes: HEADER_BYTES,
            packets: accounting,
            total_bytes: artifact.len(),
        },
    ))
}

pub fn decode(artifact: &[u8]) -> Result<Vec<Packet>> {
    scan(artifact).map(|(packets, _)| packets)
}

/// Full validation precedes inspection; every precision/framing byte is charged.
pub fn inspect(artifact: &[u8]) -> Result<Accounting> {
    scan(artifact).map(|(_, accounting)| accounting)
}

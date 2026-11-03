//! Bounded independent packet envelope. Header+body count toward complete cap.
use super::{Packet, Result, WireError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
/// Canonical whole-file byte caps. Historical 4k belongs to unchanged DML3 readers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Level {
    #[serde(rename = "2k")]
    L2k,
    #[serde(rename = "16k")]
    L16k,
    #[serde(rename = "64k")]
    L64k,
}
impl Level {
    pub const fn cap(self) -> usize {
        match self {
            Self::L2k => 2048,
            Self::L16k => 16384,
            Self::L64k => 65536,
        }
    }
}
pub const HEADER_BYTES: usize = 13;
const VERSION: u8 = 1;
/// Encode valid numerical body. Safe typed refusal instead of over-cap success.
pub fn encode(packet: &Packet) -> Result<Vec<u8>> {
    packet.validate()?;
    let body = serde_json::to_vec(packet).map_err(|_| WireError::new("serialization_failed"))?;
    if body
        .len()
        .checked_add(HEADER_BYTES)
        .is_none_or(|n| n > packet.level.cap())
    {
        return Err(WireError::new("over_budget"));
    }
    let length = u32::try_from(body.len()).map_err(|_| WireError::new("over_budget"))?;
    let mut out = Vec::with_capacity(body.len() + HEADER_BYTES);
    out.extend_from_slice(b"DML4");
    out.push(VERSION);
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(&crc32(&body).to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}
/// Refuse corrupt/truncated/oversize input before JSON parsing or unbounded allocation.
pub fn decode(bytes: &[u8]) -> Result<Packet> {
    if bytes.len() < HEADER_BYTES
        || bytes.len() > 65536
        || &bytes[..4] != b"DML4"
        || bytes[4] != VERSION
    {
        return Err(WireError::new("invalid_envelope"));
    }
    let n = u32::from_le_bytes(
        bytes[5..9]
            .try_into()
            .map_err(|_| WireError::new("invalid_envelope"))?,
    ) as usize;
    if n.checked_add(HEADER_BYTES) != Some(bytes.len()) {
        return Err(WireError::new("length_mismatch"));
    }
    let expected = u32::from_le_bytes(
        bytes[9..13]
            .try_into()
            .map_err(|_| WireError::new("invalid_envelope"))?,
    );
    let body = &bytes[HEADER_BYTES..];
    if crc32(body) != expected {
        return Err(WireError::new("checksum_mismatch"));
    }
    let packet: Packet =
        serde_json::from_slice(body).map_err(|_| WireError::new("invalid_packet_json"))?;
    packet.validate()?;
    if bytes.len() > packet.level.cap() {
        return Err(WireError::new("over_budget"));
    }
    Ok(packet)
}
/// Actual byte identity, never a claim about privacy or memorization capacity.
pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
/// IEEE CRC32 for framing corruption detection, not authentication.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb88320u32 & (0u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

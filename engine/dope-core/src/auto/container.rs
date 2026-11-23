//! The `DPA1` container of `dope fit --format auto`.
//!
//! ```text
//! offset  bytes  field
//!      0      4  magic "DPA1"
//!      4      1  encoder tag: 1 = Format 1 artifact, 7 = DSE7 v2 packet with its column schema
//!      5      8  first 8 bytes of SHA-256 of the SingleEncV7 weights (all zero for tag 1)
//!     13      -  payload (validated by the payload's own decoder)
//! ```
//!
//! The whole file, header included, is what the byte budget caps.

use crate::error::{DopeError, Result};

pub const DPA1_MAGIC: &[u8; 4] = b"DPA1";
/// Magic, encoder tag and weights fingerprint.
pub const DPA1_HEADER_BYTES: usize = 13;
/// Fingerprint field of a payload that needs no V7 weights.
pub const NO_FINGERPRINT: [u8; 8] = [0; 8];

/// Which encoder produced the payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoder {
    /// A Format 1 artifact (`dope fit --format 1`), capped at the budget minus the header.
    Format1,
    /// A self-contained DSE7 v2 packet decoded with the SingleEncV7 weights it names.
    V7,
}

impl Encoder {
    pub fn tag(self) -> u8 {
        match self {
            Encoder::Format1 => 1,
            Encoder::V7 => 7,
        }
    }

    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Encoder::Format1),
            7 => Some(Encoder::V7),
            _ => None,
        }
    }

    /// Stable name used in reports.
    pub fn name(self) -> &'static str {
        match self {
            Encoder::Format1 => "format1",
            Encoder::V7 => "v7",
        }
    }
}

/// A parsed container; `payload` borrows the file bytes after the header.
#[derive(Clone, Copy, Debug)]
pub struct Dpa1<'a> {
    pub encoder: Encoder,
    pub fingerprint: [u8; 8],
    pub payload: &'a [u8],
}

/// True when `bytes` start with the `DPA1` magic (the payload may still be invalid).
pub fn is_dpa1(bytes: &[u8]) -> bool {
    bytes.len() >= DPA1_MAGIC.len() && &bytes[..DPA1_MAGIC.len()] == DPA1_MAGIC
}

/// Prefix `payload` with the container header.
pub fn wrap_dpa1(encoder: Encoder, fingerprint: [u8; 8], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(DPA1_HEADER_BYTES + payload.len());
    out.extend_from_slice(DPA1_MAGIC);
    out.push(encoder.tag());
    out.extend_from_slice(&fingerprint);
    out.extend_from_slice(payload);
    out
}

/// Parse the header. Fails closed on a short file, an unknown tag, a Format 1 header that
/// names weights, or an empty payload; the payload itself is checked by its decoder.
pub fn parse_dpa1(bytes: &[u8]) -> Result<Dpa1<'_>> {
    let bad = |what: &str| DopeError::Artifact(format!("not a valid DPA1 artifact ({what})"));
    if !is_dpa1(bytes) {
        return Err(bad("magic"));
    }
    if bytes.len() <= DPA1_HEADER_BYTES {
        return Err(bad("truncated header or empty payload"));
    }
    let encoder = Encoder::from_tag(bytes[4]).ok_or_else(|| bad("unknown encoder tag"))?;
    let mut fingerprint = [0u8; 8];
    fingerprint.copy_from_slice(&bytes[5..DPA1_HEADER_BYTES]);
    match encoder {
        Encoder::Format1 if fingerprint != NO_FINGERPRINT => {
            return Err(bad("Format 1 payload with a weights fingerprint"))
        }
        Encoder::V7 if fingerprint == NO_FINGERPRINT => {
            return Err(bad("V7 payload without a weights fingerprint"))
        }
        _ => {}
    }
    Ok(Dpa1 {
        encoder,
        fingerprint,
        payload: &bytes[DPA1_HEADER_BYTES..],
    })
}

/// Lowercase hex of a fingerprint (weights identity, never data).
pub fn fingerprint_hex(fingerprint: &[u8; 8]) -> String {
    fingerprint.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip_and_fail_closed_cases() {
        let fp = [1, 2, 3, 4, 5, 6, 7, 8];
        let file = wrap_dpa1(Encoder::V7, fp, b"payload");
        assert_eq!(file.len(), DPA1_HEADER_BYTES + 7);
        let parsed = parse_dpa1(&file).unwrap();
        assert_eq!(parsed.encoder, Encoder::V7);
        assert_eq!(parsed.fingerprint, fp);
        assert_eq!(parsed.payload, b"payload");
        assert_eq!(fingerprint_hex(&fp), "0102030405060708");

        let f1 = wrap_dpa1(Encoder::Format1, NO_FINGERPRINT, b"x");
        assert_eq!(parse_dpa1(&f1).unwrap().encoder, Encoder::Format1);
        for bad in [
            &file[..DPA1_HEADER_BYTES],
            &file[..3],
            &wrap_dpa1(Encoder::Format1, fp, b"x")[..],
            &wrap_dpa1(Encoder::V7, NO_FINGERPRINT, b"x")[..],
        ] {
            assert!(parse_dpa1(bad).is_err());
        }
        let mut unknown = file.clone();
        unknown[4] = 2;
        let err = parse_dpa1(&unknown).unwrap_err().to_string();
        assert!(err.contains("unknown encoder tag"), "{err}");
        assert!(!is_dpa1(b"DOPE2\0\0\0") && is_dpa1(b"DPA1"));
    }
}

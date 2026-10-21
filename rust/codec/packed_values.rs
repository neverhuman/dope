// Packed DPK3 values.
//
// A DPK3 artifact records its quantization grid in the header (`bits`), yet the
// historical value codec writes every unit-interval parameter as a u16 and every
// unbounded parameter as varint(round(1e6 * v)) whatever the grid. Header flag
// bit 1 selects the packed value codec instead: a unit parameter is written as
// its grid code (one byte up to 8 bits, two bytes above), and an unbounded
// parameter as the zigzag varint of its grid index k, where v = k * 2^-(bits-2).
// Values that are not on the grid (for example correlations clamped to 0.995)
// keep the historical 1e-6 encoding behind a one-bit escape. The section layout,
// rANS stage and header are unchanged, so packed artifacts decode to the same
// quantized program; historical artifacts never set the flag and are untouched.

use std::cell::Cell;

/// Header flag bit marking packed value encoding.
pub const PACKED_FLAG: u16 = 2;
/// Quantization grids the packed codec accepts.
const QUANTIZATION_GRID_BITS: [u8; 6] = [4, 6, 8, 10, 12, 16];

thread_local! {
    static PACKED_BITS: Cell<Option<u8>> = const { Cell::new(None) };
}

/// Scope guard for the packed value mode of the current encode or decode call.
struct PackedScope(Option<u8>);

impl PackedScope {
    fn enter(bits: Option<u8>) -> Self {
        Self(PACKED_BITS.with(|cell| cell.replace(bits)))
    }
}

impl Drop for PackedScope {
    fn drop(&mut self) {
        PACKED_BITS.with(|cell| cell.set(self.0));
    }
}

fn packed_bits() -> Option<u8> {
    PACKED_BITS.with(Cell::get)
}

fn fixed_step(bits: u8) -> f32 {
    2.0f32.powi(-(i32::from(bits) - 2))
}

fn unit_levels(bits: u8) -> u32 {
    (1u32 << bits.min(16)) - 1
}

fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn unzigzag(value: u64) -> i64 {
    (value >> 1) as i64 ^ -((value & 1) as i64)
}

impl Writer {
    fn packed_fixed(&mut self, value: f32, bits: u8) {
        let step = fixed_step(bits);
        let index = (value / step).round();
        if index.abs() < (1u64 << 40) as f32 && index * step == value {
            self.varint(zigzag(index as i64) << 1);
        } else {
            let historical = (f64::from(value) * FIXED_SCALE).round() as i64;
            self.varint((zigzag(historical) << 1) | 1);
        }
    }

    fn packed_unit(&mut self, value: f32, bits: u8) {
        let code = (value.clamp(0.0, 1.0) * unit_levels(bits) as f32).round() as u16;
        if bits <= 8 {
            self.byte(code as u8);
        } else {
            self.u16(code);
        }
    }
}

impl Reader<'_> {
    fn packed_fixed(&mut self, bits: u8) -> Result<f32> {
        let raw = self.varint()?;
        let value = unzigzag(raw >> 1);
        Ok(if raw & 1 == 0 {
            value as f32 * fixed_step(bits)
        } else {
            (value as f64 / FIXED_SCALE) as f32
        })
    }

    fn packed_unit(&mut self, bits: u8) -> Result<f32> {
        let code = if bits <= 8 {
            u32::from(self.byte()?)
        } else {
            u32::from(self.u16()?)
        };
        if code > unit_levels(bits) {
            return Err(DopeError::Codec("packed unit code exceeds its grid".into()));
        }
        Ok(code as f32 / unit_levels(bits) as f32)
    }
}

/// Encode a symbolic kernel with packed values. The call is refused, so the
/// caller can keep the historical encoding, when the kernel is not symbolic or
/// when its decoded parameters would differ from the historical encoding's
/// decoded parameters by more than that codec's own resolution.
pub fn encode_kernel_packed(kernel: &Kernel) -> Result<Vec<u8>> {
    if !matches!(kernel.program, KernelProgram::Symbolic { .. })
        || !QUANTIZATION_GRID_BITS.contains(&kernel.quantization_bits)
    {
        return Err(DopeError::Codec(
            "packed values need a symbolic kernel on a DPK3 quantization grid".into(),
        ));
    }
    let packed = encode_kernel_with(kernel, true)?;
    let historical = encode_kernel_with(kernel, false)?;
    let left = serde_json::to_value(decode_kernel(&packed)?)?;
    let right = serde_json::to_value(decode_kernel(&historical)?)?;
    if !same_parameters(&left, &right) {
        return Err(DopeError::Codec(
            "kernel parameters are off their quantization grid; packed encoding refused".into(),
        ));
    }
    // Release and evaluation re-encode artifacts canonically; a packed artifact
    // that did not round-trip would fail there instead of here.
    if reencode_canonical(&packed)? != packed {
        return Err(DopeError::Codec("packed artifact is not canonical; packed encoding refused".into()));
    }
    Ok(packed)
}

/// Encode with packed values when asked and the kernel allows it, else with the
/// historical codec. The header flag (`is_packed`) records which one was used,
/// so callers that report the codec must read it from the artifact.
pub fn encode_kernel_with_codec(kernel: &Kernel, packed: bool) -> Result<Vec<u8>> {
    if packed {
        if let Ok(artifact) = encode_kernel_packed(kernel) {
            return Ok(artifact);
        }
    }
    encode_kernel(kernel)
}

/// Whether a DPK3 artifact uses packed values.
pub fn is_packed(artifact: &[u8]) -> bool {
    artifact.len() >= HEADER_BYTES
        && artifact.starts_with(MAGIC)
        && u16::from_le_bytes([artifact[10], artifact[11]]) & PACKED_FLAG != 0
}

/// Decode and re-encode an artifact in its own value codec. Canonical
/// artifacts, packed or historical, return their exact bytes.
pub fn reencode_canonical(artifact: &[u8]) -> Result<Vec<u8>> {
    let kernel = decode_kernel(artifact)?;
    if is_packed(artifact) {
        encode_kernel_with(&kernel, true)
    } else {
        encode_kernel(&kernel)
    }
}

fn same_parameters(left: &serde_json::Value, right: &serde_json::Value) -> bool {
    use serde_json::Value;
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => match (a.as_f64(), b.as_f64()) {
            (Some(a), Some(b)) => (a - b).abs() <= 2e-5 * b.abs().max(1.0),
            _ => a == b,
        },
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_parameters(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter().all(|(key, x)| b.get(key).is_some_and(|y| same_parameters(x, y)))
        }
        _ => left == right,
    }
}

#[cfg(test)]
mod packed_value_tests {
    use super::*;
    use crate::model::{ColumnSchema, SchemaKind, TriangularTerm};

    fn kernel(bits: u8) -> Kernel {
        let step = fixed_step(bits);
        let unit = |code: u32| code as f32 / unit_levels(bits) as f32;
        let schema = (0..3)
            .map(|_| ColumnSchema {
                kind: SchemaKind::Continuous,
                missing_probability: unit(1),
                impute: unit(unit_levels(bits) / 2),
                transform: Transform::Identity,
            })
            .collect();
        Kernel {
            task: Task::Regression,
            rows_fitted: 400,
            features: 3,
            seed: 11,
            seed_policy: 0,
            quantization_bits: bits,
            compliant: false,
            schema,
            marginals: (0..3)
                .map(|_| Marginal::QuantileSpline {
                    values: (0..5).map(|i| unit(i * unit_levels(bits) / 4)).collect(),
                })
                .collect(),
            program: KernelProgram::Symbolic {
                dependence: Dependence::Triangular {
                    terms: vec![
                        TriangularTerm { parent: 0, child: 1, coefficient: 3.0 * step },
                        TriangularTerm { parent: 1, child: 2, coefficient: 0.995 },
                    ],
                },
                target: Target::SparseLinear {
                    intercept: 3.0 * step,
                    terms: vec![LinearTerm { feature: 2, coefficient: -40.0 * step }],
                },
                noise: Noise::Homoscedastic { sigma: unit(9) },
            },
        }
    }

    #[test]
    fn packed_artifacts_round_trip_and_shrink() {
        for bits in [4u8, 6, 8, 10, 12, 16] {
            let source = kernel(bits);
            let packed = encode_kernel_packed(&source).unwrap();
            let historical = encode_kernel(&source).unwrap();
            assert!(is_packed(&packed) && !is_packed(&historical));
            assert!(packed.len() < historical.len(), "bits {bits}");
            let decoded = decode_kernel(&packed).unwrap();
            assert_eq!(decoded.schema, source.schema);
            assert_eq!(decoded.marginals, source.marginals);
            assert_eq!(reencode_canonical(&packed).unwrap(), packed);
            assert_eq!(reencode_canonical(&historical).unwrap(), historical);
            assert_eq!(packed_bits(), None);
        }
    }

    #[test]
    fn off_grid_unit_parameters_are_refused() {
        let mut source = kernel(8);
        source.schema[0].missing_probability = 0.0123;
        assert!(encode_kernel_packed(&source).is_err());
        assert!(encode_kernel(&source).is_ok());
    }

    #[test]
    fn historical_decoders_reject_the_packed_flag() {
        let packed = encode_kernel_packed(&kernel(8)).unwrap();
        let flags = u16::from_le_bytes([packed[10], packed[11]]);
        assert_eq!(flags & PACKED_FLAG, PACKED_FLAG);
        let mut corrupt = packed.clone();
        corrupt[10] |= 4;
        assert!(decode_kernel(&corrupt).is_err());
    }
}

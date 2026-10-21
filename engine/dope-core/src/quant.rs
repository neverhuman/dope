//! Uniform quantization and bit packing.
//!
//! Parameters are stored as b-bit integer codes (2 ≤ b ≤ 8) packed
//! little-endian into `U8` safetensors buffers, with one `f32` scale per
//! tensor. Bit depths are chosen per tensor by a measured rate–distortion
//! search (see `target::allocate_bits`).

/// Symmetric quantization: code ∈ [-(2^(b-1)-1), 2^(b-1)-1], value = code·scale.
#[derive(Clone, Debug, PartialEq)]
pub struct QTensor {
    pub codes: Vec<i32>,
    pub bits: u8,
    pub scale: f32,
}

impl QTensor {
    pub fn quantize(values: &[f64], bits: u8) -> Self {
        let levels = ((1i32 << (bits - 1)) - 1).max(1);
        let maxabs = values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let scale = if maxabs > 0.0 {
            (maxabs / levels as f64) as f32
        } else {
            1.0
        };
        let codes = values
            .iter()
            .map(|v| ((v / scale as f64).round() as i32).clamp(-levels, levels))
            .collect();
        Self { codes, bits, scale }
    }

    pub fn values(&self) -> Vec<f64> {
        self.codes
            .iter()
            .map(|&c| c as f64 * self.scale as f64)
            .collect()
    }

    pub fn packed(&self) -> Vec<u8> {
        let offset = 1i32 << (self.bits - 1);
        let unsigned: Vec<u32> = self.codes.iter().map(|&c| (c + offset) as u32).collect();
        pack(&unsigned, self.bits)
    }

    pub fn unpack(bytes: &[u8], n: usize, bits: u8, scale: f32) -> Self {
        let offset = 1i32 << (bits - 1);
        let codes = unpack(bytes, n, bits)
            .into_iter()
            .map(|u| u as i32 - offset)
            .collect();
        Self { codes, bits, scale }
    }
}

pub fn packed_len(n: usize, bits: u8) -> usize {
    (n * bits as usize).div_ceil(8)
}

pub fn pack(codes: &[u32], bits: u8) -> Vec<u8> {
    let mut out = vec![0u8; packed_len(codes.len(), bits)];
    let mut bit = 0usize;
    for &c in codes {
        for b in 0..bits as usize {
            if (c >> b) & 1 == 1 {
                out[(bit + b) / 8] |= 1 << ((bit + b) % 8);
            }
        }
        bit += bits as usize;
    }
    out
}

pub fn unpack(bytes: &[u8], n: usize, bits: u8) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    let mut bit = 0usize;
    for _ in 0..n {
        let mut c = 0u32;
        for b in 0..bits as usize {
            let at = bit + b;
            if at / 8 < bytes.len() && (bytes[at / 8] >> (at % 8)) & 1 == 1 {
                c |= 1 << b;
            }
        }
        out.push(c);
        bit += bits as usize;
    }
    out
}

/// Unsigned code of `v` in `[lo, hi]` with `bits` bits.
pub fn unit_code(v: f64, lo: f64, hi: f64, bits: u8) -> u32 {
    let top = ((1u32 << bits) - 1) as f64;
    if hi <= lo {
        return 0;
    }
    (((v - lo) / (hi - lo)).clamp(0.0, 1.0) * top).round() as u32
}

pub fn unit_value(code: u32, lo: f64, hi: f64, bits: u8) -> f64 {
    let top = ((1u32 << bits) - 1) as f64;
    lo + (hi - lo) * code as f64 / top
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip_all_widths() {
        for bits in 2..=8u8 {
            let codes: Vec<u32> = (0..37).map(|i| (i * 7) % (1 << bits)).collect();
            let packed = pack(&codes, bits);
            assert_eq!(packed.len(), packed_len(37, bits));
            assert_eq!(unpack(&packed, 37, bits), codes);
        }
    }

    #[test]
    fn symmetric_error_is_bounded() {
        let v: Vec<f64> = (0..50).map(|i| (i as f64 - 25.0) / 7.0).collect();
        for bits in 3..=8u8 {
            let q = QTensor::quantize(&v, bits);
            let back = QTensor::unpack(&q.packed(), v.len(), bits, q.scale).values();
            for (a, b) in v.iter().zip(back) {
                assert!((a - b).abs() <= q.scale as f64 * 0.5 + 1e-6);
            }
        }
    }
}

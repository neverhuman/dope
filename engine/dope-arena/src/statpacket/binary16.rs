//! IEEE 754 binary16 conversion for the atom values and target scales of the packet.

/// IEEE 754 binary16 bits to `f64` (exact).
pub fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f64;
    let mag = match exp {
        0 => frac * 2f64.powi(-24),
        31 if frac == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        e => (1.0 + frac / 1024.0) * 2f64.powi(e - 15),
    };
    sign * mag
}

/// Nearest IEEE 754 binary16 (ties to even), saturating at the largest finite value; NaN maps to +0.
pub fn f64_to_f16(x: f64) -> u16 {
    if x.is_nan() || x == 0.0 {
        return 0;
    }
    let sign = if x < 0.0 { 0x8000u16 } else { 0 };
    let a = x.abs();
    if a >= 65_520.0 {
        return sign | 0x7bff;
    }
    let mut e = (((a.to_bits() >> 52) & 0x7ff) as i32 - 1023).max(-15);
    if e < -14 {
        // Subnormal range: 10-bit fraction of 2^-14.
        let q = (a * 2f64.powi(24)).round_ties_even() as u16;
        return sign | q;
    }
    let mut q = ((a / 2f64.powi(e) - 1.0) * 1024.0).round_ties_even() as u32;
    if q == 1024 {
        q = 0;
        e += 1;
    }
    if e > 15 {
        return sign | 0x7bff;
    }
    sign | (((e + 15) as u16) << 10) | q as u16
}

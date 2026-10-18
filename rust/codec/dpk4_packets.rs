//! Lossless semantic model binding for the candidate packet format.
//!
//! The layout is canonical serde model JSON with floats replaced by tensor
//! placeholders. Integers (including seeds) retain their exact JSON values.
//! Already quantized unit floats use q4/q8; every other float retains its f32
//! bits. This avoids rounding signed parameters or collapsing ordered knots.
//! Layout and tensor framing are charged by `inspect`; no compactness or
//! certification claim follows from selecting this candidate format.

use serde_json::Value;

use super::{MAX_BYTES, MAX_VALUES, Packet, decode, encode, invalid};
use crate::error::Result;
use crate::model::Kernel;

const QUANTIZED: &str = "$DPK4.unit";
const FLOAT: &str = "$DPK4.f32";

fn visit(
    value: &mut Value,
    depth: usize,
    leaf: &mut impl FnMut(&mut Value) -> Result<()>,
) -> Result<()> {
    if depth > 64 {
        return Err(invalid("DPK4 model nesting exceeds its bound"));
    }
    match value {
        Value::Array(values) => {
            for value in values {
                visit(value, depth + 1, leaf)?;
            }
        }
        Value::Object(values) => {
            for value in values.values_mut() {
                visit(value, depth + 1, leaf)?;
            }
        }
        _ => leaf(value)?,
    }
    Ok(())
}

/// Preserve the supplied semantic model, including unaligned float parameters.
/// Fitting/rounding is explicit at the compiler boundary; this writer never
/// silently reduces precision. Only q4/q8 model profiles are advertised here.
pub fn encode_kernel(kernel: &Kernel) -> Result<Vec<u8>> {
    kernel
        .validate()
        .map_err(|_| invalid("DPK4 invalid model"))?;
    let maximum = match kernel.quantization_bits {
        4 => 15.0f32,
        8 => 255.0f32,
        _ => return Err(invalid("DPK4 unsupported model precision")),
    };
    let mut layout =
        serde_json::to_value(kernel).map_err(|_| invalid("DPK4 model serialization failed"))?;
    let mut units = Vec::new();
    let mut floats = Vec::new();
    visit(&mut layout, 0, &mut |value| {
        if value.as_number().is_some_and(|number| number.is_f64()) {
            let number = value
                .as_f64()
                .ok_or_else(|| invalid("DPK4 invalid model number"))?
                as f32;
            let level = (number * maximum).round();
            // Bit comparison preserves negative zero as well as ordinary floats.
            if (0.0..=1.0).contains(&number)
                && !number.is_sign_negative()
                && (level / maximum).to_bits() == number.to_bits()
            {
                units.push(level as u8);
                *value = Value::String(QUANTIZED.into());
            } else {
                floats.push(number);
                *value = Value::String(FLOAT.into());
            }
            if units.len() + floats.len() > MAX_VALUES {
                return Err(invalid("DPK4 numeric allocation exceeds its bound"));
            }
        }
        Ok(())
    })?;
    let layout =
        serde_json::to_vec(&layout).map_err(|_| invalid("DPK4 model serialization failed"))?;
    if layout.len() > MAX_BYTES {
        return Err(invalid("DPK4 model layout exceeds its bound"));
    }
    let units = if kernel.quantization_bits == 4 {
        Packet::Q4(units)
    } else {
        Packet::Q8(units)
    };
    encode(&[Packet::Model(layout), units, Packet::F32(floats)])
}

pub fn decode_kernel(artifact: &[u8]) -> Result<Kernel> {
    let packets = decode(artifact)?;
    let [Packet::Model(layout), units, Packet::F32(floats)] = packets.as_slice() else {
        return Err(invalid("DPK4 incomplete model packets"));
    };
    let (levels, maximum, bits) = match units {
        Packet::Q4(values) => (values, 15.0f32, 4),
        Packet::Q8(values) => (values, 255.0f32, 8),
        _ => return Err(invalid("DPK4 invalid model precision packet")),
    };
    let mut model: Value =
        serde_json::from_slice(layout).map_err(|_| invalid("DPK4 invalid model layout"))?;
    let mut levels = levels.iter();
    let mut floats = floats.iter();
    visit(&mut model, 0, &mut |value| {
        let number = match value.as_str() {
            Some(QUANTIZED) => {
                f32::from(*levels.next().ok_or_else(|| invalid("DPK4 missing unit"))?) / maximum
            }
            Some(FLOAT) => *floats.next().ok_or_else(|| invalid("DPK4 missing float"))?,
            _ => return Ok(()),
        };
        *value = Value::from(number);
        Ok(())
    })?;
    if levels.next().is_some() || floats.next().is_some() {
        return Err(invalid("DPK4 unused model parameters"));
    }
    let kernel: Kernel =
        serde_json::from_value(model).map_err(|_| invalid("DPK4 invalid semantic model"))?;
    kernel
        .validate()
        .map_err(|_| invalid("DPK4 invalid model"))?;
    if kernel.quantization_bits != bits || encode_kernel(&kernel)? != artifact {
        return Err(invalid("DPK4 non-canonical model encoding"));
    }
    Ok(kernel)
}

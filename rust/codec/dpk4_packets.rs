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

#[cfg(test)]
mod bitpacking_study {
    use super::*;
    use crate::codec;
    use crate::model::{
        ColumnSchema, Dependence, KernelProgram, LinearTerm, Marginal, Noise, SchemaKind, Target,
        Task, Transform,
    };

    fn study_model(features: usize, bits: u8, seed: u64, task: Task) -> Kernel {
        let schema = (0..features)
            .map(|_| ColumnSchema {
                kind: SchemaKind::Continuous,
                missing_probability: 0.0,
                impute: 0.5,
                transform: Transform::Identity,
            })
            .collect();
        let marginals = (0..features)
            .map(|_| Marginal::QuantileSpline {
                values: vec![0.0, 0.2, 0.5, 0.8, 1.0],
            })
            .collect();
        let terms = (0..features)
            .map(|index| LinearTerm {
                feature: index as u32,
                coefficient: ((index as f32 + 1.0) / features as f32)
                    * if (index as u64 + seed) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    },
            })
            .collect();
        let target = match task {
            Task::Regression => Target::SparseLinear {
                intercept: -0.375,
                terms,
            },
            Task::Binary => Target::SparseLogistic {
                intercept: 0.375,
                terms,
            },
        };
        Kernel {
            task,
            rows_fitted: 512,
            features: features as u32,
            seed,
            seed_policy: 0,
            quantization_bits: bits,
            compliant: false,
            schema,
            marginals,
            program: KernelProgram::Symbolic {
                dependence: Dependence::Independent,
                target,
                noise: match task {
                    Task::Regression => Noise::Homoscedastic { sigma: 0.125 },
                    Task::Binary => Noise::BernoulliCalibration { base_rate: 0.5 },
                },
            },
        }
    }

    #[test]
    fn preregistered_full_artifact_bitpacking_pilot() {
        let mut measurements = Vec::new();
        let mut win_count = 0usize;
        let mut count = 0usize;
        for task in [Task::Regression, Task::Binary] {
            for features in [4, 12, 24] {
                for bits in [4, 8] {
                    for seed in [17, 29, 43] {
                        let model = study_model(features, bits, seed, task);
                        model.validate().unwrap();
                        let baseline = codec::encode_kernel(&model).unwrap();
                        let candidate = encode_kernel(&model).unwrap();
                        let decoded = decode_kernel(&candidate).unwrap();
                        assert_eq!(encode_kernel(&decoded).unwrap(), candidate);
                        assert_eq!(decoded, model);
                        assert_eq!(
                            codec::encode_kernel(&codec::decode_kernel(&baseline).unwrap())
                                .unwrap(),
                            baseline
                        );
                        let savings = (baseline.len() as f64 - candidate.len() as f64)
                            / baseline.len() as f64;
                        win_count += usize::from(savings >= 0.20);
                        count += 1;
                        measurements.push((
                            task.as_str(),
                            features,
                            bits,
                            seed,
                            baseline.len(),
                            candidate.len(),
                        ));
                    }
                }
            }
        }
        eprintln!("R01 pilot rows: {measurements:?}");
        eprintln!("R01 pilot >=20% rows: {win_count}/{count}");
        assert_eq!(count, 36);
    }
}

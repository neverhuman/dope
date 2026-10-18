fn decode_target(data: &[u8], features: usize) -> Result<Target> {
    let mut reader = Reader::new(data);
    let tag = reader.byte()?;
    let intercept = reader.fixed()?;
    let target = match tag {
        0 | 1 => {
            let count = reader.usize(features.min(64))?;
            let mut terms = Vec::with_capacity(count);
            let mut prior = 0u32;
            for index in 0..count {
                let delta = reader.usize(features.saturating_sub(1))? as u32;
                let feature = if index == 0 {
                    delta
                } else {
                    prior
                        .checked_add(delta)
                        .ok_or_else(|| DopeError::Codec("target feature gap overflow".into()))?
                };
                terms.push(LinearTerm {
                    feature,
                    coefficient: reader.fixed()?,
                });
                prior = feature;
            }
            if tag == 0 {
                Target::SparseLinear { intercept, terms }
            } else {
                Target::SparseLogistic { intercept, terms }
            }
        }
        2 => {
            let logistic = reader.byte()? != 0;
            let count = reader.usize(64)?;
            Target::SparseGam {
                intercept,
                logistic,
                terms: (0..count)
                    .map(|_| decode_additive(&mut reader, features))
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        3 => {
            let logistic = reader.byte()? != 0;
            let main_count = reader.usize(64)?;
            let main_terms = (0..main_count)
                .map(|_| decode_additive(&mut reader, features))
                .collect::<Result<Vec<_>>>()?;
            let interaction_count = reader.usize(64)?;
            let mut interactions = Vec::with_capacity(interaction_count);
            for _ in 0..interaction_count {
                let left = reader.usize(features.saturating_sub(1))? as u32;
                let right = reader.usize(features.saturating_sub(1))? as u32;
                let left_length = reader.usize(9)?;
                let right_length = reader.usize(9)?;
                interactions.push(InteractionTerm {
                    left,
                    right,
                    left_knots: (0..left_length)
                        .map(|_| reader.unit())
                        .collect::<Result<Vec<_>>>()?,
                    right_knots: (0..right_length)
                        .map(|_| reader.unit())
                        .collect::<Result<Vec<_>>>()?,
                    values: (0..left_length * right_length)
                        .map(|_| reader.fixed())
                        .collect::<Result<Vec<_>>>()?,
                });
            }
            Target::Ga2m {
                intercept,
                logistic,
                main_terms,
                interactions,
            }
        }
        4 => {
            let logistic = reader.byte()? != 0;
            let count = reader.usize(128)?;
            let mut terms = Vec::with_capacity(count);
            for _ in 0..count {
                let coefficient = reader.fixed()?;
                let factor_count = reader.usize(3)?;
                let factors = (0..factor_count)
                    .map(|_| {
                        Ok(MarsFactor {
                            feature: reader.usize(features.saturating_sub(1))? as u32,
                            knot: reader.unit()?,
                            direction: if reader.byte()? == 0 { -1 } else { 1 },
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                terms.push(MarsTerm {
                    coefficient,
                    factors,
                });
            }
            Target::Mars {
                intercept,
                logistic,
                terms,
            }
        }
        5 => {
            let logistic = reader.byte()? != 0;
            let depth = reader.usize(8)?;
            let mut features_out = Vec::with_capacity(depth);
            let mut thresholds = Vec::with_capacity(depth);
            for _ in 0..depth {
                features_out.push(reader.usize(features.saturating_sub(1))? as u32);
                thresholds.push(reader.unit()?);
            }
            Target::ObliviousTree {
                logistic,
                features: features_out,
                thresholds,
                leaves: (0..1usize << depth)
                    .map(|_| reader.fixed())
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        6 => {
            let logistic = reader.byte()? != 0;
            let linear_count = reader.usize(features.min(64))?;
            let linear_terms = (0..linear_count)
                .map(|_| {
                    Ok(LinearTerm {
                        feature: reader.usize(features.saturating_sub(1))? as u32,
                        coefficient: reader.fixed()?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let input_count = reader.usize(features.min(24))?;
            let hidden_width = reader.byte()?;
            if !(1..=16).contains(&hidden_width) {
                return Err(DopeError::Codec(
                    "compact neural hidden width is outside 1..=16".into(),
                ));
            }
            let hidden_features = (0..input_count)
                .map(|_| {
                    reader
                        .usize(features.saturating_sub(1))
                        .map(|value| value as u32)
                })
                .collect::<Result<Vec<_>>>()?;
            let parameter_count = input_count * usize::from(hidden_width);
            Target::CompactNeuralResidual {
                intercept,
                logistic,
                linear_terms,
                hidden_features,
                hidden_width,
                input_weights: (0..parameter_count)
                    .map(|_| reader.fixed())
                    .collect::<Result<Vec<_>>>()?,
                hidden_biases: (0..hidden_width)
                    .map(|_| reader.fixed())
                    .collect::<Result<Vec<_>>>()?,
                output_weights: (0..hidden_width)
                    .map(|_| reader.fixed())
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        _ => return Err(DopeError::Codec(format!("unknown target opcode {tag}"))),
    };
    reader.finished()?;
    Ok(target)
}

fn decode_noise(data: &[u8]) -> Result<Noise> {
    let mut reader = Reader::new(data);
    let noise = match reader.byte()? {
        0 => Noise::Homoscedastic {
            sigma: reader.unit()?,
        },
        1 => Noise::BernoulliCalibration {
            base_rate: reader.unit()?,
        },
        2 => {
            let intercept = reader.fixed()?;
            let minimum_sigma = reader.unit()?;
            let maximum_sigma = reader.unit()?;
            let count = reader.usize(64)?;
            Noise::Heteroscedastic {
                intercept,
                minimum_sigma,
                maximum_sigma,
                terms: (0..count)
                    .map(|_| {
                        Ok(LinearTerm {
                            feature: reader.varint()? as u32,
                            coefficient: reader.fixed()?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        3 => {
            let count = reader.usize(33)?;
            Noise::IsotonicCalibration {
                knots: (0..count)
                    .map(|_| reader.unit())
                    .collect::<Result<Vec<_>>>()?,
                probabilities: (0..count)
                    .map(|_| reader.unit())
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        4 => {
            if reader.byte()? != 1 {
                return Err(DopeError::Codec("unsupported residual version".into()));
            }
            let distribution = match reader.byte()? {
                0 => ResidualV1::NoNoise,
                1 => ResidualV1::StudentT4 {
                    location: reader.fixed()?,
                    scale: reader.fixed()?,
                },
                2 => ResidualV1::GaussianMixture2 {
                    weight: reader.unit()?,
                    locations: [reader.fixed()?, reader.fixed()?],
                    scales: [reader.fixed()?, reader.fixed()?],
                },
                3 => ResidualV1::MonotoneQuantile17 {
                    values: (0..17)
                        .map(|_| reader.fixed())
                        .collect::<Result<Vec<_>>>()?,
                },
                _ => return Err(DopeError::Codec("unsupported residual profile".into())),
            };
            Noise::ResidualV1 { distribution }
        }
        tag => return Err(DopeError::Codec(format!("unknown noise opcode {tag}"))),
    };
    reader.finished()?;
    Ok(noise)
}
/// A development-frozen order of complete, independently decodable DPK artifacts.
/// The envelope stores no source rows and every boundary ends after a packet digest.
const PREFIX_MAGIC: &[u8; 4] = b"DPP1";
const PREFIX_HEADER_BYTES: usize = 8;
const PREFIX_PACKET_HEADER_BYTES: usize = 40;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PacketBoundary {
    pub packet_count: usize,
    pub end_offset: usize,
    pub charged_bytes: usize,
}

pub fn encode_progressive_packets(packets: &[Vec<u8>]) -> Result<(Vec<u8>, Vec<PacketBoundary>)> {
    if packets.is_empty() || packets.len() > 1024 {
        return Err(DopeError::Codec("invalid progressive packet count".into()));
    }
    let mut output = Vec::new();
    output.extend_from_slice(PREFIX_MAGIC);
    output.extend_from_slice(&(packets.len() as u32).to_le_bytes());
    let mut boundaries = Vec::with_capacity(packets.len());
    for packet in packets {
        // Reject invalid packets before they become advertised boundaries.
        decode_kernel(packet)?;
        let length = u64::try_from(packet.len())
            .map_err(|_| DopeError::Codec("progressive packet is too large".into()))?;
        output.extend_from_slice(&length.to_le_bytes());
        output.extend_from_slice(blake3::hash(packet).as_bytes());
        output.extend_from_slice(packet);
        boundaries.push(PacketBoundary {
            packet_count: boundaries.len() + 1,
            end_offset: output.len(),
            charged_bytes: output.len(),
        });
    }
    Ok((output, boundaries))
}

/// Decode only a complete advertised packet boundary. In-flight partial packets fail closed.
pub fn decode_progressive_prefix(data: &[u8], packet_count: usize) -> Result<Vec<Kernel>> {
    if data.len() < PREFIX_HEADER_BYTES || &data[..4] != PREFIX_MAGIC {
        return Err(DopeError::Codec("invalid progressive packet envelope".into()));
    }
    let count = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    if count == 0 || count > 1024 || packet_count == 0 || packet_count > count {
        return Err(DopeError::Codec("unadvertised progressive packet boundary".into()));
    }
    let mut cursor = PREFIX_HEADER_BYTES;
    let mut decoded = Vec::with_capacity(packet_count);
    for _ in 0..packet_count {
        let header_end = cursor.checked_add(PREFIX_PACKET_HEADER_BYTES)
            .ok_or_else(|| DopeError::Codec("progressive packet length overflow".into()))?;
        let header = data.get(cursor..header_end)
            .ok_or_else(|| DopeError::Codec("truncated progressive packet header".into()))?;
        let length = usize::try_from(u64::from_le_bytes(header[..8].try_into().unwrap()))
            .map_err(|_| DopeError::Codec("progressive packet length overflow".into()))?;
        if length > MAX_ARTIFACT_BYTES {
            return Err(DopeError::Codec("progressive packet exceeds maximum length".into()));
        }
        cursor = header_end;
        let end = cursor.checked_add(length)
            .ok_or_else(|| DopeError::Codec("progressive packet length overflow".into()))?;
        let packet = data.get(cursor..end)
            .ok_or_else(|| DopeError::Codec("truncated progressive packet".into()))?;
        if blake3::hash(packet).as_bytes() != &header[8..40] {
            return Err(DopeError::Codec("progressive packet checksum mismatch".into()));
        }
        decoded.push(decode_kernel(packet)?);
        cursor = end;
    }
    if cursor != data.len() {
        return Err(DopeError::Codec("prefix is not an exact packet boundary".into()));
    }
    Ok(decoded)
}

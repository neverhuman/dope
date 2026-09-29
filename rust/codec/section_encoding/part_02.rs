

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
        tag => return Err(DopeError::Codec(format!("unknown noise opcode {tag}"))),
    };
    reader.finished()?;
    Ok(noise)
}


fn append_section(payload: &mut Writer, tag: u8, raw: &[u8]) -> Result<()> {
    let compressed = if raw.len() >= 64 {
        let candidate = rans_encode(raw);
        (candidate.len() + 4 < raw.len()).then_some(candidate)
    } else {
        None
    };
    if let Some(compressed) = compressed {
        payload.byte(tag | 0x80);
        payload.varint(compressed.len() as u64);
        payload.varint(raw.len() as u64);
        payload.0.extend_from_slice(&compressed);
    } else {
        payload.byte(tag);
        payload.varint(raw.len() as u64);
        payload.0.extend_from_slice(raw);
    }
    Ok(())
}

pub fn encode_kernel(kernel: &Kernel) -> Result<Vec<u8>> {
    kernel.validate().map_err(DopeError::Codec)?;
    let mut payload = Writer::default();
    append_section(&mut payload, 1, &encode_schema(kernel))?;
    append_section(&mut payload, 2, &encode_marginals(kernel))?;
    let limit = match &kernel.program {
        KernelProgram::Symbolic {
            dependence,
            target,
            noise,
        } => {
            append_section(&mut payload, 3, &encode_dependence(dependence))?;
            append_section(&mut payload, 4, &encode_target(target))?;
            append_section(&mut payload, 5, &encode_noise(noise))?;
            MAX_ARTIFACT_BYTES
        }
        KernelProgram::NeuralJoint(generator) => {
            let neural = serde_json::to_vec(generator)?;
            append_section(&mut payload, 6, &neural)?;
            MAX_NEURAL_ARTIFACT_BYTES
        }
    };
    if HEADER_BYTES + payload.0.len() > limit {
        return Err(DopeError::Codec(if limit == MAX_NEURAL_ARTIFACT_BYTES {
            "neural pilot artifact exceeds the 16 MiB cap".into()
        } else {
            "artifact exceeds the 2 GiB cap".into()
        }));
    }
    let checksum = blake3::hash(&payload.0);
    let mut artifact = Vec::with_capacity(HEADER_BYTES + payload.0.len());
    artifact.extend_from_slice(MAGIC);
    artifact.extend_from_slice(&[
        VERSION_MAJOR,
        match &kernel.program {
            KernelProgram::NeuralJoint(generator)
                if !generator.profile.is_full()
                    || generator.architecture == crate::model::NeuralArchitecture::TabDdpm =>
            {
                VERSION_MINOR
            }
            _ => 2,
        },
        DECODER_ID,
        kernel.task.code(),
        kernel.quantization_bits,
        kernel.seed_policy,
    ]);
    artifact.extend_from_slice(&(kernel.compliant as u16).to_le_bytes());
    artifact.extend_from_slice(&kernel.features.to_le_bytes());
    artifact.extend_from_slice(&kernel.rows_fitted.to_le_bytes());
    artifact.extend_from_slice(&kernel.seed.to_le_bytes());
    artifact.extend_from_slice(&(payload.0.len() as u64).to_le_bytes());
    artifact.extend_from_slice(&checksum.as_bytes()[..16]);
    debug_assert_eq!(artifact.len(), HEADER_BYTES);
    artifact.extend_from_slice(&payload.0);
    Ok(artifact)
}

fn decode_section(
    tag: u8,
    encoded: &[u8],
    raw_length: Option<usize>,
    legacy_zstd: bool,
) -> Result<Vec<u8>> {
    if tag & 0x80 == 0 {
        return Ok(encoded.to_vec());
    }
    let raw_length = raw_length
        .ok_or_else(|| DopeError::Codec("compressed section has no decoded length".into()))?;
    if legacy_zstd {
        zstd::bulk::decompress(encoded, raw_length)
            .map_err(|error| DopeError::Codec(format!("zstd decode failure: {error}")))
    } else {
        rans_decode(encoded, raw_length)
    }
}

fn decode_schema(data: &[u8], features: usize, has_transforms: bool) -> Result<Vec<ColumnSchema>> {
    let mut reader = Reader::new(data);
    let count = reader.usize(features)?;
    if count != features {
        return Err(DopeError::Codec("schema width differs from header".into()));
    }
    let mut schema = Vec::with_capacity(count);
    for _ in 0..count {
        let kind = match reader.byte()? {
            0 => SchemaKind::Constant,
            1 => SchemaKind::Binary,
            2 => SchemaKind::Ordinal,
            3 => SchemaKind::CategoricalGrid,
            4 => SchemaKind::CountLike,
            5 => SchemaKind::Continuous,
            6 => SchemaKind::Inflated,
            tag => return Err(DopeError::Codec(format!("unknown schema opcode {tag}"))),
        };
        let missing_probability = reader.unit()?;
        let impute = reader.unit()?;
        let transform = if !has_transforms {
            Transform::Identity
        } else {
            match reader.byte()? {
                0 => Transform::Identity,
                1 => Transform::Log1p {
                    scale: reader.fixed()?,
                },
                2 => Transform::Power {
                    exponent: reader.fixed()?,
                },
                3 => Transform::Logit {
                    epsilon: reader.unit()?,
                },
                4 => Transform::Winsorize {
                    lower: reader.unit()?,
                    upper: reader.unit()?,
                },
                tag => return Err(DopeError::Codec(format!("unknown transform opcode {tag}"))),
            }
        };
        schema.push(ColumnSchema {
            kind,
            missing_probability,
            impute,
            transform,
        });
    }
    reader.finished()?;
    Ok(schema)
}

fn decode_marginal(reader: &mut Reader<'_>, depth: usize) -> Result<Marginal> {
    if depth > 2 {
        return Err(DopeError::Codec(
            "marginal nesting exceeds its bound".into(),
        ));
    }
    Ok(match reader.byte()? {
        0 => Marginal::Constant {
            value: reader.unit()?,
        },
        1 => Marginal::Bernoulli {
            probability: reader.unit()?,
        },
        2 => {
            let length = reader.usize(65_536)?;
            if length == 0 {
                return Err(DopeError::Codec("empty categorical grid".into()));
            }
            let values = (0..length)
                .map(|_| reader.unit())
                .collect::<Result<Vec<_>>>()?;
            let probabilities = (0..length)
                .map(|_| reader.unit())
                .collect::<Result<Vec<_>>>()?;
            Marginal::Grid {
                values,
                probabilities,
            }
        }
        3 => {
            let length = reader.usize(65)?;
            if !matches!(length, 5 | 9 | 17 | 33 | 65) {
                return Err(DopeError::Codec(
                    "quantile spline violates the knot ladder".into(),
                ));
            }
            Marginal::QuantileSpline {
                values: (0..length)
                    .map(|_| reader.unit())
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        4 => Marginal::Beta {
            alpha: reader.fixed()?,
            beta: reader.fixed()?,
        },
        5 => Marginal::Gaussian {
            mean: reader.unit()?,
            sigma: reader.fixed()?,
        },
        6 => {
            let length = reader.usize(257)?;
            if length < 2 {
                return Err(DopeError::Codec("histogram requires two edges".into()));
            }
            Marginal::Histogram {
                edges: (0..length)
                    .map(|_| reader.unit())
                    .collect::<Result<Vec<_>>>()?,
                probabilities: (0..length - 1)
                    .map(|_| reader.unit())
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        7 => Marginal::ZeroInflated {
            point: reader.unit()?,
            probability: reader.unit()?,
            base: Box::new(decode_marginal(reader, depth + 1)?),
        },
        tag => return Err(DopeError::Codec(format!("unknown marginal opcode {tag}"))),
    })
}

fn decode_marginals(data: &[u8], features: usize) -> Result<Vec<Marginal>> {
    let mut reader = Reader::new(data);
    let count = reader.usize(features)?;
    if count != features {
        return Err(DopeError::Codec(
            "marginal width differs from header".into(),
        ));
    }
    let mut marginals = Vec::with_capacity(count);
    for _ in 0..count {
        marginals.push(decode_marginal(&mut reader, 0)?);
    }
    reader.finished()?;
    Ok(marginals)
}

fn decode_edges(reader: &mut Reader<'_>, features: usize, bound: usize) -> Result<Vec<CopulaEdge>> {
    let count = reader.usize(bound)?;
    (0..count)
        .map(|_| {
            Ok(CopulaEdge {
                parent: reader.usize(features.saturating_sub(1))? as u32,
                child: reader.usize(features.saturating_sub(1))? as u32,
                correlation: reader.u16()? as f32 / u16::MAX as f32 * 2.0 - 1.0,
            })
        })
        .collect()
}

fn decode_dependence_value(
    reader: &mut Reader<'_>,
    features: usize,
    depth: usize,
) -> Result<Dependence> {
    if depth > 2 {
        return Err(DopeError::Codec(
            "dependence mixture nesting exceeds its bound".into(),
        ));
    }
    Ok(match reader.byte()? {
        0 => Dependence::Independent,
        1 => {
            let root = reader.usize(features.saturating_sub(1))? as u32;
            let edges = decode_edges(reader, features, features.saturating_sub(1))?;
            Dependence::ChowLiu { root, edges }
        }
        2 => {
            let length = reader.usize(features.saturating_mul(features))?;
            Dependence::GaussianCopula {
                cholesky: (0..length)
                    .map(|_| reader.fixed())
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        3 => Dependence::SparseGraph {
            edges: decode_edges(reader, features, features.saturating_mul(8))?,
        },
        4 => {
            let rank = reader.byte()?;
            let count = reader.usize(features.saturating_mul(16))?;
            let loadings = (0..count)
                .map(|_| {
                    Ok(FactorLoading {
                        feature: reader.usize(features.saturating_sub(1))? as u32,
                        factor: reader.byte()?,
                        loading: reader.fixed()?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Dependence::Poet {
                rank,
                loadings,
                residual_edges: decode_edges(reader, features, features.saturating_mul(4))?,
            }
        }
        5 => {
            let count = reader.usize(features.saturating_mul(8))?;
            Dependence::Vine {
                edges: (0..count)
                    .map(|_| {
                        Ok(VineEdge {
                            left: reader.usize(features.saturating_sub(1))? as u32,
                            right: reader.usize(features.saturating_sub(1))? as u32,
                            conditioning_depth: reader.byte()?,
                            parameter: reader.fixed()?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        6 => {
            let count = reader.usize(8)?;
            if count == 0 {
                return Err(DopeError::Codec("empty dependence mixture".into()));
            }
            let weights = (0..count)
                .map(|_| reader.unit())
                .collect::<Result<Vec<_>>>()?;
            let components = (0..count)
                .map(|_| decode_dependence_value(reader, features, depth + 1))
                .collect::<Result<Vec<_>>>()?;
            Dependence::Mixture {
                weights,
                components,
            }
        }
        7 => {
            let count = reader.usize(features.saturating_mul(8))?;
            Dependence::Triangular {
                terms: (0..count)
                    .map(|_| {
                        Ok(TriangularTerm {
                            parent: reader.usize(features.saturating_sub(1))? as u32,
                            child: reader.usize(features.saturating_sub(1))? as u32,
                            coefficient: reader.fixed()?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            }
        }
        tag => return Err(DopeError::Codec(format!("unknown dependence opcode {tag}"))),
    })
}

fn decode_dependence(data: &[u8], features: usize) -> Result<Dependence> {
    let mut reader = Reader::new(data);
    let dependence = decode_dependence_value(&mut reader, features, 0)?;
    reader.finished()?;
    Ok(dependence)
}

fn decode_additive(reader: &mut Reader<'_>, features: usize) -> Result<AdditiveTerm> {
    let feature = reader.usize(features.saturating_sub(1))? as u32;
    let length = reader.usize(17)?;
    Ok(AdditiveTerm {
        feature,
        knots: (0..length)
            .map(|_| reader.unit())
            .collect::<Result<Vec<_>>>()?,
        values: (0..length)
            .map(|_| reader.fixed())
            .collect::<Result<Vec<_>>>()?,
    })
}
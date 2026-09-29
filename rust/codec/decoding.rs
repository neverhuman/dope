
pub fn decode_kernel(artifact: &[u8]) -> Result<Kernel> {
    if artifact.len() < HEADER_BYTES || artifact.len() > MAX_ARTIFACT_BYTES {
        return Err(DopeError::Codec("artifact violates the size bounds".into()));
    }
    let v3 = &artifact[..4] == MAGIC;
    let v2 = &artifact[..4] == V2_MAGIC;
    if !v3 && !v2 {
        return Err(DopeError::Codec("not a V2 or V3 .dpk artifact".into()));
    }
    let supported_version = (v3 && artifact[4] == VERSION_MAJOR && artifact[5] <= VERSION_MINOR)
        || (v2 && artifact[4] == V2_VERSION_MAJOR && artifact[5] <= V2_VERSION_MINOR);
    if !supported_version || artifact[6] != DECODER_ID {
        return Err(DopeError::Codec(
            "unsupported bytecode or decoder version".into(),
        ));
    }
    let task = match artifact[7] {
        0 => Task::Regression,
        1 => Task::Binary,
        _ => return Err(DopeError::Codec("unknown task code".into())),
    };
    let quantization_bits = artifact[8];
    let seed_policy = artifact[9];
    let flags = u16::from_le_bytes(artifact[10..12].try_into().unwrap());
    if flags & !1 != 0 {
        return Err(DopeError::Codec("unknown header flags".into()));
    }
    let features = u32::from_le_bytes(artifact[12..16].try_into().unwrap());
    let rows_fitted = u64::from_le_bytes(artifact[16..24].try_into().unwrap());
    let seed = u64::from_le_bytes(artifact[24..32].try_into().unwrap());
    let payload_length = u64::from_le_bytes(artifact[32..40].try_into().unwrap()) as usize;
    if payload_length != artifact.len() - HEADER_BYTES {
        return Err(DopeError::Codec("payload length mismatch".into()));
    }
    let payload = &artifact[HEADER_BYTES..];
    if blake3::hash(payload).as_bytes()[..16] != artifact[40..56] {
        return Err(DopeError::Codec("payload checksum mismatch".into()));
    }

    let mut payload_reader = Reader::new(payload);
    let mut sections: [Option<Vec<u8>>; 6] = Default::default();
    let mut prior = 0u8;
    while payload_reader.remaining() > 0 {
        let encoded_tag = payload_reader.byte()?;
        let tag = encoded_tag & 0x7f;
        let maximum_tag = if v3 && artifact[5] >= 2 { 6 } else { 5 };
        if !(1..=maximum_tag).contains(&tag) || tag <= prior {
            return Err(DopeError::Codec(
                "sections are unknown, duplicated, or non-canonical".into(),
            ));
        }
        if tag == 6 && artifact.len() > MAX_NEURAL_ARTIFACT_BYTES {
            return Err(DopeError::Codec(
                "neural pilot artifact exceeds the 16 MiB cap".into(),
            ));
        }
        let encoded_length = payload_reader.usize(payload_reader.remaining())?;
        let raw_length = if encoded_tag & 0x80 != 0 {
            Some(payload_reader.usize(if tag == 6 {
                MAX_NEURAL_ARTIFACT_BYTES
            } else {
                MAX_ARTIFACT_BYTES
            })?)
        } else {
            None
        };
        let encoded = payload_reader.take(encoded_length)?;
        sections[tag as usize - 1] = Some(decode_section(
            encoded_tag,
            encoded,
            raw_length,
            v2 && artifact[5] == 0,
        )?);
        prior = tag;
    }
    let [schema, marginals, dependence, target, noise, neural] = sections;
    let schema = schema.ok_or_else(|| DopeError::Codec("missing schema section".into()))?;
    let marginals = marginals.ok_or_else(|| DopeError::Codec("missing marginal section".into()))?;
    let symbolic_count = [&dependence, &target, &noise]
        .into_iter()
        .filter(|section| section.is_some())
        .count();
    let program = match (symbolic_count, neural) {
        (3, None) => KernelProgram::Symbolic {
            dependence: decode_dependence(&dependence.unwrap(), features as usize)?,
            target: decode_target(&target.unwrap(), features as usize)?,
            noise: decode_noise(&noise.unwrap())?,
        },
        (0, Some(encoded)) if v3 && artifact[5] >= 2 => {
            let generator: crate::model::JointGenerator = serde_json::from_slice(&encoded)
                .map_err(|error| DopeError::Codec(format!("invalid neural section: {error}")))?;
            if serde_json::to_vec(&generator)? != encoded {
                return Err(DopeError::Codec(
                    "neural section is non-canonical or has unknown fields".into(),
                ));
            }
            let historical_shape = generator.profile.is_full()
                && generator.architecture != crate::model::NeuralArchitecture::TabDdpm;
            if (artifact[5] == 2) != historical_shape {
                return Err(DopeError::Codec(
                    "neural profile requires its canonical DPK3 minor version".into(),
                ));
            }
            KernelProgram::NeuralJoint(generator)
        }
        (0, None) => return Err(DopeError::Codec("missing kernel program section".into())),
        _ => {
            return Err(DopeError::Codec(
                "ambiguous or incomplete symbolic/neural payload".into(),
            ));
        }
    };
    let width = features as usize;
    let kernel = Kernel {
        task,
        rows_fitted,
        features,
        seed,
        seed_policy,
        quantization_bits,
        compliant: flags & 1 != 0,
        schema: decode_schema(&schema, width, v3 || artifact[5] >= 1)?,
        marginals: decode_marginals(&marginals, width)?,
        program,
    };
    if matches!(kernel.program, KernelProgram::NeuralJoint(_))
        && artifact.len() > MAX_NEURAL_ARTIFACT_BYTES
    {
        return Err(DopeError::Codec(
            "neural pilot artifact exceeds the 16 MiB cap".into(),
        ));
    }
    kernel.validate().map_err(DopeError::Codec)?;
    Ok(kernel)
}

pub fn load_kernel(path: &Path) -> Result<LoadedKernel> {
    let bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    if bytes.starts_with(MAGIC) {
        Ok(LoadedKernel::V3(Box::new(decode_kernel(&bytes)?)))
    } else if bytes.starts_with(V2_MAGIC) {
        Ok(LoadedKernel::V2(Box::new(decode_kernel(&bytes)?)))
    } else {
        let value: Value = serde_json::from_slice(&bytes)?;
        if value.get("format").and_then(Value::as_str) != Some("dope-kernel")
            || value.get("version").and_then(Value::as_u64) != Some(1)
        {
            return Err(DopeError::Codec(
                "not a supported V1 JSON, V2 .dpk, or V3 .dpk artifact".into(),
            ));
        }
        Ok(LoadedKernel::V1(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn kernel() -> Kernel {
        Kernel {
            task: Task::Regression,
            rows_fitted: 100,
            features: 2,
            seed: 9,
            seed_policy: 0,
            quantization_bits: 8,
            compliant: false,
            schema: vec![
                ColumnSchema {
                    kind: SchemaKind::Continuous,
                    missing_probability: 0.0,
                    impute: 0.5,
                    transform: Transform::Identity,
                },
                ColumnSchema {
                    kind: SchemaKind::Binary,
                    missing_probability: 0.1,
                    impute: 0.0,
                    transform: Transform::Identity,
                },
            ],
            marginals: vec![
                Marginal::QuantileSpline {
                    values: vec![0.0, 0.2, 0.5, 0.8, 1.0],
                },
                Marginal::Bernoulli { probability: 0.25 },
            ],
            program: KernelProgram::Symbolic {
                dependence: Dependence::ChowLiu {
                    root: 0,
                    edges: vec![CopulaEdge {
                        parent: 0,
                        child: 1,
                        correlation: 0.4,
                    }],
                },
                target: Target::SparseLinear {
                    intercept: 0.1,
                    terms: vec![LinearTerm {
                        feature: 0,
                        coefficient: 0.7,
                    }],
                },
                noise: Noise::Homoscedastic { sigma: 0.05 },
            },
        }
    }

    #[test]
    fn canonical_round_trip_and_corruption_detection() {
        let encoded = encode_kernel(&kernel()).unwrap();
        let decoded = decode_kernel(&encoded).unwrap();
        assert_eq!(encode_kernel(&decoded).unwrap(), encoded);
        let mut corrupt = encoded;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(
            decode_kernel(&corrupt)
                .unwrap_err()
                .to_string()
                .contains("checksum")
        );
        let mut unsupported = encode_kernel(&kernel()).unwrap();
        unsupported[4] = 4;
        assert!(
            decode_kernel(&unsupported)
                .unwrap_err()
                .to_string()
                .contains("unsupported")
        );
    }

    #[test]
    fn section_accounting_reconciles_exact_encoded_length() {
        let encoded = encode_kernel(&kernel()).unwrap();
        let accounting = account_artifact(&encoded).unwrap();
        assert_eq!(accounting.header_bytes, HEADER_BYTES);
        assert_eq!(accounting.total_bytes, encoded.len());
        assert_eq!(
            accounting
                .sections
                .iter()
                .map(|section| section.framing_bytes + section.encoded_bytes)
                .sum::<usize>()
                + HEADER_BYTES,
            encoded.len()
        );
        assert!(
            accounting
                .sections
                .iter()
                .all(|section| section.raw_bytes >= section.encoded_bytes || !section.compressed)
        );
    }

    proptest! {
        #[test]
        fn encoded_section_totals_and_roundtrip_hold_for_valid_parameters(
            seed in any::<u64>(),
            rows in 1u64..100_000,
            probability in 0u16..=u16::MAX,
        ) {
            let mut candidate = kernel();
            candidate.seed = seed;
            candidate.rows_fitted = rows;
            candidate.marginals[1] = Marginal::Bernoulli {
                probability: f32::from(probability) / f32::from(u16::MAX),
            };
            let encoded = encode_kernel(&candidate).unwrap();
            let accounting = account_artifact(&encoded).unwrap();
            prop_assert_eq!(accounting.total_bytes, encoded.len());
            prop_assert_eq!(
                accounting.header_bytes + accounting.sections.iter()
                    .map(|section| section.framing_bytes + section.encoded_bytes)
                    .sum::<usize>(),
                encoded.len()
            );
            let decoded = decode_kernel(&encoded).unwrap();
            prop_assert_eq!(encode_kernel(&decoded).unwrap(), encoded);
        }

        #[test]
        fn truncated_or_arbitrary_short_bytes_never_decode_as_a_kernel(
            bytes in proptest::collection::vec(any::<u8>(), 0..HEADER_BYTES),
        ) {
            prop_assert!(decode_kernel(&bytes).is_err());
            prop_assert!(account_artifact(&bytes).is_err());
        }
    }

    #[test]
    fn reads_v2_1_payloads_and_emits_v3() {
        let encoded_v3 = encode_kernel(&kernel()).unwrap();
        let expected = decode_kernel(&encoded_v3).unwrap();
        let mut encoded_v2 = encoded_v3;
        encoded_v2[..4].copy_from_slice(V2_MAGIC);
        encoded_v2[4] = V2_VERSION_MAJOR;
        encoded_v2[5] = V2_VERSION_MINOR;
        assert_eq!(decode_kernel(&encoded_v2).unwrap(), expected);
        assert!(encode_kernel(&expected).unwrap().starts_with(MAGIC));
    }

    #[test]
    fn reads_v3_0_and_v3_1_symbolic_payloads() {
        let encoded = encode_kernel(&kernel()).unwrap();
        let expected = decode_kernel(&encoded).unwrap();
        for minor in [0, 1] {
            let mut legacy = encoded.clone();
            legacy[5] = minor;
            assert_eq!(decode_kernel(&legacy).unwrap(), expected);
        }
    }

    fn assert_canonical(mut value: Kernel) {
        value.compliant = false;
        let encoded = encode_kernel(&value).unwrap();
        let decoded = decode_kernel(&encoded).unwrap();
        assert_eq!(encode_kernel(&decoded).unwrap(), encoded);
    }

    #[test]
    fn every_bounded_v3_operator_has_a_canonical_codec() {
        for transform in [
            Transform::Identity,
            Transform::Log1p { scale: 3.0 },
            Transform::Power { exponent: 2.0 },
            Transform::Logit { epsilon: 0.01 },
            Transform::Winsorize {
                lower: 0.1,
                upper: 0.9,
            },
        ] {
            let mut value = kernel();
            value.schema[0].transform = transform;
            assert_canonical(value);
        }

        for marginal in [
            Marginal::Beta {
                alpha: 2.0,
                beta: 3.0,
            },
            Marginal::Gaussian {
                mean: 0.5,
                sigma: 0.1,
            },
            Marginal::Histogram {
                edges: vec![0.0, 0.25, 1.0],
                probabilities: vec![0.3, 0.7],
            },
            Marginal::ZeroInflated {
                point: 0.0,
                probability: 0.2,
                base: Box::new(Marginal::Beta {
                    alpha: 2.0,
                    beta: 2.0,
                }),
            },
        ] {
            let mut value = kernel();
            value.marginals[0] = marginal;
            assert_canonical(value);
        }

        for dependence in [
            Dependence::GaussianCopula {
                cholesky: vec![1.0, 0.0, 0.3, 0.95],
            },
            Dependence::SparseGraph {
                edges: vec![CopulaEdge {
                    parent: 0,
                    child: 1,
                    correlation: 0.3,
                }],
            },
            Dependence::Poet {
                rank: 1,
                loadings: vec![FactorLoading {
                    feature: 1,
                    factor: 0,
                    loading: 0.4,
                }],
                residual_edges: vec![],
            },
            Dependence::Vine {
                edges: vec![VineEdge {
                    left: 0,
                    right: 1,
                    conditioning_depth: 0,
                    parameter: 0.4,
                }],
            },
            Dependence::Mixture {
                weights: vec![0.4, 0.6],
                components: vec![
                    Dependence::Independent,
                    Dependence::Triangular {
                        terms: vec![TriangularTerm {
                            parent: 0,
                            child: 1,
                            coefficient: 0.2,
                        }],
                    },
                ],
            },
            Dependence::Triangular {
                terms: vec![TriangularTerm {
                    parent: 0,
                    child: 1,
                    coefficient: 0.2,
                }],
            },
        ] {
            let mut value = kernel();
            value.symbolic_mut().unwrap().0.clone_from(&dependence);
            assert_canonical(value);
        }

        let additive = AdditiveTerm {
            feature: 0,
            knots: vec![0.0, 0.5, 1.0],
            values: vec![-0.2, 0.0, 0.2],
        };
        for target in [
            Target::SparseGam {
                intercept: 0.2,
                logistic: false,
                terms: vec![additive.clone()],
            },
            Target::Ga2m {
                intercept: 0.2,
                logistic: false,
                main_terms: vec![additive],
                interactions: vec![InteractionTerm {
                    left: 0,
                    right: 1,
                    left_knots: vec![0.0, 0.5, 1.0],
                    right_knots: vec![0.0, 0.5, 1.0],
                    values: vec![0.0; 9],
                }],
            },
            Target::Mars {
                intercept: 0.2,
                logistic: false,
                terms: vec![MarsTerm {
                    coefficient: 0.3,
                    factors: vec![MarsFactor {
                        feature: 0,
                        knot: 0.5,
                        direction: 1,
                    }],
                }],
            },
            Target::ObliviousTree {
                logistic: false,
                features: vec![0, 1],
                thresholds: vec![0.5, 0.5],
                leaves: vec![0.1, 0.3, 0.7, 0.9],
            },
            Target::CompactNeuralResidual {
                intercept: 0.2,
                logistic: false,
                linear_terms: vec![LinearTerm {
                    feature: 0,
                    coefficient: 0.3,
                }],
                hidden_features: vec![0, 1],
                hidden_width: 2,
                input_weights: vec![0.4, -0.2, -0.1, 0.5],
                hidden_biases: vec![0.1, -0.1],
                output_weights: vec![0.25, -0.3],
            },
        ] {
            let mut value = kernel();
            value.symbolic_mut().unwrap().1.clone_from(&target);
            assert_canonical(value);
        }

        for noise in [
            Noise::Heteroscedastic {
                intercept: -3.0,
                terms: vec![LinearTerm {
                    feature: 0,
                    coefficient: 0.5,
                }],
                minimum_sigma: 0.01,
                maximum_sigma: 0.2,
            },
            Noise::IsotonicCalibration {
                knots: vec![0.0, 0.5, 1.0],
                probabilities: vec![0.0, 0.4, 1.0],
            },
        ] {
            let mut value = kernel();
            value.symbolic_mut().unwrap().2.clone_from(&noise);
            assert_canonical(value);
        }
    }

    #[test]
    fn canonical_rans_round_trip() {
        let raw: Vec<u8> = (0..4096).map(|index| (index % 7) as u8).collect();
        let encoded = rans_encode(&raw);
        assert!(encoded.len() < raw.len());
        assert_eq!(rans_decode(&encoded, raw.len()).unwrap(), raw);
        assert_eq!(rans_encode(&raw), encoded);
    }
}

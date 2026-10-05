

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
    fn compact_research_target_dimensions_round_trip_without_widening_codec() {
        for (inputs, width) in [(12, 16), (16, 16), (12, 8), (24, 8)] {
            let target = Target::CompactNeuralResidual {
                intercept: 0.25,
                logistic: false,
                linear_terms: vec![],
                hidden_features: (0..inputs).collect(),
                hidden_width: width,
                input_weights: vec![0.125; inputs as usize * usize::from(width)],
                hidden_biases: vec![0.125; usize::from(width)],
                output_weights: vec![0.25; usize::from(width)],
            };
            let encoded = encode_target(&target);
            let decoded = decode_target(&encoded, 24).unwrap();
            assert_eq!(encode_target(&decoded), encoded);
        }
        for width in [0, 17, 32] {
            let target = Target::CompactNeuralResidual {
                intercept: 0.25,
                logistic: false,
                linear_terms: vec![],
                hidden_features: vec![0],
                hidden_width: width,
                input_weights: vec![0.125; usize::from(width)],
                hidden_biases: vec![0.125; usize::from(width)],
                output_weights: vec![0.25; usize::from(width)],
            };
            assert!(decode_target(&encode_target(&target), 24).is_err());
        }
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

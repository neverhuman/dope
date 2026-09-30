

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{LoadedKernel, decode_kernel, encode_kernel};
    use crate::model::{
        AutoregressiveTransformer, ColumnSchema, JointGenerator, JointNetwork, Kernel,
        KernelProgram, LatentDenoiser, LayerNormParameters, Marginal, NeuralArchitecture,
        RankNormalization, SchemaKind, TRANSFORMER_WIDTH, Task, Transform, TransformerBlock,
    };
    use crate::sample::{SampleOptions, sample_kernel};

    fn zero_linear(input: usize, output: usize, biases: Vec<f32>) -> QuantizedLinear {
        QuantizedLinear {
            input_dim: input as u32,
            output_dim: output as u32,
            weights: vec![0; input * output],
            scales: vec![1.0; output],
            biases,
        }
    }

    fn zero_decoder_profile(tokens: usize, latent: usize, hidden: usize) -> Box<JointDecoder> {
        let mut value_biases = Vec::with_capacity(tokens * 2);
        for _ in 0..tokens {
            value_biases.extend([0.0, -2.0]);
        }
        Box::new(JointDecoder {
            hidden_1: zero_linear(latent, hidden, vec![0.0; hidden]),
            hidden_2: zero_linear(hidden, hidden, vec![0.0; hidden]),
            value_head: zero_linear(hidden, tokens * 2, value_biases),
            missing_head: zero_linear(hidden, tokens, vec![-20.0; tokens]),
            variance_floor: 0.01,
        })
    }

    fn zero_decoder(tokens: usize) -> Box<JointDecoder> {
        zero_decoder_profile(tokens, NEURAL_LATENT_WIDTH, 128)
    }

    fn normalization(tokens: usize) -> Vec<RankNormalization> {
        vec![
            RankNormalization {
                location: 0.0,
                scale: 1.0,
                missing_probability: 0.0,
                randomized_discrete: true,
            };
            tokens
        ]
    }

    fn tvae_generator(features: usize) -> JointGenerator {
        let tokens = features + 1;
        JointGenerator {
            architecture: NeuralArchitecture::Tvae,
            profile: crate::model::NeuralProfile::Full,
            feature_permutation: (0..features as u32).collect(),
            target_marginal: Marginal::Bernoulli { probability: 0.4 },
            normalization: normalization(tokens),
            network: JointNetwork::Tvae {
                decoder: zero_decoder(tokens),
            },
            training_hash: "a".repeat(64),
            implementation_hash: "b".repeat(64),
        }
    }

    #[test]
    fn int8_linear_uses_per_output_scales() {
        let layer = QuantizedLinear {
            input_dim: 2,
            output_dim: 2,
            weights: vec![2, -1, 3, 4],
            scales: vec![0.5, 0.25],
            biases: vec![1.0, -1.0],
        };
        assert_eq!(int8_linear(&layer, &[2.0, 4.0]).unwrap(), vec![1.0, 4.5]);
    }

    #[test]
    fn softmax_is_stable_and_normalized() {
        let mut values = vec![1_000.0, 1_001.0, 999.0];
        softmax(&mut values).unwrap();
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(values[1] > values[0] && values[0] > values[2]);
    }

    #[test]
    fn layer_normalization_is_bounded() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        layer_normalize(
            &mut values,
            &LayerNormParameters {
                weight: vec![1.0; 4],
                bias: vec![0.0; 4],
            },
        )
        .unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().sum::<f32>().abs() < 1e-5);
    }

    #[test]
    fn neural_section_round_trips_and_sampling_never_falls_back() {
        let kernel = Kernel {
            task: Task::Binary,
            rows_fitted: 64,
            features: 1,
            seed: 11,
            seed_policy: 0,
            quantization_bits: 8,
            compliant: false,
            schema: vec![ColumnSchema {
                kind: SchemaKind::Binary,
                missing_probability: 0.0,
                impute: 0.0,
                transform: Transform::Identity,
            }],
            marginals: vec![Marginal::Bernoulli { probability: 0.6 }],
            program: KernelProgram::NeuralJoint(tvae_generator(1)),
        };
        let encoded = encode_kernel(&kernel).unwrap();
        assert_eq!(&encoded[..6], b"DPK3\x03\x02");
        let decoded = decode_kernel(&encoded).unwrap();
        assert_eq!(decoded, kernel);
        let loaded = LoadedKernel::V3(Box::new(decoded));
        let first = sample_kernel(
            &loaded,
            SampleOptions {
                rows: 128,
                seed: Some(9),
            },
        )
        .unwrap();
        let second = sample_kernel(
            &loaded,
            SampleOptions {
                rows: 128,
                seed: Some(9),
            },
        )
        .unwrap();
        assert_eq!(first, second);
        assert!(first.iter().all(|value| matches!(*value, 0.0 | 1.0)));
    }

    #[test]
    fn compact_profiles_encode_as_dpk33_and_sample_in_rust() {
        use crate::model::NeuralProfile;
        for profile in [
            NeuralProfile::MicroTvae4,
            NeuralProfile::MicroTvae8,
            NeuralProfile::MicroTvae12,
        ] {
            let (latent, hidden) = profile.tvae_dimensions().unwrap();
            let mut generator = tvae_generator(1);
            generator.profile = profile;
            generator.network = JointNetwork::Tvae {
                decoder: zero_decoder_profile(2, latent, hidden),
            };
            let kernel = Kernel {
                task: Task::Binary,
                rows_fitted: 32,
                features: 1,
                seed: 7,
                seed_policy: 0,
                quantization_bits: 8,
                compliant: false,
                schema: vec![ColumnSchema {
                    kind: SchemaKind::Binary,
                    missing_probability: 0.0,
                    impute: 0.0,
                    transform: Transform::Identity,
                }],
                marginals: vec![Marginal::Bernoulli { probability: 0.5 }],
                program: KernelProgram::NeuralJoint(generator.clone()),
            };
            let encoded = encode_kernel(&kernel).unwrap();
            assert_eq!(&encoded[..6], b"DPK3\x03\x03");
            assert_eq!(
                encode_kernel(&decode_kernel(&encoded).unwrap()).unwrap(),
                encoded
            );
            assert_eq!(encoded, encode_kernel(&kernel).unwrap());
            assert_eq!(
                sample_joint(&generator, 1, 73).unwrap(),
                sample_joint(&generator, 1, 73).unwrap()
            );
            if profile == NeuralProfile::MicroTvae4 {
                assert!(
                    encoded.len() < 10_240,
                    "micro TVAE encoded to {} bytes",
                    encoded.len()
                );
            }
            if let JointNetwork::Tvae { decoder } = &mut generator.network {
                decoder.hidden_1.input_dim += 1;
            }
            assert!(generator.validate(1, Task::Binary).is_err());
        }
        for profile in [NeuralProfile::TinyMat16, NeuralProfile::TinyMat24] {
            let (width, _, ff) = profile.transformer_dimensions().unwrap();
            let norm = LayerNormParameters {
                weight: vec![1.0; width],
                bias: vec![0.0; width],
            };
            let block = || TransformerBlock {
                query: zero_linear(width, width, vec![0.0; width]),
                key: zero_linear(width, width, vec![0.0; width]),
                value: zero_linear(width, width, vec![0.0; width]),
                attention_output: zero_linear(width, width, vec![0.0; width]),
                attention_norm: norm.clone(),
                feed_forward_1: zero_linear(width, ff, vec![0.0; ff]),
                feed_forward_2: zero_linear(ff, width, vec![0.0; width]),
                feed_forward_norm: norm.clone(),
            };
            let mut generator = tvae_generator(1);
            generator.architecture = NeuralArchitecture::MaskedAutoregressiveTransformer;
            generator.profile = profile;
            generator.network = JointNetwork::MaskedAutoregressiveTransformer {
                transformer: Box::new(AutoregressiveTransformer {
                    input_projection: zero_linear(2, width, vec![0.0; width]),
                    positional_embeddings: vec![0.0; 2 * width],
                    blocks: vec![block(), block()],
                    value_head: zero_linear(width, 2, vec![0.0, -2.0]),
                    missing_head: zero_linear(width, 1, vec![-20.0]),
                }),
            };
            assert_eq!(
                sample_joint(&generator, 1, 73).unwrap(),
                sample_joint(&generator, 1, 73).unwrap()
            );
            let kernel = Kernel {
                task: Task::Binary,
                rows_fitted: 32,
                features: 1,
                seed: 7,
                seed_policy: 0,
                quantization_bits: 8,
                compliant: false,
                schema: vec![ColumnSchema {
                    kind: SchemaKind::Binary,
                    missing_probability: 0.0,
                    impute: 0.0,
                    transform: Transform::Identity,
                }],
                marginals: vec![Marginal::Bernoulli { probability: 0.5 }],
                program: KernelProgram::NeuralJoint(generator),
            };
            let encoded = encode_kernel(&kernel).unwrap();
            assert_eq!(&encoded[..6], b"DPK3\x03\x03");
            assert_eq!(
                encode_kernel(&decode_kernel(&encoded).unwrap()).unwrap(),
                encoded
            );
        }
    }

    #[test]
    fn architecture_payload_mismatch_is_rejected() {
        let mut generator = tvae_generator(2);
        generator.architecture = NeuralArchitecture::TabSyn;
        assert!(generator.validate(2, Task::Binary).is_err());
        generator.architecture = NeuralArchitecture::Tvae;
        generator.feature_permutation = vec![0, 0];
        assert!(generator.validate(2, Task::Binary).is_err());
    }

    #[test]
    fn neural_sampling_respects_observed_missingness_support() {
        let mut generator = tvae_generator(2);
        for seed in 0..64 {
            let (_, missing) = sample_joint(&generator, 2, seed).unwrap();
            assert_eq!(missing, vec![false, false, false]);
        }
        generator.normalization[0].missing_probability = 1.0;
        generator.feature_permutation = vec![1, 0];
        let (_, missing) = sample_joint(&generator, 2, 11).unwrap();
        assert_eq!(missing, vec![true, false, false]);
    }

    #[test]
    fn transformer_and_ddim_sampling_are_deterministic_and_finite() {
        let features = 2;
        let tokens = features + 1;
        let norm = LayerNormParameters {
            weight: vec![1.0; TRANSFORMER_WIDTH],
            bias: vec![0.0; TRANSFORMER_WIDTH],
        };
        let block = || TransformerBlock {
            query: zero_linear(64, 64, vec![0.0; 64]),
            key: zero_linear(64, 64, vec![0.0; 64]),
            value: zero_linear(64, 64, vec![0.0; 64]),
            attention_output: zero_linear(64, 64, vec![0.0; 64]),
            attention_norm: norm.clone(),
            feed_forward_1: zero_linear(64, 128, vec![0.0; 128]),
            feed_forward_2: zero_linear(128, 64, vec![0.0; 64]),
            feed_forward_norm: norm.clone(),
        };
        let transformer = JointGenerator {
            architecture: NeuralArchitecture::MaskedAutoregressiveTransformer,
            profile: crate::model::NeuralProfile::Full,
            feature_permutation: vec![1, 0],
            target_marginal: Marginal::Bernoulli { probability: 0.5 },
            normalization: normalization(tokens),
            network: JointNetwork::MaskedAutoregressiveTransformer {
                transformer: Box::new(AutoregressiveTransformer {
                    input_projection: zero_linear(2, 64, vec![0.0; 64]),
                    positional_embeddings: vec![0.0; tokens * 64],
                    blocks: vec![block(), block()],
                    value_head: zero_linear(64, 2, vec![0.0, -2.0]),
                    missing_head: zero_linear(64, 1, vec![-20.0]),
                }),
            },
            training_hash: "c".repeat(64),
            implementation_hash: "d".repeat(64),
        };
        let schedule = (0..100)
            .map(|step| 1.0 - (step + 1) as f32 / 101.0)
            .collect::<Vec<_>>();
        let tabsyn = JointGenerator {
            architecture: NeuralArchitecture::TabSyn,
            profile: crate::model::NeuralProfile::Full,
            feature_permutation: vec![0, 1],
            target_marginal: Marginal::Bernoulli { probability: 0.5 },
            normalization: normalization(tokens),
            network: JointNetwork::TabSyn {
                decoder: zero_decoder(tokens),
                denoiser: Box::new(LatentDenoiser {
                    hidden_1: zero_linear(64, 128, vec![0.0; 128]),
                    hidden_2: zero_linear(128, 128, vec![0.0; 128]),
                    hidden_3: zero_linear(128, 128, vec![0.0; 128]),
                    output: zero_linear(128, 32, vec![0.0; 32]),
                }),
                alpha_cumprod: schedule.clone(),
                inference_timesteps: (0..32).map(|index| (99 - index * 99 / 31) as u8).collect(),
            },
            training_hash: "e".repeat(64),
            implementation_hash: "f".repeat(64),
        };
        let tabddpm = JointGenerator {
            architecture: NeuralArchitecture::TabDdpm,
            profile: crate::model::NeuralProfile::Full,
            feature_permutation: vec![0, 1],
            target_marginal: Marginal::Bernoulli { probability: 0.5 },
            normalization: normalization(tokens),
            network: JointNetwork::TabDdpm {
                denoiser: Box::new(LatentDenoiser {
                    hidden_1: zero_linear(tokens + 32, 128, vec![0.0; 128]),
                    hidden_2: zero_linear(128, 128, vec![0.0; 128]),
                    hidden_3: zero_linear(128, 128, vec![0.0; 128]),
                    output: zero_linear(128, tokens, vec![0.0; tokens]),
                }),
                alpha_cumprod: schedule,
                inference_timesteps: (0..32).map(|index| (99 - index * 99 / 31) as u8).collect(),
            },
            training_hash: "1".repeat(64),
            implementation_hash: "2".repeat(64),
        };
        let tabddpm_kernel = Kernel {
            task: Task::Binary,
            rows_fitted: 32,
            features: features as u32,
            seed: 7,
            seed_policy: 0,
            quantization_bits: 8,
            compliant: false,
            schema: vec![
                ColumnSchema {
                    kind: SchemaKind::Binary,
                    missing_probability: 0.0,
                    impute: 0.0,
                    transform: Transform::Identity,
                };
                features
            ],
            marginals: vec![Marginal::Bernoulli { probability: 0.5 }; features],
            program: KernelProgram::NeuralJoint(tabddpm.clone()),
        };
        let encoded = encode_kernel(&tabddpm_kernel).unwrap();
        assert_eq!(&encoded[..6], b"DPK3\x03\x03");
        assert_eq!(
            encode_kernel(&decode_kernel(&encoded).unwrap()).unwrap(),
            encoded
        );
        for generator in [transformer, tabsyn, tabddpm] {
            let first = sample_joint(&generator, features, 123).unwrap();
            let second = sample_joint(&generator, features, 123).unwrap();
            assert_eq!(first, second);
            assert!(first.0.iter().all(|value| value.is_finite()));
            assert!(first.1.iter().all(|missing| !missing));
        }
    }
}

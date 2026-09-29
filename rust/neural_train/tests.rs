#[cfg(all(test, feature = "gpu-training"))]
mod tests {
    use super::*;
    use crate::codec::encode_kernel;
    use crate::model::{
        ColumnSchema, Kernel, KernelProgram, Marginal, QuantizedLinear, RankNormalization,
        SchemaKind, Task, Transform,
    };
    use std::time::Duration;
    use tch::{Device, Tensor};

    fn fixture(rows: usize, features: usize) -> NeuralTrainingData {
        let tokens = features + 1;
        NeuralTrainingData {
            rows,
            features,
            ranks: (0..rows * tokens)
                .map(|index| {
                    let row = index / tokens;
                    let column = index % tokens;
                    (((row * 37 + column * 101) % 997) as f32 / 997.0 - 0.5) * 4.0
                })
                .collect(),
            missing: vec![0.0; rows * tokens],
            row_weights: vec![1.0; rows],
            feature_permutation: (0..features as u32).collect(),
            target_marginal: Marginal::Gaussian {
                mean: 0.5,
                sigma: 0.2,
            },
            normalization: vec![
                RankNormalization {
                    location: 0.0,
                    scale: 1.0,
                    missing_probability: 0.0,
                    randomized_discrete: false,
                };
                tokens
            ],
            training_hash: "a".repeat(64),
        }
    }

    #[test]
    fn seeded_tvae_training_repeats_exactly() {
        let data = fixture(32, 4);
        let train = || {
            let started = Instant::now();
            fit_joint_generator(
                &data,
                NeuralTrainingConfig {
                    architecture: NeuralArchitecture::Tvae,
                    profile: NeuralProfile::Full,
                    target_weight: 2.0,
                    structural_penalty: 0.0,
                    seed: 17_729,
                    deadline: started + Duration::from_secs(60),
                },
            )
            .unwrap()
        };
        assert_eq!(train(), train());
    }

    #[test]
    fn micro_tvae_and_direct_diffusion_repeat_and_export() {
        let data = fixture(32, 2);
        for (architecture, profile) in [
            (NeuralArchitecture::Tvae, NeuralProfile::MicroTvae4),
            (NeuralArchitecture::TabDdpm, NeuralProfile::Full),
        ] {
            let train = || {
                fit_joint_generator(
                    &data,
                    NeuralTrainingConfig {
                        architecture,
                        profile,
                        target_weight: 2.0,
                        structural_penalty: 0.0,
                        seed: 17_729,
                        deadline: Instant::now() + Duration::from_secs(120),
                    },
                )
                .unwrap()
            };
            let first = train();
            let second = train();
            assert_eq!(first, second);
            let kernel = Kernel {
                task: Task::Regression,
                rows_fitted: data.rows as u64,
                features: data.features as u32,
                seed: 17_729,
                seed_policy: 0,
                quantization_bits: 8,
                compliant: false,
                schema: vec![
                    ColumnSchema {
                        kind: SchemaKind::Continuous,
                        missing_probability: 0.0,
                        impute: 0.5,
                        transform: Transform::Identity,
                    };
                    data.features
                ],
                marginals: vec![
                    Marginal::Gaussian {
                        mean: 0.5,
                        sigma: 0.2,
                    };
                    data.features
                ],
                program: KernelProgram::NeuralJoint(first),
            };
            let bytes = encode_kernel(&kernel).unwrap();
            assert_eq!(&bytes[..6], b"DPK3\x03\x03");
            if architecture == NeuralArchitecture::Tvae {
                assert!(bytes.len() < 10_240, "micro TVAE is {} bytes", bytes.len());
            }
        }
    }

    fn parity_layer(input: usize, output: usize, salt: usize) -> QuantizedLinear {
        QuantizedLinear {
            input_dim: input as u32,
            output_dim: output as u32,
            weights: (0..input * output)
                .map(|index| ((index * 17 + salt * 31) % 31) as i8 - 15)
                .collect(),
            scales: (0..output)
                .map(|channel| 0.0005 + (channel + salt) as f32 * 0.00001)
                .collect(),
            biases: (0..output)
                .map(|channel| (channel as f32 - output as f32 * 0.5) * 0.001)
                .collect(),
        }
    }

    fn parity_splitmix64(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn libtorch_linear(input: &Tensor, layer: &QuantizedLinear) -> Tensor {
        let device = input.device();
        let weights = Tensor::from_slice(
            &layer
                .weights
                .iter()
                .map(|&weight| f32::from(weight))
                .collect::<Vec<_>>(),
        )
        .view([i64::from(layer.output_dim), i64::from(layer.input_dim)])
        .to_device(device);
        let scales = Tensor::from_slice(&layer.scales).to_device(device);
        let biases = Tensor::from_slice(&layer.biases).to_device(device);
        input.matmul(&weights.transpose(0, 1)) * scales + biases
    }

    #[test]
    fn rust_libtorch_quantized_golden_matches_for_100000_draws() {
        const DRAWS: usize = 100_000;
        const BATCH: usize = 2_048;
        let first = parity_layer(32, 8, 3);
        let second = parity_layer(8, 2, 7);
        first.validate().unwrap();
        second.validate().unwrap();
        crate::libtorch::with_seeded_libtorch(91_773, || {
            let device = Device::Cuda(0);
            let mut maximum_difference = 0.0f64;
            let mut decisions_agree = 0usize;
            let mut decisions = 0usize;
            for start in (0..DRAWS).step_by(BATCH) {
                let rows = (DRAWS - start).min(BATCH);
                let inputs = (0..rows * 32)
                    .map(|offset| {
                        let bits =
                            parity_splitmix64((start * 32 + offset) as u64 ^ 0xa076_1d64_78bd_642f);
                        ((bits >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
                    })
                    .collect::<Vec<_>>();
                let reference_input = Tensor::from_slice(&inputs)
                    .view([rows as i64, 32])
                    .to_device(device);
                let reference = libtorch_linear(
                    &libtorch_linear(&reference_input, &first).gelu("tanh"),
                    &second,
                );
                let reference =
                    Vec::<f32>::try_from(reference.to_device(Device::Cpu).view([-1]))
                        .map_err(|error| DopeError::Data(format!("parity output: {error}")))?;
                for (row, expected) in inputs.chunks_exact(32).zip(reference.chunks_exact(2)) {
                    let hidden = crate::neural::int8_linear(&first, row)?
                        .into_iter()
                        .map(crate::neural::gelu)
                        .collect::<Vec<_>>();
                    let actual = crate::neural::int8_linear(&second, &hidden)?;
                    for (&actual, &expected) in actual.iter().zip(expected) {
                        maximum_difference =
                            maximum_difference.max(f64::from((actual - expected).abs()));
                        decisions_agree += usize::from((actual >= 0.0) == (expected >= 0.0));
                        decisions += 1;
                    }
                }
            }
            assert!(maximum_difference <= 1e-3, "{maximum_difference}");
            assert!(decisions_agree as f64 / decisions as f64 >= 0.999);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn transformer_486_by_273_completes_with_bounded_batches() {
        let rows = 486usize;
        let features = 273usize;
        let data = fixture(rows, features);
        let started = Instant::now();
        let generator = fit_joint_generator(
            &data,
            NeuralTrainingConfig {
                architecture: NeuralArchitecture::MaskedAutoregressiveTransformer,
                profile: NeuralProfile::Full,
                target_weight: 2.0,
                structural_penalty: 0.0,
                seed: 17_729,
                deadline: started + Duration::from_secs(600),
            },
        )
        .unwrap();
        assert!(started.elapsed() < Duration::from_secs(600));
        generator
            .validate(features, crate::model::Task::Regression)
            .unwrap();
    }
}

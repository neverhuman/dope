use super::*;
use dope_kernel::model::*;

#[test]
fn corrected_activation_uses_independent_training_formula_goldens() {
    for (x, expected) in [
        (-2.0, -0.045_402_307),
        (-1.0, -0.158_808),
        (0.0, 0.0),
        (1.0, 0.841_192),
        (2.0, 1.954_597_7),
    ] {
        let actual = dope_kernel::neural::deployment::gelu_tanh(x);
        assert!((actual - expected).abs() < 3e-7);
    }
}

/// This test deliberately fails without a real working CUDA device. Enabling
/// the feature and passing CPU fixtures alone must never qualify GPU execution.
#[cfg(feature = "cuda")]
#[test]
fn cuda_execution_matches_cpu_for_both_numerical_profiles() {
    let native = dope_kernel::encode_kernel(&fixture()).unwrap();
    let options = SampleOptions {
        rows: 259,
        seed: Some(u64::MAX),
    };
    for activation in [Activation::LegacyV1, Activation::TanhV2] {
        let carrier = Carrier::load(&pack(&native, activation).unwrap()).unwrap();
        let (reference, _) = carrier.sample(options, DeviceRequest::Cpu).unwrap();
        let (actual, used) = carrier.sample(options, DeviceRequest::Cuda).unwrap();
        assert_eq!(used, "cuda-candle");
        assert_eq!(actual.len(), reference.len());
        assert!(actual.iter().all(|v| v.is_finite()));
        let error = actual
            .iter()
            .zip(reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        eprintln!("activation={activation:?} max_abs_error={error:e}");
        assert!(
            error <= 1e-5,
            "CUDA synthetic-fixture parity exceeded declared tolerance"
        );
    }
}

fn linear(input: usize, output: usize, phase: i32) -> QuantizedLinear {
    QuantizedLinear {
        input_dim: input as u32,
        output_dim: output as u32,
        weights: (0..input * output)
            .map(|i| ((i as i32 * 7 + phase) % 19 - 9) as i8)
            .collect(),
        scales: vec![0.025; output],
        biases: (0..output).map(|i| (i as f32 - 2.0) * 0.04).collect(),
    }
}

fn fixture() -> Kernel {
    Kernel {
        task: Task::Regression,
        rows_fitted: 64,
        features: 2,
        seed: 11,
        seed_policy: 0,
        quantization_bits: 8,
        compliant: false,
        schema: vec![
            ColumnSchema {
                kind: SchemaKind::Continuous,
                missing_probability: 0.0,
                impute: 0.5,
                transform: Transform::Identity
            };
            2
        ],
        marginals: vec![
            Marginal::QuantileSpline {
                values: vec![0.0, 0.2, 0.4, 0.7, 1.0]
            };
            2
        ],
        program: KernelProgram::NeuralJoint(JointGenerator {
            architecture: NeuralArchitecture::Tvae,
            profile: NeuralProfile::MicroTvae4,
            feature_permutation: vec![1, 0],
            target_marginal: Marginal::QuantileSpline {
                values: vec![0.0, 0.1, 0.4, 0.6, 1.0],
            },
            normalization: vec![
                RankNormalization {
                    location: 0.0,
                    scale: 1.0,
                    missing_probability: 0.0,
                    randomized_discrete: false
                };
                3
            ],
            network: JointNetwork::Tvae {
                decoder: Box::new(JointDecoder {
                    hidden_1: linear(4, 16, 3),
                    hidden_2: linear(16, 16, 5),
                    value_head: linear(16, 6, 7),
                    missing_head: linear(16, 3, 11),
                    variance_floor: 0.01,
                }),
            },
            training_hash: "a".repeat(64),
            implementation_hash: "b".repeat(64),
        }),
    }
}

#[test]
fn roundtrip_charges_entire_file_and_legacy_cpu_draws_are_bit_exact() {
    let native = dope_kernel::encode_kernel(&fixture()).unwrap();
    let bytes = pack(&native, Activation::LegacyV1).unwrap();
    let loaded = Carrier::load(&bytes).unwrap();
    assert_eq!(loaded.charged_bytes, bytes.len());
    assert!(bytes.len() > native.len());
    let options = SampleOptions {
        rows: 259,
        seed: Some(u64::MAX),
    };
    let (actual, label) = loaded.sample(options, DeviceRequest::Cpu).unwrap();
    let expected = sample_kernel(&LoadedKernel::V3(Box::new(loaded.kernel)), options).unwrap();
    assert_eq!(label, "cpu-native");
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn nonzero_tensor_heads_and_native_rng_match_scalar_decoder() {
    let kernel = fixture();
    let KernelProgram::NeuralJoint(g) = &kernel.program else {
        unreachable!()
    };
    let d = decoder(&kernel).unwrap();
    let layers = TensorDecoder::new(d, &Device::Cpu).unwrap();
    let seeds = joint_row_seeds(
        &kernel,
        SampleOptions {
            rows: 259,
            seed: Some(83),
        },
    )
    .unwrap();
    for seed in seeds {
        let latent = tvae_latent(g, 2, seed).unwrap();
        let x = Tensor::from_vec(latent, (1, 4), &Device::Cpu).unwrap();
        let (p, m) = layers.forward(&x, Activation::LegacyV1).unwrap();
        let (ranks, masks) = finish_tvae(
            g,
            2,
            seed,
            &p.to_vec2::<f32>().unwrap()[0],
            &m.to_vec2::<f32>().unwrap()[0],
        )
        .unwrap();
        let (reference, reference_masks) = dope_kernel::neural::sample_joint(g, 2, seed).unwrap();
        assert_eq!(masks, reference_masks);
        assert!(
            ranks
                .iter()
                .zip(reference)
                .all(|(a, b)| (a - b).abs() < 2e-6)
        );
    }
}

#[test]
fn corrected_profile_is_finite_deterministic_and_changes_nonzero_network() {
    let native = dope_kernel::encode_kernel(&fixture()).unwrap();
    let legacy = Carrier::load(&pack(&native, Activation::LegacyV1).unwrap()).unwrap();
    let corrected = Carrier::load(&pack(&native, Activation::TanhV2).unwrap()).unwrap();
    let options = SampleOptions {
        rows: 259,
        seed: Some(83),
    };
    let (first, device) = corrected.sample(options, DeviceRequest::Cpu).unwrap();
    assert_eq!(device, "cpu-candle");
    assert!(first.iter().all(|v| v.is_finite()));
    assert_eq!(
        first,
        corrected.sample(options, DeviceRequest::Cpu).unwrap().0
    );
    assert!(
        first
            .iter()
            .zip(legacy.sample(options, DeviceRequest::Cpu).unwrap().0)
            .any(|(a, b)| (a - b).abs() > 1e-5)
    );
}

#[test]
fn mismatched_weights_truncation_and_budgets_refuse() {
    let native = dope_kernel::encode_kernel(&fixture()).unwrap();
    let mut bytes = pack(&native, Activation::LegacyV1).unwrap();
    let view = SafeTensors::deserialize(&bytes).unwrap();
    let weight = view.tensor("hidden_1.weight").unwrap().data();
    let offset = weight.as_ptr() as usize - bytes.as_ptr() as usize;
    bytes[offset] ^= 1;
    assert_eq!(
        Carrier::load(&bytes).err(),
        Some("tensor_inventory_mismatch")
    );
    assert!(Carrier::load(&bytes[..bytes.len() - 1]).is_err());
    let good = Carrier::load(&pack(&native, Activation::LegacyV1).unwrap()).unwrap();
    assert_eq!(
        good.sample(
            SampleOptions {
                rows: 0,
                seed: None
            },
            DeviceRequest::Cpu
        )
        .err(),
        Some("sampling_budget")
    );
    assert_eq!(
        good.sample(
            SampleOptions {
                rows: 100_001,
                seed: None
            },
            DeviceRequest::Cpu
        )
        .err(),
        Some("sampling_budget")
    );
    #[cfg(not(feature = "cuda"))]
    assert_eq!(
        good.sample(
            SampleOptions {
                rows: 1,
                seed: None
            },
            DeviceRequest::Cuda
        )
        .err(),
        Some("cuda_unavailable")
    );
}
